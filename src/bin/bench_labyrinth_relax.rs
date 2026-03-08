// src/bin/bench_labyrinth_relax.rs
//
// Labyrinth Domain Benchmark — Flagship AMR + Composite Demag Demonstration
// =========================================================================
//
// A PMA + DMI thin film initialised from random magnetisation relaxes under
// damped LLG into a labyrinth/maze stripe domain pattern.  The chiral Néel
// domain walls (δ ≈ 5 nm) are sub-cell features on the coarse grid that the
// AMR mesh must track as extended, winding structures.
//
// Four demag solvers are compared:
//
//   1. Fine FFT         — Newell-tensor FFT on the fine-equivalent grid (reference)
//   2. Coarse FFT only  — FFT on L0, no AMR, no patches
//   3. Coarse FFT + AMR — FFT on L0 with M-restriction from patches
//   4. Composite MG+AMR — Composite V-cycle demag with fine-resolution B in patches
//
// The benchmark produces:
//   - Per-step terminal output (RMSE, coverage, patch counts, timing)
//   - CSV logs (rmse_log.csv, timing_log.csv, coverage_log.csv)
//   - mz colour maps with AMR patch overlays (García-Cervera Fig. 4/5 style)
//   - Patch grid-cell detail plots
//   - Demag profile cuts through domain walls
//   - Summary table comparing all four solvers
//
// Run:
//   cargo run --release --bin bench_labyrinth_relax -- --plots
//
// With composite V-cycle demag:
//   LLG_DEMAG_COMPOSITE_VCYCLE=1 cargo run --release --bin bench_labyrinth_relax -- --plots
//
// Skip fine reference (fast, timing-only):
//   cargo run --release --bin bench_labyrinth_relax -- --skip-fine-ref --plots
//
// Custom parameters:
//   LLG_LAB_NX=256 LLG_LAB_DZ=4e-9 cargo run --release --bin bench_labyrinth_relax -- --plots

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use plotters::prelude::*;

use llg_sim::effective_field::{FieldMask, coarse_fft_demag, demag_fft_uniform, mg_composite};
use llg_sim::geometry_mask::{MaskShape, cell_center_xy_centered};
use llg_sim::grid::Grid2D;
use llg_sim::initial_states;
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material, MU0};
use llg_sim::vector_field::VectorField2D;

use llg_sim::amr::indicator::IndicatorKind;
use llg_sim::amr::regrid::maybe_regrid_nested_levels;
use llg_sim::amr::{
    AmrHierarchy2D, AmrStepperRK4, ClusterPolicy, Connectivity, Rect2i, RegridPolicy,
};

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn ensure_dir(path: &str) {
    if !Path::new(path).exists() {
        fs::create_dir_all(path).unwrap();
    }
}

fn idx(i: usize, j: usize, nx: usize) -> usize {
    j * nx + i
}

fn pow_usize(mut base: usize, mut exp: usize) -> usize {
    let mut out = 1usize;
    while exp > 0 {
        if (exp & 1) == 1 { out = out.saturating_mul(base); }
        exp >>= 1;
        if exp > 0 { base = base.saturating_mul(base); }
    }
    out
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Flatten AMR hierarchy to a uniform fine grid for comparison.
fn flatten_to_fine(h: &AmrHierarchy2D, target: Grid2D) -> VectorField2D {
    let m = h.flatten_to_uniform_fine();
    if m.grid.nx == target.nx && m.grid.ny == target.ny {
        m
    } else {
        m.resample_to_grid(target)
    }
}

/// Compute RMSE and max delta between two B-fields (normalised by max_b).
#[allow(dead_code)]
fn rmse_and_max(a: &VectorField2D, b: &VectorField2D, max_b: f64) -> (f64, f64) {
    let n = a.data.len();
    let mut sum2 = 0.0f64;
    let mut maxd = 0.0f64;
    for i in 0..n {
        let dx = a.data[i][0] - b.data[i][0];
        let dy = a.data[i][1] - b.data[i][1];
        let dz = a.data[i][2] - b.data[i][2];
        let d2 = dx * dx + dy * dy + dz * dz;
        sum2 += d2;
        maxd = maxd.max(d2.sqrt());
    }
    let rmse = (sum2 / n as f64).sqrt();
    (rmse / max_b * 100.0, maxd / max_b * 100.0)
}

/// Count cells where |mz| < threshold (= wall region).
fn count_wall_cells(m: &VectorField2D, mz_thresh: f64) -> usize {
    m.data.iter().filter(|v| v[2].abs() < mz_thresh).count()
}

/// Compute average |∇m| over the domain (indicator of texture complexity).
fn average_grad_m(m: &VectorField2D) -> f64 {
    let nx = m.grid.nx;
    let ny = m.grid.ny;
    let dx = m.grid.dx;
    let dy = m.grid.dy;
    let mut total = 0.0f64;
    let mut n = 0usize;
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let _c = m.data[j * nx + i];
            let xp = m.data[j * nx + (i + 1)];
            let xm = m.data[j * nx + (i - 1)];
            let yp = m.data[(j + 1) * nx + i];
            let ym = m.data[(j - 1) * nx + i];
            let mut g2 = 0.0f64;
            for k in 0..3 {
                let gx = (xp[k] - xm[k]) / (2.0 * dx);
                let gy = (yp[k] - ym[k]) / (2.0 * dy);
                g2 += gx * gx + gy * gy;
            }
            total += g2.sqrt();
            n += 1;
        }
    }
    if n > 0 { total / n as f64 } else { 0.0 }
}

/// Compute coverage: fraction of fine-equivalent cells covered by AMR patches.
fn compute_coverage(h: &AmrHierarchy2D, amr_max_level: usize, refine_ratio: usize) -> (usize, f64) {
    let fine_cells_total = h.base_grid.nx * pow_usize(refine_ratio, amr_max_level)
        * h.base_grid.ny * pow_usize(refine_ratio, amr_max_level);
    let mut amr_fine_cells = 0usize;
    for p in &h.patches {
        let inx = p.interior_i1() - p.interior_i0();
        let iny = p.interior_j1() - p.interior_j0();
        amr_fine_cells += inx * iny * pow_usize(refine_ratio, amr_max_level - 1);
    }
    for (li, lvl) in h.patches_l2plus.iter().enumerate() {
        let level_ratio = pow_usize(refine_ratio, amr_max_level - (li + 2));
        for p in lvl {
            let inx = p.interior_i1() - p.interior_i0();
            let iny = p.interior_j1() - p.interior_j0();
            amr_fine_cells += inx * iny * level_ratio;
        }
    }
    let frac = amr_fine_cells as f64 / fine_cells_total as f64;
    (amr_fine_cells, frac)
}

/// Extract Rect2i for all patches at a given AMR level.
fn level_rects(h: &AmrHierarchy2D, level_idx: usize) -> Vec<Rect2i> {
    if level_idx == 0 { return Vec::new(); }
    if level_idx == 1 {
        return h.patches.iter().map(|p| p.coarse_rect.clone()).collect();
    }
    let li = level_idx - 2;
    h.patches_l2plus.get(li)
        .map(|lvl| lvl.iter().map(|p| p.coarse_rect.clone()).collect())
        .unwrap_or_default()
}

/// Count patches at each level.
fn patch_counts(h: &AmrHierarchy2D) -> (usize, usize, usize) {
    let l1 = h.patches.len();
    let l2 = h.patches_l2plus.get(0).map(|v| v.len()).unwrap_or(0);
    let l3 = h.patches_l2plus.get(1).map(|v| v.len()).unwrap_or(0);
    (l1, l2, l3)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Visualisation
// ═══════════════════════════════════════════════════════════════════════════

/// Blue-White-Red colour map for mz ∈ [-1, +1].
fn mz_to_bwr(mz: f64) -> RGBColor {
    let t = ((mz + 1.0) * 0.5).clamp(0.0, 1.0);
    if t < 0.5 {
        let a = t / 0.5;
        RGBColor((255.0 * a) as u8, (255.0 * a) as u8, 255)
    } else {
        let a = (t - 0.5) / 0.5;
        RGBColor(255, (255.0 * (1.0 - a)) as u8, (255.0 * (1.0 - a)) as u8)
    }
}

/// Save mz colour map with AMR patch boundary overlays (García-Cervera Fig. 4 style).
fn save_mz_with_patches(
    m: &VectorField2D,
    l1_rects: &[Rect2i], l2_rects: &[Rect2i], l3_rects: &[Rect2i],
    path: &str, caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = m.grid.nx as i32;
    let ny = m.grid.ny as i32;
    let root = BitMapBackend::new(path, (1024, 1024)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..nx, 0..ny)?;
    chart.configure_mesh().disable_mesh().draw()?;

    // mz colour cells.
    chart.draw_series((0..m.grid.ny).flat_map(|j| {
        (0..m.grid.nx).map(move |i| {
            let mz = m.data[idx(i, j, m.grid.nx)][2];
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                mz_to_bwr(mz).filled(),
            )
        })
    }))?;

    // Overlay AMR patch rectangles: L1 = yellow, L2 = green, L3 = blue.
    for (rects, colour, width) in [
        (l1_rects, RGBColor(255, 200, 0), 2u32),
        (l2_rects, RGBColor(0, 200, 0), 2u32),
        (l3_rects, RGBColor(0, 100, 255), 2u32),
    ] {
        for r in rects {
            let x0 = r.i0 as i32;
            let y0 = r.j0 as i32;
            let x1 = (r.i0 + r.nx) as i32;
            let y1 = (r.j0 + r.ny) as i32;
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                colour.stroke_width(width),
            ))).ok();
        }
    }
    root.present()?;
    Ok(())
}

/// Save patch grid-cell detail plot (García-Cervera Fig. 5 style).
///
/// Shows mz colour at fine resolution within a sub-region, overlaid with
/// grid cell boundaries at each AMR level.  This reveals the multi-resolution
/// structure.
fn save_patch_detail(
    m: &VectorField2D,
    h: &AmrHierarchy2D,
    center_i: usize, center_j: usize,
    half_extent: usize,
    path: &str, caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let i0 = center_i.saturating_sub(half_extent);
    let j0 = center_j.saturating_sub(half_extent);
    let i1 = (center_i + half_extent).min(m.grid.nx);
    let j1 = (center_j + half_extent).min(m.grid.ny);
    let wnx = (i1 - i0) as i32;
    let wny = (j1 - j0) as i32;

    let root = BitMapBackend::new(path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..wnx, 0..wny)?;
    chart.configure_mesh().disable_mesh().draw()?;

    // mz colour for the sub-region.
    for j in j0..j1 {
        for i in i0..i1 {
            let mz = m.data[idx(i, j, m.grid.nx)][2];
            let pi = (i - i0) as i32;
            let pj = (j - j0) as i32;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(pi, pj), (pi + 1, pj + 1)],
                mz_to_bwr(mz).filled(),
            ))).ok();
        }
    }

    // Draw coarse grid lines (grey, thin).
    let grey = RGBColor(100, 100, 100);
    for i in i0..=i1 {
        let pi = (i - i0) as i32;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(pi, 0), (pi, wny)],
            grey.stroke_width(1),
        ))).ok();
    }
    for j in j0..=j1 {
        let pj = (j - j0) as i32;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(0, pj), (wnx, pj)],
            grey.stroke_width(1),
        ))).ok();
    }

    // Draw patch boundaries with thicker coloured outlines.
    let _ratio = h.ratio;
    for (level, rects, colour) in [
        (1usize, level_rects(h, 1), RGBColor(255, 200, 0)),
        (2, level_rects(h, 2), RGBColor(0, 200, 0)),
        (3, level_rects(h, 3), RGBColor(0, 100, 255)),
    ] {
        let _ = level; // suppress unused warning
        for r in &rects {
            let rx0 = (r.i0 as i32 - i0 as i32).max(0);
            let ry0 = (r.j0 as i32 - j0 as i32).max(0);
            let rx1 = ((r.i0 + r.nx) as i32 - i0 as i32).min(wnx);
            let ry1 = ((r.j0 + r.ny) as i32 - j0 as i32).min(wny);
            if rx1 > rx0 && ry1 > ry0 {
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1), (rx0, ry0)],
                    colour.stroke_width(3),
                ))).ok();
            }
        }
    }

    root.present()?;
    Ok(())
}

/// Save a Bx profile along a horizontal cut through the domain, comparing
/// multiple solvers.
fn save_bx_profile(
    profiles: &[(&str, &[f64], RGBColor)],
    x_coords_nm: &[f64],
    path: &str, caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (1000, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let xmin = x_coords_nm.first().copied().unwrap_or(0.0);
    let xmax = x_coords_nm.last().copied().unwrap_or(1.0);
    let (ymin, ymax) = profiles.iter().flat_map(|(_, data, _)| data.iter())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let ymarg = (ymax - ymin).abs() * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(60)
        .build_cartesian_2d(xmin..xmax, (ymin - ymarg)..(ymax + ymarg))?;
    chart.configure_mesh()
        .x_desc("x (nm)")
        .y_desc("Bx (T)")
        .draw()?;

    for (label, data, colour) in profiles {
        let pts: Vec<(f64, f64)> = x_coords_nm.iter().zip(data.iter())
            .map(|(&x, &y)| (x, y)).collect();
        chart.draw_series(LineSeries::new(pts, colour.stroke_width(2)))?
            .label(*label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colour.stroke_width(2)));
    }
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let do_plots = args.iter().any(|a| a == "--plots");
    let skip_fine = args.iter().any(|a| a == "--skip-fine-ref");

    // ── Parameters (env var overridable) ──────────────────────────────────

    let base_nx = env_usize("LLG_LAB_NX", 128);
    let base_ny = env_usize("LLG_LAB_NY", 128);
    let lx = env_f64("LLG_LAB_LX", 1.0e-6);
    let ly = env_f64("LLG_LAB_LY", 1.0e-6);
    let dz = env_f64("LLG_LAB_DZ", 3.0e-9);
    let alpha = env_f64("LLG_LAB_ALPHA", 0.3);
    let steps_total = env_usize("LLG_LAB_STEPS", 2000);
    let seed = env_usize("LLG_LAB_SEED", 42) as u64;
    let regrid_interval = env_usize("LLG_LAB_REGRID", 20);
    let output_interval = env_usize("LLG_LAB_OUTPUT", 50);

    let dx = lx / base_nx as f64;
    let dy = ly / base_ny as f64;
    let dt = env_f64("LLG_LAB_DT", 5.0e-14);

    let amr_max_level = env_usize("LLG_AMR_MAX_LEVEL", 3);
    let refine_ratio = 2usize;
    let ghost = 2usize;
    let ref_ratio_total = pow_usize(refine_ratio, amr_max_level);
    let fine_nx = base_nx * ref_ratio_total;
    let fine_ny = base_ny * ref_ratio_total;

    // ── Material: Co/Pt with interfacial DMI ─────────────────────────────

    let ms = env_f64("LLG_LAB_MS", 580.0e3);
    let a_ex = env_f64("LLG_LAB_AEX", 15.0e-12);
    let d_dmi = env_f64("LLG_LAB_DMI", 3.0e-3);
    let k_u = env_f64("LLG_LAB_KU", 0.7e6);

    let k_eff = k_u - MU0 * ms * ms / 2.0;
    let delta_dw = if k_eff > 0.0 { (a_ex / k_eff).sqrt() } else { f64::NAN };
    let l_ex = (2.0 * a_ex / (MU0 * ms * ms)).sqrt();
    let d_c = if k_eff > 0.0 {
        4.0 * (a_ex * k_eff).sqrt() / std::f64::consts::PI
    } else { f64::NAN };
    let q_factor = 2.0 * k_u / (MU0 * ms * ms);

    let out_dir = "out/labyrinth";
    ensure_dir(out_dir);

    let total_t0 = Instant::now();

    // ── Print header ─────────────────────────────────────────────────────

    let bar = "═".repeat(66);
    let thin = "─".repeat(66);
    println!("╔{bar}╗");
    println!("║{:^66}║", "Labyrinth Domain Benchmark — AMR Demag Demonstration");
    println!("╚{bar}╝");
    println!();
    println!("  Domain:      {:.0} nm × {:.0} nm × {:.1} nm",
        lx * 1e9, ly * 1e9, dz * 1e9);
    println!("  Base grid:   {} × {}, dx = {:.2} nm, dx/dz = {:.1}",
        base_nx, base_ny, dx * 1e9, dx / dz);
    println!("  Fine grid:   {} × {} ({}× refinement, {} AMR levels)",
        fine_nx, fine_ny, ref_ratio_total, amr_max_level);
    println!("  Mode:        LLG relaxation → frozen-M demag comparison");
    println!();
    println!("  Material:    Ms={:.0} kA/m, A={:.0} pJ/m, D={:.1} mJ/m², Ku={:.2} MJ/m³",
        ms / 1e3, a_ex / 1e-12, d_dmi / 1e-3, k_u / 1e6);
    println!("  Derived:     l_ex={:.1} nm, δ_DW={:.1} nm, K_eff={:.3} MJ/m³, Q={:.2}",
        l_ex * 1e9, delta_dw * 1e9, k_eff / 1e6, q_factor);
    println!("               D/Dc={:.2}, dx/dz={:.1}", d_dmi / d_c, dx / dz);
    println!();
    println!("  Solver:      α={:.2}, dt={:.0e} s, steps={}, seed={}",
        alpha, dt, steps_total, seed);
    println!("  AMR:         regrid every {} steps, output every {} steps",
        regrid_interval, output_interval);
    println!();

    if k_eff <= 0.0 {
        println!("  ⚠  WARNING: K_eff ≤ 0 — no PMA. Reduce Ku or increase Ms.");
    }

    // ── Grids ────────────────────────────────────────────────────────────

    let base_grid = Grid2D::new(base_nx, base_ny, dx, dy, dz);
    let fine_grid = Grid2D::new(
        fine_nx, fine_ny,
        dx / ref_ratio_total as f64,
        dy / ref_ratio_total as f64,
        dz,
    );

    // ── Material struct ──────────────────────────────────────────────────

    let mat = Material {
        ms,
        a_ex,
        k_u,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: Some(d_dmi),
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let llg = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha,
        dt,
        b_ext: [0.0, 0.0, 0.0],
    };

    // ── Initialise: random magnetisation ─────────────────────────────────

    let mut m_coarse = VectorField2D::new(base_grid);
    initial_states::init_random(&mut m_coarse, seed);

    println!("  Initial state: random (seed={})", seed);
    println!("  Wall cells (|mz|<0.5): {}/{} ({:.1}%)",
        count_wall_cells(&m_coarse, 0.5),
        base_grid.n_cells(),
        count_wall_cells(&m_coarse, 0.5) as f64 / base_grid.n_cells() as f64 * 100.0);
    println!();

    // ── AMR hierarchy ────────────────────────────────────────────────────

    let mut m_coarse_amr = VectorField2D::new(base_grid);
    initial_states::init_random(&mut m_coarse_amr, seed);
    let mut h = AmrHierarchy2D::new(base_grid, m_coarse_amr, refine_ratio, ghost);
    h.set_geom_shape(MaskShape::Full);

    let indicator_kind = IndicatorKind::from_env();
    let regrid_policy = RegridPolicy {
        indicator: indicator_kind,
        buffer_cells: 3,
        boundary_layer: 0,
        min_change_cells: 1,
        min_area_change_frac: 0.02,
    };
    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind,
        buffer_cells: 3,
        boundary_layer: 0,
        min_patch_area: 36,
        merge_distance: 2,
        max_patches: 0,
        connectivity: Connectivity::Eight,
        min_efficiency: 0.60,
        max_flagged_fraction: 0.55,
    };

    // Initial regrid.
    let mut current_patches: Vec<Rect2i> = Vec::new();
    if let Some((new_rects, stats)) =
        maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
    {
        current_patches = new_rects;
        println!("  Initial regrid: {} cells flagged", stats.flagged_cells);
    }

    let (l1, l2, l3) = patch_counts(&h);
    let (_, cov) = compute_coverage(&h, amr_max_level, refine_ratio);
    println!("  Patches:  L1={}, L2={}, L3={}", l1, l2, l3);
    println!("  Coverage: {:.1}%", cov * 100.0);
    println!();

    // ── Write config ─────────────────────────────────────────────────────
    {
        let mut f = File::create(format!("{out_dir}/config.txt")).unwrap();
        writeln!(f, "Labyrinth Benchmark Config").unwrap();
        writeln!(f, "domain: {:.0}nm x {:.0}nm x {:.1}nm", lx*1e9, ly*1e9, dz*1e9).unwrap();
        writeln!(f, "L0: {}x{}, dx={:.2}nm", base_nx, base_ny, dx*1e9).unwrap();
        writeln!(f, "fine: {}x{}, dx={:.3}nm", fine_nx, fine_ny, dx/ref_ratio_total as f64 * 1e9).unwrap();
        writeln!(f, "Ms={:.0} A/m, A={:.0e} J/m, D={:.0e} J/m2, Ku={:.0e} J/m3",
            ms, a_ex, d_dmi, k_u).unwrap();
        writeln!(f, "alpha={}, dt={:.0e}, steps={}, seed={}", alpha, dt, steps_total, seed).unwrap();
        writeln!(f, "K_eff={:.3e}, delta_DW={:.2}nm, l_ex={:.2}nm, D/Dc={:.3}",
            k_eff, delta_dw*1e9, l_ex*1e9, d_dmi/d_c).unwrap();
    }

    // ── CSV headers ──────────────────────────────────────────────────────
    {
        let mut f = File::create(format!("{out_dir}/evolution_log.csv")).unwrap();
        writeln!(f, "step,time_ps,wall_cells_pct,avg_grad_m,L1,L2,L3,coverage_pct,step_ms").unwrap();
    }
    {
        let mut f = File::create(format!("{out_dir}/demag_comparison.csv")).unwrap();
        writeln!(f, "step,rmse_coarse_only_pct,rmse_cfft_amr_pct,rmse_composite_pct,max_b_T").unwrap();
    }

    // ══════════════════════════════════════════════════════════════════════
    //  PHASE 1: LLG Relaxation with AMR tracking
    // ══════════════════════════════════════════════════════════════════════

    println!("  ┌──────────────────────────────────────────────────────────────┐");
    println!("  │  Phase 1: LLG relaxation — random → labyrinth              │");
    println!("  └──────────────────────────────────────────────────────────────┘");
    println!();
    println!("  {:>6}  {:>8}  {:>8}  {:>6}  {:>6}  {:>6}  {:>7}  {:>8}",
        "step", "wall%", "⟨|∇m|⟩", "L1", "L2", "L3", "cov%", "ms/step");
    println!("  {thin}");

    let mut stepper = AmrStepperRK4::new(&h, true);
    let subcycle_active = stepper.is_subcycling();
    let subcycle_ratio: usize = if subcycle_active {
        (stepper.coarse_dt(&llg, &h) / llg.dt).round() as usize
    } else { 1 };

    // Snap step intervals to subcycle ratio.
    let snap = |v: usize, r: usize| -> usize {
        if r <= 1 { v } else { ((v + r - 1) / r) * r }
    };
    let steps = snap(steps_total, subcycle_ratio);
    let out_every = snap(output_interval, subcycle_ratio);
    let regrid_every = snap(regrid_interval, subcycle_ratio);

    if subcycle_active {
        eprintln!("[labyrinth] subcycling: ratio={}, steps adjusted {} → {}",
            subcycle_ratio, steps_total, steps);
    }

    let local_mask = FieldMask::ExchAnis;

    // Save initial state.
    if do_plots {
        let l1r = level_rects(&h, 1);
        let l2r = level_rects(&h, 2);
        let l3r = level_rects(&h, 3);
        save_mz_with_patches(&h.coarse, &l1r, &l2r, &l3r,
            &format!("{out_dir}/mz_step_0000.png"),
            "mz (step 0, random init)")
            .ok();
    }

    let t_evolve_start = Instant::now();

    for step in (subcycle_ratio..=steps).step_by(subcycle_ratio) {
        let t_step = Instant::now();
        stepper.step(&mut h, &llg, &mat, local_mask);
        let step_ms = t_step.elapsed().as_secs_f64() * 1e3;

        // Regrid.
        if step % regrid_every == 0 {
            if let Some((new_rects, _stats)) =
                maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
            {
                current_patches = new_rects;
            }
        }

        // Output.
        if step % out_every == 0 || step == steps {
            let wall_pct = count_wall_cells(&h.coarse, 0.5) as f64
                / base_grid.n_cells() as f64 * 100.0;
            let avg_grad = average_grad_m(&h.coarse);
            let (l1, l2, l3) = patch_counts(&h);
            let (_, cov) = compute_coverage(&h, amr_max_level, refine_ratio);

            println!("  {:>6}  {:>7.1}%  {:>8.2e}  {:>6}  {:>6}  {:>6}  {:>6.1}%  {:>7.1}",
                step, wall_pct, avg_grad, l1, l2, l3, cov * 100.0, step_ms);

            // CSV.
            {
                let mut f = OpenOptions::new().append(true)
                    .open(format!("{out_dir}/evolution_log.csv")).unwrap();
                writeln!(f, "{},{:.4},{:.1},{:.4e},{},{},{},{:.1},{:.1}",
                    step, step as f64 * dt * 1e12,
                    wall_pct, avg_grad, l1, l2, l3, cov * 100.0, step_ms).unwrap();
            }

            // Plots.
            if do_plots {
                let l1r = level_rects(&h, 1);
                let l2r = level_rects(&h, 2);
                let l3r = level_rects(&h, 3);
                save_mz_with_patches(&h.coarse, &l1r, &l2r, &l3r,
                    &format!("{out_dir}/mz_step_{step:04}.png"),
                    &format!("mz step {} + AMR patches", step))
                    .ok();

                // Patch detail plot at an interesting location (centre of domain).
                if step == steps {
                    let ci = base_nx / 2;
                    let cj = base_ny / 2;
                    let he = base_nx.min(base_ny) / 4;
                    save_patch_detail(&h.coarse, &h, ci, cj, he,
                        &format!("{out_dir}/patch_detail_final.png"),
                        "Grid cells near domain centre (final)")
                        .ok();
                }
            }
        }
    }

    let t_evolve = t_evolve_start.elapsed().as_secs_f64();
    println!();
    println!("  Relaxation complete: {:.1} s wall time, {} steps", t_evolve, steps);
    println!();

    // ══════════════════════════════════════════════════════════════════════
    //  PHASE 2: Four-way Demag Comparison (frozen M)
    // ══════════════════════════════════════════════════════════════════════

    println!("  ┌──────────────────────────────────────────────────────────────┐");
    println!("  │  Phase 2: Frozen-M demag comparison (4 solvers)             │");
    println!("  └──────────────────────────────────────────────────────────────┘");
    println!();

    // Solver 1: Fine FFT (reference).
    let (b_fine_opt, t_fine, max_b) = if !skip_fine {
        println!("  [1/4] Fine FFT ({} × {}) ...", fine_nx, fine_ny);
        let mut m_fine = VectorField2D::new(fine_grid);
        // Use the evolved magnetisation, resampled to fine grid.
        let m_composite = flatten_to_fine(&h, fine_grid);
        m_fine.data.clone_from(&m_composite.data);

        let mut b_fine = VectorField2D::new(fine_grid);
        let t0 = Instant::now();
        demag_fft_uniform::compute_demag_field(&fine_grid, &m_fine, &mut b_fine, &mat);
        let t = t0.elapsed().as_secs_f64();
        let max_b = b_fine.data.iter()
            .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
            .fold(0.0f64, f64::max);
        println!("         {:.2} s, max|B| = {:.4e} T", t, max_b);
        (Some(b_fine), t, max_b)
    } else {
        println!("  [1/4] Fine FFT — SKIPPED (--skip-fine-ref)");
        (None, 0.0, 1.0)
    };

    // Solver 2: Coarse FFT only (no AMR, no patches — just L0).
    println!("  [2/4] Coarse FFT only ({} × {}) ...", base_nx, base_ny);
    let mut b_coarse_only = VectorField2D::new(base_grid);
    let t0 = Instant::now();
    demag_fft_uniform::compute_demag_field(&base_grid, &h.coarse, &mut b_coarse_only, &mat);
    let t_coarse_only = t0.elapsed().as_secs_f64();
    println!("         {:.1} ms", t_coarse_only * 1e3);

    // Solver 3: Coarse FFT + AMR (M-restriction from patches, FFT on L0).
    println!("  [3/4] Coarse FFT + AMR ...");
    let mut b_cfft_amr = VectorField2D::new(base_grid);
    let t0 = Instant::now();
    let (_bl1_cfft, _bl2_cfft) = coarse_fft_demag::compute_coarse_fft_demag(
        &h, &mat, &mut b_cfft_amr);
    let t_cfft_amr = t0.elapsed().as_secs_f64();
    println!("         {:.1} ms", t_cfft_amr * 1e3);

    // Solver 4: Composite MG + AMR (V-cycle with fine-resolution B in patches).
    println!("  [4/4] Composite MG + AMR ...");
    let mut b_composite = VectorField2D::new(base_grid);
    let t0 = Instant::now();
    let (_bl1_comp, _bl2_comp) = mg_composite::compute_composite_demag(
        &h, &mat, &mut b_composite);
    let t_composite = t0.elapsed().as_secs_f64();
    println!("         {:.1} ms", t_composite * 1e3);
    println!();

    // ── Compare all solvers against fine FFT reference ────────────────────

    println!("╔{bar}╗");
    println!("║{:^66}║", "Demag Accuracy — Labyrinth Benchmark");
    println!("╚{bar}╝");
    println!();

    if let Some(ref b_fine) = b_fine_opt {
        println!("  (normalised to max|B| = {:.4e} T)", max_b);
        println!();

        // Area-average fine B to coarse for comparison.
        let mut b_fine_avg = VectorField2D::new(base_grid);
        for j in 0..base_ny {
            for i in 0..base_nx {
                let mut avg = [0.0f64; 3];
                for dj in 0..ref_ratio_total {
                    for di in 0..ref_ratio_total {
                        let fi = i * ref_ratio_total + di;
                        let fj = j * ref_ratio_total + dj;
                        if fi < fine_nx && fj < fine_ny {
                            let v = b_fine.data[fj * fine_nx + fi];
                            avg[0] += v[0]; avg[1] += v[1]; avg[2] += v[2];
                        }
                    }
                }
                let n = (ref_ratio_total * ref_ratio_total) as f64;
                avg[0] /= n; avg[1] /= n; avg[2] /= n;
                b_fine_avg.data[j * base_nx + i] = avg;
            }
        }

        // Compute RMSE for each solver: all cells + wall cells only.
        let mut log_all = Vec::new();
        let mut log_wall = Vec::new();

        for (name, b_test) in [
            ("Coarse FFT only", &b_coarse_only),
            ("Coarse FFT + AMR", &b_cfft_amr),
            ("Composite MG+AMR", &b_composite),
        ] {
            let mut sum2_all = 0.0f64;
            let mut sum2_wall = 0.0f64;
            let mut n_all = 0usize;
            let mut n_wall = 0usize;

            for idx in 0..base_grid.n_cells() {
                let ref_b = b_fine_avg.data[idx];
                let test_b = b_test.data[idx];
                let e2 = (ref_b[0]-test_b[0]).powi(2)
                    + (ref_b[1]-test_b[1]).powi(2)
                    + (ref_b[2]-test_b[2]).powi(2);
                sum2_all += e2;
                n_all += 1;

                if h.coarse.data[idx][2].abs() < 0.5 {
                    sum2_wall += e2;
                    n_wall += 1;
                }
            }

            let rmse_all = (sum2_all / n_all as f64).sqrt() / max_b * 100.0;
            let rmse_wall = if n_wall > 0 {
                (sum2_wall / n_wall as f64).sqrt() / max_b * 100.0
            } else { 0.0 };

            log_all.push((name, rmse_all));
            log_wall.push((name, rmse_wall, n_wall));
        }

        println!("  ALL CELLS ({} coarse cells):", base_grid.n_cells());
        for (name, rmse) in &log_all {
            println!("    {:<22} RMSE: {:.2}%", name, rmse);
        }
        println!();

        let n_wall = log_wall[0].2;
        println!("  WALL CELLS ({} cells where |mz| < 0.5):", n_wall);
        for (name, rmse, _) in &log_wall {
            println!("    {:<22} RMSE: {:.2}%", name, rmse);
        }
        println!();

        // Write to CSV.
        {
            let mut f = OpenOptions::new().append(true)
                .open(format!("{out_dir}/demag_comparison.csv")).unwrap();
            writeln!(f, "{},{:.4},{:.4},{:.4},{:.6e}",
                steps,
                log_all[0].1, log_all[1].1, log_all[2].1,
                max_b).unwrap();
        }

        // Bx profile cut through middle row.
        if do_plots {
            let mid_j = base_ny / 2;
            let x_coords: Vec<f64> = (0..base_nx)
                .map(|i| {
                    let (x, _) = cell_center_xy_centered(&base_grid, i, mid_j);
                    x * 1e9
                })
                .collect();
            let bx_fine: Vec<f64> = (0..base_nx)
                .map(|i| b_fine_avg.data[mid_j * base_nx + i][0]).collect();
            let bx_coarse: Vec<f64> = (0..base_nx)
                .map(|i| b_coarse_only.data[mid_j * base_nx + i][0]).collect();
            let bx_cfft: Vec<f64> = (0..base_nx)
                .map(|i| b_cfft_amr.data[mid_j * base_nx + i][0]).collect();
            let bx_comp: Vec<f64> = (0..base_nx)
                .map(|i| b_composite.data[mid_j * base_nx + i][0]).collect();

            save_bx_profile(
                &[
                    ("Fine FFT (ref)", &bx_fine, RGBColor(0, 0, 0)),
                    ("Coarse only", &bx_coarse, RGBColor(200, 50, 50)),
                    ("Coarse FFT+AMR", &bx_cfft, RGBColor(50, 150, 50)),
                    ("Composite MG+AMR", &bx_comp, RGBColor(50, 50, 200)),
                ],
                &x_coords,
                &format!("{out_dir}/bx_profile_mid.png"),
                "Bx profile through domain centre (y = Ly/2)",
            ).ok();
            println!("  Wrote {out_dir}/bx_profile_mid.png");
        }
    } else {
        println!("  (Fine reference skipped — accuracy comparison not available)");
    }

    // ── Timing summary ───────────────────────────────────────────────────

    println!();
    println!("  TIMING SUMMARY");
    println!("  {thin}");
    println!("    Fine FFT:          {:>10.1} ms", t_fine * 1e3);
    println!("    Coarse FFT only:   {:>10.1} ms", t_coarse_only * 1e3);
    println!("    Coarse FFT + AMR:  {:>10.1} ms", t_cfft_amr * 1e3);
    println!("    Composite MG+AMR:  {:>10.1} ms", t_composite * 1e3);
    if t_fine > 0.0 {
        println!();
        println!("    Speedup (composite vs fine): {:.0}×", t_fine / t_composite);
        println!("    Speedup (cfft+amr vs fine):  {:.0}×", t_fine / t_cfft_amr);
    }
    println!();

    // ── Write summary ────────────────────────────────────────────────────
    {
        let mut f = File::create(format!("{out_dir}/summary.txt")).unwrap();
        writeln!(f, "Labyrinth Benchmark — Summary").unwrap();
        writeln!(f, "Domain: {:.0}nm x {:.0}nm x {:.1}nm", lx*1e9, ly*1e9, dz*1e9).unwrap();
        writeln!(f, "L0: {}x{}, Fine: {}x{}", base_nx, base_ny, fine_nx, fine_ny).unwrap();
        writeln!(f, "Steps: {}, alpha={}, dt={:.0e}", steps, alpha, dt).unwrap();
        writeln!(f, "").unwrap();
        let (l1, l2, l3) = patch_counts(&h);
        let (_, cov) = compute_coverage(&h, amr_max_level, refine_ratio);
        writeln!(f, "Final patches: L1={} L2={} L3={}", l1, l2, l3).unwrap();
        writeln!(f, "Final coverage: {:.1}%", cov * 100.0).unwrap();
        writeln!(f, "Final wall cells: {:.1}%",
            count_wall_cells(&h.coarse, 0.5) as f64 / base_grid.n_cells() as f64 * 100.0).unwrap();
        writeln!(f, "").unwrap();
        writeln!(f, "Timing:").unwrap();
        writeln!(f, "  fine_fft={:.1}ms  coarse_only={:.1}ms  cfft_amr={:.1}ms  composite={:.1}ms",
            t_fine*1e3, t_coarse_only*1e3, t_cfft_amr*1e3, t_composite*1e3).unwrap();
        if t_fine > 0.0 {
            writeln!(f, "  speedup_composite={:.0}x  speedup_cfft_amr={:.0}x",
                t_fine/t_composite, t_fine/t_cfft_amr).unwrap();
        }
        writeln!(f, "").unwrap();
        writeln!(f, "Evolution: {:.1}s ({} steps)", t_evolve, steps).unwrap();
    }

    // ── Final plots ──────────────────────────────────────────────────────

    if do_plots {
        println!("  Generating final plots ...");
        let l1r = level_rects(&h, 1);
        let l2r = level_rects(&h, 2);
        let l3r = level_rects(&h, 3);

        // Hero image: final labyrinth with patches.
        save_mz_with_patches(&h.coarse, &l1r, &l2r, &l3r,
            &format!("{out_dir}/mz_final_patches.png"),
            &format!("Labyrinth ({}×{}) + AMR patches", base_nx, base_ny))
            .ok();
        println!("    Wrote {out_dir}/mz_final_patches.png");

        // Bare mz (no patches) for clean comparison.
        save_mz_with_patches(&h.coarse, &[], &[], &[],
            &format!("{out_dir}/mz_final_bare.png"),
            &format!("mz — labyrinth equilibrium ({}×{})", base_nx, base_ny))
            .ok();
        println!("    Wrote {out_dir}/mz_final_bare.png");
    }

    println!();
    println!("  Output directory: {out_dir}/");
    println!("  Total wall time:  {:.1} s", total_t0.elapsed().as_secs_f64());
    println!();
}
