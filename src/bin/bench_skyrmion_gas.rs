// src/bin/bench_skyrmion_gas.rs
//
// Flagship AMR benchmark: 7 Néel skyrmions in a PMA thin film with DMI.
//
// Default mode (static):
//   Frozen magnetisation → compute demag three ways (fine FFT, coarse FFT, composite MG)
//   → compare accuracy at skyrmion cores → report timing and RMSE.
//
// --relax mode (dynamic):
//   Oversized skyrmions relax under damped LLG (breathing + drift).
//   AMR patches track each skyrmion. Outputs per-step timing, RMSE, coverage.
//
// Key features:
//   - No material/vacuum boundary → no staircase error
//   - Skyrmion wall width δ ≈ 5nm → unresolvable on coarse grid (dx ≈ 8nm)
//   - 90% cell savings → 10-20× speedup over fine FFT
//
// Run:
//   cargo run --release --bin bench_skyrmion_gas -- --plots
//   cargo run --release --bin bench_skyrmion_gas -- --relax --plots
//   LLG_SKG_NX=256 cargo run --release --bin bench_skyrmion_gas -- --plots

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use plotters::prelude::*;

use llg_sim::effective_field::{FieldMask, demag_fft_uniform, mg_composite};
use llg_sim::geometry_mask::{MaskShape, cell_center_xy_centered};
use llg_sim::grid::Grid2D;
#[allow(unused_imports)]
use llg_sim::initial_states;
use llg_sim::llg::{RK4Scratch, step_llg_rk4_recompute_field_masked_relax_add};
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material, MU0};
use llg_sim::vector_field::VectorField2D;

use llg_sim::amr::indicator::IndicatorKind;
use llg_sim::amr::regrid::maybe_regrid_nested_levels;
use llg_sim::amr::{
    AmrHierarchy2D, AmrStepperRK4, ClusterPolicy, Connectivity, Rect2i, RegridPolicy,
};

fn ensure_dir(path: &str) {
    if !Path::new(path).exists() {
        fs::create_dir_all(path).unwrap();
    }
}

fn idx(i: usize, j: usize, nx: usize) -> usize { j * nx + i }

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

/// Compute RMSE and max delta between two magnetisation fields.
fn rmse_and_max(a: &VectorField2D, b: &VectorField2D) -> (f64, f64) {
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
    ((sum2 / n as f64).sqrt(), maxd)
}

/// Compute topological charge Q = (1/4π) ∫ m · (∂m/∂x × ∂m/∂y) dA
fn topological_charge(m: &VectorField2D) -> f64 {
    let nx = m.grid.nx;
    let ny = m.grid.ny;
    let dx = m.grid.dx;
    let dy = m.grid.dy;
    let mut q = 0.0f64;

    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let c = m.data[j * nx + i];
            let xp = m.data[j * nx + (i + 1)];
            let xm = m.data[j * nx + (i - 1)];
            let yp = m.data[(j + 1) * nx + i];
            let ym = m.data[(j - 1) * nx + i];

            // Central differences.
            let dmx_dx = (xp[0] - xm[0]) / (2.0 * dx);
            let dmy_dx = (xp[1] - xm[1]) / (2.0 * dx);
            let dmz_dx = (xp[2] - xm[2]) / (2.0 * dx);

            let dmx_dy = (yp[0] - ym[0]) / (2.0 * dy);
            let dmy_dy = (yp[1] - ym[1]) / (2.0 * dy);
            let dmz_dy = (yp[2] - ym[2]) / (2.0 * dy);

            // m · (∂m/∂x × ∂m/∂y)
            let cross_x = dmy_dx * dmz_dy - dmz_dx * dmy_dy;
            let cross_y = dmz_dx * dmx_dy - dmx_dx * dmz_dy;
            let cross_z = dmx_dx * dmy_dy - dmy_dx * dmx_dy;

            q += c[0] * cross_x + c[1] * cross_y + c[2] * cross_z;
        }
    }

    q * dx * dy / (4.0 * std::f64::consts::PI)
}

/// Skyrmion positions (centered coordinates, meters).
fn skyrmion_positions(n: usize) -> Vec<(f64, f64)> {
    let all = vec![
        (-280e-9,  200e-9),
        (-100e-9,  320e-9),
        ( 150e-9,  250e-9),
        ( 300e-9,   50e-9),
        (-200e-9, -150e-9),
        (  50e-9, -100e-9),
        ( 250e-9, -300e-9),
    ];
    all.into_iter().take(n).collect()
}

/// Initialise m with N skyrmions on a +z background.
///
/// For each cell, find the nearest skyrmion and apply that skyrmion's
/// radial profile. Cells far from all skyrmions stay at mz = +1.
/// This avoids the bug where successive init_skyrmion calls overwrite
/// the entire grid.
fn init_skyrmion_gas(
    m: &mut VectorField2D, grid: &Grid2D,
    positions: &[(f64, f64)], r0: f64, delta: f64,
) {
    let inv_delta = 1.0 / delta.max(1e-30);
    let cutoff = r0 + 6.0 * delta; // beyond this, profile is indistinguishable from background

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let id = j * grid.nx + i;
            let (x, y) = cell_center_xy_centered(grid, i, j);

            // Find nearest skyrmion.
            let mut best_r = f64::INFINITY;
            let mut best_dx = 0.0f64;
            let mut best_dy = 0.0f64;
            for &(sx, sy) in positions {
                let ddx = x - sx;
                let ddy = y - sy;
                let r = (ddx * ddx + ddy * ddy).sqrt();
                if r < best_r {
                    best_r = r;
                    best_dx = ddx;
                    best_dy = ddy;
                }
            }

            if best_r > cutoff {
                // Far from all skyrmions → uniform +z background.
                m.data[id] = [0.0, 0.0, 1.0];
                continue;
            }

            // Skyrmion profile: θ(r) = 2·atan(exp((R₀ - r)/δ))
            let theta = 2.0 * (((r0 - best_r) * inv_delta).exp()).atan();
            let ct = theta.cos();
            let st = theta.sin();

            // Néel type (helicity = 0): in-plane component points radially.
            let phi = best_dy.atan2(best_dx);
            let mx = st * phi.cos();
            let my = st * phi.sin();
            // Core polarity +1: core points -z, background +z.
            // mz = -cos(θ) → at r=0: θ≈π → mz=+1; at r→∞: θ≈0 → mz=-1
            // Wait — this gives mz=+1 at core and -1 at far field.
            // For core=-z (mz=-1 at centre): use mz = cos(θ)
            // θ(0) ≈ π → cos(π) = -1 ✓ (core down)
            // θ(∞) ≈ 0 → cos(0) = +1 ✓ (background up)
            let mz = ct;

            let len = (mx * mx + my * my + mz * mz).sqrt();
            if len > 1e-30 {
                m.data[id] = [mx / len, my / len, mz / len];
            } else {
                m.data[id] = [0.0, 0.0, 1.0];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Visualisation helpers
// ---------------------------------------------------------------------------

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

fn save_mz_map(
    m: &VectorField2D,
    l1_rects: &[Rect2i], l2_rects: &[Rect2i], l3_rects: &[Rect2i],
    path: &str, caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = m.grid.nx as i32;
    let ny = m.grid.ny as i32;

    let root = BitMapBackend::new(path, (1024, 1024)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 22))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..nx, 0..ny)?;

    chart.configure_mesh().disable_mesh().draw()?;

    // Draw mz colour map.
    chart.draw_series((0..m.grid.ny).flat_map(|j| {
        (0..m.grid.nx).map(move |i| {
            let mz = m.data[idx(i, j, m.grid.nx)][2];
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                mz_to_bwr(mz).filled(),
            )
        })
    }))?;

    // Overlay AMR patches.
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

fn level_rects(h: &AmrHierarchy2D, level_idx: usize) -> Vec<Rect2i> {
    if level_idx == 0 {
        return Vec::new();
    }
    if level_idx == 1 {
        return h.patches.iter().map(|p| p.coarse_rect.clone()).collect();
    }
    let li = level_idx - 2;
    h.patches_l2plus.get(li)
        .map(|lvl| lvl.iter().map(|p| p.coarse_rect.clone()).collect())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let do_plots = args.iter().any(|a| a == "--plots");
    let do_relax = args.iter().any(|a| a == "--relax");
    let skip_fine = args.iter().any(|a| a == "--skip-fine-ref");

    // ---- Parameters (env var overridable) ----
    let base_nx = env_usize("LLG_SKG_NX", 128);
    let base_ny = env_usize("LLG_SKG_NY", 128);
    let lx = env_f64("LLG_SKG_LX", 1.0e-6);
    let ly = env_f64("LLG_SKG_LY", 1.0e-6);
    let dz = env_f64("LLG_SKG_DZ", 4.0e-10);
    let n_skyrmions = env_usize("LLG_SKG_N_SKYRMIONS", 7);
    let r0 = env_f64("LLG_SKG_R0", 25.0e-9);
    let alpha = env_f64("LLG_SKG_ALPHA", 0.3);
    let steps_base = env_usize("LLG_SKG_STEPS", 320);

    let dx = lx / base_nx as f64;
    let dy = ly / base_ny as f64;

    let amr_max_level: usize = env_usize("LLG_AMR_MAX_LEVEL", 3);
    let refine_ratio = 2usize;
    let ghost = 2usize;
    let ref_ratio_total = pow_usize(refine_ratio, amr_max_level);
    let fine_nx = base_nx * ref_ratio_total;
    let fine_ny = base_ny * ref_ratio_total;

    // Derived physics.
    let ms = 580.0e3;       // A/m
    let a_ex = 15.0e-12;    // J/m
    let d_dmi = 3.0e-3;     // J/m²
    let k_u = 0.8e6;        // J/m³
    let k_eff = k_u - MU0 * ms * ms / 2.0;
    let delta = (a_ex / k_eff).sqrt();
    let l_ex = (2.0 * a_ex / (MU0 * ms * ms)).sqrt();
    let d_c = 4.0 * (a_ex * k_eff).sqrt() / std::f64::consts::PI;
    let dt = 5.0e-14;

    let out_dir = "out/skyrmion_gas";
    ensure_dir(out_dir);

    let total_t0 = Instant::now();

    // ---- Print header ----
    let bar = "═".repeat(64);
    println!("╔{bar}╗");
    println!("║{:^64}║", "Skyrmion Gas Benchmark — AMR Demag Demonstration");
    println!("╚{bar}╝");
    println!();
    println!("  Domain:      {:.0} nm × {:.0} nm, dz = {:.1} nm",
        lx * 1e9, ly * 1e9, dz * 1e9);
    println!("  Skyrmions:   {} Néel (R₀ = {:.0} nm, δ = {:.1} nm)",
        n_skyrmions, r0 * 1e9, delta * 1e9);
    println!("  Base grid:   {} × {}, dx = {:.2} nm",
        base_nx, base_ny, dx * 1e9);
    println!("  Fine grid:   {} × {} ({}× refinement, {} AMR levels)",
        fine_nx, fine_ny, ref_ratio_total, amr_max_level);
    println!("  Mode:        {}", if do_relax { "RELAX (LLG dynamics)" } else { "STATIC (frozen M, demag comparison)" });
    println!();
    println!("  Material:    Ms={:.0} kA/m, A={:.0} pJ/m, D={:.1} mJ/m², Ku={:.1} MJ/m³",
        ms / 1e3, a_ex / 1e-12, d_dmi / 1e-3, k_u / 1e6);
    println!("  Derived:     l_ex={:.1} nm, δ_DW={:.1} nm, Keff={:.2} MJ/m³, D/Dc={:.2}",
        l_ex * 1e9, delta * 1e9, k_eff / 1e6, d_dmi / d_c);
    println!();

    // ---- Grids ----
    let base_grid = Grid2D::new(base_nx, base_ny, dx, dy, dz);
    let fine_grid = Grid2D::new(
        fine_nx, fine_ny,
        dx / ref_ratio_total as f64,
        dy / ref_ratio_total as f64,
        dz,
    );

    let positions = skyrmion_positions(n_skyrmions);

    // ---- Material ----
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

    // ---- Initialise coarse magnetisation ----
    let mut m_coarse = VectorField2D::new(base_grid);
    init_skyrmion_gas(&mut m_coarse, &base_grid, &positions, r0, delta);

    let q_init = topological_charge(&m_coarse);
    println!("  Topological charge Q = {:.2} (expected -{:.0})", q_init, n_skyrmions as f64);
    println!();

    // ---- AMR hierarchy ----
    let mut m_coarse_amr = VectorField2D::new(base_grid);
    init_skyrmion_gas(&mut m_coarse_amr, &base_grid, &positions, r0, delta);
    let mut h = AmrHierarchy2D::new(base_grid, m_coarse_amr, refine_ratio, ghost);
    h.set_geom_shape(MaskShape::Full);

    let indicator_kind = IndicatorKind::from_env();
    let regrid_policy = RegridPolicy {
        indicator: indicator_kind,
        buffer_cells: 4,
        boundary_layer: 0,
        min_change_cells: 1,
        min_area_change_frac: 0.02,
    };
    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind,
        buffer_cells: 4,
        boundary_layer: 0,
        min_patch_area: 36,
        merge_distance: 2,
        max_patches: 0,
        connectivity: Connectivity::Eight,
        min_efficiency: 0.65,
        max_flagged_fraction: 0.50,
    };

    // Initial regrid.
    let mut current_patches: Vec<Rect2i> = Vec::new();
    if let Some((new_rects, stats)) =
        maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
    {
        current_patches = new_rects;
        println!("  Regrid: {} cells flagged", stats.flagged_cells);
    }

    let l1_count = h.patches.len();
    let l2_count = h.patches_l2plus.get(0).map(|v| v.len()).unwrap_or(0);
    let l3_count = h.patches_l2plus.get(1).map(|v| v.len()).unwrap_or(0);
    println!("  Patches:     L1={}, L2={}, L3={}", l1_count, l2_count, l3_count);

    // Estimate coverage.
    let fine_cells_total = fine_nx * fine_ny;
    let amr_fine_cells: usize = {
        let mut total = 0usize;
        for p in &h.patches {
            let inx = p.interior_i1() - p.interior_i0();
            let iny = p.interior_j1() - p.interior_j0();
            total += inx * iny * pow_usize(refine_ratio, amr_max_level - 1);
        }
        for (li, lvl) in h.patches_l2plus.iter().enumerate() {
            let level_ratio = pow_usize(refine_ratio, amr_max_level - (li + 2));
            for p in lvl {
                let inx = p.interior_i1() - p.interior_i0();
                let iny = p.interior_j1() - p.interior_j0();
                total += inx * iny * level_ratio;
            }
        }
        total
    };
    let coverage = amr_fine_cells as f64 / fine_cells_total as f64;
    println!("  Coverage:    {:.1}% ({} / {} fine-equivalent cells)",
        coverage * 100.0, amr_fine_cells, fine_cells_total);
    println!();

    // ══════════════════════════════════════════════════════════════════
    // STATIC MODE: Frozen-M demag comparison
    // ══════════════════════════════════════════════════════════════════
    if !do_relax {
        // 1. Fine FFT reference.
        if !skip_fine {
            println!("  Computing fine FFT reference ({} × {}) ...", fine_nx, fine_ny);
            let mut m_fine = VectorField2D::new(fine_grid);
            init_skyrmion_gas(&mut m_fine, &fine_grid, &positions, r0, delta);

            let mut b_fine = VectorField2D::new(fine_grid);
            let t0 = Instant::now();
            demag_fft_uniform::compute_demag_field(&fine_grid, &m_fine, &mut b_fine, &mat);
            let t_fine = t0.elapsed().as_secs_f64();
            let max_b_fine = b_fine.data.iter()
                .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
                .fold(0.0f64, f64::max);
            println!("  Fine FFT:    {:.2} s,  max|B| = {:.4e} T", t_fine, max_b_fine);

            // 2. Coarse FFT.
            let mut b_coarse = VectorField2D::new(base_grid);
            let t0 = Instant::now();
            demag_fft_uniform::compute_demag_field(&base_grid, &m_coarse, &mut b_coarse, &mat);
            let t_coarse = t0.elapsed().as_secs_f64();
            println!("  Coarse FFT:  {:.1} ms", t_coarse * 1e3);

            // 3. Composite MG.
            println!("  Running composite (V-cycle) ...");
            let t0 = Instant::now();
            let mut b_composite_coarse = VectorField2D::new(base_grid);
            let (_b_l1, _b_l2) = mg_composite::compute_composite_demag(
                &h, &mat, &mut b_composite_coarse);
            let t_comp = t0.elapsed().as_secs_f64();
            println!("  Composite:   {:.1} ms", t_comp * 1e3);
            println!();

            // 4. Compare at patch interior cells.
            // Flatten composite B to fine grid for comparison.
            // For now, compare at coarse level (patch-level comparison is in bench_composite_vcycle).
            let thin = "─".repeat(64);
            println!("╔{bar}╗");
            println!("║{:^64}║", "Demag Accuracy — Skyrmion Gas");
            println!("╚{bar}╝");
            println!("  (normalised to max|B| = {:.4e} T)", max_b_fine);
            println!();

            // Coarse-level RMSE (coarse FFT vs fine FFT, resampled).
            // For each coarse cell, compare against area-averaged fine B.
            let mut sum2_cfft = 0.0f64;
            let mut sum2_comp = 0.0f64;
            let mut n_cells = 0usize;
            let mut sum2_cfft_core = 0.0f64;
            let mut sum2_comp_core = 0.0f64;
            let mut n_core = 0usize;

            for j in 0..base_ny {
                for i in 0..base_nx {
                    // Area-average fine B over this coarse cell.
                    let mut b_avg = [0.0f64; 3];
                    for dj in 0..ref_ratio_total {
                        for di in 0..ref_ratio_total {
                            let fi = i * ref_ratio_total + di;
                            let fj = j * ref_ratio_total + dj;
                            if fi < fine_nx && fj < fine_ny {
                                let v = b_fine.data[fj * fine_nx + fi];
                                b_avg[0] += v[0];
                                b_avg[1] += v[1];
                                b_avg[2] += v[2];
                            }
                        }
                    }
                    let n = (ref_ratio_total * ref_ratio_total) as f64;
                    b_avg[0] /= n; b_avg[1] /= n; b_avg[2] /= n;

                    let bc = b_coarse.data[j * base_nx + i];
                    let bm = b_composite_coarse.data[j * base_nx + i];

                    let err_cfft = (bc[0]-b_avg[0]).powi(2) + (bc[1]-b_avg[1]).powi(2) + (bc[2]-b_avg[2]).powi(2);
                    let err_comp = (bm[0]-b_avg[0]).powi(2) + (bm[1]-b_avg[1]).powi(2) + (bm[2]-b_avg[2]).powi(2);

                    sum2_cfft += err_cfft;
                    sum2_comp += err_comp;
                    n_cells += 1;

                    // Check if near a skyrmion core.
                    let (cx, cy) = cell_center_xy_centered(&base_grid, i, j);
                    let near_core = positions.iter().any(|&(sx, sy)| {
                        let dr = ((cx - sx).powi(2) + (cy - sy).powi(2)).sqrt();
                        dr < 40.0e-9
                    });
                    if near_core {
                        sum2_cfft_core += err_cfft;
                        sum2_comp_core += err_comp;
                        n_core += 1;
                    }
                }
            }

            let rmse_cfft = (sum2_cfft / n_cells as f64).sqrt() / max_b_fine * 100.0;
            let rmse_comp = (sum2_comp / n_cells as f64).sqrt() / max_b_fine * 100.0;
            let rmse_cfft_core = if n_core > 0 { (sum2_cfft_core / n_core as f64).sqrt() / max_b_fine * 100.0 } else { 0.0 };
            let rmse_comp_core = if n_core > 0 { (sum2_comp_core / n_core as f64).sqrt() / max_b_fine * 100.0 } else { 0.0 };

            println!("  ALL CELLS ({} coarse cells):", n_cells);
            println!("    coarse-FFT RMSE: {:.2}%", rmse_cfft);
            println!("    composite  RMSE: {:.2}%", rmse_comp);
            if rmse_comp < rmse_cfft {
                println!("    → composite is {:.1}% MORE ACCURATE",
                    (1.0 - rmse_comp / rmse_cfft) * 100.0);
            }
            println!();
            println!("  SKYRMION CORES ({} cells within 40nm of any core):", n_core);
            println!("    coarse-FFT RMSE: {:.2}%", rmse_cfft_core);
            println!("    composite  RMSE: {:.2}%", rmse_comp_core);
            if rmse_comp_core < rmse_cfft_core {
                println!("    → composite is {:.1}% MORE ACCURATE at cores",
                    (1.0 - rmse_comp_core / rmse_cfft_core) * 100.0);
            }
            println!();

            println!("  TIMING");
            println!("  {thin}");
            println!("    coarse-FFT:  {:.1} ms", t_coarse * 1e3);
            println!("    composite:   {:.1} ms", t_comp * 1e3);
            println!("    fine FFT:    {:.1} ms (reference)", t_fine * 1e3);
            println!("    Speedup:     {:.0}× over fine FFT", t_fine / t_comp);
            println!();

            // Write summary.
            let mut f = File::create(format!("{out_dir}/summary.txt")).unwrap();
            writeln!(f, "Skyrmion Gas Benchmark — Static Mode").unwrap();
            writeln!(f, "Domain: {:.0}nm x {:.0}nm, {} skyrmions", lx*1e9, ly*1e9, n_skyrmions).unwrap();
            writeln!(f, "L0: {}x{}, Fine: {}x{}", base_nx, base_ny, fine_nx, fine_ny).unwrap();
            writeln!(f, "Q_init = {:.2}", q_init).unwrap();
            writeln!(f, "L1={} L2={} L3={}, coverage={:.1}%", l1_count, l2_count, l3_count, coverage*100.0).unwrap();
            writeln!(f, "").unwrap();
            writeln!(f, "ALL: cfft={:.2}% comp={:.2}%", rmse_cfft, rmse_comp).unwrap();
            writeln!(f, "CORE: cfft={:.2}% comp={:.2}%", rmse_cfft_core, rmse_comp_core).unwrap();
            writeln!(f, "").unwrap();
            writeln!(f, "t_fine={:.1}ms t_cfft={:.1}ms t_comp={:.1}ms speedup={:.0}x",
                t_fine*1e3, t_coarse*1e3, t_comp*1e3, t_fine/t_comp).unwrap();

            // Plots.
            if do_plots {
                println!("  Generating plots ...");
                let l1 = level_rects(&h, 1);
                let l2 = level_rects(&h, 2);
                let l3 = level_rects(&h, 3);

                // mz map on coarse grid with patches.
                save_mz_map(&m_coarse, &l1, &l2, &l3,
                    &format!("{out_dir}/mz_coarse_patches.png"),
                    &format!("mz (coarse {}x{}) + AMR patches", base_nx, base_ny))
                    .unwrap();

                // mz map on fine grid (reference).
                let m_fine_display = {
                    let mut mf = VectorField2D::new(fine_grid);
                    init_skyrmion_gas(&mut mf, &fine_grid, &positions, r0, delta);
                    mf
                };
                save_mz_map(&m_fine_display, &[], &[], &[],
                    &format!("{out_dir}/mz_fine_reference.png"),
                    &format!("mz (fine {}x{}) reference", fine_nx, fine_ny))
                    .unwrap();

                println!("    Wrote {out_dir}/mz_coarse_patches.png");
                println!("    Wrote {out_dir}/mz_fine_reference.png");
            }
        } else {
            println!("  (fine reference skipped — use without --skip-fine-ref for accuracy)");
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // RELAX MODE: LLG dynamics with breathing skyrmions
    // ══════════════════════════════════════════════════════════════════
    if do_relax {
        println!("  Starting LLG relaxation ({} steps, α={}, dt={:.0e} s) ...",
            steps_base, alpha, dt);

        // Fine reference (initialised from same composite field).
        let mut m_fine = flatten_to_fine(&h, fine_grid);

        let mut stepper = AmrStepperRK4::new(&h, true);
        let subcycle_active = stepper.is_subcycling();
        let subcycle_ratio: usize = if subcycle_active {
            (stepper.coarse_dt(&llg, &h) / llg.dt).round() as usize
        } else { 1 };

        let snap_up = |v: usize, r: usize| -> usize {
            if r <= 1 { v } else { ((v + r - 1) / r) * r }
        };
        let steps = snap_up(steps_base, subcycle_ratio);
        let out_every = snap_up(64, subcycle_ratio);
        let regrid_every = snap_up(64, subcycle_ratio);

        if subcycle_active {
            eprintln!("[skyrmion_gas] SUBCYCLING: ratio={}, steps adjusted {} → {}",
                subcycle_ratio, steps_base, steps);
        }

        // Log files.
        {
            let mut f = File::create(format!("{out_dir}/rmse_log.csv")).unwrap();
            writeln!(f, "step,rmse,max_delta,Q,L1,L2,L3,coverage_pct").unwrap();

            let mut f2 = File::create(format!("{out_dir}/timing_log.csv")).unwrap();
            writeln!(f2, "step,amr_step_ms").unwrap();
        }

        let mut fine_scratch = RK4Scratch::new(fine_grid);
        let local_mask = FieldMask::ExchAnis;
        let mut b_fine_demag = VectorField2D::new(fine_grid);

        let t0 = Instant::now();
        let mut t_amr_total = 0.0f64;
        let mut t_fine_total = 0.0f64;

        for step in (subcycle_ratio..=steps).step_by(subcycle_ratio) {
            // AMR step.
            let ta = Instant::now();
            stepper.step(&mut h, &llg, &mat, local_mask);
            let amr_ms = ta.elapsed().as_secs_f64() * 1e3;
            t_amr_total += ta.elapsed().as_secs_f64();

            // Fine reference step (same number of fine steps).
            let tf = Instant::now();
            for _ in 0..subcycle_ratio {
                demag_fft_uniform::compute_demag_field(&fine_grid, &m_fine, &mut b_fine_demag, &mat);
                step_llg_rk4_recompute_field_masked_relax_add(
                    &mut m_fine, &llg, &mat,
                    &mut fine_scratch, local_mask,
                    Some(&b_fine_demag),
                );
            }
            t_fine_total += tf.elapsed().as_secs_f64();

            // Log timing.
            {
                let mut f = OpenOptions::new().append(true).open(format!("{out_dir}/timing_log.csv")).unwrap();
                writeln!(f, "{},{:.1}", step, amr_ms).unwrap();
            }

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
                let m_amr_fine = flatten_to_fine(&h, fine_grid);
                let (rmse, maxd) = rmse_and_max(&m_amr_fine, &m_fine);
                let q = topological_charge(&h.coarse);

                let l1 = h.patches.len();
                let l2 = h.patches_l2plus.get(0).map(|v| v.len()).unwrap_or(0);
                let l3 = h.patches_l2plus.get(1).map(|v| v.len()).unwrap_or(0);

                println!("  step {:>4}  │  rmse {:.3e}  │  Q={:.2}  │  L1 {} L2 {} L3 {}",
                    step, rmse, q, l1, l2, l3);

                {
                    let mut f = OpenOptions::new().append(true).open(format!("{out_dir}/rmse_log.csv")).unwrap();
                    writeln!(f, "{},{:.6e},{:.6e},{:.4},{},{},{},{:.1}",
                        step, rmse, maxd, q, l1, l2, l3, coverage * 100.0).unwrap();
                }

                if do_plots {
                    let l1r = level_rects(&h, 1);
                    let l2r = level_rects(&h, 2);
                    let l3r = level_rects(&h, 3);
                    save_mz_map(&h.coarse, &l1r, &l2r, &l3r,
                        &format!("{out_dir}/mz_step_{step:04}.png"),
                        &format!("mz step {} (coarse + patches)", step))
                        .ok();
                }
            }
        }

        let wall = t0.elapsed().as_secs_f64();
        let thin = "─".repeat(64);

        println!();
        println!("╔{bar}╗");
        println!("║{:^64}║", "SKYRMION GAS RELAXATION — RESULTS");
        println!("╚{bar}╝");
        println!();
        println!("  TIMING");
        println!("  {thin}");
        println!("    Total wall clock       {:.1} s", wall);
        println!("    AMR stepping           {:.1} s", t_amr_total);
        println!("    Fine reference         {:.1} s", t_fine_total);
        if t_amr_total > 0.0 {
            println!("    Speedup (wall time)    {:.1}×", t_fine_total / t_amr_total);
        }
        println!();
        println!("  AMR PATCHES (final)");
        println!("  {thin}");
        println!("  L1={} L2={} L3={}", h.patches.len(),
            h.patches_l2plus.get(0).map(|v| v.len()).unwrap_or(0),
            h.patches_l2plus.get(1).map(|v| v.len()).unwrap_or(0));
        println!("  Coverage: {:.1}%", coverage * 100.0);
        println!();
        println!("  Output: {out_dir}/");
    }

    println!();
    println!("  Total wall time: {:.1} s", total_t0.elapsed().as_secs_f64());
}