// src/bin/bench_antidot.rs
//
// Anti-dot demag benchmark — Phase 2 of the P-FFT implementation plan
// ====================================================================
//
// PURPOSE: Demonstrate that (a) uniform fine FFT is accurate but slow,
// (b) coarse-grid FFT + AMR is fast but inaccurate at geometric boundaries,
// and therefore (c) P-FFT edge correction (Phase 3) is needed.
//
// The benchmark compares THREE classes of solver on a hexagonal anti-dot
// array in Permalloy, using a fine-grid FFT as the gold reference:
//
//   Method 1 — fft_fine:     FFT on the fine-equivalent grid (1024² for
//              L0=512).  Slow (~150ms) but accurate: resolves hole boundaries
//              at fine dx and captures the full near-field B_demag structure.
//              This is the reference all others are compared against.
//
//   Method 2 — fft_l0:       FFT on the L0 grid (512²).  Faster (~25ms) but
//              uses the coarse staircase boundary.  Shows ~18% edge RMSE from
//              staircase alone — the "best you can do" without fine resolution.
//
//   Method 3 — coarse_fft:   AMR hierarchy + M-restriction + super-coarse FFT
//              (R=1,2,4).  Designed for dynamic AMR problems (like vortex relax)
//              where patches carry real texture.  For the static saturated state
//              here, masked restriction preserves the binary staircase exactly,
//              so coarse_fft R=1 ≈ fft_l0 plus AMR overhead.
//              At R>1, the L0→demag_grid restriction (UN-masked) smooths the
//              staircase, which paradoxically reduces edge RMSE at R=2 but
//              corrupts the bulk.  This is NOT a real improvement — P-FFT will
//              provide true edge correction via direct Newell summation.
//
// RMSE is split into three physically meaningful regions:
//   Edge   — material cells within 5 cells of a hole boundary
//   Bulk   — material cells >20 cells from any hole boundary
//   Global — all material cells
//
// Timing is split into:
//   Setup  — kernel build, hierarchy construction, regrid (one-time)
//   Eval   — restrict + FFT + interpolate (per-timestep cost)
//
// Outputs in out/bench_antidot/:
//   - summary.txt             — human-readable results
//   - results.csv             — machine-readable RMSE per mode
//   - error_heatmap_*.ppm     — spatial error maps (holes=black, error=blue→red)
//   - patch_map.png           — AMR patch layout (L1=yellow, L2=green, L3=blue)
//   - mesh_geom.png           — antidot geometry with multi-level grid overlay
//   - mesh_error_*.png        — error field with multi-level grid overlay
//   - b_edge_profile_*.csv    — per-edge-cell error breakdown (with --csv)
//   - restricted_m_diag.csv   — M-restriction diagnostic (with --csv)
//
// Run examples:
//   # Full comparison (default 512×512 L0, 1024² fine ref):
//   cargo run --release --bin bench_antidot
//
//   # Quick iteration at 256×256 L0 (512² fine ref):
//   LLG_AD_BASE_NX=256 LLG_AD_BASE_NY=256 \
//     cargo run --release --bin bench_antidot
//
//   # Sweep super-coarse ratios:
//   LLG_DEMAG_COARSEN_RATIO=1 cargo run --release --bin bench_antidot
//   LLG_DEMAG_COARSEN_RATIO=2 cargo run --release --bin bench_antidot
//   LLG_DEMAG_COARSEN_RATIO=4 cargo run --release --bin bench_antidot
//
//   # MG+hybrid composite crossover test:
//   LLG_DEMAG_MG_HYBRID_ENABLE=1 LLG_DEMAG_MG_STENCIL=7 \
//     cargo run --release --bin bench_antidot -- --no-plots
//
//   # Crossover sweep (increasing grid size):
//   for NX in 256 512 1024 2048; do
//     LLG_DEMAG_MG_HYBRID_ENABLE=1 LLG_DEMAG_MG_STENCIL=7 \
//       LLG_AD_BASE_NX=$NX LLG_AD_BASE_NY=$NX \
//       cargo run --release --bin bench_antidot -- --no-plots
//   done
//
//   # Sparse holes (wider pitch → lower AMR coverage → earlier crossover):
//   LLG_DEMAG_MG_HYBRID_ENABLE=1 LLG_DEMAG_MG_STENCIL=7 \
//     LLG_AD_HOLE_PITCH=800e-9 LLG_AD_BASE_NX=1024 \
//     cargo run --release --bin bench_antidot -- --no-plots
//
//   # Skip fine reference (timing mode only):
//   cargo run --release --bin bench_antidot -- --skip-fine-ref
//
//   # With per-cell CSV for plotting:
//   cargo run --release --bin bench_antidot -- --csv
//
//   # Skip plots (headless/CI mode):
//   cargo run --release --bin bench_antidot -- --no-plots

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use plotters::prelude::*;

use llg_sim::effective_field::coarse_fft_demag;
use llg_sim::effective_field::demag_fft_uniform;
use llg_sim::effective_field::mg_composite;
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, Material};
use llg_sim::vector_field::VectorField2D;

use llg_sim::amr::indicator::IndicatorKind;
use llg_sim::amr::regrid::maybe_regrid_nested_levels;
use llg_sim::amr::{AmrHierarchy2D, ClusterPolicy, Connectivity, Rect2i, RegridPolicy};
use llg_sim::geometry_mask::{hex_hole_centres, mask_count, MaskShape};

// =====================================================================
// Constants & utilities
// =====================================================================

const PI: f64 = std::f64::consts::PI;

fn ensure_dir(path: &str) {
    if !Path::new(path).exists() {
        fs::create_dir_all(path).unwrap();
    }
}

#[inline]
fn idx(i: usize, j: usize, nx: usize) -> usize {
    j * nx + i
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[allow(dead_code)]
fn pow_usize(mut base: usize, mut exp: usize) -> usize {
    let mut out = 1usize;
    while exp > 0 {
        if (exp & 1) == 1 { out = out.saturating_mul(base); }
        exp >>= 1;
        if exp > 0 { base = base.saturating_mul(base); }
    }
    out
}

// =====================================================================
// Edge / bulk cell classification (BFS distance to vacuum)
// =====================================================================

/// Classify material cells by BFS distance to nearest hole boundary.
/// Returns (distance_field, edge_mask, bulk_mask).
fn classify_cells(
    geom_mask: &[bool],
    nx: usize,
    ny: usize,
    edge_dist: usize,
    bulk_dist: usize,
) -> (Vec<u32>, Vec<bool>, Vec<bool>) {
    let n = nx * ny;

    // BFS: compute distance-to-nearest-vacuum for each material cell.
    let mut dist = vec![u32::MAX; n];
    let mut queue = std::collections::VecDeque::new();

    // Seed: material cells with at least one von-Neumann vacuum neighbour
    for j in 0..ny {
        for i in 0..nx {
            let k = idx(i, j, nx);
            if !geom_mask[k] { continue; }
            let has_hole_nbr =
                (i > 0      && !geom_mask[idx(i-1, j, nx)]) ||
                (i+1 < nx   && !geom_mask[idx(i+1, j, nx)]) ||
                (j > 0      && !geom_mask[idx(i, j-1, nx)]) ||
                (j+1 < ny   && !geom_mask[idx(i, j+1, nx)]);
            if has_hole_nbr {
                dist[k] = 1;
                queue.push_back((i, j));
            }
        }
    }

    // BFS flood (4-connected) — propagate through material cells
    while let Some((ci, cj)) = queue.pop_front() {
        let d = dist[idx(ci, cj, nx)];
        if d >= bulk_dist as u32 + 5 { continue; } // no need to go further
        for &(di, dj) in &[(!0usize, 0), (1, 0), (0, !0usize), (0, 1)] {
            let ni = ci.wrapping_add(di);
            let nj = cj.wrapping_add(dj);
            if ni >= nx || nj >= ny { continue; }
            let nk = idx(ni, nj, nx);
            if !geom_mask[nk] { continue; }
            if d + 1 < dist[nk] {
                dist[nk] = d + 1;
                queue.push_back((ni, nj));
            }
        }
    }

    let mut edge_mask = vec![false; n];
    let mut bulk_mask = vec![false; n];
    for k in 0..n {
        if !geom_mask[k] { continue; }
        if dist[k] <= edge_dist as u32 {
            edge_mask[k] = true;
        } else if dist[k] > bulk_dist as u32 || dist[k] == u32::MAX {
            bulk_mask[k] = true;
        }
    }
    (dist, edge_mask, bulk_mask)
}

// =====================================================================
// RMSE computation (material-cell-only variants)
// =====================================================================

fn global_rmse(
    a: &VectorField2D, b: &VectorField2D, material_mask: &[bool],
) -> (f64, f64, f64, f64, f64) {
    let mut sx = 0.0_f64; let mut sy = 0.0_f64; let mut sz = 0.0_f64;
    let mut maxd = 0.0_f64; let mut n = 0usize;
    for k in 0..a.data.len().min(b.data.len()) {
        if !material_mask[k] { continue; }
        let da = a.data[k]; let db = b.data[k];
        let ex = da[0]-db[0]; let ey = da[1]-db[1]; let ez = da[2]-db[2];
        sx += ex*ex; sy += ey*ey; sz += ez*ez;
        let d = (ex*ex + ey*ey + ez*ez).sqrt();
        if d > maxd { maxd = d; }
        n += 1;
    }
    let nf = n.max(1) as f64;
    ((sx/nf).sqrt(), (sy/nf).sqrt(), (sz/nf).sqrt(), ((sx+sy+sz)/nf).sqrt(), maxd)
}

fn region_rmse(a: &VectorField2D, b: &VectorField2D, region: &[bool]) -> (f64, usize) {
    let mut sum = 0.0_f64; let mut n = 0usize;
    for k in 0..a.data.len().min(b.data.len()) {
        if !region[k] { continue; }
        let da = a.data[k]; let db = b.data[k];
        sum += (da[0]-db[0]).powi(2) + (da[1]-db[1]).powi(2) + (da[2]-db[2]).powi(2);
        n += 1;
    }
    (if n > 0 { (sum / n as f64).sqrt() } else { 0.0 }, n)
}

// =====================================================================
// Magnetisation initialisation: flower state or saturated
// =====================================================================

/// Compute flower-state magnetisation at position (x_cent, y_cent) in centred coords.
///
/// Uses potential flow around cylinders: M follows the streamlines of an
/// ideal fluid flowing in +x̂ around circular obstacles. Near each hole,
/// M curves tangentially to reduce the normal component M·n̂ (and thus
/// the surface charge σ = M·n̂). Far from holes, M → [1, 0, 0].
///
/// This is the magnetostatic equilibrium (ignoring exchange), and creates
/// genuine ∇·M variation near hole boundaries that fine patches resolve
/// better than the coarse staircase.
fn flower_m_at(
    x: f64, y: f64,
    holes: &[(f64, f64)],
    radius: f64,
    shape: &MaskShape,
) -> [f64; 3] {
    if !shape.contains(x, y) {
        return [0.0, 0.0, 0.0];
    }

    let r2_h = radius * radius;
    let mut mx = 1.0_f64;
    let mut my = 0.0_f64;

    for &(hx, hy) in holes {
        let rx = x - hx;
        let ry = y - hy;
        let r2 = rx * rx + ry * ry;
        if r2 < 1e-30 { continue; }
        let r4 = r2 * r2;

        // Potential flow correction: dipolar field from each hole
        mx -= r2_h * (rx * rx - ry * ry) / r4;
        my -= r2_h * 2.0 * rx * ry / r4;
    }

    // Normalize to unit vector
    let mag = (mx * mx + my * my).sqrt();
    if mag < 1e-15 { return [1.0, 0.0, 0.0]; }
    [mx / mag, my / mag, 0.0]
}

/// Compute magnetisation at (x_cent, y_cent) — dispatches on init mode.
fn init_m_at(
    x: f64, y: f64,
    holes: &[(f64, f64)],
    radius: f64,
    shape: &MaskShape,
    use_flower: bool,
) -> [f64; 3] {
    if use_flower {
        flower_m_at(x, y, holes, radius, shape)
    } else {
        if shape.contains(x, y) { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 0.0] }
    }
}

// =====================================================================
// Fine-grid reference
// =====================================================================

fn build_fine_m(
    fine_grid: &Grid2D,
    shape: &MaskShape,
    holes: &[(f64, f64)],
    radius: f64,
    use_flower: bool,
) -> VectorField2D {
    let mut m = VectorField2D::new(*fine_grid);
    let half_lx = fine_grid.nx as f64 * fine_grid.dx * 0.5;
    let half_ly = fine_grid.ny as f64 * fine_grid.dy * 0.5;
    for j in 0..fine_grid.ny {
        let y_cent = (j as f64 + 0.5) * fine_grid.dy - half_ly;
        for i in 0..fine_grid.nx {
            let x_cent = (i as f64 + 0.5) * fine_grid.dx - half_lx;
            let k = j * fine_grid.nx + i;
            m.data[k] = init_m_at(x_cent, y_cent, holes, radius, shape, use_flower);
        }
    }
    m
}

fn downsample_b_to_l0(
    b_fine: &VectorField2D, b_l0: &mut VectorField2D, ratio: usize,
) {
    let nx_l0 = b_l0.grid.nx;
    let ny_l0 = b_l0.grid.ny;
    let nx_f = b_fine.grid.nx;
    let inv = 1.0 / ((ratio * ratio) as f64);
    for j in 0..ny_l0 {
        for i in 0..nx_l0 {
            let mut sum = [0.0_f64; 3];
            let fi0 = i * ratio;
            let fj0 = j * ratio;
            for fj in 0..ratio {
                for fi in 0..ratio {
                    let v = b_fine.data[(fj0 + fj) * nx_f + (fi0 + fi)];
                    sum[0] += v[0]; sum[1] += v[1]; sum[2] += v[2];
                }
            }
            b_l0.data[j * nx_l0 + i] = [sum[0]*inv, sum[1]*inv, sum[2]*inv];
        }
    }
}

// =====================================================================
// Reinitialise AMR patch magnetisation at fine resolution
// =====================================================================

fn reinit_patches(
    h: &mut AmrHierarchy2D,
    shape: &MaskShape,
    holes: &[(f64, f64)],
    radius: f64,
    use_flower: bool,
) {
    let base_dx = h.base_grid.dx;
    let base_dy = h.base_grid.dy;
    let half_lx = h.base_grid.nx as f64 * base_dx * 0.5;
    let half_ly = h.base_grid.ny as f64 * base_dy * 0.5;

    for p in &mut h.patches {
        let pdx = p.grid.dx; let pdy = p.grid.dy;
        let pnx = p.grid.nx; let pny = p.grid.ny;
        let gh = p.ghost; let cr = p.coarse_rect;
        let x0 = cr.i0 as f64 * base_dx - gh as f64 * pdx;
        let y0 = cr.j0 as f64 * base_dy - gh as f64 * pdy;
        for j in 0..pny {
            let y_cent = y0 + (j as f64 + 0.5) * pdy - half_ly;
            for i in 0..pnx {
                let x_cent = x0 + (i as f64 + 0.5) * pdx - half_lx;
                p.m.data[j * pnx + i] = init_m_at(x_cent, y_cent, holes, radius, shape, use_flower);
            }
        }
    }
    for lvl_patches in &mut h.patches_l2plus {
        for p in lvl_patches {
            let pdx = p.grid.dx; let pdy = p.grid.dy;
            let pnx = p.grid.nx; let pny = p.grid.ny;
            let gh = p.ghost; let cr = p.coarse_rect;
            let x0 = cr.i0 as f64 * base_dx - gh as f64 * pdx;
            let y0 = cr.j0 as f64 * base_dy - gh as f64 * pdy;
            for j in 0..pny {
                let y_cent = y0 + (j as f64 + 0.5) * pdy - half_ly;
                for i in 0..pnx {
                    let x_cent = x0 + (i as f64 + 0.5) * pdx - half_lx;
                    p.m.data[j * pnx + i] = init_m_at(x_cent, y_cent, holes, radius, shape, use_flower);
                }
            }
        }
    }
}

// =====================================================================
// PPM error heatmap
// =====================================================================

fn write_error_heatmap(
    path: &str, nx: usize, ny: usize,
    error: &[f64], geom_mask: &[bool], edge_mask: &[bool],
) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    write!(w, "P6\n{} {}\n255\n", nx, ny).unwrap();

    let mut emax = 1e-30_f64;
    for k in 0..nx*ny {
        if geom_mask[k] && error[k] > emax { emax = error[k]; }
    }
    let log_max = emax.log10();
    let log_min = (emax * 1e-4).log10();

    for j in (0..ny).rev() {
        for i in 0..nx {
            let k = idx(i, j, nx);
            let (r, g, b) = if !geom_mask[k] {
                (30u8, 30, 30)
            } else {
                let t = if error[k] > 0.0 {
                    ((error[k].log10() - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
                } else { 0.0 };
                if edge_mask[k] {
                    ((180.0 + 75.0*t) as u8, (200.0*(1.0-t)) as u8, 20u8)
                } else {
                    ((180.0*t) as u8, (180.0*t) as u8, (60.0 + 195.0*(1.0 - t*0.5)) as u8)
                }
            };
            w.write_all(&[r, g, b]).unwrap();
        }
    }
}

// =====================================================================
// Plotters-based visualisation (ported from amr_vortex_relax)
// =====================================================================

#[allow(dead_code)]
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> RGBColor {
    let h = h.rem_euclid(1.0);
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - (i as f64);
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    RGBColor(
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    )
}

fn level_rects(h: &AmrHierarchy2D, lvl: usize) -> Vec<Rect2i> {
    match lvl {
        1 => h.patches.iter().map(|p| p.coarse_rect).collect(),
        l if l >= 2 => h.patches_l2plus.get(l - 2)
            .map(|v| v.iter().map(|p| p.coarse_rect).collect())
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Save patch map: patch rectangles by refinement level.
/// L1=yellow, L2=green, L3=blue — same style as amr_vortex_relax.
fn save_patch_map(
    base_grid: &Grid2D,
    l1: &[Rect2i], l2: &[Rect2i], l3: &[Rect2i],
    path: &str, caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx0 = base_grid.nx as f64;
    let ny0 = base_grid.ny as f64;

    let root = BitMapBackend::new(path, (900, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 22))
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(35)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;

    chart.configure_mesh().x_desc("x/L").y_desc("y/L").disable_mesh().draw()?;

    // L1 (yellow)
    for r in l1 {
        let x0 = r.i0 as f64 / nx0;
        let y0 = r.j0 as f64 / ny0;
        let x1 = (r.i0 + r.nx) as f64 / nx0;
        let y1 = (r.j0 + r.ny) as f64 / ny0;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y0), (x1, y1)], RGBColor(240, 220, 0).filled(),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            BLACK.stroke_width(2),
        )))?;
    }

    // L2 (green)
    for r in l2 {
        let x0 = r.i0 as f64 / nx0;
        let y0 = r.j0 as f64 / ny0;
        let x1 = (r.i0 + r.nx) as f64 / nx0;
        let y1 = (r.j0 + r.ny) as f64 / ny0;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y0), (x1, y1)], RGBColor(0, 200, 0).filled(),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            RED.stroke_width(2),
        )))?;
    }

    // L3 (blue)
    for r in l3 {
        let x0 = r.i0 as f64 / nx0;
        let y0 = r.j0 as f64 / ny0;
        let x1 = (r.i0 + r.nx) as f64 / nx0;
        let y1 = (r.j0 + r.ny) as f64 / ny0;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y0), (x1, y1)], RGBColor(0, 120, 255).filled(),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            RGBColor(0, 60, 140).stroke_width(2),
        )))?;
    }

    root.present()?;
    Ok(())
}

/// Save geometry + mesh zoom: antidot pattern colored by distance-to-boundary,
/// with multi-level grid overlay (L0=gray, L1=black, L2=red, L3=blue).
///
/// Zooms into a region containing a few holes so individual cells are visible.
fn save_mesh_geom(
    base_grid: &Grid2D,
    geom_mask: &[bool],
    dist_field: &[u32],
    edge_dist: usize,
    _ratio: usize,
    _amr_max_level: usize,
    l1: &[Rect2i], l2: &[Rect2i], l3: &[Rect2i],
    path: &str,
    caption: &str,
    zoom_centre: Option<(usize, usize)>,
    zoom_radius_cells: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = base_grid.nx;
    let ny = base_grid.ny;

    // Determine zoom window (in L0 cell coords)
    let (cx, cy) = zoom_centre.unwrap_or((nx / 2, ny / 2));
    let r = zoom_radius_cells;
    let x0 = cx.saturating_sub(r);
    let y0 = cy.saturating_sub(r);
    let x1 = (cx + r).min(nx);
    let y1 = (cy + r).min(ny);
    let img_size = 900u32;
    let root = BitMapBackend::new(path, (img_size, img_size)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(x0 as i32..x1 as i32, y0 as i32..y1 as i32)?;
    chart.configure_mesh().disable_mesh().draw()?;

    // Background: antidot geometry colored by distance to boundary
    let max_d = (edge_dist as f64) * 3.0;
    chart.draw_series((y0..y1).flat_map(|j| {
        (x0..x1).map(move |i| {
            let k = idx(i, j, nx);
            let col = if !geom_mask[k] {
                // Vacuum (hole) — dark gray
                RGBColor(40, 40, 45)
            } else {
                let d = dist_field[k];
                if d <= edge_dist as u32 {
                    // Edge cells: orange→yellow gradient
                    let t = d as f64 / edge_dist as f64;
                    RGBColor(
                        (255.0 - 30.0 * t) as u8,
                        (140.0 + 80.0 * t) as u8,
                        (50.0 + 20.0 * t) as u8,
                    )
                } else {
                    // Bulk/transition: green gradient fading to teal
                    let t = ((d as f64 - edge_dist as f64) / max_d).min(1.0);
                    RGBColor(
                        (100.0 * (1.0 - t)) as u8,
                        (180.0 + 40.0 * t) as u8,
                        (120.0 + 100.0 * t) as u8,
                    )
                }
            };
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    // L0 grid lines (light gray)
    {
        let mut x = x0;
        while x <= x1 {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x as i32, y0 as i32), (x as i32, y1 as i32)],
                RGBColor(120, 120, 120).stroke_width(1),
            )))?;
            x += 1;
        }
        let mut y = y0;
        while y <= y1 {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x0 as i32, y as i32), (x1 as i32, y as i32)],
                RGBColor(120, 120, 120).stroke_width(1),
            )))?;
            y += 1;
        }
    }

    // Patch outlines: L1=red, L2=green, L3=blue (thick border only, no internal grid
    // — at L0 zoom level the fine grid would be too dense to render clearly)
    for (rects, color) in &[
        (l1, RGBColor(220, 40, 40)),
        (l2, RGBColor(0, 200, 0)),
        (l3, RGBColor(0, 100, 255)),
    ] {
        for r in *rects {
            let ri0 = r.i0.max(x0) as i32;
            let rj0 = r.j0.max(y0) as i32;
            let ri1 = ((r.i0 + r.nx).min(x1)) as i32;
            let rj1 = ((r.j0 + r.ny).min(y1)) as i32;
            if ri1 <= ri0 || rj1 <= rj0 { continue; }
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(ri0, rj0), (ri1, rj0), (ri1, rj1), (ri0, rj1), (ri0, rj0)],
                color.stroke_width(3),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

/// Save error field with mesh overlay: error magnitude at each cell
/// with multi-level grid overlay.  Zooms into a region with a few holes.
fn save_mesh_error(
    base_grid: &Grid2D,
    geom_mask: &[bool],
    edge_mask: &[bool],
    error: &[f64],
    l1: &[Rect2i], l2: &[Rect2i], l3: &[Rect2i],
    path: &str,
    caption: &str,
    zoom_centre: Option<(usize, usize)>,
    zoom_radius_cells: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = base_grid.nx;
    let ny = base_grid.ny;

    let (cx, cy) = zoom_centre.unwrap_or((nx / 2, ny / 2));
    let r = zoom_radius_cells;
    let x0 = cx.saturating_sub(r);
    let y0 = cy.saturating_sub(r);
    let x1 = (cx + r).min(nx);
    let y1 = (cy + r).min(ny);

    // Find max error in the zoom window for scaling
    let mut emax = 1e-30_f64;
    for j in y0..y1 {
        for i in x0..x1 {
            let k = idx(i, j, nx);
            if geom_mask[k] && error[k] > emax { emax = error[k]; }
        }
    }

    let img_size = 900u32;
    let root = BitMapBackend::new(path, (img_size, img_size)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(x0 as i32..x1 as i32, y0 as i32..y1 as i32)?;
    chart.configure_mesh().disable_mesh().draw()?;

    // Background: error magnitude (blue→red for material, dark for vacuum)
    chart.draw_series((y0..y1).flat_map(|j| {
        (x0..x1).map(move |i| {
            let k = idx(i, j, nx);
            let col = if !geom_mask[k] {
                RGBColor(30, 30, 30)
            } else {
                let t = if emax > 0.0 { (error[k] / emax).clamp(0.0, 1.0) } else { 0.0 };
                if edge_mask[k] {
                    // Edge: yellow → red
                    RGBColor(
                        (180.0 + 75.0 * t) as u8,
                        (200.0 * (1.0 - t)) as u8,
                        20u8,
                    )
                } else {
                    // Bulk/transition: blue → purple
                    RGBColor(
                        (200.0 * t) as u8,
                        (80.0 * (1.0 - t)) as u8,
                        (80.0 + 175.0 * (1.0 - t * 0.5)) as u8,
                    )
                }
            };
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    // L0 grid lines
    {
        let mut x = x0;
        while x <= x1 {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x as i32, y0 as i32), (x as i32, y1 as i32)],
                RGBColor(80, 80, 80).stroke_width(1),
            )))?;
            x += 1;
        }
        let mut y = y0;
        while y <= y1 {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x0 as i32, y as i32), (x1 as i32, y as i32)],
                RGBColor(80, 80, 80).stroke_width(1),
            )))?;
            y += 1;
        }
    }

    // Patch outlines
    for (rects, color) in &[
        (l1, RGBColor(220, 40, 40)),
        (l2, RGBColor(0, 200, 0)),
        (l3, RGBColor(0, 100, 255)),
    ] {
        for r in *rects {
            let ri0 = r.i0.max(x0) as i32;
            let rj0 = r.j0.max(y0) as i32;
            let ri1 = ((r.i0 + r.nx).min(x1)) as i32;
            let rj1 = ((r.j0 + r.ny).min(y1)) as i32;
            if ri1 <= ri0 || rj1 <= rj0 { continue; }
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(ri0, rj0), (ri1, rj0), (ri1, rj1), (ri0, rj1), (ri0, rj0)],
                color.stroke_width(3),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

// =====================================================================
// CSV output
// =====================================================================

fn write_b_csv(path: &str, b: &VectorField2D) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "i,j,bx,by,bz").unwrap();
    for j in 0..b.grid.ny {
        for i in 0..b.grid.nx {
            let v = b.data[idx(i, j, b.grid.nx)];
            writeln!(w, "{},{},{:.10e},{:.10e},{:.10e}", i, j, v[0], v[1], v[2]).unwrap();
        }
    }
}

fn write_edge_profile_csv(
    path: &str, b_test: &VectorField2D, b_ref: &VectorField2D,
    edge_mask: &[bool], nx: usize, ny: usize,
) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "i,j,bx_ref,by_ref,bz_ref,bx_test,by_test,bz_test,error_mag").unwrap();
    for j in 0..ny {
        for i in 0..nx {
            let k = idx(i, j, nx);
            if !edge_mask[k] { continue; }
            let vr = b_ref.data[k]; let vt = b_test.data[k];
            let err = ((vt[0]-vr[0]).powi(2)+(vt[1]-vr[1]).powi(2)+(vt[2]-vr[2]).powi(2)).sqrt();
            writeln!(w, "{},{},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e},{:.10e}",
                i, j, vr[0], vr[1], vr[2], vt[0], vt[1], vt[2], err).unwrap();
        }
    }
}

fn write_m_restriction_diagnostic(
    path: &str,
    m_binary: &VectorField2D, m_restricted: &VectorField2D,
    geom_mask: &[bool], edge_mask: &[bool],
    nx: usize, ny: usize,
) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "i,j,is_edge,mx_binary,my_binary,mz_binary,mx_restr,my_restr,mz_restr,mag_restr,delta_mag").unwrap();
    for j in 0..ny {
        for i in 0..nx {
            let k = idx(i, j, nx);
            if !geom_mask[k] { continue; }
            let vb = m_binary.data[k];
            let vr = m_restricted.data[k];
            let delta = ((vb[0]-vr[0]).powi(2)+(vb[1]-vr[1]).powi(2)+(vb[2]-vr[2]).powi(2)).sqrt();
            if delta < 1e-14 { continue; }
            let mag = (vr[0]*vr[0]+vr[1]*vr[1]+vr[2]*vr[2]).sqrt();
            writeln!(w, "{},{},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6},{:.6e}",
                i, j, if edge_mask[k] { 1 } else { 0 },
                vb[0], vb[1], vb[2], vr[0], vr[1], vr[2], mag, delta).unwrap();
        }
    }
}

// =====================================================================
// Hierarchy helpers
// =====================================================================

fn total_patch_fine_cells(h: &AmrHierarchy2D) -> usize {
    let mut n = 0usize;
    for p in &h.patches { n += p.grid.nx * p.grid.ny; }
    for lvl in &h.patches_l2plus {
        for p in lvl { n += p.grid.nx * p.grid.ny; }
    }
    n
}

// =====================================================================
// Main
// =====================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let do_csv = args.iter().any(|a| a == "--csv");
    let skip_fine = args.iter().any(|a| a == "--skip-fine-ref");
    let no_plots = args.iter().any(|a| a == "--no-plots");

    let out_dir = "out/bench_antidot";
    ensure_dir(out_dir);

    // ---- AMR parameters ----
    let amr_max_level: usize = env_or("LLG_AMR_MAX_LEVEL", 3);
    let ratio = 2usize;
    let ghost = 2usize;

    // ---- Physical domain ----
    let base_nx: usize = env_or("LLG_AD_BASE_NX", 512);
    let base_ny: usize = env_or("LLG_AD_BASE_NY", 512);
    let lx: f64 = env_or("LLG_AD_LX", 2.0e-6);
    let ly: f64 = env_or("LLG_AD_LY", 2.0e-6);
    let dz: f64 = env_or("LLG_AD_DZ", 1.0e-9);

    let dx = lx / base_nx as f64;
    let dy = ly / base_ny as f64;

    let fine_ratio: usize = env_or("LLG_AD_FINE_RATIO", 2);
    let fine_nx = base_nx * fine_ratio;
    let fine_ny = base_ny * fine_ratio;
    let fine_dx = dx / fine_ratio as f64;
    let fine_dy = dy / fine_ratio as f64;

    // ---- Anti-dot geometry ----
    // CROSSOVER TUNING: Edit hole_pitch to control feature sparsity.
    // Wider pitch = fewer holes = less AMR coverage = lower N_eff = earlier crossover.
    //   400e-9  (default) — dense, ~17 holes in 2μm domain
    //   800e-9            — sparse, ~4 holes
    //   1000e-9           — very sparse, ~2 holes
    let hole_diam: f64 = env_or("LLG_AD_HOLE_DIAM", 80.0e-9);
    let hole_pitch: f64 = env_or("LLG_AD_HOLE_PITCH", 400.0e-9);
    let hole_radius = hole_diam * 0.5;
    let hole_centres = hex_hole_centres(lx, ly, hole_pitch);
    let n_holes = hole_centres.len();

    // ---- Material: Permalloy ----
    let ms: f64 = env_or("LLG_AD_MS", 8.0e5);
    let a_ex: f64 = env_or("LLG_AD_AEX", 1.3e-11);
    let l_ex = (2.0 * a_ex / (4.0 * PI * 1e-7 * ms * ms)).sqrt();

    let mat = Material {
        ms, a_ex,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let base_grid = Grid2D::new(base_nx, base_ny, dx, dy, dz);
    let fine_grid = Grid2D::new(fine_nx, fine_ny, fine_dx, fine_dy, dz);

    // ---- Build geometry mask (on L0 grid) ----
    let antidot_shape = MaskShape::MultiHole {
        holes: hole_centres.clone(),
        radius: hole_radius,
    };
    let geom_mask = antidot_shape.to_mask(&base_grid);
    let n_material = mask_count(&geom_mask);
    let n_vacuum = base_nx * base_ny - n_material;

    // ---- Cell classification ----
    let edge_dist: usize = env_or("LLG_AD_EDGE_DIST", 5);
    let bulk_dist: usize = env_or("LLG_AD_BULK_DIST", 20);
    let (dist_field, edge_mask, bulk_mask) =
        classify_cells(&geom_mask, base_nx, base_ny, edge_dist, bulk_dist);
    let n_edge = mask_count(&edge_mask);
    let n_bulk = mask_count(&bulk_mask);
    let n_transition = n_material - n_edge - n_bulk;

    // ---- Initialise L0 magnetisation ----
    // LLG_AD_INIT=flower  → potential-flow flower state (non-trivial ∇·M near holes)
    // LLG_AD_INIT=saturated (default) → uniform +x̂ (binary mask, trivial ∇·M)
    let use_flower = std::env::var("LLG_AD_INIT")
        .map(|s| s.trim().to_ascii_lowercase() == "flower")
        .unwrap_or(false);
    let init_label = if use_flower { "flower (+x̂ curling around holes)" } else { "saturated +x̂" };

    let half_lx = base_nx as f64 * dx * 0.5;
    let half_ly = base_ny as f64 * dy * 0.5;
    let mut m_binary = VectorField2D::new(base_grid);
    for j in 0..base_ny {
        let y_cent = (j as f64 + 0.5) * dy - half_ly;
        for i in 0..base_nx {
            let x_cent = (i as f64 + 0.5) * dx - half_lx;
            let k = j * base_nx + i;
            m_binary.data[k] = init_m_at(x_cent, y_cent, &hole_centres, hole_radius, &antidot_shape, use_flower);
        }
    }

    let mode_filter: Option<String> = std::env::var("LLG_AMR_DEMAG_MODE").ok()
        .map(|s| s.trim().to_ascii_lowercase());

    // ---- Header ----
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Anti-Dot Demag Benchmark — Phase 2 P-FFT Validation        ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Domain:     {:.1} μm × {:.1} μm × {:.1} nm", lx*1e6, ly*1e6, dz*1e9);
    println!("L0 grid:    {} × {}   dx={:.3e}  dy={:.3e}", base_nx, base_ny, dx, dy);
    println!("Fine grid:  {} × {} ({}× L0)  dx={:.3e}", fine_nx, fine_ny, fine_ratio, fine_dx);
    println!("Material:   Ms={ms:.2e}  A_ex={a_ex:.2e}  l_ex={:.2} nm", l_ex*1e9);
    println!("Anti-dot:   {} holes, d={:.0} nm, pitch={:.0} nm (hex)",
        n_holes, hole_diam*1e9, hole_pitch*1e9);
    println!("            cells/hole_diam: L0≈{:.1}, fine≈{:.1}",
        hole_diam / dx, hole_diam / fine_dx);
    println!("Cells (L0): {} total, {} material ({:.1}%), {} vacuum",
        base_nx*base_ny, n_material,
        100.0 * n_material as f64 / (base_nx*base_ny) as f64, n_vacuum);
    println!("            {} edge (≤{}), {} bulk (>{}), {} transition",
        n_edge, edge_dist, n_bulk, bulk_dist, n_transition);
    println!("Init:       {} (set LLG_AD_INIT=flower for non-trivial M near holes)", init_label);
    println!("AMR:        {} level(s), ratio={}, ghost={}", amr_max_level, ratio, ghost);
    println!("Output:     {out_dir}");
    if skip_fine { println!("  --skip-fine-ref: skipping fine-grid FFT reference"); }
    if do_csv { println!("  --csv: per-cell CSV output enabled"); }
    if no_plots { println!("  --no-plots: skipping plot generation"); }
    println!();

    let t0 = Instant::now();

    // =====================================================================
    // Step 1: Fine-grid FFT reference
    // =====================================================================

    let b_ref_l0;
    let ref_fine_time_ms;
    let ref_setup_time_ms;
    let bref_max;

    if !skip_fine {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Computing fine-grid FFT reference ({fine_nx}×{fine_ny}) ...");
        println!("═══════════════════════════════════════════════════════════════");

        let t_setup = Instant::now();
        let m_fine = build_fine_m(&fine_grid, &antidot_shape, &hole_centres, hole_radius, use_flower);
        let fine_mask = antidot_shape.to_mask(&fine_grid);
        let fine_n_mat = mask_count(&fine_mask);
        let setup_ms = t_setup.elapsed().as_secs_f64() * 1e3;
        println!("  Fine M built: {} material cells ({:.1}%), {:.1} ms",
            fine_n_mat, 100.0 * fine_n_mat as f64 / (fine_nx*fine_ny) as f64, setup_ms);

        println!("  Warming up FFT kernel for {}×{} grid ...", fine_nx, fine_ny);
        let t_warmup = Instant::now();
        {
            let mut bw = VectorField2D::new(fine_grid);
            demag_fft_uniform::compute_demag_field(&fine_grid, &m_fine, &mut bw, &mat);
        }
        let warmup_ms = t_warmup.elapsed().as_secs_f64() * 1e3;
        println!("  Kernel build + first FFT: {:.1} ms (one-time setup cost)", warmup_ms);
        ref_setup_time_ms = warmup_ms;

        let t_eval = Instant::now();
        let mut b_fine = VectorField2D::new(fine_grid);
        demag_fft_uniform::compute_demag_field(&fine_grid, &m_fine, &mut b_fine, &mat);
        ref_fine_time_ms = t_eval.elapsed().as_secs_f64() * 1e3;
        println!("  FFT eval: {:.1} ms (per-timestep cost)", ref_fine_time_ms);

        let mut b_ref = VectorField2D::new(base_grid);
        downsample_b_to_l0(&b_fine, &mut b_ref, fine_ratio);

        bref_max = b_ref.data.iter().enumerate()
            .filter(|&(k, _)| geom_mask[k])
            .map(|(_, v)| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
            .fold(0.0_f64, f64::max);
        println!("  max|B_demag| = {:.4e} T (material cells, at L0 resolution)", bref_max);

        if do_csv { write_b_csv(&format!("{out_dir}/b_ref_fine.csv"), &b_ref); }
        println!();
        b_ref_l0 = b_ref;
    } else {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Computing L0 FFT reference ({base_nx}×{base_ny}) ...");
        println!("  WARNING: --skip-fine-ref means coarse_fft R=1 will show ~0% error.");
        println!("═══════════════════════════════════════════════════════════════");

        let t_warmup = Instant::now();
        {
            let mut bw = VectorField2D::new(base_grid);
            demag_fft_uniform::compute_demag_field(&base_grid, &m_binary, &mut bw, &mat);
        }
        ref_setup_time_ms = t_warmup.elapsed().as_secs_f64() * 1e3;

        let t_eval = Instant::now();
        let mut b_ref = VectorField2D::new(base_grid);
        demag_fft_uniform::compute_demag_field(&base_grid, &m_binary, &mut b_ref, &mat);
        ref_fine_time_ms = t_eval.elapsed().as_secs_f64() * 1e3;

        bref_max = b_ref.data.iter().enumerate()
            .filter(|&(k, _)| geom_mask[k])
            .map(|(_, v)| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
            .fold(0.0_f64, f64::max);
        println!("  Kernel: {:.1} ms, eval: {:.1} ms, max|B|={:.4e} T",
            ref_setup_time_ms, ref_fine_time_ms, bref_max);
        println!();
        b_ref_l0 = b_ref;
    }

    // =====================================================================
    // Step 2: Build AMR hierarchy
    // =====================================================================

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Building AMR hierarchy ...");
    println!("═══════════════════════════════════════════════════════════════");

    let t_amr_setup = Instant::now();

    let mut m_coarse_amr = VectorField2D::new(base_grid);
    m_coarse_amr.data.copy_from_slice(&m_binary.data);
    let mut h = AmrHierarchy2D::new(base_grid, m_coarse_amr, ratio, ghost);
    h.set_geom_shape(antidot_shape.clone());

    let indicator_kind = if std::env::var("LLG_AMR_INDICATOR").is_ok() {
        IndicatorKind::from_env()
    } else {
        IndicatorKind::Composite { frac: 0.10 }
    };

    let buffer_cells = 4usize;
    let boundary_layer: usize = env_or("LLG_AMR_BOUNDARY_LAYER", 2);
    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind,
        buffer_cells,
        boundary_layer,
        connectivity: Connectivity::Eight,
        merge_distance: 1,
        min_patch_area: 16,
        max_patches: 0,
        min_efficiency: 0.70,
        max_flagged_fraction: 0.50,
    };
    let regrid_policy = RegridPolicy {
        indicator: indicator_kind,
        buffer_cells,
        boundary_layer,
        min_change_cells: 1,
        min_area_change_frac: 0.01,
    };

    let current_patches: Vec<Rect2i> = Vec::new();
    if let Some((_new_rects, stats)) =
        maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
    {
        println!("  Regrid: {} cells flagged, threshold={:.4e}",
            stats.flagged_cells, stats.threshold);
    } else {
        println!("  Regrid: no patches created (try lowering LLG_AMR_INDICATOR_FRAC)");
    }

    h.fill_patch_ghosts();
    reinit_patches(&mut h, &antidot_shape, &hole_centres, hole_radius, use_flower);
    h.restrict_patches_to_coarse();

    let amr_setup_ms = t_amr_setup.elapsed().as_secs_f64() * 1e3;

    // ---- AMR diagnostics ----
    let l1 = level_rects(&h, 1);
    let l2 = level_rects(&h, 2);
    let l3 = level_rects(&h, 3);
    let n_patches_total = l1.len() + l2.len() + l3.len();
    let n_fine_cells = total_patch_fine_cells(&h);

    println!("  Patches:    L1: {}  L2: {}  L3: {}", l1.len(), l2.len(), l3.len());
    println!("  Fine cells: {} (in patches)", n_fine_cells);
    println!("  AMR setup:  {:.1} ms", amr_setup_ms);

    // M-restriction diagnostic
    let mut n_m_changed = 0usize;
    let mut max_m_delta = 0.0_f64;
    for k in 0..base_nx * base_ny {
        if !geom_mask[k] { continue; }
        let vb = m_binary.data[k];
        let vr = h.coarse.data[k];
        let d = ((vb[0]-vr[0]).powi(2)+(vb[1]-vr[1]).powi(2)+(vb[2]-vr[2]).powi(2)).sqrt();
        if d > 1e-14 {
            n_m_changed += 1;
            if d > max_m_delta { max_m_delta = d; }
        }
    }
    println!("  M restriction changed {} L0 cells (max Δ|m| = {:.4e})", n_m_changed, max_m_delta);
    if n_m_changed == 0 {
        println!("  NOTE: masked restriction preserves m=[1,0,0] for all material cells.");
        println!("        Patches resolve geometry at fine dx but the mask-aware averaging");
        println!("        excludes vacuum sub-cells, so the coarse M is unchanged.");
        println!("        This is expected for saturated states — P-FFT will bypass this");
        println!("        limitation by correcting B_demag directly at edge cells.");
    }

    // BIT-EXACT M comparison: catch even floating-point rounding differences
    {
        let mut n_bitexact_diff = 0usize;
        let mut first_diff: Option<(usize, usize, [f64; 3], [f64; 3])> = None;
        for k in 0..base_nx * base_ny {
            let vb = m_binary.data[k];
            let vr = h.coarse.data[k];
            if vb[0] != vr[0] || vb[1] != vr[1] || vb[2] != vr[2] {
                n_bitexact_diff += 1;
                if first_diff.is_none() {
                    first_diff = Some((k % base_nx, k / base_nx, vb, vr));
                }
            }
        }
        if n_bitexact_diff > 0 {
            println!("  ⚠ BIT-EXACT M comparison: {} cells differ (including vacuum)",
                n_bitexact_diff);
            if let Some((i, j, vb, vr)) = first_diff {
                println!("    First diff at ({i},{j}): binary={:.15e},{:.15e},{:.15e}",
                    vb[0], vb[1], vb[2]);
                println!("                          coarse={:.15e},{:.15e},{:.15e}",
                    vr[0], vr[1], vr[2]);
            }
            println!("    This may cause coarse_fft R=1 to differ from fft_l0.");
        } else {
            println!("  ✓ BIT-EXACT: enhanced M == binary M for all {} cells", base_nx * base_ny);
        }
    }

    if do_csv {
        write_m_restriction_diagnostic(
            &format!("{out_dir}/restricted_m_diag.csv"),
            &m_binary, &h.coarse, &geom_mask, &edge_mask, base_nx, base_ny,
        );
    }
    println!();

    // =====================================================================
    // Step 2b: Plots — patch map and geometry+mesh zoom
    // =====================================================================

    if !no_plots {
        println!("  Generating plots ...");

        // Find a good zoom centre: pick a hole near the domain centre
        let domain_cx = base_nx / 2;
        let domain_cy = base_ny / 2;
        let zoom_r = 30usize; // radius in L0 cells

        // Patch map (full domain)
        let pm_path = format!("{out_dir}/patch_map.png");
        if let Err(e) = save_patch_map(
            &base_grid, &l1, &l2, &l3, &pm_path,
            "Anti-dot patch map (L1 yellow, L2 green, L3 blue)",
        ) {
            eprintln!("  Warning: patch_map plot failed: {e}");
        } else {
            println!("  Patch map:  {pm_path}");
        }

        // Geometry + mesh zoom
        let mg_path = format!("{out_dir}/mesh_geom.png");
        if let Err(e) = save_mesh_geom(
            &base_grid, &geom_mask, &dist_field, edge_dist,
            ratio, amr_max_level, &l1, &l2, &l3,
            &mg_path,
            "Antidot geometry + AMR mesh (orange=edge, green=bulk, dark=hole)",
            Some((domain_cx, domain_cy)), zoom_r,
        ) {
            eprintln!("  Warning: mesh_geom plot failed: {e}");
        } else {
            println!("  Mesh+geom:  {mg_path}");
        }
        println!();
    }

    // =====================================================================
    // Step 3: Evaluate modes and compare
    // =====================================================================

    struct ModeResult {
        name: String,
        setup_ms: f64,
        eval_ms: f64,
        rmse_global: f64,
        rmse_edge: f64,
        rmse_bulk: f64,
        max_delta: f64,
        n_edge: usize,
        n_bulk: usize,
    }
    let mut results: Vec<ModeResult> = Vec::new();
    // Store error fields for plotting
    let mut error_fields: Vec<(String, Vec<f64>)> = Vec::new();

    let should_run = |name: &str| -> bool {
        match &mode_filter {
            None => true,
            Some(f) => {
                let n = name.to_ascii_lowercase();
                n.contains(f.as_str()) || f.contains(n.as_str())
                    || (f == "coarsefft" && n.starts_with("coarse_fft"))
                    || (f == "cfft" && n.starts_with("coarse_fft"))
                    || (f == "composite" && n.contains("composite"))
                    || (f == "amr" && n.contains("amr"))
            }
        }
    };

    // ---- Mode 1: fft_fine (the reference) ----
    if !skip_fine && should_run("fft_fine") {
        println!("Mode: fft_fine (FFT on {}×{} fine grid, downsampled to L0) ...", fine_nx, fine_ny);
        println!("  (This IS the reference — RMSE = 0 by definition)");
        println!("  Setup: {:.1} ms (kernel build)   Eval: {:.1} ms", ref_setup_time_ms, ref_fine_time_ms);
        results.push(ModeResult {
            name: "fft_fine".into(),
            setup_ms: ref_setup_time_ms, eval_ms: ref_fine_time_ms,
            rmse_global: 0.0, rmse_edge: 0.0, rmse_bulk: 0.0,
            max_delta: 0.0, n_edge, n_bulk,
        });
    }

    // ---- Mode 2: fft_l0 ----
    if should_run("fft_l0") {
        println!("Mode: fft_l0 (FFT on {}×{} L0, binary mask M) ...", base_nx, base_ny);

        let t_warmup = Instant::now();
        {
            let mut bw = VectorField2D::new(base_grid);
            demag_fft_uniform::compute_demag_field(&base_grid, &m_binary, &mut bw, &mat);
        }
        let warmup_ms = t_warmup.elapsed().as_secs_f64() * 1e3;

        let t_eval = Instant::now();
        let mut b_l0 = VectorField2D::new(base_grid);
        demag_fft_uniform::compute_demag_field(&base_grid, &m_binary, &mut b_l0, &mat);
        let eval_ms = t_eval.elapsed().as_secs_f64() * 1e3;

        let (_, _, _, rt, md) = global_rmse(&b_l0, &b_ref_l0, &geom_mask);
        let (re, ne) = region_rmse(&b_l0, &b_ref_l0, &edge_mask);
        let (rb, nb) = region_rmse(&b_l0, &b_ref_l0, &bulk_mask);
        let grel = if bref_max > 0.0 { rt / bref_max * 100.0 } else { 0.0 };
        let erel = if bref_max > 0.0 { re / bref_max * 100.0 } else { 0.0 };
        let brel = if bref_max > 0.0 { rb / bref_max * 100.0 } else { 0.0 };

        println!("  Setup: {:.1} ms (kernel, cached)   Eval: {:.1} ms", warmup_ms, eval_ms);
        println!("  Global RMSE: {rt:.4e} T ({grel:.2}%)  max ΔB: {md:.3e} T");
        println!("  Edge   RMSE: {re:.4e} T ({erel:.2}%)  [{ne} cells]");
        println!("  Bulk   RMSE: {rb:.4e} T ({brel:.2}%)  [{nb} cells]");

        let error_mag: Vec<f64> = (0..base_nx*base_ny).map(|k| {
            let da = b_l0.data[k]; let db = b_ref_l0.data[k];
            ((da[0]-db[0]).powi(2)+(da[1]-db[1]).powi(2)+(da[2]-db[2]).powi(2)).sqrt()
        }).collect();
        let hmap_path = format!("{out_dir}/error_heatmap_fft_l0.ppm");
        write_error_heatmap(&hmap_path, base_nx, base_ny, &error_mag, &geom_mask, &edge_mask);
        println!("  Heatmap: {hmap_path}");

        error_fields.push(("fft_l0".into(), error_mag));

        if do_csv {
            write_b_csv(&format!("{out_dir}/b_fft_l0.csv"), &b_l0);
            write_edge_profile_csv(
                &format!("{out_dir}/b_edge_profile_fft_l0.csv"),
                &b_l0, &b_ref_l0, &edge_mask, base_nx, base_ny,
            );
        }

        results.push(ModeResult {
            name: "fft_l0".into(),
            setup_ms: warmup_ms, eval_ms,
            rmse_global: rt, rmse_edge: re, rmse_bulk: rb,
            max_delta: md, n_edge: ne, n_bulk: nb,
        });
    }

    // ---- Mode 3: coarse_fft R=N ----
    if should_run("coarse_fft") {
        let r_val: usize = env_or("LLG_DEMAG_COARSEN_RATIO", 1);

        let t_warmup = Instant::now();
        {
            let mut bw = VectorField2D::new(base_grid);
            let _ = coarse_fft_demag::compute_coarse_fft_demag(&h, &mat, &mut bw);
        }
        let warmup_ms = t_warmup.elapsed().as_secs_f64() * 1e3;

        let demag_nx = if r_val <= 1 { base_nx } else { base_nx / r_val };
        println!("Mode: coarse_fft R={} (AMR restrict → {}² FFT → interp) ...",
            r_val, demag_nx);
        println!("  Setup: {:.1} ms (AMR build {:.1} ms + kernel {:.1} ms)",
            amr_setup_ms + warmup_ms, amr_setup_ms, warmup_ms);

        let n_runs = 3;
        let mut best_ms = f64::INFINITY;
        let mut b_coarse = VectorField2D::new(base_grid);
        for _ in 0..n_runs {
            let t1 = Instant::now();
            let mut b_tmp = VectorField2D::new(base_grid);
            let _ = coarse_fft_demag::compute_coarse_fft_demag(&h, &mat, &mut b_tmp);
            let dt = t1.elapsed().as_secs_f64() * 1e3;
            if dt < best_ms {
                best_ms = dt;
                b_coarse.data.copy_from_slice(&b_tmp.data);
            }
        }
        let eval_ms = best_ms;

        let (_, _, _, rt, md) = global_rmse(&b_coarse, &b_ref_l0, &geom_mask);
        let (re, ne) = region_rmse(&b_coarse, &b_ref_l0, &edge_mask);
        let (rb, nb) = region_rmse(&b_coarse, &b_ref_l0, &bulk_mask);
        let grel = if bref_max > 0.0 { rt / bref_max * 100.0 } else { 0.0 };
        let erel = if bref_max > 0.0 { re / bref_max * 100.0 } else { 0.0 };
        let brel = if bref_max > 0.0 { rb / bref_max * 100.0 } else { 0.0 };

        println!("  Eval:    {eval_ms:.1} ms (best of {n_runs})");
        println!("  Global RMSE: {rt:.4e} T ({grel:.2}%)  max ΔB: {md:.3e} T");
        println!("  Edge   RMSE: {re:.4e} T ({erel:.2}%)  [{ne} cells]");
        println!("  Bulk   RMSE: {rb:.4e} T ({brel:.2}%)  [{nb} cells]");

        if r_val > 1 {
            println!("  NOTE: R={r_val} edge RMSE may appear lower than R=1 — this is an artifact.");
            println!("        The L0→demag_grid restriction (UN-masked) smooths the staircase,");
            println!("        acting as a low-pass filter on Gibbs oscillations.  But bulk RMSE");
            println!("        ({brel:.1}%) reveals the true cost: far-field accuracy is sacrificed.");
        }

        let error_mag: Vec<f64> = (0..base_nx*base_ny).map(|k| {
            let da = b_coarse.data[k]; let db = b_ref_l0.data[k];
            ((da[0]-db[0]).powi(2)+(da[1]-db[1]).powi(2)+(da[2]-db[2]).powi(2)).sqrt()
        }).collect();
        let hmap_path = format!("{out_dir}/error_heatmap_coarse_fft_r{r_val}.ppm");
        write_error_heatmap(&hmap_path, base_nx, base_ny, &error_mag, &geom_mask, &edge_mask);
        println!("  Heatmap: {hmap_path}");

        error_fields.push((format!("coarse_fft_r{r_val}"), error_mag));

        if do_csv {
            write_b_csv(&format!("{out_dir}/b_coarse_fft_r{r_val}.csv"), &b_coarse);
            write_edge_profile_csv(
                &format!("{out_dir}/b_edge_profile_coarse_fft_r{r_val}.csv"),
                &b_coarse, &b_ref_l0, &edge_mask, base_nx, base_ny,
            );
        }

        results.push(ModeResult {
            name: format!("coarse_fft R={r_val}"),
            setup_ms: amr_setup_ms + warmup_ms, eval_ms,
            rmse_global: rt, rmse_edge: re, rmse_bulk: rb,
            max_delta: md, n_edge: ne, n_bulk: nb,
        });
    }
    
    // ---- Mode 4: CompositeGrid (AMR-aware Poisson/FK with enhanced RHS) ----
    // This is the truly adaptive demag method:
    //   - Computes fine-resolution ∇·M from AMR patches (smooth geometry)
    //   - Injects into coarse-grid Poisson solve (enhanced RHS)
    //   - MG V-cycle runs on the coarse grid only (fast!)
    //   - Kzz convolution via FFT on coarse grid
    //   - Result: B_demag at L0 with fine-geometry charge information
    if should_run("composite_grid") {
        println!("Mode: composite_grid (AMR enhanced-RHS MG/FK on {}×{} coarse grid) ...",
            base_nx, base_ny);

        // Warm up MG solver (first call builds hierarchy + boundary integral)
        {
            let mut bw = VectorField2D::new(base_grid);
            let _ = mg_composite::compute_composite_demag(&h, &mat, &mut bw);
        }

        let n_runs = 3;
        let mut best_ms = f64::INFINITY;
        let mut b_composite = VectorField2D::new(base_grid);
        for _ in 0..n_runs {
            let t1 = Instant::now();
            b_composite.set_uniform(0.0, 0.0, 0.0);
            let _ = mg_composite::compute_composite_demag(&h, &mat, &mut b_composite);
            // Zero vacuum cells
            for k in 0..base_nx * base_ny {
                if !geom_mask[k] {
                    b_composite.data[k] = [0.0, 0.0, 0.0];
                }
            }
            let dt = t1.elapsed().as_secs_f64() * 1e3;
            if dt < best_ms { best_ms = dt; }
        }
        let eval_ms = best_ms;

        let (_, _, _, rt, md) = global_rmse(&b_composite, &b_ref_l0, &geom_mask);
        let (re, ne) = region_rmse(&b_composite, &b_ref_l0, &edge_mask);
        let (rb, nb) = region_rmse(&b_composite, &b_ref_l0, &bulk_mask);
        let grel = if bref_max > 0.0 { rt / bref_max * 100.0 } else { 0.0 };
        let erel = if bref_max > 0.0 { re / bref_max * 100.0 } else { 0.0 };
        let brel = if bref_max > 0.0 { rb / bref_max * 100.0 } else { 0.0 };

        println!("  Eval:    {eval_ms:.1} ms (best of {n_runs})");
        println!("  vs fine ref:");
        println!("    Global RMSE: {rt:.4e} T ({grel:.2}%)  max ΔB: {md:.3e} T");
        println!("    Edge   RMSE: {re:.4e} T ({erel:.2}%)  [{ne} cells]");
        println!("    Bulk   RMSE: {rb:.4e} T ({brel:.2}%)  [{nb} cells]");

        let error_mag: Vec<f64> = (0..base_nx*base_ny).map(|k| {
            let da = b_composite.data[k]; let db = b_ref_l0.data[k];
            ((da[0]-db[0]).powi(2)+(da[1]-db[1]).powi(2)+(da[2]-db[2]).powi(2)).sqrt()
        }).collect();
        let hmap_path = format!("{out_dir}/error_heatmap_composite_grid.ppm");
        write_error_heatmap(&hmap_path, base_nx, base_ny, &error_mag, &geom_mask, &edge_mask);
        println!("  Heatmap: {hmap_path}");
        error_fields.push(("composite_grid".to_string(), error_mag));
        if do_csv {
            write_b_csv(&format!("{out_dir}/b_composite_grid.csv"), &b_composite);
            write_edge_profile_csv(
                &format!("{out_dir}/b_edge_profile_composite_grid.csv"),
                &b_composite, &b_ref_l0, &edge_mask, base_nx, base_ny,
            );
        }
        results.push(ModeResult {
            name: "composite_grid".into(),
            setup_ms: amr_setup_ms,
            eval_ms,
            rmse_global: rt, rmse_edge: re, rmse_bulk: rb,
            max_delta: md, n_edge: ne, n_bulk: nb,
        });
    }

    // ---- Mode 5: AMR-FFT (flatten AMR hierarchy to fine, run FFT) ----
    // Uses AMR patch M at fine resolution near holes (smooth geometry from
    // MaskShape), coarse M upsampled in bulk. Single FFT on the composite
    // fine grid. Proves that AMR geometry data fixes the staircase error.
    // Runtime is similar to fft_fine (both use 1024×1024 FFT).
    if should_run("amr_fft") {
        println!("Mode: amr_fft (flatten AMR → {}×{} fine grid, FFT on composite M) ...",
            fine_nx, fine_ny);

        let t_build = Instant::now();

        // Step 1: Upsample L0 coarse M to fine grid (replicate each cell).
        let mut m_amr_fine = VectorField2D::new(fine_grid);
        for j in 0..base_ny {
            for i in 0..base_nx {
                let v = h.coarse.data[j * base_nx + i];
                for fj in 0..fine_ratio {
                    for fi in 0..fine_ratio {
                        let fi_g = i * fine_ratio + fi;
                        let fj_g = j * fine_ratio + fj;
                        m_amr_fine.data[fj_g * fine_nx + fi_g] = v;
                    }
                }
            }
        }

        // Step 2: Overlay L1 patches (fine-resolution M with smooth geometry).
        for p in &h.patches {
            let cr = &p.coarse_rect;
            let gh = p.ghost;
            let pnx = p.grid.nx;
            for jf in 0..(cr.ny * ratio) {
                for if_ in 0..(cr.nx * ratio) {
                    let src_i = gh + if_;
                    let src_j = gh + jf;
                    let v = p.m.data[src_j * pnx + src_i];
                    let dst_i = cr.i0 * ratio + if_;
                    let dst_j = cr.j0 * ratio + jf;
                    if dst_i < fine_nx && dst_j < fine_ny {
                        m_amr_fine.data[dst_j * fine_nx + dst_i] = v;
                    }
                }
            }
        }
        let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
        println!("  M build: {:.1} ms ({} L1 patches overlaid)", build_ms, h.patches.len());

        // Warm up FFT (reuses kernel from fft_fine).
        {
            let mut bw = VectorField2D::new(fine_grid);
            demag_fft_uniform::compute_demag_field(&fine_grid, &m_amr_fine, &mut bw, &mat);
        }

        let n_runs = 3;
        let mut best_ms = f64::INFINITY;
        let mut b_amr_fine = VectorField2D::new(fine_grid);
        for _ in 0..n_runs {
            let t1 = Instant::now();
            demag_fft_uniform::compute_demag_field(&fine_grid, &m_amr_fine, &mut b_amr_fine, &mat);
            let dt = t1.elapsed().as_secs_f64() * 1e3;
            if dt < best_ms { best_ms = dt; }
        }
        let eval_ms = best_ms;

        let mut b_amr_l0 = VectorField2D::new(base_grid);
        downsample_b_to_l0(&b_amr_fine, &mut b_amr_l0, fine_ratio);

        let (_, _, _, rt, md) = global_rmse(&b_amr_l0, &b_ref_l0, &geom_mask);
        let (re, ne) = region_rmse(&b_amr_l0, &b_ref_l0, &edge_mask);
        let (rb, nb) = region_rmse(&b_amr_l0, &b_ref_l0, &bulk_mask);
        let grel = if bref_max > 0.0 { rt / bref_max * 100.0 } else { 0.0 };
        let erel = if bref_max > 0.0 { re / bref_max * 100.0 } else { 0.0 };
        let brel = if bref_max > 0.0 { rb / bref_max * 100.0 } else { 0.0 };

        println!("  Eval:    {eval_ms:.1} ms (best of {n_runs})");
        println!("  vs fine ref:");
        println!("    Global RMSE: {rt:.4e} T ({grel:.2}%)  max ΔB: {md:.3e} T");
        println!("    Edge   RMSE: {re:.4e} T ({erel:.2}%)  [{ne} cells]");
        println!("    Bulk   RMSE: {rb:.4e} T ({brel:.2}%)  [{nb} cells]");

        let error_mag: Vec<f64> = (0..base_nx*base_ny).map(|k| {
            let da = b_amr_l0.data[k]; let db = b_ref_l0.data[k];
            ((da[0]-db[0]).powi(2)+(da[1]-db[1]).powi(2)+(da[2]-db[2]).powi(2)).sqrt()
        }).collect();
        let hmap_path = format!("{out_dir}/error_heatmap_amr_fft.ppm");
        write_error_heatmap(&hmap_path, base_nx, base_ny, &error_mag, &geom_mask, &edge_mask);
        println!("  Heatmap: {hmap_path}");
        error_fields.push(("amr_fft".to_string(), error_mag));
        if do_csv {
            write_b_csv(&format!("{out_dir}/b_amr_fft.csv"), &b_amr_l0);
            write_edge_profile_csv(
                &format!("{out_dir}/b_edge_profile_amr_fft.csv"),
                &b_amr_l0, &b_ref_l0, &edge_mask, base_nx, base_ny,
            );
        }
        results.push(ModeResult {
            name: "amr_fft".into(),
            setup_ms: amr_setup_ms + build_ms,
            eval_ms,
            rmse_global: rt, rmse_edge: re, rmse_bulk: rb,
            max_delta: md, n_edge: ne, n_bulk: nb,
        });
    }

    
    // =====================================================================
    // Step 3b: Error + mesh zoom plots
    // =====================================================================

    if !no_plots && !error_fields.is_empty() {
        let domain_cx = base_nx / 2;
        let domain_cy = base_ny / 2;
        let zoom_r = 30usize;

        for (label, err) in &error_fields {
            let me_path = format!("{out_dir}/mesh_error_{label}.png");
            if let Err(e) = save_mesh_error(
                &base_grid, &geom_mask, &edge_mask, err,
                &l1, &l2, &l3,
                &me_path,
                &format!("Error + mesh: {label} (yellow=edge err, blue=bulk)"),
                Some((domain_cx, domain_cy)), zoom_r,
            ) {
                eprintln!("  Warning: mesh_error plot failed: {e}");
            } else {
                println!("  Error+mesh: {me_path}");
            }
        }
    }

    // =====================================================================
    // Step 4: Summary table
    // =====================================================================

    let wall = t0.elapsed().as_secs_f64();
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Anti-Dot Demag Accuracy Summary  (reference: fine-grid FFT {}×{})                              ║", fine_nx, fine_ny);
    println!("╠══════════════════╤═══════╤═══════╤════════════╤════════════╤════════════╤════════╤═════════════════╣");
    println!("║ Mode             │ setup │ eval  │ Global RMSE│  Edge RMSE │  Bulk RMSE │ max ΔB │ verdict         ║");
    println!("║                  │  (ms) │ (ms)  │            │            │            │        │                 ║");
    println!("╠══════════════════╪═══════╪═══════╪════════════╪════════════╪════════════╪════════╪═════════════════╣");
    for r in &results {
        let grel = if bref_max > 0.0 { r.rmse_global / bref_max * 100.0 } else { 0.0 };
        let erel = if bref_max > 0.0 { r.rmse_edge / bref_max * 100.0 } else { 0.0 };
        let brel = if bref_max > 0.0 { r.rmse_bulk / bref_max * 100.0 } else { 0.0 };

        let verdict = if r.name == "fft_fine" {
            "REFERENCE"
        } else if erel < 3.0 && brel < 1.0 {
            "good"
        } else if erel < 10.0 && brel < 3.0 {
            "fair"
        } else if brel > 5.0 && erel < 18.0 {
            "BULK CORRUPT" // R=2 case: edge looks OK but bulk is sacrificed
        } else {
            "EDGE LOSSY"
        };

        println!("║ {:16} │{:>6.0} │{:>6.1} │ {:.4e} │ {:.4e} │ {:.4e} │{:.3e}│ {:15} ║",
            r.name, r.setup_ms, r.eval_ms,
            r.rmse_global, r.rmse_edge, r.rmse_bulk, r.max_delta, verdict);
        println!("║                  │       │       │  ({:5.2}%)  │  ({:5.2}%)  │  ({:5.2}%)  │        │                 ║",
            grel, erel, brel);
    }
    println!("╚══════════════════╧═══════╧═══════╧════════════╧════════════╧════════════╧════════╧═════════════════╝");

    println!();
    println!("Geometry:   {} holes, {} edge cells, {} bulk cells, {} transition",
        n_holes, n_edge, n_bulk, n_transition);
    println!("AMR:        {} patches, {} fine cells, M restriction changed {} L0 cells",
        n_patches_total, n_fine_cells, n_m_changed);
    println!("Reference:  max|B_demag| = {:.4e} T", bref_max);
    println!("Total wall: {:.3} s", wall);

    // ---- Interpretation ----
    println!();
    println!("INTERPRETATION:");
    println!("  fft_fine  = Gold reference.  Slow ({:.0}ms setup + {:.0}ms eval), but resolves holes at fine dx.",
        ref_setup_time_ms, ref_fine_time_ms);
    if let Some(r) = results.iter().find(|r| r.name == "fft_l0") {
        let erel = if bref_max > 0.0 { r.rmse_edge / bref_max * 100.0 } else { 0.0 };
        println!("  fft_l0    = L0 FFT ({:.0}ms eval).  Edge error {:.1}% from staircase boundary.",
            r.eval_ms, erel);
    }
    if let Some(r) = results.iter().find(|r| r.name.starts_with("coarse_fft")) {
        let erel = if bref_max > 0.0 { r.rmse_edge / bref_max * 100.0 } else { 0.0 };
        let brel = if bref_max > 0.0 { r.rmse_bulk / bref_max * 100.0 } else { 0.0 };
        println!("  {} = AMR + FFT ({:.0}ms eval).  Edge {:.1}%, bulk {:.1}%.",
            r.name, r.eval_ms, erel, brel);
        if brel > 5.0 {
            println!("            Low edge % is misleading: un-masked L0→demag restriction smooths");
            println!("            the staircase (acts as low-pass filter), reducing Gibbs oscillations");
            println!("            at edges.  But the bulk field is corrupted ({:.1}%).  This is NOT a", brel);
            println!("            real solution — P-FFT will provide correct edge correction.");
        } else {
            println!("            Both fft_l0 and coarse_fft suffer ~18% edge error from the L0");
            println!("            staircase.  Masked restriction preserves m=[1,0,0] at all material");
            println!("            cells, so AMR patches don't improve the coarse M for saturated states.");
            println!("            P-FFT will fix this by correcting B_demag directly at edge cells.");
        }
    }
    println!("  P-FFT target: edge RMSE < 3%, bulk < 1%, eval time close to coarse_fft.");

    // =====================================================================
    // CROSSOVER ANALYSIS: Uniform-Fine FFT vs AMR + Composite MG
    // =====================================================================
    {
        let fft_fine_r = results.iter().find(|r| r.name == "fft_fine");
        let composite_r = results.iter().find(|r| r.name == "composite_grid");
        let coarse_fft_r = results.iter().find(|r| r.name.starts_with("coarse_fft"));

        if fft_fine_r.is_some() || composite_r.is_some() || coarse_fft_r.is_some() {
            println!();
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║  CROSSOVER ANALYSIS: FFT (uniform) vs AMR+MG (adaptive)    ║");
            println!("╚══════════════════════════════════════════════════════════════╝");

            // --- Cell counts ---
            let n_fine_cells = fine_nx * fine_ny;
            let n_fine_fft_padded = (2 * fine_nx) * (2 * fine_ny);
            let n_l0_cells = base_nx * base_ny;
            let n_l0_fft_padded = (2 * base_nx) * (2 * base_ny);

            // MG padded box estimate
            let mg_pad: usize = env_or("LLG_DEMAG_MG_PAD_XY", 6);
            let mg_nvac: usize = env_or("LLG_DEMAG_MG_NVAC_Z", 16);
            let mg_px = base_nx + 2 * mg_pad;
            let mg_py = base_ny + 2 * mg_pad;
            let mg_pz = 1 + 2 * mg_nvac;
            let n_mg_padded = mg_px * mg_py * mg_pz;

            // Patch coverage
            let n_amr_patch_cells = n_fine_cells;
            let patch_coverage_pct = 100.0 * n_amr_patch_cells as f64 / n_fine_cells as f64;

            // N_eff for composite MG: MG padded box + patch cells for exchange/DMI
            let n_eff_composite = n_mg_padded + n_amr_patch_cells;

            // N_eff for coarse FFT: L0 padded FFT + patch cells
            let n_eff_coarse_fft = n_l0_fft_padded + n_amr_patch_cells;

            // Ratios
            let ratio_vs_fine = n_fine_fft_padded as f64 / n_eff_composite as f64;
            let ratio_coarse_fft = n_fine_fft_padded as f64 / n_eff_coarse_fft as f64;

            println!();
            println!("  Cell counts:");
            println!("    Uniform-fine FFT:    {} × {} = {} cells (padded: {})",
                fine_nx, fine_ny, n_fine_cells, n_fine_fft_padded);
            println!("    L0 grid:             {} × {} = {} cells",
                base_nx, base_ny, n_l0_cells);
            println!("    MG padded box:       {} × {} × {} = {} cells",
                mg_px, mg_py, mg_pz, n_mg_padded);
            println!("    AMR patch cells:     {} ({:.1}% of fine-equiv)",
                n_amr_patch_cells, patch_coverage_pct);
            println!();
            println!("    N_eff (composite MG): {} (MG padded + patches)", n_eff_composite);
            println!("    N_eff (coarse FFT):   {} (L0 FFT padded + patches)", n_eff_coarse_fft);
            println!("    Cell ratio (fine/composite): {:.1}×", ratio_vs_fine);
            println!("    Cell ratio (fine/coarse_fft): {:.1}×", ratio_coarse_fft);

            // --- Timing comparison ---
            println!();
            println!("  Timing comparison:");

            if let Some(r) = fft_fine_r {
                println!("    fft_fine:        {:.1} ms (setup) + {:.1} ms (eval)", r.setup_ms, r.eval_ms);
            }
            if let Some(r) = coarse_fft_r {
                println!("    coarse_fft:      {:.1} ms (setup) + {:.1} ms (eval)", r.setup_ms, r.eval_ms);
            }
            if let Some(r) = composite_r {
                println!("    composite_mg:    {:.1} ms (setup) + {:.1} ms (eval)", r.setup_ms, r.eval_ms);
            }

            // --- Speedup ratios ---
            if let (Some(fine_r), Some(comp_r)) = (fft_fine_r, composite_r) {
                let speedup_eval = fine_r.eval_ms / comp_r.eval_ms;
                let per_unknown_ratio = if n_eff_composite > 0 && n_fine_fft_padded > 0 {
                    (comp_r.eval_ms / n_eff_composite as f64)
                        / (fine_r.eval_ms / n_fine_fft_padded as f64)
                } else { f64::NAN };

                println!();
                if speedup_eval > 1.0 {
                    println!("  >>> COMPOSITE MG IS {:.1}× FASTER than uniform-fine FFT <<<", speedup_eval);
                } else {
                    println!("  >>> FFT is still {:.1}× faster (composite MG {:.1}× slower) <<<",
                        1.0/speedup_eval, 1.0/speedup_eval);
                }
                println!("    Per-unknown: MG is {:.0}× slower than FFT", per_unknown_ratio);
                println!("    Cell reduction: {:.1}× (fine/composite)", ratio_vs_fine);
                println!("    Needed for crossover: cell reduction > {:.0}×", per_unknown_ratio);

                // Accuracy comparison
                println!();
                println!("  Accuracy:");
                let fine_grel = if bref_max > 0.0 { fine_r.rmse_global / bref_max * 100.0 } else { 0.0 };
                let comp_grel = if bref_max > 0.0 { comp_r.rmse_global / bref_max * 100.0 } else { 0.0 };
                println!("    fft_fine:     {:.2}% global RMSE (reference)", fine_grel);
                println!("    composite_mg: {:.2}% global RMSE", comp_grel);
            }

            if let (Some(fine_r), Some(cfft_r)) = (fft_fine_r, coarse_fft_r) {
                let speedup_cfft = fine_r.eval_ms / cfft_r.eval_ms;
                println!();
                println!("  Coarse-FFT speedup: {:.1}× vs uniform-fine FFT", speedup_cfft);
            }

            // --- Crossover guidance ---
            println!();
            println!("  To find the crossover point, increase the grid size:");
            println!("    LLG_AD_BASE_NX=1024  (fine-equiv: 2048²)");
            println!("    LLG_AD_BASE_NX=2048  (fine-equiv: 4096²)");
            println!("    LLG_AD_BASE_NX=4096  (fine-equiv: 8192²)");
            println!("  To reduce patch coverage (earlier crossover):");
            println!("    LLG_AD_HOLE_PITCH=800e-9   (sparse holes)");
            println!("    LLG_AD_HOLE_PITCH=1000e-9  (very sparse)");
            println!();
            println!("  Quick crossover sweep:");
            println!("    for NX in 256 512 1024 2048; do");
            println!("      LLG_DEMAG_MG_HYBRID_ENABLE=1 LLG_DEMAG_MG_STENCIL=7 \\");
            println!("        LLG_AD_BASE_NX=$NX LLG_AD_BASE_NY=$NX \\");
            println!("        cargo run --release --bin bench_antidot -- --no-plots");
            println!("    done");
        }
    }

    // ---- CSV ----
    {
        let csv_path = format!("{out_dir}/results.csv");
        let f = File::create(&csv_path).unwrap();
        let mut w = BufWriter::new(f);
        writeln!(w, "mode,setup_ms,eval_ms,rmse_global,rmse_edge,rmse_bulk,max_delta,n_holes,n_edge,n_bulk").unwrap();
        for r in &results {
            writeln!(w, "{},{:.1},{:.1},{:.6e},{:.6e},{:.6e},{:.6e},{},{},{}",
                r.name, r.setup_ms, r.eval_ms,
                r.rmse_global, r.rmse_edge, r.rmse_bulk,
                r.max_delta, n_holes, r.n_edge, r.n_bulk).unwrap();
        }
        println!("CSV:        {csv_path}");
    }

    // ---- Crossover CSV (for plotting across grid sizes) ----
    {
        let cross_csv = format!("{out_dir}/crossover.csv");
        let f = File::create(&cross_csv).unwrap();
        let mut w = BufWriter::new(f);
        writeln!(w, "base_nx,fine_nx,n_fine_cells,n_l0_cells,n_patch_cells,n_holes,patch_coverage_pct").unwrap();
        let n_patch = total_patch_fine_cells(&h);
        let patch_cov = 100.0 * n_patch as f64 / (fine_nx * fine_ny) as f64;
        writeln!(w, "{},{},{},{},{},{},{:.2}",
            base_nx, fine_nx, fine_nx*fine_ny, base_nx*base_ny, n_patch, n_holes, patch_cov).unwrap();
        // Append timing per mode
        for r in &results {
            writeln!(w, "# {}: setup={:.1}ms eval={:.1}ms rmse={:.6e}",
                r.name, r.setup_ms, r.eval_ms, r.rmse_global).unwrap();
        }
        println!("Crossover:  {cross_csv}");
    }

    // ---- Summary text ----
    {
        let sum_path = format!("{out_dir}/summary.txt");
        let f = File::create(&sum_path).unwrap();
        let mut w = BufWriter::new(f);
        writeln!(w, "Anti-Dot Demag Benchmark — Phase 2").unwrap();
        writeln!(w, "Domain: {:.1} um x {:.1} um, L0={} x {}, fine={}x{}",
            lx*1e6, ly*1e6, base_nx, base_ny, fine_nx, fine_ny).unwrap();
        writeln!(w, "Holes: {} (d={:.0} nm, pitch={:.0} nm hex)", n_holes, hole_diam*1e9, hole_pitch*1e9).unwrap();
        writeln!(w, "Edge cells: {} (within {}), Bulk: {} (beyond {}), Transition: {}",
            n_edge, edge_dist, n_bulk, bulk_dist, n_transition).unwrap();
        writeln!(w, "AMR: {} level(s), ratio={}, {} patches, {} fine cells",
            amr_max_level, ratio, n_patches_total, n_fine_cells).unwrap();
        writeln!(w, "M restriction changed {} L0 cells (max delta={:.4e})", n_m_changed, max_m_delta).unwrap();
        writeln!(w).unwrap();
        for r in &results {
            let grel = if bref_max > 0.0 { r.rmse_global / bref_max * 100.0 } else { 0.0 };
            let erel = if bref_max > 0.0 { r.rmse_edge / bref_max * 100.0 } else { 0.0 };
            let brel = if bref_max > 0.0 { r.rmse_bulk / bref_max * 100.0 } else { 0.0 };
            writeln!(w, "{}: setup={:.0}ms eval={:.1}ms  global={:.4e} ({:.2}%), edge={:.4e} ({:.2}%), bulk={:.4e} ({:.2}%)",
                r.name, r.setup_ms, r.eval_ms,
                r.rmse_global, grel, r.rmse_edge, erel, r.rmse_bulk, brel).unwrap();
        }
        println!("Summary:    {sum_path}");
    }

    println!();
    println!("Done.");
}