// src/bin/amr_skyrmion_relax.rs
//
// AMR skyrmion bubble relaxation benchmark (all 5 field terms)
// =============================================================
//
// Tests the full AMR pipeline with Berger-Colella v2 subcycling on a
// Co/Pt-like PMA thin film with interfacial DMI, seeded with 5 Néel
// skyrmion bubbles on a 1µm × 1µm domain.
//
// This is the most complex AMR benchmark in the suite:
//   - Exchange + Anisotropy + DMI (local) + Demag (global FFT) + Zeeman
//   - Multiple disjoint features (skyrmion walls) → multiple AMR patches
//   - Ring-shaped gradient structures (not just point-like vortex cores)
//   - Features evolve: oversized bubbles contract to equilibrium → patches track
//   - Topological charge conservation as a physics diagnostic
//   - Berger-Colella v2 subcycling (intermediate restriction + parent ghost fill)
//
// Physics:
//   - Co/Pt-like thin film with perpendicular magnetic anisotropy (PMA)
//   - K_u easy-axis along ẑ → existing code produces correct B_ani = (2K_u/Ms)(m·ẑ)ẑ
//   - Interfacial (Néel-type) DMI stabilises skyrmion chirality
//   - Zeeman field Bz along +ẑ stabilises isolated bubbles against collapse
//   - Demag: thin-film FFT (nz=1), provides shape anisotropy opposing mz
//   - Moderate damping (α=0.2) gives genuine transient dynamics
//   - Oversized initial radius (R₀=50nm) → bubbles contract during relaxation
//
// Expected behaviour:
//   5 isolated Néel skyrmion bubbles contract from R₀=50nm to equilibrium
//   R_eq ≈ 25–35 nm over ~2000–5000 steps.  AMR patches track the
//   inward-moving walls, forcing periodic regrids.  At steady state,
//   5 separate L1 patches (one per skyrmion) with nested L2 patches.
//   Patch coverage ≈ 5–15% of domain.
//
// Grid setup:
//   - Base (level 0): 200×200 cells → dx = dy = 5 nm  (1µm × 1µm domain)
//   - AMR: up to 2 levels, ratio=2 → finest 800×800, dx_fine = 1.25 nm
//   - Exchange length l_ex ≈ 5.1 nm → ~4 cells at finest level
//   - Wall width δ ≈ 5 nm → ~4 cells at finest level (marginal at base)
//
// Outputs in out/amr_skyrmion_relax:
//   - patch_map_stepXXXX.png              : patch rectangles by refinement level [--plots]
//   - mz_amr_stepXXXX.png                : mz colour map of AMR composite [--plots]
//   - mz_coarse_stepXXXX.png             : mz colour map of coarse baseline [--plots]
//   - mesh_zoom_stepXXXX.png             : in-plane angle + multi-level grid overlay [--plots]
//   - regrid_log.csv                      : accepted regrid events
//   - regrid_levels.csv                   : per-accept per-level summaries
//   - regrid_attempts.csv                 : per-check diagnostics
//   - regrid_patches.csv                  : per-patch rectangles
//   - rmse_log.csv                        : AMR vs uniform fine RMSE vs step
//   - energy_log.csv                      : magnetisation diagnostics vs step
//   - skyrmion_log.csv                    : topological charge + per-skyrmion tracking
//   - ovf_coarse/mXXXXXXX.ovf            : coarse OVFs [--ovf]
//   - ovf_fine/mXXXXXXX.ovf              : fine reference OVFs [--ovf]
//   - ovf_amr/mXXXXXXX.ovf              : AMR composite OVFs [--ovf]
//   - *_final.csv                         : final states
//   - lineout_*_mid_y.csv                 : midline profiles
//   - run_info.txt                        : full parameter dump
//
// Run examples:
//   # Quick test (2 AMR levels, 3000 steps, subcycling auto):
//   cargo run --release --bin amr_skyrmion_relax
//
//   # Full benchmark with plots + OVF:
//   LLG_SKY_STEPS=5000 cargo run --release --bin amr_skyrmion_relax -- --plots --ovf
//
//   # Custom parameters:
//   LLG_SKY_BZ=0.08 LLG_SKY_ALPHA=0.1 LLG_SKY_N_SKYRMIONS=3 \
//     cargo run --release --bin amr_skyrmion_relax -- --plots
//
//   # Fast AMR-only check (skip uniform fine reference):
//   LLG_AMR_MAX_LEVEL=3 cargo run --release --bin amr_skyrmion_relax -- --plots --no-fine

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use plotters::prelude::*;

use llg_sim::effective_field::{FieldMask, demag_fft_uniform};
use llg_sim::grid::Grid2D;
use llg_sim::initial_states;
use llg_sim::llg::{RK4Scratch, step_llg_rk4_recompute_field_masked_relax_add};
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

use llg_sim::amr::indicator::IndicatorKind;
use llg_sim::amr::regrid::maybe_regrid_nested_levels;
use llg_sim::amr::{
    AmrHierarchy2D, AmrStepperRK4, ClusterPolicy, Connectivity, Rect2i, RegridPolicy,
};

// =====================================================================
// Physical constants
// =====================================================================
const MU_0: f64 = 1.2566370614359173e-6; // µ₀ in T·m/A
const PI: f64 = std::f64::consts::PI;

// =====================================================================
// Utility functions
// =====================================================================

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
        if (exp & 1) == 1 {
            out = out.saturating_mul(base);
        }
        exp >>= 1;
        if exp > 0 {
            base = base.saturating_mul(base);
        }
    }
    out
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// =====================================================================
// OVF writer (OOMMF OVF 2.0 text)
// =====================================================================

fn write_ovf_text(path: &str, m: &VectorField2D, title: &str) {
    let mut f = File::create(path).unwrap();
    let nx = m.grid.nx;
    let ny = m.grid.ny;
    let dx = m.grid.dx;
    let dy = m.grid.dy;
    let dz = m.grid.dz;

    writeln!(f, "# OOMMF OVF 2.0").unwrap();
    writeln!(f, "# Segment count: 1").unwrap();
    writeln!(f, "# Begin: Segment").unwrap();
    writeln!(f, "# Begin: Header").unwrap();
    writeln!(f, "# Title: {}", title).unwrap();
    writeln!(f, "# meshunit: m").unwrap();
    writeln!(f, "# meshtype: rectangular").unwrap();
    writeln!(f, "# xmin: 0.0").unwrap();
    writeln!(f, "# ymin: 0.0").unwrap();
    writeln!(f, "# zmin: 0.0").unwrap();
    writeln!(f, "# xmax: {:.12e}", nx as f64 * dx).unwrap();
    writeln!(f, "# ymax: {:.12e}", ny as f64 * dy).unwrap();
    writeln!(f, "# zmax: {:.12e}", dz).unwrap();
    writeln!(f, "# xnodes: {}", nx).unwrap();
    writeln!(f, "# ynodes: {}", ny).unwrap();
    writeln!(f, "# znodes: 1").unwrap();
    writeln!(f, "# xstepsize: {:.12e}", dx).unwrap();
    writeln!(f, "# ystepsize: {:.12e}", dy).unwrap();
    writeln!(f, "# zstepsize: {:.12e}", dz).unwrap();
    writeln!(f, "# valueunit: A/m").unwrap();
    writeln!(f, "# valuemultiplier: 1.0").unwrap();
    writeln!(f, "# valuerangemaxmag: 1.0").unwrap();
    writeln!(f, "# valuerangeminmag: 1.0").unwrap();
    writeln!(f, "# valuedim: 3").unwrap();
    writeln!(f, "# End: Header").unwrap();
    writeln!(f, "# Begin: Data Text").unwrap();

    for j in 0..ny {
        for i in 0..nx {
            let v = m.data[idx(i, j, nx)];
            writeln!(f, "{:.8e} {:.8e} {:.8e}", v[0], v[1], v[2]).unwrap();
        }
    }

    writeln!(f, "# End: Data Text").unwrap();
    writeln!(f, "# End: Segment").unwrap();
}

// =====================================================================
// Physics diagnostics
// =====================================================================

fn avg_mz(m: &VectorField2D) -> f64 {
    let n = m.data.len();
    if n == 0 {
        return 0.0;
    }
    let s: f64 = m.data.iter().map(|v| v[2]).sum();
    s / n as f64
}

fn max_abs_mz(m: &VectorField2D) -> f64 {
    m.data.iter().map(|v| v[2].abs()).fold(0.0f64, f64::max)
}

fn min_mz(m: &VectorField2D) -> f64 {
    m.data.iter().map(|v| v[2]).fold(f64::INFINITY, f64::min)
}

/// Compute topological charge Q = (1/4π) Σ m · (∂m/∂x × ∂m/∂y) dx dy.
///
/// Uses centred finite differences. Q = –N_sk for standard Néel skyrmions
/// (Q = –1 per skyrmion with core mz < 0 in a mz > 0 background).
fn topological_charge(m: &VectorField2D) -> f64 {
    let nx = m.grid.nx;
    let ny = m.grid.ny;
    let dx = m.grid.dx;
    let dy = m.grid.dy;

    let mut q = 0.0;

    // Interior cells only (centred differences)
    for j in 1..ny.saturating_sub(1) {
        for i in 1..nx.saturating_sub(1) {
            let c = m.data[idx(i, j, nx)];

            let xp = m.data[idx(i + 1, j, nx)];
            let xm = m.data[idx(i - 1, j, nx)];
            let yp = m.data[idx(i, j + 1, nx)];
            let ym = m.data[idx(i, j - 1, nx)];

            // ∂m/∂x ≈ (m(i+1,j) - m(i-1,j)) / (2 dx)
            let dmdx = [
                (xp[0] - xm[0]) / (2.0 * dx),
                (xp[1] - xm[1]) / (2.0 * dx),
                (xp[2] - xm[2]) / (2.0 * dx),
            ];
            // ∂m/∂y ≈ (m(i,j+1) - m(i,j-1)) / (2 dy)
            let dmdy = [
                (yp[0] - ym[0]) / (2.0 * dy),
                (yp[1] - ym[1]) / (2.0 * dy),
                (yp[2] - ym[2]) / (2.0 * dy),
            ];

            // cross = ∂m/∂x × ∂m/∂y
            let cross = [
                dmdx[1] * dmdy[2] - dmdx[2] * dmdy[1],
                dmdx[2] * dmdy[0] - dmdx[0] * dmdy[2],
                dmdx[0] * dmdy[1] - dmdx[1] * dmdy[0],
            ];

            // m · (∂m/∂x × ∂m/∂y)
            let density = c[0] * cross[0] + c[1] * cross[1] + c[2] * cross[2];

            q += density * dx * dy;
        }
    }

    q / (4.0 * PI)
}

/// Count approximate number of skyrmions by counting connected regions
/// where mz < threshold (default: 0).
fn count_skyrmion_cores(m: &VectorField2D, mz_threshold: f64) -> usize {
    let nx = m.grid.nx;
    let ny = m.grid.ny;
    let n = nx * ny;

    let mut visited = vec![false; n];
    let mut count = 0usize;

    for j in 0..ny {
        for i in 0..nx {
            let id = idx(i, j, nx);
            if visited[id] || m.data[id][2] >= mz_threshold {
                continue;
            }

            // BFS flood fill
            count += 1;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((i, j));
            visited[id] = true;

            while let Some((ci, cj)) = queue.pop_front() {
                for (di, dj) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let ni = ci as i32 + di;
                    let nj = cj as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= nx as i32 || nj >= ny as i32 {
                        continue;
                    }
                    let nid = idx(ni as usize, nj as usize, nx);
                    if !visited[nid] && m.data[nid][2] < mz_threshold {
                        visited[nid] = true;
                        queue.push_back((ni as usize, nj as usize));
                    }
                }
            }
        }
    }

    count
}

/// RMSE between two fields (cell-by-cell vector difference).
fn rmse_and_max_delta(a: &VectorField2D, b: &VectorField2D) -> (f64, f64) {
    let n = a.data.len().min(b.data.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut sum_sq = 0.0;
    let mut max_d = 0.0_f64;
    for k in 0..n {
        let d0 = a.data[k][0] - b.data[k][0];
        let d1 = a.data[k][1] - b.data[k][1];
        let d2 = a.data[k][2] - b.data[k][2];
        let dsq = d0 * d0 + d1 * d1 + d2 * d2;
        sum_sq += dsq;
        let d = dsq.sqrt();
        if d > max_d {
            max_d = d;
        }
    }
    ((sum_sq / n as f64).sqrt(), max_d)
}

// =====================================================================
// AMR helpers
// =====================================================================

fn flatten_to_target_grid(h: &AmrHierarchy2D, target: Grid2D) -> VectorField2D {
    let m = h.flatten_to_uniform_fine();
    if m.grid.nx == target.nx
        && m.grid.ny == target.ny
        && m.grid.dx == target.dx
        && m.grid.dy == target.dy
        && m.grid.dz == target.dz
    {
        m
    } else {
        m.resample_to_grid(target)
    }
}

fn patches_to_fine_rects(patches: &[Rect2i], r: usize) -> Vec<Rect2i> {
    patches
        .iter()
        .map(|p| Rect2i {
            i0: p.i0 * r,
            j0: p.j0 * r,
            nx: p.nx * r,
            ny: p.ny * r,
        })
        .collect()
}

fn union_rect(rects: &[Rect2i]) -> Option<Rect2i> {
    if rects.is_empty() {
        return None;
    }
    let mut i0 = rects[0].i0;
    let mut j0 = rects[0].j0;
    let mut i1 = rects[0].i0 + rects[0].nx;
    let mut j1 = rects[0].j0 + rects[0].ny;
    for r in rects.iter().skip(1) {
        i0 = i0.min(r.i0);
        j0 = j0.min(r.j0);
        i1 = i1.max(r.i0 + r.nx);
        j1 = j1.max(r.j0 + r.ny);
    }
    Some(Rect2i {
        i0,
        j0,
        nx: i1 - i0,
        ny: j1 - j0,
    })
}

fn union_rect_or_zero(rects: &[Rect2i]) -> Rect2i {
    union_rect(rects).unwrap_or(Rect2i {
        i0: 0,
        j0: 0,
        nx: 0,
        ny: 0,
    })
}

fn level_rects(h: &AmrHierarchy2D, lvl: usize) -> Vec<Rect2i> {
    match lvl {
        1 => h.patches.iter().map(|p| p.coarse_rect).collect(),
        l if l >= 2 => h
            .patches_l2plus
            .get(l - 2)
            .map(|v| v.iter().map(|p| p.coarse_rect).collect())
            .unwrap_or_else(Vec::new),
        _ => Vec::new(),
    }
}

fn level_patch_count(h: &AmrHierarchy2D, lvl: usize) -> usize {
    match lvl {
        1 => h.patches.len(),
        l if l >= 2 => h.patches_l2plus.get(l - 2).map(|v| v.len()).unwrap_or(0),
        _ => 0,
    }
}

fn all_level_rects(h: &AmrHierarchy2D, max_level: usize) -> Vec<Vec<Rect2i>> {
    (1..=max_level).map(|lvl| level_rects(h, lvl)).collect()
}

// =====================================================================
// Colour maps
// =====================================================================

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

/// Blue-white-red colour map for mz ∈ [–1, +1].
/// Blue = mz = –1 (skyrmion core), Red = mz = +1 (background).
fn mz_to_bwr(mz: f64) -> RGBColor {
    let t = ((mz + 1.0) * 0.5).clamp(0.0, 1.0);
    if t < 0.5 {
        let a = t / 0.5;
        RGBColor((255.0 * a) as u8, (255.0 * a) as u8, 255u8)
    } else {
        let a = (t - 0.5) / 0.5;
        RGBColor(255u8, (255.0 * (1.0 - a)) as u8, (255.0 * (1.0 - a)) as u8)
    }
}

// =====================================================================
// Plot helpers
// =====================================================================

fn save_patch_map(
    base_grid: &Grid2D,
    levels: &[Vec<Rect2i>],
    path: &str,
    caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx0 = base_grid.nx as f64;
    let ny0 = base_grid.ny as f64;

    let root = BitMapBackend::new(path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 22))
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(35)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;

    chart
        .configure_mesh()
        .x_desc("x / L")
        .y_desc("y / L")
        .draw()?;

    let colors: &[RGBColor] = &[
        RGBColor(200, 200, 0),  // L1 yellow
        RGBColor(0, 180, 0),    // L2 green
        RGBColor(0, 100, 255),  // L3 blue
        RGBColor(160, 80, 255), // L4 purple
    ];

    for (k, rects) in levels.iter().enumerate() {
        let col = colors[k.min(colors.len() - 1)];
        for r in rects {
            let x0 = r.i0 as f64 / nx0;
            let y0 = r.j0 as f64 / ny0;
            let x1 = (r.i0 + r.nx) as f64 / nx0;
            let y1 = (r.j0 + r.ny) as f64 / ny0;
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                col.stroke_width(3),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

fn save_mesh_zoom_multilevel(
    m_fine: &VectorField2D,
    _base_grid: &Grid2D,
    ratio: usize,
    max_level: usize,
    levels: &[Vec<Rect2i>],
    path: &str,
    caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let ref_ratio_total = pow_usize(ratio, max_level);
    let nx = m_fine.grid.nx;
    let ny = m_fine.grid.ny;

    let img_w = 800u32;
    let img_h = ((img_w as f64) * (ny as f64 / nx as f64)).max(200.0) as u32;

    let root = BitMapBackend::new(path, (img_w, img_h + 60)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..nx as i32, 0..ny as i32)?;

    chart.configure_mesh().disable_mesh().draw()?;

    // Background: in-plane angle colour map (HSV) — shows Néel wall chirality
    chart.draw_series((0..ny).flat_map(|j| {
        (0..nx).map(move |i| {
            let v = m_fine.data[idx(i, j, nx)];
            let phi = v[1].atan2(v[0]);
            let h = (phi + PI) / (2.0 * PI);
            // Reduce saturation where mz is large (show mostly uniform where mz ≈ ±1)
            let mz_frac = v[2].abs();
            let sat = (1.0 - mz_frac).max(0.0);
            let col = hsv_to_rgb(h, sat, 1.0);
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    // Grid overlay per level
    let colors: &[RGBColor] = &[
        BLACK,                 // L1
        RED,                   // L2
        RGBColor(0, 120, 255), // L3
    ];

    let level_spacing = |lvl: usize| -> usize {
        let r_lvl = pow_usize(ratio, lvl);
        (ref_ratio_total / r_lvl).max(1)
    };

    // L0 coarse grid (light grey)
    {
        let s0 = ref_ratio_total.max(1);
        let mut xx = s0;
        while xx < nx {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(xx as i32, 0i32), (xx as i32, ny as i32)],
                RGBColor(180, 180, 180).stroke_width(1),
            )))?;
            xx += s0;
        }
        let mut yy = s0;
        while yy < ny {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(0i32, yy as i32), (nx as i32, yy as i32)],
                RGBColor(180, 180, 180).stroke_width(1),
            )))?;
            yy += s0;
        }
    }

    // Higher-level grids inside their patches
    for (k, rects) in levels.iter().enumerate() {
        let lvl = k + 1;
        let sp = level_spacing(lvl);
        let col = colors[k.min(colors.len() - 1)];
        let _r_factor = pow_usize(ratio, lvl);

        for r in rects {
            let fi0 = r.i0 * ref_ratio_total;
            let fj0 = r.j0 * ref_ratio_total;
            let fi1 = (r.i0 + r.nx) * ref_ratio_total;
            let fj1 = (r.j0 + r.ny) * ref_ratio_total;

            // Patch boundary
            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (fi0 as i32, fj0 as i32),
                    (fi1 as i32, fj0 as i32),
                    (fi1 as i32, fj1 as i32),
                    (fi0 as i32, fj1 as i32),
                    (fi0 as i32, fj0 as i32),
                ],
                col.stroke_width(2),
            )))?;

            // Internal grid lines
            if sp > 0 {
                let mut xx = fi0 + sp;
                while xx < fi1 {
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(xx as i32, fj0 as i32), (xx as i32, fj1 as i32)],
                        col.stroke_width(1),
                    )))?;
                    xx += sp;
                }
                let mut yy = fj0 + sp;
                while yy < fj1 {
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(fi0 as i32, yy as i32), (fi1 as i32, yy as i32)],
                        col.stroke_width(1),
                    )))?;
                    yy += sp;
                }
            }
        }
    }

    root.present()?;
    Ok(())
}

/// Save mz colour map (Blue-White-Red).
fn save_mz_map(
    m: &VectorField2D,
    path: &str,
    caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = m.grid.nx;
    let ny = m.grid.ny;

    let img_w = 800u32;
    let img_h = ((img_w as f64) * (ny as f64 / nx as f64)).max(200.0) as u32;

    let root = BitMapBackend::new(path, (img_w, img_h + 60)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..nx as i32, 0..ny as i32)?;

    chart.configure_mesh().disable_mesh().draw()?;

    chart.draw_series((0..ny).flat_map(|j| {
        (0..nx).map(move |i| {
            let v = m.data[idx(i, j, nx)];
            let col = mz_to_bwr(v[2]);
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    root.present()?;
    Ok(())
}

// =====================================================================
// CSV output
// =====================================================================

fn write_csv_ij_m(path: &str, m: &VectorField2D) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "i,j,mx,my,mz").unwrap();
    for j in 0..m.grid.ny {
        for i in 0..m.grid.nx {
            let v = m.data[idx(i, j, m.grid.nx)];
            writeln!(w, "{},{},{:.8e},{:.8e},{:.8e}", i, j, v[0], v[1], v[2]).unwrap();
        }
    }
    w.flush().unwrap();
}

fn write_midline_y(path: &str, m: &VectorField2D) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "i,mx,my,mz").unwrap();
    let j = m.grid.ny / 2;
    for i in 0..m.grid.nx {
        let v = m.data[idx(i, j, m.grid.nx)];
        writeln!(w, "{},{:.8e},{:.8e},{:.8e}", i, v[0], v[1], v[2]).unwrap();
    }
    w.flush().unwrap();
}

fn write_midline_x(path: &str, m: &VectorField2D) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "j,mx,my,mz").unwrap();
    let i = m.grid.nx / 2;
    for j in 0..m.grid.ny {
        let v = m.data[idx(i, j, m.grid.nx)];
        writeln!(w, "{},{:.8e},{:.8e},{:.8e}", j, v[0], v[1], v[2]).unwrap();
    }
    w.flush().unwrap();
}

fn append_line(path: &str, line: &str) {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    f.write_all(line.as_bytes()).unwrap();
}

fn append_regrid_patches_csv(path: &str, h: &AmrHierarchy2D, max_level: usize, step: usize) {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();

    for (pid, p) in h.patches.iter().enumerate() {
        let r = p.coarse_rect;
        writeln!(
            f,
            "{},{},{},{},{},{},{}",
            step, 1, pid, r.i0, r.j0, r.nx, r.ny
        )
        .unwrap();
    }

    for lvl in 2..=max_level {
        let li = lvl - 2;
        if let Some(patches) = h.patches_l2plus.get(li) {
            for (pid, p) in patches.iter().enumerate() {
                let r = p.coarse_rect;
                writeln!(
                    f,
                    "{},{},{},{},{},{},{}",
                    step, lvl, pid, r.i0, r.j0, r.nx, r.ny
                )
                .unwrap();
            }
        }
    }
}

fn write_run_info(
    path: &str,
    base_grid: &Grid2D,
    fine_grid: &Grid2D,
    mat: &Material,
    llg: &LLGParams,
    amr_max_level: usize,
    ratio: usize,
    ghost: usize,
    steps: usize,
    regrid_every: usize,
    indicator_kind: IndicatorKind,
    n_skyrmions: usize,
    sky_r0: f64,
    sky_wall_w: f64,
) {
    let mut f = File::create(path).unwrap();
    writeln!(f, "AMR Skyrmion Bubble Relaxation Benchmark").unwrap();
    writeln!(f, "=========================================").unwrap();
    writeln!(f, "Co/Pt-like PMA thin film with interfacial DMI").unwrap();
    writeln!(
        f,
        "All 5 field terms: Exchange + Anisotropy + DMI + Demag + Zeeman"
    )
    .unwrap();
    writeln!(f, "").unwrap();
    writeln!(
        f,
        "Domain: {:.0}nm × {:.0}nm × {:.1}nm (thin film)",
        base_grid.nx as f64 * base_grid.dx * 1e9,
        base_grid.ny as f64 * base_grid.dy * 1e9,
        base_grid.dz * 1e9
    )
    .unwrap();
    writeln!(
        f,
        "Base grid:  {} × {}  dx={:.6e} dy={:.6e} dz={:.6e}",
        base_grid.nx, base_grid.ny, base_grid.dx, base_grid.dy, base_grid.dz
    )
    .unwrap();
    writeln!(
        f,
        "Fine grid:  {} × {}  dx={:.6e} dy={:.6e}",
        fine_grid.nx, fine_grid.ny, fine_grid.dx, fine_grid.dy
    )
    .unwrap();
    writeln!(
        f,
        "AMR levels: {}  ratio={}  ghost={}",
        amr_max_level, ratio, ghost
    )
    .unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "Material:").unwrap();
    writeln!(f, "  Ms   = {:.6e} A/m", mat.ms).unwrap();
    writeln!(f, "  A_ex = {:.6e} J/m", mat.a_ex).unwrap();
    writeln!(f, "  K_u  = {:.6e} J/m³ (PMA, easy axis = ẑ)", mat.k_u).unwrap();
    writeln!(
        f,
        "  DMI  = {:.6e} J/m² (interfacial Néel)",
        mat.dmi.unwrap_or(0.0)
    )
    .unwrap();
    writeln!(
        f,
        "  demag = {} (method: {:?})",
        mat.demag, mat.demag_method
    )
    .unwrap();
    writeln!(f, "").unwrap();

    let l_ex = (mat.a_ex / (MU_0 * mat.ms * mat.ms)).sqrt();
    let k_eff = mat.k_u - 0.5 * MU_0 * mat.ms * mat.ms;
    let delta_w = if k_eff > 0.0 {
        (mat.a_ex / k_eff).sqrt()
    } else {
        f64::INFINITY
    };
    let d_val = mat.dmi.unwrap_or(0.0).abs();
    let kappa = if k_eff > 0.0 && mat.a_ex > 0.0 {
        PI * d_val / (4.0 * (mat.a_ex * k_eff).sqrt())
    } else {
        0.0
    };

    writeln!(f, "Derived quantities:").unwrap();
    writeln!(f, "  l_ex   = {:.4e} m = {:.2} nm", l_ex, l_ex * 1e9).unwrap();
    writeln!(f, "  K_eff  = K_u - µ₀Ms²/2 = {:.4e} J/m³", k_eff).unwrap();
    writeln!(
        f,
        "  δ_wall = √(A_ex/K_eff) = {:.4e} m = {:.2} nm",
        delta_w,
        delta_w * 1e9
    )
    .unwrap();
    writeln!(f, "  κ      = πD/(4√(A·K_eff)) = {:.4}", kappa).unwrap();
    writeln!(f, "  finest dx / l_ex = {:.2}", fine_grid.dx / l_ex).unwrap();
    writeln!(f, "  finest dx / δ_wall = {:.2}", fine_grid.dx / delta_w).unwrap();
    writeln!(f, "").unwrap();

    writeln!(f, "LLG params:").unwrap();
    writeln!(f, "  gamma = {:.6e} rad/(s·T)", llg.gamma).unwrap();
    writeln!(f, "  alpha = {:.6e}", llg.alpha).unwrap();
    writeln!(f, "  dt    = {:.6e} s", llg.dt).unwrap();
    writeln!(
        f,
        "  B_ext = [0, 0, {:.6e}] T (stabilising Zeeman)",
        llg.b_ext[2]
    )
    .unwrap();
    writeln!(f, "  steps = {}", steps).unwrap();
    writeln!(
        f,
        "  total_time = {:.6e} s = {:.3} ns",
        steps as f64 * llg.dt,
        steps as f64 * llg.dt * 1e9
    )
    .unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "Initial condition:").unwrap();
    writeln!(
        f,
        "  {} Néel skyrmion bubbles on mz=+1 background",
        n_skyrmions
    )
    .unwrap();
    writeln!(
        f,
        "  R0 = {:.1} nm, wall_width = {:.1} nm",
        sky_r0 * 1e9,
        sky_wall_w * 1e9
    )
    .unwrap();
    writeln!(f, "  helicity = 0 (Néel/radial, matches interfacial DMI)").unwrap();
    writeln!(f, "").unwrap();
    writeln!(f, "Refinement:").unwrap();
    writeln!(
        f,
        "  indicator = {} (threshold param = {:.4})",
        indicator_kind.label(),
        indicator_kind.threshold_param()
    )
    .unwrap();
    writeln!(f, "  regrid_every = {}", regrid_every).unwrap();
}

// =====================================================================
// Skyrmion positions: spread across the domain (centered coordinates)
// =====================================================================

/// Generate skyrmion center positions in centered coordinates (meters).
/// Distributes N skyrmions in an aesthetically pleasing pattern:
///   1 → center only
///   2 → side by side
///   3 → triangle
///   4 → square arrangement
///   5 → center + 4 corners (quincunx)
///   6+ → center + ring
fn skyrmion_positions(n: usize, lx: f64, ly: f64) -> Vec<(f64, f64)> {
    let mut centers = Vec::with_capacity(n);
    if n == 0 {
        return centers;
    }

    let margin = 0.2; // fraction of half-width to stay away from edge
    let rx = (0.5 - margin) * lx; // usable half-width
    let ry = (0.5 - margin) * ly;

    match n {
        1 => {
            centers.push((0.0, 0.0));
        }
        2 => {
            centers.push((-rx * 0.5, 0.0));
            centers.push((rx * 0.5, 0.0));
        }
        3 => {
            centers.push((0.0, ry * 0.5));
            centers.push((-rx * 0.5, -ry * 0.4));
            centers.push((rx * 0.5, -ry * 0.4));
        }
        4 => {
            let s = 0.55;
            centers.push((-rx * s, -ry * s));
            centers.push((rx * s, -ry * s));
            centers.push((-rx * s, ry * s));
            centers.push((rx * s, ry * s));
        }
        5 => {
            // Quincunx: center + 4 at diagonal positions
            centers.push((0.0, 0.0));
            let s = 0.65;
            centers.push((-rx * s, -ry * s));
            centers.push((rx * s, -ry * s));
            centers.push((-rx * s, ry * s));
            centers.push((rx * s, ry * s));
        }
        _ => {
            // Center + N-1 evenly around a ring
            centers.push((0.0, 0.0));
            let ring_n = n - 1;
            let ring_r = rx.min(ry) * 0.65;
            for k in 0..ring_n {
                let angle = 2.0 * PI * k as f64 / ring_n as f64;
                centers.push((ring_r * angle.cos(), ring_r * angle.sin()));
            }
        }
    }

    centers
}

// =====================================================================
// Main
// =====================================================================

fn main() {
    // ---- CLI flags ----
    let args: Vec<String> = std::env::args().collect();
    let do_plots = args.iter().any(|a| a == "--plots");
    let do_ovf = args.iter().any(|a| a == "--ovf");
    let do_fine = !args.iter().any(|a| a == "--no-fine");

    let out_dir = "out/amr_skyrmion_relax";
    ensure_dir(out_dir);

    // ---- Tunable parameters (env-var overridable) ----
    let amr_max_level: usize = env_or("LLG_AMR_MAX_LEVEL", 3);
    let ratio = 2usize;
    let ghost = 2usize;

    // ---- Physical domain ----
    // Square thin film: 1µm × 1µm × 1nm  (large enough for well-separated skyrmions)
    let lx: f64 = env_or("LLG_SKY_LX", 1000.0e-9); // 1 µm
    let ly: f64 = env_or("LLG_SKY_LY", 1000.0e-9); // 1 µm
    let dz: f64 = env_or("LLG_SKY_DZ", 1.0e-9); // 1 nm (ultrathin film)

    let base_nx: usize = env_or("LLG_SKY_BASE_NX", 200);
    let base_ny: usize = env_or("LLG_SKY_BASE_NY", 200);
    let dx = lx / base_nx as f64; // 5 nm
    let dy = ly / base_ny as f64; // 5 nm

    let ref_ratio_total = pow_usize(ratio, amr_max_level);
    let fine_nx = base_nx * ref_ratio_total;
    let fine_ny = base_ny * ref_ratio_total;

    // ---- Time stepping ----
    let alpha: f64 = env_or("LLG_SKY_ALPHA", 0.2);  // moderate damping → genuine dynamics
    let dt: f64 = env_or("LLG_SKY_DT", 5.0e-14);
    let steps_base: usize = env_or("LLG_SKY_STEPS", 3000);
    // These may be snapped to multiples of subcycle_ratio later.
    let out_every_base: usize = env_or("LLG_SKY_OUT_EVERY", 200);
    let regrid_every_base: usize = env_or("LLG_SKY_REGRID_EVERY", 100);

    let pbcx = 0usize;
    let pbcy = 0usize;

    // ---- Material: Co/Pt-like PMA thin film with interfacial DMI ----
    let ms: f64 = env_or("LLG_SKY_MS", 5.8e5); // A/m (Co/Pt)
    let a_ex: f64 = env_or("LLG_SKY_AEX", 1.5e-11); // J/m
    let k_u: f64 = env_or("LLG_SKY_KU", 8.0e5); // J/m³ (PMA)
    let dmi_d: f64 = env_or("LLG_SKY_DMI", 3.0e-3); // J/m² (interfacial)
    let bz: f64 = env_or("LLG_SKY_BZ", 0.05); // T (moderate stabilising Zeeman)

    let mat = Material {
        ms,
        a_ex,
        k_u,
        easy_axis: [0.0, 0.0, 1.0], // PMA along ẑ
        dmi: Some(dmi_d),           // Interfacial (Néel) DMI
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let llg = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha,
        dt,
        b_ext: [0.0, 0.0, bz], // Stabilising field along +ẑ
    };

    // ---- Grids ----
    let base_grid = Grid2D::new(base_nx, base_ny, dx, dy, dz);
    let fine_grid = Grid2D::new(
        fine_nx,
        fine_ny,
        dx / ref_ratio_total as f64,
        dy / ref_ratio_total as f64,
        dz,
    );

    // ---- Skyrmion initial condition ----
    let n_skyrmions: usize = env_or("LLG_SKY_N_SKYRMIONS", 5);
    let sky_r0: f64 = env_or("LLG_SKY_R0", 50.0e-9); // 50 nm — oversized → contracts to ~25–35nm eq.
    let sky_wall_w: f64 = env_or("LLG_SKY_WALL_W", 10.0e-9); // 10 nm transition width
    let sky_helicity: f64 = 0.0; // Néel (radial) — matches interfacial DMI convention
    let sky_outer_polarity: f64 = 1.0; // Background mz = +1, skyrmion core mz = –1

    let sky_centers = skyrmion_positions(n_skyrmions, lx, ly);

    // ---- Uniform coarse baseline ----
    let mut m_coarse = VectorField2D::new(base_grid);
    initial_states::seed_smooth_bubbles(
        &mut m_coarse,
        &base_grid,
        &sky_centers,
        sky_r0,
        sky_wall_w,
        sky_helicity,
        sky_outer_polarity,
        None,
    );

    // ---- AMR hierarchy (starts from its own coarse field) ----
    let mut m_coarse_amr = VectorField2D::new(base_grid);
    initial_states::seed_smooth_bubbles(
        &mut m_coarse_amr,
        &base_grid,
        &sky_centers,
        sky_r0,
        sky_wall_w,
        sky_helicity,
        sky_outer_polarity,
        None,
    );

    let mut h = AmrHierarchy2D::new(base_grid, m_coarse_amr, ratio, ghost);

    // ---- Uniform fine reference ----
    let mut m_fine = VectorField2D::new(fine_grid);
    if do_fine {
        initial_states::seed_smooth_bubbles(
            &mut m_fine,
            &fine_grid,
            &sky_centers,
            sky_r0,
            sky_wall_w,
            sky_helicity,
            sky_outer_polarity,
            None,
        );
    }

    // ---- AMR policies ----
    // Composite indicator captures gradient magnitude, in-plane divergence, and curl,
    // all of which are large at skyrmion walls.
    let indicator_kind = if std::env::var("LLG_AMR_INDICATOR").is_ok() {
        IndicatorKind::from_env()
    } else {
        // Default: composite at frac=0.15 — catches all skyrmion walls
        IndicatorKind::Composite { frac: 0.15 }
    };

    let boundary_layer: usize = env_or("LLG_AMR_BOUNDARY_LAYER", 0);

    let buffer_cells = 6usize;
    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind,
        buffer_cells,
        boundary_layer,
        connectivity: Connectivity::Eight,
        merge_distance: 2,         // low → keeps separate skyrmions as separate patches
        min_patch_area: 64,        // filter tiny slivers (< 8×8 coarse cells)
        max_patches: 0,            // no limit — let bisection + efficiency control count
        min_efficiency: 0.70,
        max_flagged_fraction: 0.50, // auto-raise threshold if >50% flagged
    };
    let regrid_policy = RegridPolicy {
        indicator: indicator_kind,
        buffer_cells,
        boundary_layer,
        min_change_cells: 2,
        min_area_change_frac: 0.05,
    };

    // ---- Write run info ----
    write_run_info(
        &format!("{out_dir}/run_info.txt"),
        &base_grid,
        &fine_grid,
        &mat,
        &llg,
        amr_max_level,
        ratio,
        ghost,
        steps_base,
        regrid_every_base,
        indicator_kind,
        n_skyrmions,
        sky_r0,
        sky_wall_w,
    );

    // ---- Log files ----
    let regrid_log_path = format!("{out_dir}/regrid_log.csv");
    let regrid_levels_path = format!("{out_dir}/regrid_levels.csv");
    let regrid_attempts_path = format!("{out_dir}/regrid_attempts.csv");
    let regrid_patches_path = format!("{out_dir}/regrid_patches.csv");
    let rmse_log_path = format!("{out_dir}/rmse_log.csv");
    let energy_log_path = format!("{out_dir}/energy_log.csv");
    let skyrmion_log_path = format!("{out_dir}/skyrmion_log.csv");

    {
        let mut f = File::create(&regrid_log_path).unwrap();
        writeln!(
            f,
            "step,max_indicator,threshold,flagged_cells,patches,union_i0,union_j0,union_nx,union_ny"
        )
        .unwrap();

        let mut f2 = File::create(&rmse_log_path).unwrap();
        writeln!(f2, "step,rmse,max_delta,patches").unwrap();

        let mut f3 = File::create(&regrid_levels_path).unwrap();
        let mut hdr = String::from("step");
        for lvl in 1..=amr_max_level {
            hdr.push_str(&format!(
                ",l{lvl}_count,l{lvl}_i0,l{lvl}_j0,l{lvl}_nx,l{lvl}_ny"
            ));
        }
        hdr.push('\n');
        f3.write_all(hdr.as_bytes()).unwrap();

        let mut f4 = File::create(&regrid_attempts_path).unwrap();
        let mut hdr2 = String::from("step,max_indicator");
        for lvl in 1..=amr_max_level {
            hdr2.push_str(&format!(",l{lvl}_count"));
        }
        hdr2.push('\n');
        f4.write_all(hdr2.as_bytes()).unwrap();

        let mut f5 = File::create(&regrid_patches_path).unwrap();
        writeln!(f5, "step,level,patch_id,i0,j0,nx,ny").unwrap();

        let mut f6 = File::create(&energy_log_path).unwrap();
        writeln!(f6, "step,avg_mz_coarse,avg_mz_fine,avg_mz_amr,max_abs_mz_coarse,max_abs_mz_fine,max_abs_mz_amr").unwrap();

        let mut f7 = File::create(&skyrmion_log_path).unwrap();
        writeln!(
            f7,
            "step,Q_fine,Q_amr,n_sk_fine,n_sk_amr,min_mz_fine,min_mz_amr"
        )
        .unwrap();
    }

    // ---- Initial regrid ----
    let mut current_patches: Vec<Rect2i> = Vec::new();
    if let Some((new_rects, stats)) =
        maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
    {
        current_patches = new_rects;
        let u = union_rect_or_zero(&current_patches);
        append_line(
            &regrid_log_path,
            &format!(
                "0,{:.8e},{:.8e},{},{},{},{},{},{}\n",
                stats.max_indicator,
                stats.threshold,
                stats.flagged_cells,
                current_patches.len(),
                u.i0,
                u.j0,
                u.nx,
                u.ny
            ),
        );

        let mut row = String::from("0");
        for lvl in 1..=amr_max_level {
            let rects = level_rects(&h, lvl);
            let uu = union_rect_or_zero(&rects);
            row.push_str(&format!(
                ",{},{},{},{},{}",
                rects.len(),
                uu.i0,
                uu.j0,
                uu.nx,
                uu.ny
            ));
        }
        row.push('\n');
        append_line(&regrid_levels_path, &row);

        let mut row2 = format!("0,{:.8e}", stats.max_indicator);
        for lvl in 1..=amr_max_level {
            row2.push_str(&format!(",{}", level_patch_count(&h, lvl)));
        }
        row2.push('\n');
        append_line(&regrid_attempts_path, &row2);

        append_regrid_patches_csv(&regrid_patches_path, &h, amr_max_level, 0);
    }

    // ---- Fair comparison: start uniform fine from AMR composite at step 0 ----
    if do_fine {
        m_fine = flatten_to_target_grid(&h, fine_grid);
    }

    // ---- Stepper + subcycling wiring ----
    let mut stepper = AmrStepperRK4::new(&h, true);

    // Subcycling awareness: one stepper.step() advances by dt_coarse = dt × subcycle_ratio.
    // We only call stepper.step() at multiples of subcycle_ratio, keeping AMR and reference in sync.
    let subcycle_active = stepper.is_subcycling();
    let subcycle_ratio: usize = if subcycle_active {
        (stepper.coarse_dt(&llg, &h) / llg.dt).round() as usize
    } else {
        1
    };
    // Snap step count and output cadence to multiples of subcycle_ratio so that
    // output and regrid always happen on a coarse-step boundary.
    let snap_up = |v: usize, r: usize| -> usize {
        if r <= 1 { v } else { ((v + r - 1) / r) * r }
    };
    let steps = snap_up(steps_base, subcycle_ratio);
    let out_every = snap_up(out_every_base, subcycle_ratio);
    let regrid_every = snap_up(regrid_every_base, subcycle_ratio);
    if subcycle_active {
        eprintln!(
            "[amr_skyrmion_relax] SUBCYCLING ACTIVE: n_levels={}, subcycle_ratio={}, steps={}, out_every={}, regrid_every={}",
            h.num_levels(), subcycle_ratio, steps, out_every, regrid_every
        );
    }
    if subcycle_active && steps != steps_base {
        eprintln!(
            "[amr_skyrmion_relax] steps adjusted: {} → {} (multiple of subcycle_ratio={})",
            steps_base, steps, subcycle_ratio
        );
    }

    let mut scratch_fine = if do_fine { Some(RK4Scratch::new(fine_grid)) } else { None };
    let mut scratch_coarse = RK4Scratch::new(base_grid);
    let mut b_fine = if do_fine { Some(VectorField2D::new(fine_grid)) } else { None };
    let mut b_coarse = VectorField2D::new(base_grid);

    // Local field mask: Exchange + Anisotropy + DMI (demag added globally via Bridge B)
    let local_mask = FieldMask::ExchAnisDmi;

    // ---- OVF directories ----
    if do_ovf {
        ensure_dir(&format!("{out_dir}/ovf_coarse"));
        if do_fine {
            ensure_dir(&format!("{out_dir}/ovf_fine"));
        }
        ensure_dir(&format!("{out_dir}/ovf_amr"));
    }

    // ---- Banner ----
    let l_ex = (a_ex / (MU_0 * ms * ms)).sqrt();
    let k_eff = k_u - 0.5 * MU_0 * ms * ms;
    let delta_w = if k_eff > 0.0 {
        (a_ex / k_eff).sqrt()
    } else {
        f64::INFINITY
    };
    let kappa = if k_eff > 0.0 && a_ex > 0.0 {
        PI * dmi_d / (4.0 * (a_ex * k_eff).sqrt())
    } else {
        0.0
    };

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  AMR Skyrmion Bubble Relaxation Benchmark                    ║");
    println!("║  Co/Pt-like PMA film · Interfacial DMI · All 5 field terms   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Domain:  {:.0}nm × {:.0}nm × {:.1}nm   (thin film)",
        lx * 1e9,
        ly * 1e9,
        dz * 1e9
    );
    println!(
        "Base:    {} × {}   dx={:.3e}  dy={:.3e}  dz={:.3e}",
        base_nx, base_ny, dx, dy, dz
    );
    println!(
        "Fine:    {} × {}   dx={:.3e}  dy={:.3e}",
        fine_nx, fine_ny, fine_grid.dx, fine_grid.dy
    );
    println!(
        "AMR:     {} levels, ratio={}, ghost={}",
        amr_max_level, ratio, ghost
    );
    println!();
    println!(
        "Material: Ms={:.2e}  A_ex={:.2e}  K_u={:.2e}  DMI={:.2e}",
        ms, a_ex, k_u, dmi_d
    );
    println!(
        "  l_ex = {:.2} nm   δ_wall = {:.2} nm   κ = {:.3}",
        l_ex * 1e9,
        delta_w * 1e9,
        kappa
    );
    println!(
        "  K_eff = {:.3e} J/m³  (PMA dominant: K_u > µ₀Ms²/2)",
        k_eff
    );
    println!(
        "  finest dx/l_ex = {:.2}   dx/δ = {:.2}",
        fine_grid.dx / l_ex,
        fine_grid.dx / delta_w
    );
    println!();
    println!(
        "LLG:     α={:.4}  dt={:.2e}  steps={}  total={:.3} ns",
        alpha,
        dt,
        steps,
        steps as f64 * dt * 1e9
    );
    println!("Zeeman:  Bz = {:.3} T  (stabilising field along +ẑ)", bz);
    println!("Fields:  Exchange + Anisotropy + DMI (local) + Demag (FFT) + Zeeman");
    if subcycle_active {
        println!(
            "Subcycling: ON  ratio={}  dt_coarse={:.2e}  coarse_steps={}",
            subcycle_ratio,
            subcycle_ratio as f64 * dt,
            steps / subcycle_ratio.max(1)
        );
    } else {
        println!("Subcycling: OFF");
    }
    println!();
    println!(
        "Skyrmions: {} bubbles, R₀ = {:.0} nm, wall_w = {:.0} nm, Néel helicity",
        n_skyrmions,
        sky_r0 * 1e9,
        sky_wall_w * 1e9
    );
    for (k, &(cx, cy)) in sky_centers.iter().enumerate() {
        println!(
            "  Sk{}: center = ({:.1} nm, {:.1} nm)",
            k,
            cx * 1e9,
            cy * 1e9
        );
    }
    println!();
    println!(
        "Indicator: {} (threshold={:.4})",
        indicator_kind.label(),
        indicator_kind.threshold_param()
    );
    println!("Regrid every {} steps", regrid_every);
    println!("Output:  {out_dir}");
    if do_plots {
        println!("  --plots enabled");
    }
    if do_ovf {
        println!("  --ovf enabled");
    }
    if !do_fine {
        println!("  --no-fine: uniform fine reference SKIPPED (AMR + coarse only)");
    }
    println!();

    // ---- Timings ----
    let t0 = Instant::now();
    let mut t_demag_fine = 0.0;
    let mut t_demag_coarse = 0.0;
    let mut t_amr_step = 0.0;

    // ---- Step 0 outputs ----
    {
        let m_amr_fine = flatten_to_target_grid(&h, fine_grid);
        let (rmse, maxd) = if do_fine {
            rmse_and_max_delta(&m_amr_fine, &m_fine)
        } else {
            (f64::NAN, f64::NAN)
        };
        append_line(
            &rmse_log_path,
            &format!("0,{:.8e},{:.8e},{}\n", rmse, maxd, current_patches.len()),
        );

        let (avg_mz_fine_val, max_mz_fine_val) = if do_fine {
            (avg_mz(&m_fine), max_abs_mz(&m_fine))
        } else {
            (f64::NAN, f64::NAN)
        };
        append_line(
            &energy_log_path,
            &format!(
                "0,{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}\n",
                avg_mz(&m_coarse),
                avg_mz_fine_val,
                avg_mz(&m_amr_fine),
                max_abs_mz(&m_coarse),
                max_mz_fine_val,
                max_abs_mz(&m_amr_fine),
            ),
        );

        let q_fine = if do_fine { topological_charge(&m_fine) } else { f64::NAN };
        let q_amr = topological_charge(&m_amr_fine);
        let nsk_fine = if do_fine { count_skyrmion_cores(&m_fine, 0.0) } else { 0 };
        let nsk_amr = count_skyrmion_cores(&m_amr_fine, 0.0);
        let min_mz_fine_val = if do_fine { min_mz(&m_fine) } else { f64::NAN };
        append_line(
            &skyrmion_log_path,
            &format!(
                "0,{:.6},{:.6},{},{},{:.6},{:.6}\n",
                q_fine,
                q_amr,
                nsk_fine,
                nsk_amr,
                min_mz_fine_val,
                min_mz(&m_amr_fine),
            ),
        );

        if do_ovf {
            write_ovf_text(
                &format!("{out_dir}/ovf_coarse/m0000000.ovf"),
                &m_coarse,
                "m_coarse",
            );
            if do_fine {
                write_ovf_text(
                    &format!("{out_dir}/ovf_fine/m0000000.ovf"),
                    &m_fine,
                    "m_fine",
                );
            }
            write_ovf_text(
                &format!("{out_dir}/ovf_amr/m0000000.ovf"),
                &m_amr_fine,
                "m_amr",
            );
        }

        if do_plots {
            let levels = all_level_rects(&h, amr_max_level);
            save_patch_map(
                &base_grid,
                &levels,
                &format!("{out_dir}/patch_map_step0000.png"),
                "Patch map (step 0)",
            )
            .unwrap();
            save_mz_map(
                &m_amr_fine,
                &format!("{out_dir}/mz_amr_step0000.png"),
                "mz (AMR, step 0)",
            )
            .unwrap();
            save_mz_map(
                &m_coarse,
                &format!("{out_dir}/mz_coarse_step0000.png"),
                "mz (coarse, step 0)",
            )
            .unwrap();
            save_mesh_zoom_multilevel(
                &m_amr_fine,
                &base_grid,
                ratio,
                amr_max_level,
                &levels,
                &format!("{out_dir}/mesh_zoom_step0000.png"),
                "In-plane angle + grid (step 0)",
            )
            .unwrap();
        }

        println!(
            "step     0/{} | Q_fine={:.3} Q_amr={:.3} | #sk_fine={} #sk_amr={} | <mz>={:.4} | patches={}",
            steps,
            q_fine,
            q_amr,
            nsk_fine,
            nsk_amr,
            avg_mz(&m_amr_fine),
            current_patches.len()
        );
    }

    // =====================================================================
    // Time loop
    // =====================================================================
    for step in 1..=steps {
        // ---- Uniform fine: demag + relax step ----
        if do_fine {
            let bf = b_fine.as_mut().unwrap();
            let sf = scratch_fine.as_mut().unwrap();
            let t1 = Instant::now();
            bf.set_uniform(0.0, 0.0, 0.0);
            demag_fft_uniform::compute_demag_field_pbc(
                &fine_grid,
                &m_fine,
                bf,
                &mat,
                pbcx,
                pbcy,
            );
            t_demag_fine += t1.elapsed().as_secs_f64();
            step_llg_rk4_recompute_field_masked_relax_add(
                &mut m_fine,
                &llg,
                &mat,
                sf,
                local_mask,
                Some(bf),
            );
        }

        // ---- Uniform coarse: demag + relax step (baseline) ----
        let t2 = Instant::now();
        b_coarse.set_uniform(0.0, 0.0, 0.0);
        demag_fft_uniform::compute_demag_field_pbc(
            &base_grid,
            &m_coarse,
            &mut b_coarse,
            &mat,
            pbcx,
            pbcy,
        );
        t_demag_coarse += t2.elapsed().as_secs_f64();
        step_llg_rk4_recompute_field_masked_relax_add(
            &mut m_coarse,
            &llg,
            &mat,
            &mut scratch_coarse,
            local_mask,
            Some(&b_coarse),
        );

        // ---- AMR step (Bridge B: FFT demag on flattened fine composite) ----
        // With subcycling, one stepper.step() advances by dt_coarse = dt × subcycle_ratio.
        // We only call it at multiples of subcycle_ratio, keeping AMR and reference in sync.
        let amr_due = step % subcycle_ratio == 0;
        if amr_due {
            let t3 = Instant::now();
            stepper.step(&mut h, &llg, &mat, local_mask);
            t_amr_step += t3.elapsed().as_secs_f64();
        }

        // ---- Regrid periodically (only on coarse-step boundaries) ----
        if amr_due && regrid_every > 0 && step % regrid_every == 0 {
            let mut row2 = format!("{},{:.8e}", step, 0.0f64);
            for lvl in 1..=amr_max_level {
                row2.push_str(&format!(",{}", level_patch_count(&h, lvl)));
            }
            row2.push('\n');
            append_line(&regrid_attempts_path, &row2);

            if let Some((new_rects, stats)) =
                maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
            {
                current_patches = new_rects;
                let u = union_rect_or_zero(&current_patches);
                append_line(
                    &regrid_log_path,
                    &format!(
                        "{},{:.8e},{:.8e},{},{},{},{},{},{}\n",
                        step,
                        stats.max_indicator,
                        stats.threshold,
                        stats.flagged_cells,
                        current_patches.len(),
                        u.i0,
                        u.j0,
                        u.nx,
                        u.ny
                    ),
                );

                let mut row = format!("{}", step);
                for lvl in 1..=amr_max_level {
                    let rects = level_rects(&h, lvl);
                    let uu = union_rect_or_zero(&rects);
                    row.push_str(&format!(
                        ",{},{},{},{},{}",
                        rects.len(),
                        uu.i0,
                        uu.j0,
                        uu.nx,
                        uu.ny
                    ));
                }
                row.push('\n');
                append_line(&regrid_levels_path, &row);

                append_regrid_patches_csv(&regrid_patches_path, &h, amr_max_level, step);
            }
        }

        // ---- Diagnostics output ----
        if step % out_every == 0 || step == steps {
            let m_amr_fine = flatten_to_target_grid(&h, fine_grid);
            let (rmse, maxd) = if do_fine {
                rmse_and_max_delta(&m_amr_fine, &m_fine)
            } else {
                (f64::NAN, f64::NAN)
            };

            append_line(
                &rmse_log_path,
                &format!(
                    "{},{:.8e},{:.8e},{}\n",
                    step,
                    rmse,
                    maxd,
                    current_patches.len()
                ),
            );

            let (avg_mz_fine_val, max_mz_fine_val) = if do_fine {
                (avg_mz(&m_fine), max_abs_mz(&m_fine))
            } else {
                (f64::NAN, f64::NAN)
            };
            append_line(
                &energy_log_path,
                &format!(
                    "{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}\n",
                    step,
                    avg_mz(&m_coarse),
                    avg_mz_fine_val,
                    avg_mz(&m_amr_fine),
                    max_abs_mz(&m_coarse),
                    max_mz_fine_val,
                    max_abs_mz(&m_amr_fine),
                ),
            );

            // Skyrmion diagnostics (on fine and AMR)
            let q_fine = if do_fine { topological_charge(&m_fine) } else { f64::NAN };
            let q_amr = topological_charge(&m_amr_fine);
            let nsk_fine = if do_fine { count_skyrmion_cores(&m_fine, 0.0) } else { 0 };
            let nsk_amr = count_skyrmion_cores(&m_amr_fine, 0.0);
            let min_mz_fine_val = if do_fine { min_mz(&m_fine) } else { f64::NAN };
            append_line(
                &skyrmion_log_path,
                &format!(
                    "{},{:.6},{:.6},{},{},{:.6},{:.6}\n",
                    step,
                    q_fine,
                    q_amr,
                    nsk_fine,
                    nsk_amr,
                    min_mz_fine_val,
                    min_mz(&m_amr_fine),
                ),
            );

            if do_ovf {
                let fname = format!("m{:07}.ovf", step);
                write_ovf_text(
                    &format!("{out_dir}/ovf_coarse/{fname}"),
                    &m_coarse,
                    "m_coarse",
                );
                if do_fine {
                    write_ovf_text(&format!("{out_dir}/ovf_fine/{fname}"), &m_fine, "m_fine");
                }
                write_ovf_text(&format!("{out_dir}/ovf_amr/{fname}"), &m_amr_fine, "m_amr");
            }

            let mut lvl_counts = String::new();
            for lvl in 1..=amr_max_level {
                if lvl > 1 {
                    lvl_counts.push_str(" | ");
                }
                lvl_counts.push_str(&format!("L{} {:2}", lvl, level_patch_count(&h, lvl)));
            }
            let t_elapsed = t0.elapsed().as_secs_f64();
            println!(
                "step {:5}/{} | rmse {:.3e} | Q={:.3} #sk={} | <mz> {:.4} | {} | fine={:.0}s coarse={:.0}s amr={:.0}s | {:.1}s",
                step,
                steps,
                rmse,
                q_amr,
                nsk_amr,
                avg_mz(&m_amr_fine),
                lvl_counts,
                t_demag_fine,
                t_demag_coarse,
                t_amr_step,
                t_elapsed,
            );

            // Plots at every output step
            if do_plots {
                let levels = all_level_rects(&h, amr_max_level);

                save_patch_map(
                    &base_grid,
                    &levels,
                    &format!("{out_dir}/patch_map_step{step:04}.png"),
                    &format!("Patch map (step {})", step),
                )
                .unwrap();

                save_mz_map(
                    &m_amr_fine,
                    &format!("{out_dir}/mz_amr_step{step:04}.png"),
                    &format!("mz (AMR composite, step {})", step),
                )
                .unwrap();

                save_mz_map(
                    &m_coarse,
                    &format!("{out_dir}/mz_coarse_step{step:04}.png"),
                    &format!("mz (coarse {base_nx}×{base_ny}, step {step})"),
                )
                .unwrap();

                save_mesh_zoom_multilevel(
                    &m_amr_fine,
                    &base_grid,
                    ratio,
                    amr_max_level,
                    &levels,
                    &format!("{out_dir}/mesh_zoom_step{step:04}.png"),
                    &format!("In-plane angle + grid (step {})", step),
                )
                .unwrap();
            }
        }
    }

    let wall = t0.elapsed().as_secs_f64();

    // =====================================================================
    // Final outputs
    // =====================================================================
    let m_amr_fine_final = flatten_to_target_grid(&h, fine_grid);
    write_csv_ij_m(&format!("{out_dir}/uniform_coarse_final.csv"), &m_coarse);
    if do_fine {
        write_csv_ij_m(&format!("{out_dir}/uniform_fine_final.csv"), &m_fine);
    }
    write_csv_ij_m(&format!("{out_dir}/amr_fine_final.csv"), &m_amr_fine_final);

    if do_fine {
        write_midline_y(&format!("{out_dir}/lineout_uniform_mid_y.csv"), &m_fine);
    }
    write_midline_y(
        &format!("{out_dir}/lineout_amr_mid_y.csv"),
        &m_amr_fine_final,
    );
    if do_fine {
        write_midline_x(&format!("{out_dir}/lineout_uniform_mid_x.csv"), &m_fine);
    }
    write_midline_x(
        &format!("{out_dir}/lineout_amr_mid_x.csv"),
        &m_amr_fine_final,
    );

    let fine_patches = patches_to_fine_rects(&current_patches, ref_ratio_total);
    let fine_cells_total = fine_grid.nx * fine_grid.ny;
    let fine_cells_in_patches: usize = fine_patches.iter().map(|r| r.nx * r.ny).sum();
    let coverage = fine_cells_in_patches as f64 / fine_cells_total as f64;

    let (rmse_final, maxd_final) = if do_fine {
        rmse_and_max_delta(&m_amr_fine_final, &m_fine)
    } else {
        (f64::NAN, f64::NAN)
    };
    let q_final_fine = if do_fine { topological_charge(&m_fine) } else { f64::NAN };
    let q_final_amr = topological_charge(&m_amr_fine_final);
    let nsk_final_fine = if do_fine { count_skyrmion_cores(&m_fine, 0.0) } else { 0 };
    let nsk_final_amr = count_skyrmion_cores(&m_amr_fine_final, 0.0);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  AMR Skyrmion Bubble Relaxation — Final Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "Domain:  {:.0}nm × {:.0}nm × {:.1}nm",
        lx * 1e9,
        ly * 1e9,
        dz * 1e9
    );
    println!("Base grid:    {base_nx} × {base_ny}   dx={dx:.3e}  dy={dy:.3e}  dz={dz:.3e}");
    println!(
        "Fine grid:    {fine_nx} × {fine_ny}   dx={:.3e}  dy={:.3e}",
        fine_grid.dx, fine_grid.dy
    );
    println!("AMR levels:   {}   ratio={}", amr_max_level, ratio);
    if subcycle_active {
        println!(
            "Subcycling:   ON  ratio={}  dt_coarse={:.2e}  coarse_steps={}",
            subcycle_ratio,
            subcycle_ratio as f64 * dt,
            steps / subcycle_ratio.max(1)
        );
    }
    println!();
    println!("Material: Ms={ms:.2e}  A_ex={a_ex:.2e}  K_u={k_u:.2e}  DMI={dmi_d:.2e}");
    println!(
        "  K_eff={k_eff:.3e}  l_ex={:.2}nm  δ={:.2}nm  κ={kappa:.3}",
        l_ex * 1e9,
        delta_w * 1e9
    );
    println!();
    println!(
        "Steps: {steps}   dt={dt:.3e}   total_time={:.3} ns   Bz={bz:.3} T",
        steps as f64 * dt * 1e9
    );
    println!("α = {alpha:.4}");
    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Skyrmion Physics                                       ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!(
        "║  Topological charge Q (fine):    {:>8.4}  (expect ≈ –{})  ║",
        q_final_fine, n_skyrmions
    );
    println!(
        "║  Topological charge Q (AMR):     {:>8.4}                 ║",
        q_final_amr
    );
    println!(
        "║  Skyrmion count (fine):          {:>3}       (seeded: {})   ║",
        nsk_final_fine, n_skyrmions
    );
    println!(
        "║  Skyrmion count (AMR):           {:>3}                      ║",
        nsk_final_amr
    );
    if do_fine {
        println!(
            "║  min(mz) fine:                   {:>8.4}  (core depth)    ║",
            min_mz(&m_fine)
        );
    }
    println!(
        "║  min(mz) AMR:                    {:>8.4}                  ║",
        min_mz(&m_amr_fine_final)
    );
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Final RMSE(|Δm|):  {:.6e}", rmse_final);
    println!("Final max |Δm|:    {:.6e}", maxd_final);
    println!("Final <mz> coarse: {:.6}", avg_mz(&m_coarse));
    if do_fine {
        println!("Final <mz> fine:   {:.6}", avg_mz(&m_fine));
    }
    println!("Final <mz> AMR:    {:.6}", avg_mz(&m_amr_fine_final));
    println!();
    println!("Fine cells (uniform):   {fine_cells_total}");
    println!("Fine cells in patches:  {fine_cells_in_patches}");
    println!(
        "Patch coverage fraction: {:.4} ({:.1}%)",
        coverage,
        coverage * 100.0
    );
    println!("  → Expected: ~5–15% (skyrmion walls + buffer only)");
    println!();
    println!("Timing:");
    println!("  total wall time:       {:.3} s", wall);
    println!(
        "  fine demag FFT:        {:.3} s ({:.1}%)",
        t_demag_fine,
        100.0 * t_demag_fine / wall
    );
    println!(
        "  coarse demag FFT:      {:.3} s ({:.1}%)",
        t_demag_coarse,
        100.0 * t_demag_coarse / wall
    );
    println!(
        "  AMR step (incl. demag):{:.3} s ({:.1}%)",
        t_amr_step,
        100.0 * t_amr_step / wall
    );
    let other = (wall - t_demag_fine - t_demag_coarse - t_amr_step).max(0.0);
    println!(
        "  other/overhead:        {:.3} s ({:.1}%)",
        other,
        100.0 * other / wall
    );
    println!();
    println!("Outputs: {out_dir}");
}