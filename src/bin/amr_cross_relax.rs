// src/bin/amr_cross_relax.rs
//
// AMR benchmark: cross-domain-wall relaxation on a disk
// =====================================================
//
// Tests the AMR pipeline with Berger-Colella v2 subcycling on a
// Permalloy thin-film disk initialised with four 90° Néel domain walls
// in a cross pattern.  This is a better AMR showcase than the diamond
// benchmark because:
//
//   1. Strong gradients from step 0 → patches immediately refine the walls
//   2. Features evolve: walls smooth, vortex core nucleates, patches track
//   3. Circular disk exercises the geometry_mask analytical shape system
//   4. Visually striking: four distinct in-plane angle regions on the disk
//
// Physics:
//   - Exchange + Demag (FFT uniform via Bridge B)
//   - No anisotropy, no DMI, no Zeeman — simplest setup for domain walls
//   - Permalloy soft magnet: Ms = 8×10⁵ A/m, A_ex = 1.3×10⁻¹¹ J/m
//   - Thin film: dz = 10 nm → moderate shape anisotropy keeps m in-plane
//   - α = 0.1 (moderate damping → rich precessional dynamics)
//
// Initial condition:
//   Four-quadrant "cross" pattern in centred coordinates on a disk:
//     Q1 (+x, +y) → m = +x̂
//     Q2 (−x, +y) → m = +ŷ
//     Q3 (−x, −y) → m = −x̂
//     Q4 (+x, −y) → m = −ŷ
//   This creates four sharp 90° Néel domain walls along the x and y axes.
//   Small random perturbation in mz (ε = 0.01) seeds vortex core polarity.
//   Outside the disk: m = (0, 0, 0) via geometry mask (vacuum).
//
// Expected dynamics:
//   0–500 steps:   Sharp walls create large exchange torque → rapid smoothing.
//                   AMR patches concentrate on four wall regions.
//   500–2000:      Walls relax to smooth 90° Néel profiles.  Vortex core
//                   nucleates at the centre (intersection of all four walls).
//                   Patches track narrowing wall profiles + core.
//   2000–5000:     Precessional ringing damps out.  System settles toward
//                   Landau flux-closure ground state (4 domains, 4 walls,
//                   single vortex core).  Patches shrink as gradients diminish.
//
// Grid setup:
//   - Base (level 0): 128×128 cells → dx = dy ≈ 3.91 nm  (500 nm domain)
//   - AMR: 3 levels (default), ratio=2 → finest 1024×1024, dx_fine ≈ 0.49 nm
//   - Exchange length l_ex ≈ 4.0 nm → ~8 cells at finest level (well resolved)
//   - Boundary layer: 2 cells (flags disk edge for fine resolution → accurate RMSE)
//   - Subcycle ratio: 8 (3 levels of ratio-2)
//
// Outputs in out/amr_cross_relax:
//   - patch_map_stepXXXX.png              : patch rectangles by refinement level [--plots]
//   - mesh_zoom_stepXXXX.png              : in-plane angle + multi-level grid overlay [--plots]
//   - angle_amr_stepXXXX.png              : full-domain in-plane angle [--plots]
//   - regrid_log.csv                      : accepted regrid events
//   - regrid_levels.csv                   : per-accept per-level summaries
//   - regrid_attempts.csv                 : per-check diagnostics
//   - regrid_patches.csv                  : per-patch rectangles
//   - rmse_log.csv                        : AMR vs uniform fine RMSE vs step
//   - energy_log.csv                      : magnetisation diagnostics vs step
//   - ovf_coarse/mXXXXXXX.ovf            : coarse OVFs [--ovf]
//   - ovf_fine/mXXXXXXX.ovf              : fine reference OVFs [--ovf]
//   - ovf_amr/mXXXXXXX.ovf               : AMR composite OVFs [--ovf]
//   - *_final.csv                         : final states
//   - lineout_*_mid_y.csv                 : midline profiles
//   - run_info.txt                        : full parameter dump
//
// Run examples:
//   # Default (2 AMR levels, 5000 steps, --no-fine for speed):
//   cargo run --release --bin amr_cross_relax -- --plots --no-fine
//
//   # Full benchmark with RMSE comparison:
//   cargo run --release --bin amr_cross_relax -- --plots
//
//   # Custom parameters:
//   LLG_CROSS_ALPHA=0.2 LLG_CROSS_STEPS=3000 \
//     cargo run --release --bin amr_cross_relax -- --plots --no-fine

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use plotters::prelude::*;

use llg_sim::effective_field::{FieldMask, demag_fft_uniform};
use llg_sim::geometry_mask::{MaskShape, cell_center_xy_centered};
use llg_sim::grid::Grid2D;
use llg_sim::llg::{RK4Scratch, step_llg_rk4_recompute_field_masked_relax_add};
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

use llg_sim::amr::indicator::{IndicatorKind, indicator_grad2_forward};
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

fn append_line(path: &str, line: &str) {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    f.write_all(line.as_bytes()).unwrap();
}

// =====================================================================
// OVF / CSV I/O (matching skyrmion benchmark)
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
    writeln!(f, "# meshtype: rectangular").unwrap();
    writeln!(f, "# meshunit: m").unwrap();
    writeln!(f, "# xbase: {:.16e}", 0.5 * dx).unwrap();
    writeln!(f, "# ybase: {:.16e}", 0.5 * dy).unwrap();
    writeln!(f, "# zbase: {:.16e}", 0.5 * dz).unwrap();
    writeln!(f, "# xstepsize: {:.16e}", dx).unwrap();
    writeln!(f, "# ystepsize: {:.16e}", dy).unwrap();
    writeln!(f, "# zstepsize: {:.16e}", dz).unwrap();
    writeln!(f, "# xnodes: {}", nx).unwrap();
    writeln!(f, "# ynodes: {}", ny).unwrap();
    writeln!(f, "# znodes: 1").unwrap();
    writeln!(f, "# valuedim: 3").unwrap();
    writeln!(f, "# End: Header").unwrap();
    writeln!(f, "# Begin: Data Text").unwrap();

    for j in 0..ny {
        for i in 0..nx {
            let v = m.data[m.grid.idx(i, j)];
            writeln!(f, "{:.16e} {:.16e} {:.16e}", v[0], v[1], v[2]).unwrap();
        }
    }

    writeln!(f, "# End: Data Text").unwrap();
    writeln!(f, "# End: Segment").unwrap();
}

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

// =====================================================================
// Magnetisation diagnostics
// =====================================================================

fn avg_mz(m: &VectorField2D) -> f64 {
    let n = m.data.len() as f64;
    m.data.iter().map(|v| v[2]).sum::<f64>() / n
}

fn max_abs_mz(m: &VectorField2D) -> f64 {
    m.data.iter().map(|v| v[2].abs()).fold(0.0_f64, f64::max)
}

fn avg_m_magnitude(m: &VectorField2D) -> f64 {
    let n = m.data.len() as f64;
    m.data.iter().map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt()).sum::<f64>() / n
}

/// Count magnetic cells (|m| > 0.01) for masked geometry diagnostics.
fn count_magnetic_cells(m: &VectorField2D) -> usize {
    m.data.iter().filter(|v| {
        (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt() > 0.01
    }).count()
}

// =====================================================================
// Cross-domain-wall initial condition
// =====================================================================

/// Initialise four-quadrant cross pattern on a disk.
///
/// In centred coordinates (x, y) relative to grid centre:
///   Q1 (+x, +y) → m = +x̂
///   Q2 (−x, +y) → m = +ŷ
///   Q3 (−x, −y) → m = −x̂
///   Q4 (+x, −y) → m = −ŷ
///
/// This creates four sharp 90° Néel domain walls along the axes.
/// Cells outside the disk mask are set to m = (0,0,0).
/// A small random mz perturbation (amplitude ε) seeds vortex core polarity.
fn init_cross_pattern(
    m: &mut VectorField2D,
    grid: &Grid2D,
    shape: &MaskShape,
    perturb_amp: f64,
) {
    // Deterministic pseudo-random: hash-based for reproducibility.
    let rand_f64 = |seed: usize| -> f64 {
        let h = seed.wrapping_mul(2654435761).wrapping_add(1013904243);
        let h = (h >> 16) ^ h;
        let h = h.wrapping_mul(2246822519).wrapping_add(3266489917);
        let h = (h >> 13) ^ h;
        (h as f64 / (usize::MAX as f64)) * 2.0 - 1.0
    };

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let (x, y) = cell_center_xy_centered(grid, i, j);
            let k = idx(i, j, grid.nx);

            if !shape.contains(x, y) {
                // Vacuum cell: outside disk
                m.data[k] = [0.0, 0.0, 0.0];
                continue;
            }

            // Determine quadrant from centred coordinates.
            // Use (x, y) directly; cells exactly on axes get assigned to
            // the positive-x or positive-y quadrant (arbitrary tiebreak).
            let (mx, my) = if x >= 0.0 && y >= 0.0 {
                // Q1: +x̂
                (1.0, 0.0)
            } else if x < 0.0 && y >= 0.0 {
                // Q2: +ŷ
                (0.0, 1.0)
            } else if x < 0.0 && y < 0.0 {
                // Q3: −x̂
                (-1.0, 0.0)
            } else {
                // Q4: −ŷ
                (0.0, -1.0)
            };

            // Small random mz perturbation breaks symmetry for vortex core selection.
            let mz = perturb_amp * rand_f64(k * 3 + 2);
            let len = (mx * mx + my * my + mz * mz).sqrt();
            m.data[k] = [mx / len, my / len, mz / len];
        }
    }
}

// =====================================================================
// AMR hierarchy utilities (matching skyrmion benchmark)
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

fn append_regrid_patches_csv(path: &str, h: &AmrHierarchy2D, max_level: usize, step: usize) {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();

    // Level 1
    for (pid, p) in h.patches.iter().enumerate() {
        let r = p.coarse_rect;
        writeln!(
            f,
            "{},{},{},{},{},{},{}",
            step, 1, pid, r.i0, r.j0, r.nx, r.ny
        )
        .unwrap();
    }

    // Levels 2+
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

fn max_indicator_coarse(coarse: &VectorField2D) -> f64 {
    let mut max_ind = 0.0_f64;
    for j in 0..coarse.grid.ny {
        for i in 0..coarse.grid.nx {
            max_ind = max_ind.max(indicator_grad2_forward(coarse, i, j));
        }
    }
    max_ind
}

fn rmse_and_max_delta(a: &VectorField2D, b: &VectorField2D) -> (f64, f64) {
    assert_eq!(a.grid.nx, b.grid.nx);
    assert_eq!(a.grid.ny, b.grid.ny);

    let mut s2: f64 = 0.0;
    let mut maxd: f64 = 0.0;
    let n = (a.grid.nx * a.grid.ny) as f64;

    for k in 0..a.data.len() {
        let da = a.data[k];
        let db = b.data[k];
        let dx = da[0] - db[0];
        let dy = da[1] - db[1];
        let dz = da[2] - db[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        let d = d2.sqrt();
        s2 += d2;
        if d > maxd {
            maxd = d;
        }
    }
    ((s2 / n).sqrt(), maxd)
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

/// In-plane angle colour with vacuum handling.
/// Vacuum cells (|m| < 0.01) → dark grey.
/// Magnetic cells → HSV hue from atan2(my, mx), saturation reduced by |mz|.
fn angle_color(v: &[f64; 3]) -> RGBColor {
    let mag2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if mag2 < 1e-4 {
        // Vacuum
        return RGBColor(60, 60, 60);
    }
    let phi = v[1].atan2(v[0]);
    let hue = (phi + PI) / (2.0 * PI);
    let mz_frac = v[2].abs();
    let sat = (1.0 - mz_frac).max(0.0);
    hsv_to_rgb(hue, sat, 1.0)
}

// =====================================================================
// Plot helpers (matching two_bubbles style)
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

/// Full-domain in-plane angle colour map (no grid overlay).
fn save_angle_map(
    m: &VectorField2D,
    path: &str,
    caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = m.grid.nx;
    let ny = m.grid.ny;

    let root = BitMapBackend::new(path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 22))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..nx as i32, 0..ny as i32)?;

    chart.configure_mesh().disable_mesh().draw()?;

    chart.draw_series((0..ny).flat_map(|j| {
        (0..nx).map(move |i| {
            let v = m.data[idx(i, j, nx)];
            let col = angle_color(&v);
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    root.present()?;
    Ok(())
}

/// Mesh zoom: in-plane angle background + multi-level grid overlay.
/// Matches the two_bubbles save_mesh_zoom_multilevel style:
///   - L0 coarse grid in light grey
///   - L1 patches + internal grid in black
///   - L2 patches + internal grid in red
///   - L3+ outline only in blue/purple
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

    // Background: in-plane angle colour map with vacuum handling
    chart.draw_series((0..ny).flat_map(|j| {
        (0..nx).map(move |i| {
            let v = m_fine.data[idx(i, j, nx)];
            let col = angle_color(&v);
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

    // L1, L2: solid boundary + internal grid lines
    for (k, rects) in levels.iter().enumerate() {
        let lvl = k + 1;
        if lvl > 2 {
            break; // L3+ handled separately
        }
        let sp = level_spacing(lvl);
        let col = colors[k.min(colors.len() - 1)];

        for r in rects {
            let fi0 = r.i0 * ref_ratio_total;
            let fj0 = r.j0 * ref_ratio_total;
            let fi1 = (r.i0 + r.nx) * ref_ratio_total;
            let fj1 = (r.j0 + r.ny) * ref_ratio_total;

            // Patch boundary (thick)
            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (fi0 as i32, fj0 as i32),
                    (fi1 as i32, fj0 as i32),
                    (fi1 as i32, fj1 as i32),
                    (fi0 as i32, fj1 as i32),
                    (fi0 as i32, fj0 as i32),
                ],
                col.stroke_width(3),
            )))?;

            // Internal grid lines (thin)
            let mut xg = fi0 + sp;
            while xg < fi1 {
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(xg as i32, fj0 as i32), (xg as i32, fj1 as i32)],
                    col.stroke_width(1),
                )))?;
                xg += sp;
            }
            let mut yg = fj0 + sp;
            while yg < fj1 {
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(fi0 as i32, yg as i32), (fi1 as i32, yg as i32)],
                    col.stroke_width(1),
                )))?;
                yg += sp;
            }
        }
    }

    // L3+ dashed outlines only
    for (k, rects) in levels.iter().enumerate() {
        let lvl = k + 1;
        if lvl < 3 {
            continue;
        }
        let c = if lvl == 3 {
            RGBColor(0, 120, 255)
        } else {
            RGBColor(160, 80, 255)
        };

        for r in rects {
            let fi0 = r.i0 * ref_ratio_total;
            let fj0 = r.j0 * ref_ratio_total;
            let fi1 = (r.i0 + r.nx) * ref_ratio_total;
            let fj1 = (r.j0 + r.ny) * ref_ratio_total;

            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (fi0 as i32, fj0 as i32),
                    (fi1 as i32, fj0 as i32),
                    (fi1 as i32, fj1 as i32),
                    (fi0 as i32, fj1 as i32),
                    (fi0 as i32, fj0 as i32),
                ],
                c.stroke_width(2),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

// =====================================================================
// run_info.txt
// =====================================================================

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
    indicator: IndicatorKind,
    disk_radius: f64,
) {
    let mut f = File::create(path).unwrap();
    writeln!(f, "AMR Cross-Domain-Wall Relaxation on Disk").unwrap();
    writeln!(f, "=========================================").unwrap();
    writeln!(f).unwrap();
    writeln!(f, "Domain: {:.0} nm × {:.0} nm × {:.1} nm",
        base_grid.nx as f64 * base_grid.dx * 1e9,
        base_grid.ny as f64 * base_grid.dy * 1e9,
        base_grid.dz * 1e9
    ).unwrap();
    writeln!(f, "Disk radius: {:.0} nm", disk_radius * 1e9).unwrap();
    writeln!(f).unwrap();
    writeln!(f, "Base grid: {} × {}  dx={:.6e}  dy={:.6e}  dz={:.6e}",
        base_grid.nx, base_grid.ny, base_grid.dx, base_grid.dy, base_grid.dz
    ).unwrap();
    writeln!(f, "Fine grid: {} × {}  dx={:.6e}  dy={:.6e}",
        fine_grid.nx, fine_grid.ny, fine_grid.dx, fine_grid.dy
    ).unwrap();
    writeln!(f, "AMR: {} levels, ratio={}, ghost={}", amr_max_level, ratio, ghost).unwrap();
    writeln!(f).unwrap();
    writeln!(f, "Material (Permalloy soft magnet):").unwrap();
    writeln!(f, "  Ms   = {:.6e} A/m", mat.ms).unwrap();
    writeln!(f, "  A_ex = {:.6e} J/m", mat.a_ex).unwrap();
    writeln!(f, "  K_u  = {:.6e} J/m³ (no anisotropy)", mat.k_u).unwrap();
    writeln!(f, "  DMI  = None").unwrap();
    writeln!(f, "  Demag: ON (FFT uniform)").unwrap();
    writeln!(f).unwrap();
    writeln!(f, "LLG parameters:").unwrap();
    writeln!(f, "  alpha = {:.6e}", llg.alpha).unwrap();
    writeln!(f, "  gamma = {:.6e} rad/(s·T)", llg.gamma).unwrap();
    writeln!(f, "  dt    = {:.6e} s", llg.dt).unwrap();
    writeln!(f, "  steps = {}", steps).unwrap();
    writeln!(f, "  B_ext = [{:.4}, {:.4}, {:.4}] T", llg.b_ext[0], llg.b_ext[1], llg.b_ext[2]).unwrap();
    writeln!(f).unwrap();
    writeln!(f, "Initial condition: four-quadrant cross pattern on disk").unwrap();
    writeln!(f, "  Q1 (+x,+y) → m = +x̂").unwrap();
    writeln!(f, "  Q2 (−x,+y) → m = +ŷ").unwrap();
    writeln!(f, "  Q3 (−x,−y) → m = −x̂").unwrap();
    writeln!(f, "  Q4 (+x,−y) → m = −ŷ").unwrap();
    writeln!(f, "  mz perturbation: ε = 0.01").unwrap();
    writeln!(f).unwrap();
    writeln!(f, "Indicator: {} (threshold={:.4})", indicator.label(), indicator.threshold_param()).unwrap();
    writeln!(f, "Regrid every {} steps", regrid_every).unwrap();

    let l_ex = (mat.a_ex / (MU_0 * mat.ms * mat.ms)).sqrt();
    writeln!(f).unwrap();
    writeln!(f, "Exchange length: l_ex = {:.2} nm", l_ex * 1e9).unwrap();
    writeln!(f, "Finest dx / l_ex = {:.2}", fine_grid.dx / l_ex).unwrap();
}

// =====================================================================
// main
// =====================================================================

fn main() {
    // ---- CLI flags ----
    let args: Vec<String> = std::env::args().collect();
    let do_plots = args.iter().any(|a| a == "--plots");
    let do_ovf = args.iter().any(|a| a == "--ovf");
    let amr_only = args.iter().any(|a| a == "--amr-only");
    let do_fine = !amr_only && !args.iter().any(|a| a == "--skip-fine-ref" || a == "--no-fine");
    let skip_coarse_ref = amr_only;

    // Read the AMR demag mode for display (stepper reads it independently via from_env).
    let amr_demag_mode_label = std::env::var("LLG_AMR_DEMAG_MODE")
        .unwrap_or_else(|_| "all_fft".to_string());

    let out_dir = "out/amr_cross_relax";
    ensure_dir(out_dir);

    // ---- Tunable parameters (env-var overridable) ----
    let amr_max_level: usize = env_or("LLG_AMR_MAX_LEVEL", 3);
    let ratio = 2usize;
    let ghost = 2usize;

    // ---- Physical domain ----
    // Square domain with circular disk geometry.
    // 500 nm × 500 nm, thin film dz = 10 nm.
    let base_nx: usize = env_or("LLG_CROSS_BASE_NX", 128);
    let base_ny: usize = env_or("LLG_CROSS_BASE_NY", 128);
    let lx: f64 = env_or("LLG_CROSS_LX", 500.0e-9); // 500 nm
    let ly: f64 = env_or("LLG_CROSS_LY", 500.0e-9); // 500 nm
    let dz: f64 = env_or("LLG_CROSS_DZ", 10.0e-9);  // 10 nm thin film

    let dx = lx / base_nx as f64; // ≈ 3.91 nm
    let dy = ly / base_ny as f64; // ≈ 3.91 nm

    let ref_ratio_total = pow_usize(ratio, amr_max_level);
    let fine_nx = base_nx * ref_ratio_total;
    let fine_ny = base_ny * ref_ratio_total;

    // ---- Time stepping ----
    let alpha: f64 = env_or("LLG_CROSS_ALPHA", 0.1);
    let dt: f64 = env_or("LLG_CROSS_DT", 5.0e-14);
    let steps_base: usize = env_or("LLG_CROSS_STEPS", 5000);
    let out_every_base: usize = env_or("LLG_CROSS_OUT_EVERY", 250);
    let regrid_every_base: usize = env_or("LLG_CROSS_REGRID_EVERY", 50);

    let pbcx = 0usize;
    let pbcy = 0usize;

    // ---- Material: Permalloy soft magnet ----
    let ms: f64 = env_or("LLG_CROSS_MS", 8.0e5);
    let a_ex: f64 = env_or("LLG_CROSS_AEX", 1.3e-11);

    let mat = Material {
        ms,
        a_ex,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let llg = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha,
        dt,
        b_ext: [0.0, 0.0, 0.0],
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

    // ---- Geometry: circular disk ----
    let disk_radius = 0.5 * lx;
    let disk_shape = MaskShape::Disk {
        center: (0.0, 0.0),
        radius: disk_radius,
    };

    // ---- Initial condition: four-quadrant cross pattern ----
    let perturb_amp = 0.01;

    let mut m_coarse = VectorField2D::new(base_grid);
    init_cross_pattern(&mut m_coarse, &base_grid, &disk_shape, perturb_amp);

    let mut m_coarse_amr = VectorField2D::new(base_grid);
    init_cross_pattern(&mut m_coarse_amr, &base_grid, &disk_shape, perturb_amp);

    let mut h = AmrHierarchy2D::new(base_grid, m_coarse_amr, ratio, ghost);

    let mut m_fine = VectorField2D::new(fine_grid);
    if do_fine {
        init_cross_pattern(&mut m_fine, &fine_grid, &disk_shape, perturb_amp);
    }

    h.set_geom_shape(disk_shape);

    // ---- AMR policies ----
    let indicator_kind = if std::env::var("LLG_AMR_INDICATOR").is_ok() {
        IndicatorKind::from_env()
    } else {
        // 0.08 is lower than skyrmion's 0.15 — needed because the strong domain-wall
        // gradients would otherwise consume the entire flagging budget, leaving the
        // weaker disk-edge gradient (m→0 at mask boundary) unflagged.
        IndicatorKind::Composite { frac: 0.08 }
    };

    let boundary_layer: usize = env_or("LLG_AMR_BOUNDARY_LAYER", 2);
    let buffer_cells = 6usize;

    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind,
        buffer_cells,
        boundary_layer,
        connectivity: Connectivity::Eight,
        merge_distance: 1,
        min_patch_area: 32,
        max_patches: 0,
        min_efficiency: 0.70,
        max_flagged_fraction: 0.50,
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
        disk_radius,
    );

    // ---- Log files ----
    let regrid_log_path = format!("{out_dir}/regrid_log.csv");
    let regrid_levels_path = format!("{out_dir}/regrid_levels.csv");
    let regrid_attempts_path = format!("{out_dir}/regrid_attempts.csv");
    let regrid_patches_path = format!("{out_dir}/regrid_patches.csv");
    let rmse_log_path = format!("{out_dir}/rmse_log.csv");
    let energy_log_path = format!("{out_dir}/energy_log.csv");
    let timing_log_path = format!("{out_dir}/timing_log.csv");

    {
        let mut f = File::create(&regrid_log_path).unwrap();
        writeln!(f,
            "step,max_indicator,threshold,flagged_cells,patches,union_i0,union_j0,union_nx,union_ny"
        ).unwrap();

        let mut f2 = File::create(&rmse_log_path).unwrap();
        writeln!(f2, "step,rmse,max_delta,patches").unwrap();

        let mut f3 = File::create(&regrid_levels_path).unwrap();
        let mut hdr = String::from("step");
        for lvl in 1..=amr_max_level {
            hdr.push_str(&format!(",l{lvl}_count,l{lvl}_i0,l{lvl}_j0,l{lvl}_nx,l{lvl}_ny"));
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
        writeln!(f6, "step,avg_mz_coarse,avg_mz_fine,avg_mz_amr,max_abs_mz_coarse,max_abs_mz_fine,max_abs_mz_amr,avg_mag_amr,mag_cells_amr").unwrap();

        let mut f7 = File::create(&timing_log_path).unwrap();
        writeln!(f7, "step,amr_step_ms,fine_step_ms,coarse_step_ms").unwrap();
    }

    // ---- Initial regrid ----
    let mut current_patches: Vec<Rect2i> = Vec::new();
    if let Some((new_rects, stats)) =
        maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
    {
        current_patches = new_rects;
        let u = union_rect_or_zero(&current_patches);
        append_line(&regrid_log_path, &format!(
            "0,{:.8e},{:.8e},{},{},{},{},{},{}\n",
            stats.max_indicator, stats.threshold, stats.flagged_cells,
            current_patches.len(), u.i0, u.j0, u.nx, u.ny
        ));

        let mut row = String::from("0");
        for lvl in 1..=amr_max_level {
            let rects = level_rects(&h, lvl);
            let uu = union_rect_or_zero(&rects);
            row.push_str(&format!(",{},{},{},{},{}", rects.len(), uu.i0, uu.j0, uu.nx, uu.ny));
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

    let subcycle_active = stepper.is_subcycling();
    let subcycle_ratio: usize = if subcycle_active {
        (stepper.coarse_dt(&llg, &h) / llg.dt).round() as usize
    } else {
        1
    };
    let snap_up = |v: usize, r: usize| -> usize {
        if r <= 1 { v } else { ((v + r - 1) / r) * r }
    };
    let steps = snap_up(steps_base, subcycle_ratio);
    let out_every = snap_up(out_every_base, subcycle_ratio);
    let regrid_every = snap_up(regrid_every_base, subcycle_ratio);
    if subcycle_active {
        eprintln!(
            "[amr_cross_relax] SUBCYCLING ACTIVE: n_levels={}, subcycle_ratio={}, steps={}, out_every={}, regrid_every={}",
            h.num_levels(), subcycle_ratio, steps, out_every, regrid_every
        );
    }

    let mut scratch_fine = if do_fine { Some(RK4Scratch::new(fine_grid)) } else { None };
    let mut scratch_coarse = RK4Scratch::new(base_grid);
    let mut b_fine = if do_fine { Some(VectorField2D::new(fine_grid)) } else { None };
    let mut b_coarse = VectorField2D::new(base_grid);

    // Local field mask: Exchange + Anisotropy (K_u=0 so anis is a no-op, but matches API)
    let local_mask = FieldMask::ExchAnis;

    // ---- OVF directories ----
    if do_ovf {
        ensure_dir(&format!("{out_dir}/ovf_coarse"));
        if do_fine { ensure_dir(&format!("{out_dir}/ovf_fine")); }
        ensure_dir(&format!("{out_dir}/ovf_amr"));
    }

    // ---- Banner ----
    let l_ex = (a_ex / (MU_0 * ms * ms)).sqrt();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  AMR Cross-Domain-Wall Relaxation on Disk                    ║");
    println!("║  Permalloy · Exchange + Demag · Geometry Mask                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Domain:  {:.0}nm × {:.0}nm × {:.1}nm   (thin film disk)", lx*1e9, ly*1e9, dz*1e9);
    println!("Disk:    R = {:.0} nm  (MaskShape::Disk)", disk_radius * 1e9);
    println!("Base:    {} × {}   dx={:.3e}  dy={:.3e}  dz={:.3e}", base_nx, base_ny, dx, dy, dz);
    println!("Fine:    {} × {}   dx={:.3e}  dy={:.3e}", fine_nx, fine_ny, fine_grid.dx, fine_grid.dy);
    println!("AMR:     {} levels, ratio={}, ghost={}", amr_max_level, ratio, ghost);
    println!();
    println!("Material: Ms={:.2e}  A_ex={:.2e}  K_u=0  DMI=None", ms, a_ex);
    println!("  l_ex = {:.2} nm   finest dx/l_ex = {:.2}", l_ex*1e9, fine_grid.dx / l_ex);
    println!("  Fields: Exchange + Demag (FFT uniform)");
    println!();
    println!("LLG:     α={:.4}  dt={:.2e}  steps={}  total={:.3} ns",
        alpha, dt, steps, steps as f64 * dt * 1e9);
    if subcycle_active {
        println!("Subcycling: ON  ratio={}  dt_coarse={:.2e}  coarse_steps={}",
            subcycle_ratio, subcycle_ratio as f64 * dt, steps / subcycle_ratio.max(1));
    } else {
        println!("Subcycling: OFF");
    }
    println!();
    println!("Initial: four-quadrant cross walls on disk  (ε_mz = {:.4})", perturb_amp);
    println!("  Magnetic cells: {} / {} ({:.1}%)",
        count_magnetic_cells(&m_coarse), base_nx * base_ny,
        100.0 * count_magnetic_cells(&m_coarse) as f64 / (base_nx * base_ny) as f64);
    println!();
    println!("Indicator: {} (threshold={:.4})", indicator_kind.label(), indicator_kind.threshold_param());
    println!("Regrid every {} steps", regrid_every);
    println!("Output:  {out_dir}");
    if do_plots { println!("  --plots enabled"); }
    if do_ovf { println!("  --ovf enabled"); }
    if !do_fine { println!("  --skip-fine-ref: uniform fine reference SKIPPED"); }
    if skip_coarse_ref { println!("  --amr-only: coarse baseline SKIPPED"); }
    println!("  AMR demag mode: {}", amr_demag_mode_label);
    println!();

    // ---- Timings ----
    let t0 = Instant::now();
    let mut t_demag_fine = 0.0;
    let mut t_demag_coarse = 0.0;
    let mut t_amr_step = 0.0;
    let mut amr_step_count = 0usize;
    let mut recent_amr_ms: Vec<f64> = Vec::with_capacity(16);

    // ---- Step 0 outputs ----
    {
        let m_amr_fine = flatten_to_target_grid(&h, fine_grid);
        let (rmse, maxd) = if do_fine { rmse_and_max_delta(&m_amr_fine, &m_fine) } else { (f64::NAN, f64::NAN) };
        append_line(&rmse_log_path, &format!("0,{:.8e},{:.8e},{}\n", rmse, maxd, current_patches.len()));

        let (avg_mz_fine_val, max_mz_fine_val) = if do_fine { (avg_mz(&m_fine), max_abs_mz(&m_fine)) } else { (f64::NAN, f64::NAN) };
        append_line(&energy_log_path, &format!(
            "0,{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{}\n",
            avg_mz(&m_coarse), avg_mz_fine_val, avg_mz(&m_amr_fine),
            max_abs_mz(&m_coarse), max_mz_fine_val, max_abs_mz(&m_amr_fine),
            avg_m_magnitude(&m_amr_fine), count_magnetic_cells(&m_amr_fine),
        ));

        if do_ovf {
            write_ovf_text(&format!("{out_dir}/ovf_coarse/m0000000.ovf"), &m_coarse, "m_coarse");
            if do_fine { write_ovf_text(&format!("{out_dir}/ovf_fine/m0000000.ovf"), &m_fine, "m_fine"); }
            write_ovf_text(&format!("{out_dir}/ovf_amr/m0000000.ovf"), &m_amr_fine, "m_amr");
        }

        if do_plots {
            let levels = all_level_rects(&h, amr_max_level);
            save_patch_map(&base_grid, &levels, &format!("{out_dir}/patch_map_step0000.png"), "Patch map (step 0)").unwrap();
            save_angle_map(&m_amr_fine, &format!("{out_dir}/angle_amr_step0000.png"), "In-plane angle (AMR, step 0)").unwrap();
            save_mesh_zoom_multilevel(&m_amr_fine, &base_grid, ratio, amr_max_level, &levels,
                &format!("{out_dir}/mesh_zoom_step0000.png"), "In-plane angle + grid (step 0)").unwrap();
        }

        let mut lvl_counts = String::new();
        for lvl in 1..=amr_max_level {
            if lvl > 1 { lvl_counts.push_str(" | "); }
            lvl_counts.push_str(&format!("L{} {:2}", lvl, level_patch_count(&h, lvl)));
        }
        println!("step     0/{} | <mz>={:.4} | {} | patches={}", steps, avg_mz(&m_amr_fine), lvl_counts, current_patches.len());
    }

    // =====================================================================
    // Time loop
    // =====================================================================
    for step in 1..=steps {
        // ---- Uniform fine ----
        let mut dt_fine_ms = f64::NAN;
        if do_fine {
            let bf = b_fine.as_mut().unwrap();
            let sf = scratch_fine.as_mut().unwrap();
            let t1 = Instant::now();
            bf.set_uniform(0.0, 0.0, 0.0);
            demag_fft_uniform::compute_demag_field_pbc(&fine_grid, &m_fine, bf, &mat, pbcx, pbcy);
            t_demag_fine += t1.elapsed().as_secs_f64();
            step_llg_rk4_recompute_field_masked_relax_add(&mut m_fine, &llg, &mat, sf, local_mask, Some(bf));
            dt_fine_ms = t1.elapsed().as_secs_f64() * 1e3;
        }

        // ---- Uniform coarse ----
        let mut dt_coarse_ms = f64::NAN;
        if !skip_coarse_ref {
            let t2 = Instant::now();
            b_coarse.set_uniform(0.0, 0.0, 0.0);
            demag_fft_uniform::compute_demag_field_pbc(&base_grid, &m_coarse, &mut b_coarse, &mat, pbcx, pbcy);
            t_demag_coarse += t2.elapsed().as_secs_f64();
            step_llg_rk4_recompute_field_masked_relax_add(&mut m_coarse, &llg, &mat, &mut scratch_coarse, local_mask, Some(&b_coarse));
            dt_coarse_ms = t2.elapsed().as_secs_f64() * 1e3;
        }

        // ---- AMR step (demag mode controlled by LLG_AMR_DEMAG_MODE) ----
        let amr_due = step % subcycle_ratio == 0;
        if amr_due {
            let t3 = Instant::now();
            stepper.step(&mut h, &llg, &mat, local_mask);
            let elapsed = t3.elapsed().as_secs_f64();
            t_amr_step += elapsed;
            let dt_amr_ms = elapsed * 1e3;
            amr_step_count += 1;

            recent_amr_ms.push(dt_amr_ms);
            if recent_amr_ms.len() > 10 { recent_amr_ms.remove(0); }

            append_line(&timing_log_path, &format!(
                "{},{:.3},{:.3},{:.3}\n", step, dt_amr_ms, dt_fine_ms, dt_coarse_ms
            ));

            if amr_step_count <= 3 || amr_step_count % 10 == 0 {
                let avg: f64 = recent_amr_ms.iter().sum::<f64>() / recent_amr_ms.len() as f64;
                let l1 = h.patches.len();
                let l2 = h.patches_l2plus.get(0).map(|v| v.len()).unwrap_or(0);
                eprintln!("[step {:4}] AMR {:.1}ms (avg {:.1}ms) | L1 {} L2 {} | mode={}",
                    step, dt_amr_ms, avg, l1, l2, amr_demag_mode_label);
            }
        }

        // ---- Regrid ----
        if amr_due && regrid_every > 0 && step % regrid_every == 0 {
            let mut row2 = format!("{},{:.8e}", step, max_indicator_coarse(&h.coarse));
            for lvl in 1..=amr_max_level { row2.push_str(&format!(",{}", level_patch_count(&h, lvl))); }
            row2.push('\n');
            append_line(&regrid_attempts_path, &row2);

            if let Some((new_rects, stats)) =
                maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
            {
                current_patches = new_rects;
                let u = union_rect_or_zero(&current_patches);
                append_line(&regrid_log_path, &format!(
                    "{},{:.8e},{:.8e},{},{},{},{},{},{}\n",
                    step, stats.max_indicator, stats.threshold, stats.flagged_cells,
                    current_patches.len(), u.i0, u.j0, u.nx, u.ny
                ));

                let mut row = format!("{}", step);
                for lvl in 1..=amr_max_level {
                    let rects = level_rects(&h, lvl);
                    let uu = union_rect_or_zero(&rects);
                    row.push_str(&format!(",{},{},{},{},{}", rects.len(), uu.i0, uu.j0, uu.nx, uu.ny));
                }
                row.push('\n');
                append_line(&regrid_levels_path, &row);
                append_regrid_patches_csv(&regrid_patches_path, &h, amr_max_level, step);
            }
        }

        // ---- Diagnostics output ----
        if step % out_every == 0 || step == steps {
            let m_amr_fine = flatten_to_target_grid(&h, fine_grid);
            let (rmse, maxd) = if do_fine { rmse_and_max_delta(&m_amr_fine, &m_fine) } else { (f64::NAN, f64::NAN) };

            append_line(&rmse_log_path, &format!("{},{:.8e},{:.8e},{}\n", step, rmse, maxd, current_patches.len()));

            let (avg_mz_fine_val, max_mz_fine_val) = if do_fine { (avg_mz(&m_fine), max_abs_mz(&m_fine)) } else { (f64::NAN, f64::NAN) };
            append_line(&energy_log_path, &format!(
                "{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{}\n",
                step, avg_mz(&m_coarse), avg_mz_fine_val, avg_mz(&m_amr_fine),
                max_abs_mz(&m_coarse), max_mz_fine_val, max_abs_mz(&m_amr_fine),
                avg_m_magnitude(&m_amr_fine), count_magnetic_cells(&m_amr_fine),
            ));

            if do_ovf {
                let fname = format!("m{:07}.ovf", step);
                write_ovf_text(&format!("{out_dir}/ovf_coarse/{fname}"), &m_coarse, "m_coarse");
                if do_fine { write_ovf_text(&format!("{out_dir}/ovf_fine/{fname}"), &m_fine, "m_fine"); }
                write_ovf_text(&format!("{out_dir}/ovf_amr/{fname}"), &m_amr_fine, "m_amr");
            }

            let mut lvl_counts = String::new();
            for lvl in 1..=amr_max_level {
                if lvl > 1 { lvl_counts.push_str(" | "); }
                lvl_counts.push_str(&format!("L{} {:2}", lvl, level_patch_count(&h, lvl)));
            }
            let t_elapsed = t0.elapsed().as_secs_f64();
            println!(
                "step {:5}/{} | rmse {:.3e} | <mz> {:.4} | {} | fine={:.0}s coarse={:.0}s amr={:.0}s | {:.1}s",
                step, steps, rmse, avg_mz(&m_amr_fine), lvl_counts,
                t_demag_fine, t_demag_coarse, t_amr_step, t_elapsed,
            );

            if do_plots {
                let levels = all_level_rects(&h, amr_max_level);
                save_patch_map(&base_grid, &levels, &format!("{out_dir}/patch_map_step{step:04}.png"),
                    &format!("Patch map (step {})", step)).unwrap();
                save_angle_map(&m_amr_fine, &format!("{out_dir}/angle_amr_step{step:04}.png"),
                    &format!("In-plane angle (AMR, step {})", step)).unwrap();
                save_mesh_zoom_multilevel(&m_amr_fine, &base_grid, ratio, amr_max_level, &levels,
                    &format!("{out_dir}/mesh_zoom_step{step:04}.png"),
                    &format!("In-plane angle + grid (step {})", step)).unwrap();
            }
        }
    }

    let wall = t0.elapsed().as_secs_f64();

    // =====================================================================
    // Final outputs
    // =====================================================================
    let m_amr_fine_final = flatten_to_target_grid(&h, fine_grid);
    write_csv_ij_m(&format!("{out_dir}/uniform_coarse_final.csv"), &m_coarse);
    if do_fine { write_csv_ij_m(&format!("{out_dir}/uniform_fine_final.csv"), &m_fine); }
    write_csv_ij_m(&format!("{out_dir}/amr_fine_final.csv"), &m_amr_fine_final);

    if do_fine { write_midline_y(&format!("{out_dir}/lineout_uniform_mid_y.csv"), &m_fine); }
    write_midline_y(&format!("{out_dir}/lineout_amr_mid_y.csv"), &m_amr_fine_final);
    if do_fine { write_midline_x(&format!("{out_dir}/lineout_uniform_mid_x.csv"), &m_fine); }
    write_midline_x(&format!("{out_dir}/lineout_amr_mid_x.csv"), &m_amr_fine_final);

    // ---- Summary ----
    println!();
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Run complete                                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Base grid: {} × {}   dx={:.3e} dy={:.3e} dz={:.3e}", base_grid.nx, base_grid.ny, base_grid.dx, base_grid.dy, base_grid.dz);
    println!("Fine grid: {} × {}   dx={:.3e} dy={:.3e}", fine_grid.nx, fine_grid.ny, fine_grid.dx, fine_grid.dy);
    println!("Steps: {}   dt={:.3e}", steps, dt);
    if subcycle_active {
        println!("Subcycling: ON  n_levels={}  ratio={}  AMR coarse steps: ~{}",
            h.num_levels(), subcycle_ratio, steps / subcycle_ratio.max(1));
    } else {
        println!("Subcycling: OFF");
    }
    println!("Outputs: {out_dir}");
    println!();
    println!("Demag FFT grid sizes:");
    println!("  uniform fine FFT:      {} × {} = {} cells", fine_grid.nx, fine_grid.ny, fine_grid.nx * fine_grid.ny);
    println!("  coarse_fft FFT:        {} × {} = {} cells", base_grid.nx, base_grid.ny, base_grid.nx * base_grid.ny);
    println!("  cell ratio (fine/coarse): {}×", (fine_grid.nx * fine_grid.ny) / (base_grid.nx * base_grid.ny));
    println!();
    println!("Timing (demag mode: {}):", amr_demag_mode_label);
    println!("  total wall time:       {:.3} s", wall);
    if do_fine {
        println!("  fine demag time:       {:.3} s", t_demag_fine);
    } else {
        println!("  fine demag time:       (skipped --skip-fine-ref)");
    }
    if !skip_coarse_ref {
        println!("  coarse demag time:     {:.3} s", t_demag_coarse);
    } else {
        println!("  coarse demag time:     (skipped --amr-only)");
    }
    println!("  AMR step time:         {:.3} s  ({} AMR steps)", t_amr_step, amr_step_count);
    if amr_step_count > 0 {
        let avg_amr_ms = (t_amr_step / amr_step_count as f64) * 1e3;
        println!("  AMR avg per step:      {:.1} ms/step", avg_amr_ms);
        if do_fine && steps > 0 {
            let avg_fine_ms = (t_demag_fine / steps as f64) * 1e3;
            let speedup = avg_fine_ms / avg_amr_ms;
            println!("  fine avg per step:     {:.1} ms/step  (demag only)", avg_fine_ms);
            println!("  speedup (fine/AMR):    {:.1}×", speedup);
        }
    }
    let other = (wall - t_demag_fine - t_demag_coarse - t_amr_step).max(0.0);
    println!("  other/unaccounted:     {:.3} s", other);

    let mut lvl_counts = String::new();
    for lvl in 1..=amr_max_level {
        if lvl > 1 { lvl_counts.push_str(" | "); }
        lvl_counts.push_str(&format!("L{}: {} patches", lvl, level_patch_count(&h, lvl)));
    }
    println!("  Final patches: {}", lvl_counts);
    println!();
    println!("Timing log: {out_dir}/timing_log.csv");
}