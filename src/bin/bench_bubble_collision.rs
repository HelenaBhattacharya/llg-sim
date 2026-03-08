// src/bin/bench_bubble_collision.rs
//
// AMR Dynamics Showcase: 7 magnetic bubble domains in a PMA + DMI thin film.
//
// Three simulations run in parallel from the same initial state:
//   1. Uniform coarse (L0 only) — fast but WRONG wall physics
//   2. AMR + coarse FFT demag + subcycling — fast AND correct
//   3. Uniform fine (reference) — slow but correct
//
// Oversized bubbles (R₀=150nm) shrink under relaxation. Nearby pairs merge,
// creating complex wall topology that AMR tracks dynamically.
//
// Output structure mirrors amr_two_bubbles_relax.rs / amr_vortex_relax.rs
// so it can be inspected with scripts/amr_viewer.py.
//
// Outputs in out/bubble_collision/:
//   patch_map_stepXXXX.png       : patch rectangles by refinement level [--plots]
//   mesh_zoom_stepXXXX.png       : in-plane angle + multi-level grid overlay [--plots]
//   regrid_log.csv               : regrid accept events
//   regrid_levels.csv            : per-accept union summaries by level
//   regrid_attempts.csv          : per-check summary
//   regrid_patches.csv           : per-patch rectangles by level
//   rmse_log.csv                 : AMR vs fine + coarse vs fine RMSE
//   timing_log.csv               : per-step wall times
//   ovf_coarse/mXXXXXXX.ovf     : coarse OVFs [--ovf]
//   ovf_fine/mXXXXXXX.ovf       : fine reference OVFs [--ovf]
//   ovf_amr/mXXXXXXX.ovf        : AMR composite OVFs [--ovf]
//   *_final.csv                  : final states
//   lineout_*_mid_y.csv          : midline profiles
//
// Run:
//   cargo run --release --bin bench_bubble_collision -- --plots
//   cargo run --release --bin bench_bubble_collision -- --ovf --plots
//   cargo run --release --bin bench_bubble_collision -- --skip-fine-ref --plots
//   cargo run --release --bin bench_bubble_collision -- --amr-only --plots

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use plotters::prelude::*;

use llg_sim::amr::indicator::{IndicatorKind, indicator_grad2_forward};
use llg_sim::amr::interp::sample_bilinear;
use llg_sim::amr::regrid::maybe_regrid_nested_levels;
use llg_sim::amr::{
    AmrHierarchy2D, AmrStepperRK4, ClusterPolicy, Connectivity, Rect2i, RegridPolicy,
};
use llg_sim::effective_field::{FieldMask, demag_fft_uniform};
use llg_sim::geometry_mask::MaskShape;
use llg_sim::grid::Grid2D;
use llg_sim::initial_states::seed_smooth_bubbles;
use llg_sim::llg::{RK4Scratch, step_llg_rk4_recompute_field_masked_relax_add};
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material, MU0};
use llg_sim::vector_field::VectorField2D;

// ═══════════════════════════════════════════════════════════════════════════
//  Utility functions (same as amr_two_bubbles_relax.rs)
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

// ── OVF / CSV writers ────────────────────────────────────────────────────

fn write_ovf_text(path: &str, m: &VectorField2D, title: &str) {
    let mut f = File::create(path).unwrap();
    let (nx, ny) = (m.grid.nx, m.grid.ny);
    let (dx, dy, dz) = (m.grid.dx, m.grid.dy, m.grid.dz);

    writeln!(f, "# OOMMF OVF 2.0").unwrap();
    writeln!(f, "# Segment count: 1").unwrap();
    writeln!(f, "# Begin: Segment").unwrap();
    writeln!(f, "# Begin: Header").unwrap();
    writeln!(f, "# Title: {title}").unwrap();
    writeln!(f, "# meshtype: rectangular").unwrap();
    writeln!(f, "# meshunit: m").unwrap();
    writeln!(f, "# xbase: {:.16e}", 0.5 * dx).unwrap();
    writeln!(f, "# ybase: {:.16e}", 0.5 * dy).unwrap();
    writeln!(f, "# zbase: {:.16e}", 0.5 * dz).unwrap();
    writeln!(f, "# xstepsize: {:.16e}", dx).unwrap();
    writeln!(f, "# ystepsize: {:.16e}", dy).unwrap();
    writeln!(f, "# zstepsize: {:.16e}", dz).unwrap();
    writeln!(f, "# xnodes: {nx}").unwrap();
    writeln!(f, "# ynodes: {ny}").unwrap();
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

fn append_line(path: &str, line: &str) {
    let mut f = OpenOptions::new().create(true).append(true).open(path).unwrap();
    f.write_all(line.as_bytes()).unwrap();
}

// ── AMR helpers ──────────────────────────────────────────────────────────

fn append_regrid_patches_csv(path: &str, h: &AmrHierarchy2D, max_level: usize, step: usize) {
    let mut f = OpenOptions::new().create(true).append(true).open(path).unwrap();
    for (pid, p) in h.patches.iter().enumerate() {
        let r = p.coarse_rect;
        writeln!(f, "{},{},{},{},{},{},{}", step, 1, pid, r.i0, r.j0, r.nx, r.ny).unwrap();
    }
    for lvl in 2..=max_level {
        if let Some(patches) = h.patches_l2plus.get(lvl - 2) {
            for (pid, p) in patches.iter().enumerate() {
                let r = p.coarse_rect;
                writeln!(f, "{},{},{},{},{},{},{}", step, lvl, pid, r.i0, r.j0, r.nx, r.ny).unwrap();
            }
        }
    }
}

fn rmse_and_max_delta(a: &VectorField2D, b: &VectorField2D) -> (f64, f64) {
    assert_eq!(a.grid.nx, b.grid.nx);
    assert_eq!(a.grid.ny, b.grid.ny);
    let mut s2 = 0.0f64;
    let mut maxd = 0.0f64;
    let n = (a.grid.nx * a.grid.ny) as f64;
    for k in 0..a.data.len() {
        let da = a.data[k];
        let db = b.data[k];
        let d2 = (da[0]-db[0]).powi(2) + (da[1]-db[1]).powi(2) + (da[2]-db[2]).powi(2);
        s2 += d2;
        maxd = maxd.max(d2.sqrt());
    }
    ((s2 / n).sqrt(), maxd)
}

fn resample_to_grid_bilinear(src: &VectorField2D, dst_grid: Grid2D) -> VectorField2D {
    let mut out = VectorField2D::new(dst_grid);
    for j in 0..out.grid.ny {
        let y = (j as f64 + 0.5) * out.grid.dy;
        for i in 0..out.grid.nx {
            let x = (i as f64 + 0.5) * out.grid.dx;
            out.data[out.grid.idx(i, j)] = sample_bilinear(src, x, y);
        }
    }
    out
}

fn ensure_grid(src: &VectorField2D, dst_grid: Grid2D) -> VectorField2D {
    if src.grid.nx == dst_grid.nx && src.grid.ny == dst_grid.ny
        && src.grid.dx == dst_grid.dx && src.grid.dy == dst_grid.dy
        && src.grid.dz == dst_grid.dz
    {
        let mut out = VectorField2D::new(dst_grid);
        out.data.clone_from(&src.data);
        out
    } else {
        resample_to_grid_bilinear(src, dst_grid)
    }
}

fn union_rect(rects: &[Rect2i]) -> Option<Rect2i> {
    if rects.is_empty() { return None; }
    let (mut i0, mut j0) = (rects[0].i0, rects[0].j0);
    let (mut i1, mut j1) = (i0 + rects[0].nx, j0 + rects[0].ny);
    for r in rects.iter().skip(1) {
        i0 = i0.min(r.i0); j0 = j0.min(r.j0);
        i1 = i1.max(r.i0 + r.nx); j1 = j1.max(r.j0 + r.ny);
    }
    Some(Rect2i { i0, j0, nx: i1 - i0, ny: j1 - j0 })
}

fn union_rect_or_zero(rects: &[Rect2i]) -> Rect2i {
    union_rect(rects).unwrap_or(Rect2i { i0: 0, j0: 0, nx: 0, ny: 0 })
}

fn level_patch_count(h: &AmrHierarchy2D, level: usize) -> usize {
    match level {
        1 => h.patches.len(),
        l if l >= 2 => h.patches_l2plus.get(l - 2).map(|v| v.len()).unwrap_or(0),
        _ => 0,
    }
}

fn level_rects(h: &AmrHierarchy2D, level: usize) -> Vec<Rect2i> {
    match level {
        1 => h.patches.iter().map(|p| p.coarse_rect).collect(),
        l if l >= 2 => h.patches_l2plus.get(l - 2)
            .map(|v| v.iter().map(|p| p.coarse_rect).collect())
            .unwrap_or_else(Vec::new),
        _ => Vec::new(),
    }
}

fn all_level_rects(h: &AmrHierarchy2D, max_level: usize) -> Vec<Vec<Rect2i>> {
    (1..=max_level).map(|lvl| level_rects(h, lvl)).collect()
}

fn max_indicator_coarse(coarse: &VectorField2D) -> f64 {
    let mut max_ind = 0.0f64;
    for j in 0..coarse.grid.ny {
        for i in 0..coarse.grid.nx {
            max_ind = max_ind.max(indicator_grad2_forward(coarse, i, j));
        }
    }
    max_ind
}

// ── Plotters (same as two_bubbles) ───────────────────────────────────────

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> RGBColor {
    let h = h.rem_euclid(1.0);
    let i = (h * 6.0).floor() as i32;
    let f = h * 6.0 - i as f64;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match i.rem_euclid(6) {
        0 => (v, t, p), 1 => (q, v, p), 2 => (p, v, t),
        3 => (p, q, v), 4 => (t, p, v), _ => (v, p, q),
    };
    RGBColor(
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    )
}

fn save_patch_map(
    base_grid: &Grid2D,
    levels: &[Vec<Rect2i>],
    path: &str,
    caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx0 = base_grid.nx as f64;
    let ny0 = base_grid.ny as f64;
    let aspect = ny0 / nx0;

    let root = BitMapBackend::new(path, (900, (700.0 * aspect).max(300.0) as u32)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 22))
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(35)
        .build_cartesian_2d(0f64..1f64, 0f64..aspect)?;

    chart
        .configure_mesh()
        .x_desc("x/L")
        .y_desc("y/L")
        .disable_mesh()
        .draw()?;

    fn fill_color(level: usize) -> RGBColor {
        match level {
            1 => RGBColor(240, 220, 0),
            2 => RGBColor(0, 200, 0),
            3 => RGBColor(0, 120, 255),
            4 => RGBColor(160, 80, 255),
            _ => RGBColor(220, 0, 220),
        }
    }

    fn stroke_color(level: usize) -> RGBColor {
        match level {
            1 => BLACK,
            2 => RED,
            3 => RGBColor(0, 60, 140),
            4 => RGBColor(90, 40, 140),
            _ => BLACK,
        }
    }

    for (k, rects) in levels.iter().enumerate() {
        let lvl = k + 1;
        let fill = fill_color(lvl).filled();
        let stroke = stroke_color(lvl).stroke_width(2);

        for r in rects {
            let x0 = (r.i0 as f64) / nx0;
            let y0 = (r.j0 as f64) / ny0 * aspect;
            let x1 = ((r.i0 + r.nx) as f64) / nx0;
            let y1 = ((r.j0 + r.ny) as f64) / ny0 * aspect;

            chart.draw_series(std::iter::once(Rectangle::new([(x0, y0), (x1, y1)], fill)))?;
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                stroke.clone(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

#[allow(unused_variables)]
fn save_mesh_zoom_multilevel(
    m_fine: &VectorField2D,
    base_grid: &Grid2D,
    ratio: usize,
    amr_max_level: usize,
    levels: &[Vec<Rect2i>],
    path: &str,
    caption: &str,
    margin_cells_finest: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let ref_ratio_total = pow_usize(ratio, amr_max_level);

    // Keep zoom stable: anchor to L2 union if present, else L1.
    let target: &[Rect2i] = if levels.len() >= 2 && !levels[1].is_empty() {
        levels[1].as_slice()
    } else if !levels.is_empty() && !levels[0].is_empty() {
        levels[0].as_slice()
    } else {
        levels
            .iter()
            .rev()
            .find(|v| !v.is_empty())
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };

    let u = match union_rect(target) {
        Some(u) => u,
        None => return Ok(()),
    };

    // Convert union rect (base coords) to finest coords.
    let fi0 = u.i0 * ref_ratio_total;
    let fj0 = u.j0 * ref_ratio_total;
    let fi1 = fi0 + u.nx * ref_ratio_total;
    let fj1 = fj0 + u.ny * ref_ratio_total;

    let nx = m_fine.grid.nx;
    let ny = m_fine.grid.ny;

    let x0 = fi0.saturating_sub(margin_cells_finest);
    let y0 = fj0.saturating_sub(margin_cells_finest);
    let x1 = (fi1 + margin_cells_finest).min(nx);
    let y1 = (fj1 + margin_cells_finest).min(ny);

    let root = BitMapBackend::new(path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 22))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(x0 as i32..x1 as i32, y0 as i32..y1 as i32)?;

    chart.configure_mesh().disable_mesh().draw()?;

    // Background: in-plane angle as HSV colour.
    chart.draw_series((y0..y1).flat_map(|j| {
        (x0..x1).map(move |i| {
            let v = m_fine.data[m_fine.grid.idx(i, j)];
            let phi = v[1].atan2(v[0]);
            let h = (phi + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
            let col = hsv_to_rgb(h, 1.0, 1.0);
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    let level_spacing = |lvl: usize| -> usize {
        let r_lvl = pow_usize(ratio, lvl);
        let s = ref_ratio_total / r_lvl;
        s.max(1)
    };

    // L0 grid (coarse) in light gray.
    let s0 = ref_ratio_total.max(1);
    let mut xx = ((x0 + s0 - 1) / s0) * s0;
    while xx <= x1 {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(xx as i32, y0 as i32), (xx as i32, y1 as i32)],
            RGBColor(180, 180, 180).stroke_width(1),
        )))?;
        xx = xx.saturating_add(s0);
        if s0 == 0 { break; }
    }
    let mut yy = ((y0 + s0 - 1) / s0) * s0;
    while yy <= y1 {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0 as i32, yy as i32), (x1 as i32, yy as i32)],
            RGBColor(180, 180, 180).stroke_width(1),
        )))?;
        yy = yy.saturating_add(s0);
        if s0 == 0 { break; }
    }

    // Draw solid grid lines for L1/L2 patches.
    {
        let mut draw_grid_for_level = |rects: &[Rect2i],
                                       lvl: usize,
                                       style: ShapeStyle|
         -> Result<(), Box<dyn std::error::Error>> {
            let s = level_spacing(lvl);
            for r in rects {
                let gi0 = r.i0 * ref_ratio_total;
                let gj0 = r.j0 * ref_ratio_total;
                let gi1 = gi0 + r.nx * ref_ratio_total;
                let gj1 = gj0 + r.ny * ref_ratio_total;

                let xi0 = gi0.max(x0);
                let yi0 = gj0.max(y0);
                let xi1 = gi1.min(x1);
                let yi1 = gj1.min(y1);
                if xi1 <= xi0 || yi1 <= yi0 { continue; }

                // Patch outline.
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![
                        (xi0 as i32, yi0 as i32), (xi1 as i32, yi0 as i32),
                        (xi1 as i32, yi1 as i32), (xi0 as i32, yi1 as i32),
                        (xi0 as i32, yi0 as i32),
                    ],
                    style.clone().stroke_width(3),
                )))?;

                // Cell grid lines within this patch.
                let mut xg = xi0;
                while xg <= xi1 {
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(xg as i32, yi0 as i32), (xg as i32, yi1 as i32)],
                        style.clone().stroke_width(1),
                    )))?;
                    xg = xg.saturating_add(s);
                    if s == 0 { break; }
                }
                let mut yg = yi0;
                while yg <= yi1 {
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(xi0 as i32, yg as i32), (xi1 as i32, yg as i32)],
                        style.clone().stroke_width(1),
                    )))?;
                    yg = yg.saturating_add(s);
                    if s == 0 { break; }
                }
            }
            Ok(())
        };

        for (k, rects) in levels.iter().enumerate() {
            let lvl = k + 1;
            if lvl == 1 {
                draw_grid_for_level(rects, lvl, BLACK.mix(0.25).filled())?;
            } else if lvl == 2 {
                draw_grid_for_level(rects, lvl, RED.mix(0.3).filled())?;
            }
        }
    }

    // L3+ dashed outlines only (no cell grid lines — too dense).
    {
        let mut draw_outline =
            |rects: &[Rect2i], color: RGBColor| -> Result<(), Box<dyn std::error::Error>> {
                for r in rects {
                    let gi0 = r.i0 * ref_ratio_total;
                    let gj0 = r.j0 * ref_ratio_total;
                    let gi1 = gi0 + r.nx * ref_ratio_total;
                    let gj1 = gj0 + r.ny * ref_ratio_total;

                    let xi0 = gi0.max(x0);
                    let yi0 = gj0.max(y0);
                    let xi1 = gi1.min(x1);
                    let yi1 = gj1.min(y1);
                    if xi1 <= xi0 || yi1 <= yi0 { continue; }

                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![
                            (xi0 as i32, yi0 as i32), (xi1 as i32, yi0 as i32),
                            (xi1 as i32, yi1 as i32), (xi0 as i32, yi1 as i32),
                            (xi0 as i32, yi0 as i32),
                        ],
                        color.stroke_width(2),
                    )))?;
                }
                Ok(())
            };

        for (k, rects) in levels.iter().enumerate() {
            let lvl = k + 1;
            if lvl >= 3 {
                let c = if lvl == 3 { RGBColor(0, 120, 255) } else { RGBColor(160, 80, 255) };
                draw_outline(rects, c)?;
            }
        }
    }

    root.present()?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let do_plots = args.iter().any(|a| a == "--plots");
    let do_ovf = args.iter().any(|a| a == "--ovf" || a == "--ovfs");
    let amr_only = args.iter().any(|a| a == "--amr-only");
    let do_fine = !amr_only && !args.iter().any(|a| a == "--skip-fine-ref" || a == "--no-fine");
    let skip_coarse_ref = amr_only;

    let amr_demag_mode_label = std::env::var("LLG_AMR_DEMAG_MODE")
        .unwrap_or_else(|_| "coarse_fft".to_string());

    let amr_max_level = env_usize("LLG_AMR_MAX_LEVEL", 3);

    let out_dir = "out/bubble_collision";
    ensure_dir(out_dir);

    // ── Grid ─────────────────────────────────────────────────────────────

    let base_nx = env_usize("LLG_BC_NX", 256);
    let base_ny = env_usize("LLG_BC_NY", 128);
    let lx = env_f64("LLG_BC_LX", 2.0e-6);
    let ly = env_f64("LLG_BC_LY", 1.0e-6);
    let dz = env_f64("LLG_BC_DZ", 4.0e-10);

    let dx = lx / base_nx as f64;
    let dy = ly / base_ny as f64;
    let base_grid = Grid2D::new(base_nx, base_ny, dx, dy, dz);

    let ratio = 2usize;
    let ghost = 2usize;
    let ref_ratio_total = pow_usize(ratio, amr_max_level);
    let fine_grid = Grid2D::new(
        base_nx * ref_ratio_total, base_ny * ref_ratio_total,
        dx / ref_ratio_total as f64, dy / ref_ratio_total as f64, dz,
    );

    let steps_base = env_usize("LLG_BC_STEPS", 2000);
    let dt = env_f64("LLG_BC_DT", 5.0e-14);
    let out_every_base = env_usize("LLG_BC_OUTPUT", 100);
    let regrid_every_base = env_usize("LLG_BC_REGRID", 50);
    let alpha = env_f64("LLG_BC_ALPHA", 0.3);

    // ── Material: Co/Pt with interfacial DMI ─────────────────────────────

    let ms = env_f64("LLG_BC_MS", 580.0e3);
    let a_ex = env_f64("LLG_BC_AEX", 15.0e-12);
    let d_dmi = env_f64("LLG_BC_DMI", 3.0e-3);
    let k_u = env_f64("LLG_BC_KU", 0.8e6);

    let k_eff = k_u - MU0 * ms * ms / 2.0;
    let delta_dw = if k_eff > 0.0 { (a_ex / k_eff).sqrt() } else { f64::NAN };
    let l_ex = (2.0 * a_ex / (MU0 * ms * ms)).sqrt();
    let d_c = if k_eff > 0.0 { 4.0 * (a_ex * k_eff).sqrt() / std::f64::consts::PI } else { f64::NAN };

    let mat = Material {
        ms, a_ex, k_u,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: Some(d_dmi),
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let llg = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha, dt,
        b_ext: [0.0, 0.0, 0.0],
    };

    // ── Bubble placement ─────────────────────────────────────────────────

    let r0 = env_f64("LLG_BC_R0", 150.0e-9);
    let wall_width = env_f64("LLG_BC_WALL_WIDTH", 10.0e-9);
    let helicity = 0.0; // Néel

    // 7 bubbles: tight groups that WILL collide + isolated for contrast.
    //
    // Group A (triple → Y-junction): bubbles 0,1,2 in tight triangle
    //   centre-to-centre ~200nm, R₀=150nm → 100nm overlap
    // Group B (head-on pair): bubbles 3,4
    //   centre-to-centre ~220nm along x
    // Isolated: bubbles 5,6
        let centers: Vec<(f64, f64)> = vec![
        // Group A — tight triangle (will form Y-junction)
        (-400e-9,  200e-9),   // 0
        (-200e-9,  250e-9),   // 1: 200nm from 0
        (-350e-9,   30e-9),   // 2: ~180nm from 0, ~230nm from 1

        // Group B — head-on pair (will merge into elongated domain)
        ( 200e-9,  150e-9),   // 3
        ( 420e-9,  150e-9),   // 4: 220nm from 3

        // Isolated (shrink independently — contrast)
        (-500e-9, -300e-9),   // 5
        ( 600e-9, -250e-9),   // 6
    ];

    // ── Print header ─────────────────────────────────────────────────────

    let bar = "═".repeat(68);
    let thin = "─".repeat(68);

    println!("╔{bar}╗");
    println!("║{:^68}║", "Bubble Collision Benchmark — AMR Dynamics Showcase");
    println!("╚{bar}╝");
    println!();
    println!("  Domain:      {:.0} nm × {:.0} nm × {:.1} nm",
        lx*1e9, ly*1e9, dz*1e9);
    println!("  Base grid:   {} × {}, dx = {:.2} nm", base_nx, base_ny, dx*1e9);
    println!("  Fine grid:   {} × {} ({}× refinement, {} AMR levels)",
        fine_grid.nx, fine_grid.ny, ref_ratio_total, amr_max_level);
    println!("  Bubbles:     {} smooth (R₀={:.0}nm, w={:.0}nm, Néel)",
        centers.len(), r0*1e9, wall_width*1e9);
    println!();
    println!("  Material:    Ms={:.0} kA/m, A={:.0} pJ/m, D={:.1} mJ/m², Ku={:.2} MJ/m³",
        ms/1e3, a_ex/1e-12, d_dmi/1e-3, k_u/1e6);
    println!("  Derived:     l_ex={:.1}nm, δ_DW={:.1}nm, K_eff={:.3}MJ/m³, D/Dc={:.2}",
        l_ex*1e9, delta_dw*1e9, k_eff/1e6, d_dmi/d_c);
    println!();

    // ── Initialise three identical states ─────────────────────────────────

    let mut m_coarse = VectorField2D::new(base_grid);
    seed_smooth_bubbles(&mut m_coarse, &base_grid, &centers, r0, wall_width, helicity, 1.0, None);

    let mut m_coarse_amr = VectorField2D::new(base_grid);
    seed_smooth_bubbles(&mut m_coarse_amr, &base_grid, &centers, r0, wall_width, helicity, 1.0, None);

    let mut m_fine = VectorField2D::new(fine_grid);
    if do_fine {
        seed_smooth_bubbles(&mut m_fine, &fine_grid, &centers, r0, wall_width, helicity, 1.0, None);
    }

    // ── AMR hierarchy ────────────────────────────────────────────────────

    let mut h = AmrHierarchy2D::new(base_grid, m_coarse_amr, ratio, ghost);
    h.set_geom_shape(MaskShape::Full);

    let indicator_kind = IndicatorKind::from_env();
    let boundary_layer: usize = std::env::var("LLG_AMR_BOUNDARY_LAYER")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let buffer_cells = 4usize;

    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind, buffer_cells, boundary_layer,
        connectivity: Connectivity::Eight,
        merge_distance: 4,
        min_patch_area: 64,
        max_patches: 0,
        min_efficiency: 0.65,
        max_flagged_fraction: 0.50,
    };
    let regrid_policy = RegridPolicy {
        indicator: indicator_kind, buffer_cells, boundary_layer,
        min_change_cells: 2,
        min_area_change_frac: 0.05,
    };

    // ── CSV log headers ──────────────────────────────────────────────────

    let regrid_log_path = format!("{out_dir}/regrid_log.csv");
    let regrid_levels_path = format!("{out_dir}/regrid_levels.csv");
    let regrid_attempts_path = format!("{out_dir}/regrid_attempts.csv");
    let rmse_log_path = format!("{out_dir}/rmse_log.csv");
    let regrid_patches_path = format!("{out_dir}/regrid_patches.csv");
    let timing_log_path = format!("{out_dir}/timing_log.csv");

    {
        let mut f = File::create(&regrid_log_path).unwrap();
        writeln!(f, "step,max_indicator,threshold,flagged_cells,patches,union_i0,union_j0,union_nx,union_ny").unwrap();

        let mut f2 = File::create(&regrid_levels_path).unwrap();
        let mut hdr = String::from("step");
        for lvl in 1..=amr_max_level {
            hdr.push_str(&format!(",l{lvl}_count,l{lvl}_i0,l{lvl}_j0,l{lvl}_nx,l{lvl}_ny"));
        }
        hdr.push('\n');
        f2.write_all(hdr.as_bytes()).unwrap();

        let mut f3 = File::create(&regrid_attempts_path).unwrap();
        let mut hdr2 = String::from("step,max_indicator");
        for lvl in 1..=amr_max_level { hdr2.push_str(&format!(",l{lvl}_count")); }
        hdr2.push('\n');
        f3.write_all(hdr2.as_bytes()).unwrap();

        let mut f4 = File::create(&rmse_log_path).unwrap();
        writeln!(f4, "step,rmse_amr,max_delta_amr,rmse_coarse,max_delta_coarse,patches").unwrap();

        let mut f5 = File::create(&regrid_patches_path).unwrap();
        writeln!(f5, "step,level,patch_id,i0,j0,nx,ny").unwrap();

        let mut f6 = File::create(&timing_log_path).unwrap();
        writeln!(f6, "step,amr_step_ms,fine_step_ms,coarse_step_ms").unwrap();
    }

    // ── Initial regrid ───────────────────────────────────────────────────

    let mut current_patches: Vec<Rect2i> = Vec::new();
    if let Some((new_rects, stats)) =
        maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
    {
        current_patches = new_rects;
        let u = union_rect_or_zero(&current_patches);
        append_line(&regrid_log_path, &format!(
            "0,{:.8e},{:.8e},{},{},{},{},{},{}\n",
            stats.max_indicator, stats.threshold, stats.flagged_cells,
            current_patches.len(), u.i0, u.j0, u.nx, u.ny));

        let mut row = String::from("0");
        for lvl in 1..=amr_max_level {
            let rects = level_rects(&h, lvl);
            let u = union_rect_or_zero(&rects);
            row.push_str(&format!(",{},{},{},{},{}", rects.len(), u.i0, u.j0, u.nx, u.ny));
        }
        row.push('\n');
        append_line(&regrid_levels_path, &row);
        append_regrid_patches_csv(&regrid_patches_path, &h, amr_max_level, 0);
    }

    // ── Stepper + subcycling ─────────────────────────────────────────────

    let mut stepper = AmrStepperRK4::new(&h, true);
    let subcycle_active = stepper.is_subcycling();
    let subcycle_ratio: usize = if subcycle_active {
        (stepper.coarse_dt(&llg, &h) / llg.dt).round() as usize
    } else { 1 };

    let snap_up = |v: usize, r: usize| -> usize {
        if r <= 1 { v } else { ((v + r - 1) / r) * r }
    };
    let steps = snap_up(steps_base, subcycle_ratio);
    let out_every = snap_up(out_every_base, subcycle_ratio);
    let regrid_every = snap_up(regrid_every_base, subcycle_ratio);

    if subcycle_active {
        eprintln!("[bubble_collision] SUBCYCLING: n_levels={}, ratio={}, steps={}, out_every={}, regrid_every={}",
            h.num_levels(), subcycle_ratio, steps, out_every, regrid_every);
    }

    let mut scratch_fine = RK4Scratch::new(fine_grid);
    let mut scratch_coarse = RK4Scratch::new(base_grid);
    let mut b_fine = VectorField2D::new(fine_grid);
    let mut b_coarse = VectorField2D::new(base_grid);
    let local_mask = FieldMask::ExchAnis;

    if do_plots { println!("[bubble_collision] --plots enabled"); }
    if !do_fine { println!("[bubble_collision] --skip-fine-ref: fine reference DISABLED"); }
    if skip_coarse_ref { println!("[bubble_collision] --amr-only: coarse baseline DISABLED"); }
    println!("[bubble_collision] AMR demag mode: {amr_demag_mode_label}");
    if do_ovf {
        println!("[bubble_collision] --ovf enabled");
        ensure_dir(&format!("{out_dir}/ovf_coarse"));
        ensure_dir(&format!("{out_dir}/ovf_fine"));
        ensure_dir(&format!("{out_dir}/ovf_amr"));
    }

    // ── Step 0 outputs ───────────────────────────────────────────────────
    {
        let m_amr_comp = h.flatten_to_uniform_fine();
        let m_amr_fine = ensure_grid(&m_amr_comp, fine_grid);

        let (rmse_amr, maxd_amr) = if do_fine { rmse_and_max_delta(&m_amr_fine, &m_fine) } else { (f64::NAN, f64::NAN) };
        let m_coarse_up = resample_to_grid_bilinear(&m_coarse, fine_grid);
        let (rmse_c, maxd_c) = if do_fine { rmse_and_max_delta(&m_coarse_up, &m_fine) } else { (f64::NAN, f64::NAN) };

        append_line(&rmse_log_path, &format!(
            "0,{:.8e},{:.8e},{:.8e},{:.8e},{}\n", rmse_amr, maxd_amr, rmse_c, maxd_c, current_patches.len()));

        if do_ovf {
            write_ovf_text(&format!("{out_dir}/ovf_coarse/m0000000.ovf"), &m_coarse, "m_coarse");
            if do_fine { write_ovf_text(&format!("{out_dir}/ovf_fine/m0000000.ovf"), &m_fine, "m_fine"); }
            write_ovf_text(&format!("{out_dir}/ovf_amr/m0000000.ovf"), &m_amr_fine, "m_amr");
        }
        if do_plots {
            let levels = all_level_rects(&h, amr_max_level);
            let _ = save_patch_map(&base_grid, &levels,
                &format!("{out_dir}/patch_map_step0000.png"),
                &format!("Patch map (L1 yellow, L2 green, L3 blue)"));
            let _ = save_mesh_zoom_multilevel(&m_amr_fine, &base_grid, ratio, amr_max_level,
                &levels, &format!("{out_dir}/mesh_zoom_step0000.png"),
                "Zoom mesh: in-plane angle + multi-level grid", 30);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Main stepping loop (three simulations in parallel)
    // ══════════════════════════════════════════════════════════════════════

    let t0 = Instant::now();
    let mut t_fine_total = 0.0f64;
    let mut t_coarse_total = 0.0f64;
    let mut t_amr_total = 0.0f64;
    let mut amr_step_count = 0usize;
    let mut recent_amr_ms: Vec<f64> = Vec::with_capacity(16);

    for step in 1..=steps {
        // ── Uniform fine ─────────────────────────────────────────────
        let mut dt_fine_ms = f64::NAN;
        if do_fine {
            let t1 = Instant::now();
            b_fine.set_uniform(0.0, 0.0, 0.0);
            demag_fft_uniform::compute_demag_field_pbc(&fine_grid, &m_fine, &mut b_fine, &mat, 0, 0);
            step_llg_rk4_recompute_field_masked_relax_add(
                &mut m_fine, &llg, &mat, &mut scratch_fine, local_mask, Some(&b_fine));
            let e = t1.elapsed().as_secs_f64();
            t_fine_total += e;
            dt_fine_ms = e * 1e3;
        }

        // ── Uniform coarse ───────────────────────────────────────────
        let mut dt_coarse_ms = f64::NAN;
        if !skip_coarse_ref {
            let t2 = Instant::now();
            b_coarse.set_uniform(0.0, 0.0, 0.0);
            demag_fft_uniform::compute_demag_field_pbc(&base_grid, &m_coarse, &mut b_coarse, &mat, 0, 0);
            step_llg_rk4_recompute_field_masked_relax_add(
                &mut m_coarse, &llg, &mat, &mut scratch_coarse, local_mask, Some(&b_coarse));
            let e = t2.elapsed().as_secs_f64();
            t_coarse_total += e;
            dt_coarse_ms = e * 1e3;
        }

        // ── AMR ──────────────────────────────────────────────────────
        let amr_due = step % subcycle_ratio == 0;
        if amr_due {
            let t3 = Instant::now();
            stepper.step(&mut h, &llg, &mat, local_mask);
            let e = t3.elapsed().as_secs_f64();
            t_amr_total += e;
            amr_step_count += 1;

            let dt_amr_ms = e * 1e3;
            recent_amr_ms.push(dt_amr_ms);
            if recent_amr_ms.len() > 10 { recent_amr_ms.remove(0); }

            append_line(&timing_log_path, &format!(
                "{},{:.3},{:.3},{:.3}\n", step, dt_amr_ms, dt_fine_ms, dt_coarse_ms));

            if amr_step_count <= 3 || amr_step_count % 10 == 0 {
                let avg: f64 = recent_amr_ms.iter().sum::<f64>() / recent_amr_ms.len() as f64;
                let l1 = h.patches.len();
                let l2 = h.patches_l2plus.get(0).map(|v| v.len()).unwrap_or(0);
                let l3 = h.patches_l2plus.get(1).map(|v| v.len()).unwrap_or(0);
                eprintln!("[step {:4}] AMR {:.1}ms (avg {:.1}ms) | L1 {} L2 {} L3 {}",
                    step, dt_amr_ms, avg, l1, l2, l3);
            }
        }

        // ── Regrid ───────────────────────────────────────────────────
        if amr_due && regrid_every > 0 && step % regrid_every == 0 {
            let max_ind = max_indicator_coarse(&h.coarse);
            {
                let mut row = format!("{},{:.8e}", step, max_ind);
                for lvl in 1..=amr_max_level { row.push_str(&format!(",{}", level_patch_count(&h, lvl))); }
                row.push('\n');
                append_line(&regrid_attempts_path, &row);
            }

            if let Some((new_rects, stats)) =
                maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
            {
                current_patches = new_rects;
                let u = union_rect_or_zero(&current_patches);
                append_line(&regrid_log_path, &format!(
                    "{},{:.8e},{:.8e},{},{},{},{},{},{}\n",
                    step, stats.max_indicator, stats.threshold, stats.flagged_cells,
                    current_patches.len(), u.i0, u.j0, u.nx, u.ny));

                let mut row = format!("{step}");
                for lvl in 1..=amr_max_level {
                    let rects = level_rects(&h, lvl);
                    let u = union_rect_or_zero(&rects);
                    row.push_str(&format!(",{},{},{},{},{}", rects.len(), u.i0, u.j0, u.nx, u.ny));
                }
                row.push('\n');
                append_line(&regrid_levels_path, &row);
                append_regrid_patches_csv(&regrid_patches_path, &h, amr_max_level, step);
            }
        }

        // ── Diagnostics + outputs ────────────────────────────────────
        if step % out_every == 0 || step == steps {
            let m_amr_comp = h.flatten_to_uniform_fine();
            let m_amr_fine = ensure_grid(&m_amr_comp, fine_grid);

            let (rmse_amr, maxd_amr) = if do_fine { rmse_and_max_delta(&m_amr_fine, &m_fine) } else { (f64::NAN, f64::NAN) };
            let m_coarse_up = resample_to_grid_bilinear(&m_coarse, fine_grid);
            let (rmse_c, maxd_c) = if do_fine { rmse_and_max_delta(&m_coarse_up, &m_fine) } else { (f64::NAN, f64::NAN) };

            append_line(&rmse_log_path, &format!(
                "{},{:.8e},{:.8e},{:.8e},{:.8e},{}\n",
                step, rmse_amr, maxd_amr, rmse_c, maxd_c, current_patches.len()));

            let mut lvl_counts = String::new();
            for lvl in 1..=amr_max_level {
                if lvl > 1 { lvl_counts.push_str(" | "); }
                lvl_counts.push_str(&format!("L{} {:2}", lvl, level_patch_count(&h, lvl)));
            }
            println!("step {:4} | rmse_amr {:.3e} | rmse_coarse {:.3e} | maxΔ {:.3e} | {}",
                step, rmse_amr, rmse_c, maxd_amr, lvl_counts);

            if do_ovf {
                let fname = format!("m{:07}.ovf", step);
                write_ovf_text(&format!("{out_dir}/ovf_coarse/{fname}"), &m_coarse, "m_coarse");
                if do_fine { write_ovf_text(&format!("{out_dir}/ovf_fine/{fname}"), &m_fine, "m_fine"); }
                write_ovf_text(&format!("{out_dir}/ovf_amr/{fname}"), &m_amr_fine, "m_amr");
            }
            if do_plots {
                let levels = all_level_rects(&h, amr_max_level);
                let _ = save_patch_map(&base_grid, &levels,
                    &format!("{out_dir}/patch_map_step{step:04}.png"),
                    &format!("Patch map step {step} (L1 yellow, L2 green, L3 blue)"));
                let _ = save_mesh_zoom_multilevel(&m_amr_fine, &base_grid, ratio, amr_max_level,
                    &levels, &format!("{out_dir}/mesh_zoom_step{step:04}.png"),
                    &format!("Zoom mesh step {step}: in-plane angle + multi-level grid"), 30);
            }
        }
    }

    let wall = t0.elapsed().as_secs_f64();

    // ── Final CSVs ───────────────────────────────────────────────────────

    let m_amr_comp_final = h.flatten_to_uniform_fine();
    let m_amr_fine_final = ensure_grid(&m_amr_comp_final, fine_grid);
    write_csv_ij_m(&format!("{out_dir}/uniform_coarse_final.csv"), &m_coarse);
    if do_fine { write_csv_ij_m(&format!("{out_dir}/uniform_fine_final.csv"), &m_fine); }
    write_csv_ij_m(&format!("{out_dir}/amr_fine_final.csv"), &m_amr_fine_final);

    if do_fine { write_midline_y(&format!("{out_dir}/lineout_uniform_mid_y.csv"), &m_fine); }
    write_midline_y(&format!("{out_dir}/lineout_amr_mid_y.csv"), &m_amr_fine_final);
    write_midline_y(&format!("{out_dir}/lineout_coarse_mid_y.csv"), &m_coarse);

    // ── Summary ──────────────────────────────────────────────────────────

    let fine_cells_total = fine_grid.nx * fine_grid.ny;
    let fine_cells_in_patches: usize = current_patches.iter()
        .map(|r| (r.nx * ref_ratio_total) * (r.ny * ref_ratio_total)).sum();
    let coverage = fine_cells_in_patches as f64 / fine_cells_total as f64;

    let (rmse_amr_final, maxd_amr_final) = if do_fine {
        rmse_and_max_delta(&m_amr_fine_final, &m_fine)
    } else { (f64::NAN, f64::NAN) };

    let m_coarse_up = resample_to_grid_bilinear(&m_coarse, fine_grid);
    let (rmse_coarse_final, maxd_coarse_final) = if do_fine {
        rmse_and_max_delta(&m_coarse_up, &m_fine)
    } else { (f64::NAN, f64::NAN) };

    let avg_amr_ms = if amr_step_count > 0 { (t_amr_total / amr_step_count as f64) * 1e3 } else { f64::NAN };
    let avg_fine_ms = if do_fine && steps > 0 { (t_fine_total / steps as f64) * 1e3 } else { f64::NAN };
    let speedup_wall = if t_amr_total > 0.0 && do_fine { t_fine_total / t_amr_total } else { f64::NAN };

    println!();
    println!("╔{bar}╗");
    println!("║{:^68}║", "BUBBLE COLLISION — RESULTS SUMMARY");
    println!("╚{bar}╝");

    println!();
    println!("  PROBLEM SETUP");
    println!("  {thin}");
    println!("  Domain             {:.0} nm × {:.0} nm × {:.1} nm", lx*1e9, ly*1e9, dz*1e9);
    println!("  Coarse grid        {:>6} × {:<6}    dx = {:.2} nm", base_nx, base_ny, dx*1e9);
    println!("  Fine-equivalent    {:>6} × {:<6}    dx = {:.3} nm", fine_grid.nx, fine_grid.ny, fine_grid.dx*1e9);
    println!("  Refinement         {} levels ({}:1 total)", amr_max_level, ref_ratio_total);
    println!("  Bubbles            {} (R₀={:.0}nm, w={:.0}nm, Néel)", centers.len(), r0*1e9, wall_width*1e9);
    println!("  Material           Ms={:.0}kA/m A={:.0}pJ/m D={:.1}mJ/m² Ku={:.2}MJ/m³",
        ms/1e3, a_ex/1e-12, d_dmi/1e-3, k_u/1e6);
    println!("  δ_DW = {:.1} nm → spans {:.1} coarse cells, {:.1} fine cells",
        delta_dw*1e9, delta_dw/dx, delta_dw/(dx/ref_ratio_total as f64));
    println!("  Steps              {} (dt = {:.0e} s)", steps, dt);
    if subcycle_active {
        println!("  Subcycling         ON ratio={} → {} coarse steps", subcycle_ratio, steps/subcycle_ratio.max(1));
    }

    println!();
    println!("  AMR PATCHES (final)");
    println!("  {thin}");
    for lvl in 1..=amr_max_level {
        println!("  Level {} patches    {}", lvl, level_patch_count(&h, lvl));
    }
    println!("  Coverage           {:.1}% ({} / {} fine cells)", coverage*100.0, fine_cells_in_patches, fine_cells_total);

    println!();
    println!("  ACCURACY (vs uniform fine reference)");
    println!("  {thin}");
    if do_fine {
        println!("  Uniform coarse     RMSE = {:.4e}   max|Δm| = {:.4e}", rmse_coarse_final, maxd_coarse_final);
        println!("  AMR                RMSE = {:.4e}   max|Δm| = {:.4e}", rmse_amr_final, maxd_amr_final);
        if rmse_coarse_final > 0.0 && rmse_amr_final > 0.0 {
            println!("  AMR is {:.1}× more accurate than coarse", rmse_coarse_final / rmse_amr_final);
        }
    } else {
        println!("  (fine reference skipped)");
    }

    println!();
    println!("  TIMING");
    println!("  {thin}");
    println!("  Total wall clock           {:>8.1} s", wall);
    if do_fine {
        println!("  Uniform fine               {:>8.1} s  ({:.1} ms/step)", t_fine_total, avg_fine_ms);
    }
    if !skip_coarse_ref {
        println!("  Uniform coarse             {:>8.1} s", t_coarse_total);
    }
    println!("  AMR (coarse FFT + patches) {:>8.1} s  ({:.1} ms/step)", t_amr_total, avg_amr_ms);
    if speedup_wall.is_finite() {
        println!();
        println!("  SPEEDUP: {:.1}× (AMR vs uniform fine)", speedup_wall);
    }

    println!();
    println!("  OUTPUT: {out_dir}/");
    println!();
}