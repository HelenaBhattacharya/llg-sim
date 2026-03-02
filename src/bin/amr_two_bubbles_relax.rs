// src/bin/amr_two_bubbles_relax.rs
//
// AMR benchmark: two separated smooth "bubbles" (localized domains), demag ON.
//
// This benchmark mirrors the output structure of `amr_vortex_relax.rs` so it can be
// inspected with `scripts/amr_viewer.py`.
//
// Physics:
//  - exchange + uniaxial anisotropy + demag
//  - uniform coarse/fine demag: FFT uniform (Newell)
//  - AMR demag: controlled by `LLG_AMR_DEMAG_MODE` (run with `all_fft` / `fft` for now)
//
// Outputs in out/amr_two_bubbles_relax:
//  - patch_map_stepXXXX.png              : patch rectangles by refinement level [--plots]
//  - mesh_zoom_stepXXXX.png              : in-plane angle + multi-level grid overlay [--plots]
//  - regrid_log.csv                      : regrid accept events
//  - regrid_levels.csv                   : per-accept union summaries by level (L1..Lmax)
//  - regrid_attempts.csv                 : per-check summary (max_indicator + per-level counts)
//  - rmse_log.csv                        : AMR composite vs uniform fine error vs time
//  - ovf_coarse/mXXXXXXX.ovf             : coarse OVFs [--ovf]
//  - ovf_fine/mXXXXXXX.ovf               : uniform fine OVFs [--ovf]
//  - ovf_amr/mXXXXXXX.ovf                : AMR composite OVFs [--ovf]
//  - *_final.csv                         : final states (coarse/fine/amr)
//  - lineout_*_mid_y.csv                 : midline profiles
//
// Run examples:
//   LLG_AMR_MAX_LEVEL=3 LLG_AMR_DEMAG_MODE=all_fft cargo run --release --bin amr_two_bubbles_relax -- --ovf --plots
//

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
use llg_sim::grid::Grid2D;
use llg_sim::initial_states::seed_smooth_bubbles;
use llg_sim::llg::{RK4Scratch, step_llg_rk4_recompute_field_masked_relax_add};
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

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

fn write_ovf_text(path: &str, m: &VectorField2D, title: &str) {
    // Minimal OOMMF OVF 2.0 ASCII writer for 2D fields (znodes=1).
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

fn append_line(path: &str, line: &str) {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    f.write_all(line.as_bytes()).unwrap();
}

fn append_regrid_patches_csv(path: &str, h: &AmrHierarchy2D, max_level: usize, step: usize) {
    // One row per patch per level (coarse-grid coordinates).
    // CSV columns: step,level,patch_id,i0,j0,nx,ny
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

fn resample_to_grid_bilinear(src: &VectorField2D, dst_grid: Grid2D) -> VectorField2D {
    let mut out = VectorField2D::new(dst_grid);
    for j in 0..out.grid.ny {
        let y = (j as f64 + 0.5) * out.grid.dy;
        for i in 0..out.grid.nx {
            let x = (i as f64 + 0.5) * out.grid.dx;
            let v = sample_bilinear(src, x, y);
            let k = out.grid.idx(i, j);
            out.data[k] = v;
        }
    }
    out
}

fn ensure_grid(src: &VectorField2D, dst_grid: Grid2D) -> VectorField2D {
    if src.grid.nx == dst_grid.nx
        && src.grid.ny == dst_grid.ny
        && src.grid.dx == dst_grid.dx
        && src.grid.dy == dst_grid.dy
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
        l if l >= 2 => h
            .patches_l2plus
            .get(l - 2)
            .map(|v| v.iter().map(|p| p.coarse_rect).collect())
            .unwrap_or_else(Vec::new),
        _ => Vec::new(),
    }
}

fn all_level_rects(h: &AmrHierarchy2D, max_level: usize) -> Vec<Vec<Rect2i>> {
    (1..=max_level).map(|lvl| level_rects(h, lvl)).collect()
}

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

fn save_patch_map(
    base_grid: &Grid2D,
    levels: &[Vec<Rect2i>],
    path: &str,
    caption: &str,
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
            let y0 = (r.j0 as f64) / ny0;
            let x1 = ((r.i0 + r.nx) as f64) / nx0;
            let y1 = ((r.j0 + r.ny) as f64) / ny0;

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

    // Background: in-plane angle
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

    // L0 grid (coarse) in light gray
    let s0 = ref_ratio_total.max(1);
    let mut xx = ((x0 + s0 - 1) / s0) * s0;
    while xx <= x1 {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(xx as i32, y0 as i32), (xx as i32, y1 as i32)],
            RGBColor(180, 180, 180).stroke_width(1),
        )))?;
        xx = xx.saturating_add(s0);
        if s0 == 0 {
            break;
        }
    }
    let mut yy = ((y0 + s0 - 1) / s0) * s0;
    while yy <= y1 {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0 as i32, yy as i32), (x1 as i32, yy as i32)],
            RGBColor(180, 180, 180).stroke_width(1),
        )))?;
        yy = yy.saturating_add(s0);
        if s0 == 0 {
            break;
        }
    }

    // Helpers to draw solid and dashed grids per patch
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
                if xi1 <= xi0 || yi1 <= yi0 {
                    continue;
                }

                chart.draw_series(std::iter::once(PathElement::new(
                    vec![
                        (xi0 as i32, yi0 as i32),
                        (xi1 as i32, yi0 as i32),
                        (xi1 as i32, yi1 as i32),
                        (xi0 as i32, yi1 as i32),
                        (xi0 as i32, yi0 as i32),
                    ],
                    style.clone().stroke_width(3),
                )))?;

                let mut xg = xi0;
                while xg <= xi1 {
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(xg as i32, yi0 as i32), (xg as i32, yi1 as i32)],
                        style.clone().stroke_width(1),
                    )))?;
                    xg = xg.saturating_add(s);
                    if s == 0 {
                        break;
                    }
                }

                let mut yg = yi0;
                while yg <= yi1 {
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(xi0 as i32, yg as i32), (xi1 as i32, yg as i32)],
                        style.clone().stroke_width(1),
                    )))?;
                    yg = yg.saturating_add(s);
                    if s == 0 {
                        break;
                    }
                }
            }
            Ok(())
        };

        // L1/L2 solid
        for (k, rects) in levels.iter().enumerate() {
            let lvl = k + 1;
            if lvl == 1 {
                draw_grid_for_level(rects, lvl, BLACK.filled())?;
            } else if lvl == 2 {
                draw_grid_for_level(rects, lvl, RED.filled())?;
            }
        }
    }

    // L3+ dashed outlines only (light)
    {
        let mut draw_outline_dashed =
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
                    if xi1 <= xi0 || yi1 <= yi0 {
                        continue;
                    }

                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![
                            (xi0 as i32, yi0 as i32),
                            (xi1 as i32, yi0 as i32),
                            (xi1 as i32, yi1 as i32),
                            (xi0 as i32, yi1 as i32),
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
                let c = if lvl == 3 {
                    RGBColor(0, 120, 255)
                } else {
                    RGBColor(160, 80, 255)
                };
                draw_outline_dashed(rects, c)?;
            }
        }
    }

    root.present()?;
    Ok(())
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

fn main() {
    // Flags
    let args: Vec<String> = std::env::args().collect();
    let do_plots = args.iter().any(|a| a == "--plots");
    let do_ovf = args.iter().any(|a| a == "--ovf");

    // AMR max level (nested)
    let amr_max_level: usize = std::env::var("LLG_AMR_MAX_LEVEL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);

    let out_dir = "out/amr_two_bubbles_relax";
    ensure_dir(out_dir);

    // Grids
    let base_grid = Grid2D::new(256, 128, 5e-9, 5e-9, 1e-9);
    let ratio = 2usize;
    let ghost = 2usize;

    let ref_ratio_total = pow_usize(ratio, amr_max_level);
    let fine_grid = Grid2D::new(
        base_grid.nx * ref_ratio_total,
        base_grid.ny * ref_ratio_total,
        base_grid.dx / (ref_ratio_total as f64),
        base_grid.dy / (ref_ratio_total as f64),
        base_grid.dz,
    );

    let steps_base = 2000usize;
    let dt = 5e-14;
    let out_every_base = 100usize;
    let regrid_every_base = 100usize;

    // Material + params (demag ON)
    let mat = Material {
        ms: 8.0e5,
        a_ex: 13e-12,
        k_u: 500.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let llg = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha: 0.5,
        dt,
        b_ext: [0.0, 0.0, 0.0],
    };

    // Two bubbles far apart (centered coordinates)
    let lx = base_grid.nx as f64 * base_grid.dx;
    let centers = [(-0.25 * lx, 0.0), (0.25 * lx, 0.0)];
    let bubble_radius = 50e-9;
    let wall_width = 60e-9;
    let helicity = 0.0;

    // Initial states
    let mut m_coarse = VectorField2D::new(base_grid);
    seed_smooth_bubbles(
        &mut m_coarse,
        &base_grid,
        &centers,
        bubble_radius,
        wall_width,
        helicity,
        1.0,
        None,
    );

    let mut m_coarse_amr = VectorField2D::new(base_grid);
    seed_smooth_bubbles(
        &mut m_coarse_amr,
        &base_grid,
        &centers,
        bubble_radius,
        wall_width,
        helicity,
        1.0,
        None,
    );

    let mut m_fine = VectorField2D::new(fine_grid);
    seed_smooth_bubbles(
        &mut m_fine,
        &fine_grid,
        &centers,
        bubble_radius,
        wall_width,
        helicity,
        1.0,
        None,
    );

    // AMR hierarchy
    let mut h = AmrHierarchy2D::new(base_grid, m_coarse_amr, ratio, ghost);

    // Policies — default grad² relative mode, overridable via env.
    let indicator_kind = IndicatorKind::from_env(); // defaults to Composite { frac: 0.10 }
    let boundary_layer: usize = std::env::var("LLG_AMR_BOUNDARY_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);


    let buffer_cells = 6usize;
    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind,
        buffer_cells,
        boundary_layer,
        connectivity: Connectivity::Eight,
        merge_distance: 4,
        min_patch_area: 64,  // filter tiny slivers (< 8×8 coarse cells)
        max_patches: 0,      // no limit — let bisection + efficiency control count
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

    // Logs
    let regrid_log_path = format!("{out_dir}/regrid_log.csv");
    let regrid_levels_path = format!("{out_dir}/regrid_levels.csv");
    let regrid_attempts_path = format!("{out_dir}/regrid_attempts.csv");
    let rmse_log_path = format!("{out_dir}/rmse_log.csv");
    let regrid_patches_path = format!("{out_dir}/regrid_patches.csv");

    {
        let mut f = File::create(&regrid_log_path).unwrap();
        writeln!(
            f,
            "step,max_indicator,threshold,flagged_cells,patches,union_i0,union_j0,union_nx,union_ny"
        )
        .unwrap();

        let mut f2 = File::create(&regrid_levels_path).unwrap();
        let mut hdr = String::from("step");
        for lvl in 1..=amr_max_level {
            hdr.push_str(&format!(
                ",l{lvl}_count,l{lvl}_i0,l{lvl}_j0,l{lvl}_nx,l{lvl}_ny"
            ));
        }
        hdr.push('\n');
        f2.write_all(hdr.as_bytes()).unwrap();

        let mut f3 = File::create(&regrid_attempts_path).unwrap();
        let mut hdr2 = String::from("step,max_indicator");
        for lvl in 1..=amr_max_level {
            hdr2.push_str(&format!(",l{lvl}_count"));
        }
        hdr2.push('\n');
        f3.write_all(hdr2.as_bytes()).unwrap();

        let mut f4 = File::create(&rmse_log_path).unwrap();
        writeln!(f4, "step,rmse,max_delta,patches").unwrap();

        let mut f5 = File::create(&regrid_patches_path).unwrap();
        writeln!(f5, "step,level,patch_id,i0,j0,nx,ny").unwrap();
    }

    // Initial nested regrid
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

        // levels row
        let mut row = String::from("0");
        for lvl in 1..=amr_max_level {
            let rects = level_rects(&h, lvl);
            let u = union_rect_or_zero(&rects);
            row.push_str(&format!(
                ",{},{},{},{},{}",
                rects.len(),
                u.i0,
                u.j0,
                u.nx,
                u.ny
            ));
        }
        row.push('\n');
        append_line(&regrid_levels_path, &row);
        // Per-patch rectangles by level (step 0)
        append_regrid_patches_csv(&regrid_patches_path, &h, amr_max_level, 0);
    }

    // Stepper
    let mut stepper = AmrStepperRK4::new(&h, true);

    // Subcycling awareness
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
            "[amr_two_bubbles_relax] SUBCYCLING ACTIVE: n_levels={}, subcycle_ratio={}, steps={}, out_every={}, regrid_every={}",
            h.num_levels(), subcycle_ratio, steps, out_every, regrid_every
        );
    }

    let mut scratch_fine = RK4Scratch::new(fine_grid);
    let mut scratch_coarse = RK4Scratch::new(base_grid);
    let mut b_fine = VectorField2D::new(fine_grid);
    let mut b_coarse = VectorField2D::new(base_grid);

    let local_mask = FieldMask::ExchAnis;

    if do_plots {
        println!("[amr_two_bubbles_relax] --plots enabled: will write PNGs to {out_dir}");
    }
    if do_ovf {
        ensure_dir(&format!("{out_dir}/ovf_coarse"));
        ensure_dir(&format!("{out_dir}/ovf_fine"));
        ensure_dir(&format!("{out_dir}/ovf_amr"));
    }

    // Step 0 outputs
    {
        let m_amr_comp = h.flatten_to_uniform_fine();
        let m_amr_fine = ensure_grid(&m_amr_comp, fine_grid);
        let (rmse, maxd) = rmse_and_max_delta(&m_amr_fine, &m_fine);
        append_line(
            &rmse_log_path,
            &format!("0,{:.8e},{:.8e},{}\n", rmse, maxd, current_patches.len()),
        );

        if do_ovf {
            write_ovf_text(
                &format!("{out_dir}/ovf_coarse/m0000000.ovf"),
                &m_coarse,
                "m_coarse",
            );
            write_ovf_text(
                &format!("{out_dir}/ovf_fine/m0000000.ovf"),
                &m_fine,
                "m_fine",
            );
            write_ovf_text(
                &format!("{out_dir}/ovf_amr/m0000000.ovf"),
                &m_amr_fine,
                "m_amr",
            );
        }

        if do_plots {
            let levels = all_level_rects(&h, amr_max_level);
            let _ = save_patch_map(
                &base_grid,
                &levels,
                &format!("{out_dir}/patch_map_step0000.png"),
                "Patch map (L1..Lmax)",
            );
            let _ = save_mesh_zoom_multilevel(
                &m_amr_fine,
                &base_grid,
                ratio,
                amr_max_level,
                &levels,
                &format!("{out_dir}/mesh_zoom_step0000.png"),
                "Zoom mesh: in-plane angle + multi-level grid",
                30,
            );
        }
    }

    // Timings
    let t0 = Instant::now();
    let mut t_demag_fine = 0.0;
    let mut t_demag_coarse = 0.0;
    let mut t_amr_step = 0.0;

    // Main loop
    for step in 1..=steps {
        // Uniform fine
        let t1 = Instant::now();
        b_fine.set_uniform(0.0, 0.0, 0.0);
        demag_fft_uniform::compute_demag_field_pbc(&fine_grid, &m_fine, &mut b_fine, &mat, 0, 0);
        t_demag_fine += t1.elapsed().as_secs_f64();
        step_llg_rk4_recompute_field_masked_relax_add(
            &mut m_fine,
            &llg,
            &mat,
            &mut scratch_fine,
            local_mask,
            Some(&b_fine),
        );

        // Uniform coarse
        let t2 = Instant::now();
        b_coarse.set_uniform(0.0, 0.0, 0.0);
        demag_fft_uniform::compute_demag_field_pbc(
            &base_grid,
            &m_coarse,
            &mut b_coarse,
            &mat,
            0,
            0,
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

        // AMR
        let amr_due = step % subcycle_ratio == 0;
        if amr_due {
            let t3 = Instant::now();
            stepper.step(&mut h, &llg, &mat, local_mask);
            t_amr_step += t3.elapsed().as_secs_f64();
        }

        // Regrid
        if amr_due && regrid_every > 0 && step % regrid_every == 0 {
            let max_ind = max_indicator_coarse(&h.coarse);
            {
                let mut row = format!("{},{:.8e}", step, max_ind);
                for lvl in 1..=amr_max_level {
                    row.push_str(&format!(",{}", level_patch_count(&h, lvl)));
                }
                row.push('\n');
                append_line(&regrid_attempts_path, &row);
            }

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
                    let u = union_rect_or_zero(&rects);
                    row.push_str(&format!(
                        ",{},{},{},{},{}",
                        rects.len(),
                        u.i0,
                        u.j0,
                        u.nx,
                        u.ny
                    ));
                }
                row.push('\n');
                append_line(&regrid_levels_path, &row);
                // Per-patch rectangles by level for this accepted regrid
                append_regrid_patches_csv(&regrid_patches_path, &h, amr_max_level, step);
            }
        }

        // Diagnostics + outputs
        if step % out_every == 0 || step == steps {
            let m_amr_comp = h.flatten_to_uniform_fine();
            let m_amr_fine = ensure_grid(&m_amr_comp, fine_grid);
            let (rmse, maxd) = rmse_and_max_delta(&m_amr_fine, &m_fine);
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

            let mut lvl_counts = String::new();
            for lvl in 1..=amr_max_level {
                if lvl > 1 {
                    lvl_counts.push_str(" | ");
                }
                lvl_counts.push_str(&format!("L{} {:2}", lvl, level_patch_count(&h, lvl)));
            }
            println!(
                "step {:4} | rmse {:.3e} | maxΔ {:.3e} | {}",
                step, rmse, maxd, lvl_counts
            );

            if do_ovf {
                let fname = format!("m{:07}.ovf", step);
                write_ovf_text(
                    &format!("{out_dir}/ovf_coarse/{fname}"),
                    &m_coarse,
                    "m_coarse",
                );
                write_ovf_text(&format!("{out_dir}/ovf_fine/{fname}"), &m_fine, "m_fine");
                write_ovf_text(&format!("{out_dir}/ovf_amr/{fname}"), &m_amr_fine, "m_amr");
            }

            if do_plots {
                let levels = all_level_rects(&h, amr_max_level);
                let _ = save_patch_map(
                    &base_grid,
                    &levels,
                    &format!("{out_dir}/patch_map_step{step:04}.png"),
                    "Patch map (L1..Lmax)",
                );
                let _ = save_mesh_zoom_multilevel(
                    &m_amr_fine,
                    &base_grid,
                    ratio,
                    amr_max_level,
                    &levels,
                    &format!("{out_dir}/mesh_zoom_step{step:04}.png"),
                    "Zoom mesh: in-plane angle + multi-level grid",
                    30,
                );
            }
        }
    }

    let wall = t0.elapsed().as_secs_f64();

    // Final CSVs
    let m_amr_comp_final = h.flatten_to_uniform_fine();
    let m_amr_fine_final = ensure_grid(&m_amr_comp_final, fine_grid);
    write_csv_ij_m(&format!("{out_dir}/uniform_coarse_final.csv"), &m_coarse);
    write_csv_ij_m(&format!("{out_dir}/uniform_fine_final.csv"), &m_fine);
    write_csv_ij_m(&format!("{out_dir}/amr_fine_final.csv"), &m_amr_fine_final);

    write_midline_y(&format!("{out_dir}/lineout_uniform_mid_y.csv"), &m_fine);
    write_midline_y(
        &format!("{out_dir}/lineout_amr_mid_y.csv"),
        &m_amr_fine_final,
    );

    println!();
    println!("AMR two-bubbles relaxation benchmark (demag ON)");
    println!(
        "Base grid: {} x {}   dx={:.3e} dy={:.3e} dz={:.3e}",
        base_grid.nx, base_grid.ny, base_grid.dx, base_grid.dy, base_grid.dz
    );
    println!(
        "Fine grid: {} x {}   dx={:.3e} dy={:.3e}",
        fine_grid.nx, fine_grid.ny, fine_grid.dx, fine_grid.dy
    );
    println!("Steps: {}   dt={:.3e}", steps, dt);
    if subcycle_active {
        println!(
            "Subcycling: ON  n_levels={}  ratio={}  AMR coarse steps: ~{}",
            h.num_levels(), subcycle_ratio, steps / subcycle_ratio.max(1)
        );
    } else {
        println!("Subcycling: OFF");
    }
    println!("Outputs: {out_dir}");
    println!();
    println!("Timing:");
    println!("  total wall time:   {:.3} s", wall);
    println!("  fine demag time:   {:.3} s", t_demag_fine);
    println!("  coarse demag time: {:.3} s", t_demag_coarse);
    println!("  AMR step time:     {:.3} s", t_amr_step);
}