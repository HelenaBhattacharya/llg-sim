// src/bin/amr_masked_disk_relax.rs
//
// AMR masked-disk relaxation benchmark (demag OFF)
// ------------------------------------------------
// Standardised AMR benchmark output contract matching:
//   - amr_vortex_relax.rs
//   - amr_two_bubbles_relax.rs
//
// Physics:
//   - exchange + uniaxial anisotropy (FieldMask::ExchAnis)
//   - demag OFF
//   - geometry mask: disk (vacuum outside disk forced to m=(0,0,0))
//
// AMR:
//   - nested refinement levels L1..LLG_AMR_MAX_LEVEL (ratio=2)
//   - periodic regrid with a grad^2 relative indicator on the coarse grid,
//     evaluated with the geometry mask so vacuum does not dominate.
//
// Outputs in out/amr_masked_disk_relax:
//   - patch_map_stepXXXX.png              : patch rectangles by refinement level [--plots]
//   - mesh_zoom_stepXXXX.png              : in-plane angle + multi-level grid overlay [--plots]
//   - regrid_log.csv                      : accepted regrid events summary
//   - regrid_levels.csv                   : per-accept union summaries for L1..Lmax
//   - regrid_attempts.csv                 : per-check diagnostics (max_indicator + per-level counts)
//   - regrid_patches.csv                  : per-patch rectangles (step,level,patch_id,i0,j0,nx,ny)
//   - rmse_log.csv                        : masked AMR vs uniform fine RMSE vs time
//   - ovf_coarse/mXXXXXXX.ovf             : uniform coarse OVFs [--ovf]
//   - ovf_fine/mXXXXXXX.ovf               : uniform fine reference OVFs [--ovf]
//   - ovf_amr/mXXXXXXX.ovf                : AMR composite OVFs (masked) [--ovf]
//   - *_final.csv                         : final states (coarse/fine/amr)
//   - lineout_*_mid_y.csv                 : midline profiles
//
// Run:
//   LLG_AMR_MAX_LEVEL=3 cargo run --release --bin amr_masked_disk_relax -- --ovf --plots
//

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use plotters::prelude::*;

use llg_sim::amr::indicator::IndicatorKind;
use llg_sim::amr::regrid::maybe_regrid_nested_levels;
use llg_sim::amr::{
    AmrHierarchy2D, AmrStepperRK4, ClusterPolicy, Connectivity, Rect2i, RegridPolicy,
};
use llg_sim::effective_field::FieldMask;
use llg_sim::geometry_mask::{Mask2D, MaskShape, mask_disk};
use llg_sim::grid::Grid2D;
use llg_sim::initial_states::{apply_mask_zero, init_vortex};
use llg_sim::llg::{RK4Scratch, step_llg_rk4_recompute_field_masked_relax_geom};
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
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    // L1
    for (pid, p) in h.patches.iter().enumerate() {
        let r = p.coarse_rect;
        writeln!(
            f,
            "{},{},{},{},{},{},{}",
            step, 1, pid, r.i0, r.j0, r.nx, r.ny
        )
        .unwrap();
    }
    // L2+
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

fn rmse_and_max_delta_masked(a: &VectorField2D, b: &VectorField2D, mask: &Mask2D) -> (f64, f64) {
    assert_eq!(a.grid.nx, b.grid.nx);
    assert_eq!(a.grid.ny, b.grid.ny);
    assert_eq!(mask.len(), a.grid.n_cells());

    let mut sum_sq = 0.0_f64;
    let mut max_dm = 0.0_f64;
    let mut n_used = 0usize;

    for idx in 0..a.grid.n_cells() {
        if !mask[idx] {
            continue;
        }
        n_used += 1;
        let va = a.data[idx];
        let vb = b.data[idx];
        let dx = va[0] - vb[0];
        let dy = va[1] - vb[1];
        let dz = va[2] - vb[2];
        let dm2 = dx * dx + dy * dy + dz * dz;
        sum_sq += dm2;
        let dm = dm2.sqrt();
        if dm > max_dm {
            max_dm = dm;
        }
    }

    let n = (n_used as f64).max(1.0);
    ((sum_sq / n).sqrt(), max_dm)
}

fn max_norm_outside_mask(field: &VectorField2D, mask: &Mask2D) -> f64 {
    assert_eq!(mask.len(), field.grid.n_cells());
    let mut maxn = 0.0_f64;
    for idx in 0..field.grid.n_cells() {
        if mask[idx] {
            continue;
        }
        let v = field.data[idx];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n > maxn {
            maxn = n;
        }
    }
    maxn
}

// Upsample base mask (coarse) to finest uniform grid used for OVFs/metrics.
#[warn(unused_variables)]
#[allow(dead_code)]
fn upsample_mask_from_base(
    base_mask: &Mask2D,
    base: &Grid2D,
    ratio: usize,
    fine_grid: &Grid2D,
) -> Mask2D {
    assert_eq!(base_mask.len(), base.n_cells());
    assert_eq!(fine_grid.nx, base.nx * ratio);
    assert_eq!(fine_grid.ny, base.ny * ratio);

    let fine_nx = fine_grid.nx;
    let mut out: Mask2D = vec![false; fine_grid.n_cells()];

    for j in 0..base.ny {
        for i in 0..base.nx {
            let v = base_mask[i + base.nx * j];
            let i0 = i * ratio;
            let j0 = j * ratio;
            for fj in 0..ratio {
                for fi in 0..ratio {
                    let ii = i0 + fi;
                    let jj = j0 + fj;
                    out[ii + fine_nx * jj] = v;
                }
            }
        }
    }

    out
}

// ---------- Plot helpers (like the other AMR benchmarks) ----------
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

fn save_mesh_zoom_multilevel(
    m_fine: &VectorField2D,
    _base_grid: &Grid2D,
    ratio: usize,
    amr_max_level: usize,
    levels: &[Vec<Rect2i>],
    path: &str,
    caption: &str,
    margin_cells_finest: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let ref_ratio_total = pow_usize(ratio, amr_max_level);

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
        (ref_ratio_total / r_lvl).max(1)
    };

    // L0 grid
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

    // L1/L2 solid grids, L3 outlines
    for (k, rects) in levels.iter().enumerate() {
        let lvl = k + 1;
        if rects.is_empty() {
            continue;
        }
        let s = level_spacing(lvl);
        let col = match lvl {
            1 => BLACK,
            2 => RED,
            3 => RGBColor(0, 120, 255),
            _ => RGBColor(160, 80, 255),
        };

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
                col.stroke_width(3),
            )))?;

            if lvl <= 2 {
                let mut xg = xi0;
                while xg <= xi1 {
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(xg as i32, yi0 as i32), (xg as i32, yi1 as i32)],
                        col.stroke_width(1),
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
                        col.stroke_width(1),
                    )))?;
                    yg = yg.saturating_add(s);
                    if s == 0 {
                        break;
                    }
                }
            }
        }
    }

    root.present()?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let do_plots = args.iter().any(|a| a == "--plots");
    let do_ovf = args.iter().any(|a| a == "--ovf" || a == "--ovfs");

    let out_dir = "out/amr_masked_disk_relax";
    ensure_dir(out_dir);

    let amr_max_level: usize = std::env::var("LLG_AMR_MAX_LEVEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let ratio = 2usize;
    let ghost = 2usize;

    // Base grid
    let base_nx = 192usize;
    let base_ny = 192usize;
    let dx = 5e-9;
    let dy = 5e-9;
    let dz = 1e-9;
    let base_grid = Grid2D::new(base_nx, base_ny, dx, dy, dz);

    let ref_ratio_total = pow_usize(ratio, amr_max_level);
    let fine_grid = Grid2D::new(
        base_nx * ref_ratio_total,
        base_ny * ref_ratio_total,
        dx / (ref_ratio_total as f64),
        dy / (ref_ratio_total as f64),
        dz,
    );

    // Parameters
    let steps_base = 2000usize;
    let dt = 5e-14;
    let out_every_base = 200usize;
    let regrid_every_base = 100usize;

    // demag OFF
    let mat = Material {
        ms: 8.0e5,
        a_ex: 13e-12,
        k_u: -1.0e5,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: false,
        demag_method: DemagMethod::FftUniform,
    };

    let llg = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha: 0.5,
        dt,
        b_ext: [0.0, 0.0, 0.0],
    };

    // Disk geometry
    let lx = base_grid.nx as f64 * base_grid.dx;
    let half = 0.5 * lx;
    let disk_radius = 0.90 * half;
    let mask_base: Mask2D = mask_disk(&base_grid, disk_radius, (0.0, 0.0));

    // Vortex init
    let vortex_center = (0.0, 0.0);
    let polarity = 1.0;
    let chirality = 1.0;
    let core_radius = 30e-9;

    // Uniform coarse baseline
    let mut m_coarse = VectorField2D::new(base_grid);
    init_vortex(
        &mut m_coarse,
        &base_grid,
        vortex_center,
        polarity,
        chirality,
        core_radius,
        Some(&mask_base),
    );
    apply_mask_zero(&mut m_coarse, &mask_base);

    // AMR hierarchy coarse state
    let mut m_coarse_amr = VectorField2D::new(base_grid);
    init_vortex(
        &mut m_coarse_amr,
        &base_grid,
        vortex_center,
        polarity,
        chirality,
        core_radius,
        Some(&mask_base),
    );
    apply_mask_zero(&mut m_coarse_amr, &mask_base);

    let mut h = AmrHierarchy2D::new(base_grid, m_coarse_amr, ratio, ghost);
    let disk_shape = MaskShape::Disk {
        center: (0.0, 0.0),
        radius: disk_radius,
    };
    h.set_geom_shape(disk_shape);

    // Uniform fine reference (will be overwritten to AMR composite at step 0)
    let mut m_fine = VectorField2D::new(fine_grid);
    let fine_mask = MaskShape::Disk {
        center: (0.0, 0.0),
        radius: disk_radius,
    }
    .to_mask(&fine_grid);
    init_vortex(
        &mut m_fine,
        &fine_grid,
        vortex_center,
        polarity,
        chirality,
        core_radius,
        Some(&fine_mask),
    );
    apply_mask_zero(&mut m_fine, &fine_mask);

    // Regrid policies — default grad² relative mode, overridable via env.
    let indicator_kind = IndicatorKind::from_env(); // defaults to Composite { frac: 0.10 }
    let boundary_layer: usize = std::env::var("LLG_AMR_BOUNDARY_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let regrid_policy = RegridPolicy {
        indicator: indicator_kind,
        buffer_cells: 6,
        boundary_layer,
        min_change_cells: 2,
        min_area_change_frac: 0.05,
    };

    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind,
        buffer_cells: 2,
        boundary_layer,
        min_patch_area: 32,
        merge_distance: 1,
        max_patches: 0,
        connectivity: Connectivity::Eight,
        min_efficiency: 0.70,
        max_flagged_fraction: 0.50,  // ADD
        confine_dilation: false,
    };

    // Logs
    let regrid_log_path = format!("{out_dir}/regrid_log.csv");
    let regrid_levels_path = format!("{out_dir}/regrid_levels.csv");
    let regrid_attempts_path = format!("{out_dir}/regrid_attempts.csv");
    let regrid_patches_path = format!("{out_dir}/regrid_patches.csv");
    let rmse_log_path = format!("{out_dir}/rmse_log.csv");

    {
        let mut f = File::create(&regrid_log_path).unwrap();
        writeln!(
            f,
            "step,max_indicator,threshold,flagged_cells,patches,union_i0,union_j0,union_nx,union_ny"
        )
        .unwrap();
        let mut f2 = File::create(&rmse_log_path).unwrap();
        writeln!(f2, "step,rmse,max_delta,patches,leak_amr,leak_fine").unwrap();

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

    // Fair: uniform-fine starts from AMR composite
    m_fine = flatten_to_target_grid(&h, fine_grid);
    apply_mask_zero(&mut m_fine, &fine_mask);

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
            "[amr_masked_disk_relax] SUBCYCLING ACTIVE: n_levels={}, subcycle_ratio={}, steps={}, out_every={}, regrid_every={}",
            h.num_levels(), subcycle_ratio, steps, out_every, regrid_every
        );
    }

    let mut scratch_fine = RK4Scratch::new(fine_grid);
    let mut scratch_coarse = RK4Scratch::new(base_grid);

    if do_ovf {
        ensure_dir(&format!("{out_dir}/ovf_coarse"));
        ensure_dir(&format!("{out_dir}/ovf_fine"));
        ensure_dir(&format!("{out_dir}/ovf_amr"));
    }

    // Step 0 outputs
    {
        let mut m_amr_fine = flatten_to_target_grid(&h, fine_grid);
        apply_mask_zero(&mut m_amr_fine, &fine_mask);
        let (rmse, maxd) = rmse_and_max_delta_masked(&m_amr_fine, &m_fine, &fine_mask);
        let leak_amr = max_norm_outside_mask(&m_amr_fine, &fine_mask);
        let leak_fine = max_norm_outside_mask(&m_fine, &fine_mask);
        append_line(
            &rmse_log_path,
            &format!(
                "0,{:.8e},{:.8e},{},{:.8e},{:.8e}\n",
                rmse,
                maxd,
                current_patches.len(),
                leak_amr,
                leak_fine
            ),
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
            save_patch_map(
                &base_grid,
                &levels,
                &format!("{out_dir}/patch_map_step0000.png"),
                "Patch map (L1..Lmax)",
            )
            .unwrap();
            save_mesh_zoom_multilevel(
                &m_amr_fine,
                &base_grid,
                ratio,
                amr_max_level,
                &levels,
                &format!("{out_dir}/mesh_zoom_step0000.png"),
                "Zoom mesh: in-plane angle + grid",
                30,
            )
            .unwrap();
        }
    }

    if do_plots {
        println!("[amr_masked_disk_relax] --plots enabled: writing PNGs to {out_dir}");
    }
    if do_ovf {
        println!("[amr_masked_disk_relax] --ovf enabled: will write OVFs to {out_dir}/ovf_*");
    }

    let t0 = Instant::now();
    let mut t_amr_step = 0.0;
    let mut amr_step_count = 0usize;
    let mut t_fine_step = 0.0;

    for step in 1..=steps {
        // Uniform fine + coarse updates
        let tf = Instant::now();
        step_llg_rk4_recompute_field_masked_relax_geom(
            &mut m_fine,
            &llg,
            &mat,
            &mut scratch_fine,
            FieldMask::ExchAnis,
            Some(&fine_mask),
        );
        apply_mask_zero(&mut m_fine, &fine_mask);
        t_fine_step += tf.elapsed().as_secs_f64();

        step_llg_rk4_recompute_field_masked_relax_geom(
            &mut m_coarse,
            &llg,
            &mat,
            &mut scratch_coarse,
            FieldMask::ExchAnis,
            Some(&mask_base),
        );
        apply_mask_zero(&mut m_coarse, &mask_base);

        // AMR step
        let amr_due = step % subcycle_ratio == 0;
        if amr_due {
            let t3 = Instant::now();
            stepper.step(&mut h, &llg, &mat, FieldMask::ExchAnis);
            apply_mask_zero(&mut h.coarse, &mask_base);
            t_amr_step += t3.elapsed().as_secs_f64();
            amr_step_count += 1;
        }

        // Regrid
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

        // Output
        if step % out_every == 0 || step == steps {
            let mut m_amr_fine = flatten_to_target_grid(&h, fine_grid);
            apply_mask_zero(&mut m_amr_fine, &fine_mask);

            let (rmse, maxd) = rmse_and_max_delta_masked(&m_amr_fine, &m_fine, &fine_mask);
            let leak_amr = max_norm_outside_mask(&m_amr_fine, &fine_mask);
            let leak_fine = max_norm_outside_mask(&m_fine, &fine_mask);
            append_line(
                &rmse_log_path,
                &format!(
                    "{},{:.8e},{:.8e},{},{:.8e},{:.8e}\n",
                    step,
                    rmse,
                    maxd,
                    current_patches.len(),
                    leak_amr,
                    leak_fine
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
                "step {:4} | rmse(mask) {:.3e} | maxΔ {:.3e} | leak_amr {:.2e} | {}",
                step, rmse, maxd, leak_amr, lvl_counts
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

            if do_plots && step == steps {
                let levels = all_level_rects(&h, amr_max_level);
                save_patch_map(
                    &base_grid,
                    &levels,
                    &format!("{out_dir}/patch_map_step{step:04}.png"),
                    "Patch map (L1..Lmax)",
                )
                .unwrap();
                save_mesh_zoom_multilevel(
                    &m_amr_fine,
                    &base_grid,
                    ratio,
                    amr_max_level,
                    &levels,
                    &format!("{out_dir}/mesh_zoom_step{step:04}.png"),
                    "Zoom mesh: in-plane angle + grid",
                    30,
                )
                .unwrap();
            }
        }
    }

    let wall = t0.elapsed().as_secs_f64();

    // Final CSVs
    let mut m_amr_fine_final = flatten_to_target_grid(&h, fine_grid);
    apply_mask_zero(&mut m_amr_fine_final, &fine_mask);

    write_csv_ij_m(&format!("{out_dir}/uniform_coarse_final.csv"), &m_coarse);
    write_csv_ij_m(&format!("{out_dir}/uniform_fine_final.csv"), &m_fine);
    write_csv_ij_m(&format!("{out_dir}/amr_fine_final.csv"), &m_amr_fine_final);

    write_midline_y(&format!("{out_dir}/lineout_uniform_mid_y.csv"), &m_fine);
    write_midline_y(
        &format!("{out_dir}/lineout_amr_mid_y.csv"),
        &m_amr_fine_final,
    );

    let (rmse_final, maxd_final) =
        rmse_and_max_delta_masked(&m_amr_fine_final, &m_fine, &fine_mask);
    let leak_final = max_norm_outside_mask(&m_amr_fine_final, &fine_mask);

    let fine_cells_total = fine_grid.nx * fine_grid.ny;
    let fine_cells_in_patches: usize = current_patches.iter()
        .map(|r| (r.nx * ref_ratio_total) * (r.ny * ref_ratio_total)).sum();
    let coverage = fine_cells_in_patches as f64 / fine_cells_total as f64;

    let bar = "═".repeat(64);
    let thin = "─".repeat(64);

    let avg_amr_ms = if amr_step_count > 0 { (t_amr_step / amr_step_count as f64) * 1e3 } else { f64::NAN };
    let avg_fine_ms = if steps > 0 { (t_fine_step / steps as f64) * 1e3 } else { f64::NAN };
    let speedup_per_step = if avg_amr_ms > 0.0 && avg_fine_ms.is_finite() { avg_fine_ms / avg_amr_ms } else { f64::NAN };
    let speedup_wall = if t_amr_step > 0.0 { t_fine_step / t_amr_step } else { f64::NAN };

    println!();
    println!("╔{}╗", bar);
    println!("║{:^64}║", "AMR MASKED DISK RELAXATION — RESULTS SUMMARY");
    println!("╚{}╝", bar);

    println!();
    println!("  PROBLEM SETUP");
    println!("  {thin}");
    println!("  Coarse grid        {:>6} × {:<6}    dx = {:.2e} m", base_grid.nx, base_grid.ny, base_grid.dx);
    println!("  Fine-equivalent    {:>6} × {:<6}    dx = {:.2e} m", fine_grid.nx, fine_grid.ny, fine_grid.dx);
    println!("  Thickness                              dz = {:.2e} m", base_grid.dz);
    println!("  Refinement         {} levels (ratio {}:1 per level, {}:1 total)",
        amr_max_level, ratio, ref_ratio_total);
    println!("  Time steps         {}  (dt = {:.2e} s)", steps, dt);
    if subcycle_active {
        println!("  Subcycling         ON   ratio={}   coarse steps={}",
            subcycle_ratio, steps / subcycle_ratio.max(1));
    } else {
        println!("  Subcycling         OFF (flat stepping)");
    }
    println!("  Demag              OFF (exchange + anisotropy only)");
    println!("  Geometry           circular disk mask");

    println!();
    println!("  AMR PATCHES (final state)");
    println!("  {thin}");
    for lvl in 1..=amr_max_level {
        println!("  Level {} patches    {}", lvl, level_patch_count(&h, lvl));
    }
    println!("  Fine cells total   {:>12}   ({} × {})", fine_cells_total, fine_grid.nx, fine_grid.ny);
    println!("  Fine cells in AMR  {:>12}   ({:.1}% coverage)", fine_cells_in_patches, coverage * 100.0);
    println!("  Cell savings       {:>12}   ({:.1}% fewer cells)",
        fine_cells_total - fine_cells_in_patches, (1.0 - coverage) * 100.0);

    println!();
    println!("  ACCURACY (AMR vs uniform fine reference)");
    println!("  {thin}");
    println!("  Final RMSE(mask)   {:.4e}", rmse_final);
    println!("  Final max |Δm|     {:.4e}", maxd_final);
    println!("  Mask leak (AMR)    {:.3e}", leak_final);

    println!();
    println!("  TIMING");
    println!("  {thin}");
    println!("  Total wall clock           {:>8.1} s", wall);
    println!();
    println!("  Uniform fine (reference)   {:>8.1} s   ({} steps × {:.1} ms/step)",
        t_fine_step, steps, avg_fine_ms);
    println!("  AMR (subcycled)            {:>8.1} s   ({} steps × {:.1} ms/step)",
        t_amr_step, amr_step_count, avg_amr_ms);

    println!();
    println!("  SPEEDUP");
    println!("  {thin}");
    if speedup_per_step.is_finite() {
        println!("  Per-step:  {:.1} ms (fine)  vs  {:.1} ms (AMR)  →  {:.1}× faster",
            avg_fine_ms, avg_amr_ms, speedup_per_step);
    }
    if speedup_wall.is_finite() {
        println!("  Wall time: {:.1} s  (fine)  vs  {:.1} s  (AMR)  →  {:.1}× faster",
            t_fine_step, t_amr_step, speedup_wall);
    }

    println!();
    println!("  OUTPUT: {out_dir}/");
    println!("  {thin}");
    println!("  rmse_log.csv       RMSE vs step");
    println!("  regrid_log.csv     regrid events");
    if do_ovf {
        println!("  ovf_coarse/        coarse OVF snapshots");
        println!("  ovf_fine/          fine reference OVF snapshots");
        println!("  ovf_amr/           AMR composite OVF snapshots");
    }
    if do_plots {
        println!("  *.png              patch maps + mesh zoom plots");
    }
    println!();
}