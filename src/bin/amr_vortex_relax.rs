// src/bin/amr_vortex_relax.rs
//
// AMR vortex relaxation benchmark with:
//  - uniform coarse baseline
//  - uniform fine reference
//  - AMR (coarse + patches) using Demag Bridge B (FFT on flattened fine composite)
//
// Outputs in out/amr_vortex_relax:
//  - patch_map_stepXXXX.png              : patch rectangles by refinement level (Fig.4-style) [--plots only]
//  - mesh_zoom_stepXXXX.png              : in-plane angle + multi-level grid overlay (Fig.5-style) [--plots only]
//  - regrid_log.csv                      : level-1 regrid events (existing ClusterStats)
//  - regrid_levels.csv                   : per-event summary of L1/L2 patch rectangles
//  - regrid_attempts.csv                 : per-check summary (includes max_theta, L1/L2 counts)
//  - rmse_log.csv                        : AMR vs uniform reference error vs time
//  - ovf_coarse/mXXXXXXX.ovf             : coarse OVFs for 3D post-processing (Fig.8/9-style) [--ovf]
//  - ovf_fine/mXXXXXXX.ovf               : uniform reference OVFs [--ovf]
//  - ovf_amr/mXXXXXXX.ovf                : AMR composite OVFs [--ovf]
//  - *_final.csv                         : final states (coarse/fine/amr)
//  - lineout_*_mid_y.csv                 : midline profiles for quick sanity checks

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use plotters::prelude::*;

use llg_sim::effective_field::{demag_fft_uniform, FieldMask};
use llg_sim::geometry_mask::Mask2D;
use llg_sim::grid::Grid2D;
use llg_sim::initial_states;
use llg_sim::llg::{step_llg_rk4_recompute_field_masked_relax_add, RK4Scratch};
use llg_sim::params::{DemagMethod, LLGParams, Material, GAMMA_E_RAD_PER_S_T};
use llg_sim::vector_field::VectorField2D;

use llg_sim::amr::{
    AmrHierarchy2D, AmrStepperRK4, ClusterPolicy, Connectivity, Rect2i, RegridPolicy,
};
use llg_sim::amr::regrid::maybe_regrid_nested_levels;
use llg_sim::amr::indicator::indicator_angle_max_forward_geom;

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

// ---------- Metrics ----------
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

fn max_theta_coarse(coarse: &VectorField2D, geom_mask: Option<&[bool]>) -> f64 {
    let nx = coarse.grid.nx;
    let ny = coarse.grid.ny;
    let mut max_th = 0.0_f64;
    for j in 0..ny {
        for i in 0..nx {
            let th = indicator_angle_max_forward_geom(coarse, i, j, geom_mask);
            if th > max_th {
                max_th = th;
            }
        }
    }
    max_th
}

fn union_rect_or_zero(rects: &[Rect2i]) -> Rect2i {
    union_rect(rects).unwrap_or(Rect2i { i0: 0, j0: 0, nx: 0, ny: 0 })
}

fn level_rects_l2(h: &AmrHierarchy2D) -> Vec<Rect2i> {
    h.patches_l2plus
        .get(0)
        .map(|v| v.iter().map(|p| p.coarse_rect).collect())
        .unwrap_or_else(Vec::new)
}

fn level_rects_l3(h: &AmrHierarchy2D) -> Vec<Rect2i> {
    h.patches_l2plus
        .get(1)
        .map(|v| v.iter().map(|p| p.coarse_rect).collect())
        .unwrap_or_else(Vec::new)
}

// ---------- Local indicator (only used for plotting indicator maps) ----------
#[allow(dead_code)]
fn indicator_grad2_forward_geom_local(
    field: &VectorField2D,
    i: usize,
    j: usize,
    geom_mask: Option<&[bool]>,
) -> f64 {
    let nx = field.grid.nx;
    let ny = field.grid.ny;

    let idx0 = idx(i, j, nx);
    if let Some(mask) = geom_mask {
        if !mask[idx0] {
            return 0.0;
        }
    }

    let v0 = field.data[idx0];

    let sample = |ii: usize, jj: usize| -> [f64; 3] {
        let ii = ii.min(nx - 1);
        let jj = jj.min(ny - 1);
        let k = idx(ii, jj, nx);
        if let Some(mask) = geom_mask {
            if !mask[k] {
                return v0;
            }
        }
        field.data[k]
    };

    let vxp = sample(i + 1, j);
    let vyp = sample(i, j + 1);

    let dmx_dx = vxp[0] - v0[0];
    let dmy_dx = vxp[1] - v0[1];
    let dmz_dx = vxp[2] - v0[2];

    let dmx_dy = vyp[0] - v0[0];
    let dmy_dy = vyp[1] - v0[1];
    let dmz_dy = vyp[2] - v0[2];

    dmx_dx * dmx_dx
        + dmy_dx * dmy_dx
        + dmz_dx * dmz_dx
        + dmx_dy * dmx_dy
        + dmy_dy * dmy_dy
        + dmz_dy * dmz_dy
}

// ---------- Colour maps ----------
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

#[allow(dead_code)]
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

// ---------- Plot helpers ----------
#[allow(dead_code)]
fn save_angle_map(
    m: &VectorField2D,
    fine_patches: &[Rect2i],
    path: &str,
    caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = m.grid.nx as i32;
    let ny = m.grid.ny as i32;

    let root = BitMapBackend::new(path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 24))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..nx, 0..ny)?;

    chart.configure_mesh().disable_mesh().draw()?;

    chart.draw_series((0..m.grid.ny).flat_map(|j| {
        (0..m.grid.nx).map(move |i| {
            let v = m.data[idx(i, j, m.grid.nx)];
            let phi = v[1].atan2(v[0]);
            let h = (phi + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
            let col = hsv_to_rgb(h, 1.0, 1.0);
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    for r in fine_patches {
        let x0 = r.i0 as i32;
        let y0 = r.j0 as i32;
        let x1 = (r.i0 + r.nx) as i32;
        let y1 = (r.j0 + r.ny) as i32;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            BLACK.stroke_width(3),
        )))?;
    }

    root.present()?;
    Ok(())
}

#[allow(dead_code)]
fn save_mz_map(
    m: &VectorField2D,
    fine_patches: &[Rect2i],
    path: &str,
    caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = m.grid.nx as i32;
    let ny = m.grid.ny as i32;

    let root = BitMapBackend::new(path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 24))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..nx, 0..ny)?;

    chart.configure_mesh().disable_mesh().draw()?;

    chart.draw_series((0..m.grid.ny).flat_map(|j| {
        (0..m.grid.nx).map(move |i| {
            let v = m.data[idx(i, j, m.grid.nx)];
            let col = mz_to_bwr(v[2]);
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    for r in fine_patches {
        let x0 = r.i0 as i32;
        let y0 = r.j0 as i32;
        let x1 = (r.i0 + r.nx) as i32;
        let y1 = (r.j0 + r.ny) as i32;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            BLACK.stroke_width(3),
        )))?;
    }

    root.present()?;
    Ok(())
}

#[allow(dead_code)]
fn save_refine_layout(
    fine_grid: &Grid2D,
    fine_patches: &[Rect2i],
    path: &str,
    caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nx = fine_grid.nx as i32;
    let ny = fine_grid.ny as i32;

    let mut covered = vec![0u8; fine_grid.nx * fine_grid.ny];
    for r in fine_patches {
        for j in r.j0..(r.j0 + r.ny) {
            for i in r.i0..(r.i0 + r.nx) {
                covered[idx(i, j, fine_grid.nx)] = 255u8;
            }
        }
    }

    let root = BitMapBackend::new(path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 24))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..nx, 0..ny)?;

    chart.configure_mesh().disable_mesh().draw()?;

    let covered_ref = &covered;
    chart.draw_series((0..fine_grid.ny).flat_map(|j| {
        (0..fine_grid.nx).map(move |i| {
            let v = covered_ref[idx(i, j, fine_grid.nx)] as f64 / 255.0;
            let g = (v * 255.0) as u8;
            let col = RGBColor(g, g, g);
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    root.present()?;
    Ok(())
}

#[allow(dead_code)]
fn save_indicator_grad2(
    coarse: &VectorField2D,
    geom_mask: Option<&[bool]>,
    path: &str,
    caption: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let nxu = coarse.grid.nx;
    let nyu = coarse.grid.ny;
    let nx = nxu as i32;
    let ny = nyu as i32;

    let mut max_ind: f64 = 0.0;
    let mut ind = vec![0.0_f64; nxu * nyu];
    for j in 0..nyu {
        for i in 0..nxu {
            let v = indicator_grad2_forward_geom_local(coarse, i, j, geom_mask);
            ind[idx(i, j, nxu)] = v;
            if v > max_ind {
                max_ind = v;
            }
        }
    }
    if max_ind <= 0.0 {
        max_ind = 1.0;
    }

    let root = BitMapBackend::new(path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 24))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..nx, 0..ny)?;

    chart.configure_mesh().disable_mesh().draw()?;

    let ind_ref = &ind;
    chart.draw_series((0..nyu).flat_map(|j| {
        (0..nxu).map(move |i| {
            let v = (ind_ref[idx(i, j, nxu)] / max_ind).clamp(0.0, 1.0);
            let g = (v * 255.0) as u8;
            let col = RGBColor(g, g, g);
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    root.present()?;
    Ok(())
}

#[allow(dead_code)]
fn save_angle_zoom_with_patch_grid(
    m_fine: &VectorField2D,
    fine_patches: &[Rect2i],
    path: &str,
    caption: &str,
    margin: usize,
    grid_step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let u = match union_rect(fine_patches) {
        Some(u) => u,
        None => return Ok(()),
    };

    let nx = m_fine.grid.nx;
    let ny = m_fine.grid.ny;

    let x0 = u.i0.saturating_sub(margin);
    let y0 = u.j0.saturating_sub(margin);
    let x1 = (u.i0 + u.nx + margin).min(nx);
    let y1 = (u.j0 + u.ny + margin).min(ny);

    let root = BitMapBackend::new(path, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 24))
        .margin(10)
        .set_all_label_area_size(0)
        .build_cartesian_2d(x0 as i32..x1 as i32, y0 as i32..y1 as i32)?;

    chart.configure_mesh().disable_mesh().draw()?;

    chart.draw_series((y0..y1).flat_map(|j| {
        (x0..x1).map(move |i| {
            let v = m_fine.data[idx(i, j, m_fine.grid.nx)];
            let phi = v[1].atan2(v[0]);
            let h = (phi + std::f64::consts::PI) / (2.0 * std::f64::consts::PI);
            let col = hsv_to_rgb(h, 1.0, 1.0);
            Rectangle::new(
                [(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)],
                col.filled(),
            )
        })
    }))?;

    for r in fine_patches {
        let rx0 = r.i0 as i32;
        let ry0 = r.j0 as i32;
        let rx1 = (r.i0 + r.nx) as i32;
        let ry1 = (r.j0 + r.ny) as i32;

        chart.draw_series(std::iter::once(PathElement::new(
            vec![(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1), (rx0, ry0)],
            BLACK.stroke_width(3),
        )))?;

        if grid_step > 0 {
            for xi in (r.i0..=r.i0 + r.nx).step_by(grid_step) {
                let x = xi as i32;
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(x, ry0), (x, ry1)],
                    BLACK.stroke_width(1),
                )))?;
            }
            for yi in (r.j0..=r.j0 + r.ny).step_by(grid_step) {
                let y = yi as i32;
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(rx0, y), (rx1, y)],
                    BLACK.stroke_width(1),
                )))?;
            }
        }
    }

    root.present()?;
    Ok(())
}

fn save_patch_map(
    base_grid: &Grid2D,
    l1: &[Rect2i],
    l2: &[Rect2i],
    l3: &[Rect2i],
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

    // Draw L1 as filled yellow rectangles, L2 as filled green rectangles.
    // Order: L1 first, then L2 on top.
    for r in l1 {
        let x0 = (r.i0 as f64) / nx0;
        let y0 = (r.j0 as f64) / ny0;
        let x1 = ((r.i0 + r.nx) as f64) / nx0;
        let y1 = ((r.j0 + r.ny) as f64) / ny0;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y0), (x1, y1)],
            RGBColor(240, 220, 0).filled(),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            BLACK.stroke_width(2),
        )))?;
    }

    for r in l2 {
        let x0 = (r.i0 as f64) / nx0;
        let y0 = (r.j0 as f64) / ny0;
        let x1 = ((r.i0 + r.nx) as f64) / nx0;
        let y1 = ((r.j0 + r.ny) as f64) / ny0;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y0), (x1, y1)],
            RGBColor(0, 200, 0).filled(),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            RED.stroke_width(2),
        )))?;
    }

    for r in l3 {
        let x0 = (r.i0 as f64) / nx0;
        let y0 = (r.j0 as f64) / ny0;
        let x1 = ((r.i0 + r.nx) as f64) / nx0;
        let y1 = ((r.j0 + r.ny) as f64) / ny0;

        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y0), (x1, y1)],
            RGBColor(0, 120, 255).filled(),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            RGBColor(0, 60, 140).stroke_width(2),
        )))?;
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
    l1: &[Rect2i],
    l2: &[Rect2i],
    l3: &[Rect2i],
    path: &str,
    caption: &str,
    margin_cells_finest: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Plot on the finest uniform grid (m_fine.grid). We create a zoom window around the L2 union
    // if present, else around the L1 union.
    let ref_ratio_total = pow_usize(ratio, amr_max_level);

    let target = if !l2.is_empty() { l2 } else if !l1.is_empty() { l1 } else { l3 };
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

    // Background: in-plane angle (HSV)
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

    // Draw grid lines for each level inside its patches.
    // In finest coordinates, a level-L cell spans s = ref_ratio_total / ratio^L.
    // L1: black thin; L2: red thin.

    let level_spacing = |lvl: usize| -> usize {
        let r_lvl = pow_usize(ratio, lvl);
        let s = ref_ratio_total / r_lvl;
        s.max(1)
    };

    // L0 (coarse) grid overlay in light gray across the zoom window.
    // Each coarse cell spans `ref_ratio_total` cells on the finest uniform grid.
    let s0 = ref_ratio_total.max(1);

    // Vertical L0 lines
    let mut x = ((x0 + s0 - 1) / s0) * s0;
    while x <= x1 {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x as i32, y0 as i32), (x as i32, y1 as i32)],
            RGBColor(180, 180, 180).stroke_width(1),
        )))?;
        x = x.saturating_add(s0);
        if s0 == 0 {
            break;
        }
    }

    // Horizontal L0 lines
    let mut y = ((y0 + s0 - 1) / s0) * s0;
    while y <= y1 {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0 as i32, y as i32), (x1 as i32, y as i32)],
            RGBColor(180, 180, 180).stroke_width(1),
        )))?;
        y = y.saturating_add(s0);
        if s0 == 0 {
            break;
        }
    }

    // ---- L1/L2 solid grids ----
{
    let mut draw_grid_for_level =
        |rects: &[Rect2i], lvl: usize, style: ShapeStyle| -> Result<(), Box<dyn std::error::Error>> {
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

                let mut xx = xi0;
                while xx <= xi1 {
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(xx as i32, yi0 as i32), (xx as i32, yi1 as i32)],
                        style.clone().stroke_width(1),
                    )))?;
                    xx = xx.saturating_add(s);
                    if s == 0 {
                        break;
                    }
                }

                let mut yy = yi0;
                while yy <= yi1 {
                    chart.draw_series(std::iter::once(PathElement::new(
                        vec![(xi0 as i32, yy as i32), (xi1 as i32, yy as i32)],
                        style.clone().stroke_width(1),
                    )))?;
                    yy = yy.saturating_add(s);
                    if s == 0 {
                        break;
                    }
                }
            }
            Ok(())
        };

    draw_grid_for_level(l1, 1, BLACK.filled())?;
    draw_grid_for_level(l2, 2, RED.filled())?;
} // closure dropped here

// ---- L3 dashed grid ----
{
    let mut draw_grid_for_level_dashed =
        |rects: &[Rect2i], lvl: usize, color: RGBColor| -> Result<(), Box<dyn std::error::Error>> {
            let s = level_spacing(lvl);
            let dash = 6usize;
            let gap = 4usize;

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
                    color.stroke_width(3),
                )))?;

                let mut xx = xi0;
                while xx <= xi1 {
                    let mut yy = yi0;
                    while yy < yi1 {
                        let y_end = (yy + dash).min(yi1);
                        chart.draw_series(std::iter::once(PathElement::new(
                            vec![(xx as i32, yy as i32), (xx as i32, y_end as i32)],
                            color.stroke_width(1),
                        )))?;
                        yy = yy.saturating_add(dash + gap);
                        if dash + gap == 0 {
                            break;
                        }
                    }
                    xx = xx.saturating_add(s);
                    if s == 0 {
                        break;
                    }
                }

                let mut yy = yi0;
                while yy <= yi1 {
                    let mut xx2 = xi0;
                    while xx2 < xi1 {
                        let x_end = (xx2 + dash).min(xi1);
                        chart.draw_series(std::iter::once(PathElement::new(
                            vec![(xx2 as i32, yy as i32), (x_end as i32, yy as i32)],
                            color.stroke_width(1),
                        )))?;
                        xx2 = xx2.saturating_add(dash + gap);
                        if dash + gap == 0 {
                            break;
                        }
                    }
                    yy = yy.saturating_add(s);
                    if s == 0 {
                        break;
                    }
                }
            }
            Ok(())
        };

    draw_grid_for_level_dashed(l3, 3, RGBColor(0, 120, 255))?;
}

    root.present()?;
    Ok(())
}

// ---------- CSV output ----------
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

fn main() {
    // ---- CLI flags (small + explicit) ----
    // Default: NO plots (fast), only CSV outputs.
    // Enable plots with:
    //   cargo run --release --bin amr_vortex_relax -- --plots
    let args: Vec<String> = std::env::args().collect();
    let do_plots = args.iter().any(|a| a == "--plots");

    let do_ovf = args.iter().any(|a| a == "--ovf");

    // Nested refinement control (default: level 1 only).
    let amr_max_level: usize = std::env::var("LLG_AMR_MAX_LEVEL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);

    // ---------- parameters ----------
    let out_dir = "out/amr_vortex_relax";
    ensure_dir(out_dir);

    let base_nx = 192usize;
    let base_ny = 192usize;
    let dx = 5.0e-9;
    let dy = 5.0e-9;
    let dz = 1.0e-9;
    let refine_ratio = 2usize;
    let ghost = 2usize;

    let ref_ratio_total = pow_usize(refine_ratio, amr_max_level);
    let fine_nx = base_nx * ref_ratio_total;
    let fine_ny = base_ny * ref_ratio_total;

    let dt = 5.0e-14;
    let steps = 300usize;

    // Logging cadence (always on); plotting cadence is tied to this when --plots is enabled.
    let out_every = 100usize;
    let regrid_every = 100usize;

    let pbcx = 0usize;
    let pbcy = 0usize;

    // ---------- grids ----------
    let base_grid = Grid2D::new(base_nx, base_ny, dx, dy, dz);
    let fine_grid = Grid2D::new(
        fine_nx,
        fine_ny,
        dx / ref_ratio_total as f64,
        dy / ref_ratio_total as f64,
        dz,
    );

    let base_mask: Mask2D = vec![true; base_grid.n_cells()];

    // ---------- material + params ----------
    let mat = Material {
        ms: 8.0e5,
        a_ex: 1.3e-11,
        k_u: 0.0,
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

    // ---------- initial states ----------
    let vortex_center = (0.0, 0.0);
    let polarity = 1.0;
    let chirality = 1.0;
    let core_radius = 3.0 * base_grid.dx;

    // Uniform coarse baseline
    let mut m_coarse = VectorField2D::new(base_grid);
    initial_states::init_vortex(
        &mut m_coarse,
        &base_grid,
        vortex_center,
        polarity,
        chirality,
        core_radius,
        None,
    );

    // AMR hierarchy starts from its own coarse field
    let mut m_coarse_amr = VectorField2D::new(base_grid);
    initial_states::init_vortex(
        &mut m_coarse_amr,
        &base_grid,
        vortex_center,
        polarity,
        chirality,
        core_radius,
        None,
    );

    let mut h = AmrHierarchy2D::new(base_grid, m_coarse_amr, refine_ratio, ghost);
    h.set_geom_mask(base_mask);

    // Uniform fine reference (temporary init; will be overwritten for fair comparison)
    let mut m_fine = VectorField2D::new(fine_grid);
    initial_states::init_vortex(
        &mut m_fine,
        &fine_grid,
        vortex_center,
        polarity,
        chirality,
        core_radius,
        None,
    );

    // ---------- AMR policies (Milestone 1: angle-based absolute threshold) ----------
    //
    // We use the new dual-semantics indicator policy:
    //   - indicator_frac >= 0.0 : grad^2 relative threshold (frac * max)
    //   - indicator_frac < 0.0  : absolute angle threshold (radians), theta_refine = -indicator_frac
    //
    // Typical coarse-grid (r=2) refine angles: 20°–30°.
    let theta_refine_deg: f64 = 25.0;
    let theta_refine: f64 = theta_refine_deg.to_radians();

    let regrid_policy = RegridPolicy {
        // Negative => angle mode.
        indicator_frac: -theta_refine,
        buffer_cells: 6,
        min_change_cells: 1,
        min_area_change_frac: 0.02,
    };
    let cluster_policy = ClusterPolicy {
        // Match the regrid policy mode/threshold.
        indicator_frac: regrid_policy.indicator_frac,
        buffer_cells: regrid_policy.buffer_cells,
        min_patch_area: 64,
        merge_distance: 2,
        max_patches: 8,
        connectivity: Connectivity::Eight,
    };

    // Logs
    let regrid_log_path = format!("{out_dir}/regrid_log.csv");
    let rmse_log_path = format!("{out_dir}/rmse_log.csv");
    {
        let mut f = File::create(&regrid_log_path).unwrap();
        writeln!(
            f,
            "step,max_indicator,threshold,flagged_cells,patches,union_i0,union_j0,union_nx,union_ny"
        )
        .unwrap();
        let mut f2 = File::create(&rmse_log_path).unwrap();
        writeln!(f2, "step,rmse,max_delta,patches").unwrap();

        let mut f3 = File::create(format!("{out_dir}/regrid_levels.csv")).unwrap();
        writeln!(
            f3,
            "step,l1_count,l2_count,l3_count,l1_i0,l1_j0,l1_nx,l1_ny,l2_i0,l2_j0,l2_nx,l2_ny,l3_i0,l3_j0,l3_nx,l3_ny"
        )
        .unwrap();

        let mut f4 = File::create(format!("{out_dir}/regrid_attempts.csv")).unwrap();
        writeln!(f4, "step,max_theta_rad,max_theta_deg,l1_count,l2_count,l3_count").unwrap();
    }

    // initial regrid
    let mut current_patches: Vec<Rect2i> = Vec::new();
    if let Some((new_rects, stats)) =
        maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
    {
        current_patches = new_rects.clone();
        let u = union_rect(&new_rects);
        let (ui0, uj0, unx, uny) = u
            .map(|r| (r.i0, r.j0, r.nx, r.ny))
            .unwrap_or((0, 0, 0, 0));
        append_line(
            &regrid_log_path,
            &format!(
                "0,{:.8e},{:.8e},{},{},{},{},{},{}\n",
                stats.max_indicator,
                stats.threshold,
                stats.flagged_cells,
                new_rects.len(),
                ui0,
                uj0,
                unx,
                uny
            ),
        );
        // Level summary log (step 0)
        let l1 = current_patches.clone();
        let l2 = level_rects_l2(&h);
        let l3 = level_rects_l3(&h);
        let u1 = union_rect_or_zero(&l1);
        let u2 = union_rect_or_zero(&l2);
        let u3 = union_rect_or_zero(&l3);
        
        append_line(
            &format!("{out_dir}/regrid_levels.csv"),
            &format!(
                "0,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                l1.len(),
                l2.len(),
                l3.len(),
                u1.i0,
                u1.j0,
                u1.nx,
                u1.ny,
                u2.i0,
                u2.j0,
                u2.nx,
                u2.ny,
                u3.i0,
                u3.j0,
                u3.nx,
                u3.ny,
            ),
        );

        let th = max_theta_coarse(&h.coarse, h.geom_mask());
        append_line(
            &format!("{out_dir}/regrid_attempts.csv"),
            &format!(
                "0,{:.8e},{:.4},{},{},{}\n",
                th,
                th.to_degrees(),
                l1.len(),
                l2.len(),
                l3.len()
            ),
        );
    }

    // ---- FAIR COMPARISON FIX ----
    // Start the uniform-fine reference from the *same* composite field as AMR at step 0.
    // This makes RMSE(step=0) ~ 0, so RMSE tracks *algorithmic* differences, not init mismatch.
    m_fine = flatten_to_target_grid(&h, fine_grid);

    // ---------- stepper ----------
    let mut stepper = AmrStepperRK4::new(&h, true);

    let mut scratch_fine = RK4Scratch::new(fine_grid);
    let mut scratch_coarse = RK4Scratch::new(base_grid);
    let mut b_fine = VectorField2D::new(fine_grid);
    let mut b_coarse = VectorField2D::new(base_grid);

    let local_mask = FieldMask::ExchAnis;

    // ---------- timings ----------
    let t0 = Instant::now();
    let mut t_demag_fine = 0.0;
    let mut t_demag_coarse = 0.0;
    let mut t_amr_step = 0.0;

    if do_plots {
        println!("[amr_vortex_relax] --plots enabled: will write PNGs to {out_dir}");
    } else {
        println!("[amr_vortex_relax] plots disabled (default): writing CSV outputs only. Use `-- --plots` to enable PNGs.");
    }

    if regrid_policy.indicator_frac < 0.0 {
        let theta_refine = -regrid_policy.indicator_frac;
        println!(
            "[amr_vortex_relax] refinement indicator: angle threshold {:.3} rad ({:.1} deg)",
            theta_refine,
            theta_refine.to_degrees()
        );
    } else {
        println!(
            "[amr_vortex_relax] refinement indicator: grad^2 relative threshold frac={:.3}",
            regrid_policy.indicator_frac
        );
    }

    // output at step 0 (RMSE always; plots only if requested)
    {
        let fine_patches = patches_to_fine_rects(&current_patches, ref_ratio_total);

        let m_amr_fine = flatten_to_target_grid(&h, fine_grid);
        let (rmse, maxd) = rmse_and_max_delta(&m_amr_fine, &m_fine);
        append_line(
            &rmse_log_path,
            &format!("0,{:.8e},{:.8e},{}\n", rmse, maxd, fine_patches.len()),
        );

        if do_ovf {
            ensure_dir(&format!("{out_dir}/ovf_coarse"));
            ensure_dir(&format!("{out_dir}/ovf_fine"));
            ensure_dir(&format!("{out_dir}/ovf_amr"));
            write_ovf_text(&format!("{out_dir}/ovf_coarse/m0000000.ovf"), &m_coarse, "m_coarse");
            write_ovf_text(&format!("{out_dir}/ovf_fine/m0000000.ovf"), &m_fine, "m_fine");
            write_ovf_text(&format!("{out_dir}/ovf_amr/m0000000.ovf"), &m_amr_fine, "m_amr");
        }

        if do_plots {
            let l1 = current_patches.clone();
            let l2 = level_rects_l2(&h);
            let l3 = level_rects_l3(&h);

            save_patch_map(
                &base_grid,
                &l1,
                &l2,
                &l3,
                &format!("{out_dir}/patch_map_step0000.png"),
                "Patch map (L1 yellow, L2 green, L3 blue)",
            )
            .unwrap();

            let m_amr_fine = flatten_to_target_grid(&h, fine_grid);
            save_mesh_zoom_multilevel(
                &m_amr_fine,
                &base_grid,
                refine_ratio,
                amr_max_level,
                &l1,
                &l2,
                &l3,
                &format!("{out_dir}/mesh_zoom_step0000.png"),
                "Zoom mesh: in-plane angle + multi-level grid",
                30,
            )
            .unwrap();
        }
    }

    // ---------- time loop ----------
    for step in 1..=steps {
        // uniform fine demag + relax step
        let t1 = Instant::now();
        b_fine.set_uniform(0.0, 0.0, 0.0);
        demag_fft_uniform::compute_demag_field_pbc(&fine_grid, &m_fine, &mut b_fine, &mat, pbcx, pbcy);
        t_demag_fine += t1.elapsed().as_secs_f64();
        step_llg_rk4_recompute_field_masked_relax_add(
            &mut m_fine,
            &llg,
            &mat,
            &mut scratch_fine,
            local_mask,
            Some(&b_fine),
        );

        // uniform coarse demag + relax step (baseline)
        let t2 = Instant::now();
        b_coarse.set_uniform(0.0, 0.0, 0.0);
        demag_fft_uniform::compute_demag_field_pbc(&base_grid, &m_coarse, &mut b_coarse, &mat, pbcx, pbcy);
        t_demag_coarse += t2.elapsed().as_secs_f64();
        step_llg_rk4_recompute_field_masked_relax_add(
            &mut m_coarse,
            &llg,
            &mat,
            &mut scratch_coarse,
            local_mask,
            Some(&b_coarse),
        );

        // AMR step (Bridge B demag inside)
        let t3 = Instant::now();
        stepper.step(&mut h, &llg, &mat, local_mask);
        t_amr_step += t3.elapsed().as_secs_f64();

        // Regrid periodically
        if step % regrid_every == 0 {
            // Always log a regrid attempt (even if no change), so behaviour is visible.
            let th = max_theta_coarse(&h.coarse, h.geom_mask());
            let l1c = current_patches.len();
            let l2c = h.patches_l2plus.get(0).map(|v| v.len()).unwrap_or(0);
            let l3c = h.patches_l2plus.get(1).map(|v| v.len()).unwrap_or(0);
            append_line(
                &format!("{out_dir}/regrid_attempts.csv"),
                &format!(
                    "{},{:.8e},{:.4},{},{},{}\n",
                    step,
                    th,
                    th.to_degrees(),
                    l1c,
                    l2c,
                    l3c
                ),
            );

            if let Some((new_rects, stats)) =
                maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
            {
                current_patches = new_rects.clone();

                let u = union_rect(&new_rects);
                let (ui0, uj0, unx, uny) = u
                    .map(|r| (r.i0, r.j0, r.nx, r.ny))
                    .unwrap_or((0, 0, 0, 0));
                append_line(
                    &regrid_log_path,
                    &format!(
                        "{},{:.8e},{:.8e},{},{},{},{},{},{}\n",
                        step,
                        stats.max_indicator,
                        stats.threshold,
                        stats.flagged_cells,
                        new_rects.len(),
                        ui0,
                        uj0,
                        unx,
                        uny
                    ),
                );

                // Level summary log for this accepted regrid
                let l1 = current_patches.clone();
                let l2 = level_rects_l2(&h);
                let l3 = level_rects_l3(&h);
                let u1 = union_rect_or_zero(&l1);
                let u2 = union_rect_or_zero(&l2);
                let u3 = union_rect_or_zero(&l3);
                append_line(
                    &format!("{out_dir}/regrid_levels.csv"),
                    &format!(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                        step,
                        l1.len(),
                        l2.len(),
                        l3.len(),
                        u1.i0,
                        u1.j0,
                        u1.nx,
                        u1.ny,
                        u2.i0,
                        u2.j0,
                        u2.nx,
                        u2.ny,
                        u3.i0,
                        u3.j0,
                        u3.nx,
                        u3.ny,  
                    ),
                );
            }
        }

        // Diagnostics output (RMSE always; plots only if requested)
        if step % out_every == 0 || step == steps {
            let fine_patches = patches_to_fine_rects(&current_patches, ref_ratio_total);
            let m_amr_fine = flatten_to_target_grid(&h, fine_grid);
            let (rmse, maxd) = rmse_and_max_delta(&m_amr_fine, &m_fine);

            append_line(
                &rmse_log_path,
                &format!("{},{:.8e},{:.8e},{}\n", step, rmse, maxd, fine_patches.len()),
            );

            if do_ovf {
                let fname = format!("m{:07}.ovf", step);
                write_ovf_text(&format!("{out_dir}/ovf_coarse/{fname}"), &m_coarse, "m_coarse");
                write_ovf_text(&format!("{out_dir}/ovf_fine/{fname}"), &m_fine, "m_fine");
                write_ovf_text(&format!("{out_dir}/ovf_amr/{fname}"), &m_amr_fine, "m_amr");
            }

            let l1 = h.patches.len();
            let l2 = h.patches_l2plus.get(0).map(|v| v.len()).unwrap_or(0);
            let l3 = h.patches_l2plus.get(1).map(|v| v.len()).unwrap_or(0);
            println!(
                "step {:4} | rmse {:.3e} | maxΔ {:.3e} | L1 {:2} | L2 {:2} | L3 {:2}",
                step, rmse, maxd, l1, l2, l3
            );

            if do_plots && (step == steps) {
                let l1 = current_patches.clone();
                let l2 = level_rects_l2(&h);
                let l3 = level_rects_l3(&h);

                save_patch_map(
                    &base_grid,
                    &l1,
                    &l2,
                    &l3,
                    &format!("{out_dir}/patch_map_step{step:04}.png"),
                    "Patch map (L1 yellow, L2 green, L3 blue)",
                )
                .unwrap();

                let m_amr_fine = flatten_to_target_grid(&h, fine_grid);
                save_mesh_zoom_multilevel(
                    &m_amr_fine,
                    &base_grid,
                    refine_ratio,
                    amr_max_level,
                    &l1,
                    &l2,
                    &l3,
                    &format!("{out_dir}/mesh_zoom_step{step:04}.png"),
                    "Zoom mesh: in-plane angle + multi-level grid",
                    30,
                )
                .unwrap();
            }
        }
    }

    let wall = t0.elapsed().as_secs_f64();

    // ---------- write final CSVs ----------
    let m_amr_fine_final = flatten_to_target_grid(&h, fine_grid);
    write_csv_ij_m(&format!("{out_dir}/uniform_coarse_final.csv"), &m_coarse);
    write_csv_ij_m(&format!("{out_dir}/uniform_fine_final.csv"), &m_fine);
    write_csv_ij_m(&format!("{out_dir}/amr_fine_final.csv"), &m_amr_fine_final);

    write_midline_y(&format!("{out_dir}/lineout_uniform_mid_y.csv"), &m_fine);
    write_midline_y(&format!("{out_dir}/lineout_amr_mid_y.csv"), &m_amr_fine_final);

    let fine_patches = patches_to_fine_rects(&current_patches, ref_ratio_total);
    let fine_cells_total = fine_grid.nx * fine_grid.ny;
    let fine_cells_in_patches: usize = fine_patches.iter().map(|r| r.nx * r.ny).sum();
    let coverage = fine_cells_in_patches as f64 / fine_cells_total as f64;

    let (rmse_final, maxd_final) = rmse_and_max_delta(&m_amr_fine_final, &m_fine);

    println!();
    println!("AMR vortex relaxation benchmark (Bridge B demag FFT)");
    println!("Base grid: {base_nx} x {base_ny}   dx={dx:.3e}  dy={dy:.3e}  dz={dz:.3e}");
    println!(
        "Fine grid: {fine_nx} x {fine_ny}   dx={:.3e}  dy={:.3e}",
        dx / ref_ratio_total as f64,
        dy / ref_ratio_total as f64
    );
    println!("Steps: {steps}   dt={dt:.3e}");
    println!("Outputs: {out_dir}");
    println!();
    println!("Final RMSE(|Δm|): {:.6e}", rmse_final);
    println!("Final max |Δm| : {:.6e}", maxd_final);
    println!("Fine cells (uniform): {fine_cells_total}");
    println!("Fine cells covered by patches: {fine_cells_in_patches}");
    println!("Patch coverage fraction: {:.6}", coverage);
    println!();
    println!("Timing (includes demag FFT everywhere):");
    println!("  total wall time:       {:.3} s", wall);
    println!("  fine demag time:       {:.3} s", t_demag_fine);
    println!("  coarse demag time:     {:.3} s", t_demag_coarse);
    println!("  AMR step time:         {:.3} s", t_amr_step);
    let other = (wall - t_demag_fine - t_demag_coarse - t_amr_step).max(0.0);
    println!("  other/unaccounted:     {:.3} s", other);
}
