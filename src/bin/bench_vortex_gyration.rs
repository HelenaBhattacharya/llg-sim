// src/bin/bench_vortex_gyration.rs
//
// Gold Standard Benchmark: Permalloy Disk Vortex Gyration
// =======================================================
//
// Reproduces the vortex core gyration from:
//   Guslienko et al., J. Appl. Phys. 91, 8037 (2002), Fig. 3
//   Novosad et al., Phys. Rev. B 72, 024455 (2005) — experimental + OOMMF
//
// Setup:
//   - 200 nm diameter Permalloy disk, dz = 20 nm → β = L/R = 0.2
//   - Ms = 8.0×10⁵ A/m, A = 1.3×10⁻¹¹ J/m, K_u = 0
//   - Cell size: dx = dy = 3.125 nm (96² base on 300 nm box)
//   - α = 0.01 for dynamics (physical Permalloy damping)
//   - dt = 10 fs with 2 AMR levels; 3 fs with 3 AMR levels (auto-selected)
//   - B_shift = 10 mT → s_eq ≈ 0.12 (safely in the linear regime)
//     At β = 0.2 the restoring force is strong: old cfft run showed
//     r_eq ≈ 11.5nm at 10mT (s = 0.115), well within linear regime.
//
// CFL note: With 3 AMR levels (ratio=2), the finest dx = 3.125/8 = 0.39 nm.
//   Exchange CFL requires dt < ~3.5 fs at this spacing.  dt=3 fs gives
//   good margin.  With 2 levels, dx_fine=0.78nm, dt=10fs.
//   The code auto-selects dt based on AMR level count.
//
// Protocol (Guslienko's method):
//   Phase 1: Relax vortex to ground state (α = 0.5, B = 0, 1.5ns)
//   Phase 2: Apply 10 mT in-plane field, relax to shifted equilibrium (α = 0.5, 3ns)
//   Phase 3: Remove field, evolve with α = 0.01 — core oscillates ~6 periods in 8ns
//
// Expected result: nearly-circular orbits with slow decay,
//   eigenfrequency ~740 MHz for β = 0.2 (Guslienko 2002 Fig 3 curve b;
//   Novosad 2005 empirical scaling: f ≈ 3700×β MHz = 740 MHz).
//   Old cfft run (80² base) gave 816 MHz — 10% overestimate from coarser grid.
//   At 10 mT the equilibrium displacement is ~12nm (s ≈ 0.12), safely linear.
//
// AMR: 3 levels (L1 boundary arcs, L2 intermediate, L3 vortex core tracking).
//   dx_fine = 0.39 nm — 14.6 cells per exchange length.
//   dt = 3 fs (auto-selected for CFL safety at L3).
//
// AMR features:
//   - boundary_layer=4: ensures disk edge gets refinement patches
//     (García-Cervera criterion — resolves surface charges σ=M·n̂ at fine dx)
//     Matches bench_composite_vcycle settings for tight boundary conformance.
//   - Berger-Rigoutsos bisection (min_efficiency=0.65): splits the boundary
//     ring into small arc-segment patches that conform to the disk shape
//   - Nested L2/L3 patches track the moving vortex core
//
// Post-processing:
//   - Core trajectory X/R vs Y/R → Guslienko Fig 2 comparison
//   - Gyration frequency extraction via zero-crossing analysis
//   - Frequency comparison: fine vs coarse vs AMR vs Guslienko analytic
//
// Four-method comparison:
//   1. Uniform fine FFT (256², reference)
//   2. Uniform coarse FFT (64², baseline — core poorly resolved)
//   3. AMR + coarse-FFT demag
//   4. AMR + composite MG demag (optional, --skip-composite to disable)
//
// Run:
//   cargo run --release --bin bench_vortex_gyration -- --amr-only --skip-composite --plots
//     (AMR only, ~60-90 min: 2ns relax + 5ns gyration at dt=5fs)
//   cargo run --release --bin bench_vortex_gyration -- --plots
//     (Full 4-method comparison, ~3-4 hours)
//   cargo run --release --bin bench_vortex_gyration -- --skip-composite --plots

use std::f64::consts::PI;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use plotters::prelude::*;

use llg_sim::effective_field::{FieldMask, demag_fft_uniform};
use llg_sim::geometry_mask::{MaskShape, edge_smooth_n, apply_fill_fractions};
use llg_sim::grid::Grid2D;
use llg_sim::initial_states;
use llg_sim::llg::{
    RK4Scratch,
    step_llg_rk4_recompute_field_masked_geom_add,         // DYNAMICS with geometry mask
    step_llg_rk4_recompute_field_masked_relax_geom_add,    // RELAXATION with geometry mask
};
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

use llg_sim::amr::indicator::IndicatorKind;
use llg_sim::amr::regrid::maybe_regrid_nested_levels;
use llg_sim::amr::{
    AmrHierarchy2D, AmrStepperRK4, ClusterPolicy, Connectivity, Rect2i, RegridPolicy,
};

// =========================================================================
// Utilities
// =========================================================================

fn ensure_dir(p: &str) { if !Path::new(p).exists() { fs::create_dir_all(p).unwrap(); } }
fn idx(i: usize, j: usize, nx: usize) -> usize { j * nx + i }

fn pow_usize(mut b: usize, mut e: usize) -> usize {
    let mut o = 1usize;
    while e > 0 { if e & 1 == 1 { o = o.saturating_mul(b); } e >>= 1; if e > 0 { b = b.saturating_mul(b); } }
    o
}

// =========================================================================
// Metrics + helpers
// =========================================================================

fn rmse_fields(a: &VectorField2D, b: &VectorField2D) -> (f64, f64) {
    assert_eq!(a.grid.nx, b.grid.nx); assert_eq!(a.grid.ny, b.grid.ny);
    let n = (a.grid.nx * a.grid.ny) as f64;
    let (mut s2, mut mx) = (0.0_f64, 0.0_f64);
    for k in 0..a.data.len() {
        let (da, db) = (a.data[k], b.data[k]);
        let d2 = (da[0]-db[0]).powi(2)+(da[1]-db[1]).powi(2)+(da[2]-db[2]).powi(2);
        s2 += d2; mx = mx.max(d2.sqrt());
    }
    ((s2/n).sqrt(), mx)
}

fn flatten_to(h: &AmrHierarchy2D, tgt: Grid2D) -> VectorField2D {
    let m = h.flatten_to_uniform_fine();
    if m.grid.nx==tgt.nx && m.grid.ny==tgt.ny && m.grid.dx==tgt.dx && m.grid.dy==tgt.dy && m.grid.dz==tgt.dz { m }
    else { m.resample_to_grid(tgt) }
}

fn patches_fine(p: &[Rect2i], r: usize) -> Vec<Rect2i> {
    p.iter().map(|q| Rect2i { i0:q.i0*r, j0:q.j0*r, nx:q.nx*r, ny:q.ny*r }).collect()
}

fn l2_rects(h: &AmrHierarchy2D) -> Vec<Rect2i> { h.patches_l2plus.get(0).map(|v| v.iter().map(|p| p.coarse_rect).collect()).unwrap_or_default() }
fn l3_rects(h: &AmrHierarchy2D) -> Vec<Rect2i> { h.patches_l2plus.get(1).map(|v| v.iter().map(|p| p.coarse_rect).collect()).unwrap_or_default() }

// =========================================================================
// Core level detection (which AMR level covers the core?)
// =========================================================================

/// Determine the deepest AMR level whose patch contains the core position.
/// core_x, core_y are in metres (physical coords), bg is the base grid.
/// Returns 0 (coarse only), 1 (L1 patch), 2 (L2), or 3 (L3).
fn core_amr_level(h: &AmrHierarchy2D, core_x: f64, core_y: f64, bg: &Grid2D) -> usize {
    // Convert physical coords to base-grid cell index
    let hlx = bg.nx as f64 * bg.dx * 0.5;
    let hly = bg.ny as f64 * bg.dy * 0.5;
    let ci = ((core_x + hlx) / bg.dx).floor() as isize;
    let cj = ((core_y + hly) / bg.dy).floor() as isize;
    if ci < 0 || cj < 0 || ci >= bg.nx as isize || cj >= bg.ny as isize {
        return 0;
    }
    let (ci, cj) = (ci as usize, cj as usize);

    let mut level = 0usize;

    // Check L1 patches
    for p in &h.patches {
        let r = p.coarse_rect;
        if ci >= r.i0 && ci < r.i0 + r.nx && cj >= r.j0 && cj < r.j0 + r.ny {
            level = 1;
            break;
        }
    }

    // Check L2+ patches
    for (k, patches) in h.patches_l2plus.iter().enumerate() {
        let lv = k + 2;
        for p in patches {
            let r = p.coarse_rect;
            if ci >= r.i0 && ci < r.i0 + r.nx && cj >= r.j0 && cj < r.j0 + r.ny {
                level = lv;
                break;
            }
        }
    }

    level
}

// =========================================================================
// Core tracking (weighted centroid)
// =========================================================================

fn find_core(m: &VectorField2D, mask: Option<&[bool]>) -> (f64, f64, f64) {
    let (nx,ny,dx,dy) = (m.grid.nx, m.grid.ny, m.grid.dx, m.grid.dy);
    let (hlx,hly) = (nx as f64*dx*0.5, ny as f64*dy*0.5);
    let mut pk = 0.0_f64; let mut pk_mz = 0.0;
    for j in 0..ny { for i in 0..nx {
        let id = j*nx+i;
        if let Some(msk) = mask { if !msk[id] { continue; } }
        let a = m.data[id][2].abs();
        if a > pk { pk = a; pk_mz = m.data[id][2]; }
    }}
    if pk < 1e-10 { return (0.0, 0.0, 0.0); }
    let th = 0.5 * pk;
    let (mut wx, mut wy, mut ws) = (0.0_f64, 0.0_f64, 0.0_f64);
    for j in 0..ny { for i in 0..nx {
        let id = j*nx+i;
        if let Some(msk) = mask { if !msk[id] { continue; } }
        let a = m.data[id][2].abs();
        if a > th {
            let x = (i as f64+0.5)*dx - hlx;
            let y = (j as f64+0.5)*dy - hly;
            wx += a*x; wy += a*y; ws += a;
        }
    }}
    if ws > 0.0 { (wx/ws, wy/ws, pk_mz) } else { (0.0, 0.0, pk_mz) }
}

// =========================================================================
// Vacuum contamination check (Fix 0 diagnostic)
// =========================================================================

/// Check that L0 coarse grid has no non-zero M in vacuum cells.
/// Returns the count of contaminated cells.
fn check_vacuum_contamination(label: &str, h: &AmrHierarchy2D) -> usize {
    let (n_cont, n_vac) = h.count_coarse_vacuum_contamination();
    let n_total = h.base_grid.n_cells();
    let n_mat = n_total - n_vac;
    if n_cont > 0 {
        eprintln!(
            "  [WARN] {}: {} contaminated vacuum cells on L0 ({} material / {} vacuum / {} total)",
            label, n_cont, n_mat, n_vac, n_total
        );
    } else {
        println!(
            "  [geom OK] {}: L0 clean ({} material / {} vacuum / {} total)",
            label, n_mat, n_vac, n_total
        );
    }
    n_cont
}

// =========================================================================
// Frequency extraction from core trajectory
// =========================================================================

/// Estimate gyration frequency from a time series of core x-position.
/// Uses zero-crossing detection (robust for decaying spirals).
/// Returns (frequency_GHz, n_crossings_used).
fn estimate_gyration_freq(times_ns: &[f64], x_nm: &[f64]) -> (f64, usize) {
    if times_ns.len() < 4 { return (0.0, 0); }
    // Detect zero crossings of x(t)
    let mut crossings = Vec::new();
    for i in 1..x_nm.len() {
        if x_nm[i-1] * x_nm[i] < 0.0 && x_nm[i-1] != 0.0 {
            // Linear interpolation for crossing time
            let frac = x_nm[i-1].abs() / (x_nm[i-1].abs() + x_nm[i].abs());
            let tc = times_ns[i-1] + frac * (times_ns[i] - times_ns[i-1]);
            crossings.push(tc);
        }
    }
    if crossings.len() < 2 { return (0.0, crossings.len()); }
    // Period = 2 × (time between consecutive crossings)
    let mut periods = Vec::new();
    for i in 1..crossings.len() {
        let half_period = crossings[i] - crossings[i-1];
        if half_period > 0.0 { periods.push(2.0 * half_period); }
    }
    if periods.is_empty() { return (0.0, crossings.len()); }
    let avg_period_ns = periods.iter().sum::<f64>() / periods.len() as f64;
    (1.0 / avg_period_ns, crossings.len())  // GHz (since period in ns)
}

/// Read a core trajectory CSV and return (times_ns, x_nm, y_nm).
fn read_core_csv(path: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut ts = Vec::new(); let mut xs = Vec::new(); let mut ys = Vec::new();
    if let Ok(content) = std::fs::read_to_string(path) {
        for line in content.lines().skip(1) { // skip header
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() >= 4 {
                if let (Ok(t), Ok(x), Ok(y)) = (cols[1].parse::<f64>(), cols[2].parse::<f64>(), cols[3].parse::<f64>()) {
                    ts.push(t); xs.push(x); ys.push(y);
                }
            }
        }
    }
    (ts, xs, ys)
}

// =========================================================================
// Colour maps + snapshot plot
// =========================================================================

fn hsv_rgb(h: f64, s: f64, v: f64) -> RGBColor {
    let h = h.rem_euclid(1.0); let i = (h*6.0).floor() as i32; let f = h*6.0-i as f64;
    let (p,q,t) = (v*(1.0-s), v*(1.0-f*s), v*(1.0-(1.0-f)*s));
    let (r,g,b) = match i.rem_euclid(6) { 0=>(v,t,p),1=>(q,v,p),2=>(p,v,t),3=>(p,q,v),4=>(t,p,v),_=>(v,p,q) };
    RGBColor((r.clamp(0.0,1.0)*255.0) as u8,(g.clamp(0.0,1.0)*255.0) as u8,(b.clamp(0.0,1.0)*255.0) as u8)
}

fn mz_bwr(mz: f64) -> RGBColor {
    let t = ((mz+1.0)*0.5).clamp(0.0,1.0);
    if t<0.5 { let a=t/0.5; RGBColor((255.0*a) as u8,(255.0*a) as u8,255) }
    else { let a=(t-0.5)/0.5; RGBColor(255,(255.0*(1.0-a)) as u8,(255.0*(1.0-a)) as u8) }
}

fn save_snap(
    m: &VectorField2D, bg: &Grid2D, l1: &[Rect2i], l2: &[Rect2i], l3: &[Rect2i],
    shape: &MaskShape, path: &str, cap: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (nx,ny) = (m.grid.nx as i32, m.grid.ny as i32);
    let (nxf,nyf) = (m.grid.nx as f64, m.grid.ny as f64);
    let nx0 = bg.nx as f64;
    let root = BitMapBackend::new(path, (800,800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut ch = ChartBuilder::on(&root).caption(cap,("sans-serif",20)).margin(10)
        .set_all_label_area_size(0).build_cartesian_2d(0..nx,0..ny)?;
    ch.configure_mesh().disable_mesh().draw()?;
    ch.draw_series((0..m.grid.ny).flat_map(|j| (0..m.grid.nx).map(move |i| {
        let v = m.data[idx(i,j,m.grid.nx)];
        let mg = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
        let c = if mg<0.01 { RGBColor(220,220,220) }
            else if v[2].abs()>0.8 { mz_bwr(v[2]) }
            else { hsv_rgb((v[1].atan2(v[0])+PI)/(2.0*PI),1.0,1.0) };
        Rectangle::new([(i as i32,j as i32),(i as i32+1,j as i32+1)], c.filled())
    })))?;
    // Disk outline
    if let MaskShape::Disk { radius, .. } = shape {
        let rc = radius/m.grid.dx; let (cx,cy) = (nxf/2.0,nyf/2.0);
        let pts: Vec<(i32,i32)> = (0..=200).map(|k| {
            let th=2.0*PI*k as f64/200.0; ((cx+rc*th.cos()) as i32,(cy+rc*th.sin()) as i32)
        }).collect();
        ch.draw_series(std::iter::once(PathElement::new(pts, BLACK.stroke_width(2))))?;
    }
    // Patches: L1 yellow, L2 green, L3 blue
    let rr = (nxf/nx0) as usize;
    for (rects,col) in [(l1,RGBColor(240,200,0)),(l2,RGBColor(0,200,0)),(l3,RGBColor(0,120,255))] {
        for r in rects { let (x0,y0)=((r.i0*rr) as i32,(r.j0*rr) as i32);
            let (x1,y1)=(((r.i0+r.nx)*rr) as i32,((r.j0+r.ny)*rr) as i32);
            ch.draw_series(std::iter::once(PathElement::new(
                vec![(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)], col.stroke_width(2))))?;
    }}
    root.present()?; Ok(())
}

/// Save an mz-only snapshot using a coolwarm (blue-white-red) colourmap.
/// vmin/vmax control the colour scale; cells outside disk → white.
fn save_mz_snap(
    m: &VectorField2D, shape: &MaskShape, path: &str, cap: &str,
    vmin: f64, vmax: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let (nx, ny) = (m.grid.nx as i32, m.grid.ny as i32);
    let (nxf, nyf) = (m.grid.nx as f64, m.grid.ny as f64);
    let root = BitMapBackend::new(path, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut ch = ChartBuilder::on(&root)
        .caption(cap, ("sans-serif", 20))
        .margin(10)
        .right_y_label_area_size(60)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0..nx, 0..ny)?;
    ch.configure_mesh().disable_mesh().draw()?;

    // Precompute disk mask
    let (disk_cx, disk_cy, disk_r_pix) = if let MaskShape::Disk { radius, .. } = shape {
        (nxf / 2.0, nyf / 2.0, radius / m.grid.dx)
    } else {
        (nxf / 2.0, nyf / 2.0, nxf / 2.0)
    };

    ch.draw_series((0..m.grid.ny).flat_map(|j| {
        (0..m.grid.nx).map(move |i| {
            let v = m.data[j * m.grid.nx + i];
            let px = i as f64 + 0.5;
            let py = j as f64 + 0.5;
            let dist = ((px - disk_cx).powi(2) + (py - disk_cy).powi(2)).sqrt();
            let c = if dist > disk_r_pix {
                WHITE  // outside disk
            } else {
                // Coolwarm: normalise mz to [0,1] within [vmin, vmax]
                let t = ((v[2] - vmin) / (vmax - vmin)).clamp(0.0, 1.0);
                if t < 0.5 {
                    let a = t / 0.5;
                    RGBColor((59.0 + 196.0 * a) as u8, (76.0 + 179.0 * a) as u8, (192.0 + 63.0 * a) as u8)
                    // blue (59,76,192) → white (255,255,255)
                } else {
                    let a = (t - 0.5) / 0.5;
                    RGBColor(255, (255.0 - 175.0 * a) as u8, (255.0 - 191.0 * a) as u8)
                    // white (255,255,255) → red (255,80,64)
                }
            };
            Rectangle::new([(i as i32, j as i32), (i as i32 + 1, j as i32 + 1)], c.filled())
        })
    }))?;

    // Disk outline
    if let MaskShape::Disk { radius, .. } = shape {
        let rc = radius / m.grid.dx;
        let (cx, cy) = (nxf / 2.0, nyf / 2.0);
        let pts: Vec<(i32, i32)> = (0..=200)
            .map(|k| {
                let th = 2.0 * PI * k as f64 / 200.0;
                ((cx + rc * th.cos()) as i32, (cy + rc * th.sin()) as i32)
            })
            .collect();
        ch.draw_series(std::iter::once(PathElement::new(pts, BLACK.stroke_width(2))))?;
    }

    root.present()?;
    Ok(())
}

// =========================================================================
// CSV + log
// =========================================================================

fn append(p: &str, s: &str) { OpenOptions::new().create(true).append(true).open(p).unwrap().write_all(s.as_bytes()).unwrap(); }

// =========================================================================
// Uniform step helpers
// =========================================================================

fn step_relax(m: &mut VectorField2D, b: &mut VectorField2D,
    llg: &LLGParams, mat: &Material, s: &mut RK4Scratch, g: &Grid2D, geom: &[bool]) {
    b.set_uniform(0.0,0.0,0.0);
    demag_fft_uniform::compute_demag_field_pbc(g,m,b,mat,0,0);
    step_llg_rk4_recompute_field_masked_relax_geom_add(m,llg,mat,s,FieldMask::ExchAnis,Some(geom),Some(b));
}

/// Phase 3: FULL LLG dynamics WITH geometry mask.
/// Without the mask, vacuum cells outside the disk accumulate noise,
/// precession amplifies it, normalization creates fake material → blowup.
fn step_dyn(m: &mut VectorField2D, b: &mut VectorField2D,
    llg: &LLGParams, mat: &Material, s: &mut RK4Scratch, g: &Grid2D, geom: &[bool]) {
    b.set_uniform(0.0,0.0,0.0);
    demag_fft_uniform::compute_demag_field_pbc(g,m,b,mat,0,0);
    step_llg_rk4_recompute_field_masked_geom_add(m,llg,mat,s,FieldMask::ExchAnis,Some(geom),Some(b));
}

// =========================================================================
// Post-processing plots
// =========================================================================

/// Plot core trajectory X/R vs Y/R (Guslienko Fig 2 style) for all methods.
fn plot_trajectories(
    out: &str, disk_r: f64,
    cp_f: &str, cp_c: &str, cp_cf: &str, cp_co: &str,
    skip_fine: bool, skip_coarse: bool, skip_cfft: bool, skip_comp: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{out}/trajectory.png");
    let root = BitMapBackend::new(&path, (700, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let r_nm = disk_r * 1e9;
    let lim = 0.6; // plot ±0.6 R
    let mut ch = ChartBuilder::on(&root)
        .caption("Vortex Core Trajectory", ("sans-serif", 22))
        .margin(15).x_label_area_size(40).y_label_area_size(50)
        .build_cartesian_2d(-lim..lim, -lim..lim)?;
    ch.configure_mesh()
        .x_desc("X / R").y_desc("Y / R")
        .label_style(("sans-serif", 14)).draw()?;

    // Disk outline
    let circ: Vec<(f64,f64)> = (0..=200).map(|k| {
        let th = 2.0*PI*k as f64/200.0; (th.cos(), th.sin())
    }).collect();
    ch.draw_series(std::iter::once(PathElement::new(circ, BLACK.stroke_width(1))))?;

    let mut plot_csv = |path: &str, color: RGBColor, label: &str| -> Result<(), Box<dyn std::error::Error>> {
        let (_, xs, ys) = read_core_csv(path);
        if xs.is_empty() { return Ok(()); }
        let pts: Vec<(f64,f64)> = xs.iter().zip(ys.iter())
            .map(|(&x, &y)| (x/r_nm, y/r_nm)).collect();
        ch.draw_series(LineSeries::new(pts, color.stroke_width(2)))?
            .label(label).legend(move |(x,y)| PathElement::new([(x,y),(x+20,y)], color.stroke_width(2)));
        Ok(())
    };

    if !skip_fine   { plot_csv(cp_f, RGBColor(0,0,200), "Fine FFT")?; }
    if !skip_coarse { plot_csv(cp_c, RGBColor(200,0,0), "Coarse FFT")?; }
    if !skip_cfft   { plot_csv(cp_cf, RGBColor(0,160,0), "AMR+cfft")?; }
    if !skip_comp   { plot_csv(cp_co, RGBColor(200,100,0), "AMR+composite")?; }

    ch.configure_series_labels().border_style(BLACK).position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", 13)).draw()?;
    root.present()?;
    println!("  Saved {path}");
    Ok(())
}

/// Plot core x(t) for frequency visualisation.
fn plot_core_vs_time(
    out: &str,
    cp_f: &str, cp_c: &str, cp_cf: &str, cp_co: &str,
    skip_fine: bool, skip_coarse: bool, skip_cfft: bool, skip_comp: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = format!("{out}/core_x_vs_t.png");
    let root = BitMapBackend::new(&path, (900, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find time/x range from all available data
    let mut t_max = 0.0_f64; let mut x_max = 0.0_f64;
    for p in [cp_f, cp_c, cp_cf, cp_co] {
        let (ts, xs, _) = read_core_csv(p);
        if let Some(&tm) = ts.last() { t_max = t_max.max(tm); }
        for &x in &xs { x_max = x_max.max(x.abs()); }
    }
    if t_max <= 0.0 || x_max <= 0.0 { return Ok(()); }
    x_max *= 1.1;

    let mut ch = ChartBuilder::on(&root)
        .caption("Core x(t) — Gyration", ("sans-serif", 18))
        .margin(10).x_label_area_size(35).y_label_area_size(50)
        .build_cartesian_2d(0.0..t_max, -x_max..x_max)?;
    ch.configure_mesh()
        .x_desc("t (ns)").y_desc("x_core (nm)")
        .label_style(("sans-serif", 12)).draw()?;

    let mut plot_csv = |path: &str, color: RGBColor, label: &str| -> Result<(), Box<dyn std::error::Error>> {
        let (ts, xs, _) = read_core_csv(path);
        if ts.is_empty() { return Ok(()); }
        let pts: Vec<(f64,f64)> = ts.into_iter().zip(xs.into_iter()).collect();
        ch.draw_series(LineSeries::new(pts, color.stroke_width(2)))?
            .label(label).legend(move |(x,y)| PathElement::new([(x,y),(x+20,y)], color.stroke_width(2)));
        Ok(())
    };

    if !skip_fine   { plot_csv(cp_f, RGBColor(0,0,200), "Fine FFT")?; }
    if !skip_coarse { plot_csv(cp_c, RGBColor(200,0,0), "Coarse FFT")?; }
    if !skip_cfft   { plot_csv(cp_cf, RGBColor(0,160,0), "AMR+cfft")?; }
    if !skip_comp   { plot_csv(cp_co, RGBColor(200,100,0), "AMR+composite")?; }

    ch.configure_series_labels().border_style(BLACK).position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", 12)).draw()?;
    root.present()?;
    println!("  Saved {path}");
    Ok(())
}

// =========================================================================
// MAIN
// =========================================================================

#[allow(unused_assignments)]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let do_plots = args.iter().any(|a| a=="--plots");
    let amr_only = args.iter().any(|a| a=="--amr-only");
    let fine_only = args.iter().any(|a| a=="--fine-only" || a=="--fine");
    let coarse_only = args.iter().any(|a| a=="--coarse-only" || a=="--coarse");
    let uniform_only = fine_only || coarse_only; // skip all AMR paths
    let skip_fine = (amr_only || coarse_only) && !fine_only
        || args.iter().any(|a| a=="--skip-fine-ref");
    let skip_coarse = (amr_only || fine_only) && !coarse_only;
    let skip_comp = uniform_only || args.iter().any(|a| a=="--skip-composite");
    let skip_cfft = uniform_only; // skip AMR+coarse-FFT path

    let amr_lvl: usize = std::env::var("LLG_AMR_MAX_LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(3);

    let out = "out/bench_vortex_gyration";
    ensure_dir(out);

    // =====================================================================
    // GUSLIENKO PARAMETERS (J. Appl. Phys. 91, 8037, 2002)
    // =====================================================================

    // GUSLIENKO / NOVOSAD VALIDATION (β = 0.2)
    // 200 nm diameter Py disk, L = 20 nm, β = L/R = 0.2
    // Guslienko Fig 3 curve b: f ≈ 700-740 MHz
    // Novosad 2005 empirical: f = 3700×0.2 = 740 MHz
    // Novosad OOMMF (5nm cells): agrees with experiment to ~10%
    // Old cfft (80×80, dx=3.75nm, 53 cells/diam): 816 MHz (10% high)
    // This run: 96×96, dx=3.125nm, 64 cells/diam — should reduce the
    // staircase-boundary frequency overestimate.
    let disk_r = 100.0e-9;   // R = 100 nm (2R = 200 nm)
    let dz = 20.0e-9;        // L = 20 nm → β = 0.2
    let domain = 300.0e-9;   // 50% padding — 50nm vacuum gap = 16 cells at dx=3.125nm

    // Grid: 96² base → dx = 3.125 nm (< l_ex ≈ 5.7 nm ✓)
    // Disk spans 64 cells in diameter (21% more than old 80² setup's 53)
    // Vacuum gap = 50nm = 16 cells per side — safe for boundary_layer=4
    let bnx = 96_usize; let bny = 96;
    let dx = domain / bnx as f64;
    let dy = dx;
    let ratio = 2_usize; let ghost = 2;
    let rrt = pow_usize(ratio, amr_lvl);
    let (fnx, fny) = (bnx*rrt, bny*rrt);

    let disk = MaskShape::Disk { center: (0.0, 0.0), radius: disk_r };

    // Material: Permalloy
    let mat = Material {
        ms: 8.0e5, a_ex: 1.3e-11, k_u: 0.0, easy_axis: [0.0,0.0,1.0],
        dmi: None, demag: true, demag_method: DemagMethod::FftUniform,
    };
    let mu0 = 4.0*PI*1e-7;
    let lex = (2.0*mat.a_ex/(mu0*mat.ms*mat.ms)).sqrt();

    // Timing
    // With 2 AMR levels (default), dx_fine = dx_coarse/4 = 0.78 nm.
    // CFL limit at L2: dt_max ≈ 14 fs.  Using dt=10 fs for margin.
    // With 3 AMR levels (LLG_AMR_MAX_LEVEL=3), dx_fine = 0.39 nm → use dt=3fs.
    let dt: f64 = if amr_lvl >= 3 { 3.0e-15 } else { 10.0e-15 };
    let alpha_relax = 0.5;
    let alpha_dyn = 0.01;    // Physical Permalloy damping — gives ~6 clean oscillation
                              // periods in 8ns at 740 MHz.
    // Phase timing:
    //   τ_relax = 1/(α×ω₀) ≈ 1/(0.5 × 2π × 740MHz) ≈ 0.43ns
    //   Phase 1: 1.5ns = 3.5τ → adequate relaxation
    //   Phase 2: 3.0ns = 7.0τ → well equilibrated under field
    //   Phase 3: 8.0ns → ~5.9 periods at 740 MHz
    let relax_steps = (1.5e-9 / dt).ceil() as usize;
    let field_relax_steps = (3.0e-9 / dt).ceil() as usize;
    let b_shift = 10.0e-3;   // 10 mT in-plane field → s_eq ≈ 0.12 (from old cfft data)
                              // Safely in the linear regime (s < 0.15).
                              // Old cfft run showed core at (0.2, 11.5)nm under 10mT.
    let gyr_time = 8.0e-9;   // 8 ns — ~5.9 periods at 740 MHz
    let gyr_steps = (gyr_time/dt).ceil() as usize;

    // Output cadences
    // relax_out: print every 200ps during relaxation
    // dyn_out: record core position every 5ps
    // snap_every: snapshot every 200ps
    // dyn_regrid: regrid every 40ps
    let _steps_per_ns = (1.0e-9 / dt).round() as usize;
    let relax_out  = (200e-12 / dt).round() as usize;  // every 200ps
    let dyn_out    = (5e-12 / dt).round() as usize;     // every 5ps
    let snap_every = (200e-12 / dt).round() as usize;   // every 200ps
    let dyn_regrid = (40e-12 / dt).round() as usize;    // every 40ps

    // =====================================================================
    // Grids and masks
    // =====================================================================

    let bg = Grid2D::new(bnx, bny, dx, dy, dz);
    let fg = Grid2D::new(fnx, fny, dx/rrt as f64, dy/rrt as f64, dz);
    let ns = edge_smooth_n();
    let (mc, ffc) = disk.to_mask_and_fill(&bg, ns);
    let (mf, _) = disk.to_mask_and_fill(&fg, ns);

    let mut llg = LLGParams { gamma: GAMMA_E_RAD_PER_S_T, alpha: alpha_relax, dt, b_ext: [0.0;3] };
    let lm = FieldMask::ExchAnis;

    // AMR policies — matches bench_composite_vcycle boundary settings:
    //   boundary_layer=4, min_efficiency=0.65, min_patch_area=16, merge_distance=1
    // These produce tight, conforming arc-segment patches at the disk boundary
    // (same quality as the antidot benchmark).
    //
    // In regrid.rs, L2+ levels use boundary_layer=0 (indicator-only), so L2/L3
    // patches focus on sharp features (vortex cores) rather than re-refining the
    // already-resolved boundary.  The composite indicator's curl component peaks
    // at the vortex core, driving L2/L3 patch placement there.
    //
    // DYNAMIC TRACKING: The new-region acceptance in regrid.rs detects when
    // the indicator produces a genuinely new patch (>50% non-overlapping with
    // all existing patches) inside the boundary union.  This lets the core
    // L1 patch appear/move without destabilising the boundary arcs.
    let ind = IndicatorKind::from_env();
    let bl: usize = std::env::var("LLG_AMR_BOUNDARY_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let rp = RegridPolicy {
        indicator: ind, buffer_cells: 4, boundary_layer: bl,
        min_change_cells: 1, min_area_change_frac: 0.01,
    };
    let cp = ClusterPolicy {
        indicator: ind, buffer_cells: 4, boundary_layer: bl,
        min_patch_area: 16, merge_distance: 1, max_patches: 0,
        connectivity: Connectivity::Eight, min_efficiency: 0.65, max_flagged_fraction: 0.50,
        confine_dilation: true,  // Safe with 300nm domain: 50nm vacuum gap = 16 cells >> buffer=4
    };

    // =====================================================================
    // Initial states
    // =====================================================================

    let mk_coarse = || -> VectorField2D {
        let mut m = VectorField2D::new(bg);
        initial_states::init_vortex(&mut m, &bg, (0.0,0.0), 1.0, 1.0, 3.0*dx, Some(&mc));
        apply_fill_fractions(&mut m.data, &ffc);
        m
    };

    let mut m_fine = VectorField2D::new(fg);
    initial_states::init_vortex(&mut m_fine, &fg, (0.0,0.0), 1.0, 1.0, 3.0*dx, Some(&mf));
    let mut m_coarse = mk_coarse();

    // AMR hierarchies (skip if only running uniform fine/coarse)
    let mut h_cf = AmrHierarchy2D::new(bg, mk_coarse(), ratio, ghost);
    h_cf.set_geom_shape(disk.clone());
    let mut h_co = if !skip_comp {
        let mut h = AmrHierarchy2D::new(bg, mk_coarse(), ratio, ghost);
        h.set_geom_shape(disk.clone()); Some(h)
    } else { None };

    // --- Checkpoint 1: verify geometry mask zeroed vacuum on L0 after init ---
    {
        let nc = check_vacuum_contamination("cfft after set_geom_shape", &h_cf);
        assert!(nc == 0, "FATAL: cfft L0 has {} non-zero vacuum cells after set_geom_shape", nc);
    }
    if let Some(ref hc) = h_co {
        let nc = check_vacuum_contamination("comp after set_geom_shape", hc);
        assert!(nc == 0, "FATAL: comp L0 has {} non-zero vacuum cells after set_geom_shape", nc);
    }

    // Steppers
    let mut st_cf = if !skip_cfft {
        unsafe { std::env::set_var("LLG_AMR_DEMAG_MODE", "coarse_fft"); }
        Some(AmrStepperRK4::new(&h_cf, true))
    } else { None };
    let mut st_co = if let Some(hc) = h_co.as_ref() {
        unsafe { std::env::set_var("LLG_AMR_DEMAG_MODE", "composite"); }
        Some(AmrStepperRK4::new(hc, true))
    } else { None };
    unsafe { std::env::remove_var("LLG_AMR_DEMAG_MODE"); }

    // Initial regrid
    let mut pr_cf: Vec<Rect2i> = Vec::new();
    let mut pr_co: Vec<Rect2i> = Vec::new();
    if !skip_cfft { if let Some((r,_)) = maybe_regrid_nested_levels(&mut h_cf, &pr_cf, rp, cp) { pr_cf = r; } }
    if let Some(hc) = h_co.as_mut() {
        if let Some((r,_)) = maybe_regrid_nested_levels(hc, &pr_co, rp, cp) { pr_co = r; }
    }
    // NOTE: do NOT overwrite m_fine with flatten_to(&h_cf, fg) here.
    // The fine grid was initialized with init_vortex at fine resolution (dx_fine)
    // and should relax independently from that — this is the whole point of having
    // a fine-grid reference.  Overwriting it with the AMR's coarse-interpolated
    // state would degrade the fine reference to coarse-grid quality.

    let scr: usize = if let Some(ref s) = st_cf {
        if s.is_subcycling() { (s.coarse_dt(&llg,&h_cf)/llg.dt).round() as usize } else { 1 }
    } else { 64 }; // default subcycle ratio when AMR is not running

    let mut sf = RK4Scratch::new(fg);
    let mut sc = RK4Scratch::new(bg);
    let mut bf = VectorField2D::new(fg);
    let mut bc = VectorField2D::new(bg);

    // Log files
    let cp_f = format!("{out}/core_fine.csv");
    let cp_c = format!("{out}/core_coarse.csv");
    let cp_cf = format!("{out}/core_amr_cfft.csv");
    let cp_co = format!("{out}/core_amr_comp.csv");
    let rmse_p = format!("{out}/rmse_log.csv");
    let regrid_p = format!("{out}/regrid_log.csv");
    let comp_regrid_p = format!("{out}/comp_regrid_log.csv");
    for p in [&cp_f,&cp_c,&cp_cf,&cp_co] { File::create(p).unwrap(); append(p,"step,t_ns,x_nm,y_nm,mz,core_level\n"); }
    File::create(&rmse_p).unwrap(); append(&rmse_p,"step,t_ns,rmse_coarse,rmse_cfft,rmse_comp\n");
    File::create(&regrid_p).unwrap(); append(&regrid_p,"step,t_ns,L1,L2,L3,L1_area,L2_area,L3_area\n");
    if !skip_comp { File::create(&comp_regrid_p).unwrap(); append(&comp_regrid_p,"step,t_ns,L1,L2,L3,L1_area,L2_area,L3_area\n"); }

    let t0 = Instant::now();
    let (mut wf,mut wc,mut wcf,mut wco) = (0.0_f64,0.0_f64,0.0_f64,0.0_f64);

    // =====================================================================
    // HEADER
    // =====================================================================

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  GUSLIENKO VORTEX GYRATION — J. Appl. Phys. 91, 8037 (2002)   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!("  Disk: {:.0}nm diam, {:.0}nm thick, β={:.2}", 2.0*disk_r*1e9, dz*1e9, dz/disk_r);
    println!("  l_ex={:.1}nm, dx_coarse={:.1}nm ({:.1} cells/lex), dx_fine={:.2}nm",
        lex*1e9, dx*1e9, lex/dx, dx/rrt as f64*1e9);
    println!("  α_relax={alpha_relax}, α_dyn={alpha_dyn}, B_shift={:.0}mT", b_shift*1e3);
    println!("  dt={:.0}fs, scr={scr} (L0_dt={:.0}fs), gyr_steps={gyr_steps}", dt*1e15, scr as f64*dt*1e15);
    println!("  Relax: {}+{} steps ({:.0}+{:.0}ps), Gyration: {} steps ({:.1}ns)",
        relax_steps, field_relax_steps,
        relax_steps as f64*dt*1e12, field_relax_steps as f64*dt*1e12,
        gyr_steps, gyr_time*1e9);
    println!("  AMR: boundary_layer={bl} (L1 only, L2+=0), buffer={}, min_eff={:.0}%, indicator={}",
        cp.buffer_cells, cp.min_efficiency*100.0, ind.label());
    println!("  AMR: dyn_regrid={dyn_regrid} ({:.0}ps), snap_every={snap_every} ({:.0}ps)",
        dyn_regrid as f64 * dt * 1e12, snap_every as f64 * dt * 1e12);
    println!("  Methods: fine={} coarse={} cfft={} comp={}",
        if skip_fine{"SKIP"}else{"ON"}, if skip_coarse{"SKIP"}else{"ON"},
        if skip_cfft{"SKIP"}else{"ON"},
        if skip_comp{"SKIP"}else{"ON"});
    println!();

    // =====================================================================
    // PHASE 1: Relax to vortex ground state (α=0.5, B=0)
    // =====================================================================

    println!("── Phase 1: Relax vortex (α={alpha_relax}, {relax_steps} steps = {:.0}ps, B=0) ──",
        relax_steps as f64 * dt * 1e12);
    llg.alpha = alpha_relax; llg.b_ext = [0.0;3];

    for step in 1..=relax_steps {
        if !skip_fine { let t1=Instant::now(); step_relax(&mut m_fine,&mut bf,&llg,&mat,&mut sf,&fg,&mf); wf+=t1.elapsed().as_secs_f64(); }
        if !skip_coarse { let t1=Instant::now(); step_relax(&mut m_coarse,&mut bc,&llg,&mat,&mut sc,&bg,&mc); wc+=t1.elapsed().as_secs_f64(); }
        if step%scr==0 {
            if let Some(ref mut s) = st_cf { let t1=Instant::now(); s.step(&mut h_cf,&llg,&mat,lm); wcf+=t1.elapsed().as_secs_f64(); }
            if let (Some(s),Some(h)) = (&mut st_co,&mut h_co) { let t2=Instant::now(); s.step(h,&llg,&mat,lm); wco+=t2.elapsed().as_secs_f64(); }
        }
        // Regrid during relaxation so patches adapt as vortex settles
        if step%scr==0 && step%(relax_out)==0 {
            if !skip_cfft { if let Some((r,_)) = maybe_regrid_nested_levels(&mut h_cf,&pr_cf,rp,cp) { pr_cf=r; } }
            if let Some(hc) = h_co.as_mut() { if let Some((r,_)) = maybe_regrid_nested_levels(hc,&pr_co,rp,cp) { pr_co=r; } }
        }
        if step%relax_out==0 {
            let tps = step as f64 * dt * 1e12;
            let fine_str = if !skip_fine {
                let (fx,fy,fmz) = find_core(&m_fine, Some(&mf));
                format!("fine=({:.1},{:.1})nm mz={fmz:.4}  ", fx*1e9, fy*1e9)
            } else { String::new() };
            let coarse_str = if !skip_coarse {
                let (ccx,ccy,ccmz) = find_core(&m_coarse, Some(&mc));
                format!("coarse=({:.1},{:.1})nm mz={ccmz:.4}  ", ccx*1e9, ccy*1e9)
            } else { String::new() };
            let cfft_str = if !skip_cfft {
                let mfl = flatten_to(&h_cf, fg);
                let (cx,cy,cmz) = find_core(&mfl, Some(&mf));
                format!("core=({:.1},{:.1})nm mz={cmz:.4}", cx*1e9, cy*1e9)
            } else { String::new() };
            let comp_str = if let Some(hc) = h_co.as_ref() {
                let mc2 = flatten_to(hc, fg);
                let (ox,oy,omz) = find_core(&mc2, Some(&mf));
                format!("  comp=({:.1},{:.1})nm mz={omz:.4}", ox*1e9, oy*1e9)
            } else { String::new() };
            println!("  relax {step}/{relax_steps} ({tps:.0}ps) {fine_str}{coarse_str}{cfft_str}{comp_str}");

            // CSV logging during Phase 1 (for thesis plots showing relaxation trajectory)
            {
                let tns_log = step as f64 * dt * 1e9;
                if !skip_cfft {
                    let mfl = flatten_to(&h_cf, fg);
                    let (cx, cy, cmz) = find_core(&mfl, Some(&mf));
                    let cl = core_amr_level(&h_cf, cx, cy, &bg);
                    append(&cp_cf, &format!("{step},{tns_log:.6},{:.2},{:.2},{cmz:.6},{cl}\n", cx*1e9, cy*1e9));
                }
                if let Some(hc) = h_co.as_ref() {
                    let mfl = flatten_to(hc, fg);
                    let (ox, oy, omz) = find_core(&mfl, Some(&mf));
                    let cl = core_amr_level(hc, ox, oy, &bg);
                    append(&cp_co, &format!("{step},{tns_log:.6},{:.2},{:.2},{omz:.6},{cl}\n", ox*1e9, oy*1e9));
                }
            }
        }
    }
    println!("  Vortex ground state reached.");

    // --- Checkpoint 2: verify no vacuum contamination after Phase 1 ---
    {
        let nc = check_vacuum_contamination("cfft post-Phase1-relax", &h_cf);
        if nc > 0 { eprintln!("  WARNING: vacuum contamination re-appeared after Phase 1!"); }
    }
    if let Some(ref hc) = h_co {
        let nc = check_vacuum_contamination("comp post-Phase1-relax", hc);
        if nc > 0 { eprintln!("  WARNING: vacuum contamination re-appeared after Phase 1!"); }
    }

    // Save equilibrium snapshot (patch map + magnetisation at ground state)
    if do_plots {
        // Grid metadata for Python (always useful)
        { let mut f = File::create(format!("{out}/grid_info.csv")).unwrap();
          writeln!(f, "param,value").unwrap();
          writeln!(f, "base_nx,{}", bg.nx).unwrap();
          writeln!(f, "base_ny,{}", bg.ny).unwrap();
          writeln!(f, "fine_nx,{}", fg.nx).unwrap();
          writeln!(f, "fine_ny,{}", fg.ny).unwrap();
          writeln!(f, "dx_m,{:.6e}", bg.dx).unwrap();
          writeln!(f, "dy_m,{:.6e}", bg.dy).unwrap();
          writeln!(f, "dz_m,{:.6e}", bg.dz).unwrap();
          writeln!(f, "disk_r_m,{:.6e}", disk_r).unwrap();
          writeln!(f, "domain_m,{:.6e}", domain).unwrap();
          writeln!(f, "amr_levels,{}", amr_lvl).unwrap();
          writeln!(f, "ratio,{}", ratio).unwrap();
        }
        if !skip_cfft {
            if let Some((r,_)) = maybe_regrid_nested_levels(&mut h_cf,&pr_cf,rp,cp) { pr_cf=r; }
            let l1 = pr_cf.clone(); let l2 = l2_rects(&h_cf); let l3 = l3_rects(&h_cf);
            let ms = flatten_to(&h_cf, fg);
            let _ = save_snap(&ms,&bg,&l1,&l2,&l3,&disk,
                &format!("{out}/snap_equilibrium.png"), "Vortex ground state (B=0)");
            let _ = save_mz_snap(&ms, &disk,
                &format!("{out}/mz_equilibrium.png"), "mz — Equilibrium (B=0)", -0.2, 1.0);
            // CSV for Python: patches, coarse M, fine M
            { let mut f = File::create(format!("{out}/patches_eq.csv")).unwrap();
              writeln!(f, "level,i0,j0,nx,ny").unwrap();
              for r in &l1 { writeln!(f, "1,{},{},{},{}", r.i0, r.j0, r.nx, r.ny).unwrap(); }
              for r in &l2 { writeln!(f, "2,{},{},{},{}", r.i0, r.j0, r.nx, r.ny).unwrap(); }
              for r in &l3 { writeln!(f, "3,{},{},{},{}", r.i0, r.j0, r.nx, r.ny).unwrap(); }
            }
            { let mut f = File::create(format!("{out}/m_fine_eq.csv")).unwrap();
              writeln!(f, "i,j,mx,my,mz").unwrap();
              for j in 0..fg.ny { for i in 0..fg.nx {
                  let v = ms.data[j*fg.nx+i];
                  writeln!(f, "{i},{j},{:.6},{:.6},{:.6}", v[0], v[1], v[2]).unwrap();
              }}
            }
            let (l1n,l2n,l3n) = (l1.len(), l2.len(), l3.len());
            let (cx,cy,_) = find_core(&ms, Some(&mf));
            let cl = core_amr_level(&h_cf, cx, cy, &bg);
            println!("  Saved snap_equilibrium.png (L1={l1n} L2={l2n} L3={l3n} core@L{cl})");
        }
        if let Some(hc) = h_co.as_ref() {
            let cl1: Vec<Rect2i> = hc.patches.iter().map(|p| p.coarse_rect).collect();
            let cl2 = l2_rects(hc); let cl3 = l3_rects(hc);
            let cms = flatten_to(hc, fg);
            let _ = save_snap(&cms,&bg,&cl1,&cl2,&cl3,&disk,
                &format!("{out}/comp_snap_equilibrium.png"), "COMP Vortex ground state (B=0)");
            let _ = save_mz_snap(&cms, &disk,
                &format!("{out}/comp_mz_equilibrium.png"), "COMP mz — Equilibrium (B=0)", -0.2, 1.0);
            { let mut f = File::create(format!("{out}/comp_m_fine_eq.csv")).unwrap();
              writeln!(f, "i,j,mx,my,mz").unwrap();
              for j in 0..fg.ny { for i in 0..fg.nx {
                  let v = cms.data[j*fg.nx+i];
                  writeln!(f, "{i},{j},{:.6},{:.6},{:.6}", v[0], v[1], v[2]).unwrap();
              }}
            }
            let (ocx,ocy,_) = find_core(&cms, Some(&mf));
            let ocl = core_amr_level(hc, ocx, ocy, &bg);
            println!("  Saved comp_snap_equilibrium.png (core@L{ocl})");
        }
    }

    // =====================================================================
    // PHASE 2: Apply field, relax to shifted equilibrium
    // =====================================================================

    println!("\n── Phase 2: Apply {:.0}mT x-field, relax (α={alpha_relax}, {} steps = {:.0}ps) ──",
        b_shift*1e3, field_relax_steps, field_relax_steps as f64 * dt * 1e12);
    llg.b_ext = [b_shift, 0.0, 0.0];

    for step in 1..=field_relax_steps {
        if !skip_fine { let t1=Instant::now(); step_relax(&mut m_fine,&mut bf,&llg,&mat,&mut sf,&fg,&mf); wf+=t1.elapsed().as_secs_f64(); }
        if !skip_coarse { let t1=Instant::now(); step_relax(&mut m_coarse,&mut bc,&llg,&mat,&mut sc,&bg,&mc); wc+=t1.elapsed().as_secs_f64(); }
        if step%scr==0 {
            if let Some(ref mut s) = st_cf { let t1=Instant::now(); s.step(&mut h_cf,&llg,&mat,lm); wcf+=t1.elapsed().as_secs_f64(); }
            if let (Some(s),Some(h)) = (&mut st_co,&mut h_co) { let t2=Instant::now(); s.step(h,&llg,&mat,lm); wco+=t2.elapsed().as_secs_f64(); }
        }
        // Regrid during field relaxation so patches track the shifting core.
        // The new-region acceptance in regrid.rs detects the core patch appearing
        // inside the boundary-arc union and accepts it without force-refresh.
        if step%scr==0 && step%(relax_out)==0 {
            if !skip_cfft { if let Some((r,_)) = maybe_regrid_nested_levels(&mut h_cf,&pr_cf,rp,cp) { pr_cf=r; } }
            if let Some(hc) = h_co.as_mut() { if let Some((r,_)) = maybe_regrid_nested_levels(hc,&pr_co,rp,cp) { pr_co=r; } }
        }
        if step%relax_out==0 {
            let tps = step as f64 * dt * 1e12;
            let fine_str = if !skip_fine {
                let (fx,fy,fmz) = find_core(&m_fine, Some(&mf));
                let fr = (fx*fx+fy*fy).sqrt();
                format!("fine=({:.1},{:.1})nm r={:.1}nm mz={fmz:.4}  ", fx*1e9, fy*1e9, fr*1e9)
            } else { String::new() };
            let coarse_str = if !skip_coarse {
                let (ccx,ccy,ccmz) = find_core(&m_coarse, Some(&mc));
                let cr = (ccx*ccx+ccy*ccy).sqrt();
                format!("coarse=({:.1},{:.1})nm r={:.1}nm mz={ccmz:.4}  ", ccx*1e9, ccy*1e9, cr*1e9)
            } else { String::new() };
            let cfft_str = if !skip_cfft {
                let mfl = flatten_to(&h_cf, fg);
                let (cx,cy,cmz) = find_core(&mfl, Some(&mf));
                let r_cfft = (cx*cx+cy*cy).sqrt();
                format!("core=({:.1},{:.1})nm r={:.1}nm mz={cmz:.4}", cx*1e9, cy*1e9, r_cfft*1e9)
            } else { String::new() };
            let comp_str = if let Some(hc) = h_co.as_ref() {
                let mc2 = flatten_to(hc, fg);
                let (ox,oy,omz) = find_core(&mc2, Some(&mf));
                let r_comp = (ox*ox+oy*oy).sqrt();
                format!("  comp=({:.1},{:.1})nm r={:.1}nm mz={omz:.4}", ox*1e9, oy*1e9, r_comp*1e9)
            } else { String::new() };
            println!("  field-relax {step}/{field_relax_steps} ({tps:.0}ps) {fine_str}{coarse_str}{cfft_str}{comp_str}");

            // CSV logging during Phase 2 (for thesis plots showing field-displacement trajectory)
            {
                let tns_log = step as f64 * dt * 1e9;
                if !skip_cfft {
                    let mfl = flatten_to(&h_cf, fg);
                    let (cx, cy, cmz) = find_core(&mfl, Some(&mf));
                    let cl = core_amr_level(&h_cf, cx, cy, &bg);
                    append(&cp_cf, &format!("{step},{tns_log:.6},{:.2},{:.2},{cmz:.6},{cl}\n", cx*1e9, cy*1e9));
                }
                if let Some(hc) = h_co.as_ref() {
                    let mfl = flatten_to(hc, fg);
                    let (ox, oy, omz) = find_core(&mfl, Some(&mf));
                    let cl = core_amr_level(hc, ox, oy, &bg);
                    append(&cp_co, &format!("{step},{tns_log:.6},{:.2},{:.2},{omz:.6},{cl}\n", ox*1e9, oy*1e9));
                }
            }
        }
    }

    // --- Checkpoint 3: verify no vacuum contamination after Phase 2 ---
    {
        let nc = check_vacuum_contamination("cfft post-Phase2-field-relax", &h_cf);
        if nc > 0 { eprintln!("  WARNING: vacuum contamination after Phase 2!"); }
    }
    if let Some(ref hc) = h_co {
        let nc = check_vacuum_contamination("comp post-Phase2-field-relax", hc);
        if nc > 0 { eprintln!("  WARNING: vacuum contamination after Phase 2!"); }
    }

    // Regrid before dynamics
    if !skip_cfft { if let Some((r,_)) = maybe_regrid_nested_levels(&mut h_cf,&pr_cf,rp,cp) { pr_cf=r; } }
    if let Some(hc) = h_co.as_mut() { if let Some((r,_)) = maybe_regrid_nested_levels(hc,&pr_co,rp,cp) { pr_co=r; } }

    // Record shifted core positions
    if !skip_cfft {
        let mf_flat = flatten_to(&h_cf, fg);
        let (cx,cy,_) = find_core(&mf_flat, Some(&mf));
        let cl_shifted = core_amr_level(&h_cf, cx, cy, &bg);
        let (l1s,l2s,l3s) = (h_cf.patches.len(),
            h_cf.patches_l2plus.get(0).map(|v|v.len()).unwrap_or(0),
            h_cf.patches_l2plus.get(1).map(|v|v.len()).unwrap_or(0));
        let a3s: usize = h_cf.patches_l2plus.get(1).map(|v| v.iter().map(|p| p.coarse_rect.nx*p.coarse_rect.ny).sum()).unwrap_or(0);
        println!("  Core shifted to ({:.1}, {:.1}) nm  core@L{cl_shifted}", cx*1e9, cy*1e9);
        println!("  Pre-dynamics: L1={l1s} L2={l2s} L3={l3s} (L3_area={a3s} base cells)");
    }
    if !skip_fine {
        let (fx,fy,_) = find_core(&m_fine, Some(&mf));
        let fr = (fx*fx+fy*fy).sqrt();
        println!("  Fine core shifted to ({:.1}, {:.1}) nm  r={:.1}nm", fx*1e9, fy*1e9, fr*1e9);
    }
    if !skip_coarse {
        let (ccx,ccy,_) = find_core(&m_coarse, Some(&mc));
        let cr = (ccx*ccx+ccy*ccy).sqrt();
        println!("  Coarse core shifted to ({:.1}, {:.1}) nm  r={:.1}nm", ccx*1e9, ccy*1e9, cr*1e9);
    }
    if let Some(hc) = h_co.as_ref() {
        let mco_flat = flatten_to(hc, fg);
        let (ocx,ocy,_) = find_core(&mco_flat, Some(&mf));
        let ocl = core_amr_level(hc, ocx, ocy, &bg);
        let (cl1,cl2,cl3) = (hc.patches.len(),
            hc.patches_l2plus.get(0).map(|v|v.len()).unwrap_or(0),
            hc.patches_l2plus.get(1).map(|v|v.len()).unwrap_or(0));
        let r_comp = (ocx*ocx+ocy*ocy).sqrt();
        println!("  Comp shifted to ({:.1}, {:.1}) nm  r={:.1}nm  core@L{ocl}  L1={cl1} L2={cl2} L3={cl3}",
            ocx*1e9, ocy*1e9, r_comp*1e9);
    }

    // Switch to dynamics
    if let Some(ref mut s) = st_cf { s.relax = false; }
    if let Some(s) = st_co.as_mut() { s.relax = false; }

    if do_plots && !skip_cfft {
        let l1 = pr_cf.clone(); let l2 = l2_rects(&h_cf); let l3 = l3_rects(&h_cf);
        let ms = flatten_to(&h_cf, fg);
        let _ = save_snap(&ms,&bg,&l1,&l2,&l3,&disk, &format!("{out}/snap_shifted.png"),
            &format!("Core shifted by {:.0}mT", b_shift*1e3));
        if let Some(hc) = h_co.as_ref() {
            let cl1: Vec<Rect2i> = hc.patches.iter().map(|p| p.coarse_rect).collect();
            let cl2 = l2_rects(hc); let cl3 = l3_rects(hc);
            let cms = flatten_to(hc, fg);
            let _ = save_snap(&cms,&bg,&cl1,&cl2,&cl3,&disk,
                &format!("{out}/comp_snap_shifted.png"),
                &format!("COMP Core shifted by {:.0}mT", b_shift*1e3));
        }
    }

    // =====================================================================
    // PHASE 3: Remove field → free gyration (α = 0.01, physical Permalloy)
    // =====================================================================
    // With α=0.01: amplitude decays to ~80% over 5ns → 3.5 clean oscillation
    // periods for frequency extraction. Compare Guslienko Fig 2 (α=0.2,
    // shows trajectory shape) vs Fig 3 (frequency measurement).

    println!("\n── Phase 3: Free gyration (α={alpha_dyn}, {gyr_steps} steps = {:.1}ns) ──", gyr_time*1e9);
    llg.alpha = alpha_dyn;
    llg.b_ext = [0.0; 3]; // field OFF — core spirals back

    // --- Checkpoint 4: verify geometry before dynamics ---
    {
        let nc = check_vacuum_contamination("cfft pre-Phase3-dynamics", &h_cf);
        if nc > 0 { eprintln!("  WARNING: vacuum contamination before Phase 3!"); }
    }
    if let Some(ref hc) = h_co {
        let nc = check_vacuum_contamination("comp pre-Phase3-dynamics", hc);
        if nc > 0 { eprintln!("  WARNING: vacuum contamination before Phase 3!"); }
    }
    // Enforce clean L0 before first dynamics step
    h_cf.apply_geom_mask_to_coarse();
    if let Some(hc) = h_co.as_mut() { hc.apply_geom_mask_to_coarse(); }

    // Log initial patch state before dynamics begin
    if !skip_cfft {
        let (l1,l2,l3) = (h_cf.patches.len(),
            h_cf.patches_l2plus.get(0).map(|v|v.len()).unwrap_or(0),
            h_cf.patches_l2plus.get(1).map(|v|v.len()).unwrap_or(0));
        let a1: usize = pr_cf.iter().map(|r|r.nx*r.ny).sum();
        let a2: usize = h_cf.patches_l2plus.get(0).map(|v| v.iter().map(|p| p.coarse_rect.nx*p.coarse_rect.ny).sum()).unwrap_or(0);
        let a3: usize = h_cf.patches_l2plus.get(1).map(|v| v.iter().map(|p| p.coarse_rect.nx*p.coarse_rect.ny).sum()).unwrap_or(0);
        append(&regrid_p, &format!("0,0.000000,{l1},{l2},{l3},{a1},{a2},{a3}\n"));

        if let Some(hc) = h_co.as_ref() {
            let (cl1,cl2,cl3) = (hc.patches.len(),
                hc.patches_l2plus.get(0).map(|v|v.len()).unwrap_or(0),
                hc.patches_l2plus.get(1).map(|v|v.len()).unwrap_or(0));
            let ca1: usize = pr_co.iter().map(|r|r.nx*r.ny).sum();
            let ca2: usize = hc.patches_l2plus.get(0).map(|v| v.iter().map(|p| p.coarse_rect.nx*p.coarse_rect.ny).sum()).unwrap_or(0);
            let ca3: usize = hc.patches_l2plus.get(1).map(|v| v.iter().map(|p| p.coarse_rect.nx*p.coarse_rect.ny).sum()).unwrap_or(0);
            append(&comp_regrid_p, &format!("0,0.000000,{cl1},{cl2},{cl3},{ca1},{ca2},{ca3}\n"));
        }
    }

    let mut sn = 0_usize;

    for step in 1..=gyr_steps {
        let tns = step as f64 * dt * 1e9;

        // *** FULL LLG DYNAMICS (precession + damping) ***
        if !skip_fine { let t1=Instant::now(); step_dyn(&mut m_fine,&mut bf,&llg,&mat,&mut sf,&fg,&mf); wf+=t1.elapsed().as_secs_f64(); }
        if !skip_coarse { let t1=Instant::now(); step_dyn(&mut m_coarse,&mut bc,&llg,&mat,&mut sc,&bg,&mc); wc+=t1.elapsed().as_secs_f64(); }
        if step%scr==0 {
            if let Some(ref mut s) = st_cf { let t1=Instant::now(); s.step(&mut h_cf,&llg,&mat,lm); wcf+=t1.elapsed().as_secs_f64(); }
            if let (Some(s),Some(h)) = (&mut st_co,&mut h_co) { let t2=Instant::now(); s.step(h,&llg,&mat,lm); wco+=t2.elapsed().as_secs_f64(); }
        }

        // Regrid — normal hysteresis-based regrid.  The new-region acceptance
        // in regrid.rs detects when the indicator produces a genuinely new patch
        // (e.g. core patch at a new location) that doesn't overlap existing patches.
        // L2+ boundary_layer=0 keeps deeper levels indicator-focused on the core.
        if step%scr==0 && step%dyn_regrid==0 {
            if !skip_cfft { if let Some((r,_)) = maybe_regrid_nested_levels(&mut h_cf,&pr_cf,rp,cp) { pr_cf=r; } }
            if let Some(hc) = h_co.as_mut() { if let Some((r,_)) = maybe_regrid_nested_levels(hc,&pr_co,rp,cp) { pr_co=r; } }

            // Log patch counts and areas after each regrid
            if !skip_cfft {
                let (l1,l2,l3) = (h_cf.patches.len(),
                    h_cf.patches_l2plus.get(0).map(|v|v.len()).unwrap_or(0),
                    h_cf.patches_l2plus.get(1).map(|v|v.len()).unwrap_or(0));
                let a1: usize = pr_cf.iter().map(|r|r.nx*r.ny).sum();
                let a2: usize = h_cf.patches_l2plus.get(0).map(|v| v.iter().map(|p| p.coarse_rect.nx*p.coarse_rect.ny).sum()).unwrap_or(0);
                let a3: usize = h_cf.patches_l2plus.get(1).map(|v| v.iter().map(|p| p.coarse_rect.nx*p.coarse_rect.ny).sum()).unwrap_or(0);
                append(&regrid_p, &format!("{step},{tns:.6},{l1},{l2},{l3},{a1},{a2},{a3}\n"));
            }

            // Comp regrid log
            if let Some(hc) = h_co.as_ref() {
                let (cl1,cl2,cl3) = (hc.patches.len(),
                    hc.patches_l2plus.get(0).map(|v|v.len()).unwrap_or(0),
                    hc.patches_l2plus.get(1).map(|v|v.len()).unwrap_or(0));
                let ca1: usize = pr_co.iter().map(|r|r.nx*r.ny).sum();
                let ca2: usize = hc.patches_l2plus.get(0).map(|v| v.iter().map(|p| p.coarse_rect.nx*p.coarse_rect.ny).sum()).unwrap_or(0);
                let ca3: usize = hc.patches_l2plus.get(1).map(|v| v.iter().map(|p| p.coarse_rect.nx*p.coarse_rect.ny).sum()).unwrap_or(0);
                append(&comp_regrid_p, &format!("{step},{tns:.6},{cl1},{cl2},{cl3},{ca1},{ca2},{ca3}\n"));
            }

            // Enforce geometry mask after regrid+restriction (prevents vacuum contamination drift)
            h_cf.apply_geom_mask_to_coarse();
            if let Some(hc) = h_co.as_mut() { hc.apply_geom_mask_to_coarse(); }
        }

        // Core tracking
        if step%dyn_out==0 || step==gyr_steps {
            if !skip_fine { let (x,y,mz)=find_core(&m_fine,Some(&mf));
                append(&cp_f,&format!("{step},{tns:.6},{:.2},{:.2},{mz:.6},0\n",x*1e9,y*1e9)); }
            if !skip_coarse { let (x,y,mz)=find_core(&m_coarse,Some(&mc));
                append(&cp_c,&format!("{step},{tns:.6},{:.2},{:.2},{mz:.6},0\n",x*1e9,y*1e9)); }
            if !skip_cfft { let mfl=flatten_to(&h_cf,fg); let(x,y,mz)=find_core(&mfl,Some(&mf));
                let cl = core_amr_level(&h_cf, x, y, &bg);
                append(&cp_cf,&format!("{step},{tns:.6},{:.2},{:.2},{mz:.6},{cl}\n",x*1e9,y*1e9)); }
            if let Some(hc) = h_co.as_ref() { let mfl=flatten_to(hc,fg); let(x,y,mz)=find_core(&mfl,Some(&mf));
                let cl = core_amr_level(hc, x, y, &bg);
                append(&cp_co,&format!("{step},{tns:.6},{:.2},{:.2},{mz:.6},{cl}\n",x*1e9,y*1e9)); }

            // RMSE every 500 output steps
            if !skip_fine && step%(dyn_out*10)==0 {
                let rc = if !skip_coarse { format!("{:.6e}",rmse_fields(&m_coarse.resample_to_grid(fg),&m_fine).0) } else { "NaN".into() };
                let rcf = if !skip_cfft { format!("{:.6e}",rmse_fields(&flatten_to(&h_cf,fg),&m_fine).0) } else { "NaN".into() };
                let rco = if let Some(hc)=h_co.as_ref() { format!("{:.6e}",rmse_fields(&flatten_to(hc,fg),&m_fine).0) } else { "NaN".into() };
                append(&rmse_p,&format!("{step},{tns:.4},{rc},{rcf},{rco}\n"));
            }
        }

        // Console — include core_level so we can see immediately if L3 covers the core
        if step%(dyn_out*20)==0 || step==gyr_steps {
            let mut parts: Vec<String> = Vec::new();
            if !skip_fine {
                let (x,y,_) = find_core(&m_fine, Some(&mf));
                let r = (x*x+y*y).sqrt();
                parts.push(format!("fine=({:.1},{:.1})nm r={:.1}nm", x*1e9, y*1e9, r*1e9));
            }
            if !skip_coarse {
                let (x,y,_) = find_core(&m_coarse, Some(&mc));
                let r = (x*x+y*y).sqrt();
                parts.push(format!("coarse=({:.1},{:.1})nm r={:.1}nm", x*1e9, y*1e9, r*1e9));
            }
            if !skip_cfft {
                let mfl=flatten_to(&h_cf,fg); let(x,y,_)=find_core(&mfl,Some(&mf));
                let cl = core_amr_level(&h_cf, x, y, &bg);
                let (l1,l2,l3) = (h_cf.patches.len(), h_cf.patches_l2plus.get(0).map(|v|v.len()).unwrap_or(0),
                    h_cf.patches_l2plus.get(1).map(|v|v.len()).unwrap_or(0));
                let r_cfft = (x*x+y*y).sqrt();
                parts.push(format!("core=({:.1},{:.1})nm r={:.1}nm L1={l1} L2={l2} L3={l3} core@L{cl}",
                    x*1e9, y*1e9, r_cfft*1e9));
            }
            if let Some(hc) = h_co.as_ref() {
                let mc2 = flatten_to(hc, fg);
                let (ox,oy,_) = find_core(&mc2, Some(&mf));
                let ocl = core_amr_level(hc, ox, oy, &bg);
                let (cl1,cl2,cl3) = (hc.patches.len(),
                    hc.patches_l2plus.get(0).map(|v|v.len()).unwrap_or(0),
                    hc.patches_l2plus.get(1).map(|v|v.len()).unwrap_or(0));
                let r_comp = (ox*ox+oy*oy).sqrt();
                parts.push(format!("comp=({:.1},{:.1})nm r={:.1}nm L1={cl1} L2={cl2} L3={cl3} core@L{ocl}",
                    ox*1e9, oy*1e9, r_comp*1e9));
            }
            println!("  gyr {step}/{gyr_steps} t={tns:.2}ns {}", parts.join("  "));
        }

        // Snapshots
        if do_plots && (step%snap_every==0 || step==gyr_steps) {
            // ── Fine-FFT snapshot ──
            if !skip_fine {
                // CSV: fine magnetisation
                if sn % 5 == 0 || step == gyr_steps {
                    let mut f = File::create(format!("{out}/fine_m_{sn:03}.csv")).unwrap();
                    writeln!(f, "i,j,mx,my,mz").unwrap();
                    for j in 0..fg.ny { for i in 0..fg.nx {
                        let v = m_fine.data[j*fg.nx+i];
                        writeln!(f, "{i},{j},{:.6},{:.6},{:.6}", v[0], v[1], v[2]).unwrap();
                    }}
                }
            }

            // ── Coarse-FFT snapshot ──
            if !skip_coarse {
                if sn % 5 == 0 || step == gyr_steps {
                    let mut f = File::create(format!("{out}/coarse_m_{sn:03}.csv")).unwrap();
                    writeln!(f, "i,j,mx,my,mz").unwrap();
                    for j in 0..bg.ny { for i in 0..bg.nx {
                        let v = m_coarse.data[j*bg.nx+i];
                        writeln!(f, "{i},{j},{:.6},{:.6},{:.6}", v[0], v[1], v[2]).unwrap();
                    }}
                }
            }

            // ── AMR+cfft snapshots ──
            if !skip_cfft {
                let l1=pr_cf.clone(); let l2=l2_rects(&h_cf); let l3=l3_rects(&h_cf);
                let ms=flatten_to(&h_cf,fg);
                let _ = save_snap(&ms,&bg,&l1,&l2,&l3,&disk, &format!("{out}/snap_gyr_{sn:03}.png"),
                    &format!("t={tns:.2}ns (α={alpha_dyn})"));
                let _ = save_mz_snap(&ms, &disk,
                    &format!("{out}/mz_gyr_{sn:03}.png"),
                    &format!("mz — t={tns:.2}ns"), -0.2, 1.0);

                // ── CSV outputs for Python post-processing ──
                // 1. Patch rectangles (small file, every snapshot)
                { let mut f = File::create(format!("{out}/patches_{sn:03}.csv")).unwrap();
                  writeln!(f, "level,i0,j0,nx,ny").unwrap();
                  for r in &l1 { writeln!(f, "1,{},{},{},{}", r.i0, r.j0, r.nx, r.ny).unwrap(); }
                  for r in &l2 { writeln!(f, "2,{},{},{},{}", r.i0, r.j0, r.nx, r.ny).unwrap(); }
                  for r in &l3 { writeln!(f, "3,{},{},{},{}", r.i0, r.j0, r.nx, r.ny).unwrap(); }
                }
                // 2. Coarse (L0) magnetisation — 80×80, good for patch map overlays
                { let mut f = File::create(format!("{out}/m_coarse_{sn:03}.csv")).unwrap();
                  writeln!(f, "i,j,mx,my,mz").unwrap();
                  for j in 0..bg.ny { for i in 0..bg.nx {
                      let v = h_cf.coarse.data[j*bg.nx+i];
                      writeln!(f, "{i},{j},{:.6},{:.6},{:.6}", v[0], v[1], v[2]).unwrap();
                  }}
                }
                // 3. Fine (flattened) magnetisation — every 5th snapshot + final
                if sn % 5 == 0 || step == gyr_steps {
                    let mut f = File::create(format!("{out}/m_fine_{sn:03}.csv")).unwrap();
                    writeln!(f, "i,j,mx,my,mz").unwrap();
                    for j in 0..fg.ny { for i in 0..fg.nx {
                        let v = ms.data[j*fg.nx+i];
                        writeln!(f, "{i},{j},{:.6},{:.6},{:.6}", v[0], v[1], v[2]).unwrap();
                    }}
                }
            }

            // ── Composite outputs (equivalent to cfft above) ──
            if let Some(hc) = h_co.as_ref() {
                let cl1: Vec<Rect2i> = hc.patches.iter().map(|p| p.coarse_rect).collect();
                let cl2 = l2_rects(hc); let cl3 = l3_rects(hc);
                let cms = flatten_to(hc, fg);
                let _ = save_snap(&cms,&bg,&cl1,&cl2,&cl3,&disk,
                    &format!("{out}/comp_snap_gyr_{sn:03}.png"),
                    &format!("COMP t={tns:.2}ns (α={alpha_dyn})"));
                let _ = save_mz_snap(&cms, &disk,
                    &format!("{out}/comp_mz_gyr_{sn:03}.png"),
                    &format!("COMP mz — t={tns:.2}ns"), -0.2, 1.0);
                { let mut f = File::create(format!("{out}/comp_patches_{sn:03}.csv")).unwrap();
                  writeln!(f, "level,i0,j0,nx,ny").unwrap();
                  for r in &cl1 { writeln!(f, "1,{},{},{},{}", r.i0, r.j0, r.nx, r.ny).unwrap(); }
                  for r in &cl2 { writeln!(f, "2,{},{},{},{}", r.i0, r.j0, r.nx, r.ny).unwrap(); }
                  for r in &cl3 { writeln!(f, "3,{},{},{},{}", r.i0, r.j0, r.nx, r.ny).unwrap(); }
                }
                { let mut f = File::create(format!("{out}/comp_m_coarse_{sn:03}.csv")).unwrap();
                  writeln!(f, "i,j,mx,my,mz").unwrap();
                  for j in 0..bg.ny { for i in 0..bg.nx {
                      let v = hc.coarse.data[j*bg.nx+i];
                      writeln!(f, "{i},{j},{:.6},{:.6},{:.6}", v[0], v[1], v[2]).unwrap();
                  }}
                }
                if sn % 5 == 0 || step == gyr_steps {
                    let mut f = File::create(format!("{out}/comp_m_fine_{sn:03}.csv")).unwrap();
                    writeln!(f, "i,j,mx,my,mz").unwrap();
                    for j in 0..fg.ny { for i in 0..fg.nx {
                        let v = cms.data[j*fg.nx+i];
                        writeln!(f, "{i},{j},{:.6},{:.6},{:.6}", v[0], v[1], v[2]).unwrap();
                    }}
                }
            }

            sn += 1;
        }
    }

    let total = t0.elapsed().as_secs_f64();

    // =====================================================================
    // RESULTS
    // =====================================================================

    let cf = if !skip_fine { Some(find_core(&m_fine,Some(&mf))) } else { None };
    let cc = if !skip_coarse { Some(find_core(&m_coarse,Some(&mc))) } else { None };
    let ccf = if !skip_cfft { let mfl = flatten_to(&h_cf,fg); Some(find_core(&mfl,Some(&mf))) } else { None };
    let cco = h_co.as_ref().map(|h| { let m=flatten_to(h,fg); find_core(&m,Some(&mf)) });

    let rf_c = if !skip_fine&&!skip_coarse { Some(rmse_fields(&m_coarse.resample_to_grid(fg),&m_fine).0) } else { None };
    let rf_cf = if !skip_fine&&!skip_cfft { let mfl = flatten_to(&h_cf,fg); Some(rmse_fields(&mfl,&m_fine).0) } else { None };
    let rf_co = if !skip_fine { h_co.as_ref().map(|h| rmse_fields(&flatten_to(h,fg),&m_fine).0) } else { None };

    let cov = if !skip_cfft { patches_fine(&pr_cf,rrt).iter().map(|r|r.nx*r.ny).sum::<usize>() as f64 / (fnx*fny) as f64 } else { 0.0 };

    // Guslienko gyration frequency:
    // The "two-vortices" rigid-model leading-order result is ω₀ = (20/9) γ μ₀ Ms β
    // but this badly overestimates at β=0.2 (gives ~12.5 GHz vs measured ~0.7 GHz)
    // because the full expression involves magnetostatic integrals F(β) that provide
    // large corrections at finite β.
    //
    // Ground truth: Novosad et al., Phys. Rev. B 72, 024455 (2005) — experimental data
    //   on Permalloy disks with same material parameters:
    //     2.2μm diam, 40nm thick (β=0.036): f = 162 MHz
    //     1.1μm diam, 40nm thick (β=0.073): f = 272 MHz
    //     2.0μm diam, 20nm thick (β=0.020): f = 83  MHz
    //   Empirical scaling: f ≈ 3700 × β MHz for Permalloy.
    //
    // Guslienko 2002 Fig 3, curve b: f ≈ 0.7 GHz for 200nm diam, 20nm thick (β=0.2).
    let beta = dz/disk_r;
    let fgus_leading = (20.0/9.0) * GAMMA_E_RAD_PER_S_T * mu0 * mat.ms * beta / (2.0*PI) / 1e9;
    // Empirical interpolation from Novosad data + Guslienko Fig 3:
    let fgus_expected = 3.7 * beta;  // GHz — matches experimental trend
    let fgus = fgus_expected;  // Use empirical for comparison; leading-order shown for reference

    // ── Frequency extraction from core trajectories ──
    let (ft, fx, _fy) = read_core_csv(&cp_f);
    let (f_fine_ghz, f_fine_nc) = estimate_gyration_freq(&ft, &fx);
    let (ct, cx, _cy) = read_core_csv(&cp_c);
    let (f_coarse_ghz, f_coarse_nc) = estimate_gyration_freq(&ct, &cx);
    let (at, ax, _ay) = read_core_csv(&cp_cf);
    let (f_amr_ghz, f_amr_nc) = estimate_gyration_freq(&at, &ax);
    let (ot, ox, _oy) = read_core_csv(&cp_co);
    let (f_comp_ghz, f_comp_nc) = estimate_gyration_freq(&ot, &ox);

    let bar = "═".repeat(66); let thin = "─".repeat(66);
    println!("\n╔{bar}╗");
    println!("║{:^66}║", "RESULTS: GUSLIENKO VORTEX GYRATION");
    println!("╚{bar}╝");
    println!("\n  Guslienko f_gyr ≈ {fgus:.3} GHz (empirical: 3.7×β, Novosad 2005)");
    println!("  Leading-order (20/9)γμ₀Msβ/(2π) = {fgus_leading:.1} GHz (overestimates at β={beta:.2})");
    println!("  Novosad expt: 83MHz (β=0.02), 162MHz (β=0.036), 272MHz (β=0.073)\n");

    println!("  GYRATION FREQUENCY (from zero-crossing analysis)");
    println!("  {thin}");
    if f_fine_ghz > 0.0   { println!("  Fine FFT      {:.3} GHz ({} crossings)  [reference]", f_fine_ghz, f_fine_nc); }
    if f_coarse_ghz > 0.0 { println!("  Coarse FFT    {:.3} GHz ({} crossings)  err vs fine: {:.1}%",
        f_coarse_ghz, f_coarse_nc, if f_fine_ghz>0.0 {100.0*(f_coarse_ghz-f_fine_ghz).abs()/f_fine_ghz} else {f64::NAN}); }
    if f_amr_ghz > 0.0    { println!("  AMR+cfft      {:.3} GHz ({} crossings)  err vs fine: {:.1}%",
        f_amr_ghz, f_amr_nc, if f_fine_ghz>0.0 {100.0*(f_amr_ghz-f_fine_ghz).abs()/f_fine_ghz} else {f64::NAN}); }
    if f_comp_ghz > 0.0   { println!("  AMR+composite {:.3} GHz ({} crossings)  err vs fine: {:.1}%",
        f_comp_ghz, f_comp_nc, if f_fine_ghz>0.0 {100.0*(f_comp_ghz-f_fine_ghz).abs()/f_fine_ghz} else {f64::NAN}); }
    println!("  Guslienko     {fgus:.3} GHz (empirical, Novosad 2005)");
    if f_fine_ghz > 0.0 { println!("  Fine vs Guslienko: {:.1}%", 100.0*(f_fine_ghz-fgus).abs()/fgus); }
    if f_amr_ghz > 0.0  { println!("  AMR vs Guslienko:  {:.1}%", 100.0*(f_amr_ghz-fgus).abs()/fgus); }

    println!("\n  FINAL CORE (α={alpha_dyn}: amplitude decays to ~80% at 5ns, core still oscillating)");
    println!("  {thin}");
    if let Some((x,y,mz))=cf  { println!("  Fine FFT      ({:>6.1},{:>6.1}) nm  mz={mz:.4}",x*1e9,y*1e9); }
    if let Some((x,y,mz))=cc  { println!("  Coarse FFT    ({:>6.1},{:>6.1}) nm  mz={mz:.4}",x*1e9,y*1e9); }
    if let Some((x,y,mz))=ccf { println!("  AMR+cfft      ({:>6.1},{:>6.1}) nm  mz={mz:.4}",x*1e9,y*1e9); }
    if let Some((x,y,mz))=cco { println!("  AMR+composite ({:>6.1},{:>6.1}) nm  mz={mz:.4}",x*1e9,y*1e9); }

    println!("\n  RMSE vs FINE FFT");
    println!("  {thin}");
    if let Some(r)=rf_c  { println!("  Coarse FFT     {r:.4e}"); }
    if let Some(r)=rf_cf { println!("  AMR+cfft       {r:.4e}"); }
    if let Some(r)=rf_co { println!("  AMR+composite  {r:.4e}"); }
    if skip_fine { println!("  (fine ref skipped)"); }

    if !skip_cfft {
        let l1n = h_cf.patches.len();
        let l2n = h_cf.patches_l2plus.get(0).map(|v|v.len()).unwrap_or(0);
        let l3n = h_cf.patches_l2plus.get(1).map(|v|v.len()).unwrap_or(0);
        println!("\n  AMR: L1={l1n} L2={l2n} L3={l3n} cov={:.1}%", cov*100.0);
    }

    println!("\n  WALL TIME");
    println!("  {thin}");
    println!("  Total            {:>8.1}s", total);
    if !skip_fine   { println!("  Fine FFT         {:>8.1}s", wf); }
    if !skip_coarse { println!("  Coarse FFT       {:>8.1}s", wc); }
    if !skip_cfft   { println!("  AMR+cfft         {:>8.1}s", wcf); }
    if !skip_comp   { println!("  AMR+composite    {:>8.1}s", wco); }
    if !skip_fine && wcf>0.0 && !skip_cfft { println!("\n  SPEEDUP: AMR+cfft = {:.1}× vs fine FFT", wf/wcf); }

    // ── Trajectory plot (core X/R vs Y/R) ──
    if do_plots {
        let _ = plot_trajectories(out, disk_r,
            &cp_f, &cp_c, &cp_cf, &cp_co,
            skip_fine, skip_coarse, skip_cfft, skip_comp);
        let _ = plot_core_vs_time(out, &cp_f, &cp_c, &cp_cf, &cp_co,
            skip_fine, skip_coarse, skip_cfft, skip_comp);
    }

    println!("\n  OUTPUT: {out}/");
    println!("  core_*.csv         ← core position time series");
    println!("  rmse_log.csv       ← RMSE vs time (requires fine ref)");
    println!("  regrid_log.csv     ← cfft patch counts + areas at each regrid");
    println!("  comp_regrid_log.csv← comp patch counts + areas at each regrid");
    println!("  grid_info.csv      ← grid metadata for Python scripts");
    if do_plots {
        println!("  snap_*.png         ← Rust-rendered snapshots (quick preview)");
        println!("  patches_*.csv      ← patch rectangles per snapshot (for Python)");
        println!("  m_coarse_*.csv     ← L0 magnetisation per snapshot (80×80)");
        println!("  m_fine_*.csv       ← fine magnetisation at key snapshots (for 3D/mesh)");
        println!("  m_fine_eq.csv      ← fine magnetisation at equilibrium");
        println!("  trajectory.png     ← core X/R vs Y/R overlay");
        println!("  core_x_vs_t.png    ← core x(t) for frequency estimation");
    }

    // Save summary
    { let mut f=File::create(format!("{out}/summary.txt")).unwrap();
      writeln!(f,"Guslienko vortex gyration — J. Appl. Phys. 91, 8037 (2002)").unwrap();
      writeln!(f,"Disk: {:.0}nm diam, {:.0}nm thick, β={beta:.2}", 2.0*disk_r*1e9, dz*1e9).unwrap();
      writeln!(f,"f_gyr ≈ {fgus:.3} GHz (empirical: 3.7×β, Novosad PRB 72, 024455)").unwrap();
      writeln!(f,"Leading-order (20/9)γμ₀Msβ/(2π) = {fgus_leading:.1} GHz (overestimates)").unwrap();
      writeln!(f,"Grid: coarse {}², fine {}², AMR lvl {}", bnx, fnx, amr_lvl).unwrap();
      writeln!(f,"α_dyn={alpha_dyn}, B_shift={:.0}mT\n",b_shift*1e3).unwrap();
      writeln!(f,"Method           | f_gyr(GHz) | Wall(s)  | RMSE vs fine  | Speedup").unwrap();
      if !skip_fine { writeln!(f,"Fine FFT         | {:>10.3} | {:>7.1} | (reference)   | 1.0×",f_fine_ghz,wf).unwrap(); }
      if !skip_coarse { writeln!(f,"Coarse FFT       | {:>10.3} | {:>7.1} | {:.4e}    | {:.1}×",f_coarse_ghz,wc,rf_c.unwrap_or(f64::NAN),if wc>0.0&&!skip_fine{wf/wc}else{f64::NAN}).unwrap(); }
      if !skip_cfft { writeln!(f,"AMR+cfft         | {:>10.3} | {:>7.1} | {:.4e}    | {:.1}×",f_amr_ghz,wcf,rf_cf.unwrap_or(f64::NAN),if wcf>0.0&&!skip_fine{wf/wcf}else{f64::NAN}).unwrap(); }
      if !skip_comp { writeln!(f,"AMR+composite    | {:>10.3} | {:>7.1} | {:.4e}    | {:.1}×",f_comp_ghz,wco,rf_co.unwrap_or(f64::NAN),if wco>0.0&&!skip_fine{wf/wco}else{f64::NAN}).unwrap(); }
      writeln!(f,"\nGuslienko/Novosad empirical: {fgus:.3} GHz").unwrap();
    }

    println!();
}