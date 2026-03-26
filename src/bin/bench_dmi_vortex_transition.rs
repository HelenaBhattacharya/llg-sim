// src/bin/bench_dmi_vortex_transition.rs
//
// DMI-Driven Vortex → Skyrmion Transition in a Confined Disk
// ===========================================================
//
// Demonstrates the topological transition from a magnetic vortex (Q = +½)
// to a skyrmion-embedded-vortex state (Q = -3/2) driven by increasing
// interfacial Dzyaloshinskii-Moriya interaction, following:
//   Zhang et al., JMMM 630 (2025) 173354  — n-skyrmion vortex states
//   Novak et al., JMMM 451 (2018) 749     — skyrmion stability in disks
//
// Setup:
//   - 200nm diameter CoFeB disk, 2nm thick  (same diameter as vortex gyration)
//   - Ms = 1.1×10⁶ A/m, A = 1.3×10⁻¹¹ J/m  (CoFeB, from Zhang et al.)
//   - K_u = 4.5×10⁵ J/m³ (perpendicular anisotropy — required for skyrmion stability)
//   - D_ex swept from 0 to 3 mJ/m² (interfacial DMI)
//   - dx = dy = 2nm, dz = 2nm (cubic cells — ideal MG conditioning)
//
// The disk is thin (2nm = single z-cell) so interfacial DMI is at full
// strength.  PMA competes with demag shape anisotropy; DMI tilts the
// vortex core from Bloch-type towards Néel-type and eventually stabilises
// a confined skyrmion.
//
// Solver: AMR with composite multigrid demag — identical infrastructure
// to bench_vortex_gyration and bench_composite_vcycle.  The composite
// solver (Newell-direct for L0 B, MG V-cycle for patch ghost-fills)
// provides accurate boundary B where surface charges σ = M·n̂ interact
// with the DMI boundary conditions.
//
// AMR: Default 1 level for fast testing (dx_fine=1nm, 4.1 cells/l_ex).
//   Set DMI_AMR_LEVELS=2 for thesis-quality runs (dx_fine=0.5nm, 8.3 cells/l_ex).
//   L1 boundary patches resolve the curved disk edge.
//   L2 (if present) tracks the expanding vortex core / skyrmion.
//
// Subcycling: dt auto-selected for CFL safety at finest level.
//   scr computed from the stepper's coarse_dt (matches bench_vortex_gyration).
//
// Early stop: relaxation terminates when |Δ<mz>| between checks falls below
//   a threshold (default 1e-6).  This detects equilibrium without needing
//   the full H_eff (which includes demag via the stepper's b_add, not
//   accessible outside the stepper).  Disable with DMI_CONV_TOL=0.
//
// Protocol: For each D value, initialise a vortex and relax with α=0.5
// until equilibrium.  Output mz snapshot and topological charge Q.
//
// Run (fast test — ~5-10 min per D value with 1 AMR level):
//   LLG_DEMAG_COMPOSITE_VCYCLE=1 LLG_COMPOSITE_MAX_CYCLES=1 \
//     DMI_SWEEP="0,1.5,2.5" \
//     cargo run --release --bin bench_dmi_vortex_transition
//
// Run (full sweep, thesis quality):
//   LLG_DEMAG_COMPOSITE_VCYCLE=1 LLG_COMPOSITE_MAX_CYCLES=1 \
//     DMI_AMR_LEVELS=2 \
//     cargo run --release --bin bench_dmi_vortex_transition
//
// Customise:
//   DMI_SWEEP="0,0.5,1.0,1.5,2.0,2.5,3.0"  — D values in mJ/m²
//   DMI_KU=4.5e5                             — PMA constant (J/m³)
//   DMI_STEPS=200000                         — max relaxation steps (fine-level count)
//   DMI_AMR_LEVELS=2                         — 2 refinement levels for thesis quality
//   DMI_DISK_R=100e-9                        — disk radius (m)
//   DMI_CONV_TOL=1e-6                        — early-stop Δ<mz> threshold (0 to disable)

use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

use llg_sim::effective_field::FieldMask;
use llg_sim::geometry_mask::{MaskShape, edge_smooth_n, apply_fill_fractions};
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
//  Topological charge (Berg–Lüscher lattice formula)
// ═══════════════════════════════════════════════════════════════════════════

fn solid_angle_tri(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let cross = [
        b[1]*c[2] - b[2]*c[1],
        b[2]*c[0] - b[0]*c[2],
        b[0]*c[1] - b[1]*c[0],
    ];
    let numer = a[0]*cross[0] + a[1]*cross[1] + a[2]*cross[2];
    let ab = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    let bc = b[0]*c[0] + b[1]*c[1] + b[2]*c[2];
    let ca = c[0]*a[0] + c[1]*a[1] + c[2]*a[2];
    2.0 * numer.atan2(1.0 + ab + bc + ca)
}

fn topological_charge(m: &VectorField2D, mask: Option<&[bool]>) -> f64 {
    let nx = m.grid.nx;
    let ny = m.grid.ny;
    let mut q = 0.0_f64;
    for j in 0..ny.saturating_sub(1) {
        for i in 0..nx.saturating_sub(1) {
            let i00 = m.idx(i, j);
            let i10 = m.idx(i+1, j);
            let i01 = m.idx(i, j+1);
            let i11 = m.idx(i+1, j+1);
            if let Some(msk) = mask {
                if !msk[i00] || !msk[i10] || !msk[i01] || !msk[i11] { continue; }
            }
            q += solid_angle_tri(m.data[i00], m.data[i10], m.data[i01]);
            q += solid_angle_tri(m.data[i10], m.data[i11], m.data[i01]);
        }
    }
    q / (4.0 * PI)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn flatten_to(h: &AmrHierarchy2D, tgt: Grid2D) -> VectorField2D {
    let mut out = h.coarse.resample_to_grid(tgt);
    for p in &h.patches { p.scatter_into_uniform_fine(&mut out); }
    for lvl in &h.patches_l2plus {
        for p in lvl { p.scatter_into_uniform_fine(&mut out); }
    }
    out
}

/// Compute <mz> over material cells on the coarse grid (fast — no flatten needed).
fn coarse_mz_avg(h: &AmrHierarchy2D) -> f64 {
    let mask = h.geom_mask.as_deref();
    let mut sum = 0.0_f64;
    let mut cnt = 0_usize;
    for (idx, v) in h.coarse.data.iter().enumerate() {
        if let Some(msk) = mask {
            if !msk[idx] { continue; }
        }
        sum += v[2];
        cnt += 1;
    }
    if cnt > 0 { sum / cnt as f64 } else { 0.0 }
}

fn find_core(m: &VectorField2D, mask: Option<&[bool]>) -> (f64, f64, f64) {
    let nx = m.grid.nx;
    let ny = m.grid.ny;
    let cx = nx as f64 * 0.5;
    let cy = ny as f64 * 0.5;
    let mut best_mz = f64::NEG_INFINITY;
    let mut bx = 0.0_f64;
    let mut by = 0.0_f64;
    for j in 0..ny {
        for i in 0..nx {
            let idx = m.idx(i, j);
            if let Some(msk) = mask { if !msk[idx] { continue; } }
            if m.data[idx][2] > best_mz {
                best_mz = m.data[idx][2];
                bx = (i as f64 + 0.5 - cx) * m.grid.dx;
                by = (j as f64 + 0.5 - cy) * m.grid.dy;
            }
        }
    }
    (bx, by, best_mz)
}

fn append(path: &str, s: &str) {
    let mut f = std::fs::OpenOptions::new().append(true).open(path).unwrap();
    std::io::Write::write_all(&mut f, s.as_bytes()).unwrap();
}

fn env_f64(k: &str, d: f64) -> f64 { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
fn env_usize(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> std::io::Result<()> {
    let ms: f64     = 1.1e6;
    let a_ex: f64   = 1.3e-11;
    let k_u: f64    = env_f64("DMI_KU", 4.5e5);

    let disk_r: f64 = env_f64("DMI_DISK_R", 100.0e-9);
    let dz: f64     = 2.0e-9;
    let dx: f64     = 2.0e-9;
    let dy = dx;
    let domain = 2.0 * disk_r + 20.0e-9;
    let bnx = (domain / dx).ceil() as usize;
    let bny = bnx;

    let ratio: usize = 2;
    let ghost: usize = 2;
    // Default 1 AMR level for fast testing.  Set DMI_AMR_LEVELS=2 for thesis quality.
    let amr_lvl: usize = env_usize("DMI_AMR_LEVELS", 1);
    let rrt: usize = ratio.pow(amr_lvl as u32);
    let dx_fine = dx / rrt as f64;

    // CFL-safe dt at finest level
    let omega_max = GAMMA_E_RAD_PER_S_T * (2.0*a_ex/ms) * 2.0*(PI/dx_fine).powi(2);
    let dt_cfl = 2.83 / omega_max;
    let dt: f64 = (dt_cfl * 0.6).min(5.0e-14);

    let alpha: f64 = 0.5;
    let relax_steps: usize = env_usize("DMI_STEPS", 200_000);
    let regrid_every: usize = env_usize("DMI_REGRID", 10_000);
    // Convergence: stop when |Δ<mz>| between consecutive checks falls below this.
    // Checked every conv_check_interval stepper calls.
    let conv_tol: f64 = env_f64("DMI_CONV_TOL", 1e-4);
    let conv_check_interval: usize = 200;  // stepper calls between convergence checks

    let d_values_mjm2: Vec<f64> = std::env::var("DMI_SWEEP")
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);

    let lex = (2.0 * a_ex / (MU0 * ms * ms)).sqrt();
    let d_c = 4.0 * (a_ex * k_u).sqrt() / PI;
    let k_eff = k_u - MU0 * ms * ms / 2.0;
    let q_factor = 2.0 * k_u / (MU0 * ms * ms);

    let out = "out/bench_dmi_transition";
    fs::create_dir_all(out)?;

    let bg = Grid2D::new(bnx, bny, dx, dy, dz);
    let fg = Grid2D::new(bnx*rrt, bny*rrt, dx_fine, dx_fine, dz);

    let disk = MaskShape::Disk { center: (0.0, 0.0), radius: disk_r };
    let ns = edge_smooth_n();
    let (mc, ffc) = disk.to_mask_and_fill(&bg, ns);
    let (mf, _) = disk.to_mask_and_fill(&fg, ns);
    let n_mat = mc.iter().filter(|&&v| v).count();

    let ind = IndicatorKind::from_env();
    let bl: usize = env_usize("LLG_AMR_BOUNDARY_LAYER", 4);
    let rp = RegridPolicy {
        indicator: ind, buffer_cells: 4, boundary_layer: bl,
        min_change_cells: 1, min_area_change_frac: 0.01,
    };
    let cp = ClusterPolicy {
        indicator: ind, buffer_cells: 4, boundary_layer: bl,
        min_patch_area: 16, merge_distance: 1, max_patches: 0,
        connectivity: Connectivity::Eight, min_efficiency: 0.65,
        max_flagged_fraction: 1.0, confine_dilation: true,
    };

    // ── Build a temporary hierarchy + stepper to query the true subcycle ratio ──
    // scr must match the stepper's internal coarse_dt exactly (same as bench_vortex_gyration).
    let scr: usize;
    {
        let m_tmp = {
            let mut m = VectorField2D::new(bg);
            initial_states::init_vortex(&mut m, &bg, (0.0,0.0), 1.0, 1.0, 3.0*dx, Some(&mc));
            apply_fill_fractions(&mut m.data, &ffc);
            m
        };
        let h_tmp = AmrHierarchy2D::new(bg, m_tmp, ratio, ghost);
        unsafe { std::env::set_var("LLG_AMR_DEMAG_MODE", "composite"); }
        let st_tmp = AmrStepperRK4::new(&h_tmp, true);
        unsafe { std::env::remove_var("LLG_AMR_DEMAG_MODE"); }
        let llg_tmp = LLGParams {
            gamma: GAMMA_E_RAD_PER_S_T, alpha, dt, b_ext: [0.0; 3],
        };
        scr = if st_tmp.is_subcycling() {
            (st_tmp.coarse_dt(&llg_tmp, &h_tmp) / llg_tmp.dt).round() as usize
        } else {
            1
        };
    }

    let report_every: usize = (relax_steps / 10).max(scr);
    let n_stepper_calls_max = relax_steps / scr.max(1);
    let phys_time_ps = relax_steps as f64 * dt * 1e12;

    let bar = "═".repeat(66);
    println!("╔{bar}╗");
    println!("║{:^66}║", "DMI VORTEX → SKYRMION TRANSITION");
    println!("║{:^66}║", "Zhang et al., JMMM 630 (2025) 173354");
    println!("╚{bar}╝");
    println!("  Disk: {:.0}nm diam, {:.0}nm thick", 2.0*disk_r*1e9, dz*1e9);
    println!("  Material: CoFeB  Ms={:.0} kA/m, A={:.1} pJ/m", ms/1e3, a_ex/1e-12);
    println!("  PMA: Ku={:.1} kJ/m³, Q={:.2}, K_eff={:.1} kJ/m³", k_u/1e3, q_factor, k_eff/1e3);
    println!("  l_ex={:.1}nm, D_c={:.2} mJ/m²", lex*1e9, d_c*1e3);
    println!("  Grid: {}×{}, dx={:.1}nm, dx_fine={:.2}nm ({:.1} cells/lex)",
             bnx, bny, dx*1e9, dx_fine*1e9, lex/dx_fine);
    println!("  dt={:.1}fs (CFL={:.1}fs), scr={}, AMR={}lv (L0→L{})",
             dt*1e15, dt_cfl*1e15, scr, amr_lvl, amr_lvl);
    println!("  Relax: α={}, max {} steps ({:.0}ps), up to {} stepper calls",
             alpha, relax_steps, phys_time_ps, n_stepper_calls_max);
    println!("  Convergence: |Δ<mz>| < {:.0e} (check every {} calls), regrid every {} steps",
             conv_tol, conv_check_interval, regrid_every);
    println!("  DMI sweep: {:?} mJ/m²  ({} values)", d_values_mjm2, d_values_mjm2.len());
    println!("  Geometry: {} material / {} total ({:.1}%)", n_mat, bnx*bny,
             100.0*n_mat as f64/(bnx*bny) as f64);
    println!();

    let summary_path = format!("{out}/summary.csv");
    { let mut f = File::create(&summary_path)?;
      writeln!(f, "D_mJm2,Q,mz_avg,mz_core,core_x_nm,core_y_nm,L1,L2,wall_s,converged_step")?; }

    for &d_mjm2 in &d_values_mjm2 {
        let d = d_mjm2 * 1e-3;
        println!("══ D = {:.1} mJ/m² (D/Dc = {:.2}) ══════════════════════════",
                 d_mjm2, if d_c > 0.0 { d/d_c } else { 0.0 });

        let mat = Material {
            ms, a_ex, k_u, easy_axis: [0.0, 0.0, 1.0],
            dmi: if d.abs() > 1e-30 { Some(d) } else { None },
            demag: true, demag_method: DemagMethod::FftUniform,
        };
        let llg = LLGParams {
            gamma: GAMMA_E_RAD_PER_S_T, alpha, dt, b_ext: [0.0; 3],
        };

        let mk_coarse = || -> VectorField2D {
            let mut m = VectorField2D::new(bg);
            initial_states::init_vortex(&mut m, &bg, (0.0,0.0), 1.0, 1.0, 3.0*dx, Some(&mc));
            apply_fill_fractions(&mut m.data, &ffc);
            m
        };

        let mut h = AmrHierarchy2D::new(bg, mk_coarse(), ratio, ghost);
        h.set_geom_shape(disk.clone());

        unsafe { std::env::set_var("LLG_AMR_DEMAG_MODE", "composite"); }
        let mut stepper = AmrStepperRK4::new(&h, true);
        unsafe { std::env::remove_var("LLG_AMR_DEMAG_MODE"); }

        let lm = FieldMask::ExchAnisDmi;  // demag handled by composite solver via b_add
        let mut prev_rects: Vec<Rect2i> = Vec::new();
        if amr_lvl > 0 {
            if let Some((r,_)) = maybe_regrid_nested_levels(&mut h, &prev_rects, rp, cp) {
                prev_rects = r;
            }
        }

        let t0 = Instant::now();
        let mut stepper_calls = 0_usize;
        let mut converged_step = relax_steps;
        let mut prev_mz_avg = coarse_mz_avg(&h);

        for step in 1..=relax_steps {
            if step % scr == 0 {
                stepper.step(&mut h, &llg, &mat, lm);
                stepper_calls += 1;
            }
            if amr_lvl > 0 && step % scr == 0 && step % regrid_every == 0 {
                if let Some((r,_)) = maybe_regrid_nested_levels(&mut h, &prev_rects, rp, cp) {
                    prev_rects = r;
                }
            }

            // ── Early-stop: check |Δ<mz>| on coarse grid (cheap) ──
            // Don't check until at least 3000 steps (~60ps) to avoid premature trigger
            // on a vortex state where <mz>≈0 barely changes early on.
            let min_steps_before_conv = 3000_usize;
            if conv_tol > 0.0 && step >= min_steps_before_conv && step % scr == 0 && stepper_calls % conv_check_interval == 0 && stepper_calls > 0 {
                let mz_now = coarse_mz_avg(&h);
                let delta = (mz_now - prev_mz_avg).abs();
                if delta < conv_tol {
                    println!("  ✓ converged at step {step} ({:.0}ps, {} calls): |Δ<mz>|={delta:.2e} < {conv_tol:.0e}",
                             step as f64*dt*1e12, stepper_calls);
                    converged_step = step;
                    break;
                }
                prev_mz_avg = mz_now;
            }

            // ── Progress report ──
            if step % report_every == 0 {
                let mfl = flatten_to(&h, fg);
                let (cx,cy,cmz) = find_core(&mfl, Some(&mf));
                let mut mzs = 0.0_f64; let mut cnt = 0usize;
                for (i,&v) in mfl.data.iter().enumerate() { if mf[i] { mzs+=v[2]; cnt+=1; } }
                let l1 = h.patches.len();
                let l2: usize = h.patches_l2plus.iter().map(|v| v.len()).sum();
                let elapsed = t0.elapsed().as_secs_f64();
                let frac = step as f64 / relax_steps as f64;
                let eta_s = if frac > 0.01 { elapsed / frac - elapsed } else { f64::NAN };
                println!("  {step}/{relax_steps} ({:.0}ps) <mz>={:.4} core=({:.1},{:.1})nm mz={cmz:.4} L1={l1} L2={l2}  [{:.0}s, ETA {:.0}s]",
                         step as f64*dt*1e12, mzs/cnt.max(1) as f64, cx*1e9, cy*1e9, elapsed, eta_s);
            }
        }
        let wall_s = t0.elapsed().as_secs_f64();

        let mfl = flatten_to(&h, fg);
        let q = topological_charge(&mfl, Some(&mf));
        let (core_x,core_y,mz_core) = find_core(&mfl, Some(&mf));
        let mut mzs = 0.0_f64; let mut cnt = 0usize;
        for (i,&v) in mfl.data.iter().enumerate() { if mf[i] { mzs+=v[2]; cnt+=1; } }
        let mz_avg = mzs / cnt.max(1) as f64;

        println!("  ➤ Q = {q:.3}  <mz>={mz_avg:.4}  core_mz={mz_core:.4}  ({wall_s:.1}s = {:.1}min, {} stepper calls)",
                 wall_s / 60.0, stepper_calls);

        // Save fine magnetisation
        { let path = format!("{out}/m_fine_D{}.csv", format!("{:.1}",d_mjm2).replace('.', "p"));
          let mut f = File::create(&path)?;
          writeln!(f, "i,j,x_nm,y_nm,mx,my,mz")?;
          let cnx = fg.nx; let cx = cnx as f64*0.5; let cy = fg.ny as f64*0.5;
          for j in 0..fg.ny { for i in 0..fg.nx {
              let idx = mfl.idx(i,j);
              if !mf[idx] { continue; }
              let [mx,my,mz] = mfl.data[idx];
              writeln!(f, "{i},{j},{:.2},{:.2},{mx:.6},{my:.6},{mz:.6}",
                       (i as f64+0.5-cx)*fg.dx*1e9, (j as f64+0.5-cy)*fg.dy*1e9)?;
          }}
        }
        // Save patches
        { let path = format!("{out}/patches_D{}.csv", format!("{:.1}",d_mjm2).replace('.', "p"));
          let mut f = File::create(&path)?;
          writeln!(f, "level,i0,j0,nx,ny")?;
          for p in &h.patches { let r=p.coarse_rect; writeln!(f, "1,{},{},{},{}", r.i0,r.j0,r.nx,r.ny)?; }
          for (li,lvl) in h.patches_l2plus.iter().enumerate() {
              for p in lvl { let r=p.coarse_rect; writeln!(f, "{},{},{},{},{}", li+2,r.i0,r.j0,r.nx,r.ny)?; }
          }
        }

        let l1=h.patches.len();
        let l2:usize = h.patches_l2plus.iter().map(|v|v.len()).sum();
        append(&summary_path, &format!("{d_mjm2:.2},{q:.4},{mz_avg:.6},{mz_core:.6},{:.2},{:.2},{l1},{l2},{wall_s:.1},{converged_step}\n",
                                       core_x*1e9, core_y*1e9));
        println!();
    }

    println!("╔{bar}╗");
    println!("║{:^66}║", "RESULTS");
    println!("╚{bar}╝");
    println!("  Expected (Zhang et al. 2025, Fig 1):");
    println!("    D=0:    Q ≈ +0.5  (vortex)");
    println!("    D~1-2:  Q ≈ -0.5  (DMI-modified vortex, expanded Néel core)");
    println!("    D~2-3:  Q ≈ -1.5  (1-skyrmion vortex — skyrmion in vortex!)");
    println!("  CAVEAT: Zhang et al. used 400nm disk. Our 200nm disk has stronger");
    println!("  confinement — transition D values may shift. If no transition seen,");
    println!("  try DMI_DISK_R=200e-9 (400nm) or DMI_KU=4.0e5 / DMI_KU=5.0e5.");
    println!("  Output: {out}/summary.csv");
    Ok(())
}