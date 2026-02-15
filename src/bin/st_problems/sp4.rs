// src/bin/st_problems/sp4.rs
//
// Standard Problem 4 (a/b): Permalloy thin film reversal with demag.
// Matches your MuMax scripts:
//
// Nx=200 Ny=50 Nz=1
// dx=500nm/Nx dy=125nm/Ny dz=3nm
// Msat=8e5 A/m, Aex=13e-12 J/m, alpha=0.02
// m = uniform(1,0.1,0); Relax(); then apply Bext and run 1 ns
// table output every 10 ps: t_s, mx, my, mz
//
// Run:
//   cargo run --release --bin st_problems -- sp4 a
//   cargo run --release --bin st_problems -- sp4 b
//
// Can also provide additional arguments: RAYON_NUM_THREADS=8 LLG_DEMAG_TIMING=1 cargo run --release --bin st_problems -- sp4 a
//
// Post-process (Standard Problem 4 comparison: MuMax vs Rust):
//
// MuMax reference data:
//   mumax_outputs/st_problems/sp4/
//     ├── sp4a_out/table.txt
//     └── sp4b_out/table.txt
//
// Rust output data:
//   runs/st_problems/sp4/
//     ├── sp4a_rust/table.csv
//     └── sp4b_rust/table.csv
//
// Generate comparison plots for SP4a and SP4b (two-panel figure):
//
// python3 scripts/compare_sp4.py \
//   --mumax-root mumax_outputs/st_problems/sp4 \
//   --rust-root runs/st_problems/sp4 \
//   --metrics \
//   --out runs/st_problems/sp4/sp4_overlay.png
//
// Optional:
//   --mark-mx-zero
//     Mark first <m_x>=0 crossing time for MuMax (solid) and Rust (dashed).
//   --metrics --metrics-tmin 3e-9 --metrics-tmax 5e-9 \
//     Compute and display metrics insets for the time window [3 ns, 5 ns].
//   --metrics --metrics-interp mumax \
//     Interpolate Rust data to MuMax time points before computing metrics (for a fair comparison).
//
// This produces:
//   out/st_problems/sp4/sp4_overlay.png

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use llg_sim::effective_field::{FieldMask, build_h_eff_masked};
use llg_sim::grid::Grid2D;
use llg_sim::llg::RK23Scratch;
use llg_sim::llg::{RK45Scratch, step_llg_rk45_recompute_field_adaptive};
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::relax::{RelaxSettings, TorqueMetric, relax};
use llg_sim::vec3::{cross, normalize};
use llg_sim::vector_field::VectorField2D;
use llg_sim::ovf::{OvfMeta, write_ovf2_rectangular_text};

fn avg_vec(field: &VectorField2D) -> [f64; 3] {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    let n = field.data.len() as f64;
    for v in &field.data {
        sx += v[0];
        sy += v[1];
        sz += v[2];
    }
    [sx / n, sy / n, sz / n]
}

fn max_torque_inf(
    grid: &Grid2D,
    m: &VectorField2D,
    params: &LLGParams,
    material: &Material,
) -> f64 {
    // Build full effective field at current m
    let mut b_eff = VectorField2D::new(*grid);
    build_h_eff_masked(grid, m, &mut b_eff, params, material, FieldMask::Full);

    // Use |m x B| infinity norm as a simple convergence metric
    let mut maxv = 0.0;
    for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
        let t = cross(*mi, *bi);
        let mag = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        if mag > maxv {
            maxv = mag;
        }
    }
    maxv
}

fn field_for_case(case: char) -> [f64; 3] {
    match case {
        'a' | 'A' => [-24.6e-3, 4.3e-3, 0.0],
        'b' | 'B' => [-35.5e-3, -6.3e-3, 0.0],
        _ => [-24.6e-3, 4.3e-3, 0.0],
    }
}

fn out_dir_for_case(case: char) -> PathBuf {
    match case {
        'a' | 'A' => Path::new("runs")
            .join("st_problems")
            .join("sp4")
            .join("sp4a_rust"),
        'b' | 'B' => Path::new("runs")
            .join("st_problems")
            .join("sp4")
            .join("sp4b_rust"),
        _ => Path::new("runs")
            .join("st_problems")
            .join("sp4")
            .join("sp4a_rust"),
    }
}

fn ovf_name(i: usize) -> String {
    // Match MuMax naming: m0000000.ovf ... m0000010.ovf
    format!("m{:07}.ovf", i)
}

fn write_sp4_ovf_snapshot(
    out_dir: &Path,
    snap_idx: usize,
    t_s: f64,
    grid: &Grid2D,
    m: &VectorField2D,
) -> std::io::Result<()> {
    let path = out_dir.join(ovf_name(snap_idx));
    let meta = OvfMeta::magnetization().with_total_sim_time(t_s);
    write_ovf2_rectangular_text(&path, grid, m, &meta)
}

pub fn run_sp4(case: char) -> std::io::Result<()> {
    // --- match MuMax SP4 script (your working one) ---
    let nx: usize = 200;
    let ny: usize = 50;
    let dx: f64 = 500e-9 / (nx as f64);
    let dy: f64 = 125e-9 / (ny as f64);
    let dz: f64 = 3e-9;

    let ms: f64 = 8.0e5;
    let a_ex: f64 = 13e-12;

    // SP4 has no crystalline anisotropy (Permalloy) -> set ku=0
    let k_u: f64 = 0.0;
    let easy_axis = [0.0, 0.0, 1.0];

    let alpha_run: f64 = 0.02;

    // Output times 
    let dt_out: f64 = 10e-12; // 10 ps
    let t_total: f64 = 1e-9; // 1 ns
    let n_out: usize = (t_total / dt_out).round() as usize; // 100

    // RK45 controller (same style as uniform_film_field_rk45.rs)
    let dt0_run: f64 = 1e-13;
    let max_err: f64 = 1e-5;
    let headroom: f64 = 0.8;
    let dt_min: f64 = dt0_run * 1e-6;
    let dt_max: f64 = dt0_run * 100.0;

    // Relax controller (your relax stepper is RK4 step-doubling)
    let dt_relax0: f64 = 5e-14;
    let relax_dt_min: f64 = 1e-18;
    let relax_dt_max: f64 = 1e-11;
    let relax_torque_tol: f64 = 1e-4; // convergence criterion on |m x B|

    // -----------------------------------------------

    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    let material = Material {
        ms,
        a_ex,
        k_u,
        easy_axis,
        dmi: None,
        demag: true, // SP4 includes demag
    };

    // Magnetisation: uniform(1,0.1,0) (normalized)
    let mut m = VectorField2D::new(grid);
    let m0 = normalize([1.0, 0.1, 0.0]);
    m.set_uniform(m0[0], m0[1], m0[2]);

    // Output directory
    let out_dir = out_dir_for_case(case);
    create_dir_all(&out_dir)?;

    // --- RELAX stage (B_ext = 0), do NOT write table during relax ---
    let mut params_relax = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha: alpha_run, // used in relax RHS prefactor
        dt: dt_relax0,    // initial dt guess for adaptive relax
        b_ext: [0.0, 0.0, 0.0],
    };

    // MuMax-like relax: RK23 + energy->torque phases + tolerance tightening
    let mut scratch_rk23 = RK23Scratch::new(grid);
    let mut relax_settings = RelaxSettings {
        max_err: 1e-5,
        headroom: 0.8,
        dt_min: relax_dt_min,
        dt_max: relax_dt_max,

        // Keep SP4 behaviour exactly the same as before
        phase1_enabled: true,
        phase2_enabled: true,

        energy_stride: 3,
        rel_energy_tol: 1e-12,

        // SP4 convergence criterion on |m x B|
        torque_metric: TorqueMetric::Max,
        torque_threshold: Some(relax_torque_tol),
        torque_check_stride: 1,

        tighten_factor: std::f64::consts::FRAC_1_SQRT_2,
        tighten_floor: 1e-9,
        max_accepted_steps: 2_000_000,

        // Plateau logic is disabled for SP4.
        torque_plateau_checks: 0,

        // Fill any additional / newly-added settings fields with defaults.
        ..Default::default()
    };

    // Log torque before relax (same as before)
    let torque0 = max_torque_inf(&grid, &m, &params_relax, &material);
    println!(
        "SP4{} relax: start |m x B|_inf = {:.3e}, dt0={:.3e}",
        case.to_ascii_uppercase(),
        torque0,
        params_relax.dt
    );

    // Run relax controller (does not advance physical time)
    relax(
        &grid,
        &mut m,
        &mut params_relax,
        &material,
        &mut scratch_rk23,
        FieldMask::Full, // SP4 includes demag -> Full
        &mut relax_settings,
    );

    // Log torque after relax
    let torque1 = max_torque_inf(&grid, &m, &params_relax, &material);
    println!(
        "SP4{} relax: done |m x B|_inf = {:.3e}, final max_err={:.3e}, final dt={:.3e}",
        case.to_ascii_uppercase(),
        torque1,
        relax_settings.max_err,
        params_relax.dt
    );

    // --- RUN stage (reset time to 0, apply B_ext, write table.csv + OVF snapshots) ---
    let b_ext = field_for_case(case);

    let mut params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha: alpha_run,
        dt: dt0_run,
        b_ext,
    };

    let mut scratch_rk45 = RK45Scratch::new(grid);

    let file = File::create(out_dir.join("table.csv"))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "t_s,mx,my,mz")?;

    // Write t=0 AFTER relax (this matches MuMax: table starts at start of Run)
    {
        let [mx, my, mz] = avg_vec(&m);
        writeln!(w, "{:.16e},{:.16e},{:.16e},{:.16e}", 0.0, mx, my, mz)?;
    }

    let tol_time = 1e-18_f64;
    let mut t: f64 = 0.0;

    // MuMax AutoSave(m, 100e-12) equivalent: write m0000000.ovf .. m0000010.ovf
    // at t = 0, 0.1ns, 0.2ns, ..., 1.0ns (11 snapshots total).
    let dt_ovf: f64 = 100e-12; // 100 ps
    let ovf_stride: usize = (dt_ovf / dt_out).round() as usize; // 10
    debug_assert!(ovf_stride > 0);

    // Snapshot index 0 corresponds to t=0 (after relax, at start of Run)
    let mut ovf_idx: usize = 0;
    write_sp4_ovf_snapshot(&out_dir, ovf_idx, 0.0, &grid, &m)?;
    ovf_idx += 1;

    for k in 1..=n_out {
        let t_target = (k as f64) * dt_out;

        while t + tol_time < t_target {
            // Clamp dt so we land exactly on t_target
            let remaining = t_target - t;
            if params.dt > remaining {
                params.dt = remaining;
            }

            let (_eps, accepted, dt_used) = step_llg_rk45_recompute_field_adaptive(
                &mut m,
                &mut params,
                &material,
                &mut scratch_rk45,
                max_err,
                headroom,
                dt_min,
                dt_max,
            );

            if !accepted {
                continue;
            }
            t += dt_used;
        }

        t = t_target;
        let [mx, my, mz] = avg_vec(&m);
        writeln!(w, "{:.16e},{:.16e},{:.16e},{:.16e}", t, mx, my, mz)?;

        // Write OVF snapshot every 100 ps (i.e. every 10 table samples)
        if k % ovf_stride == 0 {
            write_sp4_ovf_snapshot(&out_dir, ovf_idx, t, &grid, &m)?;
            ovf_idx += 1;
        }
    }

    println!(
        "SP4{} wrote {:?} and {} OVF snapshots",
        case.to_ascii_uppercase(),
        out_dir.join("table.csv"),
        ovf_idx
    );
    Ok(())
}
