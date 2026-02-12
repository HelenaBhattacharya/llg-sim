// src/bin/st_problems/fmr.rs
//
// FMR standard problem aligned to Ubermag notebook.
// Saves OVFs:
//  - m_initial.ovf   (t=0, before relax)
//  - m_relaxed.ovf   (t=5ns, after relax under e1, alpha=1)
//  - m_dyn0.ovf      (t=0 of dynamic stage, immediately after switching field to e2)
//  - m_dyn_1ns.ovf   (t=1ns of dynamic stage, optional “early ringdown” spatial state)
//  - m_final.ovf     (t=20ns of dynamic stage)
//
// Writes table.csv (dynamic stage averages at 5 ps): t_s,mx,my,mz
//
// Run:
//   cargo run --release --bin st_problems -- fmr

use std::f64::consts::PI;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use llg_sim::grid::Grid2D;
use llg_sim::llg::{RK45Scratch, step_llg_rk45_recompute_field_adaptive};
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

fn mu0() -> f64 {
    4.0 * PI * 1e-7
}

fn out_dir() -> PathBuf {
    Path::new("runs").join("st_problems").join("fmr").join("fmr_rust")
}

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

fn write_ovf2_text_mumax_like(
    path: &Path,
    t_s: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    m: &VectorField2D,
) -> std::io::Result<()> {
    let xmin = 0.0;
    let ymin = 0.0;
    let zmin = 0.0;
    let xmax = (nx as f64) * dx;
    let ymax = (ny as f64) * dy;
    let zmax = (nz as f64) * dz;

    let xbase = 0.5 * dx;
    let ybase = 0.5 * dy;
    let zbase = 0.5 * dz;

    if m.data.len() != nx * ny {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("VectorField2D length mismatch: got {}, expected {}", m.data.len(), nx * ny),
        ));
    }

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# OOMMF OVF 2.0")?;
    writeln!(w, "# Segment count: 1")?;
    writeln!(w, "# Begin: Segment")?;
    writeln!(w, "# Begin: Header")?;
    writeln!(w, "# Title: m")?;
    writeln!(w, "# meshtype: rectangular")?;
    writeln!(w, "# meshunit: m")?;
    writeln!(w, "# xmin: {:.16e}", xmin)?;
    writeln!(w, "# ymin: {:.16e}", ymin)?;
    writeln!(w, "# zmin: {:.16e}", zmin)?;
    writeln!(w, "# xmax: {:.16e}", xmax)?;
    writeln!(w, "# ymax: {:.16e}", ymax)?;
    writeln!(w, "# zmax: {:.16e}", zmax)?;
    writeln!(w, "# valuedim: 3")?;
    writeln!(w, "# valuelabels: m_x m_y m_z")?;
    writeln!(w, "# valueunits: 1 1 1")?;
    writeln!(w, "# Desc: Total simulation time:  {:.16e}  s", t_s)?;
    writeln!(w, "# xbase: {:.16e}", xbase)?;
    writeln!(w, "# ybase: {:.16e}", ybase)?;
    writeln!(w, "# zbase: {:.16e}", zbase)?;
    writeln!(w, "# xnodes: {}", nx)?;
    writeln!(w, "# ynodes: {}", ny)?;
    writeln!(w, "# znodes: {}", nz)?;
    writeln!(w, "# xstepsize: {:.16e}", dx)?;
    writeln!(w, "# ystepsize: {:.16e}", dy)?;
    writeln!(w, "# zstepsize: {:.16e}", dz)?;
    writeln!(w, "# End: Header")?;

    writeln!(w, "# Begin: Data Text")?;
    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;
            let v = m.data[idx];
            writeln!(w, "{:.10e} {:.10e} {:.10e}", v[0], v[1], v[2])?;
        }
    }
    writeln!(w, "# End: Data Text")?;
    writeln!(w, "# End: Segment")?;
    Ok(())
}

pub fn run_fmr() -> std::io::Result<()> {
    // --- Ubermag spec geometry ---
    let lx = 120e-9;
    let ly = 120e-9;
    let lz = 10e-9;

    // Ubermag cells are 5 nm (nx=24, ny=24, nz=2). Our Grid2D is effectively a single layer,
    // so we represent thickness via dz = 10 nm (closest 2D analogue).
    let dx = 5e-9;
    let dy = 5e-9;
    let dz = lz;

    let nx: usize = (lx as f64 / dx as f64).round() as usize;
    let ny: usize = (ly as f64 / dy as f64).round() as usize;


    // --- material ---
    let ms: f64 = 8.0e5;
    let a_ex: f64 = 1.3e-11;
    let k_u: f64 = 0.0;

    // --- fields ---
    let hmag: f64 = 8.0e4;          // A/m
    let bmag: f64 = mu0() * hmag;   // Tesla

    // Unit vectors as in Ubermag notebook
    let e1 = [0.81345856316858023, 0.58162287266553481, 0.0];
    let e2 = [0.81923192051904048, 0.57346234436332832, 0.0];

    let b1 = [bmag * e1[0], bmag * e1[1], 0.0];
    let b2 = [bmag * e2[0], bmag * e2[1], 0.0];

    // --- stages ---
    let t_relax = 5e-9;
    let t_dyn = 20e-9;
    let dt_out = 5e-12; // 5 ps
    let n_out: usize = (t_dyn as f64 / dt_out as f64).round() as usize;

    // RK45 controller
    let dt0: f64 = 5e-14;
    let max_err: f64 = 1e-5;
    let headroom: f64 = 0.8;
    let dt_min: f64 = dt0 * 1e-6;
    let dt_max: f64 = dt0 * 100.0;

    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    let material = Material {
        ms,
        a_ex,
        k_u,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
    };

    // Initial magnetisation: uniform(0,0,1)
    let mut m = VectorField2D::new(grid);
    m.set_uniform(0.0, 0.0, 1.0);

    let out = out_dir();
    create_dir_all(&out)?;

    // Save initial OVF
    write_ovf2_text_mumax_like(&out.join("m_initial.ovf"), 0.0, nx, ny, 1, dx, dy, dz, &m)?;

    let mut params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha: 1.0,
        dt: dt0,
        b_ext: b1,
    };

    let mut scratch = RK45Scratch::new(grid);
    let tol_time = 1e-18_f64;

    // --- Relax stage: alpha=1, B=b1, 5 ns ---
    let mut t: f64 = 0.0;
    while t + tol_time < t_relax {
        let remaining = t_relax - t;
        if params.dt > remaining {
            params.dt = remaining;
        }
        let (_eps, accepted, dt_used) = step_llg_rk45_recompute_field_adaptive(
            &mut m,
            &mut params,
            &material,
            &mut scratch,
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

    // Save relaxed OVF at t=5 ns
    write_ovf2_text_mumax_like(&out.join("m_relaxed.ovf"), t_relax, nx, ny, 1, dx, dy, dz, &m)?;

    // --- Dynamic stage: alpha=0.008, B=b2, 20 ns ---
    params.alpha = 0.008;
    params.b_ext = b2;
    params.dt = dt0;

    // Save a snapshot right after switching the field (often differs slightly from m_relaxed)
    write_ovf2_text_mumax_like(&out.join("m_dyn0.ovf"), 0.0, nx, ny, 1, dx, dy, dz, &m)?;

    let file = File::create(out.join("table.csv"))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "t_s,mx,my,mz")?;

    // reset dynamic time axis
    t = 0.0;
    {
        let [mx, my, mz] = avg_vec(&m);
        writeln!(w, "{:.16e},{:.16e},{:.16e},{:.16e}", 0.0, mx, my, mz)?;
    }

    let mut saved_1ns = false;

    for k in 1..=n_out {
        let t_target = (k as f64) * dt_out;

        while t + tol_time < t_target {
            let remaining = t_target - t;
            if params.dt > remaining {
                params.dt = remaining;
            }

            let (_eps, accepted, dt_used) = step_llg_rk45_recompute_field_adaptive(
                &mut m,
                &mut params,
                &material,
                &mut scratch,
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

        // Save an early ringdown spatial snapshot at 1 ns (helps show dynamics vs relaxed)
        if !saved_1ns && t >= 1e-9 {
            write_ovf2_text_mumax_like(&out.join("m_dyn_1ns.ovf"), t, nx, ny, 1, dx, dy, dz, &m)?;
            saved_1ns = true;
        }
    }

    // Final state at 20 ns
    write_ovf2_text_mumax_like(&out.join("m_final.ovf"), t_dyn, nx, ny, 1, dx, dy, dz, &m)?;

    println!("Wrote outputs to {:?}", out);
    Ok(())
}