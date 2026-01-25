// src/bin/bloch_relax.rs
//
// Two-stage relax:
//  Stage 1: Exch+Anis (fixed dt)
//  Stage 2: Exch+Anis+DMI (energy-backtracking line search on dt)
//
// Outputs: out/bloch_relax/...

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;

use llg_sim::config::{
    FieldConfig, GeometryConfig, MaterialConfig, NumericsConfig, RunConfig, RunInfo,
};
use llg_sim::energy::{compute_energy, EnergyBreakdown};
use llg_sim::grid::Grid2D;
use llg_sim::llg::{RK4Scratch, step_llg_rk4_recompute_field_masked_relax};
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

use llg_sim::effective_field::FieldMask;

fn write_midrow_slice(m: &VectorField2D, grid: &Grid2D, filename: &Path) -> std::io::Result<()> {
    let j = grid.ny / 2;
    let mut f = BufWriter::new(File::create(filename)?);

    writeln!(f, "x,mx,my,mz")?;
    for i in 0..grid.nx {
        let idx = j * grid.nx + i;
        let x = (i as f64 + 0.5) * grid.dx;
        let v = m.data[idx];
        writeln!(f, "{:.6e},{:.6e},{:.6e},{:.6e}", x, v[0], v[1], v[2])?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    // --- keep in sync with MuMax script ---
    let nx: usize = 256;
    let ny: usize = 64;
    let dx: f64 = 5e-9;
    let dy: f64 = 5e-9;
    let dz: f64 = 5e-9;

    let ms: f64 = 8.0e5;
    let a_ex: f64 = 13e-12;
    let k_u: f64 = 500.0;
    let easy_axis = [0.0, 0.0, 1.0];

    let b_ext = [0.0, 0.0, 0.0];

    // High damping for relaxation
    let alpha: f64 = 0.5;

    // Stage 1 fixed dt, Stage 2 starts at same dt but backtracks if needed
    let dt0: f64 = 2e-13;
    let t_total: f64 = 20e-9;

    // Output cadence (outer steps)
    let out_stride: usize = 50;
    // -------------------------------------

    let n_steps: usize = (t_total / dt0).round() as usize;
    let stage1_steps: usize = n_steps / 2;

    let grid = Grid2D::new(nx, ny, dx, dy, dz);
    let mut m = VectorField2D::new(grid);

    // Smoothed wall IC
    let x0 = 0.5 * nx as f64 * dx;
    let width_cells: f64 = 40.0;
    m.init_bloch_wall_y(x0, width_cells * dx, 1.0);

    let mut params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha,
        dt: dt0,
        b_ext,
    };

    let dmi_strength: f64 = -1e-4;
    let material = Material {
        ms,
        a_ex,
        k_u,
        easy_axis,
        dmi: Some(dmi_strength),
    };

    let mut scratch = RK4Scratch::new(grid);
    let mut m_backup = VectorField2D::new(grid);

    // Output dirs
    let out_dir = Path::new("out").join("bloch_relax");
    let slices_dir = out_dir.join("bloch_slices");
    create_dir_all(&slices_dir)?;

    // config.json
    let run_config = RunConfig {
        geometry: GeometryConfig { nx, ny, nz: 1, dx, dy, dz },
        material: MaterialConfig { ms, aex: a_ex, ku1: k_u, easy_axis },
        fields: FieldConfig { b_ext, demag: false, dmi: material.dmi },
        numerics: NumericsConfig {
            integrator: "rk4relax_masked_staged_energy_backtracking".to_string(),
            dt: dt0,
            steps: n_steps,
            output_stride: out_stride,
        },
        run: RunInfo {
            binary: "bloch_relax".to_string(),
            run_id: "bloch_relax".to_string(),
            git_commit: None,
            timestamp_utc: None,
        },
    };
    run_config.write_to_dir(&out_dir)?;

    // Output table
    let file = File::create(out_dir.join("rust_table_bloch_relax.csv"))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "t,mx,my,mz,E_total,E_ex,E_an,E_zee,E_dmi,Bx,By,Bz")?;

    let avg_m = |field: &VectorField2D| -> [f64; 3] {
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
    };

    // t=0 row + slice
    {
        let [mx, my, mz] = avg_m(&m);
        let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.b_ext);
        writeln!(
            w,
            "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            0.0, mx, my, mz,
            e.total(), e.exchange, e.anisotropy, e.zeeman, e.dmi,
            params.b_ext[0], params.b_ext[1], params.b_ext[2],
        )?;
    }
    write_midrow_slice(&m, &grid, &slices_dir.join("rust_slice_t0.csv"))?;

    // Time accumulator (real time, since dt changes in Stage 2)
    let mut t: f64 = 0.0;

    // Stage 2 backtracking settings
    let dt_min: f64 = 2e-14;
    let dt_max: f64 = dt0;
    let rel_e_tol: f64 = 1e-5; // allow tiny numerical noise
    let grow: f64 = 1.05;      // gentle growth when successful
    let shrink: f64 = 0.5;     // backtrack factor

    for step in 1..=n_steps {
        let mask = if step <= stage1_steps {
            FieldMask::ExchAnis
        } else {
            FieldMask::ExchAnisDmi
        };

        if step <= stage1_steps {
            // Stage 1: fixed dt
            params.dt = dt0;
            step_llg_rk4_recompute_field_masked_relax(&mut m, &params, &material, &mut scratch, mask);
            t += params.dt;
        } else {
            // Stage 2: energy-backtracking line search on dt
            let e_old = compute_energy(&grid, &m, &material, params.b_ext).total();
            let e_tol = rel_e_tol * e_old.abs().max(1e-30);

            // Start with current dt (clamped)
            let mut dt_try = params.dt.clamp(dt_min, dt_max);

            // Try a few backtracks; this always terminates.
            for _attempt in 0..30 {
                // backup
                m_backup.data.clone_from(&m.data);
                params.dt = dt_try;

                step_llg_rk4_recompute_field_masked_relax(&mut m, &params, &material, &mut scratch, mask);

                let e_new = compute_energy(&grid, &m, &material, params.b_ext).total();

                if e_new <= e_old + e_tol {
                    // accept
                    t += dt_try;
                    // gentle grow for next step
                    params.dt = (dt_try * grow).min(dt_max);
                    break;
                } else {
                    // reject: restore and shrink
                    m.data.clone_from(&m_backup.data);
                    dt_try *= shrink;
                    if dt_try <= dt_min {
                        // force tiny step and accept to ensure progress
                        params.dt = dt_min;
                        step_llg_rk4_recompute_field_masked_relax(&mut m, &params, &material, &mut scratch, mask);
                        t += dt_min;
                        break;
                    }
                }
            }
        }

        // Save slices near target times
        if (t - 5.0e-9).abs() < 0.5 * dt0 {
            write_midrow_slice(&m, &grid, &slices_dir.join("rust_slice_t5ns.csv"))?;
        }
        if (t - 1.0e-8).abs() < 0.5 * dt0 {
            write_midrow_slice(&m, &grid, &slices_dir.join("rust_slice_t10ns.csv"))?;
        }

        // Output table on cadence
        if step % out_stride == 0 || step == n_steps {
            let [mx, my, mz] = avg_m(&m);
            let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.b_ext);
            writeln!(
                w,
                "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
                t, mx, my, mz,
                e.total(), e.exchange, e.anisotropy, e.zeeman, e.dmi,
                params.b_ext[0], params.b_ext[1], params.b_ext[2],
            )?;
        }
    }

    write_midrow_slice(&m, &grid, &slices_dir.join("rust_slice_final.csv"))?;
    println!("Wrote outputs to {:?}", out_dir);
    Ok(())
}