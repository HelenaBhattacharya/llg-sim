// src/bin/macrospin_fmr.rs
//
// Macrospin “FMR ringdown” benchmark: 1 cell, uniform B_ext along +z,
// small tilt initial condition. Writes a MuMax-like table CSV.
//
// This version uses adaptive RK45 (Dormand–Prince) but outputs at UNIFORM physical times,
// so comparison overlays against MuMax are straightforward.
//
// Run:
//   cargo run --release --bin macrospin_fmr
//
// Post-process (MuMax overlay + FFT):
//
//   python3 scripts/overlay_macrospin.py \
//     out/macrospin_fmr/rust_table_macrospin_fmr.csv \
//     mumax_outputs/macrospin_fmr/table.txt \
//     --col my --clip_overlap --metrics --do_fft \
//     --out out/macrospin_fmr/overlay_my_vs_time.png

// Output:
//   out/macrospin_fmr/
//     ├── config.json
//     ├── rust_table_macrospin_fmr.csv
//     └── dt_history.csv

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;

use llg_sim::config::{
    FieldConfig, GeometryConfig, MaterialConfig, NumericsConfig, RunConfig, RunInfo,
};
use llg_sim::energy::{EnergyBreakdown, compute_energy};
use llg_sim::grid::Grid2D;
use llg_sim::llg::{RK45Scratch, step_llg_rk45_recompute_field_adaptive};
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

fn main() -> std::io::Result<()> {
    // --- benchmark parameters (keep in sync with MuMax script) ---
    let dx = 5e-9;
    let dy = 5e-9;
    let dz = 5e-9;

    let b0 = 1.0_f64; // Tesla
    let alpha = 0.01_f64;
    let dt0 = 5e-14_f64; // initial dt guess (seconds)
    let t_total = 5e-9_f64; // 5 ns
    let theta_deg = 5.0_f64; // initial tilt angle

    let ms = 8.0e5_f64; // A/m
    let a_ex = 0.0_f64; // J/m (OFF for macrospin)
    let k_u = 0.0_f64; // J/m^3 (OFF for macrospin)

    // RK45 controller (MuMax-like defaults)
    let max_err = 1e-5_f64;
    let headroom = 0.8_f64;

    // Clamp dt so it can't blow up or shrink forever
    let dt_min = dt0 * 1e-6;
    let dt_max = dt0 * 100.0;

    // Output sampling period (UNIFORM physical time grid)
    let dt_out = dt0;
    // ------------------------------------------------------------

    let n_steps: usize = (t_total / dt_out).round() as usize;

    let grid = Grid2D::new(1, 1, dx, dy, dz);

    // Magnetisation
    let mut m = VectorField2D::new(grid);

    // Initial tilt from +z in the x–z plane
    let theta = theta_deg.to_radians();
    m.set_uniform(theta.sin(), 0.0, theta.cos());

    let mut params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha,
        dt: dt0, // RK45 will adapt this
        b_ext: [0.0, 0.0, b0],
    };

    let material = Material {
        ms,
        a_ex,
        k_u,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: false,
    };

    let mut scratch = RK45Scratch::new(grid);

    // -------------------------------------------------
    // Output directory
    // -------------------------------------------------
    let out_dir = Path::new("out").join("macrospin_fmr");
    create_dir_all(&out_dir)?;

    // -------------------------------------------------
    // Write config.json
    // -------------------------------------------------
    let run_config = RunConfig {
        geometry: GeometryConfig {
            nx: 1,
            ny: 1,
            nz: 1,
            dx,
            dy,
            dz,
        },
        material: MaterialConfig {
            ms,
            aex: a_ex,
            ku1: k_u,
            easy_axis: [0.0, 0.0, 1.0],
        },
        fields: FieldConfig {
            b_ext: [0.0, 0.0, b0],
            demag: material.demag,
            dmi: material.dmi,
        },
        numerics: NumericsConfig {
            integrator: "rk45".to_string(),
            dt: dt0,
            steps: n_steps,
            output_stride: 1,
            max_err: Some(max_err),
            headroom: Some(headroom),
            dt_min: Some(dt_min),
            dt_max: Some(dt_max),
        },
        run: RunInfo {
            binary: "macrospin_fmr".to_string(),
            run_id: "macrospin_fmr".to_string(),
            git_commit: None,
            timestamp_utc: None,
        },
    };
    run_config.write_to_dir(&out_dir)?;

    // -------------------------------------------------
    // Open output table
    // -------------------------------------------------
    let file = File::create(out_dir.join("rust_table_macrospin_fmr.csv"))?;
    let mut w = BufWriter::new(file);
    writeln!(w, "t,mx,my,mz,E_total,E_ex,E_an,E_zee,Bx,By,Bz")?;

    // Also write dt/eps history (debugging)
    let file_dt = File::create(out_dir.join("dt_history.csv"))?;
    let mut wdt = BufWriter::new(file_dt);
    writeln!(wdt, "attempt,t,dt_used,eps,accepted")?;

    // Write t = 0 row
    {
        let v = m.data[0];
        let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.b_ext);
        writeln!(
            w,
            "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            0.0,
            v[0],
            v[1],
            v[2],
            e.total(),
            e.exchange,
            e.anisotropy,
            e.zeeman,
            params.b_ext[0],
            params.b_ext[1],
            params.b_ext[2],
        )?;
    }

    // -------------------------------------------------
    // Integrate with RK45 but output at uniform physical times
    // -------------------------------------------------
    let tol_time = 1e-18_f64;

    let mut t: f64 = 0.0;
    let mut attempt: usize = 0;

    for k in 1..=n_steps {
        let t_target = (k as f64) * dt_out;

        // Take as many adaptive steps as needed to reach t_target
        while t + tol_time < t_target {
            attempt += 1;

            // Clamp dt so the accepted step can land exactly on the next output time.
            let remaining = t_target - t;
            if params.dt > remaining {
                params.dt = remaining;
            }

            let (eps, accepted, dt_used) = step_llg_rk45_recompute_field_adaptive(
                &mut m,
                &mut params,
                &material,
                &mut scratch,
                max_err,
                headroom,
                dt_min,
                dt_max,
            );

            writeln!(
                wdt,
                "{},{:.16e},{:.16e},{:.16e},{}",
                attempt,
                t,
                dt_used,
                eps,
                if accepted { 1 } else { 0 }
            )?;

            if !accepted {
                continue;
            }

            t += dt_used;
        }

        // Snap tiny rounding error
        t = t_target;

        // Write output row at exactly t_target
        let v = m.data[0];
        let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.b_ext);

        writeln!(
            w,
            "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            t,
            v[0],
            v[1],
            v[2],
            e.total(),
            e.exchange,
            e.anisotropy,
            e.zeeman,
            params.b_ext[0],
            params.b_ext[1],
            params.b_ext[2],
        )?;
    }

    let f_expected = (params.gamma * b0) / (2.0 * std::f64::consts::PI);
    println!("Wrote outputs to {:?}", out_dir);
    println!("Expected precession frequency ~ {:.3e} Hz", f_expected);
    Ok(())
}
