// src/bin/macrospin_fmr.rs
//
// Macrospin “FMR ringdown” benchmark: 1 cell, uniform B_ext along +z,
// small tilt initial condition. Writes a MuMax-like table CSV.
//
// Run:
//   cargo run --bin macrospin_fmr
//
// Output:
//   out/macrospin_fmr/
//     ├── config.json
//     └── rust_table_macrospin_fmr.csv

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;

use llg_sim::energy::{EnergyBreakdown, compute_energy};
use llg_sim::grid::Grid2D;
use llg_sim::llg::step_llg_with_field_rk4;
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

use llg_sim::config::{
    RunConfig,
    GeometryConfig,
    MaterialConfig,
    FieldConfig,
    NumericsConfig,
    RunInfo,
};

fn main() -> std::io::Result<()> {
    // --- benchmark parameters (keep in sync with MuMax script) ---
    let dx = 5e-9;
    let dy = 5e-9;
    let dz = 5e-9;

    let b0 = 1.0_f64;         // Tesla
    let alpha = 0.01_f64;
    let dt = 5e-14_f64;      // seconds
    let t_total = 5e-9_f64;  // 5 ns
    let theta_deg = 5.0_f64; // initial tilt angle

    let ms = 8.0e5_f64;      // A/m
    let a_ex = 0.0_f64;      // J/m (OFF for macrospin)
    let k_u = 0.0_f64;       // J/m^3 (OFF for macrospin)
    // ------------------------------------------------------------

    let n_steps: usize = (t_total / dt).round() as usize;

    let grid = Grid2D::new(1, 1, dx, dy, dz);

    // Magnetisation and effective field
    let mut m = VectorField2D::new(grid);
    let mut b_eff = VectorField2D::new(grid);

    // Initial tilt from +z in the x–z plane
    let theta = theta_deg.to_radians();
    m.set_uniform(theta.sin(), 0.0, theta.cos());

    // Constant B_ext along +z
    b_eff.set_uniform(0.0, 0.0, b0);

    let params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha,
        dt,
        b_ext: [0.0, 0.0, b0],
    };

    let material = Material {
        ms,
        a_ex,
        k_u,
        easy_axis: [0.0, 0.0, 1.0],
    };

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
            demag: false,
            dmi: None,
        },
        numerics: NumericsConfig {
            integrator: "rk4".to_string(),
            dt,
            steps: n_steps,
            output_stride: 1,
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

    // MuMax-like table columns
    writeln!(w, "t,mx,my,mz,E_total,E_ex,E_an,E_zee,Bx,By,Bz")?;

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

    for step in 1..=n_steps {
        step_llg_with_field_rk4(&mut m, &b_eff, &params);

        let t = (step as f64) * dt;
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

    // Helpful printout
    let f_expected = (params.gamma * b0) / (2.0 * std::f64::consts::PI);
    println!("Wrote outputs to {:?}", out_dir);
    println!("Expected precession frequency ~ {:.3e} Hz", f_expected);

    Ok(())
}