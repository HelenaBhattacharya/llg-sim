// src/bin/macrospin_anisotropy.rs
//
// Macrospin with uniaxial anisotropy (Ku != 0), no exchange, no demag.
// Uses RK4 with field recomputation at substeps (required because B_ani depends on m).
//
// Run:
//   cargo run --bin macrospin_anisotropy
//
// Output:
//   out/macrospin_anisotropy/
//     ├── config.json
//     └── rust_table_macrospin_anisotropy.csv

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;

use llg_sim::energy::{EnergyBreakdown, compute_energy};
use llg_sim::grid::Grid2D;
use llg_sim::llg::{RK4Scratch, step_llg_rk4_recompute_field};
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

    let alpha = 0.02_f64;
    let dt = 1e-12_f64;        // seconds
    let t_total = 100e-9_f64;  // 100 ns
    let theta_deg = 20.0_f64;  // initial tilt angle (from +z in x–z plane)

    let ms = 8.0e5_f64;        // A/m
    let a_ex = 0.0_f64;        // J/m (OFF)
    let k_u = 500.0_f64;       // J/m^3 (ON)
    let easy_axis = [0.0, 0.0, 1.0];

    // External field OFF for this test (anisotropy-only dynamics)
    let b_ext = [0.0_f64, 0.0_f64, 0.0_f64];
    // ------------------------------------------------------------

    let n_steps: usize = (t_total / dt).round() as usize;

    let grid = Grid2D::new(1, 1, dx, dy, dz);

    let mut m = VectorField2D::new(grid);

    // Initial tilt from +z in the x–z plane
    let theta = theta_deg.to_radians();
    m.set_uniform(theta.sin(), 0.0, theta.cos());

    let params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha,
        dt,
        b_ext,
    };

    let material = Material {
        ms,
        a_ex,
        k_u,
        easy_axis,
        dmi: None,
    };

    let mut scratch = RK4Scratch::new(grid);

    // -------------------------------------------------
    // Output directory
    // -------------------------------------------------
    let out_dir = Path::new("out").join("macrospin_anisotropy");
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
            easy_axis,
        },
        fields: FieldConfig {
            b_ext,
            demag: false,
            dmi: None,
        },
        numerics: NumericsConfig {
            integrator: "rk4_recompute".to_string(),
            dt,
            steps: n_steps,
            output_stride: 1,
        },
        run: RunInfo {
            binary: "macrospin_anisotropy".to_string(),
            run_id: "macrospin_anisotropy".to_string(),
            git_commit: None,
            timestamp_utc: None,
        },
    };

    run_config.write_to_dir(&out_dir)?;

    // -------------------------------------------------
    // Open output table
    // -------------------------------------------------
    let file = File::create(out_dir.join("rust_table_macrospin_anisotropy.csv"))?;
    let mut w = BufWriter::new(file);

    writeln!(w, "t,mx,my,mz,E_total,E_ex,E_an,E_zee,Bx,By,Bz")?;

    // t = 0 row
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
        step_llg_rk4_recompute_field(&mut m, &params, &material, &mut scratch);

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

    println!("Wrote outputs to {:?}", out_dir);
    Ok(())
}