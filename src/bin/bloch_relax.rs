// src/bin/bloch_relax.rs
//
// Bloch wall relaxation benchmark (exchange + uniaxial anisotropy).
// Demag is absent in Rust; MuMax demag will be disabled for fair comparison.
//
// Outputs:
//   out/bloch_relax/
//     ├── config.json
//     ├── rust_table_bloch_relax.csv
//     └── bloch_slices/
//         ├── rust_slice_t0.csv
//         ├── rust_slice_t5ns.csv
//         ├── rust_slice_t10ns.csv
//         └── rust_slice_final.csv
//
// Run:
//   cargo run --bin bloch_relax

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

fn write_midrow_slice(
    m: &VectorField2D,
    grid: &Grid2D,
    filename: &Path,
) -> std::io::Result<()> {
    let j = grid.ny / 2;
    let mut f = BufWriter::new(File::create(filename)?);
    writeln!(f, "x,mx,mz")?;
    for i in 0..grid.nx {
        let idx = j * grid.nx + i;
        let x = (i as f64 + 0.5) * grid.dx;
        let v = m.data[idx];
        writeln!(f, "{:.6e},{:.6e},{:.6e}", x, v[0], v[2])?;
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

    let ms: f64 = 8.0e5;      // A/m
    let a_ex: f64 = 13e-12;   // J/m
    let k_u: f64 = 500.0;     // J/m^3
    let easy_axis = [0.0, 0.0, 1.0];

    let b_ext = [0.0, 0.0, 0.0];

    let alpha: f64 = 0.5;
    let dt: f64 = 2e-13;
    let t_total: f64 = 20e-9;
    let out_stride: usize = 50;
    // -------------------------------------

    let n_steps: usize = (t_total / dt).round() as usize;

    let grid = Grid2D::new(nx, ny, dx, dy, dz);
    let mut m = VectorField2D::new(grid);

    // Bloch wall initial condition
    let x0 = 0.5 * nx as f64 * dx;
    let width = 60.0 * dx;
    m.init_bloch_wall(x0, width);

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
        dmi: Some(5e-4),
    };

    let mut scratch = RK4Scratch::new(grid);

    // -------------------------------------------------
    // Output directories
    // -------------------------------------------------
    let out_dir = Path::new("out").join("bloch_relax");
    let slices_dir = out_dir.join("bloch_slices");

    create_dir_all(&slices_dir)?;

    // -------------------------------------------------
    // Write config.json
    // -------------------------------------------------
    let run_config = RunConfig {
        geometry: GeometryConfig {
            nx,
            ny,
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
            dmi: material.dmi,
        },
        numerics: NumericsConfig {
            integrator: "rk4_recompute".to_string(),
            dt,
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

    // -------------------------------------------------
    // Open output table
    // -------------------------------------------------
    let file = File::create(out_dir.join("rust_table_bloch_relax.csv"))?;
    let mut w = BufWriter::new(file);

    writeln!(w, "t,mx,my,mz,E_total,E_ex,E_an,E_zee,Bx,By,Bz")?;

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

    // t = 0
    {
        let t = 0.0;
        let [mx, my, mz] = avg_m(&m);
        let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.b_ext);
        writeln!(
            w,
            "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            t,
            mx,
            my,
            mz,
            e.total(),
            e.exchange,
            e.anisotropy,
            e.zeeman,
            params.b_ext[0],
            params.b_ext[1],
            params.b_ext[2],
        )?;
    }

    write_midrow_slice(&m, &grid, &slices_dir.join("rust_slice_t0.csv"))?;

    for step in 1..=n_steps {
        step_llg_rk4_recompute_field(&mut m, &params, &material, &mut scratch);

        let t = (step as f64) * dt;

        if (t - 5.0e-9).abs() < 0.5 * dt {
            write_midrow_slice(&m, &grid, &slices_dir.join("rust_slice_t5ns.csv"))?;
        }
        if (t - 1.0e-8).abs() < 0.5 * dt {
            write_midrow_slice(&m, &grid, &slices_dir.join("rust_slice_t10ns.csv"))?;
        }

        if step % out_stride == 0 || step == n_steps {
            let [mx, my, mz] = avg_m(&m);
            let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.b_ext);
            writeln!(
                w,
                "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
                t,
                mx,
                my,
                mz,
                e.total(),
                e.exchange,
                e.anisotropy,
                e.zeeman,
                params.b_ext[0],
                params.b_ext[1],
                params.b_ext[2],
            )?;
        }
    }

    write_midrow_slice(&m, &grid, &slices_dir.join("rust_slice_final.csv"))?;

    println!("Wrote outputs to {:?}", out_dir);
    Ok(())
}