// src/bin/bloch_relax.rs
//
// Bloch wall relaxation benchmark (exchange + uniaxial anisotropy + optional B_ext).
// Demag is absent in Rust; MuMax demag will be disabled for fair comparison.
//
// Outputs:
//   out/rust_table_bloch_relax.csv
//
// Run:
//   cargo run --bin bloch_relax

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};

use llg_sim::energy::{EnergyBreakdown, compute_energy};
use llg_sim::grid::Grid2D;
use llg_sim::llg::{RK4Scratch, step_llg_rk4_recompute_field};
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

fn main() -> std::io::Result<()> {
    // --- keep in sync with MuMax script ---
    let nx: usize = 256;
    let ny: usize = 64;
    let dx: f64 = 5e-9;
    let dy: f64 = 5e-9;
    let dz: f64 = 5e-9;

    let ms: f64 = 8.0e5; // A/m
    let a_ex: f64 = 13e-12; // J/m
    let k_u: f64 = 500.0; // J/m^3
    let easy_axis = [0.0, 0.0, 1.0];

    // External field OFF for relaxation
    let b_ext = [0.0, 0.0, 0.0];

    let alpha: f64 = 0.02;
    let dt: f64 = 2e-13; // slightly larger than 1e-13 to keep runtime reasonable
    let t_total: f64 = 20e-9; // 20 ns relaxation
    let out_stride: usize = 50; // write every 50 steps (dt_out = 1e-11 s)
    // -------------------------------------

    let n_steps: usize = (t_total / dt).round() as usize;

    let grid = Grid2D::new(nx, ny, dx, dy, dz);
    let mut m = VectorField2D::new(grid);

    // Bloch wall initial condition: centred in x, uniform in y
    let x0 = 0.5 * nx as f64 * dx;
    let width = 5.0 * dx; // intentionally sharp vs equilibrium, so it relaxes
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
    };

    let mut scratch = RK4Scratch::new(grid);

    create_dir_all("out")?;
    let file = File::create("out/rust_table_bloch_relax.csv")?;
    let mut w = BufWriter::new(file);

    writeln!(w, "t,mx,my,mz,E_total,E_ex,E_an,E_zee,Bx,By,Bz")?;

    // helper avg
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

    // step 0
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

    for step in 1..=n_steps {
        step_llg_rk4_recompute_field(&mut m, &params, &material, &mut scratch);

        if step % out_stride == 0 || step == n_steps {
            let t = (step as f64) * dt;
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

    println!("Wrote out/rust_table_bloch_relax.csv");
    println!(
        "Grid: {}x{}, dt={}, T={}, steps={}",
        nx, ny, dt, t_total, n_steps
    );
    Ok(())
}
