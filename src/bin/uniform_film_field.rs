// src/bin/uniform_film_field.rs
//
// Uniform 2D film benchmark (exchange + anisotropy + uniform external field),
// with demag intentionally absent (to match current Rust physics + MuMax demag-off).
//
// Writes a MuMax-like table CSV with averages + energies.
//
// Run:
//   cargo run --bin uniform_film_field
//
// Output:
//   out/rust_table_uniform_film.csv

use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};

use llg_sim::energy::{compute_energy, EnergyBreakdown};
use llg_sim::grid::Grid2D;
use llg_sim::llg::{RK4Scratch, step_llg_rk4_recompute_field};
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

fn main() -> std::io::Result<()> {
    // --- keep in sync with MuMax script ---
    let nx: usize = 128;
    let ny: usize = 128;
    let dx: f64 = 5e-9;
    let dy: f64 = 5e-9;
    let dz: f64 = 5e-9;

    let ms: f64 = 8.0e5;        // A/m
    let a_ex: f64 = 13e-12;     // J/m
    let k_u: f64 = 500.0;       // J/m^3
    let easy_axis = [0.0, 0.0, 1.0];

    // External field along +x to drive dynamics from an initial +z state
    let b_ext = [0.01, 0.0, 0.0]; // 10 mT

    let alpha: f64 = 0.02;
    let dt: f64 = 1e-13;        // integration dt
    let t_total: f64 = 5e-9;    // 5 ns
    let out_stride: usize = 10; // write every N steps -> dt_out = out_stride*dt
    // -------------------------------------

    let n_steps: usize = (t_total / dt).round() as usize;

    let grid = Grid2D::new(nx, ny, dx, dy, dz);
    let mut m = VectorField2D::new(grid);

    // Initial condition: uniform +z
    m.set_uniform(0.0, 0.0, 1.0);

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
    let file = File::create("out/rust_table_uniform_film.csv")?;
    let mut w = BufWriter::new(file);

    writeln!(w, "t,mx,my,mz,E_total,E_ex,E_an,E_zee,Bx,By,Bz")?;

    // Helper: compute averages
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

    // Write step 0
    {
        let t = 0.0;
        let [mx, my, mz] = avg_m(&m);
        let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.b_ext);
        writeln!(
            w,
            "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            t,
            mx, my, mz,
            e.total(), e.exchange, e.anisotropy, e.zeeman,
            params.b_ext[0], params.b_ext[1], params.b_ext[2],
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
                mx, my, mz,
                e.total(), e.exchange, e.anisotropy, e.zeeman,
                params.b_ext[0], params.b_ext[1], params.b_ext[2],
            )?;
        }
    }

    let dt_out = (out_stride as f64) * dt;
    println!("Wrote out/rust_table_uniform_film.csv");
    println!("Steps: {}, dt={}, T={}", n_steps, dt, t_total);
    println!("Output stride: {} -> dt_out={}", out_stride, dt_out);
    Ok(())
}