// src/main.rs

use std::fs::File;
use std::io::{BufWriter, Write};

use llg_sim::grid::Grid2D;
use llg_sim::vector_field::VectorField2D;
use llg_sim::llg::step_llg_with_field;
use llg_sim::params::LLGParams;
use llg_sim::effective_field::zeeman::add_zeeman_field;

fn main() -> std::io::Result<()> {
    // 1. Set up a 2D grid and magnetisation
    let nx = 32;
    let ny = 32;
    let grid = Grid2D::new(nx, ny, 5e-9, 5e-9);  // 32x32 cells, 5 nm spacing
    let mut m = VectorField2D::new(grid);

    // Start along +z
    m.set_uniform(0.0, 0.0, 1.0);

    // 2. Define LLG parameters (field along +x)
    let params = LLGParams {
        gamma: 1.0,
        alpha: 0.1,
        dt: 0.01,
        h_ext: [1.0, 0.0, 0.0],
    };

    // 3. Create an H_eff field on the same grid
    let mut h_eff = VectorField2D::new(grid);

    // 4. Open CSV file for writing (average magnetisation vs time)
    let file = File::create("avg_magnetisation.csv")?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "t,mx_avg,my_avg,mz_avg")?;

    // 5. Time-stepping loop
    let n_steps: i32 = 100;

    for step in 0..=n_steps {
        // Build H_eff for this time step (only Zeeman for now)
        add_zeeman_field(&mut h_eff, params.h_ext);

        // Compute spatially averaged magnetisation over all cells
        let mut sum_mx = 0.0;
        let mut sum_my = 0.0;
        let mut sum_mz = 0.0;
        let n_cells = m.data.len() as f64;

        for cell in &m.data {
            sum_mx += cell[0];
            sum_my += cell[1];
            sum_mz += cell[2];
        }

        let mx_avg = sum_mx / n_cells;
        let my_avg = sum_my / n_cells;
        let mz_avg = sum_mz / n_cells;

        let t: f64 = step as f64 * params.dt;
        writeln!(writer, "{:.8},{:.8},{:.8},{:.8}", t, mx_avg, my_avg, mz_avg)?;

        // Advance one time step (skip on last step)
        if step < n_steps {
            step_llg_with_field(&mut m, &h_eff, &params);
        }
    }

    println!("Saved average magnetisation to avg_magnetisation.csv");

    Ok(())
}