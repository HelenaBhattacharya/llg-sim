// src/main.rs

use std::fs::File;
use std::io::{BufWriter, Write};

use llg_sim::grid::Grid2D;
use llg_sim::vector_field::VectorField2D;
use llg_sim::llg::step_llg;
use llg_sim::params::LLGParams;

fn main() -> std::io::Result<()> {
    // 1. Set up grid and magnetisation
    let grid = Grid2D::new(1, 1, 1e-9, 1e-9);  // 1x1: macrospin for now
    let mut m = VectorField2D::new(grid);

    // Start along +z
    m.set_uniform(0.0, 0.0, 1.0);

    // 2. Define LLG parameters (field along +x)
    let params = LLGParams {
        gamma: 1.0,                 // scaled gyromagnetic ratio
        alpha: 0.1,                 // damping
        dt: 0.01,                   // time step (dimensionless for now)
        h_ext: [1.0, 0.0, 0.0],     // external field along +x
    };

    // 3. Open CSV file for writing
    let file = File::create("macrospin.csv")?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "t,mx,my,mz")?;

    // 4. Time-stepping loop
    let n_steps = 100;

    for step in 0..=n_steps {
        // read magnetisation of the single cell
        let m_vec = m.data[0];
        let t = step as f64 * params.dt;

        // write one line to CSV
        writeln!(writer, "{:.8},{:.8},{:.8},{:.8}", t, m_vec[0], m_vec[1], m_vec[2])?;

        // advance one time step (skip on last step)
        if step < n_steps {
            step_llg(&mut m, &params);
        }
    }

    // Optionally also print a small confirmation to stdout
    println!("Saved macrospin trajectory to macrospin.csv");

    Ok(())
}