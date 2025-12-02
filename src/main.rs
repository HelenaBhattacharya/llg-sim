// src/main.rs

use llg_sim::grid::Grid2D;
use llg_sim::vector_field::VectorField2D;
use llg_sim::llg::{LLGParams, step_llg};

fn main() {
    // 1. Set up grid and magnetisation
    let grid = Grid2D::new(1, 1, 1e-9, 1e-9);  // 1x1: macrospin for now
    let mut m = VectorField2D::new(grid);

    // // Start slightly tilted away from +z so it precesses
    // m.set_uniform(0.1, 0.0, (1.0 - 0.1_f64.powi(2)).sqrt());

    // // 2. Define LLG parameters
    // let params = LLGParams {
    //     gamma: 1.0,                 // scaled gyromagnetic ratio
    //     alpha: 0.1,                 // some damping
    //     dt: 0.01,                   // small time step (dimensionless for now)
    //     h_ext: [0.0, 0.0, 1.0],     // external field along +z
    // };

    // Start along +z
    m.set_uniform(0.0, 0.0, 1.0);

    // LLG parameters: field along +x
    let params = LLGParams {
        gamma: 1.0,
        alpha: 0.1,
        dt: 0.01,
        h_ext: [1.0, 0.0, 0.0],   // external field along +x
    };

    // 3. Time-stepping loop
    let n_steps = 100;
    println!("t, mx, my, mz");
    for step in 0..=n_steps {
        // read magnetisation of the single cell
        let m_vec = m.data[0];
        let t = step as f64 * params.dt;
        println!("{:.5}, {:.6}, {:.6}, {:.6}", t, m_vec[0], m_vec[1], m_vec[2]);

        // advance one time step (skip on last step)
        if step < n_steps {
            step_llg(&mut m, &params);
        }
    }
}