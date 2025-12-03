// src/main.rs

use std::env;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::process::Command;

use llg_sim::grid::Grid2D;
use llg_sim::vector_field::VectorField2D;
use llg_sim::llg::step_llg_with_field;
use llg_sim::params::{LLGParams, Material};
use llg_sim::effective_field::build_h_eff;
use llg_sim::visualisation::{save_mz_plot, make_movie_with_ffmpeg};

fn main() -> std::io::Result<()> {
    // 1. Set up a 2D grid and magnetisation
    let nx: usize = 128;
    let ny: usize = 128;
    let dx: f64 = 5e-12;
    let dy: f64 = 5e-12;

    let grid: Grid2D = Grid2D::new(nx, ny, dx, dy);  // 64x64 cells, 5 nm spacing
    let mut m: VectorField2D = VectorField2D::new(grid);
    let mut h_eff: VectorField2D = VectorField2D::new(grid);

    // Print physical domain size
    println!(
        "Domain: {} x {} cells, dx = {:.2} nm, dy = {:.2} nm \
         (Lx = {:.1} nm, Ly = {:.1} nm)",
        nx,
        ny,
        dx * 1e9,
        dy * 1e9,
        nx as f64 * dx * 1e9,
        ny as f64 * dy * 1e9,
    );

    // -------- choose initial condition from command line --------
    // cargo run -- bloch    -> Bloch wall
    // cargo run -- uniform  -> uniform +z
    // cargo run             -> defaults to Bloch wall
    let args: Vec<String> = env::args().collect();
    let init_kind = args.get(1).map(String::as_str).unwrap_or("bloch");

    match init_kind {
        "uniform" => {
            println!("Initial condition: uniform +z");
            m.set_uniform(0.0, 0.0, 1.0);
        }
        "bloch" => {
            println!("Initial condition: Bloch wall");
            let x0 = 0.5 * nx as f64 * dx; // centre of sample
            let width = 5.0 * dx;          // wall width ~ 5 cells
            m.init_bloch_wall(x0, width);
        }
        other => {
            eprintln!(
                "Unknown initial condition '{}', expected 'uniform' or 'bloch'. \
                 Using Bloch wall.",
                other
            );
            let x0 = 0.5 * nx as f64 * dx;
            let width = 5.0 * dx;
            m.init_bloch_wall(x0, width);
        }
    }
    // ------------------------------------------------------------

    // 2. Define LLG parameters
    let params: LLGParams = LLGParams {
        gamma: 1.0,
        alpha: 0.1,
        dt: 0.0025,                 // slightly smaller dt for smoother evolution
        h_ext: [0.0, 0.0, 0.0],    // no external field while looking at the wall
    };

    // Material parameters (placeholder values)
    let material = Material {
        ms: 8.0e5,                 // A/m (placeholder)
        a_ex: 1.0,                 // exchange strength (toy units)
        k_u: 0.1,                  // uniaxial anisotropy strength
        easy_axis: [0.0, 0.0, 1.0],
    };

    // 4. Open CSV file for writing (average magnetisation vs time)
    let file: File = File::create("avg_magnetisation.csv")?;
    let mut writer: BufWriter<File> = BufWriter::new(file);

    // Write header
    writeln!(writer, "t,mx_avg,my_avg,mz_avg")?;

    // Ensure frames directory exists for PNG snapshots
    create_dir_all("frames")?;

    // 5. Time-stepping loop
    let n_steps: i32 = 500;       // more steps for a nicer movie

    for step in 0..=n_steps {
        // Build H_eff for this time step (Zeeman + exchange + anisotropy)
        build_h_eff(&grid, &m, &mut h_eff, &params, &material);

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
        writeln!(
            writer,
            "{:.8},{:.8},{:.8},{:.8}",
            t, mx_avg, my_avg, mz_avg
        )?;

        // Save a plot of m_z every 5 steps for smoother video
        if step % 2 == 0 {
            let filename = format!("frames/mz_{:04}.png", step);
            save_mz_plot(&m, &filename).expect("failed to save m_z plot");
            println!(
                "Saved frame {} (step = {}, t = {:.3}, mz_avg = {:.3})",
                filename, step, t, mz_avg
            );
        }

        // Advance one time step (skip on last step)
        if step < n_steps {
            step_llg_with_field(&mut m, &h_eff, &params);
        }
    }

    // Build a movie from all frames using ffmpeg (if available)
    if let Err(e) = make_movie_with_ffmpeg("frames/mz_*.png", "mz_evolution.mp4", 20) {
        eprintln!("Could not create movie with ffmpeg: {e}");
    } else {
        println!("Saved movie to mz_evolution.mp4");

        // On macOS, open the movie so you can just press “play”
        #[cfg(target_os = "macos")]
        {
            let _ = Command::new("open").arg("mz_evolution.mp4").status();
        }
    }

    println!("Saved average magnetisation to avg_magnetisation.csv and PNG frames in frames/");

    Ok(())
}

// // src/main.rs

// use std::env;
// use std::fs::{File, create_dir_all};
// use std::io::{BufWriter, Write};
// use std::process::Command;

// use llg_sim::grid::Grid2D;
// use llg_sim::vector_field::VectorField2D;
// use llg_sim::llg::step_llg_with_field;
// use llg_sim::params::{LLGParams, Material};
// use llg_sim::effective_field::build_h_eff;
// use llg_sim::visualisation::save_mz_plot;

// fn main() -> std::io::Result<()> {
//     // 1. Set up a 2D grid and magnetisation
//     let nx: usize = 64;
//     let ny: usize = 64;
//     let dx: f64 = 5e-9;
//     let dy: f64 = 5e-9;

//     let grid: Grid2D = Grid2D::new(nx, ny, dx, dy);  // 32x32 cells, 5 nm spacing
//     let mut m: VectorField2D = VectorField2D::new(grid);
//     let mut h_eff: VectorField2D = VectorField2D::new(grid);

//     // Print physical domain size
//     println!(
//         "Domain: {} x {} cells, dx = {:.2} nm, dy = {:.2} nm \
//          (Lx = {:.1} nm, Ly = {:.1} nm)",
//         nx,
//         ny,
//         dx * 1e9,
//         dy * 1e9,
//         nx as f64 * dx * 1e9,
//         ny as f64 * dy * 1e9,
//     );

//     // -------- choose initial condition from command line --------
//     // cargo run -- bloch    -> Bloch wall
//     // cargo run -- uniform  -> uniform +z
//     // cargo run             -> defaults to Bloch wall
//     let args: Vec<String> = env::args().collect();
//     let init_kind = args.get(1).map(String::as_str).unwrap_or("bloch");

//     match init_kind {
//         "uniform" => {
//             println!("Initial condition: uniform +z");
//             m.set_uniform(0.0, 0.0, 1.0);
//         }
//         "bloch" => {
//             println!("Initial condition: Bloch wall");
//             let x0 = 0.5 * nx as f64 * dx; // centre of sample
//             let width = 5.0 * dx;          // wall width ~ 5 cells
//             m.init_bloch_wall(x0, width);
//         }
//         other => {
//             eprintln!(
//                 "Unknown initial condition '{}', expected 'uniform' or 'bloch'. \
//                  Using Bloch wall.",
//                 other
//             );
//             let x0 = 0.5 * nx as f64 * dx;
//             let width = 5.0 * dx;
//             m.init_bloch_wall(x0, width);
//         }
//     }
//     // ------------------------------------------------------------

//     // 2. Define LLG parameters (field along +x)
//     let params: LLGParams = LLGParams {
//         gamma: 1.0,
//         alpha: 0.1,
//         dt: 0.001,
//         h_ext: [1.0, 0.0, 0.0],
//     };

//         // Material parameters (placeholder values)
//     let material = Material {
//         ms: 8.0e5,                 // e.g. ~Fe, in A/m (placeholder)
//         a_ex: 1.0,                 // for now still dimensionless-ish, we'll fix scaling later
//         k_u: 0.0,                  // no anisotropy yet
//         easy_axis: [0.0, 0.0, 1.0],
//     };


//     // 4. Open CSV file for writing (average magnetisation vs time)
//     let file: File = File::create("avg_magnetisation.csv")?;
//     let mut writer: BufWriter<File> = BufWriter::new(file);

//     // Write header
//     writeln!(writer, "t,mx_avg,my_avg,mz_avg")?;

//     // Ensure frames directory exists for PNG snapshots
//     create_dir_all("frames")?;

//     // 5. Time-stepping loop
//     let n_steps: i32 = 100;

//     for step in 0..=n_steps {
//         // Build H_eff for this time step (Zeeman + exchange for now)
//         build_h_eff(&grid, &m, &mut h_eff, &params, &material);

//         // Compute spatially averaged magnetisation over all cells
//         let mut sum_mx = 0.0;
//         let mut sum_my = 0.0;
//         let mut sum_mz = 0.0;
//         let n_cells = m.data.len() as f64;

//         for cell in &m.data {
//             sum_mx += cell[0];
//             sum_my += cell[1];
//             sum_mz += cell[2];
//         }

//         let mx_avg = sum_mx / n_cells;
//         let my_avg = sum_my / n_cells;
//         let mz_avg = sum_mz / n_cells;

//         let t: f64 = step as f64 * params.dt;
//         writeln!(
//             writer,
//             "{:.8},{:.8},{:.8},{:.8}",
//             t, mx_avg, my_avg, mz_avg
//         )?;

//         // Save a plot of m_z every 10 steps
//         if step % 10 == 0 {
//             let filename = format!("frames/mz_{:04}.png", step);
//             save_mz_plot(&m, &filename).expect("failed to save m_z plot");
//             println!(
//                 "Saved frame {} (step = {}, t = {:.3}, mz_avg = {:.3})",
//                 filename, step, t, mz_avg
//             );

//             // On macOS, open each frame as it is created
//             #[cfg(target_os = "macos")]
//             {
//                 let _ = Command::new("open").arg(&filename).status();
//             }
//         }

//         // Advance one time step (skip on last step)
//         if step < n_steps {
//             step_llg_with_field(&mut m, &h_eff, &params);
//         }
//     }

//     println!("Saved average magnetisation to avg_magnetisation.csv and PNG frames in frames/");

//     Ok(())
// }


// // src/main.rs

// use std::fs::File;
// use std::io::{BufWriter, Write};

// use llg_sim::grid::Grid2D;
// use llg_sim::vector_field::VectorField2D;
// use llg_sim::llg::step_llg_with_field;
// use llg_sim::params::LLGParams;
// use llg_sim::effective_field::zeeman::add_zeeman_field;
// use llg_sim::visualisation::save_mz_png;

// fn main() -> std::io::Result<()> {
//     // 1. Set up a 2D grid and magnetisation
//     let nx = 32;
//     let ny = 32;
//     let grid = Grid2D::new(nx, ny, 5e-9, 5e-9);  // 32x32 cells, 5 nm spacing
//     let mut m = VectorField2D::new(grid);

//     // Start along +z
//     m.set_uniform(0.0, 0.0, 1.0);

//     // 2. Define LLG parameters (field along +x)
//     let params = LLGParams {
//         gamma: 1.0,
//         alpha: 0.1,
//         dt: 0.01,
//         h_ext: [1.0, 0.0, 0.0],
//     };

//     // 3. Create an H_eff field on the same grid
//     let mut h_eff = VectorField2D::new(grid);

//     // 4. Open CSV file for writing (average magnetisation vs time)
//     let file = File::create("avg_magnetisation.csv")?;
//     let mut writer = BufWriter::new(file);

//     // Write header
//     writeln!(writer, "t,mx_avg,my_avg,mz_avg")?;

//     // 5. Time-stepping loop
//     let n_steps: i32 = 100;

//     for step in 0..=n_steps {
//         // Build H_eff for this time step (only Zeeman for now)
//         add_zeeman_field(&mut h_eff, params.h_ext);

//         // Compute spatially averaged magnetisation over all cells
//         let mut sum_mx = 0.0;
//         let mut sum_my = 0.0;
//         let mut sum_mz = 0.0;
//         let n_cells = m.data.len() as f64;

//         for cell in &m.data {
//             sum_mx += cell[0];
//             sum_my += cell[1];
//             sum_mz += cell[2];
//         }

//         let mx_avg = sum_mx / n_cells;
//         let my_avg = sum_my / n_cells;
//         let mz_avg = sum_mz / n_cells;

//         let t: f64 = step as f64 * params.dt;
//         writeln!(writer, "{:.8},{:.8},{:.8},{:.8}", t, mx_avg, my_avg, mz_avg)?;

//         // Advance one time step (skip on last step)
//         if step < n_steps {
//             step_llg_with_field(&mut m, &h_eff, &params);
//         }
//     }

//     println!("Saved average magnetisation to avg_magnetisation.csv");

//     Ok(())
// }