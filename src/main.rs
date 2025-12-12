use std::env;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::{BufWriter, Write};
use std::process::Command;

use llg_sim::grid::Grid2D;
use llg_sim::vector_field::VectorField2D;
use llg_sim::llg::step_llg_with_field;
use llg_sim::params::{LLGParams, Material};
use llg_sim::effective_field::build_h_eff;
use llg_sim::visualisation::{
    save_mz_plot,
    make_movie_with_ffmpeg,
    save_energy_components_plot,
    save_m_avg_plot,
    save_energy_residual_plot,
    save_m_avg_zoom_plot
};
use llg_sim::energy::{compute_energy, EnergyBreakdown};

fn main() -> std::io::Result<()> {
    // ---------- command line options ----------
    //   cargo run -- bloch
    //   cargo run -- bloch movie
    //   cargo run -- uniform
    //   cargo run -- uniform movie
    //   cargo run -- tilt
    //   cargo run -- tilt movie
    let args: Vec<String> = env::args().collect();
    let init_kind = args.get(1).map(String::as_str).unwrap_or("bloch");
    let make_movie_flag = args.get(2).map(String::as_str) == Some("movie");
    // ------------------------------------------

    // 1. Set up a 2D grid and magnetisation
    let nx: usize = 128;
    let ny: usize = 128;
    let dx: f64 = 5e-12;
    let dy: f64 = 5e-12;

    let grid: Grid2D = Grid2D::new(nx, ny, dx, dy);
    let mut m: VectorField2D = VectorField2D::new(grid);
    let mut h_eff: VectorField2D = VectorField2D::new(grid);

    println!(
        "Domain: {} x {} cells, dx = {:.3} pm, dy = {:.3} pm \
         (Lx = {:.1} pm, Ly = {:.1} pm)",
        nx,
        ny,
        dx * 1e12,
        dy * 1e12,
        nx as f64 * dx * 1e12,
        ny as f64 * dy * 1e12,
    );

    // -------- choose initial condition --------
    match init_kind {
        "uniform" => {
            println!("Initial condition: uniform +z");
            m.set_uniform(0.0, 0.0, 1.0);
        }
        "tilt" => {
            println!("Initial condition: uniform, 10° tilt in x–z plane");
            let theta = 10.0_f64.to_radians();
            // m points mostly along +z, with a small +x component
            m.set_uniform(theta.sin(), 0.0, theta.cos());
        }
        "bloch" => {
            println!("Initial condition: Bloch wall");
            let x0 = 0.5 * nx as f64 * dx; // centre of sample
            let width = 5.0 * dx;          // wall width ~ 5 cells
            m.init_bloch_wall(x0, width);
        }
        other => {
            eprintln!(
                "Unknown initial condition '{}', expected 'uniform', 'tilt', or 'bloch'. \
                 Using Bloch wall.",
                other
            );
            let x0 = 0.5 * nx as f64 * dx;
            let width = 5.0 * dx;
            m.init_bloch_wall(x0, width);
        }
    }
    // ------------------------------------------

    // 2. Define LLG parameters
    let params: LLGParams = LLGParams {
        gamma: 1.0,
        alpha: 0.1,
        dt: 0.0025,
        h_ext: [1.0, 0.0, 0.0],
    };

    // Material parameters (placeholder values)
    let material = Material {
        ms: 8.0e5,
        a_ex: 1.0,
        k_u: 0.1,
        easy_axis: [0.0, 0.0, 1.0],
    };

    // 3. Prepare output: CSVs + frames directory
    // (a) magnetisation averages
    let file_mag: File = File::create("avg_magnetisation.csv")?;
    let mut writer_mag: BufWriter<File> = BufWriter::new(file_mag);
    writeln!(writer_mag, "t,mx_avg,my_avg,mz_avg")?;

    // (b) energy vs time
    let file_energy: File = File::create("energy_vs_time.csv")?;
    let mut writer_energy: BufWriter<File> = BufWriter::new(file_energy);
    writeln!(writer_energy, "t,E_ex,E_an,E_zee,E_tot")?;

    // Clear any existing frames so runs don't mix
    if let Err(e) = remove_dir_all("frames") {
        if e.kind() != std::io::ErrorKind::NotFound {
            eprintln!("Warning: could not clear frames/: {e}");
        }
    }
    create_dir_all("frames")?;

    // Vectors to store energy components and averages for plotting
    let n_steps: i32 = 500;
    let n_pts = (n_steps + 1) as usize;

    let mut times:     Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_ex_vec:  Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_an_vec:  Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_zee_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_tot_vec: Vec<f64> = Vec::with_capacity(n_pts);

    // NEW: average magnetisation components
    let mut mx_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut my_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut mz_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);

    // 4. Time-stepping loop
    for step in 0..=n_steps {
        // Build H_eff for this time step (Zeeman + exchange + anisotropy)
        build_h_eff(&grid, &m, &mut h_eff, &params, &material);

        // Average magnetisation
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

        // Store averages for plotting
        mx_avg_vec.push(mx_avg);
        my_avg_vec.push(my_avg);
        mz_avg_vec.push(mz_avg);

        // Write magnetisation averages to CSV
        writeln!(
            writer_mag,
            "{:.8},{:.8},{:.8},{:.8}",
            t, mx_avg, my_avg, mz_avg
        )?;

        // ---- energy diagnostics each step ----
        let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.h_ext);
        let e_tot = e.total();

        // CSV with components (high precision)
        writeln!(
            writer_energy,
            "{:.8},{:.16e},{:.16e},{:.16e},{:.16e}",
            t, e.exchange, e.anisotropy, e.zeeman, e_tot
        )?;

        // Store for plotting
        times.push(t);
        e_ex_vec.push(e.exchange);
        e_an_vec.push(e.anisotropy);
        e_zee_vec.push(e.zeeman);
        e_tot_vec.push(e_tot);

        if step % 10 == 0 {
            println!(
                "step {:4}, t = {:.3}, E_ex = {:.3e}, E_an = {:.3e}, E_zee = {:.3e}, E_tot = {:.3e}",
                step, t, e.exchange, e.anisotropy, e.zeeman, e_tot
            );
        }

        // Save a plot of m_z every 2 steps
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

    // 5. Save energy vs time plot (auto-scaled, all components)
    if let Err(e) = save_energy_components_plot(
        &times,
        &e_ex_vec,
        &e_an_vec,
        &e_zee_vec,
        &e_tot_vec,
        "energy_vs_time.png",
    ) {
        eprintln!("Could not save energy_vs_time.png: {e}");
    } else {
        println!("Saved energy_vs_time.png");
    }

    // 6. Save average magnetisation vs time
    if let Err(e) = save_m_avg_plot(
        &times,
        &mx_avg_vec,
        &my_avg_vec,
        &mz_avg_vec,
        "m_avg_vs_time.png",
    ) {
        eprintln!("Could not save m_avg_vs_time.png: {e}");
    } else {
        println!("Saved m_avg_vs_time.png");
    }

    // 7. Save total energy residual vs time
    if let Err(e) = save_energy_residual_plot(
        &times,
        &e_tot_vec,
        "energy_residual_vs_time.png",
    ) {
        eprintln!("Could not save energy_residual_vs_time.png: {e}");
    } else {
        println!("Saved energy_residual_vs_time.png");
    }

    // Zoomed average magnetisation plot for early times (t <= 0.1 s)
    if let Err(e) = save_m_avg_zoom_plot(
        &times,
        &mx_avg_vec,
        &my_avg_vec,
        &mz_avg_vec,
        0.1,
        "m_avg_vs_time_zoom.png",
    ) {
        eprintln!("Could not save m_avg_vs_time_zoom.png: {e}");
    } else {
        println!("Saved m_avg_vs_time_zoom.png");
    }

    // 8. Optionally build a movie from all frames using ffmpeg
    if make_movie_flag {
        if let Err(e) = make_movie_with_ffmpeg("frames/mz_*.png", "mz_evolution.mp4", 20) {
            eprintln!("Could not create movie with ffmpeg: {e}");
        } else {
            println!("Saved movie to mz_evolution.mp4");
            #[cfg(target_os = "macos")]
            {
                let _ = Command::new("open").arg("mz_evolution.mp4").status();
            }
        }
    } else {
        println!("Movie generation skipped (no 'movie' flag).");
    }

    println!("Saved CSVs and PNG frames in current directory / frames/");

    Ok(())
}















// use std::env;
// use std::fs::{File, create_dir_all, remove_dir_all};
// use std::io::{BufWriter, Write};
// use std::process::Command;

// use llg_sim::grid::Grid2D;
// use llg_sim::vector_field::VectorField2D;
// use llg_sim::llg::step_llg_with_field;
// use llg_sim::params::{LLGParams, Material};
// use llg_sim::effective_field::build_h_eff;
// use llg_sim::visualisation::{
//     save_mz_plot,
//     make_movie_with_ffmpeg,
//     save_energy_plot,              // <<< new import
// };
// use llg_sim::energy::{compute_energy, EnergyBreakdown};

// fn main() -> std::io::Result<()> {
//     // ---------- command line options ----------
//     //   cargo run -- bloch
//     //   cargo run -- bloch movie
//     //   cargo run -- uniform
//     //   cargo run -- uniform movie
//     let args: Vec<String> = env::args().collect();
//     let init_kind = args.get(1).map(String::as_str).unwrap_or("bloch");
//     let make_movie_flag = args.get(2).map(String::as_str) == Some("movie");
//     // ------------------------------------------

//     // 1. Set up a 2D grid and magnetisation
//     let nx: usize = 128;
//     let ny: usize = 128;
//     let dx: f64 = 5e-12;
//     let dy: f64 = 5e-12;

//     let grid: Grid2D = Grid2D::new(nx, ny, dx, dy);
//     let mut m: VectorField2D = VectorField2D::new(grid);
//     let mut h_eff: VectorField2D = VectorField2D::new(grid);

//     println!(
//         "Domain: {} x {} cells, dx = {:.3} pm, dy = {:.3} pm \
//          (Lx = {:.1} pm, Ly = {:.1} pm)",
//         nx,
//         ny,
//         dx * 1e12,
//         dy * 1e12,
//         nx as f64 * dx * 1e12,
//         ny as f64 * dy * 1e12,
//     );

//     // -------- choose initial condition --------
//     match init_kind {
//     "uniform" => {
//         println!("Initial condition: uniform +z");
//         m.set_uniform(0.0, 0.0, 1.0);
//     }
//     "tilt" => {
//         println!("Initial condition: uniform, 10° tilt in x–z plane");
//         let theta = 10.0_f64.to_radians();
//         // m points mostly along +z, with a small +x component
//         m.set_uniform(theta.sin(), 0.0, theta.cos());
//     }
//     "bloch" => {
//         println!("Initial condition: Bloch wall");
//         let x0 = 0.5 * nx as f64 * dx; // centre of sample
//         let width = 5.0 * dx;          // wall width ~ 5 cells
//         m.init_bloch_wall(x0, width);
//     }
//     other => {
//         eprintln!(
//             "Unknown initial condition '{}', expected 'uniform', 'tilt', or 'bloch'. \
//              Using Bloch wall.",
//             other
//         );
//         let x0 = 0.5 * nx as f64 * dx;
//         let width = 5.0 * dx;
//         m.init_bloch_wall(x0, width);
//     }
// }
//     // ------------------------------------------

//     // 2. Define LLG parameters
//     let params: LLGParams = LLGParams {
//         gamma: 1.0,
//         alpha: 0.1,
//         dt: 0.0025,
//         h_ext: [0.0, 0.0, 0.0],
//     };

//     // Material parameters (placeholder values)
//     let material = Material {
//         ms: 8.0e5,
//         a_ex: 1.0,
//         k_u: 0.1,
//         easy_axis: [0.0, 0.0, 1.0],
//     };

//     // 3. Prepare output: CSVs + frames directory
//     // (a) magnetisation averages
//     let file_mag: File = File::create("avg_magnetisation.csv")?;
//     let mut writer_mag: BufWriter<File> = BufWriter::new(file_mag);
//     writeln!(writer_mag, "t,mx_avg,my_avg,mz_avg")?;

//     // (b) energy vs time
//     let file_energy: File = File::create("energy_vs_time.csv")?;   // <<< new
//     let mut writer_energy: BufWriter<File> = BufWriter::new(file_energy);
//     writeln!(writer_energy, "t,E_ex,E_an,E_zee,E_tot")?;

//     // Clear any existing frames so runs don't mix
//     if let Err(e) = remove_dir_all("frames") {
//         if e.kind() != std::io::ErrorKind::NotFound {
//             eprintln!("Warning: could not clear frames/: {e}");
//         }
//     }
//     create_dir_all("frames")?;

//     // Vectors to store energy for plotting
//     let mut times: Vec<f64> = Vec::with_capacity(501);    // <<< new
//     let mut energies: Vec<f64> = Vec::with_capacity(501);

//     // 4. Time-stepping loop
//     let n_steps: i32 = 500;

//     for step in 0..=n_steps {
//         // Build H_eff for this time step (Zeeman + exchange + anisotropy)
//         build_h_eff(&grid, &m, &mut h_eff, &params, &material);

//         // Average magnetisation
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

//         // Write magnetisation averages
//         writeln!(
//             writer_mag,
//             "{:.8},{:.8},{:.8},{:.8}",
//             t, mx_avg, my_avg, mz_avg
//         )?;

//         // ---- energy diagnostics each step ----
//         // let e_tot = compute_total_energy(&grid, &m, &material, params.h_ext);
//         // writeln!(writer_energy, "{:.8},{:.8}", t, e_tot)?;
//         // times.push(t);
//         // energies.push(e_tot);

//         // if step % 10 == 0 {
//         //     println!(
//         //         "step {:4}, t = {:.3}, E_tot (toy units) = {:.6e}",
//         //         step, t, e_tot
//         //     );
//         // }
//         let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.h_ext);
//         writeln!(
//             writer_energy,
//             "{:.8},{:.8},{:.8},{:.8},{:.8}",
//             t, e.exchange, e.anisotropy, e.zeeman, e.total()
//         )?;
//         times.push(t);
//         energies.push(e.total());

//         if step % 10 == 0 {
//             println!(
//                 "step {:4}, t = {:.3}, E_tot (toy units) = {:.6e}",
//                 step, t, e.total()
//             );
//         }

//         // Save a plot of m_z every 2 steps
//         if step % 2 == 0 {
//             let filename = format!("frames/mz_{:04}.png", step);
//             save_mz_plot(&m, &filename).expect("failed to save m_z plot");
//             println!(
//                 "Saved frame {} (step = {}, t = {:.3}, mz_avg = {:.3})",
//                 filename, step, t, mz_avg
//             );
//         }

//         // Advance one time step (skip on last step)
//         if step < n_steps {
//             step_llg_with_field(&mut m, &h_eff, &params);
//         }
//     }

//     // 5. Save energy vs time plot
//     if let Err(e) = save_energy_plot(&times, &energies, "energy_vs_time.png") {
//         eprintln!("Could not save energy_vs_time.png: {e}");
//     } else {
//         println!("Saved energy_vs_time.png");
//     }

//     // 6. Optionally build a movie from all frames using ffmpeg
//     if make_movie_flag {
//         if let Err(e) = make_movie_with_ffmpeg("frames/mz_*.png", "mz_evolution.mp4", 20) {
//             eprintln!("Could not create movie with ffmpeg: {e}");
//         } else {
//             println!("Saved movie to mz_evolution.mp4");
//             #[cfg(target_os = "macos")]
//             {
//                 let _ = Command::new("open").arg("mz_evolution.mp4").status();
//             }
//         }
//     } else {
//         println!("Movie generation skipped (no 'movie' flag).");
//     }

//     println!("Saved CSVs and PNG frames in current directory / frames/");

//     Ok(())
// }










// // src/main.rs

// use std::env;
// use std::fs::{File, create_dir_all, remove_dir_all};
// use std::io::{BufWriter, Write};
// use std::process::Command;

// use llg_sim::grid::Grid2D;
// use llg_sim::vector_field::VectorField2D;
// use llg_sim::llg::step_llg_with_field;
// use llg_sim::params::{LLGParams, Material};
// use llg_sim::effective_field::build_h_eff;
// use llg_sim::visualisation::{save_mz_plot, make_movie_with_ffmpeg};
// use llg_sim::energy::{compute_total_energy, compute_energy, EnergyBreakdown};

// fn main() -> std::io::Result<()> {
//     // ---------- command line options ----------
//     // Usage:
//     //   cargo run -- bloch           -> Bloch wall, frames only
//     //   cargo run -- bloch movie     -> Bloch wall, frames + MP4
//     //   cargo run -- uniform         -> uniform +z, frames only
//     //   cargo run -- uniform movie   -> uniform +z, frames + MP4
//     //
//     let args: Vec<String> = env::args().collect();
//     let init_kind = args.get(1).map(String::as_str).unwrap_or("bloch");
//     let make_movie_flag = args.get(2).map(String::as_str) == Some("movie");
//     // ------------------------------------------

//     // 1. Set up a 2D grid and magnetisation
//     let nx: usize = 128;
//     let ny: usize = 128;
//     let dx: f64 = 5e-12;
//     let dy: f64 = 5e-12;

//     let grid: Grid2D = Grid2D::new(nx, ny, dx, dy);
//     let mut m: VectorField2D = VectorField2D::new(grid);
//     let mut h_eff: VectorField2D = VectorField2D::new(grid);

//     // let e_init = compute_total_energy(&grid, &m, &material);
//     // println!("Initial E_tot ({} state) = {:.6e}", init_kind, e_init);

//     // Print physical domain size (using pm here because dx is 5e-12 m)
//     println!(
//         "Domain: {} x {} cells, dx = {:.3} pm, dy = {:.3} pm \
//          (Lx = {:.1} pm, Ly = {:.1} pm)",
//         nx,
//         ny,
//         dx * 1e12,
//         dy * 1e12,
//         nx as f64 * dx * 1e12,
//         ny as f64 * dy * 1e12,
//     );

//     // -------- choose initial condition --------
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
//     // ------------------------------------------

//     // 2. Define LLG parameters
//     let params: LLGParams = LLGParams {
//         gamma: 1.0,
//         alpha: 0.1,
//         dt: 0.0025,              // time step
//         h_ext: [0.0, 0.0, 0.0],  // no external field here
//     };

//     // Material parameters (placeholder values)
//     let material = Material {
//         ms: 8.0e5,                 // A/m (placeholder)
//         a_ex: 1.0,                 // exchange strength (toy units)
//         k_u: 0.1,                  // uniaxial anisotropy strength
//         easy_axis: [0.0, 0.0, 1.0],
//     };

//     // 3. Prepare output: CSV + frames directory
//     let file: File = File::create("avg_magnetisation.csv")?;
//     let mut writer: BufWriter<File> = BufWriter::new(file);
//     writeln!(writer, "t,mx_avg,my_avg,mz_avg")?;

//     // Clear any existing frames so runs don't mix
//     if let Err(e) = remove_dir_all("frames") {
//         if e.kind() != std::io::ErrorKind::NotFound {
//             eprintln!("Warning: could not clear frames/: {e}");
//         }
//     }
//     create_dir_all("frames")?;

//     // 4. Time-stepping loop
//     let n_steps: i32 = 500;

//     for step in 0..=n_steps {
//         // Build H_eff for this time step (Zeeman + exchange + anisotropy)
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

//         if step % 10 == 0 {
//         let e_tot = compute_total_energy(&grid, &m, &material);
//         println!(
//             "step {:4}, t = {:.3}, E_tot (toy units) = {:.6e}",
//             step, t, e_tot
//         );
//     }

//         // Save a plot of m_z every 2 steps
//         if step % 2 == 0 {
//             let filename = format!("frames/mz_{:04}.png", step);
//             save_mz_plot(&m, &filename).expect("failed to save m_z plot");
//             println!(
//                 "Saved frame {} (step = {}, t = {:.3}, mz_avg = {:.3})",
//                 filename, step, t, mz_avg
//             );
//         }

//         // Advance one time step (skip on last step)
//         if step < n_steps {
//             step_llg_with_field(&mut m, &h_eff, &params);
//         }
//     }

//     // 5. Optionally build a movie from all frames using ffmpeg
//     if make_movie_flag {
//         if let Err(e) = make_movie_with_ffmpeg("frames/mz_*.png", "mz_evolution.mp4", 20) {
//             eprintln!("Could not create movie with ffmpeg: {e}");
//         } else {
//             println!("Saved movie to mz_evolution.mp4");

//             // On macOS, open the movie so you can just press “play”
//             #[cfg(target_os = "macos")]
//             {
//                 let _ = Command::new("open").arg("mz_evolution.mp4").status();
//             }
//         }
//     } else {
//         println!("Movie generation skipped (no 'movie' flag).");
//     }

//     println!("Saved average magnetisation to avg_magnetisation.csv and PNG frames in frames/");

//     Ok(())
// }

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