// src/main.rs

use std::env;
use std::fs::{File, create_dir_all, remove_dir_all};
use std::io::{BufWriter, Write};
use std::process::Command;

use llg_sim::grid::Grid2D;
use llg_sim::vector_field::VectorField2D;
use llg_sim::llg::step_llg_with_field;
use llg_sim::params::{SimConfig, Preset, InitKind};
use llg_sim::effective_field::build_h_eff;
use llg_sim::visualisation::{
    save_mz_plot,
    make_movie_with_ffmpeg,
    save_energy_components_plot,
    save_m_avg_plot,
    save_energy_residual_plot,
    save_m_avg_zoom_plot,
};
use llg_sim::energy::{compute_energy, EnergyBreakdown};

fn main() -> std::io::Result<()> {
    let mut init: InitKind = InitKind::Bloch;
    let mut preset: Preset = Preset::Toy;
    let mut make_movie_flag = false;

    for arg in env::args().skip(1) {
        if let Some(k) = InitKind::from_arg(&arg) { init = k; continue; }
        if let Some(p) = Preset::from_arg(&arg) { preset = p; continue; }
        if arg == "movie" { make_movie_flag = true; continue; }
        eprintln!("Warning: ignoring unknown argument '{arg}'");
    }

    let cfg: SimConfig = SimConfig::new(preset, init);
    let grid_spec = cfg.grid;
    let params = cfg.llg;
    let material = cfg.material;
    let run = cfg.run;

    let grid: Grid2D = Grid2D::new(grid_spec.nx, grid_spec.ny, grid_spec.dx, grid_spec.dy, grid_spec.dz);
    let mut m: VectorField2D = VectorField2D::new(grid);
    let mut b_eff: VectorField2D = VectorField2D::new(grid);

    println!("--- llg-sim run config ---");
    println!("preset: {}", cfg.preset.as_str());
    println!("init:   {}", cfg.init.as_str());
    println!(
        "grid:   nx={} ny={} dx={:.3e} dy={:.3e} dz={:.3e} (Lx={:.3e}, Ly={:.3e})",
        grid_spec.nx, grid_spec.ny, grid_spec.dx, grid_spec.dy, grid_spec.dz,
        grid_spec.lx(), grid_spec.ly(),
    );
    println!(
        "LLG:    gamma={:.6e} alpha={:.3} dt={:.6e}  B_ext=[{:.3e},{:.3e},{:.3e}]",
        params.gamma, params.alpha, params.dt,
        params.b_ext[0], params.b_ext[1], params.b_ext[2]
    );
    println!(
        "mat:    Ms={:.3e} A={:.3e} Ku={:.3e}  u=[{:.3},{:.3},{:.3}]",
        material.ms, material.a_ex, material.k_u,
        material.easy_axis[0], material.easy_axis[1], material.easy_axis[2]
    );
    println!("run:    steps={} save_every={} fps={} zoom_t_max={}", run.n_steps, run.save_every, run.fps, run.zoom_t_max);
    println!("--------------------------");

    match init {
        InitKind::Uniform => {
            println!("Initial condition: uniform +z");
            m.set_uniform(0.0, 0.0, 1.0);
        }
        InitKind::Tilt => {
            println!("Initial condition: uniform, 10° tilt in x–z plane");
            let theta = 10.0_f64.to_radians();
            m.set_uniform(theta.sin(), 0.0, theta.cos());
        }
        InitKind::Bloch => {
            println!("Initial condition: Bloch wall");
            let x0 = 0.5 * grid_spec.nx as f64 * grid_spec.dx;
            let width = 5.0 * grid_spec.dx;
            m.init_bloch_wall(x0, width);
        }
    }

    let file_mag: File = File::create("avg_magnetisation.csv")?;
    let mut writer_mag: BufWriter<File> = BufWriter::new(file_mag);
    writeln!(writer_mag, "t,mx_avg,my_avg,mz_avg")?;

    let file_energy: File = File::create("energy_vs_time.csv")?;
    let mut writer_energy: BufWriter<File> = BufWriter::new(file_energy);
    writeln!(writer_energy, "t,E_ex,E_an,E_zee,E_tot")?;

    if let Err(e) = remove_dir_all("frames") {
        if e.kind() != std::io::ErrorKind::NotFound {
            eprintln!("Warning: could not clear frames/: {e}");
        }
    }
    create_dir_all("frames")?;

    let n_pts = run.n_steps + 1;
    let mut times: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_ex_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_an_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_zee_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_tot_vec: Vec<f64> = Vec::with_capacity(n_pts);

    let mut mx_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut my_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut mz_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);

    for step in 0..=run.n_steps {
        build_h_eff(&grid, &m, &mut b_eff, &params, &material);

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

        mx_avg_vec.push(mx_avg);
        my_avg_vec.push(my_avg);
        mz_avg_vec.push(mz_avg);
        writeln!(writer_mag, "{:.8},{:.8},{:.8},{:.8}", t, mx_avg, my_avg, mz_avg)?;

        let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.b_ext);
        let e_tot = e.total();

        writeln!(
            writer_energy,
            "{:.8},{:.16e},{:.16e},{:.16e},{:.16e}",
            t, e.exchange, e.anisotropy, e.zeeman, e_tot
        )?;

        times.push(t);
        e_ex_vec.push(e.exchange);
        e_an_vec.push(e.anisotropy);
        e_zee_vec.push(e.zeeman);
        e_tot_vec.push(e_tot);

        if step % 10 == 0 {
            println!(
                "step {:4}, t = {:.3e}, E_ex = {:.3e}, E_an = {:.3e}, E_zee = {:.3e}, E_tot = {:.3e}",
                step, t, e.exchange, e.anisotropy, e.zeeman, e_tot
            );
        }

        if step % run.save_every == 0 {
            let filename = format!("frames/mz_{:04}.png", step);
            save_mz_plot(&m, &filename).expect("failed to save m_z plot");
        }

        if step < run.n_steps {
            step_llg_with_field(&mut m, &b_eff, &params);
        }
    }

    let _ = save_energy_components_plot(&times, &e_ex_vec, &e_an_vec, &e_zee_vec, &e_tot_vec, "energy_vs_time.png");
    let _ = save_energy_residual_plot(&times, &e_tot_vec, "energy_residual_vs_time.png");
    let _ = save_m_avg_plot(&times, &mx_avg_vec, &my_avg_vec, &mz_avg_vec, "m_avg_vs_time.png");
    let _ = save_m_avg_zoom_plot(&times, &mx_avg_vec, &my_avg_vec, &mz_avg_vec, run.zoom_t_max, "m_avg_vs_time_zoom.png");

    if make_movie_flag {
        if let Err(e) = make_movie_with_ffmpeg("frames/mz_*.png", "mz_evolution.mp4", run.fps) {
            eprintln!("Could not create movie with ffmpeg: {e}");
        } else {
            println!("Saved movie to mz_evolution.mp4");
            #[cfg(target_os = "macos")]
            {
                let _ = Command::new("open").arg("mz_evolution.mp4").status();
            }
        }
    }

    println!("Done: wrote CSVs + plots + frames/");
    Ok(())
}