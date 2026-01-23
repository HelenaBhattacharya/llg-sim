// src/main.rs

use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use llg_sim::effective_field::build_h_eff;
use llg_sim::energy::{compute_energy, EnergyBreakdown};
use llg_sim::grid::Grid2D;
use llg_sim::llg::{
    step_llg_rk4_recompute_field, step_llg_with_field, step_llg_with_field_rk4, RK4Scratch,
};
use llg_sim::params::{InitKind, Preset, SimConfig};
use llg_sim::vector_field::VectorField2D;
use llg_sim::visualisation::{
    make_movie_with_ffmpeg, save_energy_components_plot, save_energy_residual_plot, save_m_avg_plot,
    save_m_avg_zoom_plot, save_mz_plot,
};
use llg_sim::config::{
    RunConfig,
    GeometryConfig,
    MaterialConfig,
    FieldConfig,
    NumericsConfig,
    RunInfo,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Integrator {
    Euler,
    /// RK4 with *frozen* B_eff across the step (ONLY safe if B_eff is independent of m).
    Rk4,
    /// RK4 where B_eff(m) is recomputed at substeps (the safe default for exchange/anisotropy).
    Rk4Recompute,
}

impl Integrator {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "euler" => Some(Self::Euler),
            "rk4" | "rk4frozen" => Some(Self::Rk4),
            "rk4recompute" | "rk4-recompute" => Some(Self::Rk4Recompute),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::Rk4 => "rk4",
            Self::Rk4Recompute => "rk4recompute",
        }
    }
}

fn print_usage() {
    eprintln!(
        r#"Usage:
  cargo run -- [uniform|tilt|bloch] [toy|mumaxlike] [movie]
             [integrator=euler|rk4|rk4recompute]
             [steps=N] [save=N] [fps=N] [dt=VAL] [frames=N] [zoom=VAL]
             [out=DIR] [run=RUN_ID]

Notes:
  - integrator=... : choose integrator (default: rk4recompute).
      euler         : builds B_eff once/step, Euler update
      rk4           : builds B_eff once/step, RK4 update (FROZEN field)
      rk4recompute  : recomputes B_eff at RK substeps (recommended for exchange/anisotropy)
  - steps=N : override number of timesteps.
  - dt=VAL  : override timestep (seconds).
  - save=N  : save a PNG every N steps (1 = every step; can be very slow).
  - frames=N: target ~N total frames (ignored if save=N is provided).
  - fps=N   : movie FPS (only used if "movie" flag given).
  - out=DIR : output root directory (default: runs).
  - run=ID  : run_id folder name (default: auto timestamp).

Examples:
  cargo run -- bloch toy
  cargo run -- bloch mumaxlike movie
  cargo run -- bloch mumaxlike movie steps=200000 frames=500 fps=30
  cargo run -- tilt mumaxlike steps=5000 save=50
  cargo run -- bloch mumaxlike integrator=rk4recompute movie frames=300
"#
    );
}

fn sanitize_run_id(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn default_run_id(preset: Preset, init: InitKind, integrator: Integrator) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0));
    // epoch seconds + millis => unique-ish, no extra deps
    let ts = format!("{}{:03}", now.as_secs(), now.subsec_millis());
    format!(
        "{}_{}_{}_{}",
        ts,
        preset.as_str(),
        init.as_str(),
        integrator.as_str()
    )
}

fn unique_run_dir(out_root: &str, run_id: &str) -> PathBuf {
    let base = PathBuf::from(out_root);
    let mut dir = base.join(run_id);
    if !dir.exists() {
        return dir;
    }
    for k in 1..1000 {
        let cand = base.join(format!("{}_{}", run_id, k));
        if !cand.exists() {
            dir = cand;
            break;
        }
    }
    dir
}

fn main() -> std::io::Result<()> {
    let argv: Vec<String> = env::args().collect();

    let mut init: InitKind = InitKind::Bloch;
    let mut preset: Preset = Preset::Toy;
    let mut make_movie_flag = false;
    let mut integrator: Integrator = Integrator::Rk4Recompute;

    // Optional overrides
    let mut steps_override: Option<usize> = None;
    let mut save_override: Option<usize> = None;
    let mut fps_override: Option<u32> = None;
    let mut dt_override: Option<f64> = None;
    let mut frames_target: Option<usize> = None;
    let mut zoom_override: Option<f64> = None;

    // Output controls
    let mut out_root_override: Option<String> = None;
    let mut run_id_override: Option<String> = None;

    for arg in argv.iter().skip(1) {
        if arg == "-h" || arg == "--help" || arg == "help" {
            print_usage();
            return Ok(());
        }

        if let Some(k) = InitKind::from_arg(arg) {
            init = k;
            continue;
        }
        if let Some(p) = Preset::from_arg(arg) {
            preset = p;
            continue;
        }
        if arg == "movie" {
            make_movie_flag = true;
            continue;
        }

        if let Some(v) = arg.strip_prefix("integrator=") {
            integrator = Integrator::from_str(v).unwrap_or_else(|| {
                eprintln!("Warning: unknown integrator '{v}', using rk4recompute");
                Integrator::Rk4Recompute
            });
            continue;
        }

        if let Some(v) = arg.strip_prefix("steps=") {
            steps_override = v.parse::<usize>().ok();
            continue;
        }
        if let Some(v) = arg.strip_prefix("save=") {
            save_override = v.parse::<usize>().ok();
            continue;
        }
        if let Some(v) = arg.strip_prefix("fps=") {
            fps_override = v.parse::<u32>().ok();
            continue;
        }
        if let Some(v) = arg.strip_prefix("dt=") {
            dt_override = v.parse::<f64>().ok();
            continue;
        }
        if let Some(v) = arg.strip_prefix("frames=") {
            frames_target = v.parse::<usize>().ok();
            continue;
        }
        if let Some(v) = arg.strip_prefix("zoom=") {
            zoom_override = v.parse::<f64>().ok();
            continue;
        }

        if let Some(v) = arg.strip_prefix("out=") {
            out_root_override = Some(v.to_string());
            continue;
        }
        if let Some(v) = arg.strip_prefix("run=") {
            run_id_override = Some(v.to_string());
            continue;
        }

        eprintln!("Warning: ignoring unknown argument '{arg}'");
    }

    let cfg: SimConfig = SimConfig::new(preset, init);
    let grid_spec = cfg.grid;
    let material = cfg.material;

    // Make mutable local copies so we can override without mutating SimConfig.
    let mut params = cfg.llg;
    let mut run = cfg.run;

    // Apply overrides
    if let Some(dt) = dt_override {
        params.dt = dt;
    }
    if let Some(n) = steps_override {
        run.n_steps = n;
    }
    if let Some(fps) = fps_override {
        run.fps = fps;
    }
    if let Some(z) = zoom_override {
        run.zoom_t_max = z;
    }

    // Output cadence:
    // - if save=N is provided, it wins
    // - else if frames=N is provided, compute save_every to hit ~N frames
    if let Some(s) = save_override {
        run.save_every = s.max(1);
    } else if let Some(target) = frames_target {
        // We always include step=0. If you ask for N frames, you want ~N-1 intervals.
        let denom = target.saturating_sub(1).max(1);
        let suggested = ((run.n_steps as f64) / (denom as f64)).ceil() as usize;
        run.save_every = suggested.max(1);
    }

    // -------- output directory setup --------
    let out_root = out_root_override.unwrap_or_else(|| "runs".to_string());
    create_dir_all(&out_root)?;

    let mut run_id = run_id_override.unwrap_or_else(|| default_run_id(preset, init, integrator));
    run_id = sanitize_run_id(&run_id);

    let run_dir = unique_run_dir(&out_root, &run_id);
    create_dir_all(&run_dir)?;
    let frames_dir = run_dir.join("frames");
    create_dir_all(&frames_dir)?;

    // -------------------------------------------------
    // Write config.json for this run
    // -------------------------------------------------
    let run_config = RunConfig {
        geometry: GeometryConfig {
            nx: grid_spec.nx,
            ny: grid_spec.ny,
            nz: 1,
            dx: grid_spec.dx,
            dy: grid_spec.dy,
            dz: grid_spec.dz,
        },
        material: MaterialConfig {
            ms: material.ms,
            aex: material.a_ex,
            ku1: material.k_u,
            easy_axis: material.easy_axis,
        },
        fields: FieldConfig {
            b_ext: params.b_ext,
            demag: false,   // demag not implemented yet
            dmi: None,      // DMI not implemented yet
        },
        numerics: NumericsConfig {
            integrator: integrator.as_str().to_string(),
            dt: params.dt,
            steps: run.n_steps,
            output_stride: run.save_every,
        },
        run: RunInfo {
            binary: "llg-sim".to_string(),
            run_id: run_id.clone(),
            git_commit: None,
            timestamp_utc: None,
        },
    };

    run_config.write_to_dir(&run_dir)?;


    // ffmpeg expects a glob here because visualisation.rs uses "-pattern_type glob"
    let ffmpeg_pattern = frames_dir.join("mz_*.png").to_string_lossy().to_string();
    // ----------------------------------------

    // Estimate frame count (we always save step=0, and also save the final step)
    let mut n_frames_est = run.n_steps / run.save_every + 1;
    if run.n_steps % run.save_every != 0 {
        n_frames_est += 1;
    }
    let frame_pad = n_frames_est.saturating_sub(1).to_string().len().max(4);

    let grid: Grid2D = Grid2D::new(
        grid_spec.nx,
        grid_spec.ny,
        grid_spec.dx,
        grid_spec.dy,
        grid_spec.dz,
    );

    let mut m: VectorField2D = VectorField2D::new(grid);

    // Needed for Euler / rk4 (frozen field) paths
    let mut b_eff: VectorField2D = VectorField2D::new(grid);

    // RK4 scratch buffers (reused each step; avoids allocations)
    let mut rk4_scratch: RK4Scratch = RK4Scratch::new(grid);

    println!("--- llg-sim run config ---");
    println!("run_dir: {}", run_dir.to_string_lossy());
    println!("preset: {}", cfg.preset.as_str());
    println!("init:   {}", cfg.init.as_str());
    println!("integrator: {}", integrator.as_str());
    println!(
        "grid:   nx={} ny={} dx={:.3e} dy={:.3e} dz={:.3e} (Lx={:.3e}, Ly={:.3e})",
        grid_spec.nx,
        grid_spec.ny,
        grid_spec.dx,
        grid_spec.dy,
        grid_spec.dz,
        grid_spec.lx(),
        grid_spec.ly(),
    );
    println!(
        "LLG:    gamma={:.6e} alpha={:.3} dt={:.6e}  B_ext=[{:.3e},{:.3e},{:.3e}]",
        params.gamma,
        params.alpha,
        params.dt,
        params.b_ext[0],
        params.b_ext[1],
        params.b_ext[2]
    );
    println!(
        "mat:    Ms={:.3e} A={:.3e} Ku={:.3e}  u=[{:.3},{:.3},{:.3}]",
        material.ms,
        material.a_ex,
        material.k_u,
        material.easy_axis[0],
        material.easy_axis[1],
        material.easy_axis[2]
    );
    println!(
        "run:    steps={} save_every={} fps={} zoom_t_max={}",
        run.n_steps, run.save_every, run.fps, run.zoom_t_max
    );

    if integrator == Integrator::Rk4 && (material.a_ex != 0.0 || material.k_u != 0.0) {
        eprintln!(
            "WARNING: integrator=rk4 uses frozen B_eff per step. For exchange/anisotropy you almost certainly want integrator=rk4recompute."
        );
    }

    if make_movie_flag {
        let dt_frame = (run.save_every as f64) * params.dt;
        println!(
            "movie:  approx frames={} (frame_dt≈{:.3e} s) glob={}",
            n_frames_est, dt_frame, ffmpeg_pattern
        );
    }
    println!("--------------------------");

    // -------- choose initial condition --------
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
    // ------------------------------------------

    // Output: CSVs
    let file_mag: File = File::create(run_dir.join("avg_magnetisation.csv"))?;
    let mut writer_mag: BufWriter<File> = BufWriter::new(file_mag);
    writeln!(writer_mag, "t,mx_avg,my_avg,mz_avg")?;

    let file_energy: File = File::create(run_dir.join("energy_vs_time.csv"))?;
    let mut writer_energy: BufWriter<File> = BufWriter::new(file_energy);
    writeln!(writer_energy, "t,E_ex,E_an,E_zee,E_tot")?;

    // Allocate vectors for plots
    let n_pts = run.n_steps + 1;
    let mut times: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_ex_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_an_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_zee_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_tot_vec: Vec<f64> = Vec::with_capacity(n_pts);

    let mut mx_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut my_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut mz_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);

    // Print about ~100 lines max
    let print_every = (run.n_steps / 100).max(10);

    // Sequential frame numbering (0,1,2,...) to keep ordering correct
    let mut frame_idx: usize = 0;

    // Time-stepping loop
    for step in 0..=run.n_steps {
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

        mx_avg_vec.push(mx_avg);
        my_avg_vec.push(my_avg);
        mz_avg_vec.push(mz_avg);
        writeln!(writer_mag, "{:.8},{:.8},{:.8},{:.8}", t, mx_avg, my_avg, mz_avg)?;

        // Energy diagnostics
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

        if step % print_every == 0 {
            println!(
                "step {:6}, t = {:.3e}, E_ex = {:.3e}, E_an = {:.3e}, E_zee = {:.3e}, E_tot = {:.3e}",
                step, t, e.exchange, e.anisotropy, e.zeeman, e_tot
            );
        }

        // Save frames at cadence, AND always save the final step
        if step % run.save_every == 0 || step == run.n_steps {
            let filename = frames_dir.join(format!("mz_{:0width$}.png", frame_idx, width = frame_pad));
            save_mz_plot(&m, filename.to_str().unwrap()).expect("failed to save m_z plot");
            frame_idx += 1;
        }

        // Advance (skip on last step)
        if step < run.n_steps {
            match integrator {
                Integrator::Rk4Recompute => {
                    step_llg_rk4_recompute_field(&mut m, &params, &material, &mut rk4_scratch);
                }
                Integrator::Euler => {
                    build_h_eff(&grid, &m, &mut b_eff, &params, &material);
                    step_llg_with_field(&mut m, &b_eff, &params);
                }
                Integrator::Rk4 => {
                    build_h_eff(&grid, &m, &mut b_eff, &params, &material);
                    step_llg_with_field_rk4(&mut m, &b_eff, &params);
                }
            }
        }
    }

    // Plots (saved inside run_dir)
    let _ = save_energy_components_plot(
        &times,
        &e_ex_vec,
        &e_an_vec,
        &e_zee_vec,
        &e_tot_vec,
        run_dir.join("energy_vs_time.png").to_str().unwrap(),
    );
    let _ = save_energy_residual_plot(
        &times,
        &e_tot_vec,
        run_dir.join("energy_residual_vs_time.png").to_str().unwrap(),
    );
    let _ = save_m_avg_plot(
        &times,
        &mx_avg_vec,
        &my_avg_vec,
        &mz_avg_vec,
        run_dir.join("m_avg_vs_time.png").to_str().unwrap(),
    );
    let _ = save_m_avg_zoom_plot(
        &times,
        &mx_avg_vec,
        &my_avg_vec,
        &mz_avg_vec,
        run.zoom_t_max,
        run_dir.join("m_avg_vs_time_zoom.png").to_str().unwrap(),
    );

    // Optional movie
    if make_movie_flag {
        let movie_path = run_dir.join("mz_evolution.mp4");
        if let Err(e) = make_movie_with_ffmpeg(&ffmpeg_pattern, movie_path.to_str().unwrap(), run.fps) {
            eprintln!("Could not create movie with ffmpeg: {e}");
        } else {
            println!("Saved movie to {}", movie_path.to_string_lossy());
            #[cfg(target_os = "macos")]
            {
                let _ = Command::new("open").arg(movie_path.as_os_str()).status();
            }
        }
    } else {
        println!("Movie generation skipped (no 'movie' flag).");
    }

    println!("Done. Outputs in {}", run_dir.to_string_lossy());
    Ok(())
}




    // // Write a small run_config.txt for traceability
    // {
    //     let mut f = BufWriter::new(File::create(run_dir.join("run_config.txt"))?);
    //     writeln!(f, "cmd: {}", argv.join(" "))?;
    //     writeln!(f, "run_dir: {}", run_dir.to_string_lossy())?;
    //     writeln!(f, "preset: {}", cfg.preset.as_str())?;
    //     writeln!(f, "init: {}", cfg.init.as_str())?;
    //     writeln!(f, "integrator: {}", integrator.as_str())?;
    //     writeln!(f, "steps: {}", run.n_steps)?;
    //     writeln!(f, "save_every: {}", run.save_every)?;
    //     writeln!(f, "fps: {}", run.fps)?;
    //     writeln!(f, "zoom_t_max: {:.6e}", run.zoom_t_max)?;
    //     writeln!(f, "dt: {:.16e}", params.dt)?;
    //     writeln!(
    //         f,
    //         "B_ext: [{:.6e},{:.6e},{:.6e}]",
    //         params.b_ext[0], params.b_ext[1], params.b_ext[2]
    //     )?;
    //     writeln!(f, "Ms: {:.6e}", material.ms)?;
    //     writeln!(f, "A_ex: {:.6e}", material.a_ex)?;
    //     writeln!(f, "Ku: {:.6e}", material.k_u)?;
    //     writeln!(
    //         f,
    //         "easy_axis: [{:.6e},{:.6e},{:.6e}]",
    //         material.easy_axis[0], material.easy_axis[1], material.easy_axis[2]
    //     )?;
    //     writeln!(
    //         f,
    //         "grid: nx={} ny={} dx={:.6e} dy={:.6e} dz={:.6e}",
    //         grid_spec.nx, grid_spec.ny, grid_spec.dx, grid_spec.dy, grid_spec.dz
    //     )?;
    // }