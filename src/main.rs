// src/main.rs
//
// This binary provides a flexible CLI for exploratory runs
// (e.g. quick tests, movies, parameter sweeps).
//
// Outputs from this driver are written to `runs/` (or the directory
// specified via `out=`) and are not committed to version control.
//
// NOTE:
// Reproducible validation benchmarks and MuMax comparisons
// are implemented as dedicated executables under `src/bin/*`.
//
// Examples:
//
//   cargo run --release -- tilt mumaxlike steps=20000 integrator=rk45 movie
//       -> exploratory macrospin-style dynamics with adaptive RK45,
//          saving frames and assembling an MP4 movie.
//
//   cargo run --release -- bloch mumaxlike demag=on movie
//       -> Bloch wall relaxation including magnetostatics,
//          saving frames and assembling a movie.
//
//   cargo run --release -- bloch mumaxlike demag=off integrator=rk4recompute \
//         steps=2000 frames=400 out=runs movie
//       -> fixed-step RK4 run (with field recomputation) for a Bloch wall,
//          targeting 400 movie frames over 2000 accepted steps.
//
// Typical outputs (per run directory):
//   runs/<run_id>/
//     ├── config.json
//     ├── avg_magnetisation.csv
//     ├── energy_vs_time.csv
//     ├── dt_history.csv          (adaptive integrators only)
//     ├── frames/mz_*.png
//     └── mz_evolution.mp4        (if `movie` is enabled)

use std::env;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use llg_sim::config::{
    FieldConfig, GeometryConfig, MaterialConfig, NumericsConfig, RunConfig, RunInfo,
};
use llg_sim::effective_field::build_h_eff;
use llg_sim::energy::{EnergyBreakdown, compute_energy};
use llg_sim::grid::Grid2D;
use llg_sim::llg::{
    RK4Scratch, RK45Scratch, step_llg_rk4_recompute_field, step_llg_rk45_recompute_field_adaptive,
    step_llg_with_field, step_llg_with_field_rk4,
};
use llg_sim::params::{InitKind, Preset, SimConfig};
use llg_sim::vector_field::VectorField2D;
use llg_sim::visualisation::{
    make_movie_with_ffmpeg, save_energy_components_plot, save_energy_residual_plot,
    save_m_avg_plot, save_mz_plot,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Integrator {
    Euler,
    /// RK4 with *frozen* B_eff across the step (ONLY safe if B_eff is independent of m).
    Rk4,
    /// RK4 where B_eff(m) is recomputed at substeps (safe default for exchange/anisotropy).
    Rk4Recompute,
    /// Adaptive Dormand–Prince RK45 (MuMax-like default for dynamics).
    Rk45,
}

impl Integrator {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "euler" => Some(Self::Euler),
            "rk4" | "rk4frozen" => Some(Self::Rk4),
            "rk4recompute" | "rk4-recompute" => Some(Self::Rk4Recompute),
            "rk45" | "rk45adaptive" | "rk45-adaptive" => Some(Self::Rk45),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Euler => "euler",
            Self::Rk4 => "rk4",
            Self::Rk4Recompute => "rk4recompute",
            Self::Rk45 => "rk45",
        }
    }
}

fn print_usage() {
    eprintln!(
        r#"Usage:
  cargo run -- [uniform|tilt|bloch] [toy|mumaxlike] [movie]
             [integrator=euler|rk4|rk4recompute|rk45]
             [demag=on|off] [maxerr=VAL] [headroom=VAL] [dtmin=VAL] [dtmax=VAL]
             [steps=N] [save=N] [fps=N] [dt=VAL] [frames=N] [zoom=VAL]
             [out=DIR] [run=RUN_ID]

Notes:
  - This driver logs one CSV sample per *accepted step* (dense, smooth curves).
  - If 'movie' is set, frames are saved at fixed physical spacing:
        frame_dt = save_every * dt0_initial
    by clamping dt to land exactly on each frame time.
  - If 'movie' is not set, frames are saved every save_every accepted steps (old behaviour).
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
    let mut dmi_override: Option<f64> = None;
    let mut dt_override: Option<f64> = None;
    let mut frames_target: Option<usize> = None;
    let mut zoom_override: Option<f64> = None;

    // Adaptive / demag overrides
    let mut demag_override: Option<bool> = None;
    let mut maxerr_override: Option<f64> = None;
    let mut headroom_override: Option<f64> = None;
    let mut dtmin_override: Option<f64> = None;
    let mut dtmax_override: Option<f64> = None;

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

        if let Some(v) = arg.strip_prefix("dmi=") {
            let v = v.trim();
            if v.eq_ignore_ascii_case("none") || v.eq_ignore_ascii_case("off") {
                dmi_override = Some(0.0);
            } else if let Ok(val) = v.parse::<f64>() {
                dmi_override = Some(val);
            } else {
                eprintln!("Warning: could not parse dmi value '{v}', ignoring");
            }
            continue;
        }

        if let Some(v) = arg.strip_prefix("demag=") {
            let v = v.trim();
            if v.eq_ignore_ascii_case("on") || v == "1" || v.eq_ignore_ascii_case("true") {
                demag_override = Some(true);
            } else if v.eq_ignore_ascii_case("off") || v == "0" || v.eq_ignore_ascii_case("false") {
                demag_override = Some(false);
            } else {
                eprintln!("Warning: could not parse demag value '{v}', expected on/off/1/0");
            }
            continue;
        }
        if arg == "demag" {
            demag_override = Some(true);
            continue;
        }

        if let Some(v) = arg.strip_prefix("maxerr=") {
            maxerr_override = v.parse::<f64>().ok();
            continue;
        }
        if let Some(v) = arg.strip_prefix("headroom=") {
            headroom_override = v.parse::<f64>().ok();
            continue;
        }
        if let Some(v) = arg.strip_prefix("dtmin=") {
            dtmin_override = v.parse::<f64>().ok();
            continue;
        }
        if let Some(v) = arg.strip_prefix("dtmax=") {
            dtmax_override = v.parse::<f64>().ok();
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
    let mut material = cfg.material;

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
    if let Some(d) = dmi_override {
        material.dmi = if d == 0.0 { None } else { Some(d) };
    }
    if let Some(on) = demag_override {
        material.demag = on;
    }

    // Output cadence
    if let Some(s) = save_override {
        run.save_every = s.max(1);
    } else if let Some(target) = frames_target {
        let denom = target.saturating_sub(1).max(1);
        let suggested = ((run.n_steps as f64) / (denom as f64)).ceil() as usize;
        run.save_every = suggested.max(1);
    }

    // For movie timing: fixed physical spacing between frames
    let dt0_initial = params.dt;
    let frame_dt = (run.save_every as f64) * dt0_initial;

    // -------- output directory setup --------
    let out_root = out_root_override.unwrap_or_else(|| "runs".to_string());
    create_dir_all(&out_root)?;

    let mut run_id = run_id_override.unwrap_or_else(|| default_run_id(preset, init, integrator));
    run_id = sanitize_run_id(&run_id);

    let run_dir = unique_run_dir(&out_root, &run_id);
    create_dir_all(&run_dir)?;
    let frames_dir = run_dir.join("frames");
    create_dir_all(&frames_dir)?;

    let ffmpeg_pattern = frames_dir.join("mz_*.png").to_string_lossy().to_string();

    // -------------------------------------------------
    // Write config.json
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
            demag: material.demag,
            dmi: material.dmi,
        },
        numerics: NumericsConfig {
            integrator: integrator.as_str().to_string(),
            dt: params.dt,
            steps: run.n_steps,
            output_stride: run.save_every,
            max_err: if integrator == Integrator::Rk45 {
                Some(maxerr_override.unwrap_or(1e-5))
            } else {
                None
            },
            headroom: if integrator == Integrator::Rk45 {
                Some(headroom_override.unwrap_or(0.8))
            } else {
                None
            },
            dt_min: if integrator == Integrator::Rk45 {
                dtmin_override
            } else {
                None
            },
            dt_max: if integrator == Integrator::Rk45 {
                dtmax_override
            } else {
                None
            },
        },
        run: RunInfo {
            binary: "llg-sim".to_string(),
            run_id: run_id.clone(),
            git_commit: None,
            timestamp_utc: None,
        },
    };
    run_config.write_to_dir(&run_dir)?;

    // Keep frame ordering stable under glob
    let frame_pad: usize = 6;

    let grid: Grid2D = Grid2D::new(
        grid_spec.nx,
        grid_spec.ny,
        grid_spec.dx,
        grid_spec.dy,
        grid_spec.dz,
    );

    let mut m: VectorField2D = VectorField2D::new(grid);
    let mut b_eff: VectorField2D = VectorField2D::new(grid);

    let mut rk4_scratch: RK4Scratch = RK4Scratch::new(grid);
    let mut rk45_scratch: RK45Scratch = RK45Scratch::new(grid);

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
        "LLG:    gamma={:.6e} alpha={:.3} dt0={:.6e}  B_ext=[{:.3e},{:.3e},{:.3e}]",
        params.gamma, params.alpha, params.dt, params.b_ext[0], params.b_ext[1], params.b_ext[2]
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
    println!("fields: demag={} dmi={:?}", material.demag, material.dmi);
    println!(
        "run:    steps={} save_every={} fps={} zoom_t_max={}",
        run.n_steps, run.save_every, run.fps, run.zoom_t_max
    );
    println!("movie frame_dt (physical) = {:.3e} s", frame_dt);

    if integrator == Integrator::Rk4 && (material.a_ex != 0.0 || material.k_u != 0.0) {
        eprintln!(
            "WARNING: integrator=rk4 uses frozen B_eff per step. For exchange/anisotropy you almost certainly want integrator=rk4recompute."
        );
    }

    // Adaptive settings
    let max_err: f64 = maxerr_override.unwrap_or(1e-5);
    let headroom: f64 = headroom_override.unwrap_or(0.8);

    let mut dt_min: f64 = dtmin_override.unwrap_or(params.dt * 1e-6);
    let mut dt_max: f64 = dtmax_override.unwrap_or(params.dt * 100.0);
    if dt_min <= 0.0 {
        dt_min = params.dt * 1e-6;
    }
    if dt_max <= dt_min {
        dt_max = (dt_min * 10.0).max(params.dt);
    }
    if integrator == Integrator::Rk45 {
        println!(
            "rk45:   MaxErr={} headroom={} dt_min={} dt_max={}",
            max_err, headroom, dt_min, dt_max
        );
    }
    println!("--------------------------");

    // -------- initial condition --------
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
            println!("Initial condition: Bloch wall (m_y bump)");
            let x0 = 0.5 * grid_spec.nx as f64 * grid_spec.dx;
            let width = 5.0 * grid_spec.dx;
            m.init_bloch_wall_y(x0, width, 1.0);
        }
    }

    // CSV outputs
    let file_mag: File = File::create(run_dir.join("avg_magnetisation.csv"))?;
    let mut writer_mag: BufWriter<File> = BufWriter::new(file_mag);
    writeln!(writer_mag, "t,mx_avg,my_avg,mz_avg")?;

    let file_energy: File = File::create(run_dir.join("energy_vs_time.csv"))?;
    let mut writer_energy: BufWriter<File> = BufWriter::new(file_energy);
    writeln!(writer_energy, "t,E_ex,E_an,E_zee,E_dmi,E_demag,E_tot")?;

    let file_dt: File = File::create(run_dir.join("dt_history.csv"))?;
    let mut writer_dt: BufWriter<File> = BufWriter::new(file_dt);
    writeln!(writer_dt, "attempt,t,dt,eps,accepted")?;

    // Vectors for plots (dense: one per accepted step)
    let n_pts = run.n_steps + 1;
    let mut times: Vec<f64> = Vec::with_capacity(n_pts);

    let mut e_ex_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_an_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_zee_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_dmi_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_demag_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut e_tot_vec: Vec<f64> = Vec::with_capacity(n_pts);

    let mut mx_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut my_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);
    let mut mz_avg_vec: Vec<f64> = Vec::with_capacity(n_pts);

    // Print about ~100 lines max
    let print_every = (run.n_steps / 100).max(10);

    // Frame logic
    let tol_time: f64 = 1e-15;
    let mut next_frame_t: f64 = 0.0;
    let mut frame_idx: usize = 0;

    // Step loop state
    let mut t: f64 = 0.0;
    let mut step: usize = 0;
    let mut attempt: usize = 0;

    // Helper: average magnetisation
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

    // --- record step 0 ---
    {
        let [mx, my, mz] = avg_m(&m);
        let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.b_ext);
        let e_tot = e.total();

        times.push(t);
        mx_avg_vec.push(mx);
        my_avg_vec.push(my);
        mz_avg_vec.push(mz);

        e_ex_vec.push(e.exchange);
        e_an_vec.push(e.anisotropy);
        e_zee_vec.push(e.zeeman);
        e_dmi_vec.push(e.dmi);
        e_demag_vec.push(e.demag);
        e_tot_vec.push(e_tot);

        writeln!(writer_mag, "{:.16e},{:.16e},{:.16e},{:.16e}", t, mx, my, mz)?;
        writeln!(
            writer_energy,
            "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            t, e.exchange, e.anisotropy, e.zeeman, e.dmi, e.demag, e_tot
        )?;

        // Always save an initial frame so ordering is clear
        let fname = frames_dir.join(format!("mz_{:0width$}.png", frame_idx, width = frame_pad));
        save_mz_plot(&m, fname.to_str().unwrap()).expect("failed to save m_z plot");
        frame_idx += 1;
        next_frame_t += frame_dt;
    }

    // --- main loop ---
    while step < run.n_steps {
        attempt += 1;

        // If movie is requested, enforce constant physical frame spacing by clamping dt to next_frame_t.
        // We do this by modifying params.dt *temporarily* for this attempt (restore for non-RK45).
        let dt_saved = params.dt;
        if make_movie_flag {
            let remaining_to_frame = next_frame_t - t;
            if remaining_to_frame > tol_time && params.dt > remaining_to_frame {
                params.dt = remaining_to_frame;
            }
        }

        let (eps, accepted, dt_used) = match integrator {
            Integrator::Rk45 => step_llg_rk45_recompute_field_adaptive(
                &mut m,
                &mut params,
                &material,
                &mut rk45_scratch,
                max_err,
                headroom,
                dt_min,
                dt_max,
            ),
            Integrator::Rk4Recompute => {
                // Fixed-step integrators should not permanently inherit a clamped dt.
                let dt_step = params.dt;
                step_llg_rk4_recompute_field(&mut m, &params, &material, &mut rk4_scratch);
                params.dt = dt_saved;
                (0.0, true, dt_step)
            }
            Integrator::Euler => {
                let dt_step = params.dt;
                build_h_eff(&grid, &m, &mut b_eff, &params, &material);
                step_llg_with_field(&mut m, &b_eff, &params);
                params.dt = dt_saved;
                (0.0, true, dt_step)
            }
            Integrator::Rk4 => {
                let dt_step = params.dt;
                build_h_eff(&grid, &m, &mut b_eff, &params, &material);
                step_llg_with_field_rk4(&mut m, &b_eff, &params);
                params.dt = dt_saved;
                (0.0, true, dt_step)
            }
        };

        writeln!(
            writer_dt,
            "{},{:.16e},{:.16e},{:.16e},{}",
            attempt,
            t,
            dt_used,
            eps,
            if accepted { 1 } else { 0 }
        )?;

        if !accepted {
            // RK45 updates params.dt itself on reject; keep it.
            if integrator != Integrator::Rk45 {
                params.dt = dt_saved;
            }
            continue;
        }

        t += dt_used;
        step += 1;

        let [mx, my, mz] = avg_m(&m);
        let e: EnergyBreakdown = compute_energy(&grid, &m, &material, params.b_ext);
        let e_tot = e.total();

        times.push(t);
        mx_avg_vec.push(mx);
        my_avg_vec.push(my);
        mz_avg_vec.push(mz);

        e_ex_vec.push(e.exchange);
        e_an_vec.push(e.anisotropy);
        e_zee_vec.push(e.zeeman);
        e_dmi_vec.push(e.dmi);
        e_demag_vec.push(e.demag);
        e_tot_vec.push(e_tot);

        writeln!(writer_mag, "{:.16e},{:.16e},{:.16e},{:.16e}", t, mx, my, mz)?;
        writeln!(
            writer_energy,
            "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            t, e.exchange, e.anisotropy, e.zeeman, e.dmi, e.demag, e_tot
        )?;

        if step % print_every == 0 {
            if integrator == Integrator::Rk45 {
                println!(
                    "step {:6}, t = {:.3e}, dt = {:.3e}, eps = {:.3e}, E_tot = {:.3e}",
                    step, t, dt_used, eps, e_tot
                );
            } else {
                println!("step {:6}, t = {:.3e}, E_tot = {:.3e}", step, t, e_tot);
            }
        }

        // Frame saving:
        // - If movie flag: save at constant physical spacing (next_frame_t schedule).
        // - Else: old behaviour (every save_every accepted steps).
        if make_movie_flag {
            if t + tol_time >= next_frame_t {
                let fname =
                    frames_dir.join(format!("mz_{:0width$}.png", frame_idx, width = frame_pad));
                save_mz_plot(&m, fname.to_str().unwrap()).expect("failed to save m_z plot");
                frame_idx += 1;
                next_frame_t += frame_dt;
            }
        } else {
            if step % run.save_every == 0 || step == run.n_steps {
                let fname =
                    frames_dir.join(format!("mz_{:0width$}.png", frame_idx, width = frame_pad));
                save_mz_plot(&m, fname.to_str().unwrap()).expect("failed to save m_z plot");
                frame_idx += 1;
            }
        }
    }

    // Plots
    let _ = save_energy_components_plot(
        &times,
        &e_ex_vec,
        &e_an_vec,
        &e_zee_vec,
        &e_dmi_vec,
        &e_demag_vec,
        &e_tot_vec,
        run_dir.join("energy_vs_time.png").to_str().unwrap(),
    );
    let _ = save_energy_residual_plot(
        &times,
        &e_tot_vec,
        run_dir
            .join("energy_residual_vs_time.png")
            .to_str()
            .unwrap(),
    );
    let _ = save_m_avg_plot(
        &times,
        &mx_avg_vec,
        &my_avg_vec,
        &mz_avg_vec,
        run_dir.join("m_avg_vs_time.png").to_str().unwrap(),
    );

    // Optional movie
    if make_movie_flag {
        let movie_path = run_dir.join("mz_evolution.mp4");
        if let Err(e) =
            make_movie_with_ffmpeg(&ffmpeg_pattern, movie_path.to_str().unwrap(), run.fps)
        {
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
