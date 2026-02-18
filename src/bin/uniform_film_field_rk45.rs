// src/bin/uniform_film_field_rk45.rs
//
// Uniform 2D film benchmark (exchange + anisotropy + uniform external field).
// RK45 adaptive integrator with UNIFORM physical-time outputs for MuMax comparison.
//
// Run:
//   cargo run --release --bin uniform_film_field_rk45 -- demag=off
//   cargo run --release --bin uniform_film_field_rk45 -- demag=on
//
// Post-process (MuMax overlay: uniform film, adaptive RK45):
//
// Demag OFF:
//   MuMax reference:
//     mumax_outputs/uniform_film_field_demag_off/table.txt
//
//   Example overlays:
//   python3 scripts/overlay_macrospin.py \
//     out/uniform_film_rk45_demag_off/rust_table_uniform_film.csv \
//     mumax_outputs/uniform_film_field_demag_off/table.txt \
//     --col my --clip_overlap --metrics \
//     --out out/uniform_film_rk45_demag_off/overlay_my_vs_time.png
//
//   python3 scripts/overlay_macrospin.py \
//     out/uniform_film_rk45_demag_off/rust_table_uniform_film.csv \
//     mumax_outputs/uniform_film_field_demag_off/table.txt \
//     --col mz --clip_overlap --metrics \
//     --out out/uniform_film_rk45_demag_off/overlay_mz_vs_time.png
//
// Demag ON:
//   MuMax reference:
//     mumax_outputs/uniform_film_field_demag_on/table.txt
//
//   Example overlays:
//   python3 scripts/overlay_macrospin.py \
//     out/uniform_film_rk45_demag_on/rust_table_uniform_film.csv \
//     mumax_outputs/uniform_film_field_demag_on/table.txt \
//     --col my --clip_overlap --metrics \
//     --out out/uniform_film_rk45_demag_on/overlay_my_vs_time.png

//   python3 scripts/overlay_macrospin.py \
//     out/uniform_film_rk45_demag_on/rust_table_uniform_film.csv \
//     mumax_outputs/uniform_film_field_demag_on/table.txt \
//     --col mz --clip_overlap --metrics \
//     --out out/uniform_film_rk45_demag_on/overlay_mz_vs_time.png

// Depending on the `demag=` flag, outputs are written to:
//
//   - demag=off -> out/uniform_film_rk45_demag_off/
//   - demag=on  -> out/uniform_film_rk45_demag_on/
//
// Magnetisation components:
//   - mx(t), my(t), mz(t)
//   - m_parallel(t)
//
// Field diagnostics (spatial averages):
//   - B_demagx/y/z(t)
//   - B_effx/y/z(t)
//
// Energy terms:
//   - E_demag(t)
//   - E_total(t)
//
// Output:
//   out/uniform_film_rk45_{demag_on|demag_off}/
//     ├── config.json
//     ├── rust_table_uniform_film.csv
//     └── dt_history.csv

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use llg_sim::config::{
    FieldConfig, GeometryConfig, MaterialConfig, NumericsConfig, RunConfig, RunInfo,
};
use llg_sim::effective_field::demag::compute_demag_field;
use llg_sim::effective_field::{FieldMask, build_h_eff_masked};
use llg_sim::energy::{EnergyBreakdown, compute_energy};
use llg_sim::grid::Grid2D;
use llg_sim::llg::{RK45Scratch, step_llg_rk45_recompute_field_adaptive};
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

fn parse_demag_flag() -> bool {
    let mut demag = true;
    for a in std::env::args().skip(1) {
        if a == "demag" {
            demag = true;
        } else if let Some(v) = a.strip_prefix("demag=") {
            let v = v.trim().to_ascii_lowercase();
            if v == "on" || v == "1" || v == "true" {
                demag = true;
            } else if v == "off" || v == "0" || v == "false" {
                demag = false;
            }
        }
    }
    demag
}

fn avg_vec(field: &VectorField2D) -> [f64; 3] {
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
}

fn write_row(
    w: &mut BufWriter<File>,
    t: f64,
    grid: &Grid2D,
    m: &VectorField2D,
    material: &Material,
    b_ext: [f64; 3],
    b_demag: &mut VectorField2D,
    b_eff_base: &mut VectorField2D,
) -> std::io::Result<()> {
    let [mx, my, mz] = avg_vec(m);

    let e: EnergyBreakdown = compute_energy(grid, m, material, b_ext);

    // Demag field only
    compute_demag_field(grid, m, b_demag, material);
    let [bdx, bdy, bdz] = avg_vec(b_demag);

    // Base effective field without demag (Zeeman + exchange + anisotropy)
    build_h_eff_masked(
        grid,
        m,
        b_eff_base,
        &LLGParams {
            // Only b_ext is used by zeeman in field builder; other params not used here.
            gamma: 0.0,
            alpha: 0.0,
            dt: 0.0,
            b_ext,
        },
        material,
        FieldMask::ExchAnis,
    );

    let [bbx, bby, bbz] = avg_vec(b_eff_base);
    let bex = bbx + bdx;
    let bey = bby + bdy;
    let bez = bbz + bdz;

    writeln!(
        w,
        "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
        t,
        mx,
        my,
        mz,
        e.total(),
        e.exchange,
        e.anisotropy,
        e.zeeman,
        e.demag,
        b_ext[0],
        b_ext[1],
        b_ext[2],
        bdx,
        bdy,
        bdz,
        bex,
        bey,
        bez
    )?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    // --- keep in sync with MuMax script ---
    let nx: usize = 128;
    let ny: usize = 128;
    let dx: f64 = 5e-9;
    let dy: f64 = 5e-9;
    let dz: f64 = 5e-9;

    let ms: f64 = 8.0e5;
    let a_ex: f64 = 13e-12;
    let k_u: f64 = 500.0;
    let easy_axis = [0.0, 0.0, 1.0];

    // External field (phase-lock in y)
    let b_ext = [0.01, 1e-4, 0.0];

    let alpha: f64 = 0.02;

    // Output sampling (match MuMax): dt_out = 1e-12
    let dt_out: f64 = 1e-12;
    let t_total: f64 = 5e-9;
    let n_out: usize = (t_total / dt_out).round() as usize;

    // RK45 initial dt guess
    let dt0: f64 = 1e-13;

    // RK45 controller
    let max_err: f64 = 1e-5;
    let headroom: f64 = 0.8;

    // Clamp dt
    let dt_min: f64 = dt0 * 1e-6;
    let dt_max: f64 = dt0 * 100.0;
    // -------------------------------------

    let enable_demag: bool = parse_demag_flag();

    let grid = Grid2D::new(nx, ny, dx, dy, dz);
    let mut m = VectorField2D::new(grid);
    m.set_uniform(0.0, 0.0, 1.0);

    let mut params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha,
        dt: dt0,
        b_ext,
    };

    let material = Material {
        ms,
        a_ex,
        k_u,
        easy_axis,
        dmi: None,
        demag: enable_demag,
        demag_method: DemagMethod::FftUniform,
    };

    let mut scratch = RK45Scratch::new(grid);

    // Output dirs
    let out_dir: PathBuf = if enable_demag {
        Path::new("out").join("uniform_film_rk45_demag_on")
    } else {
        Path::new("out").join("uniform_film_rk45_demag_off")
    };
    create_dir_all(&out_dir)?;

    // config.json
    let run_config = RunConfig {
        geometry: GeometryConfig {
            nx,
            ny,
            nz: 1,
            dx,
            dy,
            dz,
        },
        material: MaterialConfig {
            ms,
            aex: a_ex,
            ku1: k_u,
            easy_axis,
        },
        fields: FieldConfig {
            b_ext,
            demag: material.demag,
            dmi: material.dmi,
        },
        numerics: NumericsConfig {
            integrator: "rk45".to_string(),
            dt: dt0,
            steps: n_out,
            output_stride: 1,
            max_err: Some(max_err),
            headroom: Some(headroom),
            dt_min: Some(dt_min),
            dt_max: Some(dt_max),
        },
        run: RunInfo {
            binary: "uniform_film_field_rk45".to_string(),
            run_id: if enable_demag {
                "uniform_film_rk45_demag_on"
            } else {
                "uniform_film_rk45_demag_off"
            }
            .to_string(),
            git_commit: None,
            timestamp_utc: None,
        },
    };
    run_config.write_to_dir(&out_dir)?;

    // CSV output
    let file = File::create(out_dir.join("rust_table_uniform_film.csv"))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "t,mx,my,mz,E_total,E_ex,E_an,E_zee,E_demag,Bx,By,Bz,B_demagx,B_demagy,B_demagz,B_effx,B_effy,B_effz"
    )?;

    // dt/eps history
    let file_dt = File::create(out_dir.join("dt_history.csv"))?;
    let mut wdt = BufWriter::new(file_dt);
    writeln!(wdt, "attempt,t,dt_used,eps,accepted")?;

    // field scratch for diagnostics
    let mut b_demag = VectorField2D::new(grid);
    let mut b_eff_base = VectorField2D::new(grid);

    // Write t=0
    write_row(
        &mut w,
        0.0,
        &grid,
        &m,
        &material,
        b_ext,
        &mut b_demag,
        &mut b_eff_base,
    )?;

    // Integrate to each output time
    let tol_time = 1e-18_f64;
    let mut t: f64 = 0.0;
    let mut attempt: usize = 0;

    for k in 1..=n_out {
        let t_target = (k as f64) * dt_out;

        while t + tol_time < t_target {
            attempt += 1;

            // Clamp dt to land exactly on t_target
            let remaining = t_target - t;
            if params.dt > remaining {
                params.dt = remaining;
            }

            let (eps, accepted, dt_used) = step_llg_rk45_recompute_field_adaptive(
                &mut m,
                &mut params,
                &material,
                &mut scratch,
                max_err,
                headroom,
                dt_min,
                dt_max,
            );

            writeln!(
                wdt,
                "{},{:.16e},{:.16e},{:.16e},{}",
                attempt,
                t,
                dt_used,
                eps,
                if accepted { 1 } else { 0 }
            )?;

            if !accepted {
                continue;
            }
            t += dt_used;
        }

        t = t_target;
        write_row(
            &mut w,
            t,
            &grid,
            &m,
            &material,
            b_ext,
            &mut b_demag,
            &mut b_eff_base,
        )?;
    }

    println!("Wrote outputs to {:?}", out_dir);
    println!("Demag enabled: {}", enable_demag);
    println!("t_total={}, dt_out={}, n_out={}", t_total, dt_out, n_out);
    // Final state summary (spatial average)
    let [mx_f, my_f, mz_f] = avg_vec(&m);
    println!(
        "FINAL <m> at t = {:.3e} s: mx={:.6e}, my={:.6e}, mz={:.6e}",
        t_total, mx_f, my_f, mz_f
    );
    Ok(())
}
