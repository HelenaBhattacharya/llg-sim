// src/bin/uniform_film_field.rs
//
// Uniform 2D film benchmark (exchange + anisotropy + uniform external field).
// This version ENABLES demag so it can be compared against a MuMax3 run with demag on.
//
// Run:
//   cargo run --release --bin uniform_film_field
//
// Post-process (MuMax overlays: uniform film, RK4 recompute-field, demag ON):
//
//   # Overlay m_y(t):
//   python3 scripts/overlay_macrospin.py \
//     out/uniform_film/rust_table_uniform_film.csv \
//     mumax_outputs/uniform_film_field_demag_on/table.txt \
//     --col my --clip_overlap --metrics \
//     --out out/uniform_film/overlay_my_vs_time.png
//
//   # Overlay m_z(t):
//   python3 scripts/overlay_macrospin.py \
//     out/uniform_film/rust_table_uniform_film.csv \
//     mumax_outputs/uniform_film_field_demag_on/table.txt \
//     --col mz --clip_overlap --metrics \
//     --out out/uniform_film/overlay_mz_vs_time.png
//
// All plots are written to:
//   out/uniform_film/
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
// All plots are written into:
//   out/uniform_film/
//
// (e.g. overlay_my_rk4.png, overlay_B_demagz_rk4.png, overlay_E_demag_rk4.png)
//
// Output:
//   out/uniform_film/
//     ├── config.json
//     └── rust_table_uniform_film.csv

use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use llg_sim::config::{FieldConfig, GeometryConfig, MaterialConfig, NumericsConfig, RunConfig, RunInfo};
use llg_sim::effective_field::{build_h_eff_masked, FieldMask};
use llg_sim::effective_field::demag::compute_demag_field;
use llg_sim::energy::{compute_energy, EnergyBreakdown};
use llg_sim::grid::Grid2D;
use llg_sim::llg::{step_llg_rk4_recompute_field, RK4Scratch};
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

fn main() -> std::io::Result<()> {
    // --- keep in sync with MuMax script ---
    let nx: usize = 128;
    let ny: usize = 128;
    let dx: f64 = 5e-9;
    let dy: f64 = 5e-9;
    let dz: f64 = 5e-9;

    let ms: f64 = 8.0e5; // A/m
    let a_ex: f64 = 13e-12; // J/m
    let k_u: f64 = 500.0; // J/m^3
    let easy_axis = [0.0, 0.0, 1.0];

    // External field (phase-lock in y)
    let b_ext = [0.01, 1e-4, 0.0]; // 10 mT + 0.1 mT

    let alpha: f64 = 0.02;
    let dt: f64 = 1e-13; // integration dt
    let t_total: f64 = 5e-9; // 5 ns total time
    let out_stride: usize = 10; // write every N steps -> dt_out = 1e-12 (match MuMax)

    let enable_demag: bool = true;
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
        dmi: None,
        demag: enable_demag,
    };

    let mut scratch = RK4Scratch::new(grid);

    // -------------------------------------------------
    // Output directory + config
    // -------------------------------------------------
    let out_dir = Path::new("out").join("uniform_film");
    create_dir_all(&out_dir)?;

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
            integrator: "rk4_recompute".to_string(),
            dt,
            steps: n_steps,
            output_stride: out_stride,
            // Not used for this fixed-step RK4 script
            max_err: None,
            headroom: None,
            dt_min: None,
            dt_max: None,
        },
        run: RunInfo {
            binary: "uniform_film_field".to_string(),
            run_id: "uniform_film".to_string(),
            git_commit: None,
            timestamp_utc: None,
        },
    };

    run_config.write_to_dir(&out_dir)?;

    // -------------------------------------------------
    // CSV output
    // -------------------------------------------------
    let file = File::create(out_dir.join("rust_table_uniform_film.csv"))?;
    let mut w = BufWriter::new(file);

    // New columns added:
    // - B_demagx/y/z : spatial average demag induction (Tesla)
    // - B_effx/y/z   : spatial average effective induction (Tesla), where
    //                 B_eff = (Zeeman + exchange + anisotropy) + B_demag
    writeln!(
        w,
        "t,mx,my,mz,E_total,E_ex,E_an,E_zee,E_demag,Bx,By,Bz,B_demagx,B_demagy,B_demagz,B_effx,B_effy,B_effz"
    )?;

    // Helper: compute spatial average of any VectorField2D
    let avg_vec = |field: &VectorField2D| -> [f64; 3] {
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

    // Allocate buffers for field-level diagnostics (reused each output step)
    let mut b_demag = VectorField2D::new(grid);
    let mut b_eff_base = VectorField2D::new(grid); // Zeeman + exchange + anisotropy only (no demag)

    // A small helper to write one row (used for step 0 and subsequent outputs)
    let mut write_row = |t: f64, m: &VectorField2D| -> std::io::Result<()> {
        let [mx, my, mz] = avg_vec(m);

        // Energy (as you already log)
        let e: EnergyBreakdown = compute_energy(&grid, m, &material, params.b_ext);

        // Field-level diagnostics:
        // 1) Demag field only
        compute_demag_field(&grid, m, &mut b_demag, &material);
        let [bdx, bdy, bdz] = avg_vec(&b_demag);

        // 2) Base effective field without demag (Zeeman + exchange + anisotropy)
        //    Then we define B_eff = B_base + B_demag.
        build_h_eff_masked(&grid, m, &mut b_eff_base, &params, &material, FieldMask::ExchAnis);
        let [bbx, bby, bbz] = avg_vec(&b_eff_base);
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
            params.b_ext[0],
            params.b_ext[1],
            params.b_ext[2],
            bdx,
            bdy,
            bdz,
            bex,
            bey,
            bez,
        )?;
        Ok(())
    };

    // Write step 0
    write_row(0.0, &m)?;

    for step in 1..=n_steps {
        step_llg_rk4_recompute_field(&mut m, &params, &material, &mut scratch);

        if step % out_stride == 0 || step == n_steps {
            let t = (step as f64) * dt;
            write_row(t, &m)?;
        }
    }

    let dt_out = (out_stride as f64) * dt;
    println!("Wrote outputs to {:?}", out_dir);
    println!("Demag enabled: {}", enable_demag);
    println!("Steps: {}, dt={}, T={}", n_steps, dt, t_total);
    println!("Output stride: {} -> dt_out={}", out_stride, dt_out);

    Ok(())
}