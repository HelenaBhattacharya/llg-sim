// src/bin/demag_sp4a_probe.rs
//
// Purpose: fast SP4a-like *dynamic* diagnostic for demag_poisson_mg operator mismatch,
// WITHOUT trajectory divergence confounds.
//
// Strategy:
//   1) Evolve ONE reference trajectory m(t) using FFT/Newell demag only.
//   2) At sample times, evaluate BOTH demag operators on the SAME snapshot m(t):
//        B_fft(m(t)) and B_mg(m(t))
//      and record operator mismatch metrics.
//
// This isolates “operator mismatch” from integrator/trajectory mismatch.
//
// Output:
//   runs/demag_sp4a_probe/table.csv
//
// Run (defaults):
//   cargo run --release --bin demag_sp4a_probe
//
// Useful env overrides:
//   PROBE_NX, PROBE_NY, PROBE_DX, PROBE_DY, PROBE_DZ
//   PROBE_TEND, PROBE_DT, PROBE_STRIDE
//   PROBE_ALPHA, PROBE_GAMMA
//   PROBE_BEXT_X, PROBE_BEXT_Y, PROBE_BEXT_Z
//
// Demag MG uses your existing env vars (LLG_DEMAG_MG_*), including iso27 defaults.
//
// Recommended to keep dt the same as SP4a-ish probe, e.g. dt=2e-12.

use std::fs::{File, create_dir_all};
use std::io::{self, Write};
use std::path::PathBuf;

use llg_sim::effective_field::{demag_fft_uniform, demag_poisson_mg};
use llg_sim::grid::Grid2D;
use llg_sim::initial_states::init_vortex;
use llg_sim::llg::step_llg_with_field_rk4;
use llg_sim::params::{DemagMethod, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm3(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

/// Simple 2D exchange field in Tesla:
///   B_ex = 2 A / Ms * ∇² m
fn add_exchange_field_2d(
    grid: &Grid2D,
    m: &VectorField2D,
    b: &mut VectorField2D,
    a_ex: f64,
    ms: f64,
) {
    if a_ex == 0.0 || ms == 0.0 {
        return;
    }
    let cx = 1.0 / (grid.dx * grid.dx);
    let cy = 1.0 / (grid.dy * grid.dy);
    let pref = 2.0 * a_ex / ms;

    let nx = grid.nx;
    let ny = grid.ny;

    for j in 0..ny {
        let jm = if j == 0 { 0 } else { j - 1 };
        let jp = if j + 1 >= ny { ny - 1 } else { j + 1 };
        for i in 0..nx {
            let im = if i == 0 { 0 } else { i - 1 };
            let ip = if i + 1 >= nx { nx - 1 } else { i + 1 };

            let c = m.idx(i, j);
            let xm = m.idx(im, j);
            let xp = m.idx(ip, j);
            let ym = m.idx(i, jm);
            let yp = m.idx(i, jp);

            for comp in 0..3 {
                let lap = (m.data[xp][comp] - 2.0 * m.data[c][comp] + m.data[xm][comp]) * cx
                    + (m.data[yp][comp] - 2.0 * m.data[c][comp] + m.data[ym][comp]) * cy;
                b.data[c][comp] += pref * lap;
            }
        }
    }
}

fn avg_m(m: &VectorField2D) -> [f64; 3] {
    let mut s = [0.0f64; 3];
    let n = m.data.len() as f64;
    for v in &m.data {
        s[0] += v[0];
        s[1] += v[1];
        s[2] += v[2];
    }
    [s[0] / n, s[1] / n, s[2] / n]
}

fn demag_energy_j(grid: &Grid2D, m: &VectorField2D, b: &VectorField2D, ms: f64) -> f64 {
    // E_d = -(1/2) ∫ M · B dV   (SI)  with M in A/m, B in Tesla.
    let dvol = grid.dx * grid.dy * grid.dz;
    let mut sum = 0.0f64;
    for i in 0..m.data.len() {
        let mi = m.data[i];
        let n2 = mi[0] * mi[0] + mi[1] * mi[1] + mi[2] * mi[2];
        if n2 < 1e-30 {
            continue;
        }
        let m_phys = [ms * mi[0], ms * mi[1], ms * mi[2]];
        sum += dot(m_phys, b.data[i]) * dvol;
    }
    -0.5 * sum
}

#[derive(Clone, Copy)]
struct ErrStats {
    rmse: f64,
    p95: f64,
    max: f64,
}
fn err_stats(vals: &mut [f64]) -> ErrStats {
    if vals.is_empty() {
        return ErrStats {
            rmse: f64::NAN,
            p95: f64::NAN,
            max: f64::NAN,
        };
    }
    let n = vals.len() as f64;
    let rmse = (vals.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95_idx = ((0.95 * (vals.len() as f64 - 1.0)).round() as usize).min(vals.len() - 1);
    let p95 = vals[p95_idx];
    let max = *vals.last().unwrap();
    ErrStats { rmse, p95, max }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}
fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> io::Result<()> {
    // ---- config ----
    let nx: usize = env_usize("PROBE_NX", 128);
    let ny: usize = env_usize("PROBE_NY", 128);
    let dx: f64 = env_f64("PROBE_DX", 4.0e-9);
    let dy: f64 = env_f64("PROBE_DY", 4.0e-9);
    let dz: f64 = env_f64("PROBE_DZ", 20.0e-9);

    let t_end: f64 = env_f64("PROBE_TEND", 3.0e-9);
    let dt: f64 = env_f64("PROBE_DT", 2.0e-12);
    let stride: usize = env_usize("PROBE_STRIDE", 10);

    let alpha: f64 = env_f64("PROBE_ALPHA", 0.02);
    let gamma: f64 = env_f64("PROBE_GAMMA", 1.760_859e11);

    let b_ext = [
        env_f64("PROBE_BEXT_X", 0.010),
        env_f64("PROBE_BEXT_Y", 0.0),
        env_f64("PROBE_BEXT_Z", 0.0),
    ];

    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    // Material (Permalloy-ish)
    let mat = Material {
        ms: 8.0e5,
        a_ex: 1.3e-11,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let llg = LLGParams {
        gamma,
        alpha,
        dt,
        b_ext: [0.0, 0.0, 0.0],
    };

    eprintln!(
        "[probe2] grid=({}x{}), dx={:.3} nm, dz={:.3} nm | t_end={:.3} ns dt={:.3} ps stride={}",
        nx,
        ny,
        dx * 1e9,
        dz * 1e9,
        t_end * 1e9,
        dt * 1e12,
        stride
    );
    eprintln!(
        "[probe2] alpha={}  B_ext=[{:.3e},{:.3e},{:.3e}] T",
        alpha, b_ext[0], b_ext[1], b_ext[2]
    );
    eprintln!("[probe2] MG config from env (LLG_DEMAG_MG_*).");

    // ---- init vortex ----
    let mut m = VectorField2D::new(grid);
    init_vortex(&mut m, &grid, (0.0, 0.0), 1.0, 1.0, 20e-9, None);

    // ---- fields ----
    let mut b_fft = VectorField2D::new(grid);
    let mut b_mg = VectorField2D::new(grid);
    let mut b_eff = VectorField2D::new(grid);

    // ---- output ----
    let out_dir = PathBuf::from("runs/demag_sp4a_probe");
    create_dir_all(&out_dir)?;
    let mut f = File::create(out_dir.join("table.csv"))?;
    writeln!(
        f,
        "t_s,mx,my,mz,db_rmse_T,db_p95_T,db_max_T,de_over_efft,eff_rms_torque_diff"
    )?;

    let n_steps = (t_end / dt).ceil() as usize;

    for step in 0..=n_steps {
        let t = (step as f64) * dt;

        // ---- sample mismatch on current snapshot ----
        if step % stride == 0 || step == n_steps {
            // B_fft(m)
            b_fft.set_uniform(0.0, 0.0, 0.0);
            demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_fft, &mat);

            // B_mg(m)
            b_mg.set_uniform(0.0, 0.0, 0.0);
            demag_poisson_mg::compute_demag_field_poisson_mg(&grid, &m, &mut b_mg, &mat);

            // |ΔB| stats
            let mut mags: Vec<f64> = Vec::with_capacity(m.data.len());
            let mut torque_sq_sum = 0.0f64;

            for i in 0..m.data.len() {
                let db = [
                    b_mg.data[i][0] - b_fft.data[i][0],
                    b_mg.data[i][1] - b_fft.data[i][1],
                    b_mg.data[i][2] - b_fft.data[i][2],
                ];
                let dmag = norm3(db);
                mags.push(dmag);

                // A useful “dynamics relevance” scalar:
                // |m × ΔB| (torque error magnitude) RMS over grid
                let mi = m.data[i];
                let mxdb = [
                    mi[1] * db[2] - mi[2] * db[1],
                    mi[2] * db[0] - mi[0] * db[2],
                    mi[0] * db[1] - mi[1] * db[0],
                ];
                torque_sq_sum += dot(mxdb, mxdb);
            }

            let stats = err_stats(&mut mags);
            let e_fft = demag_energy_j(&grid, &m, &b_fft, mat.ms);
            let e_mg = demag_energy_j(&grid, &m, &b_mg, mat.ms);
            let de_over = if e_fft.abs() > 0.0 {
                (e_mg - e_fft) / e_fft
            } else {
                f64::NAN
            };

            let torque_rms = (torque_sq_sum / (m.data.len() as f64)).sqrt();
            let am = avg_m(&m);

            eprintln!(
                "[probe2] t={:.3} ns  |ΔB| rmse={:.3e} p95={:.3e} max={:.3e}  ΔE/E={:.3e}  torque_rms={:.3e}",
                t * 1e9,
                stats.rmse,
                stats.p95,
                stats.max,
                de_over,
                torque_rms
            );

            writeln!(
                f,
                "{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
                t, am[0], am[1], am[2], stats.rmse, stats.p95, stats.max, de_over, torque_rms
            )?;
        }

        if step == n_steps {
            break;
        }

        // ---- advance reference trajectory with FFT demag ----
        b_fft.set_uniform(0.0, 0.0, 0.0);
        demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_fft, &mat);

        // build B_eff = B_ext + B_fft + B_ex
        b_eff.set_uniform(b_ext[0], b_ext[1], b_ext[2]);
        for i in 0..b_eff.data.len() {
            b_eff.data[i][0] += b_fft.data[i][0];
            b_eff.data[i][1] += b_fft.data[i][1];
            b_eff.data[i][2] += b_fft.data[i][2];
        }
        add_exchange_field_2d(&grid, &m, &mut b_eff, mat.a_ex, mat.ms);

        // fixed-step RK4 with frozen field over the step (reference)
        step_llg_with_field_rk4(&mut m, &b_eff, &llg);
    }

    eprintln!("[probe2] wrote {}", out_dir.join("table.csv").display());
    Ok(())
}
