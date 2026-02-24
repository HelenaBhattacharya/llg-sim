// src/bin/demag_spectrum_diag.rs
//
// A1: spectrum diagnosis for demag_poisson_mg vs demag_fft_uniform.
//
// Goal:
//   Stop guessing whether the mismatch is (a) high-k/anisotropy dominated or (b) low-k dominated.
//
// Method (impulse-response):
//   1) Build impulse magnetisations (Mx/My/Mz) on a rectangle (single nonzero cell at center).
//   2) Compute B_fft and B_mg.
//   3) Take 2D FFT of each B component.
//   4) Compare spectra via:
//        (i) radial mean relative error vs |k|
//       (ii) anisotropy mean relative error: axis (kx=0 or ky=0) vs diagonal (|kx|=|ky|)
//
// IMPORTANT: Relative error is computed robustly to avoid division-by-near-zero:
//   rel(k) = |F_mg - F_fft| / max(|F_fft|, |F_mg|, eps)
//
// Output folder:
//   runs/demag_spectrum_diag/summary.txt
//   runs/demag_spectrum_diag/radial_<imp>_<bout>.csv
//   runs/demag_spectrum_diag/anisotropy_<imp>_<bout>.csv
//
// Run:
//   cargo run --release --bin demag_spectrum_diag
//
// Env overrides:
//   DIAG_NX, DIAG_NY, DIAG_DX, DIAG_DY, DIAG_DZ, DIAG_BINS
//   DIAG_EPS_REL   (default 1e-12; eps = DIAG_EPS_REL * max_k_amp_ref)
//   DIAG_VERBOSE   (0/1; default 0)
//
// Uses current MG env config (LLG_DEMAG_MG_*).

use std::fs::{File, create_dir_all};
use std::io::{self, Write};
use std::path::PathBuf;

use llg_sim::effective_field::demag_poisson_mg::DemagPoissonMGConfig;
use llg_sim::effective_field::{demag_fft_uniform, demag_poisson_mg};
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, Material};
use llg_sim::vector_field::VectorField2D;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

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
fn env_bool01(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .map(|v| v != 0)
        .unwrap_or(default)
}

fn out_dir() -> io::Result<PathBuf> {
    let dir = PathBuf::from("runs").join("demag_spectrum_diag");
    create_dir_all(&dir)?;
    Ok(dir)
}

fn impulse_m(grid: Grid2D, comp: usize) -> VectorField2D {
    let mut m = VectorField2D::new(grid);
    for v in &mut m.data {
        *v = [0.0, 0.0, 0.0];
    }
    let cx = m.grid.nx / 2;
    let cy = m.grid.ny / 2;
    let id = m.idx(cx, cy);
    let mut v = [0.0, 0.0, 0.0];
    v[comp] = 1.0; // unit impulse
    m.data[id] = v;
    m
}

fn fft2_inplace(buf: &mut [Complex<f64>], nx: usize, ny: usize) {
    // Forward FFT along x (rows), then along y (cols).
    let mut planner = FftPlanner::<f64>::new();
    let fft_x = planner.plan_fft_forward(nx);
    let fft_y = planner.plan_fft_forward(ny);

    // rows
    for j in 0..ny {
        let row = &mut buf[j * nx..(j + 1) * nx];
        fft_x.process(row);
    }

    // cols (temp buffer)
    let mut col = vec![Complex::<f64>::new(0.0, 0.0); ny];
    for i in 0..nx {
        for j in 0..ny {
            col[j] = buf[j * nx + i];
        }
        fft_y.process(&mut col);
        for j in 0..ny {
            buf[j * nx + i] = col[j];
        }
    }
}

fn fft2_of_component(b: &VectorField2D, comp: usize) -> Vec<Complex<f64>> {
    let nx = b.grid.nx;
    let ny = b.grid.ny;
    let mut buf: Vec<Complex<f64>> = Vec::with_capacity(nx * ny);
    for v in &b.data {
        buf.push(Complex::new(v[comp], 0.0));
    }
    fft2_inplace(&mut buf, nx, ny);
    buf
}

fn wrapped_index(i: usize, n: usize) -> isize {
    let ii = i as isize;
    let nn = n as isize;
    if ii <= nn / 2 { ii } else { ii - nn }
}

fn write_radial_csv(
    path: &PathBuf,
    bins: usize,
    sum_rel: &[f64],
    sum_ratio: &[f64],
    cnt: &[usize],
) -> io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "k_over_kmax,mean_rel_err,mean_amp_ratio,count")?;
    for b in 0..bins {
        let mean_rel = if cnt[b] > 0 {
            sum_rel[b] / (cnt[b] as f64)
        } else {
            f64::NAN
        };
        let mean_ratio = if cnt[b] > 0 {
            sum_ratio[b] / (cnt[b] as f64)
        } else {
            f64::NAN
        };
        let kfrac = (b as f64 + 0.5) / (bins as f64);
        writeln!(
            f,
            "{:.6e},{:.6e},{:.6e},{}",
            kfrac, mean_rel, mean_ratio, cnt[b]
        )?;
    }
    Ok(())
}

fn write_aniso_csv(
    path: &PathBuf,
    bins: usize,
    sum_axis: &[f64],
    cnt_axis: &[usize],
    sum_diag: &[f64],
    cnt_diag: &[usize],
) -> io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(
        f,
        "k_over_kmax,mean_axis_rel,mean_diag_rel,axis_minus_diag,n_axis,n_diag"
    )?;
    for b in 0..bins {
        let ma = if cnt_axis[b] > 0 {
            sum_axis[b] / (cnt_axis[b] as f64)
        } else {
            f64::NAN
        };
        let md = if cnt_diag[b] > 0 {
            sum_diag[b] / (cnt_diag[b] as f64)
        } else {
            f64::NAN
        };
        let kfrac = (b as f64 + 0.5) / (bins as f64);
        writeln!(
            f,
            "{:.6e},{:.6e},{:.6e},{:.6e},{},{}",
            kfrac,
            ma,
            md,
            ma - md,
            cnt_axis[b],
            cnt_diag[b]
        )?;
    }
    Ok(())
}

fn mean_over_bin_range(sum: &[f64], cnt: &[usize], lo: usize, hi: usize) -> f64 {
    let mut ss = 0.0;
    let mut cc = 0usize;
    for b in lo..hi {
        ss += sum[b];
        cc += cnt[b];
    }
    if cc > 0 { ss / (cc as f64) } else { f64::NAN }
}

fn main() -> io::Result<()> {
    let nx = env_usize("DIAG_NX", 256);
    let ny = env_usize("DIAG_NY", 256);
    let dx = env_f64("DIAG_DX", 2.5e-9);
    let dy = env_f64("DIAG_DY", 2.5e-9);
    let dz = env_f64("DIAG_DZ", 3.0e-9);
    let bins = env_usize("DIAG_BINS", 64);
    let eps_rel = env_f64("DIAG_EPS_REL", 1e-12);
    let verbose = env_bool01("DIAG_VERBOSE", false);

    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    let mat = Material {
        ms: 8.0e5,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let mg_cfg = DemagPoissonMGConfig::from_env();
    eprintln!(
        "[A1] grid=({}x{}), dx={:.3} nm, dy={:.3} nm, dz={:.3} nm, bins={}",
        nx,
        ny,
        dx * 1e9,
        dy * 1e9,
        dz * 1e9,
        bins
    );
    eprintln!("[A1] mg_cfg: {:?}", mg_cfg);

    // Physical k_max (corner of Brillouin zone for the grid):
    let kx_max = std::f64::consts::PI / dx;
    let ky_max = std::f64::consts::PI / dy;
    let kmax = (kx_max * kx_max + ky_max * ky_max).sqrt();
    let lx = (nx as f64) * dx;
    let ly = (ny as f64) * dy;

    let out = out_dir()?;
    let mut summary = File::create(out.join("summary.txt"))?;

    for (imp_c, imp_name) in [(0, "Mx"), (1, "My"), (2, "Mz")] {
        let m = impulse_m(grid, imp_c);

        let mut b_fft = VectorField2D::new(grid);
        let mut b_mg = VectorField2D::new(grid);

        demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_fft, &mat);
        demag_poisson_mg::compute_demag_field_poisson_mg(&grid, &m, &mut b_mg, &mat);

        for (bout_c, bout_name) in [(0, "Bx"), (1, "By"), (2, "Bz")] {
            let f_fft = fft2_of_component(&b_fft, bout_c);
            let f_mg = fft2_of_component(&b_mg, bout_c);

            // Set epsilon relative to the maximum magnitude in the reference spectrum.
            let mut max_ref = 0.0f64;
            for v in &f_fft {
                max_ref = max_ref.max(v.norm());
            }
            let eps = (eps_rel * max_ref).max(1e-30);

            let mut sum_rel = vec![0.0f64; bins];
            let mut sum_ratio = vec![0.0f64; bins];
            let mut cnt = vec![0usize; bins];

            let mut sum_axis = vec![0.0f64; bins];
            let mut cnt_axis = vec![0usize; bins];
            let mut sum_diag = vec![0.0f64; bins];
            let mut cnt_diag = vec![0usize; bins];

            for j in 0..ny {
                let wy = wrapped_index(j, ny);
                let ky = 2.0 * std::f64::consts::PI * (wy as f64) / ly;

                for i in 0..nx {
                    let wx = wrapped_index(i, nx);
                    let kx = 2.0 * std::f64::consts::PI * (wx as f64) / lx;

                    // skip DC
                    if wx == 0 && wy == 0 {
                        continue;
                    }

                    let r = (kx * kx + ky * ky).sqrt();
                    let b = ((r / kmax) * (bins as f64)).floor() as isize;
                    if b < 0 {
                        continue;
                    }
                    let b = (b as usize).min(bins - 1);

                    let id = j * nx + i;
                    let refv = f_fft[id];
                    let testv = f_mg[id];

                    let ref_amp = refv.norm();
                    let test_amp = testv.norm();
                    let denom = ref_amp.max(test_amp).max(eps);

                    let rel = (testv - refv).norm() / denom;
                    let ratio = test_amp / ref_amp.max(eps);

                    sum_rel[b] += rel;
                    sum_ratio[b] += ratio;
                    cnt[b] += 1;

                    // anisotropy sampling
                    if wx == 0 || wy == 0 {
                        sum_axis[b] += rel;
                        cnt_axis[b] += 1;
                    }
                    if wx.abs() == wy.abs() {
                        sum_diag[b] += rel;
                        cnt_diag[b] += 1;
                    }
                }
            }

            let radial_path = out.join(format!("radial_{}_{}.csv", imp_name, bout_name));
            let aniso_path = out.join(format!("anisotropy_{}_{}.csv", imp_name, bout_name));
            write_radial_csv(&radial_path, bins, &sum_rel, &sum_ratio, &cnt)?;
            write_aniso_csv(
                &aniso_path,
                bins,
                &sum_axis,
                &cnt_axis,
                &sum_diag,
                &cnt_diag,
            )?;

            // quick low/high summaries (count-weighted)
            let low_hi = ((0.20 * bins as f64).round() as usize).clamp(1, bins);
            let high_lo = ((0.80 * bins as f64).round() as usize).clamp(0, bins.saturating_sub(1));

            let low_rel = mean_over_bin_range(&sum_rel, &cnt, 0, low_hi);
            let high_rel = mean_over_bin_range(&sum_rel, &cnt, high_lo, bins);

            let low_axis = mean_over_bin_range(&sum_axis, &cnt_axis, 0, low_hi);
            let low_diag = mean_over_bin_range(&sum_diag, &cnt_diag, 0, low_hi);
            let high_axis = mean_over_bin_range(&sum_axis, &cnt_axis, high_lo, bins);
            let high_diag = mean_over_bin_range(&sum_diag, &cnt_diag, high_lo, bins);

            let low_ratio = mean_over_bin_range(&sum_ratio, &cnt, 0, low_hi);
            let high_ratio = mean_over_bin_range(&sum_ratio, &cnt, high_lo, bins);
            let aniso_low = low_axis - low_diag;
            let aniso_high = high_axis - high_diag;

            writeln!(
                summary,
                "{imp} impulse -> {bout}: rel_err low(<0.2kmax)={low_rel:.3e}  high(>0.8kmax)={high_rel:.3e}  \
amp_ratio low={low_ratio:.3e} high={high_ratio:.3e}  aniso_low(axis-diag)={aniso_low:.3e}  aniso_high(axis-diag)={aniso_high:.3e}",
                imp = imp_name,
                bout = bout_name,
            )?;

            if verbose {
                eprintln!(
                    "[A1] {imp}->{bout}  low_rel={low_rel:.3e}  high_rel={high_rel:.3e}  aniso_high(axis-diag)={aniso_high:.3e}",
                    imp = imp_name,
                    bout = bout_name
                );
            }
        }
    }

    eprintln!("[A1] wrote {}", out.display());
    Ok(())
}
