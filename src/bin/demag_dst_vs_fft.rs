// src/bin/demag_dst_vs_fft.rs
//
// Compare demag field computed by:
//   (1) FFT convolution demag (uniform FD grid)   — reference
//   (2) DST Poisson decomposition (U = v + w)     — under test
//
// Usage:
//   cargo run --release --bin demag_dst_vs_fft
//   cargo run --release --bin demag_dst_vs_fft -- 64 64 5e-9 5e-9 1e-9
//
// Pattern options:
//   --pattern random      (default; high-k / worst-case)
//   --pattern smooth      (single Fourier-like mode)
//   --pattern wall        (smooth Bloch wall along y)
//   --pattern impulse     (single-cell Mz impulse)
//
// Optional:
//   --seed <u64>               (only affects random)
//   --wall-width-cells <f64>   (only affects wall; default 8)
//   --sweep                    (resolution sweep at fixed physical size)
//   --diag                     (print DST phase timings and intermediate norms)

use llg_sim::effective_field::{demag_fft_uniform, demag_poisson_dst};
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, MU0, Material};
use llg_sim::vector_field::VectorField2D;

use std::time::Instant;

// ---------------------------------------------------------------------------
// Pattern selection (same as original)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Pattern {
    Random,
    Smooth,
    Wall,
    Impulse,
}

impl Pattern {
    fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "random" | "rand" => Some(Self::Random),
            "smooth" | "mode" => Some(Self::Smooth),
            "wall" | "bloch_wall" | "bloch" => Some(Self::Wall),
            "impulse" | "delta" | "single" => Some(Self::Impulse),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Smooth => "smooth",
            Self::Wall => "wall",
            Self::Impulse => "impulse",
        }
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn print_help_and_exit() -> ! {
    eprintln!(
        "Usage:\n  cargo run --release --bin demag_dst_vs_fft\n  cargo run --release --bin demag_dst_vs_fft -- <nx> <ny> <dx> <dy> <dz> [options]\n\nOptions:\n  --pattern random|smooth|wall|impulse\n  --seed <u64>\n  --wall-width-cells <f64>\n  --sweep          (resolution sweep at fixed physical size)\n  --diag           (print DST phase timings and intermediate norms)\n\nDefaults: nx=64, ny=64, dx=5e-9, dy=5e-9, dz=1e-9, pattern=random"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Defaults
    let mut nx: usize = 64;
    let mut ny: usize = 64;
    let mut dx: f64 = 5e-9;
    let mut dy: f64 = 5e-9;
    let mut dz: f64 = 1e-9;

    let mut pattern = Pattern::Random;
    let mut seed: u64 = 0x1234_5678_9abc_def0;
    let mut wall_width_cells: f64 = 8.0;
    let mut sweep: bool = false;
    let mut diag: bool = false;

    // Optional positional grid args: <nx> <ny> <dx> <dy> <dz>
    let mut i = 1usize;
    if args.len() >= 6 {
        let try_nx = args[1].parse::<usize>();
        let try_ny = args[2].parse::<usize>();
        let try_dx = args[3].parse::<f64>();
        let try_dy = args[4].parse::<f64>();
        let try_dz = args[5].parse::<f64>();
        if let (Ok(vnx), Ok(vny), Ok(vdx), Ok(vdy), Ok(vdz)) =
            (try_nx, try_ny, try_dx, try_dy, try_dz)
        {
            nx = vnx;
            ny = vny;
            dx = vdx;
            dy = vdy;
            dz = vdz;
            i = 6;
        }
    }

    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => print_help_and_exit(),
            "--pattern" => {
                i += 1;
                if i >= args.len() {
                    print_help_and_exit();
                }
                pattern = Pattern::from_str(&args[i]).unwrap_or_else(|| {
                    eprintln!("Unknown --pattern {}", args[i]);
                    print_help_and_exit();
                });
            }
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    print_help_and_exit();
                }
                seed = args[i].parse::<u64>().unwrap_or_else(|_| {
                    eprintln!("Invalid --seed {}", args[i]);
                    print_help_and_exit();
                });
            }
            "--wall-width-cells" => {
                i += 1;
                if i >= args.len() {
                    print_help_and_exit();
                }
                wall_width_cells = args[i].parse::<f64>().unwrap_or_else(|_| {
                    eprintln!("Invalid --wall-width-cells {}", args[i]);
                    print_help_and_exit();
                });
            }
            "--sweep" => {
                sweep = true;
            }
            "--diag" => {
                diag = true;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_help_and_exit();
            }
        }
        i += 1;
    }

    if !sweep {
        run_once(nx, ny, dx, dy, dz, pattern, seed, wall_width_cells, diag);
        return;
    }

    let lx = (nx as f64) * dx;
    let ly = (ny as f64) * dy;

    let cases = [(32usize, 32usize), (64usize, 64usize), (128usize, 128usize)];

    println!(
        "Resolution sweep at fixed size: Lx={:.3e} m, Ly={:.3e} m",
        lx, ly
    );
    println!("Pattern: {}", pattern.as_str());
    println!("--------------------------------------------");

    for (nx_i, ny_i) in cases {
        let dx_i = lx / (nx_i as f64);
        let dy_i = ly / (ny_i as f64);
        run_once(nx_i, ny_i, dx_i, dy_i, dz, pattern, seed, wall_width_cells, diag);
        println!("--------------------------------------------");
    }
}

// ---------------------------------------------------------------------------
// Single run
// ---------------------------------------------------------------------------

fn run_once(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    pattern: Pattern,
    seed: u64,
    wall_width_cells: f64,
    diag: bool,
) {
    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    let mut m = VectorField2D::new(grid);
    match pattern {
        Pattern::Random => init_random_unit_vectors(&mut m, seed),
        Pattern::Smooth => init_smooth_mode(&mut m),
        Pattern::Wall => init_bloch_wall_y(&mut m, wall_width_cells),
        Pattern::Impulse => init_impulse_mz(&mut m),
    }

    let ms = 8.0e5;

    let mat = Material {
        ms,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let mut b_fft = VectorField2D::new(grid);
    let mut b_dst = VectorField2D::new(grid);

    // Warm-up (untimed): ensures caches are built.
    demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_fft, &mat);
    demag_poisson_dst::compute_demag_field_poisson_dst(&grid, &m, &mut b_dst, &mat);

    // Timed (steady-state)
    let t0 = Instant::now();
    demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_fft, &mat);
    let t_fft = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    demag_poisson_dst::compute_demag_field_poisson_dst(&grid, &m, &mut b_dst, &mat);
    let t_dst = t1.elapsed().as_secs_f64();

    // --- Diagnostics (optional) ---
    if diag {
        let mut b_diag = VectorField2D::new(grid);
        let dd = demag_poisson_dst::solve_with_diagnostics(&grid, &m, &mat, &mut b_diag);

        let w_max = dd.w_interior.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let g_max = dd.g_boundary.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let v_max = dd.v_boundary.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let u_max = dd.u_nodes.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

        let g_rms = (dd.g_boundary.iter().map(|v| v * v).sum::<f64>()
            / dd.g_boundary.len().max(1) as f64)
            .sqrt();

        println!();
        println!("DST phase diagnostics:");
        println!(
            "  w-solve      : {:.3} ms   |w|_max = {:.6e}",
            dd.timings_ms[0], w_max
        );
        println!(
            "  boundary int : {:.3} ms   |g|_max = {:.6e}   g_rms = {:.6e}   |v_bdy|_max = {:.6e}",
            dd.timings_ms[1], g_max, g_rms, v_max
        );
        println!(
            "  v-solve      : {:.3} ms",
            dd.timings_ms[2]
        );
        println!(
            "  gradient     : {:.3} ms   |U|_max = {:.6e}",
            dd.timings_ms[3], u_max
        );
        println!(
            "  total        : {:.3} ms",
            dd.timings_ms[4]
        );
        println!();
    }

    // --- Error metrics ---
    let (rmse, max_abs, rel_rmse) = field_error_metrics(&b_fft, &b_dst);
    let avg_fft = mean_b(&b_fft);
    let avg_dst = mean_b(&b_dst);

    let dmean = [
        avg_dst[0] - avg_fft[0],
        avg_dst[1] - avg_fft[1],
        avg_dst[2] - avg_fft[2],
    ];
    let dmean_mag = (dmean[0] * dmean[0] + dmean[1] * dmean[1] + dmean[2] * dmean[2]).sqrt();

    let rms_fft = {
        let mut sum = 0.0f64;
        for v in &b_fft.data {
            sum += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        }
        (sum / (b_fft.data.len() as f64)).sqrt().max(1e-30)
    };
    let mean_bias_rel = dmean_mag / rms_fft;

    let e_fft = demag_energy(&grid, &m, &b_fft, ms);
    let e_dst = demag_energy(&grid, &m, &b_dst, ms);
    let de = e_dst - e_fft;
    let rel_e = if e_fft.abs() > 0.0 {
        de / e_fft
    } else {
        f64::NAN
    };

    // --- Print ---
    println!(
        "Grid: {}x{}, dx={:.3e}, dy={:.3e}, dz={:.3e}",
        nx, ny, dx, dy, dz
    );
    println!("Ms = {:.3e} A/m, mu0 = {:.6e}", ms, MU0);
    println!("Pattern: {}", pattern.as_str());
    println!();
    println!("Timing (steady-state field eval; after 1 warm-up call):");
    println!("  FFT  : {:.6} s", t_fft);
    println!("  DST  : {:.6} s", t_dst);
    println!(
        "  ratio: {:.2}x",
        if t_fft > 0.0 { t_dst / t_fft } else { f64::NAN }
    );
    println!();
    println!("Means <B> (Tesla):");
    println!(
        "  FFT  : [{:+.6e}, {:+.6e}, {:+.6e}]",
        avg_fft[0], avg_fft[1], avg_fft[2]
    );
    println!(
        "  DST  : [{:+.6e}, {:+.6e}, {:+.6e}]",
        avg_dst[0], avg_dst[1], avg_dst[2]
    );
    println!(
        "  Δ<B> : [{:+.6e}, {:+.6e}, {:+.6e}]  (DST - FFT)",
        dmean[0], dmean[1], dmean[2]
    );
    println!("  |Δ<B>| / RMS(B_fft) = {:.6e}", mean_bias_rel);
    println!();
    println!("Error DST vs FFT:");
    println!("  RMSE(|dB|)     = {:.6e} T", rmse);
    println!("  max(|dB|)      = {:.6e} T", max_abs);
    println!("  rel_RMSE       = {:.6e} (vs RMS(FFT))", rel_rmse);
    println!();
    println!("Demag energy (MuMax-style: -1/2 ∫ M·B dV):");
    println!("  FFT  : {:.6e} J", e_fft);
    println!("  DST  : {:.6e} J", e_dst);
    println!("  ΔE   : {:.6e} J  (ΔE/E_fft = {:.3e})", de, rel_e);

    // --- Per-component breakdown ---
    println!();
    println!("Per-component RMSE (Tesla):");
    let (rmse_x, rmse_y, rmse_z) = component_rmse(&b_fft, &b_dst);
    println!("  Bx: {:.6e}", rmse_x);
    println!("  By: {:.6e}", rmse_y);
    println!("  Bz: {:.6e}", rmse_z);

    // --- Spot-check centre cell ---
    let cx = nx / 2;
    let cy = ny / 2;
    let cid = cy * nx + cx;
    println!();
    println!("Centre cell ({},{}):", cx, cy);
    println!(
        "  B_fft = [{:+.6e}, {:+.6e}, {:+.6e}]",
        b_fft.data[cid][0], b_fft.data[cid][1], b_fft.data[cid][2]
    );
    println!(
        "  B_dst = [{:+.6e}, {:+.6e}, {:+.6e}]",
        b_dst.data[cid][0], b_dst.data[cid][1], b_dst.data[cid][2]
    );
    let db = [
        b_dst.data[cid][0] - b_fft.data[cid][0],
        b_dst.data[cid][1] - b_fft.data[cid][1],
        b_dst.data[cid][2] - b_fft.data[cid][2],
    ];
    println!(
        "  ΔB    = [{:+.6e}, {:+.6e}, {:+.6e}]",
        db[0], db[1], db[2]
    );
}

// ---------------------------------------------------------------------------
// Init patterns
// ---------------------------------------------------------------------------

fn init_random_unit_vectors(field: &mut VectorField2D, mut seed: u64) {
    for v in &mut field.data {
        let x = u01(&mut seed) * 2.0 - 1.0;
        let y = u01(&mut seed) * 2.0 - 1.0;
        let z = u01(&mut seed) * 2.0 - 1.0;
        let n = (x * x + y * y + z * z).sqrt().max(1e-30);
        v[0] = x / n;
        v[1] = y / n;
        v[2] = z / n;
    }
}

fn init_smooth_mode(field: &mut VectorField2D) {
    let nx = field.grid.nx as f64;
    let ny = field.grid.ny as f64;
    let two_pi = 2.0 * std::f64::consts::PI;

    for j in 0..field.grid.ny {
        for i in 0..field.grid.nx {
            let x = (i as f64 + 0.5) / nx;
            let y = (j as f64 + 0.5) / ny;

            let mut mx = (two_pi * x).cos();
            let mut my = (two_pi * y).sin();
            let mut mz = 0.2;

            let n = (mx * mx + my * my + mz * mz).sqrt().max(1e-30);
            mx /= n;
            my /= n;
            mz /= n;

            let id = field.idx(i, j);
            field.data[id][0] = mx;
            field.data[id][1] = my;
            field.data[id][2] = mz;
        }
    }
}

fn init_bloch_wall_y(field: &mut VectorField2D, wall_width_cells: f64) {
    let ny = field.grid.ny as f64;
    let y0 = 0.5 * ny;
    let delta = wall_width_cells.max(1.0);

    for j in 0..field.grid.ny {
        let y = j as f64 + 0.5;
        let t = (y - y0) / delta;

        let mz = t.tanh();
        let my = 1.0 / t.cosh();
        let mx = 0.0;

        for i in 0..field.grid.nx {
            let id = field.idx(i, j);
            field.data[id][0] = mx;
            field.data[id][1] = my;
            field.data[id][2] = mz;
        }
    }
}

fn init_impulse_mz(field: &mut VectorField2D) {
    for v in &mut field.data {
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = 0.0;
    }
    let ic = field.grid.nx / 2;
    let jc = field.grid.ny / 2;
    let id = field.idx(ic, jc);
    field.data[id][2] = 1.0;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

#[inline]
fn u01(seed: &mut u64) -> f64 {
    let mut x = *seed;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *seed = x;
    let y = x.wrapping_mul(0x2545F4914F6CDD1D);
    let mant = (y >> 11) | 0x3FF0_0000_0000_0000;
    f64::from_bits(mant) - 1.0
}

fn field_error_metrics(b_ref: &VectorField2D, b_test: &VectorField2D) -> (f64, f64, f64) {
    assert_eq!(b_ref.data.len(), b_test.data.len());

    let mut sum_sq: f64 = 0.0;
    let mut max_abs: f64 = 0.0;
    let mut sum_ref_sq: f64 = 0.0;

    for (a, b) in b_ref.data.iter().zip(b_test.data.iter()) {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let dz = b[2] - a[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        sum_sq += d * d;
        max_abs = max_abs.max(d);

        let ra = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        sum_ref_sq += ra * ra;
    }

    let n = b_ref.data.len() as f64;
    let rmse = (sum_sq / n).sqrt();
    let ref_rms = (sum_ref_sq / n).sqrt().max(1e-30);
    (rmse, max_abs, rmse / ref_rms)
}

fn component_rmse(b_ref: &VectorField2D, b_test: &VectorField2D) -> (f64, f64, f64) {
    assert_eq!(b_ref.data.len(), b_test.data.len());
    let n = b_ref.data.len() as f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sz = 0.0f64;

    for (a, b) in b_ref.data.iter().zip(b_test.data.iter()) {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let dz = b[2] - a[2];
        sx += dx * dx;
        sy += dy * dy;
        sz += dz * dz;
    }
    ((sx / n).sqrt(), (sy / n).sqrt(), (sz / n).sqrt())
}

fn mean_b(b: &VectorField2D) -> [f64; 3] {
    let mut s = [0.0; 3];
    for v in &b.data {
        s[0] += v[0];
        s[1] += v[1];
        s[2] += v[2];
    }
    let n = b.data.len() as f64;
    [s[0] / n, s[1] / n, s[2] / n]
}

fn demag_energy(grid: &Grid2D, m: &VectorField2D, b: &VectorField2D, ms: f64) -> f64 {
    let v = grid.cell_volume();
    let mut e = 0.0;
    for (mi, bi) in m.data.iter().zip(b.data.iter()) {
        let mdotb = mi[0] * bi[0] + mi[1] * bi[1] + mi[2] * bi[2];
        e += -0.5 * ms * mdotb * v;
    }
    e
}