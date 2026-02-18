// src/bin/demag_fft_vs_mg.rs
//
// Compare demag field computed by:
//   (1) FFT convolution demag (uniform FD grid)
//   (2) Poisson + geometric multigrid demag (experimental)
//
// Usage:
//   cargo run --release --bin demag_fft_vs_mg
//   cargo run --release --bin demag_fft_vs_mg -- 64 64 5e-9 5e-9 1e-9
//
// Pattern options (diagnostic):
//   --pattern random      (default; high-k / worst-case)
//   --pattern smooth      (single Fourier-like mode; should be much easier for MG)
//   --pattern wall        (smooth Bloch wall along y; more “realistic” than random)
//   --pattern impulse     (single-cell Mz impulse; isolates near-field mismatch)
//
// Optional:
//   --seed <u64>          (only affects random)
//   --wall-width-cells <f64>  (only affects wall; default 8)
//   --sweep               (run resolution sweep at fixed physical size)
//   --bc-check            (run boundary condition self-test and exit)

use llg_sim::effective_field::{demag_fft_uniform, demag_poisson_mg};
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, MU0, Material};
use llg_sim::vector_field::VectorField2D;

use std::time::Instant;

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

fn print_help_and_exit() -> ! {
    eprintln!(
        "Usage:\n  cargo run --release --bin demag_fft_vs_mg\n  cargo run --release --bin demag_fft_vs_mg -- <nx> <ny> <dx> <dy> <dz> [--pattern random|smooth|wall|impulse] [--seed <u64>] [--wall-width-cells <f64>] [--sweep] [--bc-check]\n\nDefaults: nx=64 ny=64 dx=5e-9 dy=5e-9 dz=1e-9, pattern=random, seed=0x123456789abcdef0, wall-width-cells=8"
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
    let mut bc_check: bool = false;

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
            "--bc-check" => {
                bc_check = true;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_help_and_exit();
            }
        }
        i += 1;
    }

    if bc_check {
        run_bc_selftest();
        return;
    }

    if !sweep {
        run_once(nx, ny, dx, dy, dz, pattern, seed, wall_width_cells);
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
        run_once(nx_i, ny_i, dx_i, dy_i, dz, pattern, seed, wall_width_cells);
        println!("--------------------------------------------");
    }
}

fn run_once(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    pattern: Pattern,
    seed: u64,
    wall_width_cells: f64,
) {
    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    // Magnetisation pattern (diagnostic): default is random (high-k).
    let mut m = VectorField2D::new(grid);
    match pattern {
        Pattern::Random => init_random_unit_vectors(&mut m, seed),
        Pattern::Smooth => init_smooth_mode(&mut m),
        Pattern::Wall => init_bloch_wall_y(&mut m, wall_width_cells),
        Pattern::Impulse => init_impulse_mz(&mut m),
    }

    let ms = 8.0e5;

    let mat_fft = Material {
        ms,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let mat_mg = Material {
        demag_method: DemagMethod::PoissonMG,
        ..mat_fft
    };

    let mut b_fft = VectorField2D::new(grid);
    let mut b_mg = VectorField2D::new(grid);

    // Warm-up (untimed): ensures any kernel/ΔK caches are built before timing.
    demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_fft, &mat_fft);
    demag_poisson_mg::compute_demag_field_poisson_mg(&grid, &m, &mut b_mg, &mat_mg);

    // Timed (steady-state): measure only the compute cost, not cache construction.
    let t0 = Instant::now();
    demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_fft, &mat_fft);
    let t_fft = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    demag_poisson_mg::compute_demag_field_poisson_mg(&grid, &m, &mut b_mg, &mat_mg);
    let t_mg = t1.elapsed().as_secs_f64();

    let (rmse, max_abs, rel_rmse) = field_error_metrics(&b_fft, &b_mg);
    let avg_fft = mean_b(&b_fft);
    let avg_mg = mean_b(&b_mg);

    let dmean = [
        avg_mg[0] - avg_fft[0],
        avg_mg[1] - avg_fft[1],
        avg_mg[2] - avg_fft[2],
    ];
    let dmean_mag = (dmean[0] * dmean[0] + dmean[1] * dmean[1] + dmean[2] * dmean[2]).sqrt();
    // Use RMS(B_fft) rather than |<B_fft>|, since mean can be near-zero for symmetric states.
    let rms_fft = {
        let mut sum = 0.0f64;
        for v in &b_fft.data {
            sum += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        }
        (sum / (b_fft.data.len() as f64)).sqrt().max(1e-30)
    };
    let mean_bias_rel = dmean_mag / rms_fft;

    let e_fft = demag_energy(&grid, &m, &b_fft, ms);
    let e_mg = demag_energy(&grid, &m, &b_mg, ms);

    println!(
        "Grid: {}x{}, dx={:.3e}, dy={:.3e}, dz={:.3e}",
        nx, ny, dx, dy, dz
    );
    println!("Ms = {:.3e} A/m, mu0 = {:.6e}", ms, MU0);
    println!("Pattern: {}", pattern.as_str());
    println!();
    println!("Timing (steady-state field eval; after 1 warm-up call):");
    println!("  FFT  : {:.6} s", t_fft);
    println!("  MG   : {:.6} s", t_mg);
    println!();
    println!("Means <B> (Tesla):");
    println!(
        "  FFT  : [{:+.6e}, {:+.6e}, {:+.6e}]",
        avg_fft[0], avg_fft[1], avg_fft[2]
    );
    println!(
        "  MG   : [{:+.6e}, {:+.6e}, {:+.6e}]",
        avg_mg[0], avg_mg[1], avg_mg[2]
    );
    println!(
        "  Δ<B> : [{:+.6e}, {:+.6e}, {:+.6e}]  (MG - FFT)",
        dmean[0], dmean[1], dmean[2]
    );
    println!("  |Δ<B>| / RMS(B_fft) = {:.6e}", mean_bias_rel);
    println!();
    println!("Error MG vs FFT:");
    println!("  RMSE(|dB|)     = {:.6e} T", rmse);
    println!("  max(|dB|)      = {:.6e} T", max_abs);
    println!("  rel_RMSE       = {:.6e} (vs RMS(FFT))", rel_rmse);
    println!();
    println!("Demag energy (MuMax-style: -1/2 ∫ M·B dV):");
    println!("  FFT  : {:.6e} J", e_fft);
    println!("  MG   : {:.6e} J", e_mg);
}

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
    // Simple smooth Bloch wall with wall normal along +y.
    // m_z transitions from -1 to +1 across y, m_y supplies the transverse component.
    let ny = field.grid.ny as f64;
    let y0 = 0.5 * ny;
    let delta = wall_width_cells.max(1.0);

    for j in 0..field.grid.ny {
        let y = j as f64 + 0.5;
        let t = (y - y0) / delta;

        // tanh/sech profile
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
    // Single-cell impulse: Mz=+1 at centre cell, else 0.
    // This is intentionally non-unit everywhere except the centre.
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

#[inline]
fn u01(seed: &mut u64) -> f64 {
    // xorshift64*
    let mut x = *seed;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *seed = x;
    let y = x.wrapping_mul(0x2545F4914F6CDD1D);
    // Map top 53 bits to [0,1)
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

// --- Added boundary condition self-test code ---

fn run_bc_selftest() {
    // Read env vars with defaults
    let pad_xy = std::env::var("LLG_DEMAG_MG_PAD_XY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2);
    let n_vac_z = std::env::var("LLG_DEMAG_MG_NVAC_Z")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2);
    let theta = std::env::var("LLG_DEMAG_MG_TREE_THETA")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.6);
    let leaf = std::env::var("LLG_DEMAG_MG_TREE_LEAF")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let signed = std::env::var("LLG_DEMAG_BC_TEST_SIGNED")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(0);

    // Base domain
    let nx = 32;
    let ny = 32;
    let nz = 1;
    let dx = 5e-9;
    let dy = 5e-9;
    let dz = 1e-9;

    // Padded domain sizes
    let nx_pad = nx + 2 * pad_xy;
    let ny_pad = ny + 2 * pad_xy;
    let nz_pad = nz + 2 * n_vac_z;

    // Magnetic region box origin in padded domain
    let ox = pad_xy as f64 * dx;
    let oy = pad_xy as f64 * dy;
    let oz = n_vac_z as f64 * dz;

    // Generate many random point charges inside the magnetic region (nx*ny*nz)
    let nsrc = 512;
    let mut rng_seed = 0xD00D_F00D_1234_5678u64;
    let mut sources = Vec::with_capacity(nsrc);
    for _ in 0..nsrc {
        let x = ox + (u01(&mut rng_seed) * (nx as f64) * dx);
        let y = oy + (u01(&mut rng_seed) * (ny as f64) * dy);
        let z = oz + (u01(&mut rng_seed) * (nz as f64) * dz);
        let qmag = 1.0;
        let q = if signed != 0 {
            let qsign = if u01(&mut rng_seed) < 0.5 { -1.0 } else { 1.0 };
            qsign * qmag
        } else {
            1.0
        };
        sources.push(Src { x, y, z, q });
    }

    // Diagnostics: net_q and sum_abs_q
    let net_q: f64 = sources.iter().map(|s| s.q).sum();
    let sum_abs_q: f64 = sources.iter().map(|s| s.q.abs()).sum();
    println!(
        "  net_q = {:.6e} (net_q/sum_abs_q = {:.3e})",
        net_q,
        if sum_abs_q.abs() > 1e-30 {
            net_q / sum_abs_q
        } else {
            0.0
        }
    );

    // Boundary evaluation points: all cell centers on outer shell of padded box
    let mut bpoints = Vec::new();

    // Helper to add boundary points on a face
    let mut add_face_points = |ix_min, ix_max, iy_min, iy_max, iz_min, iz_max| {
        for iz in iz_min..=iz_max {
            for iy in iy_min..=iy_max {
                for ix in ix_min..=ix_max {
                    // Only points on the face (at least one coord at min or max)
                    if ix == ix_min
                        || ix == ix_max
                        || iy == iy_min
                        || iy == iy_max
                        || iz == iz_min
                        || iz == iz_max
                    {
                        let x = (ix as f64 + 0.5) * dx;
                        let y = (iy as f64 + 0.5) * dy;
                        let z = (iz as f64 + 0.5) * dz;
                        bpoints.push([x, y, z]);
                    }
                }
            }
        }
    };

    add_face_points(0, nx_pad - 1, 0, ny_pad - 1, 0, nz_pad - 1);

    // Build tree
    let tree = Tree::build(&sources, leaf);

    // Evaluate potentials direct and tree
    let mut phi_direct = Vec::with_capacity(bpoints.len());
    let mut phi_tree = Vec::with_capacity(bpoints.len());

    for &p in &bpoints {
        let pd = potential_direct(&sources, p);
        let pt = tree.potential(p, theta);
        phi_direct.push(pd);
        phi_tree.push(pt);
    }

    // Compute RMS and max absolute errors, and reference RMS for relative errors
    let mut sum_sq = 0.0;
    let mut max_abs = 0.0;
    let mut sum_sq_ref = 0.0;
    let n = phi_direct.len() as f64;

    for i in 0..phi_direct.len() {
        let d = phi_tree[i] - phi_direct[i];
        let ad = d.abs();
        sum_sq += d * d;
        if ad > max_abs {
            max_abs = ad;
        }
        let refval = phi_direct[i];
        sum_sq_ref += refval * refval;
    }
    let rms = (sum_sq / n).sqrt();
    let ref_rms = (sum_sq_ref / n).sqrt().max(1e-30);
    let rms_rel = rms / ref_rms;
    let max_rel = max_abs / ref_rms;

    println!("Boundary condition self-test:");
    println!(
        "  Padded domain sizes: nx={} ny={} nz={}",
        nx_pad, ny_pad, nz_pad
    );
    println!("  Tree parameters: theta = {:.3}, leaf = {}", theta, leaf);
    println!("  Number of sources: {}", nsrc);
    println!("  Number of boundary points: {}", bpoints.len());
    println!("  signed charges: {}", signed != 0);
    println!("  RMS(|dphi|) = {:.6e}", rms);
    println!("  max(|dphi|) = {:.6e}", max_abs);
    println!("  ref_RMS(|phi_direct|) = {:.6e}", ref_rms);
    println!("  rel_RMS(|dphi|) = {:.6e}", rms_rel);
    println!("  rel_max(|dphi|) = {:.6e}", max_rel);
    println!();
    println!(
        "Suggestion: Sweep LLG_DEMAG_MG_TREE_THETA=0.8,0.6,0.4 and/or LLG_DEMAG_MG_TREE_LEAF=64,32,16,8 and confirm rel_RMS decreases"
    );
}

#[derive(Clone, Copy)]
struct Src {
    x: f64,
    y: f64,
    z: f64,
    q: f64,
}

struct Node {
    half: f64,
    q: f64,
    comx: f64,
    comy: f64,
    comz: f64,
    children: [Option<usize>; 8],
    len: usize,
}

struct Tree {
    nodes: Vec<Node>,
    sources: Vec<Src>,
    leaf: usize,
}

impl Tree {
    fn build(sources: &[Src], leaf: usize) -> Self {
        let mut tree = Tree {
            nodes: Vec::new(),
            sources: sources.to_vec(),
            leaf,
        };
        // Compute bounding cube for all sources
        let (cx, cy, cz, half) = bounding_cube(&tree.sources);

        // Build root node
        tree.build_node(0, tree.sources.len(), cx, cy, cz, half, leaf);
        tree
    }

    fn build_node(
        &mut self,
        start: usize,
        len: usize,
        cx: f64,
        cy: f64,
        cz: f64,
        half: f64,
        leaf: usize,
    ) -> usize {
        // Create node with placeholder children
        let mut node = Node {
            half,
            q: 0.0,
            comx: 0.0,
            comy: 0.0,
            comz: 0.0,
            children: [None; 8],
            len,
        };

        // Compute total charge and center of mass
        let mut qsum = 0.0;
        let mut xsum = 0.0;
        let mut ysum = 0.0;
        let mut zsum = 0.0;

        for i in start..start + len {
            let s = self.sources[i];
            qsum += s.q;
            xsum += s.q * s.x;
            ysum += s.q * s.y;
            zsum += s.q * s.z;
        }

        if qsum.abs() < 1e-30 {
            // Avoid division by zero
            node.q = 0.0;
            node.comx = cx;
            node.comy = cy;
            node.comz = cz;
        } else {
            node.q = qsum;
            node.comx = xsum / qsum;
            node.comy = ysum / qsum;
            node.comz = zsum / qsum;
        }

        let node_idx = self.nodes.len();
        self.nodes.push(node);

        // If leaf node or no subdivision needed, return
        if len <= leaf {
            return node_idx;
        }

        // Subdivide sources into 8 octants
        let mut octants: [Vec<Src>; 8] = std::array::from_fn(|_| Vec::new());

        let h = half / 2.0;

        for i in start..start + len {
            let s = self.sources[i];
            let mut oct = 0;
            if s.x >= cx {
                oct |= 1;
            }
            if s.y >= cy {
                oct |= 2;
            }
            if s.z >= cz {
                oct |= 4;
            }
            octants[oct].push(s);
        }

        // Replace sources[start..start+len] with concatenated octants
        let mut idx = start;
        let mut children_indices = [None; 8];

        for oct in 0..8 {
            let octlen = octants[oct].len();
            if octlen == 0 {
                children_indices[oct] = None;
                continue;
            }

            // Replace sources slice
            for s in &octants[oct] {
                self.sources[idx] = *s;
                idx += 1;
            }

            // Compute child's center coords
            let child_cx = cx + if (oct & 1) != 0 { h } else { -h };
            let child_cy = cy + if (oct & 2) != 0 { h } else { -h };
            let child_cz = cz + if (oct & 4) != 0 { h } else { -h };

            // Build child node recursively
            let child_idx =
                self.build_node(idx - octlen, octlen, child_cx, child_cy, child_cz, h, leaf);
            children_indices[oct] = Some(child_idx);
        }

        self.nodes[node_idx].children = children_indices;

        node_idx
    }

    fn potential(&self, pos: [f64; 3], theta: f64) -> f64 {
        self.potential_node(0, pos, theta)
    }

    fn potential_node(&self, node_idx: usize, pos: [f64; 3], theta: f64) -> f64 {
        let node = &self.nodes[node_idx];
        // Use centre-of-mass distance for the multipole approximation.
        let dx = node.comx - pos[0];
        let dy = node.comy - pos[1];
        let dz = node.comz - pos[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-30);
        let s = node.half * 2.0;

        // Accept this node if it is a leaf OR it satisfies the Barnes–Hut opening criterion.
        if node.len <= self.leaf || s / dist < theta {
            return -node.q / (4.0 * std::f64::consts::PI * dist);
        }

        // Else recurse into children
        let mut pot = 0.0;
        for &child_opt in &node.children {
            if let Some(child_idx) = child_opt {
                pot += self.potential_node(child_idx, pos, theta);
            }
        }
        pot
    }
}

fn bounding_cube(sources: &[Src]) -> (f64, f64, f64, f64) {
    // Compute bounding box
    let mut xmin = std::f64::INFINITY;
    let mut ymin = std::f64::INFINITY;
    let mut zmin = std::f64::INFINITY;
    let mut xmax = std::f64::NEG_INFINITY;
    let mut ymax = std::f64::NEG_INFINITY;
    let mut zmax = std::f64::NEG_INFINITY;

    for s in sources {
        if s.x < xmin {
            xmin = s.x;
        }
        if s.y < ymin {
            ymin = s.y;
        }
        if s.z < zmin {
            zmin = s.z;
        }
        if s.x > xmax {
            xmax = s.x;
        }
        if s.y > ymax {
            ymax = s.y;
        }
        if s.z > zmax {
            zmax = s.z;
        }
    }

    let cx = 0.5 * (xmin + xmax);
    let cy = 0.5 * (ymin + ymax);
    let cz = 0.5 * (zmin + zmax);
    let mut half = 0.5 * ((xmax - xmin).max((ymax - ymin).max(zmax - zmin)));

    if half < 1e-30 {
        half = 1e-9;
    }

    (cx, cy, cz, half)
}

fn potential_direct(sources: &[Src], pos: [f64; 3]) -> f64 {
    let mut pot = 0.0;
    for s in sources {
        let dx = s.x - pos[0];
        let dy = s.y - pos[1];
        let dz = s.z - pos[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-30);
        pot += s.q / r;
    }
    -pot / (4.0 * std::f64::consts::PI)
}
