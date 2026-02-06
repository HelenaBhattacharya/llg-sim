// src/bin/st_problems/sp2.rs
//
// Standard Problem #2 (MuMag): remanence and coercivity vs d/lex.
//
// Reference: MuMax3 paper, Appendix A script “Standard Problem 2 (Figs. 13 and 14)”.
// (Output columns mirror MuMax: d, remanence.x, remanence.y, Hc/Ms)
//
// Run (Rust):
//   cargo run --release --bin st_problems -- sp2
//
// Optional timing (prints per-relax call timing):
//   SP2_TIMING=1 cargo run --release --bin st_problems -- sp2
//
// Optional fast profiling mode (looser relax settings):
//   SP2_FAST=1 SP2_TIMING=1 cargo run --release --bin st_problems -- sp2
//
// Outputs (Rust):
//   runs/st_problems/sp2/table.csv
//
// Post-process (MuMax overlay plots):
//   python3 scripts/compare_sp2.py \
//     --mumax-table mumax_outputs/st_problems/sp2/sp2_out/table.txt \
//     --rust-table  runs/st_problems/sp2/table.csv \
//     --out-dir     out/st_problems/sp2
//
// IMPORTANT (practical note):
// SP2 is extremely expensive on CPU because it repeatedly relaxes a demag-dominated system.
// This file includes "debug-fast" knobs to make it runnable. Tighten back later.

use std::collections::HashSet;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use llg_sim::effective_field::FieldMask;
use llg_sim::grid::Grid2D;
use llg_sim::llg::RK23Scratch;
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material, MU0};
use llg_sim::relax::{relax, RelaxSettings};
use llg_sim::vec3::normalize;
use llg_sim::vector_field::VectorField2D;

// -------------------------
// Debug-fast knobs
// -------------------------

// Start with a small subset so you can see it completing.
// Set to (1, 30) when runtime is under control.
const D_MIN: usize = 30;
const D_MAX: usize = 30;

// SP2 coercivity bracketing/bisection:
// - MuMax step is 0.00005*Ms. We'll use a coarse bracket step (e.g. 20×) then bisect.
const BC_BRACKET_MULT: f64 = 20.0; // coarse step multiplier for bracketing
const BC_TARGET_MULT: f64 = 1.0;   // target resolution multiplier (1.0 => MuMax resolution)

// Relax effort caps per call (prevents "hangs" from very strict tightening on CPU)
const MAX_ACCEPTED_STEPS_REMANENCE: usize = 80_000;
const MAX_ACCEPTED_STEPS_COERCIVITY: usize = 30_000;

// Baseline relax settings (close-ish to MuMax but workable on CPU)
const RELAX_TIGHTEN_FLOOR: f64 = 1e-6;
const TORQUE_CHECK_STRIDE: usize = 200;

// Print progress:
const BRACKET_PRINT_EVERY: usize = 1;
const BISECT_PRINT_EVERY: usize = 1;

// Remanence cache location
const REM_CACHE_DIR: &str = "runs/st_problems/sp2/cache";

// -------------------------
// Helpers
// -------------------------

fn sp2_timing_enabled() -> bool {
    std::env::var("SP2_TIMING").is_ok()
}

fn sp2_fast_enabled() -> bool {
    std::env::var("SP2_FAST").is_ok()
}

fn ilogb_sp2(x: f64) -> i32 {
    if x <= 0.0 {
        return 0;
    }
    (x.log2().floor() as i32) + 1
}

fn avg_m(field: &VectorField2D) -> [f64; 3] {
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

fn read_done_d_values(table_path: &Path) -> std::io::Result<HashSet<i32>> {
    let mut done = HashSet::new();
    if !table_path.exists() {
        return Ok(done);
    }
    let f = File::open(table_path)?;
    let mut r = BufReader::new(f);
    let mut line = String::new();

    // header (ignore)
    let _ = r.read_line(&mut line)?;
    line.clear();

    while r.read_line(&mut line)? > 0 {
        let s = line.trim();
        if s.is_empty() {
            line.clear();
            continue;
        }
        if let Some(first) = s.split(',').next() {
            if let Ok(dv) = first.trim().parse::<i32>() {
                done.insert(dv);
            }
        }
        line.clear();
    }
    Ok(done)
}

// -------------------------
// Remanence cache (binary)
// -------------------------

#[derive(Clone, Copy)]
struct RemCacheHeader {
    magic: [u8; 8], // b"SP2REM\0\0"
    version: u32,   // 1
    d_lex: u32,
    nx: u32,
    ny: u32,
    dx: f64,
    dy: f64,
    dz: f64,
}

impl RemCacheHeader {
    fn new(d_lex: usize, g: &Grid2D) -> Self {
        Self {
            magic: *b"SP2REM\0\0",
            version: 1,
            d_lex: d_lex as u32,
            nx: g.nx as u32,
            ny: g.ny as u32,
            dx: g.dx,
            dy: g.dy,
            dz: g.dz,
        }
    }

    fn matches(&self, d_lex: usize, g: &Grid2D) -> bool {
        if self.magic != *b"SP2REM\0\0" || self.version != 1 {
            return false;
        }
        if self.d_lex != d_lex as u32 || self.nx != g.nx as u32 || self.ny != g.ny as u32 {
            return false;
        }
        // tolerate tiny float differences
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
        rel(self.dx, g.dx) < 1e-12 && rel(self.dy, g.dy) < 1e-12 && rel(self.dz, g.dz) < 1e-12
    }
}

fn rem_cache_path(d_lex: usize, g: &Grid2D) -> PathBuf {
    // include geometry in name for easy debugging/cleanup
    let fname = format!(
        "rem_d{}_nx{}_ny{}_dx{:.3e}_dy{:.3e}_dz{:.3e}.bin",
        d_lex, g.nx, g.ny, g.dx, g.dy, g.dz
    );
    Path::new(REM_CACHE_DIR).join(fname)
}

fn write_rem_cache(path: &Path, header: RemCacheHeader, m: &VectorField2D) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    let mut f = BufWriter::new(File::create(path)?);

    // header
    f.write_all(&header.magic)?;
    f.write_all(&header.version.to_le_bytes())?;
    f.write_all(&header.d_lex.to_le_bytes())?;
    f.write_all(&header.nx.to_le_bytes())?;
    f.write_all(&header.ny.to_le_bytes())?;
    f.write_all(&header.dx.to_le_bytes())?;
    f.write_all(&header.dy.to_le_bytes())?;
    f.write_all(&header.dz.to_le_bytes())?;

    // payload: m vectors
    for v in &m.data {
        f.write_all(&v[0].to_le_bytes())?;
        f.write_all(&v[1].to_le_bytes())?;
        f.write_all(&v[2].to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

fn read_rem_cache(path: &Path, d_lex: usize, g: &Grid2D) -> std::io::Result<Option<VectorField2D>> {
    if !path.exists() {
        return Ok(None);
    }
    let mut f = BufReader::new(File::open(path)?);

    // read header
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;

    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf4)?;
    let d_h = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf4)?;
    let nx_h = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf4)?;
    let ny_h = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf8)?;
    let dx_h = f64::from_le_bytes(buf8);

    f.read_exact(&mut buf8)?;
    let dy_h = f64::from_le_bytes(buf8);

    f.read_exact(&mut buf8)?;
    let dz_h = f64::from_le_bytes(buf8);

    let hdr = RemCacheHeader {
        magic,
        version,
        d_lex: d_h,
        nx: nx_h,
        ny: ny_h,
        dx: dx_h,
        dy: dy_h,
        dz: dz_h,
    };

    if !hdr.matches(d_lex, g) {
        return Ok(None);
    }

    let mut out = VectorField2D::new(*g);
    for i in 0..out.data.len() {
        let mut b = [0u8; 8];
        f.read_exact(&mut b)?;
        let x = f64::from_le_bytes(b);
        f.read_exact(&mut b)?;
        let y = f64::from_le_bytes(b);
        f.read_exact(&mut b)?;
        let z = f64::from_le_bytes(b);
        out.data[i] = [x, y, z];
    }

    Ok(Some(out))
}

// -------------------------
// Relax wrappers
// -------------------------

fn relax_call(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    max_steps: usize,
) {
    let fast = sp2_fast_enabled();

    // Fresh relax settings each call (MuMax-like baseline, with optional fast overrides)
    let mut settings = RelaxSettings {
        torque_threshold: Some(1e-4),
        torque_check_stride: TORQUE_CHECK_STRIDE,
        tighten_floor: RELAX_TIGHTEN_FLOOR,
        max_accepted_steps: max_steps,
        ..Default::default()
    };

    if fast {
        // Profiling mode: reduce work per relax call.
        settings.torque_threshold = Some(5e-4);
        settings.tighten_floor = 1e-5;
        settings.torque_check_stride = 1000;
        settings.max_err = 2e-5;
    }

    // Reset dt guess each relax call (MuMax-like)
    params.dt = 1e-13;

    let do_timing = sp2_timing_enabled();
    let t0 = Instant::now();

    relax(
        grid,
        m,
        params,
        material,
        rk23,
        FieldMask::Full,
        &mut settings,
    );

    if do_timing {
        println!(
            "      [sp2 timing] relax(max_steps={}, fast={}) took {:.3}s",
            max_steps,
            if fast { 1 } else { 0 },
            t0.elapsed().as_secs_f64()
        );
    }
}

fn set_bext_from_bc(params: &mut LLGParams, bc_amps_per_m: f64) {
    // Apply B_ext (Tesla) along (-1,-1,-1)/sqrt(3), matching MuMax
    let b = -bc_amps_per_m * MU0 / 3.0_f64.sqrt();
    params.b_ext = [b, b, b];
}

fn relax_and_msum_in_place(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    bc_amps_per_m: f64,
) -> f64 {
    set_bext_from_bc(params, bc_amps_per_m);

    relax_call(
        grid,
        m,
        params,
        material,
        rk23,
        MAX_ACCEPTED_STEPS_COERCIVITY,
    );

    let mm = avg_m(m);
    mm[0] + mm[1] + mm[2]
}

// Coercivity: continuation bracket scan + bisection from last-positive relaxed state.
fn find_hc_over_ms(
    grid: &Grid2D,
    m_work: &mut VectorField2D,
    m_rem: &VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    ms: f64,
) -> f64 {
    let bc0 = 0.0445 * ms; // MuMax starting point
    let bc_target_step = (0.00005 * BC_TARGET_MULT) * ms;
    let bc_bracket_step = (0.00005 * BC_BRACKET_MULT) * ms;

    // Start from remanent state once
    m_work.data.clone_from(&m_rem.data);

    // Evaluate at start
    let mut bc_low = bc0;
    let mut s_low = relax_and_msum_in_place(grid, m_work, params, material, rk23, bc_low);

    if s_low <= 0.0 {
        return bc_low / ms;
    }

    // Save last-positive relaxed state
    let mut m_low_state = VectorField2D::new(*grid);
    m_low_state.data.clone_from(&m_work.data);

    // Bracket upward (continuation)
    let mut bc_high = bc_low;
    let mut s_high = s_low;
    let mut k = 0usize;

    while s_high > 0.0 && bc_high <= 0.2 * ms {
        bc_high += bc_bracket_step;
        k += 1;

        // continuation: do not reset m_work
        s_high = relax_and_msum_in_place(grid, m_work, params, material, rk23, bc_high);

        if BRACKET_PRINT_EVERY > 0 && k % BRACKET_PRINT_EVERY == 0 {
            println!(
                "    [bracket] k={:>3} bc/Ms={:.6}  <m_sum>={:.6}",
                k,
                bc_high / ms,
                s_high
            );
        }

        if s_high > 0.0 {
            bc_low = bc_high;
            s_low = s_high;
            m_low_state.data.clone_from(&m_work.data);
        }
    }

    if s_high > 0.0 {
        println!("    WARNING: failed to bracket coercivity before bc cap; returning bc_high/Ms.");
        return bc_high / ms;
    }

    // Bisection: start each midpoint from last-positive state
    let mut iter = 0usize;
    while (bc_high - bc_low) > bc_target_step && iter < 60 {
        iter += 1;
        let bc_mid = 0.5 * (bc_low + bc_high);

        m_work.data.clone_from(&m_low_state.data);
        let s_mid = relax_and_msum_in_place(grid, m_work, params, material, rk23, bc_mid);

        if BISECT_PRINT_EVERY > 0 && iter % BISECT_PRINT_EVERY == 0 {
            println!(
                "    [bisect {:>2}] low={:.6} high={:.6} mid={:.6}  <m_sum>={:.6}",
                iter,
                bc_low / ms,
                bc_high / ms,
                bc_mid / ms,
                s_mid
            );
        }

        if s_mid > 0.0 {
            bc_low = bc_mid;
            s_low = s_mid;
            m_low_state.data.clone_from(&m_work.data);
        } else {
            bc_high = bc_mid;
        }
    }

    let _ = s_low;
    bc_high / ms
}

pub fn run_sp2() -> std::io::Result<()> {
    // MuMax script constants
    let ms: f64 = 1000e3;
    let a_ex: f64 = 10e-12;
    let k_u: f64 = 0.0;

    let lex: f64 = (2.0 * a_ex / (MU0 * ms * ms)).sqrt();

    println!("SP2: d range = [{}..{}]", D_MIN, D_MAX);
    println!("SP2: tighten_floor = {:.1e}", RELAX_TIGHTEN_FLOOR);
    println!("SP2: torque_check_stride = {}", TORQUE_CHECK_STRIDE);
    println!("SP2: max steps remanence  = {}", MAX_ACCEPTED_STEPS_REMANENCE);
    println!("SP2: max steps coercivity = {}", MAX_ACCEPTED_STEPS_COERCIVITY);
    println!("SP2: bracket mult = {}", BC_BRACKET_MULT);
    println!("SP2: target step mult = {}", BC_TARGET_MULT);
    println!("SP2: fast mode = {}", if sp2_fast_enabled() { "ON" } else { "OFF" });

    let out_dir = Path::new("runs").join("st_problems").join("sp2");
    create_dir_all(&out_dir)?;
    let table_path = out_dir.join("table.csv");

    // Resume support
    let done = read_done_d_values(&table_path)?;
    if !done.is_empty() {
        println!(
            "SP2: resuming; already have {} rows in {}",
            done.len(),
            table_path.display()
        );
    }

    let file_exists = table_path.exists();
    let f = OpenOptions::new().create(true).append(true).open(&table_path)?;
    let mut w = BufWriter::new(f);

    // Write header if file is new OR empty
    let need_header = !file_exists || std::fs::metadata(&table_path)?.len() == 0;
    if need_header {
        writeln!(w, "d_lex,mx_rem,my_rem,hc_over_ms")?;
        w.flush()?;
        println!("SP2: wrote header -> {}", table_path.display());
    }

    for d_int in (D_MIN..=D_MAX).rev() {
        if done.contains(&(d_int as i32)) {
            println!("SP2 d/lex={:>2}: already done; skipping.", d_int);
            continue;
        }

        let d = d_int as f64;

        let sizex = 5.0 * lex * d;
        let sizey = 1.0 * lex * d;
        let sizez = 0.1 * lex * d;

        let x = sizex / (5.0 * 0.5 * lex); // = 2d
        let p = ilogb_sp2(x);
        let nx: usize = (2usize.pow(p as u32)) * 5;
        let ny: usize = nx / 5;

        let dx = sizex / (nx as f64);
        let dy = sizey / (ny as f64);
        let dz = sizez;

        let grid = Grid2D::new(nx, ny, dx, dy, dz);

        println!(
            "\nSP2 d/lex={:>2.0}: start (nx={}, ny={}, dx/lex={:.3}, dy/lex={:.3}, dz/lex={:.3})",
            d,
            nx,
            ny,
            dx / lex,
            dy / lex,
            dz / lex
        );

        let material = Material {
            ms,
            a_ex,
            k_u,
            easy_axis: [0.0, 0.0, 1.0],
            dmi: None,
            demag: true,
        };

        let mut params = LLGParams {
            gamma: GAMMA_E_RAD_PER_S_T,
            alpha: 0.5,
            dt: 1e-13,
            b_ext: [0.0, 0.0, 0.0],
        };

        let mut rk23 = RK23Scratch::new(grid);

        // -------------------------
        // Remanence (cacheable)
        // -------------------------
        let cache_path = rem_cache_path(d_int, &grid);
        let m = if let Some(m_cached) = read_rem_cache(&cache_path, d_int, &grid)? {
            println!("SP2 d/lex={:>2}: remanence cache HIT -> {}", d_int, cache_path.display());
            m_cached
        } else {
            println!("SP2 d/lex={:>2}: remanence cache MISS -> relaxing", d_int);

            let mut m0_field = VectorField2D::new(grid);
            let m0 = normalize([1.0, 0.3, 0.0]);
            m0_field.set_uniform(m0[0], m0[1], m0[2]);

            let t_rem = Instant::now();
            relax_call(
                &grid,
                &mut m0_field,
                &mut params,
                &material,
                &mut rk23,
                MAX_ACCEPTED_STEPS_REMANENCE,
            );
            println!(
                "SP2 d/lex={:>2.0}: remanence relax done in {:.1}s",
                d,
                t_rem.elapsed().as_secs_f64()
            );

            let hdr = RemCacheHeader::new(d_int, &grid);
            write_rem_cache(&cache_path, hdr, &m0_field)?;
            println!("SP2 d/lex={:>2}: cached remanence -> {}", d_int, cache_path.display());

            m0_field
        };

        let rem = avg_m(&m);
        let mx_rem = rem[0];
        let my_rem = rem[1];
        let mz_rem = rem[2];
        let msum_rem = mx_rem + my_rem + mz_rem;
        println!(
            "SP2 d/lex={:>2.0}: rem(mx,my,mz,m_sum)=({:.6},{:.6},{:.6},{:.6})",
            d, mx_rem, my_rem, mz_rem, msum_rem
        );

        // -------------------------
        // Coercivity
        // -------------------------
        println!("SP2 d/lex={:>2.0}: coercivity search start", d);
        let m_rem = VectorField2D { grid, data: m.data.clone() };
        let mut m_work = VectorField2D::new(grid);

        let t_hc = Instant::now();
        let hc_over_ms = find_hc_over_ms(
            &grid,
            &mut m_work,
            &m_rem,
            &mut params,
            &material,
            &mut rk23,
            ms,
        );
        println!(
            "SP2 d/lex={:>2.0}: coercivity done in {:.1}s  hc/Ms={:.6}",
            d,
            t_hc.elapsed().as_secs_f64(),
            hc_over_ms
        );

        // Restore B_ext = 0 (matches MuMax script end-of-loop)
        params.b_ext = [0.0, 0.0, 0.0];

        writeln!(w, "{:.0},{:.16e},{:.16e},{:.16e}", d, mx_rem, my_rem, hc_over_ms)?;
        w.flush()?;
        println!("SP2 d/lex={:>2.0}: row written", d);
    }

    println!("\nSP2 complete. Output: {}", Path::new("runs/st_problems/sp2/table.csv").display());
    Ok(())
}