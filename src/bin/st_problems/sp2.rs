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
// Optional timing (prints per-equilibration timing):
//   SP2_TIMING=1 cargo run --release --bin st_problems -- sp2
//
// Optional fast profiling mode (looser settings):
//   SP2_FAST=1 SP2_TIMING=1 cargo run --release --bin st_problems -- sp2
//
// Outputs (Rust):
//   runs/st_problems/sp2/table.csv
//
// Post-process (MuMax overlay plots):
//
// python3 scripts/compare_sp2.py \
//   --mumax-root mumax_outputs/st_problems/sp2 \
//   --rust-root runs/st_problems/sp2 \
//   --metrics \
//   --out runs/st_problems/sp2/sp2_overlay.png
//
// IMPORTANT (practical note):
// SP2 is extremely expensive on CPU because it repeatedly equilibrates a demag-dominated system.
// This version introduces an SP2-scoped "minimize" (1 field build per iteration) and a two-tier
// equilibration strategy:
//   - Coarse tier: minimize only (fast) for bracket scanning
//   - Strict tier: minimize precondition + relax (robust) for remanence + bisection midpoints
//
// The demag implementation is NOT changed here; we only reduce how many times we call the full
// effective field build per equilibrium in SP2.

use std::collections::HashSet;
use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use llg_sim::effective_field::FieldMask;
use llg_sim::grid::Grid2D;
use llg_sim::llg::RK23Scratch;
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, MU0, Material};
use llg_sim::relax::{RelaxSettings, TorqueMetric, RelaxReport, RelaxStopReason, relax_with_report};

use llg_sim::vec3::normalize;
use llg_sim::vector_field::VectorField2D;

// New: SP2-only minimiser (one field build per iter)
// Requires: src/minimize.rs + pub mod minimize; in lib.rs
use llg_sim::minimize::{MinimizeReport, MinimizeSettings, minimize_damping_only};

// -------------------------
// Debug-fast knobs
// -------------------------

// Start with a small subset so you can see it completing.
// Set to (1, 30) when runtime is under control.
const D_MIN: usize = 1;
const D_MAX: usize = 30;

// SP2 coercivity bracketing/bisection:
// - MuMax step is 0.00005*Ms. We'll use a coarse bracket step (e.g. 20×) then bisect.
const BC_BRACKET_MULT: f64 = 20.0; // coarse step multiplier for bracketing
const BC_TARGET_MULT: f64 = 1.0; // target resolution multiplier (1.0 => MuMax resolution)

// Relax effort caps per call (prevents "hangs" from very strict tightening on CPU)
const MAX_ACCEPTED_STEPS_REMANENCE: usize = 80_000;
const MAX_ACCEPTED_STEPS_COERCIVITY: usize = 30_000;

// (removed unused constants for slimness)

// Baseline relax settings (close-ish to MuMax but workable on CPU)
const RELAX_TIGHTEN_FLOOR: f64 = 1e-6;
const TORQUE_CHECK_STRIDE: usize = 200;

// Minimize caps (one B_eff build per iter; cheap relative to RK23+demag per stage)
const MIN_ITERS_REMANENCE: usize = 30_000;
const MIN_ITERS_COERCIVITY: usize = 20_000;

// Torque thresholds for minimize tiers (Tesla).
// Coarse tier is used for bracketing; strict tier is used as a precondition before relax.
const MIN_TAU_COARSE: f64 = 6e-4;
const MIN_TAU_STRICT: f64 = 4e-4;

// (removed unused progress print constants)

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
// Equilibration (two-tier)
// -------------------------

fn set_bext_from_bc(params: &mut LLGParams, bc_amps_per_m: f64) {
    // Apply B_ext (Tesla) along (-1,-1,-1)/sqrt(3), matching MuMax
    let b = -bc_amps_per_m * MU0 / 3.0_f64.sqrt();
    params.b_ext = [b, b, b];
}

fn minimize_call(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &LLGParams,
    material: &Material,
    max_iters: usize,
    tau: f64,
    label: &str,
) -> MinimizeReport {
    let fast = sp2_fast_enabled();

    let mut s = MinimizeSettings {
        torque_threshold: tau,
        max_iters,
        ..Default::default()
    };

    if fast {
        // In fast mode, reduce work further (coarser convergence is ok for profiling)
        s.torque_threshold = (tau * 2.0).min(2e-3);
        s.max_iters = (max_iters / 2).max(1_000);
        s.lambda0 = 2e-2;
        s.lambda_max = 7e-2;
        s.stall_iters = (s.stall_iters / 2).max(50);
    }

    // For strict-ish contexts, prevent the minimizer from reaching tiny max_dM simply because
    // lambda has collapsed. This makes dm_converged more meaningful and improves agreement
    // with MuMax equilibria without forcing global changes.
    let strictish = label.starts_with("bc0/")
        || label.contains("verify_strict")
        || label.contains("low_strict")
        || label.starts_with("bisect/")
        || label.starts_with("remanence/");
    if strictish {
        s.lambda_min = 5e-5; // allow a bit more motion per iter (tunable)
        s.dm_stop = Some(5e-7); // stricter than default 1e-6 (tunable)
        s.dm_samples = 10; // keep MuMax-like smoothing
    }

    let do_timing = sp2_timing_enabled();
    let t0 = Instant::now();

    let rep = minimize_damping_only(grid, m, params, material, FieldMask::Full, &s);

    if do_timing {
        println!(
            "      [sp2 timing] minimize({}) iters={} conv={} dm_conv={} max_dm={:.3e} stalled={} tmax={:.3e} took {:.3}s",
            label,
            rep.iters,
            if rep.converged { 1 } else { 0 },
            if rep.dm_converged { 1 } else { 0 },
            rep.final_max_dm,
            if rep.stalled { 1 } else { 0 },
            rep.final_torque,
            t0.elapsed().as_secs_f64()
        );
    } else {
        // Light-touch info when not timing (useful while developing)
        if rep.stalled {
            println!(
                "      [minimize] {} stalled (iters={}, tmax={:.3e})",
                label, rep.iters, rep.final_torque
            );
        }
    }

    rep
}

fn relax_call(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    max_steps: usize,
    label: &str,
) -> RelaxReport {
    let fast = sp2_fast_enabled();

    // Fresh relax settings each call (MuMax-like baseline, with optional fast overrides)
    let mut settings = RelaxSettings {
        torque_threshold: Some(1e-4),
        torque_check_stride: TORQUE_CHECK_STRIDE,
        tighten_floor: RELAX_TIGHTEN_FLOOR,
        max_accepted_steps: max_steps,
        ..Default::default()
    };

    // SP2-only: default to skipping Phase 1 for speed.
    // For remanence, re-enable Phase 1 (MuMax energy-first behaviour) and disable plateau
    // stopping to better recover the symmetric remanent state (my ~ 0) in the sensitive range.
    settings.phase2_enabled = true;

    if label.starts_with("remanence/") {
        settings.phase1_enabled = true;
        settings.torque_metric = TorqueMetric::Max;
        settings.torque_plateau_checks = 0; // disable plateau for remanence robustness
        settings.torque_threshold = Some(1e-4);
    } else {
        settings.phase1_enabled = false;

        // Coercivity branch-following uses three modes:
        // - hc/coarse: cheap plateau settle used during coarse bracketing
        // - hc/fine:   slightly tighter plateau settle used during fine scan inside the bracket
        // - hc/strict: escalation only if coarse/fine hit the accepted-step cap
        if label == "hc/coarse" {
            settings.torque_metric = TorqueMetric::Mean;
            settings.torque_threshold = None;
            settings.torque_plateau_checks = 3;
            settings.torque_plateau_rel = 5e-3;      // plateau sooner (0.5% improvement required)
            settings.torque_plateau_min_checks = 3;

            settings.max_err = 5e-5;                 // larger steps
            settings.tighten_floor = 5e-5;           // single stage
            settings.max_accepted_steps = max_steps; // use caller cap
            settings.torque_check_stride = 200;
        } else if label == "hc/fine" {
            settings.torque_metric = TorqueMetric::Mean;
            settings.torque_threshold = None;
            settings.torque_plateau_checks = 4;
            settings.torque_plateau_rel = 2e-3;      // a bit tighter
            settings.torque_plateau_min_checks = 3;

            settings.max_err = 2e-5;
            settings.tighten_floor = 2e-5;
            settings.max_accepted_steps = max_steps;
            settings.torque_check_stride = 200;
        } else if label == "hc/strict" {
            // Escalation: keep plateau stopping, but smaller error and larger cap.
            // Avoid hard torque thresholds here, since they often force max-step termination.
            settings.torque_metric = TorqueMetric::Mean;
            settings.torque_threshold = None;
            settings.torque_plateau_checks = 6;
            settings.torque_plateau_rel = 1e-3;
            settings.torque_plateau_min_checks = 4;

            settings.max_err = 1e-5;
            settings.tighten_floor = RELAX_TIGHTEN_FLOOR;
            settings.max_accepted_steps = max_steps;
            settings.torque_check_stride = TORQUE_CHECK_STRIDE;
        } else {
            settings.torque_metric = TorqueMetric::Mean;

            // Plateau-only by default for other (non-remanence) relax calls.
            settings.torque_threshold = None;

            settings.torque_plateau_checks = 8;
            settings.torque_plateau_rel = 1e-3;
            settings.torque_plateau_min_checks = 5;
        }
    }

    // For bisection relax calls, prefer plateau-only stopping (no hard torque threshold)
    // to avoid wasting time chasing a strict absolute torque level.
    if label.starts_with("bisect/") {
        settings.torque_threshold = None;
    }

    if fast {
        // Profiling mode: reduce work per relax call.
        settings.torque_threshold = Some(5e-4);
        settings.tighten_floor = 1e-5;
        settings.torque_check_stride = 1000;
        settings.max_err = 2e-5;
    }

    // Reset dt guess for remanence and other one-off relax calls.
    // For hc_scan/hc continuation, keep the dt warm-start to avoid tiny-step re-ramping each field step.
    if !(label.starts_with("hc/") || label.starts_with("hc_scan/")) {
        params.dt = 1e-13;
    }

    let do_timing = sp2_timing_enabled();
    let t0 = Instant::now();

    let rep = relax_with_report(
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
            "      [sp2 timing] relax({}) max_steps={} stop={:?} took {:.3}s",
            label,
            settings.max_accepted_steps,
            rep.stop_reason,
            t0.elapsed().as_secs_f64()
        );
    }
    rep
}

/// One hysteresis continuation step at a given bc.
/// Performs a short minimization to stay on the metastable branch, then a bounded plateau relax.
fn hysteresis_step(
    grid: &Grid2D,
    m_work: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    bc: f64,
    mode_label: &str,
    relax_cap: usize,
) -> f64 {
    set_bext_from_bc(params, bc);

    // Coarse focusing: keep us on the same metastable branch.
    let pre_iters = if mode_label == "hc/coarse" { 200 } else { 300 };
    let _ = minimize_call(
        grid,
        m_work,
        params,
        material,
        pre_iters,
        MIN_TAU_COARSE,
        "hc/premin",
    );

    // Fine focusing: plateau relax. Escalate once if we hit the cap.
    let rep1 = relax_call(grid, m_work, params, material, rk23, relax_cap, mode_label);
    if rep1.stop_reason == RelaxStopReason::MaxAcceptedSteps {
        let _ = relax_call(grid, m_work, params, material, rk23, 8000, "hc/strict");
    }

    let mm = avg_m(m_work);
    mm[0] + mm[1] + mm[2]
}

// Coercivity (branch-following hybrid):
// 1) Coarse bracket using larger field steps, but still doing a local settle at each step.
// 2) Fine scan inside the bracket using MuMax step size until first <m_sum> <= 0.
fn find_hc_over_ms_branch_following_hybrid(
    grid: &Grid2D,
    m_work: &mut VectorField2D,
    m_rem: &VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    ms: f64,
) -> f64 {
    let bc0 = 0.0445 * ms; // MuMax starting point
    let bc_step_fine = (0.00005 * BC_TARGET_MULT) * ms;
    let bc_step_coarse = (0.00005 * BC_BRACKET_MULT) * ms;

    // Start from the remanent state once
    m_work.data.clone_from(&m_rem.data);

    // --- Seed at bc0 (must be on the correct branch) ---
    let mut bc = bc0;
    let msum0 = hysteresis_step(grid, m_work, params, material, rk23, bc, "hc/fine", 2500);
    if sp2_timing_enabled() {
        println!("    [hc_seed] bc/Ms={:.6}  <m_sum>={:.6}", bc / ms, msum0);
    }
    if msum0 <= 0.0 {
        return bc / ms;
    }

    // Track last-positive bracket state
    let mut bc_low = bc;
    let mut m_low_state = VectorField2D::new(*grid);
    m_low_state.data.clone_from(&m_work.data);

    // --- Coarse bracket ---
    loop {
        bc += bc_step_coarse;
        if bc >= 0.2 * ms {
            println!("    WARNING: failed to bracket coercivity before bc cap; returning bc/Ms at cap.");
            return bc / ms;
        }

        let msum = hysteresis_step(grid, m_work, params, material, rk23, bc, "hc/coarse", 1500);
        if sp2_timing_enabled() {
            println!("    [hc_bracket] bc/Ms={:.6}  <m_sum>={:.6}", bc / ms, msum);
        }

        if msum > 0.0 {
            bc_low = bc;
            m_low_state.data.clone_from(&m_work.data);
            continue;
        }

        // First negative found: bc is bc_high.
        let bc_high = bc;

        // --- Fine scan inside bracket ---
        m_work.data.clone_from(&m_low_state.data);
        bc = bc_low;

        loop {
            bc += bc_step_fine;
            if bc > bc_high + 1e-12 {
                // Safety: should not happen, but avoids infinite loops.
                return bc_high / ms;
            }

            let msum_f = hysteresis_step(grid, m_work, params, material, rk23, bc, "hc/fine", 2500);
            if sp2_timing_enabled() {
                // light print every ~10 fine steps
                let k = ((bc - bc_low) / bc_step_fine).round() as i32;
                if k == 1 || k % 10 == 0 {
                    println!("    [hc_fine] bc/Ms={:.6}  <m_sum>={:.6}", bc / ms, msum_f);
                }
            }

            if msum_f <= 0.0 {
                return bc / ms;
            }
        }
    }
}

pub fn run_sp2() -> std::io::Result<()> {
    // MuMax script constants
    let ms: f64 = 1000e3;
    let a_ex: f64 = 10e-12;
    let k_u: f64 = 0.0;

    let lex: f64 = (2.0 * a_ex / (MU0 * ms * ms)).sqrt();
    let rem_only = std::env::var("SP2_REM_ONLY").is_ok();

    println!("SP2: d range = [{}..{}]", D_MIN, D_MAX);
    println!("SP2: relax tighten_floor = {:.1e}", RELAX_TIGHTEN_FLOOR);
    println!("SP2: relax torque_check_stride = {}", TORQUE_CHECK_STRIDE);
    println!(
        "SP2: relax max steps remanence  = {}",
        MAX_ACCEPTED_STEPS_REMANENCE
    );
    println!(
        "SP2: relax max steps coercivity = {}",
        MAX_ACCEPTED_STEPS_COERCIVITY
    );
    println!("SP2: minimize iters remanence   = {}", MIN_ITERS_REMANENCE);
    println!("SP2: minimize iters coercivity  = {}", MIN_ITERS_COERCIVITY);
    println!(
        "SP2: minimize tau coarse/strict = {:.1e} / {:.1e}",
        MIN_TAU_COARSE, MIN_TAU_STRICT
    );
    println!("SP2: bracket mult = {}", BC_BRACKET_MULT);
    println!("SP2: target step mult = {}", BC_TARGET_MULT);
    println!(
        "SP2: fast mode = {}",
        if sp2_fast_enabled() { "ON" } else { "OFF" }
    );

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
    let f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&table_path)?;
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
            println!(
                "SP2 d/lex={:>2}: remanence cache HIT -> {}",
                d_int,
                cache_path.display()
            );
            m_cached
        } else {
            println!(
                "SP2 d/lex={:>2}: remanence cache MISS -> equilibrating (minimize + relax)",
                d_int
            );

            let mut m0_field = VectorField2D::new(grid);
            let m0 = normalize([1.0, 0.3, 0.0]);
            m0_field.set_uniform(m0[0], m0[1], m0[2]);

            // Remanence B_ext = 0
            params.b_ext = [0.0, 0.0, 0.0];

            let do_timing = sp2_timing_enabled();
            let t_rem = Instant::now();

            // Pre-minimize to reduce expensive relax steps
            let rep = minimize_call(
                &grid,
                &mut m0_field,
                &params,
                &material,
                MIN_ITERS_REMANENCE,
                MIN_TAU_STRICT,
                "remanence/premin",
            );
            if rep.stalled {
                println!(
                    "SP2 d/lex={:>2}: remanence pre-minimize stalled -> proceeding to relax anyway",
                    d_int
                );
            }

            // Strict relax to certify final remanence state
            let _ = relax_call(
                &grid,
                &mut m0_field,
                &mut params,
                &material,
                &mut rk23,
                MAX_ACCEPTED_STEPS_REMANENCE,
                "remanence/relax",
            );

            if do_timing {
                println!(
                    "SP2 d/lex={:>2.0}: remanence (premin+relax) done in {:.1}s",
                    d,
                    t_rem.elapsed().as_secs_f64()
                );
            }

            let hdr = RemCacheHeader::new(d_int, &grid);
            write_rem_cache(&cache_path, hdr, &m0_field)?;
            println!(
                "SP2 d/lex={:>2}: cached remanence -> {}",
                d_int,
                cache_path.display()
            );

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
        let hc_over_ms = if rem_only {
            f64::NAN
        } else {
            println!("SP2 d/lex={:>2.0}: coercivity search start", d);
            let m_rem = VectorField2D {
                grid,
                data: m.data.clone(),
            };
            let mut m_work = VectorField2D::new(grid);

            let t_hc = Instant::now();
            let hc_over_ms = find_hc_over_ms_branch_following_hybrid(
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
            hc_over_ms
        };

        // Restore B_ext = 0 (matches MuMax script end-of-loop)
        params.b_ext = [0.0, 0.0, 0.0];

        writeln!(
            w,
            "{:.0},{:.16e},{:.16e},{:.16e}",
            d, mx_rem, my_rem, hc_over_ms
        )?;
        w.flush()?;
        println!("SP2 d/lex={:>2.0}: row written", d);
    }

    println!(
        "\nSP2 complete. Output: {}",
        Path::new("runs/st_problems/sp2/table.csv").display()
    );
    Ok(())
}
	