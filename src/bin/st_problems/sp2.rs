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
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use llg_sim::effective_field::FieldMask;
use llg_sim::grid::Grid2D;
use llg_sim::llg::RK23Scratch;
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material, MU0};
use llg_sim::relax::{relax, RelaxSettings, TorqueMetric};
use llg_sim::vec3::normalize;
use llg_sim::vector_field::VectorField2D;

// New: SP2-only minimiser (one field build per iter)
// Requires: src/minimize.rs + `pub mod minimize;` in lib.rs
use llg_sim::minimize::{minimize_damping_only, MinimizeReport, MinimizeSettings};

// -------------------------
// Debug-fast knobs
// -------------------------

// Start with a small subset so you can see it completing.
// Set to (1, 30) when runtime is under control.
const D_MIN: usize = 18;
const D_MAX: usize = 20;

// SP2 coercivity bracketing/bisection:
// - MuMax step is 0.00005*Ms. We'll use a coarse bracket step (e.g. 20×) then bisect.
const BC_BRACKET_MULT: f64 = 20.0; // coarse step multiplier for bracketing
const BC_TARGET_MULT: f64 = 1.0;   // target resolution multiplier (1.0 => MuMax resolution)

// Relax effort caps per call (prevents "hangs" from very strict tightening on CPU)
const MAX_ACCEPTED_STEPS_REMANENCE: usize = 80_000;
const MAX_ACCEPTED_STEPS_COERCIVITY: usize = 30_000;

// Bisection needs only a reliable sign of <m_sum>, not a fully polished state.
// Use a smaller relax cap for bisect points to save runtime.
const MAX_ACCEPTED_STEPS_BISECT_STRICT: usize = 10_000;

// If Minimize() already converged very tightly (MuMax-style max dM), we can often skip Relax()
// at bisection points without changing the sign of <m_sum>.
const DM_SKIP_RELAX: f64 = 5e-7;

// Additional safety gates for skipping Relax() at bisection points.
// - DM criterion alone can be misleading if the minimizer step size collapses.
// - Only skip Relax if torque is also small, and we are not right on the decision boundary.
const TORQUE_SKIP_RELAX: f64 = 2e-3;      // Tesla (tunable)
const MSUM_SKIP_RELAX_MIN: f64 = 2e-2;    // dimensionless (tunable)

// During the coarse bracket scan, occasionally re-anchor the branch with a strict equilibration
// when we are close to switching. This reduces metastability drift that can bias Hc.
const MSUM_NEAR_SWITCH: f64 = 5e-2;       // dimensionless (tunable)

// NOTE: we no longer use a relax fallback in the coarse bracket scan. Keep this constant for
// experimentation (e.g. if you re-enable a fallback later) but silence unused warnings.
#[allow(dead_code)]
const MAX_ACCEPTED_STEPS_COARSE_FALLBACK: usize = 2000;

const MIN_ITERS_COARSE: usize = 400;   // start here; tune 200–1000

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
// Equilibration (two-tier)
// -------------------------

#[derive(Debug, Clone, Copy)]
enum EqTier {
    Coarse, // minimize only (fast)
    Strict, // minimize precondition + relax (robust)
}

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
        s.lambda_min = 5e-5;      // allow a bit more motion per iter (tunable)
        s.dm_stop = Some(5e-7);   // stricter than default 1e-6 (tunable)
        s.dm_samples = 10;        // keep MuMax-like smoothing
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

// Dedicated coarse minimiser wrapper: disables dM early stop and forces fixed work
fn minimize_call_coarse_fixed(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &LLGParams,
    material: &Material,
    max_iters: usize,
    label: &str,
) -> MinimizeReport {
    let fast = sp2_fast_enabled();

    // Force fixed-iteration behaviour for coarse bracketing:
    // - disable dM early stop (dm_stop=None)
    // - disable torque-threshold early stop (set to 0.0 which can never be reached)
    // - disable stall detection (iters too small to matter, but keep it safe)
    let mut s = MinimizeSettings {
        torque_threshold: 0.0,
        max_iters,
        ..Default::default()
    };
    s.dm_stop = None;
    s.stall_iters = usize::MAX;
    s.min_iters_before_stall = usize::MAX;

    // Keep the coarse minimizer moving near the switch (reduces "sticky" positive m_sum)
    s.lambda_min = 5e-5;
    s.shrink = 0.9;
    s.grow = 1.02;

    if fast {
        // In fast mode, reduce work further.
        s.max_iters = (max_iters / 4).max(50);
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
        settings.torque_metric = TorqueMetric::Mean;

        // MuMax-like: rely on average-torque plateau rather than a hard absolute threshold
        settings.torque_threshold = None;

        settings.torque_plateau_checks = 8;
        settings.torque_plateau_rel = 1e-3;
        settings.torque_plateau_min_checks = 5;
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
            "      [sp2 timing] relax({}) max_steps={} fast={} took {:.3}s",
            label,
            max_steps,
            if fast { 1 } else { 0 },
            t0.elapsed().as_secs_f64()
        );
    }
}

/// Equilibrate in-place under current params.b_ext.
/// Returns avg m_sum after equilibration.
/// Robustness policy:
/// - Coarse: minimize only; if it stalls, fall back to strict relax.
/// - Strict: minimize precondition; then relax.
fn equilibrate_and_msum_in_place(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    tier: EqTier,
    label: &str,
) -> f64 {
    match tier {
        EqTier::Coarse => {
            // Coarse bracket scan: run a fixed iteration budget (no dm_stop early exit)
            // to better follow the equilibrium branch without calling Relax().
            let _rep = minimize_call_coarse_fixed(
                grid,
                m,
                params,
                material,
                MIN_ITERS_COARSE,
                label,
            );
        }
        EqTier::Strict => {
            let rep = minimize_call(
                grid,
                m,
                params,
                material,
                MIN_ITERS_COERCIVITY,
                MIN_TAU_STRICT,
                label,
            );

            // For bisection points: skip Relax only if both dM and torque are tight, and we are not
            // exactly on the decision boundary.
            let is_bisect = label.starts_with("bisect/");
            let tight_dm = rep.dm_converged && rep.final_max_dm < DM_SKIP_RELAX;
            let tight_torque = rep.final_torque < TORQUE_SKIP_RELAX;

            // Compute current m_sum cheaply (no extra field builds) to avoid skipping Relax
            // exactly where the sign decision is delicate.
            let mm_pre = avg_m(m);
            let msum_pre = mm_pre[0] + mm_pre[1] + mm_pre[2];
            let near_boundary = msum_pre.abs() < MSUM_SKIP_RELAX_MIN;

            if is_bisect && tight_dm && tight_torque && !near_boundary {
                // Skip Relax; minimizer is genuinely converged and we are away from the boundary.
            } else {
                let steps = if is_bisect {
                    MAX_ACCEPTED_STEPS_BISECT_STRICT
                } else {
                    MAX_ACCEPTED_STEPS_COERCIVITY
                };
                relax_call(grid, m, params, material, rk23, steps, label);
            }
        }
    }

    let mm = avg_m(m);
    mm[0] + mm[1] + mm[2]
}

/// Convenience: set B_ext from bc, equilibrate, return m_sum.
fn equilibrate_bc_and_msum(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    bc_amps_per_m: f64,
    tier: EqTier,
    label: &str,
) -> f64 {
    set_bext_from_bc(params, bc_amps_per_m);
    equilibrate_and_msum_in_place(grid, m, params, material, rk23, tier, label)
}

// Coercivity: continuation bracket scan + bisection from last-positive state.
// New behaviour:
// - Bracket scan: Coarse tier (minimize-only) for speed
// - When sign flip found: verify bc_high with Strict tier starting from last-positive state
// - Before bisection: polish last-positive state with Strict tier for determinism/robustness
// - Bisection midpoints: Strict tier
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

    // Evaluate the starting field point with a STRICT seed.
    // This gives a robust continuation starting state and avoids paying repeated relax costs in the bracket scan.
    let mut bc_low = bc0;
    let s0 = equilibrate_bc_and_msum(
        grid,
        m_work,
        params,
        material,
        rk23,
        bc_low,
        EqTier::Strict,
        "bc0/strict_seed",
    );

    // If already switched at bc0, return immediately (matches the MuMax loop condition).
    if s0 <= 0.0 {
        return bc_low / ms;
    }

    // Save the strict seed state at bc0. This is guaranteed positive under strict equilibrium.
    let mut m_seed_state = VectorField2D::new(*grid);
    m_seed_state.data.clone_from(&m_work.data);

    // Save last-positive state (updated during bracketing)
    let mut m_low_state = VectorField2D::new(*grid);
    m_low_state.data.clone_from(&m_work.data);

    // Bracket upward (continuation)
    let mut bc_high = bc_low;
    let mut s_high = s0;
    let mut k = 0usize;

    while s_high > 0.0 && bc_high <= 0.2 * ms {
        bc_high += bc_bracket_step;
        k += 1;

        // continuation: do not reset m_work
        s_high = equilibrate_bc_and_msum(
            grid,
            m_work,
            params,
            material,
            rk23,
            bc_high,
            EqTier::Coarse,
            "bracket/coarse",
        );

        // If we are close to switching, re-anchor using a strict equilibration at this field.
        // This keeps the continuation state closer to the true equilibrium branch and reduces
        // systematic Hc overestimation from metastable drift.
        if s_high > 0.0 && s_high.abs() < MSUM_NEAR_SWITCH {
            let mut m_anchor = VectorField2D::new(*grid);
            m_anchor.data.clone_from(&m_work.data);
            let s_anchor = equilibrate_bc_and_msum(
                grid,
                &mut m_anchor,
                params,
                material,
                rk23,
                bc_high,
                EqTier::Strict,
                "bracket/anchor_strict",
            );
            // Continue from the strict-anchored state.
            m_work.data.clone_from(&m_anchor.data);
            s_high = s_anchor;
        }

        // Additional near-switch probe: if coarse minimize still claims small positive m_sum,
        // do a short plateau-only relax to avoid "sticky" false-positives.
        if s_high > 0.0 && s_high < 0.15 {
            let mut m_probe = VectorField2D::new(*grid);
            m_probe.data.clone_from(&m_work.data);
            relax_call(
                grid,
                &mut m_probe,
                params,
                material,
                rk23,
                2000,
                "bracket/probe_relax",
            );
            let mm = avg_m(&m_probe);
            s_high = mm[0] + mm[1] + mm[2];
            m_work.data.clone_from(&m_probe.data);
        }

        if BRACKET_PRINT_EVERY > 0 && k % BRACKET_PRINT_EVERY == 0 {
            println!(
                "    [bracket] k={:>3} bc/Ms={:.6}  <m_sum>={:.6}",
                k,
                bc_high / ms,
                s_high
            );
        }

        if s_high > 0.0 {
            // Still positive: update the last-positive state and continue
            bc_low = bc_high;
            m_low_state.data.clone_from(&m_work.data);
            continue;
        }

        // Coarse found sign flip: verify with STRICT starting from last-positive state.
        // This prevents a loose minimize from producing a false negative.
        let mut m_verify = VectorField2D::new(*grid);
        m_verify.data.clone_from(&m_low_state.data);

        let s_high_strict = equilibrate_bc_and_msum(
            grid,
            &mut m_verify,
            params,
            material,
            rk23,
            bc_high,
            EqTier::Strict,
            "bracket/verify_strict",
        );

        if s_high_strict > 0.0 {
            // Not actually switched under strict -> treat as still positive and continue scanning
            bc_low = bc_high;
            m_low_state.data.clone_from(&m_verify.data);

            // Continue from strict state (better seed)
            m_work.data.clone_from(&m_verify.data);
            s_high = s_high_strict;
            continue;
        }

        // Verified bracket: keep bc_low from last positive and bc_high as first negative.
        // Proceed to bisection.
        s_high = s_high_strict;
        break;
    }

    if s_high > 0.0 {
        println!("    WARNING: failed to bracket coercivity before bc cap; returning bc_high/Ms.");
        return bc_high / ms;
    }

    // Ensure the low bracket is truly POSITIVE under strict equilibrium.
    // If coarse bracketing stayed metastable too long, bc_low may be negative under strict relaxation.
    // In that case, reset the low bracket to the original strict seed at bc0.
    {
        let mut m_low_strict = VectorField2D::new(*grid);
        m_low_strict.data.clone_from(&m_low_state.data);

        let s_low_strict = equilibrate_bc_and_msum(
            grid,
            &mut m_low_strict,
            params,
            material,
            rk23,
            bc_low,
            EqTier::Strict,
            "bracket/low_strict",
        );

        if s_low_strict > 0.0 {
            // Great: update the last-positive state to the strict-equilibrated one.
            m_low_state.data.clone_from(&m_low_strict.data);
        } else {
            println!(
                "    [bracket] NOTE: bc_low not positive under strict; repairing low bracket by stepping downward (keep continuation seed)."
            );

            // Repair strategy: step bc_low downward and re-test under STRICT,
            // always seeding from the continuation state (m_low_state) rather than bc0.
            // This avoids biasing Hc upward by re-seeding bisection from a deep-positive state.
            let step_down = bc_bracket_step / 4.0;
            let mut repaired = false;

            let mut bc_try = bc_low;
            for _ in 0..60 {
                if bc_try <= bc0 + step_down {
                    break;
                }
                bc_try -= step_down;

                let mut m_try = VectorField2D::new(*grid);
                // Always seed from the continuation state near the switch (NOT from bc0)
                m_try.data.clone_from(&m_low_state.data);

                let s_try = equilibrate_bc_and_msum(
                    grid,
                    &mut m_try,
                    params,
                    material,
                    rk23,
                    bc_try,
                    EqTier::Strict,
                    "bracket/repair_low_strict",
                );

                if s_try > 0.0 {
                    bc_low = bc_try;
                    m_low_state.data.clone_from(&m_try.data);
                    repaired = true;
                    break;
                }
            }

            if !repaired {
                println!(
                    "    [bracket] WARNING: failed to repair low bracket; falling back to bc0 strict seed for bisection (may be biased)."
                );
                bc_low = bc0;
                m_low_state.data.clone_from(&m_seed_state.data);
            }
        }
    }

    // Polish last-positive state with STRICT at bc_low before bisection (robust seed).
    {
        let mut m_polish = VectorField2D::new(*grid);
        m_polish.data.clone_from(&m_low_state.data);

        let s_polish = equilibrate_bc_and_msum(
            grid,
            &mut m_polish,
            params,
            material,
            rk23,
            bc_low,
            EqTier::Strict,
            "bisect/seed_polish",
        );

        // If polishing keeps positive, use polished seed (good).
        // If polishing flips negative, we still proceed; bisection will handle it,
        // but keeping the polished state would be inconsistent with "last-positive seed".
        if s_polish > 0.0 {
            m_low_state.data.clone_from(&m_polish.data);
        }
    }

    // Bisection: start each midpoint from last-positive state (strict-polished)
    let mut iter = 0usize;
    while (bc_high - bc_low) > bc_target_step && iter < 60 {
        iter += 1;
        let bc_mid = 0.5 * (bc_low + bc_high);

        // Start from last-positive state
        m_work.data.clone_from(&m_low_state.data);
        let s_mid = equilibrate_bc_and_msum(
            grid,
            m_work,
            params,
            material,
            rk23,
            bc_mid,
            EqTier::Strict,
            "bisect/strict",
        );

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
            m_low_state.data.clone_from(&m_work.data);
        } else {
            bc_high = bc_mid;
        }
    }

    bc_high / ms
}

pub fn run_sp2() -> std::io::Result<()> {
    // MuMax script constants
    let ms: f64 = 1000e3;
    let a_ex: f64 = 10e-12;
    let k_u: f64 = 0.0;

    let lex: f64 = (2.0 * a_ex / (MU0 * ms * ms)).sqrt();

    println!("SP2: d range = [{}..{}]", D_MIN, D_MAX);
    println!("SP2: relax tighten_floor = {:.1e}", RELAX_TIGHTEN_FLOOR);
    println!("SP2: relax torque_check_stride = {}", TORQUE_CHECK_STRIDE);
    println!("SP2: relax max steps remanence  = {}", MAX_ACCEPTED_STEPS_REMANENCE);
    println!("SP2: relax max steps coercivity = {}", MAX_ACCEPTED_STEPS_COERCIVITY);
    println!("SP2: minimize iters remanence   = {}", MIN_ITERS_REMANENCE);
    println!("SP2: minimize iters coercivity  = {}", MIN_ITERS_COERCIVITY);
    println!("SP2: minimize tau coarse/strict = {:.1e} / {:.1e}", MIN_TAU_COARSE, MIN_TAU_STRICT);
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
            println!("SP2 d/lex={:>2}: remanence cache MISS -> equilibrating (minimize + relax)", d_int);

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
                println!("SP2 d/lex={:>2}: remanence pre-minimize stalled -> proceeding to relax anyway", d_int);
            }

            // Strict relax to certify final remanence state
            relax_call(
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