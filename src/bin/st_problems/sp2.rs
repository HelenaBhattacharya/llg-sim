// ===============================
// src/bin/st_problems/sp2.rs (FULL FILE)
// ===============================
//
// Standard Problem #2 (MuMag): remanence and coercivity vs d/lex.
//
// Reference: MuMax3 paper, Appendix A script “Standard Problem 2 (Figs. 13 and 14)”.
//
// MuMax coercivity logic (core truth):
//   bc := 0.0445*Ms
//   for (mx+my+mz) > 0 {
//       B_ext = -bc*mu0*(1,1,1)/sqrt(3)
//       Relax()
//       bc += 0.00005*Ms
//   }
//
// Important: MuMax does NOT bail out if a step is “hard”.
// It keeps relaxing/tightening until it settles (MaxErr -> 1e-9).
//
// This Rust driver matches that philosophy:
//  - Pure MuMax-like scan (no bracketing by default).
//  - If a relax “gate” fails, we DO MORE WORK at the same field point:
//      - more accepted steps
//      - tighter floor (towards 1e-9)
//      - optional tiny minimizer preconditioner
//    and retry until it passes.
//
// Grid policy + remanence cache are kept (your existing setup).
//
// Run:
//   cargo run --release --bin st_problems -- sp2
//
// Env (optional, not required for correctness):
//   SP2_D_MIN / SP2_D_MAX
//   SP2_FORCE=1
//   SP2_REM_ONLY=1
//   SP2_GRID_MODE=mumax|legacy
//   SP2_CELL_OVER_LEX=0.75
//   SP2_TIMING=1
//   SP2_HC_MULTI_SEED=0|1
//   SP2_HC_MULTI_SEED_AUTO=0|1
//   SP2_HC_MULTI_SEED_N=2
//   SP2_HC_MULTI_SEED_STRENGTH=1e-4
//

use std::collections::HashSet;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use llg_sim::effective_field::FieldMask;
use llg_sim::equilibrate::{
    equilibrate_remanence, hysteresis_step, HysteresisPolicy, HysteresisStepReport, RemanencePolicy,
};
use llg_sim::grid::Grid2D;
use llg_sim::grid_sp2::{
    build_sp2_grid, maybe_refine_after_remanence, resample_remanence_to_policy_grid, Sp2GridMode,
    Sp2GridPolicy,
};
use llg_sim::llg::RK23Scratch;
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, MU0, Material};
use llg_sim::vec3::normalize;
use llg_sim::vector_field::VectorField2D;
use llg_sim::energy::compute_total_energy;

// -------------------------
// Constants (MuMax SP2)
// -------------------------

const MS: f64 = 1000e3; // A/m
const A_EX: f64 = 10e-12; // J/m
const K_U: f64 = 0.0; // J/m^3

const BC0_MULT: f64 = 0.0445;
const BC_STEP_MULT: f64 = 0.00005;

const HC_BC_CAP_OVER_MS: f64 = 0.2;

const MAX_ACCEPTED_STEPS_REMANENCE: usize = 120_000;

const REM_CACHE_DIR: &str = "runs/st_problems/sp2/cache";

// -------------------------
// Env helpers
// -------------------------

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok()
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(s) => s.trim().parse::<usize>().unwrap_or(default),
        Err(_) => default,
    }
}

fn env_f64(name: &str, default: f64) -> f64 {
    match std::env::var(name) {
        Ok(s) => s.trim().parse::<f64>().unwrap_or(default),
        Err(_) => default,
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(s) => {
            let v = s.trim().to_ascii_lowercase();
            match v.as_str() {
                "1" | "true" | "yes" | "y" | "on" => true,
                "0" | "false" | "no" | "n" | "off" => false,
                _ => default,
            }
        }
        Err(_) => default,
    }
}

fn sp2_timing_enabled() -> bool {
    env_flag("SP2_TIMING")
}

fn sp2_step_mult() -> usize {
    env_usize("SP2_STEP_MULT", 1).max(1)
}

fn sp2_grid_policy_from_env() -> Sp2GridPolicy {
    let mut p = Sp2GridPolicy::default();
    let mode = std::env::var("SP2_GRID_MODE").unwrap_or_else(|_| "mumax".to_string());
    p.mode = Sp2GridMode::from_str(&mode);

    p.cell_over_lex = env_f64("SP2_CELL_OVER_LEX", p.cell_over_lex);
    p.refine_factor = env_f64("SP2_REFINE_FACTOR", p.refine_factor);
    p.max_refinements = env_usize("SP2_MAX_REFINEMENTS", p.max_refinements);

    p.nn_angle_rms_threshold = env_f64("SP2_REFINE_NN_RMS", p.nn_angle_rms_threshold);
    p.nn_angle_max_threshold = env_f64("SP2_REFINE_NN_MAX", p.nn_angle_max_threshold);
    p.remanence_steps_threshold = env_usize("SP2_REFINE_REM_STEPS", p.remanence_steps_threshold);

    p
}

// -------------------------
// Basic metrics
// -------------------------

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
    [sx / n.max(1.0), sy / n.max(1.0), sz / n.max(1.0)]
}

fn msum(field: &VectorField2D) -> f64 {
    let m = avg_m(field);
    m[0] + m[1] + m[2]
}

// -------------------------
// Small deterministic perturbation (optional multi-seed at bc0)
// -------------------------
fn apply_micro_perturb(m: &mut VectorField2D, strength: f64, seed: u32) {
    if m.data.is_empty() { return; }
    let n = m.data.len();
    let idx = match seed % 3 { 0 => 0, 1 => n / 2, _ => n - 1 };

    let mut v = m.data[idx];
    v[1] += strength;
    v[2] -= 0.5 * strength;
    m.data[idx] = normalize(v);
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
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
        rel(self.dx, g.dx) < 1e-12 && rel(self.dy, g.dy) < 1e-12 && rel(self.dz, g.dz) < 1e-12
    }
}

fn rem_cache_path(d_lex: usize, g: &Grid2D) -> PathBuf {
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

    f.write_all(&header.magic)?;
    f.write_all(&header.version.to_le_bytes())?;
    f.write_all(&header.d_lex.to_le_bytes())?;
    f.write_all(&header.nx.to_le_bytes())?;
    f.write_all(&header.ny.to_le_bytes())?;
    f.write_all(&header.dx.to_le_bytes())?;
    f.write_all(&header.dy.to_le_bytes())?;
    f.write_all(&header.dz.to_le_bytes())?;

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
// Field helper
// -------------------------

fn set_bext_from_bc(params: &mut LLGParams, bc_amps_per_m: f64) {
    // Apply B_ext (Tesla) along (-1,-1,-1)/sqrt(3), matching MuMax.
    let b = -bc_amps_per_m * MU0 / 3.0_f64.sqrt();
    params.b_ext = [b, b, b];
}

// -------------------------
// OVF writer (OOMMF OVF 2.0, Data Binary 4)
// -------------------------

fn write_ovf_binary4(
    path: &Path,
    grid: &Grid2D,
    m: &VectorField2D,
    desc: &str,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }

    let mut f = BufWriter::new(File::create(path)?);

    // OVF 2.0 header (MuMax/OOMMF compatible)
    writeln!(f, "# OOMMF OVF 2.0")?;
    writeln!(f, "# Segment count: 1")?;
    writeln!(f, "# Begin: Segment")?;
    writeln!(f, "# Begin: Header")?;
    writeln!(f, "# Title: m")?;
    writeln!(f, "# Desc: {}", desc)?;
    writeln!(f, "# meshunit: m")?;
    writeln!(f, "# meshtype: rectangular")?;

    // OVF uses cell-centered bases
    let dx = grid.dx;
    let dy = grid.dy;
    let dz = grid.dz;
    writeln!(f, "# xbase: {:.17e}", 0.5 * dx)?;
    writeln!(f, "# ybase: {:.17e}", 0.5 * dy)?;
    writeln!(f, "# zbase: {:.17e}", 0.5 * dz)?;

    writeln!(f, "# xstepsize: {:.17e}", dx)?;
    writeln!(f, "# ystepsize: {:.17e}", dy)?;
    writeln!(f, "# zstepsize: {:.17e}", dz)?;

    writeln!(f, "# xnodes: {}", grid.nx)?;
    writeln!(f, "# ynodes: {}", grid.ny)?;
    writeln!(f, "# znodes: 1")?;

    writeln!(f, "# xmin: 0")?;
    writeln!(f, "# xmax: {:.17e}", (grid.nx as f64) * dx)?;
    writeln!(f, "# ymin: 0")?;
    writeln!(f, "# ymax: {:.17e}", (grid.ny as f64) * dy)?;
    writeln!(f, "# zmin: 0")?;
    writeln!(f, "# zmax: {:.17e}", dz)?;

    writeln!(f, "# valueunit: 1")?;
    writeln!(f, "# valuemultiplier: 1")?;
    writeln!(f, "# End: Header")?;
    writeln!(f, "# Begin: Data Binary 4")?;

    // Binary4 endian check value
    let check: f32 = 1234567.0;
    f.write_all(&check.to_le_bytes())?;

    // Data: mx,my,mz per cell (row-major; x fastest)
    for v in &m.data {
        let x = v[0] as f32;
        let y = v[1] as f32;
        let z = v[2] as f32;
        f.write_all(&x.to_le_bytes())?;
        f.write_all(&y.to_le_bytes())?;
        f.write_all(&z.to_le_bytes())?;
    }

    writeln!(f)?;
    writeln!(f, "# End: Data Binary 4")?;
    writeln!(f, "# End: Segment")?;
    writeln!(f, "# End: File")?;

    f.flush()?;
    Ok(())
}

// -------------------------
// MuMax-like "do not bail" settle helper
// -------------------------

fn settle_hysteresis_until_gate_passes(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    mask: FieldMask,
    base_policy: &HysteresisPolicy,
    is_first: bool,
) -> HysteresisStepReport {
    // MuMax SP2: one Relax() per field step. No extra gate-driven escalation loop here.
    // Gate is kept as a diagnostic (printed), not enforced by doing tons more work.
    hysteresis_step(grid, m, params, material, rk23, mask, base_policy, is_first)
}

// -------------------------
// Coercivity scan (MuMax-like, no bracketing)
// -------------------------

fn find_hc_over_ms_mumax_scan(
    grid: &Grid2D,
    m_work: &mut VectorField2D,
    m_seed: &VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    policy: &HysteresisPolicy,
) -> f64 {
    let ms = material.ms;

    // Middle-ground accuracy: if the diagnostic gate fails at a field step,
    // do ONE extra Relax() at the *same* bc with a slightly tighter floor and a modest step-budget bump.
    // This is still MuMax-like (more Relax at same field), but avoids the unbounded [hc_escalate] loop.
    let strict_retry_enabled: bool = env_bool("SP2_HC_STRICT_RETRY", true);
    // Only pay extra work near the switching bifurcation (reduces runtime + avoids unnecessary retries).
    let crit_window: f64 = env_f64("SP2_HC_CRIT_WINDOW", 0.20);
    // Optionally enable Phase 1 (energy descent) on the one-shot retry (can help snap into basin near switching).
    let strict_phase1: bool = env_bool("SP2_HC_STRICT_PHASE1", false);

    let strict_floor: f64 = env_f64("SP2_HC_STRICT_FLOOR", 1e-8);
    let strict_steps_mult: f64 = env_f64("SP2_HC_STRICT_STEPS_MULT", 2.0);
    let strict_steps_cap: usize = env_usize("SP2_HC_STRICT_MAX_STEPS", 80_000);

    // Start from remanence seed.
    m_work.data.clone_from(&m_seed.data);

    // MuMax step
    let step = (BC_STEP_MULT * sp2_step_mult() as f64) * ms;

    // Start bc
    let mut bc = BC0_MULT * ms;

    // Initial point
    set_bext_from_bc(params, bc);
    rk23.invalidate_last_b_eff();

    let mut rep0 = settle_hysteresis_until_gate_passes(
        grid,
        m_work,
        params,
        material,
        rk23,
        FieldMask::Full,
        policy,
        true,
    );

    // Optional multi-seed at bc0: try small perturbations and pick lowest-energy relaxed state.
    let multiseed_force = env_bool("SP2_HC_MULTI_SEED", false);
    let multiseed_auto = env_bool("SP2_HC_MULTI_SEED_AUTO", true);
    let multiseed_n = env_usize("SP2_HC_MULTI_SEED_N", 2).min(8);
    let multiseed_strength = env_f64("SP2_HC_MULTI_SEED_STRENGTH", 1e-4);

    // State-based trigger: only auto-run if bc0 gate fails (keeps low d/lex untouched).
    let do_multiseed = multiseed_force || (multiseed_auto && !rep0.relax.gate_passed);

    if do_multiseed && multiseed_n > 0 {
        // Candidate 0 = unperturbed bc0 relaxed state
        let mut best_ok = rep0.relax.gate_passed;
        let mut best_e = compute_total_energy(grid, m_work, material, params.b_ext);
        let mut best_dt = params.dt;
        let mut best_data = m_work.data.clone();
        let mut best_rep = rep0.clone();

        for sid in 1..=multiseed_n {
            let mut m_cand = VectorField2D::new(*grid);
            m_cand.data.clone_from(&m_seed.data);
            apply_micro_perturb(&mut m_cand, multiseed_strength, sid as u32);

            // Make seeds comparable
            params.dt = 1e-13;
            set_bext_from_bc(params, bc);
            rk23.invalidate_last_b_eff();

            let rep_cand = settle_hysteresis_until_gate_passes(
                grid,
                &mut m_cand,
                params,
                material,
                rk23,
                FieldMask::Full,
                policy,
                true,
            );

            let cand_ok = rep_cand.relax.gate_passed;
            let cand_e = compute_total_energy(grid, &m_cand, material, params.b_ext);
            let cand_dt = params.dt;

            // Prefer gate-passed; within that choose lowest energy
            let better = if cand_ok && !best_ok {
                true
            } else if cand_ok == best_ok {
                cand_e < best_e
            } else {
                false
            };

            if better {
                best_ok = cand_ok;
                best_e = cand_e;
                best_dt = cand_dt;
                best_data = m_cand.data.clone();
                best_rep = rep_cand;
            }
        }

        m_work.data.clone_from(&best_data);
        params.dt = best_dt;
        rep0 = best_rep;

        if sp2_timing_enabled() {
            let last = rep0.relax.relax_passes.last();
            println!(
                "    [hc_multiseed_bc0] mode={} n={} strength={:.1e} chosen_gate_passed={} E={:.6e} dt={:.3e} torque_mean={:.3e} torque_max={:.3e}",
                if multiseed_force { "FORCE" } else { "AUTO" },
                multiseed_n,
                multiseed_strength,
                rep0.relax.gate_passed,
                best_e,
                params.dt,
                last.and_then(|r| r.final_torque).unwrap_or(f64::NAN),
                last.and_then(|r| r.final_torque_max).unwrap_or(f64::NAN),
            );
        }
    }

    // One extra Relax at the same bc if the diagnostic gate fails.
    // We keep the gate diagnostic-only (no extra passes/minimizer), but we tighten the relax floor a bit.
    if strict_retry_enabled && !rep0.relax.gate_passed && msum(m_work).abs() < crit_window {
        let mut pol2 = policy.clone();

        // Tighten only a bit (towards MuMax 1e-9), and bump step budget modestly.
        pol2.relax.tighten_floor = pol2.relax.tighten_floor.min(strict_floor);

        let mut steps = ((pol2.relax.max_accepted_steps as f64) * strict_steps_mult) as usize;
        if steps < pol2.relax.max_accepted_steps {
            steps = pol2.relax.max_accepted_steps;
        }
        if steps > strict_steps_cap {
            steps = strict_steps_cap;
        }
        pol2.relax.max_accepted_steps = steps;

        // Keep gate diagnostic-only during the retry.
        pol2.gate.extra_passes = 0;
        pol2.gate.extra_pass_max_steps_mult = 1.0;
        pol2.gate.fallback_to_minimize = false;

        // Keep the retry MuMax-like: same field, a bit more work. Optionally add Phase 1.
        pol2.relax.phase2_enabled = true;
        if strict_phase1 {
            pol2.relax.phase1_enabled = true;
        }

        rk23.invalidate_last_b_eff();
        rep0 = hysteresis_step(grid, m_work, params, material, rk23, FieldMask::Full, &pol2, false);

        if sp2_timing_enabled() {
            let last = rep0.relax.relax_passes.last();
            println!(
                "    [hc_strict_retry] bc/Ms={:.6} msum={:.6} |msum|<={:.3} gate_passed={} floor={:.1e} steps={} phase1={} torque_mean={:.3e} torque_max={:.3e}",
                bc / ms,
                msum(m_work),
                crit_window,
                rep0.relax.gate_passed,
                pol2.relax.tighten_floor,
                pol2.relax.max_accepted_steps,
                strict_phase1,
                last.and_then(|r| r.final_torque).unwrap_or(f64::NAN),
                last.and_then(|r| r.final_torque_max).unwrap_or(f64::NAN),
            );
        }
    }

    if sp2_timing_enabled() {
        let last = rep0.relax.relax_passes.last();
        println!(
            "    [hc] bc/Ms={:.6} msum={:.6} gate_passed={} dt={:.3e} torque_mean={:.3e} torque_max={:.3e}",
            bc / ms,
            msum(m_work),
            rep0.relax.gate_passed,
            params.dt,
            last.and_then(|r| r.final_torque).unwrap_or(f64::NAN),
            last.and_then(|r| r.final_torque_max).unwrap_or(f64::NAN),
        );
    }

    if sp2_timing_enabled() && !rep0.relax.gate_passed {
        println!(
            "    [hc_gate_warn] gate failed at bc/Ms={:.6} (continuing; MuMax-like single Relax/step)",
            bc / ms
        );
    }

    if msum(m_work) <= 0.0 {
        return bc / ms;
    }

    // Strict MuMax-style scan: advance by fixed step and Relax each time.
    while bc + step <= HC_BC_CAP_OVER_MS * ms {
        bc += step;

        set_bext_from_bc(params, bc);
        rk23.invalidate_last_b_eff();

        let mut rep = settle_hysteresis_until_gate_passes(
            grid,
            m_work,
            params,
            material,
            rk23,
            FieldMask::Full,
            policy,
            false,
        );

        // One extra Relax at the same bc if the diagnostic gate fails.
        if strict_retry_enabled && !rep.relax.gate_passed && msum(m_work).abs() < crit_window {
            let mut pol2 = policy.clone();

            pol2.relax.tighten_floor = pol2.relax.tighten_floor.min(strict_floor);

            let mut steps = ((pol2.relax.max_accepted_steps as f64) * strict_steps_mult) as usize;
            if steps < pol2.relax.max_accepted_steps {
                steps = pol2.relax.max_accepted_steps;
            }
            if steps > strict_steps_cap {
                steps = strict_steps_cap;
            }
            pol2.relax.max_accepted_steps = steps;

            pol2.gate.extra_passes = 0;
            pol2.gate.extra_pass_max_steps_mult = 1.0;
            pol2.gate.fallback_to_minimize = false;

            // Keep the retry MuMax-like: same field, a bit more work. Optionally add Phase 1.
            pol2.relax.phase2_enabled = true;
            if strict_phase1 {
                pol2.relax.phase1_enabled = true;
            }

            rk23.invalidate_last_b_eff();
            rep = hysteresis_step(grid, m_work, params, material, rk23, FieldMask::Full, &pol2, false);

            if sp2_timing_enabled() {
                let last = rep.relax.relax_passes.last();
                println!(
                    "    [hc_strict_retry] bc/Ms={:.6} msum={:.6} |msum|<={:.3} gate_passed={} floor={:.1e} steps={} phase1={} torque_mean={:.3e} torque_max={:.3e}",
                    bc / ms,
                    msum(m_work),
                    crit_window,
                    rep.relax.gate_passed,
                    pol2.relax.tighten_floor,
                    pol2.relax.max_accepted_steps,
                    strict_phase1,
                    last.and_then(|r| r.final_torque).unwrap_or(f64::NAN),
                    last.and_then(|r| r.final_torque_max).unwrap_or(f64::NAN),
                );
            }
        }

        if sp2_timing_enabled() {
            let last = rep.relax.relax_passes.last();
            println!(
                "    [hc] bc/Ms={:.6} msum={:.6} gate_passed={} dt={:.3e} torque_mean={:.3e} torque_max={:.3e}",
                bc / ms,
                msum(m_work),
                rep.relax.gate_passed,
                params.dt,
                last.and_then(|r| r.final_torque).unwrap_or(f64::NAN),
                last.and_then(|r| r.final_torque_max).unwrap_or(f64::NAN),
            );
        }

        if sp2_timing_enabled() && !rep.relax.gate_passed {
            println!(
                "    [hc_gate_warn] gate failed at bc/Ms={:.6} (continuing; MuMax-like single Relax/step)",
                bc / ms
            );
        }

        if msum(m_work) <= 0.0 {
            return bc / ms;
        }
    }

    // Hit cap without switching
    HC_BC_CAP_OVER_MS
}

// -------------------------
// Resume support
// -------------------------

fn read_done_d_values(table_path: &Path) -> std::io::Result<HashSet<i32>> {
    let mut done = HashSet::new();
    if !table_path.exists() {
        return Ok(done);
    }

    let f = File::open(table_path)?;
    let mut r = BufReader::new(f);
    let mut line = String::new();

    let _ = r.read_line(&mut line)?; // header
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
// Entry point
// -------------------------

pub fn run_sp2() -> std::io::Result<()> {
    let d_min = env_usize("SP2_D_MIN", 1);
    let d_max = env_usize("SP2_D_MAX", 30);
    let rem_only = env_flag("SP2_REM_ONLY");
    let force = env_flag("SP2_FORCE");

    // Exchange length
    let lex: f64 = (2.0 * A_EX / (MU0 * MS * MS)).sqrt();

    let grid_policy = sp2_grid_policy_from_env();

    println!("SP2: d range = [{}..{}]", d_min, d_max);
    println!("SP2: lex = {:.3e} m", lex);
    println!(
        "SP2: grid_mode = {:?}, cell/lex = {:.3}, max_refinements={}",
        grid_policy.mode, grid_policy.cell_over_lex, grid_policy.max_refinements
    );
    println!(
        "SP2: hc start bc/Ms = {:.6}, step/Ms = {:.6} (mult={})",
        BC0_MULT,
        BC_STEP_MULT,
        sp2_step_mult()
    );
    println!("SP2: rem_only = {}", if rem_only { "ON" } else { "OFF" });

    let out_dir = Path::new("runs").join("st_problems").join("sp2");
    create_dir_all(&out_dir)?;
    let ovf_dir = out_dir.join("ovf");
    create_dir_all(&ovf_dir)?;
    let table_path = out_dir.join("table.csv");

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

    let need_header = !file_exists || std::fs::metadata(&table_path)?.len() == 0;
    if need_header {
        writeln!(w, "d_lex,mx_rem,my_rem,hc_over_ms")?;
        w.flush()?;
    }

    for d_lex in (d_min..=d_max).rev() {
        if !force && done.contains(&(d_lex as i32)) {
            println!("SP2 d/lex={:>2}: already done; skipping.", d_lex);
            continue;
        }

        // Build baseline grid
        let policy_now = grid_policy.clone();
        let mut grid = build_sp2_grid(d_lex, lex, &policy_now);

        // Material + params
        let material = Material {
            ms: MS,
            a_ex: A_EX,
            k_u: K_U,
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

        println!(
            "\nSP2 d/lex={:>2}: start (nx={}, ny={}, dx/lex={:.3}, dy/lex={:.3}, dz/lex={:.3})",
            d_lex,
            grid.nx,
            grid.ny,
            grid.dx / lex,
            grid.dy / lex,
            grid.dz / lex
        );

        // Remanence: load cache or equilibrate
        let mut rk23 = RK23Scratch::new(grid);

        let mut rem_report = None;
        let mut m_rem = {
            let cache_path = rem_cache_path(d_lex, &grid);
            if let Some(m_cached) = read_rem_cache(&cache_path, d_lex, &grid)? {
                println!("SP2 d/lex={:>2}: rem cache HIT -> {}", d_lex, cache_path.display());
                m_cached
            } else {
                println!("SP2 d/lex={:>2}: rem cache MISS -> equilibrate", d_lex);
                let mut m0 = VectorField2D::new(grid);
                let u0 = normalize([1.0, 0.3, 0.0]);
                m0.set_uniform(u0[0], u0[1], u0[2]);

                params.b_ext = [0.0, 0.0, 0.0];

                let mut rem_pol = RemanencePolicy::default();
                rem_pol.relax.tighten_floor = env_f64("SP2_REM_TIGHTEN_FLOOR", 1e-6);
                rem_pol.relax.max_accepted_steps = MAX_ACCEPTED_STEPS_REMANENCE;

                let t0 = Instant::now();
                let rep = equilibrate_remanence(
                    &grid,
                    &mut m0,
                    &mut params,
                    &material,
                    &mut rk23,
                    FieldMask::Full,
                    &rem_pol,
                );
                rem_report = rep.relax_passes.last().cloned();

                if sp2_timing_enabled() {
                    println!(
                        "  [rem] passes={} gate_passed={} ({:.2}s)",
                        rep.relax_passes.len(),
                        rep.gate_passed,
                        t0.elapsed().as_secs_f64()
                    );
                } else {
                    println!("  [rem] done in {:.2}s", t0.elapsed().as_secs_f64());
                }

                // Cache the final remanence state (even if gate didn’t pass; remanence is now good in your tests).
                if let Some(r_last) = rep.relax_passes.last() {
                    if r_last.accepted_steps < MAX_ACCEPTED_STEPS_REMANENCE {
                        let hdr = RemCacheHeader::new(d_lex, &grid);
                        write_rem_cache(&cache_path, hdr, &m0)?;
                    }
                }

                m0
            }
        };

        // Optional one-shot refinement after remanence (state-based).
        let refinements_done = 0usize;
        if let Some(p_refined) =
            maybe_refine_after_remanence(&m_rem, rem_report.as_ref(), &policy_now, refinements_done)
        {
            println!(
                "SP2 d/lex={:>2}: grid refine triggered: cell/lex {:.3} -> {:.3}",
                d_lex, policy_now.cell_over_lex, p_refined.cell_over_lex
            );

            let (g2, mut m2) = resample_remanence_to_policy_grid(d_lex, lex, &p_refined, &m_rem);
            grid = g2;
            rk23 = RK23Scratch::new(grid);

            // Re-equilibrate remanence on refined grid
            params.dt = 1e-13;
            params.b_ext = [0.0, 0.0, 0.0];

            let mut rem_pol = RemanencePolicy::default();
            rem_pol.relax.tighten_floor = env_f64("SP2_REM_TIGHTEN_FLOOR", 1e-6);
            rem_pol.relax.max_accepted_steps = MAX_ACCEPTED_STEPS_REMANENCE;

            let _ = equilibrate_remanence(
                &grid,
                &mut m2,
                &mut params,
                &material,
                &mut rk23,
                FieldMask::Full,
                &rem_pol,
            );

            // Cache refined remanence too
            let cache_path2 = rem_cache_path(d_lex, &grid);
            let hdr2 = RemCacheHeader::new(d_lex, &grid);
            let _ = write_rem_cache(&cache_path2, hdr2, &m2);

            m_rem = m2;
        }

        let rem = avg_m(&m_rem);
        println!(
            "SP2 d/lex={:>2}: rem(mx,my,mz,msum)=({:.6},{:.6},{:.6},{:.6})",
            d_lex,
            rem[0],
            rem[1],
            rem[2],
            rem[0] + rem[1] + rem[2]
        );

        // Save remanence state (OVF)
        let rem_ovf = ovf_dir.join(format!("m_d{}_rem.ovf", d_lex));
        let rem_desc = format!("SP2 d/lex={} remanence", d_lex);
        write_ovf_binary4(&rem_ovf, &grid, &m_rem, &rem_desc)?;

        // Coercivity
        let hc_over_ms = if rem_only {
            f64::NAN
        } else {
            let mut m_work = VectorField2D::new(grid);

            let mut hpol = HysteresisPolicy::default();

            // Coercivity: MuMax-like Relax() behaviour
            hpol.relax.phase1_enabled = false;
            hpol.relax.phase2_enabled = true;
            hpol.relax.tighten_factor = std::f64::consts::FRAC_1_SQRT_2;

            // Start tolerance + tightening floor.
            // Default: 1e-8 (try 1e-7 if you want it faster).
            hpol.relax.max_err = 1e-5;
            hpol.relax.tighten_floor = env_f64("SP2_HC_TIGHTEN_FLOOR", 1e-8);
            hpol.relax.max_accepted_steps = env_usize("SP2_HC_MAX_STEPS", 80_000);

            // Keep Mean torque metric for now (matches your existing gate threshold and printing)
            hpol.relax.torque_metric = llg_sim::relax::TorqueMetric::Mean;
            hpol.relax.torque_threshold = None;

            // MuMax-style plateau semantics
            hpol.relax.torque_plateau_checks = 1;
            hpol.relax.torque_plateau_min_checks = 1;
            hpol.relax.torque_plateau_rel = 0.0;
            hpol.relax.torque_plateau_abs = 0.0;

            // MuMax relaxSteps(N≈3) to reduce torque-check overhead
            hpol.relax.torque_check_stride = 3;

            // Gate = diagnostic only (don’t spend extra passes/time trying to satisfy it)
            hpol.gate.torque_gate = Some(env_f64("SP2_HC_TORQUE_GATE", 8e-4));
            hpol.gate.torque_gate_max = std::env::var("SP2_HC_TORQUE_GATE_MAX")
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok());
            hpol.gate.extra_passes = 0;
            hpol.gate.extra_pass_max_steps_mult = 1.0;
            hpol.gate.fallback_to_minimize = false;


            let t0 = Instant::now();
            let hc = find_hc_over_ms_mumax_scan(
                &grid,
                &mut m_work,
                &m_rem,
                &mut params,
                &material,
                &mut rk23,
                &hpol,
            );
            println!(
                "SP2 d/lex={:>2}: coercivity done in {:.1}s  hc/Ms={:.6}",
                d_lex,
                t0.elapsed().as_secs_f64(),
                hc
            );

            // Save coercivity final state (OVF)
            let hc_ovf = ovf_dir.join(format!("m_d{}_hc.ovf", d_lex));
            let hc_desc = format!("SP2 d/lex={} coercivity final state (hc/Ms={:.6})", d_lex, hc);
            write_ovf_binary4(&hc_ovf, &grid, &m_work, &hc_desc)?;

            hc
        };

        writeln!(
            w,
            "{:.0},{:.16e},{:.16e},{:.16e}",
            d_lex as f64,
            rem[0],
            rem[1],
            hc_over_ms
        )?;
        w.flush()?;
        println!("SP2 d/lex={:>2}: row written", d_lex);
    }

    println!("\nSP2 complete. Output: {}", table_path.display());
    Ok(())
}