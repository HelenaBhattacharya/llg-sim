// ===============================
// src/bin/st_problems/sp2.rs (FULL FILE)
// ===============================
//
// Standard Problem #2 (MuMag): remanence and coercivity vs d/lex.
//
// Reference: MuMax3 paper, Appendix A script “Standard Problem 2 (Figs. 13 and 14)”.
// Key requirement for accuracy: coercivity is obtained from a *path-dependent* hysteresis scan
// where the system is relaxed at *every* field step using the previous relaxed state as the seed.
//
// This rewrite makes SP2 behave much closer to MuMax:
//
//   1) Remanence:
//        m = uniform(1, 0.3, 0);  B_ext = 0;  Relax()
//   2) Coercivity scan:
//        for bc = 0.0445*Ms; (mx+my+mz) > 0; bc += 0.00005*Ms { Relax() }
//
// CPU considerations / knobs:
//   - Each d/lex point is independent -> best run each d on a separate SCARF node.
//   - We keep a remanence cache (optional, but huge practical win).
//   - We avoid using Minimize() as the main driver of the hysteresis branch (that biases Hc).
//     Minimizer is only used as an optional *fallback* or tiny preconditioner.
//
// Run:
//   cargo run --release --bin st_problems -- sp2
//
// Optional:
//   SP2_TIMING=1                         -> extra per-step diagnostics
//   SP2_FAST=1                           -> looser settings for profiling (NOT for accuracy)
//   SP2_REM_ONLY=1                       -> only compute remanence rows (hc_over_ms = NaN)
//   SP2_D_MIN=29 SP2_D_MAX=29            -> override d-range without editing code
//   SP2_STEP_MULT=1                      -> bc step multiplier (1 = MuMax step; >1 is profiling only)
//   SP2_HC_START_MULT=20                -> starting *bracket* step multiplier for coercivity scan (relative to SP2_STEP_MULT)
//   SP2_PREMIN=1                         -> enable tiny pre-minimize per coercivity step
//   SP2_SYM_BREAK=1                      -> tiny deterministic perturbation of one cell at start of hc scan
//
// Output:
//   runs/st_problems/sp2/table.csv
//
// Post-process overlay (your existing script):
//   python3 scripts/compare_sp2.py --mumax-root ... --rust-root ... --metrics --out ...

use std::collections::HashSet;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use llg_sim::effective_field::FieldMask;
use llg_sim::grid::Grid2D;
use llg_sim::llg::RK23Scratch;
use llg_sim::minimize::{minimize_damping_only, MinimizeSettings};
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, MU0, Material};
use llg_sim::energy::compute_total_energy;
use llg_sim::relax::{relax_with_report, RelaxReport, RelaxSettings, RelaxStopReason, TorqueMetric};
use llg_sim::vec3::normalize;
use llg_sim::vector_field::VectorField2D;

// -------------------------
// Constants (MuMax SP2)
// -------------------------

const MS: f64 = 1000e3;      // A/m
const A_EX: f64 = 10e-12;    // J/m
const K_U: f64 = 0.0;        // J/m^3 (SP2 uses 0)

// MuMax SP2 coercivity scan parameters
const BC0_MULT: f64 = 0.0445;     // start at 0.0445 * Ms
const BC_STEP_MULT: f64 = 0.00005; // step = 0.00005 * Ms

// Relax caps (safety; prevents infinite loops on CPU)
const MAX_ACCEPTED_STEPS_REMANENCE: usize = 120_000;

// Default relax tightening floor (CPU-feasible). MuMax uses ~1e-9; on CPU this is often too expensive.
const RELAX_TIGHTEN_FLOOR: f64 = 1e-6;

// Remanence cache location
const REM_CACHE_DIR: &str = "runs/st_problems/sp2/cache";

// Coercivity scan behaviour (CPU-friendly but path-faithful)
//
// Strategy:
//   1) Bracket quickly with a large bc step.
//   2) Re-scan within the bracket at the MuMax step size, using a MuMax-like Relax() criterion.
const HC_BC_CAP_OVER_MS: f64 = 0.2;          // safety cap (matches earlier logic)
const HC_START_MULT_DEFAULT: usize = 20;     // starting bracket step multiplier

// Fine scan settings (MuMax step size) – bounded retries if Relax hits MaxAcceptedSteps.
const HC_FINE_RETRIES: usize = 2;

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

#[allow(dead_code)]
fn env_f64(name: &str, default: f64) -> f64 {
    match std::env::var(name) {
        Ok(s) => s.trim().parse::<f64>().unwrap_or(default),
        Err(_) => default,
    }
}

fn sp2_timing_enabled() -> bool {
    env_flag("SP2_TIMING")
}

fn sp2_fast_enabled() -> bool {
    env_flag("SP2_FAST")
}

fn step_mult() -> usize {
    env_usize("SP2_STEP_MULT", 1).max(1)
}

fn hc_start_mult() -> usize {
    env_usize("SP2_HC_START_MULT", HC_START_MULT_DEFAULT).max(1)
}

fn sp2_tighten_floor() -> f64 {
    // Allow tightening floor override without editing code.
    // Smaller = stricter relax (more tightening stages / more steps); larger = faster but less accurate.
    env_f64("SP2_TIGHTEN_FLOOR", RELAX_TIGHTEN_FLOOR)
}

fn sp2_grid_mode() -> String {
    std::env::var("SP2_GRID_MODE").unwrap_or_else(|_| "mumax".to_string()).to_lowercase()
}

fn sp2_cell_over_lex() -> f64 {
    // MuMax/NIST SP2 commonly targets cellsize ~ 0.75 * lex.
    // Override if you want to sweep discretisation sensitivity.
    env_f64("SP2_CELL_OVER_LEX", 0.75)
}

// -------------------------
// Geometry helpers
// -------------------------

fn ilogb_sp2(x: f64) -> i32 {
    if x <= 0.0 {
        return 0;
    }
    (x.log2().floor() as i32) + 1
}

fn build_sp2_grid(d_lex: usize, lex: f64) -> Grid2D {
    let d = d_lex as f64;

    // MuMax SP2 physical sizes
    let sizex = 5.0 * lex * d;
    let sizey = 1.0 * lex * d;
    let sizez = 0.1 * lex * d;

    let mode = sp2_grid_mode();

    if mode == "legacy" {
        // Legacy rule (kept for backwards-compat comparisons):
        // x = sizex / (5 * 0.5 * lex) = 2d, choose p so that 2^p >= x, then nx = 5 * 2^p, ny=nx/5.
        let x = sizex / (5.0 * 0.5 * lex);
        let p = ilogb_sp2(x);
        let nx: usize = (2usize.pow(p as u32)) * 5;
        let ny: usize = nx / 5;

        let dx = sizex / (nx as f64);
        let dy = sizey / (ny as f64);
        let dz = sizez;

        return Grid2D::new(nx, ny, dx, dy, dz);
    }

    // MuMax-style SP2 discretisation: target cellsize ≈ (cell_over_lex * lex) and use pow2 grid dims.
    // This is the closest practical CPU analogue to the common SP2 reference meshes.
    let cell = sp2_cell_over_lex().max(0.05) * lex;

    let nx_req = (sizex / cell).ceil().max(1.0) as usize;
    let ny_req = (sizey / cell).ceil().max(1.0) as usize;

    let nx = nx_req.next_power_of_two();
    let ny = ny_req.next_power_of_two();

    let dx = sizex / (nx as f64);
    let dy = sizey / (ny as f64);
    let dz = sizez;

    Grid2D::new(nx, ny, dx, dy, dz)
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
    [sx / n, sy / n, sz / n]
}

fn msum(field: &VectorField2D) -> f64 {
    let m = avg_m(field);
    m[0] + m[1] + m[2]
}

fn env_bool_default_true(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim().to_lowercase();
            !(v == "0" || v == "false" || v == "off")
        }
        Err(_) => true,
    }
}

fn apply_micro_perturb(m: &mut VectorField2D, strength: f64, seed: u32) {
    if m.data.is_empty() {
        return;
    }
    let n = m.data.len();
    let idx = match seed % 3 {
        0 => 0,
        1 => n / 2,
        _ => n - 1,
    };

    let mut v = m.data[idx];
    v[1] += strength;
    v[2] -= 0.5 * strength;
    m.data[idx] = normalize(v);
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

    // header
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
// Relax wrappers (MuMax-like settings)
// -------------------------

fn relax_remanence(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
) -> RelaxReport {
    let fast = sp2_fast_enabled();

    let mut settings = RelaxSettings {
        // Keep MuMax-ish defaults but CPU-feasible floor.
        tighten_floor: if fast { 1e-5 } else { sp2_tighten_floor() },
        max_accepted_steps: if fast { 30_000 } else { MAX_ACCEPTED_STEPS_REMANENCE },
        ..Default::default()
    };

    // Remanence: allow energy phase (MuMax style), and use plateau mode on mean torque.
    // NOTE: compute_total_energy currently recomputes demag; this is expensive but closer to MuMax.
    // If needed, you can disable Phase 1 via SP2_FAST or by flipping this default later.
    settings.phase1_enabled = !fast;
    settings.phase2_enabled = true;

    settings.torque_metric = TorqueMetric::Mean;
    settings.torque_threshold = None; // plateau mode
    settings.torque_check_stride = 1;
    settings.torque_plateau_checks = if fast { 4 } else { 8 };
    settings.torque_plateau_min_checks = if fast { 4 } else { 8 };

    // Strict MuMax plateau semantics: stop when steady or increasing.
    // Small abs floor can help deterministic CPU runs avoid chasing numerical noise forever.
    settings.torque_plateau_rel = 0.0;
    settings.torque_plateau_abs = if fast { 5e-7 } else { 1e-7 };

    // Reset dt guess for one-off remanence relax.
    params.dt = 1e-13;

    relax_with_report(grid, m, params, material, rk23, FieldMask::Full, &mut settings)
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HcRelaxMode {
    /// Cheap settle used only for coarse bracketing (NOT for the final answer).
    Bracket,
    /// MuMax-like settle used for the fine scan at MuMax step size.
    Fine,
}
fn relax_coercivity_step(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    mode: HcRelaxMode,
) -> RelaxReport {
    let fast = sp2_fast_enabled();

    // Bracket mode: cheap and bounded.
    // Fine mode: MuMax-like (plateau on mean torque + tightening).
    let mut settings = match mode {
        HcRelaxMode::Bracket => {
            let max_err = if fast { 2.0e-4 } else { 2.0e-4 };
            RelaxSettings {
                phase1_enabled: false,
                phase2_enabled: true,

                // Loose max-torque threshold so we stop quickly.
                torque_metric: TorqueMetric::Max,
                torque_threshold: Some(if fast { 5.0e-3 } else { 5.0e-3 }),

                max_err,
                tighten_floor: max_err, // single stage
                tighten_factor: std::f64::consts::FRAC_1_SQRT_2,

                torque_check_stride: if fast { 100 } else { 100 },

                // Plateau unused in threshold mode.
                torque_plateau_rel: 0.0,
                torque_plateau_abs: 0.0,
                torque_plateau_checks: 0,
                torque_plateau_min_checks: 5,

                max_accepted_steps: if fast { 800 } else { 1200 },
                ..Default::default()
            }
        }
        HcRelaxMode::Fine => {
            // CPU-friendly MuMax analogue: plateau stopping on MEAN SQUARED torque + tightening.
            RelaxSettings {
                phase1_enabled: false,
                phase2_enabled: true,

                torque_metric: TorqueMetric::MeanSq,
                torque_threshold: None, // plateau mode

                max_err: if fast { 2.0e-5 } else { 1.0e-5 },
                tighten_floor: if fast { 1.0e-5 } else { sp2_tighten_floor() },
                tighten_factor: std::f64::consts::FRAC_1_SQRT_2,

                torque_check_stride: 1,
                torque_plateau_checks: 1,
                torque_plateau_min_checks: 1,
                torque_plateau_rel: 0.0,
                torque_plateau_abs: 0.0,

                max_accepted_steps: if fast { 25_000 } else { 80_000 },
                ..Default::default()
            }
        }
    };

    // Keep dt warm-start across coercivity steps (important for continuation).
    // Do NOT reset params.dt here.

    // Defensive: ensure max_err is not below the floor.
    if settings.max_err < settings.tighten_floor {
        settings.max_err = settings.tighten_floor;
    }

    relax_with_report(grid, m, params, material, rk23, FieldMask::Full, &mut settings)
}

// Optional tiny pre-minimize: only as a preconditioner (not the main solver).
fn sp2_preminimize(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &LLGParams,
    material: &Material,
) {
    let mut s = MinimizeSettings {
        torque_threshold: if sp2_fast_enabled() { 2e-3 } else { 8e-4 },
        max_iters: if sp2_fast_enabled() { 50 } else { 100 },
        ..Default::default()
    };

    // Keep it “tiny” by design: this is only to help occasional sticking on CPU.
    s.parallel = env_flag("LLG_MINIMIZE_PAR");
    let _ = minimize_damping_only(grid, m, params, material, FieldMask::Full, &s);
}

// -------------------------
// Coercivity scan (MuMax-like)
// -------------------------

fn find_hc_over_ms_mumax_style(
    grid: &Grid2D,
    m_work: &mut VectorField2D,
    m_rem: &VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    ms: f64,
) -> f64 {
    // Start from the remanent state once.
    m_work.data.clone_from(&m_rem.data);

    // Optional symmetry breaker (OFF by default): tiny deterministic perturbation of one cell.
    if env_flag("SP2_SYM_BREAK") && !m_work.data.is_empty() {
        let i = 0usize;
        let mut v = m_work.data[i];
        v[1] += 1e-6;
        m_work.data[i] = normalize(v);
    }

    let bc0 = BC0_MULT * ms;
    let target_step = (BC_STEP_MULT * step_mult() as f64) * ms;
    let bracket_step = (hc_start_mult() as f64) * target_step;

    params.dt = params.dt.clamp(1e-18, 1e-11);

    // --- Step 0: settle at bc0 using the *fine* criterion (MuMax scan starts here).
    //
    // CPU analogue of MuMax GPU numerical noise:
    // try a couple of tiny deterministic perturbations at bc0, relax each, and choose
    // the lowest-energy relaxed state as the continuation seed.
    set_bext_from_bc(params, bc0);
    rk23.invalidate_last_b_eff();

    let multiseed = env_bool_default_true("SP2_HC_MULTI_SEED"); // default ON; set SP2_HC_MULTI_SEED=0 to disable

    let mut best_m: VectorField2D;
    let mut best_rep: RelaxReport;
    let mut best_e: f64;

    let mut try_seed = |seed_id: u32, perturb: bool| -> (VectorField2D, RelaxReport, f64) {
        // Fresh candidate from the same remanent state each time.
        let mut m_cand = VectorField2D { grid: *grid, data: m_rem.data.clone() };

        if perturb {
            apply_micro_perturb(&mut m_cand, 1e-4, seed_id);
        }

        // Make seed runs comparable.
        params.dt = 1e-13;
        rk23.invalidate_last_b_eff();

        let rep = relax_coercivity_step(grid, &mut m_cand, params, material, rk23, HcRelaxMode::Fine);
        let e = compute_total_energy(grid, &m_cand, material, params.b_ext);
        (m_cand, rep, e)
    };

    // Seed 0: unperturbed
    {
        let (m_cand, rep, e) = try_seed(0, false);
        best_m = m_cand;
        best_rep = rep;
        best_e = e;
    }

    // Seeds 1..2: micro-perturbed (optional)
    if multiseed {
        for sid in 1..=2u32 {
            let (m_cand, rep, e) = try_seed(sid, true);
            if e < best_e {
                best_m = m_cand;
                best_rep = rep;
                best_e = e;
            }
        }
    }

    // Use the best candidate as the bc0 state
    m_work.data.clone_from(&best_m.data);
    let rep0 = best_rep;
    let msum0 = msum(m_work);

    if sp2_timing_enabled() {
        println!(
            "    [hc0/fine]  bc/Ms={:.6} msum={:.6} acc={} rej={} stop={:?} stage={:?} tau={:.3e} dt={:.3e}  (multi_seed={}, E={:.6e})",
            bc0 / ms,
            msum0,
            rep0.accepted_steps,
            rep0.rejected_steps,
            rep0.stop_reason,
            rep0.last_stage_stop,
            rep0.final_torque.unwrap_or(f64::NAN),
            rep0.final_dt,
            if multiseed { "ON" } else { "OFF" },
            best_e,
        );
    }

    if msum0 <= 0.0 {
        return bc0 / ms;
    }

    // --- Step 1: bracket quickly with a coarse step.
    // Important: do NOT let the bracket phase "accept" under-relaxed states.
    // If a trial point takes 0 accepted steps (no-op) or hits MaxAcceptedSteps,
    // we shrink the bracket step and retry from the last good (bc_low, m_low).
    let mut bc_low = bc0;
    let mut m_low = m_work.data.clone();

    let mut k_bracket: usize = 0;
    let mut step = bracket_step;

    let bc_high: f64 = loop {
        k_bracket += 1;

        let bc_try = bc_low + step;

        if bc_try >= HC_BC_CAP_OVER_MS * ms {
            println!(
                "    WARNING: hc_bracket reached bc cap without switching; returning bc/Ms at cap (bc/Ms={:.6}).",
                (HC_BC_CAP_OVER_MS * ms) / ms
            );
            return (HC_BC_CAP_OVER_MS * ms) / ms;
        }

        // Always start attempt from last accepted unswitched state.
        m_work.data.clone_from(&m_low);
        set_bext_from_bc(params, bc_try);
        rk23.invalidate_last_b_eff();

        if env_flag("SP2_PREMIN") {
            sp2_preminimize(grid, m_work, params, material);
        }

        let t0 = Instant::now();
        let rep = relax_coercivity_step(grid, m_work, params, material, rk23, HcRelaxMode::Bracket);
        let ms_now = msum(m_work);

        if sp2_timing_enabled() {
            println!(
                "    [hc_bracket] k={:>3} bc/Ms={:.6} step/Ms={:.6} msum={:.6} acc={} stop={:?} tau={:.3e} ({:.3}s)",
                k_bracket,
                bc_try / ms,
                step / ms,
                ms_now,
                rep.accepted_steps,
                rep.stop_reason,
                rep.final_torque.unwrap_or(f64::NAN),
                t0.elapsed().as_secs_f64(),
            );
        }

        // If the bracket relax couldn't do work (no-op) OR hit its step cap, don't accept it.
        // Instead refine the step and retry from the same low state.
        let bad_bracket_point = rep.accepted_steps == 0 || rep.stop_reason == RelaxStopReason::MaxAcceptedSteps;
        if bad_bracket_point && step > target_step {
            step *= 0.5;
            if step < target_step {
                step = target_step;
            }
            if sp2_timing_enabled() {
                println!(
                    "    [hc_bracket_refine] rejected bc/Ms={:.6} (acc={}, stop={:?}); shrinking step -> step/Ms={:.6}",
                    bc_try / ms,
                    rep.accepted_steps,
                    rep.stop_reason,
                    step / ms
                );
            }
            continue;
        }

        // Switched?
        if ms_now <= 0.0 {
            break bc_try;
        }

        // Not switched and "good enough": accept as new low state.
        bc_low = bc_try;
        m_low.clone_from(&m_work.data);

        if k_bracket > 800 {
            println!(
                "    WARNING: hc_bracket exceeded 800 iterations; returning current bc_low/Ms={:.6}.",
                bc_low / ms
            );
            return bc_low / ms;
        }
    };

    // --- Step 2: fine scan within the bracket at MuMax step size.
    m_work.data.clone_from(&m_low);

    let n_steps = (((bc_high - bc_low) / target_step).ceil() as usize).max(1);

    for i in 0..=n_steps {
        let mut bc_fine = bc_low + (i as f64) * target_step;
        if bc_fine > bc_high {
            bc_fine = bc_high;
        }

        set_bext_from_bc(params, bc_fine);
        rk23.invalidate_last_b_eff();

        let mut rep_fine;
        let mut ms_now;
        let mut tries = 0usize;

        loop {
            let t0 = Instant::now();
            rep_fine = relax_coercivity_step(grid, m_work, params, material, rk23, HcRelaxMode::Fine);
            ms_now = msum(m_work);

            if sp2_timing_enabled() {
                println!(
                    "    [hc_fine] i={:>3}/{:>3} bc/Ms={:.6} msum={:.6} acc={} stop={:?} tau={:.3e} ({:.3}s)",
                    i,
                    n_steps,
                    bc_fine / ms,
                    ms_now,
                    rep_fine.accepted_steps,
                    rep_fine.stop_reason,
                    rep_fine.final_torque.unwrap_or(f64::NAN),
                    t0.elapsed().as_secs_f64(),
                );
            }

            if rep_fine.stop_reason != RelaxStopReason::MaxAcceptedSteps {
                break;
            }
            if tries >= HC_FINE_RETRIES {
                break;
            }

            if env_flag("SP2_PREMIN") {
                sp2_preminimize(grid, m_work, params, material);
            }

            tries += 1;
        }

        if ms_now <= 0.0 {
            return bc_fine / ms;
        }

        if bc_fine >= bc_high {
            break;
        }
    }

    println!(
        "    WARNING: hc_fine did not detect switching within bracket; returning bc_high/Ms={:.6}",
        bc_high / ms
    );
    bc_high / ms
}
// -------------------------
// Entry point
// -------------------------

pub fn run_sp2() -> std::io::Result<()> {
    let d_min = env_usize("SP2_D_MIN", 29);
    let d_max = env_usize("SP2_D_MAX", 29);
    let rem_only = env_flag("SP2_REM_ONLY");

    // Exchange length
    let lex: f64 = (2.0 * A_EX / (MU0 * MS * MS)).sqrt();

    println!("SP2: d range = [{}..{}]", d_min, d_max);
    println!("SP2: Ms = {:.3e} A/m, Aex = {:.3e} J/m, Ku = {:.3e}", MS, A_EX, K_U);
    println!("SP2: lex = {:.3e} m", lex);
    println!("SP2: tighten_floor = {:.1e}", RELAX_TIGHTEN_FLOOR);
    println!(
        "SP2: grid_mode = {} (SP2_GRID_MODE), cell/lex = {:.3} (SP2_CELL_OVER_LEX), tighten_floor = {:.1e} (SP2_TIGHTEN_FLOOR)",
        sp2_grid_mode(),
        sp2_cell_over_lex(),
        sp2_tighten_floor(),
    );
    println!("SP2: hc bc0/Ms = {:.6}, step/Ms = {:.6} (mult={})", BC0_MULT, BC_STEP_MULT, step_mult());
    println!(
        "SP2: hc bracket start mult = {} (env SP2_HC_START_MULT)",
        hc_start_mult()
    );
    println!("SP2: fast mode = {}", if sp2_fast_enabled() { "ON" } else { "OFF" });
    println!("SP2: rem_only = {}", if rem_only { "ON" } else { "OFF" });
    println!("SP2: premin per hc step = {}", if env_flag("SP2_PREMIN") { "ON" } else { "OFF" });

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

    // Open output (append), write header if needed
    let file_exists = table_path.exists();
    let f = OpenOptions::new().create(true).append(true).open(&table_path)?;
    let mut w = BufWriter::new(f);

    let need_header = !file_exists || std::fs::metadata(&table_path)?.len() == 0;
    if need_header {
        writeln!(w, "d_lex,mx_rem,my_rem,hc_over_ms")?;
        w.flush()?;
        println!("SP2: wrote header -> {}", table_path.display());
    }

    for d_lex in (d_min..=d_max).rev() {
        if done.contains(&(d_lex as i32)) {
            println!("SP2 d/lex={:>2}: already done; skipping.", d_lex);
            continue;
        }

        let grid = build_sp2_grid(d_lex, lex);

        println!(
            "\nSP2 d/lex={:>2}: start (nx={}, ny={}, dx/lex={:.3}, dy/lex={:.3}, dz/lex={:.3})",
            d_lex,
            grid.nx,
            grid.ny,
            grid.dx / lex,
            grid.dy / lex,
            grid.dz / lex
        );

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
            alpha: 0.5, // physical alpha for LLG; relax dynamics are precession-suppressed anyway
            dt: 1e-13,
            b_ext: [0.0, 0.0, 0.0],
        };

        let mut rk23 = RK23Scratch::new(grid);

        // -------------------------
        // Remanence (cacheable)
        // -------------------------
        let cache_path = rem_cache_path(d_lex, &grid);

        let m = if let Some(m_cached) = read_rem_cache(&cache_path, d_lex, &grid)? {
            println!("SP2 d/lex={:>2}: remanence cache HIT -> {}", d_lex, cache_path.display());
            m_cached
        } else {
            println!("SP2 d/lex={:>2}: remanence cache MISS -> Relax()", d_lex);

            let mut m0 = VectorField2D::new(grid);
            let u0 = normalize([1.0, 0.3, 0.0]);
            m0.set_uniform(u0[0], u0[1], u0[2]);

            params.b_ext = [0.0, 0.0, 0.0];

            let t_rem = Instant::now();
            let rep = relax_remanence(&grid, &mut m0, &mut params, &material, &mut rk23);

            if sp2_timing_enabled() {
                println!(
                    "  [rem] relax: acc={} rej={} stop={:?} stage={:?} tau={:.3e} dt={:.3e} ({:.2}s)",
                    rep.accepted_steps,
                    rep.rejected_steps,
                    rep.stop_reason,
                    rep.last_stage_stop,
                    rep.final_torque.unwrap_or(f64::NAN),
                    rep.final_dt,
                    t_rem.elapsed().as_secs_f64(),
                );
                if rep.accepted_steps == 0 {
                    println!("  WARNING: remanence relax accepted_steps==0 (no-op).");
                }
            } else {
                println!("  [rem] done in {:.2}s  stop={:?}", t_rem.elapsed().as_secs_f64(), rep.stop_reason);
            }

            // Cache only if we didn't hit the accepted-step cap.
            if rep.stop_reason != RelaxStopReason::MaxAcceptedSteps {
                let hdr = RemCacheHeader::new(d_lex, &grid);
                write_rem_cache(&cache_path, hdr, &m0)?;
                println!("SP2 d/lex={:>2}: cached remanence -> {}", d_lex, cache_path.display());
            } else {
                println!("SP2 d/lex={:>2}: WARNING: remanence relax hit MaxAcceptedSteps; NOT caching.", d_lex);
            }

            m0
        };

        let rem = avg_m(&m);
        let mx_rem = rem[0];
        let my_rem = rem[1];
        let mz_rem = rem[2];
        println!(
            "SP2 d/lex={:>2}: rem(mx,my,mz,msum)=({:.6},{:.6},{:.6},{:.6})",
            d_lex,
            mx_rem,
            my_rem,
            mz_rem,
            mx_rem + my_rem + mz_rem
        );

        // -------------------------
        // Coercivity (MuMax-like scan)
        // -------------------------
        let hc_over_ms = if rem_only {
            f64::NAN
        } else {
            let m_rem = VectorField2D { grid, data: m.data.clone() };
            let mut m_work = VectorField2D::new(grid);

            println!("SP2 d/lex={:>2}: coercivity scan start", d_lex);

            let t_hc = Instant::now();
            let hc = find_hc_over_ms_mumax_style(&grid, &mut m_work, &m_rem, &mut params, &material, &mut rk23, MS);
            println!(
                "SP2 d/lex={:>2}: coercivity done in {:.1}s  hc/Ms={:.6}",
                d_lex,
                t_hc.elapsed().as_secs_f64(),
                hc
            );
            hc
        };

        // Restore B_ext = 0 (matches MuMax script end-of-loop)
        params.b_ext = [0.0, 0.0, 0.0];

        writeln!(
            w,
            "{:.0},{:.16e},{:.16e},{:.16e}",
            d_lex as f64,
            mx_rem,
            my_rem,
            hc_over_ms
        )?;
        w.flush()?;
        println!("SP2 d/lex={:>2}: row written", d_lex);
    }

    println!("\nSP2 complete. Output: {}", table_path.display());
    Ok(())
}