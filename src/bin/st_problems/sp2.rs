// src/bin/st_problems/sp2.rs
//
// Standard Problem #2 (MuMag): remanence and coercivity vs d/lex.
//
// Core algorithm:
//   1) Inner solve (fixed bc): equilibrate m to a (meta)stable state at that field.
//   2) Outer solve: scan/bracket + bisect bc until the equilibrated state switches
//      from the “positive branch” to the “negative branch”, using sign(m_sum)
//      where m_sum = <mx> + <my> + <mz>.
//
// Practical CPU note:
//   Pure RK23 relax can be extremely slow for large d/lex (demag FFT dominates).
//   We therefore use a damping-only minimiser (1 field build / iter) as the
//   workhorse, and then a *bounded* RK23 relax in chunks.
//
// Key fix vs “minimise-only bracket scan”:
//   For coercivity, coarse bracket steps MUST include a short relax so continuation
//   actually follows the metastable equilibrium branch. Strict points always relax,
//   but stop by plateau/Δm (not by forcing torque to 1e-4 every time).
//
// Run:
//   cargo run --release --bin st_problems -- sp2
//
// Post-process (MuMax overlay plots):
//
// python3 scripts/compare_sp2.py \
//   --mumax-root mumax_outputs/st_problems/sp2 \
//   --rust-root runs/st_problems/sp2 \
//   --metrics \
//   --out runs/st_problems/sp2/sp2_overlay.png
//
// Useful env vars:
//   SP2_D_MIN=1 SP2_D_MAX=30
//   SP2_VERBOSE=1            (more prints)
//   SP2_TIMING=1             (timings)
// SP2_D_MIN=1 SP2_D_MAX=30 SP2_VERBOSE=1 SP2_TIMING=1 cargo run --release --bin st_problems -- sp2
//
//   # inner-solve thresholds
//   SP2_TORQUE_REM=1e-4      (Tesla) stop for remanence minimiser/relax
//   SP2_TORQUE_HC=1e-4       (Tesla) torque threshold used for minimiser guidance (strict)
//   SP2_DM_STOP=5e-7         (dimensionless) optional minimiser early-stop on max |Δm|
//
//   # iteration budgets
//   SP2_MIN_ITERS_REM=30000
//   SP2_MIN_ITERS_HC=20000
//   SP2_MIN_ITERS_COARSE=400
//   SP2_RELAX_STEPS_REM=12000   (0 disables remanence relax polish)
//   SP2_RELAX_STEPS_HC=8000     (0 disables coercivity relax; not recommended)
//   SP2_RELAX_CHUNK=800         (chunk size for prints + plateau logic)
//
//   # outer-solve resolution
//   SP2_BRACKET_MULT=20      (bracket step = mult * 0.00005*Ms)
//   SP2_TARGET_MULT=1        (bisection target = mult * 0.00005*Ms)
//
//   # OVF outputs (OOMMF OVF 2.0 binary4)
//   SP2_OVF=1                (default: ON; set 0/false/no to disable)
//   SP2_OVF_EVERY_NS=1       (default: 1 ns; <=0 disables interval frames; finals still written)
// NOTE: aliases supported:
//   SP2_REM_TORQUE_STOP  -> SP2_TORQUE_REM
//   SP2_HC_TORQUE_STOP   -> SP2_TORQUE_HC
//
// Output:
//   runs/st_problems/sp2/table.csv
//   runs/st_problems/sp2/ovf/...

use std::collections::HashSet;
use std::fs::{OpenOptions, create_dir_all};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use llg_sim::effective_field::FieldMask;
use llg_sim::grid::Grid2D;
use llg_sim::grid_sp2::{Sp2GridPolicy, build_sp2_grid};
use llg_sim::llg::RK23Scratch;
use llg_sim::minimize::{MinimizeReport, MinimizeSettings, minimize_damping_only};
use llg_sim::ovf::{OvfMeta, write_ovf2_rectangular_binary4};
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, MU0, Material};
use llg_sim::relax::{RelaxReport, RelaxSettings, TorqueMetric, relax_with_report};
use llg_sim::vec3::normalize;
use llg_sim::vector_field::VectorField2D;

// -------------------------
// Heuristics (few + meaningful)
// -------------------------

/// Coarse bracket steps use looser minimiser torque target than strict points.
const COARSE_TORQUE_MULT: f64 = 5.0;

/// When continuation gets close-ish to switching, do a strict re-anchor at same bc.
const MSUM_ANCHOR_THRESHOLD: f64 = 0.12;

/// Plateau detection in coercivity relax: require small improvement for N chunks.
const HC_PLATEAU_CHUNKS: usize = 6;
const HC_PLATEAU_IMPROVE_REL: f64 = 1e-4;

/// When plateauing, also require Δm between chunk endpoints to be tiny.
/// We reuse dm_stop (if enabled) as the scale; otherwise fall back to a safe default.
const DM_PLATEAU_MULT: f64 = 5.0;
const DM_PLATEAU_FALLBACK: f64 = 5e-7;

// -------------------------
// Small env helpers
// -------------------------

fn env_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no")
        })
        .unwrap_or(false)
}

fn env_bool_default(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no")
        }
        Err(_) => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_f64_any(names: &[&str], default: f64) -> f64 {
    for &n in names {
        if let Ok(v) = std::env::var(n)
            && let Ok(x) = v.trim().parse::<f64>()
        {
            return x;
        }
    }
    default
}

fn d_min_max_from_env() -> (usize, usize) {
    let d_min = env_usize("SP2_D_MIN", 1);
    let d_max = env_usize("SP2_D_MAX", 30);
    (d_min.min(d_max), d_min.max(d_max))
}

// -------------------------
// OVF naming mode (legacy vs MuMax-compatible)
// -------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sp2OvfNaming {
    LegacyTagTime,
    MumaxM7,
}

fn sp2_ovf_naming_from_env() -> Sp2OvfNaming {
    if let Ok(v) = std::env::var("SP2_OVF_NAMING") {
        let s = v.trim().to_ascii_lowercase();
        if matches!(s.as_str(), "m" | "mumax" | "m7" | "m0000000") {
            return Sp2OvfNaming::MumaxM7;
        }
        if matches!(s.as_str(), "legacy" | "tag" | "tagtime") {
            return Sp2OvfNaming::LegacyTagTime;
        }
    }
    Sp2OvfNaming::LegacyTagTime
}

// -------------------------
// OVF time-series dumper (best-effort snapshots at fixed time labels)
// -------------------------

struct OvfSeries {
    dir: PathBuf,
    tag: String,
    every: f64,
    next_dump: f64,
    frame: u64,
    naming: Sp2OvfNaming,
}

impl OvfSeries {
    fn new(dir: PathBuf, tag: String, every: f64, naming: Sp2OvfNaming) -> std::io::Result<Self> {
        create_dir_all(&dir)?;
        Ok(Self {
            dir,
            tag,
            every: every.max(0.0),
            next_dump: 0.0,
            frame: 0,
            naming,
        })
    }

    fn time_label(t: f64) -> String {
        let t_ns = t * 1e9;
        if t_ns.is_finite() && (t_ns - t_ns.round()).abs() < 1e-6 && t_ns.abs() < 1e12 {
            format!("{:06}ns", t_ns.round() as i64)
        } else {
            format!("{:.6e}s", t).replace('.', "p")
        }
    }

    fn dump_at(&mut self, grid: &Grid2D, m: &VectorField2D, t_label: f64) -> std::io::Result<()> {
        let path = match self.naming {
            Sp2OvfNaming::LegacyTagTime => {
                let tl = Self::time_label(t_label);
                self.dir
                    .join(format!("{}_t_{}_f{:06}.ovf", self.tag, tl, self.frame))
            }
            Sp2OvfNaming::MumaxM7 => self.dir.join(format!("m{:07}.ovf", self.frame)),
        };

        // Ensure directory exists (defensive)
        if let Some(parent) = path.parent() {
            create_dir_all(parent)?;
        }

        // Keep time information in a MuMax-compatible Desc line for viewers.
        let mut meta = OvfMeta::magnetization().with_total_sim_time(t_label);
        // Also keep the old human-readable description information.
        meta.push_desc_line(format!("SP2 {} t_label={:.6e}s", self.tag, t_label));

        write_ovf2_rectangular_binary4(&path, grid, m, &meta)?;
        self.frame += 1;
        Ok(())
    }

    fn dump_initial(&mut self, grid: &Grid2D, m: &VectorField2D) -> std::io::Result<()> {
        self.next_dump = 0.0;
        self.dump_at(grid, m, 0.0)?;
        self.next_dump = self.every;
        Ok(())
    }

    fn maybe_dump_due(
        &mut self,
        grid: &Grid2D,
        m: &VectorField2D,
        t_approx: f64,
    ) -> std::io::Result<()> {
        if self.every <= 0.0 {
            return Ok(());
        }
        let eps = 1e-15_f64;
        while t_approx + eps >= self.next_dump {
            let tl = self.next_dump;
            self.dump_at(grid, m, tl)?;
            self.next_dump += self.every;
        }
        Ok(())
    }
}

// -------------------------
// Magnetisation stats
// -------------------------

fn avg_m(m: &VectorField2D) -> [f64; 3] {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    let n = m.data.len() as f64;
    for v in &m.data {
        sx += v[0];
        sy += v[1];
        sz += v[2];
    }
    [sx / n, sy / n, sz / n]
}

fn msum(m_avg: [f64; 3]) -> f64 {
    m_avg[0] + m_avg[1] + m_avg[2]
}

fn max_dm_between(a: &VectorField2D, b: &VectorField2D) -> f64 {
    let mut mmax = 0.0;
    for (va, vb) in a.data.iter().zip(b.data.iter()) {
        let dx = (va[0] - vb[0]).abs();
        let dy = (va[1] - vb[1]).abs();
        let dz = (va[2] - vb[2]).abs();
        let d = dx.max(dy).max(dz);
        if d > mmax {
            mmax = d;
        }
    }
    mmax
}

// -------------------------
// CSV resume support
// -------------------------

fn read_done_d_values(table_path: &Path) -> std::io::Result<HashSet<i32>> {
    let mut done = HashSet::new();
    if !table_path.exists() {
        return Ok(done);
    }

    let f = std::fs::File::open(table_path)?;
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
        if let Some(first) = s.split(',').next()
            && let Ok(dv) = first.trim().parse::<i32>()
        {
            done.insert(dv);
        }
        line.clear();
    }

    Ok(done)
}

// -------------------------
// Field application
// -------------------------

fn set_bext_from_bc(params: &mut LLGParams, bc_amps_per_m: f64) {
    // MuMax SP2 uses a field along (-1,-1,-1)/sqrt(3).
    // We store B_ext (Tesla): B = mu0 * H.
    let b = -bc_amps_per_m * MU0 / 3.0_f64.sqrt();
    params.b_ext = [b, b, b];
}

// -------------------------
// Inner solve
// -------------------------

#[derive(Clone, Copy, Debug)]
enum InnerKind {
    Remanence,
    CoercivityStrict,
    CoercivityCoarse,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct InnerDiag {
    bc_amps_per_m: f64,
    kind: InnerKind,
    min_rep: MinimizeReport,
    relax_rep: Option<RelaxReport>,
    m_avg: [f64; 3],
    m_sum: f64,
    took_s: f64,
}

fn make_min_settings(
    kind: InnerKind,
    torque_thr_t: f64,
    max_iters: usize,
    dm_stop: Option<f64>,
) -> MinimizeSettings {
    let mut s = MinimizeSettings {
        torque_threshold: torque_thr_t,
        max_iters,
        ..Default::default()
    };

    // dm_stop is allowed but should be treated as "secondary completion".
    s.dm_stop = dm_stop;

    match kind {
        InnerKind::CoercivityCoarse | InnerKind::CoercivityStrict => {
            // Stabilise λ behaviour; avoid collapse-to-zero motion.
            s.lambda0 = 1e-2;
            s.lambda_min = 5e-5;
            s.lambda_max = 5e-2;
            s.shrink = 0.90;
            s.grow = 1.05;
            s.min_iters_before_stall = 500;
            s.stall_rel = 1e-4;
            s.stall_iters = match kind {
                InnerKind::CoercivityCoarse => 10_000,
                _ => 20_000,
            };
        }
        InnerKind::Remanence => {
            // Keep remanence behaviour as close to your current (working) setup as possible.
        }
    }

    s
}

/// Relax settings for remanence: keep threshold.
/// Relax settings for coercivity: plateau stopping (threshold=None) + mean torque metric.
fn make_relax_settings(kind: InnerKind, torque_thr_t: f64, max_steps: usize) -> RelaxSettings {
    let mut s = RelaxSettings {
        phase1_enabled: false,
        phase2_enabled: true,
        max_accepted_steps: max_steps,
        torque_check_stride: 200,
        tighten_floor: 1e-6,
        max_err: 1e-5,
        ..Default::default()
    };

    s.torque_metric = match kind {
        InnerKind::Remanence => TorqueMetric::Max,
        _ => TorqueMetric::Mean,
    };

    s.torque_threshold = match kind {
        InnerKind::Remanence => Some(torque_thr_t),
        _ => None, // plateau stopping
    };

    s.torque_plateau_checks = 8;
    s.torque_plateau_min_checks = 5;
    s.torque_plateau_rel = 1e-3;
    s.torque_plateau_abs = 0.0;

    s
}

fn fmt_opt_e(x: Option<f64>) -> String {
    match x {
        Some(v) if v.is_finite() => format!("{:.3e}", v),
        Some(v) => format!("{:?}", v),
        None => "None".to_string(),
    }
}

fn bc_label(bc_over_ms: f64) -> String {
    format!("{:.6}", bc_over_ms).replace('.', "p")
}

/// Chunked relax with plateau detection + Δm guard.
/// This is what prevents “run 30 min then hit max steps”.
#[allow(clippy::too_many_arguments)]
fn relax_in_chunks(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    kind: InnerKind,
    torque_thr_t: f64,
    total_steps: usize,
    chunk_steps: usize,
    dm_stop: Option<f64>,
    verbose: bool,
    prefix: &str,
    mut ovf: Option<&mut OvfSeries>,
) -> Option<RelaxReport> {
    if total_steps == 0 {
        return None;
    }

    // Interval dumping is best-effort; errors do not affect simulation.
    let mut t_approx = 0.0_f64;
    if let Some(series) = ovf.as_deref_mut() {
        if let Err(e) = series.dump_initial(grid, m) {
            eprintln!("      [ovf] WARNING: initial dump failed: {}", e);
        }
    }

    let dm_plateau = dm_stop.unwrap_or(DM_PLATEAU_FALLBACK) * DM_PLATEAU_MULT;

    let mut remaining = total_steps;
    let mut last_metric: Option<f64> = None;
    let mut plateau_count = 0usize;

    let mut last_report: Option<RelaxReport> = None;
    let mut m_prev = VectorField2D::new(*grid);
    m_prev.data.clone_from(&m.data);

    while remaining > 0 {
        let step_now = chunk_steps.min(remaining);
        remaining -= step_now;

        let mut rs = make_relax_settings(kind, torque_thr_t, step_now);

        if !(params.dt.is_finite()) || params.dt <= 0.0 {
            params.dt = 1e-13;
        }

        let rep = relax_with_report(grid, m, params, material, rk23, FieldMask::Full, &mut rs);
        let m_avg = avg_m(m);
        let m_sum = msum(m_avg);
        let dm_between = max_dm_between(&m_prev, m);

        // Choose a scalar "metric" from report for plateau detection:
        // prefer final_torque (metric), else fall back to final_torque_max.
        let metric_now = rep.final_torque.or(rep.final_torque_max);

        // Plateau accounting
        if let (Some(prev), Some(now)) = (last_metric, metric_now) {
            if prev.is_finite() && now.is_finite() {
                let rel = (prev - now).abs() / prev.abs().max(1e-30);
                if rel < HC_PLATEAU_IMPROVE_REL {
                    plateau_count += 1;
                } else {
                    plateau_count = 0;
                }
            }
        } else {
            plateau_count = 0;
        }
        last_metric = metric_now;

        // Approximate physical time advanced during this chunk.
        // (We only have final_dt; this is purely for OVF time-labeling.)
        if rep.final_dt.is_finite() && rep.final_dt > 0.0 {
            t_approx += (rep.accepted_steps as f64) * rep.final_dt;
        }

        if let Some(series) = ovf.as_deref_mut() {
            if let Err(e) = series.maybe_dump_due(grid, m, t_approx) {
                eprintln!("      [ovf] WARNING: interval dump failed: {}", e);
            }
        }

        if verbose {
            println!(
                "      [{}] chunk={} acc={} rej={} stop={:?} dt={:.2e} max_err={:.2e} metric={} torque_max={} dm_chunk={:.3e} m_sum={:.6} <m>=({:.6},{:.6},{:.6})",
                prefix,
                step_now,
                rep.accepted_steps,
                rep.rejected_steps,
                rep.stop_reason,
                rep.final_dt,
                rep.final_max_err,
                fmt_opt_e(rep.final_torque),
                fmt_opt_e(rep.final_torque_max),
                dm_between,
                m_sum,
                m_avg[0],
                m_avg[1],
                m_avg[2],
            );
        }

        // If relax finished "naturally" (not by max steps), stop.
        if rep.stop_reason != llg_sim::relax::RelaxStopReason::MaxAcceptedSteps {
            last_report = Some(rep);
            break;
        }

        // Plateau stop: require both (a) plateau_count and (b) Δm small.
        if plateau_count >= HC_PLATEAU_CHUNKS && dm_between <= dm_plateau {
            if verbose {
                println!(
                    "      [{}] PLATEAU_ACCEPT: plateau_chunks={} dm_chunk={:.3e} (<= {:.3e})",
                    prefix, plateau_count, dm_between, dm_plateau
                );
            }
            last_report = Some(rep);
            break;
        }

        // Prepare next chunk
        m_prev.data.clone_from(&m.data);
        last_report = Some(rep);
    }

    last_report
}

#[allow(clippy::too_many_arguments)]
fn inner_equilibrate(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    bc_amps_per_m: f64,
    kind: InnerKind,
    torque_thr_t: f64,
    dm_stop: Option<f64>,
    min_max_iters: usize,
    relax_total_steps: usize,
    relax_chunk: usize,
    verbose: bool,
    timing: bool,
    label: &str,
    ovf_dir: Option<&Path>,
    ovf_every_s: f64,
) -> InnerDiag {
    set_bext_from_bc(params, bc_amps_per_m);

    let t0 = Instant::now();

    if verbose {
        match kind {
            InnerKind::Remanence => {
                println!(
                    "  [rem] {}: minimise (tau_max<{:.1e}T OR dm_stop={})",
                    label,
                    torque_thr_t,
                    dm_stop
                        .map(|x| format!("{:.1e}", x))
                        .unwrap_or_else(|| "OFF".to_string())
                );
            }
            InnerKind::CoercivityStrict => {
                println!(
                    "    [hc] {}: STRICT inner at bc/Ms={:.6}",
                    label,
                    bc_amps_per_m / material.ms
                );
            }
            InnerKind::CoercivityCoarse => {
                println!(
                    "    [hc] {}: COARSE inner at bc/Ms={:.6}",
                    label,
                    bc_amps_per_m / material.ms
                );
            }
        }
    }

    // ---- Minimiser (precondition) ----
    let min_settings = make_min_settings(kind, torque_thr_t, min_max_iters, dm_stop);
    let min_rep = minimize_damping_only(grid, m, params, material, FieldMask::Full, &min_settings);

    let m_avg_min = avg_m(m);
    let m_sum_min = msum(m_avg_min);

    if verbose {
        println!(
            "      [min] iters={} conv={} dm_conv={} stalled={}  tau_max={:.3e} tau_mean={:.3e} dm_max={:.3e} lam={:.3e}  <m>_min=({:.6},{:.6},{:.6}) m_sum_min={:.6}",
            min_rep.iters,
            if min_rep.converged { 1 } else { 0 },
            if min_rep.dm_converged { 1 } else { 0 },
            if min_rep.stalled { 1 } else { 0 },
            min_rep.final_torque,
            min_rep.final_tmean,
            min_rep.final_max_dm,
            min_rep.final_lambda,
            m_avg_min[0],
            m_avg_min[1],
            m_avg_min[2],
            m_sum_min
        );
    }

    // ---- Relax polish ----
    // Remanence: preserve your current approach (thresholded relax).
    // Coercivity: ALWAYS relax (even coarse, but coarse uses fewer steps).
    let mut ovf_series: Option<OvfSeries> = None;
    if let Some(base_dir) = ovf_dir {
        let naming = sp2_ovf_naming_from_env();

        // Time-series go into ovf/dXX/rem and ovf/dXX/hc. Final states are written in ovf/dXX.
        let tag = match kind {
            InnerKind::Remanence => "rem".to_string(),
            InnerKind::CoercivityStrict => format!(
                "{}_strict_bc{}",
                label,
                bc_label(bc_amps_per_m / material.ms)
            ),
            InnerKind::CoercivityCoarse => format!(
                "{}_coarse_bc{}",
                label,
                bc_label(bc_amps_per_m / material.ms)
            ),
        };

        // Legacy mode keeps current filenames in ovf/dXX/rem and ovf/dXX/hc.
        // MuMax mode uses m0000000.ovf naming, so put coercivity series in its own subfolder.
        let series_dir: PathBuf = match (kind, naming) {
            (InnerKind::Remanence, _) => base_dir.join("rem"),
            (_, Sp2OvfNaming::LegacyTagTime) => base_dir.join("hc"),
            (_, Sp2OvfNaming::MumaxM7) => base_dir.join("hc").join(&tag),
        };

        if ovf_every_s.is_finite() && ovf_every_s > 0.0 {
            match OvfSeries::new(series_dir, tag, ovf_every_s, naming) {
                Ok(s) => ovf_series = Some(s),
                Err(e) => eprintln!("      [ovf] WARNING: cannot init series: {}", e),
            }
        }
    }

    let relax_rep = match kind {
        InnerKind::Remanence => {
            if relax_total_steps > 0 {
                // Keep remanence relax as-is but in chunks for prints/visibility.
                relax_in_chunks(
                    grid,
                    m,
                    params,
                    material,
                    rk23,
                    kind,
                    torque_thr_t,
                    relax_total_steps,
                    relax_chunk,
                    dm_stop,
                    verbose,
                    "relax_rem",
                    ovf_series.as_mut(),
                )
            } else {
                None
            }
        }
        InnerKind::CoercivityStrict | InnerKind::CoercivityCoarse => {
            if relax_total_steps > 0 {
                relax_in_chunks(
                    grid,
                    m,
                    params,
                    material,
                    rk23,
                    kind,
                    torque_thr_t,
                    relax_total_steps,
                    relax_chunk,
                    dm_stop,
                    verbose,
                    "relax_hc",
                    ovf_series.as_mut(),
                )
            } else {
                None
            }
        }
    };

    let m_avg = avg_m(m);
    let m_sum = msum(m_avg);
    let took_s = t0.elapsed().as_secs_f64();

    if verbose {
        println!(
            "      [inner_end] bc/Ms={:.6} <m>=({:.6},{:.6},{:.6}) m_sum={:.6}",
            bc_amps_per_m / material.ms,
            m_avg[0],
            m_avg[1],
            m_avg[2],
            m_sum
        );
    } else if timing {
        println!(
            "      [sp2 timing] {:?} {} took {:.2}s",
            kind, label, took_s
        );
    }

    InnerDiag {
        bc_amps_per_m,
        kind,
        min_rep,
        relax_rep,
        m_avg,
        m_sum,
        took_s,
    }
}

// -------------------------
// Outer solve: bracket + bisect (sign of m_sum)
// -------------------------

#[allow(clippy::too_many_arguments)]
fn find_hc_over_ms(
    grid: &Grid2D,
    m_rem: &VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    torque_thr_hc: f64,
    dm_stop: Option<f64>,
    min_iters_hc: usize,
    min_iters_coarse: usize,
    relax_steps_hc: usize,
    relax_chunk: usize,
    bracket_mult: f64,
    target_mult: f64,
    verbose: bool,
    timing: bool,
    ovf_dir: Option<&Path>,
    ovf_every_s: f64,
    mut m_hc_out: Option<&mut VectorField2D>,
    mut m_hc_pos_out: Option<&mut VectorField2D>,
    mut m_hc_best_out: Option<&mut VectorField2D>,
) -> f64 {
    let ms = material.ms;

    let bc0 = 0.0445 * ms;
    let base_step = 0.00005 * ms;
    let bc_bracket_step = (bracket_mult * base_step).max(base_step);
    let bc_target_step = (target_mult * base_step).max(base_step);
    let bc_cap = 0.25 * ms;

    // Coarse uses looser minimiser target, and fewer relax steps.
    let torque_thr_coarse = (COARSE_TORQUE_MULT * torque_thr_hc).max(torque_thr_hc);
    let relax_steps_coarse = (relax_steps_hc / 6)
        .max(relax_chunk)
        .max(800)
        .min(relax_steps_hc);

    // Continuation buffer
    let mut m_work = VectorField2D::new(*grid);
    m_work.data.clone_from(&m_rem.data);

    // bc0 strict
    let mut bc_low = bc0;
    let rep0 = inner_equilibrate(
        grid,
        &mut m_work,
        params,
        material,
        rk23,
        bc_low,
        InnerKind::CoercivityStrict,
        torque_thr_hc,
        dm_stop,
        min_iters_hc,
        relax_steps_hc,
        relax_chunk,
        verbose,
        timing,
        "bc0",
        ovf_dir,
        ovf_every_s,
    );

    if rep0.m_sum <= 0.0 {
        if verbose {
            println!("    [hc] switched already at bc0; returning bc0/Ms");
        }
        if let Some(out) = m_hc_out.as_deref_mut() {
            out.data.clone_from(&m_work.data);
        }
        if let Some(out) = m_hc_pos_out.as_deref_mut() {
            out.data.clone_from(&m_work.data);
        }
        if let Some(out) = m_hc_best_out.as_deref_mut() {
            out.data.clone_from(&m_work.data);
        }
        return bc_low / ms;
    }

    // last strict-positive seed
    let mut m_low_state = VectorField2D::new(*grid);
    m_low_state.data.clone_from(&m_work.data);

    // Track the strict-equilibrated state whose |m_sum| is smallest (diagnostic only).
    let mut best_abs_msum: f64 = rep0.m_sum.abs();
    let mut _best_bc: f64 = bc_low;
    let mut m_best_state = VectorField2D::new(*grid);
    m_best_state.data.clone_from(&m_work.data);

    // last strict-negative seed (filled when bracketing/bisecting finds negative)
    let mut m_high_state = VectorField2D::new(*grid);

    // Scan upwards until we find a verified negative
    let mut bc_high = bc_low;
    let mut k = 0usize;

    loop {
        bc_high += bc_bracket_step;
        k += 1;
        if bc_high > bc_cap {
            println!("    WARNING: failed to bracket Hc before bc_cap; returning bc_cap/Ms");
            if let Some(out) = m_hc_out.as_deref_mut() {
                out.data.clone_from(&m_work.data);
            }
            return bc_cap / ms;
        }

        // Coarse continuation step (BUT includes short relax!)
        let rep_coarse = inner_equilibrate(
            grid,
            &mut m_work,
            params,
            material,
            rk23,
            bc_high,
            InnerKind::CoercivityCoarse,
            torque_thr_coarse,
            dm_stop,
            min_iters_coarse,
            relax_steps_coarse,
            relax_chunk,
            verbose,
            timing,
            "bracket",
            ovf_dir,
            ovf_every_s,
        );

        if verbose {
            println!(
                "    [bracket {:>3}] bc/Ms={:.6} m_sum={:.6}",
                k,
                bc_high / ms,
                rep_coarse.m_sum
            );
        }

        // If coarse says flipped: verify strictly from last strict-positive.
        if rep_coarse.m_sum <= 0.0 {
            let mut m_verify = VectorField2D::new(*grid);
            m_verify.data.clone_from(&m_low_state.data);

            let rep_verify = inner_equilibrate(
                grid,
                &mut m_verify,
                params,
                material,
                rk23,
                bc_high,
                InnerKind::CoercivityStrict,
                torque_thr_hc,
                dm_stop,
                min_iters_hc,
                relax_steps_hc,
                relax_chunk,
                verbose,
                timing,
                "verify",
                ovf_dir,
                ovf_every_s,
            );
            if rep_verify.m_sum.abs() < best_abs_msum {
                best_abs_msum = rep_verify.m_sum.abs();
                _best_bc = bc_high;
                m_best_state.data.clone_from(&m_verify.data);
            }

            if rep_verify.m_sum <= 0.0 {
                // bracket found: [bc_low, bc_high]
                m_high_state.data.clone_from(&m_verify.data);
                break;
            }

            // still positive under strict -> promote
            bc_low = bc_high;
            m_low_state.data.clone_from(&m_verify.data);
            m_work.data.clone_from(&m_verify.data);
            continue;
        }

        // Still positive. If close to switch, strict re-anchor *at same bc_high*.
        if rep_coarse.m_sum.abs() < MSUM_ANCHOR_THRESHOLD {
            let rep_anchor = inner_equilibrate(
                grid,
                &mut m_work,
                params,
                material,
                rk23,
                bc_high,
                InnerKind::CoercivityStrict,
                torque_thr_hc,
                dm_stop,
                min_iters_hc,
                relax_steps_hc,
                relax_chunk,
                verbose,
                timing,
                "anchor",
                ovf_dir,
                ovf_every_s,
            );
            if rep_anchor.m_sum.abs() < best_abs_msum {
                best_abs_msum = rep_anchor.m_sum.abs();
                _best_bc = bc_high;
                m_best_state.data.clone_from(&m_work.data);
            }

            if rep_anchor.m_sum > 0.0 {
                bc_low = bc_high;
                m_low_state.data.clone_from(&m_work.data);
            } else {
                // strict flipped here -> bracket
                m_high_state.data.clone_from(&m_work.data);
                break;
            }
        } else {
            // Promote continuation seed (coarse relaxed state).
            bc_low = bc_high;
            m_low_state.data.clone_from(&m_work.data);
        }
    }

    if verbose {
        println!(
            "    [hc] bracket found: low={:.6} high={:.6}",
            bc_low / ms,
            bc_high / ms
        );
    }

    // Bisection (strict only), always seeded from last strict-positive.
    let mut iter = 0usize;
    while (bc_high - bc_low) > bc_target_step && iter < 64 {
        iter += 1;
        let bc_mid = 0.5 * (bc_low + bc_high);

        let mut m_mid = VectorField2D::new(*grid);
        m_mid.data.clone_from(&m_low_state.data);

        let rep_mid = inner_equilibrate(
            grid,
            &mut m_mid,
            params,
            material,
            rk23,
            bc_mid,
            InnerKind::CoercivityStrict,
            torque_thr_hc,
            dm_stop,
            min_iters_hc,
            relax_steps_hc,
            relax_chunk,
            verbose,
            timing,
            "bisect",
            ovf_dir,
            ovf_every_s,
        );
        if rep_mid.m_sum.abs() < best_abs_msum {
            best_abs_msum = rep_mid.m_sum.abs();
            _best_bc = bc_mid;
            m_best_state.data.clone_from(&m_mid.data);
        }

        if verbose {
            println!(
                "    [bisect {:>2}] low={:.6} high={:.6} mid={:.6}  m_sum={:.6}",
                iter,
                bc_low / ms,
                bc_high / ms,
                bc_mid / ms,
                rep_mid.m_sum
            );
        }

        if rep_mid.m_sum > 0.0 {
            bc_low = bc_mid;
            m_low_state.data.clone_from(&m_mid.data);
        } else {
            bc_high = bc_mid;
            m_high_state.data.clone_from(&m_mid.data);
        }
    }

    if let Some(out) = m_hc_out.as_deref_mut() {
        out.data.clone_from(&m_high_state.data);
    }
    if let Some(out) = m_hc_pos_out.as_deref_mut() {
        out.data.clone_from(&m_low_state.data);
    }
    if let Some(out) = m_hc_best_out.as_deref_mut() {
        out.data.clone_from(&m_best_state.data);
    }

    // Diagnostic values best_abs_msum and best_bc are tracked but do not affect selection.
    bc_high / ms
}

// -------------------------
// Main entry
// -------------------------

pub fn run_sp2() -> std::io::Result<()> {
    // MuMax SP2 constants
    let ms: f64 = 1000e3;
    let a_ex: f64 = 10e-12;
    let k_u: f64 = 0.0;
    let lex: f64 = (2.0 * a_ex / (MU0 * ms * ms)).sqrt();

    let (d_min, d_max) = d_min_max_from_env();
    let verbose = env_bool("SP2_VERBOSE");
    let timing = env_bool("SP2_TIMING");

    // Accept both canonical and alias names.
    let torque_rem = env_f64_any(&["SP2_TORQUE_REM", "SP2_REM_TORQUE_STOP"], 1e-4);
    let torque_hc = env_f64_any(&["SP2_TORQUE_HC", "SP2_HC_TORQUE_STOP"], 1e-4);

    let dm_stop_v = env_f64("SP2_DM_STOP", 5e-7);
    let dm_stop = if dm_stop_v > 0.0 {
        Some(dm_stop_v)
    } else {
        None
    };

    let min_iters_rem = env_usize("SP2_MIN_ITERS_REM", 30_000);
    let min_iters_hc = env_usize("SP2_MIN_ITERS_HC", 20_000);
    let min_iters_coarse = env_usize("SP2_MIN_ITERS_COARSE", 400);

    let relax_steps_rem = env_usize("SP2_RELAX_STEPS_REM", 12_000);
    let relax_steps_hc = env_usize("SP2_RELAX_STEPS_HC", 8_000);
    let relax_chunk = env_usize("SP2_RELAX_CHUNK", 800).max(200);

    let bracket_mult = env_f64("SP2_BRACKET_MULT", 20.0);
    let target_mult = env_f64("SP2_TARGET_MULT", 1.0);

    // OVF outputs
    let ovf_enabled = env_bool_default("SP2_OVF", true);
    let ovf_every_ns = env_f64("SP2_OVF_EVERY_NS", 1.0);
    let ovf_every_s = (ovf_every_ns * 1e-9).max(0.0);

    println!("SP2: d range = [{}..{}]", d_min, d_max);
    println!("SP2: lex = {:.3e} m", lex);
    println!(
        "SP2: torque thresholds (T): rem={:.1e}  hc={:.1e}",
        torque_rem, torque_hc
    );
    println!(
        "SP2: dm_stop = {}",
        dm_stop
            .map(|x| format!("{:.1e}", x))
            .unwrap_or_else(|| "OFF".to_string())
    );
    println!(
        "SP2: bracket step = {:.0} A/m ({:.0}× base), target step = {:.0} A/m",
        bracket_mult * 0.00005 * ms,
        bracket_mult,
        target_mult * 0.00005 * ms
    );
    println!(
        "SP2: minimise iters: rem={} hc={} coarse={}",
        min_iters_rem, min_iters_hc, min_iters_coarse
    );
    println!(
        "SP2: relax steps: rem={} hc={} chunk={}",
        relax_steps_rem, relax_steps_hc, relax_chunk
    );
    if ovf_enabled {
        if ovf_every_s > 0.0 {
            println!("SP2: ovf = ON (every {:.3} ns)", ovf_every_s * 1e9);
        } else {
            println!("SP2: ovf = ON (interval frames disabled; finals only)");
        }
    } else {
        println!("SP2: ovf = OFF");
    }
    println!("SP2: timing = {}", if timing { "ON" } else { "OFF" });
    println!("SP2: verbose = {}", if verbose { "ON" } else { "OFF" });

    // Output setup
    let out_dir = Path::new("runs").join("st_problems").join("sp2");
    create_dir_all(&out_dir)?;
    let table_path = out_dir.join("table.csv");

    // OVF root
    let ovf_root = out_dir.join("ovf");
    if ovf_enabled {
        create_dir_all(&ovf_root)?;
    }

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

    let need_header = !file_exists || std::fs::metadata(&table_path)?.len() == 0;
    if need_header {
        writeln!(
            w,
            "d_lex,mx_rem,my_rem,hc_over_ms,mx_hc,my_hc,mz_hc,msum_hc"
        )?;
        w.flush()?;
        println!("SP2: wrote header -> {}", table_path.display());
    }

    // Common material
    let material = Material {
        ms,
        a_ex,
        k_u,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    // Grid policy
    let grid_policy = Sp2GridPolicy::default();

    for d_lex in (d_min..=d_max).rev() {
        if done.contains(&(d_lex as i32)) {
            println!("\nSP2 d/lex={}: already done; skipping.", d_lex);
            continue;
        }

        let grid = build_sp2_grid(d_lex, lex, &grid_policy);
        println!(
            "\nSP2 d/lex={}: grid nx={} ny={}  dx/lex={:.3} dy/lex={:.3} dz/lex={:.3}",
            d_lex,
            grid.nx,
            grid.ny,
            grid.dx / lex,
            grid.dy / lex,
            grid.dz / lex
        );

        let ovf_d_dir = if ovf_enabled {
            // Base directory for this d/lex: ovf/dXX/
            // Time-series go into ovf/dXX/rem/ and ovf/dXX/hc/
            // Final states go into ovf/dXX/
            let p = ovf_root.join(format!("d{:02}", d_lex));
            create_dir_all(p.join("rem"))?;
            create_dir_all(p.join("hc"))?;
            Some(p)
        } else {
            None
        };

        let mut params = LLGParams {
            gamma: GAMMA_E_RAD_PER_S_T,
            alpha: 0.5,
            dt: 1e-13,
            b_ext: [0.0, 0.0, 0.0],
        };

        let mut rk23 = RK23Scratch::new(grid);

        // -------------------------
        // Remanence (keep as-is)
        // -------------------------
        let mut m = VectorField2D::new(grid);
        let m0 = normalize([1.0, 0.3001, 0.0]);
        m.set_uniform(m0[0], m0[1], m0[2]);

        params.b_ext = [0.0, 0.0, 0.0];

        let _rem_diag = inner_equilibrate(
            &grid,
            &mut m,
            &mut params,
            &material,
            &mut rk23,
            0.0,
            InnerKind::Remanence,
            torque_rem,
            dm_stop,
            min_iters_rem,
            relax_steps_rem,
            relax_chunk,
            verbose,
            timing,
            "rem",
            ovf_d_dir.as_deref(),
            ovf_every_s,
        );

        // Save final remanence OVF
        if let Some(d_dir) = ovf_d_dir.as_deref() {
            let path = d_dir.join(format!("m_d{:02}_rem.ovf", d_lex));
            let mut meta = OvfMeta::magnetization();
            meta.push_desc_line(format!("SP2 d/lex={} rem_final", d_lex));
            if let Err(e) = write_ovf2_rectangular_binary4(&path, &grid, &m, &meta) {
                eprintln!("  [ovf] WARNING: failed to write rem_final: {}", e);
            }
        }

        let rem_avg = avg_m(&m);
        let mx_rem = rem_avg[0];
        let my_rem = rem_avg[1];
        let rem_msum = msum(rem_avg);

        println!(
            "SP2 d/lex={}: rem <m>=({:.6},{:.6},{:.6})  m_sum={:.6}",
            d_lex, rem_avg[0], rem_avg[1], rem_avg[2], rem_msum
        );

        // -------------------------
        // Coercivity
        // -------------------------
        println!("SP2 d/lex={}: coercivity search start", d_lex);

        let m_rem = VectorField2D {
            grid,
            data: m.data.clone(),
        };

        // Capture the final (negative-branch) state at the returned Hc field.
        let mut m_hc_state = VectorField2D::new(grid);
        let mut m_hc_pos_state = VectorField2D::new(grid);
        let mut m_hc_best_state = VectorField2D::new(grid);

        let t_hc0 = Instant::now();
        let hc_over_ms = find_hc_over_ms(
            &grid,
            &m_rem,
            &mut params,
            &material,
            &mut rk23,
            torque_hc,
            dm_stop,
            min_iters_hc,
            min_iters_coarse,
            relax_steps_hc,
            relax_chunk,
            bracket_mult,
            target_mult,
            verbose,
            timing,
            ovf_d_dir.as_deref(),
            ovf_every_s,
            Some(&mut m_hc_state),
            Some(&mut m_hc_pos_state),
            Some(&mut m_hc_best_state),
        );
        let t_hc = t_hc0.elapsed().as_secs_f64();

        println!(
            "SP2 d/lex={}: coercivity done in {:.1}s  hc/Ms={:.6}",
            d_lex, t_hc, hc_over_ms
        );

        // Save final coercivity OVF
        if let Some(d_dir) = ovf_d_dir.as_deref() {
            let path = d_dir.join(format!("m_d{:02}_hc.ovf", d_lex));
            let mut meta = OvfMeta::magnetization();
            meta.push_desc_line(format!(
                "SP2 d/lex={} hc_final hc/Ms={:.6}",
                d_lex, hc_over_ms
            ));
            if let Err(e) = write_ovf2_rectangular_binary4(&path, &grid, &m_hc_state, &meta) {
                eprintln!("  [ovf] WARNING: failed to write hc_final: {}", e);
            }
        }

        // Save last strict-positive state at the final bracket (diagnostic)
        if let Some(d_dir) = ovf_d_dir.as_deref() {
            let path_pos = d_dir.join(format!("m_d{:02}_hc_pos.ovf", d_lex));
            let mut meta_pos = OvfMeta::magnetization();
            meta_pos.push_desc_line(format!(
                "SP2 d/lex={} hc_pos_strict hc/Ms={:.6}",
                d_lex, hc_over_ms
            ));
            if let Err(e) =
                write_ovf2_rectangular_binary4(&path_pos, &grid, &m_hc_pos_state, &meta_pos)
            {
                eprintln!("  [ovf] WARNING: failed to write hc_pos_strict: {}", e);
            }
        }

        // Save closest-to-zero strict state encountered (diagnostic)
        if let Some(d_dir) = ovf_d_dir.as_deref() {
            let path_best = d_dir.join(format!("m_d{:02}_hc_best.ovf", d_lex));
            let mut meta_best = OvfMeta::magnetization();
            meta_best.push_desc_line(format!("SP2 d/lex={} hc_best_strict", d_lex));
            if let Err(e) =
                write_ovf2_rectangular_binary4(&path_best, &grid, &m_hc_best_state, &meta_best)
            {
                eprintln!("  [ovf] WARNING: failed to write hc_best_strict: {}", e);
            }
        }

        params.b_ext = [0.0, 0.0, 0.0];

        let hc_avg = avg_m(&m_hc_state);
        let mx_hc = hc_avg[0];
        let my_hc = hc_avg[1];
        let mz_hc = hc_avg[2];
        let msum_hc = msum(hc_avg);

        writeln!(
            w,
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            d_lex, mx_rem, my_rem, hc_over_ms, mx_hc, my_hc, mz_hc, msum_hc
        )?;
        w.flush()?;
        println!("SP2 d/lex={}: row written", d_lex);
    }

    println!("\nSP2 complete. Output: {}", table_path.display());
    Ok(())
}
