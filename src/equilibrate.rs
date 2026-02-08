// ===============================
// src/equilibrate.rs
// ===============================
//
// Reusable equilibrium orchestration.
// - equilibrate(): “static” equilibrium (Minimize first, Relax only if needed).
// - hysteresis_step(): “branch-following” equilibrium for field scans (SP2 coercivity).
//
// Motivation:
// Static equilibrium problems can skip Relax most of the time.
// Hysteresis / coercivity is path dependent: you must settle on the continuation branch
// at *every* field step, but keep it bounded and fast.

use crate::effective_field::FieldMask;
use crate::grid::Grid2D;
use crate::llg::RK23Scratch;
use crate::minimize::{minimize_damping_only, MinimizeReport, MinimizeSettings};
use crate::params::{LLGParams, Material};
use crate::relax::{relax_with_report, RelaxReport, RelaxSettings, RelaxStopReason};
use crate::vector_field::VectorField2D;

// -------------------------
// Static equilibrium policy
// -------------------------

#[derive(Debug, Clone)]
pub struct EquilibratePolicy {
    /// First attempt: minimizer settings.
    pub minimize: MinimizeSettings,

    /// Escape relax settings used only when minimizer stalls / fails.
    pub relax_escape: RelaxSettings,

    /// If true, do a final “certify” relax after minimization succeeds.
    pub certify: bool,

    /// Settings for the certify relax (if enabled).
    pub relax_certify: RelaxSettings,

    /// If minimizer returns with final_torque above this, treat as not good enough and escape.
    pub torque_escape_gate: f64,

    /// If true, reset dt before each relax call (deterministic). Good for one-off relax.
    pub reset_dt_before_relax: bool,
}

impl Default for EquilibratePolicy {
    fn default() -> Self {
        let mut min = MinimizeSettings::default();
        min.torque_threshold = 4e-4;

        let mut esc = RelaxSettings::default();
        esc.phase1_enabled = false;
        esc.phase2_enabled = true;
        esc.torque_metric = crate::relax::TorqueMetric::Mean;
        esc.torque_threshold = None;
        esc.torque_plateau_checks = 5;
        esc.torque_plateau_rel = 1e-3;
        esc.torque_plateau_min_checks = 5;
        esc.tighten_floor = 1e-6;
        esc.max_accepted_steps = 5000;

        let mut cert = RelaxSettings::default();
        cert.phase1_enabled = false;
        cert.phase2_enabled = true;
        cert.torque_metric = crate::relax::TorqueMetric::Mean;
        cert.torque_threshold = None;
        cert.torque_plateau_checks = 8;
        cert.tighten_floor = 1e-6;
        cert.max_accepted_steps = 20000;

        Self {
            minimize: min,
            relax_escape: esc,
            certify: false,
            relax_certify: cert,
            torque_escape_gate: 5e-3,
            reset_dt_before_relax: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EquilibrateReport {
    pub minimize_first: MinimizeReport,
    pub relax_escape: Option<RelaxReport>,
    pub minimize_second: Option<MinimizeReport>,
    pub relax_certify: Option<RelaxReport>,
    pub converged: bool,
}

/// Static equilibrium.
/// Strategy:
///  1) Minimize
///  2) If stalled or torque too large -> short Relax escape (plateau-only) -> Minimize again
///  3) Optional certify Relax
pub fn equilibrate(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    mask: FieldMask,
    policy: &EquilibratePolicy,
) -> EquilibrateReport {
    let rep1 = minimize_damping_only(grid, m, params, material, mask, &policy.minimize);

    let needs_escape = rep1.stalled || rep1.final_torque > policy.torque_escape_gate;

    let mut rep_escape: Option<RelaxReport> = None;
    let mut rep2: Option<MinimizeReport> = None;

    if needs_escape {
        let mut esc = policy.relax_escape.clone();
        if policy.reset_dt_before_relax {
            params.dt = 1e-13;
        }
        let r = relax_with_report(grid, m, params, material, rk23, mask, &mut esc);
        rep_escape = Some(r);

        let rep_again = minimize_damping_only(grid, m, params, material, mask, &policy.minimize);
        rep2 = Some(rep_again);
    }

    let best = rep2.as_ref().unwrap_or(&rep1);
    let mut converged = best.converged && !best.stalled;

    let mut rep_certify: Option<RelaxReport> = None;
    if policy.certify {
        let mut cert = policy.relax_certify.clone();
        if policy.reset_dt_before_relax {
            params.dt = 1e-13;
        }
        let r = relax_with_report(grid, m, params, material, rk23, mask, &mut cert);
        converged = converged && (r.stop_reason != RelaxStopReason::MaxAcceptedSteps);
        rep_certify = Some(r);
    }

    EquilibrateReport {
        minimize_first: rep1,
        relax_escape: rep_escape,
        minimize_second: rep2,
        relax_certify: rep_certify,
        converged,
    }
}

// -------------------------
// Hysteresis / coercivity step
// -------------------------

#[derive(Debug, Clone)]
pub struct HysteresisPolicy {
    /// Minimize settings used for the first step on a new branch (seed).
    pub minimize_first: MinimizeSettings,
    /// Minimize settings used for subsequent continuation steps.
    pub minimize_step: MinimizeSettings,

    /// Primary plateau relax for branch settling.
    pub relax: RelaxSettings,

    /// Optional escalation relax if the primary relax hits MaxAcceptedSteps.
    pub relax_escalate: Option<RelaxSettings>,

    /// If true, keep params.dt warm between steps (recommended for scans).
    pub warm_start_dt: bool,
}

impl Default for HysteresisPolicy {
    fn default() -> Self {
        let mut min_first = MinimizeSettings::default();
        let mut min_step = MinimizeSettings::default();
        min_first.max_iters = 800;
        min_step.max_iters = 200;
        min_first.torque_threshold = 6e-4;
        min_step.torque_threshold = 6e-4;

        let mut r = RelaxSettings::default();
        r.phase1_enabled = false;
        r.phase2_enabled = true;
        r.torque_metric = crate::relax::TorqueMetric::Mean;
        r.torque_threshold = None;
        r.torque_plateau_checks = 4;
        r.torque_plateau_rel = 2e-3;
        r.torque_plateau_min_checks = 3;

        // single-stage by default
        r.max_err = 2e-5;
        r.tighten_floor = 2e-5;

        r.max_accepted_steps = 2500;
        r.torque_check_stride = 200;

        let mut esc = r.clone();
        esc.max_err = 1e-5;
        esc.tighten_floor = 1e-5; // IMPORTANT: single-stage (prevents 1e-6 tightening cost)
        esc.max_accepted_steps = 6000;

        Self {
            minimize_first: min_first,
            minimize_step: min_step,
            relax: r,
            relax_escalate: Some(esc),
            warm_start_dt: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HysteresisStepReport {
    pub minimize: MinimizeReport,
    pub relax: RelaxReport,
    pub relax_escalate: Option<RelaxReport>,
}

/// Hysteresis continuation step under the **current** params/material/mask.
/// Always does: short Minimize + bounded plateau Relax.
/// Optionally escalates if Relax hits MaxAcceptedSteps.
///
/// `is_first` controls whether we use `policy.minimize_first` or `policy.minimize_step`.
pub fn hysteresis_step(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    mask: FieldMask,
    policy: &HysteresisPolicy,
    is_first: bool,
) -> HysteresisStepReport {
    // For scans: keep dt warm (don’t reset every step).
    // For first step, reset once for determinism.
    if !policy.warm_start_dt || is_first {
        params.dt = 1e-13;
    }

    let min_settings = if is_first {
        &policy.minimize_first
    } else {
        &policy.minimize_step
    };

    let min_rep = minimize_damping_only(grid, m, params, material, mask, min_settings);

    let mut relax_settings = policy.relax.clone();
    let relax_rep = relax_with_report(grid, m, params, material, rk23, mask, &mut relax_settings);

    let mut esc_rep: Option<RelaxReport> = None;
    if relax_rep.stop_reason == RelaxStopReason::MaxAcceptedSteps {
        if let Some(escalate) = &policy.relax_escalate {
            let mut esc = escalate.clone();
            let r2 = relax_with_report(grid, m, params, material, rk23, mask, &mut esc);
            esc_rep = Some(r2);
        }
    }

    HysteresisStepReport {
        minimize: min_rep,
        relax: relax_rep,
        relax_escalate: esc_rep,
    }
}