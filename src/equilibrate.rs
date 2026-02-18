// ===============================
// src/equilibrate.rs
// ===============================
//
// MuMax-like *state-based* equilibrium orchestration.
//
// Motivation:
// - Stop encoding solver behaviour in driver scripts (e.g. SP2).
// - Provide a small set of reusable policies for:
//     (1) One-off equilibration (remanence)
//     (2) Branch-following equilibration (hysteresis scans)
//
// Key idea:
// - A Relax pass can terminate at TightenFloorReached while the configuration is still
//   not genuinely settled (e.g. due to plateau criteria or limited stage budget).
// - We enforce a **gate** on the final torque metric. If the gate fails, we run a small
//   number of extra passes *at the floor* (and optionally a tiny minimizer preconditioner).
//
// This gives you a state-based “hard vs easy regime” switch without hardcoding d/lex.

use crate::effective_field::{FieldMask, build_h_eff_masked};
use crate::grid::Grid2D;
use crate::llg::RK23Scratch;
use crate::minimize::{MinimizeReport, MinimizeSettings, minimize_damping_only};
use crate::params::{LLGParams, Material};
use crate::relax::{RelaxReport, RelaxSettings, RelaxStopReason, TorqueMetric, relax_with_report};
use crate::vec3::cross;
use crate::vector_field::VectorField2D;

// -------------------------
// Generic relax-with-gate
// -------------------------

#[derive(Debug, Clone)]
pub struct RelaxGate {
    /// Gate on the *same torque metric* used by RelaxSettings.
    /// If None, no gating is applied.
    pub torque_gate: Option<f64>,

    /// Optional additional gate on the **max torque** (Tesla), to catch hotspot-dominated states
    /// where the mean torque looks fine but a small region is still highly non-equilibrated.
    ///
    /// This is useful for large d/lex (stiffer, more nonuniform states) and is still state-based
    /// (no hardcoded d/lex thresholds).
    pub torque_gate_max: Option<f64>,

    /// Extra passes at tighten_floor if the gate fails.
    pub extra_passes: usize,

    /// If true, run a tiny minimize preconditioner between failed gate passes.
    pub fallback_to_minimize: bool,

    /// Settings for the fallback minimizer (used only if enabled).
    pub minimize: MinimizeSettings,

    /// When doing extra passes, multiply max_accepted_steps by this factor.
    pub extra_pass_max_steps_mult: f64,
}

impl Default for RelaxGate {
    fn default() -> Self {
        Self {
            // Conservative default: no gate unless the caller sets one.
            torque_gate: None,
            torque_gate_max: None,
            extra_passes: 2,
            fallback_to_minimize: false,
            minimize: {
                let mut s = MinimizeSettings::default();
                s.max_iters = 200;
                s.torque_threshold = 8e-4;
                s
            },
            extra_pass_max_steps_mult: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RelaxWithGateReport {
    /// Reports from each relax pass (first + any extras).
    pub relax_passes: Vec<RelaxReport>,

    /// Optional minimizer reports (only if fallback_to_minimize enabled).
    pub minimize_passes: Vec<MinimizeReport>,

    /// Whether the final configuration passed the gate (or no gate was set).
    pub gate_passed: bool,
}

fn torque_max_from_field(m: &VectorField2D, b_eff: &VectorField2D) -> f64 {
    debug_assert_eq!(m.data.len(), b_eff.data.len());
    let mut tmax = 0.0_f64;
    for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
        let t = cross(*mi, *bi);
        let mag = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        if mag > tmax {
            tmax = mag;
        }
    }
    tmax
}

fn torque_max_now(
    grid: &Grid2D,
    m: &VectorField2D,
    params: &LLGParams,
    material: &Material,
    rk23: &RK23Scratch,
    mask: FieldMask,
) -> f64 {
    // Prefer the last accepted RK23 effective field if available (avoids an extra demag build).
    if let Some(b_eff) = rk23.last_b_eff() {
        return torque_max_from_field(m, b_eff);
    }

    // Fallback: rebuild effective field (expensive).
    let mut b_eff = VectorField2D::new(*grid);
    build_h_eff_masked(grid, m, &mut b_eff, params, material, mask);
    torque_max_from_field(m, &b_eff)
}

fn ensure_final_torque_max(
    grid: &Grid2D,
    m: &VectorField2D,
    params: &LLGParams,
    material: &Material,
    rk23: &RK23Scratch,
    mask: FieldMask,
    _gate: &RelaxGate,
    rep: &mut RelaxReport,
) {
    // Always populate max torque if it's missing.
    // This keeps reporting consistent (useful for diagnostics) and supports optional
    // hotspot gating when callers enable `torque_gate_max`.
    if rep.final_torque_max.is_none() {
        let tmax = torque_max_now(grid, m, params, material, rk23, mask);
        rep.final_torque_max = Some(tmax);
    }
}

fn gate_ok(rep: &RelaxReport, gate: &RelaxGate) -> bool {
    let ok_metric = match gate.torque_gate {
        None => true,
        Some(tau) => rep.final_torque.unwrap_or(f64::INFINITY) <= tau,
    };

    // `final_torque_max` is populated by relax.rs (at least at the end of relax) and is
    // essential for hotspot safety checks.
    let ok_max = match gate.torque_gate_max {
        None => true,
        Some(tau) => rep.final_torque_max.unwrap_or(f64::INFINITY) <= tau,
    };

    ok_metric && ok_max
}

/// Run Relax at least once, then (optionally) repeat at tighten_floor until a torque gate passes.
pub fn relax_until_gate(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    mask: FieldMask,
    base_settings: &RelaxSettings,
    gate: &RelaxGate,
) -> RelaxWithGateReport {
    let mut relax_passes: Vec<RelaxReport> = Vec::new();
    let mut minimize_passes: Vec<MinimizeReport> = Vec::new();

    // First pass: use caller settings verbatim.
    let mut s0 = base_settings.clone();

    // Ensure relax.rs internal gate is OFF (equilibrate owns gating).
    s0.final_torque_gate = None;
    s0.final_torque_gate_max = None;
    s0.gate_max_extra_accepted_steps = 0;
    s0.gate_plateau_fails = 0;

    let mut r0 = relax_with_report(grid, m, params, material, rk23, mask, &mut s0);
    ensure_final_torque_max(grid, m, params, material, rk23, mask, gate, &mut r0);
    relax_passes.push(r0);

    if gate_ok(relax_passes.last().unwrap(), gate) {
        return RelaxWithGateReport {
            relax_passes,
            minimize_passes,
            gate_passed: true,
        };
    }

    // Extra passes: run at the floor only (single stage), optionally with a tiny minimizer.
    for _ in 0..gate.extra_passes {
        if gate.fallback_to_minimize {
            let rep_m = minimize_damping_only(grid, m, params, material, mask, &gate.minimize);
            minimize_passes.push(rep_m);
        }

        let mut s = base_settings.clone();

        // Ensure relax.rs internal gate is OFF (equilibrate owns gating).
        s.final_torque_gate = None;
        s.final_torque_gate_max = None;
        s.gate_max_extra_accepted_steps = 0;
        s.gate_plateau_fails = 0;

        // Force single-stage at the floor.
        s.phase1_enabled = false;
        s.phase2_enabled = true;
        s.max_err = s.tighten_floor;

        // Extra-pass behaviour: keep it MuMax-like and deterministic.
        // MuMax stops a stage as soon as torque is steady or increasing, then (normally) tightens.
        // Here we are already at the floor, so we use strict plateau semantics and avoid threshold-mode.
        s.torque_threshold = None;
        s.torque_plateau_checks = 1;
        s.torque_plateau_min_checks = 1;
        s.torque_plateau_rel = 0.0;
        s.torque_plateau_abs = 0.0;

        // Give extra stage budget.
        let mult = gate.extra_pass_max_steps_mult.max(1.0);
        s.max_accepted_steps = ((s.max_accepted_steps as f64) * mult) as usize;

        let mut r = relax_with_report(grid, m, params, material, rk23, mask, &mut s);
        ensure_final_torque_max(grid, m, params, material, rk23, mask, gate, &mut r);
        relax_passes.push(r);

        if gate_ok(relax_passes.last().unwrap(), gate) {
            return RelaxWithGateReport {
                relax_passes,
                minimize_passes,
                gate_passed: true,
            };
        }

        // If relax itself is hitting MaxAcceptedSteps repeatedly, no point looping forever.
        if relax_passes
            .last()
            .map(|rr| rr.stop_reason == RelaxStopReason::MaxAcceptedSteps)
            .unwrap_or(false)
        {
            break;
        }
    }

    let gate_passed = gate_ok(relax_passes.last().unwrap(), gate);

    RelaxWithGateReport {
        relax_passes,
        minimize_passes,
        gate_passed,
    }
}

// -------------------------
// Convenience policies
// -------------------------

#[derive(Debug, Clone)]
pub struct RemanencePolicy {
    pub relax: RelaxSettings,
    pub gate: RelaxGate,

    /// If true, reset dt before equilibration (recommended for one-off relax).
    pub reset_dt: bool,
}

impl Default for RemanencePolicy {
    fn default() -> Self {
        let mut r = RelaxSettings::default();
        // MuMax-like remanence: energy phase + plateau on mean torque.
        r.phase1_enabled = true;
        r.phase2_enabled = true;
        r.torque_metric = TorqueMetric::Mean;
        r.torque_threshold = None;
        r.torque_check_stride = 1;
        r.torque_plateau_checks = 8;
        r.torque_plateau_min_checks = 8;
        r.torque_plateau_rel = 0.0;
        r.torque_plateau_abs = 1e-7;

        // CPU-feasible defaults; callers can override tighten_floor.
        r.max_err = 1e-5;

        let mut g = RelaxGate::default();
        // Gate on mean torque: if still high, do a couple extra passes at the floor.
        g.torque_gate = Some(5e-4);
        g.torque_gate_max = None;
        g.extra_passes = 2;
        g.fallback_to_minimize = false;

        Self {
            relax: r,
            gate: g,
            reset_dt: true,
        }
    }
}

pub fn equilibrate_remanence(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    rk23: &mut RK23Scratch,
    mask: FieldMask,
    policy: &RemanencePolicy,
) -> RelaxWithGateReport {
    if policy.reset_dt {
        params.dt = 1e-13;
    }
    relax_until_gate(
        grid,
        m,
        params,
        material,
        rk23,
        mask,
        &policy.relax,
        &policy.gate,
    )
}

#[derive(Debug, Clone)]
pub struct HysteresisPolicy {
    /// Optional short minimizer to help continuation steps.
    pub premin: Option<MinimizeSettings>,

    /// Primary relax settings used every field step.
    pub relax: RelaxSettings,

    /// Gate policy.
    pub gate: RelaxGate,

    /// If true, keep dt warm between steps; if false, reset dt each step.
    pub warm_start_dt: bool,
}

impl Default for HysteresisPolicy {
    fn default() -> Self {
        let mut r = RelaxSettings::default();
        r.phase1_enabled = false;
        r.phase2_enabled = true;
        // In MuMax, Relax() plateau is typically based on average torque.
        // Max torque can over-stabilise rare hotspots and delay switching.
        r.torque_metric = TorqueMetric::Mean;
        r.torque_threshold = None;
        r.torque_check_stride = 1;
        r.torque_plateau_checks = 3;
        r.torque_plateau_min_checks = 5;
        r.torque_plateau_rel = 0.0;
        r.torque_plateau_abs = 0.0;

        r.max_err = 1e-5;

        let mut g = RelaxGate::default();
        g.torque_gate = Some(8e-4);

        // IMPORTANT:
        // Do NOT enable an absolute max-torque gate by default.
        // In large / stiff systems (e.g. SP2 at large d/lex), a small local hotspot can keep
        // `max |m×B|` relatively large even while the configuration is on the correct continuation
        // branch and the mean torque is already well-settled.
        // MuMax's default Relax() behaviour (RelaxTorqueThreshold < 0) does not use a max-torque
        // stopping rule either; it relies on an average-torque criterion plus tightening.
        //
        // If you want a max-torque gate for specific experiments, set it explicitly from the caller.
        g.torque_gate_max = None;

        g.extra_passes = 2;
        g.fallback_to_minimize = false;

        Self {
            premin: None,
            relax: r,
            gate: g,
            warm_start_dt: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HysteresisStepReport {
    pub premin: Option<MinimizeReport>,
    pub relax: RelaxWithGateReport,
}

/// Branch-following equilibrium step under the **current** params/material/mask.
///
/// - Optionally runs a short minimizer preconditioner.
/// - Runs Relax with the gate.
/// - Keeps dt warm by default (MuMax-like continuation behaviour).
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
    if !policy.warm_start_dt || is_first {
        params.dt = 1e-13;
    }

    let premin_rep = if let Some(pm) = &policy.premin {
        Some(minimize_damping_only(grid, m, params, material, mask, pm))
    } else {
        None
    };

    let relax_rep = relax_until_gate(
        grid,
        m,
        params,
        material,
        rk23,
        mask,
        &policy.relax,
        &policy.gate,
    );

    HysteresisStepReport {
        premin: premin_rep,
        relax: relax_rep,
    }
}
