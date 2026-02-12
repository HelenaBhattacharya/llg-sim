// ===============================
// src/relax.rs (FULL FILE)
// ===============================
//
// MuMax-like relaxation controller:
//
//  - Precession suppressed (damping-only LLG RHS)
//  - Adaptive stepping (Bogacki–Shampine RK23 relax stepper)
//  - Phase 1: energy descent until noise floor (MuMax uses N=3)
//  - Phase 2: torque descent with tolerance tightening
//
// Key design choice for Rust:
// - We keep `settings` as an input config, but we do NOT keep a long-lived mutable borrow to it.
// - `max_err` is treated like MuMax's global MaxErr: we evolve it locally and write it back at end.
// - This avoids borrow-checker issues and matches MuMax’s “tighten MaxErr during relax” logic.
//
// Notes:
// - Plateau mode is MuMax-style: stop a stage when torque is steady or increasing
//   (set plateau_checks=1, min_checks=1, rel=0, abs=0).
// - Threshold mode mimics MuMax’s RelaxTorqueThreshold>0 mode.
// - `final_torque_max` is always computed (cheap if last_b_eff is valid).

use crate::effective_field::{build_h_eff_masked, FieldMask};
use crate::energy::compute_total_energy;
use crate::grid::Grid2D;
use crate::llg::{step_llg_rk23_recompute_field_masked_relax_adaptive, RK23Scratch};
use crate::params::{LLGParams, Material};
use crate::vec3::cross;
use crate::vector_field::VectorField2D;

#[derive(Debug, Clone, Copy)]
pub enum TorqueMetric {
    Max,
    Mean,
    Rms,
    MeanSq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaxStopReason {
    MaxAcceptedSteps,
    Phase2Disabled,
    TightenFloorReached,
    TorqueGateNotSatisfied,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaxStageStop {
    BelowThreshold,
    Plateau,
    MaxAcceptedSteps,
}

#[derive(Debug, Clone)]
pub struct RelaxReport {
    pub accepted_steps: usize,
    pub rejected_steps: usize,

    pub torque_checks: usize,
    pub torque_field_rebuilds: usize,

    pub stop_reason: RelaxStopReason,
    pub last_stage_stop: Option<RelaxStageStop>,

    pub final_torque: Option<f64>,
    pub final_torque_max: Option<f64>,

    pub final_max_err: f64,
    pub final_dt: f64,
}

#[derive(Debug, Clone)]
pub struct RelaxSettings {
    pub max_err: f64,
    pub headroom: f64,
    pub dt_min: f64,
    pub dt_max: f64,

    pub phase1_enabled: bool,
    pub phase2_enabled: bool,

    pub energy_stride: usize,
    pub rel_energy_tol: f64,

    pub torque_metric: TorqueMetric,
    pub torque_threshold: Option<f64>,

    pub torque_check_stride: usize,

    pub torque_plateau_rel: f64,
    pub torque_plateau_abs: f64,
    pub torque_plateau_checks: usize,
    pub torque_plateau_min_checks: usize,

    pub tighten_factor: f64,
    pub tighten_floor: f64,

    pub max_accepted_steps: usize,

    // Optional internal gate (kept for compatibility; equilibrate.rs disables these)
    pub final_torque_gate: Option<f64>,
    pub final_torque_gate_max: Option<f64>,
    pub gate_max_extra_accepted_steps: usize,
    pub gate_plateau_fails: usize,
}

impl Default for RelaxSettings {
    fn default() -> Self {
        Self {
            max_err: 1e-5,
            headroom: 0.8,
            dt_min: 1e-18,
            dt_max: 1e-11,

            phase1_enabled: true,
            phase2_enabled: true,

            energy_stride: 3,
            rel_energy_tol: 1e-12,

            torque_metric: TorqueMetric::Max,
            torque_threshold: Some(1e-4),
            torque_check_stride: 1,

            torque_plateau_rel: 1e-3,
            torque_plateau_abs: 0.0,
            torque_plateau_checks: 0,
            torque_plateau_min_checks: 5,

            tighten_factor: std::f64::consts::FRAC_1_SQRT_2,
            tighten_floor: 1e-9,

            max_accepted_steps: 2_000_000,

            final_torque_gate: None,
            final_torque_gate_max: None,
            gate_max_extra_accepted_steps: 0,
            gate_plateau_fails: 0,
        }
    }
}

/// Compute a torque metric from an already-built effective field.
fn torque_metric_from_field(m: &VectorField2D, b_eff: &VectorField2D, metric: TorqueMetric) -> f64 {
    debug_assert_eq!(m.data.len(), b_eff.data.len());
    let n = m.data.len() as f64;

    match metric {
        TorqueMetric::Max => {
            let mut maxv = 0.0;
            for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
                let t = cross(*mi, *bi);
                let mag = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                if mag > maxv {
                    maxv = mag;
                }
            }
            maxv
        }
        TorqueMetric::Mean => {
            let mut sum = 0.0;
            for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
                let t = cross(*mi, *bi);
                let mag = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                sum += mag;
            }
            sum / n.max(1.0)
        }
        TorqueMetric::Rms => {
            let mut sum2 = 0.0;
            for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
                let t = cross(*mi, *bi);
                let mag2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
                sum2 += mag2;
            }
            (sum2 / n.max(1.0)).sqrt()
        }
        TorqueMetric::MeanSq => {
            // MuMax “avgTorque” in relax.go is cuda.Dot(k1,k1) (no sqrt)
            let mut sum2 = 0.0;
            for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
                let t = cross(*mi, *bi);
                let mag2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
                sum2 += mag2;
            }
            sum2 / n.max(1.0)
        }
    }
}

/// Compute torque metric, rebuilding B_eff if needed (expensive).
fn torque_metric_rebuild(
    grid: &Grid2D,
    m: &VectorField2D,
    params: &LLGParams,
    material: &Material,
    mask: FieldMask,
    metric: TorqueMetric,
    b_eff_scratch: &mut VectorField2D,
) -> f64 {
    build_h_eff_masked(grid, m, b_eff_scratch, params, material, mask);
    torque_metric_from_field(m, b_eff_scratch, metric)
}

/// Torque at current state; reuse last accepted RK23 field if possible.
fn torque_now(
    grid: &Grid2D,
    m: &VectorField2D,
    params: &LLGParams,
    material: &Material,
    scratch: &RK23Scratch,
    mask: FieldMask,
    metric: TorqueMetric,
    b_eff_scratch: &mut VectorField2D,
    torque_checks: &mut usize,
    torque_field_rebuilds: &mut usize,
) -> f64 {
    *torque_checks += 1;

    if let Some(b_eff) = scratch.last_b_eff() {
        torque_metric_from_field(m, b_eff, metric)
    } else {
        *torque_field_rebuilds += 1;
        torque_metric_rebuild(grid, m, params, material, mask, metric, b_eff_scratch)
    }
}

#[inline]
fn rk23_step(
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK23Scratch,
    mask: FieldMask,
    max_err: f64,
    headroom: f64,
    dt_min: f64,
    dt_max: f64,
) -> bool {
    let (_eps, ok, _dt_used) = step_llg_rk23_recompute_field_masked_relax_adaptive(
        m, params, material, scratch, mask, max_err, headroom, dt_min, dt_max,
    );
    ok
}

/// Advance by `n_accept` accepted steps. Rejects don’t count.
fn advance_accepted(
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK23Scratch,
    mask: FieldMask,
    max_err: f64,
    headroom: f64,
    dt_min: f64,
    dt_max: f64,
    max_accepted_steps: usize,
    n_accept: usize,
    accepted: &mut usize,
    rejected: &mut usize,
) -> bool {
    let target = accepted.saturating_add(n_accept.max(1));
    while *accepted < target {
        if *accepted >= max_accepted_steps {
            return false;
        }
        if rk23_step(m, params, material, scratch, mask, max_err, headroom, dt_min, dt_max) {
            *accepted += 1;
        } else {
            *rejected += 1;
        }
    }
    true
}

/// Check internal torque gates (optional).
fn gates_ok(
    grid: &Grid2D,
    m: &VectorField2D,
    params: &LLGParams,
    material: &Material,
    scratch: &RK23Scratch,
    mask: FieldMask,
    metric: TorqueMetric,
    gate_metric: Option<f64>,
    gate_max: Option<f64>,
    b_eff_scratch: &mut VectorField2D,
    torque_checks: &mut usize,
    torque_field_rebuilds: &mut usize,
) -> (bool, f64, f64) {
    let t_metric = torque_now(
        grid, m, params, material, scratch, mask, metric, b_eff_scratch, torque_checks, torque_field_rebuilds,
    );

    let need_max = gate_max.is_some() || matches!(metric, TorqueMetric::Max);
    let t_max = if need_max {
        if matches!(metric, TorqueMetric::Max) {
            t_metric
        } else {
            torque_now(
                grid, m, params, material, scratch, mask, TorqueMetric::Max, b_eff_scratch, torque_checks, torque_field_rebuilds,
            )
        }
    } else {
        f64::NAN
    };

    let ok_metric = gate_metric.map_or(true, |g| t_metric <= g);
    let ok_max = gate_max.map_or(true, |g| t_max <= g);

    (ok_metric && ok_max, t_metric, t_max)
}

pub fn relax_with_report(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK23Scratch,
    mask: FieldMask,
    settings: &mut RelaxSettings,
) -> RelaxReport {
    // Copy settings we will use frequently (avoids borrow issues, matches “config in, run local”).
    let phase1_enabled = settings.phase1_enabled;
    let phase2_enabled = settings.phase2_enabled;

    let energy_stride = settings.energy_stride.max(1);
    let rel_energy_tol = settings.rel_energy_tol;

    let torque_metric = settings.torque_metric;
    let torque_threshold = settings.torque_threshold;

    let torque_check_stride = settings.torque_check_stride.max(1);
    let plateau_rel = settings.torque_plateau_rel;
    let plateau_abs = settings.torque_plateau_abs;
    let plateau_checks = settings.torque_plateau_checks;
    let plateau_min_checks = settings.torque_plateau_min_checks.max(1);

    let tighten_factor = settings.tighten_factor;
    let tighten_floor = settings.tighten_floor;

    let max_accepted_steps = settings.max_accepted_steps;

    // Internal gate (usually disabled; equilibrate.rs turns it off explicitly).
    let gate_metric = settings.final_torque_gate;
    let gate_max = settings.final_torque_gate_max;
    let gate_extra_steps = settings.gate_max_extra_accepted_steps;
    let gate_plateau_fails_limit = settings.gate_plateau_fails;

    // Local MaxErr (MuMax-style): mutate locally, write back at end.
    let mut max_err = settings.max_err;

    params.dt = params.dt.clamp(settings.dt_min, settings.dt_max);
    scratch.invalidate_last_b_eff();

    let mut accepted: usize = 0;
    let mut rejected: usize = 0;
    let mut torque_checks: usize = 0;
    let mut torque_field_rebuilds: usize = 0;

    let mut last_stage_stop: Option<RelaxStageStop> = None;
    let mut final_torque: Option<f64> = None;
    let mut final_torque_max: Option<f64> = None;

    let headroom = settings.headroom;
    let dt_min = settings.dt_min;
    let dt_max = settings.dt_max;

    let mut b_eff_scratch = VectorField2D::new(*grid);

    // -------------------------
    // Phase 1: energy descent (MuMax: relaxSteps(N), compare E)
    // -------------------------
    if phase1_enabled {
        let mut e0 = compute_total_energy(grid, m, material, params.b_ext);

        loop {
            if !advance_accepted(
                m, params, material, scratch, mask,
                max_err, headroom, dt_min, dt_max,
                max_accepted_steps,
                energy_stride,
                &mut accepted, &mut rejected,
            ) {
                settings.max_err = max_err;
                return RelaxReport {
                    accepted_steps: accepted,
                    rejected_steps: rejected,
                    torque_checks,
                    torque_field_rebuilds,
                    stop_reason: RelaxStopReason::MaxAcceptedSteps,
                    last_stage_stop,
                    final_torque,
                    final_torque_max,
                    final_max_err: max_err,
                    final_dt: params.dt,
                };
            }

            let e1 = compute_total_energy(grid, m, material, params.b_ext);

            // If you want *exact* MuMax semantics, set rel_energy_tol=0.
            let tol = rel_energy_tol * e0.abs().max(1e-30);

            if e1 < e0 - tol {
                e0 = e1;
                continue;
            }
            break;
        }
    }

    // -------------------------
    // Phase 2: torque descent
    // -------------------------
    if !phase2_enabled {
        settings.max_err = max_err;
        return RelaxReport {
            accepted_steps: accepted,
            rejected_steps: rejected,
            torque_checks,
            torque_field_rebuilds,
            stop_reason: RelaxStopReason::Phase2Disabled,
            last_stage_stop,
            final_torque: None,
            final_torque_max: None,
            final_max_err: max_err,
            final_dt: params.dt,
        };
    }

    let plateau_enabled = torque_threshold.is_none() && plateau_checks > 0;
    let need_fails = plateau_checks.max(1);

    loop {
        // -------- Stage at current max_err --------
        let mut t_prev = torque_now(
            grid, m, params, material, scratch, mask,
            torque_metric,
            &mut b_eff_scratch,
            &mut torque_checks,
            &mut torque_field_rebuilds,
        );

        // MuMax plateau semantics use float32 avgTorque and a strict T1 < T0 condition.
        // For MeanSq plateau mode with zero tolerances, emulate that float32 “noise floor”
        // so we do not run extremely long due to tiny f64 improvements.
        let mumax_plateau_f32 = plateau_enabled
            && matches!(torque_metric, TorqueMetric::MeanSq)
            && plateau_rel == 0.0
            && plateau_abs == 0.0
            && plateau_checks == 1
            && plateau_min_checks == 1;

        // MuMax compares a float32 sum-of-squares (cuda.Dot). Our MeanSq metric is an average,
        // so scale by Ncells for the float32 quantisation path.
        let n_f: f64 = m.data.len() as f64;
        let mut t_prev_f32: f32 = if mumax_plateau_f32 {
            (t_prev * n_f) as f32
        } else {
            t_prev as f32
        };

        let mut plateau_fails: usize = 0;
        let mut comparisons: usize = 0;

        loop {
            // Threshold mode (MuMax RelaxTorqueThreshold > 0 analogue)
            if let Some(tau) = torque_threshold {
                if t_prev <= tau {
                    last_stage_stop = Some(RelaxStageStop::BelowThreshold);
                    break;
                }
            }

            // Plateau mode (MuMax default: stop when steady or increasing)
            if plateau_enabled && plateau_fails >= need_fails {
                last_stage_stop = Some(RelaxStageStop::Plateau);
                break;
            }

            if accepted >= max_accepted_steps {
                settings.max_err = max_err;
                return RelaxReport {
                    accepted_steps: accepted,
                    rejected_steps: rejected,
                    torque_checks,
                    torque_field_rebuilds,
                    stop_reason: RelaxStopReason::MaxAcceptedSteps,
                    last_stage_stop: Some(RelaxStageStop::MaxAcceptedSteps),
                    final_torque: Some(t_prev),
                    final_torque_max,
                    final_max_err: max_err,
                    final_dt: params.dt,
                };
            }

            if !advance_accepted(
                m, params, material, scratch, mask,
                max_err, headroom, dt_min, dt_max,
                max_accepted_steps,
                torque_check_stride,
                &mut accepted, &mut rejected,
            ) {
                settings.max_err = max_err;
                return RelaxReport {
                    accepted_steps: accepted,
                    rejected_steps: rejected,
                    torque_checks,
                    torque_field_rebuilds,
                    stop_reason: RelaxStopReason::MaxAcceptedSteps,
                    last_stage_stop: Some(RelaxStageStop::MaxAcceptedSteps),
                    final_torque: Some(t_prev),
                    final_torque_max,
                    final_max_err: max_err,
                    final_dt: params.dt,
                };
            }

            let t_new = torque_now(
                grid, m, params, material, scratch, mask,
                torque_metric,
                &mut b_eff_scratch,
                &mut torque_checks,
                &mut torque_field_rebuilds,
            );

            if plateau_enabled {
                comparisons += 1;

                if comparisons < plateau_min_checks {
                    plateau_fails = 0;
                } else {
                    // MuMax-like plateau: strict float32 T1 < T0.
                    if mumax_plateau_f32 {
                        let t_new_f32: f32 = (t_new * n_f) as f32;
                        let improved = t_new_f32 < t_prev_f32;
                        // Track the quantised value (MuMax compares float32 values).
                        t_prev_f32 = t_new_f32;

                        if improved {
                            plateau_fails = 0;
                        } else {
                            plateau_fails += 1;
                        }
                    } else {
                        // Generic plateau with optional tolerances.
                        let need_rel = plateau_rel * t_prev.abs().max(1e-30);
                        let need = need_rel.max(plateau_abs);
                        let improved = (t_prev - t_new) > need;

                        if improved {
                            plateau_fails = 0;
                        } else {
                            plateau_fails += 1;
                        }
                    }
                }
            }

            t_prev = t_new;
        }

        final_torque = Some(t_prev);

        // Tighten (MuMax: MaxErr /= sqrt2)
        if max_err <= tighten_floor {
            break;
        }
        max_err *= tighten_factor;
        if max_err < tighten_floor {
            max_err = tighten_floor;
        }
    }

    // Always compute final max torque (needed by equilibrate.rs max-torque gate)
    {
        let tmax = torque_now(
            grid, m, params, material, scratch, mask,
            TorqueMetric::Max,
            &mut b_eff_scratch,
            &mut torque_checks,
            &mut torque_field_rebuilds,
        );
        final_torque_max = Some(tmax);
    }

    // -------------------------
    // Optional internal gate (usually disabled)
    // -------------------------
    let gate_enabled = (gate_metric.is_some() || gate_max.is_some()) && gate_extra_steps > 0;

    if gate_enabled {
        let start_accepted = accepted;
        let mut last_t = final_torque.unwrap_or(f64::INFINITY);
        let mut gate_plateau_fails: usize = 0;

        loop {
            let (ok, t_metric, t_max) = gates_ok(
                grid, m, params, material, scratch, mask,
                torque_metric,
                gate_metric,
                gate_max,
                &mut b_eff_scratch,
                &mut torque_checks,
                &mut torque_field_rebuilds,
            );

            final_torque = Some(t_metric);
            if t_max.is_finite() {
                final_torque_max = Some(t_max);
            }

            if ok {
                break;
            }

            let extra_used = accepted.saturating_sub(start_accepted);
            if extra_used >= gate_extra_steps {
                settings.max_err = max_err;
                return RelaxReport {
                    accepted_steps: accepted,
                    rejected_steps: rejected,
                    torque_checks,
                    torque_field_rebuilds,
                    stop_reason: RelaxStopReason::TorqueGateNotSatisfied,
                    last_stage_stop,
                    final_torque,
                    final_torque_max,
                    final_max_err: max_err,
                    final_dt: params.dt,
                };
            }

            if gate_plateau_fails_limit > 0 {
                let need_rel = plateau_rel * last_t.abs().max(1e-30);
                let need = need_rel.max(plateau_abs);
                let improved = (last_t - t_metric) > need;

                if improved {
                    gate_plateau_fails = 0;
                } else {
                    gate_plateau_fails += 1;
                }

                if gate_plateau_fails >= gate_plateau_fails_limit {
                    settings.max_err = max_err;
                    return RelaxReport {
                        accepted_steps: accepted,
                        rejected_steps: rejected,
                        torque_checks,
                        torque_field_rebuilds,
                        stop_reason: RelaxStopReason::TorqueGateNotSatisfied,
                        last_stage_stop,
                        final_torque,
                        final_torque_max,
                        final_max_err: max_err,
                        final_dt: params.dt,
                    };
                }

                last_t = t_metric;
            }

            if !advance_accepted(
                m, params, material, scratch, mask,
                max_err, headroom, dt_min, dt_max,
                max_accepted_steps,
                torque_check_stride,
                &mut accepted, &mut rejected,
            ) {
                settings.max_err = max_err;
                return RelaxReport {
                    accepted_steps: accepted,
                    rejected_steps: rejected,
                    torque_checks,
                    torque_field_rebuilds,
                    stop_reason: RelaxStopReason::MaxAcceptedSteps,
                    last_stage_stop: Some(RelaxStageStop::MaxAcceptedSteps),
                    final_torque,
                    final_torque_max,
                    final_max_err: max_err,
                    final_dt: params.dt,
                };
            }
        }
    }

    // Write back final max_err (MuMax mutates MaxErr during relax; we mirror that).
    settings.max_err = max_err;

    RelaxReport {
        accepted_steps: accepted,
        rejected_steps: rejected,
        torque_checks,
        torque_field_rebuilds,
        stop_reason: RelaxStopReason::TightenFloorReached,
        last_stage_stop,
        final_torque,
        final_torque_max,
        final_max_err: max_err,
        final_dt: params.dt,
    }
}

/// Backwards-compatible wrapper.
pub fn relax(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK23Scratch,
    mask: FieldMask,
    settings: &mut RelaxSettings,
) {
    let _ = relax_with_report(grid, m, params, material, scratch, mask, settings);
}

/// Convenience: settings preset closer to MuMax defaults.
///
/// If you want MuMax’s *classic* plateau behaviour:
/// - torque_metric = MeanSq
/// - torque_threshold = None
/// - torque_plateau_checks = 1, min_checks = 1, rel=0, abs=0
pub fn mumax_like_relax_settings() -> RelaxSettings {
    let mut s = RelaxSettings::default();
    s.phase1_enabled = true;
    s.phase2_enabled = true;
    s.energy_stride = 3;
    s.rel_energy_tol = 0.0; // MuMax uses strict E1 < E0

    s.torque_metric = TorqueMetric::MeanSq;
    s.torque_threshold = None;

    s.torque_plateau_checks = 1;
    s.torque_plateau_min_checks = 1;
    s.torque_plateau_rel = 0.0;
    s.torque_plateau_abs = 0.0;

    s.tighten_factor = std::f64::consts::FRAC_1_SQRT_2;
    s.tighten_floor = 1e-9;

    s
}