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
// Design goals (SP2 / demag-heavy problems on CPU):
//  - Keep the control-flow close to the earlier “old relax.rs” stage-based structure.
//  - Reuse the last accepted RK23 effective field for torque checks when available,
//    to avoid extra build_h_eff_masked() calls (i.e., extra demag FFTs).
//  - Avoid the “silent no-op” failure mode (i.e., Relax returns without taking any steps)
//    by using a stage-based tightening loop (even if max_err == tighten_floor).
//
// Notes:
//  - Plateau stopping is optional and controlled by torque_plateau_checks.
//  - We support an absolute + relative improvement floor:
//      improved iff (t_prev - t_new) > max(torque_plateau_abs, torque_plateau_rel*|t_prev|)
//    Set both to 0 for strict MuMax “steady or increasing” semantics.
//
// IMPORTANT:
//  - This file focuses on correctness/robustness first.
//  - Big runtime wins (FSAL in RK23, demag column FFT parallelisation) come later.

use crate::effective_field::{build_h_eff_masked, FieldMask};
use crate::energy::compute_total_energy;
use crate::grid::Grid2D;
use crate::llg::{step_llg_rk23_recompute_field_masked_relax_adaptive, RK23Scratch};
use crate::params::{LLGParams, Material};
use crate::vec3::cross;
use crate::vector_field::VectorField2D;

#[derive(Debug, Clone, Copy)]
pub enum TorqueMetric {
    /// max_i |m_i × B_i|
    Max,
    /// (1/N) Σ_i |m_i × B_i|
    Mean,
    /// sqrt( (1/N) Σ_i |m_i × B_i|^2 )
    Rms,
    /// (1/N) Σ_i |m_i × B_i|^2  (MuMax-like “avg torque power” metric; no sqrt)
    MeanSq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaxStopReason {
    /// Stopped because accepted steps reached the hard cap.
    MaxAcceptedSteps,
    /// Phase 2 disabled (nothing to do).
    Phase2Disabled,
    /// Completed tightening down to `tighten_floor`.
    TightenFloorReached,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelaxStageStop {
    /// Stage ended because torque fell below the threshold.
    BelowThreshold,
    /// Stage ended because torque plateaued (stopped improving).
    Plateau,
    /// Stage ended because accepted steps hit the cap.
    MaxAcceptedSteps,
}

#[derive(Debug, Clone)]
pub struct RelaxReport {
    pub accepted_steps: usize,
    pub rejected_steps: usize,

    /// Number of torque checks performed (including initial stage checks).
    pub torque_checks: usize,

    /// Number of times we rebuilt the effective field specifically for torque checks.
    pub torque_field_rebuilds: usize,

    /// How the overall relax terminated.
    pub stop_reason: RelaxStopReason,

    /// How the last stage ended.
    pub last_stage_stop: Option<RelaxStageStop>,

    /// Final torque metric value (if Phase 2 ran).
    pub final_torque: Option<f64>,

    /// Final max_err after tightening.
    pub final_max_err: f64,

    /// Final dt in params after adaptive stepping.
    pub final_dt: f64,
}

#[derive(Debug, Clone)]
pub struct RelaxSettings {
    /// Initial adaptive error tolerance (MuMax MaxErr analogue).
    pub max_err: f64,
    /// Controller safety factor (MuMax Headroom analogue).
    pub headroom: f64,
    pub dt_min: f64,
    pub dt_max: f64,

    /// Enable Phase 1 (energy descent).
    pub phase1_enabled: bool,

    /// Enable Phase 2 (torque descent).
    pub phase2_enabled: bool,

    /// Energy check stride N (MuMax uses N=3).
    pub energy_stride: usize,
    /// Relative energy tolerance for "noise floor".
    pub rel_energy_tol: f64,

    /// Torque metric used in Phase 2.
    pub torque_metric: TorqueMetric,

    /// If Some(tau), stop when torque_metric(m×B) < tau (Tesla) at current max_err,
    /// then tighten max_err and continue until tighten_floor reached.
    ///
    /// If None, use plateau stopping (if enabled) + tightening.
    pub torque_threshold: Option<f64>,

    /// How often to compute the torque metric during Phase 2 (in accepted RK23 steps).
    pub torque_check_stride: usize,

    /// Plateau improvement floor: improved iff (t_prev - t_new) > max(abs, rel*|t_prev|).
    /// Set both to 0 for strict “steady or increasing” plateau semantics.
    pub torque_plateau_rel: f64,
    pub torque_plateau_abs: f64,

    /// Require this many consecutive “not improved” comparisons before declaring plateau.
    /// If 0, plateau stopping is disabled.
    ///
    /// MuMax’s default plateau logic corresponds to torque_plateau_checks = 1 with rel=0 and abs=0
    /// (i.e. stop the stage as soon as the torque is steady or increasing), followed by tightening.
    pub torque_plateau_checks: usize,

    /// Minimum number of comparisons before plateau can trigger.
    pub torque_plateau_min_checks: usize,

    /// Tighten factor applied to max_err each stage (MuMax uses /sqrt(2)).
    pub tighten_factor: f64,
    /// Stop tightening once max_err <= tighten_floor.
    pub tighten_floor: f64,

    /// Hard cap on total accepted steps (safety).
    pub max_accepted_steps: usize,
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

            // “Old relax” behaviour: plateau disabled by default unless caller turns it on.
            torque_plateau_rel: 1e-3,
            torque_plateau_abs: 0.0,
            torque_plateau_checks: 0,
            torque_plateau_min_checks: 5,

            tighten_factor: std::f64::consts::FRAC_1_SQRT_2,
            tighten_floor: 1e-9,
            max_accepted_steps: 2_000_000,
        }
    }
}

/// Compute torque metric from a precomputed effective field (avoids rebuilding B_eff).
fn torque_metric_from_field(m: &VectorField2D, b_eff: &VectorField2D, metric: TorqueMetric) -> f64 {
    debug_assert!(m.data.len() == b_eff.data.len());
    let n = m.data.len() as f64;

    match metric {
        TorqueMetric::Max => {
            let mut maxv = 0.0;
            for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
                let t = cross(*mi, *bi);
                let mag = (t[0]*t[0] + t[1]*t[1] + t[2]*t[2]).sqrt();
                if mag > maxv { maxv = mag; }
            }
            maxv
        }
        TorqueMetric::Mean => {
            let mut sum = 0.0;
            for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
                let t = cross(*mi, *bi);
                let mag = (t[0]*t[0] + t[1]*t[1] + t[2]*t[2]).sqrt();
                sum += mag;
            }
            sum / n.max(1.0)
        }
        TorqueMetric::Rms => {
            let mut sum2 = 0.0;
            for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
                let t = cross(*mi, *bi);
                let mag2 = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];
                sum2 += mag2;
            }
            (sum2 / n.max(1.0)).sqrt()
        }
        TorqueMetric::MeanSq => {
            let mut sum2 = 0.0;
            for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
                let t = cross(*mi, *bi);
                let mag2 = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];
                sum2 += mag2;
            }
            sum2 / n.max(1.0)
        }
    }
}

/// Compute torque metric, rebuilding B_eff (expensive).
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

/// Torque at current state; reuse last accepted RK23 field when possible.
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

/// One adaptive RK23 relax step (accepted/rejected).
#[inline]
fn rk23_step(
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK23Scratch,
    mask: FieldMask,
    settings: &RelaxSettings,
) -> bool {
    let (_eps, ok, _dt_used) = step_llg_rk23_recompute_field_masked_relax_adaptive(
        m,
        params,
        material,
        scratch,
        mask,
        settings.max_err,
        settings.headroom,
        settings.dt_min,
        settings.dt_max,
    );
    ok
}

/// Advance by `n_accept` accepted steps (rejects don’t count). Returns false if cap hit.
fn advance_accepted(
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK23Scratch,
    mask: FieldMask,
    settings: &RelaxSettings,
    n_accept: usize,
    accepted: &mut usize,
    rejected: &mut usize,
) -> bool {
    let target = accepted.saturating_add(n_accept.max(1));
    while *accepted < target {
        if *accepted >= settings.max_accepted_steps {
            return false;
        }
        if rk23_step(m, params, material, scratch, mask, settings) {
            *accepted += 1;
        } else {
            *rejected += 1;
        }
    }
    true
}

/// MuMax-like relax with a detailed report.
///
/// Structure is intentionally close to the earlier “old relax.rs”:
/// - Phase 1: energy descent (optional)
/// - Phase 2: repeated stages at fixed max_err, with optional plateau stopping
/// - After each stage: tighten max_err and repeat until tighten_floor reached
pub fn relax_with_report(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK23Scratch,
    mask: FieldMask,
    settings: &mut RelaxSettings,
) -> RelaxReport {
    params.dt = params.dt.clamp(settings.dt_min, settings.dt_max);
    scratch.invalidate_last_b_eff();

    let mut accepted: usize = 0;
    let mut rejected: usize = 0;

    let mut torque_checks: usize = 0;
    let mut torque_field_rebuilds: usize = 0;

    let mut last_stage_stop: Option<RelaxStageStop> = None;
    let mut final_torque: Option<f64> = None;

    // Scratch buffer for torque checks (used only when we must rebuild B_eff).
    let mut b_eff_scratch = VectorField2D::new(*grid);

    // -------------------------
    // Phase 1: energy descent
    // -------------------------
    if settings.phase1_enabled {
        let mut e0 = compute_total_energy(grid, m, material, params.b_ext);

        loop {
            let n = settings.energy_stride.max(1);

            if !advance_accepted(
                m,
                params,
                material,
                scratch,
                mask,
                settings,
                n,
                &mut accepted,
                &mut rejected,
            ) {
                return RelaxReport {
                    accepted_steps: accepted,
                    rejected_steps: rejected,
                    torque_checks,
                    torque_field_rebuilds,
                    stop_reason: RelaxStopReason::MaxAcceptedSteps,
                    last_stage_stop,
                    final_torque,
                    final_max_err: settings.max_err,
                    final_dt: params.dt,
                };
            }

            let e1 = compute_total_energy(grid, m, material, params.b_ext);
            let tol = settings.rel_energy_tol * e0.abs().max(1e-30);
            if e1 < e0 - tol {
                e0 = e1;
                continue;
            }
            break; // noise floor
        }
    }

    // -------------------------
    // Phase 2: torque descent
    // -------------------------
    if !settings.phase2_enabled {
        return RelaxReport {
            accepted_steps: accepted,
            rejected_steps: rejected,
            torque_checks,
            torque_field_rebuilds,
            stop_reason: RelaxStopReason::Phase2Disabled,
            last_stage_stop,
            final_torque: None,
            final_max_err: settings.max_err,
            final_dt: params.dt,
        };
    }

    // Stage stepping/measurement stride.
    let stride = settings.torque_check_stride.max(1);

    // Plateau config.
    let plateau_enabled = settings.torque_threshold.is_none() && settings.torque_plateau_checks > 0;
    let min_checks = settings.torque_plateau_min_checks.max(1);
    let need_fails = settings.torque_plateau_checks.max(1);

    loop {
        // -------------------------
        // Stage at current max_err
        // -------------------------

        // Initial torque sample (reuse last accepted field if available; else rebuild).
        let mut t_prev = torque_now(
            grid,
            m,
            params,
            material,
            scratch,
            mask,
            settings.torque_metric,
            &mut b_eff_scratch,
            &mut torque_checks,
            &mut torque_field_rebuilds,
        );

        let mut plateau_fails: usize = 0;
        let mut comparisons: usize = 0;

        // Run until threshold met OR plateau reached (if enabled).
        loop {
            // Threshold mode: stop stage if below threshold.
            if let Some(tau) = settings.torque_threshold {
                if t_prev <= tau {
                    last_stage_stop = Some(RelaxStageStop::BelowThreshold);
                    break;
                }
            }

            // Plateau mode: stop stage after enough consecutive “not improved” checks.
            if plateau_enabled && plateau_fails >= need_fails {
                last_stage_stop = Some(RelaxStageStop::Plateau);
                break;
            }

            // Cap check.
            if accepted >= settings.max_accepted_steps {
                return RelaxReport {
                    accepted_steps: accepted,
                    rejected_steps: rejected,
                    torque_checks,
                    torque_field_rebuilds,
                    stop_reason: RelaxStopReason::MaxAcceptedSteps,
                    last_stage_stop: Some(RelaxStageStop::MaxAcceptedSteps),
                    final_torque: Some(t_prev),
                    final_max_err: settings.max_err,
                    final_dt: params.dt,
                };
            }

            // Take `stride` accepted steps between torque checks (MuMax relaxSteps(N)-like).
            if !advance_accepted(
                m,
                params,
                material,
                scratch,
                mask,
                settings,
                stride,
                &mut accepted,
                &mut rejected,
            ) {
                return RelaxReport {
                    accepted_steps: accepted,
                    rejected_steps: rejected,
                    torque_checks,
                    torque_field_rebuilds,
                    stop_reason: RelaxStopReason::MaxAcceptedSteps,
                    last_stage_stop: Some(RelaxStageStop::MaxAcceptedSteps),
                    final_torque: Some(t_prev),
                    final_max_err: settings.max_err,
                    final_dt: params.dt,
                };
            }

            // Measure torque again.
            let t_new = torque_now(
                grid,
                m,
                params,
                material,
                scratch,
                mask,
                settings.torque_metric,
                &mut b_eff_scratch,
                &mut torque_checks,
                &mut torque_field_rebuilds,
            );

            // Plateau bookkeeping (only when torque_threshold is None).
            if plateau_enabled {
                comparisons += 1;

                // Apply min_checks “warmup” before allowing plateau stopping.
                if comparisons < min_checks {
                    plateau_fails = 0;
                } else {
                    let need_rel = settings.torque_plateau_rel * t_prev.abs().max(1e-30);
                    let need = need_rel.max(settings.torque_plateau_abs);
                    let improved = (t_prev - t_new) > need;

                    if improved {
                        plateau_fails = 0;
                    } else {
                        plateau_fails += 1;
                    }
                }
            }

            t_prev = t_new;
        }

        // Record torque at end of this terminal stage if we’re about to stop tightening.
        final_torque = Some(t_prev);

        // Tighten tolerance (stage-based, “old relax” style).
        if settings.max_err <= settings.tighten_floor {
            break;
        }
        settings.max_err *= settings.tighten_factor;
        if settings.max_err < settings.tighten_floor {
            settings.max_err = settings.tighten_floor;
        }
    }

    RelaxReport {
        accepted_steps: accepted,
        rejected_steps: rejected,
        torque_checks,
        torque_field_rebuilds,
        stop_reason: RelaxStopReason::TightenFloorReached,
        last_stage_stop,
        final_torque,
        final_max_err: settings.max_err,
        final_dt: params.dt,
    }
}

/// Backwards-compatible: keep the old signature for existing call sites.
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