// ===============================
// src/relax.rs (FULL FILE)
// ===============================
//
// MuMax-like relaxation controller:
//  - Precession suppressed (damping-only LLG RHS)
//  - Adaptive stepping (delegates to RK23 relax stepper)
//  - Phase 1: energy descent until noise floor
//  - Phase 2: torque descent with tolerance tightening
//
// This is intended to behave like MuMax3's Relax() (energy-first, then torque).
//
// Key SP2-runtime feature:
//  - Support MuMax-like "torque plateau" stopping (average torque stops decreasing),
//    rather than requiring max torque to fall below a hard threshold.
//
// IMPORTANT runtime optimisation:
//  - Torque checks reuse the last accepted RK23 field (scratch.last_b_eff()) when available,
//    avoiding an extra build_h_eff_masked() call (and demag FFT) at each check.

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
    /// Stage ended because torque plateaued.
    Plateau,
    /// Stage ended because accepted steps hit the cap.
    MaxAcceptedSteps,
}

#[derive(Debug, Clone)]
pub struct RelaxReport {
    pub accepted_steps: usize,
    pub rejected_steps: usize,

    /// Number of torque checks performed (including the initial stage check).
    pub torque_checks: usize,

    /// Number of times we rebuilt the effective field *specifically for torque checks*.
    /// (Calls to torque_metric_inplace.)
    pub torque_field_rebuilds: usize,

    /// How the final tightening loop terminated.
    pub stop_reason: RelaxStopReason,

    /// How the last stage ended (threshold/plateau/maxsteps).
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

    /// Enable Phase 1 (energy descent). Default true to preserve MuMax-like behaviour.
    /// For SP2 on CPU you may disable this to avoid expensive energy evaluations.
    pub phase1_enabled: bool,

    /// Enable Phase 2 (torque descent). Default true.
    pub phase2_enabled: bool,

    /// Energy check stride N (MuMax uses N=3).
    pub energy_stride: usize,
    /// Relative energy tolerance for "noise floor".
    pub rel_energy_tol: f64,

    /// Torque metric used in Phase 2.
    pub torque_metric: TorqueMetric,

    /// If Some(tau), stop when torque_metric(m×B) < tau (Tesla) at current MaxErr,
    /// then tighten max_err and continue until tighten_floor reached.
    ///
    /// If None, Phase 2 relies on plateau detection + tightening.
    pub torque_threshold: Option<f64>,

    /// How often to compute the torque metric during Phase 2.
    pub torque_check_stride: usize,

    /// Plateau stopping: if torque does not improve by at least `torque_plateau_rel`
    /// for `torque_plateau_checks` consecutive torque checks, treat as noise floor.
    pub torque_plateau_checks: usize,
    pub torque_plateau_rel: f64,
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

            torque_plateau_checks: 0, // disabled by default
            torque_plateau_rel: 1e-3,
            torque_plateau_min_checks: 5,

            tighten_factor: std::f64::consts::FRAC_1_SQRT_2,
            tighten_floor: 1e-9,
            max_accepted_steps: 2_000_000,
        }
    }
}

/// Compute a torque metric over the grid using a caller-provided scratch buffer.
/// This *builds* B_eff (expensive if demag is on).
fn torque_metric_inplace(
    grid: &Grid2D,
    m: &VectorField2D,
    params: &LLGParams,
    material: &Material,
    mask: FieldMask,
    metric: TorqueMetric,
    b_eff_scratch: &mut VectorField2D,
) -> f64 {
    debug_assert!(b_eff_scratch.grid.nx == grid.nx);
    debug_assert!(b_eff_scratch.grid.ny == grid.ny);

    build_h_eff_masked(grid, m, b_eff_scratch, params, material, mask);

    let n = m.data.len() as f64;

    match metric {
        TorqueMetric::Max => {
            let mut maxv = 0.0;
            for (mi, bi) in m.data.iter().zip(b_eff_scratch.data.iter()) {
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
            for (mi, bi) in m.data.iter().zip(b_eff_scratch.data.iter()) {
                let t = cross(*mi, *bi);
                let mag = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                sum += mag;
            }
            sum / n.max(1.0)
        }
        TorqueMetric::Rms => {
            let mut sum2 = 0.0;
            for (mi, bi) in m.data.iter().zip(b_eff_scratch.data.iter()) {
                let t = cross(*mi, *bi);
                let mag2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
                sum2 += mag2;
            }
            (sum2 / n.max(1.0)).sqrt()
        }
    }
}

/// Compute a torque metric given a precomputed effective field (avoids rebuilding B_eff).
fn torque_metric_from_field(m: &VectorField2D, b_eff: &VectorField2D, metric: TorqueMetric) -> f64 {
    debug_assert!(m.data.len() == b_eff.data.len());
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
    }
}

/// New: relax with a detailed report (preferred for higher-level orchestration).
pub fn relax_with_report(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK23Scratch,
    mask: FieldMask,
    settings: &mut RelaxSettings,
) -> RelaxReport {
    // Clamp dt initially
    params.dt = params.dt.clamp(settings.dt_min, settings.dt_max);

    let mut accepted: usize = 0;
    let mut rejected: usize = 0;

    let mut torque_checks: usize = 0;
    let mut torque_field_rebuilds: usize = 0;

    let mut last_stage_stop: Option<RelaxStageStop> = None;

    // Scratch buffer for torque checks (only used when we *must* rebuild B_eff)
    let mut b_eff_scratch = VectorField2D::new(*grid);

    // -------------------------
    // Phase 1: energy descent
    // -------------------------
    if settings.phase1_enabled {
        let mut e0 = compute_total_energy(grid, m, material, params.b_ext);

        loop {
            for _ in 0..settings.energy_stride {
                if accepted >= settings.max_accepted_steps {
                    return RelaxReport {
                        accepted_steps: accepted,
                        rejected_steps: rejected,
                        torque_checks,
                        torque_field_rebuilds,
                        stop_reason: RelaxStopReason::MaxAcceptedSteps,
                        last_stage_stop,
                        final_torque: None,
                        final_max_err: settings.max_err,
                        final_dt: params.dt,
                    };
                }

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

                if ok {
                    accepted += 1;
                } else {
                    rejected += 1;
                }
            }

            let e1 = compute_total_energy(grid, m, material, params.b_ext);
            let tol = settings.rel_energy_tol * e0.abs().max(1e-30);
            if e1 < e0 - tol {
                e0 = e1;
                continue;
            }
            break; // energy noise floor
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

    let final_torque: Option<f64>;

    loop {
        let stride = settings.torque_check_stride.max(1);

        let plateau_enabled = settings.torque_plateau_checks > 0;
        let mut plateau_count: usize = 0;
        let mut check_count: usize = 0;

        // metric at start of stage (requires a fresh B_eff build)
        torque_checks += 1;
        torque_field_rebuilds += 1;
        let mut t_prev = torque_metric_inplace(
            grid,
            m,
            params,
            material,
            mask,
            settings.torque_metric,
            &mut b_eff_scratch,
        );

        let mut since_check: usize = 0;

        loop {
            // Stop condition on threshold (if enabled)
            if let Some(tau) = settings.torque_threshold {
                if t_prev <= tau {
                    last_stage_stop = Some(RelaxStageStop::BelowThreshold);
                    break;
                }
            }

            // Stop condition on plateau
            if plateau_enabled && plateau_count >= settings.torque_plateau_checks {
                last_stage_stop = Some(RelaxStageStop::Plateau);
                break;
            }

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

            if !ok {
                rejected += 1;
                continue;
            }

            accepted += 1;
            since_check += 1;

            if since_check >= stride {
                torque_checks += 1;

                let t_new = if let Some(b_eff) = scratch.last_b_eff() {
                    // Reuse the field computed during the last accepted RK23 step.
                    torque_metric_from_field(m, b_eff, settings.torque_metric)
                } else {
                    // Fallback: rebuild effective field (e.g. if last step was rejected).
                    torque_field_rebuilds += 1;
                    torque_metric_inplace(
                        grid,
                        m,
                        params,
                        material,
                        mask,
                        settings.torque_metric,
                        &mut b_eff_scratch,
                    )
                };

                // Plateau update
                if plateau_enabled {
                    check_count += 1;
                    if check_count >= settings.torque_plateau_min_checks {
                        let need = settings.torque_plateau_rel * t_prev.abs().max(1e-30);
                        let improved = (t_prev - t_new) > need;
                        if improved {
                            plateau_count = 0;
                        } else {
                            plateau_count += 1;
                        }
                    }
                }

                t_prev = t_new;
                since_check = 0;
            }
        }

        // (final_torque will be set only if we break out of the outer loop)

        // Tighten tolerance
        if settings.max_err <= settings.tighten_floor {
            // Only record final_torque once, at the terminal stage, to avoid unused-assignment warnings.
            final_torque = Some(t_prev);
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

