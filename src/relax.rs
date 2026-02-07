// src/relax.rs
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
//    rather than requiring max torque to fall below a hard threshold.  [oai_citation:3‡GitHub](https://github.com/mumax/3/issues/146?utm_source=chatgpt.com)
//
// Defaults preserve previous behaviour: max torque threshold + no plateau.

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
    /// MuMax stopping is described in terms of average torque plateau.  [oai_citation:4‡GitHub](https://github.com/mumax/3/issues/146?utm_source=chatgpt.com)
    pub torque_metric: TorqueMetric,

    /// If Some(tau), stop when torque_metric(m×B) < tau (Tesla) at current MaxErr,
    /// then tighten max_err and continue until tighten_floor reached.
    ///
    /// If None, Phase 2 relies on plateau detection + tightening.
    pub torque_threshold: Option<f64>,

    /// How often to compute the (expensive) torque metric during Phase 2.
    /// 1 = every accepted step (most expensive).
    pub torque_check_stride: usize,

    /// Plateau stopping: if torque does not improve by at least `torque_plateau_rel`
    /// for `torque_plateau_checks` consecutive torque checks, treat as noise floor
    /// for this stage and proceed to tighten (MuMax-like).  [oai_citation:5‡GitHub](https://github.com/mumax/3/issues/146?utm_source=chatgpt.com)
    ///
    /// Set checks = 0 to disable plateau stopping (default).
    pub torque_plateau_checks: usize,
    /// Relative improvement required to *avoid* counting towards plateau.
    /// Example: 1e-3 means torque must drop by at least 0.1% between checks.
    pub torque_plateau_rel: f64,
    /// Do not allow plateau stopping until at least this many torque checks have happened in the stage.
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

            torque_plateau_checks: 0,     // disabled by default (preserve old behaviour)
            torque_plateau_rel: 1e-3,
            torque_plateau_min_checks: 5,

            tighten_factor: std::f64::consts::FRAC_1_SQRT_2,
            tighten_floor: 1e-9,
            max_accepted_steps: 2_000_000,
        }
    }
}

/// Compute a torque metric over the grid using a caller-provided scratch buffer.
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

/// Relax `m` in-place using MuMax-like strategy.
/// Does not advance any physical "time" variable; it just minimises.
pub fn relax(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK23Scratch,
    mask: FieldMask,
    settings: &mut RelaxSettings,
) {
    // Clamp dt initially
    params.dt = params.dt.clamp(settings.dt_min, settings.dt_max);

    let mut accepted: usize = 0;

    // Scratch buffer for torque checks
    let mut b_eff_scratch = VectorField2D::new(*grid);

    // -------------------------
    // Phase 1: energy descent
    // -------------------------
    if settings.phase1_enabled {
        let mut e0 = compute_total_energy(grid, m, material, params.b_ext);

        loop {
            for _ in 0..settings.energy_stride {
                if accepted >= settings.max_accepted_steps {
                    return;
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
        return;
    }

    loop {
        let stride = settings.torque_check_stride.max(1);

        let plateau_enabled = settings.torque_plateau_checks > 0;
        let mut plateau_count: usize = 0;
        let mut check_count: usize = 0;

        // metric at start of stage
        let mut t_prev = torque_metric_inplace(
            grid,
            m,
            params,
            material,
            mask,
            settings.torque_metric,
            &mut b_eff_scratch,
        );

        // -------------------------
        // If threshold is provided: run until below threshold OR plateau.
        // If threshold is None: run until plateau.
        // -------------------------
        let mut since_check: usize = 0;

        loop {
            // Stop condition on threshold (if enabled)
            if let Some(tau) = settings.torque_threshold {
                if t_prev <= tau {
                    break;
                }
            }

            // Stop condition on plateau
            if plateau_enabled && plateau_count >= settings.torque_plateau_checks {
                break;
            }

            if accepted >= settings.max_accepted_steps {
                return;
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
                continue;
            }

            accepted += 1;
            since_check += 1;

            if since_check >= stride {
                let t_new = torque_metric_inplace(
                    grid,
                    m,
                    params,
                    material,
                    mask,
                    settings.torque_metric,
                    &mut b_eff_scratch,
                );

                // ---- plateau update (inline; avoids borrow-checker issue) ----
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
                // ------------------------------------------------------------

                t_prev = t_new;
                since_check = 0;
            }
        }

        // Tighten tolerance
        if settings.max_err <= settings.tighten_floor {
            break;
        }
        settings.max_err *= settings.tighten_factor;
        if settings.max_err < settings.tighten_floor {
            settings.max_err = settings.tighten_floor;
        }
    }
}