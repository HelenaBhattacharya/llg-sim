// src/relax.rs
//
// MuMax-like relaxation controller:
//  - Precession suppressed (damping-only LLG RHS)
//  - Adaptive stepping (delegates to RK23 relax stepper)
//  - Phase 1: energy descent until noise floor
//  - Phase 2: torque descent with tolerance tightening
//
// This is intended to behave like MuMax3's Relax() (energy-first, then torque).

use crate::effective_field::{build_h_eff_masked, FieldMask};
use crate::energy::compute_total_energy;
use crate::grid::Grid2D;
use crate::llg::{step_llg_rk23_recompute_field_masked_relax_adaptive, RK23Scratch};
use crate::params::{LLGParams, Material};
use crate::vec3::cross;
use crate::vector_field::VectorField2D;

#[derive(Debug, Clone)]
pub struct RelaxSettings {
    /// Initial adaptive error tolerance (MuMax MaxErr analogue).
    pub max_err: f64,
    /// Controller safety factor (MuMax Headroom analogue).
    pub headroom: f64,
    pub dt_min: f64,
    pub dt_max: f64,

    /// Energy check stride N (MuMax uses N=3).
    pub energy_stride: usize,
    /// Relative energy tolerance for "noise floor".
    pub rel_energy_tol: f64,

    /// If Some(tau), stop when max|m×B| < tau (Tesla) at current MaxErr,
    /// then tighten max_err and continue until tighten_floor reached.
    pub torque_threshold: Option<f64>,

    /// How often to compute the (expensive) torque metric during Phase 2.
    /// 1 = every accepted step (exact previous behaviour).
    /// Larger values reduce overhead (helpful for SP2), without changing the torque definition.
    pub torque_check_stride: usize,

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
            energy_stride: 3,
            rel_energy_tol: 1e-12,
            torque_threshold: Some(1e-4),
            torque_check_stride: 1, // default preserves existing behaviour (SP4 unchanged)
            tighten_factor: std::f64::consts::FRAC_1_SQRT_2,
            tighten_floor: 1e-9,
            max_accepted_steps: 2_000_000,
        }
    }
}

/// Compute max torque = max |m × B_eff| (Tesla) over all cells.
fn max_torque_inf(
    grid: &Grid2D,
    m: &VectorField2D,
    params: &LLGParams,
    material: &Material,
    mask: FieldMask,
) -> f64 {
    let mut b_eff = VectorField2D::new(*grid);
    build_h_eff_masked(grid, m, &mut b_eff, params, material, mask);

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

    // -------------------------
    // Phase 1: energy descent
    // -------------------------
    let mut e0 = compute_total_energy(grid, m, material, params.b_ext);

    // Do a few strides until energy no longer decreases beyond noise floor
    loop {
        // Take energy_stride accepted adaptive steps
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
        // noise floor reached (energy no longer reliably decreases)
        break;
    }

    // -------------------------
    // Phase 2: torque descent
    // -------------------------
    // Tighten max_err progressively (MuMax style) while reducing torque.
    loop {
        if let Some(tau) = settings.torque_threshold {
            // Stride must be >= 1
            let stride = settings.torque_check_stride.max(1);

            // Compute torque once at the start of this tolerance stage
            let mut tmax = max_torque_inf(grid, m, params, material, mask);

            // Step until torque is below threshold, checking only every `stride` accepted steps
            let mut since_check: usize = 0;
            while tmax > tau {
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
                    tmax = max_torque_inf(grid, m, params, material, mask);
                    since_check = 0;
                }
            }

            // Ensure we finish this stage with a “fresh” torque evaluation (for determinism)
            if since_check != 0 {
                let _ = max_torque_inf(grid, m, params, material, mask);
            }
        } else {
            // If no threshold: just do some steps and require torque to decrease on average
            let t0 = max_torque_inf(grid, m, params, material, mask);

            for _ in 0..(10 * settings.energy_stride) {
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

            let t1 = max_torque_inf(grid, m, params, material, mask);
            if !(t1 < t0) {
                // torque not improving; proceed to tighten (or stop if already tight)
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