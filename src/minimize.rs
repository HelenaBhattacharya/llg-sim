// src/minimize.rs
//
// SP2-oriented damping-only minimiser.
// One effective-field build per iteration (≈ one demag FFT per iter).
//
// Update direction: d = (m × B) × m = B - m (m·B)
// This corresponds to the damping-only descent direction (up to a scalar factor).
//
// Stop: max |m × B| < torque_threshold (Tesla)

use crate::effective_field::{FieldMask, build_h_eff_masked};
use crate::grid::Grid2D;
use crate::params::{LLGParams, Material};
use crate::vec3::cross;
use crate::vector_field::VectorField2D;

use rayon::prelude::*;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct MinimizeSettings {
    pub torque_threshold: f64, // Tesla
    pub max_iters: usize,

    // Pseudo-step size for descent (dimensionless scale multiplying d).
    // Good starting point ~ alpha*gamma*dt used in relax (order ~1e-2).
    pub lambda0: f64,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub grow: f64,
    pub shrink: f64,

    // Stall detection: if torque fails to improve meaningfully for many iters -> bail.
    pub stall_iters: usize,
    pub stall_rel: f64, // e.g. 1e-4 means "no improvement > 0.01%"
    pub min_iters_before_stall: usize,

    /// Optional MuMax-like convergence on max dM (maximum per-cell change in m).
    /// If Some(x), consider converged when the last `dm_samples` values of max_dM are all < x.
    pub dm_stop: Option<f64>,

    /// Number of max dM samples used for convergence check (MuMax MinimizerSamples analogue).
    pub dm_samples: usize,

    /// Optional: require mean torque to be below this value for dm_stop to count as full convergence.
    /// If dm_stop triggers but mean torque is still above this gate, the minimizer returns early with
    /// `dm_converged=true` but `converged=false` (caller can then decide to run Relax()).
    pub dm_torque_gate: Option<f64>,

    /// Optional: require max torque to be below this value for dm_stop to count as full convergence.
    /// Works alongside `dm_torque_gate` (mean torque gate).
    pub dm_torque_gate_max: Option<f64>,

    /// Do not allow dm_stop-based early exit until at least this many iterations have run.
    pub dm_min_iters: usize,

    /// Enable Rayon parallelism for the per-cell update/metric pass.
    /// Off by default for maximal reproducibility; callers can also enable via env `LLG_MINIMIZE_PAR=1`.
    pub parallel: bool,

    // Optional: print every N iterations (0 disables)
    pub print_every: usize,
}

impl Default for MinimizeSettings {
    fn default() -> Self {
        Self {
            torque_threshold: 5e-4,
            max_iters: 20_000,

            lambda0: 2e-2,
            lambda_min: 1e-5,
            lambda_max: 5e-2,
            grow: 1.05,
            shrink: 0.8,

            stall_iters: 5000,
            stall_rel: 5e-4,
            min_iters_before_stall: 500,

            dm_stop: Some(1e-6),
            dm_samples: 10,
            dm_torque_gate: Some(2e-3),
            dm_torque_gate_max: Some(5e-3),
            dm_min_iters: 50,

            parallel: false,

            print_every: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MinimizeReport {
    pub iters: usize,
    pub final_torque: f64,
    pub converged: bool,
    pub stalled: bool,
    pub final_max_dm: f64,
    pub dm_converged: bool,
    pub final_tmean: f64,
    pub final_lambda: f64,
}

/// Minimise in-place. Returns report (converged/stalled).
pub fn minimize_damping_only(
    grid: &Grid2D,
    m: &mut VectorField2D,
    params: &LLGParams,
    material: &Material,
    mask: FieldMask,
    settings: &MinimizeSettings,
) -> MinimizeReport {
    let mut b_eff = VectorField2D::new(*grid);

    // Optional Rayon parallelism for the per-cell update/metric pass.
    // We keep reductions deterministic by aggregating per-chunk stats in a fixed order.
    let use_parallel = settings.parallel || std::env::var("LLG_MINIMIZE_PAR").is_ok();
    const CHUNK: usize = 2048;

    // Preallocate a per-chunk stats buffer for deterministic aggregation in the parallel path.
    // This avoids per-iteration allocations and avoids locking.
    let n_tot = m.data.len();
    let n_chunks = (n_tot + CHUNK - 1) / CHUNK;
    let mut stats_buf: Vec<(f64, f64, f64)> = vec![(0.0_f64, 0.0_f64, 0.0_f64); n_chunks];

    let mut lambda = settings.lambda0;
    // Track previous *mean* torque for controller decisions (more robust than max torque).
    let mut t_prev_mean = f64::INFINITY;

    let mut stall_count = 0usize;
    let mut dm_hist: VecDeque<f64> = VecDeque::new();
    let mut last_max_dm: f64 = f64::INFINITY;
    let mut last_tmax: f64 = f64::INFINITY;
    let mut last_tmean: f64 = f64::INFINITY;

    for it in 0..settings.max_iters {
        // Build effective field once (dominant cost: demag FFT)
        build_h_eff_masked(grid, m, &mut b_eff, params, material, mask);

        // Compute max torque, mean torque, and max dM, and update m in one pass.
        let mut tmax = 0.0;
        let mut tsum = 0.0;
        let mut max_dm = 0.0;

        if use_parallel {
            // Capture lambda by value for thread-safe use in Rayon closures.
            let lambda_step = lambda;

            // Deterministic aggregation: write chunk stats into preallocated buffer.
            stats_buf
                .par_iter_mut()
                .zip(
                    m.data
                        .par_chunks_mut(CHUNK)
                        .zip(b_eff.data.par_chunks(CHUNK)),
                )
                .for_each(|(slot, (m_chunk, b_chunk))| {
                    let mut ltmax = 0.0_f64;
                    let mut ltsum = 0.0_f64;
                    let mut lmax_dm = 0.0_f64;

                    for (mi, bi) in m_chunk.iter_mut().zip(b_chunk.iter()) {
                        let m0 = *mi;
                        let b0 = *bi;

                        // torque = m × B
                        let t = cross(m0, b0);
                        let tmag = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                        ltsum += tmag;
                        if tmag > ltmax {
                            ltmax = tmag;
                        }

                        // descent direction: d = (m × B) × m  (damping-only direction)
                        let d = cross(t, m0);

                        let mut x = m0[0] + lambda_step * d[0];
                        let mut y = m0[1] + lambda_step * d[1];
                        let mut z = m0[2] + lambda_step * d[2];

                        // renormalise
                        let n2 = x * x + y * y + z * z;
                        if n2 > 0.0 {
                            let inv = 1.0 / n2.sqrt();
                            x *= inv;
                            y *= inv;
                            z *= inv;
                        }

                        // Track max dM = max_i ||m_new - m_old||
                        let dx = x - m0[0];
                        let dy = y - m0[1];
                        let dz = z - m0[2];
                        let dm = (dx * dx + dy * dy + dz * dz).sqrt();
                        if dm > lmax_dm {
                            lmax_dm = dm;
                        }

                        *mi = [x, y, z];
                    }

                    *slot = (ltmax, ltsum, lmax_dm);
                });

            for &(ltmax, ltsum, lmax_dm) in stats_buf.iter() {
                tsum += ltsum;
                if ltmax > tmax {
                    tmax = ltmax;
                }
                if lmax_dm > max_dm {
                    max_dm = lmax_dm;
                }
            }
        } else {
            for (mi, bi) in m.data.iter_mut().zip(b_eff.data.iter()) {
                let m0 = *mi;
                let b0 = *bi;

                // torque = m × B
                let t = cross(m0, b0);
                let tmag = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                tsum += tmag;
                if tmag > tmax {
                    tmax = tmag;
                }

                // descent direction: d = (m × B) × m  (damping-only direction)
                let d = cross(t, m0);

                let mut x = m0[0] + lambda * d[0];
                let mut y = m0[1] + lambda * d[1];
                let mut z = m0[2] + lambda * d[2];

                // renormalise
                let n2 = x * x + y * y + z * z;
                if n2 > 0.0 {
                    let inv = 1.0 / n2.sqrt();
                    x *= inv;
                    y *= inv;
                    z *= inv;
                }

                // Track max dM = max_i ||m_new - m_old||
                let dx = x - m0[0];
                let dy = y - m0[1];
                let dz = z - m0[2];
                let dm = (dx * dx + dy * dy + dz * dz).sqrt();
                if dm > max_dm {
                    max_dm = dm;
                }

                *mi = [x, y, z];
            }
        }

        let n = m.data.len() as f64;
        let tmean = tsum / n.max(1.0);

        last_tmean = tmean;

        if settings.print_every > 0 && it % settings.print_every == 0 {
            println!(
                "      [minimize] it={}  tmax={:.3e}  tmean={:.3e}  lambda={:.3e}",
                it, tmax, tmean, lambda
            );
        }

        if tmax < settings.torque_threshold {
            return MinimizeReport {
                iters: it + 1,
                final_torque: tmax,
                converged: true,
                stalled: false,
                final_max_dm: max_dm,
                dm_converged: false,
                final_tmean: tmean,
                final_lambda: lambda,
            };
        }

        // MuMax-like convergence: stop when max dM has been below dm_stop for dm_samples consecutive samples.
        // Optionally gate this by mean torque to avoid false "dm" convergence when step size collapses.
        last_max_dm = max_dm;
        last_tmax = tmax;
        if let Some(dm_stop) = settings.dm_stop {
            if (it + 1) >= settings.dm_min_iters {
                dm_hist.push_back(max_dm);
                while dm_hist.len() > settings.dm_samples.max(1) {
                    dm_hist.pop_front();
                }
                if dm_hist.len() == settings.dm_samples.max(1)
                    && dm_hist.iter().all(|&v| v < dm_stop)
                {
                    let mean_ok = match settings.dm_torque_gate {
                        None => true,
                        Some(g) => tmean <= g,
                    };
                    let max_ok = match settings.dm_torque_gate_max {
                        None => true,
                        Some(g) => tmax <= g,
                    };
                    let torque_ok = mean_ok && max_ok;

                    // dm_converged=true means dm_stop fired; converged indicates torque gates passed
                    return MinimizeReport {
                        iters: it + 1,
                        final_torque: tmax,
                        converged: torque_ok,
                        stalled: false,
                        final_max_dm: max_dm,
                        dm_converged: true,
                        final_tmean: tmean,
                        final_lambda: lambda,
                    };
                }
            }
        }

        // Adapt lambda based on improvement in *mean* torque (no extra field builds)
        if tmean < t_prev_mean {
            lambda = (lambda * settings.grow).min(settings.lambda_max);
        } else {
            lambda = (lambda * settings.shrink).max(settings.lambda_min);
        }

        // Stall detection (only after a minimum number of iterations)
        if (it + 1) >= settings.min_iters_before_stall && t_prev_mean.is_finite() {
            let good_improve = t_prev_mean - tmean;
            let need = settings.stall_rel * t_prev_mean.abs().max(1e-30);
            if good_improve <= need {
                stall_count += 1;
            } else {
                stall_count = 0;
            }
        }

        // Do NOT treat hitting lambda_min as an automatic stall.
        // In demag-dominated problems (SP2), the step size can collapse early even while
        // the configuration continues to improve slowly. We rely on the plateau counter
        // (stall_count) to decide when progress has genuinely stopped.
        if stall_count >= settings.stall_iters {
            return MinimizeReport {
                iters: it + 1,
                final_torque: tmax,
                converged: false,
                stalled: true,
                final_max_dm: last_max_dm,
                dm_converged: false,
                final_tmean: tmean,
                final_lambda: lambda,
            };
        }

        t_prev_mean = tmean;
    }

    MinimizeReport {
        iters: settings.max_iters,
        final_torque: last_tmax,
        converged: false,
        stalled: false,
        final_max_dm: last_max_dm,
        dm_converged: false,
        final_tmean: last_tmean,
        final_lambda: lambda,
    }
}
