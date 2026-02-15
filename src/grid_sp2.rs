// ===============================
// src/grid_sp2.rs
// ===============================
//
// One-shot, state-based grid selection for SP2 (uniform grid; no true AMR).
//
// Goal:
//  - Avoid hardcoding regime switches in terms of d/lex.
//  - Decide whether to refine based on *state metrics* computed from the remanent state.
//
// This module provides:
//  - `Sp2GridMode`: MuMax SP2-Appendix grid vs a generic target-cell/pow2 grid
//  - `Sp2GridPolicy`: thresholds + refinement factor
//  - `build_sp2_grid(...)`: build baseline grid for SP2 geometry
//  - `remanence_metrics(...)`: cheap nonuniformity metrics
//  - `maybe_refine_after_remanence(...)`: decide if one-shot refine is needed

use crate::grid::Grid2D;
use crate::relax::RelaxReport;
use crate::vector_field::VectorField2D;

// -------------------------
// Public types
// -------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sp2GridMode {
    /// Generic "target cell" sizing: choose cell size ~ (cell_over_lex * lex), then round nx and ny independently to powers of two.
    Mumax,
    /// MuMax SP2 Appendix-style sizing: enforce nx = 5*2^p, ny = nx/5, and dx = dy with cellsize ~ 0.5*lex.
    Legacy,
}

impl Sp2GridMode {
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            // MuMax SP2 Appendix script-style grid
            "legacy" | "mumax_sp2" | "mumaxsp2" | "appendix" | "appendix_sp2" => Self::Legacy,

            // Generic target-cell + pow2 rounding
            "mumax" | "target" | "target_cell" | "pow2" | "pow2_cell" => Self::Mumax,

            // Default
            _ => Self::Mumax,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sp2GridPolicy {
    /// Grid construction mode.
    pub mode: Sp2GridMode,

    /// Baseline target cell size as a fraction of lex (MuMax references often use ~0.75).
    pub cell_over_lex: f64,

    /// If refining, multiply cell_over_lex by this factor (<1 means finer).
    /// Example: 0.5 halves the cell size.
    pub refine_factor: f64,

    /// Maximum number of one-shot refinements (keep at 1 for now).
    pub max_refinements: usize,

    /// Threshold on RMS nearest-neighbour angle (radians).
    pub nn_angle_rms_threshold: f64,

    /// Threshold on max nearest-neighbour angle (radians).
    pub nn_angle_max_threshold: f64,

    /// If the remanence relax took an unusually large number of accepted steps,
    /// treat as a stiffness signal and allow refinement.
    pub remanence_steps_threshold: usize,
}

impl Default for Sp2GridPolicy {
    fn default() -> Self {
        Self {
            mode: Sp2GridMode::Legacy,
            cell_over_lex: 0.75,
            refine_factor: 0.5,
            max_refinements: 0,
            // Conservative defaults (tune later if needed):
            nn_angle_rms_threshold: 0.18, // ~10.3 deg
            nn_angle_max_threshold: 0.45, // ~25.8 deg
            remanence_steps_threshold: 50_000,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RemanenceMetrics {
    pub mx: f64,
    pub my: f64,
    pub mz: f64,
    pub msum: f64,

    pub nn_angle_rms: f64,
    pub nn_angle_max: f64,
}

// -------------------------
// SP2 geometry + discretisation
// -------------------------

fn ilogb_sp2(x: f64) -> i32 {
    if !x.is_finite() || x <= 1.0 {
        return 0;
    }
    // Small epsilon avoids rare floating-point cases where x is extremely close to a power of two.
    let p = (x.log2() + 1e-12).ceil() as i32;
    p.max(0)
}

/// Build the SP2 grid for a given d/lex and exchange length lex.
///
/// SP2 physical sizes (MuMax reference):
///   sizex = 5 * lex * d
///   sizey = 1 * lex * d
///   sizez = 0.1 * lex * d
pub fn build_sp2_grid(d_lex: usize, lex: f64, policy: &Sp2GridPolicy) -> Grid2D {
    let d = d_lex as f64;

    let sizex = 5.0 * lex * d;
    let sizey = 1.0 * lex * d;
    let sizez = 0.1 * lex * d;

    match policy.mode {
        Sp2GridMode::Legacy => {
            // MuMax SP2 Appendix rule:
            // x = sizex / (5 * 0.5 * lex) = 2d
            // choose p so that 2^p >= x, then nx = 5 * 2^p, ny=nx/5.
            let x = sizex / (5.0 * 0.5 * lex);
            let p = ilogb_sp2(x);
            let nx: usize = (2usize.pow(p as u32)) * 5;
            let ny: usize = nx / 5;

            let dx = sizex / (nx as f64);
            let dy = sizey / (ny as f64);
            let dz = sizez;

            Grid2D::new(nx, ny, dx, dy, dz)
        }
        Sp2GridMode::Mumax => {
            // MuMax-like: target cell size ~ cell_over_lex * lex and use pow2 dimensions.
            let cell = policy.cell_over_lex.max(0.05) * lex;

            let nx_req = (sizex / cell).ceil().max(1.0) as usize;
            let ny_req = (sizey / cell).ceil().max(1.0) as usize;

            let nx = nx_req.next_power_of_two();
            let ny = ny_req.next_power_of_two();

            let dx = sizex / (nx as f64);
            let dy = sizey / (ny as f64);
            let dz = sizez;

            Grid2D::new(nx, ny, dx, dy, dz)
        }
    }
}

// -------------------------
// State metrics
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

    [sx / n.max(1.0), sy / n.max(1.0), sz / n.max(1.0)]
}

/// Nearest-neighbour angular nonuniformity.
///
/// Computes angles between (i,j) and (i+1,j), (i,j) and (i,j+1) (no wrap).
fn nn_angle_stats(m: &VectorField2D) -> (f64, f64) {
    let nx = m.grid.nx;
    let ny = m.grid.ny;

    let mut sum2 = 0.0;
    let mut count = 0.0;
    let mut max_a = 0.0;

    let clamp = |x: f64| x.max(-1.0).min(1.0);

    for j in 0..ny {
        for i in 0..nx {
            let idx = m.grid.idx(i, j);
            let a = m.data[idx];

            if i + 1 < nx {
                let b = m.data[m.grid.idx(i + 1, j)];
                let dot = clamp(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
                let ang = dot.acos();
                sum2 += ang * ang;
                count += 1.0;
                if ang > max_a {
                    max_a = ang;
                }
            }
            if j + 1 < ny {
                let b = m.data[m.grid.idx(i, j + 1)];
                let dot = clamp(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
                let ang = dot.acos();
                sum2 += ang * ang;
                count += 1.0;
                if ang > max_a {
                    max_a = ang;
                }
            }
        }
    }

    let rms = if count > 0.0 { (sum2 / count).sqrt() } else { 0.0 };
    (rms, max_a)
}

pub fn remanence_metrics(m_rem: &VectorField2D) -> RemanenceMetrics {
    let m = avg_m(m_rem);
    let (rms, max_a) = nn_angle_stats(m_rem);

    RemanenceMetrics {
        mx: m[0],
        my: m[1],
        mz: m[2],
        msum: m[0] + m[1] + m[2],
        nn_angle_rms: rms,
        nn_angle_max: max_a,
    }
}

// -------------------------
// One-shot refinement decision
// -------------------------

/// Decide whether to refine once after remanence.
///
/// Returns `Some(new_policy)` if refinement should be performed.
pub fn maybe_refine_after_remanence(
    m_rem: &VectorField2D,
    rem_report: Option<&RelaxReport>,
    policy: &Sp2GridPolicy,
    refinements_done: usize,
) -> Option<Sp2GridPolicy> {
    if refinements_done >= policy.max_refinements {
        return None;
    }

    let met = remanence_metrics(m_rem);

    let nonuniform = met.nn_angle_rms >= policy.nn_angle_rms_threshold
        || met.nn_angle_max >= policy.nn_angle_max_threshold;

    let stiff = rem_report
        .map(|r| r.accepted_steps >= policy.remanence_steps_threshold)
        .unwrap_or(false);

    if nonuniform || stiff {
        let mut p2 = policy.clone();
        p2.cell_over_lex = (p2.cell_over_lex * p2.refine_factor).max(0.05);
        return Some(p2);
    }

    None
}

/// Convenience: resample remanence onto a refined grid.
///
/// This does *not* run Relax again; callers should re-equilibrate after resampling.
pub fn resample_remanence_to_policy_grid(
    d_lex: usize,
    lex: f64,
    policy: &Sp2GridPolicy,
    m_rem: &VectorField2D,
) -> (Grid2D, VectorField2D) {
    let g2 = build_sp2_grid(d_lex, lex, policy);
    let m2 = m_rem.resample_to_grid(g2);
    (g2, m2)
}

/// Default SP2 grid policy.
///
/// The SP2 runner calls this as `default_sp2_grid_policy(lex)`.
/// For now it returns `Sp2GridPolicy::default()`, but keeping it as a named helper
/// lets us add env overrides later without touching the SP2 solver logic.
pub fn default_sp2_grid_policy(_lex: f64) -> Sp2GridPolicy {
    Sp2GridPolicy::default()
}