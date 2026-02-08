// src/llg.rs

use crate::effective_field::{FieldMask, build_h_eff_masked, zeeman::add_zeeman_field};
use crate::grid::Grid2D;
use crate::params::{LLGParams, Material};
use crate::vec3::{cross, normalize};
use crate::vector_field::VectorField2D;

/// Landau–Lifshitz form equivalent to Gilbert (for unit magnetisation m):
///
///   dm/dt = -(gamma/(1+alpha^2)) [ m×B + alpha m×(m×B) ]
///
/// where:
/// - gamma is |gamma_e| in rad/(s*T)
/// - alpha is dimensionless damping
/// - B is the effective induction in Tesla
#[inline]
fn llg_rhs(m: [f64; 3], b: [f64; 3], gamma: f64, alpha: f64) -> [f64; 3] {
    let denom = 1.0 + alpha * alpha;
    let pref = -gamma / denom;

    let m_cross_b = cross(m, b);
    let m_cross_m_cross_b = cross(m, m_cross_b);

    [
        pref * (m_cross_b[0] + alpha * m_cross_m_cross_b[0]),
        pref * (m_cross_b[1] + alpha * m_cross_m_cross_b[1]),
        pref * (m_cross_b[2] + alpha * m_cross_m_cross_b[2]),
    ]
}

/// Damping-only (precession-suppressed) RHS used for energy relaxation.
///
///   dm/dt = -(gamma*alpha/(1+alpha^2)) [ m × (m × B) ]
#[inline]
fn llg_rhs_relax(m: [f64; 3], b: [f64; 3], gamma: f64, alpha: f64) -> [f64; 3] {
    let denom = 1.0 + alpha * alpha;
    let pref = -gamma * alpha / denom;

    let m_cross_b = cross(m, b);
    let m_cross_m_cross_b = cross(m, m_cross_b);

    [
        pref * m_cross_m_cross_b[0],
        pref * m_cross_m_cross_b[1],
        pref * m_cross_m_cross_b[2],
    ]
}

#[inline]
fn add_scaled(a: [f64; 3], s: f64, b: [f64; 3]) -> [f64; 3] {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}

#[inline]
fn combo_rk4(k1: [f64; 3], k2: [f64; 3], k3: [f64; 3], k4: [f64; 3]) -> [f64; 3] {
    [
        (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    ]
}

/// Reusable scratch buffers for RK4 where B_eff is recomputed at substeps.
///
/// This avoids allocating large temporary arrays every timestep.
pub struct RK4Scratch {
    m1: VectorField2D,
    m2: VectorField2D,
    m3: VectorField2D,
    b1: VectorField2D,
    b2: VectorField2D,
    b3: VectorField2D,
    b4: VectorField2D,
    k1: Vec<[f64; 3]>,
    k2: Vec<[f64; 3]>,
    k3: Vec<[f64; 3]>,
    k4: Vec<[f64; 3]>,
}

impl RK4Scratch {
    pub fn new(grid: Grid2D) -> Self {
        let n = grid.n_cells();
        Self {
            m1: VectorField2D::new(grid),
            m2: VectorField2D::new(grid),
            m3: VectorField2D::new(grid),
            b1: VectorField2D::new(grid),
            b2: VectorField2D::new(grid),
            b3: VectorField2D::new(grid),
            b4: VectorField2D::new(grid),
            k1: vec![[0.0; 3]; n],
            k2: vec![[0.0; 3]; n],
            k3: vec![[0.0; 3]; n],
            k4: vec![[0.0; 3]; n],
        }
    }
}

/// Reusable scratch buffers for Dormand–Prince RK45 (5(4)) with B_eff recomputed at each substage.
///
/// We store stage magnetisations (m2..m7), stage fields (b1..b7), and stage torques (k1..k7).
pub struct RK45Scratch {
    m2: VectorField2D,
    m3: VectorField2D,
    m4: VectorField2D,
    m5: VectorField2D,
    m6: VectorField2D,
    m7: VectorField2D,

    b1: VectorField2D,
    b2: VectorField2D,
    b3: VectorField2D,
    b4: VectorField2D,
    b5: VectorField2D,
    b6: VectorField2D,
    b7: VectorField2D,

    k1: Vec<[f64; 3]>,
    k2: Vec<[f64; 3]>,
    k3: Vec<[f64; 3]>,
    k4: Vec<[f64; 3]>,
    k5: Vec<[f64; 3]>,
    k6: Vec<[f64; 3]>,
    k7: Vec<[f64; 3]>,
}

impl RK45Scratch {
    pub fn new(grid: Grid2D) -> Self {
        let n = grid.n_cells();
        Self {
            m2: VectorField2D::new(grid),
            m3: VectorField2D::new(grid),
            m4: VectorField2D::new(grid),
            m5: VectorField2D::new(grid),
            m6: VectorField2D::new(grid),
            m7: VectorField2D::new(grid),

            b1: VectorField2D::new(grid),
            b2: VectorField2D::new(grid),
            b3: VectorField2D::new(grid),
            b4: VectorField2D::new(grid),
            b5: VectorField2D::new(grid),
            b6: VectorField2D::new(grid),
            b7: VectorField2D::new(grid),

            k1: vec![[0.0; 3]; n],
            k2: vec![[0.0; 3]; n],
            k3: vec![[0.0; 3]; n],
            k4: vec![[0.0; 3]; n],
            k5: vec![[0.0; 3]; n],
            k6: vec![[0.0; 3]; n],
            k7: vec![[0.0; 3]; n],
        }
    }
}

/// Reusable scratch buffers for Bogacki–Shampine RK23 (3(2)) in relax-mode,
/// with B_eff recomputed at each substage.
///
/// This is the MuMax-style choice for Relax(): adaptive 3rd-order with an
/// embedded 2nd-order estimate for error control.
pub struct RK23Scratch {
    // Stage magnetisations
    m2: VectorField2D,
    m3: VectorField2D,
    m4: VectorField2D, // 3rd-order solution (accepted state)

    // Stage fields
    b1: VectorField2D,
    b2: VectorField2D,
    b3: VectorField2D,
    b4: VectorField2D,

    // Stage torques
    k1: Vec<[f64; 3]>,
    k2: Vec<[f64; 3]>,
    k3: Vec<[f64; 3]>,
    k4: Vec<[f64; 3]>,

    // Whether `b4` currently corresponds to the accepted state stored in `m`.
    // This lets higher-level routines reuse the last computed field for torque checks.
    last_b_eff_valid: bool,
}

impl RK23Scratch {
    pub fn new(grid: Grid2D) -> Self {
        let n = grid.n_cells();
        Self {
            m2: VectorField2D::new(grid),
            m3: VectorField2D::new(grid),
            m4: VectorField2D::new(grid),

            b1: VectorField2D::new(grid),
            b2: VectorField2D::new(grid),
            b3: VectorField2D::new(grid),
            b4: VectorField2D::new(grid),

            k1: vec![[0.0; 3]; n],
            k2: vec![[0.0; 3]; n],
            k3: vec![[0.0; 3]; n],
            k4: vec![[0.0; 3]; n],
            last_b_eff_valid: false,
        }
    }

    /// Returns the last computed effective field for the accepted state, if available.
    ///
    /// When a relax RK23 step is accepted, `b4` is the effective field evaluated at the
    /// accepted 3rd-order solution. We mark it valid so callers can avoid rebuilding the
    /// effective field just to compute torque metrics.
    pub fn last_b_eff(&self) -> Option<&VectorField2D> {
        if self.last_b_eff_valid {
            Some(&self.b4)
        } else {
            None
        }
    }
}

/// Adaptive Bogacki–Shampine RK23 (3(2)) step in relax mode (precession suppressed),
/// with recompute-field at each substage and a FieldMask.
///
/// Error estimate is based on the difference between 3rd- and 2nd-order states.
/// Controller: dt_next = dt * headroom * (max_err/eps)^(1/3), clamped.
///
/// Returns: (eps, accepted, dt_used).
pub fn step_llg_rk23_recompute_field_masked_relax_adaptive(
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK23Scratch,
    mask: FieldMask,
    max_err: f64,
    headroom: f64,
    dt_min: f64,
    dt_max: f64,
) -> (f64, bool, f64) {
    let grid = &m.grid;
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt0 = params.dt;
    let n = m.data.len();

    // Bogacki–Shampine coefficients
    const A21: f64 = 1.0 / 2.0;

    const A32: f64 = 3.0 / 4.0;

    // 3rd order solution weights
    const B1: f64 = 2.0 / 9.0;
    const B2: f64 = 1.0 / 3.0;
    const B3: f64 = 4.0 / 9.0;

    // 2nd order embedded weights (uses k4)
    const BH1: f64 = 7.0 / 24.0;
    const BH2: f64 = 1.0 / 4.0;
    const BH3: f64 = 1.0 / 3.0;
    const BH4: f64 = 1.0 / 8.0;

    // -----------------------
    // Stage 1: k1 at m
    // -----------------------
    build_h_eff_masked(grid, m, &mut scratch.b1, params, material, mask);
    for i in 0..n {
        scratch.k1[i] = llg_rhs_relax(m.data[i], scratch.b1.data[i], gamma, alpha);
    }

    // -----------------------
    // Stage 2: m2 = m + dt*(1/2)*k1
    // -----------------------
    for i in 0..n {
        let v = add_scaled(m.data[i], dt0 * A21, scratch.k1[i]);
        scratch.m2.data[i] = normalize(v);
    }
    build_h_eff_masked(grid, &scratch.m2, &mut scratch.b2, params, material, mask);
    for i in 0..n {
        scratch.k2[i] = llg_rhs_relax(scratch.m2.data[i], scratch.b2.data[i], gamma, alpha);
    }

    // -----------------------
    // Stage 3: m3 = m + dt*(3/4)*k2
    // -----------------------
    for i in 0..n {
        let v = add_scaled(m.data[i], dt0 * A32, scratch.k2[i]);
        scratch.m3.data[i] = normalize(v);
    }
    build_h_eff_masked(grid, &scratch.m3, &mut scratch.b3, params, material, mask);
    for i in 0..n {
        scratch.k3[i] = llg_rhs_relax(scratch.m3.data[i], scratch.b3.data[i], gamma, alpha);
    }

    // -----------------------
    // 3rd-order solution: m4 = m + dt*(B1*k1 + B2*k2 + B3*k3)
    // -----------------------
    for i in 0..n {
        let mut v = m.data[i];
        v = add_scaled(v, dt0 * B1, scratch.k1[i]);
        v = add_scaled(v, dt0 * B2, scratch.k2[i]);
        v = add_scaled(v, dt0 * B3, scratch.k3[i]);
        scratch.m4.data[i] = normalize(v);
    }

    // Stage 4 derivative at the 3rd-order solution
    build_h_eff_masked(grid, &scratch.m4, &mut scratch.b4, params, material, mask);
    for i in 0..n {
        scratch.k4[i] = llg_rhs_relax(scratch.m4.data[i], scratch.b4.data[i], gamma, alpha);
    }

    // -----------------------
    // 2nd-order embedded solution (reuse m2 buffer)
    // m2 = m + dt*(BH1*k1 + BH2*k2 + BH3*k3 + BH4*k4)
    // -----------------------
    for i in 0..n {
        let mut v = m.data[i];
        v = add_scaled(v, dt0 * BH1, scratch.k1[i]);
        v = add_scaled(v, dt0 * BH2, scratch.k2[i]);
        v = add_scaled(v, dt0 * BH3, scratch.k3[i]);
        v = add_scaled(v, dt0 * BH4, scratch.k4[i]);
        scratch.m2.data[i] = normalize(v);
    }

    // -----------------------
    // Error estimate: eps = max ||m3rd - m2nd||_inf
    // -----------------------
    let mut err_inf = 0.0_f64;
    for i in 0..n {
        let a = scratch.m4.data[i]; // 3rd order
        let b = scratch.m2.data[i]; // 2nd order
        let d0 = (a[0] - b[0]).abs();
        let d1 = (a[1] - b[1]).abs();
        let d2 = (a[2] - b[2]).abs();
        let d = d0.max(d1).max(d2);
        if d > err_inf {
            err_inf = d;
        }
    }
    let eps = err_inf;

    // Next dt (order=3 controller)
    let mut dt_next = if eps > 0.0 {
        dt0 * headroom * (max_err / eps).powf(1.0 / 3.0)
    } else {
        dt0 * 2.0
    };
    dt_next = dt_next.clamp(dt_min, dt_max);

    // Accept / reject (guarantee progress if at dt_min)
    let accept = (eps <= max_err) || (dt0 <= dt_min * 1.0000000001);

    if accept {
        // Accept 3rd-order solution
        m.data.clone_from(&scratch.m4.data);
        // `b4` is the field evaluated at `m4`, which is now the accepted state in `m`.
        scratch.last_b_eff_valid = true;
        params.dt = dt_next;
        (eps, true, dt0)
    } else {
        // Reject: keep m unchanged, just shrink dt
        // `b4` corresponds to a trial state, so it is not valid for the current `m`.
        scratch.last_b_eff_valid = false;
        params.dt = dt_next;
        (eps, false, dt0)
    }
}

/// Advance m by one step using explicit Euler, given precomputed B_eff (Tesla).
pub fn step_llg_with_field(m: &mut VectorField2D, b_eff: &VectorField2D, params: &LLGParams) {
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt = params.dt;

    for (cell_idx, cell) in m.data.iter_mut().enumerate() {
        let m0 = *cell;
        let b = b_eff.data[cell_idx];

        let dmdt = llg_rhs(m0, b, gamma, alpha);

        let m_new = [
            m0[0] + dt * dmdt[0],
            m0[1] + dt * dmdt[1],
            m0[2] + dt * dmdt[2],
        ];

        *cell = normalize(m_new);
    }
}

/// Advance m by one step using fixed-step RK4 with a *frozen* B_eff for the whole step.
pub fn step_llg_with_field_rk4(m: &mut VectorField2D, b_eff: &VectorField2D, params: &LLGParams) {
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt = params.dt;

    for (cell_idx, cell) in m.data.iter_mut().enumerate() {
        let m0 = *cell;
        let b = b_eff.data[cell_idx];

        let k1 = llg_rhs(m0, b, gamma, alpha);
        let m1 = normalize(add_scaled(m0, 0.5 * dt, k1));

        let k2 = llg_rhs(m1, b, gamma, alpha);
        let m2 = normalize(add_scaled(m0, 0.5 * dt, k2));

        let k3 = llg_rhs(m2, b, gamma, alpha);
        let m3 = normalize(add_scaled(m0, dt, k3));

        let k4 = llg_rhs(m3, b, gamma, alpha);

        let incr = combo_rk4(k1, k2, k3, k4);
        let m_new = normalize(add_scaled(m0, dt, incr));

        *cell = m_new;
    }
}

/// Fixed-step RK4 where the effective field B_eff(m) is recomputed at each RK substage.
pub fn step_llg_rk4_recompute_field_masked(
    m: &mut VectorField2D,
    params: &LLGParams,
    material: &Material,
    scratch: &mut RK4Scratch,
    mask: FieldMask,
) {
    let grid = &m.grid;
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt = params.dt;
    let n = m.data.len();

    // Stage 1
    build_h_eff_masked(grid, m, &mut scratch.b1, params, material, mask);
    for i in 0..n {
        scratch.k1[i] = llg_rhs(m.data[i], scratch.b1.data[i], gamma, alpha);
    }

    // Stage 2
    for i in 0..n {
        let m1 = add_scaled(m.data[i], 0.5 * dt, scratch.k1[i]);
        scratch.m1.data[i] = normalize(m1);
    }
    build_h_eff_masked(grid, &scratch.m1, &mut scratch.b2, params, material, mask);
    for i in 0..n {
        scratch.k2[i] = llg_rhs(scratch.m1.data[i], scratch.b2.data[i], gamma, alpha);
    }

    // Stage 3
    for i in 0..n {
        let m2 = add_scaled(m.data[i], 0.5 * dt, scratch.k2[i]);
        scratch.m2.data[i] = normalize(m2);
    }
    build_h_eff_masked(grid, &scratch.m2, &mut scratch.b3, params, material, mask);
    for i in 0..n {
        scratch.k3[i] = llg_rhs(scratch.m2.data[i], scratch.b3.data[i], gamma, alpha);
    }

    // Stage 4
    for i in 0..n {
        let m3 = add_scaled(m.data[i], dt, scratch.k3[i]);
        scratch.m3.data[i] = normalize(m3);
    }
    build_h_eff_masked(grid, &scratch.m3, &mut scratch.b4, params, material, mask);
    for i in 0..n {
        scratch.k4[i] = llg_rhs(scratch.m3.data[i], scratch.b4.data[i], gamma, alpha);
    }

    // Combine
    for i in 0..n {
        let incr = combo_rk4(scratch.k1[i], scratch.k2[i], scratch.k3[i], scratch.k4[i]);
        let m_new = add_scaled(m.data[i], dt, incr);
        m.data[i] = normalize(m_new);
    }
}

/// Adaptive Dormand–Prince RK45 (5(4)) with recompute-field at each substage.
///
/// MuMax-like controller:
/// - eps = dt * max |tau_high - tau_low|
/// - dt_next = dt * headroom * (max_err/eps)^(1/5), clamped to [dt_min, dt_max]
///
/// Returns: (eps, accepted, dt_used).
pub fn step_llg_rk45_recompute_field_masked_adaptive(
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK45Scratch,
    mask: FieldMask,
    max_err: f64,
    headroom: f64,
    dt_min: f64,
    dt_max: f64,
) -> (f64, bool, f64) {
    let grid = &m.grid;
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt0 = params.dt;
    let n = m.data.len();

    // Dormand–Prince (5(4)) coefficients
    const A21: f64 = 1.0 / 5.0;

    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;

    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;

    const A51: f64 = 19372.0 / 6561.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A54: f64 = -212.0 / 729.0;

    const A61: f64 = 9017.0 / 3168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5103.0 / 18656.0;

    // 5th order weights
    const B1: f64 = 35.0 / 384.0;
    const B3: f64 = 500.0 / 1113.0;
    const B4: f64 = 125.0 / 192.0;
    const B5: f64 = -2187.0 / 6784.0;
    const B6: f64 = 11.0 / 84.0;

    // 4th order embedded weights
    const BH1: f64 = 5179.0 / 57600.0;
    const BH3: f64 = 7571.0 / 16695.0;
    const BH4: f64 = 393.0 / 640.0;
    const BH5: f64 = -92097.0 / 339200.0;
    const BH6: f64 = 187.0 / 2100.0;
    const BH7: f64 = 1.0 / 40.0;

    // Stage 1: k1 at m
    build_h_eff_masked(grid, m, &mut scratch.b1, params, material, mask);
    for i in 0..n {
        scratch.k1[i] = llg_rhs(m.data[i], scratch.b1.data[i], gamma, alpha);
    }

    // Stage 2: m2 = m + dt*a21*k1
    for i in 0..n {
        let v = add_scaled(m.data[i], dt0 * A21, scratch.k1[i]);
        scratch.m2.data[i] = normalize(v);
    }
    build_h_eff_masked(grid, &scratch.m2, &mut scratch.b2, params, material, mask);
    for i in 0..n {
        scratch.k2[i] = llg_rhs(scratch.m2.data[i], scratch.b2.data[i], gamma, alpha);
    }

    // Stage 3
    for i in 0..n {
        let mut v = m.data[i];
        v = add_scaled(v, dt0 * A31, scratch.k1[i]);
        v = add_scaled(v, dt0 * A32, scratch.k2[i]);
        scratch.m3.data[i] = normalize(v);
    }
    build_h_eff_masked(grid, &scratch.m3, &mut scratch.b3, params, material, mask);
    for i in 0..n {
        scratch.k3[i] = llg_rhs(scratch.m3.data[i], scratch.b3.data[i], gamma, alpha);
    }

    // Stage 4
    for i in 0..n {
        let mut v = m.data[i];
        v = add_scaled(v, dt0 * A41, scratch.k1[i]);
        v = add_scaled(v, dt0 * A42, scratch.k2[i]);
        v = add_scaled(v, dt0 * A43, scratch.k3[i]);
        scratch.m4.data[i] = normalize(v);
    }
    build_h_eff_masked(grid, &scratch.m4, &mut scratch.b4, params, material, mask);
    for i in 0..n {
        scratch.k4[i] = llg_rhs(scratch.m4.data[i], scratch.b4.data[i], gamma, alpha);
    }

    // Stage 5
    for i in 0..n {
        let mut v = m.data[i];
        v = add_scaled(v, dt0 * A51, scratch.k1[i]);
        v = add_scaled(v, dt0 * A52, scratch.k2[i]);
        v = add_scaled(v, dt0 * A53, scratch.k3[i]);
        v = add_scaled(v, dt0 * A54, scratch.k4[i]);
        scratch.m5.data[i] = normalize(v);
    }
    build_h_eff_masked(grid, &scratch.m5, &mut scratch.b5, params, material, mask);
    for i in 0..n {
        scratch.k5[i] = llg_rhs(scratch.m5.data[i], scratch.b5.data[i], gamma, alpha);
    }

    // Stage 6
    for i in 0..n {
        let mut v = m.data[i];
        v = add_scaled(v, dt0 * A61, scratch.k1[i]);
        v = add_scaled(v, dt0 * A62, scratch.k2[i]);
        v = add_scaled(v, dt0 * A63, scratch.k3[i]);
        v = add_scaled(v, dt0 * A64, scratch.k4[i]);
        v = add_scaled(v, dt0 * A65, scratch.k5[i]);
        scratch.m6.data[i] = normalize(v);
    }
    build_h_eff_masked(grid, &scratch.m6, &mut scratch.b6, params, material, mask);
    for i in 0..n {
        scratch.k6[i] = llg_rhs(scratch.m6.data[i], scratch.b6.data[i], gamma, alpha);
    }

    // Stage 7 state (5th order solution)
    for i in 0..n {
        let mut v = m.data[i];
        v = add_scaled(v, dt0 * B1, scratch.k1[i]);
        v = add_scaled(v, dt0 * B3, scratch.k3[i]);
        v = add_scaled(v, dt0 * B4, scratch.k4[i]);
        v = add_scaled(v, dt0 * B5, scratch.k5[i]);
        v = add_scaled(v, dt0 * B6, scratch.k6[i]);
        scratch.m7.data[i] = normalize(v);
    }
    build_h_eff_masked(grid, &scratch.m7, &mut scratch.b7, params, material, mask);
    for i in 0..n {
        scratch.k7[i] = llg_rhs(scratch.m7.data[i], scratch.b7.data[i], gamma, alpha);
    }

    // Error estimate: eps = dt * max |tau_high - tau_low|
    let mut err_inf: f64 = 0.0;
    for i in 0..n {
        // high order torque (5th)
        let th0 = B1 * scratch.k1[i][0]
            + B3 * scratch.k3[i][0]
            + B4 * scratch.k4[i][0]
            + B5 * scratch.k5[i][0]
            + B6 * scratch.k6[i][0];
        let th1 = B1 * scratch.k1[i][1]
            + B3 * scratch.k3[i][1]
            + B4 * scratch.k4[i][1]
            + B5 * scratch.k5[i][1]
            + B6 * scratch.k6[i][1];
        let th2 = B1 * scratch.k1[i][2]
            + B3 * scratch.k3[i][2]
            + B4 * scratch.k4[i][2]
            + B5 * scratch.k5[i][2]
            + B6 * scratch.k6[i][2];

        // low order torque (4th)
        let tl0 = BH1 * scratch.k1[i][0]
            + BH3 * scratch.k3[i][0]
            + BH4 * scratch.k4[i][0]
            + BH5 * scratch.k5[i][0]
            + BH6 * scratch.k6[i][0]
            + BH7 * scratch.k7[i][0];
        let tl1 = BH1 * scratch.k1[i][1]
            + BH3 * scratch.k3[i][1]
            + BH4 * scratch.k4[i][1]
            + BH5 * scratch.k5[i][1]
            + BH6 * scratch.k6[i][1]
            + BH7 * scratch.k7[i][1];
        let tl2 = BH1 * scratch.k1[i][2]
            + BH3 * scratch.k3[i][2]
            + BH4 * scratch.k4[i][2]
            + BH5 * scratch.k5[i][2]
            + BH6 * scratch.k6[i][2]
            + BH7 * scratch.k7[i][2];

        let d = (th0 - tl0)
            .abs()
            .max((th1 - tl1).abs())
            .max((th2 - tl2).abs());
        if d > err_inf {
            err_inf = d;
        }
    }

    let eps = err_inf * dt0;

    // Next dt (MuMax-like)
    let mut dt_next = if eps > 0.0 {
        dt0 * headroom * (max_err / eps).powf(0.2)
    } else {
        dt0 * 2.0
    };
    dt_next = dt_next.clamp(dt_min, dt_max);

    // Accept / reject (guarantee progress if at dt_min)
    let accept = (eps <= max_err) || (dt0 <= dt_min * 1.0000000001);

    if accept {
        m.data.clone_from(&scratch.m7.data);
        params.dt = dt_next;
        (eps, true, dt0)
    } else {
        params.dt = dt_next;
        (eps, false, dt0)
    }
}

/// Backwards-compatible: adaptive RK45 recompute-field using full physics mask.
pub fn step_llg_rk45_recompute_field_adaptive(
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK45Scratch,
    max_err: f64,
    headroom: f64,
    dt_min: f64,
    dt_max: f64,
) -> (f64, bool, f64) {
    step_llg_rk45_recompute_field_masked_adaptive(
        m,
        params,
        material,
        scratch,
        FieldMask::Full,
        max_err,
        headroom,
        dt_min,
        dt_max,
    )
}

/// Fixed-step RK4 recompute-field, but using damping-only (precession suppressed) dynamics.
pub fn step_llg_rk4_recompute_field_masked_relax(
    m: &mut VectorField2D,
    params: &LLGParams,
    material: &Material,
    scratch: &mut RK4Scratch,
    mask: FieldMask,
) {
    let grid = &m.grid;
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt = params.dt;
    let n = m.data.len();

    // Stage 1
    build_h_eff_masked(grid, m, &mut scratch.b1, params, material, mask);
    for i in 0..n {
        scratch.k1[i] = llg_rhs_relax(m.data[i], scratch.b1.data[i], gamma, alpha);
    }

    // Stage 2
    for i in 0..n {
        let m1 = add_scaled(m.data[i], 0.5 * dt, scratch.k1[i]);
        scratch.m1.data[i] = normalize(m1);
    }
    build_h_eff_masked(grid, &scratch.m1, &mut scratch.b2, params, material, mask);
    for i in 0..n {
        scratch.k2[i] = llg_rhs_relax(scratch.m1.data[i], scratch.b2.data[i], gamma, alpha);
    }

    // Stage 3
    for i in 0..n {
        let m2 = add_scaled(m.data[i], 0.5 * dt, scratch.k2[i]);
        scratch.m2.data[i] = normalize(m2);
    }
    build_h_eff_masked(grid, &scratch.m2, &mut scratch.b3, params, material, mask);
    for i in 0..n {
        scratch.k3[i] = llg_rhs_relax(scratch.m2.data[i], scratch.b3.data[i], gamma, alpha);
    }

    // Stage 4
    for i in 0..n {
        let m3 = add_scaled(m.data[i], dt, scratch.k3[i]);
        scratch.m3.data[i] = normalize(m3);
    }
    build_h_eff_masked(grid, &scratch.m3, &mut scratch.b4, params, material, mask);
    for i in 0..n {
        scratch.k4[i] = llg_rhs_relax(scratch.m3.data[i], scratch.b4.data[i], gamma, alpha);
    }

    // Combine
    for i in 0..n {
        let incr = combo_rk4(scratch.k1[i], scratch.k2[i], scratch.k3[i], scratch.k4[i]);
        let m_new = add_scaled(m.data[i], dt, incr);
        m.data[i] = normalize(m_new);
    }
}

/// Adaptive timestep wrapper for the relax-mode masked RK4 stepper (step-doubling).
pub fn step_llg_rk4_recompute_field_masked_relax_adaptive(
    m: &mut VectorField2D,
    params: &mut LLGParams,
    material: &Material,
    scratch: &mut RK4Scratch,
    mask: FieldMask,
    tol: f64,
    dt_min: f64,
    dt_max: f64,
) -> (f64, bool) {
    let dt0 = params.dt;

    let mut m_big = VectorField2D::new(m.grid);
    let mut m_small = VectorField2D::new(m.grid);
    m_big.data.clone_from(&m.data);
    m_small.data.clone_from(&m.data);

    // big step
    params.dt = dt0;
    step_llg_rk4_recompute_field_masked_relax(&mut m_big, params, material, scratch, mask);

    // two half steps
    params.dt = 0.5 * dt0;
    step_llg_rk4_recompute_field_masked_relax(&mut m_small, params, material, scratch, mask);
    step_llg_rk4_recompute_field_masked_relax(&mut m_small, params, material, scratch, mask);

    params.dt = dt0;

    let mut err: f64 = 0.0;
    for (vb, vs) in m_big.data.iter().zip(m_small.data.iter()) {
        err = err.max((vs[0] - vb[0]).abs());
        err = err.max((vs[1] - vb[1]).abs());
        err = err.max((vs[2] - vb[2]).abs());
    }

    if err <= tol {
        m.data.clone_from(&m_small.data);

        let safety = 0.9;
        let grow = if err == 0.0 {
            2.0
        } else {
            (tol / err).powf(0.2)
        };
        let dt_new = (dt0 * safety * grow).min(dt_max);
        params.dt = dt_new.max(dt_min);

        (err, true)
    } else {
        let safety = 0.9;
        let shrink = (tol / err).powf(0.2);
        let dt_new = (dt0 * safety * shrink).max(dt_min);
        params.dt = dt_new;

        (err, false)
    }
}

/// Backwards-compatible: recompute-field RK4 using the full field builder.
pub fn step_llg_rk4_recompute_field(
    m: &mut VectorField2D,
    params: &LLGParams,
    material: &Material,
    scratch: &mut RK4Scratch,
) {
    step_llg_rk4_recompute_field_masked(m, params, material, scratch, FieldMask::Full);
}

/// Uniform-field wrapper (Euler).
pub fn step_llg(m: &mut VectorField2D, params: &LLGParams) {
    let grid = m.grid;
    let mut b_eff = VectorField2D::new(grid);
    add_zeeman_field(&mut b_eff, params.b_ext);
    step_llg_with_field(m, &b_eff, params);
}

/// Uniform-field wrapper (RK4, frozen field).
pub fn step_llg_rk4(m: &mut VectorField2D, params: &LLGParams) {
    let grid = m.grid;
    let mut b_eff = VectorField2D::new(grid);
    add_zeeman_field(&mut b_eff, params.b_ext);
    step_llg_with_field_rk4(m, &b_eff, params);
}
