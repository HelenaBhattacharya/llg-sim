// src/llg.rs

use crate::effective_field::{build_h_eff, zeeman::add_zeeman_field};
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

#[inline]
fn add_scaled(a: [f64; 3], s: f64, b: [f64; 3]) -> [f64; 3] {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
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

#[inline]
fn combo_rk4(k1: [f64; 3], k2: [f64; 3], k3: [f64; 3], k4: [f64; 3]) -> [f64; 3] {
    [
        (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    ]
}

/// Advance m by one step using explicit Euler, given precomputed B_eff (Tesla).
///
/// This is kept for backwards compatibility and for quick experiments.
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
///
/// Notes:
/// - This is exact for Zeeman-only macrospin where B_eff is constant in time and independent of m.
/// - For field terms that depend on m (exchange/anisotropy), use `step_llg_rk4_recompute_field`.
pub fn step_llg_with_field_rk4(m: &mut VectorField2D, b_eff: &VectorField2D, params: &LLGParams) {
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt = params.dt;

    for (cell_idx, cell) in m.data.iter_mut().enumerate() {
        let m0 = *cell;
        let b = b_eff.data[cell_idx];

        let k1 = llg_rhs(m0, b, gamma, alpha);
        let m1 = add_scaled(m0, 0.5 * dt, k1);

        let k2 = llg_rhs(m1, b, gamma, alpha);
        let m2 = add_scaled(m0, 0.5 * dt, k2);

        let k3 = llg_rhs(m2, b, gamma, alpha);
        let m3 = add_scaled(m0, dt, k3);

        let k4 = llg_rhs(m3, b, gamma, alpha);

        let m_new = [
            m0[0] + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            m0[1] + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            m0[2] + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        ];

        *cell = normalize(m_new);
    }
}

/// Fixed-step RK4 where the effective field B_eff(m) is recomputed at each RK substage.
///
/// This is the appropriate integrator when B_eff depends on m (exchange, anisotropy, etc.).
///
/// Usage pattern:
/// - Create one `RK4Scratch` once (e.g. in `main.rs`) and reuse it every step.
/// - Call this function instead of Euler.
pub fn step_llg_rk4_recompute_field(
    m: &mut VectorField2D,
    params: &LLGParams,
    material: &Material,
    scratch: &mut RK4Scratch,
) {
    let grid = &m.grid;
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt = params.dt;
    let n = m.data.len();

    // Stage 1: k1 from m
    build_h_eff(grid, m, &mut scratch.b1, params, material);
    for i in 0..n {
        scratch.k1[i] = llg_rhs(m.data[i], scratch.b1.data[i], gamma, alpha);
    }

    // Stage 2: m1 = normalize(m + dt/2*k1), then k2
    for i in 0..n {
        let m1 = add_scaled(m.data[i], 0.5 * dt, scratch.k1[i]);
        scratch.m1.data[i] = normalize(m1);
    }
    build_h_eff(grid, &scratch.m1, &mut scratch.b2, params, material);
    for i in 0..n {
        scratch.k2[i] = llg_rhs(scratch.m1.data[i], scratch.b2.data[i], gamma, alpha);
    }

    // Stage 3: m2 = normalize(m + dt/2*k2), then k3
    for i in 0..n {
        let m2 = add_scaled(m.data[i], 0.5 * dt, scratch.k2[i]);
        scratch.m2.data[i] = normalize(m2);
    }
    build_h_eff(grid, &scratch.m2, &mut scratch.b3, params, material);
    for i in 0..n {
        scratch.k3[i] = llg_rhs(scratch.m2.data[i], scratch.b3.data[i], gamma, alpha);
    }

    // Stage 4: m3 = normalize(m + dt*k3), then k4
    for i in 0..n {
        let m3 = add_scaled(m.data[i], dt, scratch.k3[i]);
        scratch.m3.data[i] = normalize(m3);
    }
    build_h_eff(grid, &scratch.m3, &mut scratch.b4, params, material);
    for i in 0..n {
        scratch.k4[i] = llg_rhs(scratch.m3.data[i], scratch.b4.data[i], gamma, alpha);
    }

    // Combine stages
    for i in 0..n {
        let incr = combo_rk4(scratch.k1[i], scratch.k2[i], scratch.k3[i], scratch.k4[i]);
        let m_new = add_scaled(m.data[i], dt, incr);
        m.data[i] = normalize(m_new);
    }
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