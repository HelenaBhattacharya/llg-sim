// src/llg.rs

use crate::vector_field::VectorField2D;

/// Parameters for the LLG equation (very simple version).
pub struct LLGParams {
    pub gamma: f64,       // gyromagnetic ratio
    pub alpha: f64,       // damping
    pub dt: f64,          // time step
    pub h_ext: [f64; 3],  // uniform external field (Tesla, in effective units)
}

// --- small helper functions for vector math ---

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n2 = dot(v, v);
    if n2 == 0.0 {
        return [0.0, 0.0, 1.0];
    }
    let inv = 1.0 / n2.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Advance the magnetisation by one explicit Euler step of the LLG equation,
/// assuming only a uniform external field H_ext (no exchange/demag yet).
pub fn step_llg(m: &mut VectorField2D, params: &LLGParams) {
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt = params.dt;
    let h = params.h_ext;

    // loop over all cells
    for cell in &mut m.data {
        // current magnetisation at this cell
        let m_vec = *cell;

        // Landau–Lifshitz–Gilbert (LLG) RHS:
        // dm/dt = -gamma m × H  +  alpha m × (m × H)
        let m_cross_h = cross(m_vec, h);
        let m_cross_m_cross_h = cross(m_vec, m_cross_h);

        let dmdt = [
            -gamma * m_cross_h[0] + alpha * m_cross_m_cross_h[0],
            -gamma * m_cross_h[1] + alpha * m_cross_m_cross_h[1],
            -gamma * m_cross_h[2] + alpha * m_cross_m_cross_h[2],
        ];

        // explicit Euler update
        let m_new = [
            m_vec[0] + dt * dmdt[0],
            m_vec[1] + dt * dmdt[1],
            m_vec[2] + dt * dmdt[2],
        ];

        // renormalize to |m| = 1 (simple but good enough for now)
        *cell = normalize(m_new);
    }
}