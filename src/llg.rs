// src/llg.rs

use crate::effective_field::zeeman::add_zeeman_field;
use crate::params::LLGParams;
use crate::vec3::{cross, normalize};
use crate::vector_field::VectorField2D;

/// Advance m by one step, given precomputed B_eff (Tesla).
///
/// Uses explicit Landau–Lifshitz form equivalent to Gilbert:
///   dm/dt = -(gamma/(1+alpha^2)) [ m×B + alpha m×(m×B) ]
pub fn step_llg_with_field(m: &mut VectorField2D, b_eff: &VectorField2D, params: &LLGParams) {
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt = params.dt;

    let denom = 1.0 + alpha * alpha;
    let pref = -gamma / denom;

    for (cell_idx, cell) in m.data.iter_mut().enumerate() {
        let m_vec = *cell;
        let b = b_eff.data[cell_idx];

        let m_cross_b = cross(m_vec, b);
        let m_cross_m_cross_b = cross(m_vec, m_cross_b);

        let dmdt = [
            pref * (m_cross_b[0] + alpha * m_cross_m_cross_b[0]),
            pref * (m_cross_b[1] + alpha * m_cross_m_cross_b[1]),
            pref * (m_cross_b[2] + alpha * m_cross_m_cross_b[2]),
        ];

        let m_new = [
            m_vec[0] + dt * dmdt[0],
            m_vec[1] + dt * dmdt[1],
            m_vec[2] + dt * dmdt[2],
        ];

        *cell = normalize(m_new);
    }
}

/// Uniform-field wrapper.
pub fn step_llg(m: &mut VectorField2D, params: &LLGParams) {
    let grid = m.grid;
    let mut b_eff = VectorField2D::new(grid);
    add_zeeman_field(&mut b_eff, params.b_ext);
    step_llg_with_field(m, &b_eff, params);
}