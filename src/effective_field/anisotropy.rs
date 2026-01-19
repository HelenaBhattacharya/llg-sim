// src/effective_field/anisotropy.rs

use crate::params::Material;
use crate::vector_field::VectorField2D;

/// Add uniaxial anisotropy contribution to B_eff (Tesla).
///
/// For w_ani = K_u [1 - (m·u)^2], we get:
///   B_ani = (2 K_u / M_s) (m·u) u
pub fn add_uniaxial_anisotropy_field(m: &VectorField2D, b_eff: &mut VectorField2D, mat: &Material) {
    let k_u = mat.k_u;
    let ms = mat.ms;
    if k_u == 0.0 || ms == 0.0 {
        return;
    }

    let coeff = 2.0 * k_u / ms;
    let u = mat.easy_axis;

    for (m_cell, b_cell) in m.data.iter().zip(b_eff.data.iter_mut()) {
        let mdotu = m_cell[0] * u[0] + m_cell[1] * u[1] + m_cell[2] * u[2];
        b_cell[0] += coeff * mdotu * u[0];
        b_cell[1] += coeff * mdotu * u[1];
        b_cell[2] += coeff * mdotu * u[2];
    }
}