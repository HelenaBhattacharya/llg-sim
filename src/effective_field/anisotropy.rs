// src/effective_field/anisotropy.rs

use crate::vector_field::VectorField2D;
use crate::params::Material;

/// Add a uniaxial anisotropy field H_ani to H_eff.
///
/// Starting from the energy density
///   E_an = k_u * [1 - (m·u)^2],
/// the functional derivative gives (up to constants)
///   H_ani ∝ 2 * k_u * (m·u) * u.
///
/// Here we fold all physical prefactors into k_u, so we simply use:
///   H_ani = 2 * k_u * (m·u) * u.
pub fn add_uniaxial_anisotropy_field(
    m: &VectorField2D,
    h_eff: &mut VectorField2D,
    mat: &Material,
) {
    let k_u = mat.k_u;
    if k_u == 0.0 {
        return;
    }

    let u = mat.easy_axis;
    let (ux, uy, uz) = (u[0], u[1], u[2]);

    for (m_cell, h_cell) in m.data.iter().zip(h_eff.data.iter_mut()) {
        let (mx, my, mz) = (m_cell[0], m_cell[1], m_cell[2]);
        let mdotu = mx * ux + my * uy + mz * uz;

        // H_ani = 2 k_u (m·u) u
        h_cell[0] += 2.0 * k_u * mdotu * ux;
        h_cell[1] += 2.0 * k_u * mdotu * uy;
        h_cell[2] += 2.0 * k_u * mdotu * uz;
    }
}