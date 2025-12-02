// src/llg.rs

use crate::vector_field::VectorField2D;
use crate::params::LLGParams;
use crate::vec3::{cross, normalize};
use crate::effective_field::zeeman::add_zeeman_field;

/// Core integrator: advance m by one step, given a precomputed H_eff field.
///
/// H_eff is supplied as a VectorField2D with the same grid as m.
pub fn step_llg_with_field(m: &mut VectorField2D, h_eff: &VectorField2D, params: &LLGParams) {
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt = params.dt;

    // loop over all cells
    for (cell_idx, cell) in m.data.iter_mut().enumerate() {
        let m_vec = *cell;
        let h = h_eff.data[cell_idx];

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

        // renormalize to |m| = 1
        *cell = normalize(m_new);
    }
}

/// Convenience wrapper for the old API:
/// builds a uniform H_eff from params.h_ext and calls step_llg_with_field.
///
/// This keeps your existing tests and simple use-cases working.
pub fn step_llg(m: &mut VectorField2D, params: &LLGParams) {
    // Create a temporary H_eff field on the same grid as m
    let grid = m.grid;
    let mut h_eff = VectorField2D::new(grid);

    // Add Zeeman field: uniform H_ext everywhere
    add_zeeman_field(&mut h_eff, params.h_ext);

    // Call the core integrator
    step_llg_with_field(m, &h_eff, params);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;
    use crate::vector_field::VectorField2D;
    use crate::params::LLGParams;

    #[test]
    fn macrospin_norm_stays_one() {
        let grid = Grid2D::new(1, 1, 1e-9, 1e-9);
        let mut m = VectorField2D::new(grid);
        m.set_uniform(0.1, 0.0, (1.0 - 0.1_f64.powi(2)).sqrt());

        let params = LLGParams {
            gamma: 1.0,
            alpha: 0.1,
            dt: 0.01,
            h_ext: [1.0, 0.0, 0.0],
        };

        for _ in 0..1000 {
            // uses the wrapper, which internally builds H_eff and calls step_llg_with_field
            step_llg(&mut m, &params);
        }

        let m_vec = m.data[0];
        let norm = (m_vec[0].powi(2) + m_vec[1].powi(2) + m_vec[2].powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}