// src/llg.rs

use crate::vector_field::VectorField2D;
use crate::params::LLGParams;
use crate::vec3::{cross, normalize};

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

// Tests for LLG live at the bottom, see Step 2.
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
            step_llg(&mut m, &params);
        }

        let m_vec = m.data[0];
        let norm = (m_vec[0].powi(2) + m_vec[1].powi(2) + m_vec[2].powi(2)).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}