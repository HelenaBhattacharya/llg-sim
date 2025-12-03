// src/effective_field/exchange.rs

use crate::grid::Grid2D;
use crate::vector_field::VectorField2D;

/// Add a simple finite-difference exchange field to H_eff.
///
/// H_exch ∝ ∇² m. Here `a_ex` is an effective exchange coefficient that
/// already includes the physical prefactor (2 A_ex / (mu0 * M_s)), so we can
/// write:
///     H_exch = a_ex * ∇² m
///
/// For now we use a 5-point stencil with free (zero-Laplacian) boundaries.
pub fn add_exchange_field(
    grid: &Grid2D,
    m: &VectorField2D,
    h_eff: &mut VectorField2D,
    a_ex: f64,
) {
    let nx = grid.nx;
    let ny = grid.ny;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    // Temporary array to hold Laplacian(m) so we don't mix updated values
    let mut lap = vec![[0.0f64; 3]; nx * ny];

    for j in 0..ny {
        for i in 0..nx {
            let idx = m.idx(i, j);
            let m_ij = m.data[idx];

            // Simple boundary condition: zero Laplacian at edges
            if i == 0 || i + 1 == nx || j == 0 || j + 1 == ny {
                lap[idx] = [0.0, 0.0, 0.0];
                continue;
            }

            let m_ip = m.data[m.idx(i + 1, j)];
            let m_im = m.data[m.idx(i - 1, j)];
            let m_jp = m.data[m.idx(i, j + 1)];
            let m_jm = m.data[m.idx(i, j - 1)];

            let mut lap_vec = [0.0; 3];
            for c in 0..3 {
                let d2x = (m_ip[c] - 2.0 * m_ij[c] + m_im[c]) / dx2;
                let d2y = (m_jp[c] - 2.0 * m_ij[c] + m_jm[c]) / dy2;
                lap_vec[c] = d2x + d2y;
            }
            lap[idx] = lap_vec;
        }
    }

    // Add H_exch = a_ex * Laplacian(m) to H_eff
    for (idx, h_cell) in h_eff.data.iter_mut().enumerate() {
        for c in 0..3 {
            h_cell[c] += a_ex * lap[idx][c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;
    use crate::vector_field::VectorField2D;

    #[test]
    fn uniform_m_gives_zero_exchange_field() {
        let nx = 16;
        let ny = 16;
        let dx = 1.0;
        let dy = 1.0;

        let grid = Grid2D::new(nx, ny, dx, dy);
        let mut m = VectorField2D::new(grid);
        // uniform +z (already the default, but be explicit)
        m.set_uniform(0.0, 0.0, 1.0);

        let mut h_eff = VectorField2D::new(grid);
        // reset to zero to avoid any leftover values
        h_eff.set_uniform(0.0, 0.0, 0.0);

        let a_ex = 1.0;
        add_exchange_field(&grid, &m, &mut h_eff, a_ex);

        for cell in &h_eff.data {
            let norm = (cell[0].powi(2) + cell[1].powi(2) + cell[2].powi(2)).sqrt();
            assert!(norm.abs() < 1e-12, "exchange field not zero for uniform m");
        }
    }
}