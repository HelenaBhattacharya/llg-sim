// src/effective_field/exchange.rs

use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

/// Add the exchange contribution to B_eff (Tesla).
///
/// B_ex = (2 A / M_s) ∇² m
pub fn add_exchange_field(grid: &Grid2D, m: &VectorField2D, b_eff: &mut VectorField2D, mat: &Material) {
    let nx = grid.nx;
    let ny = grid.ny;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    let a = mat.a_ex;
    let ms = mat.ms;
    if a == 0.0 || ms == 0.0 {
        return;
    }

    let coeff = 2.0 * a / ms;

    let mut lap = vec![[0.0f64; 3]; nx * ny];

    for j in 0..ny {
        for i in 0..nx {
            let idx = m.idx(i, j);
            let m_ij = m.data[idx];

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

    for (idx, b_cell) in b_eff.data.iter_mut().enumerate() {
        for c in 0..3 {
            b_cell[c] += coeff * lap[idx][c];
        }
    }
}