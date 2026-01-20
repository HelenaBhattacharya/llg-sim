// src/effective_field/exchange.rs

use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

/// Add the exchange contribution to B_eff (Tesla).
///
///   B_ex = (2 A / M_s) ∇² m
///
/// Boundary condition:
///   Free / Neumann (∂m/∂n = 0) implemented by mirroring the edge value:
///   ghost cell outside the domain = edge cell value.
///
/// Performance note:
///   This version avoids allocating a temporary Laplacian array every call.
///   That matters a lot because RK4 recompute-field calls build_h_eff 4x per step.
pub fn add_exchange_field(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    mat: &Material,
) {
    let nx = grid.nx;
    let ny = grid.ny;

    if nx == 0 || ny == 0 {
        return;
    }

    let a = mat.a_ex;
    let ms = mat.ms;
    if a == 0.0 || ms == 0.0 {
        return;
    }

    // If nx==1 or ny==1, the Laplacian in that direction is automatically zero under Neumann mirror.
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    let coeff = 2.0 * a / ms;

    for j in 0..ny {
        let j_m = if j == 0 { 0 } else { j - 1 };
        let j_p = if j + 1 == ny { ny - 1 } else { j + 1 };

        for i in 0..nx {
            let i_m = if i == 0 { 0 } else { i - 1 };
            let i_p = if i + 1 == nx { nx - 1 } else { i + 1 };

            let idx = m.idx(i, j);

            let m_ij = m.data[idx];
            let m_ip = m.data[m.idx(i_p, j)];
            let m_im = m.data[m.idx(i_m, j)];
            let m_jp = m.data[m.idx(i, j_p)];
            let m_jm = m.data[m.idx(i, j_m)];

            for c in 0..3 {
                let d2x = if nx > 1 {
                    (m_ip[c] - 2.0 * m_ij[c] + m_im[c]) / dx2
                } else {
                    0.0
                };

                let d2y = if ny > 1 {
                    (m_jp[c] - 2.0 * m_ij[c] + m_jm[c]) / dy2
                } else {
                    0.0
                };

                b_eff.data[idx][c] += coeff * (d2x + d2y);
            }
        }
    }
}
