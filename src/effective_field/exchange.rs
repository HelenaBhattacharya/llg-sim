// src/effective_field/exchange.rs

use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

#[inline]
fn ghost_x(m: [f64; 3], n_x: f64, eta: f64, dx: f64) -> [f64; 3] {
    let mx = m[0];
    let my = m[1];
    let mz = m[2];
    let dmx_dx = -eta * mz;
    let dmy_dx = 0.0;
    let dmz_dx = eta * mx;
    [
        mx + n_x * dx * dmx_dx,
        my + n_x * dx * dmy_dx,
        mz + n_x * dx * dmz_dx,
    ]
}

#[inline]
fn ghost_y(m: [f64; 3], n_y: f64, eta: f64, dy: f64) -> [f64; 3] {
    let mx = m[0];
    let my = m[1];
    let mz = m[2];
    let dmx_dy = 0.0;
    let dmy_dy = -eta * mz;
    let dmz_dy = eta * my;
    [
        mx + n_y * dy * dmx_dy,
        my + n_y * dy * dmy_dy,
        mz + n_y * dy * dmz_dy,
    ]
}

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

    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    let coeff = 2.0 * a / ms;

    let eta_opt: Option<f64> = match mat.dmi {
        Some(d) if d != 0.0 => Some(d / (2.0 * a)),
        _ => None,
    };
    // If DMI is enabled, exchange must use modified (chiral) boundary conditions
    // to remain consistent with the interfacial DMI formulation (MuMax-style)

    for j in 0..ny {
        for i in 0..nx {
            let idx = m.idx(i, j);
            let m_ij = m.data[idx];

            let (m_im, m_ip) = if nx == 1 {
                (m_ij, m_ij)
            } else if i == 0 {
                let left = if let Some(eta) = eta_opt {
                    ghost_x(m_ij, -1.0, eta, grid.dx)
                } else {
                    m_ij
                };
                (left, m.data[m.idx(i + 1, j)])
            } else if i == nx - 1 {
                let right = if let Some(eta) = eta_opt {
                    ghost_x(m_ij, 1.0, eta, grid.dx)
                } else {
                    m_ij
                };
                (m.data[m.idx(i - 1, j)], right)
            } else {
                (m.data[m.idx(i - 1, j)], m.data[m.idx(i + 1, j)])
            };

            let (m_jm, m_jp) = if ny == 1 {
                (m_ij, m_ij)
            } else if j == 0 {
                let down = if let Some(eta) = eta_opt {
                    ghost_y(m_ij, -1.0, eta, grid.dy)
                } else {
                    m_ij
                };
                (down, m.data[m.idx(i, j + 1)])
            } else if j == ny - 1 {
                let up = if let Some(eta) = eta_opt {
                    ghost_y(m_ij, 1.0, eta, grid.dy)
                } else {
                    m_ij
                };
                (m.data[m.idx(i, j - 1)], up)
            } else {
                (m.data[m.idx(i, j - 1)], m.data[m.idx(i, j + 1)])
            };

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
