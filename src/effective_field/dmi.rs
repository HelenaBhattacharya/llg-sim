// src/effective_field/dmi.rs
//
// Interfacial (Néel-type) DMI for thin films (MuMax3).
//
// MuMax3 Eq. (10):
//   B_DM = (2D/Msat) * ( ∂x m_z, ∂y m_z, -(∂x m_x + ∂y m_y) )
//
// Boundary conditions (Eq. 11–15) are imposed using ghost neighbours so derivatives remain central.
// In case of nonzero D, these BCs must also be applied to exchange.
//
// IMPORTANT: NO μ0 in prefactor.

use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

#[inline]
fn ghost_x(m: [f64; 3], n_x: f64, eta: f64, dx: f64) -> [f64; 3] {
    let mx = m[0];
    let my = m[1];
    let mz = m[2];

    // Eq. 11, 13, 14 (x-normal boundary)
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

    // Eq. 12, 13, 14 (y-normal boundary)
    let dmx_dy = 0.0;
    let dmy_dy = -eta * mz;
    let dmz_dy = eta * my;

    [
        mx + n_y * dy * dmx_dy,
        my + n_y * dy * dmy_dy,
        mz + n_y * dy * dmz_dy,
    ]
}

pub fn add_dmi_field(grid: &Grid2D, m: &VectorField2D, b_eff: &mut VectorField2D, mat: &Material) {
    let d = match mat.dmi {
        Some(v) if v != 0.0 => v,
        _ => return,
    };
    let ms = mat.ms;
    let a = mat.a_ex;

    if ms == 0.0 || a == 0.0 {
        return;
    }

    let nx = grid.nx;
    let ny = grid.ny;
    if nx == 0 || ny == 0 {
        return;
    }

    let dx = grid.dx;
    let dy = grid.dy;

    // MuMax3 prefactor: 2D/Msat (NO μ0)
    let pref = 2.0 * d / ms;

    // Boundary coupling eta = D/(2A)
    let eta = d / (2.0 * a);

    for j in 0..ny {
        for i in 0..nx {
            let idx = m.idx(i, j);
            let m_ij = m.data[idx];

            let (m_im, m_ip) = if nx == 1 {
                (m_ij, m_ij)
            } else if i == 0 {
                (ghost_x(m_ij, -1.0, eta, dx), m.data[m.idx(i + 1, j)])
            } else if i == nx - 1 {
                (m.data[m.idx(i - 1, j)], ghost_x(m_ij, 1.0, eta, dx))
            } else {
                (m.data[m.idx(i - 1, j)], m.data[m.idx(i + 1, j)])
            };

            let (m_jm, m_jp) = if ny == 1 {
                (m_ij, m_ij)
            } else if j == 0 {
                (ghost_y(m_ij, -1.0, eta, dy), m.data[m.idx(i, j + 1)])
            } else if j == ny - 1 {
                (m.data[m.idx(i, j - 1)], ghost_y(m_ij, 1.0, eta, dy))
            } else {
                (m.data[m.idx(i, j - 1)], m.data[m.idx(i, j + 1)])
            };

            let dmz_dx = if nx > 1 {
                (m_ip[2] - m_im[2]) / (2.0 * dx)
            } else {
                0.0
            };
            let dmz_dy = if ny > 1 {
                (m_jp[2] - m_jm[2]) / (2.0 * dy)
            } else {
                0.0
            };
            let dmx_dx = if nx > 1 {
                (m_ip[0] - m_im[0]) / (2.0 * dx)
            } else {
                0.0
            };
            let dmy_dy = if ny > 1 {
                (m_jp[1] - m_jm[1]) / (2.0 * dy)
            } else {
                0.0
            };

            b_eff.data[idx][0] += pref * dmz_dx;
            b_eff.data[idx][1] += pref * dmz_dy;
            b_eff.data[idx][2] += -pref * (dmx_dx + dmy_dy);
        }
    }
}
