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

/// Add the exchange contribution to `b_eff`, with an optional geometry mask.
///
/// Mask semantics:
/// - If `mask[idx] == false`, we add *no exchange field* at that cell.
/// - If a neighbour is outside the mask, we treat it as a free boundary (Neumann):
///   use `m_nb := m_center` so the finite-difference gradient across vacuum is zero.
/// - If interfacial DMI is enabled (MuMax-style), we apply the same chiral ghost
///   construction at both the domain boundary and any mask boundary.
pub fn add_exchange_field_masked(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    mat: &Material,
    mask: Option<&[bool]>,
) {
    let nx = grid.nx;
    let ny = grid.ny;

    if nx == 0 || ny == 0 {
        return;
    }

    if let Some(msk) = mask {
        debug_assert_eq!(
            msk.len(),
            grid.n_cells(),
            "mask length must match grid.n_cells()"
        );
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

    for j in 0..ny {
        for i in 0..nx {
            let idx_c = m.idx(i, j);

            // If this cell is vacuum, do not add exchange here.
            if let Some(msk) = mask {
                if !msk[idx_c] {
                    continue;
                }
            }

            let m_ij = m.data[idx_c];

            // --- X neighbours (left/right) ---
            let m_im = if nx == 1 {
                m_ij
            } else if i == 0 {
                // Domain boundary
                if let Some(eta) = eta_opt {
                    ghost_x(m_ij, -1.0, eta, grid.dx)
                } else {
                    m_ij
                }
            } else {
                // Interior: potentially a mask boundary
                let idx_l = m.idx(i - 1, j);
                if let Some(msk) = mask {
                    if !msk[idx_l] {
                        if let Some(eta) = eta_opt {
                            ghost_x(m_ij, -1.0, eta, grid.dx)
                        } else {
                            m_ij
                        }
                    } else {
                        m.data[idx_l]
                    }
                } else {
                    m.data[idx_l]
                }
            };

            let m_ip = if nx == 1 {
                m_ij
            } else if i == nx - 1 {
                // Domain boundary
                if let Some(eta) = eta_opt {
                    ghost_x(m_ij, 1.0, eta, grid.dx)
                } else {
                    m_ij
                }
            } else {
                // Interior: potentially a mask boundary
                let idx_r = m.idx(i + 1, j);
                if let Some(msk) = mask {
                    if !msk[idx_r] {
                        if let Some(eta) = eta_opt {
                            ghost_x(m_ij, 1.0, eta, grid.dx)
                        } else {
                            m_ij
                        }
                    } else {
                        m.data[idx_r]
                    }
                } else {
                    m.data[idx_r]
                }
            };

            // --- Y neighbours (down/up) ---
            let m_jm = if ny == 1 {
                m_ij
            } else if j == 0 {
                // Domain boundary
                if let Some(eta) = eta_opt {
                    ghost_y(m_ij, -1.0, eta, grid.dy)
                } else {
                    m_ij
                }
            } else {
                // Interior: potentially a mask boundary
                let idx_d = m.idx(i, j - 1);
                if let Some(msk) = mask {
                    if !msk[idx_d] {
                        if let Some(eta) = eta_opt {
                            ghost_y(m_ij, -1.0, eta, grid.dy)
                        } else {
                            m_ij
                        }
                    } else {
                        m.data[idx_d]
                    }
                } else {
                    m.data[idx_d]
                }
            };

            let m_jp = if ny == 1 {
                m_ij
            } else if j == ny - 1 {
                // Domain boundary
                if let Some(eta) = eta_opt {
                    ghost_y(m_ij, 1.0, eta, grid.dy)
                } else {
                    m_ij
                }
            } else {
                // Interior: potentially a mask boundary
                let idx_u = m.idx(i, j + 1);
                if let Some(msk) = mask {
                    if !msk[idx_u] {
                        if let Some(eta) = eta_opt {
                            ghost_y(m_ij, 1.0, eta, grid.dy)
                        } else {
                            m_ij
                        }
                    } else {
                        m.data[idx_u]
                    }
                } else {
                    m.data[idx_u]
                }
            };

            // Laplacian per component
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
                b_eff.data[idx_c][c] += coeff * (d2x + d2y);
            }
        }
    }
}

pub fn add_exchange_field(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    mat: &Material,
) {
    add_exchange_field_masked(grid, m, b_eff, mat, None);
}
