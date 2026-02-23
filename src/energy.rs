// src/energy.rs
//
// Energy bookkeeping for the micromagnetic solver.
//
// Conventions used across the codebase:
// - m is a unit vector (dimensionless).
// - B_* fields are inductions in Tesla.
// - M = Ms * m is magnetisation in A/m.
// - Cell volume is V = dx * dy * dz.
//
// DMI and Demag energies are computed from their effective fields in the same spirit as MuMax3:
//   E_term = -1/2 * ∫ M · B_term dV
// so that relaxation dynamics with damping is energy-monotone and comparisons are consistent.

use crate::effective_field::demag;
use crate::geometry_mask::assert_mask_len;
use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

#[derive(Debug, Copy, Clone)]
pub struct EnergyBreakdown {
    pub exchange: f64,
    pub anisotropy: f64,
    pub zeeman: f64,
    pub dmi: f64,
    pub demag: f64,
}

impl EnergyBreakdown {
    pub fn total(&self) -> f64 {
        self.exchange + self.anisotropy + self.zeeman + self.dmi + self.demag
    }
}

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

pub fn compute_energy(
    grid: &Grid2D,
    m: &VectorField2D,
    material: &Material,
    b_ext: [f64; 3],
) -> EnergyBreakdown {
    compute_energy_geom(grid, m, material, b_ext, None)
}

/// Compute energy breakdown with an optional geometry mask.
///
/// If `geom_mask` is provided, only cells with geom_mask[idx]==true contribute to the
/// energy sums. Neighbour-based terms (exchange, DMI) treat missing neighbours (outside
/// the mask) as a free boundary (Neumann) by skipping pair contributions / using ghosts.
pub fn compute_energy_geom(
    grid: &Grid2D,
    m: &VectorField2D,
    material: &Material,
    b_ext: [f64; 3],
    geom_mask: Option<&[bool]>,
) -> EnergyBreakdown {
    if let Some(msk) = geom_mask {
        assert_mask_len(msk, grid);
    }

    let nx = grid.nx;
    let ny = grid.ny;
    let dx = grid.dx;
    let dy = grid.dy;
    let v = grid.cell_volume();

    let aex = material.a_ex;
    let ku = material.k_u;
    let u = material.easy_axis;
    let ms = material.ms;

    let (bx, by, bz) = (b_ext[0], b_ext[1], b_ext[2]);

    let mut e_ex = 0.0;
    let mut e_an = 0.0;
    let mut e_zee = 0.0;
    let mut e_dmi = 0.0;
    let mut e_demag = 0.0;

    let (dmi_opt, eta_opt) = match (material.dmi, material.a_ex) {
        (Some(d), a) if d != 0.0 && a != 0.0 => (Some(d), Some(d / (2.0 * a))),
        _ => (None, None),
    };

    // If demag enabled, compute B_demag on the whole grid once.
    // NOTE: demag with masked geometries is handled separately (Stage 3).
    let mut b_demag = VectorField2D::new(*grid);
    if material.demag {
        b_demag.set_uniform(0.0, 0.0, 0.0);
        demag::add_demag_field(grid, m, &mut b_demag, material);
    }

    #[inline]
    fn inside(geom_mask: Option<&[bool]>, idx: usize) -> bool {
        match geom_mask {
            Some(msk) => msk[idx],
            None => true,
        }
    }

    for j in 0..ny {
        for i in 0..nx {
            let idx = grid.idx(i, j);

            // Skip vacuum cells.
            if !inside(geom_mask, idx) {
                continue;
            }

            let mij = m.data[idx];
            let (mx, my, mz) = (mij[0], mij[1], mij[2]);

            // Exchange energy (forward differences): only count pairs fully inside the mask.
            if aex != 0.0 {
                if i + 1 < nx {
                    let idx_r = grid.idx(i + 1, j);
                    if inside(geom_mask, idx_r) {
                        let mip = m.data[idx_r];
                        let dxm = [mip[0] - mx, mip[1] - my, mip[2] - mz];
                        let sq = dxm[0] * dxm[0] + dxm[1] * dxm[1] + dxm[2] * dxm[2];
                        e_ex += aex * (sq / (dx * dx)) * v;
                    }
                }
                if j + 1 < ny {
                    let idx_u = grid.idx(i, j + 1);
                    if inside(geom_mask, idx_u) {
                        let mjp = m.data[idx_u];
                        let dym = [mjp[0] - mx, mjp[1] - my, mjp[2] - mz];
                        let sq = dym[0] * dym[0] + dym[1] * dym[1] + dym[2] * dym[2];
                        e_ex += aex * (sq / (dy * dy)) * v;
                    }
                }
            }

            // Uniaxial anisotropy (local)
            if ku != 0.0 {
                let mdotu = mx * u[0] + my * u[1] + mz * u[2];
                e_an += ku * (1.0 - mdotu * mdotu) * v;
            }

            // Zeeman (local)
            if ms != 0.0 {
                let mdotb = mx * bx + my * by + mz * bz;
                e_zee -= ms * mdotb * v;
            }

            // DMI energy from field (MuMax-style: E = -1/2 ∫ M · B_DM dV)
            if let (Some(d), Some(eta)) = (dmi_opt, eta_opt) {
                if ms != 0.0 {
                    let pref = 2.0 * d / ms;

                    // Neighbours with mask-aware ghosting:
                    let (m_im, m_ip) = if nx == 1 {
                        (mij, mij)
                    } else {
                        // left
                        let m_im = if i == 0 {
                            ghost_x(mij, -1.0, eta, dx)
                        } else {
                            let idx_l = grid.idx(i - 1, j);
                            if inside(geom_mask, idx_l) {
                                m.data[idx_l]
                            } else {
                                ghost_x(mij, -1.0, eta, dx)
                            }
                        };
                        // right
                        let m_ip = if i == nx - 1 {
                            ghost_x(mij, 1.0, eta, dx)
                        } else {
                            let idx_r = grid.idx(i + 1, j);
                            if inside(geom_mask, idx_r) {
                                m.data[idx_r]
                            } else {
                                ghost_x(mij, 1.0, eta, dx)
                            }
                        };
                        (m_im, m_ip)
                    };

                    let (m_jm, m_jp) = if ny == 1 {
                        (mij, mij)
                    } else {
                        // down
                        let m_jm = if j == 0 {
                            ghost_y(mij, -1.0, eta, dy)
                        } else {
                            let idx_d = grid.idx(i, j - 1);
                            if inside(geom_mask, idx_d) {
                                m.data[idx_d]
                            } else {
                                ghost_y(mij, -1.0, eta, dy)
                            }
                        };
                        // up
                        let m_jp = if j == ny - 1 {
                            ghost_y(mij, 1.0, eta, dy)
                        } else {
                            let idx_u = grid.idx(i, j + 1);
                            if inside(geom_mask, idx_u) {
                                m.data[idx_u]
                            } else {
                                ghost_y(mij, 1.0, eta, dy)
                            }
                        };
                        (m_jm, m_jp)
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

                    let bdm_x = pref * dmz_dx;
                    let bdm_y = pref * dmz_dy;
                    let bdm_z = -pref * (dmx_dx + dmy_dy);

                    let mdotb = mx * bdm_x + my * bdm_y + mz * bdm_z;
                    e_dmi += -0.5 * ms * mdotb * v;
                }
            }

            // Demag energy from field (E = -1/2 ∫ M · B_demag dV)
            if material.demag && ms != 0.0 {
                let b = b_demag.data[idx];
                let mdotb = mx * b[0] + my * b[1] + mz * b[2];
                e_demag += -0.5 * ms * mdotb * v;
            }
        }
    }

    EnergyBreakdown {
        exchange: e_ex,
        anisotropy: e_an,
        zeeman: e_zee,
        dmi: e_dmi,
        demag: e_demag,
    }
}

pub fn compute_total_energy(
    grid: &Grid2D,
    m: &VectorField2D,
    material: &Material,
    b_ext: [f64; 3],
) -> f64 {
    compute_total_energy_geom(grid, m, material, b_ext, None)
}

/// Compute total energy with an optional geometry mask.
pub fn compute_total_energy_geom(
    grid: &Grid2D,
    m: &VectorField2D,
    material: &Material,
    b_ext: [f64; 3],
    geom_mask: Option<&[bool]>,
) -> f64 {
    compute_energy_geom(grid, m, material, b_ext, geom_mask).total()
}
