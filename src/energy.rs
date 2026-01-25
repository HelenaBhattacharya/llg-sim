// src/energy.rs
//
// DMI energy is computed from the effective field, as in MuMax3 Eq. (17):
//   E_DMI = -1/2 * ∫ M · B_DM dV
//
// This makes your energy-monotone relaxation meaningful and consistent.

use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

#[derive(Debug, Copy, Clone)]
pub struct EnergyBreakdown {
    pub exchange: f64,
    pub anisotropy: f64,
    pub zeeman: f64,
    pub dmi: f64,
}

impl EnergyBreakdown {
    pub fn total(&self) -> f64 {
        self.exchange + self.anisotropy + self.zeeman + self.dmi
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
    [mx + n_x * dx * dmx_dx, my + n_x * dx * dmy_dx, mz + n_x * dx * dmz_dx]
}

#[inline]
fn ghost_y(m: [f64; 3], n_y: f64, eta: f64, dy: f64) -> [f64; 3] {
    let mx = m[0];
    let my = m[1];
    let mz = m[2];
    let dmx_dy = 0.0;
    let dmy_dy = -eta * mz;
    let dmz_dy = eta * my;
    [mx + n_y * dy * dmx_dy, my + n_y * dy * dmy_dy, mz + n_y * dy * dmz_dy]
}

pub fn compute_energy(grid: &Grid2D, m: &VectorField2D, material: &Material, b_ext: [f64; 3]) -> EnergyBreakdown {
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

    let (dmi_opt, eta_opt) = match (material.dmi, material.a_ex) {
        (Some(d), a) if d != 0.0 && a != 0.0 => (Some(d), Some(d / (2.0 * a))),
        _ => (None, None),
    };

    for j in 0..ny {
        for i in 0..nx {
            let idx = grid.idx(i, j);
            let mij = m.data[idx];
            let (mx, my, mz) = (mij[0], mij[1], mij[2]);

            // Exchange (forward differences)
            if aex != 0.0 {
                if i + 1 < nx {
                    let mip = m.data[grid.idx(i + 1, j)];
                    let dxm = [mip[0] - mx, mip[1] - my, mip[2] - mz];
                    let sq = dxm[0] * dxm[0] + dxm[1] * dxm[1] + dxm[2] * dxm[2];
                    e_ex += aex * (sq / (dx * dx)) * v;
                }
                if j + 1 < ny {
                    let mjp = m.data[grid.idx(i, j + 1)];
                    let dym = [mjp[0] - mx, mjp[1] - my, mjp[2] - mz];
                    let sq = dym[0] * dym[0] + dym[1] * dym[1] + dym[2] * dym[2];
                    e_ex += aex * (sq / (dy * dy)) * v;
                }
            }

            // Uniaxial anisotropy
            if ku != 0.0 {
                let mdotu = mx * u[0] + my * u[1] + mz * u[2];
                e_an += ku * (1.0 - mdotu * mdotu) * v;
            }

            // Zeeman
            if ms != 0.0 {
                let mdotb = mx * bx + my * by + mz * bz;
                e_zee -= ms * mdotb * v;
            }

            // DMI energy from field (MuMax3 Eq. 17)
            if let (Some(d), Some(eta)) = (dmi_opt, eta_opt) {
                if ms == 0.0 {
                    continue;
                }

                let pref = 2.0 * d / ms;

                let (m_im, m_ip) = if nx == 1 {
                    (mij, mij)
                } else if i == 0 {
                    (ghost_x(mij, -1.0, eta, dx), m.data[grid.idx(i + 1, j)])
                } else if i == nx - 1 {
                    (m.data[grid.idx(i - 1, j)], ghost_x(mij, 1.0, eta, dx))
                } else {
                    (m.data[grid.idx(i - 1, j)], m.data[grid.idx(i + 1, j)])
                };

                let (m_jm, m_jp) = if ny == 1 {
                    (mij, mij)
                } else if j == 0 {
                    (ghost_y(mij, -1.0, eta, dy), m.data[grid.idx(i, j + 1)])
                } else if j == ny - 1 {
                    (m.data[grid.idx(i, j - 1)], ghost_y(mij, 1.0, eta, dy))
                } else {
                    (m.data[grid.idx(i, j - 1)], m.data[grid.idx(i, j + 1)])
                };

                let dmz_dx = if nx > 1 { (m_ip[2] - m_im[2]) / (2.0 * dx) } else { 0.0 };
                let dmz_dy = if ny > 1 { (m_jp[2] - m_jm[2]) / (2.0 * dy) } else { 0.0 };
                let dmx_dx = if nx > 1 { (m_ip[0] - m_im[0]) / (2.0 * dx) } else { 0.0 };
                let dmy_dy = if ny > 1 { (m_jp[1] - m_jm[1]) / (2.0 * dy) } else { 0.0 };

                let bdm_x = pref * dmz_dx;
                let bdm_y = pref * dmz_dy;
                let bdm_z = -pref * (dmx_dx + dmy_dy);

                let mdotb = mx * bdm_x + my * bdm_y + mz * bdm_z;
                e_dmi += -0.5 * ms * mdotb * v;
            }
        }
    }

    EnergyBreakdown { exchange: e_ex, anisotropy: e_an, zeeman: e_zee, dmi: e_dmi }
}

pub fn compute_total_energy(grid: &Grid2D, m: &VectorField2D, material: &Material, b_ext: [f64; 3]) -> f64 {
    compute_energy(grid, m, material, b_ext).total()
}