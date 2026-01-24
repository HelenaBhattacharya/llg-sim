// src/energy.rs
//
// Energy bookkeeping in SI units.
//
// Conventions:
// - m is a unit vector (dimensionless).
// - B_ext is in Tesla.
// - Energies are returned in Joules.
//
// Implemented terms:
//   Exchange:    w_ex  = A |∇m|^2                      [J/m^3]
//   Anisotropy:  w_an  = K_u [1 - (m·u)^2]             [J/m^3]
//   Zeeman:      w_zee = - M_s (m · B_ext)             [J/m^3]
//   Interfacial DMI (thin film):
//                w_dmi(areal) = D [ m_z (∂x m_x + ∂y m_y) - (m_x ∂x m_z + m_y ∂y m_z) ]   [J/m^2]
//
// Notes on DMI units:
// - Here D is treated as interfacial DMI with units J/m^2 (as in MuMax Dind).
// - The total DMI energy is an area integral, so we multiply by (dx*dy) per cell (not by dz).
//
// Discretisation choices:
// - Exchange uses forward differences (consistent with avoiding double counting).
// - DMI energy uses forward differences for derivatives in x and y.
// - At the domain boundary where a forward neighbour does not exist, the corresponding derivative
//   contribution is taken as 0 (simple Neumann-like handling).

use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

#[derive(Debug, Copy, Clone)]
pub struct EnergyBreakdown {
    pub exchange: f64,
    pub anisotropy: f64,
    pub zeeman: f64,
    pub dmi: f64, // interfacial DMI energy (J)
}

impl EnergyBreakdown {
    pub fn total(&self) -> f64 {
        self.exchange + self.anisotropy + self.zeeman + self.dmi
    }
}

/// Compute the SI-consistent energy breakdown for the current magnetisation.
pub fn compute_energy(
    grid: &Grid2D,
    m: &VectorField2D,
    material: &Material,
    b_ext: [f64; 3],
) -> EnergyBreakdown {
    let nx = grid.nx;
    let ny = grid.ny;
    let dx = grid.dx;
    let dy = grid.dy;

    // Volume per cell for volume-density terms (exchange/anisotropy/zeeman)
    let v = grid.cell_volume(); // dx*dy*dz

    // Area per cell for interfacial DMI (areal density)
    let a_cell = dx * dy;

    let aex = material.a_ex;
    let ku = material.k_u;
    let u = material.easy_axis;
    let ms = material.ms;
    let dmi_opt = material.dmi; // Option<f64> in J/m^2

    let (bx, by, bz) = (b_ext[0], b_ext[1], b_ext[2]);

    let mut e_ex = 0.0;
    let mut e_an = 0.0;
    let mut e_zee = 0.0;
    let mut e_dmi = 0.0;

    for j in 0..ny {
        for i in 0..nx {
            let idx = grid.idx(i, j);
            let mij = m.data[idx];
            let (mx, my, mz) = (mij[0], mij[1], mij[2]);

            // -------------------------
            // Exchange (forward differences)
            // -------------------------
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

            // -------------------------
            // Uniaxial anisotropy
            // -------------------------
            if ku != 0.0 {
                let mdotu = mx * u[0] + my * u[1] + mz * u[2];
                e_an += ku * (1.0 - mdotu * mdotu) * v;
            }

            // -------------------------
            // Zeeman
            // -------------------------
            if ms != 0.0 {
                let mdotb = mx * bx + my * by + mz * bz;
                e_zee -= ms * mdotb * v;
            }

            // -------------------------
            // Interfacial DMI (forward differences)
            //
            // w_dmi = D [ m_z (∂x m_x + ∂y m_y) - (m_x ∂x m_z + m_y ∂y m_z) ]
            // E_dmi ≈ Σ w_dmi * (dx*dy)
            // -------------------------
            if let Some(d) = dmi_opt {
                // forward differences; if neighbour missing, derivative contribution = 0
                let (dmx_dx, dmz_dx) = if i + 1 < nx {
                    let mip = m.data[grid.idx(i + 1, j)];
                    ((mip[0] - mx) / dx, (mip[2] - mz) / dx)
                } else {
                    (0.0, 0.0)
                };

                let (dmy_dy, dmz_dy) = if j + 1 < ny {
                    let mjp = m.data[grid.idx(i, j + 1)];
                    ((mjp[1] - my) / dy, (mjp[2] - mz) / dy)
                } else {
                    (0.0, 0.0)
                };

                let w_dmi = d * (mz * (dmx_dx + dmy_dy) - (mx * dmz_dx + my * dmz_dy));
                e_dmi += w_dmi * a_cell;
            }
        }
    }

    EnergyBreakdown {
        exchange: e_ex,
        anisotropy: e_an,
        zeeman: e_zee,
        dmi: e_dmi,
    }
}

pub fn compute_total_energy(
    grid: &Grid2D,
    m: &VectorField2D,
    material: &Material,
    b_ext: [f64; 3],
) -> f64 {
    compute_energy(grid, m, material, b_ext).total()
}