// src/energy.rs

use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

#[derive(Debug, Copy, Clone)]
pub struct EnergyBreakdown {
    pub exchange: f64,
    pub anisotropy: f64,
    pub zeeman: f64,
}

impl EnergyBreakdown {
    pub fn total(&self) -> f64 {
        self.exchange + self.anisotropy + self.zeeman
    }
}

/// SI-consistent energy model:
///   w_ex  = A |∇m|^2
///   w_ani = K_u [1 - (m·u)^2]
///   w_zee = - M_s (m · B_ext)
/// and E = ∫ w dV.
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
    let v = grid.cell_volume();

    let a = material.a_ex;
    let k_u = material.k_u;
    let u = material.easy_axis;
    let ms = material.ms;

    let (bx, by, bz) = (b_ext[0], b_ext[1], b_ext[2]);

    let mut e_ex = 0.0;
    let mut e_an = 0.0;
    let mut e_zee = 0.0;

    for j in 0..ny {
        for i in 0..nx {
            let idx = grid.idx(i, j);
            let mij = m.data[idx];
            let (mx, my, mz) = (mij[0], mij[1], mij[2]);

            // Exchange (forward differences)
            if a != 0.0 {
                if i + 1 < nx {
                    let mip = m.data[grid.idx(i + 1, j)];
                    let dxm = [mip[0] - mx, mip[1] - my, mip[2] - mz];
                    let sq = dxm[0] * dxm[0] + dxm[1] * dxm[1] + dxm[2] * dxm[2];
                    e_ex += a * (sq / (dx * dx)) * v;
                }
                if j + 1 < ny {
                    let mjp = m.data[grid.idx(i, j + 1)];
                    let dym = [mjp[0] - mx, mjp[1] - my, mjp[2] - mz];
                    let sq = dym[0] * dym[0] + dym[1] * dym[1] + dym[2] * dym[2];
                    e_ex += a * (sq / (dy * dy)) * v;
                }
            }

            // Uniaxial anisotropy
            if k_u != 0.0 {
                let mdotu = mx * u[0] + my * u[1] + mz * u[2];
                e_an += k_u * (1.0 - mdotu * mdotu) * v;
            }

            // Zeeman
            if ms != 0.0 {
                let mdotb = mx * bx + my * by + mz * bz;
                e_zee -= ms * mdotb * v;
            }
        }
    }

    EnergyBreakdown {
        exchange: e_ex,
        anisotropy: e_an,
        zeeman: e_zee,
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
