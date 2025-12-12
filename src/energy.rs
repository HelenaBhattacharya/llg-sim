// src/energy.rs

use crate::grid::Grid2D;
use crate::vector_field::VectorField2D;
use crate::params::Material;

/// Energy contributions in "toy" units, but coming from the same
/// functional as the effective field:
///
///   E_ex  ~ a_ex * ∫ |∇m|^2 dV
///   E_an  ~ k_u  * ∫ [1 - (m·u)^2] dV
///   E_zee ~ - M_s ∫ H_ext · m dV
///
/// where a_ex, k_u, M_s are the parameters stored in `Material`.
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

/// Compute exchange + anisotropy + Zeeman energies on the grid.
///
/// This uses simple one–sided finite differences for |∇m|^2 and
/// includes the cell area dx*dy so the energy scales with the domain size.
pub fn compute_energy(
    grid: &Grid2D,
    m: &VectorField2D,
    material: &Material,
    h_ext: [f64; 3],
) -> EnergyBreakdown {
    let nx = grid.nx;
    let ny = grid.ny;
    let dx = grid.dx;
    let dy = grid.dy;

    let cell_area = dx * dy; // 2D film, thickness = 1 in toy units

    let a_ex = material.a_ex;
    let k_u = material.k_u;
    let u = material.easy_axis;
    let m_s = material.ms;

    let (hx, hy, hz) = (h_ext[0], h_ext[1], h_ext[2]);

    let mut e_ex = 0.0;
    let mut e_an = 0.0;
    let mut e_zee = 0.0;

    for j in 0..ny {
        for i in 0..nx {
            let idx = grid.idx(i, j);
            let mij = m.data[idx];
            let (mx, my, mz) = (mij[0], mij[1], mij[2]);

            // ---------- exchange: |∇m|^2 ----------
            // forward differences in x,y with clamped boundaries
            let ip = if i + 1 < nx { i + 1 } else { i };
            let jp = if j + 1 < ny { j + 1 } else { j };

            let m_ip = m.data[grid.idx(ip, j)];
            let m_jp = m.data[grid.idx(i, jp)];

            let dmdx = [
                (m_ip[0] - mx) / dx,
                (m_ip[1] - my) / dx,
                (m_ip[2] - mz) / dx,
            ];
            let dmdy = [
                (m_jp[0] - mx) / dy,
                (m_jp[1] - my) / dy,
                (m_jp[2] - mz) / dy,
            ];

            let grad_sq = (0..3)
                .map(|c| dmdx[c] * dmdx[c] + dmdy[c] * dmdy[c])
                .sum::<f64>();

            e_ex += a_ex * grad_sq * cell_area;

            // ---------- uniaxial anisotropy ----------
            // E_an = k_u * (1 - (m·u)^2)
            let mdotu = mx * u[0] + my * u[1] + mz * u[2];
            e_an += k_u * (1.0 - mdotu * mdotu) * cell_area;

            // ---------- Zeeman ----------
            // E_zee = - M_s * H_ext · m  (toy units)
            let mdoth = mx * hx + my * hy + mz * hz;
            e_zee -= m_s * mdoth * cell_area;
        }
    }

    EnergyBreakdown {
        exchange: e_ex,
        anisotropy: e_an,
        zeeman: e_zee,
    }
}

/// Backwards-compatible helper: just the total energy.
///
/// NOTE: signature changed compared to your old version: we now also
/// need `h_ext` to include Zeeman energy.
pub fn compute_total_energy(
    grid: &Grid2D,
    m: &VectorField2D,
    material: &Material,
    h_ext: [f64; 3],
) -> f64 {
    compute_energy(grid, m, material, h_ext).total()
}