// src/effective_field/dmi.rs
//
// Interfacial (Néel-type) Dzyaloshinskii–Moriya interaction (DMI)
// for thin films.
//
// Energy density:
//   E_DMI = D [ m_z (∇·m) - (m · ∇) m_z ]
//
// Effective field:
//   B_DMI = (2D / (μ0 M_s)) * ( ∂x m_z, ∂y m_z, -(∂x m_x + ∂y m_y) )

use crate::grid::Grid2D;
use crate::params::MU0;
use crate::vector_field::VectorField2D;

pub fn add_dmi_field(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    dmi: f64,
    ms: f64,
) {
    let nx = grid.nx;
    let ny = grid.ny;
    let dx = grid.dx;
    let dy = grid.dy;

    let prefactor = 2.0 * dmi / (MU0 * ms);

    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;

            // --- derivatives ---
            let dmz_dx = if i == 0 {
                (m.data[idx + 1][2] - m.data[idx][2]) / dx
            } else if i == nx - 1 {
                (m.data[idx][2] - m.data[idx - 1][2]) / dx
            } else {
                (m.data[idx + 1][2] - m.data[idx - 1][2]) / (2.0 * dx)
            };

            let dmz_dy = if j == 0 {
                (m.data[idx + nx][2] - m.data[idx][2]) / dy
            } else if j == ny - 1 {
                (m.data[idx][2] - m.data[idx - nx][2]) / dy
            } else {
                (m.data[idx + nx][2] - m.data[idx - nx][2]) / (2.0 * dy)
            };

            let dmx_dx = if i == 0 {
                (m.data[idx + 1][0] - m.data[idx][0]) / dx
            } else if i == nx - 1 {
                (m.data[idx][0] - m.data[idx - 1][0]) / dx
            } else {
                (m.data[idx + 1][0] - m.data[idx - 1][0]) / (2.0 * dx)
            };

            let dmy_dy = if j == 0 {
                (m.data[idx + nx][1] - m.data[idx][1]) / dy
            } else if j == ny - 1 {
                (m.data[idx][1] - m.data[idx - nx][1]) / dy
            } else {
                (m.data[idx + nx][1] - m.data[idx - nx][1]) / (2.0 * dy)
            };

            // --- DMI field contribution ---
            b_eff.data[idx][0] += prefactor * dmz_dx;
            b_eff.data[idx][1] += prefactor * dmz_dy;
            b_eff.data[idx][2] += -prefactor * (dmx_dx + dmy_dy);
        }
    }
}