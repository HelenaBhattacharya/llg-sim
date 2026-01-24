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
//
// Notes:
// - D is in J/m^2 (interfacial DMI).
// - Uses central differences in the interior and one-sided differences at boundaries.
// - Safe for nx==1 or ny==1 (derivatives become 0 in that direction).

use crate::grid::Grid2D;
use crate::params::MU0;
use crate::vector_field::VectorField2D;

pub fn add_dmi_field(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    dmi: f64, // J/m^2
    ms: f64,  // A/m
) {
    let nx = grid.nx;
    let ny = grid.ny;
    let dx = grid.dx;
    let dy = grid.dy;

    // Prefactor to convert DMI into an effective induction B (Tesla)
    let prefactor = 2.0 * dmi / (MU0 * ms);

    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;

            // --- derivatives (robust for nx==1 or ny==1) ---

            // ∂x m_z
            let dmz_dx = if nx == 1 {
                0.0
            } else if i == 0 {
                (m.data[idx + 1][2] - m.data[idx][2]) / dx
            } else if i == nx - 1 {
                (m.data[idx][2] - m.data[idx - 1][2]) / dx
            } else {
                (m.data[idx + 1][2] - m.data[idx - 1][2]) / (2.0 * dx)
            };

            // ∂y m_z
            let dmz_dy = if ny == 1 {
                0.0
            } else if j == 0 {
                (m.data[idx + nx][2] - m.data[idx][2]) / dy
            } else if j == ny - 1 {
                (m.data[idx][2] - m.data[idx - nx][2]) / dy
            } else {
                (m.data[idx + nx][2] - m.data[idx - nx][2]) / (2.0 * dy)
            };

            // ∂x m_x
            let dmx_dx = if nx == 1 {
                0.0
            } else if i == 0 {
                (m.data[idx + 1][0] - m.data[idx][0]) / dx
            } else if i == nx - 1 {
                (m.data[idx][0] - m.data[idx - 1][0]) / dx
            } else {
                (m.data[idx + 1][0] - m.data[idx - 1][0]) / (2.0 * dx)
            };

            // ∂y m_y
            let dmy_dy = if ny == 1 {
                0.0
            } else if j == 0 {
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