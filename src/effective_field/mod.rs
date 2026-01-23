// src/effective_field/mod.rs

pub mod anisotropy;
pub mod exchange;
pub mod zeeman;
pub mod dmi;

use crate::grid::Grid2D;
use crate::params::{LLGParams, Material};
use crate::vector_field::VectorField2D;

/// Build total effective induction B_eff = B_ext + B_ex + B_ani.
pub fn build_h_eff(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    sim: &LLGParams,
    mat: &Material,
) {
    b_eff.set_uniform(0.0, 0.0, 0.0);

    zeeman::add_zeeman_field(b_eff, sim.b_ext);
    exchange::add_exchange_field(grid, m, b_eff, mat);
    anisotropy::add_uniaxial_anisotropy_field(m, b_eff, mat);

    if let Some(dmi) = mat.dmi {
        dmi::add_dmi_field(grid, m, b_eff, dmi, mat.ms);
    }
}
