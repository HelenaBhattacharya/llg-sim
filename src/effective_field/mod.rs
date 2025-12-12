// src/effective_field/mod.rs

pub mod zeeman;
pub mod exchange;
pub mod anisotropy;  // <-- add this

use crate::grid::Grid2D;
use crate::vector_field::VectorField2D;
use crate::params::{LLGParams, Material};

/// Build the total effective field H_eff = H_Zeeman + H_exch + H_ani.
pub fn build_h_eff(
    grid: &Grid2D,
    m: &VectorField2D,
    h_eff: &mut VectorField2D,
    sim: &LLGParams,
    mat: &Material,
) {
    // Start from zero field
    h_eff.set_uniform(0.0, 0.0, 0.0);

    // Zeeman
    zeeman::add_zeeman_field(h_eff, sim.h_ext);

    // Exchange
    exchange::add_exchange_field(grid, m, h_eff, mat.a_ex);

    // Uniaxial anisotropy
    anisotropy::add_uniaxial_anisotropy_field(m, h_eff, mat);
}