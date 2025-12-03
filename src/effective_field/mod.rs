// src/effective_field/mod.rs

pub mod zeeman;
pub mod exchange;

use crate::grid::Grid2D;
use crate::vector_field::VectorField2D;
use crate::params::{LLGParams, Material};

/// Build the total effective field H_eff = H_Zeeman + H_exch (for now).
///
/// - `grid`   : spatial grid (cell sizes, dimensions)
/// - `m`      : magnetisation field
/// - `h_eff`  : output effective field (overwritten)
/// - `sim`    : LLG / simulation parameters (contains h_ext)
/// - `mat`    : material parameters (contains a_ex, etc.)
pub fn build_h_eff(
    grid: &Grid2D,
    m: &VectorField2D,
    h_eff: &mut VectorField2D,
    sim: &LLGParams,
    mat: &Material,
) {
    // Start from zero field
    h_eff.set_uniform(0.0, 0.0, 0.0);

    // Zeeman from simulation parameters
    zeeman::add_zeeman_field(h_eff, sim.h_ext);

    // Exchange from material parameters
    exchange::add_exchange_field(grid, m, h_eff, mat.a_ex);
}