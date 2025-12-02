// src/effective_field/exchange.rs

use crate::grid::Grid2D;
use crate::vector_field::VectorField2D;

/// Very simple placeholder: in future, compute H_exch from micromagnetic Aex and M(x).
pub fn add_exchange_field(
    _grid: &Grid2D,
    _m: &VectorField2D,
    _h_eff: &mut VectorField2D,
    _a_ex: f64,
) {
    // TODO: implement discrete Laplacian-based exchange field
}