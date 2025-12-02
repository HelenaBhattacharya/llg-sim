// src/effective_field/zeeman.rs

use crate::vector_field::VectorField2D;

/// Fill H_eff with a uniform external field for every cell.
pub fn add_zeeman_field(h_eff: &mut VectorField2D, h_ext: [f64; 3]) {
    h_eff.set_uniform(h_ext[0], h_ext[1], h_ext[2]);
}