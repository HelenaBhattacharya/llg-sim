// src/effective_field/zeeman.rs

use crate::vector_field::VectorField2D;

/// Add a uniform external induction B_ext (Tesla) to every cell.
pub fn add_zeeman_field(b_eff: &mut VectorField2D, b_ext: [f64; 3]) {
    b_eff.set_uniform(b_ext[0], b_ext[1], b_ext[2]);
}