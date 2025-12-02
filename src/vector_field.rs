// src/vector_field.rs

use crate::grid::Grid2D;

/// Magnetisation field defined on a 2D grid.
/// Each cell stores (mx, my, mz).
pub struct VectorField2D {
    pub grid: Grid2D,
    pub data: Vec<[f64; 3]>,
}

impl VectorField2D {
    /// Create a new field on the given grid, initialised along +z.
    pub fn new(grid: Grid2D) -> Self {
        let n = grid.n_cells();
        Self {
            grid,
            data: vec![[0.0, 0.0, 1.0]; n], // initial M along +z
        }
    }

    /// Set all cells to the same magnetisation (mx, my, mz).
    pub fn set_uniform(&mut self, mx: f64, my: f64, mz: f64) {
        for cell in &mut self.data {
            *cell = [mx, my, mz];
        }
    }

    /// Get the flat index in `data` for grid indices (i, j).
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        self.grid.idx(i, j)
    }
}