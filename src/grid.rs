// src/grid.rs

/// Simple 2D finite-difference grid.
#[derive(Debug, Clone, Copy)]
pub struct Grid2D {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
}

impl Grid2D {
    /// Create a new 2D grid with nx × ny cells and spacings dx, dy.
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        Self { nx, ny, dx, dy }
    }

    /// Total number of cells.
    pub fn n_cells(&self) -> usize {
        self.nx * self.ny
    }

    /// Convert (i, j) indices to a flat index into a 1D array.
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.nx && j < self.ny);
        j * self.nx + i
    }
}