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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_indexing_is_consistent() {
        let g = Grid2D::new(4, 3, 1.0, 1.0);
        // Check a few indices by hand
        assert_eq!(g.idx(0, 0), 0);
        assert_eq!(g.idx(1, 0), 1);
        assert_eq!(g.idx(0, 1), 4);
        assert_eq!(g.idx(3, 2), 11); // (j=2)*4 + i=3 = 11
        assert_eq!(g.n_cells(), 12);
    }
}