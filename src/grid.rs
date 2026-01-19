// src/grid.rs

/// Simple 2D finite-difference grid with explicit thickness `dz`.
#[derive(Debug, Clone, Copy)]
pub struct Grid2D {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl Grid2D {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self { nx, ny, dx, dy, dz }
    }

    pub fn n_cells(&self) -> usize {
        self.nx * self.ny
    }

    pub fn cell_volume(&self) -> f64 {
        self.dx * self.dy * self.dz
    }

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
        let g = Grid2D::new(4, 3, 1.0, 1.0, 2.0);
        assert_eq!(g.idx(0, 0), 0);
        assert_eq!(g.idx(1, 0), 1);
        assert_eq!(g.idx(0, 1), 4);
        assert_eq!(g.idx(3, 2), 11);
        assert_eq!(g.n_cells(), 12);
        assert!((g.cell_volume() - 2.0).abs() < 1e-12);
    }
}