// src/vector_field.rs

use crate::grid::Grid2D;

/// Magnetisation / vector field defined on a 2D grid.
/// Each cell stores (x, y, z).
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
            data: vec![[0.0, 0.0, 1.0]; n],
        }
    }

    /// Set all cells to the same vector (x, y, z).
    pub fn set_uniform(&mut self, x: f64, y: f64, z: f64) {
        for cell in &mut self.data {
            *cell = [x, y, z];
        }
    }

    /// Flat index for (i, j).
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        self.grid.idx(i, j)
    }

    /// Initialise a simple 180° Bloch wall along x.
    pub fn init_bloch_wall(&mut self, x0: f64, width: f64) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx = self.grid.dx;

        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx;
                let u = (x - x0) / width;

                let mz = u.tanh();
                let mx = 1.0 / u.cosh();

                let norm = (mx * mx + mz * mz).sqrt();
                let mx = mx / norm;
                let mz = mz / norm;

                let idx = self.idx(i, j);
                self.data[idx] = [mx, 0.0, mz];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;

    #[test]
    fn bloch_wall_has_opposite_mz_at_edges_and_unit_norm() {
        let nx = 64;
        let ny = 1;
        let dx = 1.0;
        let dy = 1.0;

        let grid = Grid2D::new(nx, ny, dx, dy, 1.0);
        let mut m = VectorField2D::new(grid);

        let x0 = 0.5 * nx as f64 * dx;
        let width = 5.0 * dx;
        m.init_bloch_wall(x0, width);

        let left_m = m.data[m.idx(0, 0)];
        let right_m = m.data[m.idx(nx - 1, 0)];
        assert!(left_m[2] * right_m[2] < 0.0);

        for &i in &[0usize, nx / 2, nx - 1] {
            let v = m.data[m.idx(i, 0)];
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }
}