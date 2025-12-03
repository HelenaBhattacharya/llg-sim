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

    /// Initialise a simple 180° Bloch wall along x.
    ///
    /// - `x0`   : wall centre position (same units as `grid.dx`, i.e. metres)
    /// - `width`: characteristic wall width
    ///
    /// m rotates in the x–z plane: mx in-plane, mz out-of-plane.
    pub fn init_bloch_wall(&mut self, x0: f64, width: f64) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx = self.grid.dx;

        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx;
                let u = (x - x0) / width;

                // Standard Bloch profile: tanh/cosh
                let mz = u.tanh();       // goes from -1 to +1
                let mx = 1.0 / u.cosh(); // in-plane component

                // Normalise to |m| = 1 (numerically)
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

        let grid = Grid2D::new(nx, ny, dx, dy);
        let mut m = VectorField2D::new(grid);

        let x0 = 0.5 * nx as f64 * dx;
        let width = 5.0 * dx;
        m.init_bloch_wall(x0, width);

        // Leftmost and rightmost cells should have opposite mz sign
        let left_m = m.data[m.idx(0, 0)];
        let right_m = m.data[m.idx(nx - 1, 0)];
        assert!(
            left_m[2] * right_m[2] < 0.0,
            "mz at edges should have opposite sign (left={}, right={})",
            left_m[2],
            right_m[2]
        );

        // Norm should be ~1 at a few sample points
        for &i in &[0usize, nx / 2, nx - 1] {
            let v = m.data[m.idx(i, 0)];
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "norm at i={} not ~1 (got {})",
                i,
                norm
            );
        }
    }
}