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

    /// Initialise a 180° Néel wall (rotation in x–z plane), centered at x0 with characteristic width.
    ///
    /// m_z(x) = tanh((x-x0)/width), m_x(x) = ±sech((x-x0)/width), m_y = 0
    pub fn init_neel_wall_x(&mut self, x0: f64, width: f64, chirality_sign: f64) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx = self.grid.dx;

        let s = if chirality_sign >= 0.0 { 1.0 } else { -1.0 };

        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx;
                let u = (x - x0) / width;

                let mz = u.tanh();
                let mx = s * (1.0 / u.cosh());

                let norm = (mx * mx + mz * mz).sqrt();
                let mx = mx / norm;
                let mz = mz / norm;

                let idx = self.idx(i, j);
                self.data[idx] = [mx, 0.0, mz];
            }
        }
    }

    /// Initialise a 180° Bloch wall (rotation in y–z plane), centered at x0 with characteristic width.
    ///
    /// m_z(x) = tanh((x-x0)/width), m_y(x) = ±sech((x-x0)/width), m_x = 0
    pub fn init_bloch_wall_y(&mut self, x0: f64, width: f64, chirality_sign: f64) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx = self.grid.dx;

        let s = if chirality_sign >= 0.0 { 1.0 } else { -1.0 };

        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) * dx;
                let u = (x - x0) / width;

                let mz = u.tanh();
                let my = s * (1.0 / u.cosh());

                let norm = (my * my + mz * mz).sqrt();
                let my = my / norm;
                let mz = mz / norm;

                let idx = self.idx(i, j);
                self.data[idx] = [0.0, my, mz];
            }
        }
    }

    /// Backwards-compatible initializer used across the codebase.
    ///
    /// NOTE: despite its historical name, this currently initialises a Néel-type wall (mx bump)
    /// to preserve existing behaviour in earlier benchmarks and movie runs.
    ///
    /// If you want a true Bloch wall for DMI-validation (my bump), use `init_bloch_wall_y(...)`.
    pub fn init_bloch_wall(&mut self, x0: f64, width: f64) {
        self.init_neel_wall_x(x0, width, 1.0);
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
        m.init_bloch_wall(x0, width); // backwards-compatible Néel wall

        let left_m = m.data[m.idx(0, 0)];
        let right_m = m.data[m.idx(nx - 1, 0)];
        assert!(left_m[2] * right_m[2] < 0.0);

        for &i in &[0usize, nx / 2, nx - 1] {
            let v = m.data[m.idx(i, 0)];
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn bloch_wall_y_has_nonzero_my_and_unit_norm() {
        let nx = 64;
        let ny = 1;
        let dx = 1.0;
        let dy = 1.0;

        let grid = Grid2D::new(nx, ny, dx, dy, 1.0);
        let mut m = VectorField2D::new(grid);

        let x0 = 0.5 * nx as f64 * dx;
        let width = 5.0 * dx;
        m.init_bloch_wall_y(x0, width, 1.0);

        // Check centre has my bump and mx ~ 0
        let v_mid = m.data[m.idx(nx / 2, 0)];
        assert!(v_mid[1].abs() > 0.1, "expected sizable my bump at wall center");
        assert!(v_mid[0].abs() < 1e-6, "expected mx ~ 0 for Bloch(y) wall");

        // Norm check
        for &i in &[0usize, nx / 2, nx - 1] {
            let v = m.data[m.idx(i, 0)];
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }
}
