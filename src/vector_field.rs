// ===============================
// src/vector_field.rs
// ===============================

use crate::grid::Grid2D;
use crate::vec3::normalize;

/// Simple nonuniformity metrics that can be computed cheaply from m and grid.
///
/// These are intended for state-based decisions (e.g. one-shot grid refinement),
/// without requiring extra demag calls.
#[derive(Debug, Clone, Copy, Default)]
pub struct FieldNonuniformityMetrics {
    /// Maximum nearest-neighbour angle (radians) over x/y neighbours.
    pub max_nn_angle_rad: f64,
    /// RMS nearest-neighbour angle (radians) over x/y neighbours.
    pub rms_nn_angle_rad: f64,
    /// Gradient proxy: max over cells of (|m(i+1,j)-m(i,j)| + |m(i,j+1)-m(i,j)|).
    pub max_grad: f64,
}

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
    ///
    /// NOTE: This does not normalise; callers should provide a unit vector if required.
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

    /// Set one cell (i,j) to the given vector (x,y,z), normalising to unit length.
    ///
    /// MuMax analogue: m.SetCell(i,j,0, vector(x,y,z)) for Nz=1.
    pub fn set_cell(&mut self, i: usize, j: usize, x: f64, y: f64, z: f64) {
        assert!(
            i < self.grid.nx,
            "set_cell: i={} out of bounds (nx={})",
            i,
            self.grid.nx
        );
        assert!(
            j < self.grid.ny,
            "set_cell: j={} out of bounds (ny={})",
            j,
            self.grid.ny
        );
        let idx = self.idx(i, j);
        self.data[idx] = normalize([x, y, z]);
    }

    /// Set one cell (i,j) to the given vector, normalising to unit length.
    pub fn set_cell_vec(&mut self, i: usize, j: usize, v: [f64; 3]) {
        self.set_cell(i, j, v[0], v[1], v[2]);
    }

    /// Set one cell by flat index, normalising to unit length.
    pub fn set_cell_idx(&mut self, idx: usize, v: [f64; 3]) {
        assert!(
            idx < self.data.len(),
            "set_cell_idx: idx={} out of bounds (len={})",
            idx,
            self.data.len()
        );
        self.data[idx] = normalize(v);
    }

    /// Checked variant: returns false if (i,j) out of bounds; otherwise sets the cell and returns true.
    pub fn set_cell_checked(&mut self, i: usize, j: usize, v: [f64; 3]) -> bool {
        if i >= self.grid.nx || j >= self.grid.ny {
            return false;
        }
        let idx = self.idx(i, j);
        self.data[idx] = normalize(v);
        true
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

    // ===============================
    // New: state metrics + resampling
    // ===============================

    /// Compute cheap nonuniformity metrics from the current state.
    ///
    /// No effective-field builds are required.
    pub fn nonuniformity_metrics(&self) -> FieldNonuniformityMetrics {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        if nx == 0 || ny == 0 {
            return FieldNonuniformityMetrics::default();
        }

        let mut max_angle = 0.0_f64;
        let mut sum_angle2 = 0.0_f64;
        let mut n_edges = 0usize;

        let mut max_grad = 0.0_f64;

        // Helper: clamp dot to [-1, 1] for acos stability.
        #[inline]
        fn clamp_dot(x: f64) -> f64 {
            if x > 1.0 {
                1.0
            } else if x < -1.0 {
                -1.0
            } else {
                x
            }
        }

        #[inline]
        fn norm3(v: [f64; 3]) -> f64 {
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
        }

        for j in 0..ny {
            for i in 0..nx {
                let idx = self.idx(i, j);
                let m0 = self.data[idx];

                let mut grad_here = 0.0_f64;

                // Right neighbour
                if i + 1 < nx {
                    let idx_r = self.idx(i + 1, j);
                    let mr = self.data[idx_r];

                    let dot = clamp_dot(m0[0] * mr[0] + m0[1] * mr[1] + m0[2] * mr[2]);
                    let ang = dot.acos();
                    max_angle = max_angle.max(ang);
                    sum_angle2 += ang * ang;
                    n_edges += 1;

                    let diff = [mr[0] - m0[0], mr[1] - m0[1], mr[2] - m0[2]];
                    grad_here += norm3(diff);
                }

                // Up neighbour
                if j + 1 < ny {
                    let idx_u = self.idx(i, j + 1);
                    let mu = self.data[idx_u];

                    let dot = clamp_dot(m0[0] * mu[0] + m0[1] * mu[1] + m0[2] * mu[2]);
                    let ang = dot.acos();
                    max_angle = max_angle.max(ang);
                    sum_angle2 += ang * ang;
                    n_edges += 1;

                    let diff = [mu[0] - m0[0], mu[1] - m0[1], mu[2] - m0[2]];
                    grad_here += norm3(diff);
                }

                if grad_here > max_grad {
                    max_grad = grad_here;
                }
            }
        }

        let rms_angle = if n_edges > 0 {
            (sum_angle2 / (n_edges as f64)).sqrt()
        } else {
            0.0
        };

        FieldNonuniformityMetrics {
            max_nn_angle_rad: max_angle,
            rms_nn_angle_rad: rms_angle,
            max_grad,
        }
    }

    /// Resample this field onto a new grid using bilinear interpolation (2D) and per-cell renormalisation.
    ///
    /// Assumptions:
    /// - Both grids share the same origin (0,0) at the lower-left corner.
    /// - Physical extents are intended to be comparable (e.g. one-shot refinement keeps Lx,Ly fixed).
    pub fn resample_to_grid(&self, new_grid: Grid2D) -> VectorField2D {
        let nx0 = self.grid.nx;
        let ny0 = self.grid.ny;

        let mut out = VectorField2D::new(new_grid);

        if nx0 == 0 || ny0 == 0 {
            return out;
        }

        // Helper: clamp to [0, n-1] and force alpha/beta=0 at boundaries.
        #[inline]
        fn clamp_pair(i0: isize, n: usize, frac: f64) -> (usize, usize, f64) {
            if n <= 1 {
                return (0, 0, 0.0);
            }
            if i0 <= 0 {
                return (0, 0, 0.0);
            }
            let max0 = (n - 1) as isize;
            if i0 >= max0 {
                return ((n - 1), (n - 1), 0.0);
            }
            let i0u = i0 as usize;
            let i1u = (i0u + 1).min(n - 1);
            (i0u, i1u, frac)
        }

        for j in 0..out.grid.ny {
            for i in 0..out.grid.nx {
                // New cell centre in physical coordinates.
                let x = (i as f64 + 0.5) * out.grid.dx;
                let y = (j as f64 + 0.5) * out.grid.dy;

                // Map to old grid “cell index space” where cell centres are at (k+0.5)*dx.
                let u = x / self.grid.dx - 0.5;
                let v = y / self.grid.dy - 0.5;

                let i0f = u.floor();
                let j0f = v.floor();

                let ax = u - i0f;
                let by = v - j0f;

                let (i0, i1, ax) = clamp_pair(i0f as isize, nx0, ax);
                let (j0, j1, by) = clamp_pair(j0f as isize, ny0, by);

                let v00 = self.data[self.idx(i0, j0)];
                let v10 = self.data[self.idx(i1, j0)];
                let v01 = self.data[self.idx(i0, j1)];
                let v11 = self.data[self.idx(i1, j1)];

                // Bilinear interpolation.
                let lerp = |a: [f64; 3], b: [f64; 3], t: f64| -> [f64; 3] {
                    [
                        a[0] * (1.0 - t) + b[0] * t,
                        a[1] * (1.0 - t) + b[1] * t,
                        a[2] * (1.0 - t) + b[2] * t,
                    ]
                };

                let v0 = lerp(v00, v10, ax);
                let v1 = lerp(v01, v11, ax);
                let vv = lerp(v0, v1, by);

                let out_idx = out.idx(i, j);
                out.data[out_idx] = normalize(vv);

            }
        }

        out
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
        assert!(
            v_mid[1].abs() > 0.1,
            "expected sizable my bump at wall center"
        );
        assert!(v_mid[0].abs() < 1e-6, "expected mx ~ 0 for Bloch(y) wall");

        // Norm check
        for &i in &[0usize, nx / 2, nx - 1] {
            let v = m.data[m.idx(i, 0)];
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn set_cell_normalizes_and_only_changes_one_cell() {
        let grid = Grid2D::new(4, 3, 1.0, 1.0, 1.0);
        let mut m = VectorField2D::new(grid);

        // Start uniform +z
        m.set_uniform(0.0, 0.0, 1.0);

        // Set a single cell with a non-unit vector
        m.set_cell(2, 1, 0.0, 2.0, 0.0);

        // That cell should now be (0,1,0)
        let v = m.data[m.idx(2, 1)];
        assert!((v[0] - 0.0).abs() < 1e-12);
        assert!((v[1] - 1.0).abs() < 1e-12);
        assert!((v[2] - 0.0).abs() < 1e-12);

        // All other cells should still be +z
        for j in 0..m.grid.ny {
            for i in 0..m.grid.nx {
                if i == 2 && j == 1 {
                    continue;
                }
                let w = m.data[m.idx(i, j)];
                assert!((w[0] - 0.0).abs() < 1e-12);
                assert!((w[1] - 0.0).abs() < 1e-12);
                assert!((w[2] - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn nonuniformity_is_zero_for_uniform_state() {
        let grid = Grid2D::new(16, 8, 1.0, 1.0, 1.0);
        let mut m = VectorField2D::new(grid);
        m.set_uniform(1.0, 0.0, 0.0);

        let met = m.nonuniformity_metrics();
        assert!(met.max_nn_angle_rad.abs() < 1e-12);
        assert!(met.rms_nn_angle_rad.abs() < 1e-12);
        assert!(met.max_grad.abs() < 1e-12);
    }

    #[test]
    fn resample_preserves_uniform_state() {
        let g0 = Grid2D::new(16, 8, 1.0, 1.0, 1.0);
        let mut m0 = VectorField2D::new(g0);
        m0.set_uniform(0.0, 1.0, 0.0);

        let g1 = Grid2D::new(32, 16, 0.5, 0.5, 1.0);
        let m1 = m0.resample_to_grid(g1);

        // Sample a few points
        for &(i, j) in &[(0usize, 0usize), (15, 7), (31, 15)] {
            let v = m1.data[m1.idx(i, j)];
            assert!((v[0] - 0.0).abs() < 1e-10);
            assert!((v[1] - 1.0).abs() < 1e-10);
            assert!((v[2] - 0.0).abs() < 1e-10);
        }
    }
}