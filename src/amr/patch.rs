// src/amr/patch.rs

use crate::amr::interp::sample_bilinear_unit;
use crate::amr::rect::Rect2i;
use crate::grid::Grid2D;
use crate::vec3::normalize;
use crate::vector_field::VectorField2D;

/// A single 2D AMR patch at some refinement ratio relative to the base grid.
///
/// `m` is stored on a *patch-local* uniform grid that includes ghost cells.
/// The interior (active) region is the physical patch; ghost cells provide stencil
/// values so we can reuse the same finite-difference operators unchanged.
pub struct Patch2D {
    /// Patch coverage in *coarse (base-grid) cell indices*.
    pub coarse_rect: Rect2i,

    /// Refinement ratio relative to the base grid (typically 2 for level-1 patches).
    pub ratio: usize,

    /// Number of ghost cells around the patch.
    pub ghost: usize,

    /// Patch-local grid including ghosts.
    pub grid: Grid2D,

    /// Magnetisation on the patch-local grid (including ghosts).
    pub m: VectorField2D,

    /// Flat indices of the interior (active) cells (excluding ghosts).
    pub active: Vec<usize>,

    /// Interior dimensions in *fine* cells (no ghosts).
    pub interior_nx: usize,
    pub interior_ny: usize,
}

impl Patch2D {
    /// Build a level-1 patch covering `coarse_rect` on the base grid.
    ///
    /// `ratio` is the refinement ratio relative to the base grid (usually 2).
    /// `ghost` is the number of ghost cells used by FD stencils.
    pub fn new(base_grid: &Grid2D, coarse_rect: Rect2i, ratio: usize, ghost: usize) -> Self {
        assert!(ratio >= 1, "ratio must be >= 1");
        assert!(
            coarse_rect.fits_in(base_grid.nx, base_grid.ny),
            "Patch coarse_rect {:?} must fit in base grid (nx={}, ny={})",
            coarse_rect,
            base_grid.nx,
            base_grid.ny
        );

        let interior_nx = coarse_rect.nx * ratio;
        let interior_ny = coarse_rect.ny * ratio;

        let nx_tot = interior_nx + 2 * ghost;
        let ny_tot = interior_ny + 2 * ghost;

        let dx = base_grid.dx / (ratio as f64);
        let dy = base_grid.dy / (ratio as f64);

        let grid = Grid2D::new(nx_tot, ny_tot, dx, dy, base_grid.dz);
        let m = VectorField2D::new(grid);

        // Precompute active indices for the interior region.
        let mut active = Vec::with_capacity(interior_nx * interior_ny);
        for j in ghost..(ghost + interior_ny) {
            for i in ghost..(ghost + interior_nx) {
                active.push(grid.idx(i, j));
            }
        }

        Self {
            coarse_rect,
            ratio,
            ghost,
            grid,
            m,
            active,
            interior_nx,
            interior_ny,
        }
    }

    #[inline]
    pub fn interior_i0(&self) -> usize {
        self.ghost
    }
    #[inline]
    pub fn interior_j0(&self) -> usize {
        self.ghost
    }
    #[inline]
    pub fn interior_i1(&self) -> usize {
        self.ghost + self.interior_nx
    }
    #[inline]
    pub fn interior_j1(&self) -> usize {
        self.ghost + self.interior_ny
    }

    /// Patch cell centre position in global physical coordinates.
    ///
    /// `i_patch`, `j_patch` are indices on the patch-local grid (including ghosts).
    ///
    /// We map patch-local indices to the corresponding *global fine-grid index*
    /// implied by the patch's coarse_rect and refinement ratio.
    #[inline]
    pub fn cell_center_xy(&self, i_patch: usize, j_patch: usize) -> (f64, f64) {
        let gi0 = (self.coarse_rect.i0 * self.ratio) as f64;
        let gj0 = (self.coarse_rect.j0 * self.ratio) as f64;

        let li = i_patch as f64 - self.ghost as f64;
        let lj = j_patch as f64 - self.ghost as f64;

        let x = (gi0 + li + 0.5) * self.grid.dx;
        let y = (gj0 + lj + 0.5) * self.grid.dy;
        (x, y)
    }

    /// Initialise the full patch (interior + ghosts) by sampling the coarse field.
    ///
    /// Use this when creating a new patch.
    pub fn fill_all_from_coarse(&mut self, coarse: &VectorField2D) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        for j in 0..ny {
            for i in 0..nx {
                let (x, y) = self.cell_center_xy(i, j);
                let v = sample_bilinear_unit(coarse, x, y);
                let idx = self.grid.idx(i, j);
                self.m.data[idx] = v;
            }
        }
    }

    /// Refill *only* ghost cells by sampling the coarse field.
    ///
    /// Call this at the start of each timestep (or at each RK stage later) to keep
    /// stencil values consistent across the coarse–fine interface.
    pub fn fill_ghosts_from_coarse(&mut self, coarse: &VectorField2D) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let gi0 = self.interior_i0();
        let gj0 = self.interior_j0();
        let gi1 = self.interior_i1();
        let gj1 = self.interior_j1();

        for j in 0..ny {
            for i in 0..nx {
                let is_interior = i >= gi0 && i < gi1 && j >= gj0 && j < gj1;
                if is_interior {
                    continue;
                }
                let (x, y) = self.cell_center_xy(i, j);
                let v = sample_bilinear_unit(coarse, x, y);
                let idx = self.grid.idx(i, j);
                self.m.data[idx] = v;
            }
        }
    }

    /// Restrict (average) fine interior values back to the coarse grid under this patch.
    ///
    /// This is the standard fine→coarse synchronisation step: for each coarse cell
    /// covered by the patch, average the corresponding ratio×ratio fine cells.
    pub fn restrict_to_coarse(&self, coarse: &mut VectorField2D) {
        let r = self.ratio;
        let g = self.ghost;

        for jc in 0..self.coarse_rect.ny {
            for ic in 0..self.coarse_rect.nx {
                let i_coarse = self.coarse_rect.i0 + ic;
                let j_coarse = self.coarse_rect.j0 + jc;

                // Average r×r fine cells.
                let mut sum = [0.0_f64; 3];
                for fj in 0..r {
                    for fi in 0..r {
                        let i_f = g + ic * r + fi;
                        let j_f = g + jc * r + fj;
                        let v = self.m.data[self.grid.idx(i_f, j_f)];
                        sum[0] += v[0];
                        sum[1] += v[1];
                        sum[2] += v[2];
                    }
                }
                let inv = 1.0 / ((r * r) as f64);
                let avg = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
                let dst = coarse.idx(i_coarse, j_coarse);
                coarse.data[dst] = normalize(avg);
            }
        }
    }

    /// Overwrite values into a global *uniform fine grid* (resolution = base*ratio) for IO/diagnostics.
    ///
    /// `fine` must have dimensions (base_nx*ratio, base_ny*ratio) and dx=base_dx/ratio.
    pub fn scatter_into_uniform_fine(&self, fine: &mut VectorField2D) {
        let g = self.ghost;
        let nx_int = self.interior_nx;
        let ny_int = self.interior_ny;

        let gi0 = self.coarse_rect.i0 * self.ratio;
        let gj0 = self.coarse_rect.j0 * self.ratio;

        for j in 0..ny_int {
            for i in 0..nx_int {
                let i_patch = g + i;
                let j_patch = g + j;
                let v = self.m.data[self.grid.idx(i_patch, j_patch)];

                let i_global = gi0 + i;
                let j_global = gj0 + j;
                let dst = fine.idx(i_global, j_global);
                fine.data[dst] = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_fill_does_not_touch_interior() {
        let base = Grid2D::new(8, 8, 1.0, 1.0, 1.0);
        let mut coarse = VectorField2D::new(base);
        coarse.set_uniform(0.0, 0.0, 1.0);

        let rect = Rect2i::new(2, 2, 3, 3);
        let mut p = Patch2D::new(&base, rect, 2, 1);
        p.fill_all_from_coarse(&coarse);

        // Overwrite interior with +x.
        for &idx in &p.active {
            p.m.data[idx] = [1.0, 0.0, 0.0];
        }

        p.fill_ghosts_from_coarse(&coarse);

        for &idx in &p.active {
            assert_eq!(p.m.data[idx], [1.0, 0.0, 0.0]);
        }
    }

    #[test]
    fn restrict_averages_back_to_coarse() {
        let base = Grid2D::new(8, 8, 1.0, 1.0, 1.0);
        let mut coarse = VectorField2D::new(base);
        coarse.set_uniform(0.0, 0.0, 1.0);

        let rect = Rect2i::new(2, 1, 2, 3);
        let mut p = Patch2D::new(&base, rect, 2, 1);
        p.fill_all_from_coarse(&coarse);

        // Set fine interior to +x so restriction should overwrite coarse cells.
        for &idx in &p.active {
            p.m.data[idx] = [1.0, 0.0, 0.0];
        }

        p.restrict_to_coarse(&mut coarse);

        for jc in 0..rect.ny {
            for ic in 0..rect.nx {
                let i = rect.i0 + ic;
                let j = rect.j0 + jc;
                assert_eq!(coarse.data[coarse.idx(i, j)], [1.0, 0.0, 0.0]);
            }
        }
    }
}
