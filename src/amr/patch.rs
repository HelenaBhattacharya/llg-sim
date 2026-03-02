// src/amr/patch.rs

use crate::amr::interp::sample_bilinear_unit;
use crate::amr::rect::Rect2i;
use crate::geometry_mask::{MaskShape, assert_mask_len};
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

    /// Per-parent coarse-cell "material" flag for cells covered by this patch.
    ///
    /// Length = coarse_rect.nx * coarse_rect.ny, indexed in patch-local coarse coords:
    ///   parent_material[jc*coarse_rect.nx + ic]
    ///
    /// - Unmasked patches: all true.
    /// - Coarse-inherited mask: true where the parent coarse cell is inside the mask.
    /// - Analytical shape: true where *any* fine child of the parent is inside the shape.
    ///
    /// This allows `restrict_to_coarse()` and `scatter_into_uniform_fine()` to avoid
    /// writing into vacuum regions without needing to thread the mask everywhere yet.
    pub parent_material: Vec<bool>,

    /// Optional patch-local geometry mask on the *patch grid* (including ghosts).
    ///
    /// Semantics depend on how it was built:
    ///
    /// - **Coarse-inherited** (`rebuild_geom_mask_fine_from_coarse`): each fine cell
    ///   inherits its parent coarse cell's boolean — boundary is staircase at coarse dx.
    ///
    /// - **Analytical** (`rebuild_active_from_shape`): each fine cell is evaluated
    ///   independently against the analytical shape — boundary resolved at fine dx.
    ///
    /// Ghost cells whose parent coarse cell lies outside the base grid or outside the
    /// material are marked `false`.
    ///
    /// This is passed to geometry-aware field builders (e.g. `build_h_eff_masked_geom`)
    /// so patch-local exchange/DMI stencils treat material–vacuum boundaries consistently.
    pub geom_mask_fine: Option<Vec<bool>>,
}

impl Patch2D {
    #[inline]
    fn parent_idx(ic: usize, jc: usize, nx: usize) -> usize {
        jc * nx + ic
    }

    /// Return the patch-local geometry mask (including ghosts), if this patch is masked.
    #[inline]
    pub fn geom_mask_fine(&self) -> Option<&[bool]> {
        self.geom_mask_fine.as_deref()
    }

    // -----------------------------------------------------------------------
    // Coarse-inherited mask path (original, backward-compatible)
    // -----------------------------------------------------------------------

    /// Rebuild the patch-local geometry mask from the *coarse* (base-grid) mask.
    ///
    /// The mapping is done by assigning each patch cell (including ghosts) to its parent coarse cell.
    /// Cells whose parent lies outside the base grid are treated as vacuum (false).
    fn rebuild_geom_mask_fine_from_coarse(
        &mut self,
        base_grid: &Grid2D,
        coarse_mask: Option<&[bool]>,
    ) {
        if let Some(msk) = coarse_mask {
            assert_mask_len(msk, base_grid);

            let nx_p = self.grid.nx;
            let ny_p = self.grid.ny;
            let r = self.ratio as isize;
            let g = self.ghost as isize;
            let i0 = self.coarse_rect.i0 as isize;
            let j0 = self.coarse_rect.j0 as isize;

            let mut out = vec![false; self.grid.n_cells()];

            for j_patch in 0..ny_p {
                let lj = j_patch as isize - g;
                let jc_off = lj.div_euclid(r);
                let j_coarse = j0 + jc_off;

                for i_patch in 0..nx_p {
                    let li = i_patch as isize - g;
                    let ic_off = li.div_euclid(r);
                    let i_coarse = i0 + ic_off;

                    let idx_p = self.grid.idx(i_patch, j_patch);

                    if i_coarse < 0
                        || j_coarse < 0
                        || i_coarse >= base_grid.nx as isize
                        || j_coarse >= base_grid.ny as isize
                    {
                        out[idx_p] = false;
                        continue;
                    }

                    let ii = i_coarse as usize;
                    let jj = j_coarse as usize;
                    out[idx_p] = msk[base_grid.idx(ii, jj)];
                }
            }

            self.geom_mask_fine = Some(out);
        } else {
            self.geom_mask_fine = None;
        }
    }

    /// Rebuild the `active` index list and the per-parent-cell material flags
    /// from a *coarse* (base-grid) boolean mask.
    ///
    /// If `coarse_mask` is provided, only fine interior cells whose *parent coarse cell*
    /// is inside the mask are considered active.
    ///
    /// Mask is defined on the base (coarse) grid: `coarse_mask.len() == base_grid.n_cells()`.
    ///
    /// **Note:** This inherits the coarse staircase at curved boundaries.
    /// For analytical boundary resolution at the fine level, use
    /// [`rebuild_active_from_shape`] instead.
    pub fn rebuild_active_from_coarse_mask(
        &mut self,
        base_grid: &Grid2D,
        coarse_mask: Option<&[bool]>,
    ) {
        self.active.clear();

        // Reset parent-material flags (unmasked default).
        self.parent_material.fill(true);

        // Keep a patch-local copy of the geometry mask (including ghosts) so patch stepping
        // can use geometry-aware stencils when needed.
        self.rebuild_geom_mask_fine_from_coarse(base_grid, coarse_mask);

        let g = self.ghost;
        let r = self.ratio;
        let patch_nx = self.coarse_rect.nx;

        if let Some(msk) = coarse_mask {
            assert_mask_len(msk, base_grid);
            debug_assert!(
                self.coarse_rect.fits_in(base_grid.nx, base_grid.ny),
                "Patch coarse_rect must fit in base grid"
            );

            for jc in 0..self.coarse_rect.ny {
                for ic in 0..self.coarse_rect.nx {
                    let i_coarse = self.coarse_rect.i0 + ic;
                    let j_coarse = self.coarse_rect.j0 + jc;

                    let is_mat = msk[base_grid.idx(i_coarse, j_coarse)];
                    let pidx = Self::parent_idx(ic, jc, patch_nx);
                    self.parent_material[pidx] = is_mat;

                    if !is_mat {
                        continue;
                    }

                    // Include all r×r fine children of this coarse cell.
                    for fj in 0..r {
                        for fi in 0..r {
                            let i_f = g + ic * r + fi;
                            let j_f = g + jc * r + fj;
                            self.active.push(self.grid.idx(i_f, j_f));
                        }
                    }
                }
            }
        } else {
            // Unmasked: all interior cells are active.
            for j in g..(g + self.interior_ny) {
                for i in g..(g + self.interior_nx) {
                    self.active.push(self.grid.idx(i, j));
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Analytical shape path (new — resolves boundaries at fine dx)
    // -----------------------------------------------------------------------

    /// Patch cell centre position in **centered** physical coordinates.
    ///
    /// The centered system has (0,0) at the centre of the base grid domain,
    /// consistent with `geometry_mask::cell_center_xy_centered` and [`MaskShape`].
    ///
    /// Conversion from corner-origin: `x_centered = x_corner − ½ Lx`.
    #[inline]
    pub fn cell_center_xy_centered(
        &self,
        i_patch: usize,
        j_patch: usize,
        base_grid: &Grid2D,
    ) -> (f64, f64) {
        let (x, y) = self.cell_center_xy(i_patch, j_patch);
        let half_lx = 0.5 * base_grid.nx as f64 * base_grid.dx;
        let half_ly = 0.5 * base_grid.ny as f64 * base_grid.dy;
        (x - half_lx, y - half_ly)
    }

    /// Rebuild `active`, `parent_material`, and `geom_mask_fine` by evaluating an
    /// analytical [`MaskShape`] at every fine cell centre.
    ///
    /// This is the preferred path when a `MaskShape` is available, because it resolves
    /// curved boundaries at the patch's native fine resolution rather than inheriting
    /// the coarse staircase.
    ///
    /// Differences from [`rebuild_active_from_coarse_mask`]:
    /// - `geom_mask_fine` is evaluated per fine cell, not per parent block.
    /// - `active` includes only interior fine cells that are inside the shape.
    /// - `parent_material` is true if *any* fine child of the parent is material
    ///   (needed by `restrict_to_coarse` to know which parents to write).
    pub fn rebuild_active_from_shape(&mut self, base_grid: &Grid2D, shape: &MaskShape) {
        let nx_p = self.grid.nx;
        let ny_p = self.grid.ny;
        let half_lx = 0.5 * base_grid.nx as f64 * base_grid.dx;
        let half_ly = 0.5 * base_grid.ny as f64 * base_grid.dy;

        // 1. Build fine mask analytically at every patch cell (interior + ghosts).
        let mut fine_mask = vec![false; self.grid.n_cells()];
        for j_patch in 0..ny_p {
            for i_patch in 0..nx_p {
                let (x, y) = self.cell_center_xy(i_patch, j_patch);
                fine_mask[self.grid.idx(i_patch, j_patch)] =
                    shape.contains(x - half_lx, y - half_ly);
            }
        }

        // 2. Build active list from interior fine cells that are material.
        self.active.clear();
        let g = self.ghost;
        for j in g..(g + self.interior_ny) {
            for i in g..(g + self.interior_nx) {
                let idx = self.grid.idx(i, j);
                if fine_mask[idx] {
                    self.active.push(idx);
                }
            }
        }

        // 3. Build parent_material: true if ANY fine child is material.
        let r = self.ratio;
        let patch_nx = self.coarse_rect.nx;
        self.parent_material.fill(false);
        for jc in 0..self.coarse_rect.ny {
            for ic in 0..self.coarse_rect.nx {
                let pidx = Self::parent_idx(ic, jc, patch_nx);
                'children: for fj in 0..r {
                    for fi in 0..r {
                        let i_f = g + ic * r + fi;
                        let j_f = g + jc * r + fj;
                        if fine_mask[self.grid.idx(i_f, j_f)] {
                            self.parent_material[pidx] = true;
                            break 'children;
                        }
                    }
                }
            }
        }

        // 4. Store the fine mask.
        self.geom_mask_fine = Some(fine_mask);
    }

    // -----------------------------------------------------------------------
    // Construction, coordinate helpers, ghost fill, restriction
    // -----------------------------------------------------------------------

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

        // Precompute active indices for the interior region (unmasked default).
        let mut active = Vec::with_capacity(interior_nx * interior_ny);
        for j in ghost..(ghost + interior_ny) {
            for i in ghost..(ghost + interior_nx) {
                active.push(grid.idx(i, j));
            }
        }

        let parent_material = vec![true; coarse_rect.nx * coarse_rect.ny];

        Self {
            coarse_rect,
            ratio,
            ghost,
            grid,
            m,
            active,
            interior_nx,
            interior_ny,
            parent_material,
            geom_mask_fine: None,
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

    /// Patch cell centre position in global physical coordinates (corner-origin).
    ///
    /// `i_patch`, `j_patch` are indices on the patch-local grid (including ghosts).
    ///
    /// We map patch-local indices to the corresponding *global fine-grid index*
    /// implied by the patch's coarse_rect and refinement ratio.
    ///
    /// The origin (0,0) is at the bottom-left corner of the domain (corner-origin
    /// convention, used by `sample_bilinear`).  For centered coordinates consistent
    /// with `MaskShape`, use [`cell_center_xy_centered`].
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
    pub fn fill_all_from_coarse(&mut self, coarse: &VectorField2D) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        for j in 0..ny {
            for i in 0..nx {
                let (x, y) = self.cell_center_xy(i, j);
                let v = sample_bilinear_unit(coarse, x, y);
                let idx = self.grid.idx(i, j);
                self.m.data[idx] = v;

                if let Some(msk) = self.geom_mask_fine.as_deref() {
                    if !msk[idx] {
                        self.m.data[idx] = [0.0, 0.0, 0.0];
                    }
                }
            }
        }
    }

    /// Refill *only* ghost cells by sampling the coarse field.
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

                if let Some(msk) = self.geom_mask_fine.as_deref() {
                    if !msk[idx] {
                        self.m.data[idx] = [0.0, 0.0, 0.0];
                    }
                }
            }
        }
    }

    /// Restrict (average) fine interior values back to the coarse grid under this patch.
    ///
    /// Mask-aware behaviour:
    /// - Parent coarse cells with no material children are set to m=(0,0,0) and skipped.
    /// - When `geom_mask_fine` is present (analytical or coarse-inherited), the average
    ///   is taken over *material fine cells only*, so boundary parents with partial
    ///   coverage are restricted correctly.
    pub fn restrict_to_coarse(&self, coarse: &mut VectorField2D) {
        let r = self.ratio;
        let g = self.ghost;
        let patch_nx = self.coarse_rect.nx;
        let gm = self.geom_mask_fine.as_deref();

        for jc in 0..self.coarse_rect.ny {
            for ic in 0..self.coarse_rect.nx {
                let i_coarse = self.coarse_rect.i0 + ic;
                let j_coarse = self.coarse_rect.j0 + jc;
                let dst = coarse.idx(i_coarse, j_coarse);

                let pidx = Self::parent_idx(ic, jc, patch_nx);

                // If parent is entirely vacuum, enforce zero and skip averaging.
                if !self.parent_material[pidx] {
                    coarse.data[dst] = [0.0, 0.0, 0.0];
                    continue;
                }

                // Average fine cells, respecting per-cell fine mask when present.
                let mut sum = [0.0_f64; 3];
                let mut n_mat = 0usize;
                for fj in 0..r {
                    for fi in 0..r {
                        let i_f = g + ic * r + fi;
                        let j_f = g + jc * r + fj;
                        let idx_f = self.grid.idx(i_f, j_f);

                        if let Some(mask) = gm {
                            if !mask[idx_f] {
                                continue;
                            }
                        }

                        let v = self.m.data[idx_f];
                        sum[0] += v[0];
                        sum[1] += v[1];
                        sum[2] += v[2];
                        n_mat += 1;
                    }
                }

                if n_mat == 0 {
                    // All children turned out to be vacuum (shouldn't normally happen
                    // if parent_material is set correctly, but handle gracefully).
                    coarse.data[dst] = [0.0, 0.0, 0.0];
                } else {
                    let inv = 1.0 / (n_mat as f64);
                    let avg = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
                    coarse.data[dst] = normalize(avg);
                }
            }
        }
    }

    /// Overwrite values into a global *uniform fine grid* (resolution = base*ratio) for IO/diagnostics.
    ///
    /// Mask-aware behaviour:
    /// - When `geom_mask_fine` is present, only material fine cells are written
    ///   (per-cell resolution, not per-parent).
    /// - When absent, falls back to `parent_material` per-parent check.
    pub fn scatter_into_uniform_fine(&self, fine: &mut VectorField2D) {
        let g = self.ghost;
        let nx_int = self.interior_nx;
        let ny_int = self.interior_ny;
        let patch_nx = self.coarse_rect.nx;
        let gm = self.geom_mask_fine.as_deref();

        let gi0 = self.coarse_rect.i0 * self.ratio;
        let gj0 = self.coarse_rect.j0 * self.ratio;

        for j in 0..ny_int {
            for i in 0..nx_int {
                let i_patch = g + i;
                let j_patch = g + j;
                let idx_p = self.grid.idx(i_patch, j_patch);

                // Check fine-level mask if available; otherwise check parent.
                if let Some(mask) = gm {
                    if !mask[idx_p] {
                        continue;
                    }
                } else {
                    let ic = i / self.ratio;
                    let jc = j / self.ratio;
                    let pidx = Self::parent_idx(ic, jc, patch_nx);
                    if !self.parent_material[pidx] {
                        continue;
                    }
                }

                let v = self.m.data[idx_p];

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

    #[test]
    fn rebuild_active_respects_coarse_mask_and_sets_parent_material() {
        let base = Grid2D::new(4, 4, 1.0, 1.0, 1.0);

        // Mask out the left half of the domain.
        let mut mask = vec![true; base.n_cells()];
        for j in 0..base.ny {
            for i in 0..(base.nx / 2) {
                mask[base.idx(i, j)] = false;
            }
        }

        // Patch covers a 2x2 block that straddles masked/unmasked parents.
        let rect = Rect2i::new(1, 1, 2, 2);
        let mut p = Patch2D::new(&base, rect, 2, 1);

        // Unmasked active count = (2*2 coarse cells)*(2*2 children) = 16
        p.rebuild_active_from_coarse_mask(&base, None);
        assert_eq!(p.active.len(), 16);

        // With mask, one of the two coarse columns is masked out => active should be halved.
        p.rebuild_active_from_coarse_mask(&base, Some(mask.as_slice()));
        assert_eq!(p.active.len(), 8);

        // parent_material: across 2x2 parents => 4 entries, 2 true and 2 false in this setup
        let n_true = p.parent_material.iter().filter(|&&v| v).count();
        let n_false = p.parent_material.iter().filter(|&&v| !v).count();
        assert_eq!(n_true + n_false, rect.nx * rect.ny);
        assert_eq!(n_true, 2);
        assert_eq!(n_false, 2);
    }

    #[test]
    fn rebuild_active_from_shape_resolves_boundary_at_fine_dx() {
        // 8×8 coarse grid, 2nm cells → 16nm × 16nm domain.
        // Disk of radius 6nm centred at origin.
        // Coarse cell (3,3) has centre at (−1, −1) nm → inside.
        // Coarse cell (0,0) has centre at (−7, −7) nm → outside.
        // Boundary crosses through several coarse cells.
        let base = Grid2D::new(8, 8, 2e-9, 2e-9, 1e-9);
        let shape = MaskShape::Disk {
            center: (0.0, 0.0),
            radius: 6e-9,
        };

        // Patch covers the 4×4 block in the upper-right quadrant:
        // coarse cells (4,4)..(7,7) → corners at 8nm..16nm corner-origin → 0..8nm centered.
        // The disk boundary (r=6nm) cuts through this block.
        let rect = Rect2i::new(4, 4, 4, 4);
        let mut p = Patch2D::new(&base, rect, 2, 1);

        // Coarse-inherited path: each parent is wholly in or out.
        let coarse_mask = shape.to_mask(&base);
        p.rebuild_active_from_coarse_mask(&base, Some(&coarse_mask));
        let active_coarse = p.active.len();

        // Analytical path: per-fine-cell evaluation.
        p.rebuild_active_from_shape(&base, &shape);
        let active_fine = p.active.len();

        // The analytical mask must differ from the coarse-inherited one along the
        // boundary: some fine cells inside a "material" parent will be outside the
        // disk (or vice versa), so counts should differ.
        assert_ne!(
            active_coarse, active_fine,
            "analytical and coarse-inherited active counts should differ at a curved boundary"
        );

        // All active cells should genuinely lie inside the disk.
        let half_lx = 0.5 * base.nx as f64 * base.dx;
        let half_ly = 0.5 * base.ny as f64 * base.dy;
        for &idx in &p.active {
            let i = idx % p.grid.nx;
            let j = idx / p.grid.nx;
            let (x, y) = p.cell_center_xy(i, j);
            let xc = x - half_lx;
            let yc = y - half_ly;
            assert!(
                xc * xc + yc * yc <= 6e-9 * 6e-9 + 1e-20,
                "active cell at ({xc:.2e}, {yc:.2e}) lies outside the disk"
            );
        }
    }

    #[test]
    fn cell_center_xy_centered_consistent_with_geometry_mask() {
        use crate::geometry_mask::cell_center_xy_centered;

        // For a 1:1 patch covering the whole grid, fine cell centres must match
        // the base grid's centered cell centres.
        let base = Grid2D::new(8, 8, 5e-9, 5e-9, 1e-9);
        let rect = Rect2i::new(0, 0, 8, 8);
        let p = Patch2D::new(&base, rect, 1, 1);

        let g = p.ghost;
        for j in 0..base.ny {
            for i in 0..base.nx {
                let (xg, yg) = cell_center_xy_centered(&base, i, j);
                let (xp, yp) = p.cell_center_xy_centered(g + i, g + j, &base);
                assert!(
                    (xg - xp).abs() < 1e-20 && (yg - yp).abs() < 1e-20,
                    "mismatch at ({i},{j}): geom=({xg},{yg}) patch=({xp},{yp})"
                );
            }
        }
    }
}
