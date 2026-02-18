// src/amr/hierarchy.rs

use crate::amr::patch::Patch2D;
use crate::amr::rect::Rect2i;
use crate::grid::Grid2D;
use crate::vector_field::VectorField2D;

/// Minimal 2-level AMR hierarchy (Stage 1):
/// - Level 0: uniform base grid over the whole domain
/// - Level 1: disjoint refined rectangular patches, refinement ratio = `ratio`
///
/// This is the smallest useful scaffold for validating coarse–fine coupling
/// without involving demag.
pub struct AmrHierarchy2D {
    pub base_grid: Grid2D,
    pub ratio: usize,
    pub ghost: usize,

    /// Coarse (level-0) magnetisation over the whole domain.
    pub coarse: VectorField2D,

    /// Level-1 patches.
    pub patches: Vec<Patch2D>,
}

impl AmrHierarchy2D {
    pub fn new(base_grid: Grid2D, coarse: VectorField2D, ratio: usize, ghost: usize) -> Self {
        assert_eq!(
            base_grid.n_cells(),
            coarse.data.len(),
            "coarse field must match base grid"
        );
        Self {
            base_grid,
            ratio,
            ghost,
            coarse,
            patches: Vec::new(),
        }
    }

    /// Add a new level-1 patch covering `coarse_rect`.
    ///
    /// The patch is initialised by sampling from the current coarse field.
    pub fn add_patch(&mut self, coarse_rect: Rect2i) {
        let mut p = Patch2D::new(&self.base_grid, coarse_rect, self.ratio, self.ghost);
        p.fill_all_from_coarse(&self.coarse);
        self.patches.push(p);
    }

    /// Refill all level-1 patch ghost cells from the current coarse field.
    pub fn fill_patch_ghosts(&mut self) {
        for p in &mut self.patches {
            p.fill_ghosts_from_coarse(&self.coarse);
        }
    }

    /// Restrict all patch interiors back to the coarse grid (fine→coarse sync).
    pub fn restrict_patches_to_coarse(&mut self) {
        for p in &self.patches {
            p.restrict_to_coarse(&mut self.coarse);
        }
    }

    /// Build a uniform fine-grid representation for diagnostics/IO.
    ///
    /// We follow the standard AMR visualisation trick:
    /// 1) interpolate (resample) the coarse field onto the finest uniform grid
    /// 2) overwrite with the computed fine patch values
    pub fn flatten_to_uniform_fine(&self) -> VectorField2D {
        let fine_grid = Grid2D::new(
            self.base_grid.nx * self.ratio,
            self.base_grid.ny * self.ratio,
            self.base_grid.dx / (self.ratio as f64),
            self.base_grid.dy / (self.ratio as f64),
            self.base_grid.dz,
        );
        let mut out = self.coarse.resample_to_grid(fine_grid);
        for p in &self.patches {
            p.scatter_into_uniform_fine(&mut out);
        }
        out
    }

    /// Replace the entire level-1 patch set, preserving fine values on overlaps.
    ///
    /// New patches are seeded from the current coarse field (via `add_patch`).
    /// Then, for any region where a new patch overlaps an old patch (in global fine
    /// indices), we copy the old fine values into the new patch so we do not lose
    /// refinement detail during regrids.
    pub fn replace_patches_preserve_overlap(&mut self, new_rects: Vec<Rect2i>) {
        // Move old patches out so we can rebuild self.patches cleanly.
        let old_patches: Vec<Patch2D> = std::mem::take(&mut self.patches);

        if new_rects.is_empty() {
            self.patches = Vec::new();
            return;
        }

        // Create new patches seeded from coarse.
        self.patches = Vec::with_capacity(new_rects.len());
        for r in new_rects {
            self.add_patch(r);
        }

        // Copy overlap regions (global fine index space) from old -> new.
        let r = self.ratio;

        for new_p in &mut self.patches {
            let new_rect = new_p.coarse_rect;
            let new_gi0 = new_rect.i0 * r;
            let new_gj0 = new_rect.j0 * r;
            let new_gi1 = new_gi0 + new_rect.nx * r;
            let new_gj1 = new_gj0 + new_rect.ny * r;
            let g_new = new_p.ghost;

            for old_p in &old_patches {
                let old_rect = old_p.coarse_rect;
                let old_gi0 = old_rect.i0 * r;
                let old_gj0 = old_rect.j0 * r;
                let old_gi1 = old_gi0 + old_rect.nx * r;
                let old_gj1 = old_gj0 + old_rect.ny * r;

                let oi0 = old_gi0.max(new_gi0);
                let oj0 = old_gj0.max(new_gj0);
                let oi1 = old_gi1.min(new_gi1);
                let oj1 = old_gj1.min(new_gj1);

                if oi1 <= oi0 || oj1 <= oj0 {
                    continue;
                }

                let g_old = old_p.ghost;

                for jg in oj0..oj1 {
                    for ig in oi0..oi1 {
                        let old_ip = g_old + (ig - old_gi0);
                        let old_jp = g_old + (jg - old_gj0);

                        let new_ip = g_new + (ig - new_gi0);
                        let new_jp = g_new + (jg - new_gj0);

                        let v = old_p.m.data[old_p.grid.idx(old_ip, old_jp)];
                        let dst = new_p.grid.idx(new_ip, new_jp);
                        new_p.m.data[dst] = v;
                    }
                }
            }
        }
    }

    /// Replace the current single patch with `new_rect`, preserving fine detail in the overlap region.
    ///
    /// Stage 2A assumes one patch. If there are 0 patches, we simply add it.
    /// If there are >1 patches, we fall back to clearing and adding the new patch.
    pub fn replace_single_patch_preserve_overlap(&mut self, new_rect: Rect2i) {
        self.replace_patches_preserve_overlap(vec![new_rect]);
    }
}
