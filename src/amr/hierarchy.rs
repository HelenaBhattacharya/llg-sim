// src/amr/hierarchy.rs

use crate::amr::patch::Patch2D;
use crate::amr::rect::Rect2i;
use crate::geometry_mask::{Mask2D, assert_mask_len};
use crate::grid::Grid2D;
use crate::vector_field::VectorField2D;

/// Minimal 2-level AMR hierarchy:
/// - Level 0: uniform base grid over the whole domain
/// - Level 1: disjoint refined rectangular patches, refinement ratio = `ratio`
pub struct AmrHierarchy2D {
    pub base_grid: Grid2D,
    pub ratio: usize,
    pub ghost: usize,

    /// Optional geometry mask on the *base (coarse) grid*.
    pub geom_mask: Option<Mask2D>,

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
            geom_mask: None,
            coarse,
            patches: Vec::new(),
        }
    }

    /// Set (or replace) the geometry mask on the base grid.
    pub fn set_geom_mask(&mut self, mask: Mask2D) {
        assert_mask_len(&mask, &self.base_grid);
        self.geom_mask = Some(mask);

        // Update patch active sets + per-parent material flags.
        let gm = self.geom_mask.as_deref();
        for p in &mut self.patches {
            p.rebuild_active_from_coarse_mask(&self.base_grid, gm);
        }
    }

    /// Clear any geometry mask.
    pub fn clear_geom_mask(&mut self) {
        self.geom_mask = None;

        // Revert patches to unmasked behaviour.
        for p in &mut self.patches {
            p.rebuild_active_from_coarse_mask(&self.base_grid, None);
        }
    }

    /// Borrow the geometry mask as a slice (base grid), if present.
    #[inline]
    pub fn geom_mask(&self) -> Option<&[bool]> {
        self.geom_mask.as_deref()
    }

    /// True if a geometry mask is present.
    #[inline]
    pub fn has_geom_mask(&self) -> bool {
        self.geom_mask.is_some()
    }

    /// Construct a uniform-fine mask (resolution = base*ratio) by replicating each
    /// coarse cell's mask value into its ratio×ratio fine children.
    pub fn build_uniform_fine_mask(&self) -> Option<Mask2D> {
        let m0 = self.geom_mask.as_ref()?;
        assert_mask_len(m0, &self.base_grid);

        let r = self.ratio;
        let fine_nx = self.base_grid.nx * r;
        let fine_ny = self.base_grid.ny * r;
        let mut out = vec![false; fine_nx * fine_ny];

        for j in 0..self.base_grid.ny {
            for i in 0..self.base_grid.nx {
                if !m0[self.base_grid.idx(i, j)] {
                    continue;
                }
                let fi0 = i * r;
                let fj0 = j * r;
                for fj in 0..r {
                    for fi in 0..r {
                        out[(fj0 + fj) * fine_nx + (fi0 + fi)] = true;
                    }
                }
            }
        }

        Some(out)
    }

    /// Add a new level-1 patch covering `coarse_rect`.
    ///
    /// The patch is initialised by sampling from the current coarse field.
    pub fn add_patch(&mut self, coarse_rect: Rect2i) {
        let mut p = Patch2D::new(&self.base_grid, coarse_rect, self.ratio, self.ghost);

        // Respect geometry mask (updates active + parent_material + patch-local geom_mask_fine).
        p.rebuild_active_from_coarse_mask(&self.base_grid, self.geom_mask.as_deref());

        // Initialise the patch (interior + ghosts) by sampling from the current coarse field.
        // This will also enforce the patch-local geometry mask when present.
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
    ///
    /// This is mask-aware via Patch2D::parent_material.
    pub fn restrict_patches_to_coarse(&mut self) {
        for p in &self.patches {
            p.restrict_to_coarse(&mut self.coarse);
        }
    }

    /// Build a uniform fine-grid representation for diagnostics/IO.
    ///
    /// 1) resample coarse -> finest uniform
    /// 2) overwrite with patch interiors (mask-aware scatter)
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
    pub fn replace_patches_preserve_overlap(&mut self, new_rects: Vec<Rect2i>) {
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

    pub fn replace_single_patch_preserve_overlap(&mut self, new_rect: Rect2i) {
        self.replace_patches_preserve_overlap(vec![new_rect]);
    }
}
