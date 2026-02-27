// src/amr/hierarchy.rs

use crate::amr::patch::Patch2D;
use crate::amr::rect::Rect2i;
use crate::amr::interp::sample_bilinear;
use crate::geometry_mask::{Mask2D, assert_mask_len};
use crate::grid::Grid2D;
use crate::vector_field::VectorField2D;

/// Block-structured AMR hierarchy (currently supports nested multi-level patches).
///
/// Level indexing and coordinates:
/// - Level 0: uniform base grid over the whole domain.
/// - Level L>=1: disjoint refined rectangular patches.
///
/// All patch rectangles (`Rect2i`) are expressed in **base-grid (level-0) cell indices**.
/// A level-L patch uses an effective refinement ratio `ratio^L` relative to the base grid.
///
/// Notes:
/// - This file provides the multi-level *data structure* (nested patch sets) and
///   level-aware IO/diagnostic flattening.
/// - Time stepping / ghost fill between levels is handled in `stepper.rs`.
/// - For now, level-1 patches are kept in `patches` for backward compatibility.
///   Additional levels live in `patches_l2plus`.
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

    /// Level-2+ patches.
    ///
    /// Indexing: `patches_l2plus[0]` is level 2, `patches_l2plus[1]` is level 3, etc.
    /// Each level-L patch uses an effective refinement ratio `ratio^L`.
    pub patches_l2plus: Vec<Vec<Patch2D>>,
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
            patches_l2plus: Vec::new(),
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
        for lvl in &mut self.patches_l2plus {
            for p in lvl {
                p.rebuild_active_from_coarse_mask(&self.base_grid, gm);
            }
        }
    }

    /// Clear any geometry mask.
    pub fn clear_geom_mask(&mut self) {
        self.geom_mask = None;

        // Revert patches to unmasked behaviour.
        for p in &mut self.patches {
            p.rebuild_active_from_coarse_mask(&self.base_grid, None);
        }
        for lvl in &mut self.patches_l2plus {
            for p in lvl {
                p.rebuild_active_from_coarse_mask(&self.base_grid, None);
            }
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

    /// Construct a uniform-fine mask (resolution = base*finest_ratio_total) by replicating each
    /// coarse cell's mask value into its ratio×ratio fine children.
    pub fn build_uniform_fine_mask(&self) -> Option<Mask2D> {
        let m0 = self.geom_mask.as_ref()?;
        assert_mask_len(m0, &self.base_grid);

        // Match the resolution of `flatten_to_uniform_fine()` so callers can safely use
        // this mask to zero vacuum cells on the composite field even when L2+ patches exist.
        let r = self.finest_ratio_total();
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

    #[inline]
    fn ratio_pow(&self, level: usize) -> usize {
        // ratio^level, with level=0 -> 1
        let mut r = 1usize;
        for _ in 0..level {
            r = r.saturating_mul(self.ratio);
        }
        r
    }

    #[inline]
    fn rect_contains(a: Rect2i, b: Rect2i) -> bool {
        b.i0 >= a.i0 && b.j0 >= a.j0 && b.i1() <= a.i1() && b.j1() <= a.j1()
    }

    #[inline]
    fn patches_at_level(&self, level: usize) -> Option<&[Patch2D]> {
        match level {
            0 => None,
            1 => Some(&self.patches),
            l if l >= 2 => {
                let idx = l - 2;
                self.patches_l2plus.get(idx).map(|v| v.as_slice())
            }
            _ => None,
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn patches_at_level_mut(&mut self, level: usize) -> Option<&mut Vec<Patch2D>> {
        match level {
            0 => None,
            1 => Some(&mut self.patches),
            l if l >= 2 => {
                let idx = l - 2;
                self.patches_l2plus.get_mut(idx)
            }
            _ => None,
        }
    }

    fn ensure_level_storage(&mut self, level: usize) {
        if level <= 1 {
            return;
        }
        let need = level - 1; // number of entries needed in patches_l2plus
        while self.patches_l2plus.len() < need {
            self.patches_l2plus.push(Vec::new());
        }
    }

    /// Finest refinement ratio currently present among all patch levels.
    /// Returns 1 if no patches exist.
    pub fn finest_ratio_total(&self) -> usize {
        let mut max_level: usize = 0;
        if !self.patches.is_empty() {
            max_level = 1;
        }
        for (k, lvl) in self.patches_l2plus.iter().enumerate() {
            if !lvl.is_empty() {
                max_level = max_level.max(k + 2);
            }
        }
        self.ratio_pow(max_level)
    }

    /// Add a new patch at a specific refinement level.
    ///
    /// - `level=1` adds to `self.patches` (ratio = `ratio^1`).
    /// - `level>=2` adds to `self.patches_l2plus[level-2]` (ratio = `ratio^level`).
    ///
    /// All rectangles are expressed in base-grid indices. For `level>=2`, we enforce
    /// a simple nesting constraint: the new rect must be contained in at least one
    /// patch on the previous level.
    pub fn add_patch_level(&mut self, level: usize, coarse_rect: Rect2i) {
        assert!(level >= 1, "level must be >= 1");

        if level >= 2 {
            if let Some(parent) = self.patches_at_level(level - 1) {
                let mut ok = false;
                for p in parent {
                    if Self::rect_contains(p.coarse_rect, coarse_rect) {
                        ok = true;
                        break;
                    }
                }
                assert!(ok, "level-{level} patch must be contained within a level-{} patch", level - 1);
            } else {
                panic!("cannot add level-{level} patch without an existing level-{}", level - 1);
            }
        }

        self.ensure_level_storage(level);

        let r_total = self.ratio_pow(level);
        let mut p = Patch2D::new(&self.base_grid, coarse_rect, r_total, self.ghost);

        // Respect geometry mask (updates active + parent_material + patch-local geom_mask_fine).
        p.rebuild_active_from_coarse_mask(&self.base_grid, self.geom_mask.as_deref());

        // Initialise patch by sampling from the current coarse field.
        // NOTE: for level>=2 this does not yet incorporate level-(L-1) fine detail;
        // overlap preservation during `replace_*` operations handles continuity.
        p.fill_all_from_coarse(&self.coarse);

        if level == 1 {
            self.patches.push(p);
        } else {
            self.patches_l2plus[level - 2].push(p);
        }
    }

    pub fn add_patch(&mut self, coarse_rect: Rect2i) {
        self.add_patch_level(1, coarse_rect);
    }

    /// Refill patch ghost cells from the coarse field.
    ///
    /// This is kept for backward compatibility and simple debugging. For nested multi-level
    /// refinement, higher-level patches should prefer `fill_patch_ghosts_from_uniform_fine()`
    /// so their ghost values come from the parent composite rather than the coarse grid.
    pub fn fill_patch_ghosts(&mut self) {
        for p in &mut self.patches {
            p.fill_ghosts_from_coarse(&self.coarse);
        }
        for lvl in &mut self.patches_l2plus {
            for p in lvl {
                p.fill_ghosts_from_coarse(&self.coarse);
            }
        }
    }

    /// Fill patch ghost cells by sampling from a composite uniform field.
    ///
    /// This is the preferred ghost-fill for nested multi-level patches: the composite field
    /// should represent the best available solution on the domain (e.g. from
    /// `flatten_to_uniform_fine()`), so that level-(L>=2) ghost cells are consistent with
    /// level-(L-1) interiors.
    pub fn fill_patch_ghosts_from_uniform_fine(&mut self, fine: &VectorField2D) {
        // Level 1
        for p in &mut self.patches {
            fill_one_patch_ghosts_from_uniform(p, fine);
        }
        // Level 2+
        for lvl in &mut self.patches_l2plus {
            for p in lvl {
                fill_one_patch_ghosts_from_uniform(p, fine);
            }
        }
    }

    /// Restrict all patch interiors back to the coarse grid (fine→coarse sync).
    ///
    /// This is mask-aware via Patch2D::parent_material.
    pub fn restrict_patches_to_coarse(&mut self) {
        // Restrict level-1 first, then higher levels so that finer data overrides.
        for p in &self.patches {
            p.restrict_to_coarse(&mut self.coarse);
        }
        for lvl in &self.patches_l2plus {
            for p in lvl {
                p.restrict_to_coarse(&mut self.coarse);
            }
        }
    }

    /// Build a uniform fine-grid representation for diagnostics/IO.
    ///
    /// 1) resample coarse -> finest uniform
    /// 2) overwrite with patch interiors (mask-aware scatter)
    pub fn flatten_to_uniform_fine(&self) -> VectorField2D {
        let r_finest = self.finest_ratio_total();
        let fine_grid = Grid2D::new(
            self.base_grid.nx * r_finest,
            self.base_grid.ny * r_finest,
            self.base_grid.dx / (r_finest as f64),
            self.base_grid.dy / (r_finest as f64),
            self.base_grid.dz,
        );

        // Start from coarse resampled to the finest uniform grid.
        let mut out = self.coarse.resample_to_grid(fine_grid);

        // Helper: scatter a patch level into `out`, replicating cells if `out` is finer.
        #[allow(unused_mut)]
        let mut scatter_level = |level: usize, patches: &[Patch2D], out: &mut VectorField2D| {
            let r_patch = self.ratio_pow(level);
            let r_out = r_finest;
            debug_assert!(r_out % r_patch == 0);
            let s = r_out / r_patch;

            for p in patches {
                let rect = p.coarse_rect;
                let gi0 = rect.i0 * r_out;
                let gj0 = rect.j0 * r_out;

                let nx_f = rect.nx * r_patch;
                let ny_f = rect.ny * r_patch;
                let g = p.ghost;

                for jf in 0..ny_f {
                    for if_ in 0..nx_f {
                        let src_i = g + if_;
                        let src_j = g + jf;
                        let v = p.m.data[p.grid.idx(src_i, src_j)];

                        // Map to finest grid coordinates, replicating by factor s.
                        let dst_i0 = gi0 + if_ * s;
                        let dst_j0 = gj0 + jf * s;

                        for dj in 0..s {
                            for di in 0..s {
                                let dii = dst_i0 + di;
                                let djj = dst_j0 + dj;
                                let didx = out.grid.idx(dii, djj);
                                out.data[didx] = v;
                            }
                        }
                    }
                }
            }
        };

        // Scatter level-1 then higher levels (finer overwrites).
        scatter_level(1, &self.patches, &mut out);
        for (k, lvl) in self.patches_l2plus.iter().enumerate() {
            let level = k + 2;
            scatter_level(level, lvl, &mut out);
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

    /// Replace the entire patch set at a given level (>=2), preserving fine values on overlaps.
    pub fn replace_level_patches_preserve_overlap(&mut self, level: usize, new_rects: Vec<Rect2i>) {
        assert!(level >= 2, "level must be >= 2 for replace_level_patches_preserve_overlap");
        self.ensure_level_storage(level);

        let idx_lvl = level - 2;
        let old_patches: Vec<Patch2D> = std::mem::take(&mut self.patches_l2plus[idx_lvl]);

        if new_rects.is_empty() {
            self.patches_l2plus[idx_lvl] = Vec::new();
            return;
        }

        // Create new patches seeded from coarse.
        let mut new_patches: Vec<Patch2D> = Vec::with_capacity(new_rects.len());
        let r_total = self.ratio_pow(level);
        for r in new_rects {
            // Enforce nesting within previous level.
            if let Some(parent) = self.patches_at_level(level - 1) {
                let mut ok = false;
                for p in parent {
                    if Self::rect_contains(p.coarse_rect, r) {
                        ok = true;
                        break;
                    }
                }
                assert!(ok, "level-{level} patch must be contained within a level-{} patch", level - 1);
            }

            let mut p = Patch2D::new(&self.base_grid, r, r_total, self.ghost);
            p.rebuild_active_from_coarse_mask(&self.base_grid, self.geom_mask.as_deref());
            p.fill_all_from_coarse(&self.coarse);
            new_patches.push(p);
        }

        // Copy overlap regions (global fine index space at r_total) from old -> new.
        for new_p in &mut new_patches {
            let new_rect = new_p.coarse_rect;
            let new_gi0 = new_rect.i0 * r_total;
            let new_gj0 = new_rect.j0 * r_total;
            let new_gi1 = new_gi0 + new_rect.nx * r_total;
            let new_gj1 = new_gj0 + new_rect.ny * r_total;
            let g_new = new_p.ghost;

            for old_p in &old_patches {
                let old_rect = old_p.coarse_rect;
                let old_gi0 = old_rect.i0 * r_total;
                let old_gj0 = old_rect.j0 * r_total;
                let old_gi1 = old_gi0 + old_rect.nx * r_total;
                let old_gj1 = old_gj0 + old_rect.ny * r_total;

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

        self.patches_l2plus[idx_lvl] = new_patches;
    }

    pub fn replace_single_patch_preserve_overlap(&mut self, new_rect: Rect2i) {
        self.replace_patches_preserve_overlap(vec![new_rect]);
    }
}

#[inline]
fn is_ghost(i: usize, j: usize, g: usize, nx: usize, ny: usize) -> bool {
    i < g || j < g || i + g >= nx || j + g >= ny
}

fn fill_one_patch_ghosts_from_uniform(p: &mut Patch2D, fine: &VectorField2D) {
    let g = p.ghost;
    let nx = p.grid.nx;
    let ny = p.grid.ny;
    let gm = p.geom_mask_fine.as_deref();

    for j in 0..ny {
        for i in 0..nx {
            if !is_ghost(i, j, g, nx, ny) {
                continue;
            }
            let (x, y) = p.cell_center_xy(i, j);
            let v = sample_bilinear(fine, x, y);
            let id = p.grid.idx(i, j);
            p.m.data[id] = v;
            if let Some(mask) = gm {
                if !mask[id] {
                    p.m.data[id] = [0.0, 0.0, 0.0];
                }
            }
        }
    }
}