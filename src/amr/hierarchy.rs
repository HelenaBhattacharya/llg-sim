// src/amr/hierarchy.rs

use crate::amr::interp::sample_bilinear;
use crate::amr::patch::Patch2D;
use crate::amr::rect::Rect2i;
use crate::geometry_mask::{Mask2D, MaskShape, assert_mask_len};
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
    ///
    /// Always present when a geometry is active (set by either `set_geom_mask` or
    /// `set_geom_shape`).  The coarse mask is used by the coarse-level stepper and
    /// by the indicator/clustering pipeline (which operates on the coarse field).
    pub geom_mask: Option<Mask2D>,

    /// Optional analytical geometry shape.
    ///
    /// When present, patches evaluate the shape at their native fine resolution
    /// rather than inheriting the coarse staircase.  Set via [`set_geom_shape`].
    ///
    /// All coordinates are in **centered** convention (origin at domain centre),
    /// consistent with [`geometry_mask::cell_center_xy_centered`].
    pub geom_shape: Option<MaskShape>,

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
            geom_shape: None,
            coarse,
            patches: Vec::new(),
            patches_l2plus: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Geometry mask / shape API
    // -----------------------------------------------------------------------

    /// Set (or replace) the geometry using an analytical shape.
    ///
    /// This is the **preferred** path for geometries with known analytical boundaries
    /// (disk, ellipse, annulus, rectangle, CSG combinations).  Fine-level patches will
    /// evaluate the shape at their native resolution, resolving curved boundaries at
    /// fine dx rather than inheriting the coarse staircase.
    ///
    /// Also builds a coarse boolean mask for the base-grid stepper and indicator.
    pub fn set_geom_shape(&mut self, shape: MaskShape) {
        // Build coarse mask from the shape (needed by coarse stepper + indicator).
        let mask = shape.to_mask(&self.base_grid);
        self.geom_mask = Some(mask);
        self.geom_shape = Some(shape);

        // Rebuild all patch masks analytically.
        let shape_ref = self
            .geom_shape
            .as_ref()
            .expect("geom_shape must be present after set_geom_shape");
        for p in &mut self.patches {
            p.rebuild_active_from_shape(&self.base_grid, shape_ref);
        }
        for lvl in &mut self.patches_l2plus {
            for p in lvl {
                p.rebuild_active_from_shape(&self.base_grid, shape_ref);
            }
        }
    }

    /// Set (or replace) the geometry mask on the base grid (boolean, coarse-resolution).
    ///
    /// This is the **legacy** path.  Patches inherit the mask at coarse resolution,
    /// so curved boundaries will be staircase-stepped at coarse dx even where fine
    /// patches exist.  Clears any stored analytical shape.
    ///
    /// Prefer [`set_geom_shape`] for analytical geometries.
    pub fn set_geom_mask(&mut self, mask: Mask2D) {
        assert_mask_len(&mask, &self.base_grid);
        self.geom_mask = Some(mask);
        self.geom_shape = None; // clear analytical shape

        // Update patch active sets + per-parent material flags (coarse path).
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

    /// Clear any geometry mask / shape.
    pub fn clear_geom_mask(&mut self) {
        self.geom_mask = None;
        self.geom_shape = None;

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

    /// Borrow the analytical geometry shape, if present.
    #[inline]
    pub fn geom_shape(&self) -> Option<&MaskShape> {
        self.geom_shape.as_ref()
    }

    /// True if a geometry mask is present (either boolean or analytical).
    #[inline]
    pub fn has_geom_mask(&self) -> bool {
        self.geom_mask.is_some()
    }

    /// Apply geometry (shape or coarse mask) to a single patch.
    ///
    /// Helper that dispatches to the analytical or coarse-inherited path depending
    /// on whether `geom_shape` is set.  Used at every patch construction site.
    fn apply_geom_to_patch_inner(&self, p: &mut Patch2D) {
        if let Some(ref shape) = self.geom_shape {
            p.rebuild_active_from_shape(&self.base_grid, shape);
        } else {
            p.rebuild_active_from_coarse_mask(&self.base_grid, self.geom_mask.as_deref());
        }
    }

    /// Construct a uniform-fine mask at the finest resolution present in the hierarchy.
    ///
    /// When an analytical shape is available, the mask is evaluated at fine-cell centres
    /// (no staircase).  Otherwise falls back to replicating the coarse mask.
    pub fn build_uniform_fine_mask(&self) -> Option<Mask2D> {
        // Must have some geometry to build a mask.
        if self.geom_mask.is_none() {
            return None;
        }

        let r = self.finest_ratio_total();
        let fine_nx = self.base_grid.nx * r;
        let fine_ny = self.base_grid.ny * r;

        if let Some(ref shape) = self.geom_shape {
            // Analytical path: evaluate shape at every fine cell centre.
            let fine_dx = self.base_grid.dx / (r as f64);
            let fine_dy = self.base_grid.dy / (r as f64);
            let half_lx = 0.5 * fine_nx as f64 * fine_dx;
            let half_ly = 0.5 * fine_ny as f64 * fine_dy;

            let mut out = vec![false; fine_nx * fine_ny];
            for j in 0..fine_ny {
                for i in 0..fine_nx {
                    let x = (i as f64 + 0.5) * fine_dx - half_lx;
                    let y = (j as f64 + 0.5) * fine_dy - half_ly;
                    out[j * fine_nx + i] = shape.contains(x, y);
                }
            }
            Some(out)
        } else {
            // Coarse-replicate path (original behaviour).
            let m0 = self.geom_mask.as_ref().unwrap();
            assert_mask_len(m0, &self.base_grid);

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
    }

    // -----------------------------------------------------------------------
    // Level helpers
    // -----------------------------------------------------------------------

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

    /// Number of active levels in the hierarchy.
    ///
    /// Returns 1 if only the coarse grid exists (no patches),
    /// 2 if level-1 patches exist, etc.
    pub fn num_levels(&self) -> usize {
        let mut max_level: usize = 0;
        if !self.patches.is_empty() {
            max_level = 1;
        }
        for (k, lvl) in self.patches_l2plus.iter().enumerate() {
            if !lvl.is_empty() {
                max_level = max_level.max(k + 2);
            }
        }
        max_level + 1
    }

    // -----------------------------------------------------------------------
    // Patch creation / replacement
    // -----------------------------------------------------------------------

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
                assert!(
                    ok,
                    "level-{level} patch must be contained within a level-{} patch",
                    level - 1
                );
            } else {
                panic!(
                    "cannot add level-{level} patch without an existing level-{}",
                    level - 1
                );
            }
        }

        self.ensure_level_storage(level);

        let r_total = self.ratio_pow(level);
        let mut p = Patch2D::new(&self.base_grid, coarse_rect, r_total, self.ghost);

        // Respect geometry mask/shape (updates active + parent_material + geom_mask_fine).
        self.apply_geom_to_patch_inner(&mut p);

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

    // -----------------------------------------------------------------------
    // Ghost fill, restriction
    // -----------------------------------------------------------------------

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

    /// Fill ghost cells for patches at a specific `level` using temporally interpolated
    /// coarse data (Berger–Colella subcycling).
    ///
    /// `coarse_old`: coarse-grid state at time t (before coarse step).
    /// `coarse_new`: coarse-grid state at time t + dt_coarse (after coarse step).
    /// `alpha`: fractional time within the coarse step (0.0 = old, 1.0 = new).
    ///
    /// For v1 all levels interpolate from level-0 (coarse), matching the spatial
    /// accuracy of the existing ghost fill. A future v2 can interpolate from the
    /// direct parent level for better accuracy at L2+.
    pub fn fill_ghosts_time_interpolated(
        &mut self,
        level: usize,
        coarse_old: &VectorField2D,
        coarse_new: &VectorField2D,
        alpha: f64,
    ) {
        match level {
            0 => {} // coarse has no ghosts
            1 => {
                for p in &mut self.patches {
                    fill_one_patch_ghosts_temporal(p, coarse_old, coarse_new, alpha);
                }
            }
            l if l >= 2 => {
                let idx = l - 2;
                if let Some(lvl) = self.patches_l2plus.get_mut(idx) {
                    for p in lvl {
                        fill_one_patch_ghosts_temporal(p, coarse_old, coarse_new, alpha);
                    }
                }
            }
            _ => {}
        }
    }

    /// Fill ghost cells for ALL patch levels using temporal interpolation.
    ///
    /// Convenience wrapper that fills every level >= 1.
    pub fn fill_all_ghosts_time_interpolated(
        &mut self,
        coarse_old: &VectorField2D,
        coarse_new: &VectorField2D,
        alpha: f64,
    ) {
        for p in &mut self.patches {
            fill_one_patch_ghosts_temporal(p, coarse_old, coarse_new, alpha);
        }
        for lvl in &mut self.patches_l2plus {
            for p in lvl {
                fill_one_patch_ghosts_temporal(p, coarse_old, coarse_new, alpha);
            }
        }
    }

    /// Restrict all patch interiors back to the coarse grid (fine→coarse sync).
    ///
    /// This is mask-aware via Patch2D::parent_material and (when available) the
    /// per-cell fine mask for partial boundary parents.
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

    /// Restrict a single child level to its parent level (Berger–Colella intermediate restriction).
    ///
    /// - `child_level=1`: restrict L1 patches → coarse grid (L0).
    /// - `child_level=2`: restrict L2 patches → L1 patches.
    /// - `child_level=3`: restrict L3 patches → L2 patches.
    /// - etc.
    ///
    /// For child_level >= 2, each child interior cell that overlaps a parent patch
    /// is averaged (r×r child cells → 1 parent cell, where r = self.ratio) and
    /// written into the parent's `m` field. The coarse grid is also updated.
    pub fn restrict_level_to_parent(&mut self, child_level: usize) {
        assert!(child_level >= 1, "cannot restrict level 0");

        if child_level == 1 {
            // L1 → L0: use existing per-patch restrict
            for p in &self.patches {
                p.restrict_to_coarse(&mut self.coarse);
            }
            return;
        }

        // child_level >= 2: restrict to parent patches, then also to coarse.
        let r = self.ratio;
        let r_child = self.ratio_pow(child_level);
        let r_parent = self.ratio_pow(child_level - 1);

        let child_idx = child_level - 2;
        let parent_level = child_level - 1;

        // Borrow-split: take child patches immutably, parent patches mutably.
        // Since they're at different indices in patches_l2plus (or patches for L1),
        // we need careful handling.
        if parent_level == 1 {
            // Restrict from patches_l2plus[child_idx] into self.patches
            if let Some(child_patches) = self.patches_l2plus.get(child_idx) {
                if child_patches.is_empty() {
                    return;
                }
                // We need to restrict child patches into L1 parent patches.
                // Since we can't borrow both mutably/immutably from self at once,
                // collect the restriction operations and apply them.
                restrict_child_into_parent_patches(
                    child_patches,
                    &mut self.patches,
                    r,
                    r_child,
                    r_parent,
                );
            }
            // Also restrict child to coarse (finer data overwrites)
            if let Some(child_patches) = self.patches_l2plus.get(child_idx) {
                for p in child_patches {
                    p.restrict_to_coarse(&mut self.coarse);
                }
            }
        } else {
            // Both child and parent are in patches_l2plus at different indices.
            // parent_level >= 2 → parent_idx = parent_level - 2, child_idx = child_level - 2.
            let parent_idx = parent_level - 2;
            debug_assert!(child_idx != parent_idx);

            // Split borrow: we need mutable access to parent_idx and immutable to child_idx.
            // Use split_at_mut to get non-overlapping slices.
            if child_idx < self.patches_l2plus.len() && parent_idx < self.patches_l2plus.len() {
                let (lo, hi) = if parent_idx < child_idx {
                    let (a, b) = self.patches_l2plus.split_at_mut(child_idx);
                    (&b[0] as &Vec<_>, &mut a[parent_idx])
                } else {
                    let (a, b) = self.patches_l2plus.split_at_mut(parent_idx);
                    (&a[child_idx] as &Vec<_>, &mut b[0])
                };
                let child_patches = lo;
                let parent_patches = hi;

                if !child_patches.is_empty() {
                    restrict_child_into_parent_patches(
                        child_patches,
                        parent_patches,
                        r,
                        r_child,
                        r_parent,
                    );
                }
            }

            // Also restrict child to coarse
            if let Some(child_patches) = self.patches_l2plus.get(child_idx) {
                for p in child_patches {
                    p.restrict_to_coarse(&mut self.coarse);
                }
            }
        }
    }

    /// Fill ghost cells for patches at `child_level` using temporally interpolated
    /// data from the direct parent level's patches.
    ///
    /// - `child_level=1`: interpolates from coarse_old/coarse_new (L0) — same as v1.
    /// - `child_level>=2`: for each ghost cell, finds the covering parent-level patch
    ///   and interpolates between its old/new states. Falls back to coarse interpolation
    ///   if no parent patch covers the ghost cell.
    ///
    /// `parent_old_m`: saved m.data for each parent-level patch before stepping.
    /// `alpha`: fractional time within the parent's substep (0.0 = old, 1.0 = new).
    pub fn fill_ghosts_from_parent_temporal(
        &mut self,
        child_level: usize,
        parent_old_m: &[Vec<[f64; 3]>],
        coarse_old: &VectorField2D,
        coarse_new: &VectorField2D,
        alpha_local: f64,
        alpha_coarse: f64,
    ) {
        if child_level == 0 {
            return;
        }
        if child_level == 1 {
            // L1 ghosts from coarse old/new — same as v1
            for p in &mut self.patches {
                fill_one_patch_ghosts_temporal(p, coarse_old, coarse_new, alpha_coarse);
            }
            return;
        }

        // child_level >= 2: interpolate from parent-level patches
        let parent_level = child_level - 1;
        let child_idx = child_level - 2;

        if parent_level == 1 {
            // Parent is in self.patches, child is in self.patches_l2plus[child_idx].
            // These are distinct struct fields, so borrowing is straightforward.
            let parent_patches = &self.patches as &[Patch2D];
            if let Some(child_patches) = self.patches_l2plus.get_mut(child_idx) {
                for child_p in child_patches.iter_mut() {
                    fill_one_patch_ghosts_from_parent_patches_temporal(
                        child_p,
                        parent_patches,
                        parent_old_m,
                        alpha_local,
                        coarse_old,
                        coarse_new,
                        alpha_coarse,
                    );
                }
            }
        } else {
            // Both parent and child are in patches_l2plus at different indices.
            // parent_idx < child_idx always (since parent_level = child_level - 1).
            let parent_idx = parent_level - 2;
            debug_assert!(parent_idx < child_idx);

            if child_idx < self.patches_l2plus.len() && parent_idx < self.patches_l2plus.len() {
                // split_at_mut at child_idx: a[0..child_idx] is immutable-safe, b[0] is mutable.
                let (a, b) = self.patches_l2plus.split_at_mut(child_idx);
                let parent_patches = &a[parent_idx] as &[Patch2D];
                let child_patches = &mut b[0];

                for child_p in child_patches.iter_mut() {
                    fill_one_patch_ghosts_from_parent_patches_temporal(
                        child_p,
                        parent_patches,
                        parent_old_m,
                        alpha_local,
                        coarse_old,
                        coarse_new,
                        alpha_coarse,
                    );
                }
            }
        }
    }

    /// Trim `active` indices on each level's patches to exclude cells that are
    /// also covered by patches at the next deeper level.
    ///
    /// When level-(L+1) patches nest inside level-L patches, any level-L cell
    /// whose parent coarse cell falls within a level-(L+1) patch's `coarse_rect`
    /// will be overwritten during restriction anyway (finer data wins).
    /// Skipping those cells in the RK4 update avoids redundant computation.
    ///
    /// The savings apply to the per-`active`-cell RK4 arithmetic (addend
    /// application, LLG RHS evaluation, m-update, final combination) — four
    /// loops × four stages = 16 iterations per excluded cell per step.
    /// Field evaluation (`build_h_eff_masked`) still runs on the full patch
    /// grid because exchange stencils need neighbour values.
    ///
    /// This method is idempotent: calling it multiple times on an already-trimmed
    /// hierarchy is a cheap no-op (the `retain` finds nothing to remove).
    ///
    /// Call after all patch levels have been finalised (e.g. after regrid or
    /// initial setup).  For convenience, `AmrStepperRK4::step()` calls this
    /// at the start of every time step so no manual invocation is needed.
    pub fn trim_active_for_nesting(&mut self) {
        // Level-1 patches: exclude cells covered by level-2 patches.
        let l2_rects: Vec<Rect2i> = self
            .patches_l2plus
            .first()
            .map(|v| v.iter().map(|p| p.coarse_rect).collect())
            .unwrap_or_default();

        if !l2_rects.is_empty() {
            for p in &mut self.patches {
                trim_patch_active_covered(p, &l2_rects);
            }
        }

        // Level-L patches (L >= 2): exclude cells covered by level-(L+1).
        let n_deep = self.patches_l2plus.len();
        for k in 0..n_deep {
            // Collect deeper-level rects first to avoid borrow conflict.
            let deeper_rects: Vec<Rect2i> = if k + 1 < n_deep {
                self.patches_l2plus[k + 1]
                    .iter()
                    .map(|p| p.coarse_rect)
                    .collect()
            } else {
                Vec::new()
            };

            if !deeper_rects.is_empty() {
                for p in &mut self.patches_l2plus[k] {
                    trim_patch_active_covered(p, &deeper_rects);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Flattening / IO
    // -----------------------------------------------------------------------

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
            if patches.is_empty() {
                return;
            }
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

    // -----------------------------------------------------------------------
    // Patch replacement (regrid)
    // -----------------------------------------------------------------------

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
        assert!(
            level >= 2,
            "level must be >= 2 for replace_level_patches_preserve_overlap"
        );
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
                assert!(
                    ok,
                    "level-{level} patch must be contained within a level-{} patch",
                    level - 1
                );
            }

            let mut p = Patch2D::new(&self.base_grid, r, r_total, self.ghost);
            self.apply_geom_to_patch_inner(&mut p);
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

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

#[inline]
fn is_ghost(i: usize, j: usize, g: usize, nx: usize, ny: usize) -> bool {
    i < g || j < g || i + g >= nx || j + g >= ny
}

/// Remove entries from `patch.active` whose parent coarse cell is covered
/// by any rectangle in `deeper_rects` (expressed in base-grid coordinates).
///
/// Each fine cell at patch-local coords (i_patch, j_patch) maps to a parent
/// coarse cell at base-grid coords:
///
///     base_i = coarse_rect.i0 + (i_patch - ghost) / patch.ratio
///     base_j = coarse_rect.j0 + (j_patch - ghost) / patch.ratio
///
/// If that coarse cell falls inside any deeper-level patch rect, the cell
/// is removed from `active` because the deeper level will compute it at
/// higher resolution and restriction will overwrite the result.
fn trim_patch_active_covered(patch: &mut Patch2D, deeper_rects: &[Rect2i]) {
    if deeper_rects.is_empty() {
        return;
    }

    let ghost = patch.ghost;
    let r = patch.ratio;
    let i0 = patch.coarse_rect.i0;
    let j0 = patch.coarse_rect.j0;
    let nx = patch.grid.nx;

    patch.active.retain(|&idx| {
        let i_patch = idx % nx;
        let j_patch = idx / nx;

        // Interior-local fine coordinates.
        let li = i_patch - ghost;
        let lj = j_patch - ghost;

        // Parent coarse cell in base-grid coordinates.
        let base_i = i0 + li / r;
        let base_j = j0 + lj / r;

        // Keep this cell only if it is NOT covered by any deeper patch.
        !deeper_rects.iter().any(|rect| {
            base_i >= rect.i0 && base_i < rect.i1() && base_j >= rect.j0 && base_j < rect.j1()
        })
    });
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

/// Fill ghost cells of a single patch using temporally interpolated coarse data.
///
/// Bilinear-samples both `coarse_old` and `coarse_new` at each ghost cell's physical
/// coordinates, then linearly interpolates in time using `alpha` (0 = old, 1 = new),
/// and renormalises the result to unit length (magnetisation field).
fn fill_one_patch_ghosts_temporal(
    p: &mut Patch2D,
    coarse_old: &VectorField2D,
    coarse_new: &VectorField2D,
    alpha: f64,
) {
    use crate::amr::interp::sample_bilinear_temporal_unit;
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
            let v = sample_bilinear_temporal_unit(coarse_old, coarse_new, x, y, alpha);
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

/// Restrict child-level patch interiors into parent-level patch interiors.
///
/// For each parent-level fine cell that is covered by a child patch, computes the
/// average of the r×r child fine cells and writes the normalised result into the
/// parent patch's `m` field.
///
/// `r`: single-level refinement ratio (typically 2).
/// `r_child`: total refinement ratio for the child level (ratio^child_level).
/// `r_parent`: total refinement ratio for the parent level (ratio^parent_level).
fn restrict_child_into_parent_patches(
    child_patches: &[Patch2D],
    parent_patches: &mut [Patch2D],
    r: usize,
    r_child: usize,
    r_parent: usize,
) {
    use crate::vec3::normalize;

    for child in child_patches {
        // Child's coverage in global-child-fine coordinates
        let child_gi0 = child.coarse_rect.i0 * r_child;
        let child_gj0 = child.coarse_rect.j0 * r_child;
        let child_gi1 = child_gi0 + child.interior_nx;
        let child_gj1 = child_gj0 + child.interior_ny;

        // Convert to parent-fine coordinates (child is r× finer than parent)
        let parent_fi0 = child_gi0 / r;
        let parent_fj0 = child_gj0 / r;
        let parent_fi1 = child_gi1 / r;
        let parent_fj1 = child_gj1 / r;

        for parent in parent_patches.iter_mut() {
            let parent_gi0 = parent.coarse_rect.i0 * r_parent;
            let parent_gj0 = parent.coarse_rect.j0 * r_parent;
            let parent_gi1 = parent_gi0 + parent.interior_nx;
            let parent_gj1 = parent_gj0 + parent.interior_ny;

            // Overlap in parent-fine coordinates
            let oi0 = parent_fi0.max(parent_gi0);
            let oj0 = parent_fj0.max(parent_gj0);
            let oi1 = parent_fi1.min(parent_gi1);
            let oj1 = parent_fj1.min(parent_gj1);

            if oi1 <= oi0 || oj1 <= oj0 {
                continue;
            }

            let g_child = child.ghost;
            let g_parent = parent.ghost;

            // For each parent-fine cell in the overlap, average r×r child-fine cells
            for pj in oj0..oj1 {
                for pi in oi0..oi1 {
                    let mut sum = [0.0_f64; 3];
                    let mut count = 0usize;

                    for dj in 0..r {
                        for di in 0..r {
                            let ci = pi * r + di; // global child-fine index
                            let cj = pj * r + dj;

                            // Check bounds against child patch
                            if ci < child_gi0 || ci >= child_gi1
                                || cj < child_gj0 || cj >= child_gj1
                            {
                                continue;
                            }

                            // Convert to child patch-local index
                            let cli = g_child + (ci - child_gi0);
                            let clj = g_child + (cj - child_gj0);
                            let cidx = child.grid.idx(cli, clj);

                            let v = child.m.data[cidx];
                            sum[0] += v[0];
                            sum[1] += v[1];
                            sum[2] += v[2];
                            count += 1;
                        }
                    }

                    if count > 0 {
                        let inv = 1.0 / (count as f64);
                        let avg = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
                        let norm_avg = if avg[0] == 0.0 && avg[1] == 0.0 && avg[2] == 0.0 {
                            avg
                        } else {
                            normalize(avg)
                        };

                        // Write to parent patch
                        let pli = g_parent + (pi - parent_gi0);
                        let plj = g_parent + (pj - parent_gj0);
                        let pidx = parent.grid.idx(pli, plj);
                        parent.m.data[pidx] = norm_avg;
                    }
                }
            }
        }
    }
}

/// Fill ghost cells of a child patch by temporally interpolating from parent-level patches.
///
/// For each ghost cell:
/// 1. Check if any parent patch covers this location.
/// 2. If yes: sample from the parent's old and current m data, temporally interpolated.
/// 3. If no: fall back to coarse (L0) temporal interpolation.
fn fill_one_patch_ghosts_from_parent_patches_temporal(
    child: &mut Patch2D,
    parent_patches: &[Patch2D],
    parent_old_m: &[Vec<[f64; 3]>],
    alpha_local: f64,
    coarse_old: &VectorField2D,
    coarse_new: &VectorField2D,
    alpha_coarse: f64,
) {
    use crate::amr::interp::sample_bilinear_temporal_unit;
    use crate::vec3::normalize;

    let g = child.ghost;
    let nx = child.grid.nx;
    let ny = child.grid.ny;
    let gm = child.geom_mask_fine.as_deref();

    for j in 0..ny {
        for i in 0..nx {
            if !is_ghost(i, j, g, nx, ny) {
                continue;
            }
            let (x, y) = child.cell_center_xy(i, j);
            let id = child.grid.idx(i, j);

            // Try to sample from a parent patch
            let mut found = false;
            for (pidx, parent) in parent_patches.iter().enumerate() {
                // Check if (x, y) falls within parent's coverage (including ghosts for sampling)
                let p_dx = parent.grid.dx;
                let p_dy = parent.grid.dy;
                let p_x0 = (parent.coarse_rect.i0 as f64 * parent.ratio as f64) * p_dx;
                let p_y0 = (parent.coarse_rect.j0 as f64 * parent.ratio as f64) * p_dy;
                let p_x1 = p_x0 + (parent.interior_nx as f64) * p_dx;
                let p_y1 = p_y0 + (parent.interior_ny as f64) * p_dy;

                // Allow sampling slightly inside the parent (use interior bounds with margin)
                if x < p_x0 - 0.5 * p_dx || x > p_x1 + 0.5 * p_dx
                    || y < p_y0 - 0.5 * p_dy || y > p_y1 + 0.5 * p_dy
                {
                    continue;
                }

                // Sample from parent patch at this location, temporally interpolated
                if pidx < parent_old_m.len() && !parent_old_m[pidx].is_empty() {
                    // Convert (x, y) to parent patch-local coordinates for bilinear sampling
                    let p_offset_x = (parent.coarse_rect.i0 as f64 * parent.ratio as f64
                        - parent.ghost as f64) * p_dx;
                    let p_offset_y = (parent.coarse_rect.j0 as f64 * parent.ratio as f64
                        - parent.ghost as f64) * p_dy;
                    let lx = x - p_offset_x;
                    let ly = y - p_offset_y;

                    let v_old = sample_bilinear_on_data(
                        &parent_old_m[pidx], parent.grid.nx, parent.grid.ny, p_dx, p_dy, lx, ly,
                    );
                    let v_new = sample_bilinear_on_data(
                        &parent.m.data, parent.grid.nx, parent.grid.ny, p_dx, p_dy, lx, ly,
                    );

                    let one_a = 1.0 - alpha_local;
                    let v = [
                        one_a * v_old[0] + alpha_local * v_new[0],
                        one_a * v_old[1] + alpha_local * v_new[1],
                        one_a * v_old[2] + alpha_local * v_new[2],
                    ];

                    child.m.data[id] = if v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0 {
                        [0.0, 0.0, 1.0]
                    } else {
                        normalize(v)
                    };
                    found = true;
                    break;
                }
            }

            if !found {
                // Fallback: interpolate from coarse old/new
                let v = sample_bilinear_temporal_unit(coarse_old, coarse_new, x, y, alpha_coarse);
                child.m.data[id] = v;
            }

            // Zero out vacuum cells
            if let Some(mask) = gm {
                if !mask[id] {
                    child.m.data[id] = [0.0, 0.0, 0.0];
                }
            }
        }
    }
}

/// Bilinear sample on a raw `&[[f64; 3]]` array treated as an nx×ny grid.
///
/// Same logic as `sample_bilinear` but operates on raw data + dimensions
/// rather than a `VectorField2D`, so we can sample from saved old-state snapshots.
fn sample_bilinear_on_data(
    data: &[[f64; 3]],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    x: f64,
    y: f64,
) -> [f64; 3] {
    if nx == 0 || ny == 0 {
        return [0.0, 0.0, 0.0];
    }

    let fx = x / dx - 0.5;
    let fy = y / dy - 0.5;

    let i0f = fx.floor();
    let j0f = fy.floor();
    let tx = fx - i0f;
    let ty = fy - j0f;

    let clamp = |v: isize, n: usize| -> usize {
        if v <= 0 { 0 } else if v >= n as isize { n - 1 } else { v as usize }
    };

    let i0 = clamp(i0f as isize, nx);
    let j0 = clamp(j0f as isize, ny);
    let i1 = clamp(i0f as isize + 1, nx);
    let j1 = clamp(j0f as isize + 1, ny);

    let v00 = data[j0 * nx + i0];
    let v10 = data[j0 * nx + i1];
    let v01 = data[j1 * nx + i0];
    let v11 = data[j1 * nx + i1];

    let lerp = |a: [f64; 3], b: [f64; 3], t: f64| -> [f64; 3] {
        [
            a[0] * (1.0 - t) + b[0] * t,
            a[1] * (1.0 - t) + b[1] * t,
            a[2] * (1.0 - t) + b[2] * t,
        ]
    };

    let v0 = lerp(v00, v10, tx);
    let v1 = lerp(v01, v11, tx);
    lerp(v0, v1, ty)
}