#[inline]
fn rects_at_level_base(h: &AmrHierarchy2D, level: usize) -> Vec<Rect2i> {
    if level == 1 {
        return h.patches.iter().map(|p| p.coarse_rect).collect();
    }
    let idx = level.saturating_sub(2);
    h.patches_l2plus
        .get(idx)
        .map(|v| v.iter().map(|p| p.coarse_rect).collect())
        .unwrap_or_else(Vec::new)
}
// src/amr/regrid.rs
//
// Regridding logic (Stage 2A + Stage 2B).
//
// Stage 2A:
// - Single dynamic patch built from a coarse-grid indicator bbox.
// - Regrid periodically, but only apply if the patch changed "materially".
//
// Stage 2B:
// - Multi-patch dynamic regrid using clustering (clustering.rs).
// - Apply a cheap hysteresis check to avoid 1-cell jitter: compare union-bbox
//   movement/size change between old and new patch sets.

use crate::amr::clustering::{
    ClusterPolicy, ClusterStats, compute_patch_rects_clustered_from_indicator,
};
use crate::amr::hierarchy::AmrHierarchy2D;
use crate::amr::indicator::{
    IndicatorStats,
    compute_patch_bbox_from_indicator_geom,
    compute_patch_bbox_from_angle_threshold_geom,
    indicator_angle_max_forward_geom,
};
use crate::amr::rect::Rect2i;

#[derive(Clone, Copy, Debug)]
pub struct RegridPolicy {
    /// Refinement threshold control.
    ///
    /// If `indicator_frac >= 0.0`, Stage-2A uses the grad^2 indicator with a relative
    /// threshold: flag cells where ind >= indicator_frac * max(ind).
    ///
    /// If `indicator_frac < 0.0`, Stage-2A switches to an absolute *angle* threshold
    /// (radians): theta_refine = -indicator_frac, and flags cells where
    /// max_neighbour_angle(i,j) >= theta_refine.
    ///
    /// Typical coarse-grid values for r=2 refinement: 0.35–0.52 rad (≈ 20°–30°).
    pub indicator_frac: f64,
    pub buffer_cells: usize,
    pub min_change_cells: usize,
    pub min_area_change_frac: f64,
}

#[inline]
fn theta_coarsen_factor() -> f64 {
    // Allow runtime tuning without touching all benchmark policy literals.
    // Default chosen to avoid jitter while still permitting derefinement.
    if let Ok(s) = std::env::var("LLG_AMR_THETA_COARSEN_FACTOR") {
        if let Ok(v) = s.parse::<f64>() {
            // Clamp to a sensible range.
            return v.clamp(0.5, 0.95);
        }
    }
    0.75
}

#[inline]
pub fn material_change(
    old_rect: Rect2i,
    new_rect: Rect2i,
    min_change: usize,
    min_area_frac: f64,
) -> bool {
    let di0 = (new_rect.i0 as isize - old_rect.i0 as isize).abs() as usize;
    let dj0 = (new_rect.j0 as isize - old_rect.j0 as isize).abs() as usize;
    let dnx = (new_rect.nx as isize - old_rect.nx as isize).abs() as usize;
    let dny = (new_rect.ny as isize - old_rect.ny as isize).abs() as usize;

    let area_old = (old_rect.nx * old_rect.ny) as f64;
    let area_new = (new_rect.nx * new_rect.ny) as f64;

    let area_frac = if area_old > 0.0 {
        (area_new - area_old).abs() / area_old
    } else {
        1.0
    };

    di0 >= min_change
        || dj0 >= min_change
        || dnx >= min_change
        || dny >= min_change
        || area_frac >= min_area_frac
}

/// Propose a single patch from the current coarse state.
///
/// This is mask-aware via `h.geom_mask()`.
pub fn propose_single_patch(
    h: &AmrHierarchy2D,
    policy: RegridPolicy,
) -> Option<(Rect2i, IndicatorStats)> {
    // Stage-2A selection:
    // - indicator_frac >= 0: grad^2 with relative threshold (frac * max)
    // - indicator_frac < 0 : absolute angle threshold (radians)
    if policy.indicator_frac < 0.0 {
        let theta_refine = (-policy.indicator_frac).max(0.0);
        compute_patch_bbox_from_angle_threshold_geom(
            &h.coarse,
            theta_refine,
            policy.buffer_cells,
            h.geom_mask(),
        )
    } else {
        compute_patch_bbox_from_indicator_geom(
            &h.coarse,
            policy.indicator_frac,
            policy.buffer_cells,
            h.geom_mask(),
        )
    }
}

/// Apply Stage-2A regrid *if* the patch changes materially.
///
/// Returns Some((new_rect, stats)) if regrid occurred; else None.
pub fn maybe_regrid_single_patch(
    h: &mut AmrHierarchy2D,
    current_patch: Rect2i,
    policy: RegridPolicy,
) -> Option<(Rect2i, IndicatorStats)> {
    let (new_rect, stats) = propose_single_patch(h, policy)?;

    if material_change(
        current_patch,
        new_rect,
        policy.min_change_cells,
        policy.min_area_change_frac,
    ) {
        // Data transfer should preserve overlap to avoid destroying fine detail.
        h.replace_single_patch_preserve_overlap(new_rect);
        Some((new_rect, stats))
    } else {
        None
    }
}

fn union_of_rects(rects: &[Rect2i]) -> Option<Rect2i> {
    if rects.is_empty() {
        return None;
    }
    let mut i0 = rects[0].i0;
    let mut j0 = rects[0].j0;
    let mut i1 = rects[0].i1();
    let mut j1 = rects[0].j1();

    for &r in rects.iter().skip(1) {
        i0 = i0.min(r.i0);
        j0 = j0.min(r.j0);
        i1 = i1.max(r.i1());
        j1 = j1.max(r.j1());
    }

    Some(Rect2i::new(i0, j0, i1 - i0, j1 - j0))
}

#[inline]
fn max_angle_on_coarse_geom(coarse: &crate::vector_field::VectorField2D, geom_mask: Option<&[bool]>) -> f64 {
    let nx = coarse.grid.nx;
    let ny = coarse.grid.ny;
    let mut max_theta = 0.0_f64;
    for j in 0..ny {
        for i in 0..nx {
            let th = indicator_angle_max_forward_geom(coarse, i, j, geom_mask);
            if th > max_theta {
                max_theta = th;
            }
        }
    }
    max_theta
}

/// Stage-2B: clustered multi-patch regrid.
///
/// Returns Some((new_rects, stats)) if regrid occurred; else None.
///
/// Notes:
/// - We compare old vs new *union* bounding boxes to decide whether to accept a regrid.
///   This suppresses 1-cell jitter while staying cheap.
/// - If `current_patches` is empty we always accept.
pub fn maybe_regrid_multi_patch(
    h: &mut AmrHierarchy2D,
    current_patches: &[Rect2i],
    policy: RegridPolicy,
    cluster_policy: ClusterPolicy,
) -> Option<(Vec<Rect2i>, ClusterStats)> {
    // Milestone 1: true automatic adaptivity in angle-threshold mode.
    //
    // When `policy.indicator_frac < 0`, we interpret `theta_refine = -indicator_frac` (radians).
    // We then apply a simple hysteresis band:
    //   - refine when max_theta >= theta_refine
    //   - keep existing patches when theta_coarsen <= max_theta < theta_refine
    //   - de-refine (clear all patches) when max_theta < theta_coarsen
    //
    // The coarsen factor can be tuned via LLG_AMR_THETA_COARSEN_FACTOR (default 0.75).
    //
    // NOTE: For grad^2 relative-threshold mode, max(ind) always flags at least one cell,
    // so derefinement needs an additional absolute floor; we only implement derefine here
    // for the angle-mode path.
    if policy.indicator_frac < 0.0 {
        let theta_refine = (-policy.indicator_frac).max(0.0);
        // Hysteresis band (configurable via env var).
        let theta_coarsen = theta_coarsen_factor() * theta_refine;
        let max_theta = max_angle_on_coarse_geom(&h.coarse, h.geom_mask());

        // De-refine if we are currently refined but the texture is now well-resolved.
        if !current_patches.is_empty() && max_theta < theta_coarsen {
            let empty: Vec<Rect2i> = Vec::new();
            h.replace_patches_preserve_overlap(empty.clone());
            let stats = ClusterStats {
                max_indicator: max_theta,
                threshold: theta_refine,
                flagged_cells: 0,
                components: 0,
                patches_before_merge: 0,
                patches_after_merge: 0,
            };
            return Some((empty, stats));
        }

        // If we are refined and within the hysteresis band, keep current patches.
        if !current_patches.is_empty() && max_theta < theta_refine {
            return None;
        }

        // If we are unrefined and below the refine threshold, do nothing.
        if current_patches.is_empty() && max_theta < theta_refine {
            return None;
        }
    }

    let (mut new_rects, stats) =
        compute_patch_rects_clustered_from_indicator(&h.coarse, cluster_policy, h.geom_mask())?;

    new_rects.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));

    let mut cur = current_patches.to_vec();
    cur.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));

    // Fast path: identical
    if cur == new_rects {
        return None;
    }

    // If we currently have nothing, always accept.
    if cur.is_empty() {
        h.replace_patches_preserve_overlap(new_rects.clone());
        return Some((new_rects, stats));
    }

    // Hysteresis based on union bbox movement/size change
    let old_u = union_of_rects(&cur)?;
    let new_u = union_of_rects(&new_rects)?;

    if material_change(
        old_u,
        new_u,
        policy.min_change_cells,
        policy.min_area_change_frac,
    ) {
        h.replace_patches_preserve_overlap(new_rects.clone());
        Some((new_rects, stats))
    } else {
        None
    }
}

// -----------------------------------------------------------------------------
// Milestone 1 (nested refinement): optional level-2 patch generation
// -----------------------------------------------------------------------------

#[inline]
fn upsample_base_mask_to_ratio(
    base_grid: &crate::grid::Grid2D,
    base_mask: Option<&[bool]>,
    r: usize,
) -> Option<Vec<bool>> {
    let m0 = base_mask?;
    let fine_nx = base_grid.nx * r;
    let fine_ny = base_grid.ny * r;
    let mut out = vec![false; fine_nx * fine_ny];

    for j in 0..base_grid.ny {
        for i in 0..base_grid.nx {
            if !m0[base_grid.idx(i, j)] {
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
fn mark_union_region(
    base_grid: &crate::grid::Grid2D,
    r: usize,
    rects_base: &[Rect2i],
    fine_mask: &mut [bool],
) {
    let fine_nx = base_grid.nx * r;

    for rect in rects_base {
        let i0 = rect.i0;
        let j0 = rect.j0;
        let i1 = rect.i1();
        let j1 = rect.j1();

        let fi0 = i0 * r;
        let fj0 = j0 * r;
        let fi1 = i1 * r;
        let fj1 = j1 * r;

        for fj in fj0..fj1 {
            let row = fj * fine_nx;
            for fi in fi0..fi1 {
                fine_mask[row + fi] = true;
            }
        }
    }
}

#[inline]
fn rect_level_r_to_base(rect: Rect2i, r: usize, nx0: usize, ny0: usize) -> Rect2i {
    let i0 = rect.i0 / r;
    let j0 = rect.j0 / r;

    // Ceil division for end indices.
    let i1 = (rect.i1() + r - 1) / r;
    let j1 = (rect.j1() + r - 1) / r;

    let i0c = i0.min(nx0.saturating_sub(1));
    let j0c = j0.min(ny0.saturating_sub(1));
    let i1c = i1.min(nx0);
    let j1c = j1.min(ny0);

    let nx = (i1c.saturating_sub(i0c)).max(1);
    let ny = (j1c.saturating_sub(j0c)).max(1);
    Rect2i::new(i0c, j0c, nx, ny)
}

#[inline]
fn ratio_pow_local(ratio: usize, level: usize) -> usize {
    // ratio^level, with level=0 -> 1
    let mut r = 1usize;
    for _ in 0..level {
        r = r.saturating_mul(ratio);
    }
    r
}

#[inline]
fn theta_refine_at_level(theta_level1: f64, ratio: usize, level: usize) -> f64 {
    // level=1 -> theta_level1
    // level=2 -> theta_level1/ratio
    // level=3 -> theta_level1/ratio^2, etc.
    if level <= 1 {
        return theta_level1.max(0.0);
    }
    let div = ratio_pow_local(ratio, level - 1) as f64;
    (theta_level1 / div).max(0.0)
}

/// Nested regrid (levels 1 and 2) for Milestone 1.
///
/// This function:
/// 1) Performs the usual Stage-2B regrid for level-1 patches.
/// 2) Optionally proposes level-2 patches **inside the level-1 refined region** by evaluating
///    the same indicator on a level-1 composite field and clustering flagged cells.
///
/// Level-2 is created only when `LLG_AMR_MAX_LEVEL >= 2`.
///
/// Returns Some((level1_rects, stats1)) if level-1 regrid occurred, or Some((level1_rects, stats1))
/// if only level-2 changed (to allow callers to log that an AMR change happened).
pub fn maybe_regrid_nested_levels(
    h: &mut AmrHierarchy2D,
    current_level1: &[Rect2i],
    policy: RegridPolicy,
    cluster_policy_level1: ClusterPolicy,
) -> Option<(Vec<Rect2i>, ClusterStats)> {
    // ---- Level 1 (existing logic) ----
    let lvl1_res = maybe_regrid_multi_patch(h, current_level1, policy, cluster_policy_level1);

    // Current level-1 rects (base-grid indices)
    let mut level1_rects: Vec<Rect2i> = h.patches.iter().map(|p| p.coarse_rect).collect();
    level1_rects.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));

    // Read max refinement level
    let max_level = std::env::var("LLG_AMR_MAX_LEVEL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);

    // If no level-1 patches exist, clear all deeper levels (>=2)
    if level1_rects.is_empty() {
        let levels_to_clear: Vec<usize> = h
            .patches_l2plus
            .iter()
            .enumerate()
            .filter(|(_, lvl)| !lvl.is_empty())
            .map(|(k, _)| k + 2)
            .collect();

        let mut cleared = false;
        for level in levels_to_clear {
            h.replace_level_patches_preserve_overlap(level, Vec::new());
            cleared = true;
        }

        if cleared {
            let stats = ClusterStats {
                max_indicator: 0.0,
                threshold: 0.0,
                flagged_cells: 0,
                components: 0,
                patches_before_merge: 0,
                patches_after_merge: 0,
            };
            return Some((level1_rects, stats));
        }

        return lvl1_res;
    }

    // If we only allow level 1, clear deeper levels (if any) and return.
    if max_level < 2 {
        let levels_to_clear: Vec<usize> = h
            .patches_l2plus
            .iter()
            .enumerate()
            .filter(|(_, lvl)| !lvl.is_empty())
            .map(|(k, _)| k + 2)
            .collect();
        let mut cleared = false;
        for level in levels_to_clear {
            h.replace_level_patches_preserve_overlap(level, Vec::new());
            cleared = true;
        }
        if cleared {
            let stats = ClusterStats {
                max_indicator: 0.0,
                threshold: 0.0,
                flagged_cells: 0,
                components: 0,
                patches_before_merge: 0,
                patches_after_merge: 0,
            };
            return Some((level1_rects, stats));
        }
        return lvl1_res;
    }


    let mut changed_deep = false;

    // Clear levels above max_level if they exist
    let levels_to_clear_above: Vec<usize> = h
        .patches_l2plus
        .iter()
        .enumerate()
        .filter(|(k, lvl)| (k + 2) > max_level && !lvl.is_empty())
        .map(|(k, _)| k + 2)
        .collect();

    for level in levels_to_clear_above {
        h.replace_level_patches_preserve_overlap(level, Vec::new());
        changed_deep = true;
    }

    // Angle-mode: use level-1 theta as base
    let theta_level1 = if cluster_policy_level1.indicator_frac < 0.0 {
        (-cluster_policy_level1.indicator_frac).max(0.0)
    } else {
        0.0
    };

    // Build nested levels 2..=max_level
    for level in 2..=max_level {
        let parent_level = level - 1;

        // Parent rects (base indices)
        let mut parent_rects = rects_at_level_base(&*h, parent_level);
        parent_rects.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));

        // If parent is empty, clear this level and everything deeper, then stop.
        if parent_rects.is_empty() {
            for lv in level..=max_level {
                h.replace_level_patches_preserve_overlap(lv, Vec::new());
            }
            changed_deep = true;
            break;
        }

        // Composite field at parent resolution r_parent = ratio^(parent_level)
        let r_parent = ratio_pow_local(h.ratio, parent_level);
        let grid_parent = crate::grid::Grid2D::new(
            h.base_grid.nx * r_parent,
            h.base_grid.ny * r_parent,
            h.base_grid.dx / (r_parent as f64),
            h.base_grid.dy / (r_parent as f64),
            h.base_grid.dz,
        );

        let mut m_parent = h.coarse.resample_to_grid(grid_parent);

        // Scatter parent-level patches into this uniform field
        if parent_level == 1 {
            for p in &h.patches {
                p.scatter_into_uniform_fine(&mut m_parent);
            }
        } else {
            let idxp = parent_level - 2;
            if let Some(v) = h.patches_l2plus.get(idxp) {
                for p in v {
                    p.scatter_into_uniform_fine(&mut m_parent);
                }
            }
        }

        // Region mask: within union of parent rects AND within material
        let n_parent = m_parent.grid.n_cells();
        let mut region_mask = vec![false; n_parent];
        mark_union_region(&h.base_grid, r_parent, &parent_rects, &mut region_mask);

        if let Some(up) = upsample_base_mask_to_ratio(&h.base_grid, h.geom_mask(), r_parent) {
            for i in 0..n_parent {
                region_mask[i] = region_mask[i] && up[i];
            }
        }

        // Cluster policy for this level
        let mut cp = cluster_policy_level1;

        // Angle-mode: progressively tighter threshold per level
        if cp.indicator_frac < 0.0 {
            let theta_l = theta_refine_at_level(theta_level1, h.ratio, level);
            cp.indicator_frac = -theta_l;
        }

        // Compute candidate rects on the parent-resolution grid
        let rects_parent = match compute_patch_rects_clustered_from_indicator(
            &m_parent,
            cp,
            Some(&region_mask),
        ) {
            Some((rects, _stats)) => rects,
            None => Vec::new(),
        };

        // Convert from parent grid index space to base-grid rects
        let mut rects_base: Vec<Rect2i> = rects_parent
            .into_iter()
            .map(|r| rect_level_r_to_base(r, r_parent, h.base_grid.nx, h.base_grid.ny))
            .collect();

        rects_base.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));
        rects_base.dedup();

        // Current rects at this level
        let mut cur = rects_at_level_base(&*h, level);
        cur.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));

        if cur != rects_base {
            h.replace_level_patches_preserve_overlap(level, rects_base);
            changed_deep = true;
        }

        // If this level ended up empty, clear deeper and stop.
        let now = rects_at_level_base(&*h, level);
        if now.is_empty() {
            for lv in (level + 1)..=max_level {
                h.replace_level_patches_preserve_overlap(lv, Vec::new());
            }
            break;
        }
    }

    // If level-1 regridded, return it.
    if let Some((new1, stats1)) = lvl1_res {
        return Some((new1, stats1));
    }

    // If only deeper levels changed, return a dummy stats row to signal an AMR change.
    if changed_deep {
        let stats = ClusterStats {
            max_indicator: 0.0,
            threshold: 0.0,
            flagged_cells: 0,
            components: 0,
            patches_before_merge: 0,
            patches_after_merge: 0,
        };
        return Some((level1_rects, stats));
    }

    None
}