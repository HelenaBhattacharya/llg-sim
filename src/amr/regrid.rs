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
    IndicatorKind, IndicatorStats,
    compute_indicator_map_for_kind,
    compute_patch_bbox_from_indicator_geom,
    compute_patch_bbox_from_angle_threshold_geom,
    indicator_angle_max_forward_geom,
};
use crate::amr::rect::Rect2i;

// ═══════════════════════════════════════════════════════════════════════════
//  RegridPolicy
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub struct RegridPolicy {
    /// Which indicator + threshold to use.
    pub indicator: IndicatorKind,
    pub buffer_cells: usize,
    /// Flag a boundary layer of this many cells (0 = disabled).
    pub boundary_layer: usize,
    pub min_change_cells: usize,
    pub min_area_change_frac: f64,
}

impl RegridPolicy {
    /// Construct from the legacy sign-convention for backward compatibility.
    ///
    /// `indicator_frac >= 0` => Grad2 { frac }
    /// `indicator_frac <  0` => Angle { theta_refine: -indicator_frac }
    pub fn from_legacy(
        indicator_frac: f64,
        buffer_cells: usize,
        min_change_cells: usize,
        min_area_change_frac: f64,
    ) -> Self {
        Self {
            indicator: IndicatorKind::from_legacy_frac(indicator_frac),
            buffer_cells,
            boundary_layer: 0,
            min_change_cells,
            min_area_change_frac,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn theta_coarsen_factor() -> f64 {
    // Allow runtime tuning without touching all benchmark policy literals.
    // Default chosen to avoid jitter while still permitting derefinement.
    if let Ok(s) = std::env::var("LLG_AMR_THETA_COARSEN_FACTOR") {
        if let Ok(v) = s.parse::<f64>() {
            return v.clamp(0.5, 0.95);
        }
    }
    0.75
}

/// General-purpose coarsen factor for relative-threshold indicators.
/// When max(indicator) drops below `coarsen_frac_factor * frac * max_at_refine_time`,
/// we de-refine.  Since we don't track `max_at_refine_time`, we instead compare the
/// current max against the current threshold: if zero cells would be flagged, de-refine.
/// This is implemented inline in `maybe_regrid_multi_patch`.
#[inline]
#[allow(dead_code)]
fn relative_coarsen_factor() -> f64 {
    if let Ok(s) = std::env::var("LLG_AMR_COARSEN_FACTOR") {
        if let Ok(v) = s.parse::<f64>() {
            return v.clamp(0.1, 0.95);
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

#[inline]
fn max_angle_on_coarse_geom(
    coarse: &crate::vector_field::VectorField2D,
    geom_mask: Option<&[bool]>,
) -> f64 {
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

/// Compute max indicator value on the coarse grid for any IndicatorKind.
/// Uses `compute_indicator_map_for_kind` and returns `max_value`.
#[inline]
#[allow(dead_code)]
fn max_indicator_on_coarse(
    kind: IndicatorKind,
    coarse: &crate::vector_field::VectorField2D,
    geom_mask: Option<&[bool]>,
) -> f64 {
    let (_map, max_val, _thresh) = compute_indicator_map_for_kind(kind, coarse, geom_mask);
    max_val
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
fn rect_intersection(a: Rect2i, b: Rect2i) -> Option<Rect2i> {
    let i0 = a.i0.max(b.i0);
    let j0 = a.j0.max(b.j0);
    let i1 = a.i1().min(b.i1());
    let j1 = a.j1().min(b.j1());
    if i1 <= i0 || j1 <= j0 {
        return None;
    }
    Some(Rect2i::new(i0, j0, i1 - i0, j1 - j0))
}

#[inline]
fn rect_contains(outer: Rect2i, inner: Rect2i) -> bool {
    inner.i0 >= outer.i0
        && inner.j0 >= outer.j0
        && inner.i1() <= outer.i1()
        && inner.j1() <= outer.j1()
}

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

fn empty_cluster_stats() -> ClusterStats {
    ClusterStats {
        max_indicator: 0.0,
        threshold: 0.0,
        flagged_cells: 0,
        components: 0,
        patches_before_merge: 0,
        patches_after_merge: 0,
        merges_blocked_by_efficiency: 0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Stage-2A: single-patch regrid
// ═══════════════════════════════════════════════════════════════════════════

/// Propose a single patch from the current coarse state.
///
/// This is mask-aware via `h.geom_mask()`.
pub fn propose_single_patch(
    h: &AmrHierarchy2D,
    policy: RegridPolicy,
) -> Option<(Rect2i, IndicatorStats)> {
    match policy.indicator {
        IndicatorKind::Angle { theta_refine } => {
            compute_patch_bbox_from_angle_threshold_geom(
                &h.coarse,
                theta_refine,
                policy.buffer_cells,
                h.geom_mask(),
            )
        }
        IndicatorKind::Grad2 { frac } => {
            compute_patch_bbox_from_indicator_geom(
                &h.coarse,
                frac,
                policy.buffer_cells,
                h.geom_mask(),
            )
        }
        // For div/curl/composite in Stage-2A single-patch mode, compute the
        // indicator map, find the bbox of flagged cells, and return it.
        _ => {
            let nx = h.coarse.grid.nx;
            let ny = h.coarse.grid.ny;
            if nx == 0 || ny == 0 {
                return None;
            }

            let (map, max_val, thresh) =
                compute_indicator_map_for_kind(policy.indicator, &h.coarse, h.geom_mask());

            if max_val <= 0.0 {
                return None;
            }

            let mut found = false;
            let mut i_min = nx - 1;
            let mut i_max = 0usize;
            let mut j_min = ny - 1;
            let mut j_max = 0usize;

            for j in 0..ny {
                for i in 0..nx {
                    if map[j * nx + i] >= thresh {
                        found = true;
                        i_min = i_min.min(i);
                        i_max = i_max.max(i);
                        j_min = j_min.min(j);
                        j_max = j_max.max(j);
                    }
                }
            }

            if !found {
                return None;
            }

            let raw = Rect2i::new(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);
            let rect = raw.dilate_clamped(policy.buffer_cells, nx, ny);

            Some((
                rect,
                IndicatorStats {
                    max: max_val,
                    threshold: thresh,
                },
            ))
        }
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
        h.replace_single_patch_preserve_overlap(new_rect);
        Some((new_rect, stats))
    } else {
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Stage-2B: clustered multi-patch regrid
// ═══════════════════════════════════════════════════════════════════════════

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
    // ── Hysteresis / derefinement logic ───────────────────────────────────
    //
    // Angle mode:  absolute threshold with explicit hysteresis band.
    //   - refine   when max_theta >= theta_refine
    //   - hold     when theta_coarsen <= max_theta < theta_refine
    //   - derefine when max_theta < theta_coarsen
    //
    // Relative-threshold modes (Grad2 / DivInplane / CurlMag / Composite):
    //   max(ind) always flags at least one cell, so we derefine only when
    //   the clustering itself returns None (all cells below threshold).
    //   This is handled naturally: compute_patch_rects_clustered returns None.

    if let IndicatorKind::Angle { theta_refine } = policy.indicator {
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
                merges_blocked_by_efficiency: 0,
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

// ═══════════════════════════════════════════════════════════════════════════
//  Nested refinement helpers
// ═══════════════════════════════════════════════════════════════════════════

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
        let fi0 = rect.i0 * r;
        let fj0 = rect.j0 * r;
        let fi1 = rect.i1() * r;
        let fj1 = rect.j1() * r;

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
    let mut r = 1usize;
    for _ in 0..level {
        r = r.saturating_mul(ratio);
    }
    r
}

/// Adjust indicator threshold per refinement level.
///
/// **Angle mode:** progressively tighter absolute threshold.
///   - level=1 → `theta_refine`
///   - level=2 → `theta_refine / ratio`
///   - level=3 → `theta_refine / ratio²`, etc.
///
/// **Relative-threshold modes** (Grad2 / DivInplane / CurlMag / Composite):
///   progressively tighter `frac` per level so that L1 casts a wide net
///   (capturing walls, boundaries, broad features) while deeper levels
///   focus only on sharp features (vortex cores, Bloch points).
///
///   - level=1 → `frac` (unchanged)
///   - level=2 → `frac × tighten_factor`
///   - level=3 → `frac × tighten_factor²`, etc.
///
///   Default tighten_factor = 1.5, overridable via `LLG_AMR_LEVEL_TIGHTEN`.
///   With `frac = 0.10`: L1 = 0.10, L2 = 0.15, L3 = 0.225.
///   Capped at 0.95 to avoid flagging nothing.
#[inline]
fn indicator_at_level(base: IndicatorKind, ratio: usize, level: usize) -> IndicatorKind {
    if level <= 1 {
        return base;
    }

    match base {
        IndicatorKind::Angle { theta_refine } => {
            let div = ratio_pow_local(ratio, level - 1) as f64;
            IndicatorKind::Angle {
                theta_refine: (theta_refine / div).max(0.0),
            }
        }
        _ => {
            // Progressive tightening for relative-threshold modes.
            let tighten_factor: f64 = std::env::var("LLG_AMR_LEVEL_TIGHTEN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.5);

            let depth = (level - 1) as i32;
            let scale = tighten_factor.powi(depth);

            match base {
                IndicatorKind::Grad2 { frac } => {
                    IndicatorKind::Grad2 { frac: (frac * scale).min(0.95) }
                }
                IndicatorKind::DivInplane { frac } => {
                    IndicatorKind::DivInplane { frac: (frac * scale).min(0.95) }
                }
                IndicatorKind::CurlMag { frac } => {
                    IndicatorKind::CurlMag { frac: (frac * scale).min(0.95) }
                }
                IndicatorKind::Composite { frac } => {
                    IndicatorKind::Composite { frac: (frac * scale).min(0.95) }
                }
                // Angle handled above; this arm is unreachable but needed for exhaustiveness.
                other => other,
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Nested multi-level regrid
// ═══════════════════════════════════════════════════════════════════════════

/// Nested regrid (levels 1..=max_level).
///
/// This function:
/// 1) Performs the usual Stage-2B regrid for level-1 patches.
/// 2) For levels 2..=max_level, proposes patches **inside the parent refined
///    region** by evaluating the indicator on a parent-resolution composite
///    field and clustering flagged cells.
///
/// `LLG_AMR_MAX_LEVEL` env var controls the deepest level (default 1).
///
/// Returns Some((level1_rects, stats)) if any AMR change occurred.
pub fn maybe_regrid_nested_levels(
    h: &mut AmrHierarchy2D,
    current_level1: &[Rect2i],
    policy: RegridPolicy,
    cluster_policy_level1: ClusterPolicy,
) -> Option<(Vec<Rect2i>, ClusterStats)> {
    // ── Level 1 (existing logic) ─────────────────────────────────────────
    let lvl1_res = maybe_regrid_multi_patch(h, current_level1, policy, cluster_policy_level1);

    let mut level1_rects: Vec<Rect2i> = h.patches.iter().map(|p| p.coarse_rect).collect();
    level1_rects.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));

    let max_level = std::env::var("LLG_AMR_MAX_LEVEL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);

    // ── If no level-1 patches, clear all deeper levels ───────────────────
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
            return Some((level1_rects, empty_cluster_stats()));
        }

        return lvl1_res;
    }

    // ── If max_level < 2, clear deeper levels and return ─────────────────
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
            return Some((level1_rects, empty_cluster_stats()));
        }
        return lvl1_res;
    }

    let mut changed_deep = false;

    // ── Clear levels above max_level if they exist ───────────────────────
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

    // ── Build nested levels 2..=max_level ────────────────────────────────

    for level in 2..=max_level {
        let parent_level = level - 1;

        // Parent rects (base indices).
        let mut parent_rects = rects_at_level_base(h, parent_level);
        parent_rects.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));

        // If parent is empty, clear this level and everything deeper, then stop.
        if parent_rects.is_empty() {
            for lv in level..=max_level {
                h.replace_level_patches_preserve_overlap(lv, Vec::new());
            }
            changed_deep = true;
            break;
        }

        // Composite field at parent resolution: r_parent = ratio^(parent_level).
        let r_parent = ratio_pow_local(h.ratio, parent_level);
        let grid_parent = crate::grid::Grid2D::new(
            h.base_grid.nx * r_parent,
            h.base_grid.ny * r_parent,
            h.base_grid.dx / (r_parent as f64),
            h.base_grid.dy / (r_parent as f64),
            h.base_grid.dz,
        );

        let mut m_parent = h.coarse.resample_to_grid(grid_parent);

        // Scatter parent-level patches into this uniform field.
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

        // Region mask: within union of parent rects AND within material.
        let n_parent = m_parent.grid.n_cells();
        let mut region_mask = vec![false; n_parent];
        mark_union_region(&h.base_grid, r_parent, &parent_rects, &mut region_mask);

        if let Some(up) = upsample_base_mask_to_ratio(&h.base_grid, h.geom_mask(), r_parent) {
            for i in 0..n_parent {
                region_mask[i] = region_mask[i] && up[i];
            }
        }

        // Cluster policy for this level: adjust indicator for nesting depth.
        let level_indicator = indicator_at_level(cluster_policy_level1.indicator, h.ratio, level);
        let cp = ClusterPolicy {
            indicator: level_indicator,
            boundary_layer: cluster_policy_level1.boundary_layer,
            buffer_cells: cluster_policy_level1.buffer_cells,
            connectivity: cluster_policy_level1.connectivity,
            merge_distance: cluster_policy_level1.merge_distance,
            min_patch_area: cluster_policy_level1.min_patch_area,
            max_patches: cluster_policy_level1.max_patches,
            min_efficiency: cluster_policy_level1.min_efficiency,
            // Coverage cap disabled for L2+: nested regions are already focused
            // by the parent-level clustering, so most of the parent patch SHOULD
            // be refined.  The cap only makes sense at L1 where the domain is the
            // full grid and broadly distributed features can degenerate into
            // "refine everything".
            max_flagged_fraction: 1.0,
        };

        // Compute candidate rects on the parent-resolution grid.
        let rects_parent = match compute_patch_rects_clustered_from_indicator(
            &m_parent,
            cp,
            Some(&region_mask),
        ) {
            Some((rects, _stats)) => rects,
            None => Vec::new(),
        };

        // Convert from parent grid index space to base-grid rects.
        let mut rects_base: Vec<Rect2i> = rects_parent
            .into_iter()
            .map(|r| rect_level_r_to_base(r, r_parent, h.base_grid.nx, h.base_grid.ny))
            .collect();

        rects_base.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));
        rects_base.dedup();

        // Ensure strict nesting: every level-L rect must be fully contained
        // in some level-(L-1) parent rect.
        if !rects_base.is_empty() {
            let mut clamped: Vec<Rect2i> = Vec::new();
            for child in rects_base.into_iter() {
                // Fast path: already contained.
                if parent_rects.iter().any(|&p| rect_contains(p, child)) {
                    clamped.push(child);
                    continue;
                }

                // Otherwise clamp to the parent with the largest intersection.
                let mut best: Option<Rect2i> = None;
                let mut best_area: usize = 0;
                for &p in &parent_rects {
                    if let Some(int) = rect_intersection(child, p) {
                        let area = int.nx * int.ny;
                        if area > best_area {
                            best_area = area;
                            best = Some(int);
                        }
                    }
                }

                if let Some(int) = best {
                    clamped.push(int);
                }
            }

            clamped.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));
            clamped.dedup();
            rects_base = clamped;
        }

        // Current rects at this level.
        let mut cur = rects_at_level_base(h, level);
        cur.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));

        if cur != rects_base {
            h.replace_level_patches_preserve_overlap(level, rects_base);
            changed_deep = true;
        }

        // If this level ended up empty, clear deeper and stop.
        let now = rects_at_level_base(h, level);
        if now.is_empty() {
            for lv in (level + 1)..=max_level {
                h.replace_level_patches_preserve_overlap(lv, Vec::new());
            }
            break;
        }
    }

    // ── Return ───────────────────────────────────────────────────────────

    // If level-1 regridded, return it.
    if let Some((new1, stats1)) = lvl1_res {
        return Some((new1, stats1));
    }

    // If only deeper levels changed, return a dummy stats row to signal an AMR change.
    if changed_deep {
        return Some((level1_rects, empty_cluster_stats()));
    }

    None
}