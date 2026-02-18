// src/amr/regrid.rs
//
// Stage-2A single-patch regridding policy + helpers.

use crate::amr::clustering::{
    ClusterPolicy, ClusterStats, compute_patch_rects_clustered_from_indicator,
};
use crate::amr::hierarchy::AmrHierarchy2D;
use crate::amr::indicator::{IndicatorStats, compute_patch_bbox_from_indicator};
use crate::amr::rect::Rect2i;

#[derive(Clone, Copy, Debug)]
pub struct RegridPolicy {
    pub indicator_frac: f64,
    pub buffer_cells: usize,
    pub min_change_cells: usize,
    pub min_area_change_frac: f64,
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
pub fn propose_single_patch(
    h: &AmrHierarchy2D,
    policy: RegridPolicy,
) -> Option<(Rect2i, IndicatorStats)> {
    compute_patch_bbox_from_indicator(&h.coarse, policy.indicator_frac, policy.buffer_cells)
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

/// Stage-2B: clustered multi-patch regrid.
///
/// We use a simple hysteresis test based on the union-bbox of all patches.
/// This avoids 1-cell jitter while staying cheap.
pub fn maybe_regrid_multi_patch(
    h: &mut AmrHierarchy2D,
    current_patches: &[Rect2i],
    policy: RegridPolicy,
    cluster_policy: ClusterPolicy,
) -> Option<(Vec<Rect2i>, ClusterStats)> {
    let (mut new_rects, stats) =
        compute_patch_rects_clustered_from_indicator(&h.coarse, cluster_policy)?;

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
