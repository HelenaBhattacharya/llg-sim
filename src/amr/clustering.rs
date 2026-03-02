// src/amr/clustering.rs
//
// Stage-2B: Multi-patch clustering for block-structured AMR.
//
// Goal:
// - Given a coarse-grid refinement indicator, identify *multiple disjoint* regions
//   that should be refined and return a set of disjoint Rect2i patches.
//
// Approach (Berger–Rigoutsos / Berger–Colella):
// 1) Compute indicator per cell via `IndicatorKind` dispatch and threshold.
// 2–4) Flag→boundary-layer→dilate loop with post-dilation coverage cap:
//     2) Flag cells above threshold.
//     3) OR-in boundary-layer flags if `boundary_layer > 0` (García-Cervera).
//     4) Dilate the flagged mask by `buffer_cells` (Berger–Colella buffer zone).
//     4b) If post-dilation coverage > `max_flagged_fraction`, raise threshold
//         and repeat from step 2.  This is essential because buffer dilation
//         can easily double the flagged area (e.g. 30% pre → 60% post).
//    This is applied to the mask, NOT to bounding boxes — critical for preventing
//    post-bisection merge-back that would undo the splitting.
// 5) Connected-components labelling on the buffered mask -> per-component raw bbox.
// 6) **Efficiency-based bisection**: for each raw bbox, if the fraction of flagged
//    cells (efficiency) is below `min_efficiency`, recursively bisect along the
//    longest axis with tight-bbox re-fitting.  This breaks rings, L-shapes, and
//    diagonal strips into small, tightly-fitting patches.
//    (García-Cervera & Roma require 70–85% grid efficiency.)
// 7) Merge overlapping bboxes (merge_distance).
// 8) Filter tiny patches below min_patch_area.
// 9) Optionally down-select to max_patches via greedy merges.

use std::collections::VecDeque;

use crate::amr::indicator::{IndicatorKind, compute_indicator_map_for_kind, flag_boundary_layer};
use crate::amr::rect::Rect2i;
use crate::geometry_mask::assert_mask_len;
use crate::vector_field::VectorField2D;

// ═══════════════════════════════════════════════════════════════════════════
//  Connectivity
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub enum Connectivity {
    Four,
    Eight,
}

// ═══════════════════════════════════════════════════════════════════════════
//  ClusterPolicy
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub struct ClusterPolicy {
    /// Which indicator to use and how to threshold it.
    pub indicator: IndicatorKind,
    /// Expand each cluster bbox by this many coarse cells.
    pub buffer_cells: usize,
    /// Connectivity used to define a "cluster" of flagged cells.
    pub connectivity: Connectivity,
    /// Merge bboxes that overlap after expanding by merge_distance.
    pub merge_distance: usize,
    /// Drop patches whose *coarse* area (nx*ny) is below this.
    pub min_patch_area: usize,
    /// Maximum number of patches to return. If 0 => no limit.
    pub max_patches: usize,
    /// Flag a boundary layer of this many cells (0 = disabled).
    /// García-Cervera criterion 2: ensures boundary integrals are
    /// evaluated at fine resolution.  These cells are OR'd into the
    /// flagged mask before connected-components.
    pub boundary_layer: usize,
    /// Minimum grid efficiency for Berger–Rigoutsos bisection.
    ///
    /// After connected-components labelling produces raw bounding boxes,
    /// any box whose efficiency (flagged_cells / total_cells) falls below
    /// this threshold is recursively bisected along its longest axis with
    /// tight-bbox re-fitting.  This breaks rings, L-shapes, and diagonal
    /// strips into small, tightly-fitting patches instead of one
    /// domain-spanning rectangle.
    ///
    /// García-Cervera & Roma (2005) require 70–85%.
    /// Set to 0.0 to disable bisection (legacy behaviour).
    pub min_efficiency: f64,
    /// Maximum fraction of domain that refined bounding boxes may cover
    /// (post-dilation, accounting for bbox expansion from min_efficiency).
    ///
    /// The cap targets final bounding-box coverage.  Because bboxes always
    /// include some unflagged cells (efficiency < 100%), the internal
    /// flagged-cell limit is scaled by `min_efficiency`:
    ///   max_flagged = max_flagged_fraction × material_cells × min_efficiency
    ///
    /// This ensures that the resulting bbox area (≈ flagged / efficiency)
    /// stays below `max_flagged_fraction` of the domain.
    ///
    /// Set to 1.0 to disable the cap.
    pub max_flagged_fraction: f64,
}

impl Default for ClusterPolicy {
    fn default() -> Self {
        Self {
            indicator: IndicatorKind::Grad2 { frac: 0.25 },
            buffer_cells: 6,
            connectivity: Connectivity::Eight,
            merge_distance: 4,
            min_patch_area: 16,
            max_patches: 8,
            boundary_layer: 0,
            min_efficiency: 0.70,
            max_flagged_fraction: 0.50,
        }
    }
}

impl ClusterPolicy {
    /// Conservative settings (smaller buffer, 4-neighbour connectivity).
    /// Useful if you want to minimize refined area and your indicator is not fragmented.
    pub fn conservative() -> Self {
        Self {
            indicator: IndicatorKind::Grad2 { frac: 0.35 },
            buffer_cells: 2,
            connectivity: Connectivity::Four,
            merge_distance: 2,
            min_patch_area: 16,
            max_patches: 8,
            boundary_layer: 0,
            min_efficiency: 0.70,
            max_flagged_fraction: 0.50,
        }
    }

    /// Settings tuned for localized textures (bubbles/vortices/skyrmions).
    pub fn tuned_local_features() -> Self {
        Self::default()
    }

    /// García-Cervera default: div-inplane at 10% threshold + 2-cell boundary layer + 70% efficiency.
    pub fn garcia_cervera() -> Self {
        Self {
            indicator: IndicatorKind::DivInplane { frac: 0.10 },
            buffer_cells: 4,
            connectivity: Connectivity::Eight,
            merge_distance: 4,
            min_patch_area: 16,
            max_patches: 0, // no limit — let bisection control patch count
            boundary_layer: 2,
            min_efficiency: 0.70,
            max_flagged_fraction: 0.50,
        }
    }

    /// Construct from the legacy sign-convention for backward compatibility.
    ///
    /// `indicator_frac >= 0` => Grad2 { frac }
    /// `indicator_frac <  0` => Angle { theta_refine: -indicator_frac }
    ///
    /// All other fields must be supplied directly.
    pub fn from_legacy(
        indicator_frac: f64,
        buffer_cells: usize,
        connectivity: Connectivity,
        merge_distance: usize,
        min_patch_area: usize,
        max_patches: usize,
    ) -> Self {
        Self {
            indicator: IndicatorKind::from_legacy_frac(indicator_frac),
            buffer_cells,
            connectivity,
            merge_distance,
            min_patch_area,
            max_patches,
            boundary_layer: 0,
            min_efficiency: 0.70,
            max_flagged_fraction: 0.50,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  ClusterStats
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub struct ClusterStats {
    pub max_indicator: f64,
    pub threshold: f64,
    pub flagged_cells: usize,
    pub components: usize,
    pub patches_before_merge: usize,
    pub patches_after_merge: usize,
    /// Number of merge candidates blocked by the efficiency guard.
    /// Nonzero values indicate that bisection successfully split a
    /// low-efficiency region (e.g. a ring) into multiple patches.
    pub merges_blocked_by_efficiency: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Internal helpers (unchanged)
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn idx(i: usize, j: usize, nx: usize) -> usize {
    j * nx + i
}

#[inline]
fn rect_area(r: Rect2i) -> usize {
    r.nx * r.ny
}

#[inline]
fn rect_union(a: Rect2i, b: Rect2i) -> Rect2i {
    let i0 = a.i0.min(b.i0);
    let j0 = a.j0.min(b.j0);
    let i1 = a.i1().max(b.i1());
    let j1 = a.j1().max(b.j1());
    Rect2i::new(i0, j0, i1 - i0, j1 - j0)
}

#[inline]
fn rects_overlap_or_near(a: Rect2i, b: Rect2i, merge_dist: usize, nx: usize, ny: usize) -> bool {
    let a_exp = a.dilate_clamped(merge_dist, nx, ny);
    a_exp.intersect(b).is_some()
}

#[inline]
fn neighbour_offsets(conn: Connectivity) -> &'static [(isize, isize)] {
    match conn {
        Connectivity::Four => &[(1, 0), (-1, 0), (0, 1), (0, -1)],
        Connectivity::Eight => &[
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ],
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Mask dilation (Berger–Colella buffer zone)
// ═══════════════════════════════════════════════════════════════════════════

/// Morphological dilation of a boolean mask by `radius` cells (Chebyshev distance).
///
/// Every cell within `radius` steps (8-connected) of an initially-flagged cell
/// becomes flagged.  This implements the Berger–Colella "buffer zone" (Section 5,
/// step 1): "A buffer zone of unflagged points is added around every flagged
/// grid point [...] By flagging neighboring points, instead of enlarging grids
/// at a later step, the area of overlap between grids is reduced."
///
/// Applying the buffer to the mask (before connected-components and bisection)
/// rather than to bounding boxes (after bisection) is essential: it prevents
/// adjacent bisected patches from overlapping after dilation and merging back
/// into one giant box.
///
/// Returns the updated flagged-cell count.
fn dilate_mask(flagged: &mut [bool], nx: usize, ny: usize, radius: usize) -> usize {
    if radius == 0 {
        return flagged.iter().filter(|&&f| f).count();
    }

    // BFS from all initially-flagged cells, limited to `radius` steps.
    let mut dist = vec![u16::MAX; nx * ny];
    let mut queue = VecDeque::new();

    for k in 0..(nx * ny) {
        if flagged[k] {
            dist[k] = 0;
            queue.push_back(k);
        }
    }

    while let Some(k) = queue.pop_front() {
        let d = dist[k];
        if d as usize >= radius {
            continue;
        }
        let i = k % nx;
        let j = k / nx;
        let nd = d + 1;

        // 8-connected neighbours (Chebyshev ball).
        for dj in -1isize..=1 {
            for di in -1isize..=1 {
                if di == 0 && dj == 0 {
                    continue;
                }
                let ni = i as isize + di;
                let nj = j as isize + dj;
                if ni < 0 || nj < 0 || ni >= nx as isize || nj >= ny as isize {
                    continue;
                }
                let nk = nj as usize * nx + ni as usize;
                if dist[nk] > nd {
                    dist[nk] = nd;
                    flagged[nk] = true;
                    queue.push_back(nk);
                }
            }
        }
    }

    flagged.iter().filter(|&&f| f).count()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Berger–Rigoutsos efficiency-based bisection
// ═══════════════════════════════════════════════════════════════════════════

/// Count flagged cells inside a rectangle.
#[inline]
fn count_flagged_in_rect(flagged: &[bool], rect: Rect2i, nx: usize) -> usize {
    let mut count = 0;
    for j in rect.j0..rect.j1() {
        let row = j * nx;
        for i in rect.i0..rect.i1() {
            if flagged[row + i] {
                count += 1;
            }
        }
    }
    count
}

/// Compute the tight bounding box of flagged cells within a search region.
///
/// Returns None if no flagged cells exist in the region.
fn tight_bbox_of_flagged(flagged: &[bool], region: Rect2i, nx: usize) -> Option<Rect2i> {
    let mut i_min = region.i1();
    let mut i_max = region.i0;
    let mut j_min = region.j1();
    let mut j_max = region.j0;
    let mut found = false;

    for j in region.j0..region.j1() {
        let row = j * nx;
        for i in region.i0..region.i1() {
            if flagged[row + i] {
                found = true;
                if i < i_min {
                    i_min = i;
                }
                if i > i_max {
                    i_max = i;
                }
                if j < j_min {
                    j_min = j;
                }
                if j > j_max {
                    j_max = j;
                }
            }
        }
    }

    if !found {
        return None;
    }
    Some(Rect2i::new(
        i_min,
        j_min,
        i_max - i_min + 1,
        j_max - j_min + 1,
    ))
}

/// Recursively bisect a bounding box if its grid efficiency is below the threshold.
///
/// This is the core of the Berger–Rigoutsos grid generation algorithm:
///
/// 1. Compute efficiency = flagged_cells / total_cells for the bounding box.
/// 2. If efficiency ≥ `min_efficiency` (or the box is too small to split), accept it.
/// 3. Otherwise, split along the longest axis at the midpoint.
/// 4. For each half, compute the *tight* bounding box of the flagged cells
///    within that half (this is the key re-tightening step), then recurse.
///
/// The tight-bbox step is what makes this work for rings: after splitting a
/// ring's bounding box in half horizontally, each half's tight bbox collapses
/// to just the arc segment, discarding the empty interior.
///
/// `min_size` prevents infinite recursion on very small boxes.
fn bisect_if_inefficient(
    flagged: &[bool],
    rect: Rect2i,
    min_efficiency: f64,
    min_size: usize,
    nx: usize,
) -> Vec<Rect2i> {
    let area = rect.nx * rect.ny;
    if area == 0 {
        return Vec::new();
    }

    let n_flagged = count_flagged_in_rect(flagged, rect, nx);
    if n_flagged == 0 {
        return Vec::new();
    }

    let efficiency = n_flagged as f64 / area as f64;

    // Accept: efficient enough, or too small to split meaningfully.
    // Each half must be at least `min_size` cells along the split axis.
    let splittable = if rect.nx >= rect.ny {
        rect.nx >= min_size * 2
    } else {
        rect.ny >= min_size * 2
    };

    if efficiency >= min_efficiency || !splittable {
        return vec![rect];
    }

    // Split along the longest axis at the midpoint.
    let (region_a, region_b) = if rect.nx >= rect.ny {
        let mid = rect.i0 + rect.nx / 2;
        (
            Rect2i::new(rect.i0, rect.j0, mid - rect.i0, rect.ny),
            Rect2i::new(mid, rect.j0, rect.i1() - mid, rect.ny),
        )
    } else {
        let mid = rect.j0 + rect.ny / 2;
        (
            Rect2i::new(rect.i0, rect.j0, rect.nx, mid - rect.j0),
            Rect2i::new(rect.i0, mid, rect.nx, rect.j1() - mid),
        )
    };

    let mut result = Vec::new();

    // For each half, compute the tight bbox of flagged cells within it,
    // then recurse.  The tight-bbox step is critical: it re-fits the
    // bounding box so that empty strips at the edges of each half are
    // discarded, dramatically improving efficiency for ring/arc shapes.
    if let Some(tight_a) = tight_bbox_of_flagged(flagged, region_a, nx) {
        result.extend(bisect_if_inefficient(
            flagged,
            tight_a,
            min_efficiency,
            min_size,
            nx,
        ));
    }
    if let Some(tight_b) = tight_bbox_of_flagged(flagged, region_b, nx) {
        result.extend(bisect_if_inefficient(
            flagged,
            tight_b,
            min_efficiency,
            min_size,
            nx,
        ));
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main clustering entry point
// ═══════════════════════════════════════════════════════════════════════════

/// Compute clustered patch rectangles from an indicator on the *coarse* grid.
///
/// Pipeline (Berger–Colella / Berger–Rigoutsos):
///   1. Indicator map + threshold (dispatched by `IndicatorKind`).
///   2–4. Flag → boundary-layer → dilate loop with post-dilation cap:
///     2. Flag cells above threshold.
///     3. OR-in boundary-layer flags (García-Cervera criterion 2).
///     4. Dilate flagged mask by `buffer_cells` (Berger–Colella buffer).
///     4b. If post-dilation coverage > `max_flagged_fraction`, raise
///         threshold by 5% of max_indicator and retry from step 2.
///   5. Connected components → raw bounding boxes.
///   6. **Berger–Rigoutsos bisection**: split low-efficiency boxes.
///   7. Merge overlapping / near boxes.
///   8. Filter tiny patches, enforce `max_patches`.
///
/// The buffer is applied at the mask level (step 4), NOT to bounding boxes
/// after bisection.  This is essential: bbox-level dilation causes adjacent
/// bisected arc patches to overlap and merge back into one giant box.
///
/// If `geom_mask` is provided:
/// - Vacuum cells (mask=false) have indicator 0.
/// - Neighbours outside the mask are treated as equal to the center cell
///   (free boundary), so the indicator does not artificially spike at
///   material–vacuum edges.
pub fn compute_patch_rects_clustered_from_indicator(
    coarse: &VectorField2D,
    policy: ClusterPolicy,
    geom_mask: Option<&[bool]>,
) -> Option<(Vec<Rect2i>, ClusterStats)> {
    let nx = coarse.grid.nx;
    let ny = coarse.grid.ny;
    if nx == 0 || ny == 0 {
        return None;
    }

    if let Some(msk) = geom_mask {
        assert_mask_len(msk, &coarse.grid);
    }

    // ── 1) Compute indicator map + max + threshold via enum dispatch ─────

    let (indicator_map, max_ind, thresh) =
        compute_indicator_map_for_kind(policy.indicator, coarse, geom_mask);

    if max_ind <= 0.0 {
        return None;
    }

    // For Angle mode: if the absolute threshold exceeds the max angle,
    // nothing will be flagged.
    if let IndicatorKind::Angle { theta_refine } = policy.indicator {
        if theta_refine > max_ind {
            return None;
        }
        if theta_refine == 0.0 {
            return None; // would flag everything
        }
    }

    // ── 2–4) Flag → boundary-layer → dilate, with post-dilation cap ────
    //
    // The coverage cap must check the FINAL (post-dilation) flagged count,
    // not the pre-dilation count.  With buffer_cells=6, dilation can easily
    // double the flagged area: 30% pre-dilation → 60% post-dilation.
    // A pre-dilation cap would never fire, yet the patches cover most of
    // the domain.
    //
    // Pipeline per iteration:
    //   2) Flag cells where indicator ≥ thresh.
    //   3) OR-in boundary-layer flags (García-Cervera criterion 2).
    //      Boundary flags are indicator-independent and computed once.
    //   4) Dilate the mask by buffer_cells (Berger–Colella buffer zone).
    //   4b) Check post-dilation coverage against max_flagged_fraction.
    //        If exceeded, raise thresh by 5% of max_indicator and retry
    //        from step 2.
    //
    // Only relative-threshold modes (Grad2, DivInplane, Composite) are
    // subject to the cap.  Boundary-layer flags are always preserved:
    // they are OR'd in after indicator flagging, so raising the indicator
    // threshold never suppresses boundary flags.
    //
    // Berger & Colella (1989) §5 step 1: "A buffer zone of unflagged
    // points is added around every flagged grid point [...]  By flagging
    // neighboring points, instead of enlarging grids at a later step,
    // the area of overlap between grids is reduced."
    //
    // Applying the buffer to the mask (before CC and bisection) rather
    // than to bounding boxes (after bisection) prevents adjacent bisected
    // arc patches from overlapping and merging back into one giant box.

    // Pre-compute boundary-layer flags (same every iteration).
    let boundary_flags: Option<Vec<bool>> = if policy.boundary_layer > 0 {
        Some(flag_boundary_layer(nx, ny, policy.boundary_layer, geom_mask))
    } else {
        None
    };

    // Coverage cap limit (post-dilation).
    //
    // The cap targets bounding-box coverage, not raw flagged-cell count.
    // Because bboxes always include some unflagged cells (efficiency < 100%),
    // bbox_area ≈ flagged_cells / min_efficiency.  To keep bbox coverage
    // below max_flagged_fraction, we must cap flagged cells at:
    //   max_allowed = max_flagged_fraction × material_cells × min_efficiency
    //
    // Example: 50% cap, 70% efficiency, 32768-cell domain
    //   max_allowed = 0.50 × 32768 × 0.70 = 11468 flagged cells
    //   → bbox area ≈ 11468 / 0.70 ≈ 16384 = 50% of domain  ✓
    //
    // Without the efficiency correction, 50% flagged cells produce
    // ~71% bbox coverage (50% / 0.70), defeating the cap.
    let cap_enabled = policy.indicator.is_relative() && policy.max_flagged_fraction < 1.0;
    let material_cells = geom_mask.map_or(nx * ny, |m| m.iter().filter(|&&v| v).count());
    let eff_factor = if policy.min_efficiency > 0.0 {
        policy.min_efficiency.clamp(0.3, 1.0)
    } else {
        0.70 // conservative default if bisection disabled
    };
    let max_allowed = if cap_enabled {
        ((policy.max_flagged_fraction * material_cells as f64 * eff_factor) as usize).max(1)
    } else {
        nx * ny // effectively disabled
    };

    let mut thresh = thresh; // shadow to allow mutation
    let mut flagged = vec![false; nx * ny];
    let mut flagged_cells: usize;

    loop {
        // ── 2) Flag cells from indicator ─────────────────────────────────
        flagged_cells = 0;
        for k in 0..(nx * ny) {
            flagged[k] = indicator_map[k] >= thresh;
            if flagged[k] {
                flagged_cells += 1;
            }
        }

        // ── 3) OR-in boundary-layer flags ────────────────────────────────
        if let Some(ref bf) = boundary_flags {
            for k in 0..(nx * ny) {
                if bf[k] && !flagged[k] {
                    flagged[k] = true;
                    flagged_cells += 1;
                }
            }
        }

        if flagged_cells == 0 {
            return None;
        }

        // ── 4) Dilate the flagged mask by buffer_cells ───────────────────
        flagged_cells = dilate_mask(&mut flagged, nx, ny, policy.buffer_cells);

        // ── 4b) Post-dilation coverage cap ───────────────────────────────
        //
        // If the dilated mask exceeds max_flagged_fraction of material
        // cells, raise the indicator threshold and rebuild the mask.
        // This prevents broadly distributed features (domain-wall sweeps)
        // from degenerating into "refine everything" after buffer dilation,
        // while leaving localised features (vortex core, disk edge)
        // completely unaffected.
        if flagged_cells <= max_allowed || !cap_enabled || thresh >= 0.95 * max_ind {
            break;
        }

        thresh += 0.05 * max_ind;
    }

    if flagged_cells == 0 {
        return None;
    }

    // ── 5) Connected components -> per-component raw bounding boxes ──────

    let mut visited = vec![false; nx * ny];
    let mut raw_rects: Vec<Rect2i> = Vec::new();
    let neigh = neighbour_offsets(policy.connectivity);

    for j0 in 0..ny {
        for i0 in 0..nx {
            let id0 = idx(i0, j0, nx);
            if !flagged[id0] || visited[id0] {
                continue;
            }

            // BFS flood-fill for this component.
            let mut q = VecDeque::new();
            q.push_back((i0 as isize, j0 as isize));
            visited[id0] = true;

            let mut i_min = i0;
            let mut i_max = i0;
            let mut j_min = j0;
            let mut j_max = j0;

            while let Some((ii, jj)) = q.pop_front() {
                let ii_u = ii as usize;
                let jj_u = jj as usize;

                i_min = i_min.min(ii_u);
                i_max = i_max.max(ii_u);
                j_min = j_min.min(jj_u);
                j_max = j_max.max(jj_u);

                for (di, dj) in neigh {
                    let ni = ii + di;
                    let nj = jj + dj;
                    if ni < 0 || nj < 0 {
                        continue;
                    }
                    let niu = ni as usize;
                    let nju = nj as usize;
                    if niu >= nx || nju >= ny {
                        continue;
                    }
                    let nid = idx(niu, nju, nx);
                    if flagged[nid] && !visited[nid] {
                        visited[nid] = true;
                        q.push_back((ni, nj));
                    }
                }
            }

            let raw = Rect2i::new(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);
            raw_rects.push(raw);
        }
    }

    let components = raw_rects.len();

    // ── 6) Berger–Rigoutsos efficiency-based bisection ───────────────────
    //
    // For each connected-component bbox, check if its grid efficiency
    // (flagged_cells / total_cells) meets the threshold.  If not,
    // recursively bisect along the longest axis with tight-bbox
    // re-fitting until each sub-box is efficient or indivisible.
    //
    // This breaks ring-shaped flagged regions (e.g. disk boundary) into
    // multiple small arc patches instead of one domain-spanning box.

    let min_eff = policy.min_efficiency.max(0.0);

    // Minimum half-size for a split (cells along the split axis).
    // Uses max(2, sqrt(min_patch_area)) to avoid generating tiny slivers
    // that would just be filtered away later.
    let min_split_size = if policy.min_patch_area > 0 {
        2usize.max((policy.min_patch_area as f64).sqrt().ceil() as usize)
    } else {
        2
    };

    let mut bisected_rects: Vec<Rect2i> = Vec::new();

    if min_eff > 0.0 {
        for raw in &raw_rects {
            bisected_rects.extend(bisect_if_inefficient(
                &flagged,
                *raw,
                min_eff,
                min_split_size,
                nx,
            ));
        }
    } else {
        // Bisection disabled: pass raw rects through unchanged.
        bisected_rects = raw_rects;
    }

    let patches_before_merge = bisected_rects.len();

    // Bounding boxes already include the buffer zone (applied at the mask
    // level in step 4).  No bbox dilation needed.
    let mut rects: Vec<Rect2i> = bisected_rects;

    // ── 7) Merge overlapping / near rectangles (efficiency-guarded) ─────
    //
    // Berger & Colella (1989) §5: "Grids are merged if the single resulting
    // grid has a smaller cost.  The cost function [...] is proportional to
    // mn + m + n.  Grids are merged if the new grid is relatively more
    // efficient than the two smaller grids."
    //
    // Critical fix: the merge must NOT undo bisection.  If bisection split a
    // ring into 8 arc patches, merging adjacent arcs back together recreates
    // the domain-spanning box.  We guard each merge by checking that the
    // union's efficiency (flagged_cells / union_area) stays above
    // `min_efficiency`.  This prevents chain-merging arc patches back into
    // one giant box while still allowing genuinely overlapping patches to
    // consolidate.

    let mut merges_blocked = 0usize;
    let mut merged = true;
    while merged {
        merged = false;

        'outer: for a in 0..rects.len() {
            for b in (a + 1)..rects.len() {
                if rects_overlap_or_near(rects[a], rects[b], policy.merge_distance, nx, ny)
                    || rects_overlap_or_near(rects[b], rects[a], policy.merge_distance, nx, ny)
                {
                    let u = rect_union(rects[a], rects[b]);

                    // Efficiency guard: only merge if the union maintains
                    // acceptable grid efficiency.  Without this, adjacent
                    // arc patches from ring bisection chain-merge into one
                    // domain-spanning box.
                    if min_eff > 0.0 {
                        let u_area = rect_area(u);
                        let u_flagged = count_flagged_in_rect(&flagged, u, nx);
                        let u_eff = if u_area > 0 {
                            u_flagged as f64 / u_area as f64
                        } else {
                            0.0
                        };
                        if u_eff < min_eff {
                            // Merging would create an inefficient box — skip.
                            merges_blocked += 1;
                            continue;
                        }
                    }

                    rects[a] = u;
                    rects.swap_remove(b);
                    merged = true;
                    break 'outer;
                }
            }
        }
    }

    // ── 8) Filter tiny patches ───────────────────────────────────────────

    if policy.min_patch_area > 0 {
        rects.retain(|&r| rect_area(r) >= policy.min_patch_area);
    }

    // ── 9) Enforce max_patches by greedily merging cheapest pair ─────────
    //
    // When forced to reduce patch count, prefer merges that maintain good
    // efficiency.  Cost = wasted area = union_area - (area_a + area_b).
    // If the union's efficiency would drop below min_efficiency, add a
    // large penalty to discourage that merge (but don't block it — we
    // must satisfy max_patches).

    let max_p = if policy.max_patches == 0 {
        usize::MAX
    } else {
        policy.max_patches
    };

    while rects.len() > max_p {
        let mut best_i = 0usize;
        let mut best_j = 1usize;
        let mut best_cost = usize::MAX;

        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                let u = rect_union(rects[i], rects[j]);
                let u_area = rect_area(u);
                let base_cost = u_area.saturating_sub(rect_area(rects[i]) + rect_area(rects[j]));

                // Penalise merges that create inefficient boxes.
                let cost = if min_eff > 0.0 {
                    let u_flagged = count_flagged_in_rect(&flagged, u, nx);
                    let u_eff = if u_area > 0 {
                        u_flagged as f64 / u_area as f64
                    } else {
                        0.0
                    };
                    if u_eff < min_eff {
                        // Heavy penalty: prefer other merges first.
                        base_cost.saturating_add(u_area * 4)
                    } else {
                        base_cost
                    }
                } else {
                    base_cost
                };

                if cost < best_cost {
                    best_cost = cost;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let u = rect_union(rects[best_i], rects[best_j]);
        rects[best_i] = u;
        rects.swap_remove(best_j);
    }

    // ── 10) Stable ordering for deterministic logs ────────────────────────

    rects.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));

    let stats = ClusterStats {
        max_indicator: max_ind,
        threshold: thresh,
        flagged_cells,
        components,
        patches_before_merge,
        patches_after_merge: rects.len(),
        merges_blocked_by_efficiency: merges_blocked,
    };

    Some((rects, stats))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;
    use crate::vector_field::VectorField2D;

    /// Helper: create a small grid + field.
    fn make_field(nx: usize, ny: usize) -> VectorField2D {
        let grid = Grid2D::new(nx, ny, 1.0, 1.0, 1.0);
        VectorField2D::new(grid)
    }

    /// Helper: set a single cell's magnetisation.
    fn set_cell(f: &mut VectorField2D, i: usize, j: usize, v: [f64; 3]) {
        let idx = f.idx(i, j);
        f.data[idx] = v;
    }

    // ── from_legacy round-trip ───────────────────────────────────────────

    #[test]
    fn from_legacy_positive_gives_grad2() {
        let p = ClusterPolicy::from_legacy(0.25, 4, Connectivity::Eight, 2, 16, 8);
        assert_eq!(p.indicator, IndicatorKind::Grad2 { frac: 0.25 });
        assert_eq!(p.boundary_layer, 0);
        assert!((p.min_efficiency - 0.70).abs() < 1e-12);
        assert!((p.max_flagged_fraction - 0.50).abs() < 1e-12);
    }

    #[test]
    fn from_legacy_negative_gives_angle() {
        let p = ClusterPolicy::from_legacy(-0.4, 4, Connectivity::Eight, 2, 16, 8);
        match p.indicator {
            IndicatorKind::Angle { theta_refine } => {
                assert!((theta_refine - 0.4).abs() < 1e-12);
            }
            _ => panic!("expected Angle"),
        }
    }

    // ── garcia_cervera preset ────────────────────────────────────────────

    #[test]
    fn garcia_cervera_preset_uses_div() {
        let p = ClusterPolicy::garcia_cervera();
        assert_eq!(p.indicator, IndicatorKind::DivInplane { frac: 0.10 });
        assert_eq!(p.boundary_layer, 2);
        assert!((p.min_efficiency - 0.70).abs() < 1e-12);
        assert!((p.max_flagged_fraction - 0.50).abs() < 1e-12);
    }

    // ── Basic clustering with grad2 ──────────────────────────────────────

    #[test]
    fn uniform_field_returns_none() {
        let mut f = make_field(16, 16);
        for v in f.data.iter_mut() {
            *v = [0.0, 0.0, 1.0];
        }
        let policy = ClusterPolicy::default();
        let result = compute_patch_rects_clustered_from_indicator(&f, policy, None);
        assert!(result.is_none(), "uniform field should produce no patches");
    }

    #[test]
    fn single_feature_produces_one_patch() {
        // Uniform +z except a small region rotated to +x.
        let mut f = make_field(32, 32);
        for v in f.data.iter_mut() {
            *v = [0.0, 0.0, 1.0];
        }
        // Put a 4×4 block of +x at center.
        for j in 14..18 {
            for i in 14..18 {
                set_cell(&mut f, i, j, [1.0, 0.0, 0.0]);
            }
        }

        let policy = ClusterPolicy {
            indicator: IndicatorKind::Grad2 { frac: 0.25 },
            buffer_cells: 2,
            connectivity: Connectivity::Eight,
            merge_distance: 2,
            min_patch_area: 4,
            max_patches: 8,
            boundary_layer: 0,
            min_efficiency: 0.70,
            max_flagged_fraction: 0.50,
        };

        let result = compute_patch_rects_clustered_from_indicator(&f, policy, None);
        assert!(result.is_some(), "should detect the feature");
        let (rects, stats) = result.unwrap();
        assert_eq!(rects.len(), 1, "single feature -> one patch");
        assert!(stats.flagged_cells > 0);

        // Patch should contain the feature region [14..18].
        let r = rects[0];
        assert!(r.i0 <= 14 && r.i1() >= 18, "patch should cover feature x");
        assert!(r.j0 <= 14 && r.j1() >= 18, "patch should cover feature y");
    }

    #[test]
    fn two_features_produce_two_patches() {
        let mut f = make_field(64, 16);
        for v in f.data.iter_mut() {
            *v = [0.0, 0.0, 1.0];
        }
        // Feature A at left.
        for j in 6..10 {
            for i in 4..8 {
                set_cell(&mut f, i, j, [1.0, 0.0, 0.0]);
            }
        }
        // Feature B at right (well separated).
        for j in 6..10 {
            for i in 52..56 {
                set_cell(&mut f, i, j, [1.0, 0.0, 0.0]);
            }
        }

        let policy = ClusterPolicy {
            indicator: IndicatorKind::Grad2 { frac: 0.25 },
            buffer_cells: 2,
            connectivity: Connectivity::Eight,
            merge_distance: 2,
            min_patch_area: 4,
            max_patches: 8,
            boundary_layer: 0,
            min_efficiency: 0.70,
            max_flagged_fraction: 0.50,
        };

        let result = compute_patch_rects_clustered_from_indicator(&f, policy, None);
        assert!(result.is_some());
        let (rects, _stats) = result.unwrap();
        assert_eq!(rects.len(), 2, "two separated features -> two patches");
    }

    // ── DivInplane indicator dispatch ────────────────────────────────────

    #[test]
    fn div_inplane_flags_divergent_region() {
        // Build a field where the left half has M = (+x, 0, 0) and the
        // right half has M = (-x, 0, 0).  The interface at the center
        // has large |∇·(M₁, M₂)|.
        let nx = 32;
        let ny = 8;
        let mut f = make_field(nx, ny);
        for j in 0..ny {
            for i in 0..nx {
                let mx = if i < nx / 2 { 1.0 } else { -1.0 };
                set_cell(&mut f, i, j, [mx, 0.0, 0.0]);
            }
        }

        let policy = ClusterPolicy {
            indicator: IndicatorKind::DivInplane { frac: 0.10 },
            buffer_cells: 2,
            connectivity: Connectivity::Eight,
            merge_distance: 2,
            min_patch_area: 4,
            max_patches: 8,
            boundary_layer: 0,
            min_efficiency: 0.70,
            max_flagged_fraction: 0.50,
        };

        let result = compute_patch_rects_clustered_from_indicator(&f, policy, None);
        assert!(result.is_some(), "should detect the wall");
        let (rects, stats) = result.unwrap();
        assert!(rects.len() >= 1);
        assert!(stats.flagged_cells > 0);

        // The patch should straddle the interface at i = nx/2.
        let r = rects[0];
        assert!(
            r.i0 <= nx / 2 && r.i1() >= nx / 2,
            "patch should cover the divergence interface"
        );
    }

    // ── Boundary layer integration ───────────────────────────────────────

    #[test]
    fn boundary_layer_increases_flagged_cells() {
        // Uniform field — without boundary layer, no cells are flagged.
        // With boundary layer, edge cells should be flagged.
        let mut f = make_field(16, 16);
        // Add a small feature so the indicator fires at all.
        for v in f.data.iter_mut() {
            *v = [0.0, 0.0, 1.0];
        }
        set_cell(&mut f, 8, 8, [1.0, 0.0, 0.0]);

        let policy_no_bl = ClusterPolicy {
            indicator: IndicatorKind::Grad2 { frac: 0.25 },
            buffer_cells: 1,
            connectivity: Connectivity::Four,
            merge_distance: 1,
            min_patch_area: 1,
            max_patches: 0,
            boundary_layer: 0,
            min_efficiency: 0.0,
            max_flagged_fraction: 1.0,
        };

        let policy_with_bl = ClusterPolicy {
            boundary_layer: 2,
            ..policy_no_bl
        };

        let r1 = compute_patch_rects_clustered_from_indicator(&f, policy_no_bl, None);
        let r2 = compute_patch_rects_clustered_from_indicator(&f, policy_with_bl, None);

        let fc1 = r1.as_ref().map(|(_, s)| s.flagged_cells).unwrap_or(0);
        let fc2 = r2.as_ref().map(|(_, s)| s.flagged_cells).unwrap_or(0);

        assert!(
            fc2 > fc1,
            "boundary_layer=2 should flag more cells: {} vs {}",
            fc2,
            fc1
        );
    }

    // ── Composite indicator dispatch ─────────────────────────────────────

    #[test]
    fn composite_detects_wall_and_vortex() {
        // Build a field where:
        //   Left side: sharp M₁ sign change (high div, low curl)
        //   Right side: vortex-like rotation (low div, high curl/grad)
        let nx = 64;
        let ny = 16;
        let mut f = make_field(nx, ny);
        for j in 0..ny {
            for i in 0..nx {
                if i < nx / 2 {
                    // Wall region: step function in M₁.
                    let mx = if i < nx / 4 { 1.0 } else { -1.0 };
                    set_cell(&mut f, i, j, [mx, 0.0, 0.0]);
                } else {
                    // Vortex-like: M = (-y, x, 0) locally.
                    let cx = (nx / 2 + nx / 4) as f64;
                    let cy = (ny / 2) as f64;
                    let dx = i as f64 - cx;
                    let dy = j as f64 - cy;
                    let r = (dx * dx + dy * dy).sqrt().max(1.0);
                    set_cell(&mut f, i, j, [-dy / r, dx / r, 0.0]);
                }
            }
        }

        let policy = ClusterPolicy {
            indicator: IndicatorKind::Composite { frac: 0.15 },
            buffer_cells: 2,
            connectivity: Connectivity::Eight,
            merge_distance: 4,
            min_patch_area: 4,
            max_patches: 8,
            boundary_layer: 0,
            min_efficiency: 0.70,
            max_flagged_fraction: 0.50,
        };

        let result = compute_patch_rects_clustered_from_indicator(&f, policy, None);
        assert!(result.is_some(), "composite should detect features");
        let (rects, stats) = result.unwrap();
        assert!(stats.flagged_cells > 0);
        // Should have patches in both the wall region and the vortex region.
        // (Exact count depends on merge parameters, but at least 1.)
        assert!(rects.len() >= 1, "should produce at least one patch");
    }

    // ── max_patches enforcement ──────────────────────────────────────────

    #[test]
    fn max_patches_merges_down() {
        // Three well-separated features with max_patches=2.
        let mut f = make_field(96, 8);
        for v in f.data.iter_mut() {
            *v = [0.0, 0.0, 1.0];
        }
        for i in 4..8 {
            set_cell(&mut f, i, 4, [1.0, 0.0, 0.0]);
        }
        for i in 44..48 {
            set_cell(&mut f, i, 4, [1.0, 0.0, 0.0]);
        }
        for i in 84..88 {
            set_cell(&mut f, i, 4, [1.0, 0.0, 0.0]);
        }

        let policy = ClusterPolicy {
            indicator: IndicatorKind::Grad2 { frac: 0.10 },
            buffer_cells: 1,
            connectivity: Connectivity::Four,
            merge_distance: 1,
            min_patch_area: 1,
            max_patches: 2,
            boundary_layer: 0,
            min_efficiency: 0.70,
            max_flagged_fraction: 0.50,
        };

        let result = compute_patch_rects_clustered_from_indicator(&f, policy, None);
        assert!(result.is_some());
        let (rects, _) = result.unwrap();
        assert!(
            rects.len() <= 2,
            "max_patches=2 should enforce at most 2 patches, got {}",
            rects.len()
        );
    }

    // ── Bisection unit tests ────────────────────────────────────────────

    #[test]
    fn bisect_splits_inefficient_ring() {
        // 32×32 grid with only a 2-cell-wide ring flagged around the edge.
        // Ring efficiency ≈ 224/1024 ≈ 22% — well below 70%.
        let nx = 32usize;
        let ny = 32usize;
        let mut flagged = vec![false; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let on_edge = i < 2 || i >= nx - 2 || j < 2 || j >= ny - 2;
                flagged[j * nx + i] = on_edge;
            }
        }

        let full_box = Rect2i::new(0, 0, nx, ny);

        // Disabled bisection: 1 box.
        let no_bisect = bisect_if_inefficient(&flagged, full_box, 0.0, 2, nx);
        assert_eq!(no_bisect.len(), 1, "disabled bisection should return 1 box");

        // Enabled at 70%: should split into multiple smaller boxes.
        let bisected = bisect_if_inefficient(&flagged, full_box, 0.70, 2, nx);
        assert!(
            bisected.len() > 1,
            "bisection should split the ring into multiple patches, got {}",
            bisected.len()
        );

        // Each box should be smaller than the full domain.
        for r in &bisected {
            assert!(
                rect_area(*r) < rect_area(full_box),
                "each bisected patch should be smaller than the full domain"
            );
        }
    }

    #[test]
    fn bisect_preserves_efficient_solid_block() {
        // Solid 8×8 block — 100% efficiency, should not be split.
        let nx = 32usize;
        let ny = 32usize;
        let mut flagged = vec![false; nx * ny];
        for j in 12..20 {
            for i in 12..20 {
                flagged[j * nx + i] = true;
            }
        }

        let bbox = Rect2i::new(12, 12, 8, 8);
        let result = bisect_if_inefficient(&flagged, bbox, 0.70, 2, nx);
        assert_eq!(result.len(), 1, "100% efficient box should not be split");
        assert_eq!(result[0], bbox);
    }

    #[test]
    fn bisect_empty_region_gives_nothing() {
        let nx = 16usize;
        let flagged = vec![false; nx * 16];
        let bbox = Rect2i::new(0, 0, 16, 16);
        let result = bisect_if_inefficient(&flagged, bbox, 0.70, 2, nx);
        assert!(result.is_empty());
    }

    // ── Full-pipeline test: disk ring with boundary_layer ────────────────

    #[test]
    fn disk_boundary_ring_produces_multiple_patches() {
        // Simulate the masked-disk scenario:
        // 48×48 grid, circular disk mask, in-plane vortex magnetisation.
        // boundary_layer=2 flags a ring of cells around the disk edge.
        // With bisection at 70%, this ring should NOT produce one giant
        // 48×48 box; it should produce multiple small arc patches.
        let nx = 48;
        let ny = 48;
        let cx = 24.0;
        let cy = 24.0;
        let radius = 18.0;

        let mut f = make_field(nx, ny);
        let mut mask = vec![false; nx * ny];

        for j in 0..ny {
            for i in 0..nx {
                let dx = i as f64 + 0.5 - cx;
                let dy = j as f64 + 0.5 - cy;
                let r = (dx * dx + dy * dy).sqrt();
                let k = j * nx + i;

                if r <= radius {
                    mask[k] = true;
                    let rm = r.max(1.0);
                    set_cell(&mut f, i, j, [-dy / rm, dx / rm, 0.0]);
                } else {
                    set_cell(&mut f, i, j, [0.0, 0.0, 0.0]);
                }
            }
        }

        let policy = ClusterPolicy {
            indicator: IndicatorKind::Composite { frac: 0.25 },
            buffer_cells: 2,
            connectivity: Connectivity::Eight,
            merge_distance: 2,
            min_patch_area: 8,
            max_patches: 0, // no limit
            boundary_layer: 2,
            min_efficiency: 0.70,
            max_flagged_fraction: 0.50,
        };

        let result = compute_patch_rects_clustered_from_indicator(&f, policy, Some(&mask));
        assert!(result.is_some(), "should produce patches");
        let (rects, stats) = result.unwrap();

        let total_patch_area: usize = rects.iter().map(|r| rect_area(*r)).sum();
        let domain_area = nx * ny;

        // Key assertions:
        // 1) Multiple patches (ring was bisected, not wrapped in one giant box).
        assert!(
            rects.len() >= 2,
            "boundary ring should produce >=2 patches via bisection, got {}",
            rects.len()
        );
        // 2) Total patch area is much less than full domain.
        assert!(
            total_patch_area < domain_area * 3 / 4,
            "total patch area ({}) should be < 75% of domain ({})",
            total_patch_area,
            domain_area
        );

        eprintln!(
            "[test] disk ring: {} patches, {} flagged, total area {}/{} ({:.0}%)",
            rects.len(),
            stats.flagged_cells,
            total_patch_area,
            domain_area,
            100.0 * total_patch_area as f64 / domain_area as f64,
        );
    }

    // ── Regression: bisection disabled still works ───────────────────────

    #[test]
    fn bisection_disabled_gives_legacy_behaviour() {
        let mut f = make_field(32, 32);
        for v in f.data.iter_mut() {
            *v = [0.0, 0.0, 1.0];
        }
        for j in 14..18 {
            for i in 14..18 {
                set_cell(&mut f, i, j, [1.0, 0.0, 0.0]);
            }
        }

        let policy = ClusterPolicy {
            min_efficiency: 0.0, // disabled
            ..ClusterPolicy::default()
        };

        let result = compute_patch_rects_clustered_from_indicator(&f, policy, None);
        assert!(result.is_some());
        let (rects, _) = result.unwrap();
        assert_eq!(rects.len(), 1);
    }

    // ── Coverage cap ────────────────────────────────────────────────────

    #[test]
    fn coverage_cap_raises_threshold_when_too_many_flagged() {
        // Build a 32×32 field where most cells have a non-trivial
        // gradient (broad feature).  With frac=0.10 and no cap, nearly
        // everything gets flagged.  With cap=0.50, the threshold should
        // be raised to keep post-dilation coverage ≤ 50%.
        let nx = 32;
        let ny = 32;
        let mut f = make_field(nx, ny);
        for j in 0..ny {
            for i in 0..nx {
                // Smooth gradient across most of the domain.
                let mx = (i as f64) / (nx as f64);
                let my = (1.0 - mx * mx).sqrt();
                set_cell(&mut f, i, j, [mx, my, 0.0]);
            }
        }

        // Without cap: should flag most cells.
        let policy_no_cap = ClusterPolicy {
            indicator: IndicatorKind::Grad2 { frac: 0.10 },
            buffer_cells: 0,
            connectivity: Connectivity::Four,
            merge_distance: 1,
            min_patch_area: 1,
            max_patches: 0,
            boundary_layer: 0,
            min_efficiency: 0.0,
            max_flagged_fraction: 1.0, // disabled
        };

        let policy_with_cap = ClusterPolicy {
            max_flagged_fraction: 0.50,
            ..policy_no_cap
        };

        let r1 = compute_patch_rects_clustered_from_indicator(&f, policy_no_cap, None);
        let r2 = compute_patch_rects_clustered_from_indicator(&f, policy_with_cap, None);

        let fc1 = r1.as_ref().map(|(_, s)| s.flagged_cells).unwrap_or(0);
        let fc2 = r2.as_ref().map(|(_, s)| s.flagged_cells).unwrap_or(0);

        // With cap, fewer cells should be flagged.
        assert!(
            fc2 <= fc1,
            "coverage cap should not increase flagged cells: {} vs {}",
            fc2,
            fc1,
        );

        // The effective threshold should be raised (visible in stats).
        let t1 = r1.as_ref().map(|(_, s)| s.threshold).unwrap_or(0.0);
        let t2 = r2.as_ref().map(|(_, s)| s.threshold).unwrap_or(0.0);
        assert!(
            t2 >= t1,
            "coverage cap should raise effective threshold: {:.4} vs {:.4}",
            t2,
            t1,
        );

        let material_cells = nx * ny;
        // Internal max_allowed is scaled by eff_factor (0.70 default when
        // min_efficiency=0.0), so check against the corrected limit.
        let eff_factor = 0.70;
        let max_allowed = (0.50 * material_cells as f64 * eff_factor) as usize;
        assert!(
            fc2 <= max_allowed || fc2 == 0,
            "flagged cells ({}) should be <= efficiency-corrected cap ({})",
            fc2,
            max_allowed,
        );

        eprintln!(
            "[test] coverage cap: no_cap flagged={} thresh={:.4}, with_cap flagged={} thresh={:.4}",
            fc1, t1, fc2, t2,
        );
    }

    #[test]
    fn coverage_cap_fires_after_dilation() {
        // Key scenario: pre-dilation flagging is below 50% but post-dilation
        // coverage exceeds 50%.  This verifies the cap checks the correct
        // (post-dilation) metric.
        //
        // Sparse features on a 64×64 grid with buffer_cells=6 → dilation
        // roughly doubles the flagged area.
        let nx = 64;
        let ny = 64;
        let mut f = make_field(nx, ny);
        // Mostly uniform +z.
        for v in f.data.iter_mut() {
            *v = [0.0, 0.0, 1.0];
        }
        // Scatter many small features so pre-dilation coverage is ~30%
        // but post-dilation (buffer=6) expands to ~60%.
        for j in (4..ny - 4).step_by(8) {
            for i in (4..nx - 4).step_by(8) {
                // 3×3 patch of +x at each grid point.
                for dj in 0..3 {
                    for di in 0..3 {
                        set_cell(&mut f, i + di, j + dj, [1.0, 0.0, 0.0]);
                    }
                }
            }
        }

        // Without cap, large buffer causes most of the domain to be flagged.
        let policy_no_cap = ClusterPolicy {
            indicator: IndicatorKind::Grad2 { frac: 0.10 },
            buffer_cells: 6,
            connectivity: Connectivity::Eight,
            merge_distance: 2,
            min_patch_area: 1,
            max_patches: 0,
            boundary_layer: 0,
            min_efficiency: 0.0,
            max_flagged_fraction: 1.0, // disabled
        };

        let policy_with_cap = ClusterPolicy {
            max_flagged_fraction: 0.50,
            ..policy_no_cap
        };

        let r1 = compute_patch_rects_clustered_from_indicator(&f, policy_no_cap, None);
        let r2 = compute_patch_rects_clustered_from_indicator(&f, policy_with_cap, None);

        let fc1 = r1.as_ref().map(|(_, s)| s.flagged_cells).unwrap_or(0);
        let fc2 = r2.as_ref().map(|(_, s)| s.flagged_cells).unwrap_or(0);
        let domain = nx * ny;
        // Internal max_allowed uses eff_factor=0.70 (default for min_efficiency=0.0).
        let eff_factor = 0.70;
        let max_allowed = (0.50 * domain as f64 * eff_factor) as usize;

        // The no-cap run should have high coverage (dilation inflates it).
        assert!(
            fc1 > max_allowed,
            "no-cap post-dilation flagged ({}) should exceed efficiency-corrected cap ({}) for test to be valid",
            fc1,
            max_allowed,
        );

        // With cap, post-dilation flagged must be within the limit.
        assert!(
            fc2 <= max_allowed,
            "coverage cap should keep post-dilation flagged ({}) <= efficiency-corrected cap ({})",
            fc2,
            max_allowed,
        );

        // Threshold should be raised.
        let t1 = r1.as_ref().map(|(_, s)| s.threshold).unwrap_or(0.0);
        let t2 = r2.as_ref().map(|(_, s)| s.threshold).unwrap_or(0.0);
        assert!(
            t2 > t1,
            "cap should raise threshold: no_cap={:.4} vs cap={:.4}",
            t1,
            t2,
        );

        eprintln!(
            "[test] post-dilation cap: no_cap flagged={}/{} ({:.0}%), cap flagged={}/{} ({:.0}%), thresh {:.4} → {:.4}",
            fc1, domain, 100.0 * fc1 as f64 / domain as f64,
            fc2, domain, 100.0 * fc2 as f64 / domain as f64,
            t1, t2,
        );
    }
}