// src/amr/clustering.rs
//
// Stage-2B: Multi-patch clustering for block-structured AMR.
//
// Goal:
// - Given a coarse-grid refinement indicator, identify *multiple disjoint* regions
//   that should be refined and return a set of disjoint Rect2i patches.
//
// Approach (v0, practical):
// 1) Compute indicator per cell and threshold at frac * max(indicator).
// 2) Flag cells above threshold.
// 3) Connected-components labelling on the flagged mask -> per-component bbox.
// 4) Dilate each bbox by buffer_cells (clamped to domain).
// 5) Merge overlapping / near bboxes (merge_distance).
// 6) Optionally down-select to max_patches via greedy merges.
//
// This intentionally stays simple (no Berger–Rigoutsos yet). It's designed to be
// a clean stepping stone to "real" BR clustering later.

use std::collections::VecDeque;

use crate::amr::indicator::indicator_grad2_forward;
use crate::amr::rect::Rect2i;
use crate::vector_field::VectorField2D;

#[derive(Clone, Copy, Debug)]
pub enum Connectivity {
    Four,
    Eight,
}

#[derive(Clone, Copy, Debug)]
pub struct ClusterPolicy {
    /// Threshold fraction: flag cells where indicator >= frac * max(indicator).
    pub indicator_frac: f64,
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
}

impl Default for ClusterPolicy {
    fn default() -> Self {
        // Defaults tuned for localized textures (e.g. bubbles/vortices/skyrmions):
        // - slightly lower threshold to avoid tiny initial patches
        // - larger buffer so patches cover the full transition region
        // - 8-neighbour connectivity to reduce fragmentation of ring-like features
        // - modest merge distance to glue nearby fragments into one patch per feature
        Self {
            indicator_frac: 0.25,
            buffer_cells: 6,
            connectivity: Connectivity::Eight,
            merge_distance: 4,
            min_patch_area: 16, // 4x4
            max_patches: 8,
        }
    }
}

impl ClusterPolicy {
    /// Conservative settings (smaller buffer, 4-neighbour connectivity).
    /// Useful if you want to minimize refined area and your indicator is not fragmented.
    pub fn conservative() -> Self {
        Self {
            indicator_frac: 0.35,
            buffer_cells: 2,
            connectivity: Connectivity::Four,
            merge_distance: 2,
            min_patch_area: 16,
            max_patches: 8,
        }
    }

    /// Settings tuned for localized textures (bubbles/vortices/skyrmions) as validated
    /// by `amr_two_bubbles_relax`.
    pub fn tuned_local_features() -> Self {
        Self::default()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ClusterStats {
    pub max_indicator: f64,
    pub threshold: f64,
    pub flagged_cells: usize,
    pub components: usize,
    pub patches_before_merge: usize,
    pub patches_after_merge: usize,
}

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

/// Compute clustered patch rectangles from an indicator on the *coarse* grid.
pub fn compute_patch_rects_clustered_from_indicator(
    coarse: &VectorField2D,
    policy: ClusterPolicy,
) -> Option<(Vec<Rect2i>, ClusterStats)> {
    let nx = coarse.grid.nx;
    let ny = coarse.grid.ny;
    if nx == 0 || ny == 0 {
        return None;
    }

    // 1) max indicator
    let mut max_ind = 0.0_f64;
    for j in 0..ny {
        for i in 0..nx {
            let ind = indicator_grad2_forward(coarse, i, j);
            if ind > max_ind {
                max_ind = ind;
            }
        }
    }
    if max_ind <= 0.0 {
        return None;
    }

    let frac = policy.indicator_frac.max(0.0).min(1.0);
    let thresh = frac * max_ind;

    // 2) flagged mask
    let mut flagged = vec![false; nx * ny];
    let mut flagged_cells = 0usize;
    for j in 0..ny {
        for i in 0..nx {
            let ind = indicator_grad2_forward(coarse, i, j);
            if ind >= thresh {
                flagged[idx(i, j, nx)] = true;
                flagged_cells += 1;
            }
        }
    }

    if flagged_cells == 0 {
        return None;
    }

    // 3) connected components -> bboxes
    let mut visited = vec![false; nx * ny];
    let mut rects: Vec<Rect2i> = Vec::new();

    let neigh = |ii: isize, jj: isize, conn: Connectivity| -> [(isize, isize); 8] {
        match conn {
            Connectivity::Four => [
                (ii - 1, jj),
                (ii + 1, jj),
                (ii, jj - 1),
                (ii, jj + 1),
                (ii, jj), // unused slots
                (ii, jj),
                (ii, jj),
                (ii, jj),
            ],
            Connectivity::Eight => [
                (ii - 1, jj),
                (ii + 1, jj),
                (ii, jj - 1),
                (ii, jj + 1),
                (ii - 1, jj - 1),
                (ii - 1, jj + 1),
                (ii + 1, jj - 1),
                (ii + 1, jj + 1),
            ],
        }
    };

    for j0 in 0..ny {
        for i0 in 0..nx {
            let id0 = idx(i0, j0, nx);
            if !flagged[id0] || visited[id0] {
                continue;
            }

            // BFS/queue
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

                let ns = neigh(ii, jj, policy.connectivity);
                for (ni, nj) in ns {
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
            let rect = raw.dilate_clamped(policy.buffer_cells, nx, ny);
            rects.push(rect);
        }
    }

    let components = rects.len();
    let patches_before_merge = rects.len();

    // 4) merge overlapping / near rectangles
    let mut merged = true;
    while merged {
        merged = false;

        'outer: for a in 0..rects.len() {
            for b in (a + 1)..rects.len() {
                if rects_overlap_or_near(rects[a], rects[b], policy.merge_distance, nx, ny)
                    || rects_overlap_or_near(rects[b], rects[a], policy.merge_distance, nx, ny)
                {
                    let u = rect_union(rects[a], rects[b]);
                    rects[a] = u;
                    rects.swap_remove(b);
                    merged = true;
                    break 'outer;
                }
            }
        }
    }

    // 5) filter tiny patches
    if policy.min_patch_area > 0 {
        rects.retain(|&r| rect_area(r) >= policy.min_patch_area);
    }

    // 6) enforce max_patches by greedily merging the "cheapest" pair
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
                let cost = rect_area(u).saturating_sub(rect_area(rects[i]) + rect_area(rects[j]));
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

    // 7) stable ordering for deterministic logs
    rects.sort_by_key(|r| (r.i0, r.j0, r.nx, r.ny));

    let stats = ClusterStats {
        max_indicator: max_ind,
        threshold: thresh,
        flagged_cells,
        components,
        patches_before_merge,
        patches_after_merge: rects.len(),
    };

    Some((rects, stats))
}
