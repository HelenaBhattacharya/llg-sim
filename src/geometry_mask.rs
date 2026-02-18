// src/geometry_mask.rs
//
// Geometry / masking utilities for 2D thin-film problems.
//
// Design goals:
// - Match MuMax-style workflows: define a geometry (mask), then set Ms=0 (here: m=0) outside.
// - Provide composable CSG operations (union/intersect/difference).
// - Use a *centered* coordinate system (0,0) at the grid center, in meters.
//
// Notes:
// - The mask is currently a boolean per cell (true = magnetic material, false = vacuum).
// - Downstream code can treat "vacuum" by setting m=(0,0,0) in those cells.

use crate::grid::Grid2D;

/// Boolean geometry mask for a 2D grid (length = nx*ny).
pub type Mask2D = Vec<bool>;

#[inline]
fn idx(i: usize, j: usize, nx: usize) -> usize {
    j * nx + i
}

/// Cell-center coordinates, centered at the grid center, in meters.
///
/// For i∈[0,nx), x = (i+0.5 - nx/2)*dx.
#[inline]
pub fn cell_center_xy_centered(grid: &Grid2D, i: usize, j: usize) -> (f64, f64) {
    let cx = (grid.nx as f64) * 0.5;
    let cy = (grid.ny as f64) * 0.5;
    let x = (i as f64 + 0.5 - cx) * grid.dx;
    let y = (j as f64 + 0.5 - cy) * grid.dy;
    (x, y)
}

/// Build a mask from a predicate f(x,y)->bool, where x,y are centered cell centers.
pub fn mask_from_fn<F>(grid: &Grid2D, f: F) -> Mask2D
where
    F: Fn(f64, f64) -> bool,
{
    let mut mask = vec![false; grid.n_cells()];
    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let (x, y) = cell_center_xy_centered(grid, i, j);
            mask[idx(i, j, grid.nx)] = f(x, y);
        }
    }
    mask
}

/// Full (all true) mask.
pub fn mask_full(grid: &Grid2D) -> Mask2D {
    vec![true; grid.n_cells()]
}

/// Disk mask: (x-cx)^2 + (y-cy)^2 <= radius^2.
pub fn mask_disk(grid: &Grid2D, radius: f64, center: (f64, f64)) -> Mask2D {
    let (cx, cy) = center;
    let r2 = radius * radius;
    mask_from_fn(grid, move |x, y| {
        let dx = x - cx;
        let dy = y - cy;
        dx * dx + dy * dy <= r2
    })
}

/// Ring mask: inner <= r <= outer.
pub fn mask_ring(grid: &Grid2D, r_outer: f64, r_inner: f64, center: (f64, f64)) -> Mask2D {
    let (cx, cy) = center;
    let ro2 = r_outer * r_outer;
    let ri2 = r_inner * r_inner;
    mask_from_fn(grid, move |x, y| {
        let dx = x - cx;
        let dy = y - cy;
        let rr2 = dx * dx + dy * dy;
        rr2 <= ro2 && rr2 >= ri2
    })
}

/// Axis-aligned rectangle: |x-cx|<=hx and |y-cy|<=hy.
pub fn mask_rect(grid: &Grid2D, hx: f64, hy: f64, center: (f64, f64)) -> Mask2D {
    let (cx, cy) = center;
    mask_from_fn(grid, move |x, y| {
        (x - cx).abs() <= hx && (y - cy).abs() <= hy
    })
}

/// Ellipse: (x/a)^2 + (y/b)^2 <= 1 (with optional center).
pub fn mask_ellipse(grid: &Grid2D, a: f64, b: f64, center: (f64, f64)) -> Mask2D {
    let (cx, cy) = center;
    let inv_a2 = 1.0 / (a * a);
    let inv_b2 = 1.0 / (b * b);
    mask_from_fn(grid, move |x, y| {
        let dx = x - cx;
        let dy = y - cy;
        dx * dx * inv_a2 + dy * dy * inv_b2 <= 1.0
    })
}

/// Union (A ∪ B).
pub fn mask_union(a: &[bool], b: &[bool]) -> Mask2D {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&aa, &bb)| aa || bb).collect()
}

/// Intersection (A ∩ B).
pub fn mask_intersection(a: &[bool], b: &[bool]) -> Mask2D {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&aa, &bb)| aa && bb).collect()
}

/// Difference (A \ B).
pub fn mask_difference(a: &[bool], b: &[bool]) -> Mask2D {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&aa, &bb)| aa && !bb).collect()
}

/// XOR (A ⊕ B).
pub fn mask_xor(a: &[bool], b: &[bool]) -> Mask2D {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&aa, &bb)| aa ^ bb).collect()
}

/// Invert (~A).
pub fn mask_invert(a: &[bool]) -> Mask2D {
    a.iter().map(|&aa| !aa).collect()
}

/// Count "true" cells (useful for quick sanity checks).
pub fn mask_count(mask: &[bool]) -> usize {
    mask.iter().filter(|&&v| v).count()
}

/// Bounding box of the mask in (i_min, i_max, j_min, j_max) inclusive indices.
pub fn mask_bbox(mask: &[bool], nx: usize, ny: usize) -> Option<(usize, usize, usize, usize)> {
    assert_eq!(mask.len(), nx * ny);
    let mut i_min = nx;
    let mut i_max = 0usize;
    let mut j_min = ny;
    let mut j_max = 0usize;
    let mut any = false;

    for j in 0..ny {
        for i in 0..nx {
            if mask[idx(i, j, nx)] {
                any = true;
                i_min = i_min.min(i);
                i_max = i_max.max(i);
                j_min = j_min.min(j);
                j_max = j_max.max(j);
            }
        }
    }

    if any {
        Some((i_min, i_max, j_min, j_max))
    } else {
        None
    }
}
