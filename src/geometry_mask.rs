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
// - MaskShape is a stored analytical shape descriptor that can be re-evaluated at any
//   cell size, enabling fine-level AMR patches to resolve curved boundaries at their
//   native resolution rather than inheriting the coarse staircase.

use crate::grid::Grid2D;

/// Boolean geometry mask for a 2D grid (length = nx*ny).
pub type Mask2D = Vec<bool>;

// ---------------------------------------------------------------------------
// MaskShape: analytical shape descriptor (re-evaluable at any resolution)
// ---------------------------------------------------------------------------

/// An analytical geometry description in **centered** coordinates (origin at grid centre).
///
/// All coordinates and dimensions are in meters.  Every variant can be evaluated at an
/// arbitrary (x, y) point via [`MaskShape::contains`], which makes it possible to build
/// a boolean mask at any resolution without inheriting a coarser grid's staircase.
///
/// CSG composition is supported via `Union`, `Intersection`, `Difference`, and `Invert`
/// variants which box their operands.
#[derive(Clone, Debug)]
pub enum MaskShape {
    /// Entire domain is material.
    Full,

    /// Disk: (x−cx)² + (y−cy)² ≤ r².
    Disk { center: (f64, f64), radius: f64 },

    /// Axis-aligned rectangle: |x−cx| ≤ hx  and  |y−cy| ≤ hy.
    Rect {
        center: (f64, f64),
        hx: f64,
        hy: f64,
    },

    /// Ellipse: ((x−cx)/a)² + ((y−cy)/b)² ≤ 1.
    Ellipse { center: (f64, f64), a: f64, b: f64 },

    /// Ring (annulus): r_inner² ≤ (x−cx)² + (y−cy)² ≤ r_outer².
    Ring {
        center: (f64, f64),
        r_inner: f64,
        r_outer: f64,
    },

    // ---- CSG combinators ----
    /// A ∪ B
    Union(Box<MaskShape>, Box<MaskShape>),
    /// A ∩ B
    Intersection(Box<MaskShape>, Box<MaskShape>),
    /// A \ B  (in A but not in B)
    Difference(Box<MaskShape>, Box<MaskShape>),
    /// ¬A
    Invert(Box<MaskShape>),
}

impl MaskShape {
    /// Evaluate the shape at a point in **centered** coordinates (meters).
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match self {
            MaskShape::Full => true,

            MaskShape::Disk {
                center: (cx, cy),
                radius,
            } => {
                let dx = x - cx;
                let dy = y - cy;
                dx * dx + dy * dy <= radius * radius
            }

            MaskShape::Rect {
                center: (cx, cy),
                hx,
                hy,
            } => (x - cx).abs() <= *hx && (y - cy).abs() <= *hy,

            MaskShape::Ellipse {
                center: (cx, cy),
                a,
                b,
            } => {
                let dx = x - cx;
                let dy = y - cy;
                (dx * dx) / (a * a) + (dy * dy) / (b * b) <= 1.0
            }

            MaskShape::Ring {
                center: (cx, cy),
                r_inner,
                r_outer,
            } => {
                let dx = x - cx;
                let dy = y - cy;
                let r2 = dx * dx + dy * dy;
                r2 >= r_inner * r_inner && r2 <= r_outer * r_outer
            }

            MaskShape::Union(a, b) => a.contains(x, y) || b.contains(x, y),
            MaskShape::Intersection(a, b) => a.contains(x, y) && b.contains(x, y),
            MaskShape::Difference(a, b) => a.contains(x, y) && !b.contains(x, y),
            MaskShape::Invert(a) => !a.contains(x, y),
        }
    }

    /// Build a boolean mask on the given grid by evaluating at each cell centre
    /// (centered coordinates, consistent with [`cell_center_xy_centered`]).
    pub fn to_mask(&self, grid: &Grid2D) -> Mask2D {
        mask_from_fn(grid, |x, y| self.contains(x, y))
    }

    // ---- Convenience constructors for CSG ----

    /// A ∪ B
    pub fn union(self, other: MaskShape) -> MaskShape {
        MaskShape::Union(Box::new(self), Box::new(other))
    }

    /// A ∩ B
    pub fn intersection(self, other: MaskShape) -> MaskShape {
        MaskShape::Intersection(Box::new(self), Box::new(other))
    }

    /// A \ B
    pub fn difference(self, other: MaskShape) -> MaskShape {
        MaskShape::Difference(Box::new(self), Box::new(other))
    }

    /// ¬A
    pub fn invert(self) -> MaskShape {
        MaskShape::Invert(Box::new(self))
    }
}

// ---------------------------------------------------------------------------
// Mask utilities (unchanged public API)
// ---------------------------------------------------------------------------

/// Return true if `mask.len()` matches `grid.n_cells()`.
#[inline]
pub fn mask_len_ok(mask: &[bool], grid: &Grid2D) -> bool {
    mask.len() == grid.n_cells()
}

/// Assert that `mask.len()` matches `grid.n_cells()`.
#[inline]
pub fn assert_mask_len(mask: &[bool], grid: &Grid2D) {
    assert_eq!(
        mask.len(),
        grid.n_cells(),
        "mask length mismatch: mask.len()={} but grid.n_cells()={} (nx={}, ny={})",
        mask.len(),
        grid.n_cells(),
        grid.nx,
        grid.ny
    );
}

/// Assert that two masks refer to the same grid size.
#[inline]
pub fn assert_same_len(a: &[bool], b: &[bool]) {
    assert_eq!(
        a.len(),
        b.len(),
        "mask length mismatch: a.len()={} vs b.len()={}",
        a.len(),
        b.len()
    );
}

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

/// Disk mask (preferred argument order): center then radius.
pub fn mask_disk_at(grid: &Grid2D, center: (f64, f64), radius: f64) -> Mask2D {
    let (cx, cy) = center;
    let r2 = radius * radius;
    mask_from_fn(grid, move |x, y| {
        let dx = x - cx;
        let dy = y - cy;
        dx * dx + dy * dy <= r2
    })
}

/// Ring mask (preferred argument order): center then inner/outer radii.
pub fn mask_ring_at(grid: &Grid2D, center: (f64, f64), r_inner: f64, r_outer: f64) -> Mask2D {
    let (cx, cy) = center;
    let ri2 = r_inner * r_inner;
    let ro2 = r_outer * r_outer;
    mask_from_fn(grid, move |x, y| {
        let dx = x - cx;
        let dy = y - cy;
        let rr2 = dx * dx + dy * dy;
        rr2 >= ri2 && rr2 <= ro2
    })
}

/// Axis-aligned rectangle (preferred argument order): center then half-widths.
pub fn mask_rect_at(grid: &Grid2D, center: (f64, f64), hx: f64, hy: f64) -> Mask2D {
    let (cx, cy) = center;
    mask_from_fn(grid, move |x, y| {
        (x - cx).abs() <= hx && (y - cy).abs() <= hy
    })
}

/// Ellipse (preferred argument order): center then semi-axes.
pub fn mask_ellipse_at(grid: &Grid2D, center: (f64, f64), a: f64, b: f64) -> Mask2D {
    let (cx, cy) = center;
    let inv_a2 = 1.0 / (a * a);
    let inv_b2 = 1.0 / (b * b);
    mask_from_fn(grid, move |x, y| {
        let dx = x - cx;
        let dy = y - cy;
        dx * dx * inv_a2 + dy * dy * inv_b2 <= 1.0
    })
}

/// Disk mask: (x-cx)^2 + (y-cy)^2 <= radius^2.
pub fn mask_disk(grid: &Grid2D, radius: f64, center: (f64, f64)) -> Mask2D {
    mask_disk_at(grid, center, radius)
}

/// Ring mask: inner <= r <= outer.
pub fn mask_ring(grid: &Grid2D, r_outer: f64, r_inner: f64, center: (f64, f64)) -> Mask2D {
    mask_ring_at(grid, center, r_inner, r_outer)
}

/// Axis-aligned rectangle: |x-cx|<=hx and |y-cy|<=hy.
pub fn mask_rect(grid: &Grid2D, hx: f64, hy: f64, center: (f64, f64)) -> Mask2D {
    mask_rect_at(grid, center, hx, hy)
}

/// Ellipse: (x/a)^2 + (y/b)^2 <= 1 (with optional center).
pub fn mask_ellipse(grid: &Grid2D, a: f64, b: f64, center: (f64, f64)) -> Mask2D {
    mask_ellipse_at(grid, center, a, b)
}

/// Union (A ∪ B).
pub fn mask_union(a: &[bool], b: &[bool]) -> Mask2D {
    assert_same_len(a, b);
    a.iter().zip(b.iter()).map(|(&aa, &bb)| aa || bb).collect()
}

/// Intersection (A ∩ B).
pub fn mask_intersection(a: &[bool], b: &[bool]) -> Mask2D {
    assert_same_len(a, b);
    a.iter().zip(b.iter()).map(|(&aa, &bb)| aa && bb).collect()
}

/// Difference (A \ B).
pub fn mask_difference(a: &[bool], b: &[bool]) -> Mask2D {
    assert_same_len(a, b);
    a.iter().zip(b.iter()).map(|(&aa, &bb)| aa && !bb).collect()
}

/// XOR (A ⊕ B).
pub fn mask_xor(a: &[bool], b: &[bool]) -> Mask2D {
    assert_same_len(a, b);
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

/// Bounding box of the mask using grid dimensions.
pub fn mask_bbox_grid(mask: &[bool], grid: &Grid2D) -> Option<(usize, usize, usize, usize)> {
    mask_bbox(mask, grid.nx, grid.ny)
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_shape_disk_matches_mask_disk_at() {
        let grid = Grid2D::new(32, 32, 1e-9, 1e-9, 1e-9);
        let center = (0.0, 0.0);
        let radius = 10e-9;

        let from_fn = mask_disk_at(&grid, center, radius);
        let from_shape = MaskShape::Disk { center, radius }.to_mask(&grid);

        assert_eq!(from_fn, from_shape);
    }

    #[test]
    fn mask_shape_rect_matches_mask_rect_at() {
        let grid = Grid2D::new(16, 16, 2e-9, 2e-9, 1e-9);
        let center = (1e-9, -1e-9);
        let hx = 5e-9;
        let hy = 3e-9;

        let from_fn = mask_rect_at(&grid, center, hx, hy);
        let from_shape = MaskShape::Rect { center, hx, hy }.to_mask(&grid);

        assert_eq!(from_fn, from_shape);
    }

    #[test]
    fn mask_shape_csg_difference() {
        let grid = Grid2D::new(64, 64, 1e-9, 1e-9, 1e-9);
        let outer = MaskShape::Disk {
            center: (0.0, 0.0),
            radius: 20e-9,
        };
        let inner = MaskShape::Disk {
            center: (0.0, 0.0),
            radius: 10e-9,
        };
        let ring_csg = outer.difference(inner);
        let ring_direct = MaskShape::Ring {
            center: (0.0, 0.0),
            r_inner: 10e-9,
            r_outer: 20e-9,
        };

        let m_csg = ring_csg.to_mask(&grid);
        let m_direct = ring_direct.to_mask(&grid);

        assert_eq!(m_csg, m_direct);
    }

    #[test]
    fn mask_shape_csg_union() {
        let grid = Grid2D::new(64, 64, 1e-9, 1e-9, 1e-9);
        let d1 = MaskShape::Disk {
            center: (-10e-9, 0.0),
            radius: 8e-9,
        };
        let d2 = MaskShape::Disk {
            center: (10e-9, 0.0),
            radius: 8e-9,
        };
        let combined = d1.union(d2);
        let mask = combined.to_mask(&grid);

        let m1 = MaskShape::Disk {
            center: (-10e-9, 0.0),
            radius: 8e-9,
        }
        .to_mask(&grid);
        let m2 = MaskShape::Disk {
            center: (10e-9, 0.0),
            radius: 8e-9,
        }
        .to_mask(&grid);
        assert!(mask_count(&mask) >= mask_count(&m1));
        assert!(mask_count(&mask) >= mask_count(&m2));
    }
}
