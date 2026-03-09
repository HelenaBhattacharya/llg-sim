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

/// Per-cell volume (area) fill fraction ∈ [0.0, 1.0] for a 2D grid.
///
/// A value of 1.0 means the cell is fully inside the material;
/// 0.0 means fully outside (vacuum).  Intermediate values arise for
/// cells that straddle the material/vacuum boundary.
///
/// MuMax3 calls this "EdgeSmooth": `EdgeSmooth = n` means n² sub-samples
/// per cell (they use n³ in 3D; we use n² for thin-film 2D).
pub type FillFraction2D = Vec<f64>;

/// Read the sub-sampling level from `LLG_EDGE_SMOOTH` (default: 8).
///
/// `n = 0` disables smoothing (pure staircase, equivalent to boolean mask).
/// `n = 8` gives 64 sub-samples per cell — matches MuMax3 default.
/// Higher values are diminishing returns; 8 is sufficient for most purposes.
pub fn edge_smooth_n() -> usize {
    std::env::var("LLG_EDGE_SMOOTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
}

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

    /// Anti-dot / multi-hole: material everywhere EXCEPT inside any of the
    /// listed circular holes.  O(n_holes) per point — much cheaper than
    /// building a deeply nested CSG `Difference` chain.
    MultiHole {
        holes: Vec<(f64, f64)>,
        radius: f64,
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

            MaskShape::MultiHole { holes, radius } => {
                let r2 = radius * radius;
                !holes.iter().any(|&(cx, cy)| {
                    let ddx = x - cx;
                    let ddy = y - cy;
                    ddx * ddx + ddy * ddy <= r2
                })
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

    // ---- EdgeSmooth: fill-fraction computation (MuMax3 style) ----

    /// Compute the fill fraction of a single cell by sub-sampling.
    ///
    /// The cell is centred at `(cx, cy)` with size `(dx, dy)`.
    /// `n` sub-samples per dimension → n² evaluation points, uniformly
    /// distributed across the cell area.
    ///
    /// Returns a value in `[0.0, 1.0]`.
    ///
    /// If `n == 0`, returns 1.0 if the cell centre is inside, 0.0 otherwise
    /// (staircase approximation).
    pub fn fill_fraction(&self, cx: f64, cy: f64, dx: f64, dy: f64, n: usize) -> f64 {
        if n == 0 {
            return if self.contains(cx, cy) { 1.0 } else { 0.0 };
        }

        let mut count = 0u32;
        let total = (n * n) as u32;

        // Sub-sample positions: evenly spaced within the cell,
        // offset by half a sub-cell width from the cell edge.
        // For n=8: sub-positions at 1/16, 3/16, 5/16, ..., 15/16 of dx.
        let inv_n = 1.0 / n as f64;
        let x0 = cx - dx * 0.5;
        let y0 = cy - dy * 0.5;

        for sj in 0..n {
            let sy = y0 + (sj as f64 + 0.5) * inv_n * dy;
            for si in 0..n {
                let sx = x0 + (si as f64 + 0.5) * inv_n * dx;
                if self.contains(sx, sy) {
                    count += 1;
                }
            }
        }

        count as f64 / total as f64
    }

    /// Build a fill-fraction array for the entire grid.
    ///
    /// This is the 2D equivalent of MuMax3's `EdgeSmooth = n`.
    /// Each cell gets a value in `[0.0, 1.0]` representing the fraction
    /// of the cell that lies inside the material.
    ///
    /// **Optimisation:** cells whose 4 corners are ALL inside or ALL outside
    /// skip sub-sampling entirely (they get 1.0 or 0.0 directly).
    /// Only cells that straddle the boundary are sub-sampled at full `n²`
    /// resolution.  For a typical antidot with ~5% boundary cells, this
    /// makes the computation essentially free.
    pub fn to_fill_fractions(&self, grid: &Grid2D, n: usize) -> FillFraction2D {
        let nx = grid.nx;
        let ny = grid.ny;
        let dx = grid.dx;
        let dy = grid.dy;
        let mut ff = vec![0.0f64; nx * ny];

        let half_nx = nx as f64 * 0.5;
        let half_ny = ny as f64 * 0.5;

        for j in 0..ny {
            for i in 0..nx {
                let cx = (i as f64 + 0.5 - half_nx) * dx;
                let cy = (j as f64 + 0.5 - half_ny) * dy;

                if n == 0 {
                    // Pure staircase
                    ff[j * nx + i] = if self.contains(cx, cy) { 1.0 } else { 0.0 };
                    continue;
                }

                // Quick check: test the 4 cell corners.
                // If all are the same (all in or all out), skip sub-sampling.
                let c00 = self.contains(cx - dx * 0.5, cy - dy * 0.5);
                let c10 = self.contains(cx + dx * 0.5, cy - dy * 0.5);
                let c01 = self.contains(cx - dx * 0.5, cy + dy * 0.5);
                let c11 = self.contains(cx + dx * 0.5, cy + dy * 0.5);

                if c00 == c10 && c10 == c01 && c01 == c11 {
                    // All corners agree → cell is fully inside or fully outside.
                    ff[j * nx + i] = if c00 { 1.0 } else { 0.0 };
                } else {
                    // Boundary cell — do full sub-sampling.
                    ff[j * nx + i] = self.fill_fraction(cx, cy, dx, dy, n);
                }
            }
        }

        ff
    }

    /// Build both a boolean mask AND fill fractions in one pass.
    ///
    /// The boolean mask uses `fill > 0.0` as the threshold (any material
    /// in the cell counts as "material" for exchange / masking purposes).
    /// This matches MuMax3 behaviour where EdgeSmooth affects Ms but
    /// exchange neighbours still use the boolean mask.
    pub fn to_mask_and_fill(&self, grid: &Grid2D, n: usize) -> (Mask2D, FillFraction2D) {
        let ff = self.to_fill_fractions(grid, n);
        let mask = ff.iter().map(|&f| f > 0.0).collect();
        (mask, ff)
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
// Anti-dot array helpers
// ---------------------------------------------------------------------------

/// Generate centres of a hexagonal hole array within a domain.
///
/// Domain is centered at (0, 0), extending from (−lx/2, −ly/2) to (+lx/2, +ly/2).
/// `pitch` is the centre-to-centre distance between adjacent holes.
/// Returns centres in **centered** coordinates (meters), consistent with `MaskShape`.
///
/// Only holes whose centres lie within the domain (with a half-pitch inset from
/// each edge) are included — this ensures the outermost ring of holes is fully
/// inside the sample, avoiding partially-clipped holes at domain boundaries.
pub fn hex_hole_centres(lx: f64, ly: f64, pitch: f64) -> Vec<(f64, f64)> {
    hex_hole_centres_inset(lx, ly, pitch, pitch * 0.5)
}

/// Like [`hex_hole_centres`] but with explicit inset from domain edges.
pub fn hex_hole_centres_inset(lx: f64, ly: f64, pitch: f64, inset: f64) -> Vec<(f64, f64)> {
    let mut centres = Vec::new();
    let row_spacing = pitch * (3.0_f64).sqrt() * 0.5;
    let half_x = lx * 0.5 - inset;
    let half_y = ly * 0.5 - inset;

    let n_rows = (half_y / row_spacing).ceil() as i32 + 1;
    let n_cols = (half_x / pitch).ceil() as i32 + 1;

    for row in -n_rows..=n_rows {
        let y = row as f64 * row_spacing;
        if y.abs() > half_y { continue; }
        let x_offset = if row.abs() % 2 == 1 { pitch * 0.5 } else { 0.0 };
        for col in -n_cols..=n_cols {
            let x = col as f64 * pitch + x_offset;
            if x.abs() <= half_x {
                centres.push((x, y));
            }
        }
    }
    centres
}

// ---------------------------------------------------------------------------
// Fill-fraction utilities
// ---------------------------------------------------------------------------

/// Count boundary cells (cells with 0 < fill_fraction < 1).
pub fn fill_fraction_boundary_count(ff: &[f64]) -> usize {
    ff.iter().filter(|&&f| f > 0.0 && f < 1.0).count()
}

/// Apply fill fractions to a magnetisation array in-place.
///
/// For each cell: `m[k] *= fill_fraction[k]`.
/// Interior cells (f=1) are unchanged.  Vacuum cells (f=0) become zero.
/// Boundary cells get M_eff = f * M_sat * m̂, smoothing the surface charges.
///
/// This is the core of the EdgeSmooth correction: the demag solver then
/// sees a smooth ∇·M at material boundaries instead of a sharp step.
pub fn apply_fill_fractions(m: &mut [[f64; 3]], ff: &[f64]) {
    assert_eq!(m.len(), ff.len(), "m and fill_fraction length mismatch");
    for (v, &f) in m.iter_mut().zip(ff.iter()) {
        v[0] *= f;
        v[1] *= f;
        v[2] *= f;
    }
}

/// Build fill fractions using the global default from `LLG_EDGE_SMOOTH`.
pub fn fill_fractions_default(shape: &MaskShape, grid: &Grid2D) -> FillFraction2D {
    shape.to_fill_fractions(grid, edge_smooth_n())
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

    #[test]
    fn multihole_matches_csg_difference_chain() {
        let grid = Grid2D::new(64, 64, 1e-9, 1e-9, 1e-9);
        let holes = vec![(5e-9, 5e-9), (-5e-9, -5e-9), (10e-9, -10e-9)];
        let r = 3e-9;

        // Build via MultiHole
        let mh = MaskShape::MultiHole { holes: holes.clone(), radius: r };
        let mask_mh = mh.to_mask(&grid);

        // Build via chained CSG Difference
        let mut shape = MaskShape::Full;
        for &(cx, cy) in &holes {
            shape = shape.difference(MaskShape::Disk { center: (cx, cy), radius: r });
        }
        let mask_csg = shape.to_mask(&grid);

        assert_eq!(mask_mh, mask_csg);
    }

    #[test]
    fn hex_hole_centres_reasonable_count() {
        let centres = hex_hole_centres(2e-6, 2e-6, 200e-9);
        // 2 μm domain, 200 nm pitch → ~10×10 = ~100 holes
        assert!(centres.len() > 50, "expected ~100 holes, got {}", centres.len());
        assert!(centres.len() < 200, "expected ~100 holes, got {}", centres.len());
        // All centres should be within domain
        for &(x, y) in &centres {
            assert!(x.abs() <= 1e-6, "hole at x={:.0e} outside domain", x);
            assert!(y.abs() <= 1e-6, "hole at y={:.0e} outside domain", y);
        }
    }

    // ---- Fill-fraction / EdgeSmooth tests ----

    #[test]
    fn fill_fraction_interior_cell_is_one() {
        let shape = MaskShape::Disk { center: (0.0, 0.0), radius: 100e-9 };
        // Cell well inside the disk
        let f = shape.fill_fraction(0.0, 0.0, 2e-9, 2e-9, 8);
        assert!((f - 1.0).abs() < 1e-10, "interior cell should have f=1, got {f}");
    }

    #[test]
    fn fill_fraction_exterior_cell_is_zero() {
        let shape = MaskShape::Disk { center: (0.0, 0.0), radius: 10e-9 };
        // Cell well outside the disk
        let f = shape.fill_fraction(50e-9, 50e-9, 2e-9, 2e-9, 8);
        assert!(f.abs() < 1e-10, "exterior cell should have f=0, got {f}");
    }

    #[test]
    fn fill_fraction_boundary_cell_is_partial() {
        let shape = MaskShape::Disk { center: (0.0, 0.0), radius: 10e-9 };
        // Cell straddling the boundary (cell centre at r=10nm, edge of disk)
        let f = shape.fill_fraction(10e-9, 0.0, 2e-9, 2e-9, 8);
        assert!(f > 0.0, "boundary cell should have f>0, got {f}");
        assert!(f < 1.0, "boundary cell should have f<1, got {f}");
    }

    #[test]
    fn fill_fraction_n0_is_staircase() {
        let grid = Grid2D::new(32, 32, 1e-9, 1e-9, 1e-9);
        let shape = MaskShape::Disk { center: (0.0, 0.0), radius: 10e-9 };

        let ff_staircase = shape.to_fill_fractions(&grid, 0);
        let mask_bool = shape.to_mask(&grid);

        // n=0 should give exactly 0.0 or 1.0, matching the boolean mask
        for (k, (&f, &b)) in ff_staircase.iter().zip(mask_bool.iter()).enumerate() {
            let expected = if b { 1.0 } else { 0.0 };
            assert!((f - expected).abs() < 1e-10,
                "cell {k}: fill_fraction={f} but mask={b}");
        }
    }

    #[test]
    fn fill_fraction_has_boundary_cells() {
        let grid = Grid2D::new(64, 64, 1e-9, 1e-9, 1e-9);
        let shape = MaskShape::Disk { center: (0.0, 0.0), radius: 20e-9 };

        let ff = shape.to_fill_fractions(&grid, 8);
        let n_boundary = fill_fraction_boundary_count(&ff);

        // A circle of radius 20nm at 1nm cells should have ~60-130 boundary cells
        assert!(n_boundary > 20,
            "expected boundary cells for disk, got {n_boundary}");
        assert!(n_boundary < 300,
            "too many boundary cells: {n_boundary}");
    }

    #[test]
    fn fill_fraction_multihole_smooths_edges() {
        let grid = Grid2D::new(128, 128, 4e-9, 4e-9, 3e-9);
        let shape = MaskShape::MultiHole {
            holes: vec![(0.0, 0.0)],
            radius: 100e-9,
        };

        let ff = shape.to_fill_fractions(&grid, 8);
        let n_boundary = fill_fraction_boundary_count(&ff);

        // Should have partial cells around the hole boundary
        assert!(n_boundary > 10,
            "expected partial cells around hole, got {n_boundary}");
    }

    #[test]
    fn to_mask_and_fill_consistent() {
        let grid = Grid2D::new(32, 32, 2e-9, 2e-9, 1e-9);
        let shape = MaskShape::Disk { center: (0.0, 0.0), radius: 15e-9 };

        let (mask, ff) = shape.to_mask_and_fill(&grid, 8);

        // mask should be true wherever fill > 0
        for (k, (&m, &f)) in mask.iter().zip(ff.iter()).enumerate() {
            if f > 0.0 {
                assert!(m, "cell {k}: f={f} > 0 but mask=false");
            }
            if !m {
                assert!(f == 0.0, "cell {k}: mask=false but f={f}");
            }
        }
    }

    #[test]
    fn apply_fill_fractions_scales_m() {
        let mut m = vec![[1.0, 0.0, 0.0]; 4];
        let ff = vec![1.0, 0.5, 0.25, 0.0];
        apply_fill_fractions(&mut m, &ff);

        assert!((m[0][0] - 1.0).abs() < 1e-10);
        assert!((m[1][0] - 0.5).abs() < 1e-10);
        assert!((m[2][0] - 0.25).abs() < 1e-10);
        assert!((m[3][0] - 0.0).abs() < 1e-10);
    }
}