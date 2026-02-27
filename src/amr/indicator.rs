// src/amr/indicator.rs
//
// Coarse-grid refinement indicators + simple patch selection helpers.
//
// Stage 2A/2B intent:
// - Provide a cheap, robust per-cell refinement indicator on the *coarse* grid.
// - Support both unmasked domains and geometry-masked (material/vacuum) domains.
//
// The indicator implemented here is the squared forward-difference gradient magnitude:
//   |m(i+1,j) - m(i,j)|^2 + |m(i,j+1) - m(i,j)|^2
//
// Milestone 1 (true automatic adaptivity) additionally needs an *absolute* refinement
// criterion. We provide a neighbour misalignment-angle indicator in radians:
//   theta(i,j) = max( acos(clamp(m·m_x)), acos(clamp(m·m_y)) )
//
// This is scale-aware: for smooth textures, theta ~ dx |∇m|, so refinement naturally
// stops once features are resolved at the target maximum angle.
//
// For masked geometries we use a *free-boundary* interpretation for the indicator:
// - If the center cell is vacuum => indicator = 0.
// - If a forward neighbour is vacuum => treat neighbour value as equal to center,
//   so that the indicator does not spike at material–vacuum interfaces.
//
// This makes patch placement focus on *physical textures* (walls/vortices/cores),
// not on the artificial jump from material to pinned vacuum.

use crate::amr::rect::Rect2i;
use crate::geometry_mask::assert_mask_len;
use crate::vector_field::VectorField2D;

#[derive(Clone, Copy, Debug)]
pub struct IndicatorStats {
    pub max: f64,
    pub threshold: f64,
}

#[inline]
fn debug_assert_mask_len(_mask: &[bool], _field: &VectorField2D) {
    // Mask length checks are useful while developing, but too expensive to do per-cell
    // in release-mode indicator loops.
    #[cfg(debug_assertions)]
    assert_mask_len(_mask, &_field.grid);
}

#[inline]
fn clamp_pm1(x: f64) -> f64 {
    if x < -1.0 {
        -1.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// Angle between (approximately) unit vectors `a` and `b` in radians.
///
/// Assumes `a` and `b` are close to unit length (true for normalised magnetisation).
#[inline]
fn angle_between_unit(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    clamp_pm1(dot).acos()
}

/// Simple coarse-grid indicator: squared forward-difference gradient magnitude.
///
/// (Cheap + robust; good baseline for AMR patch tracking.)
#[inline]
pub fn indicator_grad2_forward(field: &VectorField2D, i: usize, j: usize) -> f64 {
    let nx = field.grid.nx;
    let ny = field.grid.ny;

    let idx = field.idx(i, j);
    let v = field.data[idx];

    let ip = if i + 1 < nx { i + 1 } else { i };
    let jp = if j + 1 < ny { j + 1 } else { j };

    let vx = field.data[field.idx(ip, j)];
    let vy = field.data[field.idx(i, jp)];

    let dx0 = vx[0] - v[0];
    let dx1 = vx[1] - v[1];
    let dx2 = vx[2] - v[2];

    let dy0 = vy[0] - v[0];
    let dy1 = vy[1] - v[1];
    let dy2 = vy[2] - v[2];

    (dx0 * dx0 + dx1 * dx1 + dx2 * dx2) + (dy0 * dy0 + dy1 * dy1 + dy2 * dy2)
}

/// Mask-aware coarse-grid indicator: squared forward-difference gradient magnitude.
///
/// Semantics when `geom_mask` is provided:
/// - If the center cell is vacuum => returns 0.
/// - If a forward neighbour is vacuum => treat neighbour value as center (free boundary),
///   so that contribution is 0.
#[inline]
pub fn indicator_grad2_forward_geom(
    field: &VectorField2D,
    i: usize,
    j: usize,
    geom_mask: Option<&[bool]>,
) -> f64 {
    // Fast early-out for vacuum cells.
    let idc = field.idx(i, j);
    if let Some(msk) = geom_mask {
        debug_assert_mask_len(msk, field);
        if !msk[idc] {
            return 0.0;
        }
    }

    let nx = field.grid.nx;
    let ny = field.grid.ny;

    let v = field.data[idc];

    // Forward neighbours (domain edge handled by clamping to self).
    let ip = if i + 1 < nx { i + 1 } else { i };
    let jp = if j + 1 < ny { j + 1 } else { j };

    // If neighbour is vacuum, treat as free boundary: v_nb := v_center.
    let vx = if ip == i {
        v
    } else {
        let idn = field.idx(ip, j);
        if let Some(msk) = geom_mask {
            if !msk[idn] { v } else { field.data[idn] }
        } else {
            field.data[idn]
        }
    };

    let vy = if jp == j {
        v
    } else {
        let idn = field.idx(i, jp);
        if let Some(msk) = geom_mask {
            if !msk[idn] { v } else { field.data[idn] }
        } else {
            field.data[idn]
        }
    };

    let dx0 = vx[0] - v[0];
    let dx1 = vx[1] - v[1];
    let dx2 = vx[2] - v[2];

    let dy0 = vy[0] - v[0];
    let dy1 = vy[1] - v[1];
    let dy2 = vy[2] - v[2];

    (dx0 * dx0 + dx1 * dx1 + dx2 * dx2) + (dy0 * dy0 + dy1 * dy1 + dy2 * dy2)
}

/// Absolute coarse-grid indicator: maximum neighbour misalignment angle (radians).
///
/// Returns max(theta_x, theta_y) where:
///   theta_x = acos(clamp(m(i,j) · m(i+1,j)))
///   theta_y = acos(clamp(m(i,j) · m(i,j+1)))
///
/// Domain edges are handled by clamping neighbours to self (theta=0 at boundary).
#[inline]
pub fn indicator_angle_max_forward(field: &VectorField2D, i: usize, j: usize) -> f64 {
    indicator_angle_max_forward_geom(field, i, j, None)
}

/// Geometry-mask-aware version of `indicator_angle_max_forward`.
///
/// Semantics when `geom_mask` is provided:
/// - If the center cell is vacuum => returns 0.
/// - If a forward neighbour is vacuum => treat neighbour value as center (free boundary),
///   so that contribution is 0.
#[inline]
pub fn indicator_angle_max_forward_geom(
    field: &VectorField2D,
    i: usize,
    j: usize,
    geom_mask: Option<&[bool]>,
) -> f64 {
    // Fast early-out for vacuum cells.
    let idc = field.idx(i, j);
    if let Some(msk) = geom_mask {
        debug_assert_mask_len(msk, field);
        if !msk[idc] {
            return 0.0;
        }
    }

    let nx = field.grid.nx;
    let ny = field.grid.ny;

    let v = field.data[idc];

    // Forward neighbours (domain edge handled by clamping to self).
    let ip = if i + 1 < nx { i + 1 } else { i };
    let jp = if j + 1 < ny { j + 1 } else { j };

    // If neighbour is vacuum, treat as free boundary: v_nb := v_center.
    let vx = if ip == i {
        v
    } else {
        let idn = field.idx(ip, j);
        if let Some(msk) = geom_mask {
            if !msk[idn] { v } else { field.data[idn] }
        } else {
            field.data[idn]
        }
    };

    let vy = if jp == j {
        v
    } else {
        let idn = field.idx(i, jp);
        if let Some(msk) = geom_mask {
            if !msk[idn] { v } else { field.data[idn] }
        } else {
            field.data[idn]
        }
    };

    let theta_x = angle_between_unit(v, vx);
    let theta_y = angle_between_unit(v, vy);
    theta_x.max(theta_y)
}

/// Compute a single coarse-grid patch as a buffered bounding box of cells where:
///   indicator >= frac * max(indicator).
///
/// Returns Some((rect, stats)) if any cells are flagged.
pub fn compute_patch_bbox_from_indicator(
    coarse: &VectorField2D,
    frac: f64,
    buffer: usize,
) -> Option<(Rect2i, IndicatorStats)> {
    compute_patch_bbox_from_indicator_geom(coarse, frac, buffer, None)
}

/// Geometry-mask-aware version of `compute_patch_bbox_from_indicator`.
///
/// If `geom_mask` is provided:
/// - Vacuum cells have indicator 0.
/// - Forward neighbours outside the mask are treated as equal to the center (free boundary),
///   so they contribute 0 to the gradient indicator.
pub fn compute_patch_bbox_from_indicator_geom(
    coarse: &VectorField2D,
    frac: f64,
    buffer: usize,
    geom_mask: Option<&[bool]>,
) -> Option<(Rect2i, IndicatorStats)> {
    if let Some(msk) = geom_mask {
        // Validate once at call site (not inside the per-cell indicator call).
        assert_mask_len(msk, &coarse.grid);
    }

    let nx = coarse.grid.nx;
    let ny = coarse.grid.ny;
    if nx == 0 || ny == 0 {
        return None;
    }

    // 1) maximum indicator
    let mut max_ind = 0.0_f64;
    for j in 0..ny {
        for i in 0..nx {
            let ind = indicator_grad2_forward_geom(coarse, i, j, geom_mask);
            if ind > max_ind {
                max_ind = ind;
            }
        }
    }
    if max_ind <= 0.0 {
        return None;
    }

    let frac = frac.max(0.0).min(1.0);
    let thresh = frac * max_ind;

    // 2) bounding box of flagged cells
    let mut found = false;
    let mut i_min = nx - 1;
    let mut i_max = 0usize;
    let mut j_min = ny - 1;
    let mut j_max = 0usize;

    for j in 0..ny {
        for i in 0..nx {
            let ind = indicator_grad2_forward_geom(coarse, i, j, geom_mask);
            if ind >= thresh {
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

    // 3) buffer + clamp using Rect2i helper
    let raw = Rect2i::new(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);
    let rect = raw.dilate_clamped(buffer, nx, ny);

    let stats = IndicatorStats {
        max: max_ind,
        threshold: thresh,
    };

    Some((rect, stats))
}

/// Compute a single coarse-grid patch as a buffered bounding box of cells where:
///   max_neighbour_angle(i,j) >= theta_refine.
///
/// `theta_refine` is in radians (typical values: 0.35–0.52 rad ≈ 20°–30° on the coarse
/// grid for r=2 refinement).
pub fn compute_patch_bbox_from_angle_threshold(
    coarse: &VectorField2D,
    theta_refine: f64,
    buffer: usize,
) -> Option<(Rect2i, IndicatorStats)> {
    compute_patch_bbox_from_angle_threshold_geom(coarse, theta_refine, buffer, None)
}

/// Geometry-mask-aware version of `compute_patch_bbox_from_angle_threshold`.
///
/// If `geom_mask` is provided:
/// - Vacuum cells have indicator 0.
/// - Forward neighbours outside the mask are treated as equal to the center (free boundary),
///   so they contribute 0.
pub fn compute_patch_bbox_from_angle_threshold_geom(
    coarse: &VectorField2D,
    theta_refine: f64,
    buffer: usize,
    geom_mask: Option<&[bool]>,
) -> Option<(Rect2i, IndicatorStats)> {
    if let Some(msk) = geom_mask {
        // Validate once at call site (not inside the per-cell indicator call).
        assert_mask_len(msk, &coarse.grid);
    }

    let nx = coarse.grid.nx;
    let ny = coarse.grid.ny;
    if nx == 0 || ny == 0 {
        return None;
    }

    // 1) maximum angle
    let mut max_theta = 0.0_f64;
    for j in 0..ny {
        for i in 0..nx {
            let th = indicator_angle_max_forward_geom(coarse, i, j, geom_mask);
            if th > max_theta {
                max_theta = th;
            }
        }
    }
    if max_theta <= 0.0 {
        return None;
    }

    // Absolute threshold (radians). If > max_theta, no cells are flagged.
    let thresh = theta_refine.max(0.0);

    // 2) bounding box of flagged cells
    let mut found = false;
    let mut i_min = nx - 1;
    let mut i_max = 0usize;
    let mut j_min = ny - 1;
    let mut j_max = 0usize;

    for j in 0..ny {
        for i in 0..nx {
            let th = indicator_angle_max_forward_geom(coarse, i, j, geom_mask);
            if th >= thresh {
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

    // 3) buffer + clamp using Rect2i helper
    let raw = Rect2i::new(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);
    let rect = raw.dilate_clamped(buffer, nx, ny);

    let stats = IndicatorStats {
        max: max_theta,
        threshold: thresh,
    };

    Some((rect, stats))
}
