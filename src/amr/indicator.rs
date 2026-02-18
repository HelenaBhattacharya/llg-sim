// src/amr/indicator.rs
//
// Coarse-grid refinement indicators + single-patch selection helpers.

use crate::amr::rect::Rect2i;
use crate::vector_field::VectorField2D;

#[derive(Clone, Copy, Debug)]
pub struct IndicatorStats {
    pub max: f64,
    pub threshold: f64,
}

/// Simple coarse-grid indicator: squared forward-difference gradient magnitude.
/// (Cheap + robust; good for Stage 2A and as a baseline for Stage 2B.)
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

/// Compute a single coarse-grid patch as a buffered bounding box of cells where:
///   indicator >= frac * max(indicator).
///
/// Returns Some((rect, stats)) if any cells are flagged.
pub fn compute_patch_bbox_from_indicator(
    coarse: &VectorField2D,
    frac: f64,
    buffer: usize,
) -> Option<(Rect2i, IndicatorStats)> {
    let nx = coarse.grid.nx;
    let ny = coarse.grid.ny;
    if nx == 0 || ny == 0 {
        return None;
    }

    // 1) maximum indicator
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
            let ind = indicator_grad2_forward(coarse, i, j);
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
