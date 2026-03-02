// src/amr/indicator.rs
//
// Coarse-grid refinement indicators + simple patch selection helpers.
//
// Provides per-cell refinement indicators on the *coarse* grid for AMR patch
// placement.  All indicators support geometry masks (material/vacuum domains).
//
// Available indicators (selectable via `IndicatorKind`):
//
//   Grad2        – squared forward-difference gradient magnitude (legacy)
//   Angle        – maximum neighbour misalignment angle in radians (legacy)
//   DivInplane   – |∇·(M₁, M₂)| — García-Cervera & Roma (2005) Eq. 8
//   CurlMag      – ||∇ × M|| — catches vortex cores with low divergence
//   Composite    – max of normalised div, curl, grad2 (DEFAULT — universal)
//
// Mask convention (shared by all indicators):
//   - If the center cell is vacuum => indicator = 0.
//   - If a neighbour is vacuum => treat as equal to center (free boundary),
//     so the indicator does not spike at material–vacuum interfaces.

use crate::amr::rect::Rect2i;
use crate::geometry_mask::assert_mask_len;
use crate::vector_field::VectorField2D;

use rayon::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
//  IndicatorKind enum
// ═══════════════════════════════════════════════════════════════════════════

/// Selects which refinement indicator to use and how to threshold it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IndicatorKind {
    /// Squared forward-difference gradient magnitude (existing).
    /// Relative threshold: flag where ind >= frac × max(ind).
    Grad2 { frac: f64 },

    /// Maximum neighbour misalignment angle in radians (existing).
    /// Absolute threshold: flag where θ >= theta_refine.
    Angle { theta_refine: f64 },

    /// |∇·(M₁, M₂, 0)| — García-Cervera Eq. 8.
    /// Relative threshold: flag where |div| >= frac × max(|div|).
    DivInplane { frac: f64 },

    /// ||∇ × M|| — catches vortex cores with low divergence.
    /// Relative threshold: flag where |curl| >= frac × max(|curl|).
    CurlMag { frac: f64 },

    /// Max of normalised divergence, curl, and grad2.
    /// flag where max(|div|/max_div, |curl|/max_curl, grad2/max_grad2) >= frac.
    /// This is the universal catch-all: a cell is flagged if it scores
    /// high on *any* of the three constituent indicators.
    /// Recommended as the default for arbitrary magnetic textures.
    Composite { frac: f64 },
}

impl IndicatorKind {
    /// Convert from the legacy sign-convention used by ClusterPolicy / RegridPolicy:
    ///   frac >= 0  =>  Grad2 { frac }
    ///   frac <  0  =>  Angle { theta_refine: -frac }
    pub fn from_legacy_frac(frac: f64) -> Self {
        if frac < 0.0 {
            IndicatorKind::Angle {
                theta_refine: -frac,
            }
        } else {
            IndicatorKind::Grad2 { frac }
        }
    }

    /// Parse from the `LLG_AMR_INDICATOR` / `LLG_AMR_INDICATOR_FRAC` env vars.
    ///
    /// Falls back to `Composite { frac: 0.25 }` if unset — the universal indicator
    /// that catches walls (div), vortex cores (curl), and general texture (grad²).
    pub fn from_env() -> Self {
        let kind = std::env::var("LLG_AMR_INDICATOR").unwrap_or_else(|_| "composite".into());
        let frac: f64 = std::env::var("LLG_AMR_INDICATOR_FRAC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| default_frac_for(&kind));
        match kind.to_ascii_lowercase().as_str() {
            "grad2" => IndicatorKind::Grad2 { frac },
            "angle" => IndicatorKind::Angle {
                theta_refine: frac.to_radians(),
            },
            "div_inplane" | "div" => IndicatorKind::DivInplane { frac },
            "curl" | "curl_mag" => IndicatorKind::CurlMag { frac },
            "composite" => IndicatorKind::Composite { frac },
            _ => IndicatorKind::Composite { frac },
        }
    }

    /// Short human-readable label for logging / run_info.
    pub fn label(&self) -> &'static str {
        match self {
            IndicatorKind::Grad2 { .. } => "grad2",
            IndicatorKind::Angle { .. } => "angle",
            IndicatorKind::DivInplane { .. } => "div_inplane",
            IndicatorKind::CurlMag { .. } => "curl_mag",
            IndicatorKind::Composite { .. } => "composite",
        }
    }

    /// The threshold fraction / absolute value stored in this variant.
    pub fn threshold_param(&self) -> f64 {
        match *self {
            IndicatorKind::Grad2 { frac } => frac,
            IndicatorKind::Angle { theta_refine } => theta_refine,
            IndicatorKind::DivInplane { frac } => frac,
            IndicatorKind::CurlMag { frac } => frac,
            IndicatorKind::Composite { frac } => frac,
        }
    }

    /// Whether this variant uses a relative threshold (frac × max) vs absolute.
    pub fn is_relative(&self) -> bool {
        !matches!(self, IndicatorKind::Angle { .. })
    }
}

/// Default threshold fraction for each indicator kind (used when env var is unset).
fn default_frac_for(kind: &str) -> f64 {
    match kind.to_ascii_lowercase().as_str() {
        "grad2" => 0.25,
        "angle" => 25.0, // degrees — converted to radians in from_env()
        "div_inplane" | "div" => 0.10, // García-Cervera default
        "curl" | "curl_mag" => 0.10,
        "composite" => 0.10,
        _ => 0.25, // composite default
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Stats
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
pub struct IndicatorStats {
    pub max: f64,
    pub threshold: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn debug_assert_mask_len(_mask: &[bool], _field: &VectorField2D) {
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
#[inline]
fn angle_between_unit(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    clamp_pm1(dot).acos()
}

/// Sample one component of M at (ii, jj), with mask-aware free-boundary fallback.
///
/// If the cell is vacuum (mask=false), returns `fallback` (typically the center
/// cell's value) so that the finite-difference contribution is zero.
#[inline]
fn sample_component(
    field: &VectorField2D,
    ii: usize,
    jj: usize,
    c: usize,
    fallback: f64,
    geom_mask: Option<&[bool]>,
) -> f64 {
    let idx = field.idx(ii, jj);
    if let Some(msk) = geom_mask {
        if !msk[idx] {
            return fallback;
        }
    }
    field.data[idx][c]
}

/// Sample the full 3-vector at (ii, jj), with mask-aware free-boundary fallback.
#[inline]
fn sample_vec(
    field: &VectorField2D,
    ii: usize,
    jj: usize,
    fallback: [f64; 3],
    geom_mask: Option<&[bool]>,
) -> [f64; 3] {
    let idx = field.idx(ii, jj);
    if let Some(msk) = geom_mask {
        if !msk[idx] {
            return fallback;
        }
    }
    field.data[idx]
}

// ═══════════════════════════════════════════════════════════════════════════
//  Per-cell indicator functions
// ═══════════════════════════════════════════════════════════════════════════

// ── Grad² (existing) ─────────────────────────────────────────────────────

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
#[inline]
pub fn indicator_grad2_forward_geom(
    field: &VectorField2D,
    i: usize,
    j: usize,
    geom_mask: Option<&[bool]>,
) -> f64 {
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

    let ip = if i + 1 < nx { i + 1 } else { i };
    let jp = if j + 1 < ny { j + 1 } else { j };

    let vx = if ip == i {
        v
    } else {
        sample_vec(field, ip, j, v, geom_mask)
    };

    let vy = if jp == j {
        v
    } else {
        sample_vec(field, i, jp, v, geom_mask)
    };

    let dx0 = vx[0] - v[0];
    let dx1 = vx[1] - v[1];
    let dx2 = vx[2] - v[2];

    let dy0 = vy[0] - v[0];
    let dy1 = vy[1] - v[1];
    let dy2 = vy[2] - v[2];

    (dx0 * dx0 + dx1 * dx1 + dx2 * dx2) + (dy0 * dy0 + dy1 * dy1 + dy2 * dy2)
}

// ── Grad² central-difference (symmetric) ─────────────────────────────────

/// Symmetric coarse-grid indicator: central-difference gradient magnitude squared.
///
/// Uses all 4 neighbours (±i, ±j) instead of just (+i, +j), giving directionally
/// symmetric flagging.  At domain/mask boundaries, falls back to one-sided differences
/// (same free-boundary convention as the forward version).
///
/// This is the version used inside the composite indicator so that masked geometries
/// (disks, annuli) don't exhibit the bottom-left/top-right asymmetry inherent in the
/// forward-only stencil.
#[inline]
pub fn indicator_grad2_central_geom(
    field: &VectorField2D,
    i: usize,
    j: usize,
    geom_mask: Option<&[bool]>,
) -> f64 {
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

    // x-direction: (m(i+1) - m(i-1))/2 when both exist, else one-sided.
    let vxp = if i + 1 < nx {
        sample_vec(field, i + 1, j, v, geom_mask)
    } else {
        v
    };
    let vxm = if i > 0 {
        sample_vec(field, i - 1, j, v, geom_mask)
    } else {
        v
    };

    // y-direction: (m(j+1) - m(j-1))/2 when both exist, else one-sided.
    let vyp = if j + 1 < ny {
        sample_vec(field, i, j + 1, v, geom_mask)
    } else {
        v
    };
    let vym = if j > 0 {
        sample_vec(field, i, j - 1, v, geom_mask)
    } else {
        v
    };

    let gx0 = (vxp[0] - vxm[0]) * 0.5;
    let gx1 = (vxp[1] - vxm[1]) * 0.5;
    let gx2 = (vxp[2] - vxm[2]) * 0.5;

    let gy0 = (vyp[0] - vym[0]) * 0.5;
    let gy1 = (vyp[1] - vym[1]) * 0.5;
    let gy2 = (vyp[2] - vym[2]) * 0.5;

    (gx0 * gx0 + gx1 * gx1 + gx2 * gx2) + (gy0 * gy0 + gy1 * gy1 + gy2 * gy2)
}

// ── Angle (existing) ─────────────────────────────────────────────────────

/// Absolute coarse-grid indicator: maximum neighbour misalignment angle (radians).
#[inline]
pub fn indicator_angle_max_forward(field: &VectorField2D, i: usize, j: usize) -> f64 {
    indicator_angle_max_forward_geom(field, i, j, None)
}

/// Geometry-mask-aware version of `indicator_angle_max_forward`.
#[inline]
pub fn indicator_angle_max_forward_geom(
    field: &VectorField2D,
    i: usize,
    j: usize,
    geom_mask: Option<&[bool]>,
) -> f64 {
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

    let ip = if i + 1 < nx { i + 1 } else { i };
    let jp = if j + 1 < ny { j + 1 } else { j };

    let vx = if ip == i {
        v
    } else {
        sample_vec(field, ip, j, v, geom_mask)
    };

    let vy = if jp == j {
        v
    } else {
        sample_vec(field, i, jp, v, geom_mask)
    };

    let theta_x = angle_between_unit(v, vx);
    let theta_y = angle_between_unit(v, vy);
    theta_x.max(theta_y)
}

// ── Divergence of in-plane components (NEW — García-Cervera Eq. 8) ───────

/// In-plane divergence indicator: |∂M₁/∂x + ∂M₂/∂y|.
///
/// This is exactly the volume magnetic charge density ρ = ∇·M that sources
/// the demagnetising field.  Cells with high |ρ| contain domain walls where
/// M rotates in-plane.
///
/// Uses central differences at interior cells, one-sided at domain / mask edges.
/// No 1/dx normalisation — the threshold is relative (frac × max), so the
/// constant factor cancels.
///
/// García-Cervera & Roma (2005) flag cells where:
///   |∇·(M₁, M₂)| ≥ 0.10 × max|∇·(M₁, M₂)|
#[inline]
pub fn indicator_div_inplane_geom(
    field: &VectorField2D,
    i: usize,
    j: usize,
    geom_mask: Option<&[bool]>,
) -> f64 {
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

    // ∂M₁/∂x  (component 0, x-direction)
    let dm1_dx = central_diff_component(field, i, j, 0, v[0], nx, ny, true, geom_mask);

    // ∂M₂/∂y  (component 1, y-direction)
    let dm2_dy = central_diff_component(field, i, j, 1, v[1], nx, ny, false, geom_mask);

    (dm1_dx + dm2_dy).abs()
}

/// Non-mask-aware convenience wrapper.
#[inline]
pub fn indicator_div_inplane(field: &VectorField2D, i: usize, j: usize) -> f64 {
    indicator_div_inplane_geom(field, i, j, None)
}

// ── Curl magnitude (NEW) ─────────────────────────────────────────────────

/// Curl magnitude indicator: ||∇ × M||.
///
/// For a 2D cell-centred grid with M = (M₁, M₂, M₃):
///   (∇ × M)_x =  ∂M₃/∂y             (no z-variation in 2D)
///   (∇ × M)_y = -∂M₃/∂x
///   (∇ × M)_z =  ∂M₂/∂x - ∂M₁/∂y
///
/// Returns sqrt(curl_x² + curl_y² + curl_z²).
///
/// This catches vortex cores where magnetisation is rotating rapidly but
/// the in-plane divergence may be small.
#[inline]
pub fn indicator_curl_mag_geom(
    field: &VectorField2D,
    i: usize,
    j: usize,
    geom_mask: Option<&[bool]>,
) -> f64 {
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

    // Partial derivatives via central differences.
    let dm3_dy = central_diff_component(field, i, j, 2, v[2], nx, ny, false, geom_mask);
    let dm3_dx = central_diff_component(field, i, j, 2, v[2], nx, ny, true, geom_mask);
    let dm2_dx = central_diff_component(field, i, j, 1, v[1], nx, ny, true, geom_mask);
    let dm1_dy = central_diff_component(field, i, j, 0, v[0], nx, ny, false, geom_mask);

    let curl_x = dm3_dy; // ∂M₃/∂y
    let curl_y = -dm3_dx; // -∂M₃/∂x
    let curl_z = dm2_dx - dm1_dy; // ∂M₂/∂x - ∂M₁/∂y

    (curl_x * curl_x + curl_y * curl_y + curl_z * curl_z).sqrt()
}

/// Non-mask-aware convenience wrapper.
#[inline]
pub fn indicator_curl_mag(field: &VectorField2D, i: usize, j: usize) -> f64 {
    indicator_curl_mag_geom(field, i, j, None)
}

// ── Central-difference helper (shared by div + curl) ─────────────────────

/// Compute a central finite-difference of component `c` in the x- or y-direction.
///
/// Uses central differences at interior cells, one-sided (forward or backward)
/// at domain edges or mask boundaries.
///
///   `x_dir = true`  => differentiate along i (x-direction)
///   `x_dir = false`  => differentiate along j (y-direction)
///
/// No division by dx/dy — the threshold is relative so constant factors cancel.
/// (The factor of 2 in the central-difference denominator also cancels in
/// the ratio ind / max_ind, so we omit it for consistency and speed.)
#[inline]
fn central_diff_component(
    field: &VectorField2D,
    i: usize,
    j: usize,
    c: usize,
    center_val: f64,
    nx: usize,
    ny: usize,
    x_dir: bool,
    geom_mask: Option<&[bool]>,
) -> f64 {
    if x_dir {
        // Differentiate along i (x-direction).
        let has_left = i > 0;
        let has_right = i + 1 < nx;

        let val_right = if has_right {
            sample_component(field, i + 1, j, c, center_val, geom_mask)
        } else {
            center_val
        };
        let val_left = if has_left {
            sample_component(field, i - 1, j, c, center_val, geom_mask)
        } else {
            center_val
        };

        // After mask fallback, check what we actually have:
        let right_ok = has_right
            && (val_right != center_val || {
                // val_right could legitimately equal center_val even for a
                // valid material cell, so we need to check the mask directly.
                geom_mask.map_or(true, |m| m[field.idx(i + 1, j)])
            });
        let left_ok = has_left
            && (val_left != center_val || { geom_mask.map_or(true, |m| m[field.idx(i - 1, j)]) });

        if right_ok && left_ok {
            // Central: (f(i+1) - f(i-1)) / 2  [omitting the /2]
            (val_right - val_left) * 0.5
        } else if right_ok {
            // Forward: f(i+1) - f(i)
            val_right - center_val
        } else if left_ok {
            // Backward: f(i) - f(i-1)
            center_val - val_left
        } else {
            0.0
        }
    } else {
        // Differentiate along j (y-direction).
        let has_below = j > 0;
        let has_above = j + 1 < ny;

        let val_above = if has_above {
            sample_component(field, i, j + 1, c, center_val, geom_mask)
        } else {
            center_val
        };
        let val_below = if has_below {
            sample_component(field, i, j - 1, c, center_val, geom_mask)
        } else {
            center_val
        };

        let above_ok = has_above
            && (val_above != center_val || { geom_mask.map_or(true, |m| m[field.idx(i, j + 1)]) });
        let below_ok = has_below
            && (val_below != center_val || { geom_mask.map_or(true, |m| m[field.idx(i, j - 1)]) });

        if above_ok && below_ok {
            (val_above - val_below) * 0.5
        } else if above_ok {
            val_above - center_val
        } else if below_ok {
            center_val - val_below
        } else {
            0.0
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Indicator map computation (bulk)
// ═══════════════════════════════════════════════════════════════════════════

/// Indicator function signature: (field, i, j, geom_mask) -> f64.
pub type IndicatorFn = fn(&VectorField2D, usize, usize, Option<&[bool]>) -> f64;

/// Compute the per-cell indicator for every cell on the grid.
///
/// Returns a flat Vec of length nx*ny (row-major, j*nx + i).
///
/// Uses Rayon `par_iter` for parallel evaluation — this is embarrassingly parallel
/// since each cell's indicator depends only on its local stencil (read-only field).
pub fn compute_indicator_map(
    field: &VectorField2D,
    geom_mask: Option<&[bool]>,
    ind_fn: IndicatorFn,
) -> Vec<f64> {
    let nx = field.grid.nx;
    let ny = field.grid.ny;
    (0..nx * ny)
        .into_par_iter()
        .map(|k| {
            let i = k % nx;
            let j = k / nx;
            ind_fn(field, i, j, geom_mask)
        })
        .collect()
}

/// Compute indicator map and return (map, max_value).
pub fn compute_indicator_map_with_max(
    field: &VectorField2D,
    geom_mask: Option<&[bool]>,
    ind_fn: IndicatorFn,
) -> (Vec<f64>, f64) {
    let map = compute_indicator_map(field, geom_mask, ind_fn);
    let max_val = map.par_iter().cloned().reduce(|| 0.0_f64, f64::max);
    (map, max_val)
}

/// Evaluate the indicator specified by `kind` for a single cell.
///
/// Dispatches to the appropriate per-cell function.
pub fn evaluate_indicator(
    kind: IndicatorKind,
    field: &VectorField2D,
    i: usize,
    j: usize,
    geom_mask: Option<&[bool]>,
) -> f64 {
    match kind {
        IndicatorKind::Grad2 { .. } => indicator_grad2_forward_geom(field, i, j, geom_mask),
        IndicatorKind::Angle { .. } => indicator_angle_max_forward_geom(field, i, j, geom_mask),
        IndicatorKind::DivInplane { .. } => indicator_div_inplane_geom(field, i, j, geom_mask),
        IndicatorKind::CurlMag { .. } => indicator_curl_mag_geom(field, i, j, geom_mask),
        // Composite is not a single per-cell function — it needs two passes.
        // Return 0.0 here; use compute_composite_map for the real thing.
        IndicatorKind::Composite { .. } => 0.0,
    }
}

/// Return the per-cell indicator function pointer for a given kind.
///
/// Returns `None` for `Composite` (which needs a two-pass approach).
pub fn indicator_fn_for(kind: IndicatorKind) -> Option<IndicatorFn> {
    match kind {
        IndicatorKind::Grad2 { .. } => Some(indicator_grad2_forward_geom),
        IndicatorKind::Angle { .. } => Some(indicator_angle_max_forward_geom),
        IndicatorKind::DivInplane { .. } => Some(indicator_div_inplane_geom),
        IndicatorKind::CurlMag { .. } => Some(indicator_curl_mag_geom),
        IndicatorKind::Composite { .. } => None,
    }
}

/// Compute the composite indicator map: max of normalised div, curl, and grad2.
///
/// Each indicator is normalised to [0, 1] by dividing by its own maximum.
/// Per-cell value = max(|div| / max_div, |curl| / max_curl, grad2 / max_grad2).
/// A cell is flagged if it scores high on ANY constituent indicator:
///   - div catches domain walls (magnetic charge ∇·M)
///   - curl catches vortex / skyrmion cores (rotation)
///   - grad² catches everything else (general texture changes)
///
/// This is the recommended universal indicator for arbitrary magnetic textures.
///
/// Returns (map, effective_max) where effective_max is always 1.0 (already normalised).
pub fn compute_composite_map(field: &VectorField2D, geom_mask: Option<&[bool]>) -> (Vec<f64>, f64) {
    let (div_map, max_div) =
        compute_indicator_map_with_max(field, geom_mask, indicator_div_inplane_geom);
    let (curl_map, max_curl) =
        compute_indicator_map_with_max(field, geom_mask, indicator_curl_mag_geom);
    let (grad_map, max_grad) =
        compute_indicator_map_with_max(field, geom_mask, indicator_grad2_central_geom);

    let n = div_map.len();

    // Noise floor: only normalise a constituent if its max is at least 1% of the
    // strongest constituent. This prevents amplification of numerical noise when a
    // particular indicator type is physically absent (e.g. div ≈ 0 for Bloch walls,
    // curl ≈ 0 for purely radial textures).
    let dominant_max = max_div.max(max_curl).max(max_grad);
    let noise_floor = 0.01 * dominant_max;

    let inv_div = if max_div > noise_floor {
        1.0 / max_div
    } else {
        0.0
    };
    let inv_curl = if max_curl > noise_floor {
        1.0 / max_curl
    } else {
        0.0
    };
    let inv_grad = if max_grad > noise_floor {
        1.0 / max_grad
    } else {
        0.0
    };

    let composite: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|k| {
            let norm_div = div_map[k] * inv_div;
            let norm_curl = curl_map[k] * inv_curl;
            let norm_grad = grad_map[k] * inv_grad;
            norm_div.max(norm_curl).max(norm_grad)
        })
        .collect();

    // The map is already in [0, 1]; effective max is 1.0.
    (composite, 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Full indicator map dispatch for any IndicatorKind
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the indicator map + max + threshold for the given `IndicatorKind`.
///
/// Returns `(map, max_value, threshold)`.
///
/// For relative indicators, threshold = frac × max.
/// For Angle, threshold = theta_refine (absolute).
pub fn compute_indicator_map_for_kind(
    kind: IndicatorKind,
    field: &VectorField2D,
    geom_mask: Option<&[bool]>,
) -> (Vec<f64>, f64, f64) {
    match kind {
        IndicatorKind::Grad2 { frac } => {
            let (map, max_val) =
                compute_indicator_map_with_max(field, geom_mask, indicator_grad2_forward_geom);
            let thresh = frac.clamp(0.0, 1.0) * max_val;
            (map, max_val, thresh)
        }
        IndicatorKind::Angle { theta_refine } => {
            let (map, max_val) =
                compute_indicator_map_with_max(field, geom_mask, indicator_angle_max_forward_geom);
            (map, max_val, theta_refine.max(0.0))
        }
        IndicatorKind::DivInplane { frac } => {
            let (map, max_val) =
                compute_indicator_map_with_max(field, geom_mask, indicator_div_inplane_geom);
            let thresh = frac.clamp(0.0, 1.0) * max_val;
            (map, max_val, thresh)
        }
        IndicatorKind::CurlMag { frac } => {
            let (map, max_val) =
                compute_indicator_map_with_max(field, geom_mask, indicator_curl_mag_geom);
            let thresh = frac.clamp(0.0, 1.0) * max_val;
            (map, max_val, thresh)
        }
        IndicatorKind::Composite { frac } => {
            let (map, max_val) = compute_composite_map(field, geom_mask);
            let thresh = frac.clamp(0.0, 1.0) * max_val;
            (map, max_val, thresh)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Boundary-layer flagging  (García-Cervera second criterion)
// ═══════════════════════════════════════════════════════════════════════════

/// Flag a layer of cells near the domain boundary for refinement.
///
/// García-Cervera & Roma (2005): "it is convenient to mark for refinement a
/// layer of cells close to the boundary of the domain. This way, we are able
/// to compute the boundary integral given by (7) with the accuracy of the
/// finest level."
///
/// For rectangular domains without a mask: flags all cells within `layer_cells`
/// of the domain edge (north / south / east / west).
///
/// For geometry-masked domains: flags all material cells that are within
/// `layer_cells` hops of a vacuum cell (i.e. near the material–vacuum
/// interface).  Uses a BFS flood from the interface.
///
/// Returns a boolean mask of length nx*ny.  OR this into the flagged-cell
/// mask during clustering, *before* connected-components.
pub fn flag_boundary_layer(
    nx: usize,
    ny: usize,
    layer_cells: usize,
    geom_mask: Option<&[bool]>,
) -> Vec<bool> {
    let n = nx * ny;
    let mut flags = vec![false; n];

    if layer_cells == 0 {
        return flags;
    }

    match geom_mask {
        None => {
            // Rectangular domain: flag cells within `layer_cells` of any edge.
            for j in 0..ny {
                for i in 0..nx {
                    if i < layer_cells
                        || i >= nx.saturating_sub(layer_cells)
                        || j < layer_cells
                        || j >= ny.saturating_sub(layer_cells)
                    {
                        flags[j * nx + i] = true;
                    }
                }
            }
        }
        Some(mask) => {
            // Geometry-masked: BFS from material–vacuum interface.
            // Seed: all material cells adjacent to at least one vacuum cell.
            use std::collections::VecDeque;
            let mut dist = vec![usize::MAX; n];
            let mut queue = VecDeque::new();

            for j in 0..ny {
                for i in 0..nx {
                    let idx = j * nx + i;
                    if !mask[idx] {
                        continue; // vacuum cell — skip
                    }
                    // Check if any 4-neighbour is vacuum or domain edge.
                    let at_edge = i == 0 || i == nx - 1 || j == 0 || j == ny - 1;
                    let near_vacuum = [
                        (i.wrapping_sub(1), j),
                        (i + 1, j),
                        (i, j.wrapping_sub(1)),
                        (i, j + 1),
                    ]
                    .iter()
                    .any(|&(ni, nj)| {
                        if ni >= nx || nj >= ny {
                            return false;
                        }
                        !mask[nj * nx + ni]
                    });

                    if at_edge || near_vacuum {
                        dist[idx] = 0;
                        queue.push_back((i, j));
                        flags[idx] = true;
                    }
                }
            }

            // BFS outward up to `layer_cells` hops (within material only).
            while let Some((ci, cj)) = queue.pop_front() {
                let cd = dist[cj * nx + ci];
                if cd >= layer_cells {
                    continue;
                }
                for &(di, dj) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let ni = ci as i32 + di;
                    let nj = cj as i32 + dj;
                    if ni < 0 || nj < 0 {
                        continue;
                    }
                    let ni = ni as usize;
                    let nj = nj as usize;
                    if ni >= nx || nj >= ny {
                        continue;
                    }
                    let nidx = nj * nx + ni;
                    if !mask[nidx] {
                        continue; // don't flood into vacuum
                    }
                    let nd = cd + 1;
                    if nd < dist[nidx] {
                        dist[nidx] = nd;
                        flags[nidx] = true;
                        queue.push_back((ni, nj));
                    }
                }
            }
        }
    }

    flags
}

// ═══════════════════════════════════════════════════════════════════════════
//  Legacy single-patch helpers (preserved for backward compatibility)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute a single coarse-grid patch as a buffered bounding box of cells where:
///   indicator >= frac * max(indicator).
pub fn compute_patch_bbox_from_indicator(
    coarse: &VectorField2D,
    frac: f64,
    buffer: usize,
) -> Option<(Rect2i, IndicatorStats)> {
    compute_patch_bbox_from_indicator_geom(coarse, frac, buffer, None)
}

/// Geometry-mask-aware version of `compute_patch_bbox_from_indicator`.
pub fn compute_patch_bbox_from_indicator_geom(
    coarse: &VectorField2D,
    frac: f64,
    buffer: usize,
    geom_mask: Option<&[bool]>,
) -> Option<(Rect2i, IndicatorStats)> {
    if let Some(msk) = geom_mask {
        assert_mask_len(msk, &coarse.grid);
    }

    let nx = coarse.grid.nx;
    let ny = coarse.grid.ny;
    if nx == 0 || ny == 0 {
        return None;
    }

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

    let raw = Rect2i::new(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);
    let rect = raw.dilate_clamped(buffer, nx, ny);

    Some((
        rect,
        IndicatorStats {
            max: max_ind,
            threshold: thresh,
        },
    ))
}

/// Compute a single coarse-grid patch from an angle threshold.
pub fn compute_patch_bbox_from_angle_threshold(
    coarse: &VectorField2D,
    theta_refine: f64,
    buffer: usize,
) -> Option<(Rect2i, IndicatorStats)> {
    compute_patch_bbox_from_angle_threshold_geom(coarse, theta_refine, buffer, None)
}

/// Geometry-mask-aware version of `compute_patch_bbox_from_angle_threshold`.
pub fn compute_patch_bbox_from_angle_threshold_geom(
    coarse: &VectorField2D,
    theta_refine: f64,
    buffer: usize,
    geom_mask: Option<&[bool]>,
) -> Option<(Rect2i, IndicatorStats)> {
    if let Some(msk) = geom_mask {
        assert_mask_len(msk, &coarse.grid);
    }

    let nx = coarse.grid.nx;
    let ny = coarse.grid.ny;
    if nx == 0 || ny == 0 {
        return None;
    }

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

    let thresh = theta_refine.max(0.0);

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

    let raw = Rect2i::new(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1);
    let rect = raw.dilate_clamped(buffer, nx, ny);

    Some((
        rect,
        IndicatorStats {
            max: max_theta,
            threshold: thresh,
        },
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
//  Unit tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;
    use crate::vector_field::VectorField2D;

    /// Helper: create a small grid + field for testing.
    fn make_field(nx: usize, ny: usize, dx: f64, dy: f64) -> VectorField2D {
        let grid = Grid2D::new(nx, ny, dx, dy, dx); // dz = dx (irrelevant for 2D)
        VectorField2D::new(grid)
    }

    // ── IndicatorKind round-trips ────────────────────────────────────────

    #[test]
    fn legacy_frac_positive_gives_grad2() {
        let kind = IndicatorKind::from_legacy_frac(0.25);
        assert_eq!(kind, IndicatorKind::Grad2 { frac: 0.25 });
    }

    #[test]
    fn legacy_frac_negative_gives_angle() {
        let kind = IndicatorKind::from_legacy_frac(-0.4);
        match kind {
            IndicatorKind::Angle { theta_refine } => {
                assert!((theta_refine - 0.4).abs() < 1e-12);
            }
            _ => panic!("expected Angle variant"),
        }
    }

    #[test]
    fn label_round_trips() {
        assert_eq!(IndicatorKind::Grad2 { frac: 0.1 }.label(), "grad2");
        assert_eq!(
            IndicatorKind::DivInplane { frac: 0.1 }.label(),
            "div_inplane"
        );
        assert_eq!(IndicatorKind::CurlMag { frac: 0.1 }.label(), "curl_mag");
        assert_eq!(IndicatorKind::Composite { frac: 0.1 }.label(), "composite");
    }

    // ── Divergence indicator ─────────────────────────────────────────────

    #[test]
    fn div_inplane_uniform_field_is_zero() {
        // Uniform M = (1, 0, 0) everywhere => ∇·(M₁, M₂) = 0.
        let mut f = make_field(8, 8, 1.0, 1.0);
        for v in f.data.iter_mut() {
            *v = [1.0, 0.0, 0.0];
        }
        for j in 0..8 {
            for i in 0..8 {
                let d = indicator_div_inplane(&f, i, j);
                assert!(
                    d.abs() < 1e-12,
                    "div should be 0 for uniform field, got {} at ({},{})",
                    d,
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn div_inplane_radial_field() {
        // M = (x - cx, y - cy, 0) (un-normalised, but indicator doesn't care).
        // ∂M₁/∂x = 1,  ∂M₂/∂y = 1  =>  div = 2  everywhere (interior).
        let nx = 16;
        let ny = 16;
        let mut f = make_field(nx, ny, 1.0, 1.0);
        let cx = nx as f64 / 2.0;
        let cy = ny as f64 / 2.0;
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 + 0.5;
                let y = j as f64 + 0.5;
                let idx = f.idx(i, j);
                f.data[idx] = [x - cx, y - cy, 0.0];
            }
        }

        // Check an interior cell (away from edges where stencil is one-sided).
        let d = indicator_div_inplane(&f, 8, 8);
        // Central diff: (f(i+1) - f(i-1)) * 0.5 for each component.
        // ∂M₁/∂x = ((8.5+1 - cx) - (8.5-1 - cx)) * 0.5 = (1.0) * 0.5 * 2 ... wait.
        // Actually M₁(i,j) = (i + 0.5) - cx.
        // M₁(i+1, j) - M₁(i-1, j) = ((i+1+0.5) - cx) - ((i-1+0.5) - cx) = 2.0
        // central_diff = 2.0 * 0.5 = 1.0   (our function multiplies by 0.5)
        // Similarly ∂M₂/∂y = 1.0.
        // div = 1.0 + 1.0 = 2.0
        assert!(
            (d - 2.0).abs() < 1e-10,
            "radial field interior div should be ~2.0, got {}",
            d
        );
    }

    #[test]
    fn div_inplane_vacuum_cell_returns_zero() {
        let mut f = make_field(4, 4, 1.0, 1.0);
        for v in f.data.iter_mut() {
            *v = [1.0, 1.0, 0.0];
        }
        // Mask: cell (1,1) is vacuum.
        let mut mask = vec![true; 16];
        mask[1 * 4 + 1] = false;

        let d = indicator_div_inplane_geom(&f, 1, 1, Some(&mask));
        assert_eq!(d, 0.0, "vacuum cell should return 0");
    }

    // ── Curl indicator ───────────────────────────────────────────────────

    #[test]
    fn curl_mag_uniform_field_is_zero() {
        let mut f = make_field(8, 8, 1.0, 1.0);
        for v in f.data.iter_mut() {
            *v = [0.0, 1.0, 0.0];
        }
        for j in 0..8 {
            for i in 0..8 {
                let c = indicator_curl_mag(&f, i, j);
                assert!(
                    c.abs() < 1e-12,
                    "curl should be 0 for uniform field, got {} at ({},{})",
                    c,
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn curl_mag_vortex_field() {
        // M = (-y, x, 0) (solid-body rotation).
        // curl_z = ∂M₂/∂x - ∂M₁/∂y = 1 - (-1) = 2.
        // curl_x = ∂M₃/∂y = 0,  curl_y = -∂M₃/∂x = 0.
        // ||curl|| = 2.
        let nx = 16;
        let ny = 16;
        let mut f = make_field(nx, ny, 1.0, 1.0);
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 + 0.5;
                let y = j as f64 + 0.5;
                let idx = f.idx(i, j);
                f.data[idx] = [-y, x, 0.0];
            }
        }

        // Interior cell.
        let c = indicator_curl_mag(&f, 8, 8);
        // ∂M₂/∂x: M₂(i,j) = (i+0.5).  central diff = ((i+1+0.5)-(i-1+0.5))*0.5 = 1.0
        // ∂M₁/∂y: M₁(i,j) = -(j+0.5). central diff = (-(j+1+0.5) - (-(j-1+0.5)))*0.5 = -1.0
        // curl_z = 1.0 - (-1.0) = 2.0
        // ||curl|| = 2.0
        assert!(
            (c - 2.0).abs() < 1e-10,
            "vortex curl should be ~2.0, got {}",
            c
        );
    }

    #[test]
    fn curl_mag_irrotational_field() {
        // M = (x, y, 0) (radial / irrotational).
        // curl_z = ∂M₂/∂x - ∂M₁/∂y = 0 - 0 = 0.
        let nx = 16;
        let ny = 16;
        let mut f = make_field(nx, ny, 1.0, 1.0);
        for j in 0..ny {
            for i in 0..nx {
                let idx = f.idx(i, j);
                f.data[idx] = [i as f64, j as f64, 0.0];
            }
        }

        let c = indicator_curl_mag(&f, 8, 8);
        assert!(
            c.abs() < 1e-12,
            "irrotational field should have zero curl, got {}",
            c
        );
    }

    // ── Boundary-layer flagging ──────────────────────────────────────────

    #[test]
    fn boundary_layer_rectangular_layer1() {
        let flags = flag_boundary_layer(8, 8, 1, None);
        // All edge cells should be flagged, interior should not.
        for j in 0..8 {
            for i in 0..8 {
                let edge = i == 0 || i == 7 || j == 0 || j == 7;
                assert_eq!(
                    flags[j * 8 + i],
                    edge,
                    "cell ({},{}) edge={} but flag={}",
                    i,
                    j,
                    edge,
                    flags[j * 8 + i]
                );
            }
        }
    }

    #[test]
    fn boundary_layer_rectangular_layer2() {
        let flags = flag_boundary_layer(8, 8, 2, None);
        // Cells within 2 of any edge should be flagged.
        for j in 0..8 {
            for i in 0..8 {
                let near_edge = i < 2 || i >= 6 || j < 2 || j >= 6;
                assert_eq!(
                    flags[j * 8 + i],
                    near_edge,
                    "cell ({},{}) near_edge={} but flag={}",
                    i,
                    j,
                    near_edge,
                    flags[j * 8 + i]
                );
            }
        }
    }

    #[test]
    fn boundary_layer_zero_returns_all_false() {
        let flags = flag_boundary_layer(8, 8, 0, None);
        assert!(flags.iter().all(|&f| !f));
    }

    #[test]
    fn boundary_layer_masked_disk() {
        // 8×8 grid, circular mask of radius 3 centered at (3.5, 3.5).
        let nx = 8;
        let ny = 8;
        let cx = 3.5;
        let cy = 3.5;
        let r = 3.0;
        let mut mask = vec![false; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let dx = i as f64 + 0.5 - cx;
                let dy = j as f64 + 0.5 - cy;
                if dx * dx + dy * dy <= r * r {
                    mask[j * nx + i] = true;
                }
            }
        }

        let flags = flag_boundary_layer(nx, ny, 1, Some(&mask));

        // Every flagged cell must be material.
        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                if flags[idx] {
                    assert!(
                        mask[idx],
                        "flagged cell ({},{}) is vacuum — should not be flagged",
                        i, j
                    );
                }
            }
        }

        // At least some cells should be flagged (the disk edge).
        let n_flagged: usize = flags.iter().filter(|&&f| f).count();
        assert!(n_flagged > 0, "should flag some boundary cells");

        // Interior cells (far from edge) should NOT be flagged for layer=1.
        // Cell (3,3) and (4,4) are well inside the disk.
        // (This is a soft check — depends on disk discretisation.)
        // Cell (3,3) is at BFS distance 3 from the boundary — truly interior.
        // Cell (4,4) is only distance 1 (adjacent to seed (4,5) which borders vacuum
        // at (4,6)), so it IS correctly flagged.  Only check the true center.
        assert!(
            !flags[3 * nx + 3],
            "center cell (3,3) should not be flagged with layer=1 (BFS distance=3)"
        );
    }

    // ── Composite indicator ──────────────────────────────────────────────

    #[test]
    fn composite_map_values_in_unit_range() {
        // Any field should produce composite values in [0, 1].
        let nx = 8;
        let ny = 8;
        let mut f = make_field(nx, ny, 1.0, 1.0);
        // Create a simple gradient field.
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 / nx as f64;
                let idx = f.idx(i, j);
                f.data[idx] = [x, 1.0 - x, 0.0];
            }
        }

        let (map, max_val) = compute_composite_map(&f, None);
        assert!(
            (max_val - 1.0).abs() < 1e-12,
            "composite max should be 1.0, got {}",
            max_val
        );
        for (k, &v) in map.iter().enumerate() {
            assert!(
                v >= -1e-12 && v <= 1.0 + 1e-12,
                "composite[{}] = {} out of [0,1]",
                k,
                v
            );
        }
    }

    // ── compute_indicator_map_for_kind ────────────────────────────────────

    #[test]
    fn indicator_map_for_kind_grad2_matches_direct() {
        let nx = 8;
        let ny = 8;
        let mut f = make_field(nx, ny, 1.0, 1.0);
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64;
                let idx = f.idx(i, j);
                f.data[idx] = [x, 0.0, 0.0];
            }
        }

        let kind = IndicatorKind::Grad2 { frac: 0.25 };
        let (map, _max, _thresh) = compute_indicator_map_for_kind(kind, &f, None);

        // Spot-check a cell.
        let direct = indicator_grad2_forward_geom(&f, 3, 3, None);
        assert!(
            (map[3 * nx + 3] - direct).abs() < 1e-14,
            "map and direct should match"
        );
    }

    #[test]
    fn indicator_map_for_kind_div_threshold() {
        // Radial field: div=2 everywhere interior.
        // With frac=0.5, threshold should be 0.5 * max ≈ 0.5 * 2.0 = 1.0.
        let nx = 16;
        let ny = 16;
        let mut f = make_field(nx, ny, 1.0, 1.0);
        let cx = 8.0;
        let cy = 8.0;
        for j in 0..ny {
            for i in 0..nx {
                let idx = f.idx(i, j);
                f.data[idx] = [i as f64 + 0.5 - cx, j as f64 + 0.5 - cy, 0.0];
            }
        }

        let kind = IndicatorKind::DivInplane { frac: 0.5 };
        let (_map, max_val, thresh) = compute_indicator_map_for_kind(kind, &f, None);

        // Interior cells have div ≈ 2.0, edge cells may differ.
        assert!(max_val > 1.5, "max div should be around 2, got {}", max_val);
        assert!(
            (thresh - 0.5 * max_val).abs() < 1e-10,
            "threshold should be 0.5 * max"
        );
    }

    // ── Existing indicators still work ───────────────────────────────────

    #[test]
    fn grad2_forward_uniform_is_zero() {
        let mut f = make_field(4, 4, 1.0, 1.0);
        for v in f.data.iter_mut() {
            *v = [0.0, 0.0, 1.0];
        }
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(indicator_grad2_forward(&f, i, j), 0.0);
            }
        }
    }

    #[test]
    fn grad2_central_symmetric_on_disk() {
        // A disk mask on a small grid: the central-diff indicator should give
        // the same value at opposite edges of the disk (unlike forward-diff).
        let nx = 16;
        let ny = 16;
        let mut f = make_field(nx, ny, 1.0, 1.0);
        // Simple radial field: m = (x, y, 0)/r — creates a vortex-like pattern.
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 - 7.5;
                let y = j as f64 - 7.5;
                let r = (x * x + y * y).sqrt().max(0.01);
                f.data[j * nx + i] = [x / r, y / r, 0.0];
            }
        }
        // Disk mask: radius 6 cells centered at (7.5, 7.5).
        let mut mask = vec![false; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = i as f64 - 7.5;
                let y = j as f64 - 7.5;
                mask[j * nx + i] = x * x + y * y <= 36.0;
            }
        }
        // Compare top edge (i=8, j=13) vs bottom edge (i=8, j=2).
        // Central-diff should give similar values; forward-diff would not.
        let top = indicator_grad2_central_geom(&f, 8, 13, Some(&mask));
        let bot = indicator_grad2_central_geom(&f, 8, 2, Some(&mask));
        let ratio = if top > 0.0 { bot / top } else { 1.0 };
        assert!(
            (ratio - 1.0).abs() < 0.3,
            "central grad² should be roughly symmetric: top={top:.4e}, bot={bot:.4e}, ratio={ratio:.3}"
        );
    }

    #[test]
    fn angle_uniform_is_zero() {
        let mut f = make_field(4, 4, 1.0, 1.0);
        for v in f.data.iter_mut() {
            *v = [1.0, 0.0, 0.0];
        }
        for j in 0..4 {
            for i in 0..4 {
                assert_eq!(indicator_angle_max_forward(&f, i, j), 0.0);
            }
        }
    }
}
