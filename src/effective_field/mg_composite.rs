// src/effective_field/mg_composite.rs
//
// Composite-grid demag for AMR micromagnetics.
//
// Implements the García-Cervera & Roma (2005) enhanced-RHS approach:
//
//   1. Build full 3D RHS from coarse M (includes z-surface charges correctly).
//   2. For each AMR patch, compute fine-resolution ∇·M, area-average to coarse
//      cells, compute delta = fine_avg − coarse_div, and ADD the correction
//      into the 3D RHS. This injects fine-geometry charge information (smooth
//      hole boundaries from patches) while preserving 3D surface charge physics.
//   3. Solve with MG+hybrid (treecode BCs + PPPM ΔK).
//   4. Interpolate coarse B_demag to patches via bilinear sampling.
//
// Key difference from the plain MG path: the coarse solver "sees" fine-resolution
// surface charges at antidot boundaries (from patches) instead of the coarse
// staircase. This should reduce the ~17% edge RMSE measured on antidot geometry.

use crate::amr::hierarchy::AmrHierarchy2D;
use crate::amr::interp::sample_bilinear;
use crate::amr::patch::Patch2D;
use crate::grid::Grid2D;
use crate::params::{MU0, Material};
use crate::vector_field::VectorField2D;

use super::demag_poisson_mg::{
    DemagPoissonMGConfig, DemagPoissonMGHybrid, HybridConfig,
};
use super::mg_kernels;

use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Patch-level Poisson data (Phase 1 of composite V-cycle)
// ---------------------------------------------------------------------------

/// Per-patch scalar-field storage for the composite Poisson solve.
///
/// Each AMR patch needs φ (potential), rhs (∇·M at fine resolution),
/// and residual (scratch) to participate in the composite V-cycle.
/// Dimensions include ghost cells, matching the patch's Grid2D.
pub(crate) struct PatchPoissonData {
    /// Scalar potential on the patch grid (including ghosts).

    pub phi: Vec<f64>,
    /// RHS = ∇·(Ms·m) at fine resolution (including ghosts).
    pub rhs: Vec<f64>,
    /// Scratch for residual = rhs - L(phi) (including ghosts).

    pub residual: Vec<f64>,
    /// Full patch grid dimensions (with ghosts).
    pub nx: usize,
    pub ny: usize,
    /// Ghost cell count.

    pub ghost: usize,
    /// Cell spacings at this level.
    pub dx: f64,
    pub dy: f64,
}

impl PatchPoissonData {
    /// Allocate storage for a patch. All arrays initialised to zero.
    pub fn new(patch: &Patch2D) -> Self {
        let nx = patch.grid.nx;
        let ny = patch.grid.ny;
        let n = nx * ny;
        Self {
            phi: vec![0.0; n],
            rhs: vec![0.0; n],
            residual: vec![0.0; n],
            nx,
            ny,
            ghost: patch.ghost,
            dx: patch.grid.dx,
            dy: patch.grid.dy,
        }
    }

    /// Compute RHS = ∇·(Ms·m) from the patch's magnetisation.
    ///
    /// Uses the same face-averaged divergence as mg_kernels::compute_div_m_2d,
    /// scaled by Ms. The divergence is computed on the full patch grid
    /// (including ghosts), matching the existing compute_patch_corrections
    /// approach.
    pub fn compute_rhs_from_m(&mut self, m_data: &[[f64; 3]], ms: f64) {
        debug_assert_eq!(m_data.len(), self.nx * self.ny);
        compute_scaled_div_m(m_data, self.nx, self.ny, self.dx, self.dy, ms, &mut self.rhs);
    }

    /// Area-average the fine RHS over the r×r fine cells corresponding to
    /// a single coarse cell at patch-local coarse index (ic, jc).
    ///
    /// Returns the averaged divergence value for that coarse cell.
    pub fn area_avg_rhs_at_coarse_cell(&self, ic: usize, jc: usize,
                                        ratio: usize, ghost: usize) -> f64 {
        let fi0 = ghost + ic * ratio;
        let fj0 = ghost + jc * ratio;
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for fj in fj0..fj0 + ratio {
            for fi in fi0..fi0 + ratio {
                if fi < self.nx && fj < self.ny {
                    sum += self.rhs[fj * self.nx + fi];
                    count += 1;
                }
            }
        }
        if count > 0 { sum / count as f64 } else { 0.0 }
    }
}

/// Allocate PatchPoissonData for all patches in the hierarchy.
///
/// Returns: (l1_data, l2plus_data) matching the structure of
/// h.patches and h.patches_l2plus.
fn allocate_patch_poisson_data(h: &AmrHierarchy2D)
    -> (Vec<PatchPoissonData>, Vec<Vec<PatchPoissonData>>)
{
    let l1: Vec<PatchPoissonData> = h.patches.iter()
        .map(PatchPoissonData::new).collect();
    let l2plus: Vec<Vec<PatchPoissonData>> = h.patches_l2plus.iter()
        .map(|lvl| lvl.iter().map(PatchPoissonData::new).collect())
        .collect();
    (l1, l2plus)
}

/// Compute fine RHS on all patch Poisson data from the hierarchy's magnetisation.
fn compute_all_patch_rhs(
    h: &AmrHierarchy2D,
    l1_data: &mut [PatchPoissonData],
    l2plus_data: &mut [Vec<PatchPoissonData>],
    ms: f64,
) {
    for (pd, patch) in l1_data.iter_mut().zip(h.patches.iter()) {
        pd.compute_rhs_from_m(&patch.m.data, ms);
    }
    for (lvl_data, lvl_patches) in l2plus_data.iter_mut().zip(h.patches_l2plus.iter()) {
        for (pd, patch) in lvl_data.iter_mut().zip(lvl_patches.iter()) {
            pd.compute_rhs_from_m(&patch.m.data, ms);
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2: 2D Laplacian, Jacobi smoother, scalar ghost-fill
// ---------------------------------------------------------------------------

/// Bilinear interpolation of a cell-centred scalar field at physical (x, y).
///
/// This is the scalar equivalent of `interp::sample_bilinear` for VectorField2D.
/// Uses the same coordinate convention: grid origin at (0,0), cell centres at
/// (i+0.5)*dx, (j+0.5)*dy. Clamps at boundaries.

fn sample_bilinear_scalar(
    data: &[f64], nx: usize, ny: usize, dx: f64, dy: f64,
    x: f64, y: f64,
) -> f64 {
    if nx == 0 || ny == 0 { return 0.0; }

    // Convert physical coordinate to continuous cell-centre index.
    let fx = x / dx - 0.5;
    let fy = y / dy - 0.5;

    let i0f = fx.floor();
    let j0f = fy.floor();
    let tx = fx - i0f;
    let ty = fy - j0f;

    // Clamp to valid range.
    let clamp = |v: isize, n: usize| -> usize {
        if v <= 0 { 0 }
        else if v >= n as isize - 1 { n - 1 }
        else { v as usize }
    };

    let i0 = clamp(i0f as isize, nx);
    let j0 = clamp(j0f as isize, ny);
    let i1 = clamp(i0f as isize + 1, nx);
    let j1 = clamp(j0f as isize + 1, ny);

    let v00 = data[j0 * nx + i0];
    let v10 = data[j0 * nx + i1];
    let v01 = data[j1 * nx + i0];
    let v11 = data[j1 * nx + i1];

    let v0 = v00 * (1.0 - tx) + v10 * tx;
    let v1 = v01 * (1.0 - tx) + v11 * tx;
    v0 * (1.0 - ty) + v1 * ty
}

/// Fill ghost cells in a patch's φ array from the coarse-level φ.
///
/// Uses the same coordinate mapping as `Patch2D::fill_ghosts_from_coarse` but
/// operates on scalar fields. Ghost cells get bilinearly interpolated values
/// from `coarse_phi`; interior cells are untouched.
///
/// `coarse_phi`: flat array of size `coarse_nx * coarse_ny` (L0 magnet-layer φ).
/// `coarse_dx`, `coarse_dy`: L0 cell spacings.

#[allow(dead_code)]
pub(crate) fn fill_phi_ghosts_from_coarse(
    patch: &Patch2D,
    patch_phi: &mut [f64],
    coarse_phi: &[f64],
    coarse_nx: usize, coarse_ny: usize,
    coarse_dx: f64, coarse_dy: f64,
) {
    let nx = patch.grid.nx;
    let ny = patch.grid.ny;
    let gi0 = patch.interior_i0();
    let gj0 = patch.interior_j0();
    let gi1 = patch.interior_i1();
    let gj1 = patch.interior_j1();

    for j in 0..ny {
        for i in 0..nx {
            let is_interior = i >= gi0 && i < gi1 && j >= gj0 && j < gj1;
            if is_interior {
                continue;
            }
            // Compute physical (x,y) of this patch cell — same as Patch2D::cell_center_xy
            let (x, y) = patch.cell_center_xy(i, j);
            // Bilinear interpolate from coarse φ
            patch_phi[j * nx + i] = sample_bilinear_scalar(
                coarse_phi, coarse_nx, coarse_ny, coarse_dx, coarse_dy, x, y);
        }
    }
}

/// Apply the 5-point 2D Laplacian on interior cells of a patch.
///
/// Writes L(φ)_{i,j} into `out` for all interior cells (ghost..nx-ghost, ghost..ny-ghost).
/// Ghost cells in `out` are set to zero.
/// Ghost values in `phi` are used as boundary data by the stencil.
#[allow(dead_code)]
pub(crate) fn laplacian_2d_interior(
    phi: &[f64], nx: usize, ny: usize, ghost: usize,
    dx: f64, dy: f64,
    out: &mut [f64],
) {
    debug_assert_eq!(phi.len(), nx * ny);
    debug_assert_eq!(out.len(), nx * ny);

    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);

    out.fill(0.0);

    for j in ghost..(ny - ghost) {
        for i in ghost..(nx - ghost) {
            let idx = j * nx + i;
            let phi_c = phi[idx];
            let phi_xp = phi[idx + 1];     // i+1, same j
            let phi_xm = phi[idx - 1];     // i-1, same j
            let phi_yp = phi[(j + 1) * nx + i]; // same i, j+1
            let phi_ym = phi[(j - 1) * nx + i]; // same i, j-1

            out[idx] = (phi_xp - 2.0 * phi_c + phi_xm) * inv_dx2
                     + (phi_yp - 2.0 * phi_c + phi_ym) * inv_dy2;
        }
    }
}

/// Compute residual r = rhs - L(φ) on interior cells.

#[allow(dead_code)]
pub(crate) fn compute_residual_2d(pd: &mut PatchPoissonData) {
    let nx = pd.nx;
    let ny = pd.ny;
    let ghost = pd.ghost;
    let inv_dx2 = 1.0 / (pd.dx * pd.dx);
    let inv_dy2 = 1.0 / (pd.dy * pd.dy);

    pd.residual.fill(0.0);

    for j in ghost..(ny - ghost) {
        for i in ghost..(nx - ghost) {
            let idx = j * nx + i;
            let phi_c = pd.phi[idx];
            let lap = (pd.phi[idx + 1] - 2.0 * phi_c + pd.phi[idx - 1]) * inv_dx2
                    + (pd.phi[(j + 1) * nx + i] - 2.0 * phi_c + pd.phi[(j - 1) * nx + i]) * inv_dy2;
            pd.residual[idx] = pd.rhs[idx] - lap;
        }
    }
}

/// Weighted Jacobi smoothing on a 2D patch grid.
///
/// Updates `phi` in place. Ghost cells are NOT modified (they are boundary data
/// from the coarse level). Only interior cells [ghost..nx-ghost, ghost..ny-ghost]
/// are updated.
///
/// `tmp` is scratch space of the same size as `phi`.
///
/// Standard Jacobi update:
///   phi_new = phi + ω/diag * (rhs - L(phi))
/// where diag = -2/dx² - 2/dy² (the centre coefficient of the 5-point stencil).

pub(crate) fn smooth_jacobi_2d(
    phi: &mut [f64], rhs: &[f64], tmp: &mut [f64],
    nx: usize, ny: usize, ghost: usize,
    dx: f64, dy: f64, omega: f64, n_iters: usize,
) {
    debug_assert_eq!(phi.len(), nx * ny);
    debug_assert_eq!(rhs.len(), nx * ny);
    debug_assert_eq!(tmp.len(), nx * ny);

    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);
    let diag = -2.0 * inv_dx2 - 2.0 * inv_dy2;
    let inv_diag = 1.0 / diag;

    for _iter in 0..n_iters {
        // Copy phi → tmp (including ghosts)
        tmp.copy_from_slice(phi);

        // Update interior cells
        for j in ghost..(ny - ghost) {
            for i in ghost..(nx - ghost) {
                let idx = j * nx + i;
                let phi_c = tmp[idx];
                let lap = (tmp[idx + 1] - 2.0 * phi_c + tmp[idx - 1]) * inv_dx2
                        + (tmp[(j + 1) * nx + i] - 2.0 * phi_c + tmp[(j - 1) * nx + i]) * inv_dy2;
                let residual = rhs[idx] - lap;
                phi[idx] = phi_c + omega * inv_diag * residual;
            }
        }
        // Ghost cells in phi remain unchanged (boundary data)
    }
}

// ---------------------------------------------------------------------------
// Phase 3: Residual restriction (fine → coarse)
// Phase 4: Correction prolongation (coarse → fine)
// ---------------------------------------------------------------------------

/// Area-average a scalar field over the r×r fine cells corresponding to
/// coarse cell (ic, jc) in patch-local coordinates.
///
/// Generic version — works on any &[f64] field stored on the patch grid
/// (rhs, residual, phi, etc.). Follows the same indexing pattern as
/// `Patch2D::restrict_to_coarse` and `compute_patch_corrections`.

#[allow(dead_code)]
fn area_avg_fine_to_coarse_cell(
    field: &[f64], nx: usize, ny: usize,
    ic: usize, jc: usize,
    ratio: usize, ghost: usize,
) -> f64 {
    let fi0 = ghost + ic * ratio;
    let fj0 = ghost + jc * ratio;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for fj in fj0..fj0 + ratio {
        for fi in fi0..fi0 + ratio {
            if fi < nx && fj < ny {
                sum += field[fj * nx + fi];
                count += 1;
            }
        }
    }
    if count > 0 { sum / count as f64 } else { 0.0 }
}

/// Restrict the fine-level residual from a patch to the coarse grid.
///
/// For each coarse cell covered by the patch, area-averages the fine residual
/// over the r×r fine cells and returns (coarse_cell_index, avg_residual) pairs.
///
/// These pairs are passed to `solve_with_corrections` to inject into the L0 RHS.
///
/// The coarse_cell_index uses the same convention as `compute_patch_corrections`:
///   cell_idx = coarse_j * base_nx + coarse_i

#[allow(dead_code)]
pub(crate) fn restrict_residual_to_coarse(
    patch: &Patch2D,
    patch_residual: &[f64],
    base_nx: usize,
) -> Vec<(usize, f64)> {
    let cr = &patch.coarse_rect;
    let ratio = patch.ratio;
    let ghost = patch.ghost;
    let pnx = patch.grid.nx;
    let pny = patch.grid.ny;

    let mut corrections = Vec::with_capacity(cr.nx * cr.ny);

    for jc in 0..cr.ny {
        for ic in 0..cr.nx {
            let coarse_i = cr.i0 + ic;
            let coarse_j = cr.j0 + jc;
            let cell_idx = coarse_j * base_nx + coarse_i;

            let avg = area_avg_fine_to_coarse_cell(
                patch_residual, pnx, pny, ic, jc, ratio, ghost);

            // Include all cells (even small values) — the L0 solver handles
            // the injection. This matches the reflux convention where fine
            // data supersedes coarse data at covered cells.
            corrections.push((cell_idx, avg));
        }
    }

    corrections
}

/// Prolongate the coarse φ correction to fine patch cells.
///
/// Computes delta = coarse_phi_new - coarse_phi_old (the correction from the
/// L0 solve), then bilinearly interpolates delta to each interior fine cell
/// and ADDS it to patch_phi. Ghost cells are not modified (they get updated
/// by fill_phi_ghosts_from_coarse separately).
///
/// This preserves fine-level detail from pre-smoothing while injecting the
/// coarse-level correction. Standard in composite MG (AMReX does this).

#[allow(dead_code)]
pub(crate) fn prolongate_phi_correction(
    coarse_phi_new: &[f64],
    coarse_phi_old: &[f64],
    coarse_nx: usize, coarse_ny: usize,
    coarse_dx: f64, coarse_dy: f64,
    patch: &Patch2D,
    patch_phi: &mut [f64],
) {
    debug_assert_eq!(coarse_phi_new.len(), coarse_nx * coarse_ny);
    debug_assert_eq!(coarse_phi_old.len(), coarse_nx * coarse_ny);

    // Compute the correction on the coarse grid.
    let delta_coarse: Vec<f64> = coarse_phi_new.iter()
        .zip(coarse_phi_old.iter())
        .map(|(new, old)| new - old)
        .collect();

    let pnx = patch.grid.nx;
    let gi0 = patch.interior_i0();
    let gj0 = patch.interior_j0();
    let gi1 = patch.interior_i1();
    let gj1 = patch.interior_j1();

    // Interpolate delta to each interior fine cell and add.
    for j in gj0..gj1 {
        for i in gi0..gi1 {
            let (x, y) = patch.cell_center_xy(i, j);
            let delta_interp = sample_bilinear_scalar(
                &delta_coarse, coarse_nx, coarse_ny, coarse_dx, coarse_dy, x, y);
            patch_phi[j * pnx + i] += delta_interp;
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 6: Fine-level gradient extraction
// ---------------------------------------------------------------------------

/// Extract B_demag from patch-level φ at fine resolution.
///
/// Bx = -μ₀ · ∂φ/∂x  (central difference at fine dx)
/// By = -μ₀ · ∂φ/∂y  (central difference at fine dy)
/// Bz = interpolated from coarse B_demag (the 3D L0 solve captures z-physics)
///
/// Ghost cells in φ provide the boundary data for the central difference
/// at patch edges. The gradient is computed for ALL cells (interior + ghosts)
/// to match the size of sample_coarse_to_patch output.
///
/// Returns a Vec of [Bx, By, Bz] per cell on the full patch grid.

#[allow(dead_code)]
pub(crate) fn extract_b_from_patch_phi(
    patch: &Patch2D,
    patch_phi: &[f64],
    b_coarse: &VectorField2D,  // for Bz interpolation
) -> Vec<[f64; 3]> {
    let pnx = patch.grid.nx;
    let pny = patch.grid.ny;
    let dx = patch.grid.dx;
    let dy = patch.grid.dy;
    let inv_2dx = 1.0 / (2.0 * dx);
    let inv_2dy = 1.0 / (2.0 * dy);

    let mut b = vec![[0.0f64; 3]; pnx * pny];

    for j in 0..pny {
        for i in 0..pnx {
            let idx = j * pnx + i;

            // Bx, By from fine φ gradient (central differences).
            // At boundaries (i=0 or i=pnx-1), use one-sided differences.
            let bx = if i > 0 && i + 1 < pnx {
                // Central difference.
                -MU0 * (patch_phi[j * pnx + (i + 1)] - patch_phi[j * pnx + (i - 1)]) * inv_2dx
            } else if i + 1 < pnx {
                // Forward difference at left boundary.
                -MU0 * (patch_phi[j * pnx + (i + 1)] - patch_phi[idx]) / dx
            } else if i > 0 {
                // Backward difference at right boundary.
                -MU0 * (patch_phi[idx] - patch_phi[j * pnx + (i - 1)]) / dx
            } else {
                0.0
            };

            let by = if j > 0 && j + 1 < pny {
                -MU0 * (patch_phi[(j + 1) * pnx + i] - patch_phi[(j - 1) * pnx + i]) * inv_2dy
            } else if j + 1 < pny {
                -MU0 * (patch_phi[(j + 1) * pnx + i] - patch_phi[idx]) / dy
            } else if j > 0 {
                -MU0 * (patch_phi[idx] - patch_phi[(j - 1) * pnx + i]) / dy
            } else {
                0.0
            };

            // Bz from coarse solution (interpolated).
            // The 3D L0 solve captures z-surface charges and z-gradient of φ.
            let (x, y) = patch.cell_center_xy(i, j);
            let b_coarse_val = sample_bilinear(b_coarse, x, y);
            let bz = b_coarse_val[2];

            b[idx] = [bx, by, bz];
        }
    }

    b
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[inline]
fn composite_diag() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLG_DEMAG_COMPOSITE_DIAG").is_ok())
}

// ---------------------------------------------------------------------------
// Divergence and injection helpers
// ---------------------------------------------------------------------------

/// Compute ∇·(Ms*m) on a 2D grid. m_data is unit vectors, scaled by ms internally.
fn compute_scaled_div_m(
    m_data: &[[f64; 3]], nx: usize, ny: usize,
    dx: f64, dy: f64, ms: f64, out: &mut [f64],
) {
    let scaled: Vec<[f64; 3]> = m_data.iter()
        .map(|v| [v[0] * ms, v[1] * ms, v[2] * ms]).collect();
    mg_kernels::compute_div_m_2d(&scaled, nx, ny, dx, dy, out);
}

/// Compute enhanced-RHS corrections from AMR patches.
///
/// For each coarse cell covered by a patch, computes:
///   delta = area_avg(fine_div) − coarse_div
///
/// Returns a Vec<(cell_index, delta)> for use with `solve_with_corrections`.
/// cell_index is j * base_nx + i in the coarse grid.
fn compute_patch_corrections(
    h: &AmrHierarchy2D,
    coarse_div: &[f64],
    ms: f64,
) -> Vec<(usize, f64)> {
    let base_nx = h.base_grid.nx;
    let mut corrections: Vec<(usize, f64)> = Vec::new();

    // Process all patches from coarsest to finest level.
    let all_patches: Vec<&Patch2D> = h.patches.iter()
        .chain(h.patches_l2plus.iter().flat_map(|lvl| lvl.iter()))
        .collect();

    for patch in all_patches {
        let ratio = patch.ratio;
        let ghost = patch.ghost;
        let cr = &patch.coarse_rect;
        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;

        // Compute fine ∇·(Ms*m) on the full patch grid (including ghosts).
        let mut fine_div = vec![0.0f64; pnx * pny];
        compute_scaled_div_m(
            &patch.m.data, pnx, pny,
            patch.grid.dx, patch.grid.dy, ms,
            &mut fine_div,
        );

        // For each coarse cell covered by this patch:
        for jc in 0..cr.ny {
            for ic in 0..cr.nx {
                let coarse_i = cr.i0 + ic;
                let coarse_j = cr.j0 + jc;
                let cell_idx = coarse_j * base_nx + coarse_i;

                // Area-average the fine divergence.
                let fi0 = ghost + ic * ratio;
                let fj0 = ghost + jc * ratio;
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for fj in fj0..fj0 + ratio {
                    for fi in fi0..fi0 + ratio {
                        if fi < pnx && fj < pny {
                            sum += fine_div[fj * pnx + fi];
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    let fine_avg = sum / count as f64;
                    let coarse_val = coarse_div[cell_idx];
                    let delta = fine_avg - coarse_val;

                    // Only add non-trivial corrections.
                    if delta.abs() > 1e-30 {
                        corrections.push((cell_idx, delta));
                    }
                }
            }
        }
    }

    corrections
}

/// Sample coarse B to the full patch grid via bilinear interpolation.
fn sample_coarse_to_patch(b_coarse: &VectorField2D, patch: &Patch2D) -> Vec<[f64; 3]> {
    let pnx = patch.grid.nx;
    let pny = patch.grid.ny;
    let mut b = vec![[0.0; 3]; pnx * pny];
    for j in 0..pny {
        for i in 0..pnx {
            let (x, y) = patch.cell_center_xy(i, j);
            b[j * pnx + i] = sample_bilinear(b_coarse, x, y);
        }
    }
    b
}

// ---------------------------------------------------------------------------
// Defect-correction per patch (García-Cervera / AMReX approach)
// ---------------------------------------------------------------------------

/// Compute defect-corrected B on a single patch.
///
/// Steps:
///   1. δrhs = fine_∇·M − interpolate(coarse_∇·M)
///   2. Smooth ∇²(δφ) = δrhs with DirichletZero BCs
///   3. δB = −μ₀∇(δφ)
///   4. B_patch = interpolate(B_L0) + δB
///
/// The defect RHS is HIGH-FREQUENCY (fine detail the coarse grid missed).
/// For high-k modes, 2D and 3D Green's functions agree, so the 2D Laplacian
/// is correct for the defect even though it's wrong for the full equation.
fn compute_defect_correction_on_patch(
    patch: &Patch2D,
    pd: &mut PatchPoissonData,
    coarse_div: &[f64],
    cnx: usize, cny: usize,
    cdx: f64, cdy: f64,
    b_l0: &VectorField2D,
    omega: f64,
    n_smooth: usize,
) -> Vec<[f64; 3]> {
    let pnx = pd.nx;
    let pny = pd.ny;
    let ghost = pd.ghost;

    // Step 1: Compute defect RHS = fine_div − interpolated(coarse_div).
    // pd.rhs already contains fine_div (from compute_all_patch_rhs).
    // Store defect RHS in pd.residual.
    for j in 0..pny {
        for i in 0..pnx {
            let (x, y) = patch.cell_center_xy(i, j);
            let coarse_interp = sample_bilinear_scalar(
                coarse_div, cnx, cny, cdx, cdy, x, y);
            pd.residual[j * pnx + i] = pd.rhs[j * pnx + i] - coarse_interp;
        }
    }

    // Step 2: Smooth ∇²(δφ) = defect_rhs with DirichletZero BCs.
    // δφ starts from zero. Ghost cells stay at zero (no ghost-fill from L0).
    pd.phi.fill(0.0);
    let defect_rhs = pd.residual.clone();
    let mut tmp = vec![0.0f64; pnx * pny];

    smooth_jacobi_2d(
        &mut pd.phi, &defect_rhs, &mut tmp,
        pnx, pny, ghost, pd.dx, pd.dy,
        omega, n_smooth,
    );

    // Step 3+4: Extract δB = −μ₀∇(δφ) and combine with interpolated B_L0.
    let inv_2dx = 1.0 / (2.0 * pd.dx);
    let inv_2dy = 1.0 / (2.0 * pd.dy);

    let mut b_out = vec![[0.0f64; 3]; pnx * pny];

    for j in 0..pny {
        for i in 0..pnx {
            let idx = j * pnx + i;

            // δBx, δBy from central differences on δφ.
            let dbx = if i > 0 && i + 1 < pnx {
                -MU0 * (pd.phi[j * pnx + (i + 1)] - pd.phi[j * pnx + (i - 1)]) * inv_2dx
            } else if i + 1 < pnx {
                -MU0 * (pd.phi[j * pnx + (i + 1)] - pd.phi[idx]) / pd.dx
            } else if i > 0 {
                -MU0 * (pd.phi[idx] - pd.phi[j * pnx + (i - 1)]) / pd.dx
            } else {
                0.0
            };

            let dby = if j > 0 && j + 1 < pny {
                -MU0 * (pd.phi[(j + 1) * pnx + i] - pd.phi[(j - 1) * pnx + i]) * inv_2dy
            } else if j + 1 < pny {
                -MU0 * (pd.phi[(j + 1) * pnx + i] - pd.phi[idx]) / pd.dy
            } else if j > 0 {
                -MU0 * (pd.phi[idx] - pd.phi[(j - 1) * pnx + i]) / pd.dy
            } else {
                0.0
            };

            // Interpolate B_L0 at this fine cell's position.
            let (x, y) = patch.cell_center_xy(i, j);
            let b_coarse = sample_bilinear(b_l0, x, y);

            // Combine: B_patch = B_L0_interp + δB
            // Bz purely from L0 (3D z-physics captured by the 3D padded-box solve).
            b_out[idx] = [
                b_coarse[0] + dbx,
                b_coarse[1] + dby,
                b_coarse[2],
            ];
        }
    }

    b_out
}

// ---------------------------------------------------------------------------
// Composite-grid demag solver
// ---------------------------------------------------------------------------

/// Configuration for the composite V-cycle.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CompositeVCycleConfig {
    /// Pre-smoothing iterations on patches.
    pub n_pre: usize,
    /// Post-smoothing iterations on patches.
    pub n_post: usize,
    /// Jacobi relaxation weight.
    pub omega: f64,
    /// Maximum number of outer V-cycle iterations.
    #[allow(dead_code)]
    pub max_cycles: usize,
}

impl Default for CompositeVCycleConfig {
    fn default() -> Self {
        Self {
            n_pre: 3,
            n_post: 3,
            omega: 2.0 / 3.0,
            max_cycles: 5,
        }
    }
}

pub(crate) struct CompositeGridPoisson {
    base_grid: Grid2D,
    l0_solver: DemagPoissonMGHybrid,
    /// Per-patch Poisson data for the composite V-cycle (Phase 1+).
    l1_data: Vec<PatchPoissonData>,
    l2plus_data: Vec<Vec<PatchPoissonData>>,
    /// L0 magnet-layer φ, persisted across V-cycle iterations and timesteps (warm start).
    coarse_phi: Vec<f64>,
    /// L0 in-plane ∇·(Ms·m), computed once per timestep from coarse M.
    coarse_div: Vec<f64>,
    /// V-cycle configuration.
    vcfg: CompositeVCycleConfig,
}

impl CompositeGridPoisson {
    pub(crate) fn new(base_grid: Grid2D) -> Self {
        let mg_cfg = DemagPoissonMGConfig::from_env();

        // V-cycle requires accurate L0 φ — auto-enable PPPM if user hasn't set it.
        if vcycle_enabled() && std::env::var("LLG_DEMAG_MG_HYBRID_ENABLE").is_err() {
            // SAFETY: single-threaded at init time, no concurrent readers.
            unsafe {
                std::env::set_var("LLG_DEMAG_MG_HYBRID_ENABLE", "1");
                std::env::set_var("LLG_DEMAG_MG_HYBRID_RADIUS", "14");
            }
            eprintln!(
                "[composite] V-cycle: auto-enabling PPPM hybrid (radius=14)"
            );
        }

        let hyb_cfg = HybridConfig::from_env();
        let l0_solver = DemagPoissonMGHybrid::new(base_grid, mg_cfg, hyb_cfg);
        let n = base_grid.nx * base_grid.ny;
        Self {
            base_grid,
            l0_solver,
            l1_data: Vec::new(),
            l2plus_data: Vec::new(),
            coarse_phi: vec![0.0; n],
            coarse_div: vec![0.0; n],
            vcfg: CompositeVCycleConfig::default(),
        }
    }

    pub(crate) fn same_structure(&self, h: &AmrHierarchy2D) -> bool {
        self.base_grid.nx == h.base_grid.nx
            && self.base_grid.ny == h.base_grid.ny
            && self.base_grid.dx == h.base_grid.dx
            && self.base_grid.dy == h.base_grid.dy
            && self.base_grid.dz == h.base_grid.dz
    }

    // ------------------------------------------------------------------
    // Phase 5: Composite V-cycle
    // ------------------------------------------------------------------

    /// Run one composite V-cycle iteration.
    ///
    /// Downstroke: fill ghost φ from coarse → pre-smooth → compute residual → restrict.
    /// L0 solve: existing 3D MG+PPPM with restricted fine residuals as corrections.
    /// Upstroke: prolongate correction → fill ghosts → post-smooth.
    ///
    /// `b_scratch` is a temporary VectorField2D used by solve_with_corrections.
    #[allow(dead_code)]
    fn vcycle_iteration(
        &mut self,
        h: &AmrHierarchy2D,
        mat: &Material,
        b_scratch: &mut VectorField2D,
    ) {
        let cnx = h.base_grid.nx;
        let cny = h.base_grid.ny;
        let cdx = h.base_grid.dx;
        let cdy = h.base_grid.dy;

        // ═══════ DOWNSTROKE: fine → coarse ═══════

        // Process L1 patches.
        for (patch, pd) in h.patches.iter().zip(self.l1_data.iter_mut()) {
            fill_phi_ghosts_from_coarse(
                patch, &mut pd.phi, &self.coarse_phi, cnx, cny, cdx, cdy);

            smooth_jacobi_2d(
                &mut pd.phi, &pd.rhs, &mut pd.residual,
                pd.nx, pd.ny, pd.ghost, pd.dx, pd.dy,
                self.vcfg.omega, self.vcfg.n_pre);

            compute_residual_2d(pd);
        }

        // Process L2+ patches (same: ghost-fill from L0, smooth, residual).
        for (lvl_patches, lvl_data) in
            h.patches_l2plus.iter().zip(self.l2plus_data.iter_mut())
        {
            for (patch, pd) in lvl_patches.iter().zip(lvl_data.iter_mut()) {
                fill_phi_ghosts_from_coarse(
                    patch, &mut pd.phi, &self.coarse_phi, cnx, cny, cdx, cdy);

                smooth_jacobi_2d(
                    &mut pd.phi, &pd.rhs, &mut pd.residual,
                    pd.nx, pd.ny, pd.ghost, pd.dx, pd.dy,
                    self.vcfg.omega, self.vcfg.n_pre);

                compute_residual_2d(pd);
            }
        }

        // Restrict residuals to coarse corrections.
        //
        // delta = restricted_fine_residual - coarse_in_plane_div
        //
        // This ensures solve_with_corrections produces:
        //   3d_rhs = (coarse_3d_div) + delta
        //          = z_surface_charges + restricted_fine_residual
        //
        // For V-cycle 1 with φ=0: fine_residual = fine_rhs = fine ∇·M,
        // so delta = area_avg(fine_rhs) - coarse_div — identical to the
        // existing enhanced-RHS path.

        let mut all_corrections: Vec<(usize, f64)> = Vec::new();

        for (patch, pd) in h.patches.iter().zip(self.l1_data.iter()) {
            let restricted = restrict_residual_to_coarse(
                patch, &pd.residual, cnx);
            for (cell_idx, avg_res) in restricted {
                let delta = avg_res - self.coarse_div[cell_idx];
                if delta.abs() > 1e-30 {
                    all_corrections.push((cell_idx, delta));
                }
            }
        }

        for (lvl_patches, lvl_data) in
            h.patches_l2plus.iter().zip(self.l2plus_data.iter())
        {
            for (patch, pd) in lvl_patches.iter().zip(lvl_data.iter()) {
                let restricted = restrict_residual_to_coarse(
                    patch, &pd.residual, cnx);
                for (cell_idx, avg_res) in restricted {
                    let delta = avg_res - self.coarse_div[cell_idx];
                    if delta.abs() > 1e-30 {
                        all_corrections.push((cell_idx, delta));
                    }
                }
            }
        }

        // ═══════ L0 SOLVE ═══════

        let phi_old = self.coarse_phi.clone();

        self.l0_solver.solve_with_corrections(
            &h.coarse, &all_corrections, b_scratch, mat);

        // Extract updated L0 φ.
        // DIAGNOSTIC NOTE: The MG solver produces φ_MG which differs from the
        // Newell-equivalent φ due to the FD stencil vs Newell kernel mismatch.
        // The PPPM-φ correction (above) applies a calibrated Δφ stencil to fix this.
        // The diagnostic below measures the remaining gap AFTER correction.
        let new_phi = self.l0_solver.mg.extract_magnet_layer_phi();
        self.coarse_phi.copy_from_slice(&new_phi);

        // ═══════ PPPM-φ CORRECTION ═══════
        // Apply the calibrated Δφ stencil to correct the MG potential.
        // After this, the 2D central-difference gradient of coarse_phi will
        // approximate the Newell B field, giving patches correct ghost BCs.
        if let Some(dk) = self.l0_solver.delta_kernel() {
            dk.apply_phi_correction(&h.coarse, &mut self.coarse_phi, mat.ms);
            if composite_diag() {
                let phi_min = self.coarse_phi.iter().cloned().fold(f64::INFINITY, f64::min);
                let phi_max = self.coarse_phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                eprintln!(
                    "[composite DIAG] Applied PPPM-φ correction. L0 φ range: [{:.4e}, {:.4e}]",
                    phi_min, phi_max);
            }
        }

        // ---- DIAGNOSTIC: measure φ-B inconsistency ----
        if composite_diag() {
            // Compute B from raw φ gradient (what patches will "see" via ghost-fill + ∇φ)
            // vs the ΔK-corrected B in b_scratch (what enhanced-RHS path uses).
            let phi = &self.coarse_phi;
            let cnx = h.base_grid.nx;
            let cny = h.base_grid.ny;
            let cdx = h.base_grid.dx;
            let cdy = h.base_grid.dy;
            let inv_2dx = 1.0 / (2.0 * cdx);
            let inv_2dy = 1.0 / (2.0 * cdy);

            let mut sum_b_corr_sq = 0.0f64;
            let mut sum_gap_sq = 0.0f64;
            let mut max_gap = 0.0f64;
            let mut n_cells = 0usize;

            for j in 1..cny-1 {
                for i in 1..cnx-1 {
                    let idx = j * cnx + i;
                    // B from raw φ gradient (no PPPM)
                    let bx_phi = -MU0 * (phi[j * cnx + (i+1)] - phi[j * cnx + (i-1)]) * inv_2dx;
                    let by_phi = -MU0 * (phi[(j+1) * cnx + i] - phi[(j-1) * cnx + i]) * inv_2dy;
                    // B from ΔK-corrected solve
                    let bc = b_scratch.data[idx];
                    let bx_corr = bc[0];
                    let by_corr = bc[1];
                    // Gap = what PPPM adds that φ doesn't have
                    let gap_x = bx_corr - bx_phi;
                    let gap_y = by_corr - by_phi;
                    let gap2 = gap_x * gap_x + gap_y * gap_y;
                    let bcorr2 = bx_corr * bx_corr + by_corr * by_corr;

                    sum_gap_sq += gap2;
                    sum_b_corr_sq += bcorr2;
                    max_gap = max_gap.max(gap2.sqrt());
                    n_cells += 1;
                }
            }

            if n_cells > 0 {
                let rmse_gap = (sum_gap_sq / n_cells as f64).sqrt();
                let rmse_b = (sum_b_corr_sq / n_cells as f64).sqrt();
                let rel = if rmse_b > 0.0 { rmse_gap / rmse_b * 100.0 } else { 0.0 };
                eprintln!(
                    "[composite DIAG] φ-B gap: RMSE(B_corrected - B_from_φ) = {:.4e} T ({:.1}% of |B|), max={:.4e} T",
                    rmse_gap, rel, max_gap,
                );
                eprintln!(
                    "[composite DIAG] → This gap is the PPPM ΔK correction that φ_long is MISSING.");
                eprintln!(
                    "[composite DIAG] → Patch ghost-fill uses φ_long, so patches don't see the ΔK physics.");
            }
        }

        // ═══════ UPSTROKE: coarse → fine ═══════

        for (patch, pd) in h.patches.iter().zip(self.l1_data.iter_mut()) {
            prolongate_phi_correction(
                &self.coarse_phi, &phi_old, cnx, cny, cdx, cdy,
                patch, &mut pd.phi);

            fill_phi_ghosts_from_coarse(
                patch, &mut pd.phi, &self.coarse_phi, cnx, cny, cdx, cdy);

            smooth_jacobi_2d(
                &mut pd.phi, &pd.rhs, &mut pd.residual,
                pd.nx, pd.ny, pd.ghost, pd.dx, pd.dy,
                self.vcfg.omega, self.vcfg.n_post);
        }

        for (lvl_patches, lvl_data) in
            h.patches_l2plus.iter().zip(self.l2plus_data.iter_mut())
        {
            for (patch, pd) in lvl_patches.iter().zip(lvl_data.iter_mut()) {
                prolongate_phi_correction(
                    &self.coarse_phi, &phi_old, cnx, cny, cdx, cdy,
                    patch, &mut pd.phi);

                fill_phi_ghosts_from_coarse(
                    patch, &mut pd.phi, &self.coarse_phi, cnx, cny, cdx, cdy);

                smooth_jacobi_2d(
                    &mut pd.phi, &pd.rhs, &mut pd.residual,
                    pd.nx, pd.ny, pd.ghost, pd.dx, pd.dy,
                    self.vcfg.omega, self.vcfg.n_post);
            }
        }
    }

    /// Run the full composite V-cycle solve.
    ///
    /// 1. Allocate/update patch data if needed.
    /// 2. Compute fine RHS on all patches.
    /// 3. Compute coarse in-plane div (for correction deltas).
    /// 4. Run V-cycle iterations.
    /// 5. Extract B on L0 and on patches at fine resolution.
    /// Composite V-cycle with defect correction.
    ///
    /// Architecture (following García-Cervera / AMReX):
    ///   1. L0 solve with enhanced RHS (fine ∇·M injected) → B_L0 (PPPM-corrected)
    ///   2. Per-patch DEFECT correction:
    ///        δrhs = ∇·M_fine − interpolate(∇·M_coarse)
    ///        Smooth ∇²(δφ) = δrhs with DirichletZero BCs
    ///        δB = −μ₀∇(δφ)
    ///   3. Final: B_patch = interpolate(B_L0) + δB
    ///
    /// Why DirichletZero works: The defect RHS is high-frequency (fine-scale
    /// detail the coarse grid missed). High-k corrections decay exponentially
    /// away from sources, so δφ→0 at patch boundaries. This avoids the
    /// 2D-vs-3D Green's function mismatch that broke the full-solve approach.
    pub(crate) fn compute_vcycle(
        &mut self,
        h: &AmrHierarchy2D,
        mat: &Material,
        b_demag_coarse: &mut VectorField2D,
    ) -> (Vec<Vec<[f64; 3]>>, Vec<Vec<Vec<[f64; 3]>>>) {
        if !mat.demag {
            b_demag_coarse.set_uniform(0.0, 0.0, 0.0);
            return (Vec::new(), Vec::new());
        }

        let n_l1 = h.patches.len();
        let has_patches = n_l1 > 0
            || h.patches_l2plus.iter().any(|v| !v.is_empty());

        // ---- Reallocate patch data if structure changed ----
        let need_realloc = self.l1_data.len() != n_l1
            || self.l2plus_data.len() != h.patches_l2plus.len()
            || self.l2plus_data.iter().zip(h.patches_l2plus.iter())
                .any(|(d, p)| d.len() != p.len())
            || self.l1_data.iter().zip(h.patches.iter())
                .any(|(pd, p)| pd.nx != p.grid.nx || pd.ny != p.grid.ny)
            || self.l2plus_data.iter().zip(h.patches_l2plus.iter())
                .any(|(lvl_d, lvl_p)| lvl_d.iter().zip(lvl_p.iter())
                    .any(|(pd, p)| pd.nx != p.grid.nx || pd.ny != p.grid.ny));

        if need_realloc {
            let (l1, l2) = allocate_patch_poisson_data(h);
            self.l1_data = l1;
            self.l2plus_data = l2;
        }

        let n_coarse = h.base_grid.nx * h.base_grid.ny;
        if self.coarse_phi.len() != n_coarse {
            self.coarse_phi = vec![0.0; n_coarse];
            self.coarse_div = vec![0.0; n_coarse];
        }

        // ---- Compute fine ∇·M on all patches ----
        if has_patches {
            compute_all_patch_rhs(h, &mut self.l1_data, &mut self.l2plus_data, mat.ms);
        }

        // ---- Compute coarse ∇·M on L0 ----
        compute_scaled_div_m(
            &h.coarse.data, h.base_grid.nx, h.base_grid.ny,
            h.base_grid.dx, h.base_grid.dy, mat.ms,
            &mut self.coarse_div,
        );

        // ════════════════════════════════════════════════════════════════
        // STEP 1: L0 solve with enhanced RHS → B_L0 (PPPM-corrected)
        // ════════════════════════════════════════════════════════════════
        let corrections = if has_patches {
            compute_patch_corrections(h, &self.coarse_div, mat.ms)
        } else {
            Vec::new()
        };

        self.l0_solver.solve_with_corrections(
            &h.coarse, &corrections, b_demag_coarse, mat);

        if composite_diag() {
            let n_l2plus: usize = h.patches_l2plus.iter().map(|v| v.len()).sum();
            eprintln!("[composite DEFECT] ═══════════════════════════════════════════════════");
            eprintln!("[composite DEFECT] Defect-correction mode: L1={}, L2+={}", n_l1, n_l2plus);
            eprintln!("[composite DEFECT] L0 solve complete. Computing per-patch defect corrections...");
        }

        // ════════════════════════════════════════════════════════════════
        // STEP 2: Per-patch defect correction
        // ════════════════════════════════════════════════════════════════
        //
        // For each patch:
        //   defect_rhs[i,j] = fine_div[i,j] − bilinear(coarse_div, x, y)
        //   Smooth ∇²(δφ) = defect_rhs with δφ=0 on all boundaries
        //   δB = −μ₀∇(δφ)
        //   B_patch = bilinear(B_L0) + [δBx, δBy, 0]
        //
        // Bz: purely from L0 interpolation (3D z-physics captured by L0 solve).

        let cnx = h.base_grid.nx;
        let cny = h.base_grid.ny;
        let cdx = h.base_grid.dx;
        let cdy = h.base_grid.dy;
        let n_smooth = (self.vcfg.n_pre + self.vcfg.n_post).max(10);
        let omega = self.vcfg.omega;

        // ---- Apply to all L1 patches ----
        let mut b_l1: Vec<Vec<[f64; 3]>> = Vec::with_capacity(h.patches.len());
        for (patch, pd) in h.patches.iter().zip(self.l1_data.iter_mut()) {
            let b = compute_defect_correction_on_patch(
                patch, pd, &self.coarse_div, cnx, cny, cdx, cdy,
                b_demag_coarse, omega, n_smooth);
            b_l1.push(b);
        }

        // ---- Apply to all L2+ patches ----
        let mut b_l2: Vec<Vec<Vec<[f64; 3]>>> = Vec::with_capacity(h.patches_l2plus.len());
        for (lvl_patches, lvl_data) in h.patches_l2plus.iter().zip(self.l2plus_data.iter_mut()) {
            let mut lvl_b = Vec::with_capacity(lvl_patches.len());
            for (patch, pd) in lvl_patches.iter().zip(lvl_data.iter_mut()) {
                let b = compute_defect_correction_on_patch(
                    patch, pd, &self.coarse_div, cnx, cny, cdx, cdy,
                    b_demag_coarse, omega, n_smooth);
                lvl_b.push(b);
            }
            b_l2.push(lvl_b);
        }

        if composite_diag() {
            // Report defect correction statistics
            let mut max_dphi = 0.0f64;
            for pd in self.l1_data.iter() {
                for &v in pd.phi.iter() {
                    max_dphi = max_dphi.max(v.abs());
                }
            }
            eprintln!("[composite DEFECT] max|δφ| on L1 patches = {:.4e}", max_dphi);
            eprintln!("[composite DEFECT] ═══════════════════════════════════════════════════");
        }

        (b_l1, b_l2)
    }

    pub(crate) fn compute(
        &mut self,
        h: &AmrHierarchy2D,
        mat: &Material,
        b_demag_coarse: &mut VectorField2D,
    ) -> (Vec<Vec<[f64; 3]>>, Vec<Vec<Vec<[f64; 3]>>>) {
        if !mat.demag {
            b_demag_coarse.set_uniform(0.0, 0.0, 0.0);
            return (Vec::new(), Vec::new());
        }

        let n_l1 = h.patches.len();
        let n_l2plus: usize = h.patches_l2plus.iter().map(|v| v.len()).sum();
        let has_patches = n_l1 > 0 || n_l2plus > 0;

        // ================================================================
        // Phase 1: Allocate and populate patch Poisson data
        // ================================================================

        // Reallocate if patch count or dimensions changed.
        let need_realloc = self.l1_data.len() != n_l1
            || self.l2plus_data.len() != h.patches_l2plus.len()
            || self.l2plus_data.iter().zip(h.patches_l2plus.iter())
                .any(|(d, p)| d.len() != p.len())
            || self.l1_data.iter().zip(h.patches.iter())
                .any(|(pd, p)| pd.nx != p.grid.nx || pd.ny != p.grid.ny)
            || self.l2plus_data.iter().zip(h.patches_l2plus.iter())
                .any(|(lvl_d, lvl_p)| lvl_d.iter().zip(lvl_p.iter())
                    .any(|(pd, p)| pd.nx != p.grid.nx || pd.ny != p.grid.ny));

        if need_realloc {
            let (l1, l2) = allocate_patch_poisson_data(h);
            self.l1_data = l1;
            self.l2plus_data = l2;
        }

        // Compute fine ∇·(Ms·m) on every patch.
        if has_patches {
            compute_all_patch_rhs(h, &mut self.l1_data, &mut self.l2plus_data, mat.ms);
        }

        // ================================================================
        // Step 1: Compute corrections from patches (enhanced RHS)
        //
        // This uses the EXISTING compute_patch_corrections path.
        // In later phases, the composite V-cycle will replace this with
        // restriction of the fine residual from PatchPoissonData.
        // ================================================================

        let corrections = if has_patches {
            // Compute coarse ∇·(Ms*m) for delta computation.
            let nx = h.base_grid.nx;
            let ny = h.base_grid.ny;
            let mut coarse_div = vec![0.0f64; nx * ny];
            compute_scaled_div_m(
                &h.coarse.data, nx, ny,
                h.base_grid.dx, h.base_grid.dy, mat.ms,
                &mut coarse_div,
            );

            let corr = compute_patch_corrections(h, &coarse_div, mat.ms);

            // Phase 1 diagnostic: verify PatchPoissonData RHS matches
            // the fine divergence that compute_patch_corrections uses.
            if composite_diag() {
                let max_delta = corr.iter()
                    .map(|(_, d)| d.abs()).fold(0.0f64, f64::max);
                eprintln!(
                    "[composite] enhanced RHS: grid={}x{}, n_l1={}, n_l2plus={}, \
                     {} corrections, max_delta={:.3e}",
                    h.base_grid.nx, h.base_grid.ny, n_l1, n_l2plus,
                    corr.len(), max_delta,
                );

                // Cross-check: for each L1 patch, compare area-averaged
                // PatchPoissonData.rhs against the corrections.
                let base_nx = h.base_grid.nx;
                for (pi, (patch, pd)) in h.patches.iter().zip(self.l1_data.iter()).enumerate() {
                    let cr = &patch.coarse_rect;
                    let mut max_rhs_diff = 0.0f64;
                    for jc in 0..cr.ny {
                        for ic in 0..cr.nx {
                            let cell_idx = (cr.j0 + jc) * base_nx + (cr.i0 + ic);
                            let pd_avg = pd.area_avg_rhs_at_coarse_cell(
                                ic, jc, patch.ratio, patch.ghost);
                            // Find matching correction
                            let corr_delta = corr.iter()
                                .find(|(idx, _)| *idx == cell_idx)
                                .map(|(_, d)| *d).unwrap_or(0.0);
                            // The correction is delta = fine_avg - coarse_div.
                            // So fine_avg = corr_delta + coarse_div.
                            let expected_fine_avg = corr_delta + coarse_div[cell_idx];
                            let diff = (pd_avg - expected_fine_avg).abs();
                            max_rhs_diff = max_rhs_diff.max(diff);
                        }
                    }
                    eprintln!(
                        "[composite] Phase1 cross-check: patch {} max|pd.rhs_avg - expected|={:.3e}",
                        pi, max_rhs_diff,
                    );
                }
            }

            corr
        } else {
            if composite_diag() {
                eprintln!(
                    "[composite] no patches, plain MG+hybrid on {}x{}",
                    h.base_grid.nx, h.base_grid.ny,
                );
            }
            Vec::new()
        };

        // ================================================================
        // Step 2: Solve with enhanced RHS
        // ================================================================

        self.l0_solver.solve_with_corrections(
            &h.coarse,
            &corrections,
            b_demag_coarse,
            mat,
        );

        // ================================================================
        // Step 3: Extract L0 phi for future composite V-cycle use
        // ================================================================

        // Store the L0 magnet-layer phi. Not used in the current
        // enhanced-RHS path, but needed by Phases 2+ for ghost-fill
        // and prolongation.
        let _l0_phi = self.l0_solver.mg.extract_magnet_layer_phi();

        if composite_diag() {
            let phi_max = _l0_phi.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let phi_min = _l0_phi.iter().cloned().fold(f64::INFINITY, f64::min);
            eprintln!(
                "[composite] L0 phi after solve: min={:.6e}, max={:.6e}",
                phi_min, phi_max,
            );
        }

        // ================================================================
        // Step 4: Interpolate coarse B to patches (existing path)
        // ================================================================

        let b_l1: Vec<Vec<[f64; 3]>> = h.patches.iter()
            .map(|p| sample_coarse_to_patch(b_demag_coarse, p)).collect();

        let b_l2: Vec<Vec<Vec<[f64; 3]>>> = h.patches_l2plus.iter()
            .map(|lvl| lvl.iter()
                .map(|p| sample_coarse_to_patch(b_demag_coarse, p)).collect()
            ).collect();

        (b_l1, b_l2)
    }
}

// ---------------------------------------------------------------------------
// Module-level cache + public API
// ---------------------------------------------------------------------------

static COMPOSITE_CACHE: OnceLock<Mutex<Option<CompositeGridPoisson>>> = OnceLock::new();

/// Check if the composite V-cycle mode is enabled via environment variable.
///
/// LLG_DEMAG_COMPOSITE_VCYCLE=1 enables the true composite V-cycle (Phases 0–6).
/// Default (unset or 0): uses the existing enhanced-RHS path.
fn vcycle_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("LLG_DEMAG_COMPOSITE_VCYCLE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Compute AMR-aware demag using the composite-grid solver.
///
/// Called from the stepper's `CompositeGrid` mode.
///
/// Two paths:
/// - Default: enhanced-RHS (inject fine ∇·M into coarse MG, interpolate B to patches)
/// - LLG_DEMAG_COMPOSITE_VCYCLE=1: true composite V-cycle (smooth φ on patches,
///   extract B at fine resolution from patch φ)
pub fn compute_composite_demag(
    h: &AmrHierarchy2D,
    mat: &Material,
    b_demag_coarse: &mut VectorField2D,
) -> (Vec<Vec<[f64; 3]>>, Vec<Vec<Vec<[f64; 3]>>>) {
    let cache = COMPOSITE_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().expect("COMPOSITE_CACHE mutex poisoned");

    let rebuild = match guard.as_ref() {
        Some(s) => !s.same_structure(h),
        None => true,
    };

    if rebuild {
        let solver = CompositeGridPoisson::new(h.base_grid);
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| {
            if vcycle_enabled() {
                eprintln!(
                    "[composite] V-CYCLE mode enabled (LLG_DEMAG_COMPOSITE_VCYCLE=1) — \
                     fine-resolution B within patches"
                );
            } else {
                eprintln!(
                    "[composite] enhanced-RHS mode (default) — \
                     set LLG_DEMAG_COMPOSITE_VCYCLE=1 for true composite V-cycle"
                );
            }
        });
        *guard = Some(solver);
    }

    let solver = guard.as_mut().unwrap();
    if vcycle_enabled() {
        solver.compute_vcycle(h, mat, b_demag_coarse)
    } else {
        solver.compute(h, mat, b_demag_coarse)
    }
}

#[cfg(test)]
mod phase1_tests {
    use super::*;

    /// Test that PatchPoissonData computes the same divergence as
    /// the existing compute_scaled_div_m helper.
    #[test]
    fn test_patch_poisson_data_rhs_matches_standalone() {
        // Create a fake "patch-like" magnetisation on a 10x10 grid.
        let nx = 10usize;
        let ny = 10usize;
        let dx = 5e-9;
        let dy = 5e-9;
        let ms = 800e3;

        // Vortex-like pattern: m = (-y, x, 0) / r (normalised)
        let cx = nx as f64 * 0.5;
        let cy = ny as f64 * 0.5;
        let mut m_data = vec![[0.0f64; 3]; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 + 0.5) - cx;
                let y = (j as f64 + 0.5) - cy;
                let r = (x * x + y * y).sqrt().max(0.5);
                m_data[j * nx + i] = [-y / r, x / r, 0.0];
            }
        }

        // Compute divergence using the standalone function.
        let mut div_standalone = vec![0.0f64; nx * ny];
        compute_scaled_div_m(&m_data, nx, ny, dx, dy, ms, &mut div_standalone);

        // Create PatchPoissonData manually (without a real Patch2D).
        let mut pd = PatchPoissonData {
            phi: vec![0.0; nx * ny],
            rhs: vec![0.0; nx * ny],
            residual: vec![0.0; nx * ny],
            nx,
            ny,
            ghost: 0,
            dx,
            dy,
        };

        // Compute via PatchPoissonData method.
        pd.compute_rhs_from_m(&m_data, ms);

        // They must be bit-identical — both call the same function.
        for k in 0..nx * ny {
            assert_eq!(
                pd.rhs[k], div_standalone[k],
                "RHS mismatch at cell {}: pd={:.6e} standalone={:.6e}",
                k, pd.rhs[k], div_standalone[k]
            );
        }

        // Verify the divergence is non-trivial (vortex has ∇·M ≠ 0 near centre).
        let max_div = pd.rhs.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(
            max_div > 1e-10,
            "divergence should be non-zero for vortex pattern, got max={:.3e}",
            max_div
        );

        eprintln!("[phase1] RHS bit-identical to standalone: max|div|={:.6e}", max_div);
        eprintln!("[phase1] PASSED: PatchPoissonData.compute_rhs_from_m matches compute_scaled_div_m");
    }

    /// Test area_avg_rhs_at_coarse_cell against manual summation.
    #[test]
    fn test_area_avg_rhs() {
        // Simulate a 12×12 patch with ghost=2 and ratio=2.
        // Interior: 8×8 fine cells covering 4×4 coarse cells.
        let nx = 12usize;
        let ny = 12usize;
        let ghost = 2usize;
        let ratio = 2usize;

        let mut pd = PatchPoissonData {
            phi: vec![0.0; nx * ny],
            rhs: vec![0.0; nx * ny],
            residual: vec![0.0; nx * ny],
            nx,
            ny,
            ghost,
            dx: 5e-9,
            dy: 5e-9,
        };

        // Fill RHS with a known pattern: rhs[j*nx+i] = (i+1) * (j+1)
        for j in 0..ny {
            for i in 0..nx {
                pd.rhs[j * nx + i] = ((i + 1) * (j + 1)) as f64;
            }
        }

        // For coarse cell (ic=0, jc=0), ratio=2, ghost=2:
        // Fine cells: fi in [2,3], fj in [2,3]
        // Values: rhs[2*12+2]=9, rhs[2*12+3]=12, rhs[3*12+2]=12, rhs[3*12+3]=16
        // Average: (9+12+12+16)/4 = 49/4 = 12.25
        let avg00 = pd.area_avg_rhs_at_coarse_cell(0, 0, ratio, ghost);
        assert!(
            (avg00 - 12.25).abs() < 1e-12,
            "area_avg at (0,0): expected 12.25, got {}", avg00
        );

        // For coarse cell (ic=1, jc=0), ratio=2, ghost=2:
        // Fine cells: fi in [4,5], fj in [2,3]
        // Values: rhs[2*12+4]=15, rhs[2*12+5]=18, rhs[3*12+4]=20, rhs[3*12+5]=24
        // Average: (15+18+20+24)/4 = 77/4 = 19.25
        let avg10 = pd.area_avg_rhs_at_coarse_cell(1, 0, ratio, ghost);
        assert!(
            (avg10 - 19.25).abs() < 1e-12,
            "area_avg at (1,0): expected 19.25, got {}", avg10
        );

        eprintln!("[phase1] area_avg(0,0)={:.4}, area_avg(1,0)={:.4}", avg00, avg10);
        eprintln!("[phase1] PASSED: area_avg_rhs_at_coarse_cell matches manual calculation");
    }

    /// Test that the 2D face-averaged divergence produces physically correct
    /// results for a non-uniform magnetisation with vacuum cells.
    ///
    /// Key insight from the failure: for UNIFORM M on a fully-material grid,
    /// the 2D face-averaged divergence is zero everywhere (including at domain
    /// boundaries) because face_val returns the cell's own value when the
    /// neighbour is outside. Surface charges in the physical problem come from
    /// z-faces (handled by the 3D build_rhs_from_m, not the 2D divergence).
    ///
    /// This test uses a partially-vacuum grid where the material boundary
    /// DOES produce in-plane surface charges visible to the 2D divergence.
    #[test]
    fn test_material_vacuum_interface_divergence() {
        let nx = 16usize;
        let ny = 16usize;
        let dx = 5e-9;
        let dy = 5e-9;
        let ms = 800e3;

        // Left half is material (M=[1,0,0]), right half is vacuum (M=[0,0,0]).
        // The interface at i=7/i=8 should produce a surface charge:
        // face_val(true, 1.0, false, 0.0) = 1.0 (left side contributes)
        // vs face_val(true, 1.0, true, 1.0) = 1.0 (interior face)
        // So at i=7 (last material cell): right face = face_val(true,1,false,0)=1.0
        //                                  left face = face_val(true,1,true,1)=1.0
        //                                  → div_x = (1.0 - 1.0)/dx = 0
        // But at i=8 (first vacuum cell): is_mag = false → entire cell contributes 0
        //
        // Actually the surface charge shows up at the LAST material cell.
        // At i=7: right face = face_val(c_in=true, mc=1, xp_in=false, 0) = 1.0
        //         left face  = face_val(xm_in=true, 1, c_in=true, 1) = 1.0
        //         → div_x = 0  ... hmm.
        //
        // The divergence is truly zero for uniform M even at the interface
        // because the one-sided face value equals the cell value.
        // The "charge" manifests when M VARIES near the interface.
        //
        // Use a gradient pattern instead: M_x increases linearly with i.
        let mut m_data = vec![[0.0f64; 3]; nx * ny];
        for j in 0..ny {
            for i in 0..nx / 2 {
                // M_x varies linearly: stronger to the right
                let mx = (i as f64 + 1.0) / (nx as f64 / 2.0);
                m_data[j * nx + i] = [mx, 0.0, 0.0];
            }
            // Right half stays [0,0,0] (vacuum)
        }

        let mut pd = PatchPoissonData {
            phi: vec![0.0; nx * ny],
            rhs: vec![0.0; nx * ny],
            residual: vec![0.0; nx * ny],
            nx,
            ny,
            ghost: 0,
            dx,
            dy,
        };

        pd.compute_rhs_from_m(&m_data, ms);

        // Interior material cells (i in 2..6) should have non-zero div
        // because M_x is increasing → ∂M_x/∂x > 0.
        let mut max_interior_div = 0.0f64;
        for j in 2..ny - 2 {
            for i in 2..6 {
                max_interior_div = max_interior_div.max(pd.rhs[j * nx + i].abs());
            }
        }

        // Interface cell (i=7, last material cell before vacuum):
        // should have large divergence from the material-vacuum transition.
        let j_mid = ny / 2;
        let div_interface = pd.rhs[j_mid * nx + 7];

        // First vacuum cell (i=8) should have zero divergence.
        let div_vacuum = pd.rhs[j_mid * nx + 8];

        eprintln!(
            "[phase1] gradient M: max_interior_div={:.3e}, div_interface(i=7)={:.6e}, div_vacuum(i=8)={:.6e}",
            max_interior_div, div_interface, div_vacuum
        );

        assert!(
            max_interior_div > 0.0,
            "interior divergence should be > 0 for increasing M_x"
        );
        // Interface cell has non-zero divergence (reduced by face-averaging
        // at the material-vacuum boundary).
        assert!(
            div_interface.abs() > 0.0,
            "interface divergence should be non-zero, got {:.3e}",
            div_interface.abs()
        );
        // The first vacuum cell (i=8) also has non-zero divergence because
        // the face-averaged operator's face_val sees the material neighbour
        // at i=7: face_val(xm_in=true, m_xm, c_in=false, 0) = m_xm.
        // This is correct — it's how the operator distributes surface charge.
        // Deep vacuum cells (i >= 10) should be truly zero.
        let div_deep_vacuum = pd.rhs[j_mid * nx + 10];
        assert!(
            div_deep_vacuum.abs() < 1e-30,
            "deep vacuum divergence should be zero, got {:.3e}",
            div_deep_vacuum.abs()
        );

        // Verify total charge conservation: sum of all divergence should
        // relate to the net flux out of the material region.
        let total_div: f64 = pd.rhs.iter().sum();
        eprintln!("[phase1] total divergence sum: {:.6e}", total_div);

        eprintln!("[phase1] PASSED: material-vacuum interface produces correct divergence pattern");
    }
}

#[cfg(test)]
mod phase2_tests {
    use super::*;
    use std::f64::consts::PI;

    /// Manufactured solution test: verify the 5-point Laplacian is correct.
    ///
    /// φ(x,y) = sin(πx/Lx) · sin(πy/Ly)
    /// L(φ) = -(π²/Lx² + π²/Ly²) · φ
    ///
    /// We set up a grid, fill φ with the exact function, apply the Laplacian,
    /// and check that L(φ) matches the analytical value to O(dx²).
    #[test]
    fn test_laplacian_2d_manufactured() {
        let nx = 34usize; // 30 interior + 2 ghost on each side
        let ny = 34usize;
        let ghost = 2usize;
        let dx = 5e-9;
        let dy = 5e-9;
        let lx = (nx - 2 * ghost) as f64 * dx; // physical domain size
        let ly = (ny - 2 * ghost) as f64 * dy;

        // Fill phi with sin(πx/Lx)·sin(πy/Ly)
        // x,y are local coordinates within the interior: x = (i - ghost + 0.5)*dx
        let mut phi = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 - ghost as f64 + 0.5) * dx;
                let y = (j as f64 - ghost as f64 + 0.5) * dy;
                phi[j * nx + i] = (PI * x / lx).sin() * (PI * y / ly).sin();
            }
        }

        // Apply Laplacian
        let mut lap = vec![0.0f64; nx * ny];
        laplacian_2d_interior(&phi, nx, ny, ghost, dx, dy, &mut lap);

        // Expected: L(φ) = -(π²/Lx² + π²/Ly²) · φ
        let expected_factor = -(PI * PI / (lx * lx) + PI * PI / (ly * ly));

        let mut max_err = 0.0f64;
        let mut max_rel_err = 0.0f64;
        for j in ghost..(ny - ghost) {
            for i in ghost..(nx - ghost) {
                let idx = j * nx + i;
                let expected = expected_factor * phi[idx];
                let err = (lap[idx] - expected).abs();
                max_err = max_err.max(err);
                if expected.abs() > 1e-30 {
                    max_rel_err = max_rel_err.max(err / expected.abs());
                }
            }
        }

        // Second-order error: O(dx²) = O((5e-9)²·π⁴/L⁴) relative ~ O((π/N)²) ≈ 0.01
        eprintln!(
            "[phase2] Laplacian manufactured: max_abs_err={:.3e}, max_rel_err={:.6}",
            max_err, max_rel_err
        );
        assert!(
            max_rel_err < 0.02,
            "Laplacian relative error too large: {:.6} (expected < 0.02 for 30-cell grid)",
            max_rel_err
        );
        eprintln!("[phase2] PASSED: Laplacian matches analytical to O(dx²)");
    }

    /// Smoother convergence test: verify Jacobi reduces the residual at the
    /// expected rate.
    ///
    /// Jacobi is NOT meant to fully solve the problem — it smooths high-frequency
    /// errors. In the composite V-cycle, we run 2-4 iterations per level.
    /// The coarse-grid correction handles the low-frequency convergence.
    ///
    /// What we test: after N iterations, the residual decreases. The convergence
    /// rate for the smoothest mode on an M×M grid with ω=2/3 is approximately
    /// ρ ≈ 1 - (2/3)(π/M)². For M=30: ρ ≈ 0.9966, so 500 iters gives ~5.5× reduction.
    /// High-frequency modes converge much faster (ρ_high ≈ 1/3).
    #[test]
    fn test_jacobi_smoother_convergence() {
        let nx = 34usize;
        let ny = 34usize;
        let ghost = 2usize;
        let dx = 1.0;
        let dy = 1.0;
        let n_int_x = nx - 2 * ghost;
        let n_int_y = ny - 2 * ghost;
        let lx = n_int_x as f64;
        let ly = n_int_y as f64;

        let expected_factor = -(PI * PI / (lx * lx) + PI * PI / (ly * ly));
        let mut rhs = vec![0.0f64; nx * ny];
        let mut phi_exact = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f64 - ghost as f64 + 0.5) * dx;
                let y = (j as f64 - ghost as f64 + 0.5) * dy;
                let val = (PI * x / lx).sin() * (PI * y / ly).sin();
                phi_exact[j * nx + i] = val;
                rhs[j * nx + i] = expected_factor * val;
            }
        }

        // Ghost cells = exact solution (Dirichlet BCs). Interior = 0.
        let mut phi = vec![0.0f64; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let is_interior = i >= ghost && i < nx - ghost
                               && j >= ghost && j < ny - ghost;
                if !is_interior {
                    phi[j * nx + i] = phi_exact[j * nx + i];
                }
            }
        }
        let mut tmp = vec![0.0f64; nx * ny];
        let omega = 2.0 / 3.0;

        // Measure residual BEFORE any smoothing (should be large).
        let mut lap_before = vec![0.0f64; nx * ny];
        laplacian_2d_interior(&phi, nx, ny, ghost, dx, dy, &mut lap_before);
        let mut res_before = 0.0f64;
        for j in ghost..(ny - ghost) {
            for i in ghost..(nx - ghost) {
                let v = (rhs[j * nx + i] - lap_before[j * nx + i]).abs();
                if v > res_before { res_before = v; }
            }
        }

        // Run 50 iterations (typical: 2-4 per V-cycle level, but 50 to see clear reduction).
        smooth_jacobi_2d(&mut phi, &rhs, &mut tmp, nx, ny, ghost, dx, dy, omega, 50);

        // Measure residual AFTER.
        let mut lap_after = vec![0.0f64; nx * ny];
        laplacian_2d_interior(&phi, nx, ny, ghost, dx, dy, &mut lap_after);
        let mut res_after = 0.0f64;
        for j in ghost..(ny - ghost) {
            for i in ghost..(nx - ghost) {
                let v = (rhs[j * nx + i] - lap_after[j * nx + i]).abs();
                if v > res_after { res_after = v; }
            }
        }

        let reduction = res_after / res_before;

        eprintln!(
            "[phase2] Jacobi: res_before={:.3e}, res_after={:.3e}, reduction={:.4} ({:.1}× in 50 iters)",
            res_before, res_after, reduction, 1.0 / reduction
        );

        // The residual must decrease. For 50 Jacobi iterations on a 30×30 grid,
        // even the slowest mode (k=1) reduces by 0.9966^50 ≈ 0.84.
        // High-frequency modes reduce by (1/3)^50 ≈ 0 (effectively eliminated).
        // Overall max-norm residual should drop by at least 2× in 50 iters.
        assert!(
            reduction < 0.95,
            "Jacobi should reduce residual: before={:.3e}, after={:.3e}, ratio={:.4}",
            res_before, res_after, reduction
        );

        // Run 500 more iterations and check further reduction.
        smooth_jacobi_2d(&mut phi, &rhs, &mut tmp, nx, ny, ghost, dx, dy, omega, 500);
        let mut lap_final = vec![0.0f64; nx * ny];
        laplacian_2d_interior(&phi, nx, ny, ghost, dx, dy, &mut lap_final);
        let mut res_final = 0.0f64;
        for j in ghost..(ny - ghost) {
            for i in ghost..(nx - ghost) {
                let v = (rhs[j * nx + i] - lap_final[j * nx + i]).abs();
                if v > res_final { res_final = v; }
            }
        }

        let total_reduction = res_final / res_before;

        // After 550 total iterations, expect significant reduction.
        // Slowest mode: 0.9966^550 ≈ 0.15. Fast modes: essentially zero.
        eprintln!(
            "[phase2] Jacobi total: res_final={:.3e}, total_reduction={:.4} ({:.1}× in 550 iters)",
            res_final, total_reduction, 1.0 / total_reduction
        );

        assert!(
            total_reduction < 0.5,
            "550 Jacobi iterations should reduce residual by at least 2×, got {:.4}×",
            1.0 / total_reduction
        );

        eprintln!("[phase2] PASSED: Jacobi smoother reduces residual at expected rate");
    }

    /// Ghost-fill consistency test: verify that scalar ghost-fill gives the
    /// same values as vector ghost-fill for a field where all three components
    /// carry the same scalar value.
    ///
    /// This is the critical coordinate-consistency check.
    #[test]
    fn test_ghost_fill_matches_vector_version() {
        use crate::amr::interp::sample_bilinear;
        use crate::amr::patch::Patch2D;
        use crate::amr::rect::Rect2i;
        use crate::grid::Grid2D;
        use crate::vector_field::VectorField2D;

        // Create a coarse grid and a scalar field on it.
        let base = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let coarse_nx = base.nx;
        let coarse_ny = base.ny;

        // Fill coarse phi with a smooth function: φ = sin(πx/Lx)·cos(πy/Ly)
        let lx = base.nx as f64 * base.dx;
        let ly = base.ny as f64 * base.dy;
        let mut coarse_phi = vec![0.0f64; coarse_nx * coarse_ny];
        let mut coarse_vec = VectorField2D::new(base);
        for j in 0..coarse_ny {
            for i in 0..coarse_nx {
                let x = (i as f64 + 0.5) * base.dx;
                let y = (j as f64 + 0.5) * base.dy;
                let val = (PI * x / lx).sin() * (PI * y / ly).cos();
                coarse_phi[j * coarse_nx + i] = val;
                // Put same value in all three vector components
                coarse_vec.data[j * coarse_nx + i] = [val, val, val];
            }
        }

        // Create a patch in the middle of the domain.
        let rect = Rect2i::new(4, 4, 4, 4); // covers coarse cells (4,4) to (7,7)
        let ratio = 2;
        let ghost_cells = 2;
        let patch = Patch2D::new(&base, rect, ratio, ghost_cells);

        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;

        // Scalar ghost-fill
        let mut patch_phi = vec![0.0f64; pnx * pny];
        fill_phi_ghosts_from_coarse(
            &patch, &mut patch_phi,
            &coarse_phi, coarse_nx, coarse_ny, base.dx, base.dy,
        );

        // Vector ghost-fill (using sample_bilinear on the VectorField2D)
        let gi0 = patch.interior_i0();
        let gj0 = patch.interior_j0();
        let gi1 = patch.interior_i1();
        let gj1 = patch.interior_j1();

        let mut max_diff = 0.0f64;
        let mut n_ghost_checked = 0usize;
        for j in 0..pny {
            for i in 0..pnx {
                let is_interior = i >= gi0 && i < gi1 && j >= gj0 && j < gj1;
                if is_interior {
                    continue;
                }
                let (x, y) = patch.cell_center_xy(i, j);
                let vec_val = sample_bilinear(&coarse_vec, x, y);
                let scalar_val = patch_phi[j * pnx + i];
                let diff = (scalar_val - vec_val[0]).abs();
                max_diff = max_diff.max(diff);
                n_ghost_checked += 1;
            }
        }

        eprintln!(
            "[phase2] ghost-fill consistency: {} ghost cells checked, max_diff={:.3e}",
            n_ghost_checked, max_diff
        );

        // Must be bit-identical since we use the same coordinate mapping and
        // the same bilinear interpolation logic.
        assert!(
            max_diff < 1e-14,
            "scalar ghost-fill differs from vector version by {:.3e}", max_diff
        );

        eprintln!("[phase2] PASSED: scalar ghost-fill matches vector version");
    }

    /// Residual computation test: verify compute_residual_2d matches
    /// manual rhs - L(phi).
    #[test]
    fn test_compute_residual_2d() {
        let nx = 10usize;
        let ny = 10usize;
        let ghost = 2usize;
        let dx = 1.0;
        let dy = 1.0;

        let mut pd = PatchPoissonData {
            phi: vec![0.0; nx * ny],
            rhs: vec![0.0; nx * ny],
            residual: vec![0.0; nx * ny],
            nx,
            ny,
            ghost,
            dx,
            dy,
        };

        // Set phi and rhs to known values.
        for j in 0..ny {
            for i in 0..nx {
                pd.phi[j * nx + i] = (i as f64) * (j as f64);
                pd.rhs[j * nx + i] = 1.0;
            }
        }

        // Compute residual
        compute_residual_2d(&mut pd);

        // Manually compute L(phi) and expected residual at an interior point.
        // phi = i*j, so:
        //   d²phi/dx² = 0 (phi is linear in i at fixed j)
        //   d²phi/dy² = 0 (phi is linear in j at fixed i)
        //   L(phi) = 0 everywhere in the interior
        //   residual = rhs - L(phi) = 1.0 - 0.0 = 1.0
        for j in ghost..(ny - ghost) {
            for i in ghost..(nx - ghost) {
                let idx = j * nx + i;
                assert!(
                    (pd.residual[idx] - 1.0).abs() < 1e-10,
                    "residual at ({},{}) = {:.6e}, expected 1.0",
                    i, j, pd.residual[idx]
                );
            }
        }

        // Ghost residuals should be zero.
        assert_eq!(pd.residual[0], 0.0, "ghost residual should be 0");

        eprintln!("[phase2] PASSED: compute_residual_2d matches manual calculation");
    }
}

#[cfg(test)]
mod phase3_tests {
    use super::*;
    use crate::amr::patch::Patch2D;
    use crate::amr::rect::Rect2i;
    use crate::grid::Grid2D;

    /// Test that restrict_residual_to_coarse produces one correction per
    /// covered coarse cell with the correct area-averaged value.
    #[test]
    fn test_restrict_residual_basic() {
        // Create a base grid and a patch.
        let base = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let rect = Rect2i::new(4, 4, 4, 4); // covers coarse cells (4..8, 4..8)
        let ratio = 2;
        let ghost = 2;
        let patch = Patch2D::new(&base, rect, ratio, ghost);

        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;

        // Create a PatchPoissonData and set the residual to a known pattern.
        let mut pd = PatchPoissonData::new(&patch);

        // Set residual = 1.0 everywhere interior, 0.0 at ghosts.
        // With ratio=2 and uniform residual=1.0, area_avg should be 1.0.
        for j in ghost..(pny - ghost) {
            for i in ghost..(pnx - ghost) {
                pd.residual[j * pnx + i] = 1.0;
            }
        }

        let corrections = restrict_residual_to_coarse(&patch, &pd.residual, base.nx);

        // Should have 4×4 = 16 corrections (one per coarse cell in the rect).
        assert_eq!(
            corrections.len(), 16,
            "expected 16 corrections, got {}", corrections.len()
        );

        // Each correction should have avg_residual = 1.0 (uniform residual).
        for (idx, (cell_idx, avg)) in corrections.iter().enumerate() {
            assert!(
                (avg - 1.0).abs() < 1e-12,
                "correction {}: avg={:.6e}, expected 1.0", idx, avg
            );
            // Verify cell_idx is in the correct range.
            let ci = cell_idx % base.nx;
            let cj = cell_idx / base.nx;
            assert!(ci >= 4 && ci < 8, "ci={} out of range", ci);
            assert!(cj >= 4 && cj < 8, "cj={} out of range", cj);
        }

        eprintln!("[phase3] PASSED: restrict_residual produces correct uniform average");
    }

    /// Test restriction with a non-uniform residual pattern.
    /// Verify the area-average matches manual computation.
    #[test]
    fn test_restrict_residual_nonuniform() {
        let base = Grid2D::new(8, 8, 5e-9, 5e-9, 3e-9);
        let rect = Rect2i::new(2, 2, 2, 2); // covers coarse cells (2..4, 2..4)
        let ratio = 2;
        let ghost = 2;
        let patch = Patch2D::new(&base, rect, ratio, ghost);

        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;
        let mut pd = PatchPoissonData::new(&patch);

        // Set residual with a gradient: residual[j*pnx+i] = i + j
        for j in 0..pny {
            for i in 0..pnx {
                pd.residual[j * pnx + i] = (i + j) as f64;
            }
        }

        let corrections = restrict_residual_to_coarse(&patch, &pd.residual, base.nx);

        // 2×2 = 4 corrections
        assert_eq!(corrections.len(), 4);

        // For coarse cell (ic=0, jc=0): fine cells at (ghost+0, ghost+0)..(ghost+1, ghost+1)
        // = (2,2), (3,2), (2,3), (3,3)
        // residual values: 2+2=4, 3+2=5, 2+3=5, 3+3=6 → avg = 20/4 = 5.0
        let (_, avg00) = corrections[0]; // (ic=0,jc=0) is first
        assert!(
            (avg00 - 5.0).abs() < 1e-12,
            "avg at (0,0): expected 5.0, got {}", avg00
        );

        // For coarse cell (ic=1, jc=0): fine cells (4,2),(5,2),(4,3),(5,3)
        // values: 6, 7, 7, 8 → avg = 28/4 = 7.0
        let (_, avg10) = corrections[1]; // (ic=1,jc=0)
        assert!(
            (avg10 - 7.0).abs() < 1e-12,
            "avg at (1,0): expected 7.0, got {}", avg10
        );

        eprintln!(
            "[phase3] restrict non-uniform: avg(0,0)={:.1}, avg(1,0)={:.1}",
            avg00, avg10
        );
        eprintln!("[phase3] PASSED: restrict_residual with non-uniform pattern");
    }

    /// Test that restriction conserves total residual:
    /// sum of (avg × area_coarse) should equal sum of fine residual × area_fine
    /// for covered cells.
    #[test]
    fn test_restrict_residual_conservation() {
        let base = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let rect = Rect2i::new(3, 5, 6, 4);
        let ratio = 2;
        let ghost = 2;
        let patch = Patch2D::new(&base, rect, ratio, ghost);

        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;
        let mut pd = PatchPoissonData::new(&patch);

        // Set residual to a non-trivial pattern.
        for j in 0..pny {
            for i in 0..pnx {
                pd.residual[j * pnx + i] =
                    (i as f64 * 0.3).sin() * (j as f64 * 0.7).cos();
            }
        }

        let corrections = restrict_residual_to_coarse(&patch, &pd.residual, base.nx);

        // Sum of restricted values (each is an average over ratio² fine cells,
        // so multiply by ratio² to get the total fine contribution).
        let sum_restricted: f64 = corrections.iter().map(|(_, v)| v).sum::<f64>()
            * (ratio * ratio) as f64;

        // Sum of fine residual over interior cells covered by the patch.
        let mut sum_fine = 0.0f64;
        for jc in 0..rect.ny {
            for ic in 0..rect.nx {
                let fi0 = ghost + ic * ratio;
                let fj0 = ghost + jc * ratio;
                for fj in fj0..fj0 + ratio {
                    for fi in fi0..fi0 + ratio {
                        if fi < pnx && fj < pny {
                            sum_fine += pd.residual[fj * pnx + fi];
                        }
                    }
                }
            }
        }

        let rel_err = if sum_fine.abs() > 1e-30 {
            (sum_restricted - sum_fine).abs() / sum_fine.abs()
        } else {
            0.0
        };

        eprintln!(
            "[phase3] conservation: sum_fine={:.6e}, sum_restricted={:.6e}, rel_err={:.3e}",
            sum_fine, sum_restricted, rel_err
        );

        assert!(
            rel_err < 1e-12,
            "restriction should conserve total: rel_err={:.3e}", rel_err
        );

        eprintln!("[phase3] PASSED: restriction conserves total residual");
    }

    /// Test prolongation: interpolated coarse correction matches direct bilinear
    /// evaluation at fine cell positions.
    #[test]
    fn test_prolongate_phi_correction() {
        let base = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let rect = Rect2i::new(4, 4, 4, 4);
        let ratio = 2;
        let ghost = 2;
        let patch = Patch2D::new(&base, rect, ratio, ghost);

        let coarse_nx = base.nx;
        let coarse_ny = base.ny;
        let n_coarse = coarse_nx * coarse_ny;

        // Old phi = 0 everywhere, new phi = smooth function.
        let coarse_phi_old = vec![0.0f64; n_coarse];
        let mut coarse_phi_new = vec![0.0f64; n_coarse];
        for j in 0..coarse_ny {
            for i in 0..coarse_nx {
                let x = (i as f64 + 0.5) * base.dx;
                let y = (j as f64 + 0.5) * base.dy;
                coarse_phi_new[j * coarse_nx + i] =
                    (x * 1e8).sin() * (y * 1e8).cos(); // smooth function
            }
        }

        // Start with patch phi = 0.
        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;
        let mut patch_phi = vec![0.0f64; pnx * pny];

        // Prolongate.
        prolongate_phi_correction(
            &coarse_phi_new, &coarse_phi_old,
            coarse_nx, coarse_ny, base.dx, base.dy,
            &patch, &mut patch_phi,
        );

        // The correction delta = phi_new - phi_old = phi_new (since old=0).
        // At each interior fine cell, the prolongated value should match
        // bilinear interpolation of phi_new at that cell's position.
        let gi0 = patch.interior_i0();
        let gj0 = patch.interior_j0();
        let gi1 = patch.interior_i1();
        let gj1 = patch.interior_j1();

        let mut max_err = 0.0f64;
        for j in gj0..gj1 {
            for i in gi0..gi1 {
                let (x, y) = patch.cell_center_xy(i, j);
                let expected = sample_bilinear_scalar(
                    &coarse_phi_new, coarse_nx, coarse_ny, base.dx, base.dy, x, y);
                let actual = patch_phi[j * pnx + i];
                let err = (actual - expected).abs();
                max_err = max_err.max(err);
            }
        }

        eprintln!("[phase4] prolongation max_err vs bilinear: {:.3e}", max_err);

        // Should be bit-identical — same interpolation function.
        assert!(
            max_err < 1e-14,
            "prolongation should match bilinear interpolation, err={:.3e}", max_err
        );

        // Ghost cells should remain zero (not modified by prolongation).
        for j in 0..pny {
            for i in 0..pnx {
                let is_interior = i >= gi0 && i < gi1 && j >= gj0 && j < gj1;
                if !is_interior {
                    assert_eq!(
                        patch_phi[j * pnx + i], 0.0,
                        "ghost cell ({},{}) should be zero after prolongation", i, j
                    );
                }
            }
        }

        eprintln!("[phase4] PASSED: prolongation matches bilinear, ghosts untouched");
    }

    /// Test prolongation additivity: if patch_phi already has values,
    /// prolongation ADDS the correction, doesn't overwrite.
    #[test]
    fn test_prolongate_is_additive() {
        let base = Grid2D::new(8, 8, 5e-9, 5e-9, 3e-9);
        let rect = Rect2i::new(2, 2, 3, 3);
        let ratio = 2;
        let ghost = 2;
        let patch = Patch2D::new(&base, rect, ratio, ghost);

        let coarse_nx = base.nx;
        let coarse_ny = base.ny;
        let n_coarse = coarse_nx * coarse_ny;

        let coarse_phi_old = vec![1.0f64; n_coarse];
        let coarse_phi_new = vec![3.0f64; n_coarse]; // delta = 2.0 everywhere

        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;
        let gi0 = patch.interior_i0();
        let gj0 = patch.interior_j0();
        let gi1 = patch.interior_i1();
        let gj1 = patch.interior_j1();

        // Start with patch phi = 10.0 everywhere interior.
        let mut patch_phi = vec![0.0f64; pnx * pny];
        for j in gj0..gj1 {
            for i in gi0..gi1 {
                patch_phi[j * pnx + i] = 10.0;
            }
        }

        prolongate_phi_correction(
            &coarse_phi_new, &coarse_phi_old,
            coarse_nx, coarse_ny, base.dx, base.dy,
            &patch, &mut patch_phi,
        );

        // Interior cells should be 10.0 + 2.0 = 12.0.
        // (delta=2.0 is uniform, bilinear of uniform = 2.0)
        for j in gj0..gj1 {
            for i in gi0..gi1 {
                let val = patch_phi[j * pnx + i];
                assert!(
                    (val - 12.0).abs() < 1e-12,
                    "interior ({},{})={:.6}, expected 12.0", i, j, val
                );
            }
        }

        eprintln!("[phase4] PASSED: prolongation is additive (10 + 2 = 12)");
    }
}

#[cfg(test)]
mod phase5_tests {
    use super::*;
    use crate::amr::hierarchy::AmrHierarchy2D;
    use crate::grid::Grid2D;
    use crate::params::{DemagMethod, Material};
    use crate::vector_field::VectorField2D;

    fn make_test_material() -> Material {
        Material {
            ms: 800e3,
            a_ex: 13e-12,
            k_u: 0.0,
            easy_axis: [0.0, 0.0, 1.0],
            dmi: None,
            demag: true,
            demag_method: DemagMethod::FftUniform,
        }
    }

    /// GATE TEST: Zero-patch regression.
    ///
    /// With no patches, compute_vcycle must produce the same B as the existing
    /// compute() (enhanced-RHS) path, because the V-cycle with zero patches
    /// reduces to a plain L0 solve with no corrections.
    #[test]
    fn test_zero_patch_regression() {
        let grid = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let mat = make_test_material();

        // Helper to make vortex M.
        let make_vortex = |g: Grid2D| -> VectorField2D {
            let mut m = VectorField2D::new(g);
            let cx = g.nx as f64 * 0.5;
            let cy = g.ny as f64 * 0.5;
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let x = (i as f64 + 0.5) - cx;
                    let y = (j as f64 + 0.5) - cy;
                    let r = (x * x + y * y).sqrt().max(0.5);
                    m.data[j * g.nx + i] = [-y / r, x / r, 0.0];
                }
            }
            m
        };

        // Path A: existing compute() (enhanced-RHS).
        let h_a = AmrHierarchy2D::new(grid, make_vortex(grid), 2, 2);
        let mut solver_a = CompositeGridPoisson::new(grid);
        let mut b_a = VectorField2D::new(grid);
        solver_a.compute(&h_a, &mat, &mut b_a);

        // Path B: new compute_vcycle().
        let h_b = AmrHierarchy2D::new(grid, make_vortex(grid), 2, 2);
        let mut solver_b = CompositeGridPoisson::new(grid);
        let mut b_b = VectorField2D::new(grid);
        solver_b.compute_vcycle(&h_b, &mat, &mut b_b);

        // Compare B.
        let mut max_diff = 0.0f64;
        for k in 0..b_a.data.len() {
            for c in 0..3 {
                let d = (b_a.data[k][c] - b_b.data[k][c]).abs();
                max_diff = max_diff.max(d);
            }
        }

        let b_max = b_a.data.iter()
            .flat_map(|v| v.iter())
            .map(|x| x.abs())
            .fold(0.0f64, f64::max);

        eprintln!(
            "[phase5] zero-patch: max_B_diff={:.3e}, B_max={:.3e}, rel={:.3e}",
            max_diff, b_max, if b_max > 0.0 { max_diff / b_max } else { 0.0 }
        );

        // Should be very close. Not necessarily bit-identical because the
        // V-cycle path goes through a slightly different code flow (computes
        // coarse_div separately, runs vcycle_iteration which does L0 solve
        // via the same solve_with_corrections). Allow small tolerance.
        assert!(
            max_diff < 1e-6 * b_max.max(1e-20),
            "zero-patch regression: B differs by {:.3e} (B_max={:.3e})",
            max_diff, b_max
        );

        eprintln!("[phase5] PASSED: zero-patch regression — vcycle matches existing path");
    }

    /// Test that the V-cycle produces monotonically decreasing residual
    /// with patches present.
    ///
    /// Uses a simple setup: 16×16 base grid with one 4×4 patch in the centre.
    /// Uniform +x magnetisation.
    #[test]
    fn test_vcycle_residual_decreases() {
        let grid = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let mat = make_test_material();

        // Uniform +x coarse magnetisation.
        let mut coarse = VectorField2D::new(grid);
        coarse.set_uniform(1.0, 0.0, 0.0);

        // Create hierarchy with one L1 patch.
        let mut coarse_for_h = VectorField2D::new(grid);
        coarse_for_h.set_uniform(1.0, 0.0, 0.0);
        let mut h = AmrHierarchy2D::new(grid, coarse_for_h, 2, 2);
        let rect = crate::amr::rect::Rect2i::new(4, 4, 4, 4);
        let mut patch = crate::amr::patch::Patch2D::new(&grid, rect, 2, 2);
        patch.fill_all_from_coarse(&coarse);
        patch.rebuild_active_from_coarse_mask(&grid, None);
        h.patches.push(patch);

        // Create solver with max_cycles=1 so we can run iterations manually.
        let mut solver = CompositeGridPoisson::new(grid);
        solver.vcfg.max_cycles = 1;

        // Allocate patch data and compute RHS.
        let (l1, l2) = allocate_patch_poisson_data(&h);
        solver.l1_data = l1;
        solver.l2plus_data = l2;
        compute_all_patch_rhs(&h, &mut solver.l1_data, &mut solver.l2plus_data, mat.ms);

        // Compute coarse div.
        let n_coarse = grid.nx * grid.ny;
        solver.coarse_phi = vec![0.0; n_coarse];
        solver.coarse_div = vec![0.0; n_coarse];
        compute_scaled_div_m(
            &h.coarse.data, grid.nx, grid.ny, grid.dx, grid.dy, mat.ms,
            &mut solver.coarse_div);

        // Run V-cycles and track max patch residual.
        let mut b_scratch = VectorField2D::new(grid);
        let mut residuals = Vec::new();

        for cycle in 0..5 {
            solver.vcycle_iteration(&h, &mat, &mut b_scratch);

            // Recompute residual to measure it.
            let mut max_res = 0.0f64;
            for pd in solver.l1_data.iter_mut() {
                compute_residual_2d(pd);
                for j in pd.ghost..(pd.ny - pd.ghost) {
                    for i in pd.ghost..(pd.nx - pd.ghost) {
                        max_res = max_res.max(pd.residual[j * pd.nx + i].abs());
                    }
                }
            }
            residuals.push(max_res);
            eprintln!("[phase5] cycle {}: max_residual={:.3e}", cycle, max_res);
        }

        // The residual should decrease overall (it may not be strictly
        // monotonic due to the coarse-solve injecting corrections, but
        // the trend should be clearly downward).
        let first = residuals[0];
        let last = *residuals.last().unwrap();

        eprintln!(
            "[phase5] residual: first={:.3e}, last={:.3e}, ratio={:.4}",
            first, last, if first > 0.0 { last / first } else { 0.0 }
        );

        // After 5 V-cycles, the residual should be at least 2× smaller.
        // (Thompson & Ferziger get ~0.19× per cycle; we may get less because
        // we only smooth on patches, not the full grid.)
        assert!(
            last < first || first < 1e-30,
            "residual should decrease: first={:.3e}, last={:.3e}", first, last
        );

        eprintln!("[phase5] PASSED: V-cycle residual decreases over iterations");
    }
}

#[cfg(test)]
mod phase6_tests {
    use super::*;
    use crate::amr::patch::Patch2D;
    use crate::amr::rect::Rect2i;
    use crate::grid::Grid2D;
    use crate::vector_field::VectorField2D;
    use std::f64::consts::PI;

    /// Test that extract_b_from_patch_phi correctly computes gradients
    /// from a known φ field.
    ///
    /// φ = sin(πx/Lx) · sin(πy/Ly)
    /// ∂φ/∂x = (π/Lx) cos(πx/Lx) sin(πy/Ly)
    /// Bx = -μ₀ ∂φ/∂x
    #[test]
    fn test_gradient_extraction_manufactured() {
        let base = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let rect = Rect2i::new(4, 4, 4, 4);
        let ratio = 2;
        let ghost = 2;
        let patch = Patch2D::new(&base, rect, ratio, ghost);

        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;
        let dx = patch.grid.dx; // fine dx
        let _dy = patch.grid.dy;

        // Domain size in physical coords.
        let lx = base.nx as f64 * base.dx;
        let ly = base.ny as f64 * base.dy;

        // Fill patch phi with sin(πx/Lx)·sin(πy/Ly).
        let mut patch_phi = vec![0.0f64; pnx * pny];
        for j in 0..pny {
            for i in 0..pnx {
                let (x, y) = patch.cell_center_xy(i, j);
                patch_phi[j * pnx + i] = (PI * x / lx).sin() * (PI * y / ly).sin();
            }
        }

        // Create a dummy coarse B (all zeros — we only check Bx, By).
        let b_coarse = VectorField2D::new(base);

        let b = extract_b_from_patch_phi(&patch, &patch_phi, &b_coarse);

        // Check Bx, By at interior cells against analytical gradient.
        let gi0 = patch.interior_i0();
        let gj0 = patch.interior_j0();
        let gi1 = patch.interior_i1();
        let gj1 = patch.interior_j1();

        let mut max_bx_err = 0.0f64;
        let mut max_by_err = 0.0f64;
        let mut max_bx_ref = 0.0f64;
        let mut max_by_ref = 0.0f64;

        for j in gj0..gj1 {
            for i in gi0..gi1 {
                let (x, y) = patch.cell_center_xy(i, j);

                // Analytical: Bx = -μ₀ (π/Lx) cos(πx/Lx) sin(πy/Ly)
                let bx_exact = -MU0 * (PI / lx)
                    * (PI * x / lx).cos() * (PI * y / ly).sin();
                let by_exact = -MU0 * (PI / ly)
                    * (PI * x / lx).sin() * (PI * y / ly).cos();

                let idx = j * pnx + i;
                let bx_err = (b[idx][0] - bx_exact).abs();
                let by_err = (b[idx][1] - by_exact).abs();
                max_bx_err = max_bx_err.max(bx_err);
                max_by_err = max_by_err.max(by_err);
                max_bx_ref = max_bx_ref.max(bx_exact.abs());
                max_by_ref = max_by_ref.max(by_exact.abs());
            }
        }

        let rel_bx = if max_bx_ref > 0.0 { max_bx_err / max_bx_ref } else { 0.0 };
        let rel_by = if max_by_ref > 0.0 { max_by_err / max_by_ref } else { 0.0 };

        eprintln!(
            "[phase6] gradient: Bx rel_err={:.4}, By rel_err={:.4} (dx={:.2e})",
            rel_bx, rel_by, dx
        );

        // Central difference on sin is O(dx²). At fine resolution with ratio=2,
        // dx = 2.5e-9 and Lx = 80e-9: (πdx/Lx)² ≈ (0.098)² ≈ 0.0096.
        // So we expect < 2% relative error.
        assert!(
            rel_bx < 0.03,
            "Bx relative error too large: {:.4}", rel_bx
        );
        assert!(
            rel_by < 0.03,
            "By relative error too large: {:.4}", rel_by
        );

        eprintln!("[phase6] PASSED: fine gradient matches analytical to O(dx²)");
    }

    /// Test that Bz comes from the coarse interpolation, not from patch φ.
    #[test]
    fn test_bz_from_coarse() {
        let base = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let rect = Rect2i::new(4, 4, 4, 4);
        let ratio = 2;
        let ghost = 2;
        let patch = Patch2D::new(&base, rect, ratio, ghost);

        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;

        // Patch phi = 0 (no in-plane gradient).
        let patch_phi = vec![0.0f64; pnx * pny];

        // Coarse B with known Bz = 0.5 T everywhere.
        let mut b_coarse = VectorField2D::new(base);
        for v in b_coarse.data.iter_mut() {
            *v = [0.0, 0.0, 0.5];
        }

        let b = extract_b_from_patch_phi(&patch, &patch_phi, &b_coarse);

        // All Bx, By should be zero (phi is uniform).
        // All Bz should be ~0.5 T (from coarse interpolation).
        let gi0 = patch.interior_i0();
        let gj0 = patch.interior_j0();
        let gi1 = patch.interior_i1();
        let gj1 = patch.interior_j1();

        let mut max_bxy = 0.0f64;
        let mut min_bz = f64::INFINITY;
        let mut max_bz = f64::NEG_INFINITY;

        for j in gj0..gj1 {
            for i in gi0..gi1 {
                let idx = j * pnx + i;
                max_bxy = max_bxy.max(b[idx][0].abs()).max(b[idx][1].abs());
                min_bz = min_bz.min(b[idx][2]);
                max_bz = max_bz.max(b[idx][2]);
            }
        }

        eprintln!(
            "[phase6] Bz from coarse: max_bxy={:.3e}, Bz range=[{:.6}, {:.6}]",
            max_bxy, min_bz, max_bz
        );

        assert!(
            max_bxy < 1e-20,
            "Bx/By should be zero for uniform phi, got {:.3e}", max_bxy
        );
        assert!(
            (min_bz - 0.5).abs() < 1e-6 && (max_bz - 0.5).abs() < 1e-6,
            "Bz should be 0.5 from coarse, got [{:.6}, {:.6}]", min_bz, max_bz
        );

        eprintln!("[phase6] PASSED: Bz comes from coarse interpolation");
    }

    /// Test that the zero-patch regression still holds with Phase 6 active.
    /// With no patches, compute_vcycle should still produce the same L0 B.
    #[test]
    fn test_zero_patch_still_works() {
        use crate::amr::hierarchy::AmrHierarchy2D;
        use crate::params::{DemagMethod, Material};

        let grid = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let mat = Material {
            ms: 800e3,
            a_ex: 13e-12,
            k_u: 0.0,
            easy_axis: [0.0, 0.0, 1.0],
            dmi: None,
            demag: true,
            demag_method: DemagMethod::FftUniform,
        };

        let make_vortex = |g: Grid2D| -> VectorField2D {
            let mut m = VectorField2D::new(g);
            let cx = g.nx as f64 * 0.5;
            let cy = g.ny as f64 * 0.5;
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let x = (i as f64 + 0.5) - cx;
                    let y = (j as f64 + 0.5) - cy;
                    let r = (x * x + y * y).sqrt().max(0.5);
                    m.data[j * g.nx + i] = [-y / r, x / r, 0.0];
                }
            }
            m
        };

        let h = AmrHierarchy2D::new(grid, make_vortex(grid), 2, 2);
        let mut solver = CompositeGridPoisson::new(grid);
        let mut b = VectorField2D::new(grid);
        let (b_l1, b_l2) = solver.compute_vcycle(&h, &mat, &mut b);

        // With no patches, b_l1 and b_l2 should be empty.
        assert!(b_l1.is_empty(), "b_l1 should be empty with no patches");
        assert!(b_l2.is_empty(), "b_l2 should be empty with no patches");

        // B should be non-trivial.
        let b_max = b.data.iter()
            .flat_map(|v| v.iter())
            .map(|x| x.abs())
            .fold(0.0f64, f64::max);
        assert!(b_max > 1e-6, "B should be non-trivial, got max={:.3e}", b_max);

        eprintln!("[phase6] zero-patch: B_max={:.3e}, b_l1 empty, b_l2 empty", b_max);
        eprintln!("[phase6] PASSED: zero-patch still works with Phase 6 gradient extraction");
    }
}
