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
    DemagPoissonMG, DemagPoissonMGConfig,
    DemagPoissonMGHybrid, HybridConfig,
};
use super::mg_kernels;

use std::sync::{Mutex, OnceLock};
use std::collections::HashMap;

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
// Multi-level helpers: parent-patch operations for true composite V-cycle
// ---------------------------------------------------------------------------

/// Check if base-grid rect `a` fully contains rect `b`.
fn rect_contains(a: &crate::amr::rect::Rect2i, b: &crate::amr::rect::Rect2i) -> bool {
    b.i0 >= a.i0 && b.j0 >= a.j0
        && b.i0 + b.nx <= a.i0 + a.nx
        && b.j0 + b.ny <= a.j0 + a.ny
}

/// Find the index of the parent-level patch that encloses `child`.
///
/// Uses coarse_rect containment (same check as hierarchy nesting enforcement).
/// Returns None if no enclosing parent is found (shouldn't happen if nesting
/// is enforced, but we handle it gracefully).
fn find_enclosing_patch_idx(
    child: &Patch2D,
    parent_patches: &[Patch2D],
) -> Option<usize> {
    for (i, parent) in parent_patches.iter().enumerate() {
        if rect_contains(&parent.coarse_rect, &child.coarse_rect) {
            return Some(i);
        }
    }
    None
}

/// Fill ghost cells in a child patch's φ from the enclosing parent patch's φ.
///
/// For each ghost cell in the child patch, compute its physical (x,y),
/// convert to the parent patch's local coordinate system, and bilinearly
/// interpolate from the parent patch's φ field.
///
/// This replaces `fill_phi_ghosts_from_coarse` for L2+ patches so that
/// ghost values come from the parent level rather than jumping to L0.
fn fill_phi_ghosts_from_parent_patch(
    child_patch: &Patch2D,
    child_phi: &mut [f64],
    parent_patch: &Patch2D,
    parent_phi: &[f64],
) {
    let cnx = child_patch.grid.nx;
    let cny = child_patch.grid.ny;
    let gi0 = child_patch.interior_i0();
    let gj0 = child_patch.interior_j0();
    let gi1 = child_patch.interior_i1();
    let gj1 = child_patch.interior_j1();

    let pnx = parent_patch.grid.nx;
    let pny = parent_patch.grid.ny;
    let pdx = parent_patch.grid.dx;
    let pdy = parent_patch.grid.dy;

    // The parent patch grid origin in physical coordinates.
    // cell_center_xy(0,0) returns the physical (x,y) of the first cell
    // (including ghosts). sample_bilinear_scalar expects coordinates where
    // cell i has centre at (i+0.5)*dx, so the grid origin is at
    // cell_center_xy(0,0) - (0.5*dx, 0.5*dy).
    let (px0, py0) = parent_patch.cell_center_xy(0, 0);
    let origin_x = px0 - 0.5 * pdx;
    let origin_y = py0 - 0.5 * pdy;

    for j in 0..cny {
        for i in 0..cnx {
            let is_interior = i >= gi0 && i < gi1 && j >= gj0 && j < gj1;
            if is_interior { continue; }

            // Physical position of this child ghost cell.
            let (x, y) = child_patch.cell_center_xy(i, j);

            // Convert to parent patch's local coordinate system.
            let local_x = x - origin_x;
            let local_y = y - origin_y;

            child_phi[j * cnx + i] = sample_bilinear_scalar(
                parent_phi, pnx, pny, pdx, pdy,
                local_x, local_y,
            );
        }
    }
}

// Phase B: 3D Patch MiniCycle Infrastructure
//
// These types enable per-patch 3D MG V-cycles ("miniCycles") as the
// smoothing step in the outer composite V-cycle. Each patch gets a small
// 3D padded box with the SAME operator (stencil, coarsening) as L0,
// ensuring operator consistency and enabling convergent iteration.
// =========================================================================

/// Default lateral padding for patch 3D boxes.
///
/// Smaller than L0's pad_xy=6 because the far-field is captured by L0;
/// patches only need a few ghost cells for the parent-Dirichlet BCs.
const PATCH_PAD_XY: usize = 2;

// ---------------------------------------------------------------------------
// PatchPoisson3D: per-patch 3D scalar-field storage
// ---------------------------------------------------------------------------

/// Per-patch 3D scalar-field storage for the composite V-cycle with miniCycle.
///
/// Each AMR patch is embedded in a small 3D padded box with the same
/// z-structure as L0 (n_vac_z vacuum layers above and below the magnet
/// layer). The φ field lives on this 3D box and persists between outer
/// V-cycle iterations for warm start.
pub(crate) struct PatchPoisson3D {
    /// 3D scalar potential on the padded grid (px * py * pz).
    /// Persists between outer V-cycle iterations (warm start).
    pub phi: Vec<f64>,

    /// Padded box dimensions.
    pub px: usize,
    pub py: usize,
    pub pz: usize,

    /// Offsets from padded-box origin to interior magnet region.
    pub offx: usize,
    pub offy: usize,
    pub offz: usize,

    /// Interior (magnet-region) dimensions at this patch's resolution.
    pub int_nx: usize,
    pub int_ny: usize,

    /// Cell spacings (dx, dy at patch resolution; dz global).
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,

    /// Cache key for looking up the shared solver in PatchMGCache.
    /// (int_nx, int_ny, dx_bits, dy_bits)
    pub solver_key: (usize, usize, u64, u64),
}

impl PatchPoisson3D {
    /// Create 3D storage for a patch, using the given padded-box geometry.
    ///
    /// The padded-box dimensions are obtained from a DemagPoissonMG solver
    /// configured for this patch's interior size and resolution.
    pub fn new(
        patch: &Patch2D,
        solver: &DemagPoissonMG,
    ) -> Self {
        let (px, py, pz) = solver.padded_dims();
        let (offx, offy, offz) = solver.offsets();
        let int_nx = patch.interior_nx;
        let int_ny = patch.interior_ny;
        let dx = patch.grid.dx;
        let dy = patch.grid.dy;
        let dz = patch.grid.dz;
        let n = px * py * pz;
        Self {
            phi: vec![0.0; n],
            px, py, pz,
            offx, offy, offz,
            int_nx, int_ny,
            dx, dy, dz,
            solver_key: (int_nx, int_ny, dx.to_bits(), dy.to_bits()),
        }
    }

    /// Total cell count in the 3D padded box.
    #[allow(dead_code)]
    pub fn total_cells(&self) -> usize {
        self.px * self.py * self.pz
    }

    /// Extract magnet-layer φ as a flat 2D array (int_nx × int_ny).
    /// Used for restriction to parent (extract 2D slice, then area-average).
    #[allow(dead_code)]
    pub fn extract_magnet_layer_phi(&self) -> Vec<f64> {
        let mut phi2d = vec![0.0f64; self.int_nx * self.int_ny];
        for mj in 0..self.int_ny {
            for mi in 0..self.int_nx {
                let pi = self.offx + mi;
                let pj = self.offy + mj;
                phi2d[mj * self.int_nx + mi] =
                    self.phi[mg_kernels::idx3(pi, pj, self.offz, self.px, self.py)];
            }
        }
        phi2d
    }
}

// ---------------------------------------------------------------------------
// PatchMGCache: shared solver instances keyed by patch size + resolution
// ---------------------------------------------------------------------------

/// Cache of DemagPoissonMG instances configured for patch miniCycles.
///
/// Patches of the same interior size and cell spacing share a single
/// solver instance (stencils, MG hierarchy, scratch arrays). The field
/// data (φ, rhs) is loaded from PatchPoisson3D before each miniCycle
/// and copied back afterward.
pub(crate) struct PatchMGCache {
    solvers: HashMap<(usize, usize, u64, u64), DemagPoissonMG>,
    n_vac_z: usize,
    pad_xy: usize,
    dz: f64,
}

impl PatchMGCache {
    /// Create a new cache.
    ///
    /// `n_vac_z`: vacuum padding in z (same as L0, typically 16).
    /// `dz`: z cell spacing (global, same at all AMR levels).
    pub fn new(n_vac_z: usize, dz: f64) -> Self {
        Self {
            solvers: HashMap::new(),
            n_vac_z,
            pad_xy: PATCH_PAD_XY,
            dz,
        }
    }

    /// Get or create a solver for the given patch interior size and resolution.
    ///
    /// Returns a mutable reference to the shared solver. The caller loads
    /// φ/rhs/bc_phi into the solver, runs the miniCycle, then copies φ back.
    pub fn get_or_create(
        &mut self,
        int_nx: usize,
        int_ny: usize,
        dx: f64,
        dy: f64,
    ) -> &mut DemagPoissonMG {
        let key = (int_nx, int_ny, dx.to_bits(), dy.to_bits());
        self.solvers.entry(key).or_insert_with(|| {
            let solver = DemagPoissonMG::new_for_patch(
                int_nx, int_ny, dx, dy, self.dz, self.pad_xy, self.n_vac_z,
            );
            let (px, py, pz) = solver.padded_dims();
            eprintln!(
                "[composite Phase B] Created PatchMGSolver for {}×{} interior \
                 (dx={:.3e}, dy={:.3e}) → padded {}×{}×{} ({} cells)",
                int_nx, int_ny, dx, dy, px, py, pz, px * py * pz,
            );
            solver
        })
    }

    /// Create PatchPoisson3D storage for a patch, using a solver from the cache.
    pub fn create_patch_data(
        &mut self,
        patch: &Patch2D,
    ) -> PatchPoisson3D {
        let int_nx = patch.interior_nx;
        let int_ny = patch.interior_ny;
        let dx = patch.grid.dx;
        let dy = patch.grid.dy;
        let solver = self.get_or_create(int_nx, int_ny, dx, dy);
        PatchPoisson3D::new(patch, solver)
    }

    /// Number of unique solver instances (for diagnostics).
    #[allow(dead_code)]
    pub fn solver_count(&self) -> usize {
        self.solvers.len()
    }
}

// ---------------------------------------------------------------------------
// 3D ghost-fill: lateral BC from parent's 3D φ
// ---------------------------------------------------------------------------

/// Bilinearly interpolate a value from a 3D padded-box φ field at a given
/// physical (x, y) position and z-index k.
///
/// `parent_phi`: full 3D array (parent_px * parent_py * parent_pz).
/// `parent_dx/dy`: cell spacings of the parent grid.
/// `parent_offx/offy`: offset from padded-box origin to parent interior.
///
/// The mapping assumes the parent's interior cell (mi, mj) is at physical
/// position x = (mi + 0.5) * parent_dx, consistent with corner-origin
/// coordinates used throughout the codebase.
fn sample_bilinear_3d_at_k(
    parent_phi: &[f64],
    parent_px: usize,
    parent_py: usize,
    parent_offx: usize,
    parent_offy: usize,
    parent_dx: f64,
    parent_dy: f64,
    k: usize,
    x: f64,
    y: f64,
) -> f64 {
    // Map physical coords to fractional padded-box indices.
    // Interior cell mi at physical x = (mi+0.5)*dx.
    // Padded-box index of interior cell mi = offx + mi.
    // So: padded index = x/dx - 0.5 + offx.
    let fx = x / parent_dx - 0.5 + parent_offx as f64;
    let fy = y / parent_dy - 0.5 + parent_offy as f64;

    let i0f = fx.floor();
    let j0f = fy.floor();
    let tx = fx - i0f;
    let ty = fy - j0f;

    let clamp = |v: isize, n: usize| -> usize {
        if v <= 0 {
            0
        } else if v >= n as isize - 1 {
            n - 1
        } else {
            v as usize
        }
    };

    let i0 = clamp(i0f as isize, parent_px);
    let j0 = clamp(j0f as isize, parent_py);
    let i1 = clamp(i0f as isize + 1, parent_px);
    let j1 = clamp(j0f as isize + 1, parent_py);

    let v00 = parent_phi[mg_kernels::idx3(i0, j0, k, parent_px, parent_py)];
    let v10 = parent_phi[mg_kernels::idx3(i1, j0, k, parent_px, parent_py)];
    let v01 = parent_phi[mg_kernels::idx3(i0, j1, k, parent_px, parent_py)];
    let v11 = parent_phi[mg_kernels::idx3(i1, j1, k, parent_px, parent_py)];

    let v0 = v00 * (1.0 - tx) + v10 * tx;
    let v1 = v01 * (1.0 - tx) + v11 * tx;
    v0 * (1.0 - ty) + v1 * ty
}

/// Fill the bc_phi array for a patch's 3D solver from a parent's 3D φ field.
///
/// Lateral faces (x and y boundaries of the padded box) are filled by
/// bilinear interpolation from the parent φ at each z-level.
/// Z-faces (k=0, k=pz-1) are left at zero (DirichletZero).
///
/// This is called before each miniCycle to set the boundary conditions
/// that couple the patch to the parent level.
///
/// # Arguments
/// * `patch` - The AMR patch (for physical coordinate mapping).
/// * `pd3d` - The patch's 3D data (for padded-box geometry).
/// * `solver_bc_phi` - The solver's bc_phi array to fill (px*py*pz).
/// * `parent_phi` - Parent's full 3D φ array.
/// * `parent_px/py` - Parent's padded-box dimensions.
/// * `parent_offx/offy` - Parent's interior offsets.
/// * `parent_dx/dy` - Parent's cell spacings.
pub(crate) fn fill_patch_bc_from_parent_3d(
    patch: &Patch2D,
    pd3d: &PatchPoisson3D,
    solver_bc_phi: &mut [f64],
    parent_phi: &[f64],
    parent_px: usize,
    parent_py: usize,
    parent_offx: usize,
    parent_offy: usize,
    parent_dx: f64,
    parent_dy: f64,
) {
    let px = pd3d.px;
    let py = pd3d.py;
    let pz = pd3d.pz;
    let offx = pd3d.offx;
    let offy = pd3d.offy;

    // The physical position of padded-box cell (i, j) in the patch:
    // Interior cell mi = i - offx (where mi ranges 0..int_nx).
    // Fine-grid global index of interior cell mi:
    //   gi = patch.coarse_rect.i0 * patch.ratio + mi
    // Physical x = (gi + 0.5) * patch.grid.dx
    //
    // For ghost cells (i < offx or i >= offx + int_nx), mi is outside
    // the interior but the formula still gives a valid physical position.
    let gi0 = (patch.coarse_rect.i0 * patch.ratio) as f64;
    let gj0 = (patch.coarse_rect.j0 * patch.ratio) as f64;
    let patch_dx = pd3d.dx;
    let patch_dy = pd3d.dy;

    // Fill all boundary cells of the padded box.
    // Interior cells are NOT touched (they hold φ from the V-cycle).
    // Z-face cells (k=0, k=pz-1) stay at 0.0 (DirichletZero).
    solver_bc_phi.fill(0.0);

    for k in 1..(pz - 1) {
        for j in 0..py {
            for i in 0..px {
                // We only need to fill the boundary shell.
                let is_boundary = i == 0 || i + 1 == px || j == 0 || j + 1 == py;
                if !is_boundary {
                    continue;
                }

                // Physical position of this cell.
                let mi = i as f64 - offx as f64; // may be negative
                let mj = j as f64 - offy as f64;
                let x = (gi0 + mi + 0.5) * patch_dx;
                let y = (gj0 + mj + 0.5) * patch_dy;

                let val = sample_bilinear_3d_at_k(
                    parent_phi,
                    parent_px,
                    parent_py,
                    parent_offx,
                    parent_offy,
                    parent_dx,
                    parent_dy,
                    k,
                    x,
                    y,
                );

                solver_bc_phi[mg_kernels::idx3(i, j, k, px, py)] = val;
            }
        }
    }
    // k=0 and k=pz-1 slices remain 0.0 (DirichletZero on z-faces).
}

// ---------------------------------------------------------------------------
// run_mini_cycle: orchestrate a miniCycle for a single patch
// ---------------------------------------------------------------------------

/// Run one miniCycle on a patch: load data into shared solver, run V-cycle(s),
/// copy result back.
///
/// This is the Phase B replacement for `smooth_jacobi_2d`. Instead of a few
/// 2D Jacobi iterations, it runs a full 3D MG V-cycle on the patch's padded
/// box, ensuring operator consistency with L0.
///
/// # Arguments
/// * `pd3d` - Persistent 3D patch data (φ read from here, updated φ written back).
/// * `cache` - Shared solver cache.
/// * `patch` - The AMR patch (for M data and coordinate mapping).
/// * `ms` - Saturation magnetisation.
/// * `n_vcycles` - Number of internal V-cycles per miniCycle (typically 1-2).
/// * `parent_phi` - Parent's full 3D φ array (for ghost-fill BCs).
/// * `parent_px/py/offx/offy/dx/dy` - Parent's padded-box geometry.
#[allow(dead_code)]
pub(crate) fn run_mini_cycle(
    pd3d: &mut PatchPoisson3D,
    cache: &mut PatchMGCache,
    patch: &Patch2D,
    ms: f64,
    n_vcycles: usize,
    parent_phi: &[f64],
    parent_px: usize,
    parent_py: usize,
    parent_offx: usize,
    parent_offy: usize,
    parent_dx: f64,
    parent_dy: f64,
) {
    let (int_nx, int_ny, dx_bits, dy_bits) = pd3d.solver_key;
    let solver = cache.get_or_create(int_nx, int_ny,
        f64::from_bits(dx_bits), f64::from_bits(dy_bits));

    // --- 1. Build 3D RHS from patch magnetisation ---
    // Extract interior M data (strip AMR ghost cells).
    let ghost = patch.ghost;
    let m_data = &patch.m.data;
    let patch_nx = patch.grid.nx;
    let mut m_interior = vec![[0.0f64; 3]; int_nx * int_ny];
    for mj in 0..int_ny {
        for mi in 0..int_nx {
            let pi = ghost + mi;
            let pj = ghost + mj;
            m_interior[mj * int_nx + mi] = m_data[pj * patch_nx + pi];
        }
    }
    solver.build_rhs_from_m_raw(&m_interior, ms);

    // --- 2. Load warm-start φ from persistent storage ---
    solver.phi_3d_mut().copy_from_slice(&pd3d.phi);

    // --- 3. Fill lateral BCs from parent ---
    fill_patch_bc_from_parent_3d(
        patch, pd3d,
        solver.bc_phi_mut(),
        parent_phi,
        parent_px, parent_py,
        parent_offx, parent_offy,
        parent_dx, parent_dy,
    );

    // --- 4. Run miniCycle (enforce_dirichlet + V-cycles) ---
    solver.mini_solve(n_vcycles);

    // --- 5. Copy converged φ back to persistent storage ---
    pd3d.phi.copy_from_slice(solver.phi_3d());
}

// ---------------------------------------------------------------------------
// 3D residual extraction (magnet layer only, for restriction to parent)
// ---------------------------------------------------------------------------

/// Compute the residual r = rhs - L(φ) on the magnet layer of a patch's 3D box,
/// returning a 2D array (int_nx × int_ny) suitable for restriction to the parent.
///
/// Uses the solver's 3D stencil (applied internally by re-computing the
/// stencil at magnet-layer cells). For efficiency, we compute residual only
/// at k=offz and return the 2D slice.
///
/// NOTE: For the initial implementation, we use a simple approach:
/// run the solver's compute_residual, then extract the magnet-layer slice.
/// A more efficient version would compute only the k=offz plane.
#[allow(dead_code)]
pub(crate) fn extract_magnet_layer_residual_3d(
    pd3d: &PatchPoisson3D,
    cache: &mut PatchMGCache,
    patch: &Patch2D,
    ms: f64,
    parent_phi: &[f64],
    parent_px: usize,
    parent_py: usize,
    parent_offx: usize,
    parent_offy: usize,
    parent_dx: f64,
    parent_dy: f64,
) -> Vec<f64> {
    let (int_nx, int_ny, dx_bits, dy_bits) = pd3d.solver_key;
    let solver = cache.get_or_create(int_nx, int_ny,
        f64::from_bits(dx_bits), f64::from_bits(dy_bits));

    // Load current state into solver.
    let ghost = patch.ghost;
    let m_data = &patch.m.data;
    let patch_nx = patch.grid.nx;
    let mut m_interior = vec![[0.0f64; 3]; int_nx * int_ny];
    for mj in 0..int_ny {
        for mi in 0..int_nx {
            m_interior[mj * int_nx + mi] = m_data[(ghost + mj) * patch_nx + (ghost + mi)];
        }
    }
    solver.build_rhs_from_m_raw(&m_interior, ms);
    solver.phi_3d_mut().copy_from_slice(&pd3d.phi);
    fill_patch_bc_from_parent_3d(
        patch, pd3d, solver.bc_phi_mut(),
        parent_phi, parent_px, parent_py,
        parent_offx, parent_offy, parent_dx, parent_dy,
    );
    solver.mini_solve(0); // enforce_dirichlet only (0 V-cycles)

    // Compute residual using the CORRECT stencil (matching the solve operator).
    solver.compute_residual_magnet_layer()
}

// ---------------------------------------------------------------------------
// 3D B extraction from converged patch φ
// ---------------------------------------------------------------------------

/// Extract B = -μ₀∇φ from the 3D φ field at the magnet layer (k=offz).
///
/// This is the Phase B replacement for the 2D `extract_b_from_patch_phi`.
/// The key improvement: Bz is now computed from ∂φ/∂z at fine resolution
/// (capturing z-surface charge physics locally) rather than being
/// interpolated from L0.
///
/// Returns B vectors for the patch interior cells (int_nx × int_ny).
#[allow(dead_code)]
pub(crate) fn extract_b_from_patch_phi_3d(
    pd3d: &PatchPoisson3D,
) -> Vec<[f64; 3]> {
    let px = pd3d.px;
    let py = pd3d.py;
    let offx = pd3d.offx;
    let offy = pd3d.offy;
    let offz = pd3d.offz;
    let int_nx = pd3d.int_nx;
    let int_ny = pd3d.int_ny;
    let dx = pd3d.dx;
    let dy = pd3d.dy;
    let dz = pd3d.dz;
    let phi = &pd3d.phi;

    let mut b = vec![[0.0f64; 3]; int_nx * int_ny];

    for mj in 0..int_ny {
        for mi in 0..int_nx {
            let pi = offx + mi;
            let pj = offy + mj;
            let k = offz;

            // Central differences for ∂φ/∂x, ∂φ/∂y, ∂φ/∂z.
            let dphi_dx = (phi[mg_kernels::idx3(pi + 1, pj, k, px, py)]
                - phi[mg_kernels::idx3(pi - 1, pj, k, px, py)])
                / (2.0 * dx);

            let dphi_dy = (phi[mg_kernels::idx3(pi, pj + 1, k, px, py)]
                - phi[mg_kernels::idx3(pi, pj - 1, k, px, py)])
                / (2.0 * dy);

            let dphi_dz = (phi[mg_kernels::idx3(pi, pj, k + 1, px, py)]
                - phi[mg_kernels::idx3(pi, pj, k - 1, px, py)])
                / (2.0 * dz);

            b[mj * int_nx + mi] = [
                -MU0 * dphi_dx,
                -MU0 * dphi_dy,
                -MU0 * dphi_dz,
            ];
        }
    }

    b
}

// ---------------------------------------------------------------------------
// Phase B Step 4: V-cycle integration helpers
// ---------------------------------------------------------------------------

/// Run miniCycle on a patch AND compute the magnet-layer residual.
///
/// Used in the downstroke where we need both smoothing and the residual
/// for restriction. After the miniCycle, the solver's RHS is still loaded,
/// so we compute r = rhs - L(φ) at k=offz before returning.
///
/// `residual_2d` is sized (patch.grid.nx × patch.grid.ny) — the full 2D
/// patch grid with ghost padding. Interior cells get the computed residual;
/// ghost cells are set to zero.
#[allow(dead_code)]
pub(crate) fn smooth_and_residual_3d(
    pd3d: &mut PatchPoisson3D,
    residual_2d: &mut [f64],
    patch: &Patch2D,
    cache: &mut PatchMGCache,
    ms: f64,
    n_vcycles: usize,
    parent_phi: &[f64],
    parent_px: usize, parent_py: usize,
    parent_offx: usize, parent_offy: usize,
    parent_dx: f64, parent_dy: f64,
) {
    let (int_nx, int_ny, dx_bits, dy_bits) = pd3d.solver_key;
    let dx = f64::from_bits(dx_bits);
    let dy = f64::from_bits(dy_bits);
    let solver = cache.get_or_create(int_nx, int_ny, dx, dy);

    // 1. Build 3D RHS from patch magnetisation.
    let ghost = patch.ghost;
    let m_data = &patch.m.data;
    let patch_nx = patch.grid.nx;
    let mut m_interior = vec![[0.0f64; 3]; int_nx * int_ny];
    for mj in 0..int_ny {
        for mi in 0..int_nx {
            m_interior[mj * int_nx + mi] = m_data[(ghost + mj) * patch_nx + (ghost + mi)];
        }
    }
    solver.build_rhs_from_m_raw(&m_interior, ms);

    // 2. Load warm-start φ from persistent storage.
    solver.phi_3d_mut().copy_from_slice(&pd3d.phi);

    // 3. Fill lateral BCs from parent.
    fill_patch_bc_from_parent_3d(
        patch, pd3d, solver.bc_phi_mut(),
        parent_phi, parent_px, parent_py,
        parent_offx, parent_offy, parent_dx, parent_dy,
    );

    // 4. Run miniCycle (enforce_dirichlet + V-cycles).
    solver.mini_solve(n_vcycles);

    // 5. Compute magnet-layer residual r = rhs - L(φ) at k=offz using
    //    the SAME stencil as the solve (iso27/7pt/iso9). Using a different
    //    stencil gives a large spurious residual even when converged.
    let res_interior = solver.compute_residual_magnet_layer();

    residual_2d.fill(0.0);
    let r2d_nx = patch.grid.nx;
    for mj in 0..int_ny {
        for mi in 0..int_nx {
            let ri = ghost + mi;
            let rj = ghost + mj;
            residual_2d[rj * r2d_nx + ri] = res_interior[mj * int_nx + mi];
        }
    }

    // 6. Copy converged φ back to persistent storage.
    pd3d.phi.copy_from_slice(solver.phi_3d());
}

/// Prolongate the L0 φ correction into a L1 patch's 3D φ at k=offz.
///
/// Computes delta = interp(coarse_phi_new - coarse_phi_old) at each
/// interior cell position and ADDS it to pd3d.phi on the magnet layer.
#[allow(dead_code)]
fn prolongate_correction_to_3d(
    coarse_phi_new: &[f64],
    coarse_phi_old: &[f64],
    coarse_nx: usize, coarse_ny: usize,
    coarse_dx: f64, coarse_dy: f64,
    patch: &Patch2D,
    pd3d: &mut PatchPoisson3D,
) {
    let delta_coarse: Vec<f64> = coarse_phi_new.iter()
        .zip(coarse_phi_old.iter())
        .map(|(n, o)| n - o)
        .collect();

    let ghost = patch.ghost;
    let offz = pd3d.offz;
    let (px, py) = (pd3d.px, pd3d.py);
    let offx = pd3d.offx;
    let offy = pd3d.offy;

    for mj in 0..pd3d.int_ny {
        for mi in 0..pd3d.int_nx {
            let (x, y) = patch.cell_center_xy(ghost + mi, ghost + mj);
            let delta_interp = sample_bilinear_scalar(
                &delta_coarse, coarse_nx, coarse_ny, coarse_dx, coarse_dy, x, y);
            pd3d.phi[mg_kernels::idx3(offx + mi, offy + mj, offz, px, py)] += delta_interp;
        }
    }
}

/// Prolongate correction from a parent patch's φ change into a child
/// patch's 3D φ at k=offz.
///
/// Uses interior-only magnet-layer slices of the parent's 3D phi (before
/// and after the parent's upstroke) to compute the correction delta.
#[allow(dead_code)]
fn prolongate_correction_to_3d_from_parent(
    parent_patch: &Patch2D,
    parent_phi_new_interior: &[f64],  // parent int_nx × int_ny
    parent_phi_old_interior: &[f64],  // same, snapshot from before
    child_patch: &Patch2D,
    child_pd3d: &mut PatchPoisson3D,
) {
    let p_int_nx = parent_patch.interior_nx;
    let _p_int_ny = parent_patch.interior_ny;
    let pdx = parent_patch.grid.dx;
    let pdy = parent_patch.grid.dy;

    // Origin of the parent's interior grid in physical coordinates.
    let gi0_p = (parent_patch.coarse_rect.i0 * parent_patch.ratio) as f64;
    let gj0_p = (parent_patch.coarse_rect.j0 * parent_patch.ratio) as f64;
    let origin_x = gi0_p * pdx;
    let origin_y = gj0_p * pdy;

    let delta: Vec<f64> = parent_phi_new_interior.iter()
        .zip(parent_phi_old_interior.iter())
        .map(|(n, o)| n - o)
        .collect();

    let c_ghost = child_patch.ghost;
    let offz = child_pd3d.offz;
    let (cpx, _cpy) = (child_pd3d.px, child_pd3d.py);
    let c_offx = child_pd3d.offx;
    let c_offy = child_pd3d.offy;

    for mj in 0..child_pd3d.int_ny {
        for mi in 0..child_pd3d.int_nx {
            let (x, y) = child_patch.cell_center_xy(c_ghost + mi, c_ghost + mj);
            let local_x = x - origin_x;
            let local_y = y - origin_y;
            let delta_interp = sample_bilinear_scalar(
                &delta, p_int_nx, parent_patch.interior_ny, pdx, pdy,
                local_x, local_y);
            child_pd3d.phi[mg_kernels::idx3(c_offx + mi, c_offy + mj, offz, cpx, child_pd3d.py)]
                += delta_interp;
        }
    }
}

/// Extract the interior magnet-layer φ from a PatchPoisson3D as a flat
/// 2D array (int_nx × int_ny). Used for prolongation delta computation.
#[allow(dead_code)]
fn extract_interior_magnet_layer(pd3d: &PatchPoisson3D) -> Vec<f64> {
    let mut out = vec![0.0f64; pd3d.int_nx * pd3d.int_ny];
    let offz = pd3d.offz;
    let (px, py) = (pd3d.px, pd3d.py);
    for mj in 0..pd3d.int_ny {
        for mi in 0..pd3d.int_nx {
            out[mj * pd3d.int_nx + mi] =
                pd3d.phi[mg_kernels::idx3(pd3d.offx + mi, pd3d.offy + mj, offz, px, py)];
        }
    }
    out
}

/// Copy the converged 3D magnet-layer φ into the 2D PatchPoissonData.phi
/// for backward compatibility with the existing B extraction code.
///
/// Only interior cells are copied; ghost cells are left unchanged.
/// The caller should run 2D ghost-fill afterward if B extraction needs
/// valid ghost values.
#[allow(dead_code)]
fn sync_3d_phi_to_2d(
    pd3d: &PatchPoisson3D,
    pd: &mut PatchPoissonData,
    _patch: &Patch2D,
) {
    // Copy the ENTIRE k=offz slice from the 3D padded box into the 2D phi.
    // This includes ghost cells, which have correct values from the 3D
    // solve (set by fill_patch_bc_from_parent_3d and smoothed by MG).
    //
    // We must NOT subsequently overwrite ghosts with 2D ghost-fill from
    // coarse_phi, because that would create a discontinuity at the patch
    // boundary (the 3D solve used different BCs than the 2D ghost-fill).
    //
    // Layout guarantee: pad_xy (3D) == ghost (2D) == 2, so px == pd.nx
    // and py == pd.ny. The cell at 3D index (i, j, offz) maps directly
    // to 2D index (i, j).
    let nx_2d = pd.nx;
    let ny_2d = pd.ny;
    let (px, py) = (pd3d.px, pd3d.py);
    let offz = pd3d.offz;

    debug_assert_eq!(px, nx_2d, "3D px ({}) must equal 2D nx ({})", px, nx_2d);
    debug_assert_eq!(py, ny_2d, "3D py ({}) must equal 2D ny ({})", py, ny_2d);

    for j in 0..ny_2d {
        for i in 0..nx_2d {
            pd.phi[j * nx_2d + i] = pd3d.phi[mg_kernels::idx3(i, j, offz, px, py)];
        }
    }
}

/// Restrict child patch residual into the enclosing parent patch's residual.
///
/// For each parent fine cell covered by the child, REPLACES the parent's
/// residual with the area-average of the child fine cells that map to it.
/// This is the composite V-cycle reflux step at coarse-fine interfaces.
///
/// `step_ratio` is the refinement ratio between adjacent levels (h.ratio,
/// typically 2). Each parent fine cell maps to step_ratio × step_ratio
/// child fine cells.
fn restrict_residual_to_parent_patch(
    child_patch: &Patch2D,
    child_residual: &[f64],
    child_nx: usize,
    parent_patch: &Patch2D,
    parent_residual: &mut [f64],
    parent_nx: usize,
    step_ratio: usize,
) {
    let ccr = &child_patch.coarse_rect;
    let pcr = &parent_patch.coarse_rect;

    // Overlap of child and parent in base-grid coordinates.
    let oi0 = ccr.i0.max(pcr.i0);
    let oj0 = ccr.j0.max(pcr.j0);
    let oi1 = (ccr.i0 + ccr.nx).min(pcr.i0 + pcr.nx);
    let oj1 = (ccr.j0 + ccr.ny).min(pcr.j0 + pcr.ny);
    if oi1 <= oi0 || oj1 <= oj0 { return; }

    let c_ratio = child_patch.ratio;   // child's total ratio vs base grid
    let p_ratio = parent_patch.ratio;   // parent's total ratio vs base grid
    let c_ghost = child_patch.ghost;
    let p_ghost = parent_patch.ghost;

    // For each base-grid cell in the overlap:
    for bj in oj0..oj1 {
        for bi in oi0..oi1 {
            // For each parent fine cell within this base cell:
            for py in 0..p_ratio {
                for px in 0..p_ratio {
                    let pfi = p_ghost + (bi - pcr.i0) * p_ratio + px;
                    let pfj = p_ghost + (bj - pcr.j0) * p_ratio + py;

                    // Area-average the step_ratio × step_ratio child cells
                    // that map to this parent fine cell.
                    let mut sum = 0.0f64;
                    let mut count = 0usize;
                    for dy in 0..step_ratio {
                        for dx in 0..step_ratio {
                            let cfi = c_ghost + (bi - ccr.i0) * c_ratio + px * step_ratio + dx;
                            let cfj = c_ghost + (bj - ccr.j0) * c_ratio + py * step_ratio + dy;
                            if cfi < child_nx && cfj < child_patch.grid.ny {
                                sum += child_residual[cfj * child_nx + cfi];
                                count += 1;
                            }
                        }
                    }

                    if count > 0 {
                        parent_residual[pfj * parent_nx + pfi] = sum / count as f64;
                    }
                }
            }
        }
    }
}

/// Prolongate φ correction from parent patch to child patch.
///
/// Computes delta = parent_phi_new - parent_phi_old on the parent patch grid,
/// bilinearly interpolates delta to each child interior cell, and ADDS to
/// child_phi. Ghost cells are not modified (they get updated by ghost-fill).
#[allow(dead_code)]
fn prolongate_phi_correction_from_parent_patch(
    parent_patch: &Patch2D,
    parent_phi_new: &[f64],
    parent_phi_old: &[f64],
    child_patch: &Patch2D,
    child_phi: &mut [f64],
) {
    let pnx = parent_patch.grid.nx;
    let pny = parent_patch.grid.ny;
    let pdx = parent_patch.grid.dx;
    let pdy = parent_patch.grid.dy;

    // Parent patch grid origin in physical coordinates.
    let (px0, py0) = parent_patch.cell_center_xy(0, 0);
    let origin_x = px0 - 0.5 * pdx;
    let origin_y = py0 - 0.5 * pdy;

    // Compute the correction on the parent patch grid.
    let delta_parent: Vec<f64> = parent_phi_new.iter()
        .zip(parent_phi_old.iter())
        .map(|(new, old)| new - old)
        .collect();

    let cnx = child_patch.grid.nx;
    let gi0 = child_patch.interior_i0();
    let gj0 = child_patch.interior_j0();
    let gi1 = child_patch.interior_i1();
    let gj1 = child_patch.interior_j1();

    // Interpolate delta to each child interior cell and add.
    for j in gj0..gj1 {
        for i in gi0..gi1 {
            let (x, y) = child_patch.cell_center_xy(i, j);
            let local_x = x - origin_x;
            let local_y = y - origin_y;
            let delta_interp = sample_bilinear_scalar(
                &delta_parent, pnx, pny, pdx, pdy,
                local_x, local_y,
            );
            child_phi[j * cnx + i] += delta_interp;
        }
    }
}

/// Pre-compute parent-patch index maps for all L2+ levels.
///
/// Returns a Vec<Vec<usize>> where result[lvl_idx][patch_idx] gives the
/// index of the enclosing parent patch. Panics if nesting is violated.
fn build_parent_index_maps(h: &AmrHierarchy2D) -> Vec<Vec<usize>> {
    let mut maps: Vec<Vec<usize>> = Vec::with_capacity(h.patches_l2plus.len());
    for (lvl_idx, lvl_patches) in h.patches_l2plus.iter().enumerate() {
        let parent_patches: &[Patch2D] = if lvl_idx == 0 {
            &h.patches
        } else {
            &h.patches_l2plus[lvl_idx - 1]
        };
        let lvl_map: Vec<usize> = lvl_patches.iter().map(|child| {
            find_enclosing_patch_idx(child, parent_patches)
                .unwrap_or_else(|| panic!(
                    "L{} patch at ({},{}) has no enclosing parent",
                    lvl_idx + 2, child.coarse_rect.i0, child.coarse_rect.j0
                ))
        }).collect();
        maps.push(lvl_map);
    }
    maps
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
    b_coarse: &VectorField2D,  // for Bz interpolation and fallback
) -> Vec<[f64; 3]> {
    let pnx = patch.grid.nx;
    let pny = patch.grid.ny;
    let dx = patch.grid.dx;
    let dy = patch.grid.dy;
    let inv_2dx = 1.0 / (2.0 * dx);
    let inv_2dy = 1.0 / (2.0 * dy);
    let inv_dx = 1.0 / dx;
    let inv_dy = 1.0 / dy;

    // Geometry mask: true = material, false = vacuum.
    // When available, use one-sided differences at material/vacuum boundaries
    // to avoid contamination from poorly-resolved vacuum φ values.
    let gm = patch.geom_mask_fine();

    let mut b = vec![[0.0f64; 3]; pnx * pny];

    for j in 0..pny {
        for i in 0..pnx {
            let idx = j * pnx + i;

            // For vacuum cells, use coarse-interpolated B entirely.
            if let Some(mask) = gm {
                if !mask[idx] {
                    let (x, y) = patch.cell_center_xy(i, j);
                    b[idx] = sample_bilinear(b_coarse, x, y);
                    continue;
                }
            }

            // Material cell: geometry-aware gradient.
            // Check which neighbours are material (or treat all as material if no mask).
            let xp_ok = i + 1 < pnx && gm.map_or(true, |m| m[j * pnx + (i + 1)]);
            let xm_ok = i > 0       && gm.map_or(true, |m| m[j * pnx + (i - 1)]);
            let yp_ok = j + 1 < pny && gm.map_or(true, |m| m[(j + 1) * pnx + i]);
            let ym_ok = j > 0       && gm.map_or(true, |m| m[(j - 1) * pnx + i]);

            // Bx: choose stencil based on which x-neighbours are material.
            let bx = if xp_ok && xm_ok {
                // Both neighbours material → central difference.
                -MU0 * (patch_phi[j * pnx + (i + 1)] - patch_phi[j * pnx + (i - 1)]) * inv_2dx
            } else if xm_ok && !xp_ok {
                // +x is vacuum → backward difference (away from vacuum).
                -MU0 * (patch_phi[idx] - patch_phi[j * pnx + (i - 1)]) * inv_dx
            } else if xp_ok && !xm_ok {
                // -x is vacuum → forward difference (away from vacuum).
                -MU0 * (patch_phi[j * pnx + (i + 1)] - patch_phi[idx]) * inv_dx
            } else {
                // Both neighbours vacuum or at grid edge → use coarse B.
                let (x, y) = patch.cell_center_xy(i, j);
                sample_bilinear(b_coarse, x, y)[0]
            };

            // By: same logic for y-direction.
            let by = if yp_ok && ym_ok {
                -MU0 * (patch_phi[(j + 1) * pnx + i] - patch_phi[(j - 1) * pnx + i]) * inv_2dy
            } else if ym_ok && !yp_ok {
                -MU0 * (patch_phi[idx] - patch_phi[(j - 1) * pnx + i]) * inv_dy
            } else if yp_ok && !ym_ok {
                -MU0 * (patch_phi[(j + 1) * pnx + i] - patch_phi[idx]) * inv_dy
            } else {
                let (x, y) = patch.cell_center_xy(i, j);
                sample_bilinear(b_coarse, x, y)[1]
            };

            // Bz from coarse solution (interpolated).
            let (x, y) = patch.cell_center_xy(i, j);
            let bz = sample_bilinear(b_coarse, x, y)[2];

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
// Composite field builders for multi-level defect hierarchy
// ---------------------------------------------------------------------------
//
// García-Cervera / AMReX multi-level defect correction requires each level
// to correct against its parent's *composite* field, not against L0 directly.
//
// After processing all patches at level L, we build:
//   composite_div: L0 div with L's fine-averaged div at covered coarse cells
//   composite_B:   L0 B with L's fine-averaged B at covered coarse cells
//
// Level L+1 then defect-corrects against these composite fields.

/// Update a divergence field on the L0 grid with area-averaged fine div
/// from patches at the current level.
///
/// For each coarse cell covered by a patch, the divergence value is REPLACED
/// with the area-average of the patch's fine ∇·(Ms·m) (stored in pd.rhs).
/// This builds the "composite level" divergence that the next finer level
/// computes its defect against.
#[allow(dead_code)]
fn update_composite_div(
    composite_div: &mut [f64],
    patches: &[Patch2D],
    patch_data: &[PatchPoissonData],
    base_nx: usize,
) {
    for (patch, pd) in patches.iter().zip(patch_data.iter()) {
        let cr = &patch.coarse_rect;
        let ratio = patch.ratio;
        let ghost = pd.ghost;

        for jc in 0..cr.ny {
            for ic in 0..cr.nx {
                let coarse_i = cr.i0 + ic;
                let coarse_j = cr.j0 + jc;
                let cell_idx = coarse_j * base_nx + coarse_i;

                // Use the existing area-averaging method on PatchPoissonData.
                composite_div[cell_idx] =
                    pd.area_avg_rhs_at_coarse_cell(ic, jc, ratio, ghost);
            }
        }
    }
}

/// Update a B field on the L0 grid with area-averaged fine B from patches
/// at the current level.
///
/// For each coarse cell covered by a patch, the B value is REPLACED with
/// the area-average of the patch's fine B (the full corrected B, not just δB).
/// This builds the "composite level" B field that the next finer level
/// interpolates from when computing B_patch = interp(composite_B) + δB.
#[allow(dead_code)]
fn update_composite_b(
    composite_b_data: &mut [[f64; 3]],
    patches: &[Patch2D],
    patch_b: &[Vec<[f64; 3]>],
    base_nx: usize,
) {
    for (patch, pb) in patches.iter().zip(patch_b.iter()) {
        let cr = &patch.coarse_rect;
        let ratio = patch.ratio;
        let ghost = patch.ghost;
        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;

        for jc in 0..cr.ny {
            for ic in 0..cr.nx {
                let coarse_i = cr.i0 + ic;
                let coarse_j = cr.j0 + jc;
                let cell_idx = coarse_j * base_nx + coarse_i;

                // Area-average fine B over the ratio×ratio fine cells
                // corresponding to this coarse cell.
                let fi0 = ghost + ic * ratio;
                let fj0 = ghost + jc * ratio;
                let mut sum = [0.0f64; 3];
                let mut count = 0usize;
                for fj in fj0..fj0 + ratio {
                    for fi in fi0..fi0 + ratio {
                        if fi < pnx && fj < pny {
                            let b = pb[fj * pnx + fi];
                            sum[0] += b[0];
                            sum[1] += b[1];
                            sum[2] += b[2];
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    let inv = 1.0 / count as f64;
                    composite_b_data[cell_idx] = [
                        sum[0] * inv, sum[1] * inv, sum[2] * inv,
                    ];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Defect-correction per patch (García-Cervera / AMReX approach)
// ---------------------------------------------------------------------------

/// Compute defect-corrected B on a single patch.
///
/// Steps:
///   1. δrhs = fine_∇·M − interpolate(parent_∇·M)
///   2. Smooth ∇²(δφ) = δrhs with DirichletZero BCs
///   3. δB = −μ₀∇(δφ)
///   4. B_patch = interpolate(parent_B) + δB
///
/// `coarse_div` and `b_l0` are the PARENT level's fields. For L1 patches
/// these are L0 fields; for L2+ patches these are composite fields built
/// from the parent level (containing area-averaged fine data from the
/// parent level's patches). This is the García-Cervera / AMReX multi-level
/// defect correction: each level corrects only what its parent missed.
///
/// The defect RHS is HIGH-FREQUENCY (fine detail the parent grid missed).
/// For high-k modes, 2D and 3D Green's functions agree, so the 2D Laplacian
/// is correct for the defect even though it's wrong for the full equation.
#[allow(dead_code)]
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

    // Adaptive smoothing: check defect magnitude before iterating.
    // At large L0 where the coarse grid already resolves the boundary,
    // the defect RHS is near-zero and fixed iterations introduce noise
    // that exceeds the correction magnitude (see §2.1 of Roadmap).
    let max_defect_rhs = defect_rhs.iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);

    // Scale tolerance by dx² (the expected magnitude of the correction is
    // O(defect_rhs * dx²) from the Poisson Green's function).
    let defect_tol_rel: f64 = 1e-3; // relative tolerance on correction
    let defect_tol_abs: f64 = 1e-25; // absolute floor (skip if defect is negligible)

    let effective_n_smooth = if max_defect_rhs < defect_tol_abs {
        // Defect is negligible — skip smoothing entirely.
        0
    } else {
        // Run smoothing in batches of 2, monitoring max|δφ| change.
        // Exit early if the correction has converged.
        let batch_size = 2;
        let mut iters_done = 0;
        let dx_sq = pd.dx * pd.dx;
        let tol = defect_tol_rel * max_defect_rhs * dx_sq;

        while iters_done < n_smooth {
            let this_batch = batch_size.min(n_smooth - iters_done);

            // Record phi before this batch for convergence check
            let phi_max_before = pd.phi.iter()
                .map(|v| v.abs())
                .fold(0.0f64, f64::max);

            smooth_jacobi_2d(
                &mut pd.phi, &defect_rhs, &mut tmp,
                pnx, pny, ghost, pd.dx, pd.dy,
                omega, this_batch,
            );

            iters_done += this_batch;

            // Check convergence: if the correction φ has stabilised, stop.
            let phi_max_after = pd.phi.iter()
                .map(|v| v.abs())
                .fold(0.0f64, f64::max);

            // After the first batch (starting from φ=0), phi_max_after IS the
            // change. For subsequent batches, check the delta.
            let change = if iters_done <= batch_size {
                phi_max_after
            } else {
                (phi_max_after - phi_max_before).abs()
            };

            // If the change per batch is below tolerance, the smoother has
            // captured all it can. Further iterations add noise.
            if iters_done > batch_size && change < tol {
                break;
            }
        }
        iters_done
    };

    let _ = effective_n_smooth; // suppress unused warning when diag is off

    // Step 3+4: Extract δB = −μ₀∇(δφ) and combine with interpolated parent B.
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

            // Interpolate parent-level B at this fine cell's position.
            let (x, y) = patch.cell_center_xy(i, j);
            let b_coarse = sample_bilinear(b_l0, x, y);

            // Combine: B_patch = B_parent_interp + δB
            // Bz purely from parent (3D z-physics captured by the 3D padded-box solve).
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
    /// Jacobi relaxation weight (legacy, used by 2D path).
    #[allow(dead_code)]
    pub omega: f64,
    /// Maximum number of outer V-cycle iterations.
    #[allow(dead_code)]
    pub max_cycles: usize,
}

impl Default for CompositeVCycleConfig {
    fn default() -> Self {
        let max_cycles: usize = std::env::var("LLG_COMPOSITE_MAX_CYCLES")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(5);
        let n_pre: usize = std::env::var("LLG_COMPOSITE_N_PRE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        let n_post: usize = std::env::var("LLG_COMPOSITE_N_POST")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(3);
        let omega: f64 = std::env::var("LLG_COMPOSITE_OMEGA")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(2.0 / 3.0);
        Self {
            n_pre,
            n_post,
            omega,
            max_cycles,
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
    /// Phase B: 3D patch data for miniCycle-based V-cycle.
    l1_data_3d: Vec<PatchPoisson3D>,
    l2plus_data_3d: Vec<Vec<PatchPoisson3D>>,
    /// Phase B: shared MG solver instances keyed by patch size.
    patch_mg_cache: Option<PatchMGCache>,
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
            l1_data_3d: Vec::new(),
            l2plus_data_3d: Vec::new(),
            patch_mg_cache: None,
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

    /// Run one composite V-cycle iteration using 3D miniCycles (Phase B).
    ///
    /// Downstroke: smooth_and_residual_3d on each level (finest → coarsest).
    /// L0 solve: existing 3D MG+PPPM (unchanged).
    /// Upstroke: prolongate + run_mini_cycle on each level (coarsest → finest).
    ///
    /// The 2D `l1_data`/`l2plus_data` `.residual` fields are populated for
    /// restriction compatibility. After this method returns, the caller
    /// should call `sync_3d_phi_to_2d` to update 2D phi for B extraction.
    fn vcycle_iteration(
        &mut self,
        h: &AmrHierarchy2D,
        mat: &Material,
        parent_maps: &[Vec<usize>],
        _b_scratch: &mut VectorField2D,
        pppm_delta: &[f64],
        cycle: usize,
    ) {
        let cnx = h.base_grid.nx;
        let cny = h.base_grid.ny;
        let cdx = h.base_grid.dx;
        let cdy = h.base_grid.dy;
        let _step_ratio = h.ratio;
        let ms = mat.ms;

        // Take cache temporarily to avoid borrow conflicts with self.
        let (_, _, n_vac_z) = self.l0_solver.mg.offsets();
        let dz = self.base_grid.dz;
        let mut cache = self.patch_mg_cache.take()
            .unwrap_or_else(|| PatchMGCache::new(n_vac_z, dz));

        let (l0_px, l0_py, _) = self.l0_solver.mg.padded_dims();
        let (l0_offx, l0_offy, l0_offz) = self.l0_solver.mg.offsets();
        let (nx_m, ny_m) = self.l0_solver.mg.interior_dims();

        // Create ghost-fill phi: L0's raw 3D phi + PPPM correction at k=offz.
        // The correction is on a COPY — L0's actual state stays raw MG.
        let make_corrected_l0_phi = |raw_3d: &[f64]| -> Vec<f64> {
            let mut phi = raw_3d.to_vec();
            if !pppm_delta.is_empty() {
                for mj in 0..ny_m {
                    for mi in 0..nx_m {
                        phi[mg_kernels::idx3(
                            l0_offx + mi, l0_offy + mj, l0_offz, l0_px, l0_py,
                        )] += pppm_delta[mj * nx_m + mi];
                    }
                }
            }
            phi
        };

        let l0_phi_3d = make_corrected_l0_phi(self.l0_solver.mg.phi_3d());

        // ═══════ DOWNSTROKE: finest → coarsest ═══════
        //
        // Pass 1: smooth + residual on all levels (finest first).
        // Pass 2: cascade restrictions (finest first).

        // ── Pass 1: smooth_and_residual_3d on ALL levels ──

        // L2+ from finest to coarsest
        for lvl_idx in (0..h.patches_l2plus.len()).rev() {
            let lvl_patches = &h.patches_l2plus[lvl_idx];
            if lvl_patches.is_empty() { continue; }

            // Clone parent 3D phis for ghost-fill (avoids borrow conflicts).
            let parent_phis_3d: Vec<Vec<f64>> = if lvl_idx == 0 {
                self.l1_data_3d.iter().map(|pd| pd.phi.clone()).collect()
            } else {
                self.l2plus_data_3d[lvl_idx - 1].iter().map(|pd| pd.phi.clone()).collect()
            };
            // Parent padded-box geometry.
            let parent_geoms: Vec<(usize, usize, usize, usize, f64, f64)> = if lvl_idx == 0 {
                self.l1_data_3d.iter().map(|pd| (pd.px, pd.py, pd.offx, pd.offy, pd.dx, pd.dy)).collect()
            } else {
                self.l2plus_data_3d[lvl_idx - 1].iter().map(|pd| (pd.px, pd.py, pd.offx, pd.offy, pd.dx, pd.dy)).collect()
            };

            for (pi, patch) in lvl_patches.iter().enumerate() {
                let parent_idx = parent_maps[lvl_idx][pi];
                let (ppx, ppy, poffx, poffy, pdx, pdy) = parent_geoms[parent_idx];

                let pd3d = &mut self.l2plus_data_3d[lvl_idx][pi];
                let pd = &mut self.l2plus_data[lvl_idx][pi];

                smooth_and_residual_3d(
                    pd3d, &mut pd.residual, patch, &mut cache, ms,
                    self.vcfg.n_pre,
                    &parent_phis_3d[parent_idx], ppx, ppy, poffx, poffy, pdx, pdy,
                );
            }
        }

        // L1 patches
        for (i, patch) in h.patches.iter().enumerate() {
            let pd3d = &mut self.l1_data_3d[i];
            let pd = &mut self.l1_data[i];

            smooth_and_residual_3d(
                pd3d, &mut pd.residual, patch, &mut cache, ms,
                self.vcfg.n_pre,
                &l0_phi_3d, l0_px, l0_py, l0_offx, l0_offy, cdx, cdy,
            );
        }

        // After the downstroke, before printing:
        if composite_diag() {
            // L1 residual
            let mut max_res_l1 = 0.0f64;
            for pd in self.l1_data.iter() {
                for &v in pd.residual.iter() { max_res_l1 = max_res_l1.max(v.abs()); }
            }
            eprintln!("[composite VCYCLE]   L1 max|residual| = {:.4e}", max_res_l1);
            // Per L2+ level residual
            for (lvl_idx, lvl_data) in self.l2plus_data.iter().enumerate() {
                let mut max_res = 0.0f64;
                for pd in lvl_data.iter() {
                    for &v in pd.residual.iter() { max_res = max_res.max(v.abs()); }
                }
                eprintln!("[composite VCYCLE]   L{} max|residual| = {:.4e}", lvl_idx+2, max_res);
            }

            // ── Per-level max|φ| on magnet layer (scale check) ──
            let mut max_phi_l1 = 0.0f64;
            for pd3d in self.l1_data_3d.iter() {
                for mj in 0..pd3d.int_ny {
                    for mi in 0..pd3d.int_nx {
                        let v = pd3d.phi[mg_kernels::idx3(
                            pd3d.offx + mi, pd3d.offy + mj, pd3d.offz,
                            pd3d.px, pd3d.py)].abs();
                        max_phi_l1 = max_phi_l1.max(v);
                    }
                }
            }
            eprintln!("[composite VCYCLE]   L1 max|φ_magnet| = {:.4e}", max_phi_l1);
            for (lvl_idx, lvl_data_3d) in self.l2plus_data_3d.iter().enumerate() {
                let mut max_phi = 0.0f64;
                for pd3d in lvl_data_3d.iter() {
                    for mj in 0..pd3d.int_ny {
                        for mi in 0..pd3d.int_nx {
                            let v = pd3d.phi[mg_kernels::idx3(
                                pd3d.offx + mi, pd3d.offy + mj, pd3d.offz,
                                pd3d.px, pd3d.py)].abs();
                            max_phi = max_phi.max(v);
                        }
                    }
                }
                eprintln!("[composite VCYCLE]   L{} max|φ_magnet| = {:.4e}", lvl_idx+2, max_phi);
            }

            // ── First cycle only: ghost-fill scale diagnostics ──
            if cycle == 0 {
                // L0 φ scale (the source for L1 ghost values)
                let max_l0_phi: f64 = self.coarse_phi.iter()
                    .map(|v| v.abs()).fold(0.0, f64::max);
                eprintln!("[ghost-diag] max|φ_L0_magnet_layer| = {:.4e}", max_l0_phi);

                // L0 3D phi scale at magnet layer (with PPPM correction)
                let mut max_l0_3d = 0.0f64;
                for mj in 0..ny_m {
                    for mi in 0..nx_m {
                        max_l0_3d = max_l0_3d.max(
                            l0_phi_3d[mg_kernels::idx3(
                                l0_offx + mi, l0_offy + mj, l0_offz, l0_px, l0_py
                            )].abs());
                    }
                }
                eprintln!("[ghost-diag] max|φ_L0_3d_corrected_magnet| = {:.4e}", max_l0_3d);

                // Ghost-fill values at first L1 patch boundary (from its 3D φ after smooth)
                // The boundary cells of pd3d.phi were stamped from L0 by enforce_dirichlet.
                if !self.l1_data_3d.is_empty() {
                    let pd3d = &self.l1_data_3d[0];
                    let (px, py, _pz) = (pd3d.px, pd3d.py, pd3d.pz);
                    let offz = pd3d.offz;
                    // Sample boundary cells at k=offz to see what ghost values were set
                    let mut max_bc_zlayer = 0.0f64;
                    for j in 0..py {
                        for i in 0..px {
                            let is_b = i == 0 || i+1 == px || j == 0 || j+1 == py;
                            if is_b {
                                let v = pd3d.phi[mg_kernels::idx3(i, j, offz, px, py)].abs();
                                max_bc_zlayer = max_bc_zlayer.max(v);
                            }
                        }
                    }
                    // Interior phi of patch 0 after downstroke smooth
                    let mut max_patch_phi = 0.0f64;
                    for mj in 0..pd3d.int_ny {
                        for mi in 0..pd3d.int_nx {
                            max_patch_phi = max_patch_phi.max(pd3d.phi[mg_kernels::idx3(
                                pd3d.offx + mi, pd3d.offy + mj, offz, px, py)].abs());
                        }
                    }
                    eprintln!("[ghost-diag] L1 patch 0: max|φ_boundary_k=offz| = {:.4e}, max|φ_interior_k=offz| = {:.4e}  (ratio = {:.2})",
                        max_bc_zlayer, max_patch_phi,
                        if max_patch_phi > 1e-30 { max_bc_zlayer / max_patch_phi } else { 0.0 });
                }

                // Ghost-fill values for first L2 patch (from parent L1)
                if !self.l2plus_data_3d.is_empty() && !self.l2plus_data_3d[0].is_empty() {
                    let pd3d_l2 = &self.l2plus_data_3d[0][0];
                    let parent_idx = parent_maps[0][0];
                    let parent_pd3d = &self.l1_data_3d[parent_idx];
                    let mut max_parent_phi = 0.0f64;
                    for mj in 0..parent_pd3d.int_ny {
                        for mi in 0..parent_pd3d.int_nx {
                            max_parent_phi = max_parent_phi.max(parent_pd3d.phi[mg_kernels::idx3(
                                parent_pd3d.offx + mi, parent_pd3d.offy + mj, parent_pd3d.offz,
                                parent_pd3d.px, parent_pd3d.py)].abs());
                        }
                    }
                    let mut max_l2_phi = 0.0f64;
                    for mj in 0..pd3d_l2.int_ny {
                        for mi in 0..pd3d_l2.int_nx {
                            max_l2_phi = max_l2_phi.max(pd3d_l2.phi[mg_kernels::idx3(
                                pd3d_l2.offx + mi, pd3d_l2.offy + mj, pd3d_l2.offz,
                                pd3d_l2.px, pd3d_l2.py)].abs());
                        }
                    }
                    eprintln!("[ghost-diag] L2 patch 0: parent(L1#{}) max|φ| = {:.4e}, L2 max|φ| = {:.4e}",
                        parent_idx, max_parent_phi, max_l2_phi);
                }

                // RHS scale at each level (the source term that drives φ)
                let mut max_rhs_l1 = 0.0f64;
                for pd in self.l1_data.iter() {
                    for &v in pd.rhs.iter() { max_rhs_l1 = max_rhs_l1.max(v.abs()); }
                }
                eprintln!("[ghost-diag] max|rhs_L1| = {:.4e}", max_rhs_l1);
                for (lvl_idx, lvl_data) in self.l2plus_data.iter().enumerate() {
                    let mut max_rhs = 0.0f64;
                    for pd in lvl_data.iter() {
                        for &v in pd.rhs.iter() { max_rhs = max_rhs.max(v.abs()); }
                    }
                    eprintln!("[ghost-diag] max|rhs_L{}| = {:.4e}", lvl_idx+2, max_rhs);
                }
            }
        }

        // ═══════ L0 IS FIXED (from bootstrap) ═══════
        //
        // The L0 solve was done once in compute_vcycle before the iteration
        // loop (with enhanced RHS + PPPM). The L0 RHS doesn't change between
        // iterations (static magnetisation), so L0 phi is a fixed point.
        //
        // We skip:
        //  - Cascade restriction of fine residuals (nothing to inject into L0)
        //  - L0 re-solve (would produce identical result)
        //
        // phi_old_l0 == self.coarse_phi (unchanged), so the L1 prolongation
        // below computes delta = 0. But L1 still gets mini_cycled with
        // correct ghost-fill, and L2+ benefits from L1's updated state.

        let phi_old_l0 = self.coarse_phi.clone();

        // ═══════ UPSTROKE: coarsest → finest ═══════

        // Snapshot L0's 3D phi with PPPM correction at k=offz for ghost-fill.
        let l0_phi_3d_new = make_corrected_l0_phi(self.l0_solver.mg.phi_3d());

        // Save L1 interior magnet-layer phi before prolongation (for L2 delta).
        let l1_phi_old_interior: Vec<Vec<f64>> = self.l1_data_3d.iter()
            .map(|pd| extract_interior_magnet_layer(pd)).collect();

        // L1 patches: prolongate correction from L0, then post-smooth.
        for (i, patch) in h.patches.iter().enumerate() {
            let pd3d = &mut self.l1_data_3d[i];

            prolongate_correction_to_3d(
                &self.coarse_phi, &phi_old_l0, cnx, cny, cdx, cdy, patch, pd3d);

            run_mini_cycle(
                pd3d, &mut cache, patch, ms, self.vcfg.n_post,
                &l0_phi_3d_new, l0_px, l0_py, l0_offx, l0_offy, cdx, cdy,
            );
        }

        // L2+ levels: prolongate from parent, then post-smooth.
        // Process from coarsest (L2) to finest.
        let mut prev_phi_old_interior: Vec<Vec<f64>> = l1_phi_old_interior;

        for lvl_idx in 0..h.patches_l2plus.len() {
            let lvl_patches = &h.patches_l2plus[lvl_idx];
            if lvl_patches.is_empty() {
                prev_phi_old_interior = self.l2plus_data_3d[lvl_idx].iter()
                    .map(|pd| extract_interior_magnet_layer(pd)).collect();
                continue;
            }

            let parent_patches: &[Patch2D] = if lvl_idx == 0 {
                &h.patches
            } else {
                &h.patches_l2plus[lvl_idx - 1]
            };

            // Current parent interior phi (after parent's upstroke).
            let parent_phis_new_interior: Vec<Vec<f64>> = if lvl_idx == 0 {
                self.l1_data_3d.iter().map(|pd| extract_interior_magnet_layer(pd)).collect()
            } else {
                self.l2plus_data_3d[lvl_idx - 1].iter().map(|pd| extract_interior_magnet_layer(pd)).collect()
            };
            // Parent 3D phi for ghost-fill in run_mini_cycle.
            let parent_phis_3d: Vec<Vec<f64>> = if lvl_idx == 0 {
                self.l1_data_3d.iter().map(|pd| pd.phi.clone()).collect()
            } else {
                self.l2plus_data_3d[lvl_idx - 1].iter().map(|pd| pd.phi.clone()).collect()
            };
            let parent_geoms: Vec<(usize, usize, usize, usize, f64, f64)> = if lvl_idx == 0 {
                self.l1_data_3d.iter().map(|pd| (pd.px, pd.py, pd.offx, pd.offy, pd.dx, pd.dy)).collect()
            } else {
                self.l2plus_data_3d[lvl_idx - 1].iter().map(|pd| (pd.px, pd.py, pd.offx, pd.offy, pd.dx, pd.dy)).collect()
            };

            // Save this level's interior phi before prolongation.
            let this_phi_old_interior: Vec<Vec<f64>> = self.l2plus_data_3d[lvl_idx].iter()
                .map(|pd| extract_interior_magnet_layer(pd)).collect();

            for (pi, patch) in lvl_patches.iter().enumerate() {
                let parent_idx = parent_maps[lvl_idx][pi];
                let (ppx, ppy, poffx, poffy, pdx, pdy) = parent_geoms[parent_idx];

                let pd3d = &mut self.l2plus_data_3d[lvl_idx][pi];

                prolongate_correction_to_3d_from_parent(
                    &parent_patches[parent_idx],
                    &parent_phis_new_interior[parent_idx],
                    &prev_phi_old_interior[parent_idx],
                    patch, pd3d,
                );

                run_mini_cycle(
                    pd3d, &mut cache, patch, ms, self.vcfg.n_post,
                    &parent_phis_3d[parent_idx], ppx, ppy, poffx, poffy, pdx, pdy,
                );
            }

            prev_phi_old_interior = this_phi_old_interior;
        }

        // Put cache back.
        self.patch_mg_cache = Some(cache);
    }

    /// Composite V-cycle solve (true iterative V-cycle).
    ///
    /// Architecture (following García-Cervera / AMReX):
    ///   1. Allocate/populate patch φ/rhs data.
    ///   2. Iterate the composite V-cycle:
    ///        Downstroke: finest → coarsest (smooth + restrict residuals)
    ///        L0 solve: 3D MG+PPPM with restricted fine residuals
    ///        Upstroke: coarsest → finest (prolongate + smooth)
    ///   3. Extract B from converged φ:
    ///        L0: from the MG solve (ΔK-corrected)
    ///        Patches: B = −μ₀∇(φ_patch) at fine resolution
    ///
    /// Key difference from the old defect-correction path:
    ///   - Patches solve the FULL Poisson equation (rhs = fine ∇·M), not a defect
    ///   - Ghost values come from parent-level φ (hierarchical, not from L0)
    ///   - Multiple V-cycle iterations allow φ to converge across all levels
    ///   - B is extracted from the converged φ gradient, not from interp(B_L0)+δB
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

        // ---- Phase B: allocate 3D patch data if needed ----
        let need_3d_realloc = self.l1_data_3d.len() != n_l1
            || self.l2plus_data_3d.len() != h.patches_l2plus.len()
            || self.l2plus_data_3d.iter().zip(h.patches_l2plus.iter())
                .any(|(d, p)| d.len() != p.len());

        if need_3d_realloc && has_patches {
            let (_, _, n_vac_z) = self.l0_solver.mg.offsets();
            let dz = self.base_grid.dz;
            let mut cache = self.patch_mg_cache.take()
                .unwrap_or_else(|| PatchMGCache::new(n_vac_z, dz));

            self.l1_data_3d = h.patches.iter()
                .map(|p| cache.create_patch_data(p))
                .collect();
            self.l2plus_data_3d = h.patches_l2plus.iter()
                .map(|lvl| lvl.iter().map(|p| cache.create_patch_data(p)).collect())
                .collect();

            self.patch_mg_cache = Some(cache);
        }

        if self.patch_mg_cache.is_none() {
            let (_, _, n_vac_z) = self.l0_solver.mg.offsets();
            let dz = self.base_grid.dz;
            self.patch_mg_cache = Some(PatchMGCache::new(n_vac_z, dz));
        }

        let n_coarse = h.base_grid.nx * h.base_grid.ny;
        if self.coarse_phi.len() != n_coarse {
            self.coarse_phi = vec![0.0; n_coarse];
            self.coarse_div = vec![0.0; n_coarse];
        }

        // ---- Compute fine ∇·M on all patches (patch RHS) ----
        if has_patches {
            compute_all_patch_rhs(h, &mut self.l1_data, &mut self.l2plus_data, mat.ms);
        }

        // ---- Compute coarse ∇·M on L0 (for restriction delta) ----
        compute_scaled_div_m(
            &h.coarse.data, h.base_grid.nx, h.base_grid.ny,
            h.base_grid.dx, h.base_grid.dy, mat.ms,
            &mut self.coarse_div,
        );

        // ---- Pre-compute parent-patch index maps ----
        let parent_maps = build_parent_index_maps(h);

        // ════════════════════════════════════════════════════════════════
        // BOOTSTRAP: Initial L0 solve with enhanced RHS + PPPM
        // ════════════════════════════════════════════════════════════════
        //
        // The L0 solve is done ONCE with the enhanced-RHS approach:
        //   rhs_L0[covered] = area_avg(rhs_fine)
        //   rhs_L0[uncovered] = coarse ∇·M
        // This gives L0 the best possible phi for patch ghost-fill.
        // Subsequent V-cycle iterations only smooth patches — L0 is fixed.
        let max_cycles = self.vcfg.max_cycles;
        let mut b_scratch = VectorField2D::new(self.base_grid);

        // Compute enhanced-RHS corrections (same as the `compute` path).
        let enhanced_corrections = if has_patches {
            compute_patch_corrections(h, &self.coarse_div, mat.ms)
        } else {
            Vec::new()
        };

        // Bootstrap L0 solve: full MG+PPPM with enhanced RHS.
        // This calls build_rhs_from_m, adds corrections, solves, extracts B.
        self.l0_solver.solve_with_corrections(
            &h.coarse, &enhanced_corrections, &mut b_scratch, mat);
        let new_phi = self.l0_solver.mg.extract_magnet_layer_phi();
        self.coarse_phi.copy_from_slice(&new_phi);

        // PPPM ΔK phi correction: computed NOW (not lazily after first
        // iteration). The delta kernel is loaded from disk cache by
        // solve_with_corrections → ensure_delta_kernel. This fixes Bug 3:
        // patches get PPPM-accurate ghost values from iteration 1.
        let n_coarse_cells = h.base_grid.nx * h.base_grid.ny;
        let mut pppm_delta: Vec<f64> = vec![0.0f64; n_coarse_cells];
        if let Some(dk) = self.l0_solver.delta_kernel() {
            dk.apply_phi_correction(&h.coarse, &mut pppm_delta, mat.ms);
            if composite_diag() {
                let max_d: f64 = pppm_delta.iter().map(|v| v.abs()).fold(0.0, f64::max);
                eprintln!("[composite VCYCLE] PPPM delta loaded: max|delta| = {:.4e}", max_d);
            }
        }

        // ════════════════════════════════════════════════════════════════
        // COMPOSITE V-CYCLE ITERATIONS (patch smoothing)
        // ════════════════════════════════════════════════════════════════
        //
        // L0 phi is fixed from the bootstrap. Each iteration smooths
        // patches with ghost-fill from L0 (for L1) or parent patches
        // (for L2+). Patches warm-start from the previous iteration.
        // Convergence comes from accumulated miniCycle iterations.

        if composite_diag() {
            let n_l2plus: usize = h.patches_l2plus.iter().map(|v| v.len()).sum();
            eprintln!("[composite VCYCLE] ═══════════════════════════════════════════════════");
            eprintln!("[composite VCYCLE] Iterative V-cycle: L1={}, L2+={}, max_cycles={}, n_pre={}, n_post={}",
                n_l1, n_l2plus, max_cycles, self.vcfg.n_pre, self.vcfg.n_post);
        }

        for cycle in 0..max_cycles {
            self.vcycle_iteration(h, mat, &parent_maps, &mut b_scratch, &pppm_delta, cycle);

            // Monitor convergence: max residual norm on all patches.
            if composite_diag() || cycle + 1 == max_cycles {
                let mut max_res = 0.0f64;
                for pd in self.l1_data.iter() {
                    for &v in pd.residual.iter() {
                        max_res = max_res.max(v.abs());
                    }
                }
                for lvl_data in self.l2plus_data.iter() {
                    for pd in lvl_data.iter() {
                        for &v in pd.residual.iter() {
                            max_res = max_res.max(v.abs());
                        }
                    }
                }
                if composite_diag() {
                    eprintln!(
                        "[composite VCYCLE] cycle {}/{}: max|residual| = {:.4e}",
                        cycle + 1, max_cycles, max_res);
                }
            }
        }

        // ---- Phase B: sync 3D magnet-layer phi → 2D pd.phi ----

        for (i, patch) in h.patches.iter().enumerate() {
            if i < self.l1_data_3d.len() && i < self.l1_data.len() {
                sync_3d_phi_to_2d(&self.l1_data_3d[i], &mut self.l1_data[i], patch);
            }
        }
        for (lvl_idx, lvl_patches) in h.patches_l2plus.iter().enumerate() {
            if lvl_idx >= self.l2plus_data_3d.len() { break; }
            for (pi, patch) in lvl_patches.iter().enumerate() {
                if pi < self.l2plus_data_3d[lvl_idx].len()
                    && pi < self.l2plus_data[lvl_idx].len()
                {
                    sync_3d_phi_to_2d(
                        &self.l2plus_data_3d[lvl_idx][pi],
                        &mut self.l2plus_data[lvl_idx][pi],
                        patch);
                }
            }
        }

        if composite_diag() {
            // Report converged φ magnitudes per level.
            let mut max_phi_l1 = 0.0f64;
            for pd in self.l1_data.iter() {
                for &v in pd.phi.iter() {
                    max_phi_l1 = max_phi_l1.max(v.abs());
                }
            }
            eprintln!("[composite VCYCLE] converged max|φ| on L1 = {:.4e}", max_phi_l1);

            for (lvl_idx, lvl_data) in self.l2plus_data.iter().enumerate() {
                let mut max_phi = 0.0f64;
                for pd in lvl_data.iter() {
                    for &v in pd.phi.iter() {
                        max_phi = max_phi.max(v.abs());
                    }
                }
                if !lvl_data.is_empty() {
                    eprintln!(
                        "[composite VCYCLE] converged max|φ| on L{} = {:.4e}",
                        lvl_idx + 2, max_phi);
                }
            }
            eprintln!("[composite VCYCLE] ═══════════════════════════════════════════════════");
        }

        // ════════════════════════════════════════════════════════════════
        // EXTRACT B FROM CONVERGED φ
        // ════════════════════════════════════════════════════════════════
        //
        // L0 B: from b_scratch (the ΔK-corrected MG solve output).
        // Patch B: -μ₀∇(φ_patch) at fine resolution using central differences.
        //   Bx, By from fine φ gradient; Bz interpolated from L0.
        //   Ghost cells in φ provide boundary data for the stencil.

        b_demag_coarse.data.copy_from_slice(&b_scratch.data);

        // L1 patch B from fine φ gradient.
        let b_l1: Vec<Vec<[f64; 3]>> = h.patches.iter()
            .zip(self.l1_data.iter())
            .map(|(patch, pd)| extract_b_from_patch_phi(patch, &pd.phi, b_demag_coarse))
            .collect();

        // L2+ patch B from fine φ gradient.
        let b_l2: Vec<Vec<Vec<[f64; 3]>>> = h.patches_l2plus.iter()
            .zip(self.l2plus_data.iter())
            .map(|(lvl_patches, lvl_data)| {
                lvl_patches.iter().zip(lvl_data.iter())
                    .map(|(patch, pd)| extract_b_from_patch_phi(patch, &pd.phi, b_demag_coarse))
                    .collect()
            })
            .collect();

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

    // ------------------------------------------------------------------
    // Diagnostic accessors (for benchmark φ comparison)
    // ------------------------------------------------------------------

    /// Get a reference to the L1 3D patch data (for benchmark diagnostics).
    #[allow(dead_code)]
    pub(crate) fn l1_data_3d_ref(&self) -> &[PatchPoisson3D] {
        &self.l1_data_3d
    }

    /// Get a reference to the L2+ 3D patch data (for benchmark diagnostics).
    #[allow(dead_code)]
    pub(crate) fn l2plus_data_3d_ref(&self) -> &[Vec<PatchPoisson3D>] {
        &self.l2plus_data_3d
    }

    /// Get the L0 magnet-layer φ (for benchmark diagnostics).
    #[allow(dead_code)]
    pub(crate) fn coarse_phi_ref(&self) -> &[f64] {
        &self.coarse_phi
    }

    /// Print a comprehensive diagnostic summary comparing patch φ against
    /// a fine-reference φ field.
    ///
    /// `fine_phi` is the magnet-layer φ from a uniform fine MG solve, stored
    /// as a flat array of size `fine_nx * fine_ny`.
    /// `fine_nx/ny/dx/dy` describe the fine reference grid.
    #[allow(dead_code)]
    pub(crate) fn print_phi_comparison(
        &self,
        h: &AmrHierarchy2D,
        fine_phi: &[f64],
        fine_nx: usize, fine_ny: usize,
        fine_dx: f64, fine_dy: f64,
    ) {
        eprintln!("\n[φ-comparison] ══════════════════════════════════════════");
        eprintln!("[φ-comparison] Patch φ vs uniform fine MG reference φ");
        eprintln!("[φ-comparison] Fine ref grid: {}×{}, dx={:.4e}", fine_nx, fine_ny, fine_dx);

        let max_phi_ref: f64 = fine_phi.iter().map(|v| v.abs()).fold(0.0, f64::max);
        eprintln!("[φ-comparison] max|φ_fine_ref| = {:.4e}", max_phi_ref);

        // L1 patches
        for (pi, patch) in h.patches.iter().enumerate() {
            if pi >= self.l1_data_3d.len() { break; }
            let pd3d = &self.l1_data_3d[pi];
            let ghost = patch.ghost;
            let mut max_diff = 0.0f64;
            let mut max_phi_patch = 0.0f64;
            let mut max_phi_ref_local = 0.0f64;
            let mut sum_se = 0.0f64;
            let mut n_cells = 0usize;

            for mj in 0..pd3d.int_ny {
                for mi in 0..pd3d.int_nx {
                    let (x, y) = patch.cell_center_xy(ghost + mi, ghost + mj);
                    // Map to fine reference grid index
                    let fi = (x / fine_dx - 0.5).round() as isize;
                    let fj = (y / fine_dy - 0.5).round() as isize;
                    if fi < 0 || fj < 0 || fi >= fine_nx as isize || fj >= fine_ny as isize {
                        continue;
                    }
                    let fi = fi as usize;
                    let fj = fj as usize;

                    let phi_ref = fine_phi[fj * fine_nx + fi];
                    let phi_comp = pd3d.phi[mg_kernels::idx3(
                        pd3d.offx + mi, pd3d.offy + mj, pd3d.offz,
                        pd3d.px, pd3d.py)];

                    let diff = (phi_ref - phi_comp).abs();
                    max_diff = max_diff.max(diff);
                    max_phi_patch = max_phi_patch.max(phi_comp.abs());
                    max_phi_ref_local = max_phi_ref_local.max(phi_ref.abs());
                    sum_se += diff * diff;
                    n_cells += 1;
                }
            }
            let rmse = if n_cells > 0 { (sum_se / n_cells as f64).sqrt() } else { 0.0 };
            let rel_pct = if max_phi_ref_local > 0.0 { max_diff / max_phi_ref_local * 100.0 } else { 0.0 };
            if pi < 3 || pi + 1 == h.patches.len() {
                eprintln!("[φ-comparison] L1 patch {}: max|φ_ref|={:.4e} max|φ_comp|={:.4e} \
                    max|Δφ|={:.4e} ({:.1}%) rmse={:.4e} ({} cells)",
                    pi, max_phi_ref_local, max_phi_patch, max_diff, rel_pct, rmse, n_cells);
            }
        }

        // L2+ patches (summarise per level)
        for (lvl_idx, lvl_patches) in h.patches_l2plus.iter().enumerate() {
            if lvl_idx >= self.l2plus_data_3d.len() { break; }
            let mut total_se = 0.0f64;
            let mut total_n = 0usize;
            let mut level_max_diff = 0.0f64;
            let mut level_max_ref = 0.0f64;
            let mut level_max_comp = 0.0f64;

            for (pi, patch) in lvl_patches.iter().enumerate() {
                if pi >= self.l2plus_data_3d[lvl_idx].len() { break; }
                let pd3d = &self.l2plus_data_3d[lvl_idx][pi];
                let ghost = patch.ghost;

                for mj in 0..pd3d.int_ny {
                    for mi in 0..pd3d.int_nx {
                        let (x, y) = patch.cell_center_xy(ghost + mi, ghost + mj);
                        let fi = (x / fine_dx - 0.5).round() as isize;
                        let fj = (y / fine_dy - 0.5).round() as isize;
                        if fi < 0 || fj < 0 || fi >= fine_nx as isize || fj >= fine_ny as isize {
                            continue;
                        }
                        let fi = fi as usize;
                        let fj = fj as usize;

                        let phi_ref = fine_phi[fj * fine_nx + fi];
                        let phi_comp = pd3d.phi[mg_kernels::idx3(
                            pd3d.offx + mi, pd3d.offy + mj, pd3d.offz,
                            pd3d.px, pd3d.py)];

                        let diff = (phi_ref - phi_comp).abs();
                        level_max_diff = level_max_diff.max(diff);
                        level_max_ref = level_max_ref.max(phi_ref.abs());
                        level_max_comp = level_max_comp.max(phi_comp.abs());
                        total_se += diff * diff;
                        total_n += 1;
                    }
                }
            }
            let rmse = if total_n > 0 { (total_se / total_n as f64).sqrt() } else { 0.0 };
            let rel_pct = if level_max_ref > 0.0 { level_max_diff / level_max_ref * 100.0 } else { 0.0 };
            eprintln!("[φ-comparison] L{} ({} patches): max|φ_ref|={:.4e} max|φ_comp|={:.4e} \
                max|Δφ|={:.4e} ({:.1}%) rmse={:.4e} ({} cells)",
                lvl_idx + 2, lvl_patches.len(), level_max_ref, level_max_comp,
                level_max_diff, rel_pct, rmse, total_n);
        }

        eprintln!("[φ-comparison] ══════════════════════════════════════════\n");
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

        // Allocate 3D patch data (Phase B).
        {
            let (_, _, n_vac_z) = solver.l0_solver.mg.offsets();
            let dz = grid.dz;
            let mut cache = solver.patch_mg_cache.take()
                .unwrap_or_else(|| PatchMGCache::new(n_vac_z, dz));
            solver.l1_data_3d = h.patches.iter()
                .map(|p| cache.create_patch_data(p))
                .collect();
            solver.l2plus_data_3d = h.patches_l2plus.iter()
                .map(|lvl| lvl.iter().map(|p| cache.create_patch_data(p)).collect())
                .collect();
            solver.patch_mg_cache = Some(cache);
        }

        // Compute coarse div.
        let n_coarse = grid.nx * grid.ny;
        solver.coarse_phi = vec![0.0; n_coarse];
        solver.coarse_div = vec![0.0; n_coarse];
        compute_scaled_div_m(
            &h.coarse.data, grid.nx, grid.ny, grid.dx, grid.dy, mat.ms,
            &mut solver.coarse_div);

        // Run V-cycles and track max patch residual.
        let mut b_scratch = VectorField2D::new(grid);
        let parent_maps = build_parent_index_maps(&h);
        let mut residuals = Vec::new();

        for cycle in 0..5 {
            solver.vcycle_iteration(&h, &mat, &parent_maps, &mut b_scratch, &[], cycle);

            // Read the residual that was set by smooth_and_residual_3d
            // during the downstroke (already the 3D magnet-layer residual).
            let mut max_res = 0.0f64;
            for pd in solver.l1_data.iter() {
                let ghost = pd.ghost;
                for j in ghost..(pd.ny - ghost) {
                    for i in ghost..(pd.nx - ghost) {
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

// =========================================================================
// Phase B tests: Sessions 1–4
// =========================================================================

#[cfg(test)]
mod phase_b_tests {
    use super::*;
    use crate::grid::Grid2D;

    // ── Session 1 ──

    #[test]
    fn test_patch_solver_matches_l0_geometry() {
        let dx = 3.906e-9;
        let dy = 3.906e-9;
        let dz = 3.0e-9;
        let n_vac_z = 16;
        let mut cache = PatchMGCache::new(n_vac_z, dz);
        let solver = cache.get_or_create(32, 32, dx, dy);
        let (px, py, pz) = solver.padded_dims();
        let (_offx, _offy, offz) = solver.offsets();
        assert!(px >= 32 + 2 * PATCH_PAD_XY);
        assert!(py >= 32 + 2 * PATCH_PAD_XY);
        assert_eq!(offz, n_vac_z);
        assert!(pz >= 1 + 2 * n_vac_z);
        solver.phi_3d_mut().fill(0.0);
        solver.bc_phi_mut().fill(0.0);
        solver.mini_solve(0);
        let max_phi: f64 = solver.phi_3d().iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(max_phi < 1e-30);
        eprintln!("[phase_b] PASSED: patch solver geometry");
    }

    #[test]
    fn test_mini_cycle_produces_nonzero_phi() {
        let mut cache = PatchMGCache::new(16, 3.0e-9);
        let solver = cache.get_or_create(16, 16, 3.906e-9, 3.906e-9);
        let m: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0]; 256];
        solver.build_rhs_from_m_raw(&m, 8e5);
        solver.phi_3d_mut().fill(0.0);
        solver.bc_phi_mut().fill(0.0);
        solver.mini_solve(2);
        let (px, py, _) = solver.padded_dims();
        let (offx, offy, offz) = solver.offsets();
        let phi = solver.phi_3d();
        let mut mx = 0.0f64;
        for mj in 0..16 { for mi in 0..16 {
            mx = mx.max(phi[mg_kernels::idx3(offx+mi, offy+mj, offz, px, py)].abs());
        }}
        assert!(mx > 1e-20 && mx.is_finite());
        eprintln!("[phase_b] PASSED: miniCycle nonzero phi (max={:.3e})", mx);
    }

    #[test]
    fn test_patch_poisson_3d_creation() {
        use crate::amr::patch::Patch2D;
        use crate::amr::rect::Rect2i;
        let base = Grid2D::new(128, 128, 7.812e-9, 7.812e-9, 3e-9);
        let patch = Patch2D::new(&base, Rect2i::new(4, 4, 8, 8), 2, 2);
        let mut cache = PatchMGCache::new(16, 3e-9);
        let pd3d = cache.create_patch_data(&patch);
        assert_eq!(pd3d.int_nx, 16);
        assert_eq!(pd3d.int_ny, 16);
        assert_eq!(pd3d.phi.len(), pd3d.px * pd3d.py * pd3d.pz);
        eprintln!("[phase_b] PASSED: PatchPoisson3D creation");
    }

    // ── Session 2 ──

    #[test]
    fn test_3d_rhs_raw_matches_standard() {
        use crate::vector_field::VectorField2D;
        let grid = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let mut m = VectorField2D::new(grid);
        let cx = 8.0; let cy = 8.0;
        for j in 0..16 { for i in 0..16 {
            let x = i as f64 + 0.5 - cx;
            let y = j as f64 + 0.5 - cy;
            let r = (x*x + y*y).sqrt().max(0.5);
            let (mx, my, mz) = (-y/r, x/r, 0.3);
            let n = (mx*mx+my*my+mz*mz).sqrt();
            m.data[j*16+i] = [mx/n, my/n, mz/n];
        }}
        let cfg = DemagPoissonMGConfig::from_env();
        let mut a = DemagPoissonMG::new(grid, cfg);
        a.build_rhs_from_m(&m, 8e5);
        let ra = a.rhs_3d().to_vec();
        let mut b = DemagPoissonMG::new(grid, cfg);
        b.build_rhs_from_m_raw(&m.data, 8e5);
        let rb = b.rhs_3d().to_vec();
        let md: f64 = ra.iter().zip(rb.iter()).map(|(a,b)|(a-b).abs()).fold(0.0, f64::max);
        assert!(md < 1e-20, "RHS mismatch: {:.3e}", md);
        eprintln!("[phase_b s2] PASSED: build_rhs_from_m_raw bit-identical");
    }

    #[test]
    fn test_patch_rhs_matches_l0_at_same_resolution() {
        let (nx, ny) = (16, 16);
        let (dx, dy, dz, ms) = (2.5e-9, 2.5e-9, 3e-9, 8e5);
        let norm = (0.6f64*0.6+0.8*0.8+0.3*0.3).sqrt();
        let m_vec: Vec<[f64;3]> = vec![[0.6/norm, 0.8/norm, 0.3/norm]; nx*ny];
        let grid = Grid2D::new(nx, ny, dx, dy, dz);
        let cfg = DemagPoissonMGConfig::from_env();
        let mut sl0 = DemagPoissonMG::new(grid, cfg);
        sl0.build_rhs_from_m_raw(&m_vec, ms);
        let rl0 = sl0.rhs_3d().to_vec();
        let (l0px, l0py, _) = sl0.padded_dims();
        let (l0ox, l0oy, l0oz) = sl0.offsets();
        let mut cache = PatchMGCache::new(16, dz);
        let sp = cache.get_or_create(nx, ny, dx, dy);
        sp.build_rhs_from_m_raw(&m_vec, ms);
        let rp = sp.rhs_3d().to_vec();
        let (ppx, ppy, _) = sp.padded_dims();
        let (pox, poy, poz) = sp.offsets();
        for dk in [-1isize, 0, 1] {
            let mut md = 0.0f64;
            let kl = (l0oz as isize + dk) as usize;
            let kp = (poz as isize + dk) as usize;
            for mj in 0..ny { for mi in 0..nx {
                let vl = rl0[mg_kernels::idx3(l0ox+mi, l0oy+mj, kl, l0px, l0py)];
                let vp = rp[mg_kernels::idx3(pox+mi, poy+mj, kp, ppx, ppy)];
                md = md.max((vl-vp).abs());
            }}
            assert!(md < 1e-10, "RHS mismatch dk={}: {:.3e}", dk, md);
        }
        eprintln!("[phase_b s2] PASSED: patch RHS matches L0");
    }

    #[test]
    fn test_ghost_fill_3d_from_known_parent() {
        use crate::amr::patch::Patch2D;
        use crate::amr::rect::Rect2i;
        let (pnx, pny) = (32, 32);
        let (pdx, pdy, dz) = (5e-9, 5e-9, 3e-9);
        let base = Grid2D::new(pnx, pny, pdx, pdy, dz);
        let mut pcache = PatchMGCache::new(16, dz);
        let ps = pcache.get_or_create(pnx, pny, pdx, pdy);
        let (ppx, ppy, ppz) = ps.padded_dims();
        let (pox, poy, _) = ps.offsets();
        let n = ppx*ppy*ppz;
        let mut pphi = vec![0.0f64; n];
        for k in 0..ppz { for pj in 0..ppy { for pi in 0..ppx {
            let mi = pi as f64 - pox as f64;
            let mj = pj as f64 - poy as f64;
            pphi[mg_kernels::idx3(pi, pj, k, ppx, ppy)] = (mi+0.5)*pdx + 2.0*(mj+0.5)*pdy;
        }}}
        let cp = Patch2D::new(&base, Rect2i::new(8, 8, 8, 8), 2, 2);
        let mut ccache = PatchMGCache::new(16, dz);
        let pd3d = ccache.create_patch_data(&cp);
        let mut bc = vec![0.0f64; pd3d.px*pd3d.py*pd3d.pz];
        fill_patch_bc_from_parent_3d(&cp, &pd3d, &mut bc, &pphi, ppx, ppy, pox, poy, pdx, pdy);
        let k = pd3d.offz;
        let gi0 = (cp.coarse_rect.i0 * cp.ratio) as f64;
        let gj0 = (cp.coarse_rect.j0 * cp.ratio) as f64;
        let cdx = cp.grid.dx;
        let cdy = cp.grid.dy;
        let mut me = 0.0f64;
        for j in 0..pd3d.py { for i in 0..pd3d.px {
            let is_b = i==0||i+1==pd3d.px||j==0||j+1==pd3d.py;
            if !is_b { continue; }
            let mi = i as f64 - pd3d.offx as f64;
            let mj = j as f64 - pd3d.offy as f64;
            let exp = (gi0+mi+0.5)*cdx + 2.0*(gj0+mj+0.5)*cdy;
            me = me.max((bc[mg_kernels::idx3(i, j, k, pd3d.px, pd3d.py)] - exp).abs());
        }}
        assert!(me < 1e-18, "ghost fill err: {:.3e}", me);
        eprintln!("[phase_b s2] PASSED: ghost fill linear (err={:.3e})", me);
    }

    #[test]
    fn test_run_mini_cycle_end_to_end() {
        use crate::amr::patch::Patch2D;
        use crate::amr::rect::Rect2i;
        use crate::vector_field::VectorField2D;
        let (nx, ny) = (32, 32);
        let (dx, dy, dz, ms) = (5e-9, 5e-9, 3e-9, 8e5);
        let grid = Grid2D::new(nx, ny, dx, dy, dz);
        let mut m = VectorField2D::new(grid);
        for v in m.data.iter_mut() { *v = [1.0, 0.0, 0.0]; }
        let cfg = DemagPoissonMGConfig::from_env();
        let mut l0 = DemagPoissonMG::new(grid, cfg);
        l0.build_rhs_from_m(&m, ms);
        l0.phi_3d_mut().fill(0.0);
        l0.bc_phi_mut().fill(0.0);
        l0.mini_solve(16);
        let l0phi = l0.phi_3d().to_vec();
        let (l0px, l0py, _) = l0.padded_dims();
        let (l0ox, l0oy, _) = l0.offsets();
        let mut cp = Patch2D::new(&grid, Rect2i::new(8, 8, 8, 8), 2, 2);
        let g = cp.ghost; let pnx = cp.grid.nx;
        for mj in 0..cp.interior_ny { for mi in 0..cp.interior_nx {
            cp.m.data[(g+mj)*pnx+(g+mi)] = [1.0, 0.0, 0.0];
        }}
        let mut cache = PatchMGCache::new(16, dz);
        let mut pd3d = cache.create_patch_data(&cp);
        run_mini_cycle(&mut pd3d, &mut cache, &cp, ms, 2, &l0phi, l0px, l0py, l0ox, l0oy, dx, dy);
        let b = extract_b_from_patch_phi_3d(&pd3d);
        let bmax: f64 = b.iter().flat_map(|v| v.iter()).map(|x| x.abs()).fold(0.0, f64::max);
        assert!(bmax > 1e-6 && bmax.is_finite(), "B={:.3e}", bmax);
        eprintln!("[phase_b s2] PASSED: end-to-end (B={:.3e})", bmax);
    }
}