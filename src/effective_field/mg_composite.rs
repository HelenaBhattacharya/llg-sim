// src/effective_field/mg_composite.rs
//
// Composite-grid Poisson wrapper for AMR-aware demagnetisation.
//
// Implements the García-Cervera & Roma (2005) enhanced-RHS approach using the
// 2D MG + boundary integral solver (Fredkin-Koehler decomposition).
//
// Algorithm:
//   1. Compute ∇·M on the L0 (coarse) grid from coarse magnetisation.
//   2. For each AMR patch, compute fine-resolution ∇·M at the patch's native
//      resolution, then REPLACE the coarse-grid RHS cells covered by the patch
//      with the area-average of the fine divergence. This injects fine-scale
//      charge information into the coarse solver.
//   3. Run the full Fredkin-Koehler pipeline (w-solve + boundary integral +
//      v-solve + Kzz convolution) on the coarse grid with the enhanced RHS.
//   4. Interpolate the resulting B_demag to patch cells via bilinear sampling.
//
// The key insight is step 2: the coarse MG solve "sees" the correct integrated
// charge distribution from patches even though it operates on the coarse grid.
// Without this step, the coarse solver only has the restricted (averaged)
// magnetisation, which destroys fine-scale features like vortex cores.
//
// Validation contract:
//   Zero patches (L0 only) must reproduce the standalone MG solver exactly.
//   The enhanced RHS reduces to the standard coarse ∇·M when no patches exist.

use crate::amr::hierarchy::AmrHierarchy2D;
use crate::amr::interp::sample_bilinear;
use crate::amr::patch::Patch2D;
use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

use super::demag_poisson_mg;
use super::mg_kernels;

use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[inline]
fn composite_diag() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLG_DEMAG_COMPOSITE_DIAG").is_ok())
}

// ---------------------------------------------------------------------------
// Enhanced-RHS divergence injection
// ---------------------------------------------------------------------------

/// Compute ∇·M on the coarse grid from `coarse_m`, scaled by `ms`.
///
/// Returns a flat Vec<f64> of length nx * ny (the RHS for the Poisson solve).
fn compute_coarse_rhs(grid: &Grid2D, coarse_m: &VectorField2D, ms: f64) -> Vec<f64> {
    let nx = grid.nx;
    let ny = grid.ny;
    let mut rhs = vec![0.0f64; nx * ny];

    mg_kernels::compute_div_m_2d(&coarse_m.data, nx, ny, grid.dx, grid.dy, &mut rhs);

    // Scale by Ms: RHS = Ms · ∇·m
    for v in &mut rhs {
        *v *= ms;
    }
    rhs
}

/// Inject fine-resolution ∇·M from a single patch into the coarse RHS.
///
/// For each coarse cell covered by the patch, compute the area-average of
/// the fine-resolution ∇·M over the ratio×ratio fine cells that correspond
/// to that coarse cell, then REPLACE the coarse RHS value.
///
/// This is the core of the García-Cervera enhanced-RHS algorithm.
fn inject_fine_divergence(
    coarse_rhs: &mut [f64],
    base_grid: &Grid2D,
    patch: &Patch2D,
    ms: f64,
) {
    let ratio = patch.ratio;
    let ghost = patch.ghost;
    let cr = &patch.coarse_rect;

    // The patch grid includes ghosts. Interior fine cells for coarse cell
    // (ic, jc) in patch-local coordinates are:
    //   fine_i = ghost + ic * ratio + 0..ratio-1
    //   fine_j = ghost + jc * ratio + 0..ratio-1
    //
    // The patch's magnetisation `patch.m.data` is stored on the full patch
    // grid (including ghosts), indexed as patch.grid.nx * j + i.

    let pnx = patch.grid.nx; // full patch width including ghosts
    let pny = patch.grid.ny;
    let pdx = patch.grid.dx;
    let pdy = patch.grid.dy;

    // Step 1: Compute ∇·m on the full patch grid (including ghost cells,
    // which provide the stencil halo for boundary fine cells).
    let n_patch = pnx * pny;
    let mut fine_div = vec![0.0f64; n_patch];
    mg_kernels::compute_div_m_2d(&patch.m.data, pnx, pny, pdx, pdy, &mut fine_div);

    // Scale by Ms
    for v in &mut fine_div {
        *v *= ms;
    }

    // Step 2: For each coarse cell covered by this patch, compute the
    // area-average of fine ∇·M and replace the coarse RHS.
    let cnx = base_grid.nx;
    for jc in 0..cr.ny {
        for ic in 0..cr.nx {
            let coarse_i = cr.i0 + ic;
            let coarse_j = cr.j0 + jc;

            // Fine cell indices on the patch grid (interior region starts at ghost)
            let fi0 = ghost + ic * ratio;
            let fj0 = ghost + jc * ratio;

            // Area-average the fine divergence over this coarse cell
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
                coarse_rhs[coarse_j * cnx + coarse_i] = sum / count as f64;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Composite-grid demag solver
// ---------------------------------------------------------------------------

pub(crate) struct CompositeGridPoisson {
    base_grid: Grid2D,
}

impl CompositeGridPoisson {
    pub(crate) fn new(base_grid: Grid2D) -> Self {
        Self { base_grid }
    }

    pub(crate) fn same_structure(&self, h: &AmrHierarchy2D) -> bool {
        self.base_grid.nx == h.base_grid.nx
            && self.base_grid.ny == h.base_grid.ny
            && self.base_grid.dx == h.base_grid.dx
            && self.base_grid.dy == h.base_grid.dy
            && self.base_grid.dz == h.base_grid.dz
    }

    // -----------------------------------------------------------------------
    // Sample coarse demag onto patches via bilinear interpolation
    // -----------------------------------------------------------------------

    fn sample_demag_to_patch(
        b_coarse: &VectorField2D,
        patch: &Patch2D,
    ) -> Vec<[f64; 3]> {
        let pnx = patch.grid.nx;
        let pny = patch.grid.ny;
        let n = pnx * pny;
        let mut b = vec![[0.0; 3]; n];

        for j in 0..pny {
            for i in 0..pnx {
                let (x, y) = patch.cell_center_xy(i, j);
                let v = sample_bilinear(b_coarse, x, y);
                b[j * pnx + i] = v;
            }
        }
        b
    }

    // -----------------------------------------------------------------------
    // Public API: enhanced-RHS composite solve
    // -----------------------------------------------------------------------

    pub(crate) fn compute(
        &self,
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

        if composite_diag() {
            eprintln!(
                "[composite] enhanced-RHS solve: grid={}x{}, n_l1_patches={}, n_l2plus={}",
                h.base_grid.nx, h.base_grid.ny, n_l1, n_l2plus,
            );
        }

        // Step 1: Compute coarse ∇·M (baseline RHS)
        let mut rhs = compute_coarse_rhs(&h.base_grid, &h.coarse, mat.ms);

        // Step 2: Inject fine-resolution ∇·M from all patches
        //
        // Process patches from coarsest to finest level so that deeper
        // refinement levels overwrite the previous level's injection.
        // L1 patches refine the coarse grid directly:
        for patch in &h.patches {
            inject_fine_divergence(&mut rhs, &h.base_grid, patch, mat.ms);
        }
        // L2+ patches: each level's patches are at progressively finer resolution.
        // Their divergence is still injected into the coarse RHS (we average over
        // ratio^(level) fine cells per coarse cell). For L2+ patches, the effective
        // ratio to the coarse grid is ratio^(level+1), but inject_fine_divergence
        // uses the patch's own coarse_rect (which maps to the base grid).
        for lvl in &h.patches_l2plus {
            for patch in lvl {
                inject_fine_divergence(&mut rhs, &h.base_grid, patch, mat.ms);
            }
        }

        if composite_diag() {
            let rhs_max = rhs.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            let rhs_l2: f64 = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
            eprintln!(
                "[composite] enhanced RHS: max={:.3e}, L2={:.3e}",
                rhs_max, rhs_l2,
            );
        }

        // Step 3: Solve FK with enhanced RHS
        demag_poisson_mg::solve_fk_with_external_rhs(
            &h.base_grid,
            &rhs,
            &h.coarse,
            b_demag_coarse,
            mat,
        );

        // Step 4: Sample coarse B_demag onto patches via bilinear interpolation
        let b_patches_l1: Vec<Vec<[f64; 3]>> = h
            .patches
            .iter()
            .map(|p| Self::sample_demag_to_patch(b_demag_coarse, p))
            .collect();

        let b_patches_l2plus: Vec<Vec<Vec<[f64; 3]>>> = h
            .patches_l2plus
            .iter()
            .map(|lvl| {
                lvl.iter()
                    .map(|p| Self::sample_demag_to_patch(b_demag_coarse, p))
                    .collect()
            })
            .collect();

        (b_patches_l1, b_patches_l2plus)
    }
}

// ---------------------------------------------------------------------------
// Module-level cache + public API
// ---------------------------------------------------------------------------

use std::sync::Mutex;

static COMPOSITE_CACHE: OnceLock<Mutex<Option<CompositeGridPoisson>>> = OnceLock::new();

/// Compute AMR-aware demag using the composite-grid solver with enhanced RHS.
///
/// Called from the stepper's `CompositeGrid` mode.
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
        *guard = Some(CompositeGridPoisson::new(h.base_grid));
    }

    guard.as_ref().unwrap().compute(h, mat, b_demag_coarse)
}