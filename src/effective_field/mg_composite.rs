// src/effective_field/mg_composite.rs
//
// Composite-grid Poisson wrapper for AMR-aware demagnetisation.
//
// Computes demag on the coarse (L0) grid using the 3D padded-box MG solver,
// then interpolates the result to AMR patches via bilinear sampling.
//
// This is the MG-based counterpart to coarse_fft_demag.rs:
//   - coarse_fft_demag: FFT on L0, interpolate to patches (production, fast)
//   - mg_composite:     MG on L0, interpolate to patches (experimental, AMR-friendly)
//
// Future enhancement: inject fine-resolution ∇·M from patches into the coarse
// RHS before the MG solve (García-Cervera enhanced-RHS algorithm). This would
// improve accuracy near patch boundaries without requiring a fine-grid solve.

use crate::amr::hierarchy::AmrHierarchy2D;
use crate::amr::interp::sample_bilinear;
use crate::amr::patch::Patch2D;
use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

use super::demag_poisson_mg;

use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[inline]
fn composite_diag() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLG_DEMAG_COMPOSITE_DIAG").is_ok())
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
    // Public API: compute demag on coarse grid, interpolate to patches
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
                "[composite] MG demag on coarse grid: {}x{}, n_l1_patches={}, n_l2plus={}",
                h.base_grid.nx, h.base_grid.ny, n_l1, n_l2plus,
            );
        }

        // Step 1: Compute demag on the coarse (L0) grid using 3D MG solver.
        //
        // This uses the coarse magnetisation directly. The 3D solver handles
        // the full Poisson equation with open BCs via treecode.
        demag_poisson_mg::compute_demag_field_poisson_mg(
            &h.base_grid,
            &h.coarse,
            b_demag_coarse,
            mat,
        );

        // Step 2: Sample coarse B_demag onto patches via bilinear interpolation
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

static COMPOSITE_CACHE: OnceLock<Mutex<Option<CompositeGridPoisson>>> = OnceLock::new();

/// Compute AMR-aware demag using the 3D MG solver on the coarse grid.
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