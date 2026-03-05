// src/effective_field/demag.rs
//
// Demag dispatcher: selects between demagnetising-field implementations.
//
// This file intentionally contains *no new physics*.
// It routes calls to one of:
//   - demag_fft_uniform.rs    (FFT convolution on a uniform FD grid)
//   - demag_poisson_mg.rs     (3D padded-box MG + treecode BCs)
//
// Runtime override:
//   export LLG_DEMAG_METHOD=fft   (FFT convolution — default)
//   export LLG_DEMAG_METHOD=mg    (3D multigrid on padded box)

use crate::grid::Grid2D;
use crate::params::{DemagMethod, Material};
use crate::vector_field::VectorField2D;

use super::demag_fft_uniform;
use super::demag_poisson_mg;

use rayon::current_num_threads;
use std::sync::{Once, OnceLock};

static WARN_MG_PBC_FALLBACK: Once = Once::new();
static PRINT_DEMAG_METHOD_ONCE: Once = Once::new();

static DEMAG_METHOD_OVERRIDE: OnceLock<Option<DemagMethod>> = OnceLock::new();

/// Resolve the demag method for this run.
///
/// Priority:
/// 1) `LLG_DEMAG_METHOD` environment override (if present)
/// 2) `mat.demag_method`
#[inline]
pub fn resolved_demag_method(mat: &Material) -> DemagMethod {
    if let Some(m) = DEMAG_METHOD_OVERRIDE.get_or_init(|| {
        std::env::var("LLG_DEMAG_METHOD")
            .ok()
            .and_then(|s| DemagMethod::from_str(s.trim()))
    }) {
        return *m;
    }
    mat.demag_method
}

/// Backwards-compatible: open boundaries in x/y (no PBC).
pub fn add_demag_field(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    mat: &Material,
) {
    add_demag_field_pbc(grid, m, b_eff, mat, 0, 0);
}

/// Backwards-compatible: open boundaries in x/y (no PBC).
pub fn compute_demag_field(
    grid: &Grid2D,
    m: &VectorField2D,
    out: &mut VectorField2D,
    mat: &Material,
) {
    compute_demag_field_pbc(grid, m, out, mat, 0, 0);
}

/// Add demag field with periodic boundary conditions in x/y.
///
/// *FFT method*: supports MuMax-style finite-image PBC sums.
/// *MG method*: open-BC only (3D padded box); falls back to FFT for PBC.
pub fn add_demag_field_pbc(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    mat: &Material,
    pbc_x: usize,
    pbc_y: usize,
) {
    if !mat.demag {
        return;
    }

    let method = resolved_demag_method(mat);

    PRINT_DEMAG_METHOD_ONCE.call_once(|| {
        match std::env::var("LLG_DEMAG_METHOD") {
            Ok(raw) => {
                let parsed = DemagMethod::from_str(raw.trim());
                if parsed.is_none() {
                    eprintln!(
                        "[demag] LLG_DEMAG_METHOD='{}' not recognized; falling back to Material.demag_method={:?}",
                        raw,
                        mat.demag_method
                    );
                }
                eprintln!(
                    "[demag] resolved method={:?} (material={:?}, env='{}'), pbc_x={}, pbc_y={}, rayon_threads={}",
                    method, mat.demag_method, raw, pbc_x, pbc_y, current_num_threads()
                );
            }
            Err(_) => {
                eprintln!(
                    "[demag] resolved method={:?} (material={:?}, env=<unset>), pbc_x={}, pbc_y={}, rayon_threads={}",
                    method, mat.demag_method, pbc_x, pbc_y, current_num_threads()
                );
            }
        }
    });

    match method {
        DemagMethod::FftUniform => {
            demag_fft_uniform::add_demag_field_pbc(grid, m, b_eff, mat, pbc_x, pbc_y)
        }
        DemagMethod::PoissonMG => {
            if pbc_x > 0 || pbc_y > 0 {
                WARN_MG_PBC_FALLBACK.call_once(|| {
                    eprintln!(
                        "[demag] WARN: demag_method=mg does not support PBC (pbc_x={}, pbc_y={}); \
                         falling back to FFT.",
                        pbc_x, pbc_y
                    );
                });
                demag_fft_uniform::add_demag_field_pbc(grid, m, b_eff, mat, pbc_x, pbc_y)
            } else {
                demag_poisson_mg::add_demag_field_poisson_mg(grid, m, b_eff, mat)
            }
        }
        // DST variant retired — route to MG (3D padded box).
        DemagMethod::PoissonDst => {
            static WARN_DST_RETIRED: Once = Once::new();
            WARN_DST_RETIRED.call_once(|| {
                eprintln!(
                    "[demag] INFO: demag_method=dst has been retired. \
                     Using demag_method=mg (3D padded-box MG) instead."
                );
            });
            if pbc_x > 0 || pbc_y > 0 {
                demag_fft_uniform::add_demag_field_pbc(grid, m, b_eff, mat, pbc_x, pbc_y)
            } else {
                demag_poisson_mg::add_demag_field_poisson_mg(grid, m, b_eff, mat)
            }
        }
    }
}

/// Compute demag induction B_demag (Tesla) into `out` (overwrites out), with PBC in x/y.
pub fn compute_demag_field_pbc(
    grid: &Grid2D,
    m: &VectorField2D,
    out: &mut VectorField2D,
    mat: &Material,
    pbc_x: usize,
    pbc_y: usize,
) {
    out.set_uniform(0.0, 0.0, 0.0);
    add_demag_field_pbc(grid, m, out, mat, pbc_x, pbc_y);
}