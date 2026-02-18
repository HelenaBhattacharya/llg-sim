// src/effective_field/demag.rs
//
// Demag dispatcher: selects between multiple demagnetising-field implementations.
//
// This file intentionally contains *no new physics*.
// It routes calls to one of:
//   - demag_fft_uniform.rs  (FFT convolution on a uniform FD grid)
//   - demag_poisson_mg.rs   (Poisson + geometric multigrid; experimental)
//
// The goal is that call sites keep importing `effective_field::demag` and calling
// `add_demag_field(...)` / `compute_demag_field(...)` without caring about the method.
//
// Implementation note:
// - `DemagMethod` is intended to live in `src/params.rs` and be stored on `Material`.
// - Additionally, you can override the method at runtime using `LLG_DEMAG_METHOD`:
//     export LLG_DEMAG_METHOD=fft
//     export LLG_DEMAG_METHOD=mg

use crate::grid::Grid2D;
use crate::params::{DemagMethod, Material};
use crate::vector_field::VectorField2D;

use super::demag_fft_uniform;
use super::demag_poisson_mg;

use rayon::current_num_threads;
use std::sync::{Once, OnceLock};

static WARN_MG_PBC_FALLBACK: Once = Once::new();
static PRINT_DEMAG_METHOD_ONCE: Once = Once::new();

// Cache the parsed environment override so hot loops (e.g. SP2 demag calls) don't
// pay a getenv/parse cost every field evaluation.
//
// NOTE: This is intentionally process-wide. Set LLG_DEMAG_METHOD before running.
static DEMAG_METHOD_OVERRIDE: OnceLock<Option<DemagMethod>> = OnceLock::new();

/// Resolve the demag method for this run.
///
/// Priority:
/// 1) `LLG_DEMAG_METHOD` environment override (if present)
/// 2) `mat.demag_method`
#[inline]
pub fn resolved_demag_method(mat: &Material) -> DemagMethod {
    // Read + parse once per process.
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
/// *MG method*: not PBC-aware yet; if `pbc_x>0 || pbc_y>0`, we fall back to FFT.
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
        // Print once per process to avoid polluting hot-loop output.
        // This helps confirm which backend is actually being used.
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
                    method,
                    mat.demag_method,
                    raw,
                    pbc_x,
                    pbc_y,
                    current_num_threads()
                );
            }
            Err(_) => {
                eprintln!(
                    "[demag] resolved method={:?} (material={:?}, env=<unset>), pbc_x={}, pbc_y={}, rayon_threads={}",
                    method,
                    mat.demag_method,
                    pbc_x,
                    pbc_y,
                    current_num_threads()
                );
            }
        }
    });

    match method {
        DemagMethod::FftUniform => {
            demag_fft_uniform::add_demag_field_pbc(grid, m, b_eff, mat, pbc_x, pbc_y)
        }
        DemagMethod::PoissonMG => {
            // MG implementation is not PBC-aware yet.
            // If periodic images are requested, fall back to FFT to preserve semantics.
            if pbc_x > 0 || pbc_y > 0 {
                WARN_MG_PBC_FALLBACK.call_once(|| {
                    eprintln!(
                        "[llg-sim] WARN: demag_method=mg does not support PBC (pbc_x={}, pbc_y={}); falling back to FFT.",
                        pbc_x, pbc_y
                    );
                });
                demag_fft_uniform::add_demag_field_pbc(grid, m, b_eff, mat, pbc_x, pbc_y)
            } else {
                demag_poisson_mg::add_demag_field_poisson_mg(grid, m, b_eff, mat)
            }
        }
    }
}

/// Compute demag induction B_demag (Tesla) into `out` (overwrites out), with PBC in x/y.
///
/// *FFT method*: supports MuMax-style finite-image PBC sums.
/// *MG method*: not PBC-aware yet; if `pbc_x>0 || pbc_y>0`, we fall back to FFT.
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
