// src/effective_field/coarse_fft_demag.rs
//
// Coarse-FFT AMR demag: exact Newell-tensor FFT on the L0 grid with
// M-restriction from patches, then bilinear interpolation to patches.
//
// This replaces the composite-grid Poisson approach (mg_composite.rs) with a
// fundamentally simpler and more accurate strategy:
//
//   1. Restrict fine M from all patches onto the coarse (L0) grid via
//      area-weighted averaging (UN-normalised, preserving charge structure).
//   2. Run the existing `demag_fft_uniform` FFT convolution on the L0 grid.
//      This uses the exact 3D Newell tensor — no Poisson reformulation, no
//      boundary integral, no MG iterations.
//   3. Bilinear-sample the resulting coarse B_demag onto each patch cell.
//
// Key insight (from Architecture Direction document):
//   Demag is a convolution (B = N * M), not a Poisson problem.  FFT evaluates
//   the native convolution directly, encoding all finite-thickness physics in
//   the precomputed Newell tensor kernel.  The Poisson reformulation discards
//   the kernel structure and forces reconstruction of open-boundary physics
//   through complex mechanisms.
//
// Why M-restriction rather than ∇·M injection (García-Cervera):
//   The Newell tensor handles ∇·M internally.  Injecting M directly into the
//   FFT source is both simpler and more accurate than computing ∇·M on mixed
//   coarse/fine grids and injecting it into a Poisson RHS.
//
// Performance:
//   For a 192×192 base grid with ~16% patch coverage:
//     all_fft  : FFT on 1536×1536 (padded 3072²) ≈ 29s
//     coarse_fft: FFT on 192×192  (padded 384²)  ≈ 0.3–0.5s + 0.1s interp
//   Expected ~50× speedup over all_fft.
//
// Validation contract:
//   - Zero patches: must reproduce demag_fft_uniform on L0 exactly (bit-identical).
//   - Single patch: coarse-FFT B at L0 cells should differ <1% from all_fft.
//   - Full AMR:     RMSE vs all_fft < 3%, patches at vortex core (not edges).

use crate::amr::hierarchy::AmrHierarchy2D;
use crate::amr::interp::sample_bilinear;
use crate::amr::patch::Patch2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

use super::demag_fft_uniform;

use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Configuration / diagnostics
// ---------------------------------------------------------------------------

#[inline]
fn coarse_fft_diag() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLG_DEMAG_COARSE_FFT_DIAG").is_ok())
}

// ---------------------------------------------------------------------------
// Step 1: Restrict fine M onto the coarse grid (un-normalised)
// ---------------------------------------------------------------------------

/// Restrict patch magnetisation onto the coarse grid WITHOUT renormalising.
///
/// For each coarse cell covered by a patch, the area-weighted average of the
/// `ratio × ratio` fine interior cells replaces the coarse value.  Patches are
/// processed coarsest-to-finest so deeper levels overwrite shallower ones.
///
/// **Unlike** `Patch2D::restrict_to_coarse()`, this does NOT renormalise the
/// averaged vector to unit length.  The un-normalised average preserves the
/// correct magnetic charge distribution (∇·M):
///
/// - At a domain wall where adjacent fine cells have opposing m, the average
///   magnitude |m_avg| < 1, correctly representing partial charge cancellation.
/// - At a vortex core where m rotates rapidly, the average direction and
///   magnitude encode the net winding.
///
/// Renormalising to |m| = 1 would destroy this charge information, which is
/// exactly what made the old composite-MG approach inaccurate.
fn restrict_m_to_coarse(h: &AmrHierarchy2D, coarse_m: &mut VectorField2D) {
    // Start from the current coarse field (already unit-normalised from the
    // last restriction step — this is the baseline for uncovered cells).
    coarse_m.data.clone_from(&h.coarse.data);

    // Closure: restrict a single patch into coarse_m (no normalisation).
    let restrict_patch = |p: &Patch2D, out: &mut VectorField2D| {
        let r = p.ratio;
        let g = p.ghost;
        let patch_cnx = p.coarse_rect.nx;
        let gm = p.geom_mask_fine.as_deref();

        for jc in 0..p.coarse_rect.ny {
            for ic in 0..p.coarse_rect.nx {
                let i_coarse = p.coarse_rect.i0 + ic;
                let j_coarse = p.coarse_rect.j0 + jc;
                let dst = out.idx(i_coarse, j_coarse);

                // Skip vacuum parents.
                if !p.parent_material[jc * patch_cnx + ic] {
                    out.data[dst] = [0.0, 0.0, 0.0];
                    continue;
                }

                // Area-average fine cells covering this coarse cell.
                let mut sum = [0.0_f64; 3];
                let mut n_mat = 0usize;

                for fj in 0..r {
                    for fi in 0..r {
                        let i_f = g + ic * r + fi;
                        let j_f = g + jc * r + fj;
                        let idx_f = p.grid.idx(i_f, j_f);

                        if let Some(mask) = gm {
                            if !mask[idx_f] {
                                continue;
                            }
                        }

                        let v = p.m.data[idx_f];
                        sum[0] += v[0];
                        sum[1] += v[1];
                        sum[2] += v[2];
                        n_mat += 1;
                    }
                }

                if n_mat == 0 {
                    out.data[dst] = [0.0, 0.0, 0.0];
                } else {
                    let inv = 1.0 / (n_mat as f64);
                    // NO normalisation — preserve charge structure.
                    out.data[dst] = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
                }
            }
        }
    };

    // Process coarsest-to-finest: L1, then L2, L3, ... (finer overwrites).
    for p in &h.patches {
        restrict_patch(p, coarse_m);
    }
    for lvl in &h.patches_l2plus {
        for p in lvl {
            restrict_patch(p, coarse_m);
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3: Interpolate coarse B_demag onto patches
// ---------------------------------------------------------------------------

/// Bilinear-sample the coarse B_demag field onto every cell of a patch
/// (including ghost cells, so exchange stencils at patch boundaries see
/// a smooth demag contribution).
fn sample_demag_to_patch(b_coarse: &VectorField2D, patch: &Patch2D) -> Vec<[f64; 3]> {
    let pnx = patch.grid.nx;
    let pny = patch.grid.ny;
    let n = pnx * pny;
    let mut b = vec![[0.0; 3]; n];

    for j in 0..pny {
        for i in 0..pnx {
            let (x, y) = patch.cell_center_xy(i, j);
            b[j * pnx + i] = sample_bilinear(b_coarse, x, y);
        }
    }
    b
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute AMR-aware demag using coarse-FFT: exact Newell-tensor FFT on L0
/// with M-restriction from patches, then bilinear interpolation to patches.
///
/// Returns `(b_patches_l1, b_patches_l2plus)` in the same format as
/// `mg_composite::compute_composite_demag`, so it can be wired into the
/// stepper as a drop-in replacement for the `CompositeGrid` code path.
///
/// # Arguments
/// * `h` — the AMR hierarchy (coarse grid + all patch levels)
/// * `mat` — material parameters (Ms, demag flag, etc.)
/// * `b_demag_coarse` — output: coarse-grid B_demag field (overwritten)
///
/// # Returns
/// * `b_patches_l1[i]` — flat Vec of [f64; 3] for L1 patch `i` (full patch grid incl. ghosts)
/// * `b_patches_l2plus[lvl][i]` — same for L2+ patches
pub fn compute_coarse_fft_demag(
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

    // ------------------------------------------------------------------
    // Step 1: Restrict fine M onto L0 (un-normalised area-weighted avg)
    // ------------------------------------------------------------------
    let mut enhanced_m = VectorField2D::new(h.base_grid);
    restrict_m_to_coarse(h, &mut enhanced_m);

    if coarse_fft_diag() {
        let n = h.base_grid.n_cells();
        let (mag_min, mag_max, mag_sum) = enhanced_m.data[..n].iter().fold(
            (f64::INFINITY, 0.0_f64, 0.0_f64),
            |(mn, mx, sm), v| {
                let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                (mn.min(mag), mx.max(mag), sm + mag)
            },
        );
        eprintln!(
            "[coarse_fft] enhanced M: |m| range [{:.4}, {:.4}], avg={:.4}, \
             grid={}x{}, n_l1={}, n_l2plus={}",
            mag_min,
            mag_max,
            mag_sum / n as f64,
            h.base_grid.nx,
            h.base_grid.ny,
            n_l1,
            n_l2plus,
        );
    }

    // ------------------------------------------------------------------
    // Step 2: FFT demag on the L0 grid with enhanced (restricted) M
    // ------------------------------------------------------------------
    // This is a single call to the existing, validated FFT solver.
    // Zero-padding, Newell tensor, open BCs — all handled internally.
    demag_fft_uniform::compute_demag_field(
        &h.base_grid,
        &enhanced_m,
        b_demag_coarse,
        mat,
    );

    if coarse_fft_diag() {
        let n = h.base_grid.n_cells();
        let bz_max = b_demag_coarse.data[..n]
            .iter()
            .map(|v| v[2].abs())
            .fold(0.0_f64, f64::max);
        let bxy_max = b_demag_coarse.data[..n]
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "[coarse_fft] B_demag coarse: max|Bz|={:.3e} T, max|Bxy|={:.3e} T",
            bz_max, bxy_max,
        );
    }

    // ------------------------------------------------------------------
    // Step 3: Interpolate coarse B_demag onto all patches
    // ------------------------------------------------------------------
    let b_patches_l1: Vec<Vec<[f64; 3]>> = h
        .patches
        .iter()
        .map(|p| sample_demag_to_patch(b_demag_coarse, p))
        .collect();

    let b_patches_l2plus: Vec<Vec<Vec<[f64; 3]>>> = h
        .patches_l2plus
        .iter()
        .map(|lvl| {
            lvl.iter()
                .map(|p| sample_demag_to_patch(b_demag_coarse, p))
                .collect()
        })
        .collect();

    (b_patches_l1, b_patches_l2plus)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amr::hierarchy::AmrHierarchy2D;
    use crate::grid::Grid2D;
    use crate::params::{DemagMethod, Material};
    use crate::vector_field::VectorField2D;

    fn test_material() -> Material {
        Material {
            ms: 8.0e5,
            a_ex: 0.0,
            k_u: 0.0,
            easy_axis: [0.0, 0.0, 1.0],
            dmi: None,
            demag: true,
            demag_method: DemagMethod::FftUniform,
        }
    }

    /// Zero-patch test: coarse-FFT with no patches must reproduce
    /// demag_fft_uniform on L0 bit-identically.
    #[test]
    fn zero_patches_reproduces_fft_exactly() {
        let grid = Grid2D::new(32, 32, 5e-9, 5e-9, 1e-9);
        let mut m = VectorField2D::new(grid);
        // Vortex-like initial state
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let x = (i as f64 + 0.5) / grid.nx as f64 - 0.5;
                let y = (j as f64 + 0.5) / grid.ny as f64 - 0.5;
                let r = (x * x + y * y).sqrt().max(1e-12);
                let idx = grid.idx(i, j);
                m.data[idx] = crate::vec3::normalize([-y / r, x / r, 0.3]);
            }
        }

        let mat = test_material();
        let mut m_copy = VectorField2D::new(grid);
        m_copy.data.copy_from_slice(&m.data);
        let h = AmrHierarchy2D::new(grid, m_copy, 2, 2);

        // Reference: direct FFT on L0
        let mut b_ref = VectorField2D::new(grid);
        demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_ref, &mat);

        // Coarse-FFT with no patches
        let mut b_test = VectorField2D::new(grid);
        let (bl1, bl2) = compute_coarse_fft_demag(&h, &mat, &mut b_test);

        assert!(bl1.is_empty());
        assert!(bl2.is_empty());

        // Must be bit-identical (same input M, same FFT call)
        for idx in 0..grid.n_cells() {
            assert_eq!(
                b_ref.data[idx], b_test.data[idx],
                "mismatch at cell {}: ref={:?}, test={:?}",
                idx, b_ref.data[idx], b_test.data[idx],
            );
        }
    }
}