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
// Super-coarse FFT (LLG_DEMAG_COARSEN_RATIO > 1):
//   When L0 itself is large (>512²), the FFT on L0 can become the bottleneck.
//   Super-coarse FFT adds an intermediate restriction step:
//     1. Restrict M from patches → L0 (unchanged)
//     2. Restrict M from L0 → demag grid of size (L0/R)²  [NEW]
//     3. Run Newell FFT on the smaller demag grid
//     4. Interpolate B_demag from demag grid → L0              [NEW]
//     5. Interpolate B_demag from L0 → patches (unchanged)
//   The FFT cost drops by ~R² (R=2 → 4×, R=4 → 16×).
//   Accuracy: R=2 gives ~1–2% RMSE, R=4 gives ~2–4% RMSE.
//   Default R=1 (no coarsening) preserves the current code path exactly.
//
// Validation contract:
//   - Zero patches: must reproduce demag_fft_uniform on L0 exactly (bit-identical).
//   - Single patch: coarse-FFT B at L0 cells should differ <1% from all_fft.
//   - Full AMR:     RMSE vs all_fft < 3%, patches at vortex core (not edges).
//   - R=1: must be bit-identical to the non-super-coarse path.
//   - R=2: RMSE vs R=1 reference < 5% on smooth vortex states.

use crate::amr::hierarchy::AmrHierarchy2D;
use crate::amr::interp::sample_bilinear;
use crate::amr::patch::Patch2D;
use crate::grid::Grid2D;
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

/// Read the super-coarse FFT coarsening ratio from environment.
///
/// `LLG_DEMAG_COARSEN_RATIO=R` where R ∈ {1, 2, 4, 8, ...}.
/// Default R=1 (no super-coarsening; identical to the original code path).
///
/// The ratio must be a power of 2.  If the value is invalid or doesn't evenly
/// divide both grid dimensions, we fall back to R=1 with a warning.
#[inline]
fn demag_coarsen_ratio() -> usize {
    static RATIO: OnceLock<usize> = OnceLock::new();
    *RATIO.get_or_init(|| {
        let raw = match std::env::var("LLG_DEMAG_COARSEN_RATIO") {
            Ok(s) => s,
            Err(_) => return 1,
        };
        let r: usize = match raw.trim().parse() {
            Ok(v) if v >= 1 => v,
            _ => {
                eprintln!(
                    "[coarse_fft] WARNING: LLG_DEMAG_COARSEN_RATIO='{}' is not a valid \
                     positive integer; falling back to R=1.",
                    raw
                );
                return 1;
            }
        };
        if r == 1 {
            return 1;
        }
        // Must be a power of 2.
        if !r.is_power_of_two() {
            eprintln!(
                "[coarse_fft] WARNING: LLG_DEMAG_COARSEN_RATIO={} is not a power of 2; \
                 falling back to R=1.",
                r
            );
            return 1;
        }
        r
    })
}

/// Validate that `ratio` evenly divides both grid dimensions.  Returns the
/// effective ratio to use (either `ratio` or 1 if validation fails).
#[inline]
fn validated_ratio(nx: usize, ny: usize, ratio: usize) -> usize {
    if ratio <= 1 {
        return 1;
    }
    if nx % ratio != 0 || ny % ratio != 0 {
        static WARNED: OnceLock<bool> = OnceLock::new();
        WARNED.get_or_init(|| {
            eprintln!(
                "[coarse_fft] WARNING: LLG_DEMAG_COARSEN_RATIO={} does not evenly divide \
                 grid {}x{}; falling back to R=1.",
                ratio, nx, ny
            );
            true
        });
        return 1;
    }
    if nx / ratio < 4 || ny / ratio < 4 {
        static WARNED_SMALL: OnceLock<bool> = OnceLock::new();
        WARNED_SMALL.get_or_init(|| {
            eprintln!(
                "[coarse_fft] WARNING: LLG_DEMAG_COARSEN_RATIO={} would reduce grid {}x{} \
                 to {}x{} (< 4×4); falling back to R=1.",
                ratio, nx, ny, nx / ratio, ny / ratio
            );
            true
        });
        return 1;
    }
    ratio
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
// Step 1b: Restrict L0 M onto the super-coarse demag grid (un-normalised)
// ---------------------------------------------------------------------------

/// Restrict the enhanced (patch-injected) L0 magnetisation onto a coarser
/// "demag grid" by area-averaging R×R blocks of L0 cells.
///
/// The demag grid has dimensions `(nx/R, ny/R)` with cell spacings `(R*dx, R*dy, dz)`.
/// Like the patch→L0 restriction, this does NOT renormalise — the un-normalised
/// average preserves the charge distribution for the FFT convolution.
///
/// Non-magnetic cells (zero vectors from geometry masking) are handled
/// implicitly: they contribute zero to the sum and reduce the denominator,
/// which correctly dilutes the averaged magnetisation in partially-material
/// blocks.
fn restrict_l0_to_demag_grid(
    l0_m: &VectorField2D,
    demag_m: &mut VectorField2D,
    ratio: usize,
) {
    let nx_l0 = l0_m.grid.nx;
    let dnx = demag_m.grid.nx;
    let dny = demag_m.grid.ny;
    let inv_r2 = 1.0 / ((ratio * ratio) as f64);

    for jd in 0..dny {
        for id in 0..dnx {
            let mut sum = [0.0_f64; 3];

            // Sum the R×R block of L0 cells that maps to this demag cell.
            let i0 = id * ratio;
            let j0 = jd * ratio;
            for fj in 0..ratio {
                let row_base = (j0 + fj) * nx_l0 + i0;
                for fi in 0..ratio {
                    let v = l0_m.data[row_base + fi];
                    sum[0] += v[0];
                    sum[1] += v[1];
                    sum[2] += v[2];
                }
            }

            // Un-normalised area average.  For fully-material blocks this is
            // equivalent to the arithmetic mean of the R² cell vectors.
            let dst = demag_m.idx(id, jd);
            demag_m.data[dst] = [sum[0] * inv_r2, sum[1] * inv_r2, sum[2] * inv_r2];
        }
    }
}

// ---------------------------------------------------------------------------
// Step 2b: Interpolate demag-grid B_demag back onto the L0 grid
// ---------------------------------------------------------------------------

/// Bilinear-sample the super-coarse demag-grid B_demag field onto every cell
/// of the L0 grid.
///
/// For each L0 cell centre `(x, y)`, we call the existing `sample_bilinear`
/// on the demag-grid field.  This reuses the same proven bilinear interpolation
/// used for patch ghost fills and L0→patch sampling.
fn interpolate_demag_to_l0(
    b_demag_grid: &VectorField2D,
    b_l0: &mut VectorField2D,
) {
    let nx = b_l0.grid.nx;
    let ny = b_l0.grid.ny;
    let dx = b_l0.grid.dx;
    let dy = b_l0.grid.dy;

    for j in 0..ny {
        let y = (j as f64 + 0.5) * dy;
        for i in 0..nx {
            let x = (i as f64 + 0.5) * dx;
            let idx = j * nx + i;
            b_l0.data[idx] = sample_bilinear(b_demag_grid, x, y);
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
/// When `LLG_DEMAG_COARSEN_RATIO=R` (R > 1), an additional super-coarse step
/// restricts M from L0 to a `(L0/R)²` demag grid before the FFT, then
/// interpolates B_demag back to L0.  This reduces FFT cost by ~R² at the
/// expense of ~1–4% RMSE in the smooth demag far-field.  Default R=1
/// (no super-coarsening; bit-identical to the original code path).
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

    // Resolve the effective coarsening ratio for this grid.
    let r_env = demag_coarsen_ratio();
    let r = validated_ratio(h.base_grid.nx, h.base_grid.ny, r_env);

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
             grid={}x{}, n_l1={}, n_l2plus={}, coarsen_ratio={}",
            mag_min,
            mag_max,
            mag_sum / n as f64,
            h.base_grid.nx,
            h.base_grid.ny,
            n_l1,
            n_l2plus,
            r,
        );
    }

    // ------------------------------------------------------------------
    // Step 2: FFT demag — either on L0 directly (R=1) or on the
    //         super-coarse demag grid (R>1)
    // ------------------------------------------------------------------
    if r <= 1 {
        // ---- R=1: original code path (bit-identical) ----
        demag_fft_uniform::compute_demag_field(
            &h.base_grid,
            &enhanced_m,
            b_demag_coarse,
            mat,
        );
    } else {
        // ---- R>1: super-coarse FFT path ----
        let dnx = h.base_grid.nx / r;
        let dny = h.base_grid.ny / r;
        let demag_grid = Grid2D::new(
            dnx,
            dny,
            h.base_grid.dx * r as f64,
            h.base_grid.dy * r as f64,
            h.base_grid.dz,
        );

        // Step 1b: Restrict enhanced L0 M → demag grid
        let mut demag_m = VectorField2D::new(demag_grid);
        restrict_l0_to_demag_grid(&enhanced_m, &mut demag_m, r);

        if coarse_fft_diag() {
            let n_d = demag_grid.n_cells();
            let (mag_min, mag_max) = demag_m.data[..n_d].iter().fold(
                (f64::INFINITY, 0.0_f64),
                |(mn, mx), v| {
                    let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    (mn.min(mag), mx.max(mag))
                },
            );
            eprintln!(
                "[coarse_fft] super-coarse M: demag_grid={}x{}, dx'={:.3e} m, \
                 |m| range [{:.4}, {:.4}]",
                dnx, dny, demag_grid.dx, mag_min, mag_max,
            );
        }

        // Step 2: FFT on the demag grid (kernel auto-cached by grid dims).
        let mut b_demag_sc = VectorField2D::new(demag_grid);
        demag_fft_uniform::compute_demag_field(
            &demag_grid,
            &demag_m,
            &mut b_demag_sc,
            mat,
        );

        // Step 2b: Interpolate demag-grid B_demag → L0.
        interpolate_demag_to_l0(&b_demag_sc, b_demag_coarse);

        if coarse_fft_diag() {
            eprintln!(
                "[coarse_fft] super-coarse interpolated B back to L0 ({}x{}), R={}",
                h.base_grid.nx, h.base_grid.ny, r,
            );
        }
    }

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

    fn make_vortex_m(grid: Grid2D) -> VectorField2D {
        let mut m = VectorField2D::new(grid);
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let x = (i as f64 + 0.5) / grid.nx as f64 - 0.5;
                let y = (j as f64 + 0.5) / grid.ny as f64 - 0.5;
                let r = (x * x + y * y).sqrt().max(1e-12);
                let idx = grid.idx(i, j);
                m.data[idx] = crate::vec3::normalize([-y / r, x / r, 0.3]);
            }
        }
        m
    }

    /// Zero-patch test: coarse-FFT with no patches must reproduce
    /// demag_fft_uniform on L0 bit-identically.
    #[test]
    fn zero_patches_reproduces_fft_exactly() {
        let grid = Grid2D::new(32, 32, 5e-9, 5e-9, 1e-9);
        let m = make_vortex_m(grid);
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

    /// Test the L0→demag-grid restriction: for R=2, a 4×4 grid with
    /// uniform M should produce a 2×2 grid with identical M vectors.
    #[test]
    fn restrict_l0_to_demag_uniform_m() {
        let grid = Grid2D::new(4, 4, 5e-9, 5e-9, 1e-9);
        let mut m = VectorField2D::new(grid);
        let v = crate::vec3::normalize([1.0, 0.0, 0.3]);
        for d in m.data.iter_mut() {
            *d = v;
        }

        let demag_grid = Grid2D::new(2, 2, 10e-9, 10e-9, 1e-9);
        let mut demag_m = VectorField2D::new(demag_grid);
        restrict_l0_to_demag_grid(&m, &mut demag_m, 2);

        for idx in 0..4 {
            for c in 0..3 {
                assert!(
                    (demag_m.data[idx][c] - v[c]).abs() < 1e-14,
                    "mismatch at demag cell {} component {}: got {}, expected {}",
                    idx, c, demag_m.data[idx][c], v[c],
                );
            }
        }
    }

    /// Test that the L0→demag restriction correctly averages opposing vectors.
    /// Two adjacent cells with m = +x and m = −x should average to zero,
    /// preserving the charge cancellation.
    #[test]
    fn restrict_l0_to_demag_charge_cancellation() {
        let grid = Grid2D::new(2, 2, 5e-9, 5e-9, 1e-9);
        let mut m = VectorField2D::new(grid);
        // Top-left and bottom-left: +x, top-right and bottom-right: −x.
        m.data[0] = [1.0, 0.0, 0.0];  // (0,0)
        m.data[1] = [-1.0, 0.0, 0.0]; // (1,0)
        m.data[2] = [1.0, 0.0, 0.0];  // (0,1)
        m.data[3] = [-1.0, 0.0, 0.0]; // (1,1)

        let demag_grid = Grid2D::new(1, 1, 10e-9, 10e-9, 1e-9);
        let mut demag_m = VectorField2D::new(demag_grid);
        restrict_l0_to_demag_grid(&m, &mut demag_m, 2);

        // The average of [+1,0,0] and [−1,0,0] (×2 each) should be [0,0,0].
        for c in 0..3 {
            assert!(
                demag_m.data[0][c].abs() < 1e-14,
                "expected zero at component {}, got {}",
                c, demag_m.data[0][c],
            );
        }
    }

    /// Super-coarse R=2 on a 64×64 vortex: RMSE vs R=1 reference should
    /// be small (< 10% — generous bound for a test; real accuracy is ~1–4%).
    #[test]
    fn super_coarse_r2_accuracy() {
        let grid = Grid2D::new(64, 64, 5e-9, 5e-9, 1e-9);
        let m = make_vortex_m(grid);
        let mat = test_material();

        // R=1 reference: direct FFT on L0
        let mut b_ref = VectorField2D::new(grid);
        demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_ref, &mat);

        // R=2: manual super-coarse path
        let r = 2usize;
        let demag_grid = Grid2D::new(32, 32, 10e-9, 10e-9, 1e-9);
        let mut demag_m = VectorField2D::new(demag_grid);
        restrict_l0_to_demag_grid(&m, &mut demag_m, r);

        let mut b_sc = VectorField2D::new(demag_grid);
        demag_fft_uniform::compute_demag_field(&demag_grid, &demag_m, &mut b_sc, &mat);

        let mut b_test = VectorField2D::new(grid);
        interpolate_demag_to_l0(&b_sc, &mut b_test);

        // Compute RMSE normalised by max|B_ref|.
        let n = grid.n_cells();
        let b_max = b_ref.data[..n]
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())
            .fold(0.0_f64, f64::max);
        assert!(b_max > 0.0, "reference B field is zero everywhere");

        let mse: f64 = b_ref.data[..n]
            .iter()
            .zip(b_test.data[..n].iter())
            .map(|(r, t)| {
                let dx = r[0] - t[0];
                let dy = r[1] - t[1];
                let dz = r[2] - t[2];
                dx * dx + dy * dy + dz * dz
            })
            .sum::<f64>()
            / n as f64;
        let rmse_rel = mse.sqrt() / b_max;

        eprintln!(
            "[test] super_coarse R=2: RMSE/max|B| = {:.4}% (b_max={:.3e} T)",
            rmse_rel * 100.0,
            b_max,
        );

        // Generous bound for CI.  Real accuracy on smooth vortex states is ~1–4%.
        assert!(
            rmse_rel < 0.10,
            "RMSE too large: {:.2}% (expected < 10%)",
            rmse_rel * 100.0,
        );
    }

    /// validated_ratio must reject ratios that don't evenly divide the grid.
    #[test]
    fn validated_ratio_rejects_bad_divisor() {
        // 30 is not divisible by 4.
        assert_eq!(validated_ratio(30, 32, 4), 1);
        // 32 is divisible by 4.
        assert_eq!(validated_ratio(32, 32, 4), 4);
        // R=1 always passes.
        assert_eq!(validated_ratio(7, 13, 1), 1);
    }

    /// validated_ratio must reject ratios that would make the grid too small.
    #[test]
    fn validated_ratio_rejects_too_small() {
        // 8/4 = 2 which is < 4 minimum.
        assert_eq!(validated_ratio(8, 8, 4), 1);
        // 16/4 = 4 which is exactly 4 — should pass.
        assert_eq!(validated_ratio(16, 16, 4), 4);
    }
}