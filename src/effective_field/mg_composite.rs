// src/effective_field/mg_composite.rs
//
// Component 1: Composite-grid Poisson wrapper for AMR-aware demagnetisation.
//
// Implements the García-Cervera & Roma (2005) approach adapted for our thin-film
// multigrid solver.  Instead of flattening the entire AMR hierarchy to a single
// uniform fine grid (which defeats AMR efficiency), we:
//
//   1. Build the divergence RHS on each AMR level independently.
//   2. *Enhance* the L0 (coarse) RHS by restricting the fine-level divergence
//      into the coarse cells covered by patches — giving the L0 solver the
//      benefit of fine-resolution charge distribution.
//   3. Run the existing DemagPoissonMG V-cycle on L0 (3D padded box with
//      treecode/dipole BCs).  This captures all long-range interactions.
//   4. Extract H_demag on L0 for coarse cells.
//   5. For patch cells, interpolate L0's φ (or H_demag) to patch resolution.
//
// This is the "enhanced-RHS" variant of the composite-grid approach.  It is the
// minimum-viable composite step that:
//   - Runs the MG solve on the *coarse* grid (not the flattened fine grid)
//   - Gets fine-resolution RHS accuracy in patch regions
//   - Scales as O(N_coarse) not O(N_fine) for the expensive V-cycle
//
// Future additions (Components 2–3):
//   - Per-patch near-field PPPM/Ewald ΔK correction (Component 2)
//   - Far-field subtraction at coarse–fine interfaces (Component 3)
//
// Validation contract:
//   Zero patches (L0 only) must reproduce DemagPoissonMG bit-exactly.
//   This is enforced by Test 1 in the validation hierarchy.

use crate::amr::hierarchy::AmrHierarchy2D;
use crate::amr::interp::sample_bilinear;
use crate::amr::patch::Patch2D;
use crate::grid::Grid2D;
use crate::params::Material;
use crate::vector_field::VectorField2D;

use super::demag_poisson_mg::{DemagPoissonMG, DemagPoissonMGConfig};
use super::mg_kernels as kernels;

use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Configuration (env-var controlled)
// ---------------------------------------------------------------------------

/// Whether to print diagnostics during composite-grid operations.
#[inline]
fn composite_diag() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLG_DEMAG_COMPOSITE_DIAG").is_ok())
}

// ---------------------------------------------------------------------------
// Composite-grid Poisson solver
// ---------------------------------------------------------------------------

/// AMR-aware composite-grid demag Poisson solver.
///
/// Wraps the existing `DemagPoissonMG` for the L0 (base-grid) solve and
/// enhances its RHS with fine-resolution divergence from AMR patches.
pub(crate) struct CompositeGridPoisson {
    /// L0 solver — the standard 3D padded-box multigrid with treecode BC.
    mg: DemagPoissonMG,
}

impl CompositeGridPoisson {
    /// Create a new composite solver from the base grid and MG config.
    pub(crate) fn new(base_grid: Grid2D, cfg: DemagPoissonMGConfig) -> Self {
        let mg = DemagPoissonMG::new(base_grid, cfg);
        Self { mg }
    }

    /// Check if the existing solver can be reused for the given hierarchy.
    pub(crate) fn same_structure(
        &self,
        h: &AmrHierarchy2D,
        _cfg: &DemagPoissonMGConfig,
    ) -> bool {
        let g = self.mg.base_grid();
        g.nx == h.base_grid.nx
            && g.ny == h.base_grid.ny
            && g.dx == h.base_grid.dx
            && g.dy == h.base_grid.dy
            && g.dz == h.base_grid.dz
    }

    // -----------------------------------------------------------------------
    // Step 1: Build the L0 RHS from coarse M
    // -----------------------------------------------------------------------

    fn build_l0_rhs(&mut self, coarse_m: &VectorField2D, ms: f64) {
        self.mg.build_rhs_from_m(coarse_m, ms);
    }

    // -----------------------------------------------------------------------
    // Step 2: Enhance L0 RHS with fine-level divergence from patches
    // -----------------------------------------------------------------------

    /// Restrict the fine-level ∇·M from a single patch onto the L0 RHS,
    /// *replacing* the coarse-resolution divergence in the covered region.
    ///
    /// For each coarse cell (ci, cj) covered by the patch, we compute
    /// ∇·M at fine resolution on the ratio×ratio sub-block and average
    /// it down.  This overwrites the coarse ∇·M in that cell with a
    /// more accurate value.
    fn enhance_rhs_from_patch(
        &mut self,
        patch: &Patch2D,
        ratio: usize,
        ms: f64,
    ) {
        let (px, py, _pz) = self.mg.padded_dims();
        let (offx, offy, offz) = self.mg.magnet_offsets();

        let rect = patch.coarse_rect;
        let g = patch.ghost;
        let dx_fine = patch.grid.dx;
        let dy_fine = patch.grid.dy;
        let inv_2dx = 1.0 / (2.0 * dx_fine);
        let inv_2dy = 1.0 / (2.0 * dy_fine);
        let inv_r2 = 1.0 / (ratio * ratio) as f64;
        let pnx = patch.grid.nx;

        let rhs = self.mg.finest_rhs_mut();

        // For each coarse cell covered by this patch
        for jc in 0..rect.ny {
            let cj = rect.j0 + jc;
            for ic in 0..rect.nx {
                let ci = rect.i0 + ic;

                // Average fine-level ∇·M over the ratio × ratio sub-block
                let mut div_sum = 0.0;
                for fj in 0..ratio {
                    let pj = g + jc * ratio + fj;
                    for fi in 0..ratio {
                        let pi = g + ic * ratio + fi;
                        let idx = pj * pnx + pi;

                        // Central-difference ∇·M at fine resolution
                        let mx_p = ms * patch.m.data[pj * pnx + (pi + 1)][0];
                        let mx_m = if pi > 0 {
                            ms * patch.m.data[pj * pnx + (pi - 1)][0]
                        } else {
                            ms * patch.m.data[idx][0]
                        };
                        let my_p = ms * patch.m.data[(pj + 1) * pnx + pi][1];
                        let my_m = if pj > 0 {
                            ms * patch.m.data[(pj - 1) * pnx + pi][1]
                        } else {
                            ms * patch.m.data[idx][1]
                        };

                        let div_m = (mx_p - mx_m) * inv_2dx + (my_p - my_m) * inv_2dy;
                        div_sum += div_m;
                    }
                }

                // Replace the coarse-grid RHS at this cell.  The existing
                // L0 build_rhs_from_m has already set a coarse-resolution
                // value here; we overwrite with the fine-level average.
                let pad_i = offx + ci;
                let pad_j = offy + cj;
                let pad_k = offz;
                let pad_idx = kernels::idx3(pad_i, pad_j, pad_k, px, py);
                rhs[pad_idx] = div_sum * inv_r2;
            }
        }
    }

    /// Enhance the L0 RHS with fine-level divergence from ALL patches.
    fn enhance_rhs_all_patches(&mut self, h: &AmrHierarchy2D, ms: f64) {
        // L1 patches
        for p in &h.patches {
            self.enhance_rhs_from_patch(p, h.ratio, ms);
        }

        // L2+ patches (overwrite L1's contribution where they overlap —
        // finer levels are more accurate)
        for (k, lvl) in h.patches_l2plus.iter().enumerate() {
            let level = k + 2;
            let r_total = {
                let mut r = 1usize;
                for _ in 0..level {
                    r *= h.ratio;
                }
                r
            };
            for p in lvl {
                self.enhance_rhs_from_patch(p, r_total, ms);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Solve
    // -----------------------------------------------------------------------

    fn solve_l0(&mut self) {
        self.mg.solve();
    }

    // -----------------------------------------------------------------------
    // Step 4: Extract H_demag on coarse grid
    // -----------------------------------------------------------------------

    fn extract_coarse_demag(&mut self, b_out: &mut VectorField2D) {
        b_out.set_uniform(0.0, 0.0, 0.0);
        self.mg.add_b_from_phi_on_magnet_layer_all(b_out);
    }

    // -----------------------------------------------------------------------
    // Step 5: Sample H_demag onto patches via interpolation of L0 field
    // -----------------------------------------------------------------------

    /// Sample the coarse-grid demag field onto a patch's cells using bilinear
    /// interpolation.  Returns a flat array of [Bx, By, Bz] for the entire
    /// patch grid (including ghosts, which get the interpolated value).
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
    // Public API
    // -----------------------------------------------------------------------

    /// Full composite-grid demag computation.
    ///
    /// Writes the coarse-grid demag into `b_demag_coarse`.
    /// Returns per-patch sampled demag fields: (L1 patches, L2+ patches).
    ///
    /// Each inner `Vec<[f64; 3]>` has length = patch.grid.nx * patch.grid.ny
    /// (full patch grid including ghosts).
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

        let ms = mat.ms;

        // Step 1: Build L0 RHS from coarse M
        self.build_l0_rhs(&h.coarse, ms);

        // Step 2: Enhance L0 RHS with fine-level divergence from patches
        //         (skipped when no patches — gives zero-patch regression)
        if !h.patches.is_empty() || h.patches_l2plus.iter().any(|v| !v.is_empty()) {
            self.enhance_rhs_all_patches(h, ms);

            if composite_diag() {
                let (px, py, pz) = self.mg.padded_dims();
                let rhs = self.mg.finest_rhs_mut();
                let rhs_max: f64 = rhs.iter().map(|v| v.abs()).fold(0.0, f64::max);
                eprintln!(
                    "[composite] enhanced RHS: grid={}×{}×{} max_rhs={:.3e} n_l1_patches={} n_l2plus={}",
                    px, py, pz, rhs_max,
                    h.patches.len(),
                    h.patches_l2plus.iter().map(|v| v.len()).sum::<usize>(),
                );
            }
        }

        // Step 3: Solve L0 (full 3D V-cycle with treecode BC)
        self.solve_l0();

        // Step 4: Extract H_demag on coarse grid
        self.extract_coarse_demag(b_demag_coarse);

        // Step 5: Sample coarse demag onto patches
        let b_patches_l1: Vec<Vec<[f64; 3]>> = h.patches.iter()
            .map(|p| Self::sample_demag_to_patch(b_demag_coarse, p))
            .collect();

        let b_patches_l2plus: Vec<Vec<Vec<[f64; 3]>>> = h.patches_l2plus.iter()
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
// Module-level cache + public API (matches demag_poisson_mg pattern)
// ---------------------------------------------------------------------------

use std::sync::Mutex;

static COMPOSITE_CACHE: OnceLock<Mutex<Option<CompositeGridPoisson>>> = OnceLock::new();

/// Compute AMR-aware demag using the composite-grid Poisson solver.
///
/// Called from the stepper's `CompositeGrid` mode.  Writes:
///   - `b_demag_coarse`: coarse-grid demag field
/// Returns:
///   - per-L1-patch B arrays (full patch grid including ghosts)
///   - per-L2+-patch B arrays (nested by level then patch)
pub(crate) fn compute_composite_demag(
    h: &AmrHierarchy2D,
    mat: &Material,
    cfg: &DemagPoissonMGConfig,
    b_demag_coarse: &mut VectorField2D,
) -> (Vec<Vec<[f64; 3]>>, Vec<Vec<Vec<[f64; 3]>>>) {
    let cache = COMPOSITE_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().expect("COMPOSITE_CACHE mutex poisoned");

    let rebuild = match guard.as_ref() {
        Some(s) => !s.same_structure(h, cfg),
        None => true,
    };

    if rebuild {
        *guard = Some(CompositeGridPoisson::new(h.base_grid, *cfg));
    }

    guard.as_mut().unwrap().compute(h, mat, b_demag_coarse)
}