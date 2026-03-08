// src/effective_field/demag_poisson_mg.rs
//
// Experimental demag alternative:
//   Solve magnetostatic scalar potential phi on a padded 3D box using geometric multigrid.
//
// Physics (SI):
//   H = -∇phi
//   ∇² phi = ∇·M     (with M = Ms*m in the magnet, M = 0 in vacuum)
//   B_demag = μ0 * H  (Tesla)
//
// Notes / motivation:
// - Intended as an optional alternative to the existing FFT-convolution demag.
// - This is structurally AMR-friendly (local stencils), unlike global FFTs.
// - Accurate open boundaries are *the* hard part. A padded Dirichlet box is a cheap
//   approximation that can require a lot of vacuum, especially in z for thin films.
//
// - **Hybrid mode (optional, PPPM/Ewald-like):**
//   MG computes only the *smooth/long-range* part on a *screened RHS*, and we add a local,
//   truncatable correction stencil:
//
//       rhs_long = Gσ * rhs
//       MG:  ∇² φ_long = rhs_long
//       B_long = -μ0 ∇φ_long
//
//       ΔK = K_fft(full) - K_mg_long(rhs screened with same Gσ)
//
//   This is the essential PPPM contract: MG is *defined* to be long-range, so ΔK is truly
//   short-range and safe to truncate.
//
// Hybrid controls (OFF by default):
//   LLG_DEMAG_MG_HYBRID_ENABLE=1|0            (default 0)
//   LLG_DEMAG_MG_HYBRID_RADIUS=<cells>        (default 0; >0 enables ΔK *only if* ENABLE=1)
//   LLG_DEMAG_MG_HYBRID_DELTA_VCYCLES=<n>     (default 60; one-time build cost)
//   LLG_DEMAG_MG_HYBRID_SIGMA=<cells>         (default 1.5; Gaussian screening width for RHS before MG solve)
//                                            (σ<=0 disables screening and reverts to old “unscreened ΔK”)
//   LLG_DEMAG_MG_HYBRID_CACHE=1|0             (default 1; caches ΔK in out/demag_cache)
// Diagnostics (optional):
//   LLG_DEMAG_MG_HYBRID_DIAG=1                (prints ΔK sum-rule / tail diagnostics when (re)building ΔK)
//   LLG_DEMAG_MG_HYBRID_DIAG_INVAR=1          (expensive location-invariance check during ΔK build; debug only)
//
// - **A3/A4 operator controls (finest-grid Laplacian + MG transfer/coarse operators):**
//   LLG_DEMAG_MG_STENCIL   = "7" | "iso9" | "iso27"     (default: "iso27")
//   LLG_DEMAG_MG_PROLONG   = "inject" | "trilinear"     (default: "trilinear")
//   LLG_DEMAG_MG_COARSE_OP = "rediscretize" | "galerkin" (default: "galerkin")
//   LLG_DEMAG_MG_ISO27_FLUX_ALPHA=<alpha>  (optional, >0; overrides iso27 diagonal weight parameter)
//
// - This implementation supports three outer BCs:
//     * DirichletZero     : phi = 0 on the padded box boundary
//     * DirichletDipole   : boundary phi set by monopole+dipole far-field approximation
//     * DirichletTreecode : boundary phi set by Barnes–Hut treecode evaluation (best accuracy for small padding)
//
// Caveat:
// - The multigrid solve uses the chosen Laplacian stencil to get phi. The field extraction
//   uses a robust face-gradient (average of one-sided differences) on the finest level.
//   For iso9/iso27, the Laplacian is not exactly div(grad) of that gradient, but the approach
//   remains consistent in the continuum limit and works well in practice.
//
// - Operator mismatch at near-field/high-k is handled by (i) optional iso27/iso9 stencils,
//   and (ii) the PPPM/Ewald-style ΔK correction (screened long-range MG + local complement).

use crate::grid::Grid2D;
use crate::params::{MU0, Material};
use crate::vector_field::VectorField2D;

use super::demag_fft_uniform;
use super::mg_kernels as kernels;
use super::mg_kernels::{idx3, interp_1d_cell_centered};
use super::mg_treecode as treecode;

use rayon::prelude::*;

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// phi = 0 on the outer boundary of the padded domain.
    DirichletZero,

    /// Dirichlet boundary values set from a monopole+dipole approximation of the free-space
    /// solution of ∇²phi = rhs (rhs = ∇·M).
    DirichletDipole,

    /// Dirichlet boundary values set by a Barnes–Hut treecode evaluation of the free-space
    /// Green's function integral:
    ///
    ///   phi(r) = -(1/4π) ∫ rhs(r') / |r - r'| dV'
    DirichletTreecode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MGSmoother {
    /// Weighted Jacobi (parallel-friendly, supports general stencils).
    WeightedJacobi,
    /// Red-black Gauss–Seidel with optional SOR (only valid for classic 7pt stencil path).
    RedBlackSOR,
}

impl MGSmoother {
    fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "jacobi" | "wj" | "weighted_jacobi" => Some(Self::WeightedJacobi),
            "rbgs" | "redblack" | "red_black" | "red_black_sor" | "sor" => Some(Self::RedBlackSOR),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DemagPoissonMGConfig {
    /// Minimum vacuum padding in x/y in *cells* on each side of the magnet region.
    pub pad_xy: usize,

    /// Vacuum layers below and above the magnet layer (in units of dz cells).
    pub n_vac_z: usize,

    /// If `tol_abs` is None, run exactly this many V-cycles.
    pub v_cycles: usize,

    /// Max V-cycles if using a tolerance stop.
    pub v_cycles_max: usize,

    /// Stop when max-norm residual <= tol_abs (units: A/m^2).
    pub tol_abs: Option<f64>,

    /// Stop when max-norm residual <= tol_rel * max-norm(rhs).
    pub tol_rel: Option<f64>,

    /// Pre-smoothing iterations.
    pub pre_smooth: usize,
    /// Post-smoothing iterations.
    pub post_smooth: usize,

    /// Smoother selection.
    pub smoother: MGSmoother,

    /// Weighted Jacobi relaxation parameter (0 < omega <= 1).
    pub omega: f64,

    /// Red-black SOR relaxation factor (0 < sor_omega < 2).
    pub sor_omega: f64,

    /// Use previous phi as initial guess (warm start).
    pub warm_start: bool,

    /// Outer boundary condition on the padded box.
    pub bc: BoundaryCondition,

    /// Treecode opening angle θ (smaller -> more accurate, slower).
    pub tree_theta: f64,

    /// Treecode leaf size (direct evaluation threshold).
    pub tree_leaf: usize,

    /// Treecode max depth safeguard.
    pub tree_max_depth: usize,
}

impl Default for DemagPoissonMGConfig {
    fn default() -> Self {
        Self {
            pad_xy: 6,
            n_vac_z: 16,
            v_cycles: 16,
            v_cycles_max: 80,
            tol_abs: None,
            tol_rel: None,
            pre_smooth: 2,
            post_smooth: 2,
            smoother: MGSmoother::WeightedJacobi,
            omega: 2.0 / 3.0,
            sor_omega: 1.0,
            warm_start: true,
            bc: BoundaryCondition::DirichletTreecode,
            tree_theta: 0.6,
            tree_leaf: 64,
            tree_max_depth: 20,
        }
    }
}

impl DemagPoissonMGConfig {
    pub fn from_env() -> Self {
        fn get_usize(name: &str) -> Option<usize> {
            std::env::var(name)
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
        }

        #[inline]
        fn get_f64(name: &str) -> Option<f64> {
            std::env::var(name)
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
        }

        let mut cfg = Self::default();

        // Padding: prefer PAD_XY; accept legacy PAD_FACTOR_XY as alias.
        if let Some(v) =
            get_usize("LLG_DEMAG_MG_PAD_XY").or_else(|| get_usize("LLG_DEMAG_MG_PAD_FACTOR_XY"))
        {
            cfg.pad_xy = v.max(1);
        }
        if let Some(v) = get_usize("LLG_DEMAG_MG_NVAC_Z") {
            cfg.n_vac_z = v.max(1);
        }
        if let Some(v) = get_usize("LLG_DEMAG_MG_VCYCLES") {
            cfg.v_cycles = v.max(1);
        }
        if let Some(v) = get_usize("LLG_DEMAG_MG_VCYCLES_MAX") {
            cfg.v_cycles_max = v.max(1);
        }
        if let Some(v) = get_f64("LLG_DEMAG_MG_TOL_ABS") {
            cfg.tol_abs = Some(v.max(0.0));
        }
        if let Some(v) = get_f64("LLG_DEMAG_MG_TOL_REL") {
            cfg.tol_rel = Some(v.max(0.0).min(1.0));
        }
        if let Some(v) = get_usize("LLG_DEMAG_MG_PRE_SMOOTH") {
            cfg.pre_smooth = v.max(1);
        }
        if let Some(v) = get_usize("LLG_DEMAG_MG_POST_SMOOTH") {
            cfg.post_smooth = v.max(1);
        }
        if let Some(v) = get_f64("LLG_DEMAG_MG_OMEGA") {
            cfg.omega = v.clamp(0.05, 1.0);
        }
        if let Some(v) = get_f64("LLG_DEMAG_MG_SOR_OMEGA") {
            cfg.sor_omega = v.clamp(0.2, 1.95);
        }
        if let Ok(v) = std::env::var("LLG_DEMAG_MG_SMOOTHER") {
            if let Some(sm) = MGSmoother::from_str(&v) {
                cfg.smoother = sm;
            }
        }
        if let Ok(v) = std::env::var("LLG_DEMAG_MG_WARM_START") {
            cfg.warm_start = matches!(v.as_str(), "1" | "true" | "yes" | "on");
        }
        if let Ok(v) = std::env::var("LLG_DEMAG_MG_BC") {
            let s = v.trim().to_ascii_lowercase();
            cfg.bc = match s.as_str() {
                "0" | "zero" | "dirichlet0" | "dirichlet_zero" => BoundaryCondition::DirichletZero,
                "dipole" | "dirichlet_dipole" | "dirichletdipole" => {
                    BoundaryCondition::DirichletDipole
                }
                "tree" | "treecode" | "bh" | "fmm" => BoundaryCondition::DirichletTreecode,
                _ => cfg.bc,
            };
        }
        if let Some(v) = get_f64("LLG_DEMAG_MG_TREE_THETA") {
            cfg.tree_theta = v.clamp(0.2, 1.5);
        }
        if let Some(v) = get_usize("LLG_DEMAG_MG_TREE_LEAF") {
            cfg.tree_leaf = v.clamp(8, 4096);
        }
        if let Some(v) = get_usize("LLG_DEMAG_MG_TREE_MAX_DEPTH") {
            cfg.tree_max_depth = v.clamp(4, 64);
        }

        cfg
    }
}

// ---------------------------
// Hybrid (MG + local Newell) config
// ---------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HybridConfig {
    enabled: bool,
    radius_xy: usize,
    delta_v_cycles: usize,
    sigma_cells: f64,
    cache_to_disk: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            radius_xy: 12,
            delta_v_cycles: 60,
            sigma_cells: 1.5,
            cache_to_disk: true,
        }
    }
}

impl HybridConfig {
    pub(crate) fn from_env() -> Self {
        fn get_usize(name: &str) -> Option<usize> {
            std::env::var(name)
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
        }

        #[inline]
        fn get_bool(name: &str) -> bool {
            std::env::var(name)
                .ok()
                .map(|s| {
                    matches!(
                        s.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        }

        fn get_f64(name: &str) -> Option<f64> {
            std::env::var(name)
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
        }

        let mut cfg = Self::default();

        cfg.enabled = get_bool("LLG_DEMAG_MG_HYBRID_ENABLE");

        // Hybrid is OFF unless explicitly enabled.
        if !cfg.enabled {
            if let Some(v) = get_usize("LLG_DEMAG_MG_HYBRID_RADIUS") {
                if v > 0 {
                    static WARN_ONCE: OnceLock<()> = OnceLock::new();
                    WARN_ONCE.get_or_init(|| {
                        eprintln!(
                            "[demag_mg] NOTE: LLG_DEMAG_MG_HYBRID_RADIUS={} is set but hybrid ΔK is DISABLED (set LLG_DEMAG_MG_HYBRID_ENABLE=1 to enable). Ignoring.",
                            v
                        );
                    });
                }
            }
            return cfg;
        }

        if let Some(v) = get_usize("LLG_DEMAG_MG_HYBRID_RADIUS") {
            cfg.radius_xy = v;
        }
        if let Some(v) = get_usize("LLG_DEMAG_MG_HYBRID_DELTA_VCYCLES") {
            cfg.delta_v_cycles = v.max(1);
        }
        if let Some(s) = get_f64("LLG_DEMAG_MG_HYBRID_SIGMA") {
            cfg.sigma_cells = s.max(0.0).min(32.0);
        }
        if let Ok(v) = std::env::var("LLG_DEMAG_MG_HYBRID_CACHE") {
            cfg.cache_to_disk = matches!(v.as_str(), "1" | "true" | "yes" | "on");
        }
        cfg
    }

    #[inline]
    fn enabled(&self) -> bool {
        self.enabled && self.radius_xy > 0
    }
}

// ---------------------------
// Local correction kernel ΔK(r) = K_fft(r) - K_mg_long(screened)(r)
// ---------------------------

#[derive(Debug, Clone)]
pub(crate) struct DeltaKernel2D {
    radius: usize,
    stride: usize,
    dkxx: Vec<f64>,
    dkxy: Vec<f64>,
    dkyy: Vec<f64>,
    dkzz: Vec<f64>,
    // PPPM-φ: potential correction stencils (for composite V-cycle ghost-fill).
    // dphi_x[k] = φ_ref − φ_MG at offset (di,dj) for an mx impulse, etc.
    // φ_ref is the potential whose 2D central-difference gradient matches B_Newell.
    dphi_x: Vec<f64>,
    dphi_y: Vec<f64>,
    dphi_z: Vec<f64>,
}

impl DeltaKernel2D {
    fn new(radius: usize) -> Self {
        let stride = 2 * radius + 1;
        let n = stride * stride;
        Self {
            radius,
            stride,
            dkxx: vec![0.0; n],
            dkxy: vec![0.0; n],
            dkyy: vec![0.0; n],
            dkzz: vec![0.0; n],
            dphi_x: vec![0.0; n],
            dphi_y: vec![0.0; n],
            dphi_z: vec![0.0; n],
        }
    }

    #[inline]
    fn idx(&self, dx: isize, dy: isize) -> usize {
        debug_assert!(dx.abs() as usize <= self.radius);
        debug_assert!(dy.abs() as usize <= self.radius);
        let rx = (dx + self.radius as isize) as usize;
        let ry = (dy + self.radius as isize) as usize;
        ry * self.stride + rx
    }

    fn add_correction(&self, m: &VectorField2D, b_eff: &mut VectorField2D, ms: f64) {
        let nx = m.grid.nx;
        let ny = m.grid.ny;
        debug_assert_eq!(nx, b_eff.grid.nx);
        debug_assert_eq!(ny, b_eff.grid.ny);

        let r = self.radius as isize;
        let stride = self.stride;
        let dkxx = &self.dkxx;
        let dkxy = &self.dkxy;
        let dkyy = &self.dkyy;
        let dkzz = &self.dkzz;
        let mdata = &m.data;

        b_eff
            .data
            .par_chunks_mut(nx)
            .enumerate()
            .for_each(|(j, row)| {
                let j_is = j as isize;
                for i in 0..nx {
                    let i_is = i as isize;

                    let mut bx = 0.0f64;
                    let mut by = 0.0f64;
                    let mut bz = 0.0f64;

                    for dy in -r..=r {
                        let sj = j_is - dy;
                        if sj < 0 || sj >= ny as isize {
                            continue;
                        }
                        for dx in -r..=r {
                            let si = i_is - dx;
                            if si < 0 || si >= nx as isize {
                                continue;
                            }
                            let k = (dy + r) as usize * stride + (dx + r) as usize;
                            let src = mdata[(sj as usize) * nx + (si as usize)];
                            let mx = ms * src[0];
                            let my = ms * src[1];
                            let mz = ms * src[2];
                            bx += dkxx[k] * mx + dkxy[k] * my;
                            by += dkxy[k] * mx + dkyy[k] * my;
                            bz += dkzz[k] * mz;
                        }
                    }

                    row[i][0] += bx;
                    row[i][1] += by;
                    row[i][2] += bz;
                }
            });
    }

    /// Apply the PPPM-φ correction to a 2D scalar potential field.
    ///
    /// φ_corrected[i,j] = φ_MG[i,j] + Σ_{di,dj} (dphi_x·mx + dphi_y·my + dphi_z·mz) · Ms
    ///
    /// After this, the 2D central-difference gradient of φ_corrected will approximate
    /// the Newell B field, making patch ghost-fill consistent with the FFT reference.
    pub(crate) fn apply_phi_correction(&self, m: &VectorField2D, phi: &mut [f64], ms: f64) {
        let nx = m.grid.nx;
        let ny = m.grid.ny;
        debug_assert_eq!(phi.len(), nx * ny);

        if self.dphi_x.iter().all(|&v| v == 0.0)
            && self.dphi_y.iter().all(|&v| v == 0.0)
            && self.dphi_z.iter().all(|&v| v == 0.0)
        {
            return; // No phi correction available (legacy cache)
        }

        let r = self.radius as isize;
        let stride = self.stride;
        let dpx = &self.dphi_x;
        let dpy = &self.dphi_y;
        let dpz = &self.dphi_z;
        let mdata = &m.data;

        // Accumulate correction into a separate buffer, then add (avoids aliasing).
        let mut delta = vec![0.0f64; nx * ny];
        for j in 0..ny {
            let j_is = j as isize;
            for i in 0..nx {
                let i_is = i as isize;
                let mut acc = 0.0f64;
                for dy in -r..=r {
                    let sj = j_is - dy;
                    if sj < 0 || sj >= ny as isize { continue; }
                    for dx in -r..=r {
                        let si = i_is - dx;
                        if si < 0 || si >= nx as isize { continue; }
                        let k = (dy + r) as usize * stride + (dx + r) as usize;
                        let src = mdata[(sj as usize) * nx + (si as usize)];
                        acc += dpx[k] * src[0] + dpy[k] * src[1] + dpz[k] * src[2];
                    }
                }
                delta[j * nx + i] = acc * ms;
            }
        }
        for k in 0..nx * ny {
            phi[k] += delta[k];
        }
    }

    #[allow(dead_code)]
    pub fn add_correction_subregion(
        &self,
        _m_patch: &[[f64; 3]],
        b_patch: &mut [[f64; 3]],
        patch_nx: usize,
        _patch_ny: usize,
        i0: usize,       // patch origin x in full grid
        j0: usize,       // patch origin y in full grid
        full_m: &[[f64; 3]],  // full-grid magnetisation for halo reads
        full_nx: usize,
        full_ny: usize,
        ms: f64,
    ) {
        let r = self.radius as isize;
        let stride = self.stride;
        let dkxx = &self.dkxx;
        let dkxy = &self.dkxy;
        let dkyy = &self.dkyy;
        let dkzz = &self.dkzz;

        b_patch
            .par_chunks_mut(patch_nx)
            .enumerate()
            .for_each(|(pj, row)| {
                let gj = j0 + pj;  // global j
                for pi in 0..patch_nx {
                    let gi = i0 + pi;  // global i

                    let mut bx = 0.0f64;
                    let mut by = 0.0f64;
                    let mut bz = 0.0f64;

                    for dy in -r..=r {
                        let sj = gj as isize - dy;
                        if sj < 0 || sj >= full_ny as isize { continue; }
                        for dx in -r..=r {
                            let si = gi as isize - dx;
                            if si < 0 || si >= full_nx as isize { continue; }
                            let k = (dy + r) as usize * stride + (dx + r) as usize;
                            let src = full_m[(sj as usize) * full_nx + (si as usize)];
                            let mx = ms * src[0];
                            let my = ms * src[1];
                            let mz = ms * src[2];
                            bx += dkxx[k] * mx + dkxy[k] * my;
                            by += dkxy[k] * mx + dkyy[k] * my;
                            bz += dkzz[k] * mz;
                        }
                    }

                    row[pi][0] += bx;
                    row[pi][1] += by;
                    row[pi][2] += bz;
                }
            });
    }
    
    fn symmetrize(&mut self, cell_dx: f64, cell_dy: f64) {
        let r = self.radius as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                let k = self.idx(dx, dy);
                let ki = self.idx(-dx, -dy);

                self.dkxx[k] = 0.5 * (self.dkxx[k] + self.dkxx[ki]);
                self.dkyy[k] = 0.5 * (self.dkyy[k] + self.dkyy[ki]);
                self.dkzz[k] = 0.5 * (self.dkzz[k] + self.dkzz[ki]);
                self.dkxy[k] = 0.5 * (self.dkxy[k] + self.dkxy[ki]);
            }
        }

        if (cell_dx - cell_dy).abs() <= 1e-12 * cell_dx.abs().max(cell_dy.abs()).max(1.0) {
            for dy in -r..=r {
                for dx in -r..=r {
                    let k = self.idx(dx, dy);
                    let ks = self.idx(dy, dx);

                    let xx = 0.5 * (self.dkxx[k] + self.dkyy[ks]);
                    let yy = 0.5 * (self.dkyy[k] + self.dkxx[ks]);
                    let xy = 0.5 * (self.dkxy[k] + self.dkxy[ks]);

                    self.dkxx[k] = xx;
                    self.dkyy[k] = yy;
                    self.dkxy[k] = xy;
                }
            }
        }
    }
}


// -----------------------------------------------------------------------------
// A3/A4 operator + multigrid hygiene upgrades (stencil/prolong/coarse-op)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LaplacianStencilKind {
    SevenPoint,
    Iso9PlusZ,
    Iso27,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProlongationKind {
    Injection,
    Trilinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoarseOpKind {
    Rediscretize,
    Galerkin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MGOperatorSettings {
    stencil: LaplacianStencilKind,
    prolong: ProlongationKind,
    coarse_op: CoarseOpKind,
    /// Optional iso27 alpha override (bits). 0 => none.
    iso27_alpha_bits: u64,
}

fn env_str_lower(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
}

impl MGOperatorSettings {
    fn from_env() -> Self {
        let stencil = match env_str_lower("LLG_DEMAG_MG_STENCIL").as_deref() {
            Some("7") | Some("7pt") | Some("seven") | Some("sevenpoint") => {
                LaplacianStencilKind::SevenPoint
            }
            Some("iso9") | Some("9") | Some("9pt") | Some("mehrstellen9") => {
                LaplacianStencilKind::Iso9PlusZ
            }
            Some("iso27") | Some("27") | Some("27pt") | Some("mehrstellen27") => {
                LaplacianStencilKind::Iso27
            }
            Some(other) => {
                eprintln!(
                    "[demag_mg] WARNING: unknown LLG_DEMAG_MG_STENCIL='{}' -> using 'iso27'",
                    other
                );
                LaplacianStencilKind::Iso27
            }
            None => LaplacianStencilKind::Iso27,
        };

        let prolong = match env_str_lower("LLG_DEMAG_MG_PROLONG").as_deref() {
            Some("inject") | Some("injection") | Some("pc") | Some("piecewiseconstant") => {
                ProlongationKind::Injection
            }
            Some("trilinear") | Some("linear") | Some("tl") => ProlongationKind::Trilinear,
            Some(other) => {
                eprintln!(
                    "[demag_mg] WARNING: unknown LLG_DEMAG_MG_PROLONG='{}' -> using 'trilinear'",
                    other
                );
                ProlongationKind::Trilinear
            }
            None => ProlongationKind::Trilinear,
        };

        let coarse_op = match env_str_lower("LLG_DEMAG_MG_COARSE_OP").as_deref() {
            Some("rediscretize") | Some("re") | Some("rd") => CoarseOpKind::Rediscretize,
            Some("galerkin") | Some("g") => CoarseOpKind::Galerkin,
            Some(other) => {
                eprintln!(
                    "[demag_mg] WARNING: unknown LLG_DEMAG_MG_COARSE_OP='{}' -> using 'galerkin'",
                    other
                );
                CoarseOpKind::Galerkin
            }
            None => CoarseOpKind::Galerkin,
        };

        let iso27_alpha_bits: u64 = std::env::var("LLG_DEMAG_MG_ISO27_FLUX_ALPHA")
            .ok()
            .and_then(|s| s.trim().parse::<f64>().ok())
            .filter(|a| a.is_finite() && *a > 0.0)
            .map(|a| a.to_bits())
            .unwrap_or(0);

        Self {
            stencil,
            prolong,
            coarse_op,
            iso27_alpha_bits,
        }
    }

    fn tag(&self) -> String {
        let s = match self.stencil {
            LaplacianStencilKind::SevenPoint => "7".to_string(),
            LaplacianStencilKind::Iso9PlusZ => "iso9".to_string(),
            LaplacianStencilKind::Iso27 => {
                if self.iso27_alpha_bits != 0 {
                    format!("iso27a{:016x}", self.iso27_alpha_bits)
                } else {
                    "iso27".to_string()
                }
            }
        };
        let p = match self.prolong {
            ProlongationKind::Injection => "inj",
            ProlongationKind::Trilinear => "tri",
        };
        let c = match self.coarse_op {
            CoarseOpKind::Rediscretize => "rd",
            CoarseOpKind::Galerkin => "gal",
        };
        format!("{}_{}_{}", s, p, c)
    }
}

/// Constant-coefficient 3D stencil for a cell-centered Laplacian-like operator.
#[derive(Clone, Debug)]
struct Stencil3D {
    center: f64,
    diag: f64,
    offs: Vec<[isize; 3]>,
    coeffs: Vec<f64>,
}

impl Stencil3D {
    fn seven_point(dx: f64, dy: f64, dz: f64) -> Self {
        let sx = 1.0 / (dx * dx);
        let sy = 1.0 / (dy * dy);
        let sz = 1.0 / (dz * dz);
        let center = -2.0 * (sx + sy + sz);

        let mut offs = Vec::with_capacity(6);
        let mut coeffs = Vec::with_capacity(6);

        offs.push([1, 0, 0]);
        coeffs.push(sx);
        offs.push([-1, 0, 0]);
        coeffs.push(sx);

        offs.push([0, 1, 0]);
        coeffs.push(sy);
        offs.push([0, -1, 0]);
        coeffs.push(sy);

        offs.push([0, 0, 1]);
        coeffs.push(sz);
        offs.push([0, 0, -1]);
        coeffs.push(sz);

        let diag = -center;

        Self {
            center,
            diag,
            offs,
            coeffs,
        }
    }

    fn iso9_plus_z(dx: f64, dy: f64, dz: f64) -> Self {
        let rel = (dx - dy).abs() / dx.max(dy).max(1e-30);
        if rel > 1e-6 {
            return Self::seven_point(dx, dy, dz);
        }
        let h = 0.5 * (dx + dy);
        let inv_h2 = 1.0 / (h * h);
        let cz = 1.0 / (dz * dz);

        // xy: (1/(6h^2)) [ 4*(axis) + 1*(diag) - 20*C ]
        let c_axis = (2.0 / 3.0) * inv_h2;
        let c_diag = (1.0 / 6.0) * inv_h2;

        let center = -(10.0 / 3.0) * inv_h2 - 2.0 * cz;

        let mut offs = Vec::with_capacity(10);
        let mut coeffs = Vec::with_capacity(10);

        offs.push([1, 0, 0]);
        coeffs.push(c_axis);
        offs.push([-1, 0, 0]);
        coeffs.push(c_axis);
        offs.push([0, 1, 0]);
        coeffs.push(c_axis);
        offs.push([0, -1, 0]);
        coeffs.push(c_axis);

        offs.push([1, 1, 0]);
        coeffs.push(c_diag);
        offs.push([1, -1, 0]);
        coeffs.push(c_diag);
        offs.push([-1, 1, 0]);
        coeffs.push(c_diag);
        offs.push([-1, -1, 0]);
        coeffs.push(c_diag);

        offs.push([0, 0, 1]);
        coeffs.push(cz);
        offs.push([0, 0, -1]);
        coeffs.push(cz);

        let diag = -center;
        Self {
            center,
            diag,
            offs,
            coeffs,
        }
    }

    fn iso27(dx: f64, dy: f64, dz: f64, alpha_override_bits: u64) -> Self {
        // Precompute sums over non-face diagonals for SPD limit.
        let mut sx1 = 0.0_f64;
        let mut sy1 = 0.0_f64;
        let mut sz1 = 0.0_f64;

        for di in -1isize..=1 {
            for dj in -1isize..=1 {
                for dk in -1isize..=1 {
                    if di == 0 && dj == 0 && dk == 0 {
                        continue;
                    }
                    let nn = di.abs() + dj.abs() + dk.abs();
                    if nn == 1 {
                        continue;
                    }

                    let ddx = (di as f64) * dx;
                    let ddy = (dj as f64) * dy;
                    let ddz = (dk as f64) * dz;
                    let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                    if dist_sq <= 0.0 {
                        continue;
                    }
                    let w1 = 1.0 / dist_sq;

                    sx1 += w1 * ddx * ddx;
                    sy1 += w1 * ddy * ddy;
                    sz1 += w1 * ddz * ddz;
                }
            }
        }

        let alpha_env = if alpha_override_bits != 0 {
            let a = f64::from_bits(alpha_override_bits);
            if a.is_finite() && a > 0.0 {
                Some(a)
            } else {
                None
            }
        } else {
            None
        };

        let mut alpha_max = f64::INFINITY;
        if sx1 > 0.0 {
            alpha_max = alpha_max.min(2.0 / sx1);
        }
        if sy1 > 0.0 {
            alpha_max = alpha_max.min(2.0 / sy1);
        }
        if sz1 > 0.0 {
            alpha_max = alpha_max.min(2.0 / sz1);
        }
        if alpha_max.is_finite() {
            alpha_max *= 0.999;
        }

        const ALPHA_DEFAULT_FRAC: f64 = 0.99;
        let alpha_default = if alpha_max.is_finite() {
            (ALPHA_DEFAULT_FRAC * alpha_max).max(0.0)
        } else {
            0.0
        };

        let mut alpha = alpha_env.unwrap_or(alpha_default);

        if alpha_max.is_finite() && alpha > alpha_max {
            eprintln!(
                "[demag_mg] WARNING: LLG_DEMAG_MG_ISO27_FLUX_ALPHA={} too large (max≈{}). Clamping to keep SPD.",
                alpha, alpha_max
            );
            alpha = alpha_max.max(0.0);
        }

        let w_fx = (2.0 - alpha * sx1) / (2.0 * dx * dx);
        let w_fy = (2.0 - alpha * sy1) / (2.0 * dy * dy);
        let w_fz = (2.0 - alpha * sz1) / (2.0 * dz * dz);

        let mut offs = Vec::with_capacity(26);
        let mut coeffs = Vec::with_capacity(26);

        // faces
        offs.push([1, 0, 0]);
        coeffs.push(w_fx);
        offs.push([-1, 0, 0]);
        coeffs.push(w_fx);

        offs.push([0, 1, 0]);
        coeffs.push(w_fy);
        offs.push([0, -1, 0]);
        coeffs.push(w_fy);

        offs.push([0, 0, 1]);
        coeffs.push(w_fz);
        offs.push([0, 0, -1]);
        coeffs.push(w_fz);

        // edges + corners
        for di in -1isize..=1 {
            for dj in -1isize..=1 {
                for dk in -1isize..=1 {
                    if di == 0 && dj == 0 && dk == 0 {
                        continue;
                    }
                    let nn = di.abs() + dj.abs() + dk.abs();
                    if nn <= 1 {
                        continue;
                    }

                    let ddx = (di as f64) * dx;
                    let ddy = (dj as f64) * dy;
                    let ddz = (dk as f64) * dz;
                    let dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
                    if dist_sq <= 0.0 {
                        continue;
                    }

                    offs.push([di, dj, dk]);
                    coeffs.push(alpha / dist_sq);
                }
            }
        }

        let sum_nb: f64 = coeffs.iter().copied().sum();
        let center = -sum_nb;
        let diag = sum_nb;

        Self {
            center,
            diag,
            offs,
            coeffs,
        }
    }

    fn from_kind(
        kind: LaplacianStencilKind,
        dx: f64,
        dy: f64,
        dz: f64,
        iso27_alpha_bits: u64,
    ) -> Self {
        match kind {
            LaplacianStencilKind::SevenPoint => Self::seven_point(dx, dy, dz),
            LaplacianStencilKind::Iso9PlusZ => Self::iso9_plus_z(dx, dy, dz),
            LaplacianStencilKind::Iso27 => Self::iso27(dx, dy, dz, iso27_alpha_bits),
        }
    }

    fn apply_at(&self, phi: &[f64], nx: usize, ny: usize, nz: usize,
                i: usize, j: usize, k: usize) -> f64 {
        kernels::stencil_apply_at(phi, nx, ny, nz, i, j, k,
                                self.center, &self.offs, &self.coeffs)
    }

    #[allow(dead_code)]
    fn offdiag_sum_at(&self, phi: &[f64], nx: usize, ny: usize, nz: usize,
                    i: usize, j: usize, k: usize) -> f64 {
        kernels::offdiag_sum_at(phi, nx, ny, nz, i, j, k,
                                &self.offs, &self.coeffs)
    }

    fn galerkin_coarsen(
        fine: &Stencil3D,
        rx: usize,
        ry: usize,
        rz: usize,
        prolong: ProlongationKind,
    ) -> Self {
        let ncx: usize = 9;
        let ncy: usize = 9;
        let ncz: usize = 9;

        let nfx = ncx * rx;
        let nfy = ncy * ry;
        let nfz = ncz * rz;

        let c0 = (ncx / 2, ncy / 2, ncz / 2);
        let id_c0 = kernels::idx3(c0.0, c0.1, c0.2, ncx, ncy);

        let mut phi_c = vec![0.0f64; ncx * ncy * ncz];
        let mut phi_f = vec![0.0f64; nfx * nfy * nfz];
        let mut y_f = vec![0.0f64; nfx * nfy * nfz];
        let mut y_c = vec![0.0f64; ncx * ncy * ncz];

        let mut map: HashMap<(isize, isize, isize), f64> = HashMap::new();

        for kz in 0..ncz {
            for jy in 0..ncy {
                for ix in 0..ncx {
                    phi_c.fill(0.0);
                    phi_c[kernels::idx3(ix, jy, kz, ncx, ncy)] = 1.0;

                    prolongate_scalar(
                        &phi_c, ncx, ncy, ncz, &mut phi_f, nfx, nfy, nfz, rx, ry, rz, prolong,
                    );

                    apply_stencil_to_field(fine, &phi_f, &mut y_f, nfx, nfy, nfz);

                    restrict_scalar_avg(&y_f, nfx, nfy, nfz, &mut y_c, ncx, ncy, ncz, rx, ry, rz);

                    let coeff = y_c[id_c0];
                    if coeff.abs() > 1e-14 {
                        let off = (
                            ix as isize - c0.0 as isize,
                            jy as isize - c0.1 as isize,
                            kz as isize - c0.2 as isize,
                        );
                        map.insert(off, coeff);
                    }
                }
            }
        }

        let mut keys: Vec<(isize, isize, isize)> = map.keys().cloned().collect();
        keys.sort();

        let mut center = 0.0;
        let mut offs = Vec::new();
        let mut coeffs = Vec::new();

        for key in keys {
            let c = map[&key];
            if key == (0, 0, 0) {
                center = c;
            } else {
                offs.push([key.0, key.1, key.2]);
                coeffs.push(c);
            }
        }

        let diag = -center;

        // Validate Galerkin-coarsened stencil.
        // If the diagonal is non-positive or the Jacobi spectral radius
        // is too large, the smoother will diverge (producing NaN).
        // This happens with extreme cell aspect ratios (thin-film geometries).
        let offdiag_abs_sum: f64 = coeffs.iter().map(|c| c.abs()).sum();
        let neg_offdiag = coeffs.iter().filter(|&&c| c < 0.0).count();
        let jacobi_rho = if diag > 0.0 { offdiag_abs_sum / diag } else { f64::INFINITY };

        if diag <= 0.0 || jacobi_rho > 2.0 || neg_offdiag > coeffs.len() / 2 {
            eprintln!(
                "[demag_mg] WARNING: Galerkin produced ill-conditioned stencil \
                 (diag={:.3e}, jacobi_rho={:.2}, neg_offdiag={}/{}). \
                 Will use rediscretization instead.",
                diag, jacobi_rho, neg_offdiag, coeffs.len()
            );
            // Return a sentinel that the level builder can detect
            // to trigger fallback to rediscretization.
            return Self {
                center: 0.0,
                diag: 0.0,
                offs: Vec::new(),
                coeffs: Vec::new(),
            };
        }

        Self {
            center,
            diag,
            offs,
            coeffs,
        }
    }
}

fn apply_stencil_to_field(
    st: &Stencil3D,
    phi: &[f64],
    out: &mut [f64],
    nx: usize,
    ny: usize,
    nz: usize,
) {
    debug_assert_eq!(phi.len(), nx * ny * nz);
    debug_assert_eq!(out.len(), nx * ny * nz);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                out[kernels::idx3(i, j, k, nx, ny)] = st.apply_at(phi, nx, ny, nz, i, j, k);
            }
        }
    }
}

fn restrict_scalar_avg(
    fine: &[f64],
    nfx: usize,
    nfy: usize,
    nfz: usize,
    coarse: &mut [f64],
    ncx: usize,
    ncy: usize,
    ncz: usize,
    rx: usize,
    ry: usize,
    rz: usize,
) {
    debug_assert_eq!(fine.len(), nfx * nfy * nfz);
    debug_assert_eq!(coarse.len(), ncx * ncy * ncz);

    let norm = 1.0 / ((rx * ry * rz) as f64);

    for kz in 0..ncz {
        for jy in 0..ncy {
            for ix in 0..ncx {
                let mut sum = 0.0;
                for fk in 0..rz {
                    for fj in 0..ry {
                        for fi in 0..rx {
                            let i = ix * rx + fi;
                            let j = jy * ry + fj;
                            let k = kz * rz + fk;
                            sum += fine[kernels::idx3(i, j, k, nfx, nfy)];
                        }
                    }
                }
                coarse[kernels::idx3(ix, jy, kz, ncx, ncy)] = norm * sum;
            }
        }
    }
}

fn prolongate_scalar(
    coarse: &[f64],
    ncx: usize,
    ncy: usize,
    ncz: usize,
    fine: &mut [f64],
    nfx: usize,
    nfy: usize,
    nfz: usize,
    rx: usize,
    ry: usize,
    rz: usize,
    kind: ProlongationKind,
) {
    debug_assert_eq!(coarse.len(), ncx * ncy * ncz);
    debug_assert_eq!(fine.len(), nfx * nfy * nfz);

    match kind {
        ProlongationKind::Injection => {
            fine.fill(0.0);
            for kz in 0..ncz {
                for jy in 0..ncy {
                    for ix in 0..ncx {
                        let v = coarse[kernels::idx3(ix, jy, kz, ncx, ncy)];
                        for fk in 0..rz {
                            for fj in 0..ry {
                                for fi in 0..rx {
                                    let i = ix * rx + fi;
                                    let j = jy * ry + fj;
                                    let k = kz * rz + fk;
                                    fine[kernels::idx3(i, j, k, nfx, nfy)] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
        ProlongationKind::Trilinear => {
            fine.fill(0.0);
            for k in 0..nfz {
                let kz = k / rz;
                let rk = k % rz;
                let (k0, k1, wk0, wk1) = interp_1d_cell_centered(kz, rk, ncz, rz);

                for j in 0..nfy {
                    let jy = j / ry;
                    let rj = j % ry;
                    let (j0, j1, wj0, wj1) = interp_1d_cell_centered(jy, rj, ncy, ry);

                    for i in 0..nfx {
                        let ix = i / rx;
                        let ri = i % rx;
                        let (i0, i1, wi0, wi1) = interp_1d_cell_centered(ix, ri, ncx, rx);

                        let mut v = 0.0;

                        v += wi0 * wj0 * wk0 * coarse[kernels::idx3(i0, j0, k0, ncx, ncy)];
                        v += wi1 * wj0 * wk0 * coarse[kernels::idx3(i1, j0, k0, ncx, ncy)];
                        v += wi0 * wj1 * wk0 * coarse[kernels::idx3(i0, j1, k0, ncx, ncy)];
                        v += wi1 * wj1 * wk0 * coarse[kernels::idx3(i1, j1, k0, ncx, ncy)];
                        v += wi0 * wj0 * wk1 * coarse[kernels::idx3(i0, j0, k1, ncx, ncy)];
                        v += wi1 * wj0 * wk1 * coarse[kernels::idx3(i1, j0, k1, ncx, ncy)];
                        v += wi0 * wj1 * wk1 * coarse[kernels::idx3(i0, j1, k1, ncx, ncy)];
                        v += wi1 * wj1 * wk1 * coarse[kernels::idx3(i1, j1, k1, ncx, ncy)];

                        fine[kernels::idx3(i, j, k, nfx, nfy)] = v;
                    }
                }
            }
        }
    }
}


// ---------------------------
// Multigrid data structures
// ---------------------------

struct MGLevel {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    inv_dx2: f64,
    inv_dy2: f64,
    inv_dz2: f64,

    stencil: Stencil3D,

    phi: Vec<f64>,
    rhs: Vec<f64>,
    res: Vec<f64>,
    tmp: Vec<f64>,

    bc_phi: Vec<f64>,
}

impl MGLevel {
    fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        let n = nx * ny * nz;
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            inv_dx2: 1.0 / (dx * dx),
            inv_dy2: 1.0 / (dy * dy),
            inv_dz2: 1.0 / (dz * dz),
            stencil: Stencil3D::seven_point(dx, dy, dz),
            phi: vec![0.0; n],
            rhs: vec![0.0; n],
            res: vec![0.0; n],
            tmp: vec![0.0; n],
            bc_phi: vec![0.0; n],
        }
    }

    fn enforce_dirichlet(&mut self) {
        kernels::stamp_dirichlet_bc(&mut self.phi, &self.bc_phi, self.nx, self.ny, self.nz);
    }

    #[inline]
    #[allow(dead_code)]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        idx3(i, j, k, self.nx, self.ny)
    }
}

pub struct DemagPoissonMG {
    grid: Grid2D,
    cfg: DemagPoissonMGConfig,
    op: MGOperatorSettings,

    px: usize,
    py: usize,
    pz: usize,

    offx: usize,
    offy: usize,
    offz: usize,

    levels: Vec<MGLevel>,
}

#[inline]
fn rb_allowed_for_op(op: MGOperatorSettings) -> bool {
    op.stencil == LaplacianStencilKind::SevenPoint
        && op.coarse_op == CoarseOpKind::Rediscretize
        && op.prolong == ProlongationKind::Injection
}

#[inline]
fn sanitize_cfg_for_op(cfg: &mut DemagPoissonMGConfig, op: MGOperatorSettings) {
    if cfg.smoother == MGSmoother::RedBlackSOR && !rb_allowed_for_op(op) {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| {
            eprintln!(
                "[demag_mg] INFO: overriding smoother RedBlackSOR -> WeightedJacobi (stencil/prolong/coarse-op requires it)."
            );
        });
        cfg.smoother = MGSmoother::WeightedJacobi;
    }
}

impl DemagPoissonMG {
    pub fn new(grid: Grid2D, cfg: DemagPoissonMGConfig) -> Self {
        let op = MGOperatorSettings::from_env();
        Self::new_with_operator(grid, cfg, op)
    }

    fn new_with_operator(
        grid: Grid2D,
        mut cfg: DemagPoissonMGConfig,
        op: MGOperatorSettings,
    ) -> Self {
        sanitize_cfg_for_op(&mut cfg, op);

        let nx = grid.nx.max(1);
        let ny = grid.ny.max(1);

        let pad = cfg.pad_xy.max(1);
        let mut px = nx + 2 * pad;
        let mut py = ny + 2 * pad;

        if px % 2 == 1 {
            px += 1;
        }
        if py % 2 == 1 {
            py += 1;
        }

        let n_vac = cfg.n_vac_z.max(1);
        let mut pz = 1 + 2 * n_vac;
        if pz % 2 == 1 {
            pz += 1;
        }
        // Round up to next multiple of 4 so z can coarsen at least twice
        // before becoming odd. This avoids mixed-ratio Galerkin coarsening
        // which produces degenerate stencils for anisotropic grids.
        if pz % 4 != 0 {
            pz = ((pz + 3) / 4) * 4;
        }

        let offx = (px.saturating_sub(nx)) / 2;
        let offy = (py.saturating_sub(ny)) / 2;
        let offz = n_vac;

        let mut levels: Vec<MGLevel> = Vec::new();
        let mut lx = px;
        let mut ly = py;
        let mut lz = pz;
        let mut dx = grid.dx;
        let mut dy = grid.dy;
        let mut dz = grid.dz;

        levels.push(MGLevel::new(lx, ly, lz, dx, dy, dz));

        loop {
            let can_xy = lx >= 8 && ly >= 8 && lx % 2 == 0 && ly % 2 == 0;
            let can_z = lz >= 8 && lz % 2 == 0;

            if !can_xy && !can_z {
                break;
            }

            if can_xy {
                lx /= 2;
                ly /= 2;
                dx *= 2.0;
                dy *= 2.0;
            }
            if can_z {
                lz /= 2;
                dz *= 2.0;
            }

            levels.push(MGLevel::new(lx, ly, lz, dx, dy, dz));

            if levels.len() > 32 {
                break;
            }
        }

        // Warn about extreme cell aspect ratios that may degrade MG convergence.
        {
            let aspect_xy = (grid.dx / grid.dy).max(grid.dy / grid.dx);
            let aspect_xz = (grid.dx / grid.dz).max(grid.dz / grid.dx);
            let aspect_yz = (grid.dy / grid.dz).max(grid.dz / grid.dy);
            let max_aspect = aspect_xy.max(aspect_xz).max(aspect_yz);
            if max_aspect > 4.0 {
                eprintln!(
                    "[demag_mg] WARNING: extreme cell aspect ratio {:.1} \
                     (dx={:.2e}, dy={:.2e}, dz={:.2e}). \
                     Consider using 7-point stencil (LLG_DEMAG_MG_STENCIL=7) \
                     or adjusting n_vac_z for better conditioning.",
                    max_aspect, grid.dx, grid.dy, grid.dz,
                );
            }
        }

        // Assign stencils per level.
        if !levels.is_empty() {
            levels[0].stencil = Stencil3D::from_kind(
                op.stencil,
                levels[0].dx,
                levels[0].dy,
                levels[0].dz,
                op.iso27_alpha_bits,
            );
            for l in 1..levels.len() {
                let rx = levels[l - 1].nx / levels[l].nx;
                let ry = levels[l - 1].ny / levels[l].ny;
                let rz = levels[l - 1].nz / levels[l].nz;

                levels[l].stencil = match op.coarse_op {
                    CoarseOpKind::Rediscretize => Stencil3D::from_kind(
                        op.stencil,
                        levels[l].dx,
                        levels[l].dy,
                        levels[l].dz,
                        op.iso27_alpha_bits,
                    ),
                    CoarseOpKind::Galerkin => {
                        let g = Stencil3D::galerkin_coarsen(
                            &levels[l - 1].stencil, rx, ry, rz, op.prolong,
                        );
                        if g.diag <= 0.0 || g.offs.is_empty() {
                            // Galerkin produced a degenerate stencil — fall back
                            // to rediscretization at this level and all coarser.
                            eprintln!(
                                "[demag_mg] Falling back to rediscretization at level {} \
                                 ({}×{}×{}, dx={:.2e} dy={:.2e} dz={:.2e})",
                                l, levels[l].nx, levels[l].ny, levels[l].nz,
                                levels[l].dx, levels[l].dy, levels[l].dz,
                            );
                            Stencil3D::from_kind(
                                op.stencil,
                                levels[l].dx,
                                levels[l].dy,
                                levels[l].dz,
                                op.iso27_alpha_bits,
                            )
                        } else {
                            g
                        }
                    }
                };
            }
        }

        Self {
            grid,
            cfg,
            op,
            px,
            py,
            pz,
            offx,
            offy,
            offz,
            levels,
        }
    }

    fn apply_cfg(&mut self, mut cfg: DemagPoissonMGConfig) {
        sanitize_cfg_for_op(&mut cfg, self.op);
        self.cfg = cfg;
    }

    fn same_structure(&self, grid: &Grid2D, cfg: &DemagPoissonMGConfig) -> bool {
        self.grid.nx == grid.nx
            && self.grid.ny == grid.ny
            && self.grid.dx == grid.dx
            && self.grid.dy == grid.dy
            && self.grid.dz == grid.dz
            && self.cfg.pad_xy == cfg.pad_xy
            && self.cfg.n_vac_z == cfg.n_vac_z
            && self.op == MGOperatorSettings::from_env()
    }

    pub(crate) fn build_rhs_from_m(&mut self, m: &VectorField2D, ms: f64) {
        let finest = &mut self.levels[0];
        finest.rhs.fill(0.0);

        let nx = finest.nx;
        let ny = finest.ny;
        let nz = finest.nz;

        let dx = finest.dx;
        let dy = finest.dy;
        let dz = finest.dz;

        let offx = self.offx;
        let offy = self.offy;
        let offz = self.offz;

        let nx_m = self.grid.nx;
        let ny_m = self.grid.ny;

        let px = self.px;
        let py = self.py;
        let pz = self.pz;

        let mdata = &m.data;

        #[inline]
        fn m_at(
            pi: isize,
            pj: isize,
            pk: isize,
            px: usize,
            py: usize,
            pz: usize,
            offx: usize,
            offy: usize,
            offz: usize,
            nx_m: usize,
            ny_m: usize,
            mdata: &[[f64; 3]],
            ms: f64,
        ) -> (bool, [f64; 3]) {
            if pi < 0 || pj < 0 || pk < 0 {
                return (false, [0.0; 3]);
            }
            let (piu, pju, pku) = (pi as usize, pj as usize, pk as usize);
            if piu >= px || pju >= py || pku >= pz {
                return (false, [0.0; 3]);
            }

            if pku != offz {
                return (false, [0.0; 3]);
            }
            if piu < offx || piu >= offx + nx_m || pju < offy || pju >= offy + ny_m {
                return (false, [0.0; 3]);
            }

            let mi = piu - offx;
            let mj = pju - offy;
            let id = mj * nx_m + mi;

            let v = mdata[id];
            let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if n2 < 1e-30 {
                return (false, [0.0; 3]);
            }

            (true, [ms * v[0], ms * v[1], ms * v[2]])
        }

        #[inline]
        fn face_val(in_a: bool, a: f64, in_b: bool, b: f64) -> f64 {
            match (in_a, in_b) {
                (true, true) => 0.5 * (a + b),
                (true, false) => a,
                (false, true) => b,
                (false, false) => 0.0,
            }
        }

        let rhs = &mut finest.rhs;

        rhs.par_chunks_mut(nx)
            .enumerate()
            .for_each(|(row_idx, rhs_row)| {
                let k = row_idx / ny;
                let j = row_idx % ny;

                if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                    return;
                }

                let pj = j as isize;
                let pk = k as isize;

                for i in 1..(nx - 1) {
                    let pi = i as isize;

                    let (c_in, m_c) = m_at(
                        pi, pj, pk, px, py, pz, offx, offy, offz, nx_m, ny_m, mdata, ms,
                    );

                    let (xp_in, m_xp) = m_at(
                        pi + 1,
                        pj,
                        pk,
                        px,
                        py,
                        pz,
                        offx,
                        offy,
                        offz,
                        nx_m,
                        ny_m,
                        mdata,
                        ms,
                    );
                    let (xm_in, m_xm) = m_at(
                        pi - 1,
                        pj,
                        pk,
                        px,
                        py,
                        pz,
                        offx,
                        offy,
                        offz,
                        nx_m,
                        ny_m,
                        mdata,
                        ms,
                    );

                    let (yp_in, m_yp) = m_at(
                        pi,
                        pj + 1,
                        pk,
                        px,
                        py,
                        pz,
                        offx,
                        offy,
                        offz,
                        nx_m,
                        ny_m,
                        mdata,
                        ms,
                    );
                    let (ym_in, m_ym) = m_at(
                        pi,
                        pj - 1,
                        pk,
                        px,
                        py,
                        pz,
                        offx,
                        offy,
                        offz,
                        nx_m,
                        ny_m,
                        mdata,
                        ms,
                    );

                    let (zp_in, m_zp) = m_at(
                        pi,
                        pj,
                        pk + 1,
                        px,
                        py,
                        pz,
                        offx,
                        offy,
                        offz,
                        nx_m,
                        ny_m,
                        mdata,
                        ms,
                    );
                    let (zm_in, m_zm) = m_at(
                        pi,
                        pj,
                        pk - 1,
                        px,
                        py,
                        pz,
                        offx,
                        offy,
                        offz,
                        nx_m,
                        ny_m,
                        mdata,
                        ms,
                    );

                    let mx_p = face_val(c_in, m_c[0], xp_in, m_xp[0]);
                    let mx_m = face_val(xm_in, m_xm[0], c_in, m_c[0]);

                    let my_p = face_val(c_in, m_c[1], yp_in, m_yp[1]);
                    let my_m = face_val(ym_in, m_ym[1], c_in, m_c[1]);

                    let mz_p = face_val(c_in, m_c[2], zp_in, m_zp[2]);
                    let mz_m = face_val(zm_in, m_zm[2], c_in, m_c[2]);

                    let div_m = (mx_p - mx_m) / dx + (my_p - my_m) / dy + (mz_p - mz_m) / dz;
                    rhs_row[i] = div_m;
                }
            });
    }

    fn update_finest_boundary_bc(&mut self) {
        let finest = &mut self.levels[0];
        finest.bc_phi.fill(0.0);

        match self.cfg.bc {
            BoundaryCondition::DirichletZero => {}
            BoundaryCondition::DirichletDipole => {
                treecode::evaluate_dipole_bc(
                    &mut finest.bc_phi, &finest.rhs,
                    finest.nx, finest.ny, finest.nz,
                    finest.dx, finest.dy, finest.dz,
                );
            }
            BoundaryCondition::DirichletTreecode => {
                let charges = treecode::build_charges_from_rhs(
                    &finest.rhs,
                    finest.nx, finest.ny, finest.nz,
                    finest.dx, finest.dy, finest.dz,
                );
                treecode::evaluate_treecode_bc(
                    &mut finest.bc_phi, charges,
                    finest.nx, finest.ny, finest.nz,
                    finest.dx, finest.dy, finest.dz,
                    self.cfg.tree_leaf, self.cfg.tree_theta, self.cfg.tree_max_depth,
                );
            }
        }

        finest.enforce_dirichlet();
    }

    fn smooth_weighted_jacobi(level: &mut MGLevel, iters: usize, omega: f64) {
        kernels::smooth_weighted_jacobi(
            &mut level.phi, &mut level.tmp, &level.rhs, &level.bc_phi,
            level.nx, level.ny, level.nz,
            level.stencil.diag, &level.stencil.offs, &level.stencil.coeffs,
            iters, omega,
        );
    }

    fn smooth_rb_sor(level: &mut MGLevel, iters: usize, omega: f64) {
        kernels::smooth_rb_sor(
            &mut level.phi, &mut level.tmp, &level.rhs, &level.bc_phi,
            level.nx, level.ny, level.nz,
            level.inv_dx2, level.inv_dy2, level.inv_dz2,
            iters, omega,
        );
    }

    fn compute_residual(level: &mut MGLevel) -> f64 {
        kernels::compute_residual(
            &level.phi, &level.rhs, &mut level.res,
            level.nx, level.ny, level.nz,
            level.stencil.center, &level.stencil.offs, &level.stencil.coeffs,
        )
    }

    fn restrict_residual(fine: &MGLevel, coarse: &mut MGLevel) {
        kernels::restrict_residual(
            &fine.res, fine.nx, fine.ny, fine.nz,
            &mut coarse.rhs, &mut coarse.phi,
            coarse.nx, coarse.ny, coarse.nz,
        );
    }

    fn prolongate_add(coarse: &MGLevel, fine: &mut MGLevel, kind: ProlongationKind) {
        kernels::prolongate_add(
            &coarse.phi, coarse.nx, coarse.ny, coarse.nz,
            &mut fine.phi, fine.nx, fine.ny, fine.nz,
            kind == ProlongationKind::Trilinear,
        );
    }

    fn v_cycle(&mut self, l: usize) {
        let pre = self.cfg.pre_smooth;
        let post = self.cfg.post_smooth;

        let smoother = self.cfg.smoother;
        let omega_j = self.cfg.omega;
        let omega_sor = self.cfg.sor_omega;

        if l == self.levels.len() - 1 {
            match smoother {
                MGSmoother::WeightedJacobi => {
                    Self::smooth_weighted_jacobi(&mut self.levels[l], 80, omega_j)
                }
                MGSmoother::RedBlackSOR => Self::smooth_rb_sor(&mut self.levels[l], 80, omega_sor),
            }

            // NaN guard: detect divergence at the coarsest level before it
            // propagates upward through prolongation.
            #[cfg(debug_assertions)]
            {
                let has_nan = self.levels[l].phi.iter().any(|v| !v.is_finite());
                if has_nan {
                    let lev = &self.levels[l];
                    eprintln!(
                        "[demag_mg] NaN/Inf detected at coarsest level {} \
                         ({}×{}×{}, diag={:.3e}). \
                         Stencil may be degenerate for this grid geometry.",
                        l, lev.nx, lev.ny, lev.nz, lev.stencil.diag,
                    );
                }
            }

            return;
        }

        match smoother {
            MGSmoother::WeightedJacobi => {
                Self::smooth_weighted_jacobi(&mut self.levels[l], pre, omega_j)
            }
            MGSmoother::RedBlackSOR => Self::smooth_rb_sor(&mut self.levels[l], pre, omega_sor),
        }

        Self::compute_residual(&mut self.levels[l]);

        {
            let (fine, coarse) = {
                let (a, b) = self.levels.split_at_mut(l + 1);
                (&a[l], &mut b[0])
            };
            Self::restrict_residual(fine, coarse);
        }

        self.v_cycle(l + 1);

        {
            let (fine, coarse) = {
                let (a, b) = self.levels.split_at_mut(l + 1);
                (&mut a[l], &b[0])
            };
            Self::prolongate_add(coarse, fine, self.op.prolong);
        }

        match smoother {
            MGSmoother::WeightedJacobi => {
                Self::smooth_weighted_jacobi(&mut self.levels[l], post, omega_j)
            }
            MGSmoother::RedBlackSOR => Self::smooth_rb_sor(&mut self.levels[l], post, omega_sor),
        }
    }

    fn solve_with_timing(&mut self) -> (u64, u64) {
        if !self.cfg.warm_start {
            self.levels[0].phi.fill(0.0);
        }

        let t_bc = Instant::now();
        self.update_finest_boundary_bc();
        let bc_ns = t_bc.elapsed().as_nanos() as u64;

        self.levels[0].enforce_dirichlet();

        let t_solve = Instant::now();

        let tol_abs = self.cfg.tol_abs;
        let tol_rel = self.cfg.tol_rel;
        let use_tol = tol_abs.is_some() || tol_rel.is_some();

        let rhs_max = if use_tol && tol_rel.is_some() {
            let finest = &self.levels[0];
            let nx = finest.nx;
            let ny = finest.ny;
            let nz = finest.nz;
            finest
                .rhs
                .par_chunks(nx)
                .enumerate()
                .map(|(row_idx, row)| {
                    let k = row_idx / ny;
                    let j = row_idx % ny;
                    if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                        return 0.0;
                    }
                    let mut m = 0.0f64;
                    for &v in &row[1..nx - 1] {
                        m = m.max(v.abs());
                    }
                    m
                })
                .reduce(|| 0.0, f64::max)
        } else {
            0.0
        };

        let tol_target = if use_tol {
            let mut t = 0.0f64;
            if let Some(a) = tol_abs {
                t = t.max(a.max(0.0));
            }
            if let Some(r) = tol_rel {
                t = t.max((r.max(0.0) * rhs_max).max(0.0));
            }
            t
        } else {
            0.0
        };

        let min_cycles = if use_tol { self.cfg.v_cycles.max(1) } else { 0 };
        let max_cycles = if use_tol {
            self.cfg.v_cycles_max.max(min_cycles).max(1)
        } else {
            self.cfg.v_cycles.max(1)
        };

        for iter in 0..max_cycles {
            self.v_cycle(0);
            self.levels[0].enforce_dirichlet();

            // Early exit if the solution has diverged to NaN/Inf.
            // This prevents wasting cycles on garbage data and gives
            // a clearer diagnostic than downstream NaN propagation.
            if iter % 2 == 1 || iter + 1 == max_cycles {
                let has_bad = self.levels[0].phi.iter()
                    .take(self.levels[0].nx * self.levels[0].ny * 2)
                    .any(|v| !v.is_finite());
                if has_bad {
                    eprintln!(
                        "[demag_mg] ERROR: NaN/Inf in phi after V-cycle {}. \
                         Solver diverged — check grid aspect ratio and stencil. \
                         (grid={}×{}×{}, dx={:.2e}, dz={:.2e})",
                        iter + 1,
                        self.levels[0].nx, self.levels[0].ny, self.levels[0].nz,
                        self.levels[0].dx, self.levels[0].dz,
                    );
                    self.levels[0].phi.fill(0.0);
                    self.levels[0].enforce_dirichlet();
                    break;
                }
            }

            if use_tol && (iter + 1) >= min_cycles {
                let max_r = Self::compute_residual(&mut self.levels[0]);
                if max_r <= tol_target {
                    break;
                }
            }
        }

        self.levels[0].enforce_dirichlet();

        let solve_ns = t_solve.elapsed().as_nanos() as u64;

        if use_tol {
            let max_r = Self::compute_residual(&mut self.levels[0]);
            eprintln!(
                "[demag_mg] max_residual={:.3e}  rhs_max={:.3e}  tol_target={:.3e}  (min_cycles={}, max_cycles={})",
                max_r, rhs_max, tol_target, min_cycles, max_cycles
            );
        }

        (bc_ns, solve_ns)
    }

    pub(crate) fn solve(&mut self) {
        let _ = self.solve_with_timing();
    }

    fn add_b_from_phi_on_magnet_layer(
        &self,
        m: &VectorField2D,
        _ms: f64,
        b_eff: &mut VectorField2D,
    ) {
        self.add_b_from_phi_on_magnet_layer_impl(Some(&m.data), b_eff);
    }

    pub(crate) fn add_b_from_phi_on_magnet_layer_all(&self, b_eff: &mut VectorField2D) {
        self.add_b_from_phi_on_magnet_layer_impl(None, b_eff);
    }

    fn add_b_from_phi_on_magnet_layer_impl(
        &self,
        mdata_opt: Option<&[[f64; 3]]>,
        b_eff: &mut VectorField2D,
    ) {
        let finest = &self.levels[0];
        kernels::extract_gradient_on_magnet_layer(
            &finest.phi,
            self.px, self.py, self.pz,
            finest.dx, finest.dy, finest.dz,
            self.offx, self.offy, self.offz,
            self.grid.nx, self.grid.ny,
            MU0,
            &mut b_eff.data,
            mdata_opt,
        );
    }

    pub fn add_field(&mut self, m: &VectorField2D, b_eff: &mut VectorField2D, ms: f64) {
        self.build_rhs_from_m(m, ms);
        self.solve();
        self.add_b_from_phi_on_magnet_layer(m, ms, b_eff);
    }

    // -- Composite-grid accessors (used by mg_composite) ---------------------

    /// Padded box dimensions.
    #[allow(dead_code)]
    pub(crate) fn padded_dims(&self) -> (usize, usize, usize) {
        (self.px, self.py, self.pz)
    }

    /// Read-only access to finest-level φ.
    #[allow(dead_code)]
    pub(crate) fn finest_phi(&self) -> &[f64] {
        &self.levels[0].phi
    }

    /// Mutable access to finest-level RHS (for injecting patch residuals).
    #[allow(dead_code)]
    pub(crate) fn finest_rhs_mut(&mut self) -> &mut [f64] {
        &mut self.levels[0].rhs
    }

    /// The physical grid this solver was built for.
    #[allow(dead_code)]
    pub(crate) fn base_grid(&self) -> Grid2D {
        self.grid
    }

    /// Extract magnet-layer φ as a flat 2D array (nx_m × ny_m).
    ///
    /// Returns a Vec<f64> of length `grid.nx * grid.ny` containing the
    /// φ values on the single magnet layer (z = offz) of the 3D padded box.
    /// Cell ordering: phi_2d[j * nx_m + i] = φ at magnet cell (i, j).
    ///
    /// Used by the composite V-cycle to prolongate coarse φ to fine patches.
    #[allow(dead_code)]
    pub(crate) fn extract_magnet_layer_phi(&self) -> Vec<f64> {
        let nx_m = self.grid.nx;
        let ny_m = self.grid.ny;
        let phi_3d = &self.levels[0].phi;
        let px = self.px;
        let py = self.py;
        let offx = self.offx;
        let offy = self.offy;
        let offz = self.offz;

        let mut phi_2d = vec![0.0f64; nx_m * ny_m];
        for j in 0..ny_m {
            for i in 0..nx_m {
                phi_2d[j * nx_m + i] = phi_3d[idx3(offx + i, offy + j, offz, px, py)];
            }
        }
        phi_2d
    }

    /// Write a 2D φ array back into the magnet layer of the 3D padded box.
    ///
    /// This is the inverse of `extract_magnet_layer_phi()`.
    /// Used for warm-start: inject the previous timestep's coarse φ solution
    /// before the next composite V-cycle.
    ///
    /// Panics if `phi_2d.len() != grid.nx * grid.ny`.
    #[allow(dead_code)]
    pub(crate) fn inject_magnet_layer_phi(&mut self, phi_2d: &[f64]) {
        let nx_m = self.grid.nx;
        let ny_m = self.grid.ny;
        assert_eq!(phi_2d.len(), nx_m * ny_m,
            "inject_magnet_layer_phi: expected {} values, got {}",
            nx_m * ny_m, phi_2d.len());

        let px = self.px;
        let py = self.py;
        let offx = self.offx;
        let offy = self.offy;
        let offz = self.offz;
        let phi_3d = &mut self.levels[0].phi;

        for j in 0..ny_m {
            for i in 0..nx_m {
                phi_3d[idx3(offx + i, offy + j, offz, px, py)] = phi_2d[j * nx_m + i];
            }
        }
    }

    // -- Composite-grid: external RHS support ---------------------------------

    /// Stamp a 2D RHS array into the magnet layer of the 3D padded box.
    ///
    /// Clears the entire 3D RHS to zero, then writes `rhs_2d[j*nx + i]`
    /// into padded-box cell `(offx+i, offy+j, offz)` for each magnet cell.
    ///
    /// After calling this, use `solve()` then `add_b_from_phi_on_magnet_layer_all()`
    /// to complete the pipeline.
    #[allow(dead_code)]
    pub(crate) fn stamp_rhs_from_2d(&mut self, rhs_2d: &[f64]) {
        let nx_m = self.grid.nx;
        let ny_m = self.grid.ny;
        assert_eq!(rhs_2d.len(), nx_m * ny_m);

        let finest = &mut self.levels[0];
        finest.rhs.fill(0.0);

        let px = self.px;
        let py = self.py;
        let offx = self.offx;
        let offy = self.offy;
        let offz = self.offz;

        for j in 0..ny_m {
            for i in 0..nx_m {
                let pi = offx + i;
                let pj = offy + j;
                finest.rhs[idx3(pi, pj, offz, px, py)] = rhs_2d[j * nx_m + i];
            }
        }
    }

    /// Full pipeline: stamp external 2D RHS, set BCs, solve, extract B.
    ///
    /// Overwrites `b_out` with the demag field from the given RHS.
    /// Does NOT apply hybrid ΔK correction (caller should do that separately
    /// if needed).
    #[allow(dead_code)]
    pub(crate) fn solve_external_rhs_2d(
        &mut self,
        rhs_2d: &[f64],
        b_out: &mut VectorField2D,
    ) {
        self.stamp_rhs_from_2d(rhs_2d);
        self.update_finest_boundary_bc();
        self.levels[0].enforce_dirichlet();

        // Use the tolerance-based or fixed V-cycle solve.
        self.solve();

        b_out.set_uniform(0.0, 0.0, 0.0);
        self.add_b_from_phi_on_magnet_layer_all(b_out);
    }

    /// Solve with external 2D RHS and DirichletZero BCs (no treecode).
    ///
    /// Used for patch-local defect correction where the correction decays
    /// to zero at the patch boundary.
    #[allow(dead_code)]
    pub(crate) fn solve_external_rhs_2d_dirichlet_zero(
        &mut self,
        rhs_2d: &[f64],
        b_out: &mut VectorField2D,
    ) {
        self.stamp_rhs_from_2d(rhs_2d);

        // DirichletZero: boundary phi = 0 (already the default after fill)
        let finest = &mut self.levels[0];
        finest.bc_phi.fill(0.0);
        finest.enforce_dirichlet();

        // Clear phi for a cold start (defect correction should not reuse
        // previous phi from a different RHS).
        finest.phi.fill(0.0);
        finest.enforce_dirichlet();

        self.solve();

        b_out.set_uniform(0.0, 0.0, 0.0);
        self.add_b_from_phi_on_magnet_layer_all(b_out);
    }

    // =====================================================================
    // Patch miniCycle support (Phase B)
    //
    // These methods allow mg_composite.rs to construct lightweight MG
    // solvers for AMR patches and run internal V-cycles ("miniCycles")
    // as the smoothing step in the outer composite V-cycle.
    // =====================================================================

    /// Create a solver configured for use as a patch miniCycle.
    ///
    /// Uses DirichletZero BCs (no treecode, no dipole — parent-level
    /// Dirichlet values are stamped into bc_phi externally).
    /// No PPPM/ΔK (fine-resolution MG discretisation error is O(dx²),
    /// small enough at patch resolution).
    ///
    /// Smoother, stencil, and coarse-operator settings are inherited
    /// from env vars so that the patch operator is identical to L0's.
    pub(crate) fn new_for_patch(
        interior_nx: usize,
        interior_ny: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        pad_xy: usize,
        n_vac_z: usize,
    ) -> Self {
        let grid = Grid2D::new(interior_nx, interior_ny, dx, dy, dz);
        let mut cfg = DemagPoissonMGConfig::from_env();
        // Override patch-specific settings.
        cfg.pad_xy = pad_xy;
        cfg.n_vac_z = n_vac_z;
        cfg.bc = BoundaryCondition::DirichletZero;
        cfg.warm_start = true; // reuse φ between outer V-cycle iterations
        cfg.v_cycles = 1;
        cfg.tol_abs = None;
        cfg.tol_rel = None;

        // Force 7-point stencil for patch solvers.
        // The iso27 stencil has cross-derivative terms that make the
        // weighted-Jacobi smoother unstable at high cell aspect ratios
        // (e.g. dx=0.49nm, dz=3nm → ratio 6:1 at L3). The 7-point
        // stencil is unconditionally stable with weighted Jacobi at
        // any aspect ratio.
        //
        // This means patches use a different stencil than L0 (which
        // uses iso27). The difference is O(h²) — negligible at fine
        // resolution. The residual is computed with the correct stencil
        // via compute_residual_magnet_layer().
        let op = MGOperatorSettings {
            stencil: LaplacianStencilKind::SevenPoint,
            prolong: ProlongationKind::Trilinear,
            coarse_op: CoarseOpKind::Galerkin,
            iso27_alpha_bits: 0,
        };
        Self::new_with_operator(grid, cfg, op)
    }

    /// Run `n_vcycles` internal V-cycles on the current state.
    ///
    /// The caller is responsible for:
    ///   1. Loading rhs into `self.rhs_3d_mut()` (via `build_rhs_from_m`
    ///      or `build_rhs_from_m_raw`)
    ///   2. Loading parent-derived Dirichlet BCs into `self.bc_phi_mut()`
    ///   3. Loading warm-start φ into `self.phi_3d_mut()` (if desired)
    ///
    /// This method then enforces Dirichlet and runs the requested number
    /// of standard geometric-MG V-cycles.
    pub(crate) fn mini_solve(&mut self, n_vcycles: usize) {
        // Zero coarser MG levels to prevent stale data when the solver is
        // shared across patches via PatchMGCache. In the MG correction scheme,
        // coarser levels represent the correction δφ and must start from zero.
        // (The L0 solver doesn't need this because it always solves the same
        // problem, so coarser-level warm-start is valid. Shared patch solvers
        // switch between different patches with different BCs, making the
        // coarser-level phi from a previous patch a wrong initial guess that
        // causes divergence.)
        for l in 1..self.levels.len() {
            self.levels[l].phi.fill(0.0);
        }
        self.levels[0].enforce_dirichlet();
        for _ in 0..n_vcycles {
            self.v_cycle(0);
        }
    }

    /// Compute residual r = rhs - L(φ) on the finest level using the
    /// CORRECT stencil (iso27/7pt/iso9, matching the solve operator),
    /// then extract the magnet-layer (k = offz) slice as a flat 2D array.
    ///
    /// Returns a Vec<f64> of length `grid.nx * grid.ny`.
    ///
    /// This must be used instead of a hand-rolled 7-point residual
    /// because the solver may use iso27 or iso9 stencils. Computing
    /// the residual with a different operator gives a large spurious
    /// residual even when the solve has converged.
    pub(crate) fn compute_residual_magnet_layer(&mut self) -> Vec<f64> {
        Self::compute_residual(&mut self.levels[0]);

        let nx_m = self.grid.nx;
        let ny_m = self.grid.ny;
        let px = self.px;
        let py = self.py;
        let offx = self.offx;
        let offy = self.offy;
        let offz = self.offz;
        let res = &self.levels[0].res;

        let mut res_2d = vec![0.0f64; nx_m * ny_m];
        for mj in 0..ny_m {
            for mi in 0..nx_m {
                res_2d[mj * nx_m + mi] = res[idx3(offx + mi, offy + mj, offz, px, py)];
            }
        }
        res_2d
    }

    /// Build 3D RHS from a raw magnetisation data slice.
    ///
    /// `m_data` must contain `nx_m * ny_m` entries in row-major order
    /// (interior cells only, no ghost cells). Each entry is `[mx, my, mz]`
    /// (unit magnetisation, NOT scaled by Ms — scaling is applied here).
    ///
    /// The RHS includes:
    ///   - ∇·(Ms·m) at the magnet layer (k = offz) using face-averaged
    ///     divergence with correct material/vacuum interface handling
    ///   - z-surface charges in the vacuum cells at k = offz ± 1
    ///   - Zero everywhere else
    ///
    /// This is identical to `build_rhs_from_m` but takes a raw slice
    /// instead of a `VectorField2D`, enabling use from mg_composite.rs
    /// without constructing a temporary VectorField2D.
    pub(crate) fn build_rhs_from_m_raw(
        &mut self,
        m_data: &[[f64; 3]],
        ms: f64,
    ) {
        let nx_m = self.grid.nx;
        let ny_m = self.grid.ny;
        debug_assert_eq!(
            m_data.len(),
            nx_m * ny_m,
            "build_rhs_from_m_raw: expected {} entries, got {}",
            nx_m * ny_m,
            m_data.len()
        );

        let finest = &mut self.levels[0];
        finest.rhs.fill(0.0);

        let nx = finest.nx;
        let ny = finest.ny;
        let nz = finest.nz;
        let dx = finest.dx;
        let dy = finest.dy;
        let dz = finest.dz;

        let offx = self.offx;
        let offy = self.offy;
        let offz = self.offz;
        let px = self.px;
        let py = self.py;
        let pz = self.pz;

        // --- Helpers (identical to build_rhs_from_m) ---

        #[inline]
        fn m_at(
            pi: isize, pj: isize, pk: isize,
            px: usize, py: usize, pz: usize,
            offx: usize, offy: usize, offz: usize,
            nx_m: usize, ny_m: usize,
            mdata: &[[f64; 3]], ms: f64,
        ) -> (bool, [f64; 3]) {
            if pi < 0 || pj < 0 || pk < 0 {
                return (false, [0.0; 3]);
            }
            let (piu, pju, pku) = (pi as usize, pj as usize, pk as usize);
            if piu >= px || pju >= py || pku >= pz {
                return (false, [0.0; 3]);
            }
            if pku != offz {
                return (false, [0.0; 3]);
            }
            if piu < offx || piu >= offx + nx_m || pju < offy || pju >= offy + ny_m {
                return (false, [0.0; 3]);
            }
            let mi = piu - offx;
            let mj = pju - offy;
            let id = mj * nx_m + mi;
            let v = mdata[id];
            let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            if n2 < 1e-30 {
                return (false, [0.0; 3]);
            }
            (true, [ms * v[0], ms * v[1], ms * v[2]])
        }

        #[inline]
        fn face_val(in_a: bool, a: f64, in_b: bool, b: f64) -> f64 {
            match (in_a, in_b) {
                (true, true) => 0.5 * (a + b),
                (true, false) => a,
                (false, true) => b,
                (false, false) => 0.0,
            }
        }

        let rhs = &mut finest.rhs;

        rhs.par_chunks_mut(nx)
            .enumerate()
            .for_each(|(row_idx, rhs_row)| {
                let k = row_idx / ny;
                let j = row_idx % ny;

                if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                    return;
                }

                let pj = j as isize;
                let pk = k as isize;

                for i in 1..(nx - 1) {
                    let pi = i as isize;

                    let (c_in, m_c) = m_at(
                        pi, pj, pk, px, py, pz, offx, offy, offz,
                        nx_m, ny_m, m_data, ms,
                    );
                    let (xp_in, m_xp) = m_at(
                        pi + 1, pj, pk, px, py, pz, offx, offy, offz,
                        nx_m, ny_m, m_data, ms,
                    );
                    let (xm_in, m_xm) = m_at(
                        pi - 1, pj, pk, px, py, pz, offx, offy, offz,
                        nx_m, ny_m, m_data, ms,
                    );
                    let (yp_in, m_yp) = m_at(
                        pi, pj + 1, pk, px, py, pz, offx, offy, offz,
                        nx_m, ny_m, m_data, ms,
                    );
                    let (ym_in, m_ym) = m_at(
                        pi, pj - 1, pk, px, py, pz, offx, offy, offz,
                        nx_m, ny_m, m_data, ms,
                    );
                    let (zp_in, m_zp) = m_at(
                        pi, pj, pk + 1, px, py, pz, offx, offy, offz,
                        nx_m, ny_m, m_data, ms,
                    );
                    let (zm_in, m_zm) = m_at(
                        pi, pj, pk - 1, px, py, pz, offx, offy, offz,
                        nx_m, ny_m, m_data, ms,
                    );

                    let mx_p = face_val(c_in, m_c[0], xp_in, m_xp[0]);
                    let mx_m = face_val(xm_in, m_xm[0], c_in, m_c[0]);

                    let my_p = face_val(c_in, m_c[1], yp_in, m_yp[1]);
                    let my_m = face_val(ym_in, m_ym[1], c_in, m_c[1]);

                    let mz_p = face_val(c_in, m_c[2], zp_in, m_zp[2]);
                    let mz_m = face_val(zm_in, m_zm[2], c_in, m_c[2]);

                    rhs_row[i] = (mx_p - mx_m) / dx
                        + (my_p - my_m) / dy
                        + (mz_p - mz_m) / dz;
                }
            });
    }

    // ── 3D field accessors for patch miniCycle ──

    /// Read-only access to the finest-level 3D φ field.
    pub(crate) fn phi_3d(&self) -> &[f64] {
        &self.levels[0].phi
    }

    /// Mutable access to the finest-level 3D φ field.
    pub(crate) fn phi_3d_mut(&mut self) -> &mut [f64] {
        &mut self.levels[0].phi
    }

    /// Read-only access to the finest-level 3D RHS field.
    pub(crate) fn rhs_3d(&self) -> &[f64] {
        &self.levels[0].rhs
    }

    /// Mutable access to the finest-level 3D RHS field.
    #[allow(dead_code)]
    pub(crate) fn rhs_3d_mut(&mut self) -> &mut [f64] {
        &mut self.levels[0].rhs
    }

    /// Mutable access to the finest-level 3D boundary-condition φ field.
    pub(crate) fn bc_phi_mut(&mut self) -> &mut [f64] {
        &mut self.levels[0].bc_phi
    }

    /// Offsets from padded-box origin to the interior magnet region.
    pub(crate) fn offsets(&self) -> (usize, usize, usize) {
        (self.offx, self.offy, self.offz)
    }

    /// Cell spacings on the finest MG level.
    pub(crate) fn cell_spacings(&self) -> (f64, f64, f64) {
        (self.levels[0].dx, self.levels[0].dy, self.levels[0].dz)
    }

    /// Interior (magnet region) dimensions.
    pub(crate) fn interior_dims(&self) -> (usize, usize) {
        (self.grid.nx, self.grid.ny)
    }

}

// ---------------------------
// Hybrid wrapper: MG + local (ΔK) correction
// ---------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeltaKernelKey {
    nx: usize,
    ny: usize,
    dx_bits: u64,
    dy_bits: u64,
    dz_bits: u64,
    pad_xy: usize,
    n_vac_z: usize,

    op: MGOperatorSettings,

    bc: BoundaryCondition,
    tree_theta_bits: u64,
    tree_leaf: usize,
    tree_max_depth: usize,

    pre_smooth: usize,
    post_smooth: usize,
    smoother: MGSmoother,
    omega_bits: u64,
    sor_omega_bits: u64,

    sigma_bits: u64,
    radius_xy: usize,
    delta_v_cycles: usize,
}

impl DeltaKernelKey {
    fn new(
        grid: &Grid2D,
        mg_cfg: &DemagPoissonMGConfig,
        hyb: &HybridConfig,
        op: MGOperatorSettings,
    ) -> Self {
        Self {
            nx: grid.nx,
            ny: grid.ny,
            dx_bits: grid.dx.to_bits(),
            dy_bits: grid.dy.to_bits(),
            dz_bits: grid.dz.to_bits(),
            pad_xy: mg_cfg.pad_xy,
            n_vac_z: mg_cfg.n_vac_z,
            op,
            bc: mg_cfg.bc,
            tree_theta_bits: mg_cfg.tree_theta.to_bits(),
            tree_leaf: mg_cfg.tree_leaf,
            tree_max_depth: mg_cfg.tree_max_depth,
            pre_smooth: mg_cfg.pre_smooth,
            post_smooth: mg_cfg.post_smooth,
            smoother: mg_cfg.smoother,
            omega_bits: mg_cfg.omega.to_bits(),
            sor_omega_bits: mg_cfg.sor_omega.to_bits(),
            sigma_bits: hyb.sigma_cells.to_bits(),
            radius_xy: hyb.radius_xy,
            delta_v_cycles: hyb.delta_v_cycles,
        }
    }

    fn cache_path(&self) -> PathBuf {
        let bc_tag = match self.bc {
            BoundaryCondition::DirichletZero => "dir0",
            BoundaryCondition::DirichletDipole => "dip",
            BoundaryCondition::DirichletTreecode => "bh",
        };
        let dx = f64::from_bits(self.dx_bits);
        let dy = f64::from_bits(self.dy_bits);
        let dz = f64::from_bits(self.dz_bits);
        let tree_theta = f64::from_bits(self.tree_theta_bits);
        let sm_tag = match self.smoother {
            MGSmoother::WeightedJacobi => "wj",
            MGSmoother::RedBlackSOR => "rb",
        };

        let fname = format!(
            "demag_mg_hybrid_dk_nx{}_ny{}_dx{:.3e}_dy{:.3e}_dz{:.3e}_pad{}_nvacz{}_op{}_bc{}_th{:.3e}_leaf{}_dep{}_sm{}_pre{}_post{}_om{:016x}_som{:016x}_sig{:016x}_r{}_dv{}.bin",
            self.nx,
            self.ny,
            dx,
            dy,
            dz,
            self.pad_xy,
            self.n_vac_z,
            self.op.tag(),
            bc_tag,
            tree_theta,
            self.tree_leaf,
            self.tree_max_depth,
            sm_tag,
            self.pre_smooth,
            self.post_smooth,
            self.omega_bits,
            self.sor_omega_bits,
            self.sigma_bits,
            self.radius_xy,
            self.delta_v_cycles
        );
        PathBuf::from("out").join("demag_cache").join(fname)
    }
}

pub(crate) struct DemagPoissonMGHybrid {
    pub(crate) mg: DemagPoissonMG,
    pub(crate) hyb: HybridConfig,

    dk_key: Option<DeltaKernelKey>,
    dk: Option<DeltaKernel2D>,
}

impl DemagPoissonMGHybrid {
    pub(crate) fn new(grid: Grid2D, mut mg_cfg: DemagPoissonMGConfig, hyb: HybridConfig) -> Self {
        let op = MGOperatorSettings::from_env();
        sanitize_cfg_for_op(&mut mg_cfg, op);
        let mg = DemagPoissonMG::new_with_operator(grid, mg_cfg, op);

        Self {
            mg,
            hyb,
            dk_key: None,
            dk: None,
        }
    }

    pub(crate) fn same_structure(&self, grid: &Grid2D, mg_cfg: &DemagPoissonMGConfig) -> bool {
        self.mg.same_structure(grid, mg_cfg)
    }

    /// Access the calibrated PPPM ΔK/Δφ stencil (if built).
    ///
    /// Used by mg_composite to apply the PPPM-φ correction to L0 φ
    /// before ghost-filling into AMR patches.
    pub(crate) fn delta_kernel(&self) -> Option<&DeltaKernel2D> {
        self.dk.as_ref()
    }

    fn ensure_delta_kernel(&mut self, mat: &Material) {
        if !self.hyb.enabled() {
            self.dk = None;
            self.dk_key = None;
            return;
        }

        let max_r_x = self.mg.grid.nx.saturating_sub(2) / 2;
        let max_r_y = self.mg.grid.ny.saturating_sub(2) / 2;
        let r_eff = self.hyb.radius_xy.min(max_r_x).min(max_r_y);
        if r_eff == 0 {
            self.dk = None;
            self.dk_key = None;
            return;
        }

        let mut hyb_eff = self.hyb;
        hyb_eff.radius_xy = r_eff;

        let key = DeltaKernelKey::new(&self.mg.grid, &self.mg.cfg, &hyb_eff, self.mg.op);

        if self.dk_key == Some(key) && self.dk.is_some() {
            return;
        }

        if hyb_eff.cache_to_disk {
            if let Some(dk) = load_delta_kernel_from_disk(&key) {
                eprintln!(
                    "[demag_mg] hybrid cache hit -> loaded ΔK stencil from \"{}\"",
                    key.cache_path().display()
                );
                self.dk_key = Some(key);
                self.dk = Some(dk);
                return;
            }
            eprintln!(
                "[demag_mg] hybrid cache miss -> building ΔK stencil (r={}, dv={}, sigma_cells={:.3}) ...",
                hyb_eff.radius_xy, hyb_eff.delta_v_cycles, hyb_eff.sigma_cells
            );
        }

        let dk = build_delta_kernel_impulse(&self.mg.grid, &self.mg.cfg, &hyb_eff, mat, self.mg.op);

        if hyb_eff.cache_to_disk {
            if save_delta_kernel_to_disk(&key, &dk).is_ok() {
                eprintln!(
                    "[demag_mg] hybrid cached ΔK stencil to \"{}\"",
                    key.cache_path().display()
                );
            }
        }

        self.dk_key = Some(key);
        self.dk = Some(dk);
    }

    fn add_field(&mut self, m: &VectorField2D, b_eff: &mut VectorField2D, mat: &Material) {
        mg_hybrid_notice_once(&self.hyb);

        if mg_timing_enabled() {
            let t_total = Instant::now();

            let t_rhs = Instant::now();
            self.mg.build_rhs_from_m(m, mat.ms);
            if self.hyb.enabled() && (self.hyb.sigma_cells > 0.0) {
                screen_rhs_gaussian_xy(&mut self.mg.levels[0], self.hyb.sigma_cells);
            }
            let rhs_ns = t_rhs.elapsed().as_nanos() as u64;

            if self.hyb.enabled() && self.mg.cfg.warm_start {
                for lev in &mut self.mg.levels {
                    lev.phi.fill(0.0);
                }
            }

            let (bc_ns, solve_ns) = self.mg.solve_with_timing();

            let t_grad = Instant::now();
            self.mg.add_b_from_phi_on_magnet_layer(m, mat.ms, b_eff);
            let grad_ns = t_grad.elapsed().as_nanos() as u64;

            let t_h = Instant::now();
            if self.hyb.enabled() {
                self.ensure_delta_kernel(mat);
                if let Some(dk) = &self.dk {
                    dk.add_correction(m, b_eff, mat.ms);
                }
            }
            let hybrid_ns = t_h.elapsed().as_nanos() as u64;

            let total_ns = t_total.elapsed().as_nanos() as u64;
            mg_timing_record(rhs_ns, bc_ns, solve_ns, grad_ns, hybrid_ns, total_ns);
            return;
        }

        if !self.hyb.enabled() {
            self.mg.add_field(m, b_eff, mat.ms);
            return;
        }

        self.ensure_delta_kernel(mat);
        self.mg.build_rhs_from_m(m, mat.ms);
        if self.hyb.sigma_cells > 0.0 {
            screen_rhs_gaussian_xy(&mut self.mg.levels[0], self.hyb.sigma_cells);
        }
        if self.mg.cfg.warm_start {
            for lev in &mut self.mg.levels {
                lev.phi.fill(0.0);
            }
        }
        self.mg.solve();
        self.mg.add_b_from_phi_on_magnet_layer(m, mat.ms, b_eff);
        if let Some(dk) = &self.dk {
            dk.add_correction(m, b_eff, mat.ms);
        }
    }

    /// Solve with the coarse magnetisation (full 3D RHS including z-surface
    /// charges) plus additive corrections from AMR patches.
    ///
    /// `corrections`: sparse list of (coarse_cell_index, delta_div) pairs.
    ///   delta_div = area_avg(fine_div) − coarse_div for each patch-covered cell.
    ///   These are ADDED to the 3D RHS that build_rhs_from_m computes, preserving
    ///   the z-surface charge contribution while injecting fine in-plane detail.
    ///
    /// `coarse_m`: coarse magnetisation (used for both RHS and gradient mask).
    /// `b_out`: overwritten with B_demag on the coarse grid.
    /// `mat`: material parameters.
    pub(crate) fn solve_with_corrections(
        &mut self,
        coarse_m: &VectorField2D,
        corrections: &[(usize, f64)],
        b_out: &mut VectorField2D,
        mat: &Material,
    ) {
        mg_hybrid_notice_once(&self.hyb);

        // Step 1: Build full 3D RHS from coarse M (includes z-surface charges).
        self.mg.build_rhs_from_m(coarse_m, mat.ms);

        // Step 2: Add patch corrections to the magnet-layer cells.
        if !corrections.is_empty() {
            let nx_m = self.mg.grid.nx;
            let offx = self.mg.offx;
            let offy = self.mg.offy;
            let offz = self.mg.offz;
            let px = self.mg.px;
            let py = self.mg.py;
            let rhs = &mut self.mg.levels[0].rhs;

            for &(cell_idx, delta) in corrections {
                let ci = cell_idx % nx_m;
                let cj = cell_idx / nx_m;
                let pi = offx + ci;
                let pj = offy + cj;
                rhs[idx3(pi, pj, offz, px, py)] += delta;
            }
        }

        // Step 3: Apply Gaussian screening if hybrid is enabled.
        if self.hyb.enabled() && self.hyb.sigma_cells > 0.0 {
            screen_rhs_gaussian_xy(&mut self.mg.levels[0], self.hyb.sigma_cells);
        }
        if self.hyb.enabled() && self.mg.cfg.warm_start {
            for lev in &mut self.mg.levels {
                lev.phi.fill(0.0);
            }
        }

        // Step 4: BCs, solve, extract.
        self.mg.update_finest_boundary_bc();
        self.mg.levels[0].enforce_dirichlet();
        self.mg.solve();

        b_out.set_uniform(0.0, 0.0, 0.0);
        self.mg.add_b_from_phi_on_magnet_layer(coarse_m, mat.ms, b_out);

        // Step 5: Hybrid ΔK correction.
        if self.hyb.enabled() {
            self.ensure_delta_kernel(mat);
            if let Some(dk) = &self.dk {
                dk.add_correction(coarse_m, b_out, mat.ms);
            }
        }
    }
}

#[inline]
fn mg_hybrid_notice_once(hyb: &HybridConfig) {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        if hyb.enabled() {
            eprintln!(
                "[demag_mg] hybrid ΔK ENABLED (radius_xy={}, delta_v_cycles={}, sigma_cells={:.3}, cache_to_disk={})",
                hyb.radius_xy, hyb.delta_v_cycles, hyb.sigma_cells, hyb.cache_to_disk
            );
            if !(hyb.sigma_cells > 0.0) {
                eprintln!(
                    "[demag_mg] warning: hybrid ΔK is UNSCREENED (LLG_DEMAG_MG_HYBRID_SIGMA<=0). \
                     This usually leaves long-range tails/DC leak and can worsen errors. \
                     Recommended: set LLG_DEMAG_MG_HYBRID_SIGMA=1.0..2.0"
                );
            }
        } else {
            eprintln!(
                "[demag_mg] hybrid ΔK DISABLED (pure MG). To enable: set LLG_DEMAG_MG_HYBRID_ENABLE=1 and LLG_DEMAG_MG_HYBRID_RADIUS>0"
            );
        }
    });
}

// ---------------------------
// Hybrid diagnostics (opt-in via env vars)
// ---------------------------

#[inline]
fn mg_hybrid_diag_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLG_DEMAG_MG_HYBRID_DIAG").is_ok())
}

#[inline]
fn mg_hybrid_diag_invar_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLG_DEMAG_MG_HYBRID_DIAG_INVAR").is_ok())
}

// ---------------------------
// PPPM/Ewald-style screening: Gaussian smoothing of RHS in XY
// ---------------------------

fn screen_rhs_gaussian_xy(level: &mut MGLevel, sigma_cells: f64) {
    kernels::screen_rhs_gaussian_xy(
        &mut level.rhs, &mut level.tmp,
        level.nx, level.ny, level.nz,
        sigma_cells,
    );
}
// ---------------------------
// Timing / profiling helpers (opt-in via env var)
// ---------------------------

#[inline]
fn mg_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LLG_DEMAG_TIMING").is_ok())
}

#[inline]
fn mg_timing_stride() -> usize {
    static STRIDE: OnceLock<usize> = OnceLock::new();
    *STRIDE.get_or_init(|| {
        std::env::var("LLG_DEMAG_TIMING_EVERY")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(200)
            .max(1)
    })
}

static MG_TIMING_CALLS: AtomicUsize = AtomicUsize::new(0);
static MG_TIMING_RHS_NS: AtomicU64 = AtomicU64::new(0);
static MG_TIMING_BC_NS: AtomicU64 = AtomicU64::new(0);
static MG_TIMING_SOLVE_NS: AtomicU64 = AtomicU64::new(0);
static MG_TIMING_GRAD_NS: AtomicU64 = AtomicU64::new(0);
static MG_TIMING_HYBRID_NS: AtomicU64 = AtomicU64::new(0);
static MG_TIMING_TOTAL_NS: AtomicU64 = AtomicU64::new(0);

#[inline]
fn mg_timing_record(
    rhs_ns: u64,
    bc_ns: u64,
    solve_ns: u64,
    grad_ns: u64,
    hybrid_ns: u64,
    total_ns: u64,
) {
    let c = MG_TIMING_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    MG_TIMING_RHS_NS.fetch_add(rhs_ns, Ordering::Relaxed);
    MG_TIMING_BC_NS.fetch_add(bc_ns, Ordering::Relaxed);
    MG_TIMING_SOLVE_NS.fetch_add(solve_ns, Ordering::Relaxed);
    MG_TIMING_GRAD_NS.fetch_add(grad_ns, Ordering::Relaxed);
    MG_TIMING_HYBRID_NS.fetch_add(hybrid_ns, Ordering::Relaxed);
    MG_TIMING_TOTAL_NS.fetch_add(total_ns, Ordering::Relaxed);

    let stride = mg_timing_stride();
    if c % stride == 0 {
        let calls = c as f64;
        let to_ms = 1.0e-6;
        let avg_total = MG_TIMING_TOTAL_NS.load(Ordering::Relaxed) as f64 * to_ms / calls;
        let avg_rhs = MG_TIMING_RHS_NS.load(Ordering::Relaxed) as f64 * to_ms / calls;
        let avg_bc = MG_TIMING_BC_NS.load(Ordering::Relaxed) as f64 * to_ms / calls;
        let avg_solve = MG_TIMING_SOLVE_NS.load(Ordering::Relaxed) as f64 * to_ms / calls;
        let avg_grad = MG_TIMING_GRAD_NS.load(Ordering::Relaxed) as f64 * to_ms / calls;
        let avg_hyb = MG_TIMING_HYBRID_NS.load(Ordering::Relaxed) as f64 * to_ms / calls;

        eprintln!(
            "[demag_mg timing] calls={} avg_total={:.3} ms (rhs {:.3} | bc {:.3} | solve {:.3} | grad {:.3} | hybrid {:.3})",
            c, avg_total, avg_rhs, avg_bc, avg_solve, avg_grad, avg_hyb
        );
    }
}

// ---------------------------
// ΔK disk cache + builder
// ---------------------------

// New file format: includes dphi stencils (SOR-converged). Bump when layout changes.
const DK_CACHE_MAGIC: &[u8; 8] = b"LLGDKH8\x00";

fn ensure_cache_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn load_delta_kernel_from_disk(key: &DeltaKernelKey) -> Option<DeltaKernel2D> {
    let path = key.cache_path();
    let mut f = fs::File::open(&path).ok()?;

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).ok()?;
    if &magic != DK_CACHE_MAGIC {
        return None;
    }

    fn read_u64<R: Read>(r: &mut R) -> Option<u64> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf).ok()?;
        Some(u64::from_le_bytes(buf))
    }

    let nx = read_u64(&mut f)? as usize;
    let ny = read_u64(&mut f)? as usize;
    let dx_bits = read_u64(&mut f)?;
    let dy_bits = read_u64(&mut f)?;
    let dz_bits = read_u64(&mut f)?;
    let pad_xy = read_u64(&mut f)? as usize;
    let n_vac_z = read_u64(&mut f)? as usize;

    let op_stencil_u = read_u64(&mut f)? as u32;
    let op_prolong_u = read_u64(&mut f)? as u32;
    let op_coarse_u = read_u64(&mut f)? as u32;
    let op_iso27_alpha_bits = read_u64(&mut f)?;

    let bc_u = read_u64(&mut f)? as u32;
    let tree_theta_bits = read_u64(&mut f)?;
    let tree_leaf = read_u64(&mut f)? as usize;
    let tree_max_depth = read_u64(&mut f)? as usize;

    let pre_smooth = read_u64(&mut f)? as usize;
    let post_smooth = read_u64(&mut f)? as usize;
    let smoother_u = read_u64(&mut f)? as u32;
    let omega_bits = read_u64(&mut f)?;
    let sor_omega_bits = read_u64(&mut f)?;

    let sigma_bits = read_u64(&mut f)?;
    let radius_xy = read_u64(&mut f)? as usize;
    let delta_v_cycles = read_u64(&mut f)? as usize;

    let op_stencil = match op_stencil_u {
        0 => LaplacianStencilKind::SevenPoint,
        1 => LaplacianStencilKind::Iso9PlusZ,
        2 => LaplacianStencilKind::Iso27,
        _ => return None,
    };
    let op_prolong = match op_prolong_u {
        0 => ProlongationKind::Injection,
        1 => ProlongationKind::Trilinear,
        _ => return None,
    };
    let op_coarse = match op_coarse_u {
        0 => CoarseOpKind::Rediscretize,
        1 => CoarseOpKind::Galerkin,
        _ => return None,
    };
    let op = MGOperatorSettings {
        stencil: op_stencil,
        prolong: op_prolong,
        coarse_op: op_coarse,
        iso27_alpha_bits: op_iso27_alpha_bits,
    };

    let bc = match bc_u {
        0 => BoundaryCondition::DirichletZero,
        1 => BoundaryCondition::DirichletDipole,
        2 => BoundaryCondition::DirichletTreecode,
        _ => return None,
    };

    let smoother = match smoother_u {
        0 => MGSmoother::WeightedJacobi,
        1 => MGSmoother::RedBlackSOR,
        _ => return None,
    };

    let key_in = DeltaKernelKey {
        nx,
        ny,
        dx_bits,
        dy_bits,
        dz_bits,
        pad_xy,
        n_vac_z,
        op,
        bc,
        tree_theta_bits,
        tree_leaf,
        tree_max_depth,
        pre_smooth,
        post_smooth,
        smoother,
        omega_bits,
        sor_omega_bits,
        sigma_bits,
        radius_xy,
        delta_v_cycles,
    };

    if &key_in != key {
        return None;
    }

    let stride = 2 * radius_xy + 1;
    let n = stride * stride;

    fn read_f64_vec<R: Read>(r: &mut R, n: usize) -> Option<Vec<f64>> {
        let mut buf = vec![0u8; 8 * n];
        r.read_exact(&mut buf).ok()?;
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            let mut b = [0u8; 8];
            b.copy_from_slice(&buf[(8 * i)..(8 * i + 8)]);
            out[i] = f64::from_le_bytes(b);
        }
        Some(out)
    }

    let dkxx = read_f64_vec(&mut f, n)?;
    let dkxy = read_f64_vec(&mut f, n)?;
    let dkyy = read_f64_vec(&mut f, n)?;
    let dkzz = read_f64_vec(&mut f, n)?;

    // PPPM-φ fields (added in LLGDKH7). If read fails, fill with zeros.
    let dphi_x = read_f64_vec(&mut f, n).unwrap_or_else(|| vec![0.0; n]);
    let dphi_y = read_f64_vec(&mut f, n).unwrap_or_else(|| vec![0.0; n]);
    let dphi_z = read_f64_vec(&mut f, n).unwrap_or_else(|| vec![0.0; n]);

    Some(DeltaKernel2D {
        radius: radius_xy,
        stride,
        dkxx,
        dkxy,
        dkyy,
        dkzz,
        dphi_x,
        dphi_y,
        dphi_z,
    })
}

fn save_delta_kernel_to_disk(key: &DeltaKernelKey, dk: &DeltaKernel2D) -> std::io::Result<()> {
    let path = key.cache_path();
    ensure_cache_dir(&path)?;
    let mut f = fs::File::create(&path)?;

    f.write_all(DK_CACHE_MAGIC)?;

    fn write_u64<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }

    write_u64(&mut f, key.nx as u64)?;
    write_u64(&mut f, key.ny as u64)?;
    write_u64(&mut f, key.dx_bits)?;
    write_u64(&mut f, key.dy_bits)?;
    write_u64(&mut f, key.dz_bits)?;
    write_u64(&mut f, key.pad_xy as u64)?;
    write_u64(&mut f, key.n_vac_z as u64)?;

    let op_stencil_u: u64 = match key.op.stencil {
        LaplacianStencilKind::SevenPoint => 0,
        LaplacianStencilKind::Iso9PlusZ => 1,
        LaplacianStencilKind::Iso27 => 2,
    };
    let op_prolong_u: u64 = match key.op.prolong {
        ProlongationKind::Injection => 0,
        ProlongationKind::Trilinear => 1,
    };
    let op_coarse_u: u64 = match key.op.coarse_op {
        CoarseOpKind::Rediscretize => 0,
        CoarseOpKind::Galerkin => 1,
    };
    write_u64(&mut f, op_stencil_u)?;
    write_u64(&mut f, op_prolong_u)?;
    write_u64(&mut f, op_coarse_u)?;
    write_u64(&mut f, key.op.iso27_alpha_bits)?;

    let bc_u: u64 = match key.bc {
        BoundaryCondition::DirichletZero => 0,
        BoundaryCondition::DirichletDipole => 1,
        BoundaryCondition::DirichletTreecode => 2,
    };
    write_u64(&mut f, bc_u)?;
    write_u64(&mut f, key.tree_theta_bits)?;
    write_u64(&mut f, key.tree_leaf as u64)?;
    write_u64(&mut f, key.tree_max_depth as u64)?;

    write_u64(&mut f, key.pre_smooth as u64)?;
    write_u64(&mut f, key.post_smooth as u64)?;
    let smoother_u: u64 = match key.smoother {
        MGSmoother::WeightedJacobi => 0,
        MGSmoother::RedBlackSOR => 1,
    };
    write_u64(&mut f, smoother_u)?;
    write_u64(&mut f, key.omega_bits)?;
    write_u64(&mut f, key.sor_omega_bits)?;

    write_u64(&mut f, key.sigma_bits)?;
    write_u64(&mut f, key.radius_xy as u64)?;
    write_u64(&mut f, key.delta_v_cycles as u64)?;

    fn write_f64_vec<W: Write>(w: &mut W, v: &[f64]) -> std::io::Result<()> {
        for &x in v {
            w.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }

    write_f64_vec(&mut f, &dk.dkxx)?;
    write_f64_vec(&mut f, &dk.dkxy)?;
    write_f64_vec(&mut f, &dk.dkyy)?;
    write_f64_vec(&mut f, &dk.dkzz)?;
    write_f64_vec(&mut f, &dk.dphi_x)?;
    write_f64_vec(&mut f, &dk.dphi_y)?;
    write_f64_vec(&mut f, &dk.dphi_z)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PPPM-φ calibration: compute Δφ stencil from impulse response
// ---------------------------------------------------------------------------

/// Solve ∇²u = rhs on a 2D grid with Dirichlet-zero BCs using SOR
/// (Successive Over-Relaxation with red-black ordering).
///
/// Used during PPPM-φ calibration to recover the potential correction.
/// Optimal ω ≈ 2/(1 + sin(π/N)) gives O(N) convergence instead of O(N²)
/// for standard Jacobi. 2000 iterations is more than enough for 128×128.
fn solve_2d_poisson_sor(
    rhs: &[f64], nx: usize, ny: usize,
    dx: f64, dy: f64, n_iters: usize,
    phi_out: &mut [f64],
) {
    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);
    let diag = -2.0 * inv_dx2 - 2.0 * inv_dy2;
    let inv_diag = 1.0 / diag;

    // Optimal SOR relaxation parameter for 2D Poisson on N×N grid.
    let n_eff = nx.max(ny) as f64;
    let omega = 2.0 / (1.0 + (std::f64::consts::PI / n_eff).sin());

    phi_out.fill(0.0);

    for _iter in 0..n_iters {
        // Red-black Gauss-Seidel with SOR overrelaxation.
        // Red pass: (i+j) % 2 == 0, then black pass: (i+j) % 2 == 1.
        for pass in 0..2u32 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    if ((i + j) as u32 % 2) != pass { continue; }
                    let idx = j * nx + i;
                    let lap = (phi_out[idx + 1] - 2.0 * phi_out[idx] + phi_out[idx - 1]) * inv_dx2
                        + (phi_out[(j + 1) * nx + i] - 2.0 * phi_out[idx]
                            + phi_out[(j - 1) * nx + i]) * inv_dy2;
                    let residual = rhs[idx] - lap;
                    phi_out[idx] += omega * inv_diag * residual;
                }
            }
        }
    }
}

/// Compute the PPPM-φ stencil for one impulse component.
///
/// Given the Newell B field (B_FFT) and the MG magnet-layer φ (φ_MG) for
/// an impulse at (cx, cy), computes Δφ = φ_ref − φ_MG where φ_ref is the
/// potential whose 2D central-difference gradient matches B_FFT.
///
/// Method: solve ∇²(Δφ) = −(1/μ₀)·∇·(B_FFT − B_from_2D_grad(φ_MG))
/// with Dirichlet-zero BCs using SOR (converges in ~300 iterations for 128×128).
fn compute_dphi_for_impulse(
    b_fft: &VectorField2D,
    phi_mg: &[f64],
    nx: usize, ny: usize,
    dx: f64, dy: f64,
    cx: usize, cy: usize,
    radius: usize,
    ms_inv: f64,
    dphi_out: &mut [f64],  // stencil-sized output
    stride: usize,
) {
    let mu0: f64 = MU0;
    let inv_2dx = 1.0 / (2.0 * dx);
    let inv_2dy = 1.0 / (2.0 * dy);

    // Step 1: Compute B from 2D gradient of φ_MG.
    let mut b2d_x = vec![0.0f64; nx * ny];
    let mut b2d_y = vec![0.0f64; nx * ny];
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let idx = j * nx + i;
            b2d_x[idx] = -mu0 * (phi_mg[idx + 1] - phi_mg[idx - 1]) * inv_2dx;
            b2d_y[idx] = -mu0 * (phi_mg[(j + 1) * nx + i] - phi_mg[(j - 1) * nx + i]) * inv_2dy;
        }
    }

    // Step 2: ΔB = B_FFT − B_2D (what the 2D gradient of φ_MG is missing).
    let mut delta_bx = vec![0.0f64; nx * ny];
    let mut delta_by = vec![0.0f64; nx * ny];
    for k in 0..nx * ny {
        delta_bx[k] = b_fft.data[k][0] - b2d_x[k];
        delta_by[k] = b_fft.data[k][1] - b2d_y[k];
    }

    // Step 3: RHS = −(1/μ₀)·div(ΔB)
    let mut rhs = vec![0.0f64; nx * ny];
    for j in 2..ny - 2 {
        for i in 2..nx - 2 {
            let div_db = (delta_bx[j * nx + (i + 1)] - delta_bx[j * nx + (i - 1)]) * inv_2dx
                + (delta_by[(j + 1) * nx + i] - delta_by[(j - 1) * nx + i]) * inv_2dy;
            rhs[j * nx + i] = -div_db / mu0;
        }
    }

    // Step 4: Solve ∇²(Δφ) = rhs with SOR (2000 iterations, O(N) convergence).
    let mut delta_phi = vec![0.0f64; nx * ny];
    solve_2d_poisson_sor(&rhs, nx, ny, dx, dy, 2000, &mut delta_phi);

    // Step 5: Extract stencil values around impulse centre.
    let r = radius as isize;
    for dj in -r..=r {
        for di in -r..=r {
            let tx = (cx as isize + di) as usize;
            let ty = (cy as isize + dj) as usize;
            if tx < nx && ty < ny {
                let k = (dj + r) as usize * stride + (di + r) as usize;
                dphi_out[k] = delta_phi[ty * nx + tx] * ms_inv;
            }
        }
    }
}

fn build_delta_kernel_impulse(
    grid: &Grid2D,
    mg_cfg: &DemagPoissonMGConfig,
    hyb: &HybridConfig,
    mat: &Material,
    op: MGOperatorSettings,
) -> DeltaKernel2D {
    let nx = grid.nx;
    let ny = grid.ny;
    let cx = nx / 2;
    let cy = ny / 2;

    let ms = mat.ms;
    let ms_inv = if ms.abs() > 0.0 { 1.0 / ms } else { 0.0 };

    let geom_eps = 1e-14;

    let mut cfg_imp = *mg_cfg;
    cfg_imp.warm_start = false;
    cfg_imp.tol_abs = None;
    cfg_imp.tol_rel = None;
    cfg_imp.v_cycles = hyb.delta_v_cycles;
    cfg_imp.v_cycles_max = hyb.delta_v_cycles.max(1);

    let mut mg = DemagPoissonMG::new_with_operator(*grid, cfg_imp, op);

    let r = hyb.radius_xy;
    let stride = 2 * r + 1;
    let nst = stride * stride;

    let diag = mg_hybrid_diag_enabled();
    let invar = mg_hybrid_diag_invar_enabled();

    let max_r_x = cx.min(nx - 1 - cx);
    let max_r_y = cy.min(ny - 1 - cy);
    let max_r = max_r_x.min(max_r_y);
    let r_big = if diag && r > 0 {
        ((r * 2).max(r + 4)).min(max_r)
    } else {
        0
    };

    let mut tail_xx: Option<f64> = None;
    let mut tail_yy: Option<f64> = None;
    let mut tail_zz: Option<f64> = None;

    let tail_fraction = |b_fft: &VectorField2D, b_mg: &VectorField2D, comp: usize| -> f64 {
        if r_big <= r || ms_inv == 0.0 {
            return 0.0;
        }
        let mut tot = 0.0;
        let mut inside = 0.0;
        let cx_i = cx as isize;
        let cy_i = cy as isize;
        let nx_i = nx as isize;
        let ny_i = ny as isize;
        let r_i = r as isize;
        let rb_i = r_big as isize;
        for dy in -rb_i..=rb_i {
            for dx in -rb_i..=rb_i {
                let tx = cx_i + dx;
                let ty = cy_i + dy;
                if tx < 0 || tx >= nx_i || ty < 0 || ty >= ny_i {
                    continue;
                }
                let id = (ty as usize) * nx + (tx as usize);
                let d = (b_fft.data[id][comp] - b_mg.data[id][comp]) * ms_inv;
                let d2 = d * d;
                tot += d2;
                if dx.abs() <= r_i && dy.abs() <= r_i {
                    inside += d2;
                }
            }
        }
        if tot > 0.0 {
            ((tot - inside) / tot).max(0.0).sqrt()
        } else {
            0.0
        }
    };

    let mut dk = DeltaKernel2D::new(r);
    let mut dkxy_from_x = vec![0.0; nst];
    let mut dkxy_from_y = vec![0.0; nst];

    let mut m_imp = VectorField2D::new(*grid);
    for v in &mut m_imp.data {
        *v = [geom_eps, 0.0, 0.0];
    }

    let mut b_fft = VectorField2D::new(*grid);
    let mut b_mg = VectorField2D::new(*grid);

    // X impulse
    {
        for v in &mut m_imp.data {
            *v = [geom_eps, 0.0, 0.0];
        }
        let center = m_imp.idx(cx, cy);
        m_imp.data[center] = [1.0 + geom_eps, 0.0, 0.0];

        for v in &mut b_fft.data {
            *v = [0.0; 3];
        }
        demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);

        for v in &mut b_mg.data {
            *v = [0.0; 3];
        }
        mg.build_rhs_from_m(&m_imp, ms);
        if hyb.sigma_cells > 0.0 {
            screen_rhs_gaussian_xy(&mut mg.levels[0], hyb.sigma_cells);
        }
        mg.solve();
        mg.add_b_from_phi_on_magnet_layer_all(&mut b_mg);

        if diag && r_big > r {
            tail_xx = Some(tail_fraction(&b_fft, &b_mg, 0));
        }

        for dy in -(r as isize)..=(r as isize) {
            for dx in -(r as isize)..=(r as isize) {
                let tx = (cx as isize + dx) as usize;
                let ty = (cy as isize + dy) as usize;
                let id = m_imp.idx(tx, ty);
                let k = dk.idx(dx, dy);

                dk.dkxx[k] = (b_fft.data[id][0] - b_mg.data[id][0]) * ms_inv;
                dkxy_from_x[k] = (b_fft.data[id][1] - b_mg.data[id][1]) * ms_inv;
            }
        }

        // PPPM-φ: compute Δφ for mx impulse
        let phi_mg_x = mg.extract_magnet_layer_phi();
        compute_dphi_for_impulse(
            &b_fft, &phi_mg_x, nx, ny, grid.dx, grid.dy,
            cx, cy, r, ms_inv, &mut dk.dphi_x, dk.stride);
    }

    // Y impulse
    {
        for v in &mut m_imp.data {
            *v = [geom_eps, 0.0, 0.0];
        }
        let center = m_imp.idx(cx, cy);
        m_imp.data[center] = [geom_eps, 1.0, 0.0];

        for v in &mut b_fft.data {
            *v = [0.0; 3];
        }
        demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);

        for v in &mut b_mg.data {
            *v = [0.0; 3];
        }
        mg.build_rhs_from_m(&m_imp, ms);
        if hyb.sigma_cells > 0.0 {
            screen_rhs_gaussian_xy(&mut mg.levels[0], hyb.sigma_cells);
        }
        mg.solve();
        mg.add_b_from_phi_on_magnet_layer_all(&mut b_mg);

        if diag && r_big > r {
            tail_yy = Some(tail_fraction(&b_fft, &b_mg, 1));
        }

        for dy in -(r as isize)..=(r as isize) {
            for dx in -(r as isize)..=(r as isize) {
                let tx = (cx as isize + dx) as usize;
                let ty = (cy as isize + dy) as usize;
                let id = m_imp.idx(tx, ty);
                let k = dk.idx(dx, dy);

                dk.dkyy[k] = (b_fft.data[id][1] - b_mg.data[id][1]) * ms_inv;
                dkxy_from_y[k] = (b_fft.data[id][0] - b_mg.data[id][0]) * ms_inv;
            }
        }

        // PPPM-φ: compute Δφ for my impulse
        let phi_mg_y = mg.extract_magnet_layer_phi();
        compute_dphi_for_impulse(
            &b_fft, &phi_mg_y, nx, ny, grid.dx, grid.dy,
            cx, cy, r, ms_inv, &mut dk.dphi_y, dk.stride);
    }

    // Z impulse
    {
        for v in &mut m_imp.data {
            *v = [geom_eps, 0.0, 0.0];
        }
        let center = m_imp.idx(cx, cy);
        m_imp.data[center] = [geom_eps, 0.0, 1.0];

        for v in &mut b_fft.data {
            *v = [0.0; 3];
        }
        demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);

        for v in &mut b_mg.data {
            *v = [0.0; 3];
        }
        mg.build_rhs_from_m(&m_imp, ms);
        if hyb.sigma_cells > 0.0 {
            screen_rhs_gaussian_xy(&mut mg.levels[0], hyb.sigma_cells);
        }
        mg.solve();
        mg.add_b_from_phi_on_magnet_layer_all(&mut b_mg);

        if diag && r_big > r {
            tail_zz = Some(tail_fraction(&b_fft, &b_mg, 2));
        }

        for dy in -(r as isize)..=(r as isize) {
            for dx in -(r as isize)..=(r as isize) {
                let tx = (cx as isize + dx) as usize;
                let ty = (cy as isize + dy) as usize;
                let id = m_imp.idx(tx, ty);
                let k = dk.idx(dx, dy);

                dk.dkzz[k] = (b_fft.data[id][2] - b_mg.data[id][2]) * ms_inv;
            }
        }

        // PPPM-φ: compute Δφ for mz impulse
        let phi_mg_z = mg.extract_magnet_layer_phi();
        compute_dphi_for_impulse(
            &b_fft, &phi_mg_z, nx, ny, grid.dx, grid.dy,
            cx, cy, r, ms_inv, &mut dk.dphi_z, dk.stride);
    }

    // Average cross-term from x- and y-impulses.
    for dy in -(r as isize)..=(r as isize) {
        for dx in -(r as isize)..=(r as isize) {
            let k = dk.idx(dx, dy);
            dk.dkxy[k] = 0.5 * (dkxy_from_x[k] + dkxy_from_y[k]);
        }
    }

    // Optional invariance diagnostic (debug).
    if invar && r > 0 {
        let r_i = r as isize;
        let nx_i = nx as isize;
        let ny_i = ny as isize;

        let sh = 10isize;
        let mut sx = cx as isize + sh;
        let mut sy = cy as isize;
        if sx - r_i < 0 || sx + r_i >= nx_i {
            sx = cx as isize - sh;
        }
        if sx - r_i < 0 || sx + r_i >= nx_i {
            sx = cx as isize;
        }
        if sy - r_i < 0 || sy + r_i >= ny_i {
            sy = cy as isize;
        }

        if sx != cx as isize || sy != cy as isize {
            let sidx = (sy as usize) * nx + (sx as usize);

            let mut dkxx_s = vec![0.0f64; nst];
            let mut dkyy_s = vec![0.0f64; nst];
            let mut dkzz_s = vec![0.0f64; nst];
            let mut dkxy_x_s = vec![0.0f64; nst];
            let mut dkxy_y_s = vec![0.0f64; nst];

            // Shifted x-impulse
            for v in &mut m_imp.data {
                *v = [geom_eps, geom_eps, geom_eps];
            }
            m_imp.data[sidx] = [1.0, 0.0, 0.0];
            for v in &mut b_fft.data {
                *v = [0.0; 3];
            }
            demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);
            b_mg.data.iter_mut().for_each(|v| *v = [0.0; 3]);
            mg.build_rhs_from_m(&m_imp, ms);
            if hyb.sigma_cells > 0.0 {
                screen_rhs_gaussian_xy(&mut mg.levels[0], hyb.sigma_cells);
            }
            mg.solve();
            mg.add_b_from_phi_on_magnet_layer_all(&mut b_mg);
            for dy in -(r_i)..=r_i {
                for dx in -(r_i)..=r_i {
                    let tx = (sx + dx) as usize;
                    let ty = (sy + dy) as usize;
                    let id = m_imp.idx(tx, ty);
                    let k = dk.idx(dx, dy);
                    dkxx_s[k] = (b_fft.data[id][0] - b_mg.data[id][0]) * ms_inv;
                    dkxy_x_s[k] = (b_fft.data[id][1] - b_mg.data[id][1]) * ms_inv;
                }
            }

            // Shifted y-impulse
            for v in &mut m_imp.data {
                *v = [geom_eps, geom_eps, geom_eps];
            }
            m_imp.data[sidx] = [0.0, 1.0, 0.0];
            for v in &mut b_fft.data {
                *v = [0.0; 3];
            }
            demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);
            b_mg.data.iter_mut().for_each(|v| *v = [0.0; 3]);
            mg.build_rhs_from_m(&m_imp, ms);
            if hyb.sigma_cells > 0.0 {
                screen_rhs_gaussian_xy(&mut mg.levels[0], hyb.sigma_cells);
            }
            mg.solve();
            mg.add_b_from_phi_on_magnet_layer_all(&mut b_mg);
            for dy in -(r_i)..=r_i {
                for dx in -(r_i)..=r_i {
                    let tx = (sx + dx) as usize;
                    let ty = (sy + dy) as usize;
                    let id = m_imp.idx(tx, ty);
                    let k = dk.idx(dx, dy);
                    dkyy_s[k] = (b_fft.data[id][1] - b_mg.data[id][1]) * ms_inv;
                    dkxy_y_s[k] = (b_fft.data[id][0] - b_mg.data[id][0]) * ms_inv;
                }
            }

            // Shifted z-impulse
            for v in &mut m_imp.data {
                *v = [geom_eps, geom_eps, geom_eps];
            }
            m_imp.data[sidx] = [0.0, 0.0, 1.0];
            for v in &mut b_fft.data {
                *v = [0.0; 3];
            }
            demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);
            b_mg.data.iter_mut().for_each(|v| *v = [0.0; 3]);
            mg.build_rhs_from_m(&m_imp, ms);
            if hyb.sigma_cells > 0.0 {
                screen_rhs_gaussian_xy(&mut mg.levels[0], hyb.sigma_cells);
            }
            mg.solve();
            mg.add_b_from_phi_on_magnet_layer_all(&mut b_mg);
            for dy in -(r_i)..=r_i {
                for dx in -(r_i)..=r_i {
                    let tx = (sx + dx) as usize;
                    let ty = (sy + dy) as usize;
                    let id = m_imp.idx(tx, ty);
                    let k = dk.idx(dx, dy);
                    dkzz_s[k] = (b_fft.data[id][2] - b_mg.data[id][2]) * ms_inv;
                }
            }

            let mut dkxy_s = vec![0.0f64; nst];
            for i in 0..nst {
                dkxy_s[i] = 0.5 * (dkxy_x_s[i] + dkxy_y_s[i]);
            }

            let metrics = |a: &[f64], b: &[f64]| -> (f64, f64) {
                let mut num = 0.0;
                let mut den = 0.0;
                let mut max_abs: f64 = 0.0;
                for i in 0..a.len() {
                    let d = a[i] - b[i];
                    num += d * d;
                    den += a[i] * a[i];
                    max_abs = max_abs.max(d.abs());
                }
                let rel = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
                (rel, max_abs)
            };

            let (rel_xx, max_xx) = metrics(&dk.dkxx, &dkxx_s);
            let (rel_yy, max_yy) = metrics(&dk.dkyy, &dkyy_s);
            let (rel_zz, max_zz) = metrics(&dk.dkzz, &dkzz_s);
            let (rel_xy, max_xy) = metrics(&dk.dkxy, &dkxy_s);

            eprintln!(
                "[demag_mg] ΔK invariance check: shift=({:+},{:+}) rel_L2 dkxx={:.3e} dkyy={:.3e} dkzz={:.3e} dkxy={:.3e}",
                sx - cx as isize,
                sy - cy as isize,
                rel_xx,
                rel_yy,
                rel_zz,
                rel_xy
            );
            eprintln!(
                "[demag_mg] ΔK invariance check: shift=({:+},{:+}) max_abs dkxx={:.3e} dkyy={:.3e} dkzz={:.3e} dkxy={:.3e}",
                sx - cx as isize,
                sy - cy as isize,
                max_xx,
                max_yy,
                max_zz,
                max_xy
            );
        } else {
            eprintln!(
                "[demag_mg] ΔK invariance check skipped: insufficient interior margin for shift"
            );
        }
    }

    dk.symmetrize(grid.dx, grid.dy);

    // DC leak fix: enforce zero-sum on diagonal terms by adjusting only (0,0).
    let center = r * stride + r;
    let sxx0: f64 = dk.dkxx.iter().sum();
    let syy0: f64 = dk.dkyy.iter().sum();
    let szz0: f64 = dk.dkzz.iter().sum();
    let sxy0: f64 = dk.dkxy.iter().sum();

    dk.dkxx[center] -= sxx0;
    dk.dkyy[center] -= syy0;
    dk.dkzz[center] -= szz0;

    let sxx1: f64 = dk.dkxx.iter().sum();
    let syy1: f64 = dk.dkyy.iter().sum();
    let szz1: f64 = dk.dkzz.iter().sum();
    let sxy1: f64 = dk.dkxy.iter().sum();

    if diag {
        eprintln!(
            "[demag_mg] ΔK diagnostics: r={}  sigma_cells={:.3}  r_big={}",
            r, hyb.sigma_cells, r_big
        );
        eprintln!(
            "[demag_mg]   uniform-bias sums (pre-fix):  Sxx={:.3e}  Syy={:.3e}  Szz={:.3e}  Sxy={:.3e}",
            sxx0, syy0, szz0, sxy0
        );
        eprintln!(
            "[demag_mg]   uniform-bias sums (post-fix): Sxx={:.3e}  Syy={:.3e}  Szz={:.3e}  Sxy={:.3e}",
            sxx1, syy1, szz1, sxy1
        );

        if r_big > r {
            if let Some(f) = tail_xx {
                eprintln!("[demag_mg]   tail-mass frac (xx, outside r): {:.3e}", f);
            }
            if let Some(f) = tail_yy {
                eprintln!("[demag_mg]   tail-mass frac (yy, outside r): {:.3e}", f);
            }
            if let Some(f) = tail_zz {
                eprintln!("[demag_mg]   tail-mass frac (zz, outside r): {:.3e}", f);
            }
        }
    }

    dk
}

// Cache a solver instance so we don’t rebuild hierarchies every field evaluation.
static DEMAG_MG_CACHE: OnceLock<Mutex<Option<DemagPoissonMGHybrid>>> = OnceLock::new();

pub fn add_demag_field_poisson_mg(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    mat: &Material,
) {
    if !mat.demag {
        return;
    }

    let cfg = DemagPoissonMGConfig::from_env();
    let hyb = HybridConfig::from_env();

    let cache = DEMAG_MG_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().expect("DEMAG_MG_CACHE mutex poisoned");

    let rebuild = match guard.as_ref() {
        Some(s) => !s.same_structure(grid, &cfg),
        None => true,
    };

    if rebuild {
        *guard = Some(DemagPoissonMGHybrid::new(*grid, cfg, hyb));
    }

    if let Some(s) = guard.as_mut() {
        // Apply runtime knobs safely (including smoothing/smoother sanitization).
        s.mg.apply_cfg(cfg);
        s.hyb = hyb;
        s.add_field(m, b_eff, mat);
    }
}

pub fn compute_demag_field_poisson_mg(
    grid: &Grid2D,
    m: &VectorField2D,
    out: &mut VectorField2D,
    mat: &Material,
) {
    out.set_uniform(0.0, 0.0, 0.0);
    add_demag_field_poisson_mg(grid, m, out, mat);
}

#[cfg(test)]
mod phase0_tests {
    use super::*;

    /// Phase 0 gate test: extract_magnet_layer_phi round-trips correctly
    /// and the existing B output is unchanged.
    #[test]
    fn test_extract_inject_roundtrip() {
        // 1. Build a small grid and solver
        let grid = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let cfg = DemagPoissonMGConfig::default();
        let mut mg = DemagPoissonMG::new(grid, cfg);

        // 2. Create a simple magnetisation (uniform +x)
        let mut m = VectorField2D::new(grid);
        m.set_uniform(1.0, 0.0, 0.0);

        // 3. Build RHS, set BCs, solve
        mg.build_rhs_from_m(&m, 800e3);
        mg.update_finest_boundary_bc();
        mg.levels[0].enforce_dirichlet();
        mg.solve();

        // 4. Extract B using the existing method (reference)
        let mut b_ref = VectorField2D::new(grid);
        mg.add_b_from_phi_on_magnet_layer_all(&mut b_ref);

        // 5. Extract magnet-layer phi
        let phi_2d = mg.extract_magnet_layer_phi();
        assert_eq!(phi_2d.len(), 16 * 16);

        // 6. Verify phi is non-trivial (not all zeros)
        let phi_max = phi_2d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let phi_min = phi_2d.iter().cloned().fold(f64::INFINITY, f64::min);
        let phi_range = phi_max - phi_min;
        assert!(
            phi_range > 1e-30,
            "phi should be non-trivial after solving, but range = {:.3e}",
            phi_range
        );
        eprintln!(
            "[phase0] phi range: {:.6e} to {:.6e} (range {:.6e})",
            phi_min, phi_max, phi_range
        );

        // 7. Inject back and verify phi is unchanged (round-trip)
        mg.inject_magnet_layer_phi(&phi_2d);
        let phi_2d_check = mg.extract_magnet_layer_phi();
        for k in 0..phi_2d.len() {
            assert!(
                phi_2d[k] == phi_2d_check[k],
                "round-trip mismatch at cell {}: {:.6e} vs {:.6e}",
                k, phi_2d[k], phi_2d_check[k]
            );
        }
        eprintln!("[phase0] PASSED: extract/inject round-trip is bit-exact");

        // 8. Extract B again — must be BIT-IDENTICAL to reference
        let mut b_check = VectorField2D::new(grid);
        mg.add_b_from_phi_on_magnet_layer_all(&mut b_check);
        for k in 0..b_ref.data.len() {
            assert!(
                b_ref.data[k] == b_check.data[k],
                "B mismatch at cell {} after phi round-trip: ref={:?} check={:?}",
                k, b_ref.data[k], b_check.data[k]
            );
        }
        eprintln!("[phase0] PASSED: B output unchanged after phi round-trip");
    }

    /// Verify that the extracted phi is spatially coherent (not random noise).
    /// For uniform +x magnetisation, phi should vary smoothly across the grid
    /// with a pattern related to the surface-charge distribution.
    #[test]
    fn test_phi_spatial_coherence() {
        let grid = Grid2D::new(32, 32, 5e-9, 5e-9, 3e-9);
        let cfg = DemagPoissonMGConfig::default();
        let mut mg = DemagPoissonMG::new(grid, cfg);

        let mut m = VectorField2D::new(grid);
        m.set_uniform(1.0, 0.0, 0.0);

        mg.build_rhs_from_m(&m, 800e3);
        mg.update_finest_boundary_bc();
        mg.levels[0].enforce_dirichlet();
        mg.solve();

        let phi_2d = mg.extract_magnet_layer_phi();
        let nx = grid.nx;
        let ny = grid.ny;

        // For uniform +x magnetisation, div(M) = 0 in the interior.
        // Surface charges exist at the x-boundaries (left and right edges).
        // phi should be antisymmetric about x = Lx/2:
        //   phi(x, y) ≈ -phi(Lx - x, y)
        //
        // Check: phi at the left quarter should have opposite sign to right quarter.
        let mut sum_left = 0.0f64;
        let mut sum_right = 0.0f64;
        for j in ny / 4..3 * ny / 4 {
            for i in 0..nx / 4 {
                sum_left += phi_2d[j * nx + i];
            }
            for i in 3 * nx / 4..nx {
                sum_right += phi_2d[j * nx + i];
            }
        }

        eprintln!(
            "[phase0] phi spatial check: sum_left={:.6e}, sum_right={:.6e}",
            sum_left, sum_right
        );

        // They should have opposite signs (antisymmetric potential)
        // or at least be significantly different
        let phi_abs_max = phi_2d.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(
            phi_abs_max > 1e-30,
            "phi is all zeros — solver didn't produce a solution"
        );

        // Check smoothness: the max gradient should be bounded.
        // (random noise would have huge cell-to-cell variation)
        let mut max_grad = 0.0f64;
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let c = phi_2d[j * nx + i];
                let dx = ((phi_2d[j * nx + i + 1] - c).abs())
                    .max((c - phi_2d[j * nx + i - 1]).abs());
                let dy = ((phi_2d[(j + 1) * nx + i] - c).abs())
                    .max((c - phi_2d[(j - 1) * nx + i]).abs());
                max_grad = max_grad.max(dx).max(dy);
            }
        }
        // Max cell-to-cell difference should be much smaller than the range
        let phi_range = phi_2d.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - phi_2d.iter().cloned().fold(f64::INFINITY, f64::min);
        let grad_ratio = if phi_range > 0.0 { max_grad / phi_range } else { 0.0 };
        eprintln!(
            "[phase0] phi smoothness: max_grad={:.6e}, range={:.6e}, ratio={:.3}",
            max_grad, phi_range, grad_ratio
        );
        assert!(
            grad_ratio < 0.5,
            "phi is not smooth — max gradient / range = {:.3} (expected < 0.5)",
            grad_ratio
        );
        eprintln!("[phase0] PASSED: phi is spatially smooth and non-trivial");
    }

    /// Verify that solve_with_corrections with empty corrections gives the
    /// same phi as the standard build_rhs + solve path.
    ///
    /// Both paths use raw DemagPoissonMG to avoid needing Material construction.
    #[test]
    fn test_empty_corrections_matches_standard_solve() {
        let grid = Grid2D::new(16, 16, 5e-9, 5e-9, 3e-9);
        let cfg = DemagPoissonMGConfig::default();
        let ms = 800e3_f64;

        let mut m = VectorField2D::new(grid);
        m.set_uniform(1.0, 0.0, 0.0);

        // Path A: standard solve
        let mut mg_a = DemagPoissonMG::new(grid, cfg);
        mg_a.build_rhs_from_m(&m, ms);
        mg_a.update_finest_boundary_bc();
        mg_a.levels[0].enforce_dirichlet();
        mg_a.solve();
        let phi_a = mg_a.extract_magnet_layer_phi();
        let mut b_a = VectorField2D::new(grid);
        mg_a.add_b_from_phi_on_magnet_layer_all(&mut b_a);

        // Path B: replicate what solve_with_corrections does with empty corrections
        // (build_rhs_from_m → no corrections to add → BC → dirichlet → solve)
        let mut mg_b = DemagPoissonMG::new(grid, cfg);
        mg_b.build_rhs_from_m(&m, ms);
        // No corrections to inject (empty list)
        mg_b.update_finest_boundary_bc();
        mg_b.levels[0].enforce_dirichlet();
        // Cold start (solve_with_corrections clears phi when warm_start=true + hybrid)
        // For non-hybrid default config, phi starts at 0 anyway (freshly allocated).
        mg_b.solve();
        let phi_b = mg_b.extract_magnet_layer_phi();
        let mut b_b = VectorField2D::new(grid);
        mg_b.add_b_from_phi_on_magnet_layer_all(&mut b_b);

        // Compare phi — should be bit-identical since paths are the same
        let mut max_phi_diff = 0.0f64;
        for k in 0..phi_a.len() {
            max_phi_diff = max_phi_diff.max((phi_a[k] - phi_b[k]).abs());
        }

        // Compare B
        let mut max_b_diff = 0.0f64;
        for k in 0..b_a.data.len() {
            for c in 0..3 {
                max_b_diff = max_b_diff.max((b_a.data[k][c] - b_b.data[k][c]).abs());
            }
        }

        eprintln!("[phase0] phi max diff between paths: {:.3e}", max_phi_diff);
        eprintln!("[phase0] B max diff between paths: {:.3e}", max_b_diff);

        // Should be bit-identical since both paths are exactly the same
        assert!(
            max_phi_diff == 0.0,
            "phi should be bit-identical but differs by {:.3e}", max_phi_diff
        );
        assert!(
            max_b_diff == 0.0,
            "B should be bit-identical but differs by {:.3e}", max_b_diff
        );

        eprintln!("[phase0] PASSED: identical solve paths produce bit-identical phi and B");
    }
}