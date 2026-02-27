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

use rayon::prelude::*;

use std::collections::HashMap;
use std::f64::consts::PI;
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
struct HybridConfig {
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
    fn from_env() -> Self {
        fn get_usize(name: &str) -> Option<usize> {
            std::env::var(name)
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
        }

        #[inline]
        fn get_bool(name: &str) -> bool {
            std::env::var(name)
                .ok()
                .map(|s| matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
                .unwrap_or(false)
        }

        fn get_f64(name: &str) -> Option<f64> {
            std::env::var(name).ok().and_then(|s| s.trim().parse::<f64>().ok())
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

#[inline]
fn idx3(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    (k * ny + j) * nx + i
}

#[inline]
fn stamp_dirichlet_bc(arr: &mut [f64], bc_phi: &[f64], nx: usize, ny: usize, nz: usize) {
    for k in 0..nz {
        for j in 0..ny {
            let id0 = idx3(0, j, k, nx, ny);
            let id1 = idx3(nx - 1, j, k, nx, ny);
            arr[id0] = bc_phi[id0];
            arr[id1] = bc_phi[id1];
        }
        for i in 0..nx {
            let id0 = idx3(i, 0, k, nx, ny);
            let id1 = idx3(i, ny - 1, k, nx, ny);
            arr[id0] = bc_phi[id0];
            arr[id1] = bc_phi[id1];
        }
    }

    for j in 0..ny {
        for i in 0..nx {
            let id0 = idx3(i, j, 0, nx, ny);
            let id1 = idx3(i, j, nz - 1, nx, ny);
            arr[id0] = bc_phi[id0];
            arr[id1] = bc_phi[id1];
        }
    }
}

// ---------------------------
// Local correction kernel ΔK(r) = K_fft(r) - K_mg_long(screened)(r)
// ---------------------------

#[derive(Debug, Clone)]
struct DeltaKernel2D {
    radius: usize,
    stride: usize,
    dkxx: Vec<f64>,
    dkxy: Vec<f64>,
    dkyy: Vec<f64>,
    dkzz: Vec<f64>,
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

// ---------------------------
// Barnes–Hut treecode for open-boundary Dirichlet values
// ---------------------------

#[derive(Clone, Copy)]
struct Charge {
    pos: [f64; 3],
    q: f64,
}

#[derive(Clone)]
struct BhNode {
    center: [f64; 3],
    half: f64,
    q: f64,
    p: [f64; 3],
    children: [Option<usize>; 8],
    indices: Vec<usize>,
}

impl BhNode {
    fn new(center: [f64; 3], half: f64) -> Self {
        Self {
            center,
            half,
            q: 0.0,
            p: [0.0; 3],
            children: [None; 8],
            indices: Vec::new(),
        }
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        self.children.iter().all(|c| c.is_none())
    }
}

struct BarnesHutTree {
    charges: Vec<Charge>,
    nodes: Vec<BhNode>,
    theta: f64,
    leaf_size: usize,
    max_depth: usize,
}

impl BarnesHutTree {
    fn build(
        charges: Vec<Charge>,
        root_center: [f64; 3],
        root_half: f64,
        leaf_size: usize,
        theta: f64,
        max_depth: usize,
    ) -> Self {
        let mut tree = Self {
            charges,
            nodes: Vec::new(),
            theta,
            leaf_size,
            max_depth,
        };
        tree.nodes.push(BhNode::new(root_center, root_half));

        let n = tree.charges.len();
        for idx in 0..n {
            tree.insert(0, idx, 0);
        }

        tree.compute_moments_rec(0);
        tree
    }

    #[inline]
    fn octant(center: [f64; 3], pos: [f64; 3]) -> usize {
        let mut o = 0usize;
        if pos[0] >= center[0] {
            o |= 1;
        }
        if pos[1] >= center[1] {
            o |= 2;
        }
        if pos[2] >= center[2] {
            o |= 4;
        }
        o
    }

    fn ensure_child(&mut self, node_idx: usize, oct: usize) -> usize {
        if let Some(ci) = self.nodes[node_idx].children[oct] {
            return ci;
        }

        let parent_center = self.nodes[node_idx].center;
        let parent_half = self.nodes[node_idx].half;

        let child_half = parent_half * 0.5;
        let cx = parent_center[0]
            + if (oct & 1) != 0 {
                child_half
            } else {
                -child_half
            };
        let cy = parent_center[1]
            + if (oct & 2) != 0 {
                child_half
            } else {
                -child_half
            };
        let cz = parent_center[2]
            + if (oct & 4) != 0 {
                child_half
            } else {
                -child_half
            };

        let child_idx = self.nodes.len();
        self.nodes.push(BhNode::new([cx, cy, cz], child_half));
        self.nodes[node_idx].children[oct] = Some(child_idx);
        child_idx
    }

    fn insert(&mut self, node_idx: usize, charge_idx: usize, depth: usize) {
        if depth >= self.max_depth {
            self.nodes[node_idx].indices.push(charge_idx);
            return;
        }

        let is_leaf = self.nodes[node_idx].is_leaf();
        if is_leaf {
            self.nodes[node_idx].indices.push(charge_idx);

            if self.nodes[node_idx].indices.len() > self.leaf_size {
                let indices = std::mem::take(&mut self.nodes[node_idx].indices);
                for ci in indices {
                    let pos = self.charges[ci].pos;
                    let oct = Self::octant(self.nodes[node_idx].center, pos);
                    let child = self.ensure_child(node_idx, oct);
                    self.insert(child, ci, depth + 1);
                }
            }
        } else {
            let pos = self.charges[charge_idx].pos;
            let oct = Self::octant(self.nodes[node_idx].center, pos);
            let child = self.ensure_child(node_idx, oct);
            self.insert(child, charge_idx, depth + 1);
        }
    }

    fn compute_moments_rec(&mut self, node_idx: usize) -> (f64, [f64; 3]) {
        let center = self.nodes[node_idx].center;
        if self.nodes[node_idx].is_leaf() {
            let indices = self.nodes[node_idx].indices.clone();
            let mut q = 0.0f64;
            let mut p = [0.0f64; 3];
            for ci in indices {
                let c = self.charges[ci];
                q += c.q;
                p[0] += c.q * (c.pos[0] - center[0]);
                p[1] += c.q * (c.pos[1] - center[1]);
                p[2] += c.q * (c.pos[2] - center[2]);
            }
            self.nodes[node_idx].q = q;
            self.nodes[node_idx].p = p;
            (q, p)
        } else {
            let children = self.nodes[node_idx].children;
            let mut q = 0.0f64;
            let mut p = [0.0f64; 3];

            for &child_opt in &children {
                if let Some(ci) = child_opt {
                    let (cq, cp) = self.compute_moments_rec(ci);
                    let child_center = self.nodes[ci].center;
                    q += cq;
                    p[0] += cp[0] + cq * (child_center[0] - center[0]);
                    p[1] += cp[1] + cq * (child_center[1] - center[1]);
                    p[2] += cp[2] + cq * (child_center[2] - center[2]);
                }
            }

            self.nodes[node_idx].q = q;
            self.nodes[node_idx].p = p;
            (q, p)
        }
    }

    fn eval_phi(&self, target: [f64; 3]) -> f64 {
        let sum = self.eval_node(0, target);
        -sum / (4.0 * PI)
    }

    fn eval_node(&self, node_idx: usize, target: [f64; 3]) -> f64 {
        let node = &self.nodes[node_idx];
        if node.q == 0.0 && node.indices.is_empty() && node.is_leaf() {
            return 0.0;
        }

        let rx = target[0] - node.center[0];
        let ry = target[1] - node.center[1];
        let rz = target[2] - node.center[2];

        let ax = rx.abs();
        let ay = ry.abs();
        let az = rz.abs();
        let dx = (ax - node.half).max(0.0);
        let dy = (ay - node.half).max(0.0);
        let dz = (az - node.half).max(0.0);
        let d2 = dx * dx + dy * dy + dz * dz;
        let d = d2.sqrt();

        let r2 = rx * rx + ry * ry + rz * rz;
        let r = r2.sqrt();
        let size = node.half * 2.0;

        let accept = if node.is_leaf() {
            true
        } else if d > 0.0 {
            size / d < self.theta
        } else {
            false
        };

        if accept {
            if node.is_leaf() {
                let mut sum = 0.0f64;
                for &ci in &node.indices {
                    let c = self.charges[ci];
                    let dx = target[0] - c.pos[0];
                    let dy = target[1] - c.pos[1];
                    let dz = target[2] - c.pos[2];
                    let rr2 = dx * dx + dy * dy + dz * dz;
                    if rr2 > 0.0 {
                        sum += c.q / rr2.sqrt();
                    }
                }
                sum
            } else {
                let inv_r = 1.0 / r;
                let inv_r3 = inv_r * inv_r * inv_r;
                let pr = node.p[0] * rx + node.p[1] * ry + node.p[2] * rz;
                node.q * inv_r + pr * inv_r3
            }
        } else {
            let mut sum = 0.0f64;
            for &child_opt in &node.children {
                if let Some(ci) = child_opt {
                    sum += self.eval_node(ci, target);
                }
            }
            sum
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

    fn from_kind(kind: LaplacianStencilKind, dx: f64, dy: f64, dz: f64, iso27_alpha_bits: u64) -> Self {
        match kind {
            LaplacianStencilKind::SevenPoint => Self::seven_point(dx, dy, dz),
            LaplacianStencilKind::Iso9PlusZ => Self::iso9_plus_z(dx, dy, dz),
            LaplacianStencilKind::Iso27 => Self::iso27(dx, dy, dz, iso27_alpha_bits),
        }
    }

    #[inline]
    fn clamp_idx(v: isize, lo: isize, hi: isize) -> isize {
        if v < lo {
            lo
        } else if v > hi {
            hi
        } else {
            v
        }
    }

    #[inline]
    fn idx(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> usize {
        (k * ny + j) * nx + i
    }

    fn apply_at(&self, phi: &[f64], nx: usize, ny: usize, nz: usize, i: usize, j: usize, k: usize) -> f64 {
        let idc = Self::idx(nx, ny, i, j, k);
        let mut sum = self.center * phi[idc];

        let i0 = i as isize;
        let j0 = j as isize;
        let k0 = k as isize;

        let nxm = nx as isize - 1;
        let nym = ny as isize - 1;
        let nzm = nz as isize - 1;

        for (off, &c) in self.offs.iter().zip(self.coeffs.iter()) {
            let ii = Self::clamp_idx(i0 + off[0], 0, nxm) as usize;
            let jj = Self::clamp_idx(j0 + off[1], 0, nym) as usize;
            let kk = Self::clamp_idx(k0 + off[2], 0, nzm) as usize;
            sum += c * phi[Self::idx(nx, ny, ii, jj, kk)];
        }
        sum
    }

    fn offdiag_sum_at(&self, phi: &[f64], nx: usize, ny: usize, nz: usize, i: usize, j: usize, k: usize) -> f64 {
        let i0 = i as isize;
        let j0 = j as isize;
        let k0 = k as isize;

        let nxm = nx as isize - 1;
        let nym = ny as isize - 1;
        let nzm = nz as isize - 1;

        let mut sum = 0.0;
        for (off, &c) in self.offs.iter().zip(self.coeffs.iter()) {
            let ii = Self::clamp_idx(i0 + off[0], 0, nxm) as usize;
            let jj = Self::clamp_idx(j0 + off[1], 0, nym) as usize;
            let kk = Self::clamp_idx(k0 + off[2], 0, nzm) as usize;
            sum += c * phi[Self::idx(nx, ny, ii, jj, kk)];
        }
        sum
    }

    fn galerkin_coarsen(fine: &Stencil3D, rx: usize, ry: usize, rz: usize, prolong: ProlongationKind) -> Self {
        let ncx: usize = 9;
        let ncy: usize = 9;
        let ncz: usize = 9;

        let nfx = ncx * rx;
        let nfy = ncy * ry;
        let nfz = ncz * rz;

        let c0 = (ncx / 2, ncy / 2, ncz / 2);
        let id_c0 = Self::idx(ncx, ncy, c0.0, c0.1, c0.2);

        let mut phi_c = vec![0.0f64; ncx * ncy * ncz];
        let mut phi_f = vec![0.0f64; nfx * nfy * nfz];
        let mut y_f = vec![0.0f64; nfx * nfy * nfz];
        let mut y_c = vec![0.0f64; ncx * ncy * ncz];

        let mut map: HashMap<(isize, isize, isize), f64> = HashMap::new();

        for kz in 0..ncz {
            for jy in 0..ncy {
                for ix in 0..ncx {
                    phi_c.fill(0.0);
                    phi_c[Self::idx(ncx, ncy, ix, jy, kz)] = 1.0;

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
        Self {
            center,
            diag,
            offs,
            coeffs,
        }
    }
}

fn apply_stencil_to_field(st: &Stencil3D, phi: &[f64], out: &mut [f64], nx: usize, ny: usize, nz: usize) {
    debug_assert_eq!(phi.len(), nx * ny * nz);
    debug_assert_eq!(out.len(), nx * ny * nz);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                out[Stencil3D::idx(nx, ny, i, j, k)] = st.apply_at(phi, nx, ny, nz, i, j, k);
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
                            sum += fine[Stencil3D::idx(nfx, nfy, i, j, k)];
                        }
                    }
                }
                coarse[Stencil3D::idx(ncx, ncy, ix, jy, kz)] = norm * sum;
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
                        let v = coarse[Stencil3D::idx(ncx, ncy, ix, jy, kz)];
                        for fk in 0..rz {
                            for fj in 0..ry {
                                for fi in 0..rx {
                                    let i = ix * rx + fi;
                                    let j = jy * ry + fj;
                                    let k = kz * rz + fk;
                                    fine[Stencil3D::idx(nfx, nfy, i, j, k)] = v;
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

                        v += wi0 * wj0 * wk0 * coarse[Stencil3D::idx(ncx, ncy, i0, j0, k0)];
                        v += wi1 * wj0 * wk0 * coarse[Stencil3D::idx(ncx, ncy, i1, j0, k0)];
                        v += wi0 * wj1 * wk0 * coarse[Stencil3D::idx(ncx, ncy, i0, j1, k0)];
                        v += wi1 * wj1 * wk0 * coarse[Stencil3D::idx(ncx, ncy, i1, j1, k0)];
                        v += wi0 * wj0 * wk1 * coarse[Stencil3D::idx(ncx, ncy, i0, j0, k1)];
                        v += wi1 * wj0 * wk1 * coarse[Stencil3D::idx(ncx, ncy, i1, j0, k1)];
                        v += wi0 * wj1 * wk1 * coarse[Stencil3D::idx(ncx, ncy, i0, j1, k1)];
                        v += wi1 * wj1 * wk1 * coarse[Stencil3D::idx(ncx, ncy, i1, j1, k1)];

                        fine[Stencil3D::idx(nfx, nfy, i, j, k)] = v;
                    }
                }
            }
        }
    }
}

fn interp_1d_cell_centered(i_coarse: usize, r_i: usize, n_coarse: usize, r: usize) -> (usize, usize, f64, f64) {
    if r == 1 {
        let i0 = i_coarse.min(n_coarse - 1);
        return (i0, i0, 1.0, 0.0);
    }
    debug_assert!(r == 2);

    let i0 = i_coarse.min(n_coarse - 1);
    if r_i == 0 {
        let i1 = if i0 > 0 { i0 - 1 } else { 0 };
        if i1 == i0 {
            (i0, i0, 1.0, 0.0)
        } else {
            (i0, i1, 0.75, 0.25)
        }
    } else {
        let i1 = (i0 + 1).min(n_coarse - 1);
        if i1 == i0 {
            (i0, i0, 1.0, 0.0)
        } else {
            (i0, i1, 0.75, 0.25)
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
        stamp_dirichlet_bc(&mut self.phi, &self.bc_phi, self.nx, self.ny, self.nz);
    }

    #[inline]
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

    fn new_with_operator(grid: Grid2D, mut cfg: DemagPoissonMGConfig, op: MGOperatorSettings) -> Self {
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

        // Assign stencils per level.
        if !levels.is_empty() {
            levels[0].stencil = Stencil3D::from_kind(op.stencil, levels[0].dx, levels[0].dy, levels[0].dz, op.iso27_alpha_bits);
            for l in 1..levels.len() {
                let rx = levels[l - 1].nx / levels[l].nx;
                let ry = levels[l - 1].ny / levels[l].ny;
                let rz = levels[l - 1].nz / levels[l].nz;

                levels[l].stencil = match op.coarse_op {
                    CoarseOpKind::Rediscretize => {
                        Stencil3D::from_kind(op.stencil, levels[l].dx, levels[l].dy, levels[l].dz, op.iso27_alpha_bits)
                    }
                    CoarseOpKind::Galerkin => {
                        Stencil3D::galerkin_coarsen(&levels[l - 1].stencil, rx, ry, rz, op.prolong)
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

    fn build_rhs_from_m(&mut self, m: &VectorField2D, ms: f64) {
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

                    let (c_in, m_c) =
                        m_at(pi, pj, pk, px, py, pz, offx, offy, offz, nx_m, ny_m, mdata, ms);

                    let (xp_in, m_xp) =
                        m_at(pi + 1, pj, pk, px, py, pz, offx, offy, offz, nx_m, ny_m, mdata, ms);
                    let (xm_in, m_xm) =
                        m_at(pi - 1, pj, pk, px, py, pz, offx, offy, offz, nx_m, ny_m, mdata, ms);

                    let (yp_in, m_yp) =
                        m_at(pi, pj + 1, pk, px, py, pz, offx, offy, offz, nx_m, ny_m, mdata, ms);
                    let (ym_in, m_ym) =
                        m_at(pi, pj - 1, pk, px, py, pz, offx, offy, offz, nx_m, ny_m, mdata, ms);

                    let (zp_in, m_zp) =
                        m_at(pi, pj, pk + 1, px, py, pz, offx, offy, offz, nx_m, ny_m, mdata, ms);
                    let (zm_in, m_zm) =
                        m_at(pi, pj, pk - 1, px, py, pz, offx, offy, offz, nx_m, ny_m, mdata, ms);

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
                let dx = finest.dx;
                let dy = finest.dy;
                let dz = finest.dz;
                let dvol = dx * dy * dz;

                let cx = (finest.nx as f64) * 0.5;
                let cy = (finest.ny as f64) * 0.5;
                let cz = (finest.nz as f64) * 0.5;

                let mut q = 0.0f64;
                let mut pxm = 0.0f64;
                let mut pym = 0.0f64;
                let mut pzm = 0.0f64;

                for k in 0..finest.nz {
                    let z = (k as f64 + 0.5 - cz) * dz;
                    for j in 0..finest.ny {
                        let y = (j as f64 + 0.5 - cy) * dy;
                        let row = (k * finest.ny + j) * finest.nx;
                        for i in 0..finest.nx {
                            let x = (i as f64 + 0.5 - cx) * dx;
                            let rho = finest.rhs[row + i];
                            let w = rho * dvol;
                            q += w;
                            pxm += w * x;
                            pym += w * y;
                            pzm += w * z;
                        }
                    }
                }

                let inv4pi = 1.0 / (4.0 * PI);

                let nx = finest.nx;
                let ny = finest.ny;
                let nz = finest.nz;

                for k in 0..nz {
                    let z = (k as f64 + 0.5 - cz) * dz;
                    for j in 0..ny {
                        let y = (j as f64 + 0.5 - cy) * dy;

                        {
                            let i = 0usize;
                            let x = (i as f64 + 0.5 - cx) * dx;
                            let r2 = x * x + y * y + z * z;
                            if r2 > 0.0 {
                                let r = r2.sqrt();
                                let pr = pxm * x + pym * y + pzm * z;
                                finest.bc_phi[idx3(i, j, k, nx, ny)] =
                                    -inv4pi * (q / r + pr / (r * r * r));
                            }
                        }
                        {
                            let i = nx - 1;
                            let x = (i as f64 + 0.5 - cx) * dx;
                            let r2 = x * x + y * y + z * z;
                            if r2 > 0.0 {
                                let r = r2.sqrt();
                                let pr = pxm * x + pym * y + pzm * z;
                                finest.bc_phi[idx3(i, j, k, nx, ny)] =
                                    -inv4pi * (q / r + pr / (r * r * r));
                            }
                        }
                    }

                    for i in 0..nx {
                        let x = (i as f64 + 0.5 - cx) * dx;

                        {
                            let j = 0usize;
                            let y = (j as f64 + 0.5 - cy) * dy;
                            let r2 = x * x + y * y + z * z;
                            if r2 > 0.0 {
                                let r = r2.sqrt();
                                let pr = pxm * x + pym * y + pzm * z;
                                finest.bc_phi[idx3(i, j, k, nx, ny)] =
                                    -inv4pi * (q / r + pr / (r * r * r));
                            }
                        }
                        {
                            let j = ny - 1;
                            let y = (j as f64 + 0.5 - cy) * dy;
                            let r2 = x * x + y * y + z * z;
                            if r2 > 0.0 {
                                let r = r2.sqrt();
                                let pr = pxm * x + pym * y + pzm * z;
                                finest.bc_phi[idx3(i, j, k, nx, ny)] =
                                    -inv4pi * (q / r + pr / (r * r * r));
                            }
                        }
                    }
                }

                for j in 0..ny {
                    let y = (j as f64 + 0.5 - cy) * dy;
                    for i in 0..nx {
                        let x = (i as f64 + 0.5 - cx) * dx;

                        {
                            let k = 0usize;
                            let z = (k as f64 + 0.5 - cz) * dz;
                            let r2 = x * x + y * y + z * z;
                            if r2 > 0.0 {
                                let r = r2.sqrt();
                                let pr = pxm * x + pym * y + pzm * z;
                                finest.bc_phi[idx3(i, j, k, nx, ny)] =
                                    -inv4pi * (q / r + pr / (r * r * r));
                            }
                        }
                        {
                            let k = nz - 1;
                            let z = (k as f64 + 0.5 - cz) * dz;
                            let r2 = x * x + y * y + z * z;
                            if r2 > 0.0 {
                                let r = r2.sqrt();
                                let pr = pxm * x + pym * y + pzm * z;
                                finest.bc_phi[idx3(i, j, k, nx, ny)] =
                                    -inv4pi * (q / r + pr / (r * r * r));
                            }
                        }
                    }
                }
            }

            BoundaryCondition::DirichletTreecode => {
                let dx = finest.dx;
                let dy = finest.dy;
                let dz = finest.dz;
                let dvol = dx * dy * dz;

                let cx = (finest.nx as f64) * 0.5;
                let cy = (finest.ny as f64) * 0.5;
                let cz = (finest.nz as f64) * 0.5;

                let mut charges: Vec<Charge> = Vec::new();
                charges.reserve(finest.rhs.len() / 8);

                for k in 0..finest.nz {
                    let z = (k as f64 + 0.5 - cz) * dz;
                    for j in 0..finest.ny {
                        let y = (j as f64 + 0.5 - cy) * dy;
                        let row = (k * finest.ny + j) * finest.nx;
                        for i in 0..finest.nx {
                            let rho = finest.rhs[row + i];
                            if rho.abs() < 1e-40 {
                                continue;
                            }
                            let x = (i as f64 + 0.5 - cx) * dx;
                            charges.push(Charge {
                                pos: [x, y, z],
                                q: rho * dvol,
                            });
                        }
                    }
                }

                if !charges.is_empty() {
                    let lx = (finest.nx as f64) * dx;
                    let ly = (finest.ny as f64) * dy;
                    let lz = (finest.nz as f64) * dz;
                    let root_half = 0.5 * lx.max(ly).max(lz) + 1e-12;

                    let tree = BarnesHutTree::build(
                        charges,
                        [0.0, 0.0, 0.0],
                        root_half,
                        self.cfg.tree_leaf,
                        self.cfg.tree_theta,
                        self.cfg.tree_max_depth,
                    );

                    let nx = finest.nx;
                    let ny = finest.ny;
                    let nz = finest.nz;

                    finest
                        .bc_phi
                        .par_chunks_mut(nx)
                        .enumerate()
                        .for_each(|(row_idx, bc_row)| {
                            let k = row_idx / ny;
                            let j = row_idx % ny;

                            let z = (k as f64 + 0.5 - cz) * dz;
                            let y = (j as f64 + 0.5 - cy) * dy;

                            if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                                for i in 0..nx {
                                    let x = (i as f64 + 0.5 - cx) * dx;
                                    bc_row[i] = tree.eval_phi([x, y, z]);
                                }
                            } else {
                                let i0 = 0usize;
                                let x0 = (i0 as f64 + 0.5 - cx) * dx;
                                bc_row[i0] = tree.eval_phi([x0, y, z]);

                                let i1 = nx - 1;
                                let x1 = (i1 as f64 + 0.5 - cx) * dx;
                                bc_row[i1] = tree.eval_phi([x1, y, z]);
                            }
                        });
                }
            }
        }

        finest.enforce_dirichlet();
    }

    fn smooth_weighted_jacobi(level: &mut MGLevel, iters: usize, omega: f64) {
        let nx = level.nx;
        let ny = level.ny;
        let nz = level.nz;

        let st = level.stencil.clone();

        for _ in 0..iters {
            level.tmp.copy_from_slice(&level.phi);

            {
                let tmp: &[f64] = &level.tmp;
                let rhs: &[f64] = &level.rhs;
                let st_ref = &st;

                level
                    .phi
                    .par_chunks_mut(nx)
                    .enumerate()
                    .for_each(|(row_idx, phi_row)| {
                        let k = row_idx / ny;
                        let j = row_idx % ny;

                        if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                            return;
                        }

                        let base = row_idx * nx;
                        for i in 1..(nx - 1) {
                            let id = base + i;
                            let off = st_ref.offdiag_sum_at(tmp, nx, ny, nz, i, j, k);
                            let phi_gs = (off - rhs[id]) / st_ref.diag;
                            phi_row[i] = (1.0 - omega) * tmp[id] + omega * phi_gs;
                        }
                    });
            }

            level.enforce_dirichlet();
        }
    }

    fn smooth_rb_sor(level: &mut MGLevel, iters: usize, omega: f64) {
        let nx = level.nx;
        let ny = level.ny;
        let nz = level.nz;

        let sx = level.inv_dx2;
        let sy = level.inv_dy2;
        let sz = level.inv_dz2;

        let denom = 2.0 * (sx + sy + sz);
        let plane = nx * ny;

        for _ in 0..iters {
            for color in 0..2usize {
                let phi_ro: &[f64] = &level.phi;
                let rhs_ro: &[f64] = &level.rhs;

                level
                    .tmp
                    .par_chunks_mut(nx)
                    .enumerate()
                    .for_each(|(row_idx, tmp_row)| {
                        let k = row_idx / ny;
                        let j = row_idx % ny;

                        if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                            return;
                        }

                        let base = row_idx * nx;

                        for i in 1..(nx - 1) {
                            if ((i + j + k) & 1) != color {
                                continue;
                            }
                            let id = base + i;

                            let xm = phi_ro[id - 1];
                            let xp = phi_ro[id + 1];
                            let ym = phi_ro[id - nx];
                            let yp = phi_ro[id + nx];
                            let zm = phi_ro[id - plane];
                            let zp = phi_ro[id + plane];

                            let off = sx * (xm + xp) + sy * (ym + yp) + sz * (zm + zp);
                            let rhs = rhs_ro[id];
                            let phi_new = (off - rhs) / denom;

                            let phi_old = phi_ro[id];
                            tmp_row[i] = phi_old + omega * (phi_new - phi_old);
                        }
                    });

                let tmp_ro: &[f64] = &level.tmp;
                level
                    .phi
                    .par_chunks_mut(nx)
                    .enumerate()
                    .for_each(|(row_idx, phi_row)| {
                        let k = row_idx / ny;
                        let j = row_idx % ny;

                        if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                            return;
                        }

                        let base = row_idx * nx;

                        for i in 1..(nx - 1) {
                            if ((i + j + k) & 1) != color {
                                continue;
                            }
                            let id = base + i;
                            phi_row[i] = tmp_ro[id];
                        }
                    });
            }

            level.enforce_dirichlet();
        }
    }

    fn compute_residual(level: &mut MGLevel) -> f64 {
        let nx = level.nx;
        let ny = level.ny;
        let nz = level.nz;

        let phi: &[f64] = &level.phi;
        let rhs: &[f64] = &level.rhs;
        let st = &level.stencil;

        level
            .res
            .par_chunks_mut(nx)
            .enumerate()
            .map(|(row_idx, res_row)| {
                let k = row_idx / ny;
                let j = row_idx % ny;

                if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                    res_row.fill(0.0);
                    return 0.0f64;
                }

                let base = row_idx * nx;
                let mut max_abs: f64 = 0.0;

                res_row[0] = 0.0;
                res_row[nx - 1] = 0.0;

                for i in 1..(nx - 1) {
                    let id = base + i;
                    let aphi = st.apply_at(phi, nx, ny, nz, i, j, k);
                    let r = rhs[id] - aphi;

                    res_row[i] = r;
                    max_abs = max_abs.max(r.abs());
                }

                max_abs
            })
            .reduce(|| 0.0, |a, b| a.max(b))
    }

    fn restrict_residual(fine: &MGLevel, coarse: &mut MGLevel) {
        coarse.rhs.fill(0.0);
        coarse.phi.fill(0.0);

        let nxc = coarse.nx;
        let nyc = coarse.ny;
        let nzc = coarse.nz;

        let rxf = fine.nx / coarse.nx;
        let ryf = fine.ny / coarse.ny;
        let rzf = fine.nz / coarse.nz;

        debug_assert!(rxf == 1 || rxf == 2);
        debug_assert!(ryf == 1 || ryf == 2);
        debug_assert!(rzf == 1 || rzf == 2);

        coarse
            .rhs
            .par_chunks_mut(nxc)
            .enumerate()
            .for_each(|(rowc_idx, rhs_row)| {
                let kc = rowc_idx / nyc;
                let jc = rowc_idx % nyc;
                if kc == 0 || kc + 1 == nzc || jc == 0 || jc + 1 == nyc {
                    return;
                }

                for ic in 1..(nxc - 1) {
                    let fi0 = rxf * ic;
                    let fj0 = ryf * jc;
                    let fk0 = rzf * kc;

                    let mut sum = 0.0;
                    for dk in 0..rzf {
                        for dj in 0..ryf {
                            for di in 0..rxf {
                                let id_f = idx3(fi0 + di, fj0 + dj, fk0 + dk, fine.nx, fine.ny);
                                sum += fine.res[id_f];
                            }
                        }
                    }

                    let w = 1.0 / ((rxf * ryf * rzf) as f64);
                    rhs_row[ic] = w * sum;
                }
            });
    }

    fn prolongate_add(coarse: &MGLevel, fine: &mut MGLevel, kind: ProlongationKind) {
        let rx = fine.nx / coarse.nx;
        let ry = fine.ny / coarse.ny;
        let rz = fine.nz / coarse.nz;

        let nfx = fine.nx;
        let nfy = fine.ny;
        let nfz = fine.nz;

        match kind {
            ProlongationKind::Injection => {
                let cphi: &[f64] = &coarse.phi;

                fine.phi
                    .par_chunks_mut(nfx)
                    .enumerate()
                    .for_each(|(row_idx, phi_row)| {
                        let k = row_idx / nfy;
                        let j = row_idx % nfy;

                        if k == 0 || k + 1 == nfz || j == 0 || j + 1 == nfy {
                            return;
                        }

                        let kc = k / rz;
                        let jc = j / ry;

                        for i in 1..(nfx - 1) {
                            let ic = i / rx;

                            if ic == 0
                                || ic + 1 >= coarse.nx
                                || jc == 0
                                || jc + 1 >= coarse.ny
                                || kc == 0
                                || kc + 1 >= coarse.nz
                            {
                                continue;
                            }

                            let v = cphi[coarse.idx(ic, jc, kc)];
                            phi_row[i] += v;
                        }
                    });
            }

            ProlongationKind::Trilinear => {
                let cphi: &[f64] = &coarse.phi;

                fine.phi
                    .par_chunks_mut(nfx)
                    .enumerate()
                    .for_each(|(row_idx, phi_row)| {
                        let k = row_idx / nfy;
                        let j = row_idx % nfy;

                        if k == 0 || k + 1 == nfz || j == 0 || j + 1 == nfy {
                            return;
                        }

                        let kz = k / rz;
                        let rk = k % rz;
                        let (k0, k1, wk0, wk1) = interp_1d_cell_centered(kz, rk, coarse.nz, rz);

                        let jy = j / ry;
                        let rj = j % ry;
                        let (j0, j1, wj0, wj1) = interp_1d_cell_centered(jy, rj, coarse.ny, ry);

                        for i in 1..(nfx - 1) {
                            let ix = i / rx;
                            let ri = i % rx;
                            let (i0, i1, wi0, wi1) = interp_1d_cell_centered(ix, ri, coarse.nx, rx);

                            let mut v = 0.0;

                            v += wi0 * wj0 * wk0 * cphi[coarse.idx(i0, j0, k0)];
                            v += wi1 * wj0 * wk0 * cphi[coarse.idx(i1, j0, k0)];
                            v += wi0 * wj1 * wk0 * cphi[coarse.idx(i0, j1, k0)];
                            v += wi1 * wj1 * wk0 * cphi[coarse.idx(i1, j1, k0)];
                            v += wi0 * wj0 * wk1 * cphi[coarse.idx(i0, j0, k1)];
                            v += wi1 * wj0 * wk1 * cphi[coarse.idx(i1, j0, k1)];
                            v += wi0 * wj1 * wk1 * cphi[coarse.idx(i0, j1, k1)];
                            v += wi1 * wj1 * wk1 * cphi[coarse.idx(i1, j1, k1)];

                            phi_row[i] += v;
                        }
                    });
            }
        }
    }

    fn v_cycle(&mut self, l: usize) {
        let pre = self.cfg.pre_smooth;
        let post = self.cfg.post_smooth;

        let smoother = self.cfg.smoother;
        let omega_j = self.cfg.omega;
        let omega_sor = self.cfg.sor_omega;

        if l == self.levels.len() - 1 {
            match smoother {
                MGSmoother::WeightedJacobi => Self::smooth_weighted_jacobi(&mut self.levels[l], 80, omega_j),
                MGSmoother::RedBlackSOR => Self::smooth_rb_sor(&mut self.levels[l], 80, omega_sor),
            }
            return;
        }

        match smoother {
            MGSmoother::WeightedJacobi => Self::smooth_weighted_jacobi(&mut self.levels[l], pre, omega_j),
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
            MGSmoother::WeightedJacobi => Self::smooth_weighted_jacobi(&mut self.levels[l], post, omega_j),
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

    fn solve(&mut self) {
        let _ = self.solve_with_timing();
    }

    fn add_b_from_phi_on_magnet_layer(&self, m: &VectorField2D, _ms: f64, b_eff: &mut VectorField2D) {
        self.add_b_from_phi_on_magnet_layer_impl(Some(&m.data), b_eff);
    }

    fn add_b_from_phi_on_magnet_layer_all(&self, b_eff: &mut VectorField2D) {
        self.add_b_from_phi_on_magnet_layer_impl(None, b_eff);
    }

    fn add_b_from_phi_on_magnet_layer_impl(&self, mdata_opt: Option<&[[f64; 3]]>, b_eff: &mut VectorField2D) {
        let finest = &self.levels[0];
        let phi = &finest.phi;

        let nx_m = self.grid.nx;

        let dx = finest.dx;
        let dy = finest.dy;
        let dz = finest.dz;

        let k = self.offz;

        #[inline]
        fn is_mag(mv: [f64; 3]) -> bool {
            let n2 = mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2];
            n2 > 1e-30
        }

        b_eff
            .data
            .par_chunks_mut(nx_m)
            .enumerate()
            .for_each(|(j, row)| {
                let pj = self.offy + j;
                for i in 0..nx_m {
                    let idx2 = j * nx_m + i;
                    if let Some(mdata) = mdata_opt {
                        if !is_mag(mdata[idx2]) {
                            continue;
                        }
                    }

                    let pi = self.offx + i;
                    let phi_c = phi[idx3(pi, pj, k, self.px, self.py)];

                    let mut dphi_dx = 0.0;
                    let mut wdx = 0.0;
                    if pi + 1 < self.px {
                        let phi_p = phi[idx3(pi + 1, pj, k, self.px, self.py)];
                        dphi_dx += (phi_p - phi_c) / dx;
                        wdx += 1.0;
                    }
                    if pi > 0 {
                        let phi_m = phi[idx3(pi - 1, pj, k, self.px, self.py)];
                        dphi_dx += (phi_c - phi_m) / dx;
                        wdx += 1.0;
                    }
                    if wdx > 0.0 {
                        dphi_dx /= wdx;
                    }

                    let mut dphi_dy = 0.0;
                    let mut wdy = 0.0;
                    if pj + 1 < self.py {
                        let phi_p = phi[idx3(pi, pj + 1, k, self.px, self.py)];
                        dphi_dy += (phi_p - phi_c) / dy;
                        wdy += 1.0;
                    }
                    if pj > 0 {
                        let phi_m = phi[idx3(pi, pj - 1, k, self.px, self.py)];
                        dphi_dy += (phi_c - phi_m) / dy;
                        wdy += 1.0;
                    }
                    if wdy > 0.0 {
                        dphi_dy /= wdy;
                    }

                    let mut dphi_dz = 0.0;
                    let mut wdz = 0.0;
                    if k + 1 < self.pz {
                        let phi_p = phi[idx3(pi, pj, k + 1, self.px, self.py)];
                        dphi_dz += (phi_p - phi_c) / dz;
                        wdz += 1.0;
                    }
                    if k > 0 {
                        let phi_m = phi[idx3(pi, pj, k - 1, self.px, self.py)];
                        dphi_dz += (phi_c - phi_m) / dz;
                        wdz += 1.0;
                    }
                    if wdz > 0.0 {
                        dphi_dz /= wdz;
                    }

                    row[i][0] += -MU0 * dphi_dx;
                    row[i][1] += -MU0 * dphi_dy;
                    row[i][2] += -MU0 * dphi_dz;
                }
            });
    }

    pub fn add_field(&mut self, m: &VectorField2D, b_eff: &mut VectorField2D, ms: f64) {
        self.build_rhs_from_m(m, ms);
        self.solve();
        self.add_b_from_phi_on_magnet_layer(m, ms, b_eff);
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
    fn new(grid: &Grid2D, mg_cfg: &DemagPoissonMGConfig, hyb: &HybridConfig, op: MGOperatorSettings) -> Self {
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

struct DemagPoissonMGHybrid {
    mg: DemagPoissonMG,
    hyb: HybridConfig,

    dk_key: Option<DeltaKernelKey>,
    dk: Option<DeltaKernel2D>,
}

impl DemagPoissonMGHybrid {
    fn new(grid: Grid2D, mut mg_cfg: DemagPoissonMGConfig, hyb: HybridConfig) -> Self {
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

    fn same_structure(&self, grid: &Grid2D, mg_cfg: &DemagPoissonMGConfig) -> bool {
        self.mg.same_structure(grid, mg_cfg)
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

fn gaussian_kernel_1d(sigma_cells: f64) -> (Vec<f64>, isize) {
    if !(sigma_cells > 0.0) {
        return (vec![1.0], 0);
    }
    let r = (3.0 * sigma_cells).ceil() as isize;
    if r <= 0 {
        return (vec![1.0], 0);
    }
    let mut w = Vec::with_capacity((2 * r + 1) as usize);
    let inv2s2 = 1.0 / (2.0 * sigma_cells * sigma_cells);
    for i in -r..=r {
        let x = i as f64;
        w.push((-x * x * inv2s2).exp());
    }
    let sum: f64 = w.iter().sum();
    if sum > 0.0 {
        for wi in &mut w {
            *wi /= sum;
        }
    }
    (w, r)
}

fn screen_rhs_gaussian_xy(level: &mut MGLevel, sigma_cells: f64) {
    if !(sigma_cells > 0.0) {
        return;
    }

    let (w, r) = gaussian_kernel_1d(sigma_cells);
    if r <= 0 {
        return;
    }

    let nx = level.nx;
    let ny = level.ny;
    let nz = level.nz;

    if nx < 3 || ny < 3 || nz < 3 {
        return;
    }

    // Pass 1: X convolution (rhs -> tmp)
    {
        let rhs = &level.rhs;
        let tmp = &mut level.tmp;
        tmp.par_chunks_mut(nx)
            .enumerate()
            .for_each(|(row, tmp_row)| {
                let k = row / ny;
                let j = row - k * ny;
                if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                    tmp_row.fill(0.0);
                    return;
                }
                let base = row * nx;
                tmp_row[0] = 0.0;
                tmp_row[nx - 1] = 0.0;
                for i in 1..(nx - 1) {
                    let mut acc = 0.0;
                    let ii = i as isize;
                    for (t, wi) in (-r..=r).zip(w.iter()) {
                        let mut x = ii + t;
                        if x < 0 {
                            x = 0;
                        } else if x > (nx as isize - 1) {
                            x = nx as isize - 1;
                        }
                        acc += wi * rhs[base + x as usize];
                    }
                    tmp_row[i] = acc;
                }
            });
    }

    // Pass 2: Y convolution (tmp -> rhs)
    {
        let tmp = &level.tmp;
        let rhs = &mut level.rhs;
        rhs.par_chunks_mut(nx)
            .enumerate()
            .for_each(|(row, rhs_row)| {
                let k = row / ny;
                let j = row - k * ny;
                if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                    rhs_row.fill(0.0);
                    return;
                }
                rhs_row[0] = 0.0;
                rhs_row[nx - 1] = 0.0;
                let jj = j as isize;
                for i in 1..(nx - 1) {
                    let mut acc = 0.0;
                    for (t, wi) in (-r..=r).zip(w.iter()) {
                        let mut y = jj + t;
                        if y < 0 {
                            y = 0;
                        } else if y > (ny as isize - 1) {
                            y = ny as isize - 1;
                        }
                        let src_row = k * ny + y as usize;
                        acc += wi * tmp[src_row * nx + i];
                    }
                    rhs_row[i] = acc;
                }
            });
    }
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
fn mg_timing_record(rhs_ns: u64, bc_ns: u64, solve_ns: u64, grad_ns: u64, hybrid_ns: u64, total_ns: u64) {
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

// New file format (includes op + sigma + MG iteration params). Bump if layout changes.
const DK_CACHE_MAGIC: &[u8; 8] = b"LLGDKH6\x00";

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

    Some(DeltaKernel2D {
        radius: radius_xy,
        stride,
        dkxx,
        dkxy,
        dkyy,
        dkzz,
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
    Ok(())
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
    let r_big = if diag && r > 0 { ((r * 2).max(r + 4)).min(max_r) } else { 0 };

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
            for v in &mut b_fft.data { *v = [0.0; 3]; }
            demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);
            b_mg.data.iter_mut().for_each(|v| *v = [0.0; 3]);
            mg.build_rhs_from_m(&m_imp, ms);
            if hyb.sigma_cells > 0.0 { screen_rhs_gaussian_xy(&mut mg.levels[0], hyb.sigma_cells); }
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
            for v in &mut m_imp.data { *v = [geom_eps, geom_eps, geom_eps]; }
            m_imp.data[sidx] = [0.0, 1.0, 0.0];
            for v in &mut b_fft.data { *v = [0.0; 3]; }
            demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);
            b_mg.data.iter_mut().for_each(|v| *v = [0.0; 3]);
            mg.build_rhs_from_m(&m_imp, ms);
            if hyb.sigma_cells > 0.0 { screen_rhs_gaussian_xy(&mut mg.levels[0], hyb.sigma_cells); }
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
            for v in &mut m_imp.data { *v = [geom_eps, geom_eps, geom_eps]; }
            m_imp.data[sidx] = [0.0, 0.0, 1.0];
            for v in &mut b_fft.data { *v = [0.0; 3]; }
            demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);
            b_mg.data.iter_mut().for_each(|v| *v = [0.0; 3]);
            mg.build_rhs_from_m(&m_imp, ms);
            if hyb.sigma_cells > 0.0 { screen_rhs_gaussian_xy(&mut mg.levels[0], hyb.sigma_cells); }
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
                sx - cx as isize, sy - cy as isize, rel_xx, rel_yy, rel_zz, rel_xy
            );
            eprintln!(
                "[demag_mg] ΔK invariance check: shift=({:+},{:+}) max_abs dkxx={:.3e} dkyy={:.3e} dkzz={:.3e} dkxy={:.3e}",
                sx - cx as isize, sy - cy as isize, max_xx, max_yy, max_zz, max_xy
            );
        } else {
            eprintln!("[demag_mg] ΔK invariance check skipped: insufficient interior margin for shift");
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
