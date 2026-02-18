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
// - **Hybrid mode (recommended):** uses MG for the long-range field and a small local
//   correction stencil ΔK = K_fft - K_mg (Newell/prism near-field) to fix the high-k
//   / near-field operator mismatch. This is the classic particle-mesh / PPPM idea.
//
// Hybrid controls:
//   LLG_DEMAG_MG_HYBRID_RADIUS=<cells>        (default 6; set 0 to disable)
//   LLG_DEMAG_MG_HYBRID_DELTA_VCYCLES=<n>     (default 60; one-time build cost)
//   LLG_DEMAG_MG_HYBRID_CACHE=1|0            (default 1; caches ΔK in out/demag_cache)
// - This implementation supports three outer BCs:
//     * DirichletZero     : phi = 0 on the padded box boundary (cheap, needs lots of vacuum)
//     * DirichletDipole   : boundary phi set by a monopole+dipole far-field approximation
//                           computed from the RHS moments (helps reduce required vacuum)
//     * DirichletTreecode : boundary phi set by an FMM-like Barnes–Hut treecode evaluation
//                           of the free-space Green's function (best accuracy with small padding)
//
// Caveat:
// - This discretisation is a standard 7-point Laplacian with cell-centered phi.
//   It will not exactly reproduce MuMax's prism-integrated kernel on coarse grids,
//   but it should converge with refinement and is AMR-friendly structurally.

use crate::grid::Grid2D;
use crate::params::{MU0, Material};
use crate::vector_field::VectorField2D;

use super::demag_fft_uniform;

use rayon::prelude::*;

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
    ///
    /// This reduces explicit vacuum padding, but is still only an approximation (higher
    /// multipoles are neglected).
    DirichletDipole,

    /// Dirichlet boundary values set by a Barnes–Hut treecode evaluation of the free-space
    /// Green's function integral:
    ///
    ///   phi(r) = -(1/4π) ∫ rhs(r') / |r - r'| dV'
    ///
    /// This is "FMM-like": it hierarchically groups sources and evaluates the potential at
    /// the outer boundary in ~O(N log N). With this BC, we can aggressively reduce explicit
    /// vacuum padding while retaining good far-field accuracy.
    DirichletTreecode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MGSmoother {
    /// Weighted Jacobi (parallel-friendly, but weaker than Gauss–Seidel).
    WeightedJacobi,
    /// Red-black Gauss–Seidel with optional SOR over-relaxation (stronger smoother).
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
    ///
    /// The actual padded size is:
    ///   px = nx + 2*pad_xy (+1 if needed to make px even for coarsening)
    ///   py = ny + 2*pad_xy (+1 if needed to make py even for coarsening)
    ///
    /// This replaces the old "pad_factor_xy" approach which scaled padding with grid size.
    pub pad_xy: usize,

    /// Vacuum layers below and above the magnet layer (in units of dz cells).
    pub n_vac_z: usize,

    /// If `tol_abs` is None, run exactly this many V-cycles.
    pub v_cycles: usize,

    /// Max V-cycles if using a tolerance stop.
    pub v_cycles_max: usize,

    /// Stop when max-norm residual <= tol_abs (units: A/m^2).
    /// If None, run fixed v_cycles.
    pub tol_abs: Option<f64>,

    /// Pre-smoothing iterations.
    pub pre_smooth: usize,
    /// Post-smoothing iterations.
    pub post_smooth: usize,

    /// Smoother selection.
    pub smoother: MGSmoother,

    /// Weighted Jacobi relaxation parameter (0 < omega <= 1).
    pub omega: f64,

    /// Red-black SOR relaxation factor (0 < sor_omega < 2). Use 1.0 for plain RBGS.
    pub sor_omega: f64,

    /// Use previous phi as initial guess (warm start).
    pub warm_start: bool,

    /// Outer boundary condition on the padded box.
    pub bc: BoundaryCondition,

    /// Treecode opening angle θ (smaller -> more accurate, slower). Typical 0.4..0.8.
    pub tree_theta: f64,

    /// Treecode leaf size (direct evaluation threshold).
    pub tree_leaf: usize,

    /// Treecode max depth safeguard.
    pub tree_max_depth: usize,
}

impl Default for DemagPoissonMGConfig {
    fn default() -> Self {
        Self {
            // Aggressive by default: with Treecode BC we don't need "factor-of-two" padding.
            pad_xy: 2,
            n_vac_z: 2,
            // With a stronger smoother, we can usually run fewer V-cycles.
            v_cycles: 2,
            v_cycles_max: 80,
            tol_abs: None,
            pre_smooth: 2,
            post_smooth: 2,
            smoother: MGSmoother::RedBlackSOR,
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
    /// Optional: configure via env vars so you can experiment without plumbing through Material yet.
    pub fn from_env() -> Self {
        fn get_usize(name: &str) -> Option<usize> {
            std::env::var(name)
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
        }
        fn get_f64(name: &str) -> Option<f64> {
            std::env::var(name)
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
        }

        let mut cfg = Self::default();

        // Padding: prefer PAD_XY; accept legacy PAD_FACTOR_XY as an alias for PAD_XY.
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
            // Clamp to a conservative stable range.
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
    /// Local correction stencil radius in cells (XY).
    ///
    /// 0 disables the hybrid correction and uses pure MG.
    /// Typical useful values: 2..10.
    radius_xy: usize,

    /// When building the local correction stencil (ΔK = K_fft - K_mg),
    /// run this many MG V-cycles per impulse solve to make the stencil
    /// stable and not dependent on the runtime MG iteration count.
    delta_v_cycles: usize,

    /// Cache the computed local correction stencil to disk (out/demag_cache).
    cache_to_disk: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            // Enable hybrid by default: MG alone is known to be inaccurate for high-k patterns.
            // You can disable by setting LLG_DEMAG_MG_HYBRID_RADIUS=0.
            radius_xy: 6,
            // A moderately-large, one-time build cost that pays off by making ΔK stable.
            delta_v_cycles: 60,
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

        let mut cfg = Self::default();
        if let Some(v) = get_usize("LLG_DEMAG_MG_HYBRID_RADIUS") {
            cfg.radius_xy = v;
        }
        if let Some(v) = get_usize("LLG_DEMAG_MG_HYBRID_DELTA_VCYCLES") {
            cfg.delta_v_cycles = v.max(1);
        }
        if let Ok(v) = std::env::var("LLG_DEMAG_MG_HYBRID_CACHE") {
            cfg.cache_to_disk = matches!(v.as_str(), "1" | "true" | "yes" | "on");
        }
        cfg
    }

    #[inline]
    fn enabled(&self) -> bool {
        self.radius_xy > 0
    }
}

#[inline]
fn idx3(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    (k * ny + j) * nx + i
}

#[inline]
fn stamp_dirichlet_bc(arr: &mut [f64], bc_phi: &[f64], nx: usize, ny: usize, nz: usize) {
    // Copy stored Dirichlet boundary values into `arr`.
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
// Local correction kernel ΔK(r) = K_fft(r) - K_mg(r), truncated to a square stencil.
// ---------------------------

/// Stored in real space, per unit magnetisation amplitude (Tesla per A/m).
///
/// The field correction is:
///   B_corr_x = Σ [ΔKxx(r)*Mx(src) + ΔKxy(r)*My(src)]
///   B_corr_y = Σ [ΔKxy(r)*Mx(src) + ΔKyy(r)*My(src)]
///   B_corr_z = Σ [ΔKzz(r)*Mz(src)]
/// with r = (target - src).
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

                    // Loop over displacement r = (target - src).
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
}

// ---------------------------
// Barnes–Hut treecode (FMM-like) for open-boundary Dirichlet values
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
    // Only populated for leaf nodes (<= leaf_size). Internal nodes keep this empty.
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
                // Split the leaf: redistribute all indices into children.
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
            // Leaf: direct moments about node center.
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
            // Internal: accumulate children, shifting dipoles to this center.
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

        // Barnes–Hut acceptance:
        // Use the distance from the target point to the node's *bounding cube* (not the center).
        // This avoids approximating a large node whose center is far away but which extends
        // close to the target (a common source of large errors near boundaries).
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
            // target is inside / touching this node cube: do not approximate
            false
        };

        if accept {
            if node.is_leaf() {
                // Direct sum within leaf.
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
                // Multipole (monopole + dipole).
                // (r should be > 0 here; for d>0, r cannot be zero.)
                let inv_r = 1.0 / r;
                let inv_r3 = inv_r * inv_r * inv_r;
                let pr = node.p[0] * rx + node.p[1] * ry + node.p[2] * rz;
                node.q * inv_r + pr * inv_r3
            }
        } else {
            // Descend.
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

    /// Current solution on this level.
    phi: Vec<f64>,
    /// Right-hand side (Poisson source).
    rhs: Vec<f64>,
    /// Residual (rhs - L phi).
    res: Vec<f64>,
    /// Scratch array for smoothers.
    tmp: Vec<f64>,

    /// Dirichlet boundary values (only boundary cells are used; interior may remain 0).
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
            phi: vec![0.0; n],
            rhs: vec![0.0; n],
            res: vec![0.0; n],
            tmp: vec![0.0; n],
            bc_phi: vec![0.0; n],
        }
    }

    /// Enforce Dirichlet boundary on `self.phi`.
    fn enforce_dirichlet(&mut self) {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let (phi, bc_phi) = (&mut self.phi, &self.bc_phi);
        stamp_dirichlet_bc(phi, bc_phi, nx, ny, nz);
    }
}

pub struct DemagPoissonMG {
    grid: Grid2D,
    cfg: DemagPoissonMGConfig,

    // padded domain size
    px: usize,
    py: usize,
    pz: usize,

    // offset to embed the magnet region inside padded domain
    offx: usize,
    offy: usize,
    offz: usize, // magnet layer index in z

    levels: Vec<MGLevel>,
}

impl DemagPoissonMG {
    pub fn new(grid: Grid2D, cfg: DemagPoissonMGConfig) -> Self {
        let nx = grid.nx.max(1);
        let ny = grid.ny.max(1);

        let pad = cfg.pad_xy.max(1);
        let mut px = nx + 2 * pad;
        let mut py = ny + 2 * pad;

        // Make even so we can coarsen cleanly.
        if px % 2 == 1 {
            px += 1;
        }
        if py % 2 == 1 {
            py += 1;
        }

        // Use a 3D box: one magnet layer + vacuum above/below.
        //
        // IMPORTANT: For multigrid coarsening, having at least one even dimension in z helps.
        // With the old choice pz = 1 + 2*n_vac, pz is always odd -> no z-coarsening possible.
        // Here we pad by +1 vacuum layer so pz becomes even (still roughly symmetric).
        let n_vac = cfg.n_vac_z.max(1);
        let mut pz = 1 + 2 * n_vac;
        if pz % 2 == 1 {
            pz += 1; // add one extra vacuum layer on top
        }

        let offx = (px.saturating_sub(nx)) / 2;
        let offy = (py.saturating_sub(ny)) / 2;
        let offz = n_vac;

        // Build multigrid levels by (semi-)coarsening by ~2 while possible.
        //
        // We allow semi-coarsening: always coarsen x/y together, and coarsen z when even/large enough.
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

            // Safety: avoid pathological level counts.
            if levels.len() > 32 {
                break;
            }
        }

        Self {
            grid,
            cfg,
            px,
            py,
            pz,
            offx,
            offy,
            offz,
            levels,
        }
    }

    /// Only compare settings that change the *allocation / hierarchy shape*.
    fn same_structure(&self, grid: &Grid2D, cfg: &DemagPoissonMGConfig) -> bool {
        self.grid.nx == grid.nx
            && self.grid.ny == grid.ny
            && self.grid.dx == grid.dx
            && self.grid.dy == grid.dy
            && self.grid.dz == grid.dz
            && self.cfg.pad_xy == cfg.pad_xy
            && self.cfg.n_vac_z == cfg.n_vac_z
    }

    fn build_rhs_from_m(&mut self, m: &VectorField2D, ms: f64) {
        let finest = &mut self.levels[0];
        finest.rhs.fill(0.0);

        // Discrete divergence of M on the padded grid.
        //
        // Key detail: handling magnet/vacuum interfaces.
        // If we use naive face-averaging, surface "magnetic charges" (from M·n at interfaces)
        // are under-represented (often by ~1/2). Here we use:
        //  - average(M) on faces shared by two magnet cells
        //  - one-sided M (from the magnet cell) on magnet-vacuum faces
        //  - 0 on vacuum-vacuum faces
        let nx = finest.nx;
        let ny = finest.ny;
        let nz = finest.nz;

        let dx = finest.dx;
        let dy = finest.dy;
        let dz = finest.dz;

        // Capture embedding parameters.
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

            // Magnet exists only on one layer: pk == offz, and within magnet XY window.
            if pku != offz {
                return (false, [0.0; 3]);
            }
            if piu < offx || piu >= offx + nx_m || pju < offy || pju >= offy + ny_m {
                return (false, [0.0; 3]);
            }

            let mi = piu - offx;
            let mj = pju - offy;
            let id = mj * nx_m + mi;

            // Treat masked-out cells as vacuum by encoding them as m = (0,0,0).
            // This matches MuMax-style masking where Ms=0 outside the geometry.
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
                (true, true) => 0.5 * (a + b), // interior magnet face
                (true, false) => a,            // magnet-vacuum face: take magnet side
                (false, true) => b,            // magnet-vacuum face: take magnet side
                (false, false) => 0.0,         // vacuum-vacuum
            }
        }

        let rhs = &mut finest.rhs;

        // Parallelise over contiguous X-rows (k,j rows). This gives good parallelism for thin films (small nz).
        rhs.par_chunks_mut(nx)
            .enumerate()
            .for_each(|(row_idx, rhs_row)| {
                let k = row_idx / ny;
                let j = row_idx % ny;

                // Outer boundary rows are unused (Dirichlet).
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

                    // Face values (x+, x-, y+, y-, z+, z-)
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

    /// Update outer Dirichlet boundary values on the finest level.
    ///
    /// For `DirichletZero` this is all zeros.
    /// For `DirichletDipole` we compute a monopole+dipole approximation from the RHS moments.
    fn update_finest_boundary_bc(&mut self) {
        let finest = &mut self.levels[0];
        finest.bc_phi.fill(0.0);

        match self.cfg.bc {
            BoundaryCondition::DirichletZero => {
                // bc_phi already zero
            }
            BoundaryCondition::DirichletDipole => {
                // Compute monopole and dipole moments of rhs: q = ∫rhs dV, p = ∫rhs r dV
                //
                // Free-space solution of ∇²phi = rhs:
                //   phi(r) = -(1/4π) ∫ rhs(r') / |r - r'| dV'
                // Far-field multipole approximation (about origin):
                //   phi(r) ≈ -(1/4π) ( q/r + (p·r)/r^3 + ... )
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

                // Sequential moment accumulation (cheap relative to the solve).
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

                // Set boundary cell-centered values.
                let nx = finest.nx;
                let ny = finest.ny;
                let nz = finest.nz;

                for k in 0..nz {
                    let z = (k as f64 + 0.5 - cz) * dz;
                    for j in 0..ny {
                        let y = (j as f64 + 0.5 - cy) * dy;

                        // i = 0
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
                        // i = nx-1
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

                        // j = 0
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
                        // j = ny-1
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

                        // k = 0
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
                        // k = nz-1
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
                // Barnes–Hut treecode evaluation of the free-space boundary potential.
                //
                // phi(r) = -(1/4π) ∫ rhs(r') / |r-r'| dV'
                //
                // We treat each cell as a point source with strength q = rhs * dV located at the
                // cell center. This is consistent with our cell-centered Laplacian discretisation.
                let dx = finest.dx;
                let dy = finest.dy;
                let dz = finest.dz;
                let dvol = dx * dy * dz;

                let cx = (finest.nx as f64) * 0.5;
                let cy = (finest.ny as f64) * 0.5;
                let cz = (finest.nz as f64) * 0.5;

                // Build the source list (charges).
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

                    // Parallelise over (k,j) rows to get good parallelism for thin-film domains (small nz).
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
                                // Entire row lies on a boundary face (y or z face): all i are boundary cells.
                                for i in 0..nx {
                                    let x = (i as f64 + 0.5 - cx) * dx;
                                    bc_row[i] = tree.eval_phi([x, y, z]);
                                }
                            } else {
                                // Interior row: only i = 0 and i = nx-1 are boundary cells.
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

        // Ensure the current phi respects the boundary.
        finest.enforce_dirichlet();
    }

    fn smooth_weighted_jacobi(level: &mut MGLevel, iters: usize, omega: f64) {
        let nx = level.nx;
        let ny = level.ny;
        let nz = level.nz;

        let sx = level.inv_dx2;
        let sy = level.inv_dy2;
        let sz = level.inv_dz2;

        let denom = 2.0 * (sx + sy + sz);
        let plane = nx * ny;

        for _ in 0..iters {
            // Start with zeros, then stamp in Dirichlet boundary values.
            level.tmp.fill(0.0);
            stamp_dirichlet_bc(&mut level.tmp, &level.bc_phi, nx, ny, nz);

            let phi_ro: &[f64] = &level.phi;
            let rhs_ro: &[f64] = &level.rhs;

            // Parallelise over rows for good scaling when nz is small.
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
                        let id = base + i;

                        let xm = phi_ro[id - 1];
                        let xp = phi_ro[id + 1];
                        let ym = phi_ro[id - nx];
                        let yp = phi_ro[id + nx];
                        let zm = phi_ro[id - plane];
                        let zp = phi_ro[id + plane];

                        let off = sx * (xm + xp) + sy * (ym + yp) + sz * (zm + zp);
                        let phi_new = (off - rhs_ro[id]) / denom;

                        tmp_row[i] = (1.0 - omega) * phi_ro[id] + omega * phi_new;
                    }
                });

            std::mem::swap(&mut level.phi, &mut level.tmp);
        }
    }

    /// Parallel red-black Gauss–Seidel / SOR update.
    ///
    /// Key property: within each colour sweep, updates are independent (depend only on opposite colour).
    /// We exploit this by computing new values into `tmp` in parallel, then applying them back to `phi`.
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
                // Compute updated values for this colour into tmp.
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

                // Apply updates back into phi for this colour.
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

            // Keep Dirichlet boundaries pinned after each full red+black sweep.
            level.enforce_dirichlet();
        }
    }

    fn compute_residual(level: &mut MGLevel) -> f64 {
        let nx = level.nx;
        let ny = level.ny;
        let nz = level.nz;

        let sx = level.inv_dx2;
        let sy = level.inv_dy2;
        let sz = level.inv_dz2;
        let diag = -2.0 * (sx + sy + sz);

        level.res.fill(0.0);

        let plane = nx * ny;
        let phi_ro: &[f64] = &level.phi;
        let rhs_ro: &[f64] = &level.rhs;

        level
            .res
            .par_chunks_mut(nx)
            .enumerate()
            .map(|(row_idx, res_row)| {
                let k = row_idx / ny;
                let j = row_idx % ny;
                if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                    return 0.0f64;
                }

                let base = row_idx * nx;
                let mut max_abs: f64 = 0.0;

                for i in 1..(nx - 1) {
                    let id = base + i;

                    let xm = phi_ro[id - 1];
                    let xp = phi_ro[id + 1];
                    let ym = phi_ro[id - nx];
                    let yp = phi_ro[id + nx];
                    let zm = phi_ro[id - plane];
                    let zp = phi_ro[id + plane];

                    let aphi = sx * (xm + xp) + sy * (ym + yp) + sz * (zm + zp) + diag * phi_ro[id];
                    let r = rhs_ro[id] - aphi;

                    res_row[i] = r;
                    max_abs = max_abs.max(r.abs());
                }
                max_abs
            })
            .reduce(|| 0.0f64, f64::max)
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

    fn prolongate_add(coarse: &MGLevel, fine: &mut MGLevel) {
        // Piecewise-constant prolongation (injection):
        // each fine cell receives the correction of its parent coarse cell.
        //
        // This is robust for cell-centered multigrid, and works with semi-coarsening.
        let nxf = fine.nx;
        let nyf = fine.ny;
        let nzf = fine.nz;

        let rxf = fine.nx / coarse.nx;
        let ryf = fine.ny / coarse.ny;
        let rzf = fine.nz / coarse.nz;

        debug_assert!(rxf == 1 || rxf == 2);
        debug_assert!(ryf == 1 || ryf == 2);
        debug_assert!(rzf == 1 || rzf == 2);

        fine.phi
            .par_chunks_mut(nxf)
            .enumerate()
            .for_each(|(rowf_idx, phi_row)| {
                let kf = rowf_idx / nyf;
                let jf = rowf_idx % nyf;

                if kf == 0 || kf + 1 == nzf || jf == 0 || jf + 1 == nyf {
                    return;
                }

                let kc = kf / rzf;
                let jc = jf / ryf;

                for if_ in 1..(nxf - 1) {
                    let ic = if_ / rxf;
                    let corr = coarse.phi[idx3(ic, jc, kc, coarse.nx, coarse.ny)];
                    phi_row[if_] += corr;
                }
            });
    }

    fn v_cycle(&mut self, l: usize) {
        let pre = self.cfg.pre_smooth;
        let post = self.cfg.post_smooth;

        let smoother = self.cfg.smoother;
        let omega_j = self.cfg.omega;
        let omega_sor = self.cfg.sor_omega;

        if l == self.levels.len() - 1 {
            // Coarsest: do extra smoothing (acts as a cheap coarse solve).
            match smoother {
                MGSmoother::WeightedJacobi => {
                    Self::smooth_weighted_jacobi(&mut self.levels[l], 80, omega_j)
                }
                MGSmoother::RedBlackSOR => Self::smooth_rb_sor(&mut self.levels[l], 80, omega_sor),
            }
            return;
        }

        match smoother {
            MGSmoother::WeightedJacobi => {
                Self::smooth_weighted_jacobi(&mut self.levels[l], pre, omega_j)
            }
            MGSmoother::RedBlackSOR => Self::smooth_rb_sor(&mut self.levels[l], pre, omega_sor),
        }

        // residual on level l
        Self::compute_residual(&mut self.levels[l]);

        // restrict residual to rhs on level l+1
        {
            let (fine, coarse) = {
                let (a, b) = self.levels.split_at_mut(l + 1);
                (&a[l], &mut b[0])
            };
            Self::restrict_residual(fine, coarse);
        }

        // recurse on coarse
        self.v_cycle(l + 1);

        // prolongate correction back
        {
            let (fine, coarse) = {
                let (a, b) = self.levels.split_at_mut(l + 1);
                (&mut a[l], &b[0])
            };
            Self::prolongate_add(coarse, fine);
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

        // Update and enforce Dirichlet boundary values on the finest level.
        let t_bc = Instant::now();
        self.update_finest_boundary_bc();
        let bc_ns = t_bc.elapsed().as_nanos() as u64;

        let t_solve = Instant::now();

        if let Some(tol) = self.cfg.tol_abs {
            for _ in 0..self.cfg.v_cycles_max {
                self.v_cycle(0);
                // Keep boundary pinned (defensive).
                self.levels[0].enforce_dirichlet();

                let max_r = Self::compute_residual(&mut self.levels[0]);
                if max_r <= tol {
                    break;
                }
            }
        } else {
            for _ in 0..self.cfg.v_cycles {
                self.v_cycle(0);
                // Keep boundary pinned (defensive).
                self.levels[0].enforce_dirichlet();
            }
        }

        // Final BC enforcement.
        self.levels[0].enforce_dirichlet();

        let solve_ns = t_solve.elapsed().as_nanos() as u64;
        (bc_ns, solve_ns)
    }

    fn solve(&mut self) {
        let _ = self.solve_with_timing();
    }

    fn add_b_from_phi_on_magnet_layer(&self, b_eff: &mut VectorField2D) {
        let finest = &self.levels[0];
        let phi = &finest.phi;

        let nx_m = self.grid.nx;

        let dx = finest.dx;
        let dy = finest.dy;
        let dz = finest.dz;

        let k = self.offz; // magnet layer index

        b_eff
            .data
            .par_chunks_mut(nx_m)
            .enumerate()
            .for_each(|(j, row)| {
                for i in 0..nx_m {
                    let pi = self.offx + i;
                    let pj = self.offy + j;

                    // Central differences (magnet should be away from outer boundary due to padding).
                    let ip = (pi + 1).min(self.px - 1);
                    let im = pi.saturating_sub(1);
                    let jp = (pj + 1).min(self.py - 1);
                    let jm = pj.saturating_sub(1);
                    let kp = (k + 1).min(self.pz - 1);
                    let km = k.saturating_sub(1);

                    let dphi_dx = (phi[idx3(ip, pj, k, self.px, self.py)]
                        - phi[idx3(im, pj, k, self.px, self.py)])
                        / (2.0 * dx);
                    let dphi_dy = (phi[idx3(pi, jp, k, self.px, self.py)]
                        - phi[idx3(pi, jm, k, self.px, self.py)])
                        / (2.0 * dy);
                    let dphi_dz = (phi[idx3(pi, pj, kp, self.px, self.py)]
                        - phi[idx3(pi, pj, km, self.px, self.py)])
                        / (2.0 * dz);

                    // H = -grad(phi); B = μ0 H
                    let bx = -MU0 * dphi_dx;
                    let by = -MU0 * dphi_dy;
                    let bz = -MU0 * dphi_dz;

                    row[i][0] += bx;
                    row[i][1] += by;
                    row[i][2] += bz;
                }
            });
    }

    pub fn add_field(&mut self, m: &VectorField2D, b_eff: &mut VectorField2D, ms: f64) {
        self.build_rhs_from_m(m, ms);
        self.solve();
        self.add_b_from_phi_on_magnet_layer(b_eff);
    }

    fn add_field_timed(
        &mut self,
        m: &VectorField2D,
        b_eff: &mut VectorField2D,
        ms: f64,
    ) -> (u64, u64, u64, u64) {
        let t_rhs = Instant::now();
        self.build_rhs_from_m(m, ms);
        let rhs_ns = t_rhs.elapsed().as_nanos() as u64;

        let (bc_ns, solve_ns) = self.solve_with_timing();

        let t_grad = Instant::now();
        self.add_b_from_phi_on_magnet_layer(b_eff);
        let grad_ns = t_grad.elapsed().as_nanos() as u64;

        (rhs_ns, bc_ns, solve_ns, grad_ns)
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
    bc: BoundaryCondition,
    // Only relevant for DirichletTreecode, but included so disk caches are safe.
    tree_theta_bits: u64,
    tree_leaf: usize,
    tree_max_depth: usize,
    radius_xy: usize,
    delta_v_cycles: usize,
}

impl DeltaKernelKey {
    fn new(grid: &Grid2D, mg_cfg: &DemagPoissonMGConfig, hyb: &HybridConfig) -> Self {
        Self {
            nx: grid.nx,
            ny: grid.ny,
            dx_bits: grid.dx.to_bits(),
            dy_bits: grid.dy.to_bits(),
            dz_bits: grid.dz.to_bits(),
            pad_xy: mg_cfg.pad_xy,
            n_vac_z: mg_cfg.n_vac_z,
            bc: mg_cfg.bc,
            tree_theta_bits: mg_cfg.tree_theta.to_bits(),
            tree_leaf: mg_cfg.tree_leaf,
            tree_max_depth: mg_cfg.tree_max_depth,
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
        let fname = format!(
            "demag_mg_hybrid_dk_nx{}_ny{}_dx{:.3e}_dy{:.3e}_dz{:.3e}_pad{}_nvacz{}_bc{}_th{:.3e}_leaf{}_dep{}_r{}_dv{}.bin",
            self.nx,
            self.ny,
            dx,
            dy,
            dz,
            self.pad_xy,
            self.n_vac_z,
            bc_tag,
            tree_theta,
            self.tree_leaf,
            self.tree_max_depth,
            self.radius_xy,
            self.delta_v_cycles
        );
        PathBuf::from("out").join("demag_cache").join(fname)
    }
}

struct DemagPoissonMGHybrid {
    mg: DemagPoissonMG,
    hyb: HybridConfig,

    // cached ΔK stencil
    dk_key: Option<DeltaKernelKey>,
    dk: Option<DeltaKernel2D>,
}

impl DemagPoissonMGHybrid {
    fn new(grid: Grid2D, mg_cfg: DemagPoissonMGConfig, hyb: HybridConfig) -> Self {
        Self {
            mg: DemagPoissonMG::new(grid, mg_cfg),
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

        // Clamp radius so the impulse stencil fits within the grid.
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
        let key = DeltaKernelKey::new(&self.mg.grid, &self.mg.cfg, &hyb_eff);

        if self.dk_key == Some(key) && self.dk.is_some() {
            return;
        }

        // Try disk cache.
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
                "[demag_mg] hybrid cache miss -> building ΔK stencil (r={}, dv={}) ...",
                hyb_eff.radius_xy, hyb_eff.delta_v_cycles
            );
        }

        // Build ΔK by impulse response: ΔK = K_fft - K_mg.
        let dk = build_delta_kernel_impulse(&self.mg.grid, &self.mg.cfg, &hyb_eff, mat);

        if hyb_eff.cache_to_disk {
            // Best-effort: ignore IO errors.
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
        if mg_timing_enabled() {
            let t_total = Instant::now();

            // MG far field (timed)
            let (rhs_ns, bc_ns, solve_ns, grad_ns) = self.mg.add_field_timed(m, b_eff, mat.ms);

            // Local near-field correction (timed)
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
        } else {
            // MG far field
            self.mg.add_field(m, b_eff, mat.ms);

            // Local near-field correction
            if self.hyb.enabled() {
                self.ensure_delta_kernel(mat);
                if let Some(dk) = &self.dk {
                    dk.add_correction(m, b_eff, mat.ms);
                }
            }
        }
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

// Bump this if the MG operator / hybrid stencil definition changes in a way that should
// invalidate on-disk cached ΔK files.
const DK_CACHE_MAGIC: &[u8; 8] = b"LLGDKH2\0";

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

    // Helper to read little-endian u64.
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
    let bc_u = read_u64(&mut f)? as u32;
    let tree_theta_bits = read_u64(&mut f)?;
    let tree_leaf = read_u64(&mut f)? as usize;
    let tree_max_depth = read_u64(&mut f)? as usize;
    let radius_xy = read_u64(&mut f)? as usize;
    let delta_v_cycles = read_u64(&mut f)? as usize;

    let bc = match bc_u {
        0 => BoundaryCondition::DirichletZero,
        1 => BoundaryCondition::DirichletDipole,
        2 => BoundaryCondition::DirichletTreecode,
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
        bc,
        tree_theta_bits,
        tree_leaf,
        tree_max_depth,
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
    let bc_u: u64 = match key.bc {
        BoundaryCondition::DirichletZero => 0,
        BoundaryCondition::DirichletDipole => 1,
        BoundaryCondition::DirichletTreecode => 2,
    };
    write_u64(&mut f, bc_u)?;
    write_u64(&mut f, key.tree_theta_bits)?;
    write_u64(&mut f, key.tree_leaf as u64)?;
    write_u64(&mut f, key.tree_max_depth as u64)?;
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
) -> DeltaKernel2D {
    let nx = grid.nx;
    let ny = grid.ny;
    let cx = nx / 2;
    let cy = ny / 2;

    let ms = mat.ms;
    let ms_inv = if ms.abs() > 0.0 { 1.0 / ms } else { 0.0 };

    // Use a dedicated MG solver for stencil building so we don't perturb the runtime warm-start state.
    let mut cfg_imp = *mg_cfg;
    cfg_imp.warm_start = false;
    cfg_imp.tol_abs = None;
    cfg_imp.v_cycles = hyb.delta_v_cycles;
    cfg_imp.v_cycles_max = hyb.delta_v_cycles;

    let mut mg = DemagPoissonMG::new(*grid, cfg_imp);

    let r = hyb.radius_xy;
    let stride = 2 * r + 1;
    let nst = stride * stride;

    let mut dk = DeltaKernel2D::new(r);
    let mut dkxy_from_x = vec![0.0f64; nst];
    let mut dkxy_from_y = vec![0.0f64; nst];

    // Helper: build impulse magnetisation.
    let mut m_imp = VectorField2D::new(*grid);
    for v in &mut m_imp.data {
        *v = [0.0, 0.0, 0.0];
    }

    let mut b_fft = VectorField2D::new(*grid);
    let mut b_mg = VectorField2D::new(*grid);

    // X impulse
    {
        let center = m_imp.idx(cx, cy);
        m_imp.data[center] = [1.0, 0.0, 0.0];
    }
    demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);
    for v in &mut b_mg.data {
        *v = [0.0, 0.0, 0.0];
    }
    mg.add_field(&m_imp, &mut b_mg, ms);
    for dy in -(r as isize)..=(r as isize) {
        for dx in -(r as isize)..=(r as isize) {
            let tx = (cx as isize + dx) as usize;
            let ty = (cy as isize + dy) as usize;
            let id = b_fft.idx(tx, ty);
            let k = dk.idx(dx, dy);
            dk.dkxx[k] = (b_fft.data[id][0] - b_mg.data[id][0]) * ms_inv;
            dkxy_from_x[k] = (b_fft.data[id][1] - b_mg.data[id][1]) * ms_inv;
        }
    }

    // Y impulse
    for v in &mut m_imp.data {
        *v = [0.0, 0.0, 0.0];
    }
    {
        let center = m_imp.idx(cx, cy);
        m_imp.data[center] = [0.0, 1.0, 0.0];
    }
    demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);
    for v in &mut b_mg.data {
        *v = [0.0, 0.0, 0.0];
    }
    mg.add_field(&m_imp, &mut b_mg, ms);
    for dy in -(r as isize)..=(r as isize) {
        for dx in -(r as isize)..=(r as isize) {
            let tx = (cx as isize + dx) as usize;
            let ty = (cy as isize + dy) as usize;
            let id = b_fft.idx(tx, ty);
            let k = dk.idx(dx, dy);
            dk.dkyy[k] = (b_fft.data[id][1] - b_mg.data[id][1]) * ms_inv;
            dkxy_from_y[k] = (b_fft.data[id][0] - b_mg.data[id][0]) * ms_inv;
        }
    }

    // Z impulse
    for v in &mut m_imp.data {
        *v = [0.0, 0.0, 0.0];
    }
    {
        let center = m_imp.idx(cx, cy);
        m_imp.data[center] = [0.0, 0.0, 1.0];
    }
    demag_fft_uniform::compute_demag_field(grid, &m_imp, &mut b_fft, mat);
    for v in &mut b_mg.data {
        *v = [0.0, 0.0, 0.0];
    }
    mg.add_field(&m_imp, &mut b_mg, ms);
    for dy in -(r as isize)..=(r as isize) {
        for dx in -(r as isize)..=(r as isize) {
            let tx = (cx as isize + dx) as usize;
            let ty = (cy as isize + dy) as usize;
            let id = b_fft.idx(tx, ty);
            let k = dk.idx(dx, dy);
            dk.dkzz[k] = (b_fft.data[id][2] - b_mg.data[id][2]) * ms_inv;
        }
    }

    // Enforce symmetry for ΔKxy by averaging the two ways of extracting it.
    for i in 0..nst {
        dk.dkxy[i] = 0.5 * (dkxy_from_x[i] + dkxy_from_y[i]);
    }

    dk
}

// Cache a solver instance so we don’t rebuild hierarchies every field evaluation.
static DEMAG_MG_CACHE: OnceLock<Mutex<Option<DemagPoissonMGHybrid>>> = OnceLock::new();

/// Add demag field using Poisson+MG alternative.
///
/// This is intended to be called by a demag dispatcher (demag.rs) based on a method option.
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
        // Even if we didn't rebuild the hierarchy, update runtime knobs.
        s.mg.cfg = cfg;
        s.hyb = hyb;
        s.add_field(m, b_eff, mat);
    }
}

/// Compute demag B field using Poisson+MG into `out` (overwrites out).
pub fn compute_demag_field_poisson_mg(
    grid: &Grid2D,
    m: &VectorField2D,
    out: &mut VectorField2D,
    mat: &Material,
) {
    out.set_uniform(0.0, 0.0, 0.0);
    add_demag_field_poisson_mg(grid, m, out, mat);
}
