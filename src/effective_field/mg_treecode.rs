// src/effective_field/mg_treecode.rs
//
// Change B: Barnes–Hut treecode for open-boundary Dirichlet values,
// extracted from demag_poisson_mg.rs into a standalone module.
//
// This computes:
//   phi(r) = -(1/4π) ∫ rhs(r') / |r - r'| dV'
//
// on the boundary faces of a padded 3D box using monopole + dipole moments.
//
// Standalone so that:
//   1. The composite-grid solver can call it for L0 boundaries only.
//   2. Future García-Cervera boundary integral (Change D) can be added
//      alongside as an alternative BC method.
//
// NO NUMERICS CHANGE — the treecode evaluation must be bit-exact with the
// original inline implementation.

use std::f64::consts::PI;

use rayon::prelude::*;

use super::mg_kernels::idx3;

// ---------------------------------------------------------------------------
// Charge representation (volume element with position and charge)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Charge {
    pub pos: [f64; 3],
    pub q: f64,
}

// ---------------------------------------------------------------------------
// Tree node
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Barnes–Hut tree
// ---------------------------------------------------------------------------

pub struct BarnesHutTree {
    charges: Vec<Charge>,
    nodes: Vec<BhNode>,
    theta: f64,
    leaf_size: usize,
    max_depth: usize,
}

impl BarnesHutTree {
    pub fn build(
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
        if pos[0] >= center[0] { o |= 1; }
        if pos[1] >= center[1] { o |= 2; }
        if pos[2] >= center[2] { o |= 4; }
        o
    }

    fn ensure_child(&mut self, node_idx: usize, oct: usize) -> usize {
        if let Some(ci) = self.nodes[node_idx].children[oct] {
            return ci;
        }

        let parent_center = self.nodes[node_idx].center;
        let parent_half = self.nodes[node_idx].half;
        let child_half = parent_half * 0.5;
        let cx = parent_center[0] + if (oct & 1) != 0 { child_half } else { -child_half };
        let cy = parent_center[1] + if (oct & 2) != 0 { child_half } else { -child_half };
        let cz = parent_center[2] + if (oct & 4) != 0 { child_half } else { -child_half };

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

    pub fn eval_phi(&self, target: [f64; 3]) -> f64 {
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
                    let ddx = target[0] - c.pos[0];
                    let ddy = target[1] - c.pos[1];
                    let ddz = target[2] - c.pos[2];
                    let rr2 = ddx * ddx + ddy * ddy + ddz * ddz;
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

// ---------------------------------------------------------------------------
// High-level BC evaluation functions
// ---------------------------------------------------------------------------

/// Build charges from the RHS field on the finest 3D padded grid and return them.
///
/// Each interior cell with |rhs| > threshold becomes a point charge at
/// its cell-centred position with q = rhs * dV.
///
/// The coordinate system is centred on the box so the treecode monopole/dipole
/// moments are well-conditioned.
pub fn build_charges_from_rhs(
    rhs: &[f64],
    nx: usize, ny: usize, nz: usize,
    dx: f64, dy: f64, dz: f64,
) -> Vec<Charge> {
    let dvol = dx * dy * dz;
    let cx = (nx as f64) * 0.5;
    let cy = (ny as f64) * 0.5;
    let cz = (nz as f64) * 0.5;

    let mut charges = Vec::new();
    charges.reserve(rhs.len() / 8);

    for k in 0..nz {
        let z = (k as f64 + 0.5 - cz) * dz;
        for j in 0..ny {
            let y = (j as f64 + 0.5 - cy) * dy;
            let row = (k * ny + j) * nx;
            for i in 0..nx {
                let rho = rhs[row + i];
                if rho.abs() < 1e-40 { continue; }
                let x = (i as f64 + 0.5 - cx) * dx;
                charges.push(Charge { pos: [x, y, z], q: rho * dvol });
            }
        }
    }
    charges
}

/// Evaluate treecode Dirichlet BCs on the boundary faces of the padded box
/// and write them into `bc_phi`.
///
/// This is the extracted version of DemagPoissonMG::update_finest_boundary_bc
/// for the DirichletTreecode case. It builds the tree from `charges`, then
/// evaluates phi on all six boundary faces in parallel.
pub fn evaluate_treecode_bc(
    bc_phi: &mut [f64],
    charges: Vec<Charge>,
    nx: usize, ny: usize, nz: usize,
    dx: f64, dy: f64, dz: f64,
    tree_leaf: usize,
    tree_theta: f64,
    tree_max_depth: usize,
) {
    bc_phi.fill(0.0);

    if charges.is_empty() {
        return;
    }

    let lx = (nx as f64) * dx;
    let ly = (ny as f64) * dy;
    let lz = (nz as f64) * dz;
    let root_half = 0.5 * lx.max(ly).max(lz) + 1e-12;

    let cx = (nx as f64) * 0.5;
    let cy = (ny as f64) * 0.5;
    let cz = (nz as f64) * 0.5;

    let tree = BarnesHutTree::build(
        charges,
        [0.0, 0.0, 0.0],
        root_half,
        tree_leaf,
        tree_theta,
        tree_max_depth,
    );

    bc_phi
        .par_chunks_mut(nx)
        .enumerate()
        .for_each(|(row_idx, bc_row)| {
            let k = row_idx / ny;
            let j = row_idx % ny;

            let z = (k as f64 + 0.5 - cz) * dz;
            let y = (j as f64 + 0.5 - cy) * dy;

            if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                // Entire row is on a boundary face — evaluate all cells.
                for i in 0..nx {
                    let x = (i as f64 + 0.5 - cx) * dx;
                    bc_row[i] = tree.eval_phi([x, y, z]);
                }
            } else {
                // Interior row — only x-face endpoints.
                let x0 = (0.5 - cx) * dx;
                bc_row[0] = tree.eval_phi([x0, y, z]);

                let x1 = (nx as f64 - 0.5 - cx) * dx;
                bc_row[nx - 1] = tree.eval_phi([x1, y, z]);
            }
        });
}

/// Evaluate monopole+dipole Dirichlet BCs on the boundary faces.
///
/// This is the extracted version of the DirichletDipole case.
pub fn evaluate_dipole_bc(
    bc_phi: &mut [f64],
    rhs: &[f64],
    nx: usize, ny: usize, nz: usize,
    dx: f64, dy: f64, dz: f64,
) {
    bc_phi.fill(0.0);

    let dvol = dx * dy * dz;
    let cx = (nx as f64) * 0.5;
    let cy = (ny as f64) * 0.5;
    let cz = (nz as f64) * 0.5;

    // Accumulate monopole + dipole moments.
    let mut q = 0.0f64;
    let mut pxm = 0.0f64;
    let mut pym = 0.0f64;
    let mut pzm = 0.0f64;

    for k in 0..nz {
        let z = (k as f64 + 0.5 - cz) * dz;
        for j in 0..ny {
            let y = (j as f64 + 0.5 - cy) * dy;
            let row = (k * ny + j) * nx;
            for i in 0..nx {
                let x = (i as f64 + 0.5 - cx) * dx;
                let rho = rhs[row + i];
                let w = rho * dvol;
                q += w;
                pxm += w * x;
                pym += w * y;
                pzm += w * z;
            }
        }
    }

    let inv4pi = 1.0 / (4.0 * PI);

    // Helper to evaluate monopole+dipole potential at a point.
    let eval = |x: f64, y: f64, z: f64| -> f64 {
        let r2 = x * x + y * y + z * z;
        if r2 <= 0.0 { return 0.0; }
        let r = r2.sqrt();
        let pr = pxm * x + pym * y + pzm * z;
        -inv4pi * (q / r + pr / (r * r * r))
    };

    // x-faces
    for k in 0..nz {
        let z = (k as f64 + 0.5 - cz) * dz;
        for j in 0..ny {
            let y = (j as f64 + 0.5 - cy) * dy;
            let x0 = (0.5 - cx) * dx;
            bc_phi[idx3(0, j, k, nx, ny)] = eval(x0, y, z);
            let x1 = (nx as f64 - 0.5 - cx) * dx;
            bc_phi[idx3(nx - 1, j, k, nx, ny)] = eval(x1, y, z);
        }
        // y-faces
        for i in 0..nx {
            let x = (i as f64 + 0.5 - cx) * dx;
            let y0 = (0.5 - cy) * dy;
            bc_phi[idx3(i, 0, k, nx, ny)] = eval(x, y0, z);
            let y1 = (ny as f64 - 0.5 - cy) * dy;
            bc_phi[idx3(i, ny - 1, k, nx, ny)] = eval(x, y1, z);
        }
    }
    // z-faces
    for j in 0..ny {
        let y = (j as f64 + 0.5 - cy) * dy;
        for i in 0..nx {
            let x = (i as f64 + 0.5 - cx) * dx;
            let z0 = (0.5 - cz) * dz;
            bc_phi[idx3(i, j, 0, nx, ny)] = eval(x, y, z0);
            let z1 = (nz as f64 - 0.5 - cz) * dz;
            bc_phi[idx3(i, j, nz - 1, nx, ny)] = eval(x, y, z1);
        }
    }
}