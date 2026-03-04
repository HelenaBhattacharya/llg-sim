// src/effective_field/boundary_integral_2d.rs
//
// 2D boundary integral for the open-BC Poisson decomposition (U = v + w).
//
// After solving ∇²w = ∇·M with w=0 on ∂V (the w-solve), we need boundary
// values for the v-solve (∇²v = 0 with Dirichlet BC from boundary integral).
//
// The boundary source density is:
//   g(y) = −M·ν̂(y) + ∂w/∂ν(y)   on ∂V
//
// where ν̂ is the outward unit normal.
//
// The single-layer potential evaluated at boundary points gives v:
//   v(x) = ∫_∂V N(x − y) g(y) dσ(y)
//
// where N(r) = −(1/(2π)) ln|r| is the 2D Newtonian potential (free-space Green's fn).
//
// For a rectangular domain ∂V has 4 edges. Boundary cells are discretised as
// line segments of length ds = dx or dy. Direct summation is O(N_bdy²),
// which is trivially fast for 2D grids (e.g. 256×256 → ~2048² ≈ 4M ops).
//
// Grid convention:
//   Cells (i,j) have centers at ((i+0.5)*dx, (j+0.5)*dy), i=0..nx-1, j=0..ny-1.
//   Nodes are at (i*dx, j*dy), i=0..nx, j=0..ny.
//   The domain boundary ∂V consists of the rectangle edges:
//     bottom: y = 0,  x ∈ [0, Lx]
//     top:    y = Ly,  x ∈ [0, Lx]
//     left:   x = 0,  y ∈ [0, Ly]
//     right:  x = Lx,  y ∈ [0, Ly]
//
//   Boundary nodes:
//     bottom: (i, 0)    for i = 0..nx       (nx+1 nodes)
//     top:    (i, ny)   for i = 0..nx       (nx+1 nodes)
//     left:   (0, j)    for j = 1..ny-1     (ny-1 interior nodes, corners already counted)
//     right:  (nx, j)   for j = 1..ny-1     (ny-1 interior nodes)
//
//   Total boundary nodes = 2*(nx+1) + 2*(ny-1) = 2*(nx + ny)

use std::f64::consts::PI;

/// A boundary node with position, outward normal, and source density.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryNode {
    /// Position (x, y) in physical coordinates.
    pub x: f64,
    pub y: f64,
    /// Outward unit normal.
    pub nx: f64,
    pub ny: f64,
    /// Integration weight (arc-length element ds).
    pub ds: f64,
    /// Node index in full (nx+1)×(ny+1) node grid: j_node * (grid_nx+1) + i_node.
    pub node_idx: usize,
}

/// Enumerate all boundary nodes of a rectangular domain.
///
/// Returns nodes in order: bottom → right → top → left, traversed counter-clockwise.
/// Corner nodes are included exactly once.
pub fn enumerate_boundary_nodes(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Vec<BoundaryNode> {
    let lx = nx as f64 * dx;
    let ly = ny as f64 * dy;
    let stride = nx + 1;

    let mut nodes = Vec::with_capacity(2 * (nx + ny));

    // Bottom edge: y = 0, outward normal = (0, -1), nodes (i, 0) for i = 0..nx
    for i in 0..=nx {
        let x = i as f64 * dx;
        // Integration weight: half-cells at corners, full cell at interior
        let ds = if i == 0 || i == nx { dx * 0.5 } else { dx };
        nodes.push(BoundaryNode {
            x,
            y: 0.0,
            nx: 0.0,
            ny: -1.0,
            ds,
            node_idx: 0 * stride + i,
        });
    }

    // Right edge: x = Lx, outward normal = (1, 0), nodes (nx, j) for j = 1..ny-1
    for j in 1..ny {
        let y = j as f64 * dy;
        let ds = dy;
        nodes.push(BoundaryNode {
            x: lx,
            y,
            nx: 1.0,
            ny: 0.0,
            ds,
            node_idx: j * stride + nx,
        });
    }

    // Top edge: y = Ly, outward normal = (0, 1), nodes (i, ny) for i = nx..0 (reversed for CCW)
    for ii in 0..=nx {
        let i = nx - ii; // reverse order for counter-clockwise traversal
        let x = i as f64 * dx;
        let ds = if i == 0 || i == nx { dx * 0.5 } else { dx };
        nodes.push(BoundaryNode {
            x,
            y: ly,
            nx: 0.0,
            ny: 1.0,
            ds,
            node_idx: ny * stride + i,
        });
    }

    // Left edge: x = 0, outward normal = (-1, 0), nodes (0, j) for j = ny-1..1 (reversed for CCW)
    for jj in 1..ny {
        let j = ny - jj; // reverse
        let y = j as f64 * dy;
        let ds = dy;
        nodes.push(BoundaryNode {
            x: 0.0,
            y,
            nx: -1.0,
            ny: 0.0,
            ds,
            node_idx: j * stride + 0,
        });
    }

    // Fix corner integration weights: each corner is shared between two edges.
    // The current scheme gives dx/2 from horizontal edge.
    // We also need dy/2 from the vertical edge. Since corners appear only once
    // (from the bottom/top edges), add the vertical contribution.
    // Bottom-left (i=0, j=0): index 0 in nodes
    nodes[0].ds = 0.5 * dx + 0.5 * dy;
    // Bottom-right (i=nx, j=0): index nx
    nodes[nx].ds = 0.5 * dx + 0.5 * dy;
    // Top-right (i=nx, j=ny): first node of top edge = index (nx+1) + (ny-1)
    let top_start = (nx + 1) + (ny - 1);
    nodes[top_start].ds = 0.5 * dx + 0.5 * dy;
    // Top-left (i=0, j=ny): last node of top edge = top_start + nx
    nodes[top_start + nx].ds = 0.5 * dx + 0.5 * dy;

    nodes
}

/// Compute the source density g at each boundary node.
///
/// g(y) = −M·ν̂(y) + ∂w/∂ν(y)
///
/// - `m_data`: cell-centered magnetisation m (unit vector), stored as `[[mx,my,mz]; nx*ny]`
///   in row-major order (cell (i,j) at index j*nx + i).
/// - `w_nodes`: potential w at all (nx+1)*(ny+1) nodes (from the w-solve; boundary nodes = 0).
/// - `boundary`: the boundary node list from `enumerate_boundary_nodes`.
/// - `ms`: saturation magnetisation (A/m).
///
/// Returns: vector of g values, one per boundary node, in the same order as `boundary`.
pub fn compute_source_density(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    m_data: &[[f64; 3]],
    ms: f64,
    w_nodes: &[f64],
    boundary: &[BoundaryNode],
) -> Vec<f64> {
    assert_eq!(m_data.len(), nx * ny);
    assert_eq!(w_nodes.len(), (nx + 1) * (ny + 1));

    let mut g = Vec::with_capacity(boundary.len());

    for bn in boundary {
        // 1) −M·ν̂: average M from adjacent cells onto the boundary node,
        //    then dot with outward normal.
        let m_dot_n = boundary_m_dot_normal(nx, ny, m_data, ms, bn);

        // 2) ∂w/∂ν: outward normal derivative of w at the boundary node.
        let dw_dn = boundary_dw_dn(nx, ny, dx, dy, w_nodes, bn);

        g.push(m_dot_n - dw_dn);    // g = +M·ν̂ - ∂w/∂ν  ← CORRECT
    }

    g
}

/// Compute M·ν̂ at a boundary node by averaging adjacent cell M values.
fn boundary_m_dot_normal(
    nx: usize,
    ny: usize,
    m_data: &[[f64; 3]],
    ms: f64,
    bn: &BoundaryNode,
) -> f64 {
    // Determine which cells are adjacent to this boundary node.
    // Node at grid position (i_node, j_node):
    let i_node = bn.node_idx % (nx + 1);
    let j_node = bn.node_idx / (nx + 1);

    // Adjacent cells: (i_node-1, j_node-1), (i_node, j_node-1),
    //                 (i_node-1, j_node),   (i_node, j_node)
    // Only cells within [0,nx)×[0,ny) are valid.
    let mut mx_sum = 0.0;
    let mut my_sum = 0.0;
    let mut count = 0.0;

    for dj in [-1i32, 0i32] {
        let cj = j_node as i32 + dj;
        if cj < 0 || cj >= ny as i32 {
            continue;
        }
        for di in [-1i32, 0i32] {
            let ci = i_node as i32 + di;
            if ci < 0 || ci >= nx as i32 {
                continue;
            }
            let idx = cj as usize * nx + ci as usize;
            mx_sum += m_data[idx][0];
            my_sum += m_data[idx][1];
            count += 1.0;
        }
    }

    if count > 0.0 {
        mx_sum /= count;
        my_sum /= count;
    }

    // M·ν̂ = Ms * (mx * nx + my * ny)
    ms * (mx_sum * bn.nx + my_sum * bn.ny)
}

/// Compute ∂w/∂ν at a boundary node using one-sided finite differences.
fn boundary_dw_dn(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    w_nodes: &[f64],
    bn: &BoundaryNode,
) -> f64 {
    let stride = nx + 1;
    let i = bn.node_idx % stride;
    let j = bn.node_idx / stride;

    // ∂w/∂ν = ∇w · ν̂
    // We compute ∂w/∂x and ∂w/∂y at the boundary using one-sided differences,
    // then dot with the outward normal.

    // ∂w/∂x: use one-sided difference pointing inward
    let dwdx = if bn.nx < -0.5 {
        // Left boundary (x=0): forward difference
        let w_right = w_nodes[j * stride + (i + 1)];
        let w_here = w_nodes[j * stride + i]; // = 0 for homogeneous Dirichlet
        (w_right - w_here) / dx
    } else if bn.nx > 0.5 {
        // Right boundary (x=Lx): backward difference
        let w_left = w_nodes[j * stride + (i - 1)];
        let w_here = w_nodes[j * stride + i];
        (w_here - w_left) / dx
    } else if i > 0 && i < nx {
        // Interior x-position (top/bottom edges): central difference
        let w_left = w_nodes[j * stride + (i - 1)];
        let w_right = w_nodes[j * stride + (i + 1)];
        (w_right - w_left) / (2.0 * dx)
    } else if i == 0 {
        let w_right = w_nodes[j * stride + 1];
        let w_here = w_nodes[j * stride + 0];
        (w_right - w_here) / dx
    } else {
        let w_left = w_nodes[j * stride + (nx - 1)];
        let w_here = w_nodes[j * stride + nx];
        (w_here - w_left) / dx
    };

    // ∂w/∂y: similar
    let dwdy = if bn.ny < -0.5 {
        // Bottom boundary (y=0): forward difference
        let w_up = w_nodes[(j + 1) * stride + i];
        let w_here = w_nodes[j * stride + i];
        (w_up - w_here) / dy
    } else if bn.ny > 0.5 {
        // Top boundary (y=Ly): backward difference
        let w_down = w_nodes[(j - 1) * stride + i];
        let w_here = w_nodes[j * stride + i];
        (w_here - w_down) / dy
    } else if j > 0 && j < ny {
        // Interior y-position (left/right edges): central difference
        let w_down = w_nodes[(j - 1) * stride + i];
        let w_up = w_nodes[(j + 1) * stride + i];
        (w_up - w_down) / (2.0 * dy)
    } else if j == 0 {
        let w_up = w_nodes[1 * stride + i];
        let w_here = w_nodes[0 * stride + i];
        (w_up - w_here) / dy
    } else {
        let w_down = w_nodes[(ny - 1) * stride + i];
        let w_here = w_nodes[ny * stride + i];
        (w_here - w_down) / dy
    };

    // ∂w/∂ν = ∇w · ν̂
    dwdx * bn.nx + dwdy * bn.ny
}

/// Evaluate the single-layer potential at all boundary nodes.
///
/// v(x_i) = Σ_j N(x_i − y_j) g(y_j) ds_j
///
/// where N(r) = −(1/(2π)) ln|r| and the sum runs over all boundary nodes j ≠ i.
///
/// For the self-interaction (i = j), we use the regularised formula for a segment
/// of length ds:  N_self ≈ −(1/(2π)) [ln(ds/2) − 1] * ds
/// (from the integral of −(1/(2π)) ln|t| over t ∈ [−ds/2, ds/2]).
///
/// Returns: v values at all boundary nodes, in the same order as `boundary`.
pub fn evaluate_single_layer_potential(
    boundary: &[BoundaryNode],
    g: &[f64],
) -> Vec<f64> {
    let n = boundary.len();
    assert_eq!(g.len(), n);

    let inv_2pi = 1.0 / (2.0 * PI);
    let mut v = vec![0.0f64; n];

    for i in 0..n {
        let xi = boundary[i].x;
        let yi = boundary[i].y;
        let mut sum = 0.0;

        for j in 0..n {
            if i == j {
                // Self-interaction: regularised integral of −(1/2π)ln|r| over segment.
                // ∫_{-ds/2}^{ds/2} −(1/2π) ln|t| dt = −(ds/2π)(ln(ds/2) − 1)
                let ds = boundary[j].ds;
                if ds > 0.0 {
                    let self_val = -inv_2pi * ds * ((ds * 0.5).ln() - 1.0);
                    sum += self_val * g[j];
                }
                continue;
            }

            let xj = boundary[j].x;
            let yj = boundary[j].y;
            let rx = xi - xj;
            let ry = yi - yj;
            let r2 = rx * rx + ry * ry;

            if r2 < 1e-30 {
                // Degenerate (shouldn't happen for i≠j with distinct positions)
                continue;
            }

            // N(r) = −(1/2π) ln|r| = −(1/4π) ln(r²)
            let kernel = -inv_2pi * 0.5 * r2.ln();
            sum += kernel * g[j] * boundary[j].ds;
        }

        v[i] = sum;
    }

    v
}

/// Set boundary node values on the full (nx+1)×(ny+1) node grid.
///
/// `v_boundary`: values at boundary nodes (from `evaluate_single_layer_potential`).
/// `bc_full`: the full node grid (length (nx+1)*(ny+1)), initially zero.
///
/// After this call, `bc_full[node_idx]` = v_boundary[k] for each boundary node k.
pub fn set_boundary_values_on_node_grid(
    boundary: &[BoundaryNode],
    v_boundary: &[f64],
    bc_full: &mut [f64],
) {
    assert_eq!(v_boundary.len(), boundary.len());
    for (k, bn) in boundary.iter().enumerate() {
        bc_full[bn.node_idx] = v_boundary[k];
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_node_count() {
        let nx = 8;
        let ny = 6;
        let nodes = enumerate_boundary_nodes(nx, ny, 1.0, 1.0);
        // Expected: 2*(nx+ny) boundary nodes (corners counted once each)
        assert_eq!(nodes.len(), 2 * (nx + ny));
    }

    #[test]
    fn boundary_node_unique_indices() {
        let nx = 4;
        let ny = 3;
        let nodes = enumerate_boundary_nodes(nx, ny, 1.0, 1.0);
        let mut indices: Vec<usize> = nodes.iter().map(|n| n.node_idx).collect();
        indices.sort();
        indices.dedup();
        // All boundary node indices should be unique
        assert_eq!(indices.len(), nodes.len());
    }

    #[test]
    fn single_layer_constant_density() {
        // If g = constant on a square, the potential should be finite and smooth.
        let nx = 4;
        let ny = 4;
        let dx = 1.0;
        let dy = 1.0;
        let boundary = enumerate_boundary_nodes(nx, ny, dx, dy);
        let g = vec![1.0; boundary.len()];
        let v = evaluate_single_layer_potential(&boundary, &g);

        // Just check no NaN or Inf
        for val in &v {
            assert!(val.is_finite(), "Got non-finite value in potential: {}", val);
        }
    }
}