// src/effective_field/demag_poisson_dst.rs
//
// Demagnetising field via U = v + w potential decomposition with DST-based
// Poisson solver and open-boundary-condition boundary integral.
//
// For 2D thin films (Nz = 1), the magnetostatic potential U satisfies:
//   ∇²U = ∇·M    inside V
//   ∇²U = 0       outside V
//   [U]  = 0      on ∂V   (continuity)
//   [∂U/∂ν] = M·ν̂ on ∂V   (jump in normal derivative)
//
// Decomposition:
//   U = w + v
//
// w-solve:  ∇²w = ∇·M  in V,  w = 0 on ∂V     (homogeneous Dirichlet Poisson)
// boundary: g(y) = −M·ν̂(y) + ∂w/∂ν(y)          (source density on ∂V)
// integral: v(x) = ∫_∂V N(x−y) g(y) dσ(y)      (single-layer potential, gives BC)
// v-solve:  ∇²v = 0    in V,  v = integral vals  (inhomogeneous Dirichlet Laplace)
// compose:  U = w + v
// gradient: B_demag = −μ₀ ∇U  (in-plane) + thin-film Bz = −μ₀ Ms mz
//
// Public API:
//   add_demag_field_poisson_dst(grid, m, b_eff, mat)      — adds to b_eff
//   compute_demag_field_poisson_dst(grid, m, out, mat)     — overwrites out

use crate::grid::Grid2D;
use crate::params::{MU0, Material};
use crate::vector_field::VectorField2D;

use super::boundary_integral_2d::{
    BoundaryNode, compute_source_density, enumerate_boundary_nodes,
    evaluate_single_layer_potential, set_boundary_values_on_node_grid,
};
use super::dst_poisson_2d::DstPoisson2D;

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Caching: one solver per grid shape (like the FFT demag cache)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DstKey {
    nx: usize,
    ny: usize,
    dx_bits: u64,
    dy_bits: u64,
}

impl DstKey {
    fn new(grid: &Grid2D) -> Self {
        Self {
            nx: grid.nx,
            ny: grid.ny,
            dx_bits: grid.dx.to_bits(),
            dy_bits: grid.dy.to_bits(),
        }
    }
}

struct DstDemag {
    poisson: DstPoisson2D,
    boundary: Vec<BoundaryNode>,
    // Scratch buffers
    w_nodes: Vec<f64>,       // (nx+1)*(ny+1) node values of w
    rhs_interior: Vec<f64>,  // mx*my interior node RHS
    v_boundary: Vec<f64>,    // boundary node values of v
    bc_full: Vec<f64>,       // (nx+1)*(ny+1) full BC for v-solve
    v_nodes: Vec<f64>,       // mx*my interior node values of v
    u_nodes: Vec<f64>,       // (nx+1)*(ny+1) composed potential U = w + v
}

impl DstDemag {
    fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        let mx = nx - 1;
        let my = ny - 1;
        let nn = (nx + 1) * (ny + 1);

        Self {
            poisson: DstPoisson2D::new(nx, ny, dx, dy),
            boundary: enumerate_boundary_nodes(nx, ny, dx, dy),
            w_nodes: vec![0.0; nn],
            rhs_interior: vec![0.0; mx * my],
            v_boundary: Vec::new(),
            bc_full: vec![0.0; nn],
            v_nodes: vec![0.0; mx * my],
            u_nodes: vec![0.0; nn],
        }
    }
}

static DST_CACHE: OnceLock<Mutex<HashMap<DstKey, DstDemag>>> = OnceLock::new();

fn dst_timing_enabled() -> bool {
    std::env::var("LLG_DEMAG_DST_TIMING").is_ok()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute B_demag and add it to `b_eff`.
pub fn add_demag_field_poisson_dst(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    mat: &Material,
) {
    let t0 = if dst_timing_enabled() {
        Some(Instant::now())
    } else {
        None
    };

    let key = DstKey::new(grid);
    let cache = DST_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    let solver = map
        .entry(key)
        .or_insert_with(|| DstDemag::new(grid.nx, grid.ny, grid.dx, grid.dy));

    let nx = grid.nx;
    let ny = grid.ny;
    let dx = grid.dx;
    let dy = grid.dy;
    let ms = mat.ms;
    let mx = nx - 1;
    let my = ny - 1;
    let stride = nx + 1;

    // Phase 1: ∇·M at interior nodes → w-solve
    build_divergence_at_nodes(nx, ny, dx, dy, &m.data, ms, &mut solver.rhs_interior);
    solver.poisson.solve_homogeneous(&mut solver.rhs_interior);

    // Pack w into full node grid
    solver.w_nodes.fill(0.0);
    for q in 0..my {
        let j = q + 1;
        for p in 0..mx {
            let i = p + 1;
            solver.w_nodes[j * stride + i] = solver.rhs_interior[q * mx + p];
        }
    }

    // Phase 2: boundary integral
    let g = compute_source_density(
        nx, ny, dx, dy, &m.data, ms, &solver.w_nodes, &solver.boundary,
    );
    solver.v_boundary = evaluate_single_layer_potential(&solver.boundary, &g);

    // Phase 3: v-solve
    solver.bc_full.fill(0.0);
    set_boundary_values_on_node_grid(&solver.boundary, &solver.v_boundary, &mut solver.bc_full);
    solver.v_nodes.resize(mx * my, 0.0);
    solver
        .poisson
        .solve_inhomogeneous_laplace(&solver.bc_full, &mut solver.v_nodes);

    // Phase 4: compose U = w + v on full node grid
    // Boundary: U = v (since w=0 on boundary)
    solver.u_nodes.copy_from_slice(&solver.bc_full);
    // Interior: U = w + v
    for q in 0..my {
        let j = q + 1;
        for p in 0..mx {
            let i = p + 1;
            let nidx = j * stride + i;
            solver.u_nodes[nidx] =
                solver.w_nodes[nidx] + solver.v_nodes[q * mx + p];
        }
    }

    // Phase 5: gradient + z-component → B_demag
    add_demag_to_beff(nx, ny, dx, dy, ms, &solver.u_nodes, &m.data, b_eff);

    if let Some(t0) = t0 {
        eprintln!(
            "[demag_dst] {}x{} solve: {:.3} ms",
            nx, ny,
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
}

/// Compute B_demag into `out` (overwrites).
pub fn compute_demag_field_poisson_dst(
    grid: &Grid2D,
    m: &VectorField2D,
    out: &mut VectorField2D,
    mat: &Material,
) {
    out.set_uniform(0.0, 0.0, 0.0);
    add_demag_field_poisson_dst(grid, m, out, mat);
}

// ---------------------------------------------------------------------------
// Internal: build ∇·M at interior nodes from cell-centred m
// ---------------------------------------------------------------------------

/// Compute ∇·(Ms*m) at interior nodes of the (nx+1)×(ny+1) grid.
///
/// Interior node (i,j) for i=1..nx-1, j=1..ny-1.
/// The divergence at a node is computed by averaging face-normal fluxes
/// from surrounding cells:
///
///   ∂(Ms*mx)/∂x at node (i,j) ≈ Ms * [mx(i,·) − mx(i-1,·)] / dx
///   averaged over the two adjacent y-rows:
///   = Ms * 0.5 * [(mx(i,j-1) + mx(i,j)) − (mx(i-1,j-1) + mx(i-1,j))] / dx
///
/// Similarly for ∂(Ms*my)/∂y.
fn build_divergence_at_nodes(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    m_data: &[[f64; 3]],
    ms: f64,
    out: &mut Vec<f64>,
) {
    let mx = nx - 1;
    let my = ny - 1;
    out.resize(mx * my, 0.0);

    let inv_dx = ms / dx;
    let inv_dy = ms / dy;

    for q in 0..my {
        let j = q + 1; // node row
        for p in 0..mx {
            let i = p + 1; // node col

            // Cell indices of the 4 surrounding cells
            let c_bl = (j - 1) * nx + (i - 1); // bottom-left
            let c_br = (j - 1) * nx + i;        // bottom-right
            let c_tl = j * nx + (i - 1);         // top-left
            let c_tr = j * nx + i;                // top-right

            // ∂(Mx)/∂x averaged over the 2 adjacent y-rows
            let dmx_dx = 0.5 * inv_dx
                * ((m_data[c_br][0] + m_data[c_tr][0])
                    - (m_data[c_bl][0] + m_data[c_tl][0]));

            // ∂(My)/∂y averaged over the 2 adjacent x-columns
            let dmy_dy = 0.5 * inv_dy
                * ((m_data[c_tl][1] + m_data[c_tr][1])
                    - (m_data[c_bl][1] + m_data[c_br][1]));

            out[q * mx + p] = dmx_dx + dmy_dy;
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: gradient of U at cell centres → B_demag
// ---------------------------------------------------------------------------

/// Compute B_demag = −μ₀ ∇U at cell centres from node-centred U,
/// and add to b_eff. Also adds thin-film z self-demagnetisation.
///
/// Cell (i,j) centre is at ((i+0.5)*dx, (j+0.5)*dy).
/// Gradient computed by averaging the 4 surrounding node differences:
///
///   ∂U/∂x at cell (i,j) = 0.5*[(U(i+1,j) + U(i+1,j+1)) − (U(i,j) + U(i,j+1))] / dx
///   ∂U/∂y at cell (i,j) = 0.5*[(U(i,j+1) + U(i+1,j+1)) − (U(i,j) + U(i+1,j))] / dy
fn add_demag_to_beff(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    ms: f64,
    u_nodes: &[f64],
    m_data: &[[f64; 3]],
    b_eff: &mut VectorField2D,
) {
    let stride = nx + 1;
    let inv_dx = 1.0 / dx;
    let inv_dy = 1.0 / dy;

    // B_demag = −μ₀ ∇U for in-plane (U already includes Ms from RHS construction)
    let factor_xy = -MU0;

    // Thin-film self-demagnetisation: B_z = −μ₀ Ms mz
    let factor_z = -MU0 * ms;

    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;

            // 4 surrounding nodes of cell (i,j)
            let u00 = u_nodes[j * stride + i];
            let u10 = u_nodes[j * stride + (i + 1)];
            let u01 = u_nodes[(j + 1) * stride + i];
            let u11 = u_nodes[(j + 1) * stride + (i + 1)];

            let du_dx = 0.5 * ((u10 + u11) - (u00 + u01)) * inv_dx;
            let du_dy = 0.5 * ((u01 + u11) - (u00 + u10)) * inv_dy;

            b_eff.data[idx][0] += factor_xy * du_dx;
            b_eff.data[idx][1] += factor_xy * du_dy;
            b_eff.data[idx][2] += factor_z * m_data[idx][2];
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostic: expose intermediate results for validation
// ---------------------------------------------------------------------------

/// Diagnostic struct for inspecting intermediate results.
#[derive(Debug)]
pub struct DstDemagDiagnostics {
    /// w potential at interior nodes (mx × my).
    pub w_interior: Vec<f64>,
    /// Source density g at boundary nodes.
    pub g_boundary: Vec<f64>,
    /// v potential at boundary nodes (from single-layer integral).
    pub v_boundary: Vec<f64>,
    /// Composed U = w + v at all (nx+1)×(ny+1) nodes.
    pub u_nodes: Vec<f64>,
    /// Timing: [w_solve_ms, boundary_integral_ms, v_solve_ms, gradient_ms, total_ms]
    pub timings_ms: [f64; 5],
}

/// Run the DST demag pipeline with diagnostics (not cached, allocates fresh each time).
pub fn solve_with_diagnostics(
    grid: &Grid2D,
    m: &VectorField2D,
    mat: &Material,
    b_out: &mut VectorField2D,
) -> DstDemagDiagnostics {
    let nx = grid.nx;
    let ny = grid.ny;
    let dx = grid.dx;
    let dy = grid.dy;
    let ms = mat.ms;
    let mx = nx - 1;
    let my = ny - 1;
    let stride = nx + 1;
    let nn = (nx + 1) * (ny + 1);

    let t_total = Instant::now();

    // Phase 1: w-solve
    let t1 = Instant::now();
    let mut poisson = DstPoisson2D::new(nx, ny, dx, dy);
    let mut rhs = vec![0.0f64; mx * my];
    build_divergence_at_nodes(nx, ny, dx, dy, &m.data, ms, &mut rhs);
    poisson.solve_homogeneous(&mut rhs);
    let w_interior = rhs.clone();

    let mut w_nodes = vec![0.0f64; nn];
    for q in 0..my {
        let j = q + 1;
        for p in 0..mx {
            let i = p + 1;
            w_nodes[j * stride + i] = w_interior[q * mx + p];
        }
    }
    let t_w = t1.elapsed().as_secs_f64() * 1e3;

    // Phase 2: boundary integral
    let t2 = Instant::now();
    let boundary = enumerate_boundary_nodes(nx, ny, dx, dy);
    let g = compute_source_density(nx, ny, dx, dy, &m.data, ms, &w_nodes, &boundary);
    let v_bdy = evaluate_single_layer_potential(&boundary, &g);
    let t_bi = t2.elapsed().as_secs_f64() * 1e3;

    // Phase 3: v-solve
    let t3 = Instant::now();
    let mut bc_full = vec![0.0f64; nn];
    set_boundary_values_on_node_grid(&boundary, &v_bdy, &mut bc_full);
    let mut v_interior = vec![0.0f64; mx * my];
    poisson.solve_inhomogeneous_laplace(&bc_full, &mut v_interior);
    let t_v = t3.elapsed().as_secs_f64() * 1e3;

    // Phase 4: compose
    let mut u_nodes = bc_full.clone(); // boundary of U = v boundary
    for q in 0..my {
        let j = q + 1;
        for p in 0..mx {
            let i = p + 1;
            let nidx = j * stride + i;
            u_nodes[nidx] = w_nodes[nidx] + v_interior[q * mx + p];
        }
    }

    // Phase 5: gradient
    let t4 = Instant::now();
    b_out.set_uniform(0.0, 0.0, 0.0);
    add_demag_to_beff(nx, ny, dx, dy, ms, &u_nodes, &m.data, b_out);
    let t_grad = t4.elapsed().as_secs_f64() * 1e3;

    let t_total_ms = t_total.elapsed().as_secs_f64() * 1e3;

    DstDemagDiagnostics {
        w_interior,
        g_boundary: g,
        v_boundary: v_bdy,
        u_nodes,
        timings_ms: [t_w, t_bi, t_v, t_grad, t_total_ms],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform magnetisation (all +x): ∇·M = 0 everywhere.
    /// Check no NaN and that the field is finite.
    #[test]
    fn uniform_mx_no_nan() {
        let grid = Grid2D::new(8, 8, 5e-9, 5e-9, 1e-9);
        let mut m = VectorField2D::new(grid);
        for v in &mut m.data {
            *v = [1.0, 0.0, 0.0];
        }

        let mat = Material {
            ms: 8e5,
            a_ex: 0.0,
            k_u: 0.0,
            easy_axis: [0.0, 0.0, 1.0],
            dmi: None,
            demag: true,
            demag_method: crate::params::DemagMethod::FftUniform,
        };

        let mut b = VectorField2D::new(grid);
        compute_demag_field_poisson_dst(&grid, &m, &mut b, &mat);

        for (idx, v) in b.data.iter().enumerate() {
            assert!(
                v[0].is_finite() && v[1].is_finite() && v[2].is_finite(),
                "NaN/Inf at cell {}: {:?}",
                idx, v
            );
        }
    }

    /// Uniform Mz: in thin film, B_z = -μ₀ Ms at interior. In-plane components
    /// should be small for interior cells.
    #[test]
    fn uniform_mz_thin_film() {
        let grid = Grid2D::new(16, 16, 5e-9, 5e-9, 1e-9);
        let mut m = VectorField2D::new(grid);
        for v in &mut m.data {
            *v = [0.0, 0.0, 1.0];
        }

        let ms = 8e5;
        let mat = Material {
            ms,
            a_ex: 0.0,
            k_u: 0.0,
            easy_axis: [0.0, 0.0, 1.0],
            dmi: None,
            demag: true,
            demag_method: crate::params::DemagMethod::FftUniform,
        };

        let mut b = VectorField2D::new(grid);
        compute_demag_field_poisson_dst(&grid, &m, &mut b, &mat);

        // Central cell: Bz should be approximately -μ₀ Ms
        let cx = 8;
        let cy = 8;
        let idx = cy * 16 + cx;
        let expected_bz = -MU0 * ms;
        let rel_err = ((b.data[idx][2] - expected_bz) / expected_bz).abs();

        assert!(
            rel_err < 0.15,
            "Bz at center: got {:.6e}, expected {:.6e}, rel_err={:.4}",
            b.data[idx][2], expected_bz, rel_err
        );
    }
}