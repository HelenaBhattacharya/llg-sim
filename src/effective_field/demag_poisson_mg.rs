// src/effective_field/demag_poisson_mg.rs
//
// Demagnetising field via 2D multigrid with Fredkin-Koehler (φ = w + v)
// decomposition and boundary integral for open boundary conditions.
//
// Physics (SI):
//   ∇²φ = ∇·M   in the domain
//   φ → 0        as |r| → ∞  (open boundary conditions)
//   H = −∇φ      (demagnetising field)
//   B = μ₀ H     (Tesla)
//
// Fredkin-Koehler decomposition:
//   φ = w + v
//   ∇²w = ∇·M   with  w = 0 on ∂V       (homogeneous Dirichlet)
//   ∇²v = 0      with  v = v_bdy on ∂V   (inhomogeneous Dirichlet)
//
// where v_bdy is computed from the single-layer boundary integral:
//   v(x) = ∫_∂V N(x−y) g(y) dσ(y)
//   g(y) = −M·ν̂(y) + ∂w/∂ν(y)
//
// Architecture:
//   The MG solver operates on a CELL-CENTRED grid (nx × ny), which coarsens
//   cleanly for power-of-2 grids (64 → 32 → 16 → 8).  The boundary integral
//   operates on the NODE grid ((nx+1) × (ny+1)), with nodes at positions
//   (i·dx, j·dy) sitting on the physical domain boundary ∂V.
//
//   Two bridging operations connect the representations:
//     cells_to_nodes:  w on cell centres → w at nodes (for ∂w/∂ν in source density)
//     boundary_v_to_cell_bc:  v at boundary nodes → v at perimeter cells (for v-solve BC)
//
// Out-of-plane component Bz:
//   The Fredkin-Koehler decomposition solves for the in-plane scalar potential.
//   Bz requires the full Kzz(r) kernel — not just the self-term Nzz(0,0,0).
//
//   We compute Bz via FFT convolution on the L0 grid:
//     Bz = IFFT[ K̃zz · FFT(Ms · mz) ]
//   using the exact same Newell tensor (3D prism face-charge integration) that
//   the FFT solver uses.  This gives bit-identical Bz to the full FFT solver,
//   eliminating the 76% Bz error that the self-term-only approach had.
//
//   For AMR: the Kzz convolution runs on L0 and is interpolated to patches
//   (Bz is a smooth field, so interpolation error is negligible).
//
//   Controlled by LLG_DEMAG_MG_KZZ_MODE:
//     "fft"  (default) — full Kzz FFT convolution (exact Bz)
//     "self"           — Nzz self-term only (fast but 76% Bz error; for debugging)
//
// Grid convention:
//   Physical domain: nx × ny cells, centres at ((i+0.5)*dx, (j+0.5)*dy).
//   MG solver: nx × ny cells; perimeter ring = Dirichlet BC.
//   Node grid: (nx+1) × (ny+1); boundary nodes on ∂V.

use crate::grid::Grid2D;
use crate::params::{MU0, Material};
use crate::vector_field::VectorField2D;

use super::boundary_integral_2d;
use super::demag_fft_uniform;
use super::mg_config::PoissonMG2DConfig;
use super::mg_diagnostics::{self, PhaseTimer, TIMING};
use super::mg_kernels;
use super::mg_solver::PoissonMG2D;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Kzz mode selection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KzzMode {
    /// Full Kzz FFT convolution — exact Bz, identical to FFT solver.
    Fft,
    /// Nzz self-term only — fast but ~76% Bz error.  For debugging/comparison.
    SelfTermOnly,
}

fn kzz_mode() -> KzzMode {
    static MODE: OnceLock<KzzMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("LLG_DEMAG_MG_KZZ_MODE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "self" | "nzz_only" | "self_term" => {
                eprintln!("[demag_mg2d] Kzz mode: self-term only (LLG_DEMAG_MG_KZZ_MODE=self)");
                KzzMode::SelfTermOnly
            }
            _ => {
                // Default: FFT convolution
                KzzMode::Fft
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Cached Nzz value (for self-term fallback mode)
// ---------------------------------------------------------------------------

static NZZ_CACHE: OnceLock<f64> = OnceLock::new();

fn cached_nzz(dx: f64, dy: f64, dz: f64) -> f64 {
    *NZZ_CACHE.get_or_init(|| {
        let nzz = mg_kernels::newell_nzz_self(dx, dy, dz);
        eprintln!(
            "[demag_mg2d] Nzz self-demag factor = {:.6} (dx={:.2e}, dy={:.2e}, dz={:.2e})",
            nzz, dx, dy, dz,
        );
        nzz
    })
}

// ===========================================================================
// Kzz FFT Convolver
// ===========================================================================
//
// Computes Bz = IFFT[ K̃zz · FFT(Ms·mz) ] using the exact Newell Kzz tensor.
//
// This is a lightweight single-component convolver — it only handles Kzz,
// not the full 4-component tensor.  The in-plane components (Bx, By) are
// computed by the FK/MG pipeline above.
//
// Cost: 2 FFTs (forward mz + inverse bz) ≈ 100μs for 256×256.

struct KzzConvolver {
    nx: usize,
    ny: usize,
    px: usize,   // padded x = 2*nx (open BC)
    py: usize,   // padded y = 2*ny (open BC)

    /// Kzz kernel in k-space (length px*py).
    kzz_k: Vec<Complex<f64>>,

    /// Scratch buffers for forward/inverse FFT (length px*py each).
    mz_buf: Vec<Complex<f64>>,
    bz_buf: Vec<Complex<f64>>,
    fft_tmp: Vec<Complex<f64>>,

    /// FFT plans.
    fft_x_fwd: Arc<dyn rustfft::Fft<f64>>,
    fft_x_inv: Arc<dyn rustfft::Fft<f64>>,
    fft_y_fwd: Arc<dyn rustfft::Fft<f64>>,
    fft_y_inv: Arc<dyn rustfft::Fft<f64>>,
}

impl KzzConvolver {
    /// Build the Kzz convolver for a given grid.
    ///
    /// Constructs the real-space Kzz kernel at all displacements, FFTs it to
    /// k-space, and creates FFT plans for the forward/inverse transforms.
    fn new(grid: &Grid2D) -> Self {
        let nx = grid.nx;
        let ny = grid.ny;
        let px = 2 * nx;  // open BC: zero-pad to 2N
        let py = 2 * ny;
        let n_pad = px * py;

        let zero = Complex::new(0.0, 0.0);

        // Build real-space Kzz kernel with MuMax parity enforcement.
        // Only Kzz needed — we discard Kxx, Kxy, Kyy.
        let mut kzz_r = vec![zero; n_pad];
        let accuracy = demag_fft_uniform::DEMAG_ACCURACY;

        let nx_i = nx as isize;
        let ny_i = ny as isize;

        // Fill base quadrant sx>=0, sy>=0, then reflect with even parity.
        // Kzz is even in both x and y (diagonal tensor component).
        for sy in 0..ny_i {
            for sx in 0..nx_i {
                let (_kxx, _kxy, _kyy, k_zz) =
                    demag_fft_uniform::kernel_2d_components_mumax_like(
                        grid.dx, grid.dy, grid.dz, sx, sy, accuracy,
                    );

                // Reflect into all four quadrants (Kzz is even-even).
                for &sx_sign in &[1isize, -1isize] {
                    for &sy_sign in &[1isize, -1isize] {
                        let sx_s = sx_sign * sx;
                        let sy_s = sy_sign * sy;
                        let ix = demag_fft_uniform::wrap_index(sx_s, px);
                        let iy = demag_fft_uniform::wrap_index(sy_s, py);
                        kzz_r[iy * px + ix].re = k_zz;
                    }
                }
            }
        }

        // FFT plans
        let mut planner = FftPlanner::<f64>::new();
        let fft_x_fwd = planner.plan_fft_forward(px);
        let fft_x_inv = planner.plan_fft_inverse(px);
        let fft_y_fwd = planner.plan_fft_forward(py);
        let fft_y_inv = planner.plan_fft_inverse(py);

        // FFT kernel to k-space
        let mut fft_tmp = vec![zero; n_pad];
        demag_fft_uniform::fft2_forward_in_place(
            &mut kzz_r, px, py, &fft_x_fwd, &fft_y_fwd, &mut fft_tmp,
        );

        eprintln!(
            "[demag_mg2d] KzzConvolver built: {}x{} -> {}x{} padded, \
             |Kzz(0)|={:.6e} T/(A/m)",
            nx, ny, px, py, kzz_r[0].re,
        );

        Self {
            nx,
            ny,
            px,
            py,
            kzz_k: kzz_r,  // now in k-space after the FFT
            mz_buf: vec![zero; n_pad],
            bz_buf: vec![zero; n_pad],
            fft_tmp,
            fft_x_fwd,
            fft_x_inv,
            fft_y_fwd,
            fft_y_inv,
        }
    }

    /// Compute Bz = Kzz * (Ms·mz) via FFT convolution.
    ///
    /// `m_data`: cell-centred unit magnetisation, row-major (nx*ny).
    /// `ms`: saturation magnetisation (A/m).
    /// `bz_out`: output field at cell centres (nx*ny), Bz ADDED to [2] component.
    fn convolve_add(&mut self, m_data: &[[f64; 3]], ms: f64, bz_out: &mut [[f64; 3]]) {
        let nx = self.nx;
        let ny = self.ny;
        let px = self.px;
        debug_assert_eq!(m_data.len(), nx * ny);
        debug_assert_eq!(bz_out.len(), nx * ny);

        let zero = Complex::new(0.0, 0.0);

        // 1. Pack Ms*mz into padded buffer (top-left = physical, rest = zero)
        self.mz_buf.fill(zero);
        for j in 0..ny {
            for i in 0..nx {
                self.mz_buf[j * px + i].re = ms * m_data[j * nx + i][2];
            }
        }

        // 2. Forward FFT
        demag_fft_uniform::fft2_forward_in_place(
            &mut self.mz_buf, px, self.py,
            &self.fft_x_fwd, &self.fft_y_fwd, &mut self.fft_tmp,
        );

        // 3. Pointwise multiply: Bz_k = Kzz_k * Mz_k
        for idx in 0..self.kzz_k.len() {
            self.bz_buf[idx] = self.kzz_k[idx] * self.mz_buf[idx];
        }

        // 4. Inverse FFT
        demag_fft_uniform::fft2_inverse_in_place(
            &mut self.bz_buf, px, self.py,
            &self.fft_x_inv, &self.fft_y_inv, &mut self.fft_tmp,
        );

        // 5. Extract physical region → add to Bz
        for j in 0..ny {
            for i in 0..nx {
                bz_out[j * nx + i][2] += self.bz_buf[j * px + i].re;
            }
        }
    }
}

// ===========================================================================
// Cell ↔ Node bridging
// ===========================================================================

/// Interpolate cell-centred phi (nx × ny) to node grid ((nx+1) × (ny+1)).
///
/// For interior node (i,j), average the up-to-4 surrounding cells:
///   (i-1,j-1), (i,j-1), (i-1,j), (i,j)  [only those that exist].
///
/// For boundary nodes on ∂V, the surrounding cells are only the ones that
/// exist (corner nodes have 1 cell, edge nodes have 2 cells).
///
/// For the w-solve (perimeter cells = 0), boundary nodes naturally get values
/// close to 0 — the perimeter cells contribute 0, and only interior cells
/// adjacent to the boundary give a small contribution through averaging.
fn cells_to_nodes(
    phi_cells: &[f64],
    nx: usize,
    ny: usize,
    phi_nodes: &mut [f64],
) {
    let nnx = nx + 1;
    let nny = ny + 1;
    debug_assert_eq!(phi_cells.len(), nx * ny);
    debug_assert_eq!(phi_nodes.len(), nnx * nny);

    for j in 0..nny {
        for i in 0..nnx {
            let mut sum = 0.0f64;
            let mut count = 0u32;

            // Cell (i-1, j-1)
            if i > 0 && j > 0 {
                sum += phi_cells[(j - 1) * nx + (i - 1)];
                count += 1;
            }
            // Cell (i, j-1)
            if i < nx && j > 0 {
                sum += phi_cells[(j - 1) * nx + i];
                count += 1;
            }
            // Cell (i-1, j)
            if i > 0 && j < ny {
                sum += phi_cells[j * nx + (i - 1)];
                count += 1;
            }
            // Cell (i, j)
            if i < nx && j < ny {
                sum += phi_cells[j * nx + i];
                count += 1;
            }

            phi_nodes[j * nnx + i] = if count > 0 { sum / count as f64 } else { 0.0 };
        }
    }
}

/// Map boundary-integral v-values (at boundary nodes) to cell-centred Dirichlet
/// values for perimeter cells.
///
/// For each perimeter cell, interpolate v along its nearest boundary edge at
/// the cell centre's projected position:
///   Bottom (j=0):  v_cell = 0.5·(v_node(i,0) + v_node(i+1,0))
///   Top (j=ny-1):  v_cell = 0.5·(v_node(i,ny) + v_node(i+1,ny))
///   Left (i=0):    v_cell = 0.5·(v_node(0,j) + v_node(0,j+1))
///   Right (i=nx-1): v_cell = 0.5·(v_node(nx,j) + v_node(nx,j+1))
///
/// Corner cells get the average of both edge interpolations.
fn boundary_v_to_cell_bc(
    v_node_grid: &[f64],
    nx: usize,
    ny: usize,
    bc_cells: &mut [f64],
) {
    let nnx = nx + 1;
    debug_assert_eq!(v_node_grid.len(), nnx * (ny + 1));
    debug_assert_eq!(bc_cells.len(), nx * ny);

    bc_cells.fill(0.0);

    // Bottom row: j = 0
    for i in 0..nx {
        bc_cells[i] = 0.5 * (v_node_grid[i] + v_node_grid[i + 1]);
    }

    // Top row: j = ny - 1
    for i in 0..nx {
        bc_cells[(ny - 1) * nx + i] =
            0.5 * (v_node_grid[ny * nnx + i] + v_node_grid[ny * nnx + (i + 1)]);
    }

    // Left column: i = 0 (interior rows only — corners already set)
    for j in 1..ny - 1 {
        bc_cells[j * nx] = 0.5 * (v_node_grid[j * nnx] + v_node_grid[(j + 1) * nnx]);
    }

    // Right column: i = nx - 1 (interior rows only)
    for j in 1..ny - 1 {
        bc_cells[j * nx + (nx - 1)] =
            0.5 * (v_node_grid[j * nnx + nx] + v_node_grid[(j + 1) * nnx + nx]);
    }

    // Fix corner cells: average both edge interpolations.
    // Bottom-left (0,0):
    {
        let v_bot = 0.5 * (v_node_grid[0] + v_node_grid[1]);
        let v_lft = 0.5 * (v_node_grid[0] + v_node_grid[nnx]);
        bc_cells[0] = 0.5 * (v_bot + v_lft);
    }
    // Bottom-right (nx-1,0):
    {
        let v_bot = 0.5 * (v_node_grid[nx - 1] + v_node_grid[nx]);
        let v_rgt = 0.5 * (v_node_grid[nx] + v_node_grid[nnx + nx]);
        bc_cells[nx - 1] = 0.5 * (v_bot + v_rgt);
    }
    // Top-left (0,ny-1):
    {
        let v_top = 0.5 * (v_node_grid[ny * nnx] + v_node_grid[ny * nnx + 1]);
        let v_lft = 0.5 * (v_node_grid[(ny - 1) * nnx] + v_node_grid[ny * nnx]);
        bc_cells[(ny - 1) * nx] = 0.5 * (v_top + v_lft);
    }
    // Top-right (nx-1,ny-1):
    {
        let v_top = 0.5 * (v_node_grid[ny * nnx + nx - 1] + v_node_grid[ny * nnx + nx]);
        let v_rgt = 0.5 * (v_node_grid[(ny - 1) * nnx + nx] + v_node_grid[ny * nnx + nx]);
        bc_cells[(ny - 1) * nx + (nx - 1)] = 0.5 * (v_top + v_rgt);
    }
}

// ===========================================================================
// Solver state (cached between timesteps)
// ===========================================================================

struct DemagMG2DState {
    grid: Grid2D,

    /// MG solvers for w and v — operate on the nx × ny CELL grid.
    w_solver: PoissonMG2D,
    v_solver: PoissonMG2D,

    /// Cell-grid arrays (nx × ny, reused across calls).
    rhs_cells: Vec<f64>,
    phi_cells: Vec<f64>,

    /// Node-grid arrays ((nx+1) × (ny+1), for boundary integral bridging).
    w_nodes: Vec<f64>,
    v_node_bc: Vec<f64>,
    v_cell_bc: Vec<f64>,

    /// Pre-computed boundary node list (on node grid).
    boundary: Vec<boundary_integral_2d::BoundaryNode>,

    /// Kzz FFT convolver for exact Bz (None if using self-term-only mode).
    kzz_convolver: Option<KzzConvolver>,
}

impl DemagMG2DState {
    fn new(grid: Grid2D, cfg: PoissonMG2DConfig) -> Self {
        let nx = grid.nx;
        let ny = grid.ny;
        let nc = nx * ny;
        let nn = (nx + 1) * (ny + 1);

        let w_solver = PoissonMG2D::new(nx, ny, grid.dx, grid.dy, cfg);
        let v_solver = PoissonMG2D::new(nx, ny, grid.dx, grid.dy, cfg);

        let boundary = boundary_integral_2d::enumerate_boundary_nodes(nx, ny, grid.dx, grid.dy);

        // Build Kzz convolver unless self-term-only mode is requested.
        let kzz_convolver = match kzz_mode() {
            KzzMode::Fft => Some(KzzConvolver::new(&grid)),
            KzzMode::SelfTermOnly => {
                // Pre-compute Nzz for the self-term fallback.
                let _ = cached_nzz(grid.dx, grid.dy, grid.dz);
                None
            }
        };

        static LOG_ONCE: OnceLock<()> = OnceLock::new();
        LOG_ONCE.get_or_init(|| {
            eprintln!(
                "[demag_mg2d] Fredkin-Koehler 2D: cells={}x{}, nodes={}x{}, \
                 boundary_nodes={}, dx={:.2e}, dy={:.2e}, dz={:.2e}, kzz_mode={:?}",
                nx, ny, nx + 1, ny + 1, boundary.len(),
                grid.dx, grid.dy, grid.dz, kzz_mode(),
            );
        });

        Self {
            grid,
            w_solver,
            v_solver,
            rhs_cells: vec![0.0; nc],
            phi_cells: vec![0.0; nc],
            w_nodes: vec![0.0; nn],
            v_node_bc: vec![0.0; nn],
            v_cell_bc: vec![0.0; nc],
            boundary,
            kzz_convolver,
        }
    }

    fn same_structure(&self, grid: &Grid2D) -> bool {
        self.grid.nx == grid.nx
            && self.grid.ny == grid.ny
            && self.grid.dx == grid.dx
            && self.grid.dy == grid.dy
    }

    // -----------------------------------------------------------------------
    // Step 1: Compute ∇·M on cell grid
    // -----------------------------------------------------------------------

    fn compute_rhs(&mut self, m: &VectorField2D, ms: f64) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        // compute_div_m_2d writes ∇·m (unit-vector divergence) into rhs_cells.
        mg_kernels::compute_div_m_2d(
            &m.data, nx, ny, self.grid.dx, self.grid.dy, &mut self.rhs_cells,
        );

        // Scale by Ms: RHS = Ms · ∇·m
        for v in &mut self.rhs_cells {
            *v *= ms;
        }
    }

    // -----------------------------------------------------------------------
    // Step 2: w-solve (∇²w = ∇·M, w = 0 on perimeter cells)
    // -----------------------------------------------------------------------

    fn solve_w(&mut self) {
        let nc = self.grid.nx * self.grid.ny;
        let bc_zero = vec![0.0f64; nc];
        self.w_solver.solve_with_rhs_and_bc(&self.rhs_cells, &bc_zero);
    }

    // -----------------------------------------------------------------------
    // Step 3: Bridge w from cells to nodes
    // -----------------------------------------------------------------------

    fn bridge_w_to_nodes(&mut self) {
        cells_to_nodes(
            self.w_solver.phi(),
            self.grid.nx,
            self.grid.ny,
            &mut self.w_nodes,
        );
    }

    // -----------------------------------------------------------------------
    // Step 4: Boundary integral (source density + single-layer potential)
    // -----------------------------------------------------------------------

    fn compute_boundary_v(&mut self, m: &VectorField2D, ms: f64) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;

        let g = boundary_integral_2d::compute_source_density(
            nx, ny, self.grid.dx, self.grid.dy,
            &m.data, ms, &self.w_nodes, &self.boundary,
        );

        let v_bdy = boundary_integral_2d::evaluate_single_layer_potential(&self.boundary, &g);

        self.v_node_bc.fill(0.0);
        boundary_integral_2d::set_boundary_values_on_node_grid(
            &self.boundary, &v_bdy, &mut self.v_node_bc,
        );
    }

    // -----------------------------------------------------------------------
    // Step 5: Bridge v from boundary nodes to cell-grid BC
    // -----------------------------------------------------------------------

    fn bridge_v_to_cell_bc(&mut self) {
        boundary_v_to_cell_bc(
            &self.v_node_bc, self.grid.nx, self.grid.ny, &mut self.v_cell_bc,
        );
    }

    // -----------------------------------------------------------------------
    // Step 6: v-solve (∇²v = 0, v = v_bdy on perimeter cells)
    // -----------------------------------------------------------------------

    fn solve_v(&mut self) {
        let nc = self.grid.nx * self.grid.ny;
        let rhs_zero = vec![0.0f64; nc];
        self.v_solver.solve_with_rhs_and_bc(&rhs_zero, &self.v_cell_bc);
    }

    // -----------------------------------------------------------------------
    // Step 7: Combine φ = w + v on cell grid
    // -----------------------------------------------------------------------

    fn combine_phi(&mut self) {
        let w = self.w_solver.phi();
        let v = self.v_solver.phi();
        for (i, phi) in self.phi_cells.iter_mut().enumerate() {
            *phi = w[i] + v[i];
        }
    }

    // -----------------------------------------------------------------------
    // Step 8: Extract gradient → B_demag at cell centres
    // -----------------------------------------------------------------------

    fn extract_gradient(&self, m: &VectorField2D, b_eff: &mut VectorField2D) {
        mg_kernels::extract_gradient_2d(
            &self.phi_cells,
            self.grid.nx, self.grid.ny,
            self.grid.dx, self.grid.dy,
            MU0,
            &mut b_eff.data,
            Some(&m.data),
        );
    }

    // -----------------------------------------------------------------------
    // Step 9: Bz via Kzz FFT convolution (or Nzz self-term fallback)
    // -----------------------------------------------------------------------

    fn apply_kzz_bz(&mut self, m: &VectorField2D, b_eff: &mut VectorField2D, ms: f64) {
        match &mut self.kzz_convolver {
            Some(conv) => {
                // Full Kzz FFT convolution — exact Bz.
                conv.convolve_add(&m.data, ms, &mut b_eff.data);
            }
            None => {
                // Nzz self-term fallback.
                let nzz = cached_nzz(self.grid.dx, self.grid.dy, self.grid.dz);
                let coeff = -MU0 * nzz * ms;
                let n = self.grid.nx * self.grid.ny;
                for idx in 0..n {
                    b_eff.data[idx][2] += coeff * m.data[idx][2];
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Full pipeline
    // -----------------------------------------------------------------------

    fn add_field(&mut self, m: &VectorField2D, b_eff: &mut VectorField2D, mat: &Material) {
        let ms = mat.ms;
        let do_timing = mg_diagnostics::timing_enabled();
        let total_start = if do_timing { Some(std::time::Instant::now()) } else { None };

        // Step 1: ∇·M at cell centres
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.div_ns));
            self.compute_rhs(m, ms);
        }

        // Step 2: w-solve
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.w_solve_ns));
            self.solve_w();
        }

        // Steps 3-4: Bridge to nodes → boundary integral
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.boundary_integral_ns));
            self.bridge_w_to_nodes();
            self.compute_boundary_v(m, ms);
        }

        // Steps 5-6: Bridge to cell BC → v-solve
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.v_solve_ns));
            self.bridge_v_to_cell_bc();
            self.solve_v();
        }

        // Step 7: φ = w + v
        self.combine_phi();

        // Step 8: Gradient → B_demag (in-plane: Bx, By)
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.gradient_ns));
            self.extract_gradient(m, b_eff);
        }

        // Step 9: Bz via Kzz FFT convolution (or Nzz self-term fallback)
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.nzz_ns));
            self.apply_kzz_bz(m, b_eff, ms);
        }

        if let Some(start) = total_start {
            let elapsed = start.elapsed().as_nanos() as u64;
            TIMING.total_ns.fetch_add(elapsed, Ordering::Relaxed);
            let n = TIMING.call_count.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 100 == 0 {
                TIMING.print_summary();
            }
        }
    }


    /// Run the FK pipeline with a pre-computed RHS (for composite-grid enhanced-RHS).
    ///
    /// Identical to add_field() except step 1 (compute_rhs) is replaced
    /// by copying the externally provided RHS into self.rhs_cells.
    ///
    /// The magnetisation m is still needed for:
    ///   - The boundary integral source density (-M.nu term in step 4)
    ///   - The Kzz Bz convolution (step 9)
    ///   - The gradient extraction mag mask (step 8)
    fn add_field_with_external_rhs(
        &mut self,
        rhs_cells: &[f64],
        m: &VectorField2D,
        b_eff: &mut VectorField2D,
        mat: &Material,
    ) {
        let ms = mat.ms;
        let nc = self.grid.nx * self.grid.ny;
        debug_assert_eq!(rhs_cells.len(), nc);

        let do_timing = mg_diagnostics::timing_enabled();
        let total_start = if do_timing { Some(std::time::Instant::now()) } else { None };

        // Step 1: Use externally provided RHS (skip compute_rhs)
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.div_ns));
            self.rhs_cells.copy_from_slice(rhs_cells);
        }

        // Steps 2-9: identical to add_field()

        // Step 2: w-solve
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.w_solve_ns));
            self.solve_w();
        }

        // Steps 3-4: Bridge to nodes -> boundary integral
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.boundary_integral_ns));
            self.bridge_w_to_nodes();
            self.compute_boundary_v(m, ms);
        }

        // Steps 5-6: Bridge to cell BC -> v-solve
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.v_solve_ns));
            self.bridge_v_to_cell_bc();
            self.solve_v();
        }

        // Step 7: phi = w + v
        self.combine_phi();

        // Step 8: Gradient -> B_demag (in-plane: Bx, By)
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.gradient_ns));
            self.extract_gradient(m, b_eff);
        }

        // Step 9: Bz via Kzz FFT convolution (or Nzz self-term fallback)
        {
            let _t = do_timing.then(|| PhaseTimer::start(&TIMING.nzz_ns));
            self.apply_kzz_bz(m, b_eff, ms);
        }

        if let Some(start) = total_start {
            let elapsed = start.elapsed().as_nanos() as u64;
            TIMING.total_ns.fetch_add(elapsed, Ordering::Relaxed);
            let n = TIMING.call_count.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 100 == 0 {
                TIMING.print_summary();
            }
        }
    }

}

// ===========================================================================
// Module-level cache + public API
// ===========================================================================

static DEMAG_MG2D_CACHE: OnceLock<Mutex<Option<DemagMG2DState>>> = OnceLock::new();

/// Add the demagnetising field to `b_eff` using the 2D MG + boundary integral solver.
pub fn add_demag_field_poisson_mg(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    mat: &Material,
) {
    if !mat.demag { return; }

    let cfg = PoissonMG2DConfig::from_env();
    let cache = DEMAG_MG2D_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().expect("DEMAG_MG2D_CACHE mutex poisoned");

    let rebuild = match guard.as_ref() {
        Some(s) => !s.same_structure(grid),
        None => true,
    };
    if rebuild {
        *guard = Some(DemagMG2DState::new(*grid, cfg));
    }

    if let Some(state) = guard.as_mut() {
        state.add_field(m, b_eff, mat);
    }
}

/// Compute B_demag into `out` (overwrites).
pub fn compute_demag_field_poisson_mg(
    grid: &Grid2D,
    m: &VectorField2D,
    out: &mut VectorField2D,
    mat: &Material,
) {
    out.set_uniform(0.0, 0.0, 0.0);
    add_demag_field_poisson_mg(grid, m, out, mat);
}

/// Convenience for composite-grid integration.
pub fn compute_demag_on_grid(
    grid: &Grid2D,
    m: &VectorField2D,
    b_out: &mut VectorField2D,
    mat: &Material,
) {
    b_out.set_uniform(0.0, 0.0, 0.0);
    add_demag_field_poisson_mg(grid, m, b_out, mat);
}

/// Solve the FK pipeline with an externally provided RHS (enhanced for composite grid).
///
/// This is the entry point for the Garcia-Cervera enhanced-RHS algorithm.
/// The rhs_cells array (length nx*ny) should contain div(M) at cell centres,
/// possibly enhanced with fine-resolution divergence from AMR patches.
///
/// The magnetisation m is still needed for the boundary integral (M.nu source
/// term) and the Kzz Bz convolution.
///
/// b_out is overwritten with the computed B_demag field.
pub fn solve_fk_with_external_rhs(
    grid: &Grid2D,
    rhs_cells: &[f64],
    m: &VectorField2D,
    b_out: &mut VectorField2D,
    mat: &Material,
) {
    b_out.set_uniform(0.0, 0.0, 0.0);

    if !mat.demag {
        return;
    }

    let cfg = PoissonMG2DConfig::from_env();
    let cache = DEMAG_MG2D_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().expect("DEMAG_MG2D_CACHE mutex poisoned");

    let rebuild = match guard.as_ref() {
        Some(s) => !s.same_structure(grid),
        None => true,
    };
    if rebuild {
        *guard = Some(DemagMG2DState::new(*grid, cfg));
    }

    if let Some(state) = guard.as_mut() {
        state.add_field_with_external_rhs(rhs_cells, m, b_out, mat);
    }
}
