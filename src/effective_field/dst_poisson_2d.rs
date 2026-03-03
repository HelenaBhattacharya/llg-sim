// src/effective_field/dst_poisson_2d.rs
//
// 2D Poisson solver using DST-I (Discrete Sine Transform Type I) via FFT.
//
// Solves  ∇²u = f  on a rectangular domain [0, Lx] × [0, Ly]
// with Dirichlet boundary conditions.
//
// Node-centered grid:
//   Interior nodes at (i*hx, j*hy) for i=1..mx, j=1..my
//   where mx = nx-1 interior nodes in x (nx cells → nx+1 nodes, 2 boundary = nx-1 interior)
//         my = ny-1 interior nodes in y
//   hx = Lx/nx = dx (cell spacing)
//   hy = Ly/ny = dy
//
// The DST-I exactly diagonalises the 5-point Laplacian with homogeneous Dirichlet BCs.
// For inhomogeneous Dirichlet, boundary contributions are folded into the RHS.
//
// DST-I of length N is computed via FFT of length 2(N+1) using odd-symmetric embedding.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// DST-I via FFT
// ---------------------------------------------------------------------------

/// Compute DST-I of `x[0..n-1]` in-place, overwriting `x`.
///
/// DST-I definition:
///   X[k] = Σ_{j=0}^{N-1} x[j] * sin(π(j+1)(k+1)/(N+1))   for k=0..N-1
///
/// Implemented via FFT of length M = 2(N+1):
///   Embed x in odd-symmetric sequence z, FFT(z), extract imaginary parts.
fn dst1_inplace(x: &mut [f64], fft: &Arc<dyn Fft<f64>>, scratch: &mut [Complex<f64>]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let m = 2 * (n + 1); // FFT length
    debug_assert!(scratch.len() >= m);

    // Build odd-symmetric embedding:
    //   z[0] = 0
    //   z[k] = x[k-1]           for k = 1..N
    //   z[N+1] = 0
    //   z[N+1+k] = -x[N-k]     for k = 1..N
    let z = &mut scratch[..m];
    z[0] = Complex::new(0.0, 0.0);
    for k in 0..n {
        z[k + 1] = Complex::new(x[k], 0.0);
    }
    z[n + 1] = Complex::new(0.0, 0.0);
    for k in 0..n {
        z[n + 2 + k] = Complex::new(-x[n - 1 - k], 0.0);
    }

    // Forward FFT
    fft.process(z);

    // Extract DST-I coefficients: X[k] = -Im(Z[k+1]) / 2
    // The factor of 2 arises because the odd-symmetric embedding of length 2(N+1)
    // distributes energy into both halves of the spectrum.
    for k in 0..n {
        x[k] = -z[k + 1].im * 0.5;
    }
}

/// Inverse DST-I: same as forward DST-I scaled by 2/(N+1).
///
/// Since DST-I is its own inverse up to this factor:
///   x[j] = (2/(N+1)) Σ_{k=0}^{N-1} X[k] * sin(π(j+1)(k+1)/(N+1))
fn idst1_inplace(x: &mut [f64], fft: &Arc<dyn Fft<f64>>, scratch: &mut [Complex<f64>]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    dst1_inplace(x, fft, scratch);
    let scale = 2.0 / (n + 1) as f64;
    for v in x.iter_mut() {
        *v *= scale;
    }
}

// ---------------------------------------------------------------------------
// 2D Poisson solver
// ---------------------------------------------------------------------------

/// Cached DST Poisson solver for a fixed grid size.
///
/// Solves ∇²u = f on interior nodes of an nx×ny cell grid
/// (i.e. (nx-1)×(ny-1) interior node unknowns).
pub struct DstPoisson2D {
    /// Number of cells in x, y.
    pub nx: usize,
    pub ny: usize,

    /// Cell spacing.
    pub dx: f64,
    pub dy: f64,

    /// Number of interior nodes: mx = nx-1, my = ny-1.
    mx: usize,
    my: usize,

    /// Precomputed eigenvalues λ(p,q) for p=0..mx-1, q=0..my-1.
    /// λ(p,q) = (2/dx²)(cos(π(p+1)/nx) - 1) + (2/dy²)(cos(π(q+1)/ny) - 1)
    eigenvalues: Vec<f64>,

    /// FFT plans for DST in x (length 2*nx) and y (length 2*ny).
    fft_x: Arc<dyn Fft<f64>>,
    fft_y: Arc<dyn Fft<f64>>,

    /// Scratch buffers.
    scratch_x: Vec<Complex<f64>>,
    scratch_y: Vec<Complex<f64>>,
    /// Row/column buffer for transposed operations.
    work: Vec<f64>,
}

impl DstPoisson2D {
    /// Create a new solver for an nx×ny cell grid with spacings dx, dy.
    ///
    /// The solver operates on (nx-1)×(ny-1) interior nodes.
    /// Requires nx ≥ 2 and ny ≥ 2 (at least 1 interior node in each direction).
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        assert!(nx >= 2, "DstPoisson2D requires nx >= 2, got {}", nx);
        assert!(ny >= 2, "DstPoisson2D requires ny >= 2, got {}", ny);

        let mx = nx - 1;
        let my = ny - 1;

        // Precompute eigenvalues of the 5-point Laplacian.
        let inv_dx2 = 1.0 / (dx * dx);
        let inv_dy2 = 1.0 / (dy * dy);
        let mut eigenvalues = vec![0.0f64; mx * my];
        for q in 0..my {
            let ly = 2.0 * inv_dy2 * ((PI * (q + 1) as f64 / ny as f64).cos() - 1.0);
            for p in 0..mx {
                let lx = 2.0 * inv_dx2 * ((PI * (p + 1) as f64 / nx as f64).cos() - 1.0);
                eigenvalues[q * mx + p] = lx + ly;
            }
        }

        // FFT plans
        let mut planner = FftPlanner::<f64>::new();
        let fft_x = planner.plan_fft_forward(2 * nx);
        let fft_y = planner.plan_fft_forward(2 * ny);

        let scratch_x = vec![Complex::new(0.0, 0.0); 2 * nx];
        let scratch_y = vec![Complex::new(0.0, 0.0); 2 * ny];
        let work = vec![0.0f64; mx.max(my)];

        Self {
            nx,
            ny,
            dx,
            dy,
            mx,
            my,
            eigenvalues,
            fft_x,
            fft_y,
            scratch_x,
            scratch_y,
            work,
        }
    }

    /// Number of interior nodes in x.
    #[inline]
    pub fn mx(&self) -> usize {
        self.mx
    }

    /// Number of interior nodes in y.
    #[inline]
    pub fn my(&self) -> usize {
        self.my
    }

    /// Index into the mx×my interior-node array.
    #[inline]
    pub fn node_idx(&self, p: usize, q: usize) -> usize {
        debug_assert!(p < self.mx && q < self.my);
        q * self.mx + p
    }

    /// Solve ∇²u = rhs with **homogeneous** Dirichlet BCs (u = 0 on ∂V).
    ///
    /// `rhs` must be length mx*my (interior nodes only).
    /// Solution is written in-place into `rhs`.
    pub fn solve_homogeneous(&mut self, rhs: &mut [f64]) {
        let mx = self.mx;
        let my = self.my;
        assert_eq!(rhs.len(), mx * my);

        // Forward DST-I along rows (x-direction)
        for q in 0..my {
            let row = &mut rhs[q * mx..(q + 1) * mx];
            dst1_inplace(row, &self.fft_x, &mut self.scratch_x);
        }

        // Forward DST-I along columns (y-direction)
        for p in 0..mx {
            self.work.resize(my, 0.0);
            for q in 0..my {
                self.work[q] = rhs[q * mx + p];
            }
            dst1_inplace(&mut self.work[..my], &self.fft_y, &mut self.scratch_y);
            for q in 0..my {
                rhs[q * mx + p] = self.work[q];
            }
        }

        // Divide by eigenvalues: û(p,q) = f̂(p,q) / λ(p,q)
        for idx in 0..mx * my {
            let ev = self.eigenvalues[idx];
            if ev.abs() < 1e-30 {
                rhs[idx] = 0.0;
            } else {
                rhs[idx] /= ev;
            }
        }

        // Inverse DST-I along columns
        for p in 0..mx {
            self.work.resize(my, 0.0);
            for q in 0..my {
                self.work[q] = rhs[q * mx + p];
            }
            idst1_inplace(&mut self.work[..my], &self.fft_y, &mut self.scratch_y);
            for q in 0..my {
                rhs[q * mx + p] = self.work[q];
            }
        }

        // Inverse DST-I along rows
        for q in 0..my {
            let row = &mut rhs[q * mx..(q + 1) * mx];
            idst1_inplace(row, &self.fft_x, &mut self.scratch_x);
        }
    }

    /// Solve ∇²u = 0 with **inhomogeneous** Dirichlet BCs.
    ///
    /// Boundary values are provided on the (nx+1)×(ny+1) full node grid.
    /// Only the boundary nodes are read; interior values in `bc_nodes` are ignored.
    ///
    /// `bc_nodes` layout: node (i,j) at index j*(nx+1) + i, for i=0..nx, j=0..ny.
    ///
    /// The solution (interior nodes only) is written into `solution` (length mx*my).
    pub fn solve_inhomogeneous_laplace(
        &mut self,
        bc_nodes: &[f64],
        solution: &mut [f64],
    ) {
        let mx = self.mx;
        let my = self.my;
        let nx = self.nx;
        let ny = self.ny;
        assert_eq!(bc_nodes.len(), (nx + 1) * (ny + 1));
        assert_eq!(solution.len(), mx * my);

        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let inv_dy2 = 1.0 / (self.dy * self.dy);

        // Build effective RHS by moving boundary contributions.
        for q in 0..my {
            let j = q + 1; // node row in full grid
            for p in 0..mx {
                let i = p + 1; // node col in full grid
                let mut f = 0.0;

                if i == 1 {
                    f -= inv_dx2 * bc_nodes[j * (nx + 1) + 0];
                }
                if i == nx - 1 {
                    f -= inv_dx2 * bc_nodes[j * (nx + 1) + nx];
                }
                if j == 1 {
                    f -= inv_dy2 * bc_nodes[0 * (nx + 1) + i];
                }
                if j == ny - 1 {
                    f -= inv_dy2 * bc_nodes[ny * (nx + 1) + i];
                }

                solution[q * mx + p] = f;
            }
        }

        self.solve_homogeneous(solution);
    }

    /// Solve ∇²u = rhs with **inhomogeneous** Dirichlet BCs.
    ///
    /// Combines a nonzero RHS with nonzero boundary values.
    /// `rhs_interior` has length mx*my (values at interior nodes).
    /// `bc_nodes` has length (nx+1)*(ny+1).
    /// Solution is written into `rhs_interior` (in-place).
    pub fn solve_inhomogeneous(
        &mut self,
        rhs_interior: &mut [f64],
        bc_nodes: &[f64],
    ) {
        let mx = self.mx;
        let my = self.my;
        let nx = self.nx;
        let ny = self.ny;
        assert_eq!(rhs_interior.len(), mx * my);
        assert_eq!(bc_nodes.len(), (nx + 1) * (ny + 1));

        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let inv_dy2 = 1.0 / (self.dy * self.dy);

        // Fold boundary values into the RHS.
        for q in 0..my {
            let j = q + 1;
            for p in 0..mx {
                let i = p + 1;

                if i == 1 {
                    rhs_interior[q * mx + p] -= inv_dx2 * bc_nodes[j * (nx + 1) + 0];
                }
                if i == nx - 1 {
                    rhs_interior[q * mx + p] -= inv_dx2 * bc_nodes[j * (nx + 1) + nx];
                }
                if j == 1 {
                    rhs_interior[q * mx + p] -= inv_dy2 * bc_nodes[0 * (nx + 1) + i];
                }
                if j == ny - 1 {
                    rhs_interior[q * mx + p] -= inv_dy2 * bc_nodes[ny * (nx + 1) + i];
                }
            }
        }

        self.solve_homogeneous(rhs_interior);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dst1_matches_naive() {
        let n = 7;
        let x_orig: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
        let mut x = x_orig.clone();

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(2 * (n + 1));
        let mut scratch = vec![Complex::new(0.0, 0.0); 2 * (n + 1)];

        dst1_inplace(&mut x, &fft, &mut scratch);

        let mut x_naive = vec![0.0f64; n];
        for k in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += x_orig[j] * (PI * (j + 1) as f64 * (k + 1) as f64 / (n + 1) as f64).sin();
            }
            x_naive[k] = s;
        }

        for k in 0..n {
            assert!(
                (x[k] - x_naive[k]).abs() < 1e-10,
                "DST-I mismatch at k={}: got {}, expected {}",
                k, x[k], x_naive[k]
            );
        }
    }

    #[test]
    fn dst1_roundtrip() {
        let n = 15;
        let x_orig: Vec<f64> = (0..n).map(|i| (0.3 * i as f64).cos()).collect();
        let mut x = x_orig.clone();

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(2 * (n + 1));
        let mut scratch = vec![Complex::new(0.0, 0.0); 2 * (n + 1)];

        dst1_inplace(&mut x, &fft, &mut scratch);
        idst1_inplace(&mut x, &fft, &mut scratch);

        for i in 0..n {
            assert!(
                (x[i] - x_orig[i]).abs() < 1e-10,
                "Round-trip mismatch at i={}: got {}, expected {}",
                i, x[i], x_orig[i]
            );
        }
    }

    #[test]
    fn poisson_sine_mode() {
        let n = 32;
        let h = 1.0 / n as f64;
        let mx = n - 1;
        let my = n - 1;

        let mut solver = DstPoisson2D::new(n, n, h, h);

        let mut rhs = vec![0.0f64; mx * my];
        for q in 0..my {
            let j = q + 1;
            let y = j as f64 * h;
            for p in 0..mx {
                let i = p + 1;
                let x = i as f64 * h;
                rhs[q * mx + p] = -2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
            }
        }

        solver.solve_homogeneous(&mut rhs);

        let mut max_err = 0.0f64;
        for q in 0..my {
            let j = q + 1;
            let y = j as f64 * h;
            for p in 0..mx {
                let i = p + 1;
                let x = i as f64 * h;
                let exact = (PI * x).sin() * (PI * y).sin();
                let err = (rhs[q * mx + p] - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        assert!(
            max_err < 5e-3,
            "Poisson sine mode: max_err={:.6e} (expected < 5e-3)",
            max_err
        );
    }

    #[test]
    fn laplace_linear_bc() {
        let nx = 16;
        let ny = 12;
        let dx = 1.0 / nx as f64;
        let dy = 1.0 / ny as f64;
        let mx = nx - 1;
        let my = ny - 1;

        let mut solver = DstPoisson2D::new(nx, ny, dx, dy);

        let mut bc = vec![0.0f64; (nx + 1) * (ny + 1)];
        for j in 0..=ny {
            for i in 0..=nx {
                bc[j * (nx + 1) + i] = i as f64 * dx;
            }
        }

        let mut solution = vec![0.0f64; mx * my];
        solver.solve_inhomogeneous_laplace(&bc, &mut solution);

        let mut max_err = 0.0f64;
        for q in 0..my {
            for p in 0..mx {
                let i = p + 1;
                let exact = i as f64 * dx;
                let err = (solution[q * mx + p] - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }

        assert!(
            max_err < 1e-10,
            "Laplace linear BC: max_err={:.6e}",
            max_err
        );
    }
}