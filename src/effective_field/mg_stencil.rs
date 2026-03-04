// src/effective_field/mg_stencil.rs
//
// 2D Laplacian stencils for the cell-centred multigrid Poisson solver.
//
// Provides construction, application, and Galerkin coarsening of constant-
// coefficient stencils on a 2D rectangular grid.

use std::collections::HashMap;

use super::mg_config::{LaplacianStencilKind, ProlongationKind};
use super::mg_kernels::{idx2, interp_1d_cell_centered_2d};

// ---------------------------------------------------------------------------
// Stencil2D
// ---------------------------------------------------------------------------

/// Constant-coefficient 2D stencil for a cell-centred Laplacian-like operator.
///
/// The stencil is decomposed as:  A * phi(i,j) = center * phi(i,j) + sum_k coeffs[k] * phi(neighbour_k)
/// where `diag = -center` is the denominator used in Jacobi smoothing.
#[derive(Clone, Debug)]
pub struct Stencil2D {
    pub center: f64,
    pub diag: f64,
    pub offs: Vec<[isize; 2]>,
    pub coeffs: Vec<f64>,
}

impl Stencil2D {
    /// Standard 5-point Laplacian:  (1/dx^2)(phi_{i+1} + phi_{i-1}) + (1/dy^2)(phi_{j+1} + phi_{j-1}) - 2(1/dx^2 + 1/dy^2) phi_{ij}
    pub fn five_point(dx: f64, dy: f64) -> Self {
        let sx = 1.0 / (dx * dx);
        let sy = 1.0 / (dy * dy);
        let center = -2.0 * (sx + sy);

        let offs = vec![[1, 0], [-1, 0], [0, 1], [0, -1]];
        let coeffs = vec![sx, sx, sy, sy];

        Self { center, diag: -center, offs, coeffs }
    }

    /// Isotropic 9-point Mehrstellen stencil.
    ///
    /// For dx = dy = h:
    ///   (1/(6h^2)) * [  1   4   1 ]     center = -20/(6h^2)
    ///                [  4 -20   4 ]     axis   =   4/(6h^2)
    ///                [  1   4   1 ]     diag   =   1/(6h^2)
    ///
    /// This stencil is fourth-order accurate for Laplace's equation on a
    /// square grid. Falls back to 5-point if dx != dy (anisotropic grids).
    pub fn iso9(dx: f64, dy: f64) -> Self {
        let rel = (dx - dy).abs() / dx.max(dy).max(1e-30);
        if rel > 1e-6 {
            return Self::five_point(dx, dy);
        }
        let h = 0.5 * (dx + dy);
        let inv_h2 = 1.0 / (h * h);

        let c_axis = (2.0 / 3.0) * inv_h2;
        let c_diag = (1.0 / 6.0) * inv_h2;
        let center = -(10.0 / 3.0) * inv_h2;

        let offs = vec![
            [1, 0], [-1, 0], [0, 1], [0, -1],   // axis neighbours
            [1, 1], [1, -1], [-1, 1], [-1, -1],  // diagonal neighbours
        ];
        let coeffs = vec![
            c_axis, c_axis, c_axis, c_axis,
            c_diag, c_diag, c_diag, c_diag,
        ];

        Self { center, diag: -center, offs, coeffs }
    }

    /// Build a stencil from the enum kind.
    pub fn from_kind(kind: LaplacianStencilKind, dx: f64, dy: f64) -> Self {
        match kind {
            LaplacianStencilKind::FivePoint => Self::five_point(dx, dy),
            LaplacianStencilKind::Iso9 => Self::iso9(dx, dy),
        }
    }

    /// Apply the full stencil at cell (i, j), reading from `phi`.
    /// Boundary cells are handled by clamping (Neumann-like at edges).
    #[inline]
    pub fn apply_at(&self, phi: &[f64], nx: usize, ny: usize, i: usize, j: usize) -> f64 {
        let mut sum = self.center * phi[idx2(i, j, nx)];
        let nxm = nx as isize - 1;
        let nym = ny as isize - 1;
        let ii = i as isize;
        let jj = j as isize;
        for (off, &c) in self.offs.iter().zip(self.coeffs.iter()) {
            let ni = (ii + off[0]).clamp(0, nxm) as usize;
            let nj = (jj + off[1]).clamp(0, nym) as usize;
            sum += c * phi[idx2(ni, nj, nx)];
        }
        sum
    }

    /// Compute only the off-diagonal sum at (i, j) — used in Jacobi smoother.
    #[inline]
    pub fn offdiag_at(&self, phi: &[f64], nx: usize, ny: usize, i: usize, j: usize) -> f64 {
        let nxm = nx as isize - 1;
        let nym = ny as isize - 1;
        let ii = i as isize;
        let jj = j as isize;
        let mut sum = 0.0;
        for (off, &c) in self.offs.iter().zip(self.coeffs.iter()) {
            let ni = (ii + off[0]).clamp(0, nxm) as usize;
            let nj = (jj + off[1]).clamp(0, nym) as usize;
            sum += c * phi[idx2(ni, nj, nx)];
        }
        sum
    }

    /// Galerkin coarsening: compute the coarse-grid stencil by the RAP triple product.
    ///
    /// This uses a small test domain (9x9 coarse) and applies the full
    /// restrict-apply-prolongate cycle column by column to extract the
    /// coarse operator entries numerically.
    pub fn galerkin_coarsen(fine: &Stencil2D, prolong: ProlongationKind) -> Self {
        let r: usize = 2;   // refinement ratio
        let ncx: usize = 9;
        let ncy: usize = 9;
        let nfx = ncx * r;
        let nfy = ncy * r;

        let c0 = (ncx / 2, ncy / 2);
        let id_c0 = idx2(c0.0, c0.1, ncx);

        let mut phi_c = vec![0.0f64; ncx * ncy];
        let mut phi_f = vec![0.0f64; nfx * nfy];
        let mut y_f   = vec![0.0f64; nfx * nfy];
        let mut y_c   = vec![0.0f64; ncx * ncy];

        let mut map: HashMap<(isize, isize), f64> = HashMap::new();

        for jy in 0..ncy {
            for ix in 0..ncx {
                // Unit impulse at coarse cell (ix, jy)
                phi_c.fill(0.0);
                phi_c[idx2(ix, jy, ncx)] = 1.0;

                // Prolongate to fine
                prolongate_scalar_2d(&phi_c, ncx, ncy, &mut phi_f, nfx, nfy, r, prolong);

                // Apply fine stencil
                apply_stencil_to_field_2d(fine, &phi_f, &mut y_f, nfx, nfy);

                // Restrict to coarse
                restrict_scalar_avg_2d(&y_f, nfx, nfy, &mut y_c, ncx, ncy, r);

                let coeff = y_c[id_c0];
                if coeff.abs() > 1e-14 {
                    let off = (ix as isize - c0.0 as isize, jy as isize - c0.1 as isize);
                    map.insert(off, coeff);
                }
            }
        }

        let mut keys: Vec<(isize, isize)> = map.keys().cloned().collect();
        keys.sort();

        let mut center = 0.0;
        let mut offs = Vec::new();
        let mut coeffs = Vec::new();

        for key in keys {
            let c = map[&key];
            if key == (0, 0) {
                center = c;
            } else {
                offs.push([key.0, key.1]);
                coeffs.push(c);
            }
        }

        let diag = -center;

        // Validate: if the coarsened stencil is ill-conditioned, signal it.
        let offdiag_abs_sum: f64 = coeffs.iter().map(|c| c.abs()).sum();
        let neg_offdiag = coeffs.iter().filter(|&&c| c < 0.0).count();
        let jacobi_rho = if diag > 0.0 { offdiag_abs_sum / diag } else { f64::INFINITY };

        if diag <= 0.0 || jacobi_rho > 2.0 || neg_offdiag > coeffs.len() / 2 {
            eprintln!(
                "[demag_mg2d] WARNING: Galerkin produced ill-conditioned stencil \
                 (diag={:.3e}, jacobi_rho={:.2}). Using rediscretization instead.",
                diag, jacobi_rho,
            );
            // Return sentinel (diag=0) that level builder can detect.
            return Self { center: 0.0, diag: 0.0, offs: Vec::new(), coeffs: Vec::new() };
        }

        Self { center, diag, offs, coeffs }
    }
}

// ---------------------------------------------------------------------------
// Helper functions for Galerkin coarsening (operate on flat 2D arrays)
// ---------------------------------------------------------------------------

fn apply_stencil_to_field_2d(
    st: &Stencil2D,
    phi: &[f64],
    out: &mut [f64],
    nx: usize,
    ny: usize,
) {
    debug_assert_eq!(phi.len(), nx * ny);
    debug_assert_eq!(out.len(), nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            out[idx2(i, j, nx)] = st.apply_at(phi, nx, ny, i, j);
        }
    }
}

fn restrict_scalar_avg_2d(
    fine: &[f64],
    nfx: usize, nfy: usize,
    coarse: &mut [f64],
    ncx: usize, ncy: usize,
    r: usize,
) {
    let norm = 1.0 / ((r * r) as f64);
    for jy in 0..ncy {
        for ix in 0..ncx {
            let mut sum = 0.0;
            for fj in 0..r {
                for fi in 0..r {
                    let i = ix * r + fi;
                    let j = jy * r + fj;
                    if i < nfx && j < nfy {
                        sum += fine[idx2(i, j, nfx)];
                    }
                }
            }
            coarse[idx2(ix, jy, ncx)] = norm * sum;
        }
    }
}

fn prolongate_scalar_2d(
    coarse: &[f64],
    ncx: usize, ncy: usize,
    fine: &mut [f64],
    nfx: usize, nfy: usize,
    r: usize,
    kind: ProlongationKind,
) {
    match kind {
        ProlongationKind::Injection => {
            fine.fill(0.0);
            for jy in 0..ncy {
                for ix in 0..ncx {
                    let v = coarse[idx2(ix, jy, ncx)];
                    for fj in 0..r {
                        for fi in 0..r {
                            let i = ix * r + fi;
                            let j = jy * r + fj;
                            if i < nfx && j < nfy {
                                fine[idx2(i, j, nfx)] = v;
                            }
                        }
                    }
                }
            }
        }
        ProlongationKind::Bilinear => {
            fine.fill(0.0);
            for j in 0..nfy {
                let jy = j / r;
                let rj = j % r;
                let (j0, j1, wj0, wj1) = interp_1d_cell_centered_2d(jy, rj, ncy, r);
                for i in 0..nfx {
                    let ix = i / r;
                    let ri = i % r;
                    let (i0, i1, wi0, wi1) = interp_1d_cell_centered_2d(ix, ri, ncx, r);

                    let v = wi0 * wj0 * coarse[idx2(i0, j0, ncx)]
                          + wi1 * wj0 * coarse[idx2(i1, j0, ncx)]
                          + wi0 * wj1 * coarse[idx2(i0, j1, ncx)]
                          + wi1 * wj1 * coarse[idx2(i1, j1, ncx)];
                    fine[idx2(i, j, nfx)] = v;
                }
            }
        }
    }
}