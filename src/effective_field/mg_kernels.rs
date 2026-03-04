// src/effective_field/mg_kernels.rs
//
// Free-function numerical kernels for the 2D cell-centred multigrid Poisson solver.
//
// All operations are on flat 2D arrays (nx * ny) with row-major layout.
// Boundary cells (outermost ring: i=0, i=nx-1, j=0, j=ny-1) hold Dirichlet
// values and are NOT updated by smoothers or residual computation.
//
// This replaces the old 3D mg_kernels.rs. The 3D versions are no longer needed
// because the 2D MG + boundary integral architecture eliminates the 3D padded box.

use rayon::prelude::*;

use super::mg_stencil::Stencil2D;

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn idx2(i: usize, j: usize, nx: usize) -> usize {
    j * nx + i
}

// ---------------------------------------------------------------------------
// Cell-centred 1D interpolation for bilinear prolongation (2D version)
// ---------------------------------------------------------------------------

/// For a fine-grid cell at coarse index `i_coarse` with sub-cell offset `r_i`
/// (where r_i in 0..r), returns (i0, i1, w0, w1) - the two coarse neighbours
/// and their bilinear weights.
#[inline]
pub fn interp_1d_cell_centered_2d(
    i_coarse: usize,
    r_i: usize,
    n_coarse: usize,
    r: usize,
) -> (usize, usize, f64, f64) {
    if r == 1 {
        let i0 = i_coarse.min(n_coarse - 1);
        return (i0, i0, 1.0, 0.0);
    }
    debug_assert!(r == 2);

    let i0 = i_coarse.min(n_coarse - 1);
    if r_i == 0 {
        // Fine cell is left of coarse cell centre
        let i1 = if i0 > 0 { i0 - 1 } else { 0 };
        if i1 == i0 { (i0, i0, 1.0, 0.0) } else { (i0, i1, 0.75, 0.25) }
    } else {
        // Fine cell is right of coarse cell centre
        let i1 = (i0 + 1).min(n_coarse - 1);
        if i1 == i0 { (i0, i0, 1.0, 0.0) } else { (i0, i1, 0.75, 0.25) }
    }
}

// ---------------------------------------------------------------------------
// Dirichlet BC stamping
// ---------------------------------------------------------------------------

/// Stamp Dirichlet boundary values from `bc_phi` onto `arr` for the perimeter
/// of an (nx, ny) grid.
pub fn stamp_dirichlet_bc_2d(arr: &mut [f64], bc_phi: &[f64], nx: usize, ny: usize) {
    // Bottom and top rows (j=0, j=ny-1)
    for i in 0..nx {
        arr[idx2(i, 0, nx)] = bc_phi[idx2(i, 0, nx)];
        arr[idx2(i, ny - 1, nx)] = bc_phi[idx2(i, ny - 1, nx)];
    }
    // Left and right columns (i=0, i=nx-1) for interior rows
    for j in 1..(ny - 1) {
        arr[idx2(0, j, nx)] = bc_phi[idx2(0, j, nx)];
        arr[idx2(nx - 1, j, nx)] = bc_phi[idx2(nx - 1, j, nx)];
    }
}

// ---------------------------------------------------------------------------
// Weighted Jacobi smoother (2D, general stencil)
// ---------------------------------------------------------------------------

/// Weighted Jacobi smoothing on a 2D cell-centred grid.
///
/// `phi` is updated in-place, `tmp` is scratch of same size.
/// Boundary cells are re-stamped from `bc_phi` after each sweep.
pub fn smooth_weighted_jacobi_2d(
    phi: &mut [f64],
    tmp: &mut [f64],
    rhs: &[f64],
    bc_phi: &[f64],
    nx: usize,
    ny: usize,
    stencil: &Stencil2D,
    iters: usize,
    omega: f64,
) {
    debug_assert_eq!(phi.len(), nx * ny);

    for _ in 0..iters {
        tmp.copy_from_slice(phi);

        phi.par_chunks_mut(nx)
            .enumerate()
            .for_each(|(j, phi_row)| {
                if j == 0 || j + 1 == ny {
                    return; // boundary row
                }
                let base = j * nx;
                for i in 1..(nx - 1) {
                    let id = base + i;
                    let off = stencil.offdiag_at(tmp, nx, ny, i, j);
                    let phi_gs = if stencil.diag.abs() > 1e-30 {
                        (off - rhs[id]) / stencil.diag
                    } else {
                        tmp[id] // degenerate stencil - skip
                    };
                    phi_row[i] = (1.0 - omega) * tmp[id] + omega * phi_gs;
                }
            });

        stamp_dirichlet_bc_2d(phi, bc_phi, nx, ny);
    }
}

// ---------------------------------------------------------------------------
// Red-black SOR smoother (2D, 5-point stencil only)
// ---------------------------------------------------------------------------

/// Red-black Gauss-Seidel with SOR on a 2D cell-centred grid.
///
/// Only valid for the 5-point stencil (axis-aligned neighbours).
pub fn smooth_rb_sor_2d(
    phi: &mut [f64],
    tmp: &mut [f64],
    rhs: &[f64],
    bc_phi: &[f64],
    nx: usize,
    ny: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    iters: usize,
    omega: f64,
) {
    let sx = inv_dx2;
    let sy = inv_dy2;
    let denom = 2.0 * (sx + sy);

    for _ in 0..iters {
        for color in 0..2usize {
            let phi_ro: &[f64] = phi;

            // Compute updated values into tmp
            tmp.par_chunks_mut(nx)
                .enumerate()
                .for_each(|(j, tmp_row)| {
                    if j == 0 || j + 1 == ny { return; }
                    let base = j * nx;
                    for i in 1..(nx - 1) {
                        if ((i + j) & 1) != color { continue; }
                        let id = base + i;

                        let xm = phi_ro[id - 1];
                        let xp = phi_ro[id + 1];
                        let ym = phi_ro[id - nx];
                        let yp = phi_ro[id + nx];

                        let off = sx * (xm + xp) + sy * (ym + yp);
                        let phi_new = (off - rhs[id]) / denom;
                        let phi_old = phi_ro[id];
                        tmp_row[i] = phi_old + omega * (phi_new - phi_old);
                    }
                });

            // Copy updated values back
            let tmp_ro: &[f64] = tmp;
            phi.par_chunks_mut(nx)
                .enumerate()
                .for_each(|(j, phi_row)| {
                    if j == 0 || j + 1 == ny { return; }
                    let base = j * nx;
                    for i in 1..(nx - 1) {
                        if ((i + j) & 1) != color { continue; }
                        phi_row[i] = tmp_ro[base + i];
                    }
                });
        }

        stamp_dirichlet_bc_2d(phi, bc_phi, nx, ny);
    }
}

// ---------------------------------------------------------------------------
// Residual computation (2D)
// ---------------------------------------------------------------------------

/// Compute residual r = rhs - A*phi on interior cells.
/// Returns the max-norm of the residual. Boundary cells in `res` are zeroed.
pub fn compute_residual_2d(
    phi: &[f64],
    rhs: &[f64],
    res: &mut [f64],
    nx: usize,
    ny: usize,
    stencil: &Stencil2D,
) -> f64 {
    debug_assert_eq!(phi.len(), nx * ny);

    res.par_chunks_mut(nx)
        .enumerate()
        .map(|(j, res_row)| {
            if j == 0 || j + 1 == ny {
                res_row.fill(0.0);
                return 0.0f64;
            }
            let base = j * nx;
            let mut max_abs: f64 = 0.0;
            res_row[0] = 0.0;
            res_row[nx - 1] = 0.0;
            for i in 1..(nx - 1) {
                let id = base + i;
                let aphi = stencil.apply_at(phi, nx, ny, i, j);
                let r = rhs[id] - aphi;
                res_row[i] = r;
                max_abs = max_abs.max(r.abs());
            }
            max_abs
        })
        .reduce(|| 0.0, |a, b| a.max(b))
}

// ---------------------------------------------------------------------------
// Restriction: fine residual -> coarse RHS (2D, full weighting)
// ---------------------------------------------------------------------------

/// Full-weighting restriction for 2D grids with refinement ratio 2.
/// `coarse_rhs` and `coarse_phi` are zeroed first.
pub fn restrict_residual_2d(
    fine_res: &[f64],
    fine_nx: usize,
    fine_ny: usize,
    coarse_rhs: &mut [f64],
    coarse_phi: &mut [f64],
    coarse_nx: usize,
    coarse_ny: usize,
) {
    let rx = fine_nx / coarse_nx;
    let ry = fine_ny / coarse_ny;
    debug_assert!(rx == 1 || rx == 2);
    debug_assert!(ry == 1 || ry == 2);

    coarse_rhs.fill(0.0);
    coarse_phi.fill(0.0);

    let w = 1.0 / ((rx * ry) as f64);

    coarse_rhs
        .par_chunks_mut(coarse_nx)
        .enumerate()
        .for_each(|(jc, rhs_row)| {
            if jc == 0 || jc + 1 == coarse_ny { return; }
            for ic in 1..(coarse_nx - 1) {
                let fi0 = rx * ic;
                let fj0 = ry * jc;
                let mut sum = 0.0;
                for dj in 0..ry {
                    for di in 0..rx {
                        sum += fine_res[idx2(fi0 + di, fj0 + dj, fine_nx)];
                    }
                }
                rhs_row[ic] = w * sum;
            }
        });
}

// ---------------------------------------------------------------------------
// Prolongation: coarse correction -> fine (2D)
// ---------------------------------------------------------------------------

/// Add coarse-grid correction to fine-grid phi (injection or bilinear).
/// `fine_phi` is modified in-place (correction *added*, not overwritten).
/// Boundary cells on the fine grid are skipped.
pub fn prolongate_add_2d(
    coarse_phi: &[f64],
    coarse_nx: usize,
    coarse_ny: usize,
    fine_phi: &mut [f64],
    fine_nx: usize,
    fine_ny: usize,
    bilinear: bool,
) {
    let rx = fine_nx / coarse_nx;
    let ry = fine_ny / coarse_ny;

    if bilinear {
        fine_phi
            .par_chunks_mut(fine_nx)
            .enumerate()
            .for_each(|(j, phi_row)| {
                if j == 0 || j + 1 == fine_ny { return; }
                let (j0, j1, wj0, wj1) = interp_1d_cell_centered_2d(j / ry, j % ry, coarse_ny, ry);
                for i in 1..(fine_nx - 1) {
                    let (i0, i1, wi0, wi1) = interp_1d_cell_centered_2d(i / rx, i % rx, coarse_nx, rx);
                    let c = |ii: usize, jj: usize| coarse_phi[idx2(ii, jj, coarse_nx)];
                    let v = wi0 * wj0 * c(i0, j0)
                          + wi1 * wj0 * c(i1, j0)
                          + wi0 * wj1 * c(i0, j1)
                          + wi1 * wj1 * c(i1, j1);
                    phi_row[i] += v;
                }
            });
    } else {
        // Injection
        fine_phi
            .par_chunks_mut(fine_nx)
            .enumerate()
            .for_each(|(j, phi_row)| {
                if j == 0 || j + 1 == fine_ny { return; }
                let jc = j / ry;
                for i in 1..(fine_nx - 1) {
                    let ic = i / rx;
                    if ic == 0 || ic + 1 >= coarse_nx || jc == 0 || jc + 1 >= coarse_ny {
                        continue;
                    }
                    phi_row[i] += coarse_phi[idx2(ic, jc, coarse_nx)];
                }
            });
    }
}

// ---------------------------------------------------------------------------
// Divergence of M on a 2D grid (face-averaged at material interfaces)
// ---------------------------------------------------------------------------

/// Compute div(M) on a 2D cell-centred grid using face-averaged values at
/// magnet-vacuum interfaces.
///
/// `m_data` contains magnetisation vectors already scaled by Ms.
/// `rhs_out` is overwritten.
pub fn compute_div_m_2d(
    m_data: &[[f64; 3]],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    rhs_out: &mut [f64],
) {
    debug_assert_eq!(m_data.len(), nx * ny);
    debug_assert_eq!(rhs_out.len(), nx * ny);

    #[inline]
    fn is_mag(v: [f64; 3]) -> bool {
        v[0] * v[0] + v[1] * v[1] + v[2] * v[2] > 1e-30
    }

    #[inline]
    fn face_val(in_a: bool, a: f64, in_b: bool, b: f64) -> f64 {
        match (in_a, in_b) {
            (true, true)   => 0.5 * (a + b),
            (true, false)  => a,
            (false, true)  => b,
            (false, false) => 0.0,
        }
    }

    rhs_out
        .par_chunks_mut(nx)
        .enumerate()
        .for_each(|(j, row)| {
            for i in 0..nx {
                let idx = j * nx + i;
                let c_in = is_mag(m_data[idx]);
                let mc = m_data[idx];

                let (xp_in, mxp) = if i + 1 < nx {
                    let id = j * nx + (i + 1); (is_mag(m_data[id]), m_data[id])
                } else { (false, [0.0; 3]) };
                let (xm_in, mxm) = if i > 0 {
                    let id = j * nx + (i - 1); (is_mag(m_data[id]), m_data[id])
                } else { (false, [0.0; 3]) };

                let (yp_in, myp) = if j + 1 < ny {
                    let id = (j + 1) * nx + i; (is_mag(m_data[id]), m_data[id])
                } else { (false, [0.0; 3]) };
                let (ym_in, mym) = if j > 0 {
                    let id = (j - 1) * nx + i; (is_mag(m_data[id]), m_data[id])
                } else { (false, [0.0; 3]) };

                let mx_p = face_val(c_in, mc[0], xp_in, mxp[0]);
                let mx_m = face_val(xm_in, mxm[0], c_in, mc[0]);
                let my_p = face_val(c_in, mc[1], yp_in, myp[1]);
                let my_m = face_val(ym_in, mym[1], c_in, mc[1]);

                row[i] = (mx_p - mx_m) / dx + (my_p - my_m) / dy;
            }
        });
}

// ---------------------------------------------------------------------------
// Gradient extraction: phi -> H_demag on 2D grid
// ---------------------------------------------------------------------------

/// Extract H_demag = -mu0 * grad(phi) on a 2D grid, *adding* into `b_out`.
///
/// Uses central differences for interior cells, one-sided at boundaries.
/// If `mag_mask` is Some, only cells where |m| > 0 are updated.
pub fn extract_gradient_2d(
    phi: &[f64],
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    mu0: f64,
    b_out: &mut [[f64; 3]],
    mag_mask: Option<&[[f64; 3]]>,
) {
    debug_assert_eq!(phi.len(), nx * ny);
    debug_assert_eq!(b_out.len(), nx * ny);

    #[inline]
    fn is_mag(v: [f64; 3]) -> bool {
        v[0] * v[0] + v[1] * v[1] + v[2] * v[2] > 1e-30
    }

    b_out
        .par_chunks_mut(nx)
        .enumerate()
        .for_each(|(j, row)| {
            for i in 0..nx {
                if let Some(mdata) = mag_mask {
                    if !is_mag(mdata[j * nx + i]) { continue; }
                }

                let phi_c = phi[idx2(i, j, nx)];

                // x gradient (central difference, one-sided at boundary)
                let dphi_dx = if i > 0 && i + 1 < nx {
                    (phi[idx2(i + 1, j, nx)] - phi[idx2(i - 1, j, nx)]) / (2.0 * dx)
                } else if i + 1 < nx {
                    (phi[idx2(i + 1, j, nx)] - phi_c) / dx
                } else if i > 0 {
                    (phi_c - phi[idx2(i - 1, j, nx)]) / dx
                } else {
                    0.0
                };

                // y gradient
                let dphi_dy = if j > 0 && j + 1 < ny {
                    (phi[idx2(i, j + 1, nx)] - phi[idx2(i, j - 1, nx)]) / (2.0 * dy)
                } else if j + 1 < ny {
                    (phi[idx2(i, j + 1, nx)] - phi_c) / dy
                } else if j > 0 {
                    (phi_c - phi[idx2(i, j - 1, nx)]) / dy
                } else {
                    0.0
                };

                row[i][0] += -mu0 * dphi_dx;
                row[i][1] += -mu0 * dphi_dy;
                // Bz is handled separately via Nzz self-demagnetisation (not from phi).
            }
        });
}

// ---------------------------------------------------------------------------
// Newell self-demagnetisation factor Nzz
// ---------------------------------------------------------------------------

/// Compute the Newell self-demagnetisation factor Nzz for a rectangular prism
/// of dimensions (dx, dy, dz).
///
/// This gives the fraction of the self-demagnetising field in the z direction:
///   Bz_self = -mu0 * Nzz * Ms * mz
///
/// For thin films (dz << dx, dy): Nzz -> 1.
/// For a cube: Nzz = 1/3.
///
/// Uses the analytic formula from Newell et al. (1993), J. Geophys. Res.
pub fn newell_nzz_self(dx: f64, dy: f64, dz: f64) -> f64 {
    // Use the identity Nxx + Nyy + Nzz = 1 and compute Nxx, Nyy via
    // the standard Newell self-demagnetisation integral.
    // For a single prism, Nzz = 1 - Nxx - Nyy where
    // Nxx = (1/(4*pi*V)) * integral over prism faces.
    //
    // Direct computation using the Newell f-function evaluated at the 8 corners.
    let a = dx;
    let b = dy;
    let c = dz;
    let vol = a * b * c;
    if vol <= 0.0 {
        return 1.0;
    }

    // The Newell self-interaction for component zz is:
    // Nzz = (2 / (pi * a * b * c)) * F(a, b, c)
    // where F is summed over corners with appropriate signs.
    fn f_newell(x: f64, y: f64, z: f64) -> f64 {
        let r = (x * x + y * y + z * z).sqrt();
        if r < 1e-30 { return 0.0; }

        let mut val = 0.0;

        // Term 1: (y/2)(z^2 - x^2) * asinh(y / sqrt(x^2 + z^2))
        let xz2 = x * x + z * z;
        if xz2 > 0.0 && y.abs() > 0.0 {
            val += 0.5 * y * (z * z - x * x) * (y / xz2.sqrt()).asinh();
        }

        // Term 2: (x/2)(z^2 - y^2) * asinh(x / sqrt(y^2 + z^2))
        let yz2 = y * y + z * z;
        if yz2 > 0.0 && x.abs() > 0.0 {
            val += 0.5 * x * (z * z - y * y) * (x / yz2.sqrt()).asinh();
        }

        // Term 3: -|x*y*z| * atan(x*y / (|z| * r))
        let xyz = (x * y * z).abs();
        if xyz > 1e-30 {
            val -= xyz * (x * y / (z.abs() * r)).atan();
        }

        // Term 4: (1/6)(2z^2 - x^2 - y^2) * r
        val += (2.0 * z * z - x * x - y * y) * r / 6.0;

        val
    }

    // Sum over 8 corners of the prism [0,a]×[0,b]×[0,c] with alternating signs.
    //
    // The Newell (1993) self-demagnetisation integral for the zz component is:
    //   Nzz = (2 / (π·V)) × Σ_{p,q,r ∈ {0,1}} (-1)^(p+q+r) f(p·a, q·b, r·c)
    //
    // BUG FIX: Previously evaluated at (±a, ±b, ±c), which are the corners of a
    // prism centred at the origin.  The f-function has parity symmetry that causes
    // perfect cancellation, always returning Nzz = 0.  The correct evaluation
    // points are the 8 corners of the physical prism [0,a]×[0,b]×[0,c].
    let mut nzz = 0.0;
    for pi in 0u32..2 {
        for pj in 0u32..2 {
            for pk in 0u32..2 {
                let sign = if (pi + pj + pk) % 2 == 0 { 1.0 } else { -1.0 };
                nzz += sign * f_newell(
                    pi as f64 * a,
                    pj as f64 * b,
                    pk as f64 * c,
                );
            }
        }
    }

    nzz *= 2.0 / (std::f64::consts::PI * vol);

    // Clamp to valid range [0, 1] for robustness
    nzz.clamp(0.0, 1.0)
}