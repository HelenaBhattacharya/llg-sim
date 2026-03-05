// src/effective_field/mg_kernels.rs
//
// Change A: Free-function numerical kernels for the geometric multigrid Poisson solver.
//
// Extracted from DemagPoissonMG so they operate on raw slices + explicit grid
// dimensions. The existing MGLevel methods become thin wrappers (see
// demag_poisson_mg.rs). This creates the seam for the composite-grid AMR solver
// (CompositeGridPoissonMG) where each "level" is a collection of AMR patches,
// not a single padded box.
//
// NO NUMERICS CHANGE — the uniform-grid path must remain bit-exact after refactor.

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Index helpers
// ---------------------------------------------------------------------------

#[inline]
pub fn idx3(i: usize, j: usize, k: usize, nx: usize, ny: usize) -> usize {
    (k * ny + j) * nx + i
}

#[inline]
fn clamp_idx(v: isize, lo: isize, hi: isize) -> isize {
    if v < lo { lo } else if v > hi { hi } else { v }
}

// ---------------------------------------------------------------------------
// Dirichlet BC stamping
// ---------------------------------------------------------------------------

/// Stamp Dirichlet boundary values from `bc_phi` onto `arr` for all six faces
/// of an (nx, ny, nz) box.
pub fn stamp_dirichlet_bc(arr: &mut [f64], bc_phi: &[f64], nx: usize, ny: usize, nz: usize) {
    // x-faces
    for k in 0..nz {
        for j in 0..ny {
            let id0 = idx3(0, j, k, nx, ny);
            let id1 = idx3(nx - 1, j, k, nx, ny);
            arr[id0] = bc_phi[id0];
            arr[id1] = bc_phi[id1];
        }
        // y-faces
        for i in 0..nx {
            let id0 = idx3(i, 0, k, nx, ny);
            let id1 = idx3(i, ny - 1, k, nx, ny);
            arr[id0] = bc_phi[id0];
            arr[id1] = bc_phi[id1];
        }
    }
    // z-faces
    for j in 0..ny {
        for i in 0..nx {
            let id0 = idx3(i, j, 0, nx, ny);
            let id1 = idx3(i, j, nz - 1, nx, ny);
            arr[id0] = bc_phi[id0];
            arr[id1] = bc_phi[id1];
        }
    }
}

// ---------------------------------------------------------------------------
// Stencil application on raw slices
// ---------------------------------------------------------------------------

/// Compute the off-diagonal stencil sum at cell (i,j,k) reading from `phi`.
///
/// `offs` and `coeffs` are the neighbour offsets and weights from Stencil3D.
/// This is the inner kernel of the Jacobi smoother and residual computation.
#[inline]
pub fn offdiag_sum_at(
    phi: &[f64],
    nx: usize, ny: usize, nz: usize,
    i: usize, j: usize, k: usize,
    offs: &[[isize; 3]],
    coeffs: &[f64],
) -> f64 {
    let i0 = i as isize;
    let j0 = j as isize;
    let k0 = k as isize;
    let nxm = nx as isize - 1;
    let nym = ny as isize - 1;
    let nzm = nz as isize - 1;

    let mut sum = 0.0;
    for (off, &c) in offs.iter().zip(coeffs.iter()) {
        let ii = clamp_idx(i0 + off[0], 0, nxm) as usize;
        let jj = clamp_idx(j0 + off[1], 0, nym) as usize;
        let kk = clamp_idx(k0 + off[2], 0, nzm) as usize;
        sum += c * phi[idx3(ii, jj, kk, nx, ny)];
    }
    sum
}

/// Apply full stencil (centre + off-diagonal) at cell (i,j,k).
#[inline]
pub fn stencil_apply_at(
    phi: &[f64],
    nx: usize, ny: usize, nz: usize,
    i: usize, j: usize, k: usize,
    center: f64,
    offs: &[[isize; 3]],
    coeffs: &[f64],
) -> f64 {
    center * phi[idx3(i, j, k, nx, ny)]
        + offdiag_sum_at(phi, nx, ny, nz, i, j, k, offs, coeffs)
}

// ---------------------------------------------------------------------------
// Weighted Jacobi smoother
// ---------------------------------------------------------------------------

/// Weighted Jacobi smoothing on a 3D cell-centred grid (raw-slice version).
///
/// `phi` is updated in-place, `tmp` is scratch of same size.
/// Boundary cells (outermost layer) are re-stamped from `bc_phi` after each sweep.
///
/// Stencil is passed as decomposed fields so the caller doesn't need to expose
/// the Stencil3D struct to this module.
pub fn smooth_weighted_jacobi(
    phi: &mut [f64],
    tmp: &mut [f64],
    rhs: &[f64],
    bc_phi: &[f64],
    nx: usize, ny: usize, nz: usize,
    st_diag: f64,
    st_offs: &[[isize; 3]],
    st_coeffs: &[f64],
    iters: usize,
    omega: f64,
) {
    debug_assert_eq!(phi.len(), nx * ny * nz);

    for _ in 0..iters {
        tmp.copy_from_slice(phi);

        phi.par_chunks_mut(nx)
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
                    let off = offdiag_sum_at(tmp, nx, ny, nz, i, j, k, st_offs, st_coeffs);
                    let phi_gs = if st_diag.abs() > 1e-30 {
                        (off - rhs[id]) / st_diag
                    } else {
                        // Degenerate stencil — skip update to prevent Inf/NaN
                        tmp[id]
                    };
                    phi_row[i] = (1.0 - omega) * tmp[id] + omega * phi_gs;
                }
            });

        stamp_dirichlet_bc(phi, bc_phi, nx, ny, nz);
    }
}

// ---------------------------------------------------------------------------
// Red-black SOR smoother (7-point stencil only)
// ---------------------------------------------------------------------------

/// Red-black Gauss–Seidel with SOR on a 3D cell-centred grid.
///
/// Only valid for the classic 7-point stencil (axis-aligned neighbours).
/// `tmp` is used as a communication buffer for the parallel colour sweep.
pub fn smooth_rb_sor(
    phi: &mut [f64],
    tmp: &mut [f64],
    rhs: &[f64],
    bc_phi: &[f64],
    nx: usize, ny: usize, nz: usize,
    inv_dx2: f64,
    inv_dy2: f64,
    inv_dz2: f64,
    iters: usize,
    omega: f64,
) {
    let sx = inv_dx2;
    let sy = inv_dy2;
    let sz = inv_dz2;
    let denom = 2.0 * (sx + sy + sz);
    let plane = nx * ny;

    for _ in 0..iters {
        for color in 0..2usize {
            let phi_ro: &[f64] = phi;

            tmp.par_chunks_mut(nx)
                .enumerate()
                .for_each(|(row_idx, tmp_row)| {
                    let k = row_idx / ny;
                    let j = row_idx % ny;
                    if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                        return;
                    }
                    let base = row_idx * nx;
                    for i in 1..(nx - 1) {
                        if ((i + j + k) & 1) != color { continue; }
                        let id = base + i;

                        let xm = phi_ro[id - 1];
                        let xp = phi_ro[id + 1];
                        let ym = phi_ro[id - nx];
                        let yp = phi_ro[id + nx];
                        let zm = phi_ro[id - plane];
                        let zp = phi_ro[id + plane];

                        let off = sx * (xm + xp) + sy * (ym + yp) + sz * (zm + zp);
                        let phi_new = (off - rhs[id]) / denom;
                        let phi_old = phi_ro[id];
                        tmp_row[i] = phi_old + omega * (phi_new - phi_old);
                    }
                });

            let tmp_ro: &[f64] = tmp;
            phi.par_chunks_mut(nx)
                .enumerate()
                .for_each(|(row_idx, phi_row)| {
                    let k = row_idx / ny;
                    let j = row_idx % ny;
                    if k == 0 || k + 1 == nz || j == 0 || j + 1 == ny {
                        return;
                    }
                    let base = row_idx * nx;
                    for i in 1..(nx - 1) {
                        if ((i + j + k) & 1) != color { continue; }
                        phi_row[i] = tmp_ro[base + i];
                    }
                });
        }

        stamp_dirichlet_bc(phi, bc_phi, nx, ny, nz);
    }
}

// ---------------------------------------------------------------------------
// Residual computation
// ---------------------------------------------------------------------------

/// Compute residual r = rhs − A·phi on interior cells.
///
/// Returns the max-norm of the residual. Boundary cells in `res` are zeroed.
pub fn compute_residual(
    phi: &[f64],
    rhs: &[f64],
    res: &mut [f64],
    nx: usize, ny: usize, nz: usize,
    st_center: f64,
    st_offs: &[[isize; 3]],
    st_coeffs: &[f64],
) -> f64 {
    debug_assert_eq!(phi.len(), nx * ny * nz);

    res.par_chunks_mut(nx)
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
                let aphi = stencil_apply_at(
                    phi, nx, ny, nz, i, j, k, st_center, st_offs, st_coeffs,
                );
                let r = rhs[id] - aphi;
                res_row[i] = r;
                max_abs = max_abs.max(r.abs());
            }
            max_abs
        })
        .reduce(|| 0.0, |a, b| a.max(b))
}

// ---------------------------------------------------------------------------
// Restriction (fine residual → coarse RHS)
// ---------------------------------------------------------------------------

/// Full-weighting restriction. Refinement ratios inferred from dimension ratios
/// (each must be 1 or 2). `coarse_rhs` and `coarse_phi` are zeroed first.
pub fn restrict_residual(
    fine_res: &[f64],
    fine_nx: usize, fine_ny: usize, fine_nz: usize,
    coarse_rhs: &mut [f64],
    coarse_phi: &mut [f64],
    coarse_nx: usize, coarse_ny: usize, coarse_nz: usize,
) {
    let rxf = fine_nx / coarse_nx;
    let ryf = fine_ny / coarse_ny;
    let rzf = fine_nz / coarse_nz;

    debug_assert!(rxf == 1 || rxf == 2);
    debug_assert!(ryf == 1 || ryf == 2);
    debug_assert!(rzf == 1 || rzf == 2);

    coarse_rhs.fill(0.0);
    coarse_phi.fill(0.0);

    let w = 1.0 / ((rxf * ryf * rzf) as f64);

    coarse_rhs
        .par_chunks_mut(coarse_nx)
        .enumerate()
        .for_each(|(rowc_idx, rhs_row)| {
            let kc = rowc_idx / coarse_ny;
            let jc = rowc_idx % coarse_ny;
            if kc == 0 || kc + 1 == coarse_nz || jc == 0 || jc + 1 == coarse_ny {
                return;
            }
            for ic in 1..(coarse_nx - 1) {
                let fi0 = rxf * ic;
                let fj0 = ryf * jc;
                let fk0 = rzf * kc;
                let mut sum = 0.0;
                for dk in 0..rzf {
                    for dj in 0..ryf {
                        for di in 0..rxf {
                            sum += fine_res[idx3(fi0 + di, fj0 + dj, fk0 + dk, fine_nx, fine_ny)];
                        }
                    }
                }
                rhs_row[ic] = w * sum;
            }
        });
}

// ---------------------------------------------------------------------------
// Prolongation (coarse correction → fine)
// ---------------------------------------------------------------------------

/// Cell-centred 1D interpolation helper for trilinear prolongation.
///
/// For a fine-grid cell at coarse index `i_coarse` with sub-cell offset `r_i`
/// (where r_i ∈ 0..r), returns (i0, i1, w0, w1) — the two coarse neighbours
/// and their weights.
#[inline]
pub fn interp_1d_cell_centered(
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
        let i1 = if i0 > 0 { i0 - 1 } else { 0 };
        if i1 == i0 { (i0, i0, 1.0, 0.0) } else { (i0, i1, 0.75, 0.25) }
    } else {
        let i1 = (i0 + 1).min(n_coarse - 1);
        if i1 == i0 { (i0, i0, 1.0, 0.0) } else { (i0, i1, 0.75, 0.25) }
    }
}

/// Add coarse-grid correction to fine-grid phi (injection or trilinear).
///
/// `fine_phi` is modified in-place (correction *added*, not overwritten).
/// Boundary cells on the fine grid are skipped.
pub fn prolongate_add(
    coarse_phi: &[f64],
    coarse_nx: usize, coarse_ny: usize, coarse_nz: usize,
    fine_phi: &mut [f64],
    fine_nx: usize, fine_ny: usize, fine_nz: usize,
    trilinear: bool,
) {
    let rx = fine_nx / coarse_nx;
    let ry = fine_ny / coarse_ny;
    let rz = fine_nz / coarse_nz;

    if trilinear {
        fine_phi
            .par_chunks_mut(fine_nx)
            .enumerate()
            .for_each(|(row_idx, phi_row)| {
                let k = row_idx / fine_ny;
                let j = row_idx % fine_ny;
                if k == 0 || k + 1 == fine_nz || j == 0 || j + 1 == fine_ny {
                    return;
                }
                let (k0, k1, wk0, wk1) = interp_1d_cell_centered(k / rz, k % rz, coarse_nz, rz);
                let (j0, j1, wj0, wj1) = interp_1d_cell_centered(j / ry, j % ry, coarse_ny, ry);

                for i in 1..(fine_nx - 1) {
                    let (i0, i1, wi0, wi1) = interp_1d_cell_centered(i / rx, i % rx, coarse_nx, rx);
                    let c = |ii: usize, jj: usize, kk: usize| coarse_phi[idx3(ii, jj, kk, coarse_nx, coarse_ny)];

                    let v = wi0 * wj0 * wk0 * c(i0, j0, k0)
                          + wi1 * wj0 * wk0 * c(i1, j0, k0)
                          + wi0 * wj1 * wk0 * c(i0, j1, k0)
                          + wi1 * wj1 * wk0 * c(i1, j1, k0)
                          + wi0 * wj0 * wk1 * c(i0, j0, k1)
                          + wi1 * wj0 * wk1 * c(i1, j0, k1)
                          + wi0 * wj1 * wk1 * c(i0, j1, k1)
                          + wi1 * wj1 * wk1 * c(i1, j1, k1);
                    phi_row[i] += v;
                }
            });
    } else {
        // Injection
        fine_phi
            .par_chunks_mut(fine_nx)
            .enumerate()
            .for_each(|(row_idx, phi_row)| {
                let k = row_idx / fine_ny;
                let j = row_idx % fine_ny;
                if k == 0 || k + 1 == fine_nz || j == 0 || j + 1 == fine_ny {
                    return;
                }
                let kc = k / rz;
                let jc = j / ry;
                for i in 1..(fine_nx - 1) {
                    let ic = i / rx;
                    if ic == 0 || ic + 1 >= coarse_nx
                        || jc == 0 || jc + 1 >= coarse_ny
                        || kc == 0 || kc + 1 >= coarse_nz
                    {
                        continue;
                    }
                    phi_row[i] += coarse_phi[idx3(ic, jc, kc, coarse_nx, coarse_ny)];
                }
            });
    }
}

// ---------------------------------------------------------------------------
// Gradient extraction (phi → H_demag) on the 2D magnet layer
// ---------------------------------------------------------------------------

/// Extract H_demag = −μ₀ ∇φ on a 2D magnet layer embedded at z-index `k_mag`
/// within the 3D padded box, *adding* into `b_out`.
///
/// Uses averaged one-sided differences (the existing face-gradient approach).
/// If `mag_mask` is Some, only cells where |m| > 0 are updated.
pub fn extract_gradient_on_magnet_layer(
    phi: &[f64],
    px: usize, py: usize, pz: usize,
    dx: f64, dy: f64, dz: f64,
    offset_x: usize, offset_y: usize, k_mag: usize,
    nx_m: usize, ny_m: usize,
    mu0: f64,
    b_out: &mut [[f64; 3]],
    mag_mask: Option<&[[f64; 3]]>,
) {
    debug_assert_eq!(b_out.len(), nx_m * ny_m);

    #[inline]
    fn is_mag(mv: [f64; 3]) -> bool {
        mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2] > 1e-30
    }

    let k = k_mag;

    b_out
        .par_chunks_mut(nx_m)
        .enumerate()
        .for_each(|(j, row)| {
            let pj = offset_y + j;
            for i in 0..nx_m {
                if let Some(mdata) = mag_mask {
                    if !is_mag(mdata[j * nx_m + i]) { continue; }
                }
                let pi = offset_x + i;
                let phi_c = phi[idx3(pi, pj, k, px, py)];

                // x gradient
                let mut dphi_dx = 0.0;
                let mut wdx = 0.0;
                if pi + 1 < px { dphi_dx += (phi[idx3(pi + 1, pj, k, px, py)] - phi_c) / dx; wdx += 1.0; }
                if pi > 0     { dphi_dx += (phi_c - phi[idx3(pi - 1, pj, k, px, py)]) / dx; wdx += 1.0; }
                if wdx > 0.0  { dphi_dx /= wdx; }

                // y gradient
                let mut dphi_dy = 0.0;
                let mut wdy = 0.0;
                if pj + 1 < py { dphi_dy += (phi[idx3(pi, pj + 1, k, px, py)] - phi_c) / dy; wdy += 1.0; }
                if pj > 0     { dphi_dy += (phi_c - phi[idx3(pi, pj - 1, k, px, py)]) / dy; wdy += 1.0; }
                if wdy > 0.0  { dphi_dy /= wdy; }

                // z gradient
                let mut dphi_dz = 0.0;
                let mut wdz = 0.0;
                if k + 1 < pz { dphi_dz += (phi[idx3(pi, pj, k + 1, px, py)] - phi_c) / dz; wdz += 1.0; }
                if k > 0     { dphi_dz += (phi_c - phi[idx3(pi, pj, k - 1, px, py)]) / dz; wdz += 1.0; }
                if wdz > 0.0  { dphi_dz /= wdz; }

                row[i][0] += -mu0 * dphi_dx;
                row[i][1] += -mu0 * dphi_dy;
                row[i][2] += -mu0 * dphi_dz;
            }
        });
}

// ---------------------------------------------------------------------------
// 2D divergence of M (for composite-grid per-level RHS, future use)
// ---------------------------------------------------------------------------

/// Compute ∇·M on a 2D cell-centred grid using face-averaged values at
/// magnet–vacuum interfaces. This is the same physics as `build_rhs_from_m`
/// but operates on a flat 2D grid rather than requiring the 3D padded embedding.
///
/// `m_data` contains magnetisation vectors already scaled by Ms.
/// `rhs_out` is overwritten.
pub fn compute_div_m_2d(
    m_data: &[[f64; 3]],
    nx_m: usize, ny_m: usize,
    dx: f64, dy: f64,
    rhs_out: &mut [f64],
) {
    debug_assert_eq!(m_data.len(), nx_m * ny_m);
    debug_assert_eq!(rhs_out.len(), nx_m * ny_m);

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
        .par_chunks_mut(nx_m)
        .enumerate()
        .for_each(|(j, row)| {
            for i in 0..nx_m {
                let idx = j * nx_m + i;
                let c_in = is_mag(m_data[idx]);
                let mc = m_data[idx];

                let (xp_in, mxp) = if i + 1 < nx_m {
                    let id = j * nx_m + (i + 1); (is_mag(m_data[id]), m_data[id])
                } else { (false, [0.0; 3]) };
                let (xm_in, mxm) = if i > 0 {
                    let id = j * nx_m + (i - 1); (is_mag(m_data[id]), m_data[id])
                } else { (false, [0.0; 3]) };

                let (yp_in, myp) = if j + 1 < ny_m {
                    let id = (j + 1) * nx_m + i; (is_mag(m_data[id]), m_data[id])
                } else { (false, [0.0; 3]) };
                let (ym_in, mym) = if j > 0 {
                    let id = (j - 1) * nx_m + i; (is_mag(m_data[id]), m_data[id])
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
// Gaussian screening (PPPM hybrid)
// ---------------------------------------------------------------------------

pub fn gaussian_kernel_1d(sigma_cells: f64) -> (Vec<f64>, isize) {
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
        for wi in &mut w { *wi /= sum; }
    }
    (w, r)
}

/// Separable Gaussian smoothing of the RHS in XY on raw 3D slices.
///
/// `rhs` is smoothed in-place, `tmp` is scratch (same size).
pub fn screen_rhs_gaussian_xy(
    rhs: &mut [f64],
    tmp: &mut [f64],
    nx: usize, ny: usize, nz: usize,
    sigma_cells: f64,
) {
    if !(sigma_cells > 0.0) { return; }
    let (w, r) = gaussian_kernel_1d(sigma_cells);
    if r <= 0 { return; }
    if nx < 3 || ny < 3 || nz < 3 { return; }

    // Pass 1: X convolution (rhs → tmp)
    {
        let rhs_ro: &[f64] = rhs;
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
                        if x < 0 { x = 0; } else if x > (nx as isize - 1) { x = nx as isize - 1; }
                        acc += wi * rhs_ro[base + x as usize];
                    }
                    tmp_row[i] = acc;
                }
            });
    }

    // Pass 2: Y convolution (tmp → rhs)
    {
        let tmp_ro: &[f64] = tmp;
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
                        if y < 0 { y = 0; } else if y > (ny as isize - 1) { y = ny as isize - 1; }
                        let src_row = k * ny + y as usize;
                        acc += wi * tmp_ro[src_row * nx + i];
                    }
                    rhs_row[i] = acc;
                }
            });
    }
}