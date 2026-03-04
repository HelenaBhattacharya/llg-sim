//! sk1: static demag validation problems.
//!
//! Compare Poisson+multigrid demag against FFT/Newell on representative textures.
//!
//! Dashboard additions:
//! (D1) Spatial support: distance-to-boundary scalar field + |ΔB| stats vs distance.
//! (D3) Energy consistency: demag energy from FFT vs MG on the same magnetisation.
//!
//! Note: D2 (3D padded-box RHS diagnostics) removed — the new 2D cell-centred MG
//! solver no longer uses a 3D padded domain.

use std::collections::VecDeque;
use std::fs::{File, create_dir_all};
use std::io::{self, Write};
use std::path::PathBuf;

use llg_sim::effective_field::{demag_fft_uniform, demag_poisson_mg};
use llg_sim::geometry_mask::{Mask2D, mask_disk, mask_rect};
use llg_sim::grid::Grid2D;
use llg_sim::initial_states::{init_skyrmion, init_vortex};
use llg_sim::ovf::{OvfMeta, write_ovf2_rectangular_text};
use llg_sim::params::{DemagMethod, MU0, Material};
use llg_sim::vector_field::VectorField2D;

#[derive(Clone, Copy, Debug)]
struct CompMetrics {
    rmse: f64,
    max_abs: f64,
    p95_abs: f64,
}

fn compute_metrics(
    b_ref: &VectorField2D,
    b_test: &VectorField2D,
    mask: Option<&[bool]>,
) -> [CompMetrics; 3] {
    let n = b_ref.data.len();
    assert_eq!(b_test.data.len(), n);
    if let Some(m) = mask {
        assert_eq!(m.len(), n);
    }

    let mut sumsq = [0.0f64; 3];
    let mut max_abs = [0.0f64; 3];
    let mut abs_vals: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut count: usize = 0;

    for idx in 0..n {
        if let Some(m) = mask
            && !m[idx]
        {
            continue;
        }
        count += 1;
        let dr = [
            b_test.data[idx][0] - b_ref.data[idx][0],
            b_test.data[idx][1] - b_ref.data[idx][1],
            b_test.data[idx][2] - b_ref.data[idx][2],
        ];
        for c in 0..3 {
            let a = dr[c].abs();
            sumsq[c] += dr[c] * dr[c];
            if a > max_abs[c] {
                max_abs[c] = a;
            }
            abs_vals[c].push(a);
        }
    }

    if count == 0 {
        return [
            CompMetrics {
                rmse: 0.0,
                max_abs: 0.0,
                p95_abs: 0.0,
            },
            CompMetrics {
                rmse: 0.0,
                max_abs: 0.0,
                p95_abs: 0.0,
            },
            CompMetrics {
                rmse: 0.0,
                max_abs: 0.0,
                p95_abs: 0.0,
            },
        ];
    }

    let mut out = [
        CompMetrics {
            rmse: 0.0,
            max_abs: 0.0,
            p95_abs: 0.0,
        },
        CompMetrics {
            rmse: 0.0,
            max_abs: 0.0,
            p95_abs: 0.0,
        },
        CompMetrics {
            rmse: 0.0,
            max_abs: 0.0,
            p95_abs: 0.0,
        },
    ];

    for c in 0..3 {
        abs_vals[c].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_idx =
            ((0.95 * (abs_vals[c].len() as f64 - 1.0)).round() as usize).min(abs_vals[c].len() - 1);
        out[c] = CompMetrics {
            rmse: (sumsq[c] / (count as f64)).sqrt(),
            max_abs: max_abs[c],
            p95_abs: abs_vals[c][p95_idx],
        };
    }
    out
}

fn avg_field(b: &VectorField2D, mask: Option<&[bool]>) -> [f64; 3] {
    let n = b.data.len();
    if let Some(m) = mask {
        assert_eq!(m.len(), n);
    }
    let mut s = [0.0f64; 3];
    let mut count = 0usize;
    for idx in 0..n {
        if let Some(m) = mask
            && !m[idx]
        {
            continue;
        }
        count += 1;
        for c in 0..3 {
            s[c] += b.data[idx][c];
        }
    }
    if count == 0 {
        return [0.0, 0.0, 0.0];
    }
    [
        s[0] / count as f64,
        s[1] / count as f64,
        s[2] / count as f64,
    ]
}

#[derive(Clone, Copy, Debug)]
struct RatioStats {
    n: usize,
    mean: f64,
    p50: f64,
    p95: f64,
}

/// Diagnose whether ΔBz ≈ c * (μ0 Ms m_z) for some constant c.
fn dz_ratio_stats(
    b_ref: &VectorField2D,
    b_test: &VectorField2D,
    mask: Option<&[bool]>,
    m: &VectorField2D,
    ms: f64,
) -> Option<RatioStats> {
    let n = b_ref.data.len();
    if b_test.data.len() != n || m.data.len() != n {
        return None;
    }
    if let Some(mm) = mask {
        if mm.len() != n {
            return None;
        }
    }

    // Only include points where |m_z| is large enough that the ratio is meaningful.
    let mut ratios: Vec<f64> = Vec::new();
    let mut sum = 0.0f64;

    for idx in 0..n {
        if let Some(mm) = mask {
            if !mm[idx] {
                continue;
            }
        }
        let mz = m.data[idx][2];
        if mz.abs() < 0.5 {
            continue;
        }
        let dz = b_test.data[idx][2] - b_ref.data[idx][2];
        let denom = MU0 * ms * mz;
        if denom.abs() < 1e-30 {
            continue;
        }
        let r = dz / denom;
        ratios.push(r);
        sum += r;
    }

    if ratios.len() < 32 {
        return None;
    }

    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = ratios[ratios.len() / 2];
    let p95_idx = ((0.95 * (ratios.len() as f64 - 1.0)).round() as usize).min(ratios.len() - 1);
    let p95 = ratios[p95_idx];
    let mean = sum / ratios.len() as f64;

    Some(RatioStats {
        n: ratios.len(),
        mean,
        p50,
        p95,
    })
}

fn meta_m(case_tag: &str) -> OvfMeta {
    OvfMeta {
        title: format!("sk1{} magnetization", case_tag),
        desc_lines: vec!["m is unitless".to_string()],
        valuelabels: ["m_x".into(), "m_y".into(), "m_z".into()],
        valueunits: ["".into(), "".into(), "".into()],
    }
}

fn meta_b(case_tag: &str, label: &str) -> OvfMeta {
    OvfMeta {
        title: format!("sk1{} demag field ({})", case_tag, label),
        desc_lines: vec!["B is in Tesla".to_string()],
        valuelabels: ["B_x".into(), "B_y".into(), "B_z".into()],
        valueunits: ["T".into(), "T".into(), "T".into()],
    }
}

fn meta_scalar(case_tag: &str, label: &str, unit: &str) -> OvfMeta {
    OvfMeta {
        title: format!("sk1{} scalar ({})", case_tag, label),
        desc_lines: vec![format!("scalar stored in x-component; unit={unit}")],
        valuelabels: ["s".into(), "0".into(), "0".into()],
        valueunits: [unit.into(), "".into(), "".into()],
    }
}

fn out_dir_for_case(case_tag: &str) -> io::Result<PathBuf> {
    let dir = PathBuf::from("runs/st_problems/sk1").join(format!("{}_rust", case_tag));
    create_dir_all(&dir)?;
    Ok(dir)
}

/// Encode a scalar field into the x-component of a VectorField2D.
fn scalar_to_vfield(grid: Grid2D, s: &[f64]) -> VectorField2D {
    let mut v = VectorField2D::new(grid);
    assert_eq!(v.data.len(), s.len());
    for (i, &si) in s.iter().enumerate() {
        v.data[i] = [si, 0.0, 0.0];
    }
    v
}

/// Distance (in 4-neighbour cell steps) from each magnet cell to the boundary of the mask.
/// - boundary cells have dist=0
/// - interior cells have dist>=1
/// - outside mask has dist=-1
fn distance_to_boundary_4n(mask: &[bool], grid: &Grid2D) -> Vec<i32> {
    let nx = grid.nx;
    let ny = grid.ny;
    assert_eq!(mask.len(), nx * ny);

    let mut dist = vec![-1i32; nx * ny];
    let mut q: VecDeque<usize> = VecDeque::new();

    let in_bounds =
        |i: isize, j: isize| -> bool { i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < ny };

    // seed queue with boundary cells
    for j in 0..ny {
        for i in 0..nx {
            let id = j * nx + i;
            if !mask[id] {
                continue;
            }
            let mut is_boundary = false;

            // edge of grid counts as boundary
            if i == 0 || j == 0 || i + 1 == nx || j + 1 == ny {
                is_boundary = true;
            } else {
                // any 4-neighbour outside the mask => boundary
                let nb = [
                    (i as isize - 1, j as isize),
                    (i as isize + 1, j as isize),
                    (i as isize, j as isize - 1),
                    (i as isize, j as isize + 1),
                ];
                for (ii, jj) in nb {
                    if in_bounds(ii, jj) {
                        let nid = (jj as usize) * nx + (ii as usize);
                        if !mask[nid] {
                            is_boundary = true;
                            break;
                        }
                    }
                }
            }

            if is_boundary {
                dist[id] = 0;
                q.push_back(id);
            }
        }
    }

    // BFS inward
    while let Some(id) = q.pop_front() {
        let d0 = dist[id];
        let i = (id % nx) as isize;
        let j = (id / nx) as isize;

        let nb = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)];
        for (ii, jj) in nb {
            if !in_bounds(ii, jj) {
                continue;
            }
            let nid = (jj as usize) * nx + (ii as usize);
            if !mask[nid] {
                continue;
            }
            if dist[nid] >= 0 {
                continue;
            }
            dist[nid] = d0 + 1;
            q.push_back(nid);
        }
    }

    dist
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[derive(Clone, Copy, Debug)]
struct PctStats {
    n: usize,
    mean: f64,
    p50: f64,
    p95: f64,
    max: f64,
}

fn percentile_stats(mut vals: Vec<f64>) -> Option<PctStats> {
    if vals.len() < 8 {
        return None;
    }
    let n = vals.len();
    let mean = vals.iter().sum::<f64>() / n as f64;
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = vals[n / 2];
    let p95_idx = ((0.95 * (n as f64 - 1.0)).round() as usize).min(n - 1);
    let p95 = vals[p95_idx];
    let max = *vals.last().unwrap();
    Some(PctStats {
        n,
        mean,
        p50,
        p95,
        max,
    })
}

fn demag_energy_j(
    grid: &Grid2D,
    m: &VectorField2D,
    b: &VectorField2D,
    ms: f64,
    mask: Option<&[bool]>,
) -> f64 {
    let dvol = grid.dx * grid.dy * grid.dz;
    let mut sum = 0.0f64;
    for idx in 0..m.data.len() {
        if let Some(mm) = mask {
            if !mm[idx] {
                continue;
            }
        }
        let v = m.data[idx];
        let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if n2 < 1e-30 {
            continue;
        }
        let m_phys = [ms * v[0], ms * v[1], ms * v[2]];
        let b_i = b.data[idx];
        let mdotb = m_phys[0] * b_i[0] + m_phys[1] * b_i[1] + m_phys[2] * b_i[2];
        sum += mdotb * dvol;
    }
    // Demag energy (SI): E_d = -(1/2) ∫ M · B_d dV.
    // Here M is in A/m and B is in Tesla, so M·B has units of J/m^3; no division by μ0.
    -0.5 * sum
}

// ---------------------------------------------------------------------------
/// Per-component demag energy: returns (E_xy, E_z).
///
/// E_xy = -(1/2) integral (Mx*Bx + My*By) dV   (in-plane contribution)
/// E_z  = -(1/2) integral (Mz*Bz) dV            (out-of-plane contribution)
///
/// E_total = E_xy + E_z.  Splitting lets us see that E_z is exact (Kzz FFT)
/// while E_xy carries the FK/MG approximation error.
fn demag_energy_j_split(
    grid: &Grid2D,
    m: &VectorField2D,
    b: &VectorField2D,
    ms: f64,
    mask: Option<&[bool]>,
) -> (f64, f64) {
    let dvol = grid.dx * grid.dy * grid.dz;
    let mut sum_xy = 0.0f64;
    let mut sum_z = 0.0f64;
    for idx in 0..m.data.len() {
        if let Some(mm) = mask {
            if !mm[idx] {
                continue;
            }
        }
        let v = m.data[idx];
        let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if n2 < 1e-30 {
            continue;
        }
        let b_i = b.data[idx];
        sum_xy += (ms * v[0] * b_i[0] + ms * v[1] * b_i[1]) * dvol;
        sum_z += (ms * v[2] * b_i[2]) * dvol;
    }
    (-0.5 * sum_xy, -0.5 * sum_z)
}

// Test cases
// ---------------------------------------------------------------------------

fn build_case(case: char) -> (String, Grid2D, Option<Mask2D>, VectorField2D) {
    match case {
        'a' | 'A' => {
            let grid = Grid2D::new(256, 256, 2.5e-9, 2.5e-9, 3.0e-9);
            let radius = 0.45 * (grid.nx.min(grid.ny) as f64) * grid.dx;
            let mask = mask_disk(&grid, radius, (0.0, 0.0));
            let mut m = VectorField2D::new(grid);
            init_skyrmion(
                &mut m,
                &grid,
                (0.0, 0.0),
                60e-9,
                10e-9,
                0.0,
                -1.0,
                Some(&mask),
            );
            ("sk1a".to_string(), grid, Some(mask), m)
        }
        'b' | 'B' => {
            let grid = Grid2D::new(256, 256, 2.5e-9, 2.5e-9, 3.0e-9);
            let radius = 0.45 * (grid.nx.min(grid.ny) as f64) * grid.dx;
            let mask = mask_disk(&grid, radius, (0.0, 0.0));
            let mut m = VectorField2D::new(grid);
            init_vortex(&mut m, &grid, (0.0, 0.0), 1.0, 1.0, 20e-9, Some(&mask));
            ("sk1b".to_string(), grid, Some(mask), m)
        }
        'c' | 'C' => {
            let grid = Grid2D::new(200, 50, 2.5e-9, 2.5e-9, 3.0e-9);
            let mut m = VectorField2D::new(grid);
            m.set_uniform(1.0, 0.0, 0.0);
            ("sk1c".to_string(), grid, None, m)
        }
        'd' | 'D' => {
            let grid = Grid2D::new(256, 256, 2.5e-9, 2.5e-9, 3.0e-9);
            let hx = 0.5 * 200.0 * grid.dx;
            let hy = 0.5 * 50.0 * grid.dy;
            let mask = mask_rect(&grid, hx, hy, (0.0, 0.0));
            let mut m = VectorField2D::new(grid);
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let id = m.idx(i, j);
                    m.data[id] = if mask[id] {
                        [1.0, 0.0, 0.0]
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                }
            }
            ("sk1d".to_string(), grid, Some(mask), m)
        }
        'e' | 'E' => {
            // Uniform out-of-plane magnetisation in a disk.
            let grid = Grid2D::new(256, 256, 2.5e-9, 2.5e-9, 3.0e-9);
            let radius = 0.45 * (grid.nx.min(grid.ny) as f64) * grid.dx;
            let mask = mask_disk(&grid, radius, (0.0, 0.0));
            let mut m = VectorField2D::new(grid);
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let id = m.idx(i, j);
                    m.data[id] = if mask[id] {
                        [0.0, 0.0, 1.0]
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                }
            }
            ("sk1e".to_string(), grid, Some(mask), m)
        }
        'f' | 'F' => {
            // Uniform in-plane magnetisation in a disk.
            let grid = Grid2D::new(256, 256, 2.5e-9, 2.5e-9, 3.0e-9);
            let radius = 0.45 * (grid.nx.min(grid.ny) as f64) * grid.dx;
            let mask = mask_disk(&grid, radius, (0.0, 0.0));
            let mut m = VectorField2D::new(grid);
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let id = m.idx(i, j);
                    m.data[id] = if mask[id] {
                        [1.0, 0.0, 0.0]
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                }
            }
            ("sk1f".to_string(), grid, Some(mask), m)
        }
        _ => build_case('a'),
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn run(case: char) -> io::Result<()> {
    let (case_tag, grid, mask_opt, m) = build_case(case);
    let out_dir = out_dir_for_case(&case_tag)?;

    let mat = Material {
        ms: 8.0e5,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    eprintln!(
        "[sk1] case={case_tag}, grid=({}x{}), dx={:.3} nm, dz={:.3} nm",
        grid.nx,
        grid.ny,
        grid.dx * 1e9,
        grid.dz * 1e9,
    );

    // --- Compute fields ---
    let mut b_fft = VectorField2D::new(grid);
    let mut b_mg = VectorField2D::new(grid);
    demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_fft, &mat);
    demag_poisson_mg::compute_demag_field_poisson_mg(&grid, &m, &mut b_mg, &mat);

    let mut b_diff = VectorField2D::new(grid);
    for idx in 0..b_diff.data.len() {
        b_diff.data[idx][0] = b_mg.data[idx][0] - b_fft.data[idx][0];
        b_diff.data[idx][1] = b_mg.data[idx][1] - b_fft.data[idx][1];
        b_diff.data[idx][2] = b_mg.data[idx][2] - b_fft.data[idx][2];
    }

    let mask_slice = mask_opt.as_deref();

    // --- Existing metrics ---
    let metrics = compute_metrics(&b_fft, &b_mg, mask_slice);
    let avg = avg_field(&b_mg, mask_slice);

    if let Some(rs) = dz_ratio_stats(&b_fft, &b_mg, mask_slice, &m, mat.ms) {
        eprintln!(
            "[sk1] ΔBz/(μ0 Ms m_z) stats over |m_z|>=0.5: n={} mean={:.3e} p50={:.3e} p95={:.3e}",
            rs.n, rs.mean, rs.p50, rs.p95
        );
    } else {
        eprintln!("[sk1] ΔBz/(μ0 Ms m_z) stats: insufficient samples (need |m_z|>=0.5)");
    }

    // --- D3: Energy consistency (per-component) ---
    let e_fft = demag_energy_j(&grid, &m, &b_fft, mat.ms, mask_slice);
    let e_mg = demag_energy_j(&grid, &m, &b_mg, mat.ms, mask_slice);
    let de = e_mg - e_fft;

    let (e_fft_xy, e_fft_z) = demag_energy_j_split(&grid, &m, &b_fft, mat.ms, mask_slice);
    let (e_mg_xy, e_mg_z) = demag_energy_j_split(&grid, &m, &b_mg, mat.ms, mask_slice);

    // Gate the ratio: only meaningful when |E_fft| is large enough.
    let e_threshold = 1e-17; // Joules -- below this, ratio is noise/noise
    if e_fft.abs() >= e_threshold {
        let rel = de / e_fft;
        eprintln!(
            "[sk1] E_demag: E_fft={:.6e} J  E_mg={:.6e} J  dE/E={:.3e}",
            e_fft, e_mg, rel,
        );
    } else {
        eprintln!(
            "[sk1] E_demag: E_fft={:.6e} J  E_mg={:.6e} J  dE/E: N/A (|E_fft| < {:.0e})",
            e_fft, e_mg, e_threshold,
        );
    }
    eprintln!(
        "[sk1]   E_xy: fft={:.6e}  mg={:.6e}  dE_xy={:.3e} J  (FK/MG in-plane)",
        e_fft_xy, e_mg_xy, e_mg_xy - e_fft_xy,
    );
    eprintln!(
        "[sk1]   E_z:  fft={:.6e}  mg={:.6e}  dE_z ={:.3e} J  (Kzz FFT exact)",
        e_fft_z, e_mg_z, e_mg_z - e_fft_z,
    );

    // --- Write OVFs ---
    write_ovf2_rectangular_text(&out_dir.join("m.ovf"), &grid, &m, &meta_m(&case_tag))?;
    write_ovf2_rectangular_text(
        &out_dir.join("b_fft.ovf"),
        &grid,
        &b_fft,
        &meta_b(&case_tag, "fft"),
    )?;
    write_ovf2_rectangular_text(
        &out_dir.join("b_mg_env.ovf"),
        &grid,
        &b_mg,
        &meta_b(&case_tag, "env"),
    )?;
    write_ovf2_rectangular_text(
        &out_dir.join("b_diff_env.ovf"),
        &grid,
        &b_diff,
        &meta_b(&case_tag, "diff_env"),
    )?;

    // |dB| magnitude field for spatial visualisation of error
    {
        let mut b_diff_mag = VectorField2D::new(grid);
        for idx in 0..b_diff.data.len() {
            b_diff_mag.data[idx] = [norm3(b_diff.data[idx]), 0.0, 0.0];
        }
        write_ovf2_rectangular_text(
            &out_dir.join("b_diff_mag.ovf"),
            &grid,
            &b_diff_mag,
            &meta_scalar(&case_tag, "delta_B_magnitude", "T"),
        )?;
    }

    // --- D1: distance-to-boundary + |ΔB| vs distance ---
    if let Some(mask2d) = mask_slice {
        let dist_i32 = distance_to_boundary_4n(mask2d, &grid);
        let mut dist_f = vec![-1.0f64; dist_i32.len()];
        for i in 0..dist_i32.len() {
            dist_f[i] = dist_i32[i] as f64;
        }

        let dist_field = scalar_to_vfield(grid, &dist_f);
        write_ovf2_rectangular_text(
            &out_dir.join("dist_to_boundary.ovf"),
            &grid,
            &dist_field,
            &meta_scalar(&case_tag, "dist_to_boundary_4n", "cells"),
        )?;

        // Bin |ΔB| magnitude by distance.
        let max_bin: i32 = 8; // 0..7 plus ">=8"
        let mut bins: Vec<Vec<f64>> = (0..=max_bin).map(|_| Vec::new()).collect();

        let mut sum_all = 0.0f64;
        let mut sum_d01 = 0.0f64;

        for idx in 0..mask2d.len() {
            if !mask2d[idx] {
                continue;
            }
            let d = dist_i32[idx];
            if d < 0 {
                continue;
            }
            let db = norm3(b_diff.data[idx]);
            sum_all += db;
            if d <= 1 {
                sum_d01 += db;
            }
            let b = if d >= max_bin { max_bin } else { d };
            bins[b as usize].push(db);
        }

        let frac_d01 = if sum_all > 0.0 {
            sum_d01 / sum_all
        } else {
            0.0
        };
        eprintln!(
            "[sk1] |ΔB| mass fraction in dist<=1: {:.3e}  (dist<=1 sum / total sum)",
            frac_d01
        );

        // Write per-bin stats to CSV and print summary.
        let mut f = File::create(out_dir.join("dbmag_by_dist.csv"))?;
        writeln!(f, "dist_bin,n,mean_T,p50_T,p95_T,max_T")?;
        for (bi, v) in bins.into_iter().enumerate() {
            if let Some(st) = percentile_stats(v) {
                writeln!(
                    f,
                    "{},{},{:.6e},{:.6e},{:.6e},{:.6e}",
                    bi, st.n, st.mean, st.p50, st.p95, st.max
                )?;
                eprintln!(
                    "[sk1] |ΔB| vs dist: dist={} n={} mean={:.3e} p50={:.3e} p95={:.3e} max={:.3e} (T)",
                    bi, st.n, st.mean, st.p50, st.p95, st.max
                );
            } else {
                writeln!(f, "{},{},,,,,", bi, 0)?;
            }
        }
    }

    // --- Write metrics.csv ---
    {
        let mut f = File::create(out_dir.join("metrics.csv"))?;
        writeln!(
            f,
            "variant,component,rmse_T,max_abs_T,p95_abs_T,avg_bx_T,avg_by_T,avg_bz_T"
        )?;
        for (ci, cname) in ["x", "y", "z"].iter().enumerate() {
            writeln!(
                f,
                "env,{cname},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                metrics[ci].rmse, metrics[ci].max_abs, metrics[ci].p95_abs, avg[0], avg[1], avg[2]
            )?;
        }
        // Append energy summary lines (total + per-component)
        writeln!(f, "energy,E_fft_J,{:.16e}", e_fft)?;
        writeln!(f, "energy,E_mg_J,{:.16e}", e_mg)?;
        writeln!(f, "energy,dE_J,{:.16e}", de)?;
        writeln!(f, "energy,E_fft_xy_J,{:.16e}", e_fft_xy)?;
        writeln!(f, "energy,E_fft_z_J,{:.16e}", e_fft_z)?;
        writeln!(f, "energy,E_mg_xy_J,{:.16e}", e_mg_xy)?;
        writeln!(f, "energy,E_mg_z_J,{:.16e}", e_mg_z)?;
    }

    // --- Terminal summary ---
    eprintln!(
        "[sk1] avg(B)=[{:.3e},{:.3e},{:.3e}] T",
        avg[0], avg[1], avg[2]
    );
    eprintln!(
        "[sk1] ΔB metrics (mg-fft): rmse=[{:.3e},{:.3e},{:.3e}] T  p95=[{:.3e},{:.3e},{:.3e}] T  max=[{:.3e},{:.3e},{:.3e}] T",
        metrics[0].rmse,
        metrics[1].rmse,
        metrics[2].rmse,
        metrics[0].p95_abs,
        metrics[1].p95_abs,
        metrics[2].p95_abs,
        metrics[0].max_abs,
        metrics[1].max_abs,
        metrics[2].max_abs,
    );
    eprintln!("[sk1] wrote {}", out_dir.display());
    Ok(())
}