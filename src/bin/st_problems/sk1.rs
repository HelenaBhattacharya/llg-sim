//! sk1: static demag validation problems.
//!
//! Compare Poisson+multigrid demag against FFT/Newell on representative textures.

use std::fs::{File, create_dir_all};
use std::io::{self, Write};
use std::path::PathBuf;

use llg_sim::effective_field::demag_poisson_mg::DemagPoissonMGConfig;
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

/// Diagnose whether ΔBz ≈ c * (μ0 Ms m_z) for some constant c (often ~ ±0.5 when a
/// magnet/vacuum interface term is missing in a thin-film, single-z-cell formulation).
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

fn out_dir_for_case(case_tag: &str) -> io::Result<PathBuf> {
    let dir = PathBuf::from("runs/st_problems/sk1").join(format!("{}_rust", case_tag));
    create_dir_all(&dir)?;
    Ok(dir)
}

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
            // This isolates top/bottom surface-charge handling (Bz should be accurate).
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
            // This isolates lateral edge-charge handling (Bx/By should be accurate).
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

    let mg_cfg = DemagPoissonMGConfig::from_env();
    eprintln!(
        "[sk1] case={case_tag}, grid=({}x{}), dx={:.3} nm",
        grid.nx,
        grid.ny,
        grid.dx * 1e9
    );
    eprintln!("[sk1] mg_cfg: {:?}", mg_cfg);

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
