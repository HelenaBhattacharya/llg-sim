// src/bin/bench_composite_vcycle.rs
//
// Composite V-Cycle Demag Benchmark
// ==================================
//
// Purpose-built to validate the composite V-cycle by measuring B_demag
// at PATCH-LEVEL fine cells near geometric boundaries, where the V-cycle's
// fine ∇φ should beat coarse-FFT's interpolated B.
//
// Setup:
//   - Square Permalloy domain with a single circular hole (simple antidot)
//   - Saturated +x magnetisation (frozen — no LLG dynamics)
//   - AMR patches placed around the hole boundary (boundary-layer flagging)
//   - Three solvers compared at patch cells:
//       1. Uniform fine FFT (reference)
//       2. Coarse-FFT + bilinear interpolation to patches
//       3. Composite V-cycle (defect correction) + fine δ∇φ on patches
//
// Modes:
//   Single run (default):  accuracy + timing at one grid size
//   Crossover sweep (--sweep):  timing-only across multiple L0 sizes → CSV + plot
//
// Run:
//   cargo run --release --bin bench_composite_vcycle
//
// With V-cycle + L3 patches + plots:
//   LLG_DEMAG_COMPOSITE_VCYCLE=1 cargo run --release --bin bench_composite_vcycle -- --plots
//
// Crossover sweep (timing only, no fine ref):
//   LLG_DEMAG_COMPOSITE_VCYCLE=1 cargo run --release --bin bench_composite_vcycle -- --sweep
//
// Custom grid / levels:
//   LLG_CV_BASE_NX=256 LLG_AMR_MAX_LEVEL=3 LLG_DEMAG_COMPOSITE_VCYCLE=1 \
//     cargo run --release --bin bench_composite_vcycle

use std::fs;
use std::io::{BufWriter, Write};
use std::time::Instant;

use plotters::prelude::*;

use llg_sim::effective_field::coarse_fft_demag;
use llg_sim::effective_field::demag_fft_uniform;
use llg_sim::effective_field::demag_poisson_mg;
use llg_sim::effective_field::mg_composite;
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, Material};
use llg_sim::vector_field::VectorField2D;

use llg_sim::amr::indicator::IndicatorKind;
use llg_sim::amr::interp::sample_bilinear;
use llg_sim::amr::regrid::maybe_regrid_nested_levels;
use llg_sim::amr::{AmrHierarchy2D, ClusterPolicy, Connectivity, Rect2i, RegridPolicy};
use llg_sim::geometry_mask::MaskShape;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Core benchmark: run all three solvers at a given L0 grid size.
// Returns (t_fine_ms, t_cfft_ms, t_comp_ms, edge_rmse_cfft, edge_rmse_comp, n_l1, n_l2plus, n_edge)
// If skip_fine, t_fine_ms = 0 and edge errors are NaN.
// ---------------------------------------------------------------------------
fn run_benchmark(
    base_nx: usize, base_ny: usize, amr_levels: usize,
    ratio: usize, ghost: usize,
    domain_nm: f64, hole_radius_nm: f64, dz: f64,
    mat: &Material, shape: &MaskShape,
    skip_fine: bool, verbose: bool,
) -> (f64, f64, f64, f64, f64, usize, usize, usize) {
    let dx = domain_nm * 1e-9 / base_nx as f64;
    let dy = domain_nm * 1e-9 / base_ny as f64;
    let base_grid = Grid2D::new(base_nx, base_ny, dx, dy, dz);
    let total_ratio = ratio.pow(amr_levels as u32);
    let fine_nx = base_nx * total_ratio;
    let fine_ny = base_ny * total_ratio;
    let fine_grid = Grid2D::new(fine_nx, fine_ny, dx / total_ratio as f64, dy / total_ratio as f64, dz);

    let hole_radius = hole_radius_nm * 1e-9;
    let hole_centre = (0.0, 0.0);

    // Build coarse M
    let geom_mask = shape.to_mask(&base_grid);
    let mut m_coarse = VectorField2D::new(base_grid);
    for j in 0..base_ny {
        for i in 0..base_nx {
            let k = j * base_nx + i;
            m_coarse.data[k] = if geom_mask[k] { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 0.0] };
        }
    }

    // Build AMR hierarchy
    let mut h = AmrHierarchy2D::new(base_grid, m_coarse, ratio, ghost);
    h.set_geom_shape(shape.clone());

    let indicator_kind = IndicatorKind::Composite { frac: 0.10 };
    let boundary_layer: usize = env_or("LLG_AMR_BOUNDARY_LAYER", 4);
    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind,
        buffer_cells: 4,
        boundary_layer,
        connectivity: Connectivity::Eight,
        merge_distance: 1,
        min_patch_area: 16,
        max_patches: 0,
        min_efficiency: 0.65,
        max_flagged_fraction: 0.50,
    };
    let regrid_policy = RegridPolicy {
        indicator: indicator_kind,
        buffer_cells: 4,
        boundary_layer,
        min_change_cells: 1,
        min_area_change_frac: 0.01,
    };

    let current: Vec<Rect2i> = Vec::new();
    let _ = maybe_regrid_nested_levels(&mut h, &current, regrid_policy, cluster_policy);
    h.fill_patch_ghosts();

    // Reinitialise patch M at fine resolution
    for p in &mut h.patches {
        p.rebuild_active_from_shape(&base_grid, shape);
        let pnx = p.grid.nx;
        let pny = p.grid.ny;
        for j in 0..pny {
            for i in 0..pnx {
                let (x, y) = p.cell_center_xy_centered(i, j, &base_grid);
                p.m.data[j * pnx + i] = if shape.contains(x, y) { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 0.0] };
            }
        }
    }
    for lvl in &mut h.patches_l2plus {
        for p in lvl {
            p.rebuild_active_from_shape(&base_grid, shape);
            let pnx = p.grid.nx;
            let pny = p.grid.ny;
            for j in 0..pny {
                for i in 0..pnx {
                    let (x, y) = p.cell_center_xy_centered(i, j, &base_grid);
                    p.m.data[j * pnx + i] = if shape.contains(x, y) { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 0.0] };
                }
            }
        }
    }
    h.restrict_patches_to_coarse();

    let n_l1 = h.patches.len();
    let n_l2: usize = h.patches_l2plus.iter().map(|v| v.len()).sum();

    if verbose {
        println!("  Patches: L1={}, L2+={}", n_l1, n_l2);
    }

    // Fixed physical distance for edge classification (meters).
    // Using a constant physical band (default 2nm either side of the hole boundary)
    // ensures the edge cell population is consistent across grid sizes.
    // Previous: 4.0 * patch.grid.dx — this changed with resolution, making
    // cross-resolution RMSE comparisons unreliable.
    let edge_dist_nm: f64 = env_or("LLG_CV_EDGE_DIST_NM", 8.0);
    let edge_dist = edge_dist_nm * 1e-9;

    // Fine FFT reference
    let mut t_fine_ms = 0.0f64;
    let b_fine_opt = if !skip_fine {
        if verbose { println!("  Computing fine FFT reference ({} × {}) ...", fine_nx, fine_ny); }
        let mut m_fine = VectorField2D::new(fine_grid);
        let fine_half_lx = fine_nx as f64 * fine_grid.dx * 0.5;
        let fine_half_ly = fine_ny as f64 * fine_grid.dy * 0.5;
        for j in 0..fine_ny {
            for i in 0..fine_nx {
                let x = (i as f64 + 0.5) * fine_grid.dx - fine_half_lx;
                let y = (j as f64 + 0.5) * fine_grid.dy - fine_half_ly;
                m_fine.data[j * fine_nx + i] = if shape.contains(x, y) { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 0.0] };
            }
        }
        let t1 = Instant::now();
        let mut b_fine = VectorField2D::new(fine_grid);
        demag_fft_uniform::compute_demag_field(&fine_grid, &m_fine, &mut b_fine, mat);
        t_fine_ms = t1.elapsed().as_secs_f64() * 1e3;
        if verbose { println!("  Fine FFT:    {:.1} ms", t_fine_ms); }
        Some(b_fine)
    } else {
        None
    };

    // Coarse-FFT — warm up (builds Newell kernel on first call), then time
    {
        let mut bw = VectorField2D::new(base_grid);
        let _ = coarse_fft_demag::compute_coarse_fft_demag(&h, mat, &mut bw);
    }
    let t1 = Instant::now();
    let mut b_coarse_fft = VectorField2D::new(base_grid);
    let (b_l1_cfft, b_l2_cfft) = coarse_fft_demag::compute_coarse_fft_demag(&h, mat, &mut b_coarse_fft);
    let t_cfft_ms = t1.elapsed().as_secs_f64() * 1e3;
    if verbose { println!("  coarse-FFT:  {:.1} ms", t_cfft_ms); }

    // Composite — warm up then time
    {
        let mut bw = VectorField2D::new(base_grid);
        let _ = mg_composite::compute_composite_demag(&h, mat, &mut bw);
    }
    let t2 = Instant::now();
    let mut b_coarse_comp = VectorField2D::new(base_grid);
    let (b_l1_comp, b_l2_comp) = mg_composite::compute_composite_demag(&h, mat, &mut b_coarse_comp);
    let t_comp_ms = t2.elapsed().as_secs_f64() * 1e3;
    if verbose { println!("  composite:   {:.1} ms", t_comp_ms); }

    // Compute edge RMSE across ALL levels if we have the fine reference
    let mut edge_rmse_cfft = f64::NAN;
    let mut edge_rmse_comp = f64::NAN;
    let mut n_edge = 0usize;
    if let Some(ref b_fine) = b_fine_opt {
        let b_max = b_fine.data.iter()
            .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
            .fold(0.0f64, f64::max);

        let mut se_cfft = 0.0f64;
        let mut se_comp = 0.0f64;

        // Helper: measure edge error on a single patch
        let mut measure_patch = |patch: &llg_sim::amr::patch::Patch2D, bc: &[[f64; 3]], bv: &[[f64; 3]]| {
            let pnx = patch.grid.nx;
            let gi0 = patch.interior_i0();
            let gj0 = patch.interior_j0();
            let gi1 = patch.interior_i1();
            let gj1 = patch.interior_j1();
            for j in gj0..gj1 {
                for i in gi0..gi1 {
                    let (x, y) = patch.cell_center_xy_centered(i, j, &base_grid);
                    if !shape.contains(x, y) { continue; }
                    let dist = (x - hole_centre.0).hypot(y - hole_centre.1) - hole_radius;
                    if dist.abs() >= edge_dist { continue; }
                    let (xc, yc) = patch.cell_center_xy(i, j);
                    let br = sample_bilinear(b_fine, xc, yc);
                    let idx = j * pnx + i;
                    se_cfft += (bc[idx][0]-br[0]).powi(2) + (bc[idx][1]-br[1]).powi(2) + (bc[idx][2]-br[2]).powi(2);
                    se_comp += (bv[idx][0]-br[0]).powi(2) + (bv[idx][1]-br[1]).powi(2) + (bv[idx][2]-br[2]).powi(2);
                    n_edge += 1;
                }
            }
        };

        // L1 patches
        for (pi, patch) in h.patches.iter().enumerate() {
            if pi < b_l1_cfft.len() && pi < b_l1_comp.len() {
                measure_patch(patch, &b_l1_cfft[pi], &b_l1_comp[pi]);
            }
        }

        // L2+ patches
        for (lvl_idx, lvl_patches) in h.patches_l2plus.iter().enumerate() {
            let bc_lvl = if lvl_idx < b_l2_cfft.len() { &b_l2_cfft[lvl_idx] } else { continue };
            let bv_lvl = if lvl_idx < b_l2_comp.len() { &b_l2_comp[lvl_idx] } else { continue };
            for (pi, patch) in lvl_patches.iter().enumerate() {
                if pi < bc_lvl.len() && pi < bv_lvl.len() {
                    measure_patch(patch, &bc_lvl[pi], &bv_lvl[pi]);
                }
            }
        }

        if n_edge > 0 {
            edge_rmse_cfft = (se_cfft / n_edge as f64).sqrt() / b_max * 100.0;
            edge_rmse_comp = (se_comp / n_edge as f64).sqrt() / b_max * 100.0;
        }
    }

    (t_fine_ms, t_cfft_ms, t_comp_ms, edge_rmse_cfft, edge_rmse_comp, n_l1, n_l2, n_edge)
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let do_plots = args.iter().any(|a| a == "--plots");
    let do_sweep = args.iter().any(|a| a == "--sweep");

    // ---- Configuration ----
    let base_nx: usize = env_or("LLG_CV_BASE_NX", 128);
    let base_ny: usize = env_or("LLG_CV_BASE_NY", base_nx);
    let amr_levels: usize = env_or("LLG_AMR_MAX_LEVEL", 3);
    let ratio: usize = 2;
    let ghost: usize = 2;
    let skip_fine: bool = env_or("LLG_CV_SKIP_FINE", 0usize) != 0;

    let domain_nm = 500.0;
    let hole_radius_nm = 100.0;
    let dz: f64 = env_or("LLG_CV_DZ", 3e-9);

    let ms = 8.0e5;
    let a_ex = 1.3e-11;
    let mat = Material {
        ms, a_ex, k_u: 0.0, easy_axis: [0.0, 0.0, 1.0],
        dmi: None, demag: true, demag_method: DemagMethod::FftUniform,
    };

    let hole_centre = (0.0, 0.0);
    let hole_radius = hole_radius_nm * 1e-9;
    let outer = MaskShape::Full;
    let hole = MaskShape::Disk { center: hole_centre, radius: hole_radius };
    let shape = outer.difference(hole);

    let vcycle_on = std::env::var("LLG_DEMAG_COMPOSITE_VCYCLE")
        .map(|v| v == "1").unwrap_or(false);

    // Fixed physical distance for edge classification (same as sweep mode).
    let edge_dist_nm: f64 = env_or("LLG_CV_EDGE_DIST_NM", 2.0);
    let edge_dist = edge_dist_nm * 1e-9;

    // ════════════════════════════════════════════════════════════════
    // SWEEP MODE: timing + accuracy across grid sizes → CSV + plot
    // ════════════════════════════════════════════════════════════════
    if do_sweep {
        let sweep_sizes: Vec<usize> = if let Ok(s) = std::env::var("LLG_CV_SWEEP_SIZES") {
            s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
        } else {
            vec![32, 48, 64, 96, 128, 256, 512]
        };
        let sweep_levels: usize = amr_levels;
        let sweep_skip_fine = env_or("LLG_CV_SWEEP_SKIP_FINE", 0usize) != 0;
        let edge_dist_nm: f64 = env_or("LLG_CV_EDGE_DIST_NM", 2.0);

        let diag_dir = "out/bench_vcycle_diag";
        fs::create_dir_all(diag_dir).ok();
        let csv_path = format!("{}/crossover_sweep.csv", diag_dir);

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║  Composite V-Cycle — Crossover Sweep                          ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
        println!("  AMR levels:  {} (ratio {}×, total {}×)", sweep_levels, ratio, ratio.pow(sweep_levels as u32));
        println!("  V-cycle:     {}", if vcycle_on { "ON" } else { "OFF" });
        println!("  Fine ref:    {}", if sweep_skip_fine { "SKIPPED" } else { "ON" });
        println!("  Edge dist:   {:.1} nm (fixed physical distance)", edge_dist_nm);
        println!("  Grid sizes:  {:?}", sweep_sizes);
        println!();

        let mut csv_f = BufWriter::new(fs::File::create(&csv_path).unwrap());
        writeln!(csv_f, "base_nx,fine_nx,amr_levels,t_fine_ms,t_cfft_ms,t_comp_ms,\
            edge_rmse_cfft_pct,edge_rmse_comp_pct,n_l1,n_l2plus,fine_cells,comp_cells_est,n_edge").unwrap();

        println!("  {:>6} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10} {:>7}",
            "L0", "fine", "t_fine", "t_cfft", "t_comp", "e_cfft%", "e_comp%", "n_edge");
        println!("  {:->6} {:->7} {:->10} {:->10} {:->10} {:->10} {:->10} {:->7}",
            "", "", "", "", "", "", "", "");

        struct SweepRow { _base_nx: usize, fine_nx: usize, t_fine: f64, t_cfft: f64, t_comp: f64, e_cfft: f64, e_comp: f64, _n_edge: usize }
        let mut rows: Vec<SweepRow> = Vec::new();

        for &nx in &sweep_sizes {
            let tr = ratio.pow(sweep_levels as u32);
            let fnx = nx * tr;
            let (tf, tc, tv, ec, ev, nl1, nl2, ne) = run_benchmark(
                nx, nx, sweep_levels, ratio, ghost,
                domain_nm, hole_radius_nm, dz,
                &mat, &shape, sweep_skip_fine, false,
            );
            let fine_cells = fnx * fnx;
            // Rough estimate: L0 + patch cells (actual varies)
            let comp_cells_est = nx * nx + (nl1 + nl2) * 400;

            writeln!(csv_f, "{},{},{},{:.1},{:.1},{:.1},{:.2},{:.2},{},{},{},{},{}",
                nx, fnx, sweep_levels, tf, tc, tv, ec, ev, nl1, nl2, fine_cells, comp_cells_est, ne).unwrap();

            let ec_str = if ec.is_nan() { "N/A".to_string() } else { format!("{:.2}", ec) };
            let ev_str = if ev.is_nan() { "N/A".to_string() } else { format!("{:.2}", ev) };
            println!("  {:>6} {:>7} {:>9.0}ms {:>9.1}ms {:>9.1}ms {:>9}% {:>9}% {:>7}",
                nx, fnx, tf, tc, tv, ec_str, ev_str, ne);

            rows.push(SweepRow { _base_nx: nx, fine_nx: fnx, t_fine: tf, t_cfft: tc, t_comp: tv, e_cfft: ec, e_comp: ev, _n_edge: ne });
        }

        println!();
        println!("  CSV: {}", csv_path);

        // ---- Crossover plot ----
        if do_plots && rows.len() >= 2 {
            // Plot 1: Runtime vs fine-equivalent grid size
            let plot_path = format!("{}/crossover_timing.png", diag_dir);
            let root = BitMapBackend::new(&plot_path, (900, 550)).into_drawing_area();
            root.fill(&WHITE).unwrap();

            let x_max = rows.iter().map(|r| r.fine_nx as f64).fold(0.0f64, f64::max) * 1.2;
            let t_max = rows.iter().map(|r| r.t_fine.max(r.t_cfft).max(r.t_comp))
                .fold(0.0f64, f64::max) * 1.3;
            let t_max = if t_max < 10.0 { 100.0 } else { t_max };

            let mut chart = ChartBuilder::on(&root)
                .caption("Demag Solver Runtime vs Fine-Equivalent Grid Size", ("sans-serif", 20))
                .margin(15)
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d((50f64..x_max).log_scale(), (0.1f64..t_max).log_scale())
                .unwrap();

            chart.configure_mesh()
                .x_desc("Fine-equivalent grid N (N×N)")
                .y_desc("Time (ms)")
                .draw().unwrap();

            // Fine FFT points (skip zeros)
            let fft_pts: Vec<(f64, f64)> = rows.iter()
                .filter(|r| r.t_fine > 0.0)
                .map(|r| (r.fine_nx as f64, r.t_fine))
                .collect();
            if !fft_pts.is_empty() {
                chart.draw_series(LineSeries::new(fft_pts.clone(), BLUE.stroke_width(2)))
                    .unwrap()
                    .label("fine FFT")
                    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));
                chart.draw_series(fft_pts.iter().map(|&(x, y)| Circle::new((x, y), 4, BLUE.filled())))
                    .unwrap();
            }

            // Coarse-FFT
            let cfft_pts: Vec<(f64, f64)> = rows.iter()
                .map(|r| (r.fine_nx as f64, r.t_cfft))
                .collect();
            chart.draw_series(LineSeries::new(cfft_pts.clone(), GREEN.stroke_width(2)))
                .unwrap()
                .label("coarse FFT")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(2)));
            chart.draw_series(cfft_pts.iter().map(|&(x, y)| Circle::new((x, y), 4, GREEN.filled())))
                .unwrap();

            // Composite
            let comp_pts: Vec<(f64, f64)> = rows.iter()
                .map(|r| (r.fine_nx as f64, r.t_comp))
                .collect();
            chart.draw_series(LineSeries::new(comp_pts.clone(), RED.stroke_width(2)))
                .unwrap()
                .label("composite MG")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));
            chart.draw_series(comp_pts.iter().map(|&(x, y)| Circle::new((x, y), 4, RED.filled())))
                .unwrap();

            chart.configure_series_labels()
                .background_style(WHITE.mix(0.8))
                .border_style(BLACK)
                .position(SeriesLabelPosition::UpperLeft)
                .draw().unwrap();

            root.present().unwrap();
            println!("  Plot: {}", plot_path);

            // Plot 2: Accuracy vs grid size (if we have error data)
            if rows.iter().any(|r| !r.e_cfft.is_nan()) {
                let acc_path = format!("{}/crossover_accuracy.png", diag_dir);
                let root = BitMapBackend::new(&acc_path, (900, 550)).into_drawing_area();
                root.fill(&WHITE).unwrap();

                let e_max = rows.iter()
                    .filter(|r| !r.e_cfft.is_nan())
                    .map(|r| r.e_cfft.max(r.e_comp))
                    .fold(0.0f64, f64::max) * 1.3;

                let mut chart = ChartBuilder::on(&root)
                    .caption("Edge RMSE (%) vs L0 Grid Size", ("sans-serif", 20))
                    .margin(15)
                    .x_label_area_size(40)
                    .y_label_area_size(55)
                    .build_cartesian_2d(
                        (50f64..x_max).log_scale(),
                        0.0..e_max.max(1.0),
                    ).unwrap();

                chart.configure_mesh()
                    .x_desc("Fine-equivalent grid N")
                    .y_desc("Edge RMSE (%)")
                    .draw().unwrap();

                let cfft_acc: Vec<(f64, f64)> = rows.iter()
                    .filter(|r| !r.e_cfft.is_nan())
                    .map(|r| (r.fine_nx as f64, r.e_cfft))
                    .collect();
                let comp_acc: Vec<(f64, f64)> = rows.iter()
                    .filter(|r| !r.e_comp.is_nan())
                    .map(|r| (r.fine_nx as f64, r.e_comp))
                    .collect();

                chart.draw_series(LineSeries::new(cfft_acc.clone(), GREEN.stroke_width(2)))
                    .unwrap()
                    .label("coarse-FFT edge RMSE")
                    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN.stroke_width(2)));
                chart.draw_series(cfft_acc.iter().map(|&(x, y)| Circle::new((x, y), 4, GREEN.filled())))
                    .unwrap();

                chart.draw_series(LineSeries::new(comp_acc.clone(), RED.stroke_width(2)))
                    .unwrap()
                    .label("composite edge RMSE")
                    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));
                chart.draw_series(comp_acc.iter().map(|&(x, y)| Circle::new((x, y), 4, RED.filled())))
                    .unwrap();

                chart.configure_series_labels()
                    .background_style(WHITE.mix(0.8))
                    .border_style(BLACK)
                    .position(SeriesLabelPosition::UpperRight)
                    .draw().unwrap();

                root.present().unwrap();
                println!("  Plot: {}", acc_path);
            }
        }

        let wall = t0.elapsed().as_secs_f64();
        println!();
        println!("  Total sweep time: {:.1} s", wall);
        println!();
        return;
    }

    // ════════════════════════════════════════════════════════════════
    // SINGLE-RUN MODE: detailed accuracy comparison at one grid size
    // ════════════════════════════════════════════════════════════════
    let dx = domain_nm * 1e-9 / base_nx as f64;
    let dy = domain_nm * 1e-9 / base_ny as f64;
    let base_grid = Grid2D::new(base_nx, base_ny, dx, dy, dz);
    let total_ratio = ratio.pow(amr_levels as u32);
    let fine_nx = base_nx * total_ratio;
    let fine_ny = base_ny * total_ratio;
    let fine_grid = Grid2D::new(fine_nx, fine_ny, dx / total_ratio as f64, dy / total_ratio as f64, dz);

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Composite V-Cycle Benchmark — Single Antidot Hole            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Domain:      {:.0} nm × {:.0} nm, dz = {:.1} nm", domain_nm, domain_nm, dz * 1e9);
    println!("  Hole:        r = {:.0} nm at centre", hole_radius_nm);
    println!("  Base grid:   {} × {}, dx = {:.2} nm", base_nx, base_ny, dx * 1e9);
    println!("  Fine grid:   {} × {} ({}× refinement, {} AMR levels)",
        fine_nx, fine_ny, total_ratio, amr_levels);
    println!("  V-cycle:     {}", if vcycle_on { "ON (fine ∇φ on patches)" } else { "OFF (interpolated coarse B)" });
    println!();

    // ---- Build AMR hierarchy ----
    let geom_mask = shape.to_mask(&base_grid);
    let mut m_coarse = VectorField2D::new(base_grid);
    for j in 0..base_ny {
        for i in 0..base_nx {
            let k = j * base_nx + i;
            m_coarse.data[k] = if geom_mask[k] { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 0.0] };
        }
    }

    let mut h = AmrHierarchy2D::new(base_grid, m_coarse, ratio, ghost);
    h.set_geom_shape(shape.clone());

    // Regrid with boundary-layer flagging to place patches around the hole
    let indicator_kind = IndicatorKind::Composite { frac: 0.10 };
    let boundary_layer: usize = env_or("LLG_AMR_BOUNDARY_LAYER", 4);

    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind,
        buffer_cells: 4,
        boundary_layer,
        connectivity: Connectivity::Eight,
        merge_distance: 1,
        min_patch_area: 16,
        max_patches: 0,
        min_efficiency: 0.65,
        max_flagged_fraction: 0.50,
    };
    let regrid_policy = RegridPolicy {
        indicator: indicator_kind,
        buffer_cells: 4,
        boundary_layer,
        min_change_cells: 1,
        min_area_change_frac: 0.01,
    };

    let current: Vec<Rect2i> = Vec::new();
    if let Some((_rects, stats)) = maybe_regrid_nested_levels(&mut h, &current, regrid_policy, cluster_policy) {
        println!("  Regrid: {} cells flagged", stats.flagged_cells);
    }

    h.fill_patch_ghosts();

    // Reinitialise patch M at fine resolution using the shape
    for p in &mut h.patches {
        p.rebuild_active_from_shape(&base_grid, &shape);
        let pnx = p.grid.nx;
        let pny = p.grid.ny;
        for j in 0..pny {
            for i in 0..pnx {
                let (x, y) = p.cell_center_xy_centered(i, j, &base_grid);
                p.m.data[j * pnx + i] = if shape.contains(x, y) {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 0.0]
                };
            }
        }
    }
    for lvl in &mut h.patches_l2plus {
        for p in lvl {
            p.rebuild_active_from_shape(&base_grid, &shape);
            let pnx = p.grid.nx;
            let pny = p.grid.ny;
            for j in 0..pny {
                for i in 0..pnx {
                    let (x, y) = p.cell_center_xy_centered(i, j, &base_grid);
                    p.m.data[j * pnx + i] = if shape.contains(x, y) {
                        [1.0, 0.0, 0.0]
                    } else {
                        [0.0, 0.0, 0.0]
                    };
                }
            }
        }
    }
    h.restrict_patches_to_coarse();

    let n_l1 = h.patches.len();
    let n_l2: usize = h.patches_l2plus.iter().map(|v| v.len()).sum();
    println!("  Patches:     L1={}, L2+={}", n_l1, n_l2);
    println!();

    // ---- Fine FFT reference ----
    // Build fine-resolution M (reused for both FFT and MG references)
    let m_fine_opt = if !skip_fine {
        let mut m_fine = VectorField2D::new(fine_grid);
        let fine_half_lx = fine_nx as f64 * fine_grid.dx * 0.5;
        let fine_half_ly = fine_ny as f64 * fine_grid.dy * 0.5;
        for j in 0..fine_ny {
            for i in 0..fine_nx {
                let x = (i as f64 + 0.5) * fine_grid.dx - fine_half_lx;
                let y = (j as f64 + 0.5) * fine_grid.dy - fine_half_ly;
                m_fine.data[j * fine_nx + i] = if shape.contains(x, y) {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 0.0, 0.0]
                };
            }
        }
        Some(m_fine)
    } else {
        None
    };

    let b_fine_fft_opt = if let Some(ref m_fine) = m_fine_opt {
        println!("  Computing fine FFT reference ({} × {}) ...", fine_nx, fine_ny);
        let t1 = Instant::now();
        let mut b_fine = VectorField2D::new(fine_grid);
        demag_fft_uniform::compute_demag_field(&fine_grid, m_fine, &mut b_fine, &mat);
        let t_fine = t1.elapsed().as_secs_f64();
        println!("  Fine FFT:    {:.2} s", t_fine);

        let b_max = b_fine.data.iter()
            .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
            .fold(0.0f64, f64::max);
        println!("  max|B|:      {:.4e} T", b_max);
        Some(b_fine)
    } else {
        println!("  Fine FFT:    SKIPPED (LLG_CV_SKIP_FINE=1)");
        None
    };

    // ---- Coarse-FFT solve ----
    println!("  Running coarse-FFT ...");
    // Warm up (builds Newell kernel on first call)
    {
        let mut bw = VectorField2D::new(base_grid);
        let _ = coarse_fft_demag::compute_coarse_fft_demag(&h, &mat, &mut bw);
    }
    let t1 = Instant::now();
    let mut b_coarse_fft = VectorField2D::new(base_grid);
    let (b_l1_cfft, b_l2_cfft) = coarse_fft_demag::compute_coarse_fft_demag(&h, &mat, &mut b_coarse_fft);
    let t_cfft = t1.elapsed().as_secs_f64() * 1e3;
    println!("  coarse-FFT:  {:.1} ms", t_cfft);

    // ---- Composite solve ----
    println!("  Running composite {} ...", if vcycle_on { "(V-cycle)" } else { "(enhanced-RHS)" });
    // Warm up (builds MG hierarchy + ΔK cache on first call)
    {
        let mut bw = VectorField2D::new(base_grid);
        let _ = mg_composite::compute_composite_demag(&h, &mat, &mut bw);
    }
    let t2 = Instant::now();
    let mut b_coarse_comp = VectorField2D::new(base_grid);
    let (b_l1_comp, b_l2_comp) = mg_composite::compute_composite_demag(&h, &mat, &mut b_coarse_comp);
    let t_comp = t2.elapsed().as_secs_f64() * 1e3;
    println!("  composite:   {:.1} ms", t_comp);
    println!();

    // ---- Fine MG reference (same formulation as composite, uniform fine grid) ----
    // Computed AFTER composite solve so that PPPM auto-enable (if any)
    // is active — ensuring the fine-MG uses the same PPPM config as the composite L0.
    let b_fine_mg_opt = if let Some(ref m_fine) = m_fine_opt {
        println!("  Computing fine MG reference ({} × {}) ...", fine_nx, fine_ny);
        let t1 = Instant::now();
        let mut b_fine_mg = VectorField2D::new(fine_grid);
        demag_poisson_mg::compute_demag_field_poisson_mg(
            &fine_grid, m_fine, &mut b_fine_mg, &mat);
        let t_mg = t1.elapsed().as_secs_f64();
        println!("  Fine MG:     {:.2} s", t_mg);

        let b_max_mg = b_fine_mg.data.iter()
            .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
            .fold(0.0f64, f64::max);
        println!("  max|B|_MG:   {:.4e} T", b_max_mg);
        println!();
        Some(b_fine_mg)
    } else {
        None
    };

    // ---- Patch-level accuracy comparison ----
    // Helper to compute errors against a reference B field.
    // Iterates ALL AMR levels (L1, L2, L3, ...) for comprehensive measurement.
    let compute_patch_errors = |b_ref: &VectorField2D, label: &str| {
        let b_max_global = b_ref.data.iter()
            .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
            .fold(0.0f64, f64::max);

        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║  Patch-Level B Accuracy (vs {:<36})  ║", label);
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!("  (normalised to max|B| = {:.4e} T)", b_max_global);
        println!();

        // Per-level accumulators
        struct LevelStats {
            edge_se_cfft: f64, edge_se_comp: f64,
            bulk_se_cfft: f64, bulk_se_comp: f64,
            edge_se_comp_bx: f64, edge_se_comp_by: f64, edge_se_comp_bz: f64,
            edge_cells: usize, bulk_cells: usize,
            material_cells: usize,
        }
        impl LevelStats {
            fn new() -> Self { Self {
                edge_se_cfft: 0.0, edge_se_comp: 0.0,
                bulk_se_cfft: 0.0, bulk_se_comp: 0.0,
                edge_se_comp_bx: 0.0, edge_se_comp_by: 0.0, edge_se_comp_bz: 0.0,
                edge_cells: 0, bulk_cells: 0, material_cells: 0,
            }}
        }

        // Collect patches, B arrays, and level labels into a unified list.
        // Each entry: (level_label, &[Patch2D], &[Vec<[f64;3]>] cfft, &[Vec<[f64;3]>] comp)
        let n_levels = 1 + h.patches_l2plus.len(); // 1 for L1, plus however many L2+
        let mut level_stats: Vec<LevelStats> = (0..n_levels).map(|_| LevelStats::new()).collect();

        // Process L1 patches (level index 0)
        println!("  ── Level 1 ({} patches, dx={:.2} nm) ──", h.patches.len(),
            if !h.patches.is_empty() { h.patches[0].grid.dx * 1e9 } else { 0.0 });

        for (pi, patch) in h.patches.iter().enumerate() {
            let pnx = patch.grid.nx;
            let gi0 = patch.interior_i0();
            let gj0 = patch.interior_j0();
            let gi1 = patch.interior_i1();
            let gj1 = patch.interior_j1();

            let b_cfft = if pi < b_l1_cfft.len() { &b_l1_cfft[pi] } else { continue };
            let b_comp = if pi < b_l1_comp.len() { &b_l1_comp[pi] } else { continue };
            let stats = &mut level_stats[0];

            for j in gj0..gj1 {
                for i in gi0..gi1 {
                    let (x, y) = patch.cell_center_xy_centered(i, j, &base_grid);
                    if !shape.contains(x, y) { continue; }
                    stats.material_cells += 1;

                    let dist_to_hole = (x - hole_centre.0).hypot(y - hole_centre.1) - hole_radius;
                    let is_edge = dist_to_hole.abs() < edge_dist;

                    let (xc, yc) = patch.cell_center_xy(i, j);
                    let b_r = sample_bilinear(b_ref, xc, yc);
                    let idx = j * pnx + i;

                    let b_cf = b_cfft[idx];
                    let b_co = b_comp[idx];

                    let err_cfft = (b_cf[0]-b_r[0]).powi(2) + (b_cf[1]-b_r[1]).powi(2) + (b_cf[2]-b_r[2]).powi(2);
                    let err_comp = (b_co[0]-b_r[0]).powi(2) + (b_co[1]-b_r[1]).powi(2) + (b_co[2]-b_r[2]).powi(2);

                    if is_edge {
                        stats.edge_se_cfft += err_cfft;
                        stats.edge_se_comp += err_comp;
                        stats.edge_se_comp_bx += (b_co[0]-b_r[0]).powi(2);
                        stats.edge_se_comp_by += (b_co[1]-b_r[1]).powi(2);
                        stats.edge_se_comp_bz += (b_co[2]-b_r[2]).powi(2);
                        stats.edge_cells += 1;
                    } else {
                        stats.bulk_se_cfft += err_cfft;
                        stats.bulk_se_comp += err_comp;
                        stats.bulk_cells += 1;
                    }
                }
            }
        }

        // Process L2+ patches (level indices 1, 2, ...)
        for (lvl_idx, lvl_patches) in h.patches_l2plus.iter().enumerate() {
            let level_num = lvl_idx + 2;
            println!("  ── Level {} ({} patches, dx={:.2} nm) ──", level_num, lvl_patches.len(),
                if !lvl_patches.is_empty() { lvl_patches[0].grid.dx * 1e9 } else { 0.0 });

            let b_cfft_lvl = if lvl_idx < b_l2_cfft.len() { &b_l2_cfft[lvl_idx] } else { continue };
            let b_comp_lvl = if lvl_idx < b_l2_comp.len() { &b_l2_comp[lvl_idx] } else { continue };
            let stats = &mut level_stats[lvl_idx + 1];

            for (pi, patch) in lvl_patches.iter().enumerate() {
                let pnx = patch.grid.nx;
                let gi0 = patch.interior_i0();
                let gj0 = patch.interior_j0();
                let gi1 = patch.interior_i1();
                let gj1 = patch.interior_j1();

                let b_cfft = if pi < b_cfft_lvl.len() { &b_cfft_lvl[pi] } else { continue };
                let b_comp = if pi < b_comp_lvl.len() { &b_comp_lvl[pi] } else { continue };

                for j in gj0..gj1 {
                    for i in gi0..gi1 {
                        let (x, y) = patch.cell_center_xy_centered(i, j, &base_grid);
                        if !shape.contains(x, y) { continue; }
                        stats.material_cells += 1;

                        let dist_to_hole = (x - hole_centre.0).hypot(y - hole_centre.1) - hole_radius;
                        let is_edge = dist_to_hole.abs() < edge_dist;

                        let (xc, yc) = patch.cell_center_xy(i, j);
                        let b_r = sample_bilinear(b_ref, xc, yc);
                        let idx = j * pnx + i;

                        let b_cf = b_cfft[idx];
                        let b_co = b_comp[idx];

                        let err_cfft = (b_cf[0]-b_r[0]).powi(2) + (b_cf[1]-b_r[1]).powi(2) + (b_cf[2]-b_r[2]).powi(2);
                        let err_comp = (b_co[0]-b_r[0]).powi(2) + (b_co[1]-b_r[1]).powi(2) + (b_co[2]-b_r[2]).powi(2);

                        if is_edge {
                            stats.edge_se_cfft += err_cfft;
                            stats.edge_se_comp += err_comp;
                            stats.edge_se_comp_bx += (b_co[0]-b_r[0]).powi(2);
                            stats.edge_se_comp_by += (b_co[1]-b_r[1]).powi(2);
                            stats.edge_se_comp_bz += (b_co[2]-b_r[2]).powi(2);
                            stats.edge_cells += 1;
                        } else {
                            stats.bulk_se_cfft += err_cfft;
                            stats.bulk_se_comp += err_comp;
                            stats.bulk_cells += 1;
                        }
                    }
                }
            }
        }

        // Print per-level results
        for (li, stats) in level_stats.iter().enumerate() {
            let level_num = if li == 0 { 1 } else { li + 1 };
            if stats.edge_cells > 0 {
                let e_cfft = (stats.edge_se_cfft / stats.edge_cells as f64).sqrt();
                let e_comp = (stats.edge_se_comp / stats.edge_cells as f64).sqrt();
                let n = stats.edge_cells as f64;
                let bx_r = (stats.edge_se_comp_bx / n).sqrt();
                let by_r = (stats.edge_se_comp_by / n).sqrt();
                let bz_r = (stats.edge_se_comp_bz / n).sqrt();
                println!("  L{}: {} edge, {} bulk, {} material cells",
                    level_num, stats.edge_cells, stats.bulk_cells, stats.material_cells);
                println!("    Edge:  cfft={:.2}%  comp={:.2}%  (Bx={:.2}% By={:.2}% Bz={:.2}%)",
                    e_cfft / b_max_global * 100.0,
                    e_comp / b_max_global * 100.0,
                    bx_r / b_max_global * 100.0,
                    by_r / b_max_global * 100.0,
                    bz_r / b_max_global * 100.0);
            }
            if stats.bulk_cells > 0 {
                let b_cfft = (stats.bulk_se_cfft / stats.bulk_cells as f64).sqrt();
                let b_comp = (stats.bulk_se_comp / stats.bulk_cells as f64).sqrt();
                if stats.edge_cells == 0 {
                    println!("  L{}: {} bulk, {} material cells (no edge cells)",
                        level_num, stats.bulk_cells, stats.material_cells);
                }
                println!("    Bulk:  cfft={:.2}%  comp={:.2}%",
                    b_cfft / b_max_global * 100.0,
                    b_comp / b_max_global * 100.0);
            }
        }

        // Aggregate totals across all levels
        let total_edge_se_cfft: f64 = level_stats.iter().map(|s| s.edge_se_cfft).sum();
        let total_edge_se_comp: f64 = level_stats.iter().map(|s| s.edge_se_comp).sum();
        let total_bulk_se_cfft: f64 = level_stats.iter().map(|s| s.bulk_se_cfft).sum();
        let total_bulk_se_comp: f64 = level_stats.iter().map(|s| s.bulk_se_comp).sum();
        let total_edge_cells: usize = level_stats.iter().map(|s| s.edge_cells).sum();
        let total_bulk_cells: usize = level_stats.iter().map(|s| s.bulk_cells).sum();
        let total_material: usize = level_stats.iter().map(|s| s.material_cells).sum();
        let total_edge_se_comp_bx: f64 = level_stats.iter().map(|s| s.edge_se_comp_bx).sum();
        let total_edge_se_comp_by: f64 = level_stats.iter().map(|s| s.edge_se_comp_by).sum();
        let total_edge_se_comp_bz: f64 = level_stats.iter().map(|s| s.edge_se_comp_bz).sum();

        println!();
        println!("  ────────────────────────────────────────────────────");
        println!("  ALL-LEVEL TOTALS (vs {}) — {} material cells", label, total_material);
        println!("  ────────────────────────────────────────────────────");

        if total_edge_cells > 0 {
            let edge_rmse_cfft = (total_edge_se_cfft / total_edge_cells as f64).sqrt();
            let edge_rmse_comp = (total_edge_se_comp / total_edge_cells as f64).sqrt();
            let edge_rel_cfft = edge_rmse_cfft / b_max_global * 100.0;
            let edge_rel_comp = edge_rmse_comp / b_max_global * 100.0;

            println!("  EDGE ({} cells across all levels):", total_edge_cells);
            println!("    coarse-FFT: {:.2}%", edge_rel_cfft);
            println!("    composite:  {:.2}%", edge_rel_comp);

            if edge_rmse_comp < edge_rmse_cfft {
                println!("    → composite is {:.1}% MORE ACCURATE at edges",
                    (1.0 - edge_rmse_comp / edge_rmse_cfft) * 100.0);
            } else {
                println!("    → composite is {:.1}% worse at edges",
                    (edge_rmse_comp / edge_rmse_cfft - 1.0) * 100.0);
            }

            let n = total_edge_cells as f64;
            let bx_rmse = (total_edge_se_comp_bx / n).sqrt();
            let by_rmse = (total_edge_se_comp_by / n).sqrt();
            let bz_rmse = (total_edge_se_comp_bz / n).sqrt();
            println!("    Components: Bx={:.2}% By={:.2}% Bz={:.2}%",
                bx_rmse / b_max_global * 100.0,
                by_rmse / b_max_global * 100.0,
                bz_rmse / b_max_global * 100.0);
        }

        if total_bulk_cells > 0 {
            let bulk_rmse_cfft = (total_bulk_se_cfft / total_bulk_cells as f64).sqrt();
            let bulk_rmse_comp = (total_bulk_se_comp / total_bulk_cells as f64).sqrt();
            println!("  BULK ({} cells):", total_bulk_cells);
            println!("    coarse-FFT: {:.2}%", bulk_rmse_cfft / b_max_global * 100.0);
            println!("    composite:  {:.2}%", bulk_rmse_comp / b_max_global * 100.0);
        }
        println!();
    };

    // ---- Compare against fine FFT reference ----
    if let Some(ref b_fine_fft) = b_fine_fft_opt {
        compute_patch_errors(b_fine_fft, "uniform fine FFT (Newell)");
    }

    // ---- Compare against fine MG reference ----
    // This is the KEY diagnostic: if composite vs fine-MG is small,
    // the composite algorithm works correctly, and the gap vs Newell
    // is a formulation difference (not a V-cycle bug).
    if let Some(ref b_fine_mg) = b_fine_mg_opt {
        compute_patch_errors(b_fine_mg, "uniform fine MG (same formulation)");
    }

    if b_fine_fft_opt.is_none() && b_fine_mg_opt.is_none() {
        println!("  (Accuracy comparison skipped — set LLG_CV_SKIP_FINE=0 to enable)");
        println!();
    }

    // ---- Timing summary ----
    println!("  TIMING");
    println!("  ────────────────────────────────────────────────────");
    println!("    coarse-FFT:  {:.1} ms", t_cfft);
    println!("    composite:   {:.1} ms", t_comp);
    if !skip_fine {
        println!("    fine FFT:    {:.1} ms (reference)", t0.elapsed().as_secs_f64() * 1e3);
    }

    println!();
    println!("  V-cycle mode: {}", if vcycle_on { "ON" } else { "OFF" });
    if !vcycle_on {
        println!("  → Run with LLG_DEMAG_COMPOSITE_VCYCLE=1 to test fine-resolution B");
    }

    let wall = t0.elapsed().as_secs_f64();

    // ════════════════════════════════════════════════════════════════════
    // DIAGNOSTIC CSV OUTPUT
    // ════════════════════════════════════════════════════════════════════
    let diag_dir = "out/bench_vcycle_diag";
    fs::create_dir_all(diag_dir).ok();

    // ---- 1. Patch map ----
    {
        let path = format!("{}/patch_map.csv", diag_dir);
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        writeln!(f, "patch_id,level,coarse_i0,coarse_j0,coarse_nx,coarse_ny,ratio,ghost,fine_nx,fine_ny,dx_nm,dy_nm").unwrap();
        for (pi, p) in h.patches.iter().enumerate() {
            let cr = &p.coarse_rect;
            writeln!(f, "{},1,{},{},{},{},{},{},{},{},{:.4},{:.4}",
                pi, cr.i0, cr.j0, cr.nx, cr.ny, p.ratio, p.ghost,
                p.grid.nx, p.grid.ny, p.grid.dx * 1e9, p.grid.dy * 1e9).unwrap();
        }
        for (lvl_idx, lvl) in h.patches_l2plus.iter().enumerate() {
            for (pi, p) in lvl.iter().enumerate() {
                let cr = &p.coarse_rect;
                let global_id = h.patches.len() + lvl_idx * 1000 + pi;
                writeln!(f, "{},{},{},{},{},{},{},{},{},{},{:.4},{:.4}",
                    global_id, lvl_idx + 2, cr.i0, cr.j0, cr.nx, cr.ny,
                    p.ratio, p.ghost, p.grid.nx, p.grid.ny,
                    p.grid.dx * 1e9, p.grid.dy * 1e9).unwrap();
            }
        }
        println!("  Wrote {}", path);
    }

    // ---- 2. Per-cell error map for L1 patches near the hole ----
    if let Some(ref b_fine_fft) = b_fine_fft_opt {
        let path = format!("{}/error_map_l1.csv", diag_dir);
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        writeln!(f, "patch_id,i,j,x_nm,y_nm,dist_to_hole_nm,is_material,\
            bx_fft,by_fft,bz_fft,bx_cfft,by_cfft,bz_cfft,bx_comp,by_comp,bz_comp,\
            err_cfft,err_comp").unwrap();

        for (pi, patch) in h.patches.iter().enumerate() {
            let pnx = patch.grid.nx;
            let gi0 = patch.interior_i0();
            let gj0 = patch.interior_j0();
            let gi1 = patch.interior_i1();
            let gj1 = patch.interior_j1();

            let b_cfft = if pi < b_l1_cfft.len() { &b_l1_cfft[pi] } else { continue };
            let b_comp = if pi < b_l1_comp.len() { &b_l1_comp[pi] } else { continue };

            for j in gj0..gj1 {
                for i in gi0..gi1 {
                    let (x, y) = patch.cell_center_xy_centered(i, j, &base_grid);
                    let is_mat = shape.contains(x, y);
                    let dist_nm = ((x - hole_centre.0).hypot(y - hole_centre.1) - hole_radius) * 1e9;

                    let (xc, yc) = patch.cell_center_xy(i, j);
                    let b_ref = sample_bilinear(b_fine_fft, xc, yc);
                    let idx = j * pnx + i;
                    let bc = b_cfft[idx];
                    let bv = b_comp[idx];

                    let err_c = ((bc[0]-b_ref[0]).powi(2) + (bc[1]-b_ref[1]).powi(2) + (bc[2]-b_ref[2]).powi(2)).sqrt();
                    let err_v = ((bv[0]-b_ref[0]).powi(2) + (bv[1]-b_ref[1]).powi(2) + (bv[2]-b_ref[2]).powi(2)).sqrt();

                    writeln!(f, "{},{},{},{:.4},{:.4},{:.2},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                        pi, i, j, x * 1e9, y * 1e9, dist_nm, is_mat as u8,
                        b_ref[0], b_ref[1], b_ref[2],
                        bc[0], bc[1], bc[2],
                        bv[0], bv[1], bv[2],
                        err_c, err_v).unwrap();
                }
            }
        }
        println!("  Wrote {}", path);
    }

    // ---- 3. L0-level B comparison along y=centre slice ----
    {
        let path = format!("{}/l0_slice_y_center.csv", diag_dir);
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        writeln!(f, "i,x_nm,is_material,\
            bx_cfft,by_cfft,bz_cfft,bx_comp,by_comp,bz_comp,\
            bx_fft_ref,by_fft_ref,bz_fft_ref").unwrap();

        let jc = base_ny / 2;
        let half_lx = base_nx as f64 * dx * 0.5;
        for i in 0..base_nx {
            let x_phys = (i as f64 + 0.5) * dx - half_lx;
            let y_phys = (jc as f64 + 0.5) * dy - base_ny as f64 * dy * 0.5;
            let is_mat = shape.contains(x_phys, y_phys);
            let idx = jc * base_nx + i;

            let bc_fft = b_coarse_fft.data[idx];
            let bc_comp = b_coarse_comp.data[idx];

            // Sample fine FFT reference at this coarse cell centre
            let xc = (i as f64 + 0.5) * dx;
            let yc = (jc as f64 + 0.5) * dy;
            let b_ref = if let Some(ref bf) = b_fine_fft_opt {
                sample_bilinear(bf, xc, yc)
            } else {
                [0.0; 3]
            };

            writeln!(f, "{},{:.4},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                i, x_phys * 1e9, is_mat as u8,
                bc_fft[0], bc_fft[1], bc_fft[2],
                bc_comp[0], bc_comp[1], bc_comp[2],
                b_ref[0], b_ref[1], b_ref[2]).unwrap();
        }
        println!("  Wrote {}", path);
    }

    // ---- 4. L1 patch B along a radial slice through the hole ----
    // Pick the largest L1 patch and write a slice through its centre.
    if !h.patches.is_empty() && !b_l1_comp.is_empty() {
        let path = format!("{}/patch_radial_slice.csv", diag_dir);
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        writeln!(f, "patch_id,i,j,x_nm,y_nm,r_nm,is_material,\
            bx_fft,by_fft,bx_cfft,by_cfft,bx_comp,by_comp").unwrap();

        // Use patch 5 (typically the largest near-hole patch from the output)
        let target_pi = if h.patches.len() > 5 { 5 } else { 0 };
        let patch = &h.patches[target_pi];
        let pnx = patch.grid.nx;
        let gi0 = patch.interior_i0();
        let gj0 = patch.interior_j0();
        let gi1 = patch.interior_i1();
        let gj1 = patch.interior_j1();
        let jmid = (gj0 + gj1) / 2;

        let b_cfft = &b_l1_cfft[target_pi];
        let b_comp = &b_l1_comp[target_pi];

        for i in gi0..gi1 {
            let (x, y) = patch.cell_center_xy_centered(i, jmid, &base_grid);
            let r_nm = (x - hole_centre.0).hypot(y - hole_centre.1) * 1e9;
            let is_mat = shape.contains(x, y);

            let (xc, yc) = patch.cell_center_xy(i, jmid);
            let b_ref = if let Some(ref bf) = b_fine_fft_opt {
                sample_bilinear(bf, xc, yc)
            } else {
                [0.0; 3]
            };

            let idx = jmid * pnx + i;
            let bc = b_cfft[idx];
            let bv = b_comp[idx];

            writeln!(f, "{},{},{},{:.4},{:.4},{:.2},{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                target_pi, i, jmid, x * 1e9, y * 1e9, r_nm, is_mat as u8,
                b_ref[0], b_ref[1], bc[0], bc[1], bv[0], bv[1]).unwrap();
        }
        println!("  Wrote {}", path);
    }

    // ---- 5. Summary file ----
    {
        let path = format!("{}/summary.txt", diag_dir);
        let mut f = BufWriter::new(fs::File::create(&path).unwrap());
        writeln!(f, "Composite V-Cycle Benchmark Diagnostics").unwrap();
        writeln!(f, "========================================").unwrap();
        writeln!(f, "Domain: {:.0} nm x {:.0} nm, dz = {:.1} nm", domain_nm, domain_nm, dz * 1e9).unwrap();
        writeln!(f, "Hole: r = {:.0} nm at centre", hole_radius_nm).unwrap();
        writeln!(f, "Base grid: {} x {}, dx = {:.2} nm", base_nx, base_ny, dx * 1e9).unwrap();
        writeln!(f, "Fine grid: {} x {} ({} AMR levels)", fine_nx, fine_ny, amr_levels).unwrap();
        writeln!(f, "V-cycle: {}", if vcycle_on { "ON" } else { "OFF" }).unwrap();
        writeln!(f, "Patches: L1={}, L2+={}", n_l1, n_l2).unwrap();
        writeln!(f, "coarse-FFT: {:.1} ms", t_cfft).unwrap();
        writeln!(f, "composite: {:.1} ms", t_comp).unwrap();
        writeln!(f, "").unwrap();
        writeln!(f, "Files:").unwrap();
        writeln!(f, "  patch_map.csv          — patch geometry (all levels)").unwrap();
        writeln!(f, "  error_map_l1.csv       — per-cell B and error for L1 patches").unwrap();
        writeln!(f, "  l0_slice_y_center.csv  — L0 B along y=centre slice").unwrap();
        writeln!(f, "  patch_radial_slice.csv — B along radial cut through patch 5").unwrap();
        println!("  Wrote {}", path);
    }

    println!();
    println!("  Diagnostics written to {}/", diag_dir);

    // ---- PNG PLOTS (--plots flag) ----
    if do_plots {
        println!();
        println!("  Generating plots ...");

        // Plot 1: Patch map with hole geometry
        {
            let path = format!("{}/patch_map.png", diag_dir);
            let root = BitMapBackend::new(&path, (800, 800)).into_drawing_area();
            root.fill(&WHITE).unwrap();
            let half = domain_nm * 0.5;
            let mut chart = ChartBuilder::on(&root)
                .caption(format!("Patch Map — Antidot Hole (L0={}², {} AMR levels)", base_nx, amr_levels), ("sans-serif", 18))
                .margin(15).x_label_area_size(35).y_label_area_size(45)
                .build_cartesian_2d(-half..half, -half..half).unwrap();
            chart.configure_mesh().x_desc("x (nm)").y_desc("y (nm)").draw().unwrap();

            // Hole circle
            let n_pts = 120;
            let circle: Vec<(f64, f64)> = (0..=n_pts).map(|k| {
                let th = 2.0 * std::f64::consts::PI * k as f64 / n_pts as f64;
                (hole_radius_nm * th.cos(), hole_radius_nm * th.sin())
            }).collect();
            chart.draw_series(std::iter::once(PathElement::new(circle, BLACK.stroke_width(3)))).unwrap();

            // L1 patches (yellow/orange)
            for p in h.patches.iter() {
                let cr = &p.coarse_rect;
                let x0 = cr.i0 as f64 * dx * 1e9 - half;
                let y0 = cr.j0 as f64 * dy * 1e9 - half;
                let x1 = (cr.i0 + cr.nx) as f64 * dx * 1e9 - half;
                let y1 = (cr.j0 + cr.ny) as f64 * dy * 1e9 - half;
                chart.draw_series(std::iter::once(Rectangle::new([(x0, y0), (x1, y1)], RGBColor(255, 200, 0).mix(0.25).filled()))).unwrap();
                chart.draw_series(std::iter::once(PathElement::new(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], RGBColor(200, 150, 0).stroke_width(2)))).unwrap();
            }
            // L2+ patches with distinct colors per level
            let level_colors: Vec<(RGBColor, RGBColor)> = vec![
                (RGBColor(0, 160, 0), RGBColor(0, 100, 0)),       // L2: green
                (RGBColor(0, 100, 200), RGBColor(0, 60, 150)),     // L3: blue
                (RGBColor(180, 0, 180), RGBColor(120, 0, 120)),    // L4: purple (if needed)
            ];
            for (lvl_idx, lvl) in h.patches_l2plus.iter().enumerate() {
                let (fill_c, stroke_c) = if lvl_idx < level_colors.len() {
                    level_colors[lvl_idx]
                } else {
                    (RGBColor(128, 128, 128), RGBColor(80, 80, 80))
                };
                for p in lvl.iter() {
                    let cr = &p.coarse_rect;
                    let x0 = cr.i0 as f64 * dx * 1e9 - half;
                    let y0 = cr.j0 as f64 * dy * 1e9 - half;
                    let x1 = (cr.i0 + cr.nx) as f64 * dx * 1e9 - half;
                    let y1 = (cr.j0 + cr.ny) as f64 * dy * 1e9 - half;
                    chart.draw_series(std::iter::once(Rectangle::new([(x0, y0), (x1, y1)], fill_c.mix(0.2).filled()))).unwrap();
                    chart.draw_series(std::iter::once(PathElement::new(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], stroke_c.stroke_width(1)))).unwrap();
                }
            }
            root.present().unwrap();
            println!("    Wrote {}", path);
        }

        // Plot 2: Radial Bx profile
        if b_fine_fft_opt.is_some() && !b_l1_comp.is_empty() && h.patches.len() > 5 {
            let path = format!("{}/radial_bx_profile.png", diag_dir);
            let root = BitMapBackend::new(&path, (900, 500)).into_drawing_area();
            root.fill(&WHITE).unwrap();

            let target_pi = if h.patches.len() > 5 { 5 } else { 0 };
            let patch = &h.patches[target_pi];
            let pnx = patch.grid.nx;
            let gi0 = patch.interior_i0();
            let gi1 = patch.interior_i1();
            let gj0 = patch.interior_j0();
            let gj1 = patch.interior_j1();
            let jmid = (gj0 + gj1) / 2;
            let b_cfft_p = &b_l1_cfft[target_pi];
            let b_comp_p = &b_l1_comp[target_pi];

            let mut pts_ref = Vec::new();
            let mut pts_cfft = Vec::new();
            let mut pts_comp = Vec::new();
            for i in gi0..gi1 {
                let (x, y) = patch.cell_center_xy_centered(i, jmid, &base_grid);
                let r_nm = (x - hole_centre.0).hypot(y - hole_centre.1) * 1e9;
                let idx = jmid * pnx + i;
                let (xc, yc) = patch.cell_center_xy(i, jmid);
                if let Some(ref bf) = b_fine_fft_opt {
                    let br = sample_bilinear(bf, xc, yc);
                    pts_ref.push((r_nm, br[0]));
                }
                pts_cfft.push((r_nm, b_cfft_p[idx][0]));
                pts_comp.push((r_nm, b_comp_p[idx][0]));
            }
            if !pts_ref.is_empty() {
                let r_min = pts_ref.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
                let r_max = pts_ref.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
                let b_min = pts_ref.iter().chain(pts_cfft.iter()).chain(pts_comp.iter())
                    .map(|p| p.1).fold(f64::INFINITY, f64::min) * 1.1;
                let b_max = pts_ref.iter().chain(pts_cfft.iter()).chain(pts_comp.iter())
                    .map(|p| p.1).fold(f64::NEG_INFINITY, f64::max) * 1.1;
                let mut chart = ChartBuilder::on(&root)
                    .caption("Bx vs Radial Distance (Patch 5 mid-slice)", ("sans-serif", 18))
                    .margin(10).x_label_area_size(35).y_label_area_size(55)
                    .build_cartesian_2d(r_min..r_max, b_min..b_max).unwrap();
                chart.configure_mesh().x_desc("r (nm)").y_desc("Bx (T)").draw().unwrap();
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(hole_radius_nm, b_min), (hole_radius_nm, b_max)], BLACK.stroke_width(1),
                ))).unwrap();
                chart.draw_series(LineSeries::new(pts_ref, BLUE.stroke_width(2))).unwrap()
                    .label("FFT ref").legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], BLUE.stroke_width(2)));
                chart.draw_series(LineSeries::new(pts_cfft, GREEN.stroke_width(2))).unwrap()
                    .label("coarse-FFT").legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], GREEN.stroke_width(2)));
                chart.draw_series(LineSeries::new(pts_comp, RED.stroke_width(2))).unwrap()
                    .label("composite").legend(|(x, y)| PathElement::new(vec![(x, y), (x+20, y)], RED.stroke_width(2)));
                chart.configure_series_labels()
                    .background_style(WHITE.mix(0.8)).border_style(BLACK)
                    .position(SeriesLabelPosition::UpperLeft).draw().unwrap();
                root.present().unwrap();
                println!("    Wrote {}", path);
            }
        }

        // Plot 3: Error bar chart (all levels)
        if let Some(ref b_fine_fft) = b_fine_fft_opt {
            let path = format!("{}/error_comparison.png", diag_dir);
            let root = BitMapBackend::new(&path, (900, 550)).into_drawing_area();
            root.fill(&WHITE).unwrap();
            let b_max_g = b_fine_fft.data.iter()
                .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt()).fold(0.0f64, f64::max);

            // Collect per-level edge/bulk RMSE
            struct LvlErr { edge_cfft: f64, edge_comp: f64, bulk_cfft: f64, bulk_comp: f64, ne: usize, nb: usize }
            let n_levels = 1 + h.patches_l2plus.len();
            let mut lvl_errs: Vec<LvlErr> = (0..n_levels).map(|_| LvlErr { edge_cfft: 0.0, edge_comp: 0.0, bulk_cfft: 0.0, bulk_comp: 0.0, ne: 0, nb: 0 }).collect();

            // Helper closure for a single patch
            let measure_patch = |patch: &llg_sim::amr::patch::Patch2D, b_cfft_p: &[[f64; 3]], b_comp_p: &[[f64; 3]], lvl: &mut LvlErr| {
                let pnx = patch.grid.nx;
                let (gi0, gj0, gi1, gj1) = (patch.interior_i0(), patch.interior_j0(), patch.interior_i1(), patch.interior_j1());
                for j in gj0..gj1 { for i in gi0..gi1 {
                    let (x, y) = patch.cell_center_xy_centered(i, j, &base_grid);
                    if !shape.contains(x, y) { continue; }
                    let dist = (x - hole_centre.0).hypot(y - hole_centre.1) - hole_radius;
                    let is_edge = dist.abs() < edge_dist;
                    let (xc, yc) = patch.cell_center_xy(i, j);
                    let br = sample_bilinear(b_fine_fft, xc, yc);
                    let idx = j * pnx + i;
                    let ec = (b_cfft_p[idx][0]-br[0]).powi(2)+(b_cfft_p[idx][1]-br[1]).powi(2)+(b_cfft_p[idx][2]-br[2]).powi(2);
                    let ev = (b_comp_p[idx][0]-br[0]).powi(2)+(b_comp_p[idx][1]-br[1]).powi(2)+(b_comp_p[idx][2]-br[2]).powi(2);
                    if is_edge { lvl.edge_cfft += ec; lvl.edge_comp += ev; lvl.ne += 1; }
                    else { lvl.bulk_cfft += ec; lvl.bulk_comp += ev; lvl.nb += 1; }
                }}
            };

            // L1
            for (pi, patch) in h.patches.iter().enumerate() {
                if pi < b_l1_cfft.len() && pi < b_l1_comp.len() {
                    measure_patch(patch, &b_l1_cfft[pi], &b_l1_comp[pi], &mut lvl_errs[0]);
                }
            }
            // L2+
            for (lvl_idx, lvl_patches) in h.patches_l2plus.iter().enumerate() {
                let bc_lvl = if lvl_idx < b_l2_cfft.len() { &b_l2_cfft[lvl_idx] } else { continue };
                let bv_lvl = if lvl_idx < b_l2_comp.len() { &b_l2_comp[lvl_idx] } else { continue };
                for (pi, patch) in lvl_patches.iter().enumerate() {
                    if pi < bc_lvl.len() && pi < bv_lvl.len() {
                        measure_patch(patch, &bc_lvl[pi], &bv_lvl[pi], &mut lvl_errs[lvl_idx + 1]);
                    }
                }
            }

            // Build per-level bar data
            let mut bars: Vec<(String, f64, f64, f64, f64)> = Vec::new(); // (label, edge_cfft%, edge_comp%, bulk_cfft%, bulk_comp%)
            for (li, le) in lvl_errs.iter().enumerate() {
                let lnum = if li == 0 { 1 } else { li + 1 };
                let ep_c = if le.ne > 0 { (le.edge_cfft/le.ne as f64).sqrt()/b_max_g*100.0 } else { 0.0 };
                let ep_v = if le.ne > 0 { (le.edge_comp/le.ne as f64).sqrt()/b_max_g*100.0 } else { 0.0 };
                let bp_c = if le.nb > 0 { (le.bulk_cfft/le.nb as f64).sqrt()/b_max_g*100.0 } else { 0.0 };
                let bp_v = if le.nb > 0 { (le.bulk_comp/le.nb as f64).sqrt()/b_max_g*100.0 } else { 0.0 };
                if le.ne > 0 || le.nb > 0 {
                    bars.push((format!("L{}", lnum), ep_c, ep_v, bp_c, bp_v));
                }
            }

            let m = bars.iter().flat_map(|b| vec![b.1, b.2, b.3, b.4]).fold(0.0f64, f64::max) * 1.3;
            let x_max = (bars.len() as f64) * 4.0 + 1.0;

            let mut chart = ChartBuilder::on(&root)
                .caption("Per-Level RMSE (% of max|B|) vs Newell FFT", ("sans-serif", 18))
                .margin(15).x_label_area_size(45).y_label_area_size(55)
                .build_cartesian_2d(0.0f64..x_max, 0.0..m.max(1.0)).unwrap();
            chart.configure_mesh().y_desc("RMSE (%)").disable_x_mesh().draw().unwrap();

            for (bi, (label, ep_c, ep_v, bp_c, bp_v)) in bars.iter().enumerate() {
                let x0 = bi as f64 * 4.0 + 0.5;
                // Edge bars
                chart.draw_series(std::iter::once(Rectangle::new([(x0, 0.0), (x0+0.7, *ep_c)], GREEN.filled()))).unwrap();
                chart.draw_series(std::iter::once(Rectangle::new([(x0+0.8, 0.0), (x0+1.5, *ep_v)], RED.filled()))).unwrap();
                // Bulk bars
                chart.draw_series(std::iter::once(Rectangle::new([(x0+1.8, 0.0), (x0+2.5, *bp_c)], GREEN.mix(0.5).filled()))).unwrap();
                chart.draw_series(std::iter::once(Rectangle::new([(x0+2.6, 0.0), (x0+3.3, *bp_v)], RED.mix(0.5).filled()))).unwrap();
                // Label
                chart.draw_series(std::iter::once(plotters::element::Text::new(
                    format!("{} edge", label), (x0+0.3, -m*0.03), ("sans-serif", 11),
                ))).ok();
            }

            chart.draw_series(std::iter::once(Rectangle::new([(0.0, 0.0), (0.0, 0.0)], GREEN.filled()))).unwrap()
                .label("coarse-FFT edge").legend(|(x, y)| Rectangle::new([(x, y-5), (x+15, y+5)], GREEN.filled()));
            chart.draw_series(std::iter::once(Rectangle::new([(0.0, 0.0), (0.0, 0.0)], RED.filled()))).unwrap()
                .label("composite edge").legend(|(x, y)| Rectangle::new([(x, y-5), (x+15, y+5)], RED.filled()));
            chart.draw_series(std::iter::once(Rectangle::new([(0.0, 0.0), (0.0, 0.0)], GREEN.mix(0.5).filled()))).unwrap()
                .label("coarse-FFT bulk").legend(|(x, y)| Rectangle::new([(x, y-5), (x+15, y+5)], GREEN.mix(0.5).filled()));
            chart.draw_series(std::iter::once(Rectangle::new([(0.0, 0.0), (0.0, 0.0)], RED.mix(0.5).filled()))).unwrap()
                .label("composite bulk").legend(|(x, y)| Rectangle::new([(x, y-5), (x+15, y+5)], RED.mix(0.5).filled()));

            chart.configure_series_labels().background_style(WHITE.mix(0.8)).border_style(BLACK)
                .position(SeriesLabelPosition::UpperRight).draw().unwrap();
            root.present().unwrap();
            println!("    Wrote {}", path);
        }

        println!("  Plots written to {}/", diag_dir);
    } else if !do_sweep {
        println!("  (Use `-- --plots` for PNGs, `-- --sweep` for crossover study)");
    }

    println!();
    println!("  Total wall time: {:.1} s", wall);
    println!();
}
