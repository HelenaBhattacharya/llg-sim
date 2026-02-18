// src/bin/amr_bloch_relax.rs
//
// AMR Bloch-wall relaxation benchmark (demag OFF)
// ------------------------------------------------
// Purpose:
//   Stage-1/2 AMR validation harness for local-stencil micromagnetics terms.
//   Runs:
//     (1) 2-level AMR hierarchy: coarse + dynamic refined patch (Stage 2A)
//     (2) uniform-fine reference run at the same finest resolution
//   Outputs:
//     out/amr_bloch_relax/
//       - amr_fine_final.csv
//       - uniform_fine_final.csv
//       - lineout_amr_mid_y.csv
//       - lineout_uniform_mid_y.csv
//       - metrics.txt
//       - efficiency.txt
//       - regrid_log.csv
//       - run_info.txt
//
// Notes:
//   - This benchmark uses FieldMask::ExchAnis (exchange + uniaxial anisotropy).
//   - Ghost fill is performed once per timestep (Stage-1 simplification).
//   - Dynamic AMR (Stage 2A): every N steps, compute a coarse-grid indicator and
//     rebuild the level-1 patch as a single buffered bounding box.
//   - A small +z bias field (b_ext) is applied to drive wall motion so patch tracking is exercised.

use llg_sim::amr::{AmrHierarchy2D, AmrStepperRK4, Rect2i};
use llg_sim::amr::{RegridPolicy, compute_patch_bbox_from_indicator, maybe_regrid_single_patch};
use llg_sim::effective_field::FieldMask;
use llg_sim::grid::Grid2D;
use llg_sim::llg::RK4Scratch;
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn ensure_out_dir() -> io::Result<PathBuf> {
    let out = PathBuf::from("out").join("amr_bloch_relax");
    fs::create_dir_all(&out)?;
    Ok(out)
}

fn write_run_info(
    path: &Path,
    base: &Grid2D,
    ratio: usize,
    ghost: usize,
    n_steps: usize,
    params: &LLGParams,
    mat: &Material,
    regrid_every: usize,
    indicator_frac: f64,
    buffer_cells: usize,
    init_patch: Rect2i,
) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "AMR Bloch-wall relaxation benchmark (demag OFF)")?;
    writeln!(f, "")?;
    writeln!(
        f,
        "Base grid: nx={} ny={} dx={:.6e} dy={:.6e} dz={:.6e}",
        base.nx, base.ny, base.dx, base.dy, base.dz
    )?;
    writeln!(f, "AMR: ratio={} ghost={}", ratio, ghost)?;
    writeln!(f, "Steps: {}", n_steps)?;
    writeln!(f, "")?;
    writeln!(f, "Dynamic regrid (Stage 2A):")?;
    writeln!(f, "  regrid_every_steps = {}", regrid_every)?;
    writeln!(f, "  indicator_frac     = {:.6e}", indicator_frac)?;
    writeln!(f, "  buffer_cells       = {}", buffer_cells)?;
    writeln!(f, "")?;
    writeln!(
        f,
        "Initial patch (coarse): [i0={}, j0={}, nx={}, ny={}]",
        init_patch.i0, init_patch.j0, init_patch.nx, init_patch.ny
    )?;
    writeln!(f, "")?;
    writeln!(
        f,
        "LLG params: gamma={:.6e} alpha={:.6e} dt={:.6e}",
        params.gamma, params.alpha, params.dt
    )?;
    writeln!(
        f,
        "Bext: [{:.6e}, {:.6e}, {:.6e}]",
        params.b_ext[0], params.b_ext[1], params.b_ext[2]
    )?;
    writeln!(f, "")?;
    writeln!(f, "Material:")?;
    writeln!(f, "  Ms   = {:.6e}", mat.ms)?;
    writeln!(f, "  Aex  = {:.6e}", mat.a_ex)?;
    writeln!(f, "  Ku   = {:.6e}", mat.k_u)?;
    writeln!(
        f,
        "  easy = [{:.6e}, {:.6e}, {:.6e}]",
        mat.easy_axis[0], mat.easy_axis[1], mat.easy_axis[2]
    )?;
    writeln!(f, "  demag = {}", mat.demag)?;
    writeln!(f, "  demag_method = {:?}", mat.demag_method)?;
    Ok(())
}

fn write_csv_field(path: &Path, field: &VectorField2D) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "i,j,x_m,y_m,mx,my,mz")?;
    let nx = field.grid.nx;
    let ny = field.grid.ny;
    for j in 0..ny {
        let y = (j as f64 + 0.5) * field.grid.dy;
        for i in 0..nx {
            let x = (i as f64 + 0.5) * field.grid.dx;
            let idx = field.idx(i, j);
            let v = field.data[idx];
            writeln!(
                f,
                "{},{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
                i, j, x, y, v[0], v[1], v[2]
            )?;
        }
    }
    Ok(())
}

fn write_lineout_mid_y(path: &Path, field: &VectorField2D) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "i,x_m,mx,my,mz")?;
    let nx = field.grid.nx;
    let j = field.grid.ny / 2;
    for i in 0..nx {
        let x = (i as f64 + 0.5) * field.grid.dx;
        let idx = field.idx(i, j);
        let v = field.data[idx];
        writeln!(f, "{},{:.9e},{:.9e},{:.9e},{:.9e}", i, x, v[0], v[1], v[2])?;
    }
    Ok(())
}

fn rmse_and_max_delta(a: &VectorField2D, b: &VectorField2D) -> (f64, f64) {
    assert_eq!(a.grid.nx, b.grid.nx);
    assert_eq!(a.grid.ny, b.grid.ny);
    let mut sum_sq = 0.0_f64;
    let mut max_dm = 0.0_f64;
    let n = a.grid.n_cells() as f64;

    for idx in 0..a.grid.n_cells() {
        let va = a.data[idx];
        let vb = b.data[idx];
        let dx = va[0] - vb[0];
        let dy = va[1] - vb[1];
        let dz = va[2] - vb[2];
        let dm2 = dx * dx + dy * dy + dz * dz;
        sum_sq += dm2;
        let dm = dm2.sqrt();
        if dm > max_dm {
            max_dm = dm;
        }
    }

    let rmse = (sum_sq / n).sqrt();
    (rmse, max_dm)
}

fn write_efficiency_metrics(
    path: &Path,
    base: &Grid2D,
    ratio: usize,
    ghost: usize,
    patches: &[Rect2i],
    amr_secs: f64,
    uniform_secs: f64,
) -> io::Result<()> {
    let coarse_cells = (base.nx * base.ny) as u64;
    let uniform_fine_cells = ((base.nx * ratio) * (base.ny * ratio)) as u64;

    let mut fine_total_cells: u64 = 0;
    let mut fine_active_cells: u64 = 0;
    for p in patches {
        let interior_nx = p.nx * ratio;
        let interior_ny = p.ny * ratio;
        let nx_tot = interior_nx + 2 * ghost;
        let ny_tot = interior_ny + 2 * ghost;
        fine_total_cells += (nx_tot * ny_tot) as u64;
        fine_active_cells += (interior_nx * interior_ny) as u64;
    }

    let coverage = if uniform_fine_cells > 0 {
        fine_active_cells as f64 / uniform_fine_cells as f64
    } else {
        0.0
    };
    let speedup = if amr_secs > 0.0 {
        uniform_secs / amr_secs
    } else {
        0.0
    };

    let mut f = fs::File::create(path)?;
    writeln!(f, "coarse_cells:          {}", coarse_cells)?;
    writeln!(f, "uniform_fine_cells:    {}", uniform_fine_cells)?;
    writeln!(f, "patch_count:           {}", patches.len())?;
    writeln!(f, "fine_active_cells:     {}", fine_active_cells)?;
    writeln!(f, "fine_total_cells:      {}", fine_total_cells)?;
    writeln!(f, "active_coverage_frac:  {:.6e}", coverage)?;
    writeln!(f, "amr_time_s:            {:.6e}", amr_secs)?;
    writeln!(f, "uniform_time_s:        {:.6e}", uniform_secs)?;
    writeln!(f, "speedup_uniform_over_amr: {:.6e}", speedup)?;
    Ok(())
}

fn init_regrid_log(path: &Path) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "step,i0,j0,nx,ny,max_indicator,threshold")?;
    Ok(())
}

fn append_regrid_log(
    path: &Path,
    step: usize,
    rect: Rect2i,
    max_ind: f64,
    thresh: f64,
) -> io::Result<()> {
    let mut f = fs::OpenOptions::new().append(true).open(path)?;
    writeln!(
        f,
        "{},{},{},{},{},{:.9e},{:.9e}",
        step, rect.i0, rect.j0, rect.nx, rect.ny, max_ind, thresh
    )?;
    Ok(())
}

fn run_uniform_fine(
    fine_grid: Grid2D,
    params: &LLGParams,
    mat: &Material,
    n_steps: usize,
) -> (VectorField2D, f64) {
    let mut m = VectorField2D::new(fine_grid);

    // Same physical initial condition.
    let lx = fine_grid.nx as f64 * fine_grid.dx;
    let x0 = 0.5 * lx;
    let width = (mat.a_ex / mat.k_u).sqrt();
    m.init_bloch_wall_y(x0, width, 1.0);

    let mut scratch = RK4Scratch::new(fine_grid);

    let t0 = Instant::now();
    for step in 0..n_steps {
        llg_sim::llg::step_llg_rk4_recompute_field_masked_relax(
            &mut m,
            params,
            mat,
            &mut scratch,
            FieldMask::ExchAnis,
        );
        if step % 200 == 0 {
            println!("[uniform] step {step}/{n_steps}");
        }
    }
    let dt_wall = t0.elapsed();
    println!("[uniform] done in {:.2?}", dt_wall);

    (m, dt_wall.as_secs_f64())
}

fn main() -> io::Result<()> {
    // ----------------------------
    // 1) Base grid + material/params
    // ----------------------------
    let base = Grid2D::new(
        256,  // nx
        64,   // ny
        5e-9, // dx
        5e-9, // dy
        1e-9, // dz
    );

    // Simple Permalloy-ish parameters (demag OFF for this test).
    let mat = Material {
        ms: 8.0e5,
        a_ex: 13e-12,
        k_u: 500.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: false,
        // Keep a value here for forward compatibility even though demag=false.
        demag_method: DemagMethod::FftUniform,
    };

    let n_steps = 2_000;
    let params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha: 0.5,
        dt: 5e-14,
        b_ext: [0.0, 0.0, 5e-3],
    };

    // ----------------------------
    // 2) AMR hierarchy definition (Stage 2A)
    // ----------------------------
    let ratio = 2;
    let ghost = 2;

    // Dynamic regrid settings
    let regrid_every = 100;
    let indicator_frac = 0.35;
    let buffer_cells = 2;

    // Hysteresis: only rebuild the patch if it changes materially.
    let min_change_cells: usize = 2;
    let min_area_change_frac: f64 = 0.10;

    let policy = RegridPolicy {
        indicator_frac,
        buffer_cells,
        min_change_cells,
        min_area_change_frac,
    };

    // Initial coarse state
    let mut m0 = VectorField2D::new(base);
    let lx = base.nx as f64 * base.dx;
    let x0 = 0.5 * lx;
    let width = (mat.a_ex / mat.k_u).sqrt();
    m0.init_bloch_wall_y(x0, width, 1.0);

    // Initial patch guess: indicator-based, else fallback to a centered strip.
    let fallback_patch_nx = 64;
    let fallback_i0 = base.nx / 2 - fallback_patch_nx / 2;
    let fallback_patch = Rect2i::new(fallback_i0, 0, fallback_patch_nx, base.ny);

    let (init_patch, init_stats) =
        match compute_patch_bbox_from_indicator(&m0, indicator_frac, buffer_cells) {
            Some((r, stats)) => (r, Some(stats)),
            None => (fallback_patch, None),
        };

    // ----------------------------
    // 3) Outputs + logs
    // ----------------------------
    let out_dir = ensure_out_dir()?;

    write_run_info(
        &out_dir.join("run_info.txt"),
        &base,
        ratio,
        ghost,
        n_steps,
        &params,
        &mat,
        regrid_every,
        indicator_frac,
        buffer_cells,
        init_patch,
    )?;

    let regrid_log = out_dir.join("regrid_log.csv");
    init_regrid_log(&regrid_log)?;
    // record initial patch selection if it came from indicator
    if let Some(stats) = init_stats {
        append_regrid_log(&regrid_log, 0, init_patch, stats.max, stats.threshold)?;
    }

    // ----------------------------
    // 4) Run AMR with dynamic regridding (single patch)
    // ----------------------------
    let mut h = AmrHierarchy2D::new(base, m0, ratio, ghost);
    h.add_patch(init_patch);

    let mut stepper = AmrStepperRK4::new(&h, true);
    let mut current_patch = init_patch;

    let t0 = Instant::now();
    for step in 0..n_steps {
        stepper.step(&mut h, &params, &mat, FieldMask::ExchAnis);

        if step % 200 == 0 {
            println!("[amr] step {step}/{n_steps}");
        }

        // Periodic regrid (Stage 2A)
        if regrid_every > 0 && step > 0 && (step % regrid_every == 0) {
            if let Some((new_rect, stats)) =
                maybe_regrid_single_patch(&mut h, current_patch, policy)
            {
                current_patch = new_rect;
                append_regrid_log(&regrid_log, step, new_rect, stats.max, stats.threshold)?;
            }
        }
    }
    let dt_amr = t0.elapsed();
    let amr_secs = dt_amr.as_secs_f64();
    println!("[amr] done in {:.2?}", dt_amr);

    let fine_amr = h.flatten_to_uniform_fine();

    // Uniform fine reference at the same finest resolution as the flattened AMR field.
    let fine_grid = fine_amr.grid;
    let (fine_uniform, uniform_secs) = run_uniform_fine(fine_grid, &params, &mat, n_steps);

    // ----------------------------
    // 5) Outputs + metrics
    // ----------------------------
    write_csv_field(&out_dir.join("amr_fine_final.csv"), &fine_amr)?;
    write_csv_field(&out_dir.join("uniform_fine_final.csv"), &fine_uniform)?;
    write_lineout_mid_y(&out_dir.join("lineout_amr_mid_y.csv"), &fine_amr)?;
    write_lineout_mid_y(&out_dir.join("lineout_uniform_mid_y.csv"), &fine_uniform)?;

    let (rmse, max_dm) = rmse_and_max_delta(&fine_amr, &fine_uniform);

    let mut f = fs::File::create(out_dir.join("metrics.txt"))?;
    writeln!(f, "RMSE(|Δm|_2): {:.6e}", rmse)?;
    writeln!(f, "max |Δm|:     {:.6e}", max_dm)?;

    let patches = vec![current_patch];
    let uniform_fine_cells = (base.nx * ratio) as f64 * (base.ny * ratio) as f64;
    let fine_active_cells = (current_patch.nx * ratio) as f64 * (current_patch.ny * ratio) as f64;
    let coverage = if uniform_fine_cells > 0.0 {
        fine_active_cells / uniform_fine_cells
    } else {
        0.0
    };
    let speedup = if amr_secs > 0.0 {
        uniform_secs / amr_secs
    } else {
        0.0
    };

    println!("[metrics] RMSE(|Δm|_2) = {:.6e}", rmse);
    println!("[metrics] max |Δm|     = {:.6e}", max_dm);
    println!("[metrics] active coverage   = {:.6e}", coverage);
    println!("[metrics] speedup (uniform/amr) = {:.6e}", speedup);
    println!("[output] wrote {}", out_dir.display());

    write_efficiency_metrics(
        &out_dir.join("efficiency.txt"),
        &base,
        ratio,
        ghost,
        &patches,
        amr_secs,
        uniform_secs,
    )?;
    println!(
        "[efficiency] wrote {}",
        out_dir.join("efficiency.txt").display()
    );

    // Append final patch info to run_info for convenience.
    {
        let mut rf = fs::OpenOptions::new()
            .append(true)
            .open(out_dir.join("run_info.txt"))?;
        writeln!(rf, "")?;
        writeln!(
            rf,
            "Final patch (coarse): [i0={}, j0={}, nx={}, ny={}]",
            current_patch.i0, current_patch.j0, current_patch.nx, current_patch.ny
        )?;
    }

    Ok(())
}
