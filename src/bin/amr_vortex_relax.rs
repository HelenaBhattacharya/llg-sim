// src/bin/amr_vortex_relax.rs
//
// Stage-2B AMR benchmark: vortex relaxation (demag OFF).
// Purpose:
//   - De-risk clustering generality on a non-wall, non-bubble feature.
//   - Validate multi-patch clustering machinery behaves sensibly on ring-like gradients.
//
// Output: out/amr_vortex_relax/
//
// Notes:
// - Uses exchange + uniaxial anisotropy only (FieldMask::ExchAnis).
// - Uses initial_states::init_vortex.
// - Regrids using multi-patch clustering every N steps.

use llg_sim::amr::{
    compute_patch_rects_clustered_from_indicator, maybe_regrid_multi_patch, AmrHierarchy2D,
    AmrStepperRK4, ClusterPolicy, Connectivity, Rect2i, RegridPolicy,
};
use llg_sim::effective_field::FieldMask;
use llg_sim::grid::Grid2D;
use llg_sim::initial_states::init_vortex;
use llg_sim::llg::RK4Scratch;
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn ensure_out_dir() -> io::Result<PathBuf> {
    let out = PathBuf::from("out").join("amr_vortex_relax");
    fs::create_dir_all(&out)?;
    Ok(out)
}

fn write_run_info(
    path: &Path,
    base: &Grid2D,
    ratio: usize,
    ghost: usize,
    n_steps: usize,
    regrid_every: usize,
    vortex_center: (f64, f64),
    polarity: f64,
    chirality: f64,
    core_radius: f64,
    cluster_policy: ClusterPolicy,
    regrid_policy: RegridPolicy,
    params: &LLGParams,
    mat: &Material,
) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "AMR vortex relaxation benchmark (Stage 2B, demag OFF)")?;
    writeln!(f, "")?;

    writeln!(
        f,
        "Base grid: nx={} ny={} dx={:.6e} dy={:.6e} dz={:.6e}",
        base.nx, base.ny, base.dx, base.dy, base.dz
    )?;
    writeln!(
        f,
        "AMR: ratio={} ghost={} regrid_every_steps={}",
        ratio, ghost, regrid_every
    )?;
    writeln!(f, "Steps: {}", n_steps)?;
    writeln!(f, "")?;

    writeln!(f, "Vortex initial state:")?;
    writeln!(f, "  center:       ({:.6e}, {:.6e})", vortex_center.0, vortex_center.1)?;
    writeln!(f, "  polarity:     {:.6e}", polarity)?;
    writeln!(f, "  chirality:    {:.6e}", chirality)?;
    writeln!(f, "  core_radius:  {:.6e}", core_radius)?;
    writeln!(f, "")?;

    writeln!(f, "Clustering policy:")?;
    writeln!(f, "  indicator_frac:   {:.6e}", cluster_policy.indicator_frac)?;
    writeln!(f, "  buffer_cells:     {}", cluster_policy.buffer_cells)?;
    writeln!(f, "  connectivity:     {:?}", cluster_policy.connectivity)?;
    writeln!(f, "  merge_distance:   {}", cluster_policy.merge_distance)?;
    writeln!(f, "  min_patch_area:   {}", cluster_policy.min_patch_area)?;
    writeln!(f, "  max_patches:      {}", cluster_policy.max_patches)?;
    writeln!(f, "")?;

    writeln!(f, "Regrid hysteresis:")?;
    writeln!(f, "  min_change_cells:     {}", regrid_policy.min_change_cells)?;
    writeln!(f, "  min_area_change_frac: {:.6e}", regrid_policy.min_area_change_frac)?;
    writeln!(f, "")?;

    writeln!(f, "LLG params:")?;
    writeln!(f, "  gamma: {:.6e}", params.gamma)?;
    writeln!(f, "  alpha: {:.6e}", params.alpha)?;
    writeln!(f, "  dt:    {:.6e}", params.dt)?;
    writeln!(
        f,
        "  b_ext: [{:.6e}, {:.6e}, {:.6e}]",
        params.b_ext[0], params.b_ext[1], params.b_ext[2]
    )?;
    writeln!(f, "")?;

    writeln!(f, "Material:")?;
    writeln!(f, "  Ms:   {:.6e}", mat.ms)?;
    writeln!(f, "  Aex:  {:.6e}", mat.a_ex)?;
    writeln!(f, "  Ku:   {:.6e}", mat.k_u)?;
    writeln!(
        f,
        "  easy: [{:.6e}, {:.6e}, {:.6e}]",
        mat.easy_axis[0], mat.easy_axis[1], mat.easy_axis[2]
    )?;
    writeln!(f, "  demag: {}", mat.demag)?;
    writeln!(f, "  demag_method: {:?}", mat.demag_method)?;
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

    ((sum_sq / n).sqrt(), max_dm)
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
    let speedup = if amr_secs > 0.0 { uniform_secs / amr_secs } else { 0.0 };

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
    writeln!(
        f,
        "step,patch_id,i0,j0,nx,ny,max_indicator,threshold,flagged_cells,components,patches_after_merge"
    )?;
    Ok(())
}

fn append_regrid_log(
    path: &Path,
    step: usize,
    rects: &[Rect2i],
    stats: llg_sim::amr::ClusterStats,
) -> io::Result<()> {
    let mut f = fs::OpenOptions::new().append(true).open(path)?;
    for (pid, r) in rects.iter().enumerate() {
        writeln!(
            f,
            "{},{},{},{},{},{},{:.9e},{:.9e},{},{},{}",
            step,
            pid,
            r.i0,
            r.j0,
            r.nx,
            r.ny,
            stats.max_indicator,
            stats.threshold,
            stats.flagged_cells,
            stats.components,
            stats.patches_after_merge
        )?;
    }
    Ok(())
}

fn run_uniform_fine(
    fine_grid: Grid2D,
    params: &LLGParams,
    mat: &Material,
    n_steps: usize,
    vortex_center: (f64, f64),
    polarity: f64,
    chirality: f64,
    core_radius: f64,
) -> (VectorField2D, f64) {
    let mut m = VectorField2D::new(fine_grid);
    init_vortex(
        &mut m,
        &fine_grid,
        vortex_center,
        polarity,
        chirality,
        core_radius,
        None,
    );

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
    let dt = t0.elapsed();
    println!("[uniform] done in {:.2?}", dt);
    (m, dt.as_secs_f64())
}

fn main() -> io::Result<()> {
    // ----------------------------
    // 1) Grid / material / params
    // ----------------------------
    let base = Grid2D::new(192, 192, 5e-9, 5e-9, 1e-9);

    // Easy-plane-ish by making Ku negative with easy axis along +z.
    // This discourages unwinding via out-of-plane escape and makes vortex relaxation non-trivial.
    let mat = Material {
        ms: 8.0e5,
        a_ex: 13e-12,
        k_u: -1.0e5,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: false,
        demag_method: DemagMethod::FftUniform,
    };

    let n_steps = 2_000;
    let params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha: 0.5,
        dt: 5e-14,
        b_ext: [0.0, 0.0, 0.0],
    };

    // Vortex params (centered coordinates in meters)
    let vortex_center = (0.0, 0.0);
    let polarity = 1.0;
    let chirality = 1.0;
    let core_radius = 30e-9;

    // ----------------------------
    // 2) AMR settings (Stage 2B)
    // ----------------------------
    let ratio = 2;
    let ghost = 2;
    let regrid_every = 100;

    let cluster_policy = ClusterPolicy {
        indicator_frac: 0.30,
        buffer_cells: 6,
        connectivity: Connectivity::Eight,
        merge_distance: 4,
        min_patch_area: 16,
        max_patches: 8,
    };

    let regrid_policy = RegridPolicy {
        indicator_frac: cluster_policy.indicator_frac,
        buffer_cells: cluster_policy.buffer_cells,
        min_change_cells: 2,
        min_area_change_frac: 0.05,
    };

    // ----------------------------
    // 3) Initial coarse state
    // ----------------------------
    let mut m0 = VectorField2D::new(base);
    init_vortex(
        &mut m0,
        &base,
        vortex_center,
        polarity,
        chirality,
        core_radius,
        None,
    );

    // Initial patch set from clustering (fallback if needed)
    let fallback = Rect2i::new(base.nx / 2 - 32, base.ny / 2 - 32, 64, 64);
    let (mut patch_rects, init_stats) =
        match compute_patch_rects_clustered_from_indicator(&m0, cluster_policy) {
            Some((rs, st)) if !rs.is_empty() => (rs, st),
            _ => {
                let rs = vec![fallback];
                let st = llg_sim::amr::ClusterStats {
                    max_indicator: 0.0,
                    threshold: 0.0,
                    flagged_cells: 0,
                    components: 0,
                    patches_before_merge: 1,
                    patches_after_merge: 1,
                };
                (rs, st)
            }
        };

    println!("[init] max_indicator(coarse) = {:.9e}", init_stats.max_indicator);
    println!("[init] clustered patches: {}", patch_rects.len());
    for (k, r) in patch_rects.iter().enumerate() {
        println!("[init] patch {k}: i0={} j0={} nx={} ny={}", r.i0, r.j0, r.nx, r.ny);
    }

    // ----------------------------
    // 4) Outputs
    // ----------------------------
    let out_dir = ensure_out_dir()?;

    write_run_info(
        &out_dir.join("run_info.txt"),
        &base,
        ratio,
        ghost,
        n_steps,
        regrid_every,
        vortex_center,
        polarity,
        chirality,
        core_radius,
        cluster_policy,
        regrid_policy,
        &params,
        &mat,
    )?;

    let regrid_log = out_dir.join("regrid_log.csv");
    init_regrid_log(&regrid_log)?;
    append_regrid_log(&regrid_log, 0, &patch_rects, init_stats)?;

    // ----------------------------
    // 5) Run AMR
    // ----------------------------
    let mut h_amr = AmrHierarchy2D::new(base, m0, ratio, ghost);
    for r in &patch_rects {
        h_amr.add_patch(*r);
    }

    let mut stepper = AmrStepperRK4::new(&h_amr, true);

    let t0 = Instant::now();
    for step in 0..n_steps {
        stepper.step(&mut h_amr, &params, &mat, FieldMask::ExchAnis);

        if step % 200 == 0 {
            println!("[amr] step {step}/{n_steps}");
        }

        if regrid_every > 0 && step > 0 && (step % regrid_every == 0) {
            if let Some((new_rects, stats)) =
                maybe_regrid_multi_patch(&mut h_amr, &patch_rects, regrid_policy, cluster_policy)
            {
                patch_rects = new_rects;
                append_regrid_log(&regrid_log, step, &patch_rects, stats)?;
            }
        }
    }
    let dt_amr = t0.elapsed();
    let amr_secs = dt_amr.as_secs_f64();
    println!("[amr] done in {:.2?}", dt_amr);

    println!("[final] patches: {}", patch_rects.len());
    for (k, r) in patch_rects.iter().enumerate() {
        println!("[final] patch {k}: i0={} j0={} nx={} ny={}", r.i0, r.j0, r.nx, r.ny);
    }

    let fine_amr = h_amr.flatten_to_uniform_fine();

    // ----------------------------
    // 6) Uniform-fine reference
    // ----------------------------
    let fine_grid = fine_amr.grid;
    let (fine_uniform, uniform_secs) = run_uniform_fine(
        fine_grid,
        &params,
        &mat,
        n_steps,
        vortex_center,
        polarity,
        chirality,
        core_radius,
    );

    // ----------------------------
    // 7) Metrics + outputs
    // ----------------------------
    write_csv_field(&out_dir.join("amr_fine_final.csv"), &fine_amr)?;
    write_csv_field(&out_dir.join("uniform_fine_final.csv"), &fine_uniform)?;
    write_lineout_mid_y(&out_dir.join("lineout_amr_mid_y.csv"), &fine_amr)?;
    write_lineout_mid_y(&out_dir.join("lineout_uniform_mid_y.csv"), &fine_uniform)?;

    let (rmse, max_dm) = rmse_and_max_delta(&fine_amr, &fine_uniform);

    let mut f = fs::File::create(out_dir.join("metrics.txt"))?;
    writeln!(f, "RMSE(|Δm|_2): {:.6e}", rmse)?;
    writeln!(f, "max |Δm|:     {:.6e}", max_dm)?;

    write_efficiency_metrics(
        &out_dir.join("efficiency.txt"),
        &base,
        ratio,
        ghost,
        &patch_rects,
        amr_secs,
        uniform_secs,
    )?;

    let uniform_fine_cells = ((base.nx * ratio) * (base.ny * ratio)) as f64;
    let fine_active_cells: f64 = patch_rects
        .iter()
        .map(|r| (r.nx * ratio) as f64 * (r.ny * ratio) as f64)
        .sum();
    let coverage = if uniform_fine_cells > 0.0 { fine_active_cells / uniform_fine_cells } else { 0.0 };
    let speedup = if amr_secs > 0.0 { uniform_secs / amr_secs } else { 0.0 };

    println!("[perf] active coverage = {:.6e}", coverage);
    println!("[perf] speedup (uniform/amr) = {:.6e}", speedup);
    println!("[metrics] RMSE(|Δm|_2) = {:.6e}", rmse);
    println!("[metrics] max |Δm|     = {:.6e}", max_dm);
    println!("[output] wrote {}", out_dir.display());

    Ok(())
}