// -----------------------------------------------------------------------------
// Relaxation benchmark: near-uniform tilt -> relaxed ground state (MuMax vs Rust)
//
// Purpose:
//   Validate that Rust's MuMax-like relax() controller converges to the same
//   final equilibrium as MuMax3 Relax(), for a non-degenerate problem.
//   (A small Bz bias removes the ±z degeneracy so "same ground state" is well-defined.)
//
// Physics:
//   - Exchange + uniaxial anisotropy
//   - No demag, no DMI
//   - Strong damping (precession suppressed in relax RHS)
//   - Small symmetry-breaking field: B_ext = (0, 0, 1e-6) T
//
// Initial condition:
//   Near-uniform tilt from +z:
//     m(t=0) = (eps, 0, sqrt(1-eps^2)), eps=0.05
//
// RUN (RUST)
//   cargo run --release --bin relax_uniform_noisy
//
// Outputs (Rust):
//   out/relax_uniform_noisy/
//     └── rust_slice_final.csv         # mid-row slice: x,mx,my,mz
//
// POST-PROCESS (RUST vs MUMAX OVERLAY)
//   python3 scripts/overlay_relax.py
//
// This writes comparison plots to:
//   out/relax_uniform_noisy/
//     ├── relax_uniform_overlay_xy.png       # mx,my overlays (mid-row)
//     ├── relax_uniform_overlay_mzdev.png    # (1-mz) overlays (mid-row)
//     ├── relax_uniform_overlay_deltas.png   # Δmx,Δmy,Δmz (Rust - MuMax)
//     └── relax_uniform_overlay_mperp.png    # |m_perp| overlays (mid-row)
//
// The script also prints slice-level L_inf and RMS errors to confirm agreement.
// -----------------------------------------------------------------------------

use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use llg_sim::effective_field::FieldMask;
use llg_sim::grid::Grid2D;
use llg_sim::llg::RK23Scratch;
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::relax::{relax, RelaxSettings};
use llg_sim::vec3::cross;
use llg_sim::effective_field::build_h_eff_masked;
use llg_sim::vector_field::VectorField2D;

fn write_midrow_slice(m: &VectorField2D, grid: &Grid2D, path: &Path) -> std::io::Result<()> {
    let j = grid.ny / 2;
    let mut f = BufWriter::new(File::create(path)?);

    writeln!(f, "x,mx,my,mz")?;
    for i in 0..grid.nx {
        let idx = grid.idx(i, j);
        let x = (i as f64 + 0.5) * grid.dx;
        let v = m.data[idx];
        writeln!(f, "{:.6e},{:.6e},{:.6e},{:.6e}", x, v[0], v[1], v[2])?;
    }
    Ok(())
}

/// max |m x B_eff| over grid (Tesla) for reporting convergence
fn max_torque_inf(grid: &Grid2D, m: &VectorField2D, params: &LLGParams, material: &Material, mask: FieldMask) -> f64 {
    let mut b_eff = VectorField2D::new(*grid);
    build_h_eff_masked(grid, m, &mut b_eff, params, material, mask);

    let mut maxv = 0.0;
    for (mi, bi) in m.data.iter().zip(b_eff.data.iter()) {
        let t = cross(*mi, *bi);
        let mag = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        if mag > maxv {
            maxv = mag;
        }
    }
    maxv
}

fn main() -> std::io::Result<()> {
    // ---------------- Geometry ----------------
    let nx = 64;
    let ny = 64;
    let dx = 5e-9;
    let dy = 5e-9;
    let dz = 5e-9;

    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    // ---------------- Material ----------------
    let material = Material {
        ms: 8.0e5,
        a_ex: 13e-12,
        k_u: 500.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: false,
    };

    // ---------------- Initial condition ----------------
    let eps:f64 = 0.05; // tilt magnitude
    let mut m = VectorField2D::new(grid);

    // Near-uniform tilted state (matches MuMax IC)
    let mx0:f64 = eps;
    let my0:f64 = 0.0;
    let mz0:f64 = (1.0 - eps * eps).sqrt();
    m.set_uniform(mx0, my0, mz0);

    // ---------------- Relaxation params ----------------
    // NOTE: relax() uses llg_rhs_relax internally (precession suppressed),
    // but it still needs params.alpha, params.gamma, params.b_ext, params.dt.
    let mut params_relax = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha: 0.5,
        dt: 2e-13,                 // initial dt guess (adaptive will adjust)
        b_ext: [0.0, 0.0, 1e-6],   // tiny bias to remove ±z degeneracy
    };

    let mut rk23 = RK23Scratch::new(grid);

    // MuMax-like relax settings
    let mut settings = RelaxSettings {
        // For this simple problem we can demand a tighter torque than SP4
        torque_threshold: Some(1e-6),
        dt_min: 1e-18,
        dt_max: 1e-11,
        max_err: 1e-5,
        headroom: 0.8,
        tighten_floor: 1e-9,
        ..Default::default()
    };

    let t0 = max_torque_inf(&grid, &m, &params_relax, &material, FieldMask::ExchAnis);
    println!("relax_uniform_noisy: start max|m×B| = {:.3e} T", t0);

    // Run MuMax-like relaxation controller
    relax(
        &grid,
        &mut m,
        &mut params_relax,
        &material,
        &mut rk23,
        FieldMask::ExchAnis,
        &mut settings,
    );

    let t1 = max_torque_inf(&grid, &m, &params_relax, &material, FieldMask::ExchAnis);
    println!(
        "relax_uniform_noisy: done  max|m×B| = {:.3e} T  (final max_err={:.3e}, dt={:.3e})",
        t1,
        settings.max_err,
        params_relax.dt
    );

    // ---------------- Output ----------------
    let out_dir = Path::new("out").join("relax_uniform_noisy");
    create_dir_all(&out_dir)?;

    write_midrow_slice(&m, &grid, &out_dir.join("rust_slice_final.csv"))?;
    println!("Wrote final relaxed slice to {:?}", out_dir);

    Ok(())
}