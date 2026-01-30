// -----------------------------------------------------------------------------
// Bloch wall with interfacial DMI (chirality benchmark)
//
// Tests:
//   - Chirality selection by DMI
//   - Chirality flip for D -> -D
//
// Run:
//   cargo run --release --bin bloch_dmi -- plus
//   cargo run --release --bin bloch_dmi -- minus
//
// Post-process:
// python3 scripts/bloch_dmi_analysis.py \
//   --dplus  out/bloch_dmi/Dplus \
//   --dminus out/bloch_dmi/Dminus
// Physics:
//   Exchange + uniaxial anisotropy + interfacial DMI
//   Demag: OFF
//
// Output:
//   out/bloch_dmi/Dplus/
//   out/bloch_dmi/Dminus/
// -----------------------------------------------------------------------------

use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use llg_sim::grid::Grid2D;
use llg_sim::llg::{step_llg_rk4_recompute_field_masked_relax, RK4Scratch};
use llg_sim::params::{GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;
use llg_sim::effective_field::FieldMask;

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

fn out_dir_for_sign(sign: &str) -> PathBuf {
    match sign {
        "plus"  => Path::new("out").join("bloch_dmi").join("Dplus"),
        "minus" => Path::new("out").join("bloch_dmi").join("Dminus"),
        _ => panic!("Expected 'plus' or 'minus'"),
    }
}

fn dmi_value_for_sign(sign: &str) -> f64 {
    let d0 = 1e-4;
    match sign {
        "plus"  =>  d0,
        "minus" => -d0,
        _ => panic!("Expected 'plus' or 'minus'"),
    }
}

fn main() -> std::io::Result<()> {
    let sign = std::env::args().nth(1).expect("Usage: bloch_dmi <plus|minus>");

    // Geometry
    let nx = 256;
    let ny = 64;
    let dx = 5e-9;
    let dy = 5e-9;
    let dz = 5e-9;

    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    // Material
    let material = Material {
        ms: 8.0e5,
        a_ex: 13e-12,
        k_u: 500.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: Some(dmi_value_for_sign(&sign)),
        demag: false,
    };

    // Initial Bloch wall (y–z rotation)
    let mut m = VectorField2D::new(grid);
    let x0 = 0.5 * nx as f64 * dx;
    let width = 40.0 * dx;
    m.init_bloch_wall_y(x0, width, 1.0);

    // Relaxation parameters
    let params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha: 0.5,
        dt: 2e-13,
        b_ext: [0.0, 0.0, 0.0],
    };

    let n_steps = (20e-9 / params.dt) as usize;
    let mut scratch = RK4Scratch::new(grid);

    let out_dir = out_dir_for_sign(&sign);
    create_dir_all(&out_dir)?;

    // Relax
    for _ in 0..n_steps {
        step_llg_rk4_recompute_field_masked_relax(
            &mut m,
            &params,
            &material,
            &mut scratch,
            FieldMask::ExchAnisDmi,
        );
    }

    // Save slice
    write_midrow_slice(&m, &grid, &out_dir.join("rust_slice_final.csv"))?;

    println!("Wrote Bloch–DMI slice to {:?}", out_dir);
    Ok(())
}