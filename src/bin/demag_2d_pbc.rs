// src/bin/demag_2d_pbc.rs
//
// MuMax PBC demag comparison (Nz=1), periodic along X:
//   SetPBC(32, 0, 0)
//   Two geometries: (Nx,Ny,Nz) = (2,128,1) and (3,128,1)
//   Cell: (1e-9, 1e-9, 0.5e-9)
//
// Ms = 1/MU0 so <B_demag> averages are dimensionless demag tensor entries,
// matching MuMax test convention.
//
// Run:
//   cargo run --release --bin demag_2d_pbc
//
// Output:
//   out/demag2Dpbc/rust_demag2Dpbc.csv
//
// Compare against:
//   mumax_outputs/demag2Dpbc/table.txt
//
// Note: requires compute_demag_field_pbc(...) in effective_field::demag.

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;

use llg_sim::effective_field::demag::compute_demag_field_pbc;
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, MU0, Material};
use llg_sim::vector_field::VectorField2D;

fn avg_vec(field: &VectorField2D) -> [f64; 3] {
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    let n = field.data.len() as f64;
    for v in &field.data {
        sx += v[0];
        sy += v[1];
        sz += v[2];
    }
    [sx / n, sy / n, sz / n]
}

fn run_block(
    w: &mut BufWriter<File>,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    pbc_x: usize,
    pbc_y: usize,
    ms: f64,
) -> std::io::Result<()> {
    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    let mat = Material {
        ms,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let mut m = VectorField2D::new(grid);
    let mut b_demag = VectorField2D::new(grid);

    let cases = [
        ("mx", 1.0, 0.0, 0.0),
        ("my", 0.0, 1.0, 0.0),
        ("mz", 0.0, 0.0, 1.0),
    ];

    for (case, mx, my, mz) in cases {
        m.set_uniform(mx, my, mz);

        compute_demag_field_pbc(&grid, &m, &mut b_demag, &mat, pbc_x, pbc_y);
        let [bdx, bdy, bdz] = avg_vec(&b_demag);

        writeln!(
            w,
            "{},{},{},{},{},{},{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            nx, ny, 1, pbc_x, pbc_y, case, 0.0, mx, my, mz, bdx, bdy, bdz
        )?;
    }

    Ok(())
}

fn main() -> std::io::Result<()> {
    // Geometry + PBC match MuMax script
    let dx: f64 = 1e-9;
    let dy: f64 = 1e-9;
    let dz: f64 = 0.5e-9;

    let pbc_x: usize = 32;
    let pbc_y: usize = 0;

    // Ms = 1/mu0 for dimensionless demag tensor entries
    let ms: f64 = 1.0 / MU0;

    let out_dir = Path::new("out").join("demag2Dpbc");
    create_dir_all(&out_dir)?;
    let mut w = BufWriter::new(File::create(out_dir.join("rust_demag2Dpbc.csv"))?);

    // CSV header
    writeln!(
        w,
        "nx,ny,nz,pbc_x,pbc_y,case,t,mx,my,mz,B_demagx,B_demagy,B_demagz"
    )?;

    // Block 1: Nx=2, Ny=128
    run_block(&mut w, 2, 128, dx, dy, dz, pbc_x, pbc_y, ms)?;

    // Block 2: Nx=3, Ny=128 (odd Nx regression)
    run_block(&mut w, 3, 128, dx, dy, dz, pbc_x, pbc_y, ms)?;

    println!("Wrote out/demag2Dpbc/rust_demag2Dpbc.csv");
    Ok(())
}
