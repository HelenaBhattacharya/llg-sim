// src/bin/demag_small.rs
//
// MuMax demagSmall comparison (Nz=1):
//   Grid: 3x4x1
//   Cell: 1e-9, 2e-9, 0.5e-9
//   Ms = 1/MU0 so <B_demag> averages are dimensionless demag tensor entries.
//
// Run:
//   cargo run --release --bin demag_small
//
// Output:
//   out/demagSmall/rust_demagSmall.csv
//
// Compare against:
//   mumax_outputs/demagSmall/table.txt

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;

use llg_sim::effective_field::demag::compute_demag_field;
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

fn main() -> std::io::Result<()> {
    // Match MuMax demagSmall geometry
    let nx: usize = 3;
    let ny: usize = 4;
    let dx: f64 = 1e-9;
    let dy: f64 = 2e-9;
    let dz: f64 = 0.5e-9;

    // Ms = 1/mu0 so that <B_demag> is dimensionless demag tensor entries
    let ms: f64 = 1.0 / MU0;

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

    let out_dir = Path::new("out").join("demagSmall");
    create_dir_all(&out_dir)?;
    let mut w = BufWriter::new(File::create(out_dir.join("rust_demagSmall.csv"))?);

    // Write a simple 3-row CSV (one per case), matching the MuMax TableSave idea.
    writeln!(w, "case,t,mx,my,mz,B_demagx,B_demagy,B_demagz")?;

    let cases = [
        ("mx", 1.0, 0.0, 0.0),
        ("my", 0.0, 1.0, 0.0),
        ("mz", 0.0, 0.0, 1.0),
    ];

    for (label, mx, my, mz) in cases {
        m.set_uniform(mx, my, mz);

        compute_demag_field(&grid, &m, &mut b_demag, &mat);
        let [bdx, bdy, bdz] = avg_vec(&b_demag);

        // t=0 because this is a static operator evaluation
        writeln!(
            w,
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            label, 0.0, mx, my, mz, bdx, bdy, bdz
        )?;
    }

    println!("Wrote out/demagSmall/rust_demagSmall.csv");
    Ok(())
}
