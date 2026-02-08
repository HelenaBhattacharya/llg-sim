// src/bin/demag_2d.rs
//
// MuMax demag2D comparison (Nz=1):
//   Grid: 128x128x1
//   Cell: 1e-9, 1e-9, 0.5e-9
//   Ms = 1/MU0 so <B_demag> averages become dimensionless demag tensor entries,
//   matching MuMax's demag2D test convention.
//
// Run:
//   cargo run --release --bin demag_2d
//
// Output:
//   out/demag2D/rust_demag2D.csv
//
// Compare against MuMax output:
//   mumax_outputs/demag2D/table.txt
//
// MuMax side should append 3 rows using TableSave() after each Run(0).   [oai_citation:0‡mumax.github.io](https://mumax.github.io/api.html?utm_source=chatgpt.com)

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::Path;

use llg_sim::effective_field::demag::compute_demag_field;
use llg_sim::grid::Grid2D;
use llg_sim::params::{MU0, Material};
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
    // Match MuMax demag2D geometry
    let nx: usize = 128;
    let ny: usize = 128;
    let dx: f64 = 1e-9;
    let dy: f64 = 1e-9;
    let dz: f64 = 0.5e-9;

    // Ms = 1/mu0 so <B_demag> is dimensionless demag tensor entries
    let ms: f64 = 1.0 / MU0;

    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    let mat = Material {
        ms,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
    };

    let mut m = VectorField2D::new(grid);
    let mut b_demag = VectorField2D::new(grid);

    let out_dir = Path::new("out").join("demag2D");
    create_dir_all(&out_dir)?;
    let mut w = BufWriter::new(File::create(out_dir.join("rust_demag2D.csv"))?);

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

        writeln!(
            w,
            "{},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
            label, 0.0, mx, my, mz, bdx, bdy, bdz
        )?;
    }

    println!("Wrote out/demag2D/rust_demag2D.csv");
    Ok(())
}
