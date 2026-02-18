// src/bin/demag_snapshot.rs
//
// Static demag snapshot test: compute spatially averaged B_demag and B_eff for
// three uniform magnetisation cases (mx, my, mz) on the same grid as MuMax.
//
// Run:
//   cargo run --release --bin demag_snapshot
//
// Outputs (Rust):
//   out/demag_snapshot/
//     ├── rust_demag_snapshot_x.csv
//     ├── rust_demag_snapshot_y.csv
//     └── rust_demag_snapshot_z.csv
//
// Each CSV contains a single row at t=0 with columns:
//   t,mx,my,mz,B_demagx,B_demagy,B_demagz,B_effx,B_effy,B_effz
//
// Post-process (Rust vs MuMax overlay examples):
// python3 scripts/overlay_macrospin.py \
//   out/demag_snapshot/rust_demag_snapshot_x.csv \
//   mumax_outputs/demag_snapshot_x/table.txt \
//   --col B_demagx --metrics \
//   --out out/demag_snapshot/overlay_x_B_demagx.png
//
//   python3 scripts/overlay_macrospin.py \
//     out/demag_snapshot/rust_demag_snapshot_z.csv \
//     mumax_outputs/demag_snapshot_z/table.txt \
//     --col B_demagz --clip_overlap --metrics \
//     --out out/demag_snapshot/overlay_z_B_demagz.png

use llg_sim::effective_field::demag::compute_demag_field;
use llg_sim::effective_field::{FieldMask, build_h_eff_masked};
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, GAMMA_E_RAD_PER_S_T, LLGParams, Material};
use llg_sim::vector_field::VectorField2D;
use std::fs::File;
use std::fs::create_dir_all;
use std::io::{BufWriter, Write};
use std::path::Path;

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

fn write_snapshot_csv(
    path: &Path,
    mx: f64,
    my: f64,
    mz: f64,
    b_demag: [f64; 3],
    b_eff: [f64; 3],
) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(
        f,
        "t,mx,my,mz,B_demagx,B_demagy,B_demagz,B_effx,B_effy,B_effz"
    )?;
    writeln!(
        f,
        "{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e},{:.16e}",
        0.0, mx, my, mz, b_demag[0], b_demag[1], b_demag[2], b_eff[0], b_eff[1], b_eff[2]
    )?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    // Match MuMax snapshot scripts
    let nx: usize = 128;
    let ny: usize = 128;
    let dx: f64 = 5e-9;
    let dy: f64 = 5e-9;
    let dz: f64 = 5e-9;

    let ms: f64 = 8.0e5; // A/m
    let a_ex: f64 = 13e-12; // J/m
    let k_u: f64 = 500.0; // J/m^3
    let easy_axis = [0.0, 0.0, 1.0];

    // External field must match MuMax if you want B_eff to match exactly.
    // In the snapshots we didn't specify B_ext explicitly; MuMax default is 0 unless set.
    let b_ext = [0.0, 0.0, 0.0];

    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    let params = LLGParams {
        gamma: GAMMA_E_RAD_PER_S_T,
        alpha: 0.02,
        dt: 1e-13,
        b_ext,
    };

    let material = Material {
        ms,
        a_ex,
        k_u,
        easy_axis,
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    // Buffers
    let mut m = VectorField2D::new(grid);
    let mut b_demag = VectorField2D::new(grid);
    let mut b_eff_base = VectorField2D::new(grid); // Zeeman+exch+anis only

    let out_dir = Path::new("out").join("demag_snapshot");
    create_dir_all(&out_dir)?;

    // Helper to compute, print, and write one case
    let mut run_case =
        |label: &str, mx: f64, my: f64, mz: f64, csv_name: &str| -> std::io::Result<()> {
            m.set_uniform(mx, my, mz);

            // Demag field only
            compute_demag_field(&grid, &m, &mut b_demag, &material);
            let [bdx, bdy, bdz] = avg_vec(&b_demag);

            // Base effective field without demag (Zeeman + exchange + anisotropy)
            build_h_eff_masked(
                &grid,
                &m,
                &mut b_eff_base,
                &params,
                &material,
                FieldMask::ExchAnis,
            );
            let [bbx, bby, bbz] = avg_vec(&b_eff_base);

            // Define B_eff = B_base + B_demag
            let bex = bbx + bdx;
            let bey = bby + bdy;
            let bez = bbz + bdz;

            println!("=== {} ===", label);
            println!("m = [{:.6}, {:.6}, {:.6}]", mx, my, mz);
            println!("<B_demag> = [{:.6e}, {:.6e}, {:.6e}] T", bdx, bdy, bdz);
            println!("<B_eff>   = [{:.6e}, {:.6e}, {:.6e}] T", bex, bey, bez);
            println!();

            let out_path = out_dir.join(csv_name);
            write_snapshot_csv(&out_path, mx, my, mz, [bdx, bdy, bdz], [bex, bey, bez])?;

            Ok(())
        };

    run_case("uniform +x", 1.0, 0.0, 0.0, "rust_demag_snapshot_x.csv")?;
    run_case("uniform +y", 0.0, 1.0, 0.0, "rust_demag_snapshot_y.csv")?;
    run_case("uniform +z", 0.0, 0.0, 1.0, "rust_demag_snapshot_z.csv")?;

    println!("Wrote demag snapshot CSVs to {:?}", out_dir);
    Ok(())
}
