// src/bin/demag_cube.rs
//
// Developer diagnostic: demagnetisation kernel sanity checker.
//
// Computes implied demag factors (Nxx, Nyy, Nzz) for a uniform magnetisation
// on an Nx×Ny×1 grid with given cell dimensions.
//
// This tool:
//   - prints results to stdout only
//   - does NOT write files
//   - does NOT compare against MuMax
//   - is intended for interactive inspection during demag development
//
// Usage examples:
//   cargo run --bin demag_cube
//   cargo run --bin demag_cube -- 2 2 1.0 1.0 1.0
//   cargo run --bin demag_cube -- 64 64 5e-9 5e-9 1e-9

use llg_sim::effective_field::demag::compute_demag_field;
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, MU0, Material};
use llg_sim::vector_field::VectorField2D;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Defaults: a modest thin-film-ish grid
    let (nx, ny, dx, dy, dz) = if args.len() == 6 {
        (
            args[1].parse::<usize>().expect("nx"),
            args[2].parse::<usize>().expect("ny"),
            args[3].parse::<f64>().expect("dx"),
            args[4].parse::<f64>().expect("dy"),
            args[5].parse::<f64>().expect("dz"),
        )
    } else {
        (32usize, 32usize, 5e-9, 5e-9, 5e-9)
    };

    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    // Ms cancels when we infer Nii, but keep it explicit.
    let ms = 1.0;
    let mat = Material {
        ms,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let nxx = infer_nii(&grid, &mat, 0);
    let nyy = infer_nii(&grid, &mat, 1);
    let nzz = infer_nii(&grid, &mat, 2);

    println!(
        "Grid: {}x{}, dx={:.3e}, dy={:.3e}, dz={:.3e}",
        nx, ny, dx, dy, dz
    );
    println!(
        "Implied demag factors: Nxx={:.6}, Nyy={:.6}, Nzz={:.6}",
        nxx, nyy, nzz
    );
    println!("Trace check: Nxx+Nyy+Nzz = {:.6}", nxx + nyy + nzz);
}

fn infer_nii(grid: &Grid2D, mat: &Material, component: usize) -> f64 {
    let mut m = VectorField2D::new(*grid);
    let mut b_demag = VectorField2D::new(*grid);

    // Uniform m along chosen axis
    let (mx, my, mz) = match component {
        0 => (1.0, 0.0, 0.0),
        1 => (0.0, 1.0, 0.0),
        2 => (0.0, 0.0, 1.0),
        _ => unreachable!(),
    };
    m.set_uniform(mx, my, mz);

    compute_demag_field(grid, &m, &mut b_demag, mat);

    // Average B_i over all cells (robust)
    let mut sum = 0.0;
    for b in &b_demag.data {
        sum += b[component];
    }
    let b_avg = sum / (b_demag.data.len() as f64);

    // Nii = -B_i / (mu0 * Ms * m_i). Here m_i = 1.
    -b_avg / (MU0 * mat.ms)
}
