// src/bin/st_problems/sk1.rs
//
// "Skyrmion in a disk" standard-problem-style setup inspired by sk1.mx3.
//
// This file is intentionally focused on *problem setup*: grid, geometry mask,
// initial magnetization seed. Hook it into your existing simulation / relaxation
// driver (LLG, minimizer, etc.) the same way you do for SP2/SP4 binaries.
//
// MuMax reference (sk1.mx3):
// - disk diameter: 100 nm, thickness 1 nm
// - Msat = 1000 kA/m
// - Aex = 15 pJ/m
// - Dind = 3 mJ/m^2
// - Ku1 = 1.2 MJ/m^3 (easy axis +z)
// - B_ext = (0,0,0.2) T
// - alpha = 0.3
// - seed: mostly +z with slight y tilt, then flip a small core.

use cmp11::grid::Grid2D;
use cmp11::vector_field::VectorField2D;

// New helper modules to add under src/
use cmp11::geometry_mask as geom;
use cmp11::initial_states as init;

fn main() {
    // --- Grid (match MuMax cell sizes) ---
    let nx: usize = 128;
    let ny: usize = 128;

    let disk_diam = 100e-9;
    let thickness = 1e-9;

    let dx = disk_diam / (nx as f64);
    let dy = disk_diam / (ny as f64);
    let dz = thickness; // 2D thin-film model: one layer of thickness dz

    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    // --- Geometry: disk ---
    let disk_radius = 0.5 * disk_diam;
    let mask = geom::mask_disk(&grid, disk_radius, (0.0, 0.0));

    // --- Initial magnetization seed ---
    let mut m = VectorField2D::new(grid);

    // Mostly +z, with slight y tilt to break symmetry.
    init::init_uniform(&mut m, [0.0, 0.02, 1.0]);

    // Flip a small core (MuMax uses circle(20e-9) and comments "r<10 nm").
    let core_radius = 10e-9;
    init::seed_reversed_core(
        &mut m,
        &grid,
        (0.0, 0.0),
        core_radius,
        [0.0, 0.02, 1.0],
        [0.0, 0.02, -1.0],
        Some(&mask),
    );

    // Enforce vacuum outside disk.
    init::apply_mask_zero(&mut m, &mask);

    // --- Output + simulation hook ---
    //
    // 1) Write the initial OVF (or whatever your output format is).
    //    Example (adapt to your actual IO module):
    //      cmp11::io::write_ovf("out/sk1_init.ovf", &grid, &m).unwrap();
    //
    // 2) Run a relaxation / LLG integration to equilibrium.
    //    Example (adapt to your driver):
    //      let mat = Material { ... } // set Ms, A, D, Ku, Bext, alpha, demag_method, ...
    //      let report = cmp11::relax::relax(&grid, &mut m, &mat, /*dt*/ 1e-13, /*steps*/ 50_000);
    //
    // 3) Save final state and energy table.

    println!(
        "sk1 setup complete: nx={} ny={} dx={:.3e} dy={:.3e} disk cells={}",
        nx,
        ny,
        dx,
        dy,
        geom::mask_count(&mask)
    );
}
