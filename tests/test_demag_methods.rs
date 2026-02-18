// tests/demag_methods.rs
//
// Integration tests for the demag dispatcher wiring:
// - When DemagMethod::FftUniform is selected, `effective_field::demag` should route to
//   `effective_field::demag_fft_uniform`.
// - When DemagMethod::PoissonMG is selected, `effective_field::demag` should route to
//   `effective_field::demag_poisson_mg`.
//
// These tests are about *wiring*, not about proving physical equivalence between FFT and MG.

use llg_sim::effective_field::{demag, demag_fft_uniform, demag_poisson_mg};
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, Material};
use llg_sim::vector_field::VectorField2D;

use std::sync::{Mutex, OnceLock};

static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[test]
fn demag_dispatch_routes_to_fft_uniform() {
    let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
    configure_env_for_demag_tests();

    let grid = Grid2D::new(8, 8, 5e-9, 5e-9, 1e-9);
    let mut m = VectorField2D::new(grid);
    init_pattern(&mut m);

    let mat = Material {
        ms: 8.0e5,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    let mut b_dispatch = VectorField2D::new(grid);
    let mut b_direct = VectorField2D::new(grid);

    demag::compute_demag_field(&grid, &m, &mut b_dispatch, &mat);
    demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_direct, &mat);

    let max_abs = max_abs_diff(&b_dispatch, &b_direct);
    assert!(
        max_abs < 1e-12,
        "dispatcher vs direct FFT mismatch: max_abs={}",
        max_abs
    );
}

#[test]
fn demag_dispatch_routes_to_poisson_mg() {
    let _guard = ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
    configure_env_for_demag_tests();

    let grid = Grid2D::new(8, 8, 5e-9, 5e-9, 1e-9);
    let mut m = VectorField2D::new(grid);
    init_pattern(&mut m);

    let mat = Material {
        ms: 8.0e5,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::PoissonMG,
    };

    let mut b_dispatch = VectorField2D::new(grid);
    let mut b_direct = VectorField2D::new(grid);

    demag::compute_demag_field(&grid, &m, &mut b_dispatch, &mat);
    demag_poisson_mg::compute_demag_field_poisson_mg(&grid, &m, &mut b_direct, &mat);

    let max_abs = max_abs_diff(&b_dispatch, &b_direct);
    assert!(
        max_abs < 1e-12,
        "dispatcher vs direct MG mismatch: max_abs={}",
        max_abs
    );
}

fn configure_env_for_demag_tests() {
    // NOTE: In recent Rust versions, mutating process environment variables is `unsafe`
    // because it can cause undefined behavior if other threads concurrently access `environ`.
    // These tests guard env mutations with `ENV_LOCK`, so we serialize updates.
    unsafe {
        // Ensure method selection is driven by `Material.demag_method` in these wiring tests.
        std::env::remove_var("LLG_DEMAG_METHOD");

        // Make MG runs deterministic and fast.
        // Warm start would otherwise make the second call more converged than the first.
        std::env::set_var("LLG_DEMAG_MG_WARM_START", "0");
        std::env::set_var("LLG_DEMAG_MG_PAD_FACTOR_XY", "2");
        std::env::set_var("LLG_DEMAG_MG_NVAC_Z", "2");
        std::env::set_var("LLG_DEMAG_MG_VCYCLES", "2");
        std::env::remove_var("LLG_DEMAG_MG_TOL_ABS");
    }
}

fn init_pattern(m: &mut VectorField2D) {
    let nx = m.grid.nx;
    let ny = m.grid.ny;
    let two_pi = 2.0 * std::f64::consts::PI;

    for j in 0..ny {
        for i in 0..nx {
            let x = (i as f64 + 0.5) / (nx as f64);
            let y = (j as f64 + 0.5) / (ny as f64);

            let mut mx: f64 = (two_pi * x).cos();
            let mut my: f64 = (two_pi * y).sin();
            let mut mz: f64 = 0.2;

            let n = (mx * mx + my * my + mz * mz).sqrt().max(1e-30);
            mx /= n;
            my /= n;
            mz /= n;

            let id = m.idx(i, j);
            m.data[id][0] = mx;
            m.data[id][1] = my;
            m.data[id][2] = mz;
        }
    }
}

fn max_abs_diff(a: &VectorField2D, b: &VectorField2D) -> f64 {
    assert_eq!(a.data.len(), b.data.len());
    let mut max_abs: f64 = 0.0;
    for (va, vb) in a.data.iter().zip(b.data.iter()) {
        let dx = (va[0] - vb[0]).abs();
        let dy = (va[1] - vb[1]).abs();
        let dz = (va[2] - vb[2]).abs();
        max_abs = max_abs.max(dx.max(dy).max(dz));
    }
    max_abs
}

// --- Hybrid demag accuracy regression test ---
// Requires: demag_poisson_mg implements hybrid via env vars (radius / delta_v_cycles).

fn xorshift64(seed: &mut u64) -> u64 {
    let mut x = *seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *seed = x;
    x
}

fn rand_u01(seed: &mut u64) -> f64 {
    // [0,1)
    let x = xorshift64(seed);
    (x as f64) / (u64::MAX as f64)
}

fn init_random_unit_vectors(field: &mut VectorField2D, seed0: u64) {
    let mut seed = seed0;
    for v in &mut field.data {
        // simple symmetric random vector, then normalize
        let mut x = 2.0 * rand_u01(&mut seed) - 1.0;
        let mut y = 2.0 * rand_u01(&mut seed) - 1.0;
        let mut z = 2.0 * rand_u01(&mut seed) - 1.0;

        let n = (x * x + y * y + z * z).sqrt().max(1e-30);
        x /= n;
        y /= n;
        z /= n;
        v[0] = x;
        v[1] = y;
        v[2] = z;
    }
}

fn rel_rmse_vs_ref(b_ref: &VectorField2D, b_test: &VectorField2D) -> f64 {
    assert_eq!(b_ref.data.len(), b_test.data.len());
    let mut sum_d2 = 0.0f64;
    let mut sum_ref2 = 0.0f64;

    for (a, b) in b_ref.data.iter().zip(b_test.data.iter()) {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        sum_d2 += dx * dx + dy * dy + dz * dz;

        sum_ref2 += a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    }

    if sum_ref2 <= 0.0 {
        return 0.0;
    }
    (sum_d2 / sum_ref2).sqrt()
}

#[test]
fn demag_hybrid_reduces_random_error_vs_pure_mg() {
    // IMPORTANT: env vars are global; keep this test serialized with other env tests.
    let _guard = ENV_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap();

    // Keep test small so it stays fast in debug builds.
    let grid = Grid2D::new(32, 32, 5e-9, 5e-9, 1e-9);

    let mat = Material {
        ms: 8.0e5,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform, // irrelevant for direct calls
    };

    let mut m = VectorField2D::new(grid);
    init_random_unit_vectors(&mut m, 12345);

    let mut b_fft = VectorField2D::new(grid);
    let mut b_mg0 = VectorField2D::new(grid);
    let mut b_hyb = VectorField2D::new(grid);

    // FFT reference
    demag_fft_uniform::compute_demag_field(&grid, &m, &mut b_fft, &mat);

    // Configure MG runtime knobs deterministically for the test.
    unsafe {
        std::env::set_var("LLG_DEMAG_MG_WARM_START", "0");
        std::env::set_var("LLG_DEMAG_MG_PAD_FACTOR_XY", "2");
        std::env::set_var("LLG_DEMAG_MG_NVAC_Z", "16");
        std::env::set_var("LLG_DEMAG_MG_VCYCLES", "4");
        std::env::set_var("LLG_DEMAG_MG_HYBRID_CACHE", "0");
        std::env::set_var("LLG_DEMAG_MG_HYBRID_DELTA_VCYCLES", "20");
    }

    // Pure MG (hybrid disabled)
    unsafe {
        std::env::set_var("LLG_DEMAG_MG_HYBRID_RADIUS", "0");
    }
    demag_poisson_mg::compute_demag_field_poisson_mg(&grid, &m, &mut b_mg0, &mat);

    // Hybrid MG
    unsafe {
        std::env::set_var("LLG_DEMAG_MG_HYBRID_RADIUS", "2");
    }
    demag_poisson_mg::compute_demag_field_poisson_mg(&grid, &m, &mut b_hyb, &mat);

    let rel0 = rel_rmse_vs_ref(&b_fft, &b_mg0);
    let relh = rel_rmse_vs_ref(&b_fft, &b_hyb);

    // Expect a big improvement (your real run is ~0.32 -> ~0.04).
    assert!(
        relh < 0.12,
        "hybrid too inaccurate vs FFT: rel_RMSE={:.3e} (pure MG was {:.3e})",
        relh,
        rel0
    );
    assert!(
        relh < 0.35 * rel0,
        "hybrid did not improve enough: rel_hyb={:.3e}, rel_pure={:.3e}",
        relh,
        rel0
    );

    // Clean up env vars to reduce cross-test contamination.
    unsafe {
        std::env::remove_var("LLG_DEMAG_MG_WARM_START");
        std::env::remove_var("LLG_DEMAG_MG_PAD_FACTOR_XY");
        std::env::remove_var("LLG_DEMAG_MG_NVAC_Z");
        std::env::remove_var("LLG_DEMAG_MG_VCYCLES");
        std::env::remove_var("LLG_DEMAG_MG_HYBRID_CACHE");
        std::env::remove_var("LLG_DEMAG_MG_HYBRID_DELTA_VCYCLES");
        std::env::remove_var("LLG_DEMAG_MG_HYBRID_RADIUS");
    }
}
