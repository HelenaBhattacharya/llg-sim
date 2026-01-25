// tests/validation.rs
//
// Integration-style validation tests (physics sanity checks).
// Run with: cargo test
// Or only these tests: cargo test --test validation
// To run ignored (FFT) test too: cargo test --test validation -- --ignored

use llg_sim::effective_field::build_h_eff;
use llg_sim::grid::Grid2D;
use llg_sim::llg::step_llg_with_field;
use llg_sim::params::{LLGParams, Material};
use llg_sim::vector_field::VectorField2D;
use llg_sim::energy::compute_total_energy;

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

fn sech(x: f64) -> f64 {
    1.0 / x.cosh()
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn macrospin_precession_quarter_turn_about_bz() {
    // Macrospin: 1 cell. No exchange/anisotropy effects.
    let grid = Grid2D::new(1, 1, 1.0, 1.0, 1.0);
    let mut m = VectorField2D::new(grid);
    let mut b_eff = VectorField2D::new(grid);

    // Start along +x
    m.set_uniform(1.0, 0.0, 0.0);

    // Precess about +z with alpha=0
    let b0 = 0.1; // Tesla
    let gamma = 1.760_859_630_23e11; // rad/(s*T)
    let params = LLGParams {
        gamma,
        alpha: 0.0,
        dt: 1e-14,
        b_ext: [0.0, 0.0, b0],
    };

    // No anisotropy/exchange
    let material = Material {
        ms: 8.0e5,
        a_ex: 0.0,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
    };

    // Target time: quarter period, t = (pi/2)/(gamma B)
    let t_target = std::f64::consts::FRAC_PI_2 / (gamma * b0);
    let n_steps = (t_target / params.dt).round() as usize;

    for _ in 0..n_steps {
        build_h_eff(&grid, &m, &mut b_eff, &params, &material);
        step_llg_with_field(&mut m, &b_eff, &params);
    }

    // Expected: m rotated from +x toward ±y. Direction depends on sign conventions,
    // so we assert it's *mostly* in the y-direction and still near the equator.
    let v = m.data[0];
    assert!(
        v[2].abs() < 0.1,
        "m_z should stay ~0 for pure precession, got {}",
        v[2]
    );
    assert!(
        v[1].abs() > 0.9,
        "after ~quarter turn, |m_y| should be large, got {}",
        v[1]
    );
    assert!(
        v[0].abs() < 0.3,
        "after ~quarter turn, |m_x| should be small, got {}",
        v[0]
    );
}

#[test]
fn macrospin_anisotropy_relaxes_toward_easy_axis() {
    // Macrospin with uniaxial anisotropy, no external field.
    let grid = Grid2D::new(1, 1, 1.0, 1.0, 1.0);
    let mut m = VectorField2D::new(grid);
    let mut b_eff = VectorField2D::new(grid);

    // Start tilted away from +z
    m.set_uniform(0.6, 0.0, 0.8); // already unit-ish but normalisation happens in integrator

    let params = LLGParams {
        gamma: 1.760_859_630_23e11,
        alpha: 0.2, // damping to relax
        dt: 1e-12,
        b_ext: [0.0, 0.0, 0.0],
    };

    let material = Material {
        ms: 8.0e5,
        a_ex: 0.0,
        k_u: 500.0, // J/m^3
        easy_axis: unit([0.0, 0.0, 1.0]),
        dmi: None,
    };

    let mz0 = m.data[0][2];

    // Run for a physical time (macrospin is cheap)
    let n_steps = 100_000; // 100 ns at dt=1e-12
    for _ in 0..n_steps {
        build_h_eff(&grid, &m, &mut b_eff, &params, &material);
        step_llg_with_field(&mut m, &b_eff, &params);
    }

    let mz1 = m.data[0][2];
    assert!(
        mz1 > mz0,
        "mz should increase toward +easy axis: mz0={}, mz1={}",
        mz0,
        mz1
    );
    assert!(
        mz1 > 0.95,
        "should be quite close to +z after damping, mz1={}",
        mz1
    );
}

#[test]
fn bloch_wall_init_matches_tanh_sech_profile() {
    // Analytic (tanh/sech) profile check for the *initialisation*.
    // This is a cheap spatial correctness test (no time evolution).

    let nx: usize = 128;
    let ny: usize = 1;
    let dx: f64 = 5e-9;
    let dy: f64 = 5e-9;
    let dz: f64 = 5e-9;

    let grid = Grid2D::new(nx, ny, dx, dy, dz);
    let mut m = VectorField2D::new(grid);

    let x0 = 0.5 * nx as f64 * dx;
    let width = 5.0 * dx;
    m.init_bloch_wall(x0, width);

    // Sample a few points and compare against the analytic expressions used in init_bloch_wall.
    let samples: [usize; 5] = [0, nx / 4, nx / 2, 3 * nx / 4, nx - 1];

    for &i in &samples {
        let x = (i as f64 + 0.5) * dx;
        let u = (x - x0) / width;

        let mz_expected = u.tanh();
        let mx_expected = sech(u);

        let v = m.data[m.idx(i, 0)];

        assert!(
            v[1].abs() < 1e-12,
            "Bloch init should have my=0, got {} at i={}",
            v[1],
            i
        );
        assert!(
            approx_eq(v[2], mz_expected, 1e-6),
            "mz mismatch at i={}: got {}, expected {}",
            i,
            v[2],
            mz_expected
        );
        assert!(
            approx_eq(v[0], mx_expected, 1e-6),
            "mx mismatch at i={}: got {}, expected {}",
            i,
            v[0],
            mx_expected
        );

        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(
            approx_eq(norm, 1.0, 1e-6),
            "|m| not ~1 at i={}: got {}",
            i,
            norm
        );
    }

    // Optional derivative sanity check near the wall centre.
    // mz(x) = tanh(u), u=(x-x0)/width -> dmz/dx = sech^2(u) / width
    let i_c = nx / 2;
    if i_c >= 1 && i_c + 1 < nx {
        let mz_m = m.data[m.idx(i_c - 1, 0)][2];
        let mz_p = m.data[m.idx(i_c + 1, 0)][2];

        let dmz_dx_num = (mz_p - mz_m) / (2.0 * dx);

        let x_c = (i_c as f64 + 0.5) * dx;
        let u_c = (x_c - x0) / width;
        let dmz_dx_expected = sech(u_c).powi(2) / width;

        let rel_err = (dmz_dx_num - dmz_dx_expected).abs() / dmz_dx_expected.abs().max(1e-30);
        assert!(
            rel_err < 0.1,
            "dmz/dx mismatch: num={}, expected={}, rel_err={}",
            dmz_dx_num,
            dmz_dx_expected,
            rel_err
        );
    }
}

#[test]
fn energy_gradient_consistency_exchange_anisotropy() {
    // Checks the discrete consistency:
    //   ΔE ≈ -Ms * V * B_eff(cell) · Δm
    //
    // This is *exactly* the Priority B "energy-gradient consistency" idea.

    let grid = Grid2D::new(16, 16, 5e-9, 5e-9, 5e-9);
    let mut m = VectorField2D::new(grid);

    // A non-uniform configuration (so exchange is active)
    let x0 = 0.5 * grid.nx as f64 * grid.dx;
    let width = 5.0 * grid.dx;
    m.init_bloch_wall(x0, width);

    let params = LLGParams {
        gamma: 1.760_859_630_23e11,
        alpha: 0.02,
        dt: 1e-13,
        b_ext: [0.0, 0.0, 0.0],
    };

    let material = Material {
        ms: 8.0e5,
        a_ex: 13e-12,
        k_u: 500.0,
        easy_axis: unit([0.0, 0.0, 1.0]),
        dmi: None,
    };

    // Build B_eff for the current state
    let mut b_eff = VectorField2D::new(grid);
    build_h_eff(&grid, &m, &mut b_eff, &params, &material);

    let e0 = compute_total_energy(&grid, &m, &material, params.b_ext);

    // Pick an interior cell
    let i = grid.nx / 2;
    let j = grid.ny / 2;
    let idx = grid.idx(i, j);
    let m0 = m.data[idx];

    // Choose a perturbation direction perpendicular to m0
    let a = [0.37, -0.24, 0.91];
    let mdota = m0[0] * a[0] + m0[1] * a[1] + m0[2] * a[2];
    let mut dir = [a[0] - mdota * m0[0], a[1] - mdota * m0[1], a[2] - mdota * m0[2]];
    let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    assert!(n > 1e-12, "failed to construct a perpendicular perturbation");
    dir = [dir[0] / n, dir[1] / n, dir[2] / n];

    // Small perturbation + renormalise
    let eps = 1e-6;
    let m1 = unit([m0[0] + eps * dir[0], m0[1] + eps * dir[1], m0[2] + eps * dir[2]]);
    let dm = [m1[0] - m0[0], m1[1] - m0[1], m1[2] - m0[2]];

    // Copy field and perturb one cell
    let mut m_pert = VectorField2D { grid, data: m.data.clone() };
    m_pert.data[idx] = m1;

    let e1 = compute_total_energy(&grid, &m_pert, &material, params.b_ext);

    let de_num = e1 - e0;

    let b = b_eff.data[idx];
    let de_pred = -material.ms * grid.cell_volume() * (b[0] * dm[0] + b[1] * dm[1] + b[2] * dm[2]);

    // Compare with a loose tolerance (we only need “same order and sign” right now)
    let b_norm = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    let scale = (material.ms * grid.cell_volume() * b_norm * eps).max(1e-30);

    let err = (de_num - de_pred).abs();

    assert!(
        err < 5e-2 * scale,
        "ΔE mismatch: num={:.6e}, pred={:.6e}, err={:.6e}, scale={:.6e}",
        de_num, de_pred, err, scale
    );
}

#[test]
#[ignore]
fn macrospin_fmr_fft_peak_matches_gamma_b() {
    // "Full" FMR-style check: simulate macrospin ringdown and find dominant frequency via FFT/DFT.
    // Marked #[ignore] because it is more expensive and can be sensitive to timestep/resolution.
    // Run with: cargo test --test validation -- --ignored

    let grid = Grid2D::new(1, 1, 1.0, 1.0, 1.0);
    let mut m = VectorField2D::new(grid);
    let mut b_eff = VectorField2D::new(grid);

    // Small tilt from +z to excite precession
    let theta = 5.0_f64.to_radians();
    m.set_uniform(theta.sin(), 0.0, theta.cos());

    let b0 = 1.0; // Tesla
    let gamma = 1.760_859_630_23e11; // rad/(s*T)

    let params = LLGParams {
        gamma,
        alpha: 0.01,
        dt: 5e-14,
        b_ext: [0.0, 0.0, b0],
    };

    // No anisotropy/exchange, so B_eff is just B_ext
    b_eff.set_uniform(0.0, 0.0, b0);

    // Collect m_y(t)
    let n: usize = 16384;
    let mut my: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..n {
        step_llg_with_field(&mut m, &b_eff, &params);
        my.push(m.data[0][1]);
    }

    // Naive DFT (O(N^2)) is fine for N=4096.
    // Find peak frequency for k=1..N/2.
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut best_k: usize = 1;
    let mut best_mag: f64 = 0.0;

    for k in 1..(n / 2) {
        let mut re = 0.0;
        let mut im = 0.0;
        for (n_idx, &x) in my.iter().enumerate() {
            let ang = -two_pi * (k as f64) * (n_idx as f64) / (n as f64);
            re += x * ang.cos();
            im += x * ang.sin();
        }
        let mag2 = re * re + im * im;
        if mag2 > best_mag {
            best_mag = mag2;
            best_k = k;
        }
    }

    let f_peak_hz = (best_k as f64) / ((n as f64) * params.dt);
    let f_expected_hz = (gamma * b0) / two_pi;

    // Frequency resolution is ~ 1/T where T = N*dt.
    // Use a loose tolerance.
    let rel_err = (f_peak_hz - f_expected_hz).abs() / f_expected_hz;
    assert!(
        rel_err < 0.25,
        "FFT peak freq mismatch: f_peak={} Hz, f_expected={} Hz, rel_err={} (dt={}, N={})",
        f_peak_hz,
        f_expected_hz,
        rel_err,
        params.dt,
        n
    );
}


#[test]
fn dmi_field_flips_sign_with_d() {
    use llg_sim::effective_field::build_h_eff;
    use llg_sim::grid::Grid2D;
    use llg_sim::params::{LLGParams, Material};
    use llg_sim::vector_field::VectorField2D;

    // 1D line: ny=1 so only x-derivatives matter.
    let nx: usize = 9;
    let ny: usize = 1;
    let dx: f64 = 1.0;
    let dy: f64 = 1.0;
    let dz: f64 = 1.0;
    let grid = Grid2D::new(nx, ny, dx, dy, dz);

    // Gentle mz gradient so dmz/dx != 0
    let mut m = VectorField2D::new(grid);
    for i in 0..nx {
        let mz = (i as f64 - (nx as f64 - 1.0) / 2.0) * 0.01;
        let mz = mz.max(-0.2).min(0.2);
        let mx = (1.0 - mz * mz).sqrt();

        let idx = m.idx(i, 0);
        m.data[idx] = [mx, 0.0, mz];
    }

    let params = LLGParams {
        gamma: 1.760_859_630_23e11,
        alpha: 0.0,
        dt: 1e-13,
        b_ext: [0.0, 0.0, 0.0],
    };

    let ms: f64 = 8.0e5;
    let a_ex: f64 = 13e-12;

    // +D
    let mat_plus = Material {
        ms,
        a_ex,
        k_u: 0.0,
        easy_axis: unit([0.0, 0.0, 1.0]),
        dmi: Some(1e-4),
    };
    let mut b_plus = VectorField2D::new(grid);
    build_h_eff(&grid, &m, &mut b_plus, &params, &mat_plus);

    // -D
    let mat_minus = Material {
        ms,
        a_ex,
        k_u: 0.0,
        easy_axis: unit([0.0, 0.0, 1.0]),
        dmi: Some(-1e-4),
    };
    let mut b_minus = VectorField2D::new(grid);
    build_h_eff(&grid, &m, &mut b_minus, &params, &mat_minus);

    // Interior cell (avoid boundary complications)
    let i0 = 4usize;
    let idx0 = m.idx(i0, 0);

    let bx_plus = b_plus.data[idx0][0];
    let bx_minus = b_minus.data[idx0][0];

    // Should be opposite sign: bx(+D) + bx(-D) ≈ 0
    assert!(bx_plus.abs() > 0.0, "expected nonzero DMI field component");
    assert!(
        (bx_plus + bx_minus).abs() < 1e-10 * bx_plus.abs().max(1.0),
        "DMI field should flip sign with D: bx+= {}, bx-= {}",
        bx_plus, bx_minus
    );
}