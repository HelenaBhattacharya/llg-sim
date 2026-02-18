// src/initial_states.rs
//
// Initial magnetization (seed) generators for 2D thin-film problems.
//
// Conventions:
// - m is stored per cell (nx*ny), typically unit length within the geometry.
// - Outside geometry, represent vacuum by setting m=(0,0,0). This matches the demag
//   Poisson-MG implementation which treats |m|≈0 cells as non-magnetic.
//
// Coordinate system:
// - We use centered cell centers (0,0) at the grid center (same as geometry_mask.rs).

use crate::grid::Grid2D;
use crate::vec3::normalize;
use crate::vector_field::VectorField2D;

use crate::geometry_mask::cell_center_xy_centered;

/// Apply a boolean mask by setting m=(0,0,0) where mask=false.
pub fn apply_mask_zero(m: &mut VectorField2D, mask: &[bool]) {
    assert_eq!(m.data.len(), mask.len());
    for (v, &inside) in m.data.iter_mut().zip(mask.iter()) {
        if !inside {
            *v = [0.0; 3];
        }
    }
}

/// Set a uniform direction (normalized). Does not apply a mask.
pub fn init_uniform(m: &mut VectorField2D, dir: [f64; 3]) {
    let v = normalize(dir);
    m.set_uniform(v[0], v[1], v[2]);
}

/// Uniform + optional small random tilt (useful to break symmetry).
pub fn init_uniform_with_noise(m: &mut VectorField2D, dir: [f64; 3], noise: f64, seed: u64) {
    let base = normalize(dir);
    let mut rng = XorShift64::new(seed);
    for v in &mut m.data {
        if v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0 {
            // allow pre-masked arrays
            continue;
        }
        let dx = noise * (rng.next_f64() * 2.0 - 1.0);
        let dy = noise * (rng.next_f64() * 2.0 - 1.0);
        let dz = noise * (rng.next_f64() * 2.0 - 1.0);
        *v = normalize([base[0] + dx, base[1] + dy, base[2] + dz]);
    }
}

/// Random directions (uniform-ish on the sphere) using a simple xorshift RNG.
pub fn init_random(m: &mut VectorField2D, seed: u64) {
    let mut rng = XorShift64::new(seed);
    for v in &mut m.data {
        // Marsaglia method: pick from normal-ish and normalize.
        let x = rng.next_f64() * 2.0 - 1.0;
        let y = rng.next_f64() * 2.0 - 1.0;
        let z = rng.next_f64() * 2.0 - 1.0;
        *v = normalize([x, y, z]);
    }
}

/// Seed a reversed core inside a radius (useful for skyrmion nucleation).
///
/// - `center`: (x,y) in meters (centered coordinates)
/// - `outer_dir`: default direction outside core
/// - `core_dir`: direction inside core
/// - If `mask` is provided, cells outside mask are forced to (0,0,0).
pub fn seed_reversed_core(
    m: &mut VectorField2D,
    grid: &Grid2D,
    center: (f64, f64),
    core_radius: f64,
    outer_dir: [f64; 3],
    core_dir: [f64; 3],
    mask: Option<&[bool]>,
) {
    let (cx, cy) = center;
    let r2 = core_radius * core_radius;
    let outer = normalize(outer_dir);
    let core = normalize(core_dir);

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let id = j * grid.nx + i;

            if let Some(msk) = mask {
                if !msk[id] {
                    m.data[id] = [0.0; 3];
                    continue;
                }
            }

            let (x, y) = cell_center_xy_centered(grid, i, j);
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= r2 {
                m.data[id] = core;
            } else {
                m.data[id] = outer;
            }
        }
    }
}

/// Create a vortex in-plane, with an optional out-of-plane core.
pub fn init_vortex(
    m: &mut VectorField2D,
    grid: &Grid2D,
    center: (f64, f64),
    polarity: f64,  // +1 up, -1 down
    chirality: f64, // +1 CCW, -1 CW
    core_radius: f64,
    mask: Option<&[bool]>,
) {
    let (cx, cy) = center;
    let core_r2 = core_radius * core_radius;
    let pol = polarity.signum();
    let chi = chirality.signum();

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let id = j * grid.nx + i;

            if let Some(msk) = mask {
                if !msk[id] {
                    m.data[id] = [0.0; 3];
                    continue;
                }
            }

            let (x, y) = cell_center_xy_centered(grid, i, j);
            let dx = x - cx;
            let dy = y - cy;
            let r2 = dx * dx + dy * dy;

            if r2 < 1e-30 {
                m.data[id] = [0.0, 0.0, pol];
                continue;
            }

            // Azimuthal unit vector e_phi = (-y, x)/r
            let r = r2.sqrt();
            let ex = -dy / r;
            let ey = dx / r;

            let mut mz = 0.0;
            if r2 <= core_r2 {
                // Smooth core profile: mz goes to polarity at r=0.
                let t = 1.0 - (r / core_radius).clamp(0.0, 1.0);
                mz = pol * t;
            }

            let in_plane_scale = (1.0 - mz * mz).max(0.0).sqrt();
            m.data[id] = normalize([chi * ex * in_plane_scale, chi * ey * in_plane_scale, mz]);
        }
    }
}

/// Create a skyrmion seed.
///
/// Profile:
///   θ(r) = 2 * atan( exp( (R0 - r)/Δ ) )
///   m_z  = -p * cosθ
///   m_xy = sinθ * (cosφ, sinφ) with φ = atan2(y,x) + helicity
///
/// - `core_polarity` p = +1 means core points -z, p=-1 core +z (depending on convention).
/// - `helicity` 0 = Néel (radial), π/2 = Bloch (tangential).
pub fn init_skyrmion(
    m: &mut VectorField2D,
    grid: &Grid2D,
    center: (f64, f64),
    r0: f64,
    delta: f64,
    helicity: f64,
    core_polarity: f64,
    mask: Option<&[bool]>,
) {
    let (cx, cy) = center;
    let p = core_polarity.signum();
    let inv_delta = 1.0 / delta.max(1e-30);

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let id = j * grid.nx + i;

            if let Some(msk) = mask {
                if !msk[id] {
                    m.data[id] = [0.0; 3];
                    continue;
                }
            }

            let (x, y) = cell_center_xy_centered(grid, i, j);
            let dx = x - cx;
            let dy = y - cy;
            let r = (dx * dx + dy * dy).sqrt();

            let theta = 2.0 * (((r0 - r) * inv_delta).exp()).atan();
            let ct = theta.cos();
            let st = theta.sin();

            let phi = dy.atan2(dx) + helicity;
            let mx = st * phi.cos();
            let my = st * phi.sin();
            let mz = -p * ct;

            m.data[id] = normalize([mx, my, mz]);
        }
    }
}

// ---------------------------
// Small deterministic RNG (no extra deps)
// ---------------------------

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: s }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        // Map top 53 bits to [0,1)
        let u = self.next_u64() >> 11;
        (u as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Seed multiple reversed cores in one pass (does NOT overwrite one core with another).
pub fn seed_reversed_cores(
    m: &mut VectorField2D,
    grid: &Grid2D,
    centers: &[(f64, f64)],
    core_radius: f64,
    outer_dir: [f64; 3],
    core_dir: [f64; 3],
    mask: Option<&[bool]>,
) {
    let r2 = core_radius * core_radius;
    let outer = normalize(outer_dir);
    let core = normalize(core_dir);

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let id = j * grid.nx + i;

            if let Some(msk) = mask {
                if !msk[id] {
                    m.data[id] = [0.0; 3];
                    continue;
                }
            }

            let (x, y) = cell_center_xy_centered(grid, i, j);

            let mut inside_any = false;
            for &(cx, cy) in centers {
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy <= r2 {
                    inside_any = true;
                    break;
                }
            }

            m.data[id] = if inside_any { core } else { outer };
        }
    }
}

/// Seed multiple smooth "bubble" domains with a finite wall width.
///
/// This avoids the discontinuous step in `seed_reversed_cores`, which can create
/// extremely sharp gradients (and overly sensitive AMR indicators) at t=0.
///
/// The profile is a smooth 180° rotation as a function of radius from the nearest center:
///   mz(r) = s * tanh((r - r0)/w)
/// where `s = sign(outer_polarity)` so that outside is ±z and inside is ∓z.
///
/// The in-plane component has magnitude sqrt(1 - mz^2) and direction set by
/// `phi = atan2(dy, dx) + helicity` (helicity=0 => radial/Néel-like, helicity=π/2 => tangential/Bloch-like).
///
/// - `centers`: slice of (x,y) in meters in *centered* coordinates
/// - `r0`: bubble radius in meters
/// - `wall_width`: transition width in meters (choose a few cells, e.g. 20–50 nm for dx=5 nm)
/// - `outer_polarity`: +1 means outside +z, inside -z; -1 means outside -z, inside +z
pub fn seed_smooth_bubbles(
    m: &mut VectorField2D,
    grid: &Grid2D,
    centers: &[(f64, f64)],
    r0: f64,
    wall_width: f64,
    helicity: f64,
    outer_polarity: f64,
    mask: Option<&[bool]>,
) {
    let inv_w = 1.0 / wall_width.max(1e-30);
    let s = outer_polarity.signum();

    for j in 0..grid.ny {
        for i in 0..grid.nx {
            let id = j * grid.nx + i;

            if let Some(msk) = mask {
                if !msk[id] {
                    m.data[id] = [0.0; 3];
                    continue;
                }
            }

            let (x, y) = cell_center_xy_centered(grid, i, j);

            // Find nearest bubble center (for multiple bubbles).
            let mut best_r2 = f64::INFINITY;
            let mut best_dx = 0.0;
            let mut best_dy = 0.0;
            for &(cx, cy) in centers {
                let dx = x - cx;
                let dy = y - cy;
                let rr2 = dx * dx + dy * dy;
                if rr2 < best_r2 {
                    best_r2 = rr2;
                    best_dx = dx;
                    best_dy = dy;
                }
            }

            let r = best_r2.sqrt();

            // Smooth wall profile.
            let t = (r - r0) * inv_w;
            let mz = s * t.tanh();

            // In-plane magnitude.
            let mxy = (1.0 - mz * mz).max(0.0).sqrt();

            // Direction of in-plane component (radial/tangential depending on helicity).
            let phi = best_dy.atan2(best_dx) + helicity;
            let mx = mxy * phi.cos();
            let my = mxy * phi.sin();

            m.data[id] = normalize([mx, my, mz]);
        }
    }
}
