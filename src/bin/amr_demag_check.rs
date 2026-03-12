// src/bin/amr_demag_check.rs
//
// AMR demag accuracy check — frozen-magnetisation benchmark
// ==========================================================
//
// Isolates the demagnetising-field solver from LLG dynamics by computing
// B_demag for a *fixed* analytic vortex state on an AMR hierarchy and
// comparing each solver mode against the gold-standard FFT reference on
// a uniform fine grid.
//
// This is the Phase 2 diagnostic recommended in the Architecture Diagnosis
// document: a dedicated benchmark that does NOT step LLG, so demag errors
// cannot be conflated with time-integration artefacts.
//
// Physics setup (matching García-Cervera & Roma, IEEE Trans. Magn. 2005):
//   - 500 nm × 500 nm Permalloy square, dz = dx ≈ 3.9 nm (CUBIC cells)
//   - García-Cervera's formulation assumes infinite thickness in z.
//     The 2D log-kernel Green's function is exact for dz → ∞ and
//     increasingly approximate as dz → dx.  For cubic cells (dz = dx),
//     the 2D kernel overestimates in-plane demag factors (Nxx ≈ 0.5
//     instead of 0.33).  This is a known limitation, not a bug.
//     Use LLG_CHECK_DZ to sweep dz and measure the thickness dependence.
//     Set LLG_CHECK_DZ=100e-9 to test the near-infinite-thickness regime
//     where the FK pipeline should achieve < 1% RMSE.
//   - No geometry mask (full square domain)
//   - Analytic Usov-type vortex:
//       θ = atan2(y − cy, x − cx)    (azimuthal angle)
//       r = distance from centre
//       mx = −sin(θ),  my = +cos(θ),  mz = p · exp(−r²/R_c²)
//     where R_c ≈ 2 × l_ex is the vortex-core radius and p = +1 is the
//     core polarity.  This gives a non-trivial ∇·M distribution that
//     exercises the demag solver properly.
//   - Exchange + Demag only (no anisotropy, no DMI, no Zeeman)
//
// Modes compared:
//   1. fft_fine    — FFT Newell convolution on uniform fine grid (reference)
//   2. fft_coarse  — FFT Newell on uniform coarse grid (baseline)
//   3. coarse_fft  — Phase 1 solver: M-restriction + L0 FFT
//   4. composite   — Enhanced-RHS FK/MG composite solver
//   5. mg_coarse   — Standalone FK/MG on coarse grid (no patches)
//
// Outputs in out/amr_demag_check:
//   - summary.txt       — human-readable results table
//   - results.csv       — machine-readable per-mode RMSE breakdown
//   - b_MODE_coarse.csv — per-cell Bx,By,Bz on coarse grid (with --csv)
//   - b_ref_coarse.csv  — reference B sampled to coarse grid (with --csv)
//
// Run examples:
//   # Full comparison (all 4 modes, default 128×128, cubic cells dz=dx):
//   cargo run --release --bin amr_demag_check
//
//   # Skip expensive fine reference (uses coarse-only FFT as reference):
//   cargo run --release --bin amr_demag_check -- --skip-fine-ref
//
//   # Write per-cell CSV files for plotting:
//   cargo run --release --bin amr_demag_check -- --csv
//
//   # Thin-film regime (expect FK degradation — confirms known limitation):
//   LLG_CHECK_DZ=1e-9 cargo run --release --bin amr_demag_check
//
//   # Near-infinite thickness (FK should be very accurate here):
//   LLG_CHECK_DZ=100e-9 cargo run --release --bin amr_demag_check
//
//   # Custom grid size:
//   LLG_CHECK_BASE_NX=64 LLG_CHECK_BASE_NY=64 \
//     cargo run --release --bin amr_demag_check
//
//   # Custom AMR depth:
//   LLG_AMR_MAX_LEVEL=2 cargo run --release --bin amr_demag_check
//
//   # Select specific modes via env:
//   LLG_AMR_DEMAG_MODE=composite cargo run --release --bin amr_demag_check

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use llg_sim::effective_field::{demag_fft_uniform, demag_poisson_mg};
use llg_sim::effective_field::coarse_fft_demag;
use llg_sim::effective_field::mg_composite;
use llg_sim::grid::Grid2D;
use llg_sim::params::{DemagMethod, Material};
use llg_sim::vector_field::VectorField2D;

use llg_sim::amr::indicator::IndicatorKind;
use llg_sim::amr::interp::sample_bilinear;
use llg_sim::amr::regrid::maybe_regrid_nested_levels;
use llg_sim::amr::{
    AmrHierarchy2D, ClusterPolicy, Connectivity, Rect2i, RegridPolicy,
};

// =====================================================================
// Constants
// =====================================================================

const PI: f64 = std::f64::consts::PI;

// =====================================================================
// Utility functions
// =====================================================================

fn ensure_dir(path: &str) {
    if !Path::new(path).exists() {
        fs::create_dir_all(path).unwrap();
    }
}

fn idx(i: usize, j: usize, nx: usize) -> usize {
    j * nx + i
}

fn pow_usize(mut base: usize, mut exp: usize) -> usize {
    let mut out = 1usize;
    while exp > 0 {
        if (exp & 1) == 1 {
            out = out.saturating_mul(base);
        }
        exp >>= 1;
        if exp > 0 {
            base = base.saturating_mul(base);
        }
    }
    out
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// =====================================================================
// Analytic vortex state (Usov-type)
// =====================================================================

/// Evaluate the analytic vortex at physical coordinates (x, y).
///
/// mx = −sin(θ),  my = cos(θ),  mz = p · exp(−r²/R_c²)
/// Renormalised to |m| = 1.
///
/// `cx`, `cy`: vortex centre in metres.
/// `polarity`: +1 or −1.
/// `core_radius`: R_c in metres (typically ~2 × l_ex).
fn vortex_at(x: f64, y: f64, cx: f64, cy: f64, polarity: f64, core_radius: f64) -> [f64; 3] {
    let ddx = x - cx;
    let ddy = y - cy;
    let r2 = ddx * ddx + ddy * ddy;
    let r = r2.sqrt();
    let rc2 = core_radius * core_radius;

    // Azimuthal unit vector: (−sin θ, cos θ)
    let (mx, my) = if r > 1e-20 {
        (-ddy / r, ddx / r)
    } else {
        (0.0, 0.0)
    };

    // Out-of-plane core profile
    let mz = polarity * (-r2 / rc2).exp();

    // Renormalise to |m| = 1
    let mag = (mx * mx + my * my + mz * mz).sqrt();
    let inv = if mag > 1e-30 { 1.0 / mag } else { 0.0 };
    [mx * inv, my * inv, mz * inv]
}

/// Initialise a VectorField2D with an analytic vortex centred in the domain.
fn init_vortex(grid: &Grid2D, polarity: f64, core_radius: f64) -> VectorField2D {
    let mut out = VectorField2D::new(*grid);
    let cx = grid.nx as f64 * grid.dx * 0.5;
    let cy = grid.ny as f64 * grid.dy * 0.5;

    for j in 0..grid.ny {
        let y = (j as f64 + 0.5) * grid.dy;
        for i in 0..grid.nx {
            let x = (i as f64 + 0.5) * grid.dx;
            out.data[idx(i, j, grid.nx)] = vortex_at(x, y, cx, cy, polarity, core_radius);
        }
    }
    out
}

/// Re-initialise ALL patch M from the analytic vortex at patch-native resolution.
///
/// This is critical: after regrid, patches are initialised by interpolating from
/// the coarse grid (which is under-resolved at the vortex core).  Reinitialising
/// from the analytic formula gives patches genuinely fine-scale data that the
/// coarse grid does not have — which is the entire point of AMR.
///
/// Without this step, coarse_fft would restrict the interpolated patch M back
/// to the coarse grid and get approximately the original coarse M, making the
/// test trivial.
fn reinit_patches_from_vortex(
    h: &mut AmrHierarchy2D,
    polarity: f64,
    core_radius: f64,
) {
    let base_dx = h.base_grid.dx;
    let base_dy = h.base_grid.dy;
    let cx = h.base_grid.nx as f64 * base_dx * 0.5;
    let cy = h.base_grid.ny as f64 * base_dy * 0.5;

    // L1 patches
    for p in &mut h.patches {
        let pdx = p.grid.dx;
        let pdy = p.grid.dy;
        let pnx = p.grid.nx;
        let pny = p.grid.ny;
        let gh = p.ghost;
        let cr = p.coarse_rect;

        // Physical origin of cell (0,0) on the patch grid:
        //   x_origin = cr.i0 * base_dx - gh * pdx
        let x0 = cr.i0 as f64 * base_dx - gh as f64 * pdx;
        let y0 = cr.j0 as f64 * base_dy - gh as f64 * pdy;

        for j in 0..pny {
            let y = y0 + (j as f64 + 0.5) * pdy;
            for i in 0..pnx {
                let x = x0 + (i as f64 + 0.5) * pdx;
                p.m.data[j * pnx + i] = vortex_at(x, y, cx, cy, polarity, core_radius);
            }
        }
    }

    // L2+ patches
    for lvl_patches in &mut h.patches_l2plus {
        for p in lvl_patches {
            let pdx = p.grid.dx;
            let pdy = p.grid.dy;
            let pnx = p.grid.nx;
            let pny = p.grid.ny;
            let gh = p.ghost;
            let cr = p.coarse_rect;

            let x0 = cr.i0 as f64 * base_dx - gh as f64 * pdx;
            let y0 = cr.j0 as f64 * base_dy - gh as f64 * pdy;

            for j in 0..pny {
                let y = y0 + (j as f64 + 0.5) * pdy;
                for i in 0..pnx {
                    let x = x0 + (i as f64 + 0.5) * pdx;
                    p.m.data[j * pnx + i] = vortex_at(x, y, cx, cy, polarity, core_radius);
                }
            }
        }
    }
}

// =====================================================================
// Comparison utilities
// =====================================================================

/// Sample a fine-grid VectorField2D at each cell centre of a coarse grid.
/// Returns a coarse-resolution VectorField2D.
fn sample_fine_to_coarse(fine: &VectorField2D, coarse_grid: &Grid2D) -> VectorField2D {
    let mut out = VectorField2D::new(*coarse_grid);
    for j in 0..coarse_grid.ny {
        let y = (j as f64 + 0.5) * coarse_grid.dy;
        for i in 0..coarse_grid.nx {
            let x = (i as f64 + 0.5) * coarse_grid.dx;
            let v = sample_bilinear(fine, x, y);
            out.data[idx(i, j, coarse_grid.nx)] = v;
        }
    }
    out
}

/// Compute component-wise and total RMSE between two coarse-grid B fields.
///
/// Returns (rmse_bx, rmse_by, rmse_bz, rmse_total, max_delta).
fn component_rmse(a: &VectorField2D, b: &VectorField2D) -> (f64, f64, f64, f64, f64) {
    assert_eq!(a.grid.nx, b.grid.nx);
    assert_eq!(a.grid.ny, b.grid.ny);

    let n = (a.grid.nx * a.grid.ny) as f64;
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sz = 0.0_f64;
    let mut maxd = 0.0_f64;

    for k in 0..a.data.len().min(b.data.len()) {
        let da = a.data[k];
        let db = b.data[k];
        let ex = da[0] - db[0];
        let ey = da[1] - db[1];
        let ez = da[2] - db[2];
        sx += ex * ex;
        sy += ey * ey;
        sz += ez * ez;
        let d = (ex * ex + ey * ey + ez * ez).sqrt();
        if d > maxd { maxd = d; }
    }

    let rmse_bx = (sx / n).sqrt();
    let rmse_by = (sy / n).sqrt();
    let rmse_bz = (sz / n).sqrt();
    let rmse_total = ((sx + sy + sz) / n).sqrt();
    (rmse_bx, rmse_by, rmse_bz, rmse_total, maxd)
}

/// Compute regional RMSE: cells inside patch footprints vs outside vs interface.
///
/// `inside_mask[k]` is true if coarse cell k is covered by any L1 patch.
/// `interface_mask[k]` is true if the cell is within 1 cell of a patch boundary.
fn regional_rmse(
    a: &VectorField2D,
    b: &VectorField2D,
    inside_mask: &[bool],
    interface_mask: &[bool],
) -> (f64, f64, f64) {
    let nx = a.grid.nx;
    let ny = a.grid.ny;
    let mut sum_in = 0.0_f64;
    let mut n_in = 0usize;
    let mut sum_out = 0.0_f64;
    let mut n_out = 0usize;
    let mut sum_iface = 0.0_f64;
    let mut n_iface = 0usize;

    for j in 0..ny {
        for i in 0..nx {
            let k = idx(i, j, nx);
            let da = a.data[k];
            let db = b.data[k];
            let d2 = (da[0] - db[0]).powi(2) + (da[1] - db[1]).powi(2) + (da[2] - db[2]).powi(2);

            if interface_mask[k] {
                sum_iface += d2;
                n_iface += 1;
            } else if inside_mask[k] {
                sum_in += d2;
                n_in += 1;
            } else {
                sum_out += d2;
                n_out += 1;
            }
        }
    }

    let rmse_in = if n_in > 0 { (sum_in / n_in as f64).sqrt() } else { 0.0 };
    let rmse_out = if n_out > 0 { (sum_out / n_out as f64).sqrt() } else { 0.0 };
    let rmse_iface = if n_iface > 0 { (sum_iface / n_iface as f64).sqrt() } else { 0.0 };
    (rmse_in, rmse_out, rmse_iface)
}

/// Build inside and interface masks from the current hierarchy L1 patches.
fn build_patch_masks(h: &AmrHierarchy2D) -> (Vec<bool>, Vec<bool>) {
    let nx = h.base_grid.nx;
    let ny = h.base_grid.ny;
    let mut inside = vec![false; nx * ny];

    // Mark cells covered by any L1+ patch
    for p in &h.patches {
        let r = p.coarse_rect;
        for j in r.j0..r.j0 + r.ny {
            for i in r.i0..r.i0 + r.nx {
                if i < nx && j < ny {
                    inside[idx(i, j, nx)] = true;
                }
            }
        }
    }
    // L2+ patches map to sub-regions of L1 — their coarse_rect is in L0 coords
    for lvl in &h.patches_l2plus {
        for p in lvl {
            let r = p.coarse_rect;
            for j in r.j0..r.j0 + r.ny {
                for i in r.i0..r.i0 + r.nx {
                    if i < nx && j < ny {
                        inside[idx(i, j, nx)] = true;
                    }
                }
            }
        }
    }

    // Interface: inside cells that have at least one outside neighbour
    let mut interface = vec![false; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let k = idx(i, j, nx);
            if !inside[k] { continue; }

            let at_border = (i > 0 && !inside[idx(i-1, j, nx)])
                || (i + 1 < nx && !inside[idx(i+1, j, nx)])
                || (j > 0 && !inside[idx(i, j-1, nx)])
                || (j + 1 < ny && !inside[idx(i, j+1, nx)]);

            if at_border {
                interface[k] = true;
            }
        }
    }

    (inside, interface)
}

// =====================================================================
// Level-count helpers (matching other benchmarks)
// =====================================================================

fn level_patch_count(h: &AmrHierarchy2D, lvl: usize) -> usize {
    match lvl {
        1 => h.patches.len(),
        l if l >= 2 => h.patches_l2plus.get(l - 2).map(|v| v.len()).unwrap_or(0),
        _ => 0,
    }
}

fn level_rects(h: &AmrHierarchy2D, lvl: usize) -> Vec<Rect2i> {
    match lvl {
        1 => h.patches.iter().map(|p| p.coarse_rect).collect(),
        l if l >= 2 => h
            .patches_l2plus
            .get(l - 2)
            .map(|v| v.iter().map(|p| p.coarse_rect).collect())
            .unwrap_or_else(Vec::new),
        _ => Vec::new(),
    }
}

// =====================================================================
// CSV output
// =====================================================================

fn write_b_csv(path: &str, b: &VectorField2D) {
    let f = File::create(path).unwrap();
    let mut w = BufWriter::new(f);
    writeln!(w, "i,j,bx,by,bz").unwrap();
    for j in 0..b.grid.ny {
        for i in 0..b.grid.nx {
            let v = b.data[idx(i, j, b.grid.nx)];
            writeln!(w, "{},{},{:.10e},{:.10e},{:.10e}", i, j, v[0], v[1], v[2]).unwrap();
        }
    }
}

// =====================================================================
// Main
// =====================================================================

fn main() {
    // ---- CLI flags ----
    let args: Vec<String> = std::env::args().collect();
    let do_csv = args.iter().any(|a| a == "--csv");
    let skip_fine = args.iter().any(|a| a == "--skip-fine-ref");

    let out_dir = "out/amr_demag_check";
    ensure_dir(out_dir);

    // ---- Tunable parameters (env-var overridable) ----
    let amr_max_level: usize = env_or("LLG_AMR_MAX_LEVEL", 3);
    let ratio = 2usize;
    let ghost = 2usize;

    // ---- Physical domain ----
    // 500 nm × 500 nm Permalloy square, dz = dx (cubic cells, stresses FK 2D kernel)
    // Use LLG_CHECK_DZ to override (e.g. 1e-9 for thin-film, 100e-9 for near-infinite)
    let base_nx: usize = env_or("LLG_CHECK_BASE_NX", 128);
    let base_ny: usize = env_or("LLG_CHECK_BASE_NY", 128);
    let lx: f64 = env_or("LLG_CHECK_LX", 500.0e-9); // 500 nm
    let ly: f64 = env_or("LLG_CHECK_LY", 500.0e-9); // 500 nm

    let dx = lx / base_nx as f64;
    let dy = ly / base_ny as f64;

    // Default dz = dx (cubic cells) — stresses the FK 2D kernel with finite
    // thickness.  The 2D log kernel is exact only for dz → ∞.
    // Set LLG_CHECK_DZ=100e-9 to test the near-infinite-thickness regime.
    // Set LLG_CHECK_DZ=1e-9 to test thin-film regime (expect FK degradation).
    let dz: f64 = env_or("LLG_CHECK_DZ", dx);

    let ref_ratio_total = pow_usize(ratio, amr_max_level);
    let fine_nx = base_nx * ref_ratio_total;
    let fine_ny = base_ny * ref_ratio_total;

    // ---- Material: Permalloy (no anisotropy, no DMI) ----
    let ms: f64 = env_or("LLG_CHECK_MS", 8.0e5);
    let a_ex: f64 = env_or("LLG_CHECK_AEX", 1.3e-11);
    let l_ex = (2.0 * a_ex / (4.0 * PI * 1e-7 * ms * ms)).sqrt();

    let mat = Material {
        ms,
        a_ex,
        k_u: 0.0,
        easy_axis: [0.0, 0.0, 1.0],
        dmi: None,
        demag: true,
        demag_method: DemagMethod::FftUniform,
    };

    // ---- Grids ----
    let base_grid = Grid2D::new(base_nx, base_ny, dx, dy, dz);
    let fine_grid = Grid2D::new(
        fine_nx, fine_ny,
        dx / ref_ratio_total as f64,
        dy / ref_ratio_total as f64,
        dz,
    );

    // ---- Vortex parameters ----
    let polarity = 1.0;
    let core_radius = 2.0 * l_ex; // 2 × exchange length

    // ---- Demag mode from env (for filtering) ----
    let mode_filter: Option<String> = std::env::var("LLG_AMR_DEMAG_MODE").ok()
        .map(|s| s.trim().to_ascii_lowercase());

    // ---- Print header ----
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  AMR Demag Accuracy Check — Frozen-Magnetisation Benchmark   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Domain:    {:.0} nm × {:.0} nm × {:.1} nm  (dz/dx = {:.2})", lx * 1e9, ly * 1e9, dz * 1e9, dz / dx);
    println!("Base grid: {} × {}   dx={:.3e}  dy={:.3e}", base_nx, base_ny, dx, dy);
    println!("Fine grid: {} × {}   dx={:.3e}  dy={:.3e}", fine_nx, fine_ny, fine_grid.dx, fine_grid.dy);
    println!("AMR:       {} levels, ratio={}, ghost={}", amr_max_level, ratio, ghost);
    println!("Material:  Ms={ms:.2e}  A_ex={a_ex:.2e}  l_ex={:.2} nm", l_ex * 1e9);
    println!("Vortex:    polarity=+1  R_core={:.2} nm ({:.1} × l_ex)",
        core_radius * 1e9, core_radius / l_ex);
    println!("Output:    {out_dir}");
    if skip_fine { println!("  --skip-fine-ref: uniform fine FFT reference SKIPPED"); }
    if do_csv { println!("  --csv: per-cell CSV output enabled"); }
    if let Some(ref m) = mode_filter {
        println!("  LLG_AMR_DEMAG_MODE={m}: running selected mode only");
    }
    let regime = if (dz / dx - 1.0).abs() < 0.01 {
        "cubic cells — FK 2D kernel approximate (exact only for dz → ∞)"
    } else if dz / dx < 0.5 {
        "THIN FILM — FK 2D kernel will overestimate in-plane fields"
    } else if dz / dx > 2.0 {
        "thick film — FK 2D kernel approaching exact"
    } else {
        "intermediate aspect ratio"
    };
    println!("Regime:    {}", regime);
    println!();

    // =====================================================================
    // Step 1: Create analytic vortex state on both grids
    // =====================================================================

    let t0 = Instant::now();

    println!("Vortex initialised on coarse grid ({} × {})", base_nx, base_ny);

    // =====================================================================
    // Step 2: Create AMR hierarchy + regrid
    // =====================================================================

    let m_coarse_amr = init_vortex(&base_grid, polarity, core_radius);
    let mut h = AmrHierarchy2D::new(base_grid, m_coarse_amr, ratio, ghost);

    // Regrid: place patches around the vortex core where gradients are strong
    let indicator_kind = if std::env::var("LLG_AMR_INDICATOR").is_ok() {
        IndicatorKind::from_env()
    } else {
        IndicatorKind::Composite { frac: 0.10 }
    };

    let buffer_cells = 4usize;
    let boundary_layer = 0usize; // no geometry mask → no boundary layer
    let cluster_policy = ClusterPolicy {
        indicator: indicator_kind,
        buffer_cells,
        boundary_layer,
        connectivity: Connectivity::Eight,
        merge_distance: 1,
        min_patch_area: 16,
        max_patches: 0,
        min_efficiency: 0.70,
        max_flagged_fraction: 0.50,
        confine_dilation: false,
    };
    let regrid_policy = RegridPolicy {
        indicator: indicator_kind,
        buffer_cells,
        boundary_layer,
        min_change_cells: 1,
        min_area_change_frac: 0.01,
    };

    let current_patches: Vec<Rect2i> = Vec::new();
    if let Some((_new_rects, stats)) =
        maybe_regrid_nested_levels(&mut h, &current_patches, regrid_policy, cluster_policy)
    {
        println!(
            "Regrid: {} patches flagged, threshold={:.4e}",
            stats.flagged_cells, stats.threshold
        );
    } else {
        println!("Regrid: no patches created (vortex may be too smooth for threshold)");
    }

    // Fill ghosts so patch M is properly initialised
    h.fill_patch_ghosts();

    // CRITICAL: Reinitialise patch M from the analytic formula at patch-native
    // resolution.  Without this, patch M is just interpolated from the coarse
    // grid and the coarse_fft/composite modes would trivially match fft_coarse.
    reinit_patches_from_vortex(&mut h, polarity, core_radius);

    // Also restrict the now-fine patch data back to the coarse level so that
    // h.coarse reflects the best available M at L0.
    h.restrict_patches_to_coarse();

    // Print patch summary
    let mut lvl_counts = String::new();
    for lvl in 1..=amr_max_level {
        if lvl > 1 { lvl_counts.push_str(" | "); }
        lvl_counts.push_str(&format!("L{}: {} patches", lvl, level_patch_count(&h, lvl)));
    }
    println!("Patches:   {}", lvl_counts);

    // Build regional masks
    let (inside_mask, interface_mask) = build_patch_masks(&h);
    let n_inside: usize = inside_mask.iter().filter(|&&b| b).count();
    let n_interface: usize = interface_mask.iter().filter(|&&b| b).count();
    let n_total = base_nx * base_ny;
    let n_outside = n_total - n_inside;
    println!(
        "Coverage:  {} inside ({:.1}%), {} outside, {} interface cells",
        n_inside, 100.0 * n_inside as f64 / n_total as f64,
        n_outside, n_interface,
    );
    println!();

    // =====================================================================
    // Step 3: Compute reference — FFT on uniform fine grid
    // =====================================================================

    let b_ref_coarse: VectorField2D;
    let mut b_fine_opt: Option<VectorField2D> = None;

    if !skip_fine {
        println!("Computing FFT reference on {} × {} fine grid ...", fine_nx, fine_ny);
        let t1 = Instant::now();

        // Create vortex on fine grid
        let m_fine = init_vortex(&fine_grid, polarity, core_radius);

        let mut b_fine = VectorField2D::new(fine_grid);
        demag_fft_uniform::compute_demag_field(&fine_grid, &m_fine, &mut b_fine, &mat);

        let t_fine = t1.elapsed().as_secs_f64();
        println!("  Fine FFT done in {:.3} s", t_fine);

        // Sample to coarse grid for comparison
        b_ref_coarse = sample_fine_to_coarse(&b_fine, &base_grid);

        // Stats
        let b_max = b_fine.data[..fine_grid.n_cells()].iter()
            .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
            .fold(0.0_f64, f64::max);
        println!("  max|B_demag| = {:.4e} T", b_max);

        if do_csv {
            write_b_csv(&format!("{out_dir}/b_ref_coarse.csv"), &b_ref_coarse);
        }
        
        // Save for patch-level comparison in Step 6
        b_fine_opt = Some(b_fine);
        
        println!();
    } else {
        // Use coarse-only FFT as reference
        println!("Fine reference skipped; using coarse-only FFT as reference.");
        let mut b_tmp = VectorField2D::new(base_grid);
        demag_fft_uniform::compute_demag_field(&base_grid, &h.coarse, &mut b_tmp, &mat);
        b_ref_coarse = b_tmp;
        println!();
    }

    // Reference B stats
    let bref_max = b_ref_coarse.data[..base_grid.n_cells()].iter()
        .map(|v| (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())
        .fold(0.0_f64, f64::max);
    let bref_bz_max = b_ref_coarse.data[..base_grid.n_cells()].iter()
        .map(|v| v[2].abs())
        .fold(0.0_f64, f64::max);

    // =====================================================================
    // Step 4: Compute each mode and compare
    // =====================================================================

    struct ModeResult {
        name: String,
        time_ms: f64,
        rmse_bx: f64,
        rmse_by: f64,
        rmse_bz: f64,
        rmse_total: f64,
        max_delta: f64,
        rmse_inside: f64,
        rmse_outside: f64,
        rmse_interface: f64,
    }

    let mut results: Vec<ModeResult> = Vec::new();

    let should_run = |name: &str| -> bool {
        match &mode_filter {
            None => true,
            Some(f) => {
                let n = name.to_ascii_lowercase();
                n.contains(f.as_str()) || f.contains(n.as_str())
                    || (f == "coarsefft" && n == "coarse_fft")
                    || (f == "cfft" && n == "coarse_fft")
            }
        }
    };

    // Mode 1: FFT on coarse grid (no patches)
    if should_run("fft_coarse") {
        println!("Mode: fft_coarse (FFT Newell on {} × {} coarse grid) ...", base_nx, base_ny);
        let t1 = Instant::now();
        let mut b = VectorField2D::new(base_grid);
        demag_fft_uniform::compute_demag_field(&base_grid, &h.coarse, &mut b, &mat);
        let dt = t1.elapsed().as_secs_f64() * 1e3;

        let (rx, ry, rz, rt, md) = component_rmse(&b, &b_ref_coarse);
        let (ri, ro, rif) = regional_rmse(&b, &b_ref_coarse, &inside_mask, &interface_mask);
        let rel = if bref_max > 0.0 { rt / bref_max * 100.0 } else { 0.0 };

        println!("  RMSE: {rt:.4e} T ({rel:.2}%)  Bx={rx:.3e}  By={ry:.3e}  Bz={rz:.3e}  max={md:.3e}  [{dt:.1} ms]");
        println!("  Regional: inside={ri:.3e}  outside={ro:.3e}  interface={rif:.3e}");

        if do_csv {
            write_b_csv(&format!("{out_dir}/b_fft_coarse.csv"), &b);
        }

        results.push(ModeResult {
            name: "fft_coarse".into(), time_ms: dt,
            rmse_bx: rx, rmse_by: ry, rmse_bz: rz, rmse_total: rt, max_delta: md,
            rmse_inside: ri, rmse_outside: ro, rmse_interface: rif,
        });
    }

    // Mode 2: CoarseFft (Phase 1 solver: M-restriction + L0 FFT)
    if should_run("coarse_fft") {
        // Warm-up: pre-build and cache the Demag2D operator for whatever grid
        // size compute_coarse_fft_demag will use (base grid at R=1, or the
        // super-coarse demag grid at R>1).  Without this, the first call pays
        // the one-time FFT-planning + kernel-loading cost (~0.5–1s), which
        // dominates the timing and masks the actual per-step FFT speedup.
        {
            let mut b_warmup = VectorField2D::new(base_grid);
            let _ = coarse_fft_demag::compute_coarse_fft_demag(&h, &mat, &mut b_warmup);
        }

        println!("Mode: coarse_fft (M-restriction + L0 FFT) ...");
        let t1 = Instant::now();
        let mut b_coarse = VectorField2D::new(base_grid);
        let (_b_l1, _b_l2plus) = coarse_fft_demag::compute_coarse_fft_demag(
            &h, &mat, &mut b_coarse,
        );
        let dt = t1.elapsed().as_secs_f64() * 1e3;

        let (rx, ry, rz, rt, md) = component_rmse(&b_coarse, &b_ref_coarse);
        let (ri, ro, rif) = regional_rmse(&b_coarse, &b_ref_coarse, &inside_mask, &interface_mask);
        let rel = if bref_max > 0.0 { rt / bref_max * 100.0 } else { 0.0 };

        println!("  RMSE: {rt:.4e} T ({rel:.2}%)  Bx={rx:.3e}  By={ry:.3e}  Bz={rz:.3e}  max={md:.3e}  [{dt:.1} ms]");
        println!("  Regional: inside={ri:.3e}  outside={ro:.3e}  interface={rif:.3e}");

        if do_csv {
            write_b_csv(&format!("{out_dir}/b_coarse_fft.csv"), &b_coarse);
        }

        results.push(ModeResult {
            name: "coarse_fft".into(), time_ms: dt,
            rmse_bx: rx, rmse_by: ry, rmse_bz: rz, rmse_total: rt, max_delta: md,
            rmse_inside: ri, rmse_outside: ro, rmse_interface: rif,
        });
    }

    // Mode 3: Composite (enhanced-RHS FK/MG)
    if should_run("composite") {
        println!("Mode: composite (enhanced-RHS FK/MG) ...");
        let t1 = Instant::now();
        let mut b_coarse = VectorField2D::new(base_grid);
        let (_b_l1, _b_l2plus) = mg_composite::compute_composite_demag(
            &h, &mat, &mut b_coarse,
        );
        let dt = t1.elapsed().as_secs_f64() * 1e3;

        let (rx, ry, rz, rt, md) = component_rmse(&b_coarse, &b_ref_coarse);
        let (ri, ro, rif) = regional_rmse(&b_coarse, &b_ref_coarse, &inside_mask, &interface_mask);
        let rel = if bref_max > 0.0 { rt / bref_max * 100.0 } else { 0.0 };

        println!("  RMSE: {rt:.4e} T ({rel:.2}%)  Bx={rx:.3e}  By={ry:.3e}  Bz={rz:.3e}  max={md:.3e}  [{dt:.1} ms]");
        println!("  Regional: inside={ri:.3e}  outside={ro:.3e}  interface={rif:.3e}");

        if do_csv {
            write_b_csv(&format!("{out_dir}/b_composite.csv"), &b_coarse);
        }

        results.push(ModeResult {
            name: "composite".into(), time_ms: dt,
            rmse_bx: rx, rmse_by: ry, rmse_bz: rz, rmse_total: rt, max_delta: md,
            rmse_inside: ri, rmse_outside: ro, rmse_interface: rif,
        });
    }

    // Mode 4: Standalone MG on coarse grid (no patches — FK/MG baseline)
    if should_run("mg_coarse") {
        println!("Mode: mg_coarse (standalone FK/MG on {} × {} coarse) ...", base_nx, base_ny);
        let t1 = Instant::now();
        let mut b = VectorField2D::new(base_grid);
        demag_poisson_mg::compute_demag_field_poisson_mg(&base_grid, &h.coarse, &mut b, &mat);
        let dt = t1.elapsed().as_secs_f64() * 1e3;

        let (rx, ry, rz, rt, md) = component_rmse(&b, &b_ref_coarse);
        let (ri, ro, rif) = regional_rmse(&b, &b_ref_coarse, &inside_mask, &interface_mask);
        let rel = if bref_max > 0.0 { rt / bref_max * 100.0 } else { 0.0 };

        println!("  RMSE: {rt:.4e} T ({rel:.2}%)  Bx={rx:.3e}  By={ry:.3e}  Bz={rz:.3e}  max={md:.3e}  [{dt:.1} ms]");
        println!("  Regional: inside={ri:.3e}  outside={ro:.3e}  interface={rif:.3e}");

        if do_csv {
            write_b_csv(&format!("{out_dir}/b_mg_coarse.csv"), &b);
        }

        results.push(ModeResult {
            name: "mg_coarse".into(), time_ms: dt,
            rmse_bx: rx, rmse_by: ry, rmse_bz: rz, rmse_total: rt, max_delta: md,
            rmse_inside: ri, rmse_outside: ro, rmse_interface: rif,
        });
    }

    // =====================================================================
    // Step 5: Summary
    // =====================================================================

    let wall = t0.elapsed().as_secs_f64();

    println!();
    println!("╔════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Demag Accuracy Summary                                                                     ║");
    println!("╠══════════════╤════════════╤════════════╤════════════╤════════════╤════════╤═══════════════════╣");
    println!("║ Mode         │ RMSE total │   RMSE Bx  │   RMSE By  │   RMSE Bz  │ max ΔB │ time (ms)       ║");
    println!("╠══════════════╪════════════╪════════════╪════════════╪════════════╪════════╪═══════════════════╣");

    for r in &results {
        let rel = if bref_max > 0.0 { r.rmse_total / bref_max * 100.0 } else { 0.0 };
        println!(
            "║ {:12} │ {:.4e} │ {:.4e} │ {:.4e} │ {:.4e} │{:.3e}│ {:>8.1}        ║",
            r.name, r.rmse_total, r.rmse_bx, r.rmse_by, r.rmse_bz, r.max_delta, r.time_ms,
        );
        println!(
            "║              │  ({:5.2}%)  │            │            │            │        │                 ║",
            rel,
        );
    }
    println!("╚══════════════╧════════════╧════════════╧════════════╧════════════╧════════╧═══════════════════╝");

    println!();
    println!("Regional RMSE (inside patch / outside patch / at interface):");
    for r in &results {
        println!(
            "  {:12}:  inside={:.3e}  outside={:.3e}  interface={:.3e}",
            r.name, r.rmse_inside, r.rmse_outside, r.rmse_interface,
        );
    }

    println!();
    println!("Reference:  max|B_demag| = {:.4e} T,  max|Bz| = {:.4e} T", bref_max, bref_bz_max);
    println!("Coverage:   {} inside, {} outside, {} interface ({:.1}% patched)",
        n_inside, n_outside, n_interface, 100.0 * n_inside as f64 / n_total as f64);
    println!("Total wall: {:.3} s", wall);

    // ---- Write results CSV ----
    {
        let csv_path = format!("{out_dir}/results.csv");
        let f = File::create(&csv_path).unwrap();
        let mut w = BufWriter::new(f);
        writeln!(w, "mode,rmse_total,rmse_bx,rmse_by,rmse_bz,max_delta,rmse_inside,rmse_outside,rmse_interface,time_ms,rel_pct").unwrap();
        for r in &results {
            let rel = if bref_max > 0.0 { r.rmse_total / bref_max * 100.0 } else { 0.0 };
            writeln!(w,
                "{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.2},{:.4}",
                r.name, r.rmse_total, r.rmse_bx, r.rmse_by, r.rmse_bz, r.max_delta,
                r.rmse_inside, r.rmse_outside, r.rmse_interface, r.time_ms, rel,
            ).unwrap();
        }
        println!("Results CSV: {csv_path}");
    }

    // ---- Write summary text ----
    {
        let txt_path = format!("{out_dir}/summary.txt");
        let mut f = File::create(&txt_path).unwrap();
        writeln!(f, "AMR Demag Check — Frozen-Magnetisation Benchmark").unwrap();
        writeln!(f, "================================================").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "Domain:  {:.0} nm × {:.0} nm × {:.1} nm  (dz/dx = {:.2})", lx*1e9, ly*1e9, dz*1e9, dz/dx).unwrap();
        writeln!(f, "Base:    {} × {}   dx={:.3e}", base_nx, base_ny, dx).unwrap();
        writeln!(f, "Fine:    {} × {}   dx={:.3e}", fine_nx, fine_ny, fine_grid.dx).unwrap();
        writeln!(f, "AMR:     {} levels, ratio={}", amr_max_level, ratio).unwrap();
        writeln!(f, "Ms={ms:.2e}  A_ex={a_ex:.2e}  l_ex={:.2} nm", l_ex*1e9).unwrap();
        writeln!(f, "Reference: {}",
            if skip_fine { "coarse-only FFT (fine skipped)" } else { "uniform fine FFT" }).unwrap();
        writeln!(f).unwrap();
        writeln!(f, "Patches:").unwrap();
        for lvl in 1..=amr_max_level {
            let rects = level_rects(&h, lvl);
            writeln!(f, "  L{}: {} patches", lvl, rects.len()).unwrap();
            for (pid, r) in rects.iter().enumerate() {
                writeln!(f, "    [{}] i0={} j0={} nx={} ny={}", pid, r.i0, r.j0, r.nx, r.ny).unwrap();
            }
        }
        writeln!(f).unwrap();
        writeln!(f, "Coverage: {:.1}% ({} of {} cells)", 100.0*n_inside as f64/n_total as f64, n_inside, n_total).unwrap();
        writeln!(f).unwrap();

        writeln!(f, "{:>14} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "mode", "rmse_total", "rmse_bx", "rmse_by", "rmse_bz", "max_delta",
            "inside", "outside", "interface", "ms").unwrap();
        for r in &results {
            writeln!(f, "{:>14} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>8.1}",
                r.name, r.rmse_total, r.rmse_bx, r.rmse_by, r.rmse_bz, r.max_delta,
                r.rmse_inside, r.rmse_outside, r.rmse_interface, r.time_ms).unwrap();
        }
        writeln!(f).unwrap();
        writeln!(f, "Wall time: {:.3} s", wall).unwrap();
        println!("Summary:     {txt_path}");
    }

    println!();
    println!("Done.");

    // =====================================================================
    // Step 6: Patch-level B accuracy (the composite V-cycle test)
    //
    // This compares B at FINE-RESOLUTION patch cells, not on the coarse grid.
    // This is where the composite V-cycle's advantage should appear:
    //   - coarse_fft gives interpolated coarse B at patch cells
    //   - composite vcycle gives -μ₀∇φ from patch-level φ
    //
    // Reference: b_fine sampled at each patch cell's physical position.
    // =====================================================================

    if let Some(ref b_fine) = b_fine_opt {
        println!();
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║  Patch-Level B Accuracy (fine-resolution comparison)          ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();

        // Run coarse_fft to get interpolated patch B.
        let mut b_coarse_tmp = VectorField2D::new(base_grid);
        let (b_l1_cfft, _) = coarse_fft_demag::compute_coarse_fft_demag(
            &h, &mat, &mut b_coarse_tmp);

        // Run composite (uses vcycle if LLG_DEMAG_COMPOSITE_VCYCLE=1).
        let mut b_coarse_comp = VectorField2D::new(base_grid);
        let (b_l1_comp, _) = mg_composite::compute_composite_demag(
            &h, &mat, &mut b_coarse_comp);

        // For each L1 patch, compare both b_l1 arrays against the fine reference.
        println!("  {:>5} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
            "patch", "nx", "ny", "cfft_rmse", "comp_rmse", "cfft_rel%", "comp_rel%");
        println!("  (relative errors normalised to global max|B| = {:.4e} T)", bref_max);
        println!("  {:->5} {:->8} {:->8} {:->10} {:->10} {:->10} {:->10}",
            "", "", "", "", "", "", "");

        let mut total_cfft_se = 0.0f64;
        let mut total_comp_se = 0.0f64;
        let mut total_cells = 0usize;

        for (pi, patch) in h.patches.iter().enumerate() {
            let pnx = patch.grid.nx;
            let pny = patch.grid.ny;
            let gi0 = patch.interior_i0();
            let gj0 = patch.interior_j0();
            let gi1 = patch.interior_i1();
            let gj1 = patch.interior_j1();

            let b_cfft = if pi < b_l1_cfft.len() { &b_l1_cfft[pi] } else { continue };
            let b_comp = if pi < b_l1_comp.len() { &b_l1_comp[pi] } else { continue };

            let mut se_cfft = 0.0f64;
            let mut se_comp = 0.0f64;
            let mut n_cells = 0usize;

            for j in gj0..gj1 {
                for i in gi0..gi1 {
                    let (x, y) = patch.cell_center_xy(i, j);
                    let b_ref = sample_bilinear(b_fine, x, y);
                    let idx = j * pnx + i;

                    let b_cf = b_cfft[idx];
                    let b_co = b_comp[idx];

                    // Squared error (Bx + By only — Bz is the same for both)
                    let err_cfft = (b_cf[0] - b_ref[0]).powi(2) + (b_cf[1] - b_ref[1]).powi(2);
                    let err_comp = (b_co[0] - b_ref[0]).powi(2) + (b_co[1] - b_ref[1]).powi(2);

                    se_cfft += err_cfft;
                    se_comp += err_comp;
                    n_cells += 1;
                }
            }

            if n_cells > 0 {
                let rmse_cfft = (se_cfft / n_cells as f64).sqrt();
                let rmse_comp = (se_comp / n_cells as f64).sqrt();
                // Normalize to GLOBAL B_max, not patch-local
                let rel_cfft = if bref_max > 0.0 { rmse_cfft / bref_max * 100.0 } else { 0.0 };
                let rel_comp = if bref_max > 0.0 { rmse_comp / bref_max * 100.0 } else { 0.0 };

                println!("  {:>5} {:>8} {:>8} {:>10.4e} {:>10.4e} {:>9.2}% {:>9.2}%",
                    pi, pnx, pny, rmse_cfft, rmse_comp, rel_cfft, rel_comp);

                total_cfft_se += se_cfft;
                total_comp_se += se_comp;
                total_cells += n_cells;
            }
        }

        if total_cells > 0 {
            let total_rmse_cfft = (total_cfft_se / total_cells as f64).sqrt();
            let total_rmse_comp = (total_comp_se / total_cells as f64).sqrt();
            let total_rel_cfft = if bref_max > 0.0 { total_rmse_cfft / bref_max * 100.0 } else { 0.0 };
            let total_rel_comp = if bref_max > 0.0 { total_rmse_comp / bref_max * 100.0 } else { 0.0 };

            println!();
            println!("  TOTAL ({} interior cells across {} L1 patches):", total_cells, h.patches.len());
            println!("    coarse_fft (interpolated):  RMSE = {:.4e} T  ({:.2}%)", total_rmse_cfft, total_rel_cfft);
            println!("    composite  (from patch φ):  RMSE = {:.4e} T  ({:.2}%)", total_rmse_comp, total_rel_comp);

            if total_rmse_comp < total_rmse_cfft {
                let improvement = (1.0 - total_rmse_comp / total_rmse_cfft) * 100.0;
                println!("    → composite is {:.1}% more accurate at patch cells", improvement);
            } else {
                let degradation = (total_rmse_comp / total_rmse_cfft - 1.0) * 100.0;
                println!("    → composite is {:.1}% WORSE at patch cells (L0 MG discretisation error dominates)", degradation);
                println!("    → enable PPPM hybrid: LLG_DEMAG_MG_HYBRID_ENABLE=1 LLG_DEMAG_MG_HYBRID_RADIUS=14");
            }

            let vcycle_on = std::env::var("LLG_DEMAG_COMPOSITE_VCYCLE")
                .map(|v| v == "1").unwrap_or(false);
            println!();
            println!("    V-cycle mode: {}", if vcycle_on { "ON (fine ∇φ on patches)" } else { "OFF (interpolated coarse B)" });
            if !vcycle_on {
                println!("    → To test fine-resolution B: LLG_DEMAG_COMPOSITE_VCYCLE=1");
            }
        }
        println!();
    } else {
        println!();
        println!("  (Patch-level comparison skipped — run without --skip-fine-ref to enable)");
    }
}