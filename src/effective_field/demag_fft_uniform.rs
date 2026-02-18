// src/effective_field/demag_fft_uniform.rs
//
// MuMax-like 2D demag (Nz = 1) via FFT-accelerated convolution.
//
// We compute:
//   B_demag_i = K_ij * M_j
// where M = Ms * m is magnetization in A/m and B_demag is Tesla.
//
// This implementation mirrors MuMax3's methodology for Nz=1:
//
// OPEN boundaries (pbc_x=pbc_y=0):
// - zero-padding to 2Nx × 2Ny (linear convolution / open boundaries)
// - kernel computed by brute-force face-charge integration + volume averaging
// - MuMax-style integration point selection (round-to-nearest)
// - MuMax-style staggering (nv *= 2, nw *= 2)
// - enforce parity symmetries exactly across displacements (even-even diagonals, odd-odd kxy)
//
// PERIODIC boundaries (pbc_x>0 and/or pbc_y>0):
// - no padding in periodic directions (circular convolution / wrap-around in those dims)
// - kernel sums contributions from periodic images over MuMax-style ranges and accumulates into wrapped bins
//
// Notes:
// - MuMax stores a dimensionless kernel N_ij (unit "1") and applies μ0 later.
//   Here we store K_ij = μ0 * N_ij directly (Tesla per (A/m)) to keep existing API unchanged.
// - For Nz=1, MuMax's 2D path uses only XX, YY, XY for in-plane and ZZ for out-of-plane.

use crate::grid::Grid2D;
use crate::params::{MU0, Material};
use crate::vector_field::VectorField2D;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use rayon::prelude::*;

static DEMAG_CACHE: OnceLock<Mutex<Option<Demag2D>>> = OnceLock::new();
fn demag_timing_enabled() -> bool {
    std::env::var("LLG_DEMAG_TIMING").is_ok()
}

// For small grids, the transpose+parallel-column FFT can be slower than the simple
// gather-column approach due to extra memory traffic + rayon overhead.
//
// Use a hybrid: serial columns for small padded grids; transpose+parallel columns for larger ones.
static FFT_PAR_THRESHOLD: OnceLock<usize> = OnceLock::new();
const DEFAULT_FFT_PAR_THRESHOLD: usize = 32_768;

fn demag_fft_par_threshold() -> usize {
    *FFT_PAR_THRESHOLD.get_or_init(|| {
        std::env::var("LLG_DEMAG_FFT_PAR_THRESHOLD")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_FFT_PAR_THRESHOLD)
    })
}

#[inline]
fn use_parallel_column_fft(nx: usize, ny: usize) -> bool {
    // Avoid transpose path for tiny sizes.
    if nx < 64 || ny < 32 {
        return false;
    }
    nx.saturating_mul(ny) >= demag_fft_par_threshold()
}

/// Backwards-compatible: open boundaries in x/y (no PBC).
pub fn add_demag_field(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    mat: &Material,
) {
    add_demag_field_pbc(grid, m, b_eff, mat, 0, 0);
}

/// Backwards-compatible: open boundaries in x/y (no PBC).
pub fn compute_demag_field(
    grid: &Grid2D,
    m: &VectorField2D,
    out: &mut VectorField2D,
    mat: &Material,
) {
    compute_demag_field_pbc(grid, m, out, mat, 0, 0);
}

/// Add demag field with periodic boundary conditions in x/y.
/// pbc_x/pbc_y are the number of periodic images used in that direction (MuMax-style finite sum).
pub fn add_demag_field_pbc(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    mat: &Material,
    pbc_x: usize,
    pbc_y: usize,
) {
    if !mat.demag {
        return;
    }

    let cache = DEMAG_CACHE.get_or_init(|| Mutex::new(None));
    let mut guard = cache.lock().expect("DEMAG_CACHE mutex poisoned");

    let rebuild = match guard.as_ref() {
        Some(d) => !same_grid_pbc(&d.grid, grid, d.pbc_x, d.pbc_y, pbc_x, pbc_y),
        None => true,
    };

    if rebuild {
        *guard = Some(Demag2D::new(*grid, pbc_x, pbc_y));
    }

    if let Some(d) = guard.as_mut() {
        d.add_field(m, b_eff, mat.ms);
    }
}

/// Compute demag induction B_demag (Tesla) into `out` (overwrites out), with PBC in x/y.
pub fn compute_demag_field_pbc(
    grid: &Grid2D,
    m: &VectorField2D,
    out: &mut VectorField2D,
    mat: &Material,
    pbc_x: usize,
    pbc_y: usize,
) {
    out.set_uniform(0.0, 0.0, 0.0);
    add_demag_field_pbc(grid, m, out, mat, pbc_x, pbc_y);
}

fn same_grid(a: &Grid2D, b: &Grid2D) -> bool {
    a.nx == b.nx && a.ny == b.ny && a.dx == b.dx && a.dy == b.dy && a.dz == b.dz
}

fn same_grid_pbc(a: &Grid2D, b: &Grid2D, ax: usize, ay: usize, bx: usize, by: usize) -> bool {
    same_grid(a, b) && ax == bx && ay == by
}

#[inline]
fn wrap_index(d: isize, n: usize) -> usize {
    let n = n as isize;
    let mut v = d % n;
    if v < 0 {
        v += n;
    }
    v as usize
}

#[derive(Debug, Clone, Copy)]
struct KernelCacheHeader {
    magic: [u8; 8],
    version: u32,
    nx: u32,
    ny: u32,
    pbc_x: u32,
    pbc_y: u32,
    px: u32,
    py: u32,
    dx: f64,
    dy: f64,
    dz: f64,
    accuracy: f64,
}

impl KernelCacheHeader {
    fn new(grid: Grid2D, pbc_x: usize, pbc_y: usize, px: usize, py: usize, accuracy: f64) -> Self {
        Self {
            magic: *b"LLGDMAG\0",
            version: 2,
            nx: grid.nx as u32,
            ny: grid.ny as u32,
            pbc_x: pbc_x as u32,
            pbc_y: pbc_y as u32,
            px: px as u32,
            py: py as u32,
            dx: grid.dx,
            dy: grid.dy,
            dz: grid.dz,
            accuracy,
        }
    }

    fn matches(
        &self,
        grid: Grid2D,
        pbc_x: usize,
        pbc_y: usize,
        px: usize,
        py: usize,
        accuracy: f64,
    ) -> bool {
        self.magic == *b"LLGDMAG\0"
            && self.version == 2
            && self.nx == grid.nx as u32
            && self.ny == grid.ny as u32
            && self.pbc_x == pbc_x as u32
            && self.pbc_y == pbc_y as u32
            && self.px == px as u32
            && self.py == py as u32
            && self.dx == grid.dx
            && self.dy == grid.dy
            && self.dz == grid.dz
            && self.accuracy == accuracy
    }
}

fn demag_cache_path(
    grid: Grid2D,
    pbc_x: usize,
    pbc_y: usize,
    px: usize,
    py: usize,
    accuracy: f64,
) -> PathBuf {
    let mut dir = PathBuf::from("out");
    dir.push("demag_cache");

    let fname = format!(
        "demag_kernel_nx{}_ny{}_pbcx{}_pbcy{}_px{}_py{}_dx{:.3e}_dy{:.3e}_dz{:.3e}_acc{:.2}.bin",
        grid.nx, grid.ny, pbc_x, pbc_y, px, py, grid.dx, grid.dy, grid.dz, accuracy
    );

    dir.push(fname);
    dir
}

fn write_kernel_kspace(
    path: &PathBuf,
    header: KernelCacheHeader,
    kxx: &[Complex<f64>],
    kxy: &[Complex<f64>],
    kyy: &[Complex<f64>],
    kzz: &[Complex<f64>],
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut f = File::create(path)?;

    // Header
    f.write_all(&header.magic)?;
    f.write_all(&header.version.to_le_bytes())?;
    f.write_all(&header.nx.to_le_bytes())?;
    f.write_all(&header.ny.to_le_bytes())?;
    f.write_all(&header.pbc_x.to_le_bytes())?;
    f.write_all(&header.pbc_y.to_le_bytes())?;
    f.write_all(&header.px.to_le_bytes())?;
    f.write_all(&header.py.to_le_bytes())?;
    f.write_all(&header.dx.to_le_bytes())?;
    f.write_all(&header.dy.to_le_bytes())?;
    f.write_all(&header.dz.to_le_bytes())?;
    f.write_all(&header.accuracy.to_le_bytes())?;

    fn write_complex_vec(f: &mut File, v: &[Complex<f64>]) -> std::io::Result<()> {
        for c in v {
            f.write_all(&c.re.to_le_bytes())?;
            f.write_all(&c.im.to_le_bytes())?;
        }
        Ok(())
    }

    write_complex_vec(&mut f, kxx)?;
    write_complex_vec(&mut f, kxy)?;
    write_complex_vec(&mut f, kyy)?;
    write_complex_vec(&mut f, kzz)?;
    Ok(())
}

fn try_load_kernel_kspace(
    path: &PathBuf,
    grid: Grid2D,
    pbc_x: usize,
    pbc_y: usize,
    px: usize,
    py: usize,
    accuracy: f64,
    kxx: &mut Vec<Complex<f64>>,
    kxy: &mut Vec<Complex<f64>>,
    kyy: &mut Vec<Complex<f64>>,
    kzz: &mut Vec<Complex<f64>>,
) -> std::io::Result<bool> {
    if !path.exists() {
        return Ok(false);
    }

    let mut f = File::open(path)?;

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;

    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);

    // Only support v2 cache here; if mismatch, just return false and rebuild.
    if version != 2 {
        return Ok(false);
    }

    f.read_exact(&mut buf4)?;
    let nx = u32::from_le_bytes(buf4);
    f.read_exact(&mut buf4)?;
    let ny = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf4)?;
    let pbc_x_h = u32::from_le_bytes(buf4);
    f.read_exact(&mut buf4)?;
    let pbc_y_h = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf4)?;
    let px_h = u32::from_le_bytes(buf4);
    f.read_exact(&mut buf4)?;
    let py_h = u32::from_le_bytes(buf4);

    f.read_exact(&mut buf8)?;
    let dx_h = f64::from_le_bytes(buf8);
    f.read_exact(&mut buf8)?;
    let dy_h = f64::from_le_bytes(buf8);
    f.read_exact(&mut buf8)?;
    let dz_h = f64::from_le_bytes(buf8);
    f.read_exact(&mut buf8)?;
    let acc_h = f64::from_le_bytes(buf8);

    let header = KernelCacheHeader {
        magic,
        version,
        nx,
        ny,
        pbc_x: pbc_x_h,
        pbc_y: pbc_y_h,
        px: px_h,
        py: py_h,
        dx: dx_h,
        dy: dy_h,
        dz: dz_h,
        accuracy: acc_h,
    };

    if !header.matches(grid, pbc_x, pbc_y, px, py, accuracy) {
        return Ok(false);
    }

    let n_pad = px * py;

    fn read_complex_vec(
        f: &mut File,
        out: &mut Vec<Complex<f64>>,
        n: usize,
    ) -> std::io::Result<()> {
        out.clear();
        out.reserve(n);
        let mut buf = [0u8; 8];
        for _ in 0..n {
            f.read_exact(&mut buf)?;
            let re = f64::from_le_bytes(buf);
            f.read_exact(&mut buf)?;
            let im = f64::from_le_bytes(buf);
            out.push(Complex::new(re, im));
        }
        Ok(())
    }

    read_complex_vec(&mut f, kxx, n_pad)?;
    read_complex_vec(&mut f, kxy, n_pad)?;
    read_complex_vec(&mut f, kyy, n_pad)?;
    read_complex_vec(&mut f, kzz, n_pad)?;
    Ok(true)
}

// Match your MuMax scripts (you set DemagAccuracy=10 there).
const DEMAG_ACCURACY: f64 = 10.0;

struct Demag2D {
    grid: Grid2D,
    pbc_x: usize,
    pbc_y: usize,

    px: usize,
    py: usize,
    #[allow(dead_code)]
    n_pad: usize,

    // K-space kernel (Tesla per (A/m))
    kxx: Vec<Complex<f64>>,
    kxy: Vec<Complex<f64>>,
    kyy: Vec<Complex<f64>>,
    kzz: Vec<Complex<f64>>,

    // Scratch (padded, complex)
    mx: Vec<Complex<f64>>,
    my: Vec<Complex<f64>>,
    mz: Vec<Complex<f64>>,
    bx: Vec<Complex<f64>>,
    by: Vec<Complex<f64>>,
    bz: Vec<Complex<f64>>,

    fft_x_fwd: Arc<dyn Fft<f64>>,
    fft_x_inv: Arc<dyn Fft<f64>>,
    fft_y_fwd: Arc<dyn Fft<f64>>,
    fft_y_inv: Arc<dyn Fft<f64>>,

    // Scratch buffer for parallel column FFTs via transpose (len = px*py)
    fft_tmp: Vec<Complex<f64>>,
}

impl Demag2D {
    fn new(grid: Grid2D, pbc_x: usize, pbc_y: usize) -> Self {
        let nx = grid.nx;
        let ny = grid.ny;

        // MuMax padSize behavior:
        // - periodic direction: no padding (wrap)
        // - open direction: pad to 2N for FFT performance
        let px = if pbc_x > 0 { nx } else { 2 * nx };
        let py = if pbc_y > 0 { ny } else { 2 * ny };
        let n_pad = px * py;

        let do_timing = demag_timing_enabled();
        let t_total = Instant::now();

        let mut planner = FftPlanner::<f64>::new();
        let fft_x_fwd = planner.plan_fft_forward(px);
        let fft_x_inv = planner.plan_fft_inverse(px);
        let fft_y_fwd = planner.plan_fft_forward(py);
        let fft_y_inv = planner.plan_fft_inverse(py);

        let zero = Complex::new(0.0, 0.0);

        let mut kxx = vec![zero; n_pad];
        let mut kxy = vec![zero; n_pad];
        let mut kyy = vec![zero; n_pad];
        let mut kzz = vec![zero; n_pad];

        let cache_path = demag_cache_path(grid, pbc_x, pbc_y, px, py, DEMAG_ACCURACY);
        let t_load = Instant::now();
        let loaded = try_load_kernel_kspace(
            &cache_path,
            grid,
            pbc_x,
            pbc_y,
            px,
            py,
            DEMAG_ACCURACY,
            &mut kxx,
            &mut kxy,
            &mut kyy,
            &mut kzz,
        )
        .unwrap_or(false);

        if do_timing {
            println!(
                "[demag timing] cache load: loaded={} in {:.3}s (nx={}, ny={}, px={}, py={}, pbc_x={}, pbc_y={})",
                loaded,
                t_load.elapsed().as_secs_f64(),
                grid.nx,
                grid.ny,
                px,
                py,
                pbc_x,
                pbc_y
            );
        }

        // Scratch used by fft2_* to parallelise columns via transpose.
        let mut fft_tmp = vec![zero; n_pad];

        if !loaded {
            println!(
                "[demag] cache miss -> building kernel (pbc_x={}, pbc_y={}) ...",
                pbc_x, pbc_y
            );
            let t_build = Instant::now();

            if pbc_x == 0 && pbc_y == 0 {
                // Open boundaries: keep your parity-enforced builder.
                build_kernel_realspace_open(
                    &grid,
                    px,
                    py,
                    &mut kxx,
                    &mut kxy,
                    &mut kyy,
                    &mut kzz,
                    DEMAG_ACCURACY,
                );
            } else {
                // PBC: sum over image ranges and accumulate into wrapped bins (MuMax-style).
                build_kernel_realspace_pbc(
                    &grid,
                    px,
                    py,
                    pbc_x,
                    pbc_y,
                    &mut kxx,
                    &mut kxy,
                    &mut kyy,
                    &mut kzz,
                    DEMAG_ACCURACY,
                );
            }
            if do_timing {
                println!(
                    "[demag timing] build real-space kernel took {:.3}s",
                    t_build.elapsed().as_secs_f64()
                );
            }
            let t_fft = Instant::now();
            // FFT to k-space
            fft2_forward_in_place(&mut kxx, px, py, &fft_x_fwd, &fft_y_fwd, &mut fft_tmp);
            fft2_forward_in_place(&mut kxy, px, py, &fft_x_fwd, &fft_y_fwd, &mut fft_tmp);
            fft2_forward_in_place(&mut kyy, px, py, &fft_x_fwd, &fft_y_fwd, &mut fft_tmp);
            fft2_forward_in_place(&mut kzz, px, py, &fft_x_fwd, &fft_y_fwd, &mut fft_tmp);

            if do_timing {
                println!(
                    "[demag timing] FFT kernel -> k-space took {:.3}s",
                    t_fft.elapsed().as_secs_f64()
                );
            }

            let header = KernelCacheHeader::new(grid, pbc_x, pbc_y, px, py, DEMAG_ACCURACY);
            if let Err(e) = write_kernel_kspace(&cache_path, header, &kxx, &kxy, &kyy, &kzz) {
                eprintln!(
                    "[demag] warning: failed to write cache {:?}: {}",
                    cache_path, e
                );
            } else {
                println!("[demag] cached kernel to {:?}", cache_path);
            }
        } else {
            println!("[demag] cache hit -> loaded kernel from {:?}", cache_path);
        }
        if do_timing {
            println!(
                "[demag timing] Demag2D::new total {:.3}s (loaded={})",
                t_total.elapsed().as_secs_f64(),
                loaded
            );
        }
        Self {
            grid,
            pbc_x,
            pbc_y,
            px,
            py,
            n_pad,

            kxx,
            kxy,
            kyy,
            kzz,

            mx: vec![zero; n_pad],
            my: vec![zero; n_pad],
            mz: vec![zero; n_pad],
            bx: vec![zero; n_pad],
            by: vec![zero; n_pad],
            bz: vec![zero; n_pad],

            fft_x_fwd,
            fft_x_inv,
            fft_y_fwd,
            fft_y_inv,

            fft_tmp,
        }
    }

    fn add_field(&mut self, m: &VectorField2D, b_eff: &mut VectorField2D, ms: f64) {
        debug_assert!(same_grid(&m.grid, &self.grid));

        let zero = Complex::new(0.0, 0.0);
        self.mx.fill(zero);
        self.my.fill(zero);
        self.mz.fill(zero);

        // Pack M = Ms * m into top-left (physical region), zeros elsewhere
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let px = self.px;

        // Only touch the first `ny` rows of the padded buffers.
        // This avoids any take()/zip length mismatch issues.
        let n_rows = ny * px;

        {
            let mx_buf = &mut self.mx[..n_rows];
            let my_buf = &mut self.my[..n_rows];
            let mz_buf = &mut self.mz[..n_rows];

            mx_buf
                .par_chunks_mut(px)
                .zip_eq(my_buf.par_chunks_mut(px))
                .zip_eq(mz_buf.par_chunks_mut(px))
                .enumerate()
                .for_each(|(j, ((mx_row, my_row), mz_row))| {
                    let src_row = &m.data[j * nx..(j + 1) * nx];
                    for i in 0..nx {
                        let v = src_row[i];
                        mx_row[i].re = ms * v[0];
                        my_row[i].re = ms * v[1];
                        mz_row[i].re = ms * v[2];
                    }
                });
        }

        // FFT M
        fft2_forward_in_place(
            &mut self.mx,
            self.px,
            self.py,
            &self.fft_x_fwd,
            &self.fft_y_fwd,
            &mut self.fft_tmp,
        );
        fft2_forward_in_place(
            &mut self.my,
            self.px,
            self.py,
            &self.fft_x_fwd,
            &self.fft_y_fwd,
            &mut self.fft_tmp,
        );
        fft2_forward_in_place(
            &mut self.mz,
            self.px,
            self.py,
            &self.fft_x_fwd,
            &self.fft_y_fwd,
            &mut self.fft_tmp,
        );

        // k-space multiply (parallel per idx; deterministic)
        let kxx = &self.kxx;
        let kxy = &self.kxy;
        let kyy = &self.kyy;
        let kzz = &self.kzz;

        let mxk = &self.mx;
        let myk = &self.my;
        let mzk = &self.mz;

        self.bx
            .par_iter_mut()
            .zip_eq(self.by.par_iter_mut())
            .zip_eq(self.bz.par_iter_mut())
            .enumerate()
            .for_each(|(idx, ((bx_i, by_i), bz_i))| {
                let mx = mxk[idx];
                let my = myk[idx];
                let mz = mzk[idx];

                *bx_i = kxx[idx] * mx + kxy[idx] * my;
                *by_i = kxy[idx] * mx + kyy[idx] * my;
                *bz_i = kzz[idx] * mz;
            });

        // iFFT back
        fft2_inverse_in_place(
            &mut self.bx,
            self.px,
            self.py,
            &self.fft_x_inv,
            &self.fft_y_inv,
            &mut self.fft_tmp,
        );
        fft2_inverse_in_place(
            &mut self.by,
            self.px,
            self.py,
            &self.fft_x_inv,
            &self.fft_y_inv,
            &mut self.fft_tmp,
        );
        fft2_inverse_in_place(
            &mut self.bz,
            self.px,
            self.py,
            &self.fft_x_inv,
            &self.fft_y_inv,
            &mut self.fft_tmp,
        );

        // Add physical region back into b_eff (parallel over rows)
        let bx = &self.bx;
        let by = &self.by;
        let bz = &self.bz;

        b_eff
            .data
            .par_chunks_mut(nx)
            .enumerate()
            .for_each(|(j, row)| {
                let src_base = j * px;
                for i in 0..nx {
                    let src = src_base + i;
                    row[i][0] += bx[src].re;
                    row[i][1] += by[src].re;
                    row[i][2] += bz[src].re;
                }
            });
    }
}

/// 2D forward FFT (in-place), applying 1D FFTs over rows then columns.
///
/// Hybrid strategy:
/// - For small padded grids: simple gather-column FFT (often faster).
/// - For larger padded grids: transpose + parallel column FFT.
fn fft2_forward_in_place(
    data: &mut [Complex<f64>],
    nx: usize,
    ny: usize,
    fft_x: &Arc<dyn Fft<f64>>,
    fft_y: &Arc<dyn Fft<f64>>,
    tmp: &mut [Complex<f64>],
) {
    let n = nx * ny;
    assert!(
        tmp.len() >= n,
        "fft2_forward_in_place: tmp too small (have {}, need {})",
        tmp.len(),
        n
    );

    // 1) Rows (parallel)
    data.par_chunks_mut(nx).for_each(|row| {
        fft_x.process(row);
    });

    // 2) Columns
    if !use_parallel_column_fft(nx, ny) {
        // Simple gather-column path.
        let col_buf = &mut tmp[..ny];
        for x in 0..nx {
            for y in 0..ny {
                col_buf[y] = data[y * nx + x];
            }
            fft_y.process(col_buf);
            for y in 0..ny {
                data[y * nx + x] = col_buf[y];
            }
        }
        return;
    }

    // Transpose path: tmp[x*ny + y] = data[y*nx + x]
    {
        let data_ro: &[Complex<f64>] = &*data;
        tmp[..n]
            .par_chunks_mut(ny)
            .enumerate()
            .for_each(|(x, col)| {
                for y in 0..ny {
                    col[y] = data_ro[y * nx + x];
                }
            });
    }

    // Columns become contiguous rows in tmp -> parallel FFT over length ny
    tmp[..n].par_chunks_mut(ny).for_each(|col| {
        fft_y.process(col);
    });

    // Transpose back into data (parallel over rows)
    let tmp_ro: &[Complex<f64>] = &tmp[..n];
    data.par_chunks_mut(nx).enumerate().for_each(|(y, row)| {
        for x in 0..nx {
            row[x] = tmp_ro[x * ny + y];
        }
    });
}

/// 2D inverse FFT (in-place), with standard 1/(nx*ny) scaling applied at end.
///
/// Hybrid strategy:
/// - For small padded grids: simple gather-column FFT (often faster).
/// - For larger padded grids: transpose + parallel column FFT.
fn fft2_inverse_in_place(
    data: &mut [Complex<f64>],
    nx: usize,
    ny: usize,
    fft_x_inv: &Arc<dyn Fft<f64>>,
    fft_y_inv: &Arc<dyn Fft<f64>>,
    tmp: &mut [Complex<f64>],
) {
    let n = nx * ny;
    assert!(
        tmp.len() >= n,
        "fft2_inverse_in_place: tmp too small (have {}, need {})",
        tmp.len(),
        n
    );

    // 1) Rows (parallel)
    data.par_chunks_mut(nx).for_each(|row| {
        fft_x_inv.process(row);
    });

    // 2) Columns
    if !use_parallel_column_fft(nx, ny) {
        // Simple gather-column path.
        let col_buf = &mut tmp[..ny];
        for x in 0..nx {
            for y in 0..ny {
                col_buf[y] = data[y * nx + x];
            }
            fft_y_inv.process(col_buf);
            for y in 0..ny {
                data[y * nx + x] = col_buf[y];
            }
        }

        // rustfft is unnormalised -> scale (parallel)
        let scale = 1.0 / (nx * ny) as f64;
        data.par_iter_mut().for_each(|v| {
            v.re *= scale;
            v.im *= scale;
        });
        return;
    }

    // Transpose path: tmp[x*ny + y] = data[y*nx + x]
    {
        let data_ro: &[Complex<f64>] = &*data;
        tmp[..n]
            .par_chunks_mut(ny)
            .enumerate()
            .for_each(|(x, col)| {
                for y in 0..ny {
                    col[y] = data_ro[y * nx + x];
                }
            });
    }

    // Column inverse FFTs (parallel)
    tmp[..n].par_chunks_mut(ny).for_each(|col| {
        fft_y_inv.process(col);
    });

    // Transpose back
    let tmp_ro: &[Complex<f64>] = &tmp[..n];
    data.par_chunks_mut(nx).enumerate().for_each(|(y, row)| {
        for x in 0..nx {
            row[x] = tmp_ro[x * ny + y];
        }
    });

    // rustfft is unnormalised -> scale (parallel)
    let scale = 1.0 / (nx * ny) as f64;
    data.par_iter_mut().for_each(|v| {
        v.re *= scale;
        v.im *= scale;
    });
}

/// Open-boundary kernel builder (keeps your existing parity-enforced behavior).
fn build_kernel_realspace_open(
    grid: &Grid2D,
    px: usize,
    py: usize,
    kxx: &mut [Complex<f64>],
    kxy: &mut [Complex<f64>],
    kyy: &mut [Complex<f64>],
    kzz: &mut [Complex<f64>],
    accuracy: f64,
) {
    // This is your existing open-boundary builder.
    // (Renamed to make intent clear.)
    build_kernel_realspace_mumax_2d(grid, px, py, kxx, kxy, kyy, kzz, accuracy);
}

/// PBC kernel builder: accumulate contributions from periodic images over MuMax-style ranges.
fn build_kernel_realspace_pbc(
    grid: &Grid2D,
    px: usize,
    py: usize,
    pbc_x: usize,
    pbc_y: usize,
    kxx: &mut [Complex<f64>],
    kxy: &mut [Complex<f64>],
    kyy: &mut [Complex<f64>],
    kzz: &mut [Complex<f64>],
    accuracy: f64,
) {
    let nx = grid.nx as isize;
    let ny = grid.ny as isize;

    // Range logic matches MuMax kernelRanges:
    // if pbc==0:  sx in [-(N-1), +(N-1)]
    // if pbc>0:   sx in [-(N*pbc - 1), +(N*pbc - 1)]
    let sx_max = if pbc_x == 0 {
        nx - 1
    } else {
        (nx * (pbc_x as isize)) - 1
    };
    let sy_max = if pbc_y == 0 {
        ny - 1
    } else {
        (ny * (pbc_y as isize)) - 1
    };

    // Arrays are already zero-initialised by caller.
    // Accumulate contributions into wrapped bins (+=), because many images fold to same index.
    for sy in -sy_max..=sy_max {
        let iy = wrap_index(sy, py);
        for sx in -sx_max..=sx_max {
            let ix = wrap_index(sx, px);
            let idx = iy * px + ix;

            let (k_xx, k_xy, k_yy, k_zz) =
                kernel_2d_components_mumax_like(grid.dx, grid.dy, grid.dz, sx, sy, accuracy);

            kxx[idx].re += k_xx;
            kxy[idx].re += k_xy;
            kyy[idx].re += k_yy;
            kzz[idx].re += k_zz;
        }
    }

    // Optional: enforce kxy=0 on axes (helps reduce tiny numerical drift).
    // This is safe with wrap indexing for both periodic and open directions.
    // If you prefer to be ultra-pure MuMax, you can omit this; it mainly cleans noise.
    if px > 0 {
        for sy in -sy_max..=sy_max {
            let ix0 = wrap_index(0, px);
            let iy = wrap_index(sy, py);
            kxy[iy * px + ix0].re = 0.0;
        }
    }
    if py > 0 {
        for sx in -sx_max..=sx_max {
            let iy0 = wrap_index(0, py);
            let ix = wrap_index(sx, px);
            kxy[iy0 * px + ix].re = 0.0;
        }
    }
}

/// Build MuMax-like 2D real-space kernel and enforce parity exactly (open boundaries).
///
/// We fill only displacements |sx|<=Nx-1, |sy|<=Ny-1 and leave the Nyquist planes zero.
/// Then enforce:
/// - kxx, kyy, kzz even in x and y
/// - kxy odd in x and odd in y
fn build_kernel_realspace_mumax_2d(
    grid: &Grid2D,
    px: usize,
    py: usize,
    kxx: &mut [Complex<f64>],
    kxy: &mut [Complex<f64>],
    kyy: &mut [Complex<f64>],
    kzz: &mut [Complex<f64>],
    accuracy: f64,
) {
    let nx = grid.nx as isize;
    let ny = grid.ny as isize;
    let rx_max = nx - 1;
    let ry_max = ny - 1;

    // Fill base quadrant sx>=0, sy>=0 (including axes), then reflect with exact parity.
    for sy in 0..=ry_max {
        for sx in 0..=rx_max {
            let (k_xx, k_xy, k_yy, k_zz) =
                kernel_2d_components_mumax_like(grid.dx, grid.dy, grid.dz, sx, sy, accuracy);

            let mut set_at = |sx_s: isize, sy_s: isize, xx: f64, xy: f64, yy: f64, zz: f64| {
                let ix = wrap_index(sx_s, px);
                let iy = wrap_index(sy_s, py);
                let idx = iy * px + ix;
                kxx[idx].re = xx;
                kxy[idx].re = xy;
                kyy[idx].re = yy;
                kzz[idx].re = zz;
            };

            for &sx_sign in &[1isize, -1isize] {
                for &sy_sign in &[1isize, -1isize] {
                    let sx_s = sx_sign * sx;
                    let sy_s = sy_sign * sy;

                    let xx = k_xx;
                    let yy = k_yy;
                    let zz = k_zz;

                    let xy = (sx_sign as f64) * (sy_sign as f64) * k_xy;

                    set_at(sx_s, sy_s, xx, xy, yy, zz);
                }
            }
        }
    }

    // Enforce kxy=0 on axes exactly (MuMax parity implies this).
    for sy in -ry_max..=ry_max {
        let ix0 = wrap_index(0, px);
        let iy = wrap_index(sy, py);
        kxy[iy * px + ix0].re = 0.0;
    }
    for sx in -rx_max..=rx_max {
        let iy0 = wrap_index(0, py);
        let ix = wrap_index(sx, px);
        kxy[iy0 * px + ix].re = 0.0;
    }
}

/// Compute (Kxx, Kxy, Kyy, Kzz) for 2D Nz=1.
/// Returns K = μ0 * N in Tesla/(A/m).
fn kernel_2d_components_mumax_like(
    dx: f64,
    dy: f64,
    dz: f64,
    sx: isize,
    sy: isize,
    accuracy: f64,
) -> (f64, f64, f64, f64) {
    let r_center = [sx as f64 * dx, sy as f64 * dy, 0.0_f64];
    let cell = [dx, dy, dz];

    let hx_from_x = mumax_like_h_from_unit_m(0, r_center, cell, [sx, sy, 0], accuracy);
    let hx_from_y = mumax_like_h_from_unit_m(1, r_center, cell, [sx, sy, 0], accuracy);
    let hx_from_z = mumax_like_h_from_unit_m(2, r_center, cell, [sx, sy, 0], accuracy);

    let nxx = hx_from_x[0];
    let nxy = hx_from_y[0]; // Hx from My
    let nyy = hx_from_y[1];
    let nzz = hx_from_z[2];

    (MU0 * nxx, MU0 * nxy, MU0 * nyy, MU0 * nzz)
}

/// MuMax-like face-charge integration returning H per unit M (dimensionless N tensor row).
fn mumax_like_h_from_unit_m(
    source_axis: usize,
    r_center: [f64; 3],
    cell: [f64; 3],
    disp_ijk: [isize; 3],
    accuracy: f64,
) -> [f64; 3] {
    let u = source_axis;
    let v = (u + 1) % 3;
    let w = (u + 2) % 3;

    let lmin = cell[0].min(cell[1]).min(cell[2]);

    let dx_min = delta_cell(disp_ijk[0]) * cell[0];
    let dy_min = delta_cell(disp_ijk[1]) * cell[1];
    let dz_min = delta_cell(disp_ijk[2]) * cell[2];

    let mut d = (dx_min * dx_min + dy_min * dy_min + dz_min * dz_min).sqrt();
    if d == 0.0 {
        d = lmin;
    }

    let max_size = d / accuracy;

    #[inline]
    fn mumax_n(x: f64) -> usize {
        ((x.max(1.0) + 0.5).floor()) as usize
    }

    let nx = mumax_n(cell[0] / max_size);
    let ny = mumax_n(cell[1] / max_size);
    let nz = mumax_n(cell[2] / max_size);

    let mut nv = mumax_n(cell[v] / max_size);
    let mut nw = mumax_n(cell[w] / max_size);

    nv *= 2;
    nw *= 2;

    let scale = 1.0 / ((nv * nw * nx * ny * nz) as f64);
    let face_area = cell[v] * cell[w];
    let charge = face_area * scale;

    let pu1 = 0.5 * cell[u];
    let pu2 = -pu1;

    let mut pole = [0.0_f64; 3];
    let mut h = [0.0_f64; 3];

    for i in 0..nv {
        let pv = -0.5 * cell[v] + cell[v] / (2.0 * nv as f64) + (i as f64) * (cell[v] / nv as f64);
        pole[v] = pv;

        for j in 0..nw {
            let pw =
                -0.5 * cell[w] + cell[w] / (2.0 * nw as f64) + (j as f64) * (cell[w] / nw as f64);
            pole[w] = pw;

            for ax in 0..nx {
                let rx = r_center[0] - 0.5 * cell[0]
                    + cell[0] / (2.0 * nx as f64)
                    + (ax as f64) * (cell[0] / nx as f64);

                for ay in 0..ny {
                    let ry = r_center[1] - 0.5 * cell[1]
                        + cell[1] / (2.0 * ny as f64)
                        + (ay as f64) * (cell[1] / ny as f64);

                    for az in 0..nz {
                        let rz = r_center[2] - 0.5 * cell[2]
                            + cell[2] / (2.0 * nz as f64)
                            + (az as f64) * (cell[2] / nz as f64);

                        // + pole
                        pole[u] = pu1;
                        let r1x = rx - pole[0];
                        let r1y = ry - pole[1];
                        let r1z = rz - pole[2];
                        let r1 = (r1x * r1x + r1y * r1y + r1z * r1z).sqrt();
                        let q1 = charge / (4.0 * PI * r1 * r1 * r1);

                        let hx1 = r1x * q1;
                        let hy1 = r1y * q1;
                        let hz1 = r1z * q1;

                        // - pole
                        pole[u] = pu2;
                        let r2x = rx - pole[0];
                        let r2y = ry - pole[1];
                        let r2z = rz - pole[2];
                        let r2 = (r2x * r2x + r2y * r2y + r2z * r2z).sqrt();
                        let q2 = -charge / (4.0 * PI * r2 * r2 * r2);

                        h[0] += hx1 + r2x * q2;
                        h[1] += hy1 + r2y * q2;
                        h[2] += hz1 + r2z * q2;
                    }
                }
            }
        }
    }

    h
}

/// Closest distance between cells given integer centre distance.
/// If cells touch, delta is 0.
#[inline]
fn delta_cell(d: isize) -> f64 {
    let mut a = d.abs() as f64;
    if a > 0.0 {
        a -= 1.0;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid2D;
    use crate::params::DemagMethod;

    #[test]
    fn demag_single_cell_cube_is_close_to_minus_mu0_over_3() {
        let grid = Grid2D::new(1, 1, 1.0, 1.0, 1.0);

        let mut m = VectorField2D::new(grid);
        m.set_uniform(0.0, 0.0, 1.0);

        let mat = Material {
            ms: 1.0,
            a_ex: 0.0,
            k_u: 0.0,
            easy_axis: [0.0, 0.0, 1.0],
            dmi: None,
            demag: true,
            demag_method: DemagMethod::FftUniform,
        };

        let mut b = VectorField2D::new(grid);
        b.set_uniform(0.0, 0.0, 0.0);

        add_demag_field(&grid, &m, &mut b, &mat);

        let bz = b.data[0][2];
        let expected = -MU0 / 3.0;

        // Loose tolerance to avoid brittle tests across accuracy implementations.
        assert!(
            (bz - expected).abs() < 1e-3,
            "bz={}, expected={}",
            bz,
            expected
        );
    }
}
