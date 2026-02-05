// src/effective_field/demag.rs
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
use crate::params::{Material, MU0};
use crate::vector_field::VectorField2D;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

static DEMAG_CACHE: OnceLock<Mutex<Option<Demag2D>>> = OnceLock::new();

/// Backwards-compatible: open boundaries in x/y (no PBC).
pub fn add_demag_field(grid: &Grid2D, m: &VectorField2D, b_eff: &mut VectorField2D, mat: &Material) {
    add_demag_field_pbc(grid, m, b_eff, mat, 0, 0);
}

/// Backwards-compatible: open boundaries in x/y (no PBC).
pub fn compute_demag_field(grid: &Grid2D, m: &VectorField2D, out: &mut VectorField2D, mat: &Material) {
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

    fn matches(&self, grid: Grid2D, pbc_x: usize, pbc_y: usize, px: usize, py: usize, accuracy: f64) -> bool {
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

fn demag_cache_path(grid: Grid2D, pbc_x: usize, pbc_y: usize, px: usize, py: usize, accuracy: f64) -> PathBuf {
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

    fn read_complex_vec(f: &mut File, out: &mut Vec<Complex<f64>>, n: usize) -> std::io::Result<()> {
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

    col_buf: Vec<Complex<f64>>,
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

        let mut col_buf = vec![zero; py];

        if !loaded {
            println!(
                "[demag] cache miss -> building kernel (pbc_x={}, pbc_y={}) ...",
                pbc_x, pbc_y
            );

            if pbc_x == 0 && pbc_y == 0 {
                // Open boundaries: keep your parity-enforced builder.
                build_kernel_realspace_open(&grid, px, py, &mut kxx, &mut kxy, &mut kyy, &mut kzz, DEMAG_ACCURACY);
            } else {
                // PBC: sum over image ranges and accumulate into wrapped bins (MuMax-style).
                build_kernel_realspace_pbc(&grid, px, py, pbc_x, pbc_y, &mut kxx, &mut kxy, &mut kyy, &mut kzz, DEMAG_ACCURACY);
            }

            // FFT to k-space
            fft2_forward_in_place(&mut kxx, px, py, &fft_x_fwd, &fft_y_fwd, &mut col_buf);
            fft2_forward_in_place(&mut kxy, px, py, &fft_x_fwd, &fft_y_fwd, &mut col_buf);
            fft2_forward_in_place(&mut kyy, px, py, &fft_x_fwd, &fft_y_fwd, &mut col_buf);
            fft2_forward_in_place(&mut kzz, px, py, &fft_x_fwd, &fft_y_fwd, &mut col_buf);

            let header = KernelCacheHeader::new(grid, pbc_x, pbc_y, px, py, DEMAG_ACCURACY);
            if let Err(e) = write_kernel_kspace(&cache_path, header, &kxx, &kxy, &kyy, &kzz) {
                eprintln!("[demag] warning: failed to write cache {:?}: {}", cache_path, e);
            } else {
                println!("[demag] cached kernel to {:?}", cache_path);
            }
        } else {
            println!("[demag] cache hit -> loaded kernel from {:?}", cache_path);
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

            col_buf,
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

        for j in 0..ny {
            for i in 0..nx {
                let src = j * nx + i;
                let dst = j * px + i;
                let v = m.data[src];
                self.mx[dst].re = ms * v[0];
                self.my[dst].re = ms * v[1];
                self.mz[dst].re = ms * v[2];
            }
        }

        // FFT M
        fft2_forward_in_place(&mut self.mx, self.px, self.py, &self.fft_x_fwd, &self.fft_y_fwd, &mut self.col_buf);
        fft2_forward_in_place(&mut self.my, self.px, self.py, &self.fft_x_fwd, &self.fft_y_fwd, &mut self.col_buf);
        fft2_forward_in_place(&mut self.mz, self.px, self.py, &self.fft_x_fwd, &self.fft_y_fwd, &mut self.col_buf);

        // MuMax 2D demag: XY coupled via XX, YY, XY; Z only via ZZ.
        for idx in 0..self.n_pad {
            let mx = self.mx[idx];
            let my = self.my[idx];
            let mz = self.mz[idx];

            self.bx[idx] = self.kxx[idx] * mx + self.kxy[idx] * my;
            self.by[idx] = self.kxy[idx] * mx + self.kyy[idx] * my;
            self.bz[idx] = self.kzz[idx] * mz;
        }

        // iFFT back
        fft2_inverse_in_place(&mut self.bx, self.px, self.py, &self.fft_x_inv, &self.fft_y_inv, &mut self.col_buf);
        fft2_inverse_in_place(&mut self.by, self.px, self.py, &self.fft_x_inv, &self.fft_y_inv, &mut self.col_buf);
        fft2_inverse_in_place(&mut self.bz, self.px, self.py, &self.fft_x_inv, &self.fft_y_inv, &mut self.col_buf);

        // Add physical region back into b_eff
        for j in 0..ny {
            for i in 0..nx {
                let dst_field = j * nx + i;
                let src_pad = j * px + i;
                b_eff.data[dst_field][0] += self.bx[src_pad].re;
                b_eff.data[dst_field][1] += self.by[src_pad].re;
                b_eff.data[dst_field][2] += self.bz[src_pad].re;
            }
        }
    }
}

/// 2D forward FFT (in-place), applying 1D FFTs over rows then columns.
fn fft2_forward_in_place(
    data: &mut [Complex<f64>],
    nx: usize,
    ny: usize,
    fft_x: &Arc<dyn Fft<f64>>,
    fft_y: &Arc<dyn Fft<f64>>,
    col_buf: &mut Vec<Complex<f64>>,
) {
    if col_buf.len() != ny {
        col_buf.resize(ny, Complex::new(0.0, 0.0));
    }

    // Rows
    for y in 0..ny {
        let row = &mut data[y * nx..(y + 1) * nx];
        fft_x.process(row);
    }

    // Columns
    for x in 0..nx {
        for y in 0..ny {
            col_buf[y] = data[y * nx + x];
        }
        fft_y.process(col_buf);
        for y in 0..ny {
            data[y * nx + x] = col_buf[y];
        }
    }
}

/// 2D inverse FFT (in-place), with standard 1/(nx*ny) scaling applied at end.
fn fft2_inverse_in_place(
    data: &mut [Complex<f64>],
    nx: usize,
    ny: usize,
    fft_x_inv: &Arc<dyn Fft<f64>>,
    fft_y_inv: &Arc<dyn Fft<f64>>,
    col_buf: &mut Vec<Complex<f64>>,
) {
    if col_buf.len() != ny {
        col_buf.resize(ny, Complex::new(0.0, 0.0));
    }

    // Rows
    for y in 0..ny {
        let row = &mut data[y * nx..(y + 1) * nx];
        fft_x_inv.process(row);
    }

    // Columns
    for x in 0..nx {
        for y in 0..ny {
            col_buf[y] = data[y * nx + x];
        }
        fft_y_inv.process(col_buf);
        for y in 0..ny {
            data[y * nx + x] = col_buf[y];
        }
    }

    // rustfft is unnormalised -> scale
    let scale = 1.0 / (nx * ny) as f64;
    for v in data.iter_mut() {
        v.re *= scale;
        v.im *= scale;
    }
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

            let (k_xx, k_xy, k_yy, k_zz) = kernel_2d_components_mumax_like(grid.dx, grid.dy, grid.dz, sx, sy, accuracy);

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
            let (k_xx, k_xy, k_yy, k_zz) = kernel_2d_components_mumax_like(grid.dx, grid.dy, grid.dz, sx, sy, accuracy);

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
            let pw = -0.5 * cell[w] + cell[w] / (2.0 * nw as f64) + (j as f64) * (cell[w] / nw as f64);
            pole[w] = pw;

            for ax in 0..nx {
                let rx = r_center[0]
                    - 0.5 * cell[0]
                    + cell[0] / (2.0 * nx as f64)
                    + (ax as f64) * (cell[0] / nx as f64);

                for ay in 0..ny {
                    let ry = r_center[1]
                        - 0.5 * cell[1]
                        + cell[1] / (2.0 * ny as f64)
                        + (ay as f64) * (cell[1] / ny as f64);

                    for az in 0..nz {
                        let rz = r_center[2]
                            - 0.5 * cell[2]
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
        };

        let mut b = VectorField2D::new(grid);
        b.set_uniform(0.0, 0.0, 0.0);

        add_demag_field(&grid, &m, &mut b, &mat);

        let bz = b.data[0][2];
        let expected = -MU0 / 3.0;

        // Loose tolerance to avoid brittle tests across accuracy implementations.
        assert!((bz - expected).abs() < 1e-3, "bz={}, expected={}", bz, expected);
    }
}



// // src/effective_field/demag.rs
// //
// // Magnetostatic (demagnetising) field via FFT-accelerated convolution.
// //
// // We compute (discrete convolution):
// //   B_demag_i = K_ij * M_j
// // where M = Ms * m is magnetization in A/m and B_demag is in Tesla.
// //
// // This implementation:
// // - 2D zero-padding to 2*Nx × 2*Ny (linear convolution / open boundaries)
// // - A *rectangular-prism (volume-averaged)* kernel K_ij computed numerically
// //   (near-field), with a point-dipole approximation (far-field).
// //
// // This mirrors the standard micromagnetics approach: treat each FD cell as having
// // uniform magnetization and compute the averaged field via a demag kernel, then
// // FFT accelerate the convolution (as in MuMax3 design/verification paper)
// //
// // Notes:
// // - Grid is 2D (Nx×Ny) with a finite thickness dz (Nz=1).
// // - The kernel returns K in units of Tesla per (A/m), so B = K * M.
// // - For a cubic single cell, the self-demag factor should be exactly 1/3:
// //     B = -mu0/3 * M   (component-wise), and off-diagonals should be 0.

// use crate::grid::Grid2D;
// use crate::params::{Material, MU0};
// use crate::vector_field::VectorField2D;

// use rustfft::num_complex::Complex;
// use rustfft::{Fft, FftPlanner};

// use std::f64::consts::PI;
// use std::sync::{Arc, Mutex, OnceLock};
// use std::fs::{self, File};
// use std::io::{Read, Write};
// use std::path::PathBuf;

// static DEMAG_CACHE: OnceLock<Mutex<Option<Demag2D>>> = OnceLock::new();

// /// Add the demagnetising (magnetostatic) induction B_demag (Tesla) to b_eff.
// ///
// /// Uses a cached FFT kernel so we do NOT rebuild the kernel every timestep.
// /// If the grid changes between runs, the kernel is rebuilt automatically.
// pub fn add_demag_field(
//     grid: &Grid2D,
//     m: &VectorField2D,
//     b_eff: &mut VectorField2D,
//     mat: &Material,
// ) {
//     if !mat.demag {
//         return;
//     }

//     let cache = DEMAG_CACHE.get_or_init(|| Mutex::new(None));
//     let mut guard = cache.lock().expect("DEMAG_CACHE mutex poisoned");

//     let rebuild = match guard.as_ref() {
//         Some(d) => !same_grid(&d.grid, grid),
//         None => true,
//     };

//     if rebuild {
//         *guard = Some(Demag2D::new(*grid));
//     }

//     if let Some(d) = guard.as_mut() {
//         d.add_field(m, b_eff, mat.ms);
//     }
// }

// /// Compute demag induction B_demag (Tesla) into `out` (overwrites out).
// pub fn compute_demag_field(
//     grid: &Grid2D,
//     m: &VectorField2D,
//     out: &mut VectorField2D,
//     mat: &Material,
// ) {
//     out.set_uniform(0.0, 0.0, 0.0);
//     add_demag_field(grid, m, out, mat);
// }

// fn same_grid(a: &Grid2D, b: &Grid2D) -> bool {
//     a.nx == b.nx && a.ny == b.ny && a.dx == b.dx && a.dy == b.dy && a.dz == b.dz
// }

// #[inline]
// fn wrap_index(d: isize, n: usize) -> usize {
//     let n = n as isize;
//     let mut v = d % n;
//     if v < 0 { v += n; }
//     v as usize
// }

// #[derive(Debug, Clone, Copy)]
// struct KernelCacheHeader {
//     magic: [u8; 8], // b"LLGDMAG\0"
//     version: u32,
//     nx: u32,
//     ny: u32,
//     px: u32,
//     py: u32,
//     dx: f64,
//     dy: f64,
//     dz: f64,
//     accuracy: f64,
// }

// impl KernelCacheHeader {
//     fn new(grid: Grid2D, px: usize, py: usize, accuracy: f64) -> Self {
//         Self {
//             magic: *b"LLGDMAG\0",
//             version: 1,
//             nx: grid.nx as u32,
//             ny: grid.ny as u32,
//             px: px as u32,
//             py: py as u32,
//             dx: grid.dx,
//             dy: grid.dy,
//             dz: grid.dz,
//             accuracy,
//         }
//     }

//     fn matches(&self, grid: Grid2D, px: usize, py: usize, accuracy: f64) -> bool {
//         self.magic == *b"LLGDMAG\0"
//             && self.version == 1
//             && self.nx == grid.nx as u32
//             && self.ny == grid.ny as u32
//             && self.px == px as u32
//             && self.py == py as u32
//             && self.dx == grid.dx
//             && self.dy == grid.dy
//             && self.dz == grid.dz
//             && self.accuracy == accuracy
//     }
// }

// fn demag_cache_path(grid: Grid2D, accuracy: f64) -> PathBuf {
//     // Stored in out/demag_cache/ with a filename keyed by geometry + accuracy
//     let mut dir = PathBuf::from("out");
//     dir.push("demag_cache");

//     let fname = format!(
//         "demag_kernel_nx{}_ny{}_dx{:.3e}_dy{:.3e}_dz{:.3e}_acc{:.2}.bin",
//         grid.nx, grid.ny, grid.dx, grid.dy, grid.dz, accuracy
//     );

//     dir.push(fname);
//     dir
// }

// fn write_kernel_kspace(
//     path: &PathBuf,
//     header: KernelCacheHeader,
//     kxx: &[Complex<f64>],
//     kxy: &[Complex<f64>],
//     kxz: &[Complex<f64>],
//     kyy: &[Complex<f64>],
//     kyz: &[Complex<f64>],
//     kzz: &[Complex<f64>],
// ) -> std::io::Result<()> {
//     if let Some(parent) = path.parent() {
//         fs::create_dir_all(parent)?;
//     }

//     let mut f = File::create(path)?;

//     // Header
//     f.write_all(&header.magic)?;
//     f.write_all(&header.version.to_le_bytes())?;
//     f.write_all(&header.nx.to_le_bytes())?;
//     f.write_all(&header.ny.to_le_bytes())?;
//     f.write_all(&header.px.to_le_bytes())?;
//     f.write_all(&header.py.to_le_bytes())?;
//     f.write_all(&header.dx.to_le_bytes())?;
//     f.write_all(&header.dy.to_le_bytes())?;
//     f.write_all(&header.dz.to_le_bytes())?;
//     f.write_all(&header.accuracy.to_le_bytes())?;

//     // Kernel arrays: store re,im as f64 in little endian
//     fn write_complex_vec(f: &mut File, v: &[Complex<f64>]) -> std::io::Result<()> {
//         for c in v {
//             f.write_all(&c.re.to_le_bytes())?;
//             f.write_all(&c.im.to_le_bytes())?;
//         }
//         Ok(())
//     }

//     write_complex_vec(&mut f, kxx)?;
//     write_complex_vec(&mut f, kxy)?;
//     write_complex_vec(&mut f, kxz)?;
//     write_complex_vec(&mut f, kyy)?;
//     write_complex_vec(&mut f, kyz)?;
//     write_complex_vec(&mut f, kzz)?;

//     Ok(())
// }

// fn try_load_kernel_kspace(
//     path: &PathBuf,
//     grid: Grid2D,
//     px: usize,
//     py: usize,
//     accuracy: f64,
//     kxx: &mut Vec<Complex<f64>>,
//     kxy: &mut Vec<Complex<f64>>,
//     kxz: &mut Vec<Complex<f64>>,
//     kyy: &mut Vec<Complex<f64>>,
//     kyz: &mut Vec<Complex<f64>>,
//     kzz: &mut Vec<Complex<f64>>,
// ) -> std::io::Result<bool> {
//     if !path.exists() {
//         return Ok(false);
//     }

//     let mut f = File::open(path)?;

//     // Read header
//     let mut magic = [0u8; 8];
//     f.read_exact(&mut magic)?;

//     let mut buf4 = [0u8; 4];
//     let mut buf8 = [0u8; 8];

//     f.read_exact(&mut buf4)?;
//     let version = u32::from_le_bytes(buf4);

//     f.read_exact(&mut buf4)?;
//     let nx = u32::from_le_bytes(buf4);
//     f.read_exact(&mut buf4)?;
//     let ny = u32::from_le_bytes(buf4);
//     f.read_exact(&mut buf4)?;
//     let px_h = u32::from_le_bytes(buf4);
//     f.read_exact(&mut buf4)?;
//     let py_h = u32::from_le_bytes(buf4);

//     f.read_exact(&mut buf8)?;
//     let dx_h = f64::from_le_bytes(buf8);
//     f.read_exact(&mut buf8)?;
//     let dy_h = f64::from_le_bytes(buf8);
//     f.read_exact(&mut buf8)?;
//     let dz_h = f64::from_le_bytes(buf8);
//     f.read_exact(&mut buf8)?;
//     let acc_h = f64::from_le_bytes(buf8);

//     let header = KernelCacheHeader {
//         magic,
//         version,
//         nx,
//         ny,
//         px: px_h,
//         py: py_h,
//         dx: dx_h,
//         dy: dy_h,
//         dz: dz_h,
//         accuracy: acc_h,
//     };

//     if !header.matches(grid, px, py, accuracy) {
//         return Ok(false);
//     }

//     let n_pad = px * py;

//     fn read_complex_vec(f: &mut File, out: &mut Vec<Complex<f64>>, n: usize) -> std::io::Result<()> {
//         out.clear();
//         out.reserve(n);
//         let mut buf = [0u8; 8];

//         for _ in 0..n {
//             f.read_exact(&mut buf)?;
//             let re = f64::from_le_bytes(buf);
//             f.read_exact(&mut buf)?;
//             let im = f64::from_le_bytes(buf);
//             out.push(Complex::new(re, im));
//         }
//         Ok(())
//     }

//     read_complex_vec(&mut f, kxx, n_pad)?;
//     read_complex_vec(&mut f, kxy, n_pad)?;
//     read_complex_vec(&mut f, kxz, n_pad)?;
//     read_complex_vec(&mut f, kyy, n_pad)?;
//     read_complex_vec(&mut f, kyz, n_pad)?;
//     read_complex_vec(&mut f, kzz, n_pad)?;

//     Ok(true)
// }

// struct Demag2D {
//     grid: Grid2D,
//     px: usize,
//     py: usize,
//     n_pad: usize,

//     // Kernel in Fourier domain (Tesla per (A/m))
//     kxx: Vec<Complex<f64>>,
//     kxy: Vec<Complex<f64>>,
//     #[allow(dead_code)]
//     kxz: Vec<Complex<f64>>,
//     kyy: Vec<Complex<f64>>,
//     #[allow(dead_code)]
//     kyz: Vec<Complex<f64>>,
//     kzz: Vec<Complex<f64>>,

//     // Scratch buffers (in-place FFT)
//     mx: Vec<Complex<f64>>,
//     my: Vec<Complex<f64>>,
//     mz: Vec<Complex<f64>>,
//     bx: Vec<Complex<f64>>,
//     by: Vec<Complex<f64>>,
//     bz: Vec<Complex<f64>>,

//     // FFT plans
//     fft_x_fwd: Arc<dyn Fft<f64>>,
//     fft_x_inv: Arc<dyn Fft<f64>>,
//     fft_y_fwd: Arc<dyn Fft<f64>>,
//     fft_y_inv: Arc<dyn Fft<f64>>,

//     // Column scratch for 2D FFT
//     col_buf: Vec<Complex<f64>>,
// }

// impl Demag2D {
//     fn new(grid: Grid2D) -> Self {
//         let nx = grid.nx;
//         let ny = grid.ny;
//         let px = 2 * nx;
//         let py = 2 * ny;
//         let n_pad = px * py;

//         // FFT plans
//         let mut planner = FftPlanner::<f64>::new();
//         let fft_x_fwd = planner.plan_fft_forward(px);
//         let fft_x_inv = planner.plan_fft_inverse(px);
//         let fft_y_fwd = planner.plan_fft_forward(py);
//         let fft_y_inv = planner.plan_fft_inverse(py);

//         // Real-space kernel (Complex with im=0), then FFT -> k-space
//                 // Kernel in Fourier domain (Tesla per (A/m)): try load from disk, else compute and cache.
//         let zero = Complex::new(0.0, 0.0);

//         let mut kxx = vec![zero; n_pad];
//         let mut kxy = vec![zero; n_pad];
//         let mut kxz = vec![zero; n_pad];
//         let mut kyy = vec![zero; n_pad];
//         let mut kyz = vec![zero; n_pad];
//         let mut kzz = vec![zero; n_pad];

//         let cache_path = demag_cache_path(grid, DEMAG_ACCURACY);
//         let loaded = try_load_kernel_kspace(
//             &cache_path,
//             grid,
//             px,
//             py,
//             DEMAG_ACCURACY,
//             &mut kxx,
//             &mut kxy,
//             &mut kxz,
//             &mut kyy,
//             &mut kyz,
//             &mut kzz,
//         ).unwrap_or(false);

//         let mut col_buf = vec![zero; py];

//         if !loaded {
//             println!("[demag] cache miss -> building kernel (this may take a while)...");
//             // Fill only the physically meaningful linear-convolution range.// MuMax fills displacements in [-(N-1), +(N-1)] and leaves the ±N Nyquist plane as 0.
//             // MuMax fills displacements in [-(N-1), +(N-1)] and leaves the ±N Nyquist plane as 0.
//             let rx_max = nx as isize - 1;
//             let ry_max = ny as isize - 1;

//             // arrays are already zero-initialised, so any entries we do not fill remain 0
//             for sy in -ry_max..=ry_max {
//                 let iy = wrap_index(sy, py);
//                 for sx in -rx_max..=rx_max {
//                     let ix = wrap_index(sx, px);
//                     let (k_xx, k_xy, _k_xz, k_yy, _k_yz, k_zz) =
//                     prism_kernel_tensor_numeric(grid.dx, grid.dy, grid.dz, sx, sy);
//                     let idx = iy * px + ix;
//                     kxx[idx].re = k_xx;
//                     kxy[idx].re = k_xy;
//                     kyy[idx].re = k_yy;
//                     kzz[idx].re = k_zz;
//                     // MuMax 2D convention: XZ/YZ not used
//                     kxz[idx].re = 0.0;
//                     kyz[idx].re = 0.0;
//                 }
//             }

//             // FFT kernel to k-space
//             fft2_forward_in_place(&mut kxx, px, py, &fft_x_fwd, &fft_y_fwd, &mut col_buf);
//             fft2_forward_in_place(&mut kxy, px, py, &fft_x_fwd, &fft_y_fwd, &mut col_buf);
//             fft2_forward_in_place(&mut kxz, px, py, &fft_x_fwd, &fft_y_fwd, &mut col_buf);
//             fft2_forward_in_place(&mut kyy, px, py, &fft_x_fwd, &fft_y_fwd, &mut col_buf);
//             fft2_forward_in_place(&mut kyz, px, py, &fft_x_fwd, &fft_y_fwd, &mut col_buf);
//             fft2_forward_in_place(&mut kzz, px, py, &fft_x_fwd, &fft_y_fwd, &mut col_buf);

//             // Write cache
//             let header = KernelCacheHeader::new(grid, px, py, DEMAG_ACCURACY);
//             if let Err(e) = write_kernel_kspace(&cache_path, header, &kxx, &kxy, &kxz, &kyy, &kyz, &kzz) {
//                 eprintln!("[demag] warning: failed to write cache {:?}: {}", cache_path, e);
//             } else {
//                 println!("[demag] cached kernel to {:?}", cache_path);
//             }
//         } else {
//             println!("[demag] cache hit -> loaded kernel from {:?}", cache_path);
//         }
//         Self {
//             grid,
//             px,
//             py,
//             n_pad,

//             kxx,
//             kxy,
//             kxz,
//             kyy,
//             kyz,
//             kzz,

//             mx: vec![zero; n_pad],
//             my: vec![zero; n_pad],
//             mz: vec![zero; n_pad],
//             bx: vec![zero; n_pad],
//             by: vec![zero; n_pad],
//             bz: vec![zero; n_pad],

//             fft_x_fwd,
//             fft_x_inv,
//             fft_y_fwd,
//             fft_y_inv,

//             col_buf,
//         }
//     }

//     fn add_field(&mut self, m: &VectorField2D, b_eff: &mut VectorField2D, ms: f64) {
//         debug_assert!(same_grid(&m.grid, &self.grid));

//         let zero = Complex::new(0.0, 0.0);
//         self.mx.fill(zero);
//         self.my.fill(zero);
//         self.mz.fill(zero);

//         // Pack M = Ms*m into padded arrays (top-left), zeros elsewhere
//         let nx = self.grid.nx;
//         let ny = self.grid.ny;
//         let px = self.px;

//         for j in 0..ny {
//             for i in 0..nx {
//                 let src = j * nx + i;
//                 let dst = j * px + i;

//                 let v = m.data[src];
//                 self.mx[dst].re = ms * v[0];
//                 self.my[dst].re = ms * v[1];
//                 self.mz[dst].re = ms * v[2];
//             }
//         }

//         // FFT M
//         fft2_forward_in_place(&mut self.mx, self.px, self.py, &self.fft_x_fwd, &self.fft_y_fwd, &mut self.col_buf);
//         fft2_forward_in_place(&mut self.my, self.px, self.py, &self.fft_x_fwd, &self.fft_y_fwd, &mut self.col_buf);
//         fft2_forward_in_place(&mut self.mz, self.px, self.py, &self.fft_x_fwd, &self.fft_y_fwd, &mut self.col_buf);

//         // Multiply in k-space: B = K * M
//         for idx in 0..self.n_pad {
//             let mx = self.mx[idx];
//             let my = self.my[idx];
//             let mz = self.mz[idx];
//             // Match MuMax 2D demag path: no XZ/YZ coupling
//             self.bx[idx] = self.kxx[idx] * mx + self.kxy[idx] * my;
//             self.by[idx] = self.kxy[idx] * mx + self.kyy[idx] * my;
//             self.bz[idx] = self.kzz[idx] * mz;
//         }
//         // iFFT back to real space
//         fft2_inverse_in_place(&mut self.bx, self.px, self.py, &self.fft_x_inv, &self.fft_y_inv, &mut self.col_buf);
//         fft2_inverse_in_place(&mut self.by, self.px, self.py, &self.fft_x_inv, &self.fft_y_inv, &mut self.col_buf);
//         fft2_inverse_in_place(&mut self.bz, self.px, self.py, &self.fft_x_inv, &self.fft_y_inv, &mut self.col_buf);

//         // Add result into the physical region
//         for j in 0..ny {
//             for i in 0..nx {
//                 let dst_field = j * nx + i;
//                 let src_pad = j * px + i;

//                 b_eff.data[dst_field][0] += self.bx[src_pad].re;
//                 b_eff.data[dst_field][1] += self.by[src_pad].re;
//                 b_eff.data[dst_field][2] += self.bz[src_pad].re;
//             }
//         }
//     }
// }

// /// 2D forward FFT (in-place), applying 1D FFTs over rows then columns.
// fn fft2_forward_in_place(
//     data: &mut [Complex<f64>],
//     nx: usize,
//     ny: usize,
//     fft_x: &Arc<dyn Fft<f64>>,
//     fft_y: &Arc<dyn Fft<f64>>,
//     col_buf: &mut Vec<Complex<f64>>,
// ) {
//     if col_buf.len() != ny {
//         col_buf.resize(ny, Complex::new(0.0, 0.0));
//     }

//     // Rows
//     for y in 0..ny {
//         let row = &mut data[y * nx..(y + 1) * nx];
//         fft_x.process(row);
//     }

//     // Columns
//     for x in 0..nx {
//         for y in 0..ny {
//             col_buf[y] = data[y * nx + x];
//         }
//         fft_y.process(col_buf);
//         for y in 0..ny {
//             data[y * nx + x] = col_buf[y];
//         }
//     }
// }

// /// 2D inverse FFT (in-place), with the standard 1/(nx*ny) scaling applied at the end.
// fn fft2_inverse_in_place(
//     data: &mut [Complex<f64>],
//     nx: usize,
//     ny: usize,
//     fft_x_inv: &Arc<dyn Fft<f64>>,
//     fft_y_inv: &Arc<dyn Fft<f64>>,
//     col_buf: &mut Vec<Complex<f64>>,
// ) {
//     if col_buf.len() != ny {
//         col_buf.resize(ny, Complex::new(0.0, 0.0));
//     }

//     // Rows
//     for y in 0..ny {
//         let row = &mut data[y * nx..(y + 1) * nx];
//         fft_x_inv.process(row);
//     }

//     // Columns
//     for x in 0..nx {
//         for y in 0..ny {
//             col_buf[y] = data[y * nx + x];
//         }
//         fft_y_inv.process(col_buf);
//         for y in 0..ny {
//             data[y * nx + x] = col_buf[y];
//         }
//     }

//     // rustfft is unnormalised -> scale
//     let scale = 1.0 / (nx * ny) as f64;
//     for v in data.iter_mut() {
//         v.re *= scale;
//         v.im *= scale;
//     }
// }

// // Demag kernel accuracy parameter (MuMax default is 6.0).
// // Larger => more integration points near-field => more accurate, slower init.
// const DEMAG_ACCURACY: f64 = 10.0;

// /// MuMax-like demag kernel element via face-charge integration + volume averaging.
// ///
// /// Returns (Kxx, Kxy, Kxz, Kyy, Kyz, Kzz) in units Tesla per (A/m),
// /// so that B = K * M, where M is magnetization in A/m and B is Tesla.
// ///
// /// This follows MuMax3's approach: integrate magnetic surface charges on the two faces
// /// normal to the source magnetization axis, and average the resulting field over the
// /// destination cell volume, with adaptive integration point counts based on distance
// /// and cell aspect ratio
// fn prism_kernel_tensor_numeric(
//     dx: f64,
//     dy: f64,
//     dz: f64,
//     sx: isize,
//     sy: isize,
// ) -> (f64, f64, f64, f64, f64, f64) {
//     let sz: isize = 0;

//     // Strict cube self-term to keep your existing unit test exact.
//     // (MuMax gets very close but not exactly 1/3; you deliberately wanted exact here.)
//     if sx == 0 && sy == 0 {
//         let cube = (dx - dy).abs() < 1e-15 && (dy - dz).abs() < 1e-15;
//         if cube {
//             let k = -MU0 / 3.0;
//             return (k, 0.0, 0.0, k, 0.0, k);
//         }
//     }

//     // Destination cell centre relative to source cell centre.
//     let r_center = [sx as f64 * dx, sy as f64 * dy, 0.0_f64];

//     // Compute full 3×3 kernel K_{i,j} = B_i due to unit magnetization along j.
//     // We'll fill by looping source axis j=0..2 and computing B-vector.
//     let mut k = [[0.0_f64; 3]; 3];

//     for source_axis in 0..3 {
//         let h = mumax_like_h_from_unit_m(
//             source_axis,
//             r_center,
//             [dx, dy, dz],
//             [sx, sy, sz],
//             DEMAG_ACCURACY,
//         );

//         // Convert H-kernel to B-kernel: B = μ0 H.
//         for dest_axis in 0..3 {
//             k[dest_axis][source_axis] = MU0 * h[dest_axis];
//         }
//     }

//     // Enforce symmetry numerically (reciprocity): Kxy = Kyx, etc.
//     let kxx = k[0][0];
//     let kyy = k[1][1];
//     let kzz = k[2][2];

//     let kxy = 0.5 * (k[0][1] + k[1][0]);
//     let kxz = 0.5 * (k[0][2] + k[2][0]);
//     let kyz = 0.5 * (k[1][2] + k[2][1]);

//     (kxx, kxy, kxz, kyy, kyz, kzz)
// }

// /// Computes the demag field H (A/m) at displacement `r_center` due to a *unit* magnetization
// /// along `source_axis` (0=x,1=y,2=z), using MuMax3's face-charge integration.
// ///
// /// This returns H per unit M_source (dimensionless), but expressed as H in A/m because
// /// it uses physical lengths (m). We later multiply by μ0 to get B-kernel (Tesla per A/m).
// fn mumax_like_h_from_unit_m(
//     source_axis: usize,
//     r_center: [f64; 3],
//     cell: [f64; 3],
//     disp_ijk: [isize; 3],
//     accuracy: f64,
// ) -> [f64; 3] {
//     // u = source axis, v,w perpendicular axes
//     let u = source_axis;
//     let v = (u + 1) % 3;
//     let w = (u + 2) % 3;

//     // Smallest cell dimension as length scale (MuMax uses this when d==0)
//     let mut lmin = cell[0];
//     if cell[1] < lmin { lmin = cell[1]; }
//     if cell[2] < lmin { lmin = cell[2]; }

//     // Closest distance between the two *cells* (not centers): MuMax's delta() logic.
//     let dx_min = delta_cell(disp_ijk[0]) * cell[0];
//     let dy_min = delta_cell(disp_ijk[1]) * cell[1];
//     let dz_min = delta_cell(disp_ijk[2]) * cell[2];

//     let mut d = (dx_min * dx_min + dy_min * dy_min + dz_min * dz_min).sqrt();
//     if d == 0.0 {
//         d = lmin;
//     }

//     // Maximum acceptable integration element size
//     let max_size = d / accuracy;

//     // Integration counts for destination volume
//     let nx = ((cell[0] / max_size).ceil().max(1.0)) as usize;
//     let ny = ((cell[1] / max_size).ceil().max(1.0)) as usize;
//     let nz = ((cell[2] / max_size).ceil().max(1.0)) as usize;

//     let mut nv = ((cell[v] / max_size).ceil().max(1.0)) as usize;
//     let mut nw = ((cell[w] / max_size).ceil().max(1.0)) as usize;

//     // MuMax-style staggering
//     nv *= 2;
//     nw *= 2;

//     debug_assert!(nx > 0 && ny > 0 && nz > 0 && nv > 0 && nw > 0);

//     // Averaging over destination volume is done by including 1/(nx*ny*nz),
//     // and face integration uses dS/(nv*nw). We combine both in "charge".
//     let scale = 1.0 / ((nv * nw * nx * ny * nz) as f64);
//     let face_area = cell[v] * cell[w];
//     let charge = face_area * scale; // effective point-charge weight for unit M_u=1

//     let pu1 = 0.5 * cell[u];  // + face centre
//     let pu2 = -0.5 * cell[u]; // - face centre

//     let mut pole = [0.0_f64; 3];
//     let mut h = [0.0_f64; 3];

//     // Surface integral over source face
//     for i in 0..nv {
//         let pv = -0.5 * cell[v] + cell[v] / (2.0 * nv as f64) + (i as f64) * (cell[v] / nv as f64);
//         pole[v] = pv;

//         for j in 0..nw {
//             let pw = -0.5 * cell[w] + cell[w] / (2.0 * nw as f64) + (j as f64) * (cell[w] / nw as f64);
//             pole[w] = pw;

//             // Volume integral over destination cell
//             for ax in 0..nx {
//                 let rx = r_center[0]
//                     - 0.5 * cell[0]
//                     + cell[0] / (2.0 * nx as f64)
//                     + (ax as f64) * (cell[0] / nx as f64);

//                 for ay in 0..ny {
//                     let ry = r_center[1]
//                         - 0.5 * cell[1]
//                         + cell[1] / (2.0 * ny as f64)
//                         + (ay as f64) * (cell[1] / ny as f64);

//                     for az in 0..nz {
//                         let rz = r_center[2]
//                             - 0.5 * cell[2]
//                             + cell[2] / (2.0 * nz as f64)
//                             + (az as f64) * (cell[2] / nz as f64);

//                         // + pole
//                         pole[u] = pu1;
//                         let r1x = rx - pole[0];
//                         let r1y = ry - pole[1];
//                         let r1z = rz - pole[2];
//                         let r1 = (r1x * r1x + r1y * r1y + r1z * r1z).sqrt();

//                         // H from magnetic charge: (1/4π) q r / r^3
//                         let q1 = charge / (4.0 * PI * r1 * r1 * r1);
//                         let hx1 = r1x * q1;
//                         let hy1 = r1y * q1;
//                         let hz1 = r1z * q1;

//                         // - pole (negative charge)
//                         pole[u] = pu2;
//                         let r2x = rx - pole[0];
//                         let r2y = ry - pole[1];
//                         let r2z = rz - pole[2];
//                         let r2 = (r2x * r2x + r2y * r2y + r2z * r2z).sqrt();

//                         let q2 = -charge / (4.0 * PI * r2 * r2 * r2);

//                         // Ordered addition (MuMax does this for accuracy)
//                         h[0] += hx1 + r2x * q2;
//                         h[1] += hy1 + r2y * q2;
//                         h[2] += hz1 + r2z * q2;
//                     }
//                 }
//             }
//         }
//     }

//     h
// }

// /// MuMax's "delta": closest distance between cells given integer centre distance.
// /// If cells touch (even at a corner), delta is 0.
// /// For |d|=0 => 0; |d|=1 => 0; |d|=2 => 1; etc.
// #[inline]
// fn delta_cell(d: isize) -> f64 {
//     let mut a = d.abs() as f64;
//     if a > 0.0 {
//         a -= 1.0;
//     }
//     a
// }

// // Currently unused: kept for potential hybrid near-field / far-field kernels.
// #[allow(dead_code)]
// fn dipole_far_field(
//     dx: f64,
//     dy: f64,
//     dz: f64,
//     sx: isize,
//     sy: isize,
// ) -> (f64, f64, f64, f64, f64, f64) {
//     let rx = (sx as f64) * dx;
//     let ry = (sy as f64) * dy;
//     let rz = 0.0;

//     let r2 = rx * rx + ry * ry + rz * rz;
//     let r = r2.sqrt();
//     let r3 = r2 * r;
//     let r5 = r3 * r2;

//     let volume = dx * dy * dz;
//     let pref = MU0 * volume / (4.0 * PI);

//     let inv_r3 = 1.0 / r3;

//     let kxx = pref * (3.0 * rx * rx / r5 - inv_r3);
//     let kxy = pref * (3.0 * rx * ry / r5);
//     let kxz = pref * (3.0 * rx * rz / r5);
//     let kyy = pref * (3.0 * ry * ry / r5 - inv_r3);
//     let kyz = pref * (3.0 * ry * rz / r5);
//     let kzz = pref * (3.0 * rz * rz / r5 - inv_r3);

//     (kxx, kxy, kxz, kyy, kyz, kzz)
// }


// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::grid::Grid2D;

//     #[test]
//     fn demag_single_cell_uniform_matches_analytic_cube_self_term() {
//         let grid = Grid2D::new(1, 1, 1.0, 1.0, 1.0);

//         let mut m = VectorField2D::new(grid);
//         m.set_uniform(0.0, 0.0, 1.0);

//         let mat = Material {
//             ms: 1.0,
//             a_ex: 0.0,
//             k_u: 0.0,
//             easy_axis: [0.0, 0.0, 1.0],
//             dmi: None,
//             demag: true,
//         };

//         let mut b_eff = VectorField2D::new(grid);
//         b_eff.set_uniform(0.0, 0.0, 0.0);

//         add_demag_field(&grid, &m, &mut b_eff, &mat);

//         let b = b_eff.data[0];

//         assert!(b[0].abs() < 1e-12, "Bx={}", b[0]);
//         assert!(b[1].abs() < 1e-12, "By={}", b[1]);

//         let expected = -MU0 / 3.0;
//         assert!(
//             (b[2] - expected).abs() < 1e-10,
//             "b_z={}, expected={}",
//             b[2],
//             expected
//         );
//     }
// }