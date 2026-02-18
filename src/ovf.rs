// src/ovf.rs
//
// Reusable OVF writers for llg-sim.
// Supports OOMMF OVF 2.0 rectangular meshes:
//  - text data (MuMax-like)
//  - binary4 data (fast + compact)
//
// Binary4 uses little-endian floats and starts with the OVF2 check value 1234567.0f.

use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::grid::Grid2D;
use crate::vector_field::VectorField2D;

#[derive(Clone, Debug, Default)]
pub struct OvfMeta {
    pub title: String,
    pub desc_lines: Vec<String>,
    pub valuelabels: [String; 3],
    pub valueunits: [String; 3],
}

impl OvfMeta {
    pub fn magnetization() -> Self {
        Self {
            title: "m".to_string(),
            desc_lines: vec![],
            valuelabels: ["m_x".into(), "m_y".into(), "m_z".into()],
            valueunits: ["1".into(), "1".into(), "1".into()],
        }
    }

    /// Match MuMax-style time metadata used by a lot of parsers/viewers.
    pub fn with_total_sim_time(mut self, t_s: f64) -> Self {
        self.desc_lines
            .push(format!("Total simulation time:  {:.16e}  s", t_s));
        self
    }

    pub fn push_desc_line<S: Into<String>>(&mut self, s: S) {
        self.desc_lines.push(s.into());
    }
}

fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    Ok(())
}

pub fn write_ovf2_rectangular_text(
    path: &Path,
    grid: &Grid2D,
    m: &VectorField2D,
    meta: &OvfMeta,
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;

    let nx = grid.nx;
    let ny = grid.ny;
    let nz = 1usize;

    if m.data.len() != nx * ny {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "VectorField2D length mismatch: got {}, expected {} (nx*ny)",
                m.data.len(),
                nx * ny
            ),
        ));
    }

    let dx = grid.dx;
    let dy = grid.dy;
    let dz = grid.dz;

    let xmin = 0.0;
    let ymin = 0.0;
    let zmin = 0.0;
    let xmax = (nx as f64) * dx;
    let ymax = (ny as f64) * dy;
    let zmax = (nz as f64) * dz;

    let xbase = 0.5 * dx;
    let ybase = 0.5 * dy;
    let zbase = 0.5 * dz;

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# OOMMF OVF 2.0")?;
    writeln!(w, "# Segment count: 1")?;
    writeln!(w, "# Begin: Segment")?;
    writeln!(w, "# Begin: Header")?;
    writeln!(w, "# Title: {}", meta.title)?;
    writeln!(w, "# meshtype: rectangular")?;
    writeln!(w, "# meshunit: m")?;

    writeln!(w, "# xmin: {:.16e}", xmin)?;
    writeln!(w, "# ymin: {:.16e}", ymin)?;
    writeln!(w, "# zmin: {:.16e}", zmin)?;
    writeln!(w, "# xmax: {:.16e}", xmax)?;
    writeln!(w, "# ymax: {:.16e}", ymax)?;
    writeln!(w, "# zmax: {:.16e}", zmax)?;

    writeln!(w, "# valuedim: 3")?;
    writeln!(
        w,
        "# valuelabels: {} {} {}",
        meta.valuelabels[0], meta.valuelabels[1], meta.valuelabels[2]
    )?;
    writeln!(
        w,
        "# valueunits: {} {} {}",
        meta.valueunits[0], meta.valueunits[1], meta.valueunits[2]
    )?;

    for d in &meta.desc_lines {
        writeln!(w, "# Desc: {}", d)?;
    }

    writeln!(w, "# xbase: {:.16e}", xbase)?;
    writeln!(w, "# ybase: {:.16e}", ybase)?;
    writeln!(w, "# zbase: {:.16e}", zbase)?;
    writeln!(w, "# xnodes: {}", nx)?;
    writeln!(w, "# ynodes: {}", ny)?;
    writeln!(w, "# znodes: {}", nz)?;
    writeln!(w, "# xstepsize: {:.16e}", dx)?;
    writeln!(w, "# ystepsize: {:.16e}", dy)?;
    writeln!(w, "# zstepsize: {:.16e}", dz)?;

    writeln!(w, "# End: Header")?;
    writeln!(w, "# Begin: Data Text")?;

    // x fastest, then y, then z
    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;
            let v = m.data[idx];
            writeln!(w, "{:.10e} {:.10e} {:.10e}", v[0], v[1], v[2])?;
        }
    }

    writeln!(w, "# End: Data Text")?;
    writeln!(w, "# End: Segment")?;
    Ok(())
}

pub fn write_ovf2_rectangular_binary4(
    path: &Path,
    grid: &Grid2D,
    m: &VectorField2D,
    meta: &OvfMeta,
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;

    let nx = grid.nx;
    let ny = grid.ny;
    let nz = 1usize;

    if m.data.len() != nx * ny {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "VectorField2D length mismatch: got {}, expected {} (nx*ny)",
                m.data.len(),
                nx * ny
            ),
        ));
    }

    let dx = grid.dx;
    let dy = grid.dy;
    let dz = grid.dz;

    let xbase = 0.5 * dx;
    let ybase = 0.5 * dy;
    let zbase = 0.5 * dz;

    let xmax = (nx as f64) * dx;
    let ymax = (ny as f64) * dy;

    let mut f = BufWriter::new(File::create(path)?);

    writeln!(f, "# OOMMF OVF 2.0")?;
    writeln!(f, "# Segment count: 1")?;
    writeln!(f, "# Begin: Segment")?;
    writeln!(f, "# Begin: Header")?;
    writeln!(f, "# Title: {}", meta.title)?;
    for d in &meta.desc_lines {
        writeln!(f, "# Desc: {}", d)?;
    }
    writeln!(f, "# meshunit: m")?;
    writeln!(f, "# meshtype: rectangular")?;

    writeln!(f, "# xbase: {:.17e}", xbase)?;
    writeln!(f, "# ybase: {:.17e}", ybase)?;
    writeln!(f, "# zbase: {:.17e}", zbase)?;
    writeln!(f, "# xstepsize: {:.17e}", dx)?;
    writeln!(f, "# ystepsize: {:.17e}", dy)?;
    writeln!(f, "# zstepsize: {:.17e}", dz)?;
    writeln!(f, "# xnodes: {}", nx)?;
    writeln!(f, "# ynodes: {}", ny)?;
    writeln!(f, "# znodes: {}", nz)?;

    writeln!(f, "# xmin: 0")?;
    writeln!(f, "# xmax: {:.17e}", xmax)?;
    writeln!(f, "# ymin: 0")?;
    writeln!(f, "# ymax: {:.17e}", ymax)?;
    writeln!(f, "# zmin: 0")?;
    writeln!(f, "# zmax: {:.17e}", dz)?;

    writeln!(f, "# valuedim: 3")?;
    writeln!(
        f,
        "# valuelabels: {} {} {}",
        meta.valuelabels[0], meta.valuelabels[1], meta.valuelabels[2]
    )?;
    writeln!(
        f,
        "# valueunits: {} {} {}",
        meta.valueunits[0], meta.valueunits[1], meta.valueunits[2]
    )?;

    writeln!(f, "# End: Header")?;
    writeln!(f, "# Begin: Data Binary 4")?;

    // OVF2 binary4 check value (little endian)
    let check: f32 = 1234567.0;
    f.write_all(&check.to_le_bytes())?;

    for v in &m.data {
        let x = v[0] as f32;
        let y = v[1] as f32;
        let z = v[2] as f32;
        f.write_all(&x.to_le_bytes())?;
        f.write_all(&y.to_le_bytes())?;
        f.write_all(&z.to_le_bytes())?;
    }

    writeln!(f)?;
    writeln!(f, "# End: Data Binary 4")?;
    writeln!(f, "# End: Segment")?;
    writeln!(f, "# End: File")?;
    f.flush()?;
    Ok(())
}

// Optional: a reusable “dump every X seconds” helper.
// Keep naming policy outside if you want maximum flexibility.
pub struct OvfSeries {
    pub dir: PathBuf,
    pub every: f64,
    next_dump: f64,
    frame: u64,
}

impl OvfSeries {
    pub fn new(dir: PathBuf, every_s: f64) -> std::io::Result<Self> {
        create_dir_all(&dir)?;
        Ok(Self {
            dir,
            every: every_s.max(0.0),
            next_dump: 0.0,
            frame: 0,
        })
    }

    pub fn dump_initial<F>(
        &mut self,
        grid: &Grid2D,
        m: &VectorField2D,
        mut make_path: F,
        meta: &OvfMeta,
    ) -> std::io::Result<()>
    where
        F: FnMut(u64, f64) -> PathBuf,
    {
        self.next_dump = 0.0;
        let p = make_path(self.frame, 0.0);
        write_ovf2_rectangular_binary4(&p, grid, m, meta)?;
        self.frame += 1;
        self.next_dump = self.every;
        Ok(())
    }

    pub fn maybe_dump_due<F>(
        &mut self,
        grid: &Grid2D,
        m: &VectorField2D,
        t_approx: f64,
        mut make_path: F,
        meta_builder: impl Fn(f64) -> OvfMeta,
    ) -> std::io::Result<()>
    where
        F: FnMut(u64, f64) -> PathBuf,
    {
        if self.every <= 0.0 {
            return Ok(());
        }
        let eps = 1e-15_f64;
        while t_approx + eps >= self.next_dump {
            let t = self.next_dump;
            let p = make_path(self.frame, t);
            let meta = meta_builder(t);
            write_ovf2_rectangular_binary4(&p, grid, m, &meta)?;
            self.frame += 1;
            self.next_dump += self.every;
        }
        Ok(())
    }
}
