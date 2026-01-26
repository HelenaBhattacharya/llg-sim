use serde::Serialize;
use serde_json;
use std::fs::File;
use std::path::Path;

#[derive(Serialize)]
pub struct RunConfig {
    pub geometry: GeometryConfig,
    pub material: MaterialConfig,
    pub fields: FieldConfig,
    pub numerics: NumericsConfig,
    pub run: RunInfo,
}

#[derive(Serialize)]
pub struct GeometryConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

#[derive(Serialize)]
pub struct MaterialConfig {
    pub ms: f64,
    pub aex: f64,
    pub ku1: f64,
    pub easy_axis: [f64; 3],
}

#[derive(Serialize)]
pub struct FieldConfig {
    pub b_ext: [f64; 3],
    pub demag: bool,
    pub dmi: Option<f64>,
}

#[derive(Serialize)]
pub struct NumericsConfig {
    pub integrator: String,
    /// Initial timestep (s). For adaptive integrators, dt will vary during the run; see dt_history.csv.
    pub dt: f64,
    pub steps: usize,
    pub output_stride: usize,

    // Optional adaptive-step settings (MuMax-like)
    pub max_err: Option<f64>,
    pub headroom: Option<f64>,
    pub dt_min: Option<f64>,
    pub dt_max: Option<f64>,
}

#[derive(Serialize)]
pub struct RunInfo {
    pub binary: String,
    pub run_id: String,

    // Optional provenance (can be filled later)
    pub git_commit: Option<String>,
    pub timestamp_utc: Option<String>,
}

impl RunConfig {
    pub fn write_to_dir(&self, out_dir: &Path) -> std::io::Result<()> {
        let path = out_dir.join("config.json");
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}