// src/params.rs

/// Parameters for the LLG equation (simple, dimensionless version for now).
pub struct LLGParams {
    pub gamma: f64,       // gyromagnetic ratio (scaled)
    pub alpha: f64,       // damping constant
    pub dt: f64,          // time step
    pub h_ext: [f64; 3],  // uniform external field
}