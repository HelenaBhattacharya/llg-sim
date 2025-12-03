// src/params.rs

/// Dynamical / simulation parameters for LLG.
pub struct LLGParams {
    pub gamma: f64,
    pub alpha: f64,
    pub dt: f64,
    /// External field (Tesla) – for now uniform in space.
    pub h_ext: [f64; 3],
}

/// Material parameters (single-region for now).
pub struct Material {
    /// Saturation magnetisation (A/m).
    pub ms: f64,
    /// Exchange stiffness (J/m).
    pub a_ex: f64,
    /// Uniaxial anisotropy constant (J/m^3), if we add anisotropy later.
    pub k_u: f64,
    /// Easy-axis direction (unit vector).
    pub easy_axis: [f64; 3],
}