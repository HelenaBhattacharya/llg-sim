// src/effective_field/mod.rs
/// NOTE: Despite the name build_h_eff*, this builds the effective induction B_eff (Tesla),
/// consistent with the global convention used throughout the solver.
pub mod anisotropy;
pub mod demag;
pub mod demag_fft_uniform;
pub mod demag_poisson_mg;
pub mod dmi;
pub mod exchange;
pub mod zeeman;

// 3D multigrid infrastructure (restored from pre-FK codebase)
pub mod mg_kernels;
pub mod mg_treecode;

// Diagnostics (timing infrastructure)
pub mod mg_diagnostics;

// AMR composite-grid wrapper (calls 3D MG solver on coarse grid)
pub mod mg_composite;

// AMR coarse-FFT wrapper (production — exact Newell-tensor FFT on L0)
pub mod coarse_fft_demag;

// Retired / archived modules (2D FK decomposition — broken, see project docs):
//   boundary_integral_2d.rs  — 2D single-layer potential (sign/node-cell bugs)
//   mg_stencil.rs            — 2D Laplacian stencils (FK-specific)
//   mg_solver.rs             — 2D cell-centred MG (FK-specific)
//   mg_config.rs             — 2D MG config (superseded by config in demag_poisson_mg.rs)
//   demag_poisson_dst.rs     — DST-based Poisson solver (retired; kernel mismatch)
//   dst_poisson_2d.rs        — DST spectral solver (retired)

use crate::grid::Grid2D;
use crate::params::{LLGParams, Material};
use crate::vector_field::VectorField2D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldMask {
    /// Zeeman + Exchange + Anisotropy (no DMI, no Demag)
    ExchAnis,
    /// Zeeman + Exchange + Anisotropy + DMI (no Demag)
    ExchAnisDmi,
    /// All currently implemented terms (includes DMI if mat.dmi is Some, Demag if mat.demag = true)
    Full,
}

/// Build effective induction with a mask controlling which terms are included.
pub fn build_h_eff_masked(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    sim: &LLGParams,
    mat: &Material,
    mask: FieldMask,
) {
    b_eff.set_uniform(0.0, 0.0, 0.0);

    // Zeeman (always included; may be zero)
    zeeman::add_zeeman_field(b_eff, sim.b_ext);

    // Exchange + anisotropy
    exchange::add_exchange_field_masked(grid, m, b_eff, mat, None);
    anisotropy::add_uniaxial_anisotropy_field(m, b_eff, mat);

    // DMI only if mask allows it and material has DMI enabled
    let include_dmi = matches!(mask, FieldMask::ExchAnisDmi | FieldMask::Full);
    if include_dmi && mat.dmi.is_some() {
        dmi::add_dmi_field(grid, m, b_eff, mat);
    }

    // Demag only if mask is Full and demag is enabled.
    if matches!(mask, FieldMask::Full) && mat.demag {
        demag::add_demag_field(grid, m, b_eff, mat);
    }
}

/// Build effective induction with a term mask and an optional geometry mask.
pub fn build_h_eff_masked_geom(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    sim: &LLGParams,
    mat: &Material,
    mask: FieldMask,
    geom_mask: Option<&[bool]>,
) {
    b_eff.set_uniform(0.0, 0.0, 0.0);

    zeeman::add_zeeman_field(b_eff, sim.b_ext);
    exchange::add_exchange_field_masked(grid, m, b_eff, mat, geom_mask);
    anisotropy::add_uniaxial_anisotropy_field(m, b_eff, mat);

    let include_dmi = matches!(mask, FieldMask::ExchAnisDmi | FieldMask::Full);
    if include_dmi && mat.dmi.is_some() {
        dmi::add_dmi_field_masked(grid, m, b_eff, mat, geom_mask);
    }

    if matches!(mask, FieldMask::Full) && mat.demag {
        demag::add_demag_field(grid, m, b_eff, mat);
    }
}

/// Backwards-compatible: build full B_eff.
pub fn build_h_eff(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    sim: &LLGParams,
    mat: &Material,
) {
    build_h_eff_masked(grid, m, b_eff, sim, mat, FieldMask::Full);
}