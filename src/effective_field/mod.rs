// src/effective_field/mod.rs

pub mod anisotropy;
pub mod exchange;
pub mod zeeman;
pub mod dmi;

use crate::grid::Grid2D;
use crate::params::{LLGParams, Material};
use crate::vector_field::VectorField2D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldMask {
    /// Zeeman + Exchange + Anisotropy (no DMI, no Demag)
    ExchAnis,
    /// Zeeman + Exchange + Anisotropy + DMI (no Demag)
    ExchAnisDmi,
    /// All currently implemented terms (includes DMI if mat.dmi is Some)
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
    exchange::add_exchange_field(grid, m, b_eff, mat);
    anisotropy::add_uniaxial_anisotropy_field(m, b_eff, mat);

    // DMI only if mask allows it and material has DMI enabled
    let include_dmi = matches!(mask, FieldMask::ExchAnisDmi | FieldMask::Full);
    if include_dmi && mat.dmi.is_some() {
        dmi::add_dmi_field(grid, m, b_eff, mat);
    }

    // (Demag will be added later under FieldMask::Full.)
}

/// Backwards-compatible: build full B_eff (includes DMI if mat.dmi is Some).
pub fn build_h_eff(
    grid: &Grid2D,
    m: &VectorField2D,
    b_eff: &mut VectorField2D,
    sim: &LLGParams,
    mat: &Material,
) {
    build_h_eff_masked(grid, m, b_eff, sim, mat, FieldMask::Full);
}