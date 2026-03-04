// src/amr/stepper.rs

use crate::effective_field::{
    FieldMask, build_h_eff_masked, build_h_eff_masked_geom, coarse_fft_demag,
    demag_fft_uniform, demag_poisson_mg, mg_composite,
};
use crate::grid::Grid2D;
use crate::llg::{
    RK4Scratch, step_llg_rk4_recompute_field_masked_add,
    step_llg_rk4_recompute_field_masked_geom_add, step_llg_rk4_recompute_field_masked_relax_add,
    step_llg_rk4_recompute_field_masked_relax_geom_add,
};
use crate::params::{LLGParams, Material};
use crate::vec3::{cross, normalize};
use crate::vector_field::VectorField2D;

use rayon::prelude::*;

use crate::amr::hierarchy::AmrHierarchy2D;
use crate::amr::interp::sample_bilinear;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AmrDemagMode {
    /// Bridge-B style: flatten to uniform fine, run FFT demag on that, then sample back.
    AllFft,
    /// Stage 3A: run Poisson-MG demag only on the coarse grid; patches get no demag addend.
    MixMgCoarseOnly,
    /// Stage 3B: run Poisson-MG demag on the flattened uniform-fine composite, then sample back.
    AllMgUniformFine,
    /// AMR-aware composite-grid: enhanced-RHS MG on L0 + interpolation to patches.
    /// Avoids flattening to uniform fine — V-cycle runs on the coarse grid only.
    CompositeGrid,
    /// Coarse-FFT: exact Newell-tensor FFT on L0 with M-restriction from patches,
    /// then bilinear interpolation to patches.  No Poisson reformulation — uses the
    /// same proven FFT solver as AllFft but on the small coarse grid.
    CoarseFft,
}

impl AmrDemagMode {
    fn from_env() -> Self {
        // LLG_AMR_DEMAG_MODE = all_fft | mix | all_mg | composite | coarse_fft
        // Backward-compatible aliases are accepted.
        let v = std::env::var("LLG_AMR_DEMAG_MODE").ok();
        match v.as_deref().map(|s| s.trim().to_ascii_lowercase()) {
            Some(ref s) if s == "all_fft" || s == "fft" || s == "bridgeb_fft" || s == "bridgeb" => {
                Self::AllFft
            }
            Some(ref s) if s == "mix" || s == "mg_coarse_only" || s == "mg_coarse" => {
                Self::MixMgCoarseOnly
            }
            Some(ref s)
                if s == "all_mg" || s == "mg" || s == "mg_uniform" || s == "mg_uniform_fine" =>
            {
                Self::AllMgUniformFine
            }
            Some(ref s)
                if s == "composite" || s == "composite_grid" || s == "garcia" =>
            {
                Self::CompositeGrid
            }
            Some(ref s)
                if s == "coarse_fft" || s == "coarsefft" || s == "cfft" =>
            {
                Self::CoarseFft
            }
            _ => Self::AllFft,
        }
    }
}

// ------------------------------------------------------------
// Patch-local RK4 with an explicit "active index" set.
// ------------------------------------------------------------

/// RK4 scratch buffers for patch stepping where we need to update only a subset of cells.
///
/// We cannot reuse `crate::llg::RK4Scratch` here because its fields are private.
pub struct PatchRK4Scratch {
    pub m1: VectorField2D,
    pub m2: VectorField2D,
    pub m3: VectorField2D,
    pub b1: VectorField2D,
    pub b2: VectorField2D,
    pub b3: VectorField2D,
    pub b4: VectorField2D,
    pub b_add: VectorField2D,
    pub k1: Vec<[f64; 3]>,
    pub k2: Vec<[f64; 3]>,
    pub k3: Vec<[f64; 3]>,
    pub k4: Vec<[f64; 3]>,
}

impl PatchRK4Scratch {
    pub fn new(grid: Grid2D) -> Self {
        let n = grid.n_cells();
        Self {
            m1: VectorField2D::new(grid),
            m2: VectorField2D::new(grid),
            m3: VectorField2D::new(grid),
            b1: VectorField2D::new(grid),
            b2: VectorField2D::new(grid),
            b3: VectorField2D::new(grid),
            b4: VectorField2D::new(grid),
            b_add: VectorField2D::new(grid),
            k1: vec![[0.0; 3]; n],
            k2: vec![[0.0; 3]; n],
            k3: vec![[0.0; 3]; n],
            k4: vec![[0.0; 3]; n],
        }
    }

    pub fn resize_if_needed(&mut self, grid: Grid2D) {
        if self.m1.grid.nx == grid.nx
            && self.m1.grid.ny == grid.ny
            && self.m1.grid.dx == grid.dx
            && self.m1.grid.dy == grid.dy
            && self.m1.grid.dz == grid.dz
        {
            return;
        }
        *self = PatchRK4Scratch::new(grid);
    }
}

#[inline]
fn llg_rhs(m: [f64; 3], b: [f64; 3], gamma: f64, alpha: f64) -> [f64; 3] {
    let denom = 1.0 + alpha * alpha;
    let pref = -gamma / denom;
    let m_cross_b = cross(m, b);
    let m_cross_m_cross_b = cross(m, m_cross_b);
    [
        pref * (m_cross_b[0] + alpha * m_cross_m_cross_b[0]),
        pref * (m_cross_b[1] + alpha * m_cross_m_cross_b[1]),
        pref * (m_cross_b[2] + alpha * m_cross_m_cross_b[2]),
    ]
}

#[inline]
fn llg_rhs_relax(m: [f64; 3], b: [f64; 3], gamma: f64, alpha: f64) -> [f64; 3] {
    let denom = 1.0 + alpha * alpha;
    let pref = -gamma * alpha / denom;
    let m_cross_b = cross(m, b);
    let m_cross_m_cross_b = cross(m, m_cross_b);
    [
        pref * m_cross_m_cross_b[0],
        pref * m_cross_m_cross_b[1],
        pref * m_cross_m_cross_b[2],
    ]
}

#[inline]
fn add_scaled(a: [f64; 3], s: f64, b: [f64; 3]) -> [f64; 3] {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}

#[inline]
fn combo_rk4(k1: [f64; 3], k2: [f64; 3], k3: [f64; 3], k4: [f64; 3]) -> [f64; 3] {
    [
        (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    ]
}

#[cfg(debug_assertions)]
#[inline]
fn assert_field_finite(name: &str, f: &VectorField2D) {
    for (idx, v) in f.data.iter().enumerate() {
        if !(v[0].is_finite() && v[1].is_finite() && v[2].is_finite()) {
            panic!(
                "{} contains non-finite value at idx {}: [{:.6e}, {:.6e}, {:.6e}]",
                name, idx, v[0], v[1], v[2]
            );
        }
    }
}

#[cfg(not(debug_assertions))]
#[inline(always)]
fn assert_field_finite(_name: &str, _f: &VectorField2D) {}

/// Patch RK4 step where we only advance cells listed in `active`.
///
/// Ghost cells should *not* be updated by the integrator (they are boundary data
/// coming from coarse→fine interpolation).
///
/// NOTE: For Stage 1 we keep ghosts fixed over the whole RK step.
/// Later we can upgrade to refilling ghosts at each substage using a coarse predictor.
pub fn step_patch_rk4_recompute_field_masked_active(
    m: &mut VectorField2D,
    active: &[usize],
    params: &LLGParams,
    material: &Material,
    scratch: &mut PatchRK4Scratch,
    mask: FieldMask,
    geom_mask: Option<&[bool]>,
    relax: bool,
) {
    let grid = &m.grid;
    let gamma = params.gamma;
    let alpha = params.alpha;
    let dt = params.dt;

    scratch.resize_if_needed(*grid);

    // Stage 1
    if geom_mask.is_some() {
        build_h_eff_masked_geom(grid, m, &mut scratch.b1, params, material, mask, geom_mask);
    } else {
        build_h_eff_masked(grid, m, &mut scratch.b1, params, material, mask);
    }
    for &idx in active {
        let a = scratch.b_add.data[idx];
        scratch.b1.data[idx][0] += a[0];
        scratch.b1.data[idx][1] += a[1];
        scratch.b1.data[idx][2] += a[2];
    }
    for &idx in active {
        let mi = m.data[idx];
        let bi = scratch.b1.data[idx];
        scratch.k1[idx] = if relax {
            llg_rhs_relax(mi, bi, gamma, alpha)
        } else {
            llg_rhs(mi, bi, gamma, alpha)
        };
    }

    // m1 = m + 0.5 dt k1 (active only)
    scratch.m1.data.clone_from(&m.data);
    for &idx in active {
        let v = add_scaled(m.data[idx], 0.5 * dt, scratch.k1[idx]);
        scratch.m1.data[idx] = normalize(v);
    }

    // Stage 2
    if geom_mask.is_some() {
        build_h_eff_masked_geom(
            grid,
            &scratch.m1,
            &mut scratch.b2,
            params,
            material,
            mask,
            geom_mask,
        );
    } else {
        build_h_eff_masked(grid, &scratch.m1, &mut scratch.b2, params, material, mask);
    }
    for &idx in active {
        let a = scratch.b_add.data[idx];
        scratch.b2.data[idx][0] += a[0];
        scratch.b2.data[idx][1] += a[1];
        scratch.b2.data[idx][2] += a[2];
    }
    for &idx in active {
        let mi = scratch.m1.data[idx];
        let bi = scratch.b2.data[idx];
        scratch.k2[idx] = if relax {
            llg_rhs_relax(mi, bi, gamma, alpha)
        } else {
            llg_rhs(mi, bi, gamma, alpha)
        };
    }

    // m2 = m + 0.5 dt k2
    scratch.m2.data.clone_from(&m.data);
    for &idx in active {
        let v = add_scaled(m.data[idx], 0.5 * dt, scratch.k2[idx]);
        scratch.m2.data[idx] = normalize(v);
    }

    // Stage 3
    if geom_mask.is_some() {
        build_h_eff_masked_geom(
            grid,
            &scratch.m2,
            &mut scratch.b3,
            params,
            material,
            mask,
            geom_mask,
        );
    } else {
        build_h_eff_masked(grid, &scratch.m2, &mut scratch.b3, params, material, mask);
    }
    for &idx in active {
        let a = scratch.b_add.data[idx];
        scratch.b3.data[idx][0] += a[0];
        scratch.b3.data[idx][1] += a[1];
        scratch.b3.data[idx][2] += a[2];
    }
    for &idx in active {
        let mi = scratch.m2.data[idx];
        let bi = scratch.b3.data[idx];
        scratch.k3[idx] = if relax {
            llg_rhs_relax(mi, bi, gamma, alpha)
        } else {
            llg_rhs(mi, bi, gamma, alpha)
        };
    }

    // m3 = m + dt k3
    scratch.m3.data.clone_from(&m.data);
    for &idx in active {
        let v = add_scaled(m.data[idx], dt, scratch.k3[idx]);
        scratch.m3.data[idx] = normalize(v);
    }

    // Stage 4
    if geom_mask.is_some() {
        build_h_eff_masked_geom(
            grid,
            &scratch.m3,
            &mut scratch.b4,
            params,
            material,
            mask,
            geom_mask,
        );
    } else {
        build_h_eff_masked(grid, &scratch.m3, &mut scratch.b4, params, material, mask);
    }
    for &idx in active {
        let a = scratch.b_add.data[idx];
        scratch.b4.data[idx][0] += a[0];
        scratch.b4.data[idx][1] += a[1];
        scratch.b4.data[idx][2] += a[2];
    }
    for &idx in active {
        let mi = scratch.m3.data[idx];
        let bi = scratch.b4.data[idx];
        scratch.k4[idx] = if relax {
            llg_rhs_relax(mi, bi, gamma, alpha)
        } else {
            llg_rhs(mi, bi, gamma, alpha)
        };
    }

    // Combine
    for &idx in active {
        let dk = combo_rk4(
            scratch.k1[idx],
            scratch.k2[idx],
            scratch.k3[idx],
            scratch.k4[idx],
        );
        let v = add_scaled(m.data[idx], dt, dk);
        m.data[idx] = normalize(v);
    }
}

// ------------------------------------------------------------
// Hierarchy stepper (Stage 1): coarse + level-1 patches.
// ------------------------------------------------------------

/// A simple RK4 stepper for a 2-level AMR hierarchy.
///
/// For now, we:
/// 1) fill fine ghosts from coarse
/// 2) advance fine patches (active interior only)
/// 3) advance coarse grid (whole domain)
/// 4) restrict fine back to coarse under patches
///
/// This is *not* yet a full Berger–Colella time integration scheme
/// (no subcycling, no refluxing). It's the smallest working unit
/// to validate AMR plumbing with local stencils.
pub struct AmrStepperRK4 {
    pub coarse_scratch: RK4Scratch,

    /// Level-1 patch scratch buffers (backward-compatible).
    pub patch_scratch: Vec<PatchRK4Scratch>,

    /// Level-2+ patch scratch buffers.
    ///
    /// Indexing: `patch_scratch_l2plus[0]` corresponds to hierarchy level 2,
    /// `patch_scratch_l2plus[1]` to level 3, etc.
    pub patch_scratch_l2plus: Vec<Vec<PatchRK4Scratch>>,

    pub b_demag_fine: VectorField2D,
    pub b_demag_coarse: VectorField2D,
    demag_mode: AmrDemagMode,
    pub relax: bool,

    // ---- Subcycling state (Berger–Colella) ----

    /// Whether subcycling is enabled (env `LLG_AMR_SUBCYCLE=1`).
    subcycle: bool,

    /// Maximum subcycle ratio (env `LLG_AMR_MAX_SUBCYCLE_RATIO`).
    /// Caps the effective subcycle ratio regardless of the number of levels.
    /// Default: usize::MAX (no cap).
    max_subcycle_ratio: usize,

    /// Saved coarse state *before* the coarse step (for temporal ghost interpolation).
    coarse_old: VectorField2D,

    /// Saved coarse state *after* the coarse step (for temporal ghost interpolation).
    /// Cloned from `h.coarse` post-step to avoid borrow conflicts during fine stepping.
    coarse_new: VectorField2D,

    /// Per-level old-state snapshots for parent-level temporal ghost interpolation.
    ///
    /// `level_old_m[0]` = saved L1 patch states (one `Vec<[f64; 3]>` per L1 patch).
    /// `level_old_m[1]` = saved L2 patch states.
    /// etc.
    ///
    /// Before stepping level L, we clone each patch's m.data into level_old_m[L-1].
    /// After stepping, the current patch m.data serves as the "new" state.
    /// Child-level ghosts then interpolate between old and new.
    level_old_m: Vec<Vec<Vec<[f64; 3]>>>,
}

impl AmrStepperRK4 {
    pub fn new(h: &AmrHierarchy2D, relax: bool) -> Self {
        let coarse_scratch = RK4Scratch::new(h.base_grid);
        let patch_scratch = h
            .patches
            .iter()
            .map(|p| PatchRK4Scratch::new(p.grid))
            .collect();
        let patch_scratch_l2plus = h
            .patches_l2plus
            .iter()
            .map(|lvl| lvl.iter().map(|p| PatchRK4Scratch::new(p.grid)).collect())
            .collect();
        let b_demag_coarse = VectorField2D::new(h.base_grid);
        let fine_grid = Grid2D::new(
            h.base_grid.nx * h.ratio,
            h.base_grid.ny * h.ratio,
            h.base_grid.dx / (h.ratio as f64),
            h.base_grid.dy / (h.ratio as f64),
            h.base_grid.dz,
        );
        let b_demag_fine = VectorField2D::new(fine_grid);

        // Subcycling is ON by default. Set LLG_AMR_SUBCYCLE=0 to disable.
        let subcycle = std::env::var("LLG_AMR_SUBCYCLE")
            .ok()
            .map(|s| !(s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("off")))
            .unwrap_or(true);

        let max_subcycle_ratio = std::env::var("LLG_AMR_MAX_SUBCYCLE_RATIO")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(usize::MAX);

        let coarse_old = VectorField2D::new(h.base_grid);
        let coarse_new = VectorField2D::new(h.base_grid);

        Self {
            coarse_scratch,
            patch_scratch,
            patch_scratch_l2plus,
            b_demag_fine,
            b_demag_coarse,
            demag_mode: AmrDemagMode::from_env(),
            relax,
            subcycle,
            max_subcycle_ratio,
            coarse_old,
            coarse_new,
            level_old_m: Vec::new(),
        }
    }

    pub fn sync_with_hierarchy(&mut self, h: &AmrHierarchy2D) {
        // Coarse grid shape never changes.

        // ---- Level 1 ----
        if self.patch_scratch.len() != h.patches.len() {
            self.patch_scratch = h
                .patches
                .iter()
                .map(|p| PatchRK4Scratch::new(p.grid))
                .collect();
        } else {
            for (s, p) in self.patch_scratch.iter_mut().zip(h.patches.iter()) {
                s.resize_if_needed(p.grid);
            }
        }

        // ---- Level 2+ ----
        if self.patch_scratch_l2plus.len() != h.patches_l2plus.len() {
            self.patch_scratch_l2plus = h
                .patches_l2plus
                .iter()
                .map(|lvl| lvl.iter().map(|p| PatchRK4Scratch::new(p.grid)).collect())
                .collect();
            return;
        }

        for (scratch_lvl, patches_lvl) in self
            .patch_scratch_l2plus
            .iter_mut()
            .zip(h.patches_l2plus.iter())
        {
            if scratch_lvl.len() != patches_lvl.len() {
                *scratch_lvl = patches_lvl
                    .iter()
                    .map(|p| PatchRK4Scratch::new(p.grid))
                    .collect();
                continue;
            }
            for (s, p) in scratch_lvl.iter_mut().zip(patches_lvl.iter()) {
                s.resize_if_needed(p.grid);
            }
        }
    }

    pub fn step(
        &mut self,
        h: &mut AmrHierarchy2D,
        params: &LLGParams,
        mat: &Material,
        mask: FieldMask,
    ) {
        let n_levels = h.num_levels();
        if self.subcycle && n_levels > 1 {
            self.step_subcycled(h, params, mat, mask);
        } else {
            self.step_flat(h, params, mat, mask);
        }
    }

    /// Return the effective coarse time step for the current hierarchy depth.
    ///
    /// When subcycling is active, one call to `step()` advances by this amount.
    /// When subcycling is off, `step()` advances by `params.dt` (the finest dt).
    ///
    /// Respects `LLG_AMR_MAX_SUBCYCLE_RATIO` if set: the effective ratio is capped
    /// so `dt_coarse` never exceeds `params.dt * max_ratio`.
    pub fn coarse_dt(&self, params: &LLGParams, h: &AmrHierarchy2D) -> f64 {
        if !self.subcycle {
            return params.dt;
        }
        let n_levels = h.num_levels();
        let r_sq = (h.ratio * h.ratio) as f64;
        let natural_ratio = r_sq.powi((n_levels as i32) - 1) as usize;
        let effective_ratio = natural_ratio.min(self.max_subcycle_ratio);
        params.dt * effective_ratio as f64
    }

    /// Query whether subcycling is active.
    #[inline]
    pub fn is_subcycling(&self) -> bool {
        self.subcycle
    }

    /// Enable or disable subcycling programmatically.
    ///
    /// This overrides the `LLG_AMR_SUBCYCLE` env var that was read at construction time.
    /// Useful for tests and for benchmark runners that want to switch at runtime.
    #[inline]
    pub fn set_subcycle(&mut self, enabled: bool) {
        self.subcycle = enabled;
    }

    /// Set maximum subcycle ratio programmatically.
    ///
    /// Overrides the `LLG_AMR_MAX_SUBCYCLE_RATIO` env var. Set to `usize::MAX` to disable.
    #[inline]
    pub fn set_max_subcycle_ratio(&mut self, max_ratio: usize) {
        self.max_subcycle_ratio = max_ratio;
    }

    // ================================================================
    //  Flat stepping (original algorithm — no subcycling)
    // ================================================================

    fn step_flat(
        &mut self,
        h: &mut AmrHierarchy2D,
        params: &LLGParams,
        mat: &Material,
        mask: FieldMask,
    ) {
        self.sync_with_hierarchy(h);

        // Trim active-cell lists so that level-L patches skip cells covered
        // by level-(L+1) patches — those cells will be computed at the finer
        // level and restricted back, so updating them here is wasted work.
        // This is idempotent; the first call after a regrid does the real work,
        // subsequent calls within the same regrid epoch are a cheap no-op.
        h.trim_active_for_nesting();

        // Phase-2 Bridge B: build the flattened *uniform fine composite* magnetisation.
        //
        // This composite is used both for FFT demag (gold operator) and as the source of
        // parent-consistent ghost values for nested patches.
        let mut m_fine = h.flatten_to_uniform_fine();

        // If a geometry mask exists, ensure vacuum stays m=(0,0,0) on the fine grid before demag.
        // Keep the mask around so we can also zero demag addends in vacuum.
        let fine_mask = h.build_uniform_fine_mask();
        if let Some(fm) = fine_mask.as_deref() {
            for idx in 0..m_fine.grid.n_cells() {
                if !fm[idx] {
                    m_fine.data[idx] = [0.0, 0.0, 0.0];
                }
            }
        }

        // Parent-consistent ghost fill for all patch levels.
        //
        // Use the composite field so level-2+ ghost cells are consistent with level-(L-1)
        // interiors, rather than falling back to coarse.
        h.fill_patch_ghosts_from_uniform_fine(&m_fine);

        let use_fine_demag_sampling = matches!(
            self.demag_mode,
            AmrDemagMode::AllFft | AmrDemagMode::AllMgUniformFine
        );

        // 1) Compute demag
        self.compute_demag(h, &mut m_fine, fine_mask.as_deref(), mat);

        // 2) Step fine patches (active interior only) — PARALLEL
        {
            let b_demag_fine = &self.b_demag_fine;
            let relax = self.relax;
            let demag_sets_b_add = matches!(
                self.demag_mode,
                AmrDemagMode::CompositeGrid | AmrDemagMode::CoarseFft
            );
            h.patches
                .par_iter_mut()
                .zip(self.patch_scratch.par_iter_mut())
                .for_each(|(p, s)| {
                    let geom_mask = p.geom_mask_fine.as_deref();
                    let nxp = p.grid.nx;
                    let nyp = p.grid.ny;

                    if use_fine_demag_sampling {
                        // Build patch-local demag addend by sampling the fine demag field.
                        for j in 0..nyp {
                            for i in 0..nxp {
                                let (x, y) = p.cell_center_xy(i, j);
                                let v = sample_bilinear(b_demag_fine, x, y);
                                let idx = p.grid.idx(i, j);
                                s.b_add.data[idx] = v;

                                if let Some(gm) = geom_mask {
                                    if !gm[idx] {
                                        s.b_add.data[idx] = [0.0, 0.0, 0.0];
                                    }
                                }
                            }
                        }
                    } else if !demag_sets_b_add {
                        // Mix mode: patches are stepped without demag (exchange/anis/etc only).
                        // (CompositeGrid and CoarseFft already set b_add in compute_demag.)
                        s.b_add.set_uniform(0.0, 0.0, 0.0);
                    }

                    let active = p.active.as_slice();
                    let m = &mut p.m;

                    step_patch_rk4_recompute_field_masked_active(
                        m, active, params, mat, s, mask, geom_mask, relax,
                    );
                });
        }

        // 2b) Step level-2+ patches (active interior only) — PARALLEL per level
        {
            let b_demag_fine = &self.b_demag_fine;
            let relax = self.relax;
            let demag_sets_b_add = matches!(
                self.demag_mode,
                AmrDemagMode::CompositeGrid | AmrDemagMode::CoarseFft
            );
            for (patches_lvl, scratch_lvl) in h
                .patches_l2plus
                .iter_mut()
                .zip(self.patch_scratch_l2plus.iter_mut())
            {
                patches_lvl
                    .par_iter_mut()
                    .zip(scratch_lvl.par_iter_mut())
                    .for_each(|(p, s)| {
                        let geom_mask = p.geom_mask_fine.as_deref();
                        let nxp = p.grid.nx;
                        let nyp = p.grid.ny;

                        if use_fine_demag_sampling {
                            // Build patch-local demag addend by sampling the fine demag field.
                            for j in 0..nyp {
                                for i in 0..nxp {
                                    let (x, y) = p.cell_center_xy(i, j);
                                    let v = sample_bilinear(b_demag_fine, x, y);
                                    let idx = p.grid.idx(i, j);
                                    s.b_add.data[idx] = v;

                                    if let Some(gm) = geom_mask {
                                        if !gm[idx] {
                                            s.b_add.data[idx] = [0.0, 0.0, 0.0];
                                        }
                                    }
                                }
                            }
                        } else if !demag_sets_b_add {
                            // Mix mode: patches are stepped without demag (exchange/anis/etc only).
                            // (CompositeGrid and CoarseFft already set b_add in compute_demag.)
                            s.b_add.set_uniform(0.0, 0.0, 0.0);
                        }

                        let active = p.active.as_slice();
                        let m = &mut p.m;

                        step_patch_rk4_recompute_field_masked_active(
                            m, active, params, mat, s, mask, geom_mask, relax,
                        );
                    });
            }
        }

        // 3) Step coarse (whole grid)
        // If a geometry mask is present on the hierarchy, use the mask-aware `_geom`
        // stepping functions so exchange behaves correctly at vacuum boundaries.
        let geom_mask = h.geom_mask.as_deref();
        if self.relax {
            if let Some(gm) = geom_mask {
                step_llg_rk4_recompute_field_masked_relax_geom_add(
                    &mut h.coarse,
                    params,
                    mat,
                    &mut self.coarse_scratch,
                    mask,
                    Some(gm),
                    Some(&self.b_demag_coarse),
                );
            } else {
                step_llg_rk4_recompute_field_masked_relax_add(
                    &mut h.coarse,
                    params,
                    mat,
                    &mut self.coarse_scratch,
                    mask,
                    Some(&self.b_demag_coarse),
                );
            }
        } else {
            if let Some(gm) = geom_mask {
                step_llg_rk4_recompute_field_masked_geom_add(
                    &mut h.coarse,
                    params,
                    mat,
                    &mut self.coarse_scratch,
                    mask,
                    Some(gm),
                    Some(&self.b_demag_coarse),
                );
            } else {
                step_llg_rk4_recompute_field_masked_add(
                    &mut h.coarse,
                    params,
                    mat,
                    &mut self.coarse_scratch,
                    mask,
                    Some(&self.b_demag_coarse),
                );
            }
        }

        // 4) Fine→coarse restriction under patches
        h.restrict_patches_to_coarse();
    }

    // ================================================================
    //  Demag computation (shared between flat and subcycled paths)
    // ================================================================

    fn compute_demag(
        &mut self,
        h: &AmrHierarchy2D,
        m_fine: &mut VectorField2D,
        fine_mask: Option<&[bool]>,
        mat: &Material,
    ) {
        match self.demag_mode {
            AmrDemagMode::AllFft => {
                // Ensure our cached fine demag buffer matches the fine grid.
                if self.b_demag_fine.grid.nx != m_fine.grid.nx
                    || self.b_demag_fine.grid.ny != m_fine.grid.ny
                    || self.b_demag_fine.grid.dx != m_fine.grid.dx
                    || self.b_demag_fine.grid.dy != m_fine.grid.dy
                    || self.b_demag_fine.grid.dz != m_fine.grid.dz
                {
                    self.b_demag_fine = VectorField2D::new(m_fine.grid);
                }

                self.b_demag_fine.set_uniform(0.0, 0.0, 0.0);
                demag_fft_uniform::compute_demag_field_pbc(
                    &m_fine.grid,
                    m_fine,
                    &mut self.b_demag_fine,
                    mat,
                    0,
                    0,
                );

                // Zero demag in vacuum cells if we have a mask.
                if let Some(fm) = fine_mask {
                    for idx in 0..m_fine.grid.n_cells() {
                        if !fm[idx] {
                            self.b_demag_fine.data[idx] = [0.0, 0.0, 0.0];
                        }
                    }
                }

                // Build a coarse-grid addend by sampling the fine demag field at coarse cell centres.
                self.b_demag_coarse.set_uniform(0.0, 0.0, 0.0);
                for j in 0..h.base_grid.ny {
                    let y = (j as f64 + 0.5) * h.base_grid.dy;
                    for i in 0..h.base_grid.nx {
                        let x = (i as f64 + 0.5) * h.base_grid.dx;
                        let v = sample_bilinear(&self.b_demag_fine, x, y);
                        let idx = h.base_grid.idx(i, j);
                        self.b_demag_coarse.data[idx] = v;
                    }
                }
            }
            AmrDemagMode::MixMgCoarseOnly => {
                // Stage 3A: compute demag only on the coarse grid using Poisson-MG.
                assert_field_finite("h.coarse before MG coarse demag", &h.coarse);
                demag_poisson_mg::compute_demag_field_poisson_mg(
                    &h.base_grid,
                    &h.coarse,
                    &mut self.b_demag_coarse,
                    mat,
                );

                // If a geometry mask is present, zero demag in vacuum.
                if let Some(gm) = h.geom_mask.as_deref() {
                    for idx in 0..h.base_grid.n_cells() {
                        if !gm[idx] {
                            self.b_demag_coarse.data[idx] = [0.0, 0.0, 0.0];
                        }
                    }
                }
                assert_field_finite("b_demag_coarse (MG coarse)", &self.b_demag_coarse);
            }
            AmrDemagMode::AllMgUniformFine => {
                // Stage 3B: compute demag on the flattened uniform-fine composite using Poisson-MG.
                if self.b_demag_fine.grid.nx != m_fine.grid.nx
                    || self.b_demag_fine.grid.ny != m_fine.grid.ny
                    || self.b_demag_fine.grid.dx != m_fine.grid.dx
                    || self.b_demag_fine.grid.dy != m_fine.grid.dy
                    || self.b_demag_fine.grid.dz != m_fine.grid.dz
                {
                    self.b_demag_fine = VectorField2D::new(m_fine.grid);
                }

                assert_field_finite("m_fine before MG uniform-fine demag", m_fine);
                demag_poisson_mg::compute_demag_field_poisson_mg(
                    &m_fine.grid,
                    m_fine,
                    &mut self.b_demag_fine,
                    mat,
                );

                // Zero demag in vacuum cells if we have a mask.
                if let Some(fm) = fine_mask {
                    for idx in 0..m_fine.grid.n_cells() {
                        if !fm[idx] {
                            self.b_demag_fine.data[idx] = [0.0, 0.0, 0.0];
                        }
                    }
                }
                assert_field_finite("b_demag_fine (MG uniform fine)", &self.b_demag_fine);

                // Build coarse addend by sampling the fine demag field at coarse cell centres.
                self.b_demag_coarse.set_uniform(0.0, 0.0, 0.0);
                for j in 0..h.base_grid.ny {
                    let y = (j as f64 + 0.5) * h.base_grid.dy;
                    for i in 0..h.base_grid.nx {
                        let x = (i as f64 + 0.5) * h.base_grid.dx;
                        let v = sample_bilinear(&self.b_demag_fine, x, y);
                        let idx = h.base_grid.idx(i, j);
                        self.b_demag_coarse.data[idx] = v;
                    }
                }
                assert_field_finite(
                    "b_demag_coarse (sampled from MG fine)",
                    &self.b_demag_coarse,
                );
            }
            AmrDemagMode::CompositeGrid => {
                // AMR-aware composite-grid demag: enhanced-RHS MG on the coarse
                // grid, with fine-resolution ∇·M injected from patches.  The V-cycle
                // runs on the coarse grid (not the flattened fine grid).
                let (b_l1, b_l2plus) = mg_composite::compute_composite_demag(
                    h,
                    mat,
                    &mut self.b_demag_coarse,
                );

                // Zero demag in vacuum (coarse level)
                if let Some(gm) = h.geom_mask.as_deref() {
                    for idx in 0..h.base_grid.n_cells() {
                        if !gm[idx] {
                            self.b_demag_coarse.data[idx] = [0.0, 0.0, 0.0];
                        }
                    }
                }
                assert_field_finite("b_demag_coarse (composite)", &self.b_demag_coarse);

                Self::write_patch_demag_addends(
                    &mut self.patch_scratch,
                    &mut self.patch_scratch_l2plus,
                    h,
                    &b_l1,
                    &b_l2plus,
                    "composite",
                );
            }
            AmrDemagMode::CoarseFft => {
                // Coarse-FFT demag: exact Newell-tensor FFT on L0 with
                // M-restriction from patches, then bilinear interpolation.
                let (b_l1, b_l2plus) = coarse_fft_demag::compute_coarse_fft_demag(
                    h,
                    mat,
                    &mut self.b_demag_coarse,
                );

                // Zero demag in vacuum (coarse level)
                if let Some(gm) = h.geom_mask.as_deref() {
                    for idx in 0..h.base_grid.n_cells() {
                        if !gm[idx] {
                            self.b_demag_coarse.data[idx] = [0.0, 0.0, 0.0];
                        }
                    }
                }
                assert_field_finite("b_demag_coarse (coarse_fft)", &self.b_demag_coarse);

                Self::write_patch_demag_addends(
                    &mut self.patch_scratch,
                    &mut self.patch_scratch_l2plus,
                    h,
                    &b_l1,
                    &b_l2plus,
                    "coarse_fft",
                );
            }
        }
    }

    /// Write per-patch B_demag addends into scratch buffers from the vectors
    /// returned by composite-grid or coarse-FFT solvers.
    ///
    /// Shared between `CompositeGrid` and `CoarseFft` modes to avoid duplication.
    fn write_patch_demag_addends(
        patch_scratch: &mut [PatchRK4Scratch],
        patch_scratch_l2plus: &mut [Vec<PatchRK4Scratch>],
        h: &AmrHierarchy2D,
        b_l1: &[Vec<[f64; 3]>],
        b_l2plus: &[Vec<Vec<[f64; 3]>>],
        _tag: &str,
    ) {
        // Write per-patch B addends directly into scratch (L1)
        for (pi, s) in patch_scratch.iter_mut().enumerate() {
            if pi < b_l1.len() && pi < h.patches.len() {
                let b_patch = &b_l1[pi];
                for (idx, v) in b_patch.iter().enumerate() {
                    if idx < s.b_add.data.len() {
                        s.b_add.data[idx] = *v;
                    }
                }
                // Zero vacuum cells
                if let Some(gm) = h.patches[pi].geom_mask_fine.as_deref() {
                    for idx in 0..s.b_add.data.len().min(gm.len()) {
                        if !gm[idx] {
                            s.b_add.data[idx] = [0.0, 0.0, 0.0];
                        }
                    }
                }
            } else {
                s.b_add.set_uniform(0.0, 0.0, 0.0);
            }
        }

        // Write per-patch B addends directly into scratch (L2+)
        for (li, scratch_lvl) in patch_scratch_l2plus.iter_mut().enumerate() {
            let b_lvl = b_l2plus.get(li);
            for (pi, s) in scratch_lvl.iter_mut().enumerate() {
                if let Some(bl) = b_lvl {
                    if pi < bl.len() {
                        for (idx, v) in bl[pi].iter().enumerate() {
                            if idx < s.b_add.data.len() {
                                s.b_add.data[idx] = *v;
                            }
                        }
                        let h_idx = li;
                        if let Some(patches_lvl) = h.patches_l2plus.get(h_idx) {
                            if let Some(p) = patches_lvl.get(pi) {
                                if let Some(gm) = p.geom_mask_fine.as_deref() {
                                    for idx in 0..s.b_add.data.len().min(gm.len()) {
                                        if !gm[idx] {
                                            s.b_add.data[idx] = [0.0, 0.0, 0.0];
                                        }
                                    }
                                }
                            }
                        }
                        continue;
                    }
                }
                s.b_add.set_uniform(0.0, 0.0, 0.0);
            }
        }
    }

    // ================================================================
    //  Subcycled stepping (Berger–Colella)
    // ================================================================

    /// Berger–Colella subcycled time integration (v2).
    ///
    /// One call advances the simulation by `dt_coarse = params.dt * effective_ratio`.
    /// Each level runs at its natural CFL time step: level L takes r^(2*L) substeps of
    /// `dt_coarse / r^(2*L)`.
    ///
    /// v2 improvements over v1:
    /// - **Intermediate restriction**: after child level completes its substeps,
    ///   restrict child → parent before the parent's next substep (Berger–Colella §3).
    /// - **Parent-level temporal ghost fill**: L2+ ghosts interpolate from their direct
    ///   parent level's old/new states, not from L0. L1 ghosts still from L0.
    /// - **Max subcycle ratio cap**: `LLG_AMR_MAX_SUBCYCLE_RATIO` limits dt_coarse.
    fn step_subcycled(
        &mut self,
        h: &mut AmrHierarchy2D,
        params: &LLGParams,
        mat: &Material,
        mask: FieldMask,
    ) {
        self.sync_with_hierarchy(h);
        h.trim_active_for_nesting();

        let n_levels = h.num_levels();
        let r_sq = (h.ratio * h.ratio) as f64;
        let r_sq_u = h.ratio * h.ratio;

        // Phase 3: respect max subcycle ratio
        let natural_ratio = r_sq.powi((n_levels as i32) - 1) as usize;
        let effective_ratio = natural_ratio.min(self.max_subcycle_ratio);
        let dt_coarse = params.dt * effective_ratio as f64;

        // ----------------------------------------------------------
        // 1) Save old coarse state for temporal ghost interpolation
        // ----------------------------------------------------------
        self.coarse_old.data.clone_from(&h.coarse.data);

        // ----------------------------------------------------------
        // 2) Compute demag ONCE on the flattened composite (Strategy A)
        // ----------------------------------------------------------
        let mut m_fine = h.flatten_to_uniform_fine();
        let fine_mask = h.build_uniform_fine_mask();
        if let Some(fm) = fine_mask.as_deref() {
            for idx in 0..m_fine.grid.n_cells() {
                if !fm[idx] {
                    m_fine.data[idx] = [0.0, 0.0, 0.0];
                }
            }
        }

        // Initial ghost fill at t=0 (using the composite — same as flat path).
        h.fill_patch_ghosts_from_uniform_fine(&m_fine);

        let use_fine_demag_sampling = matches!(
            self.demag_mode,
            AmrDemagMode::AllFft | AmrDemagMode::AllMgUniformFine
        );

        self.compute_demag(h, &mut m_fine, fine_mask.as_deref(), mat);

        // ----------------------------------------------------------
        // 3) Set up frozen demag addends for ALL patches (once)
        // ----------------------------------------------------------
        // CompositeGrid and CoarseFft modes already populated b_add in compute_demag;
        // other modes need the explicit sampling/zeroing step.
        if !matches!(self.demag_mode, AmrDemagMode::CompositeGrid | AmrDemagMode::CoarseFft) {
            self.setup_patch_demag_addends(h, use_fine_demag_sampling);
        }

        // ----------------------------------------------------------
        // 4) Advance coarse for ONE step of dt_coarse
        // ----------------------------------------------------------
        {
            let mut params_coarse = *params;
            params_coarse.dt = dt_coarse;
            let geom_mask = h.geom_mask.as_deref();
            if self.relax {
                if let Some(gm) = geom_mask {
                    step_llg_rk4_recompute_field_masked_relax_geom_add(
                        &mut h.coarse,
                        &params_coarse,
                        mat,
                        &mut self.coarse_scratch,
                        mask,
                        Some(gm),
                        Some(&self.b_demag_coarse),
                    );
                } else {
                    step_llg_rk4_recompute_field_masked_relax_add(
                        &mut h.coarse,
                        &params_coarse,
                        mat,
                        &mut self.coarse_scratch,
                        mask,
                        Some(&self.b_demag_coarse),
                    );
                }
            } else {
                if let Some(gm) = geom_mask {
                    step_llg_rk4_recompute_field_masked_geom_add(
                        &mut h.coarse,
                        &params_coarse,
                        mat,
                        &mut self.coarse_scratch,
                        mask,
                        Some(gm),
                        Some(&self.b_demag_coarse),
                    );
                } else {
                    step_llg_rk4_recompute_field_masked_add(
                        &mut h.coarse,
                        &params_coarse,
                        mat,
                        &mut self.coarse_scratch,
                        mask,
                        Some(&self.b_demag_coarse),
                    );
                }
            }
        }

        // ----------------------------------------------------------
        // 5) Save coarse_new (post-step) for temporal interpolation
        // ----------------------------------------------------------
        self.coarse_new.data.clone_from(&h.coarse.data);

        // ----------------------------------------------------------
        // 6) Prepare per-level old-state storage (Phase 2)
        // ----------------------------------------------------------
        // Ensure level_old_m has entries for each patch level.
        // level_old_m[0] = L1 snapshots, level_old_m[1] = L2, etc.
        {
            let n_patch_levels = if n_levels > 1 { n_levels - 1 } else { 0 };
            self.level_old_m.resize(n_patch_levels, Vec::new());

            // L1 patches
            if n_patch_levels >= 1 {
                let n_l1 = h.patches.len();
                self.level_old_m[0].resize(n_l1, Vec::new());
            }
            // L2+ patches
            for k in 0..h.patches_l2plus.len() {
                let lvl_idx = k + 1; // level_old_m index (L2 = idx 1)
                if lvl_idx < n_patch_levels {
                    let n_patches = h.patches_l2plus[k].len();
                    self.level_old_m[lvl_idx].resize(n_patches, Vec::new());
                }
            }
        }

        // ----------------------------------------------------------
        // 7) Recursively advance fine levels (Berger–Colella §3 v2)
        // ----------------------------------------------------------
        // Recursive driver — implemented as a stack-based loop.
        // When a Frame pops (all substeps done), intermediate restriction fires.
        {
            let max_level = n_levels - 1; // 0-based, highest active level

            struct Frame {
                level: usize,
                k: usize,
                n_substeps: usize,
                dt_level: f64,
                /// Fractional time within the coarse step at the start of this level's substeps.
                t_start_frac: f64,
                /// dt of the PARENT level (for computing local alpha in parent-level ghost fill).
                dt_parent: f64,
            }

            let mut stack: Vec<Frame> = Vec::with_capacity(max_level);

            // Compute the number of fine substeps for level 1 based on effective ratio.
            // With max_subcycle_ratio, the effective ratio may be less than the natural ratio.
            // Level 1 takes `effective_ratio / (natural_ratio / r²)` substeps if capped,
            // but more precisely: dt_L1 = params.dt * r^(2*(n_levels-2)) (natural for L1).
            // n_L1_substeps = dt_coarse / dt_L1.
            // But dt_L1 must remain the natural CFL-limited dt for level 1.
            let dt_l1 = if n_levels > 2 {
                // L1's natural dt = params.dt * r^(2*(n_levels-2))
                params.dt * r_sq.powi((n_levels as i32) - 2)
            } else {
                // Only 2 levels: L1 dt = params.dt
                params.dt
            };
            let n_l1_substeps = (dt_coarse / dt_l1).round() as usize;

            if max_level >= 1 && n_l1_substeps > 0 {
                stack.push(Frame {
                    level: 1,
                    k: 0,
                    n_substeps: n_l1_substeps,
                    dt_level: dt_l1,
                    t_start_frac: 0.0,
                    dt_parent: dt_coarse,
                });
            }

            while let Some(frame) = stack.last_mut() {
                if frame.k >= frame.n_substeps {
                    // All substeps at this level done.
                    let finished_level = frame.level;
                    stack.pop();

                    // Phase 1: Intermediate restriction — child → parent
                    h.restrict_level_to_parent(finished_level);

                    continue;
                }

                let level = frame.level;
                let k = frame.k;
                let dt_level = frame.dt_level;
                let dt_parent = frame.dt_parent;
                let t_start_frac = frame.t_start_frac;

                // Alpha relative to the COARSE step (for L1 ghost fill from L0)
                let alpha_coarse = t_start_frac + (k as f64) * dt_level / dt_coarse;

                // Alpha LOCAL to the parent's substep (for L2+ ghost fill from parent)
                let alpha_local = k as f64 * dt_level / dt_parent;

                // Advance substep counter NOW (before any child push).
                frame.k += 1;

                // a) Phase 2: Save old state for this level's patches
                //    (so children can interpolate ghosts temporally from this level)
                {
                    let lom_idx = level - 1; // level_old_m index
                    match level {
                        1 => {
                            for (pidx, p) in h.patches.iter().enumerate() {
                                if pidx < self.level_old_m[lom_idx].len() {
                                    self.level_old_m[lom_idx][pidx]
                                        .clone_from(&p.m.data);
                                }
                            }
                        }
                        l if l >= 2 => {
                            let h_idx = l - 2;
                            if let Some(patches_lvl) = h.patches_l2plus.get(h_idx) {
                                for (pidx, p) in patches_lvl.iter().enumerate() {
                                    if lom_idx < self.level_old_m.len()
                                        && pidx < self.level_old_m[lom_idx].len()
                                    {
                                        self.level_old_m[lom_idx][pidx]
                                            .clone_from(&p.m.data);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // b) Temporal ghost fill for this level
                //    Phase 2: L1 uses coarse old/new; L2+ uses parent-level old/new
                if level == 1 {
                    // L1 ghosts from coarse (L0) temporal interpolation
                    h.fill_ghosts_time_interpolated(
                        level,
                        &self.coarse_old,
                        &self.coarse_new,
                        alpha_coarse,
                    );
                } else {
                    // L2+ ghosts from parent level patches (Phase 2)
                    let parent_lom_idx = level - 2; // parent's level_old_m index
                    let empty_parent: Vec<Vec<[f64; 3]>> = Vec::new();
                    let parent_old: &[Vec<[f64; 3]>] = if parent_lom_idx < self.level_old_m.len() {
                        &self.level_old_m[parent_lom_idx]
                    } else {
                        &empty_parent
                    };
                    h.fill_ghosts_from_parent_temporal(
                        level,
                        parent_old,
                        &self.coarse_old,
                        &self.coarse_new,
                        alpha_local,
                        alpha_coarse,
                    );
                }

                // c) Step all patches at this level with dt_level
                {
                    let mut params_level = *params;
                    params_level.dt = dt_level;
                    let relax = self.relax;

                    match level {
                        1 => {
                            h.patches
                                .par_iter_mut()
                                .zip(self.patch_scratch.par_iter_mut())
                                .for_each(|(p, s)| {
                                    step_patch_rk4_recompute_field_masked_active(
                                        &mut p.m,
                                        p.active.as_slice(),
                                        &params_level,
                                        mat,
                                        s,
                                        mask,
                                        p.geom_mask_fine.as_deref(),
                                        relax,
                                    );
                                });
                        }
                        l if l >= 2 => {
                            let idx_lvl = l - 2;
                            if let (Some(patches_lvl), Some(scratch_lvl)) = (
                                h.patches_l2plus.get_mut(idx_lvl),
                                self.patch_scratch_l2plus.get_mut(idx_lvl),
                            ) {
                                patches_lvl
                                    .par_iter_mut()
                                    .zip(scratch_lvl.par_iter_mut())
                                    .for_each(|(p, s)| {
                                        step_patch_rk4_recompute_field_masked_active(
                                            &mut p.m,
                                            p.active.as_slice(),
                                            &params_level,
                                            mat,
                                            s,
                                            mask,
                                            p.geom_mask_fine.as_deref(),
                                            relax,
                                        );
                                    });
                            }
                        }
                        _ => {}
                    }
                }

                // d) Push child level (if it exists) for recursive advance.
                //    The child covers the SAME time interval as the parent substep
                //    just completed, but at finer temporal resolution.
                if level < max_level {
                    let child_dt = dt_level / r_sq;
                    let child_n = r_sq_u;
                    stack.push(Frame {
                        level: level + 1,
                        k: 0,
                        n_substeps: child_n,
                        dt_level: child_dt,
                        // Child's t_start_frac = alpha_coarse at the start of the
                        // parent substep that just completed.
                        t_start_frac: alpha_coarse,
                        dt_parent: dt_level,
                    });
                }
                // Intermediate restriction happens when the child Frame pops (above).
            }
        }

        // ----------------------------------------------------------
        // 8) Final restriction: ensure coarse is fully up to date
        // ----------------------------------------------------------
        // The intermediate restrictions above already restricted each level
        // to its parent. But as a safety measure, do a final full restriction
        // so the coarse grid reflects the finest data everywhere.
        h.restrict_patches_to_coarse();
    }

    /// Pre-compute and store frozen demag addends in all patch scratch buffers.
    ///
    /// Called once per coarse step before the substep loop.
    fn setup_patch_demag_addends(
        &mut self,
        h: &AmrHierarchy2D,
        use_fine_demag_sampling: bool,
    ) {
        let b_demag_fine = &self.b_demag_fine;

        // Level-1 patches
        for (p, s) in h.patches.iter().zip(self.patch_scratch.iter_mut()) {
            let geom_mask = p.geom_mask_fine.as_deref();
            let nxp = p.grid.nx;
            let nyp = p.grid.ny;

            if use_fine_demag_sampling {
                for j in 0..nyp {
                    for i in 0..nxp {
                        let (x, y) = p.cell_center_xy(i, j);
                        let v = sample_bilinear(b_demag_fine, x, y);
                        let idx = p.grid.idx(i, j);
                        s.b_add.data[idx] = v;
                        if let Some(gm) = geom_mask {
                            if !gm[idx] {
                                s.b_add.data[idx] = [0.0, 0.0, 0.0];
                            }
                        }
                    }
                }
            } else {
                s.b_add.set_uniform(0.0, 0.0, 0.0);
            }
        }

        // Level-2+ patches
        for (patches_lvl, scratch_lvl) in h
            .patches_l2plus
            .iter()
            .zip(self.patch_scratch_l2plus.iter_mut())
        {
            for (p, s) in patches_lvl.iter().zip(scratch_lvl.iter_mut()) {
                let geom_mask = p.geom_mask_fine.as_deref();
                let nxp = p.grid.nx;
                let nyp = p.grid.ny;

                if use_fine_demag_sampling {
                    for j in 0..nyp {
                        for i in 0..nxp {
                            let (x, y) = p.cell_center_xy(i, j);
                            let v = sample_bilinear(b_demag_fine, x, y);
                            let idx = p.grid.idx(i, j);
                            s.b_add.data[idx] = v;
                            if let Some(gm) = geom_mask {
                                if !gm[idx] {
                                    s.b_add.data[idx] = [0.0, 0.0, 0.0];
                                }
                            }
                        }
                    }
                } else {
                    s.b_add.set_uniform(0.0, 0.0, 0.0);
                }
            }
        }
    }
}