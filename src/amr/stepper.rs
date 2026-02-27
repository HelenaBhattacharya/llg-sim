// src/amr/stepper.rs

use crate::effective_field::{
    FieldMask, build_h_eff_masked, build_h_eff_masked_geom, demag_fft_uniform, demag_poisson_mg,
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
}

impl AmrDemagMode {
    fn from_env() -> Self {
        // LLG_AMR_DEMAG_MODE = all_fft | mix | all_mg
        // Backward-compatible aliases are accepted.
        let v = std::env::var("LLG_AMR_DEMAG_MODE").ok();
        match v.as_deref().map(|s| s.trim().to_ascii_lowercase()) {
            Some(ref s) if s == "all_fft" || s == "fft" || s == "bridgeb_fft" || s == "bridgeb" => Self::AllFft,
            Some(ref s) if s == "mix" || s == "mg_coarse_only" || s == "mg_coarse" => Self::MixMgCoarseOnly,
            Some(ref s) if s == "all_mg" || s == "mg" || s == "mg_uniform" || s == "mg_uniform_fine" => Self::AllMgUniformFine,
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
        Self {
            coarse_scratch,
            patch_scratch,
            patch_scratch_l2plus,
            b_demag_fine,
            b_demag_coarse,
            demag_mode: AmrDemagMode::from_env(),
            relax,
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
        self.sync_with_hierarchy(h);

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

        let use_fine_demag_sampling = matches!(self.demag_mode, AmrDemagMode::AllFft | AmrDemagMode::AllMgUniformFine);

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
                    &m_fine,
                    &mut self.b_demag_fine,
                    mat,
                    0,
                    0,
                );

                // Zero demag in vacuum cells if we have a mask.
                if let Some(fm) = fine_mask.as_deref() {
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
                // Stage 3B: compute demag on the flattened uniform-fine composite using Poisson-MG,
                // then sample back like Bridge-B.

                // Ensure our cached fine demag buffer matches the fine grid.
                if self.b_demag_fine.grid.nx != m_fine.grid.nx
                    || self.b_demag_fine.grid.ny != m_fine.grid.ny
                    || self.b_demag_fine.grid.dx != m_fine.grid.dx
                    || self.b_demag_fine.grid.dy != m_fine.grid.dy
                    || self.b_demag_fine.grid.dz != m_fine.grid.dz
                {
                    self.b_demag_fine = VectorField2D::new(m_fine.grid);
                }

                assert_field_finite("m_fine before MG uniform-fine demag", &m_fine);
                demag_poisson_mg::compute_demag_field_poisson_mg(
                    &m_fine.grid,
                    &m_fine,
                    &mut self.b_demag_fine,
                    mat,
                );

                // Zero demag in vacuum cells if we have a mask.
                if let Some(fm) = fine_mask.as_deref() {
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
                assert_field_finite("b_demag_coarse (sampled from MG fine)", &self.b_demag_coarse);
            }
        }

        // 2) Step fine patches (active interior only)
        for (p, s) in h.patches.iter_mut().zip(self.patch_scratch.iter_mut()) {
            let geom_mask = p.geom_mask_fine.as_deref();
            let nxp = p.grid.nx;
            let nyp = p.grid.ny;

            if use_fine_demag_sampling {
                // Build patch-local demag addend by sampling the fine demag field.
                for j in 0..nyp {
                    for i in 0..nxp {
                        let (x, y) = p.cell_center_xy(i, j);
                        let v = sample_bilinear(&self.b_demag_fine, x, y);
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
                // Mix mode: patches are stepped without demag (exchange/anis/etc only).
                s.b_add.set_uniform(0.0, 0.0, 0.0);
            }

            let active = p.active.as_slice();
            let m = &mut p.m;

            step_patch_rk4_recompute_field_masked_active(
                m, active, params, mat, s, mask, geom_mask, self.relax,
            );
        }

        // 2b) Step level-2+ patches (active interior only)
        for (patches_lvl, scratch_lvl) in h
            .patches_l2plus
            .iter_mut()
            .zip(self.patch_scratch_l2plus.iter_mut())
        {
            for (p, s) in patches_lvl.iter_mut().zip(scratch_lvl.iter_mut()) {
                let geom_mask = p.geom_mask_fine.as_deref();
                let nxp = p.grid.nx;
                let nyp = p.grid.ny;

                if use_fine_demag_sampling {
                    // Build patch-local demag addend by sampling the fine demag field.
                    for j in 0..nyp {
                        for i in 0..nxp {
                            let (x, y) = p.cell_center_xy(i, j);
                            let v = sample_bilinear(&self.b_demag_fine, x, y);
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
                    // Mix mode: patches are stepped without demag (exchange/anis/etc only).
                    s.b_add.set_uniform(0.0, 0.0, 0.0);
                }

                let active = p.active.as_slice();
                let m = &mut p.m;

                step_patch_rk4_recompute_field_masked_active(
                    m, active, params, mat, s, mask, geom_mask, self.relax,
                );
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
}
