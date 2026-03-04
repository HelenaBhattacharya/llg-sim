// src/effective_field/mg_solver.rs
//
// 2D cell-centred geometric multigrid Poisson solver.
//
// Solves:  ∇²φ = f  on a rectangular 2D grid with Dirichlet boundary conditions.
//
// Grid convention:
//   The grid has dimensions (nx, ny).  The outermost ring of cells
//   (i=0, i=nx-1, j=0, j=ny-1) holds fixed Dirichlet values from `bc_phi`.
//   Interior cells (1..nx-2, 1..ny-2) are solved by the multigrid V-cycle.
//
// This is a pure PDE solver — no physics knowledge.  The Fredkin-Koehler
// decomposition and field extraction live in demag_poisson_mg.rs.

use super::mg_config::{
    CoarseOpKind, MGOperatorSettings, MGSmoother, PoissonMG2DConfig,
    ProlongationKind,
};
use super::mg_kernels;
use super::mg_stencil::Stencil2D;

use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Per-level data
// ---------------------------------------------------------------------------

struct MGLevel2D {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    inv_dx2: f64,
    inv_dy2: f64,

    stencil: Stencil2D,

    phi: Vec<f64>,
    rhs: Vec<f64>,
    res: Vec<f64>,
    tmp: Vec<f64>,
    bc_phi: Vec<f64>,
}

impl MGLevel2D {
    fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        let n = nx * ny;
        Self {
            nx,
            ny,
            dx,
            dy,
            inv_dx2: 1.0 / (dx * dx),
            inv_dy2: 1.0 / (dy * dy),
            stencil: Stencil2D::five_point(dx, dy),
            phi: vec![0.0; n],
            rhs: vec![0.0; n],
            res: vec![0.0; n],
            tmp: vec![0.0; n],
            bc_phi: vec![0.0; n],
        }
    }

    fn enforce_dirichlet(&mut self) {
        mg_kernels::stamp_dirichlet_bc_2d(&mut self.phi, &self.bc_phi, self.nx, self.ny);
    }
}

// ---------------------------------------------------------------------------
// Public solver struct
// ---------------------------------------------------------------------------

/// 2D cell-centred multigrid Poisson solver.
///
/// Construct with `new()`, then call `solve()` or `solve_with_rhs_and_bc()`.
/// The solution is stored in the finest-level phi and can be read with `phi()`.
pub struct PoissonMG2D {
    cfg: PoissonMG2DConfig,
    op: MGOperatorSettings,
    levels: Vec<MGLevel2D>,
}

impl PoissonMG2D {
    /// Create a new solver for a grid of size (nx, ny) with cell spacing (dx, dy).
    ///
    /// The grid must have at least 4 cells in each direction (outermost ring is BC).
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64, cfg: PoissonMG2DConfig) -> Self {
        let op = MGOperatorSettings::from_env();
        Self::new_with_op(nx, ny, dx, dy, cfg, op)
    }

    fn new_with_op(
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        mut cfg: PoissonMG2DConfig,
        op: MGOperatorSettings,
    ) -> Self {
        cfg.sanitize_for_op(&op);

        let mut levels: Vec<MGLevel2D> = Vec::new();
        let mut lx = nx;
        let mut ly = ny;
        let mut ldx = dx;
        let mut ldy = dy;

        // Finest level
        levels.push(MGLevel2D::new(lx, ly, ldx, ldy));

        // Coarsen until grid is too small or non-uniform coarsening would be needed.
        //
        // BUG FIX: Previously used `!can_x && !can_y` which allowed semi-coarsening
        // (only one dimension halves).  This produces rx/ry ratios of (2,1) or (1,2)
        // between levels, causing two problems:
        //   1. Galerkin coarsening always assumes r=2 uniform, so the computed stencil
        //      doesn't match the actual transfer operators.
        //   2. Even with rediscretization, the anisotropic grid (dx ≠ dy) at coarse
        //      levels combined with mismatched transfer can cause divergence.
        //
        // Requiring both dimensions to coarsen simultaneously (uniform 2:1) is
        // conservative but correct.  Proper semi-coarsening support is a future task.
        loop {
            let can_x = lx >= 8 && lx % 2 == 0;
            let can_y = ly >= 8 && ly % 2 == 0;

            if !can_x || !can_y {
                break;
            }

            lx /= 2;
            ldx *= 2.0;
            ly /= 2;
            ldy *= 2.0;

            levels.push(MGLevel2D::new(lx, ly, ldx, ldy));

            if levels.len() > 20 {
                break;
            }
        }

        // Assign stencils per level
        if !levels.is_empty() {
            levels[0].stencil = Stencil2D::from_kind(op.stencil, levels[0].dx, levels[0].dy);

            for l in 1..levels.len() {
                levels[l].stencil = match op.coarse_op {
                    CoarseOpKind::Rediscretize => {
                        Stencil2D::from_kind(op.stencil, levels[l].dx, levels[l].dy)
                    }
                    CoarseOpKind::Galerkin => {
                        let g = Stencil2D::galerkin_coarsen(&levels[l - 1].stencil, op.prolong);
                        if g.diag <= 0.0 || g.offs.is_empty() {
                            // Galerkin failed — fall back to rediscretization
                            eprintln!(
                                "[mg_solver2d] Falling back to rediscretization at level {} \
                                 ({}x{}, dx={:.2e} dy={:.2e})",
                                l, levels[l].nx, levels[l].ny, levels[l].dx, levels[l].dy,
                            );
                            Stencil2D::from_kind(op.stencil, levels[l].dx, levels[l].dy)
                        } else {
                            g
                        }
                    }
                };
            }
        }

        static LOG_ONCE: OnceLock<()> = OnceLock::new();
        LOG_ONCE.get_or_init(|| {
            eprintln!(
                "[mg_solver2d] hierarchy: {} levels, finest={}x{}, coarsest={}x{}, \
                 stencil={:?}, prolong={:?}, coarse_op={:?}",
                levels.len(),
                levels[0].nx, levels[0].ny,
                levels.last().map(|l| l.nx).unwrap_or(0),
                levels.last().map(|l| l.ny).unwrap_or(0),
                op.stencil, op.prolong, op.coarse_op,
            );
        });

        Self { cfg, op, levels }
    }

    // -----------------------------------------------------------------------
    // Public accessors
    // -----------------------------------------------------------------------

    /// Read access to the finest-level solution phi.
    pub fn phi(&self) -> &[f64] {
        &self.levels[0].phi
    }

    /// Write access to the finest-level solution phi (for warm-start seeding).
    pub fn phi_mut(&mut self) -> &mut [f64] {
        &mut self.levels[0].phi
    }

    /// Finest grid dimensions.
    pub fn dims(&self) -> (usize, usize) {
        (self.levels[0].nx, self.levels[0].ny)
    }

    /// Grid spacing at finest level.
    pub fn spacing(&self) -> (f64, f64) {
        (self.levels[0].dx, self.levels[0].dy)
    }

    // -----------------------------------------------------------------------
    // Solve
    // -----------------------------------------------------------------------

    /// Set RHS and boundary conditions, then solve.
    ///
    /// `rhs`: right-hand side array of size nx*ny.
    /// `bc_phi`: Dirichlet boundary values, same size. Interior values are ignored.
    ///
    /// If `warm_start` is enabled in config and phi already contains a reasonable
    /// guess, the solver uses it; otherwise phi is zeroed.
    pub fn solve_with_rhs_and_bc(&mut self, rhs: &[f64], bc_phi: &[f64]) {
        let n = self.levels[0].nx * self.levels[0].ny;
        debug_assert_eq!(rhs.len(), n);
        debug_assert_eq!(bc_phi.len(), n);

        self.levels[0].rhs.copy_from_slice(rhs);
        self.levels[0].bc_phi.copy_from_slice(bc_phi);

        if !self.cfg.warm_start {
            self.levels[0].phi.fill(0.0);
        }

        self.levels[0].enforce_dirichlet();
        self.run_vcycles();
    }

    /// Re-solve using the current RHS and BC already stored in the finest level.
    /// Useful for warm-started iterative refinement.
    pub fn solve(&mut self) {
        self.levels[0].enforce_dirichlet();
        self.run_vcycles();
    }

    /// Apply updated config (e.g. from env-var hot-reload).
    pub fn apply_cfg(&mut self, mut cfg: PoissonMG2DConfig) {
        cfg.sanitize_for_op(&self.op);
        self.cfg = cfg;
    }

    // -----------------------------------------------------------------------
    // V-cycle engine
    // -----------------------------------------------------------------------

    fn run_vcycles(&mut self) {
        let use_tol = self.cfg.tol_abs.is_some() || self.cfg.tol_rel.is_some();
        let max_cycles = if use_tol { self.cfg.v_cycles_max } else { self.cfg.v_cycles };

        let rhs_norm = if use_tol {
            self.levels[0].rhs.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
        } else {
            0.0
        };

        for _cycle in 0..max_cycles {
            self.vcycle(0);

            if use_tol {
                let l = &mut self.levels[0];
                let res_norm = mg_kernels::compute_residual_2d(
                    &l.phi, &l.rhs, &mut l.res, l.nx, l.ny, &l.stencil,
                );

                if let Some(tol) = self.cfg.tol_abs {
                    if res_norm <= tol {
                        break;
                    }
                }
                if let Some(tol) = self.cfg.tol_rel {
                    if res_norm <= tol * rhs_norm.max(1e-30) {
                        break;
                    }
                }
            }
        }
    }

    fn vcycle(&mut self, level: usize) {
        let n_levels = self.levels.len();
        if level + 1 == n_levels {
            // Coarsest level: smooth heavily as direct solve approximation
            self.smooth(level, self.cfg.pre_smooth + self.cfg.post_smooth + 4);
            return;
        }

        // Pre-smooth
        self.smooth(level, self.cfg.pre_smooth);

        // Compute residual
        {
            let l = &mut self.levels[level];
            mg_kernels::compute_residual_2d(
                &l.phi, &l.rhs, &mut l.res, l.nx, l.ny, &l.stencil,
            );
        }

        // Restrict residual to coarse level
        {
            let (fine_slice, coarse_slice) = self.levels.split_at_mut(level + 1);
            let fine = &fine_slice[level];
            let coarse = &mut coarse_slice[0];

            mg_kernels::restrict_residual_2d(
                &fine.res, fine.nx, fine.ny,
                &mut coarse.rhs, &mut coarse.phi,
                coarse.nx, coarse.ny,
            );
        }

        // Recurse
        self.vcycle(level + 1);

        // Prolongate correction and add to current level
        {
            let (fine_slice, coarse_slice) = self.levels.split_at_mut(level + 1);
            let fine = &mut fine_slice[level];
            let coarse = &coarse_slice[0];

            let bilinear = self.op.prolong == ProlongationKind::Bilinear;
            mg_kernels::prolongate_add_2d(
                &coarse.phi, coarse.nx, coarse.ny,
                &mut fine.phi, fine.nx, fine.ny,
                bilinear,
            );
        }

        // Re-enforce BCs after prolongation
        self.levels[level].enforce_dirichlet();

        // Post-smooth
        self.smooth(level, self.cfg.post_smooth);
    }

    fn smooth(&mut self, level: usize, iters: usize) {
        if iters == 0 {
            return;
        }
        let l = &mut self.levels[level];
        match self.cfg.smoother {
            MGSmoother::WeightedJacobi => {
                mg_kernels::smooth_weighted_jacobi_2d(
                    &mut l.phi, &mut l.tmp, &l.rhs, &l.bc_phi,
                    l.nx, l.ny, &l.stencil, iters, self.cfg.omega,
                );
            }
            MGSmoother::RedBlackSOR => {
                mg_kernels::smooth_rb_sor_2d(
                    &mut l.phi, &mut l.tmp, &l.rhs, &l.bc_phi,
                    l.nx, l.ny, l.inv_dx2, l.inv_dy2,
                    iters, self.cfg.sor_omega,
                );
            }
        }
    }
}