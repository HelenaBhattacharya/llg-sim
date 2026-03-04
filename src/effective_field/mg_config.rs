// src/effective_field/mg_config.rs
//
// Configuration for the 2D multigrid Poisson solver.
//
// This replaces the old 3D padded-box configuration. There are no padding
// parameters, no treecode parameters, and no hybrid/PPPM parameters. The
// open-boundary physics is handled by the boundary integral in the Fredkin-
// Koehler decomposition (see demag_poisson_mg.rs), not by solver-level BCs.

use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Smoother selection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MGSmoother {
    /// Weighted Jacobi (parallel-friendly, supports general stencils).
    WeightedJacobi,
    /// Red-black Gauss-Seidel with SOR (only valid for 5-point stencil).
    RedBlackSOR,
}

impl MGSmoother {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "jacobi" | "wj" | "weighted_jacobi" => Some(Self::WeightedJacobi),
            "rbgs" | "redblack" | "red_black" | "red_black_sor" | "sor" => Some(Self::RedBlackSOR),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Stencil / prolongation / coarse-operator selection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaplacianStencilKind {
    /// Standard 5-point stencil (axis-aligned neighbours).
    FivePoint,
    /// Isotropic 9-point Mehrstellen stencil (requires dx ~ dy).
    Iso9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProlongationKind {
    /// Piecewise-constant injection.
    Injection,
    /// Cell-centred bilinear interpolation.
    Bilinear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoarseOpKind {
    /// Rediscretize the Laplacian at the coarse grid spacing.
    Rediscretize,
    /// Galerkin coarsening: R * A * P.
    Galerkin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MGOperatorSettings {
    pub stencil: LaplacianStencilKind,
    pub prolong: ProlongationKind,
    pub coarse_op: CoarseOpKind,
}

pub(crate) fn env_str_lower(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
}

impl MGOperatorSettings {
    pub fn from_env() -> Self {
        let stencil = match env_str_lower("LLG_DEMAG_MG_STENCIL").as_deref() {
            Some("5") | Some("5pt") | Some("five") | Some("fivepoint") => {
                LaplacianStencilKind::FivePoint
            }
            // Accept old 3D names as aliases for the 2D equivalents.
            Some("7") | Some("7pt") | Some("seven") | Some("sevenpoint") => {
                LaplacianStencilKind::FivePoint
            }
            Some("iso9") | Some("9") | Some("9pt") | Some("mehrstellen9") => {
                LaplacianStencilKind::Iso9
            }
            // iso27 was a 3D stencil; map to iso9 in 2D.
            Some("iso27") | Some("27") | Some("27pt") => LaplacianStencilKind::Iso9,
            Some(other) => {
                eprintln!(
                    "[demag_mg2d] WARNING: unknown LLG_DEMAG_MG_STENCIL='{}' -> using 'iso9'",
                    other
                );
                LaplacianStencilKind::Iso9
            }
            None => LaplacianStencilKind::Iso9,
        };

        let prolong = match env_str_lower("LLG_DEMAG_MG_PROLONG").as_deref() {
            Some("inject") | Some("injection") | Some("pc") => ProlongationKind::Injection,
            Some("bilinear") | Some("linear") | Some("trilinear") | Some("tl") => {
                ProlongationKind::Bilinear
            }
            Some(other) => {
                eprintln!(
                    "[demag_mg2d] WARNING: unknown LLG_DEMAG_MG_PROLONG='{}' -> using 'bilinear'",
                    other
                );
                ProlongationKind::Bilinear
            }
            None => ProlongationKind::Bilinear,
        };

        let coarse_op = match env_str_lower("LLG_DEMAG_MG_COARSE_OP").as_deref() {
            Some("rediscretize") | Some("re") | Some("rd") => CoarseOpKind::Rediscretize,
            Some("galerkin") | Some("g") => CoarseOpKind::Galerkin,
            Some(other) => {
                eprintln!(
                    "[demag_mg2d] WARNING: unknown LLG_DEMAG_MG_COARSE_OP='{}' -> using 'galerkin'",
                    other
                );
                CoarseOpKind::Galerkin
            }
            None => CoarseOpKind::Galerkin,
        };

        Self { stencil, prolong, coarse_op }
    }
}

// ---------------------------------------------------------------------------
// Main solver configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct PoissonMG2DConfig {
    /// Fixed V-cycle count (used when tol_abs/tol_rel are None).
    pub v_cycles: usize,
    /// Maximum V-cycles when using tolerance-based stopping.
    pub v_cycles_max: usize,
    /// Stop when max-norm residual <= tol_abs.
    pub tol_abs: Option<f64>,
    /// Stop when max-norm residual <= tol_rel * max-norm(rhs).
    pub tol_rel: Option<f64>,
    /// Pre-smoothing iterations per V-cycle.
    pub pre_smooth: usize,
    /// Post-smoothing iterations per V-cycle.
    pub post_smooth: usize,
    /// Smoother algorithm.
    pub smoother: MGSmoother,
    /// Weighted Jacobi relaxation parameter (0 < omega <= 1).
    pub omega: f64,
    /// Red-black SOR relaxation factor (0 < omega < 2).
    pub sor_omega: f64,
    /// Reuse phi from previous solve as initial guess.
    pub warm_start: bool,
}

impl Default for PoissonMG2DConfig {
    fn default() -> Self {
        Self {
            v_cycles: 10,
            v_cycles_max: 60,
            tol_abs: None,
            tol_rel: Some(1e-6),
            pre_smooth: 2,
            post_smooth: 2,
            smoother: MGSmoother::WeightedJacobi,
            omega: 2.0 / 3.0,
            sor_omega: 1.5,
            warm_start: true,
        }
    }
}

impl PoissonMG2DConfig {
    pub fn from_env() -> Self {
        fn get_usize(name: &str) -> Option<usize> {
            std::env::var(name).ok().and_then(|s| s.trim().parse::<usize>().ok())
        }
        fn get_f64(name: &str) -> Option<f64> {
            std::env::var(name).ok().and_then(|s| s.trim().parse::<f64>().ok())
        }

        let mut cfg = Self::default();

        if let Some(v) = get_usize("LLG_DEMAG_MG_VCYCLES")     { cfg.v_cycles = v.max(1); }
        if let Some(v) = get_usize("LLG_DEMAG_MG_VCYCLES_MAX") { cfg.v_cycles_max = v.max(1); }
        if let Some(v) = get_f64("LLG_DEMAG_MG_TOL_ABS")       { cfg.tol_abs = Some(v.max(0.0)); }
        if let Some(v) = get_f64("LLG_DEMAG_MG_TOL_REL")       { cfg.tol_rel = Some(v.max(0.0).min(1.0)); }
        if let Some(v) = get_usize("LLG_DEMAG_MG_PRE_SMOOTH")  { cfg.pre_smooth = v.max(1); }
        if let Some(v) = get_usize("LLG_DEMAG_MG_POST_SMOOTH") { cfg.post_smooth = v.max(1); }
        if let Some(s) = env_str_lower("LLG_DEMAG_MG_SMOOTHER") {
            if let Some(sm) = MGSmoother::from_str(&s) { cfg.smoother = sm; }
        }
        if let Some(v) = get_f64("LLG_DEMAG_MG_OMEGA")     { cfg.omega = v.clamp(0.01, 1.0); }
        if let Some(v) = get_f64("LLG_DEMAG_MG_SOR_OMEGA") { cfg.sor_omega = v.clamp(0.01, 1.99); }
        if let Ok(v) = std::env::var("LLG_DEMAG_MG_WARM_START") {
            cfg.warm_start = matches!(v.as_str(), "1" | "true" | "yes" | "on");
        }

        cfg
    }

    /// Sanitize smoother choice for the active operator settings.
    /// Red-black SOR only works with 5-point stencil + rediscretize.
    pub fn sanitize_for_op(&mut self, op: &MGOperatorSettings) {
        let rb_ok = op.stencil == LaplacianStencilKind::FivePoint
            && op.coarse_op == CoarseOpKind::Rediscretize;
        if self.smoother == MGSmoother::RedBlackSOR && !rb_ok {
            static ONCE: OnceLock<()> = OnceLock::new();
            ONCE.get_or_init(|| {
                eprintln!(
                    "[demag_mg2d] INFO: overriding smoother RedBlackSOR -> WeightedJacobi \
                     (stencil/coarse-op requires it)."
                );
            });
            self.smoother = MGSmoother::WeightedJacobi;
        }
    }
}