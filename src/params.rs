// src/params.rs
//
// SI-consistent parameter meanings.
//
// Conventions:
// - m is a unit vector (dimensionless).
// - B_eff and B_ext are in Tesla.
// - gamma is |gamma_e| in rad/(s*T).
// - dt is in seconds.

use std::f64::consts::PI;

pub type Vec3 = [f64; 3];

pub const MU0: f64 = 4.0 * PI * 1e-7; // T·m/A
pub const GAMMA_E_RAD_PER_S_T: f64 = 1.760_859_630_23e11; // rad/(s*T)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitKind {
    Uniform,
    Tilt,
    Bloch,
}

impl InitKind {
    pub fn from_arg(s: &str) -> Option<Self> {
        match s {
            "uniform" => Some(Self::Uniform),
            "tilt" => Some(Self::Tilt),
            "bloch" => Some(Self::Bloch),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Tilt => "tilt",
            Self::Bloch => "bloch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Toy,
    MuMaxLike,
}

impl Preset {
    pub fn from_arg(s: &str) -> Option<Self> {
        match s {
            "toy" => Some(Self::Toy),
            "mumax" | "si" | "mumaxlike" => Some(Self::MuMaxLike),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Toy => "toy",
            Self::MuMaxLike => "mumaxlike",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LLGParams {
    pub gamma: f64,
    pub alpha: f64,
    pub dt: f64,
    /// Uniform external induction B_ext in Tesla.
    pub b_ext: Vec3,
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub ms: f64,           // A/m
    pub a_ex: f64,         // J/m
    pub k_u: f64,          // J/m^3
    pub easy_axis: Vec3,
    pub dmi: Option<f64>,  // J/m^2 (interfacial DMI), None = OFF
}

#[derive(Debug, Clone, Copy)]
pub struct GridSpec {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl GridSpec {
    pub fn lx(&self) -> f64 {
        self.nx as f64 * self.dx
    }
    pub fn ly(&self) -> f64 {
        self.ny as f64 * self.dy
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RunSpec {
    pub n_steps: usize,
    pub save_every: usize,
    pub fps: u32,
    pub zoom_t_max: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    pub preset: Preset,
    pub init: InitKind,
    pub grid: GridSpec,
    pub llg: LLGParams,
    pub material: Material,
    pub run: RunSpec,
}

impl SimConfig {
    pub fn new(preset: Preset, init: InitKind) -> Self {
        match preset {
            Preset::Toy => Self::toy(init),
            Preset::MuMaxLike => Self::mumax_like(init),
        }
    }

    /// Non-physical numbers, but consistent *meanings*.
    /// External field is only applied for Tilt by default.
    pub fn toy(init: InitKind) -> Self {
        let grid = GridSpec {
            nx: 128,
            ny: 128,
            dx: 5e-12,
            dy: 5e-12,
            dz: 5e-12,
        };

        let b_ext = match init {
            InitKind::Tilt => [1.0, 0.0, 0.0],
            _ => [0.0, 0.0, 0.0],
        };

        let llg = LLGParams {
            gamma: 1.0,
            alpha: 0.1,
            dt: 0.0025,
            b_ext,
        };

        let material = Material {
            ms: 8.0e5,
            a_ex: 1.0,
            k_u: 0.1,
            easy_axis: normalise3([0.0, 0.0, 1.0]),
            dmi: None,
        };

        let run = RunSpec {
            n_steps: 500,
            save_every: 2,
            fps: 20,
            zoom_t_max: 0.1,
        };

        Self {
            preset: Preset::Toy,
            init,
            grid,
            llg,
            material,
            run,
        }
    }

    /// SI-ish starting point for MuMax comparisons (after validation).
    pub fn mumax_like(init: InitKind) -> Self {
        let grid = GridSpec {
            nx: 128,
            ny: 128,
            dx: 5e-9,
            dy: 5e-9,
            dz: 5e-9,
        };

        let b_ext = match init {
            InitKind::Tilt => [0.01, 0.0, 0.0], // 10 mT
            _ => [0.0, 0.0, 0.0],
        };

        let llg = LLGParams {
            gamma: GAMMA_E_RAD_PER_S_T,
            alpha: 0.02,
            dt: 1e-13,
            b_ext,
        };

        let material = Material {
            ms: 8.0e5,
            a_ex: 13e-12,
            k_u: 500.0,
            easy_axis: normalise3([0.0, 0.0, 1.0]),
            dmi: None,
        };

        let run = RunSpec {
            n_steps: 10_000,
            save_every: 200,
            fps: 20,
            zoom_t_max: 2e-9,
        };

        Self {
            preset: Preset::MuMaxLike,
            init,
            grid,
            llg,
            material,
            run,
        }
    }
}

fn normalise3(v: Vec3) -> Vec3 {
    let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if n2 == 0.0 {
        return [0.0, 0.0, 1.0];
    }
    let inv = 1.0 / n2.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}
