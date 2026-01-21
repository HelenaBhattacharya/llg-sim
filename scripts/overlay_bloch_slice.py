import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Tuple


# =================================================
# Load Rust Bloch slice
# =================================================
def load_rust_slice(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Rust Bloch slice CSV.

    Expected columns:
      x, mx, mz
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    x = data[:, 0]
    mx = data[:, 1]
    mz = data[:, 2]
    return x, mx, mz


# =================================================
# Load MuMax CSV slice (from mumax3-convert -csv)
# =================================================
def load_mumax_slice_csv(path: str, nx: int, ny: int, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MuMax CSV produced by `mumax3-convert -csv`.

    CSV format:
      i, j, k, value

    where:
      i = x-index
      j = y-index
      k = z-index (always 0 here)

    Returns:
      x (physical coordinate), value along mid-row j = ny//2
    """
    field = np.zeros((ny, nx))

    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            try:
                i = int(row[0])
                j = int(row[1])
                val = float(row[-1])
            except ValueError:
                continue

            if 0 <= i < nx and 0 <= j < ny:
                field[j, i] = val

    jmid = ny // 2
    x = (np.arange(nx) + 0.5) * dx
    v = field[jmid, :]

    return x, v


# =================================================
# Analytic Bloch wall (exchange + uniaxial anisotropy)
# =================================================
def analytic_bloch_wall(x: np.ndarray, A: float, Ku: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytic Bloch wall profile:

      mz(x) = tanh(x / Delta)
      mx(x) = sech(x / Delta)

    with Delta = sqrt(A / Ku)
    """
    Delta = np.sqrt(A / Ku)
    mz = np.tanh(x / Delta)
    mx = 1.0 / np.cosh(x / Delta)
    return mx, mz


# =================================================
# Utilities
# =================================================
def center_x(x: np.ndarray) -> np.ndarray:
    """Center x-axis so that x = 0 is at the wall centre."""
    return x - np.mean(x)


# =================================================
# CONFIG (must match simulations)
# =================================================
tag = "final"       # t0, t5ns, t10ns, final
nx = 256
ny = 64
dx = 5e-9           # cell size in x (meters)

# Material parameters (must match Rust & MuMax)
Aex = 13e-12        # J/m
Ku1 = 500.0         # J/m^3


# =================================================
# Load data
# =================================================
# Rust
xr, mxr, mzr = load_rust_slice(
    f"out/bloch_slices/rust_slice_{tag}.csv"
)

# MuMax
xm_z, mzm = load_mumax_slice_csv(
    f"mumax_outputs/bloch_relax/mz_{tag}.csv", nx, ny, dx
)
xm_x, mxm = load_mumax_slice_csv(
    f"mumax_outputs/bloch_relax/mx_{tag}.csv", nx, ny, dx
)

# Center all x-axes
xr = center_x(xr)
xm_z = center_x(xm_z)
xm_x = center_x(xm_x)

# Analytic solution on same x grid
mx_th, mz_th = analytic_bloch_wall(xr, Aex, Ku1)


# =================================================
# Plot m_z(x)
# =================================================
plt.figure(figsize=(6, 4))
plt.plot(xr * 1e9, mzr, label="Rust m_z")
plt.plot(xm_z * 1e9, mzm, "--", label="MuMax m_z")
plt.plot(xr * 1e9, mz_th, ":", label="Analytic m_z")
plt.xlabel("x − x₀ (nm)")
plt.ylabel("m_z")
plt.title(f"Bloch wall m_z(x) ({tag})")
plt.legend()
plt.tight_layout()
plt.show()


# =================================================
# Plot m_x(x)
# =================================================
plt.figure(figsize=(6, 4))
plt.plot(xr * 1e9, mxr, label="Rust m_x")
plt.plot(xm_x * 1e9, mxm, "--", label="MuMax m_x")
plt.plot(xr * 1e9, mx_th, ":", label="Analytic m_x")
plt.xlabel("x − x₀ (nm)")
plt.ylabel("m_x")
plt.title(f"Bloch wall m_x(x) ({tag})")
plt.legend()
plt.tight_layout()
plt.show()