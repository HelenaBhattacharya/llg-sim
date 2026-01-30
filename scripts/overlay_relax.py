import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Relaxation validation: Rust vs MuMax (final state only)
# ============================================================

def load_rust_slice(path: Path):
    """
    Rust CSV formats supported:
      (A) x,mx,my,mz
      (B) x,my,mz        (mx assumed 0)
    """
    with open(path, "r") as f:
        header = f.readline().strip()
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    cols = [h.strip() for h in header.split(",")]

    if cols == ["x", "mx", "my", "mz"] and data.shape[1] == 4:
        x, mx, my, mz = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        return x, mx, my, mz

    if cols == ["x", "my", "mz"] and data.shape[1] == 3:
        x, my, mz = data[:, 0], data[:, 1], data[:, 2]
        mx = np.zeros_like(my)
        return x, mx, my, mz

    raise ValueError(f"Unexpected Rust slice header/shape in {path}: header={cols}, shape={data.shape}")


def load_mumax_m_from_csv(path: Path, nx: int, ny: int):
    """
    Supports MuMax CSV styles:

    1) Stacked dense blocks:
         shape = (3*ny, nx)  -> split into mx,my,mz blocks

    2) Cell list:
         shape = (N, 6+) with columns i,j,k,mx,my,mz

    Returns full-field arrays: mx[ny,nx], my[ny,nx], mz[ny,nx], fmt_label
    """
    data = np.loadtxt(path, delimiter=",", comments="#")

    # Case 1: stacked blocks
    if data.ndim == 2 and data.shape == (3 * ny, nx):
        mx = data[0:ny, :]
        my = data[ny:2*ny, :]
        mz = data[2*ny:3*ny, :]
        return mx, my, mz, "stacked_3ny_by_nx"

    # Case 2: list-of-cells with mx,my,mz
    if data.ndim == 2 and data.shape[1] >= 6:
        i = data[:, 0].astype(int)
        j = data[:, 1].astype(int)
        mxv = data[:, 3]
        myv = data[:, 4]
        mzv = data[:, 5]

        mx = np.zeros((ny, nx))
        my = np.zeros((ny, nx))
        mz = np.zeros((ny, nx))

        for ii, jj, a, b, c in zip(i, j, mxv, myv, mzv):
            if 0 <= ii < nx and 0 <= jj < ny:
                mx[jj, ii] = a
                my[jj, ii] = b
                mz[jj, ii] = c
        return mx, my, mz, "cell_list_ijk_mxyz"

    if data.ndim == 2 and data.shape == (ny, nx):
        raise ValueError(
            f"{path} looks like a single scalar grid (ny,nx). "
            "For relaxation validation you must export vector m."
        )

    raise ValueError(f"Unrecognized MuMax CSV format: {path} shape={data.shape}")


def midrow_slice(field_2d: np.ndarray):
    return field_2d[field_2d.shape[0] // 2, :]


def linf(a, b):
    return float(np.max(np.abs(a - b)))


def l2_rms(a, b):
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


# ============================================================
# CONFIG
# ============================================================

NX = 64
NY = 64
DX = 5e-9

RUST_DIR  = Path("out/relax_uniform_noisy")
MUMAX_DIR = Path("mumax_outputs/relax_uniform_noisy")

rust_path  = RUST_DIR / "rust_slice_final.csv"
mumax_path = MUMAX_DIR / "m_final.csv"

out_dir = RUST_DIR
out_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load data
# ============================================================

xr, mxr, myr, mzr = load_rust_slice(rust_path)

mx2d, my2d, mz2d, fmt = load_mumax_m_from_csv(mumax_path, NX, NY)
mxm = midrow_slice(mx2d)
mym = midrow_slice(my2d)
mzm = midrow_slice(mz2d)

xm = (np.arange(NX) + 0.5) * DX

# Use physical x axis (nm) for both; no centering here
x_nm_r = xr * 1e9
x_nm_m = xm * 1e9

# Derived
mperp_r = np.sqrt(mxr**2 + myr**2)
mperp_m = np.sqrt(mxm**2 + mym**2)

# ============================================================
# Sanity prints
# ============================================================

print("\n=== Relaxation validation (final state) ===")
print(f"MuMax CSV format detected: {fmt}")

# Global MuMax stats (field)
mz_mean = float(np.mean(mz2d))
mperp_mean = float(np.mean(np.sqrt(mx2d**2 + my2d**2)))
mnorm_mean = float(np.mean(np.sqrt(mx2d**2 + my2d**2 + mz2d**2)))

print(f"MuMax global <mz>         = {mz_mean:+.6f}")
print(f"MuMax global <|m_perp|>   = {mperp_mean:.6e}")
print(f"MuMax global <|m|>        = {mnorm_mean:.6f} (should be ~1)")

print(f"MuMax mid-row mx min/max: {mxm.min():+.3e} / {mxm.max():+.3e}")
print(f"MuMax mid-row my min/max: {mym.min():+.3e} / {mym.max():+.3e}")
print(f"MuMax mid-row mz min/max: {mzm.min():+.6f} / {mzm.max():+.6f}")

if mperp_mean > 1e-2:
    print("WARNING: MuMax does not look fully relaxed (mean |m_perp| > 1e-2). "
          "If you just switched to Relax(), rerun and reconvert/copy back.")


# ============================================================
# Error metrics (slice)
# ============================================================

print("\nSlice errors (Rust vs MuMax):")
print(f"  L_inf mx  = {linf(mxr, mxm):.3e}   RMS = {l2_rms(mxr, mxm):.3e}")
print(f"  L_inf my  = {linf(myr, mym):.3e}   RMS = {l2_rms(myr, mym):.3e}")
print(f"  L_inf mz  = {linf(mzr, mzm):.3e}   RMS = {l2_rms(mzr, mzm):.3e}")
print(f"  L_inf |m_perp| = {linf(mperp_r, mperp_m):.3e}   RMS = {l2_rms(mperp_r, mperp_m):.3e}")


# ============================================================
# Plots (best for comparison)
# ============================================================

# 1) mx & my overlay (zoom-friendly; mz omitted)
plt.figure(figsize=(7.5, 4.5))
plt.plot(x_nm_r, mxr, label="Rust mx", linewidth=1.8)
plt.plot(x_nm_m, mxm, "--", label="MuMax mx", linewidth=1.8)
plt.plot(x_nm_r, myr, label="Rust my", linewidth=1.8)
plt.plot(x_nm_m, mym, "--", label="MuMax my", linewidth=1.8)
plt.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
plt.xlabel("x (nm)")
plt.ylabel("magnetisation")
plt.title("Relaxation validation: transverse components (mid-row)")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
p_xy = out_dir / "relax_uniform_overlay_xy.png"
plt.savefig(p_xy, dpi=200)
plt.close()

# 2) mz deviation from +1 (best way to see small differences)
plt.figure(figsize=(7.5, 4.5))
plt.plot(x_nm_r, 1.0 - mzr, label="Rust (1 - mz)", linewidth=2.0)
plt.plot(x_nm_m, 1.0 - mzm, "--", label="MuMax (1 - mz)", linewidth=2.0)
plt.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
plt.xlabel("x (nm)")
plt.ylabel("1 - mz")
plt.title("Relaxation validation: longitudinal deviation from +z")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
p_mzdev = out_dir / "relax_uniform_overlay_mzdev.png"
plt.savefig(p_mzdev, dpi=200)
plt.close()

# 3) Delta plot (most diagnostic)
plt.figure(figsize=(7.5, 4.5))
plt.plot(x_nm_r, mxr - mxm, label="Δmx", linewidth=1.8)
plt.plot(x_nm_r, myr - mym, label="Δmy", linewidth=1.8)
plt.plot(x_nm_r, mzr - mzm, label="Δmz", linewidth=1.8)
plt.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
plt.xlabel("x (nm)")
plt.ylabel("Rust − MuMax")
plt.title("Relaxation validation: mid-row differences")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
p_delta = out_dir / "relax_uniform_overlay_deltas.png"
plt.savefig(p_delta, dpi=200)
plt.close()

# 4) |m_perp| overlay
plt.figure(figsize=(7.5, 4.5))
plt.plot(x_nm_r, mperp_r, label="Rust |m_perp|", linewidth=2.0)
plt.plot(x_nm_m, mperp_m, "--", label="MuMax |m_perp|", linewidth=2.0)
plt.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
plt.xlabel("x (nm)")
plt.ylabel("|m_perp|")
plt.title("Relaxation validation: transverse magnitude (mid-row)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
p_mperp = out_dir / "relax_uniform_overlay_mperp.png"
plt.savefig(p_mperp, dpi=200)
plt.close()

print("\nWrote plots:")
print(f"  {p_xy}")
print(f"  {p_mzdev}")
print(f"  {p_delta}")
print(f"  {p_mperp}")