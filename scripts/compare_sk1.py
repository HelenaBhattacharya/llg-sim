#!/usr/bin/env python3
"""
compare_sk1.py — Compare SK1 outputs (MuMax vs Rust)

Computes:
  - Skyrmion number Q for each snapshot
  - Radial profiles: <m_z(r)> and <m_perp(r)> (radial average around skyrmion core)

Typical use:
  python3 scripts/compare_sk1.py \
    --mumax mumax_outputs/st_problems/sk1/sk1_out \
    --rust  runs/st_problems/sk1/sk1_rust \
    --outdir plots/sk1_compare
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import discretisedfield as df


@dataclass
class SnapshotAnalysis:
    path: Path
    Q: float
    center_nm: Tuple[float, float]
    mz_core: float
    r_nm: np.ndarray
    mz_r: np.ndarray
    mperp_r: np.ndarray
    counts: np.ndarray


def load_slice_field(path: Path) -> df.Field:
    f = df.Field.from_file(str(path))
    # Thin film => z slice
    return f.sel("z")


def field_array_and_mesh(m: df.Field) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Returns:
      vec: (nx, ny, 3) numpy array
      dx, dy: cell sizes [m]
      lx, ly: sample size [m]
    """
    arr = np.asarray(m.array)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        arr = arr[:, :, 0, :]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected field array shape: {arr.shape}")

    dx, dy = float(m.mesh.cell[0]), float(m.mesh.cell[1])
    lx, ly = float(m.mesh.region.edges[0]), float(m.mesh.region.edges[1])
    return arr, dx, dy, lx, ly


def skyrmion_number(arr: np.ndarray, dx: float, dy: float) -> float:
    """
    Q = 1/(4π) ∫ m · (∂x m × ∂y m) dx dy

    Uses np.gradient (non-periodic boundaries).
    """
    mx = arr[:, :, 0]
    my = arr[:, :, 1]
    mz = arr[:, :, 2]

    dmx_dx, dmx_dy = np.gradient(mx, dx, dy, edge_order=2)
    dmy_dx, dmy_dy = np.gradient(my, dx, dy, edge_order=2)
    dmz_dx, dmz_dy = np.gradient(mz, dx, dy, edge_order=2)

    # (∂x m × ∂y m)
    cx = dmy_dx * dmz_dy - dmz_dx * dmy_dy
    cy = dmz_dx * dmx_dy - dmx_dx * dmz_dy
    cz = dmx_dx * dmy_dy - dmy_dx * dmx_dy

    density = mx * cx + my * cy + mz * cz
    Q = float((density / (4.0 * np.pi)).sum() * dx * dy)
    return Q


def find_core_center(arr: np.ndarray, dx: float, dy: float) -> Tuple[int, int, float]:
    """
    Core center via argmin(mz). Returns (ic, jc, mz_core).
    """
    mz = arr[:, :, 2]
    flat_idx = int(np.argmin(mz))
    ic_np, jc_np = np.unravel_index(flat_idx, mz.shape)
    ic = int(ic_np)
    jc = int(jc_np)
    return ic, jc, float(mz[ic, jc])


def radial_profile(
    arr: np.ndarray,
    dx: float,
    dy: float,
    lx: float,
    ly: float,
    nbins: int = 120,
    center_idx: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Radial averages around core center:
      <mz(r)> and <m_perp(r)>

    Returns:
      r_nm (bin centers),
      mz_r,
      mperp_r,
      counts,
      center_nm
    """
    nx, ny, _ = arr.shape
    mx = arr[:, :, 0]
    my = arr[:, :, 1]
    mz = arr[:, :, 2]
    mperp = np.sqrt(mx * mx + my * my)

    if center_idx is None:
        ic, jc, _ = find_core_center(arr, dx, dy)
    else:
        ic, jc = center_idx

    # coordinates of cell centers [m]
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    xc = xs[ic]
    yc = ys[jc]

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    R = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)

    # bins from 0 to half the smaller side
    r_max = 0.5 * min(lx, ly)
    edges = np.linspace(0.0, r_max, nbins + 1)
    idx = np.digitize(R.ravel(), edges) - 1
    idx = np.clip(idx, 0, nbins - 1)

    # accumulate
    mz_sum = np.bincount(idx, weights=mz.ravel(), minlength=nbins)
    mp_sum = np.bincount(idx, weights=mperp.ravel(), minlength=nbins)
    counts = np.bincount(idx, minlength=nbins).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        mz_r = mz_sum / counts
        mp_r = mp_sum / counts

    # bin centers [nm]
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    r_nm = r_centers * 1e9

    center_nm = (float(xc * 1e9), float(yc * 1e9))
    return r_nm, mz_r, mp_r, counts, center_nm


def analyse_snapshot(path: Path, nbins: int) -> SnapshotAnalysis:
    m = load_slice_field(path)
    arr, dx, dy, lx, ly = field_array_and_mesh(m)

    Q = skyrmion_number(arr, dx, dy)
    ic, jc, mz_core = find_core_center(arr, dx, dy)
    r_nm, mz_r, mp_r, counts, center_nm = radial_profile(
        arr, dx, dy, lx, ly, nbins=nbins, center_idx=(ic, jc)
    )

    return SnapshotAnalysis(
        path=path,
        Q=Q,
        center_nm=center_nm,
        mz_core=mz_core,
        r_nm=r_nm,
        mz_r=mz_r,
        mperp_r=mp_r,
        counts=counts,
    )


def resolve_snapshot_paths(dir_path: Path) -> Dict[str, Path]:
    """
    Expects:
      m000000.ovf (initial)
      m000001.ovf (relaxed)
    """
    p0 = dir_path / "m000000.ovf"
    p1 = dir_path / "m000001.ovf"
    if not p0.exists():
        raise FileNotFoundError(f"Missing {p0}")
    if not p1.exists():
        raise FileNotFoundError(f"Missing {p1}")
    return {"init": p0, "relaxed": p1}


def plot_profiles(
    outdir: Path,
    title: str,
    mumax_init: SnapshotAnalysis,
    mumax_rel: SnapshotAnalysis,
    rust_init: SnapshotAnalysis,
    rust_rel: SnapshotAnalysis,
):
    outdir.mkdir(parents=True, exist_ok=True)

    # mz(r)
    fig, ax = plt.subplots()
    ax.plot(mumax_init.r_nm, mumax_init.mz_r, label="MuMax init")
    ax.plot(mumax_rel.r_nm, mumax_rel.mz_r, label="MuMax relaxed")
    ax.plot(rust_init.r_nm, rust_init.mz_r, label="Rust init")
    ax.plot(rust_rel.r_nm, rust_rel.mz_r, label="Rust relaxed")
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("<mz(r)>")
    ax.set_title(f"{title} — radial mz")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "sk1_radial_mz.png", dpi=300)
    plt.close(fig)

    # m_perp(r)
    fig, ax = plt.subplots()
    ax.plot(mumax_init.r_nm, mumax_init.mperp_r, label="MuMax init")
    ax.plot(mumax_rel.r_nm, mumax_rel.mperp_r, label="MuMax relaxed")
    ax.plot(rust_init.r_nm, rust_init.mperp_r, label="Rust init")
    ax.plot(rust_rel.r_nm, rust_rel.mperp_r, label="Rust relaxed")
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("<m_perp(r)>")
    ax.set_title(f"{title} — radial in-plane magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "sk1_radial_mperp.png", dpi=300)
    plt.close(fig)


def print_summary(tag: str, a: SnapshotAnalysis):
    cx, cy = a.center_nm
    print(f"[{tag}] {a.path}")
    print(f"  Q         = {a.Q:+.6f}")
    print(f"  center_nm = ({cx:.2f}, {cy:.2f})")
    print(f"  mz_core   = {a.mz_core:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mumax", type=str, required=True, help="MuMax output dir containing m000000.ovf and m000001.ovf")
    ap.add_argument("--rust", type=str, required=True, help="Rust output dir containing m000000.ovf and m000001.ovf")
    ap.add_argument("--outdir", type=str, default="plots/sk1_compare", help="Output directory for comparison plots")
    ap.add_argument("--nbins", type=int, default=120, help="Number of radial bins")
    args = ap.parse_args()

    mumax_dir = Path(args.mumax)
    rust_dir = Path(args.rust)
    outdir = Path(args.outdir)

    mumax_paths = resolve_snapshot_paths(mumax_dir)
    rust_paths = resolve_snapshot_paths(rust_dir)

    mumax_init = analyse_snapshot(mumax_paths["init"], nbins=args.nbins)
    mumax_rel = analyse_snapshot(mumax_paths["relaxed"], nbins=args.nbins)
    rust_init = analyse_snapshot(rust_paths["init"], nbins=args.nbins)
    rust_rel = analyse_snapshot(rust_paths["relaxed"], nbins=args.nbins)

    print("==============================================================")
    print("SK1 comparison: MuMax vs Rust")
    print("==============================================================")
    print_summary("MuMax init", mumax_init)
    print_summary("MuMax relaxed", mumax_rel)
    print_summary("Rust init", rust_init)
    print_summary("Rust relaxed", rust_rel)
    print("--------------------------------------------------------------")
    print(f"ΔQ (relaxed) = {rust_rel.Q - mumax_rel.Q:+.6f}   (Rust - MuMax)")
    print("==============================================================")

    plot_profiles(
        outdir=outdir,
        title="SK1",
        mumax_init=mumax_init,
        mumax_rel=mumax_rel,
        rust_init=rust_init,
        rust_rel=rust_rel,
    )

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
