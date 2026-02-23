#!/usr/bin/env python3
"""
mask_xor_ring.py

B) Compute XOR ring between:
  - fine disk mask (circle test on fine cell centers), and
  - upsampled coarse disk mask (circle test on coarse cell centers, replicated to fine)

Also checks whether the max-|Δm| location lies inside that XOR region.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def load_field_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    i = data["i"].astype(np.int64)
    j = data["j"].astype(np.int64)
    x = data["x_m"].astype(np.float64)
    y = data["y_m"].astype(np.float64)
    m = np.stack([data["mx"], data["my"], data["mz"]], axis=1).astype(np.float64)
    return i, j, x, y, m


def infer_grid(i: np.ndarray, j: np.ndarray) -> tuple[int, int]:
    return int(i.max() + 1), int(j.max() + 1)


def make_circle_mask(nx: int, ny: int, dx: float, dy: float, R: float, cx: float = 0.0, cy: float = 0.0):
    """
    Circle mask evaluated at cell centers in *centered* coordinates.
    Returns (ny, nx) boolean.
    """
    Lx = nx * dx
    Ly = ny * dy
    xs = (np.arange(nx) + 0.5) * dx - 0.5 * Lx
    ys = (np.arange(ny) + 0.5) * dy - 0.5 * Ly
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    rr2 = (X - cx) ** 2 + (Y - cy) ** 2
    return rr2 <= (R * R)


def upsample_mask(mask_coarse: np.ndarray, ratio: int) -> np.ndarray:
    """
    Replicate each coarse cell to a ratio×ratio block.
    mask_coarse: (nyc, nxc) bool
    returns: (nyc*ratio, nxc*ratio) bool
    """
    return np.repeat(np.repeat(mask_coarse, ratio, axis=0), ratio, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/amr_masked_disk_relax"))
    ap.add_argument("--disk-radius", type=float, required=True)
    ap.add_argument("--ratio", type=int, default=2)
    ap.add_argument("--coarse-nx", type=int, default=192)
    ap.add_argument("--coarse-ny", type=int, default=192)
    ap.add_argument("--coarse-dx", type=float, default=5e-9)
    ap.add_argument("--coarse-dy", type=float, default=5e-9)
    ap.add_argument("--center-x", type=float, default=0.0)
    ap.add_argument("--center-y", type=float, default=0.0)
    args = ap.parse_args()

    amr_path = args.out / "amr_fine_final.csv"
    uni_path = args.out / "uniform_fine_final.csv"

    ia, ja, xa, ya, ma = load_field_csv(amr_path)
    ib, jb, xb, yb, mb = load_field_csv(uni_path)

    nx_f, ny_f = infer_grid(ia, ja)
    dx_f = args.coarse_dx / args.ratio
    dy_f = args.coarse_dy / args.ratio

    # Masks
    fine_circle = make_circle_mask(nx_f, ny_f, dx_f, dy_f, args.disk_radius, args.center_x, args.center_y)
    coarse_circle = make_circle_mask(args.coarse_nx, args.coarse_ny, args.coarse_dx, args.coarse_dy, args.disk_radius, args.center_x, args.center_y)
    coarse_up = upsample_mask(coarse_circle, args.ratio)

    if coarse_up.shape != fine_circle.shape:
        raise RuntimeError(f"Shape mismatch: coarse_up {coarse_up.shape} vs fine_circle {fine_circle.shape}")

    xor = np.logical_xor(fine_circle, coarse_up)
    inter = np.logical_and(fine_circle, coarse_up)

    # Compute max-|Δm| location
    dm = np.linalg.norm(ma - mb, axis=1)
    k = int(np.argmax(dm))
    i_max = int(ia[k])
    j_max = int(ja[k])
    in_xor = bool(xor[j_max, i_max])      # note [j,i] indexing for (ny,nx)
    in_inter = bool(inter[j_max, i_max])

    print("=== B) Mask disagreement ring ===")
    print(f"fine grid: nx={nx_f} ny={ny_f}  dx={dx_f:.3e} dy={dy_f:.3e}")
    print(f"coarse grid: nx={args.coarse_nx} ny={args.coarse_ny}  dx={args.coarse_dx:.3e} dy={args.coarse_dy:.3e}")
    print(f"ratio: {args.ratio}")
    print(f"disk radius: {args.disk_radius:.6e}")
    print(f"xor cell count: {int(xor.sum())}  ({xor.mean()*100:.3f}% of fine cells)")
    print(f"intersection cell count: {int(inter.sum())}  ({inter.mean()*100:.3f}% of fine cells)")
    print("")
    print("Max-|Δm| point:")
    print(f"  (i,j)=({i_max},{j_max})  |Δm|={dm[k]:.6e}")
    print(f"  in XOR ring? {in_xor}")
    print(f"  in intersection? {in_inter}")

    # Optional: quantify how much error lives in XOR ring vs elsewhere
    dm_map = dm.reshape((ny_f, nx_f))
    dm_xor = dm_map[xor]
    dm_in = dm_map[inter]
    dm_else = dm_map[np.logical_and(~xor, ~inter)]

    def safe_stats(arr: np.ndarray):
        if arr.size == 0:
            return (0.0, 0.0, 0.0)
        return (float(arr.mean()), float(np.percentile(arr, 95)), float(arr.max()))

    m1, p1, mx1 = safe_stats(dm_xor)
    m2, p2, mx2 = safe_stats(dm_in)

    print("\nError stats:")
    print(f"  XOR ring: mean={m1:.3e}  p95={p1:.3e}  max={mx1:.3e}  (n={dm_xor.size})")
    print(f"  INTERIOR: mean={m2:.3e}  p95={p2:.3e}  max={mx2:.3e}  (n={dm_in.size})")


if __name__ == '__main__':
    main()