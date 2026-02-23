#!/usr/bin/env python3
"""
analyse_amr_masked_disk.py

A) Locate the max |Δm| point between AMR-flattened and uniform-fine final states,
   and report whether it's close to the disk boundary.

Inputs expected in out_dir:
  - amr_fine_final.csv
  - uniform_fine_final.csv

Assumes CSV format written by write_csv_field():
  i,j,x_m,y_m,mx,my,mz
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def load_field_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: i (N,), j (N,), x (N,), y (N,), m (N,3)
    """
    data = np.genfromtxt(path, delimiter=",", names=True)
    i = data["i"].astype(np.int64)
    j = data["j"].astype(np.int64)
    x = data["x_m"].astype(np.float64)
    y = data["y_m"].astype(np.float64)
    m = np.stack([data["mx"], data["my"], data["mz"]], axis=1).astype(np.float64)
    return i, j, x, y, m


def infer_grid(i: np.ndarray, j: np.ndarray) -> tuple[int, int]:
    nx = int(i.max() + 1)
    ny = int(j.max() + 1)
    return nx, ny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/amr_masked_disk_relax"))
    ap.add_argument("--disk-radius", type=float, required=True, help="Disk radius in meters used by the benchmark")
    ap.add_argument("--center-x", type=float, default=0.0, help="Disk center x in centered coords (meters)")
    ap.add_argument("--center-y", type=float, default=0.0, help="Disk center y in centered coords (meters)")
    ap.add_argument("--near-boundary-cells", type=float, default=3.0, help="How many cell widths counts as 'near boundary'")
    args = ap.parse_args()

    amr_path = args.out / "amr_fine_final.csv"
    uni_path = args.out / "uniform_fine_final.csv"

    ia, ja, xa, ya, ma = load_field_csv(amr_path)
    ib, jb, xb, yb, mb = load_field_csv(uni_path)

    # Basic sanity
    if ia.shape != ib.shape:
        raise RuntimeError(f"Row count mismatch: {amr_path} vs {uni_path}")
    if not (np.all(ia == ib) and np.all(ja == jb)):
        raise RuntimeError("Index grids differ between CSVs (i,j).")

    nx, ny = infer_grid(ia, ja)

    # Infer dx, dy from coordinates (assumes uniform spacing)
    # Use two adjacent points along x and y if possible.
    dx = float(np.median(np.diff(np.unique(xa)))) if nx > 1 else 0.0
    dy = float(np.median(np.diff(np.unique(ya)))) if ny > 1 else 0.0

    # Compute delta
    dm_vec = ma - mb
    dm = np.linalg.norm(dm_vec, axis=1)

    k = int(np.argmax(dm))
    max_dm = float(dm[k])

    i_max = int(ia[k])
    j_max = int(ja[k])
    x_max = float(xa[k])
    y_max = float(ya[k])

    # Convert absolute coords (0..L) to centered coords (-L/2..L/2)
    Lx = nx * dx
    Ly = ny * dy
    x_c = x_max - 0.5 * Lx
    y_c = y_max - 0.5 * Ly

    # Distance from disk center (in centered coordinates)
    rx = x_c - args.center_x
    ry = y_c - args.center_y
    r = float(np.hypot(rx, ry))
    dist_to_boundary = abs(r - args.disk_radius)

    near_thresh = args.near_boundary_cells * max(dx, dy)
    near_boundary = dist_to_boundary <= near_thresh

    print("=== A) Max |Δm| location ===")
    print(f"grid: nx={nx} ny={ny}  dx≈{dx:.3e}  dy≈{dy:.3e}")
    print(f"max |Δm| = {max_dm:.6e} at (i,j)=({i_max},{j_max})")
    print(f"pos (abs)   x={x_max:.6e}  y={y_max:.6e}")
    print(f"pos (cent)  x={x_c:.6e}  y={y_c:.6e}")
    print(f"r(centered) = {r:.6e}")
    print(f"|r - R|     = {dist_to_boundary:.6e}")
    print(f"near-boundary? {near_boundary}  (threshold {near_thresh:.3e} m ~ {args.near_boundary_cells} cells)")

    # Extra: report how concentrated the error is
    print("\n=== Extra: error distribution ===")
    print(f"p95 |Δm| = {np.percentile(dm, 95):.6e}")
    print(f"p99 |Δm| = {np.percentile(dm, 99):.6e}")
    print(f"mean|Δm| = {dm.mean():.6e}")
    print(f"RMSE(|Δm|) = {np.sqrt(np.mean(dm**2)):.6e}")


if __name__ == '__main__':
    main()