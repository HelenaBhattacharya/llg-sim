#!/usr/bin/env python3
"""
plot_bubble_cascade.py — Publication-quality figures for the bubble cascade benchmark.

Produces:
  1. García-Cervera Fig 4 style:  mz heatmap + patch boundaries at selected steps
  2. García-Cervera Fig 5 style:  zoomed mesh showing multi-level cell grids
  3. 3D surface plot:             mz as height, mesh grid overlaid
  4. RMSE comparison plot:        AMR vs coarse divergence over time
  5. Multi-panel composite:       snapshot strip for presentation slides

Usage:
  python plot_bubble_cascade.py                          # all figures, default dir
  python plot_bubble_cascade.py --root out/bubble_cascade --steps 0 1920 6016
  python plot_bubble_cascade.py --only rmse              # just the RMSE plot
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─── OVF reader ──────────────────────────────────────────────────────────────

def read_ovf(path):
    """Read an OVF 2.0 text file. Returns (m, nx, ny, dx, dy, dz)."""
    header = {}
    data_lines = []
    in_data = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("# Begin: Data"):
                in_data = True
                continue
            if s.startswith("# End: Data"):
                in_data = False
                continue
            if in_data:
                data_lines.append(s)
            elif s.startswith("#"):
                parts = s[1:].strip().split(":", 1)
                if len(parts) == 2:
                    header[parts[0].strip().lower()] = parts[1].strip()
    nx = int(header.get("xnodes", 0))
    ny = int(header.get("ynodes", 0))
    dx = float(header.get("xstepsize", 1))
    dy = float(header.get("ystepsize", 1))
    dz = float(header.get("zstepsize", 1))
    vals = []
    for line in data_lines:
        vals.extend(float(x) for x in line.split())
    m = np.array(vals).reshape(ny, nx, 3)
    return m, nx, ny, dx, dy, dz


def read_patches_at_step(csv_path, step):
    """Read regrid_patches.csv, return {level: [(i0,j0,nx,ny), ...]} for a given step."""
    patches = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["step"]) != step:
                continue
            lvl = int(row["level"])
            r = (int(row["i0"]), int(row["j0"]), int(row["nx"]), int(row["ny"]))
            patches.setdefault(lvl, []).append(r)
    return patches


def read_rmse_log(csv_path):
    """Read rmse_log.csv into dict of arrays."""
    data = {"step": [], "rmse_amr": [], "rmse_coarse": [], "mean_mz": []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["step"].append(int(row["step"]))
            data["rmse_amr"].append(float(row["rmse_amr"]))
            data["rmse_coarse"].append(float(row["rmse_coarse"]))
            data["mean_mz"].append(float(row["mean_mz"]))
    return {k: np.array(v) for k, v in data.items()}


# ─── Patch drawing ───────────────────────────────────────────────────────────

LEVEL_COLORS = {
    1: ("#FFD700", 1.8),   # yellow, linewidth
    2: ("#00AA00", 1.4),   # green
    3: ("#2266DD", 1.0),   # blue
    4: ("#9933CC", 0.8),   # purple
}

def draw_patches_on_ax(ax, patches_by_level, cnx, cny, nx_fine, ny_fine):
    """Draw patch rectangles scaled to fine-grid pixel coordinates."""
    ratio = nx_fine // cnx if cnx > 0 else 1
    legend_handles = []
    for lvl in sorted(patches_by_level.keys()):
        col, lw = LEVEL_COLORS.get(lvl, ("#999999", 1.0))
        for (i0, j0, pnx, pny) in patches_by_level[lvl]:
            x0 = i0 * ratio
            y0 = j0 * ratio
            w = pnx * ratio
            h = pny * ratio
            rect = mpatches.Rectangle((x0 - 0.5, y0 - 0.5), w, h,
                                      linewidth=lw, edgecolor=col,
                                      facecolor="none", zorder=10 + lvl)
            ax.add_patch(rect)
        legend_handles.append(mpatches.Patch(edgecolor=col, facecolor="none",
                                             linewidth=lw, label=f"L{lvl}"))
    return legend_handles


def draw_cell_grids(ax, patches_by_level, cnx, cny, nx_fine, ny_fine,
                    xlim=None, ylim=None):
    """Draw cell grid lines within each patch (for mesh zoom)."""
    ratio_total = nx_fine // cnx
    for lvl in sorted(patches_by_level.keys()):
        col, _ = LEVEL_COLORS.get(lvl, ("#999999", 1.0))
        cell_size = ratio_total // (2 ** lvl)  # fine-grid pixels per cell at this level
        if cell_size < 2:
            continue  # too fine to draw
        for (i0, j0, pnx, pny) in patches_by_level[lvl]:
            px0 = i0 * ratio_total
            py0 = j0 * ratio_total
            px1 = px0 + pnx * ratio_total
            py1 = py0 + pny * ratio_total
            # Clip to zoom region
            if xlim:
                px0 = max(px0, xlim[0]); px1 = min(px1, xlim[1])
            if ylim:
                py0 = max(py0, ylim[0]); py1 = min(py1, ylim[1])
            if px1 <= px0 or py1 <= py0:
                continue
            # Vertical cell lines
            x = px0 + cell_size - (px0 % cell_size) if px0 % cell_size else px0
            while x <= px1:
                ax.plot([x-0.5, x-0.5], [py0-0.5, py1-0.5],
                        color=col, linewidth=0.3, alpha=0.6, zorder=5+lvl)
                x += cell_size
            # Horizontal cell lines
            y = py0 + cell_size - (py0 % cell_size) if py0 % cell_size else py0
            while y <= py1:
                ax.plot([px0-0.5, px1-0.5], [y-0.5, y-0.5],
                        color=col, linewidth=0.3, alpha=0.6, zorder=5+lvl)
                y += cell_size
            # Patch outline (thicker)
            rect = mpatches.Rectangle((px0-0.5, py0-0.5), px1-px0, py1-py0,
                                      linewidth=1.2, edgecolor=col,
                                      facecolor="none", zorder=10+lvl)
            ax.add_patch(rect)


# ─── Figure 1: mz + patches (García-Cervera Fig 4 style) ────────────────────

def fig_mz_patches(root, step, cnx, cny, out_dir):
    """mz colourmap with AMR patch boundaries overlaid."""
    ovf_path = root / f"ovf_amr/m{step:07d}.ovf"
    if not ovf_path.exists():
        print(f"  SKIP {ovf_path} (not found)")
        return
    m, nx, ny, dx, dy, dz = read_ovf(ovf_path)
    mz = m[:, :, 2]

    patches = read_patches_at_step(root / "regrid_patches.csv", step)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(mz, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest", aspect="equal")
    handles = draw_patches_on_ax(ax, patches, cnx, cny, nx, ny)
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              framealpha=0.8, handlelength=1.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("$m_z$", fontsize=11)
    ax.set_xlabel("x (fine cells)", fontsize=10)
    ax.set_ylabel("y (fine cells)", fontsize=10)
    ax.set_title(f"$m_z$ with AMR patches — step {step}", fontsize=12)

    out = out_dir / f"fig4_mz_patches_step{step:04d}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.name}")


# ─── Figure 2: mesh zoom (García-Cervera Fig 5 style) ───────────────────────

def fig_mesh_zoom(root, step, cnx, cny, out_dir, zoom_frac=(0.2, 0.7, 0.15, 0.85)):
    """Zoomed view showing cell grids at each AMR level."""
    ovf_path = root / f"ovf_amr/m{step:07d}.ovf"
    if not ovf_path.exists():
        print(f"  SKIP {ovf_path} (not found)")
        return
    m, nx, ny, dx, dy, dz = read_ovf(ovf_path)

    # Compute in-plane angle for colour (more informative than just mz)
    angle = np.arctan2(m[:, :, 1], m[:, :, 0])

    patches = read_patches_at_step(root / "regrid_patches.csv", step)

    # Zoom region in fine-grid pixels
    x0 = int(zoom_frac[0] * nx)
    x1 = int(zoom_frac[1] * nx)
    y0 = int(zoom_frac[2] * ny)
    y1 = int(zoom_frac[3] * ny)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(angle, origin="lower", cmap="hsv", vmin=-np.pi, vmax=np.pi,
              interpolation="nearest", aspect="equal")
    draw_cell_grids(ax, patches, cnx, cny, nx, ny, xlim=(x0, x1), ylim=(y0, y1))
    ax.set_xlim(x0 - 0.5, x1 - 0.5)
    ax.set_ylim(y0 - 0.5, y1 - 0.5)
    ax.set_xlabel("x (fine cells)", fontsize=10)
    ax.set_ylabel("y (fine cells)", fontsize=10)
    ax.set_title(f"Mesh zoom — in-plane angle + AMR grids — step {step}", fontsize=12)

    # Add L0 grid lines (coarse)
    ratio = nx // cnx
    for xi in range(x0 // ratio, x1 // ratio + 1):
        xp = xi * ratio
        if x0 <= xp <= x1:
            ax.axvline(xp - 0.5, color="gray", linewidth=0.5, alpha=0.4, zorder=3)
    for yi in range(y0 // ratio, y1 // ratio + 1):
        yp = yi * ratio
        if y0 <= yp <= y1:
            ax.axhline(yp - 0.5, color="gray", linewidth=0.5, alpha=0.4, zorder=3)

    out = out_dir / f"fig5_mesh_zoom_step{step:04d}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.name}")


# ─── Figure 3: 3D surface with mesh (García-Cervera 3D style) ───────────────

def fig_3d_surface(root, step, cnx, cny, out_dir,
                   elev=35, azim=-60, warp=0.5, stride=4):
    """3D surface plot: mz as height, coloured by mz, mesh grid overlaid."""
    ovf_path = root / f"ovf_amr/m{step:07d}.ovf"
    if not ovf_path.exists():
        print(f"  SKIP {ovf_path}")
        return
    m, nx, ny, dx, dy, dz = read_ovf(ovf_path)
    mz = m[:, :, 2]

    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = warp * mz

    patches = read_patches_at_step(root / "regrid_patches.csv", step)
    ratio = nx // cnx

    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")  # type: ignore[attr-defined]

    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(mz)),  # type: ignore[attr-defined]
                    rstride=stride, cstride=stride,
                    linewidth=0, antialiased=True, shade=True, alpha=0.92)

    # Draw patch boundaries as 3D lines on the surface
    for lvl in sorted(patches.keys()):
        col, lw = LEVEL_COLORS.get(lvl, ("#999999", 1.0))
        for (i0, j0, pnx, pny) in patches[lvl]:
            # Convert coarse coords to normalised coords
            fx0 = (i0 * ratio) / nx
            fy0 = (j0 * ratio) / ny
            fx1 = ((i0 + pnx) * ratio) / nx
            fy1 = ((j0 + pny) * ratio) / ny
            # Sample mz at patch corners for z-coordinate
            corners_x = [fx0, fx1, fx1, fx0, fx0]
            corners_y = [fy0, fy0, fy1, fy1, fy0]
            corners_z = []
            for cx, cy in zip(corners_x, corners_y):
                ix = min(int(cx * nx), nx - 1)
                iy = min(int(cy * ny), ny - 1)
                corners_z.append(warp * mz[iy, ix] + 0.005)
            ax.plot(corners_x, corners_y, corners_z,
                    color=col, linewidth=lw * 0.8, alpha=0.7, zorder=10 + lvl)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08).set_label("$m_z$", fontsize=10)

    ax.set_xlabel("x / L", fontsize=9, labelpad=6)
    ax.set_ylabel("y / L", fontsize=9, labelpad=6)
    ax.set_zlabel(f"$m_z$ × {warp}", fontsize=9, labelpad=6)  # type: ignore[attr-defined]
    ax.set_title(f"3D AMR surface — step {step}", fontsize=12, pad=12)
    ax.view_init(elev=elev, azim=azim)  # type: ignore[attr-defined]

    out = out_dir / f"fig_3d_step{step:04d}_e{elev}a{azim}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.name}")


# ─── Figure 4: RMSE comparison ──────────────────────────────────────────────

def fig_rmse(root, out_dir):
    """RMSE vs step: AMR stays flat, coarse diverges."""
    data = read_rmse_log(root / "rmse_log.csv")
    steps = data["step"]
    rmse_a = data["rmse_amr"] * 100   # percent
    rmse_c = data["rmse_coarse"] * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, rmse_a, "r-", linewidth=1.5, label="AMR (coarse FFT + patches)")
    ax.plot(steps, rmse_c, "b--", linewidth=1.5, label="Uniform coarse")
    ax.set_xlabel("Simulation step", fontsize=11)
    ax.set_ylabel("RMSE vs fine reference (%)", fontsize=11)
    ax.set_title("Accuracy: AMR vs uniform coarse", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(0, max(rmse_c[-1] * 1.15, 25))
    ax.grid(True, alpha=0.3)

    # Annotate final values
    ax.annotate(f"AMR: {rmse_a[-1]:.1f}%", xy=(steps[-1], rmse_a[-1]),
                xytext=(-80, 10), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax.annotate(f"Coarse: {rmse_c[-1]:.1f}%", xy=(steps[-1], rmse_c[-1]),
                xytext=(-100, -15), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="blue"), color="blue")

    out = out_dir / "fig_rmse_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.name}")


# ─── Figure 5: Snapshot strip (presentation slide) ──────────────────────────

def fig_snapshot_strip(root, steps_list, cnx, cny, out_dir):
    """Horizontal strip of mz+patches at selected steps. Good for slides."""
    n = len(steps_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, step in zip(axes, steps_list):
        ovf_path = root / f"ovf_amr/m{step:07d}.ovf"
        if not ovf_path.exists():
            ax.text(0.5, 0.5, f"step {step}\nnot found", ha="center", va="center")
            continue
        m, nx, ny, dx, dy, dz = read_ovf(ovf_path)
        mz = m[:, :, 2]
        patches = read_patches_at_step(root / "regrid_patches.csv", step)

        ax.imshow(mz, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                  interpolation="nearest", aspect="equal")
        draw_patches_on_ax(ax, patches, cnx, cny, nx, ny)
        ax.set_title(f"Step {step}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Bubble Cascade — AMR Patch Evolution", fontsize=14, y=1.02)

    out = out_dir / "fig_snapshot_strip.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.name}")


# ─── Main ────────────────────────────────────────────────────────────────────

def find_available_steps(root):
    """Find all step numbers from OVF files."""
    ovf_dir = root / "ovf_amr"
    if not ovf_dir.exists():
        return []
    steps = []
    for f in sorted(ovf_dir.iterdir()):
        if f.suffix == ".ovf":
            try:
                steps.append(int(f.stem[1:]))  # m0001920.ovf → 1920
            except ValueError:
                pass
    return steps


def pick_steps(available, n=4):
    """Pick n evenly spaced steps from available list."""
    if len(available) <= n:
        return available
    idx = np.linspace(0, len(available) - 1, n, dtype=int)
    return [available[i] for i in idx]


def main():
    parser = argparse.ArgumentParser(description="Plot bubble cascade benchmark results")
    parser.add_argument("--root", default="out/bubble_cascade", help="Output directory")
    parser.add_argument("--steps", nargs="*", type=int, default=None,
                        help="Steps to plot (default: auto-pick 4)")
    parser.add_argument("--cnx", type=int, default=128, help="Coarse grid nx")
    parser.add_argument("--cny", type=int, default=64, help="Coarse grid ny")
    parser.add_argument("--only", choices=["mz", "zoom", "3d", "rmse", "strip"],
                        default=None, help="Generate only one figure type")
    parser.add_argument("--zoom-frac", nargs=4, type=float,
                        default=[0.15, 0.75, 0.1, 0.9],
                        help="Zoom region as fractions: x0 x1 y0 y1")
    parser.add_argument("--elev", type=int, default=35, help="3D elevation angle")
    parser.add_argument("--azim", type=int, default=-60, help="3D azimuth angle")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        sys.exit(1)

    out_dir = root / "figures"
    out_dir.mkdir(exist_ok=True)

    available = find_available_steps(root)
    if not available and args.only != "rmse":
        print(f"No OVF files found in {root}/ovf_amr/")
        print("Run the benchmark with --ovfs flag first.")
        sys.exit(1)

    if args.steps:
        steps = args.steps
    else:
        steps = pick_steps(available, n=4)

    print(f"Bubble Cascade Plotter")
    print(f"  root:  {root}")
    print(f"  steps: {steps}")
    print(f"  grid:  {args.cnx}×{args.cny}")
    print()

    # ── Generate figures ──────────────────────────────────────────────

    if args.only in (None, "mz"):
        print("=== Fig 4 style: mz + patches ===")
        for s in steps:
            fig_mz_patches(root, s, args.cnx, args.cny, out_dir)

    if args.only in (None, "zoom"):
        print("=== Fig 5 style: mesh zoom ===")
        # Pick the mid-point step for zoom (where merger is happening)
        zoom_step = steps[len(steps) // 2] if steps else 0
        fig_mesh_zoom(root, zoom_step, args.cnx, args.cny, out_dir,
                      zoom_frac=tuple(args.zoom_frac))

    if args.only in (None, "3d"):
        print("=== 3D surface ===")
        for s in steps:
            fig_3d_surface(root, s, args.cnx, args.cny, out_dir,
                           elev=args.elev, azim=args.azim)

    if args.only in (None, "rmse"):
        rmse_path = root / "rmse_log.csv"
        if rmse_path.exists():
            print("=== RMSE comparison ===")
            fig_rmse(root, out_dir)
        else:
            print("  SKIP rmse (no rmse_log.csv — run with fine reference)")

    if args.only in (None, "strip"):
        print("=== Snapshot strip ===")
        fig_snapshot_strip(root, steps, args.cnx, args.cny, out_dir)

    print()
    print(f"All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()