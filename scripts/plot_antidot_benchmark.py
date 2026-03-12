#!/usr/bin/env python3
"""
Publication-quality figures for the antidot composite-MG benchmark.

Reads CSV diagnostics from out/bench_vcycle_diag/ and produces
thesis-quality PDF+PNG figures matching the SP2/SP4 style.

Usage:
    python scripts/plot_antidot_benchmark.py [--dir out/bench_vcycle_diag]

Outputs:
    fig_patch_map.pdf / .png    — AMR patch hierarchy around the antidot hole
    fig_radial_bx.pdf / .png    — Radial Bx profile near hole boundary
    fig_error_cmap.pdf / .png   — Signed Bx error colormap (L2+L3)
    fig_crossover.pdf / .png    — Timing + accuracy crossover (if CSV exists)
"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# ═══════════════════════════════════════════════════════════════════
#  Global style — match SP2/SP4 thesis figures
# ═══════════════════════════════════════════════════════════════════

def setup_style() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        # Font
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size":         11,
        "mathtext.fontset":  "cm",

        # Axes
        "axes.linewidth":    0.8,
        "axes.labelsize":    12,
        "axes.titlesize":    12,
        "axes.spines.top":   True,
        "axes.spines.right": True,

        # Ticks
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.size":  4,
        "ytick.major.size":  4,
        "xtick.minor.size":  2,
        "ytick.minor.size":  2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.top":         True,
        "ytick.right":       True,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,

        # Legend
        "legend.fontsize":    10,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "0.7",
        "legend.fancybox":    False,
        "legend.handlelength": 1.8,

        # Lines
        "lines.linewidth":   1.5,

        # Figure
        "figure.dpi":        200,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
    })


def _save(fig: Figure, out_dir: str, name: str) -> None:
    """Save figure as both PDF and PNG."""
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p)
        print(f"  Wrote {p}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Figure 0: AMR Patch Map
# ═══════════════════════════════════════════════════════════════════

def plot_patch_map(diag_dir: str, out_dir: str,
                   hole_r_nm: float = 100.0,
                   domain_nm: float = 500.0,
                   base_nx: int = 256) -> None:
    """AMR patch hierarchy around the antidot hole."""
    csv_path = os.path.join(diag_dir, "patch_map.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP patch map: {csv_path} not found")
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    data = np.atleast_1d(data)

    dx_l0 = domain_nm / base_nx  # nm per L0 cell
    domain_half = domain_nm / 2.0

    # Colour palette per level
    level_style: dict[int, dict[str, object]] = {
        1: {"fc": (0.95, 0.75, 0.20, 0.18), "ec": (0.70, 0.55, 0.10), "lw": 0.8,
            "label": f"L1 (dx = {dx_l0/2:.2f} nm)"},
        2: {"fc": (0.40, 0.75, 0.40, 0.22), "ec": (0.15, 0.50, 0.15), "lw": 0.7,
            "label": f"L2 (dx = {dx_l0/4:.2f} nm)"},
        3: {"fc": (0.35, 0.60, 0.85, 0.25), "ec": (0.15, 0.35, 0.65), "lw": 0.6,
            "label": f"L3 (dx = {dx_l0/8:.3f} nm)"},
    }

    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    # Light grey background for domain
    ax.add_patch(mpatches.Rectangle(
        (-domain_half, -domain_half), domain_nm, domain_nm,
        fc="0.96", ec="0.7", lw=0.5, zorder=0))

    # Patches level by level
    legend_handles: list[mpatches.Patch | Line2D] = []
    for lvl in sorted(level_style.keys()):
        style = level_style[lvl]
        mask = data["level"].astype(int) == lvl
        drawn = False
        for row in data[mask]:
            x0 = float(row["coarse_i0"]) * dx_l0 - domain_half
            y0 = float(row["coarse_j0"]) * dx_l0 - domain_half
            w = float(row["coarse_nx"]) * dx_l0
            h = float(row["coarse_ny"]) * dx_l0
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), w, h,
                fc=style["fc"], ec=style["ec"],
                lw=float(style["lw"]),  # type: ignore[arg-type]
                zorder=lvl + 1))
            drawn = True
        if drawn:
            legend_handles.append(mpatches.Patch(
                fc=style["fc"], ec=style["ec"],
                lw=float(style["lw"]),  # type: ignore[arg-type]
                label=str(style["label"])))

    # Hole boundary
    circle = Circle((0.0, 0.0), hole_r_nm, fc="white", ec="black",
                         lw=1.2, zorder=10)
    ax.add_patch(circle)
    legend_handles.append(Line2D(
        [], [], color="black", lw=1.2, label="Hole boundary"))

    ax.set_xlim(-domain_half * 1.02, domain_half * 1.02)
    ax.set_ylim(-domain_half * 1.02, domain_half * 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (nm)")
    ax.set_ylabel("$y$ (nm)")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
              frameon=True, framealpha=0.92, borderpad=0.5,
              handlelength=1.5, handleheight=1.2)

    fig.tight_layout()
    _save(fig, out_dir, "fig_patch_map")


# ═══════════════════════════════════════════════════════════════════
#  Figure 1: Radial Bx profile near hole boundary
# ═══════════════════════════════════════════════════════════════════

def plot_radial_bx(diag_dir: str, out_dir: str,
                   hole_r_nm: float = 100.0) -> None:
    """Radial Bx near the hole boundary."""
    csv_path = os.path.join(diag_dir, "radial_bx_averaged.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP radial Bx: {csv_path} not found")
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    r = data["r_nm"]
    n = data["n_samples"]
    bx_fft = data["bx_fft_avg"]
    bx_cfft = data["bx_cfft_avg"]
    bx_comp = data["bx_comp_avg"]

    r_lo, r_hi = 85.0, 115.0
    mask = (r >= r_lo) & (r <= r_hi) & (n > 0)
    r, bx_fft = r[mask], bx_fft[mask]
    bx_cfft, bx_comp = bx_cfft[mask], bx_comp[mask]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.axvline(hole_r_nm, color="0.55", ls="--", lw=0.7, zorder=1)
    ax.axhline(0, color="0.82", ls="-", lw=0.5, zorder=0)

    if np.any(np.isfinite(bx_fft)):
        ax.plot(r, bx_fft, "-", color="#1F4E9A", lw=2.2,
                label="Fine FFT (reference)", zorder=3)
    ax.plot(r, bx_cfft, "-", color="#2D8E2D", lw=1.4,
            label="Coarse-FFT", zorder=2)
    ax.plot(r, bx_comp, "-", color="#C03030", lw=1.4,
            label="Composite MG", zorder=2)

    ax.set_xlim(r_lo, r_hi)
    ylims = ax.get_ylim()
    ypad = 0.06 * (ylims[1] - ylims[0])
    ax.text(hole_r_nm - 5.0, ylims[0] + ypad, "hole", ha="center",
            va="bottom", fontsize=9, fontstyle="italic", color="0.4")
    ax.text(hole_r_nm + 5.5, ylims[0] + ypad, "material", ha="center",
            va="bottom", fontsize=9, fontstyle="italic", color="0.4")

    ax.legend(loc="upper left", frameon=True, fontsize=9,
              borderpad=0.4, handletextpad=0.5)
    ax.set_xlabel("Radial distance from hole centre (nm)")
    ax.set_ylabel(r"$B_x$ (T)")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    fig.tight_layout()
    _save(fig, out_dir, "fig_radial_bx")


# ═══════════════════════════════════════════════════════════════════
#  Figure 2: Signed Bx error colormap (L2+L3 cells)
# ═══════════════════════════════════════════════════════════════════

def plot_error_cmap(diag_dir: str, out_dir: str,
                    hole_r_nm: float = 100.0) -> None:
    """Side-by-side Bx error colormaps: coarse-FFT vs composite MG."""
    csv_path = os.path.join(diag_dir, "error_map_l2l3.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP error cmap: {csv_path} not found")
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    x = data["x_nm"]
    y = data["y_nm"]
    dx = data["dx_nm"]
    bx_fft = data["bx_fft"]
    bx_cfft = data["bx_cfft"]
    bx_comp = data["bx_comp"]
    is_mat = data["is_material"].astype(bool)

    b_mag = np.sqrt(bx_fft**2 + data["by_fft"]**2)
    b_max = max(float(np.max(b_mag)), 1e-30)

    # % error, NaN for vacuum
    dbx_cfft_pct = np.where(is_mat, (bx_cfft - bx_fft) / b_max * 100.0, np.nan)
    dbx_comp_pct = np.where(is_mat, (bx_comp - bx_fft) / b_max * 100.0, np.nan)

    x_lo, x_hi = -107.5, -92.5
    y_lo, y_hi = -12.0, 12.0
    spatial = ((x >= x_lo - 1) & (x <= x_hi + 1) &
               (y >= y_lo - 1) & (y <= y_hi + 1))
    x_s, y_s, dx_s = x[spatial], y[spatial], dx[spatial]
    dbx_cfft_s = dbx_cfft_pct[spatial]
    dbx_comp_s = dbx_comp_pct[spatial]

    if len(x_s) == 0:
        print("  SKIP error cmap: no cells in zoom region")
        return

    # Rasterise
    px = 0.12
    nx_px = int(np.ceil((x_hi - x_lo) / px))
    ny_px = int(np.ceil((y_hi - y_lo) / px))
    img_cfft = np.full((ny_px, nx_px), np.nan)
    img_comp = np.full((ny_px, nx_px), np.nan)

    for xi, yi, dxi, vc, vv in zip(x_s, y_s, dx_s, dbx_cfft_s, dbx_comp_s):
        half = dxi * 0.5
        ix0 = max(0, int((xi - half - x_lo) / px))
        ix1 = min(nx_px, int(np.ceil((xi + half - x_lo) / px)))
        iy0 = max(0, int((yi - half - y_lo) / px))
        iy1 = min(ny_px, int(np.ceil((yi + half - y_lo) / px)))
        if np.isfinite(vc):
            img_cfft[iy0:iy1, ix0:ix1] = vc
        if np.isfinite(vv):
            img_comp[iy0:iy1, ix0:ix1] = vv

    # p99 cap (not p95) — wider range shows the error *decay* away from the
    # boundary, making the composite advantage visually clear.  p95 saturates
    # too many boundary cells, hiding the spatial pattern.
    all_abs = np.concatenate([np.abs(dbx_cfft_s), np.abs(dbx_comp_s)])
    all_abs = all_abs[np.isfinite(all_abs)]
    if len(all_abs) == 0:
        print("  SKIP error cmap: no finite error values")
        return
    cap = float(max(np.ceil(np.percentile(all_abs, 99) / 5) * 5, 5.0))

    cmap_obj = matplotlib.colormaps.get_cmap("RdBu_r").copy()
    cmap_obj.set_bad(color="white")

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True,
                              gridspec_kw={"wspace": 0.08, "right": 0.88})

    for ax, title, img in zip(axes.flat, ["Coarse-FFT", "Composite MG"],
                               [img_cfft, img_comp]):
        ax.imshow(img, origin="lower",
                  extent=[x_lo, x_hi, y_lo, y_hi],
                  cmap=cmap_obj, vmin=-cap, vmax=cap,
                  aspect="equal", interpolation="nearest")

        theta = np.linspace(np.pi * 0.55, np.pi * 1.45, 400)
        arc_x = hole_r_nm * np.cos(theta)
        arc_y = hole_r_nm * np.sin(theta)
        vis = ((arc_x >= x_lo) & (arc_x <= x_hi) &
               (arc_y >= y_lo) & (arc_y <= y_hi))
        if np.any(vis):
            ax.plot(arc_x[vis], arc_y[vis], "k-", lw=1.0, zorder=5)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("$x$ (nm)")
        ax.set_title(title, fontsize=11, pad=4)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    axes.flat[0].set_ylabel("$y$ (nm)")

    norm = Normalize(vmin=-cap, vmax=cap)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes((0.90, 0.15, 0.025, 0.70))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\Delta B_x / \mathrm{max}|B|$ (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.10, bottom=0.16, top=0.90)
    _save(fig, out_dir, "fig_error_cmap")


# ═══════════════════════════════════════════════════════════════════
#  Figure 3: Crossover timing + accuracy (sweep mode)
# ═══════════════════════════════════════════════════════════════════

def plot_crossover(diag_dir: str, out_dir: str) -> None:
    """Timing crossover and edge accuracy vs grid size."""
    csv_path = os.path.join(diag_dir, "crossover_sweep.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP crossover: {csv_path} not found")
        return

    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    N_fine = data["fine_cells"]       # N = n² total cells
    t_fine = data["t_fine_ms"] / 1000.0   # seconds
    t_cfft = data["t_cfft_ms"] / 1000.0
    t_comp = data["t_comp_ms"] / 1000.0
    e_cfft = data["edge_rmse_cfft_pct"]
    e_comp = data["edge_rmse_comp_pct"]

    has_fine = bool(np.any(t_fine > 0))
    has_acc = bool(np.any(np.isfinite(e_cfft) & (e_cfft > 0)))

    n_panels = 1 + int(has_acc)
    fig, axes_raw = plt.subplots(n_panels, 1,
                                  figsize=(4.8, 2.8 * n_panels),
                                  sharex=True)
    axes: list[plt.Axes] = [axes_raw] if n_panels == 1 else list(axes_raw.flat)  # type: ignore[list-item]
    panel = 0

    # ── Panel 1: Timing with speedup annotations ──
    ax = axes[panel]; panel += 1
    if has_fine:
        m = t_fine > 0
        ax.loglog(N_fine[m], t_fine[m], "o-", color="#1F4E9A", ms=4,
                  label="Fine FFT (uniform)", zorder=3)
    ax.loglog(N_fine, t_cfft, "s-", color="#2D8E2D", ms=4,
              label="Coarse-FFT + AMR", zorder=2)
    ax.loglog(N_fine, t_comp, "^-", color="#C03030", ms=4,
              label="Composite MG + AMR", zorder=2)

    # O(N log N) reference
    if has_fine and int(np.sum(t_fine > 0)) >= 2:
        m = t_fine > 0
        N_ref, t_ref = float(N_fine[m][-1]), float(t_fine[m][-1])
        N_line = np.logspace(np.log10(float(N_fine.min()) * 0.5),
                             np.log10(float(N_fine.max()) * 2.0), 60)
        c = t_ref / (N_ref * np.log(N_ref))
        ax.plot(N_line, c * N_line * np.log(N_line), "--",
                color="0.6", lw=0.8, label=r"$\propto N\log N$", zorder=1)

    # Speedup labels below each composite point
    if has_fine:
        m = t_fine > 0
        for ni, ti, tc in zip(N_fine[m], t_fine[m], t_comp[m]):
            sp = ti / tc
            ax.annotate(f"{sp:.0f}×",
                        (ni, tc), textcoords="offset points",
                        xytext=(0, -13), ha="center",
                        fontsize=7.5, color="#C03030", fontstyle="italic")

    ax.set_ylabel("Wall-clock time (s)")
    ax.legend(loc="upper left", fontsize=8)

    # ── Panel 2: Edge RMSE only ──
    if has_acc:
        ax2 = axes[panel]; panel += 1
        m = np.isfinite(e_cfft) & (e_cfft > 0)
        if np.any(m):
            ax2.semilogx(N_fine[m], e_cfft[m], "s-", color="#2D8E2D", ms=4,
                         label="Coarse-FFT + AMR")
        m2 = np.isfinite(e_comp) & (e_comp > 0)
        if np.any(m2):
            ax2.semilogx(N_fine[m2], e_comp[m2], "^-", color="#C03030", ms=4,
                         label="Composite MG + AMR")
        ax2.set_ylabel("Edge RMSE (%)")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_ylim(bottom=0)

    # ── X-axis ──
    n_min = int(np.sqrt(float(N_fine.min())))
    n_max = int(np.sqrt(float(N_fine.max())))
    axes[-1].set_xlabel(
        rf"$N = n^2$ total fine-equivalent cells  ($n = {n_min}$ to ${n_max}$)")

    fig.tight_layout()
    _save(fig, out_dir, "fig_crossover")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thesis-quality figures for antidot composite-MG benchmark")
    parser.add_argument("--dir", default="out/bench_vcycle_diag",
                        help="Directory containing diagnostic CSVs")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: same as --dir)")
    parser.add_argument("--hole-r", type=float, default=100.0,
                        help="Hole radius in nm")
    parser.add_argument("--domain", type=float, default=500.0,
                        help="Domain size in nm")
    parser.add_argument("--base-nx", type=int, default=256,
                        help="L0 grid side")
    args = parser.parse_args()

    diag_dir = args.dir
    out_dir = args.out or diag_dir

    if not os.path.isdir(diag_dir):
        print(f"ERROR: {diag_dir} not found. Run bench_composite_vcycle first.")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    setup_style()

    print(f"Reading from {diag_dir}/")
    print(f"Writing to {out_dir}/")
    print()

    plot_patch_map(diag_dir, out_dir, args.hole_r, args.domain, args.base_nx)
    plot_radial_bx(diag_dir, out_dir, args.hole_r)
    plot_error_cmap(diag_dir, out_dir, args.hole_r)
    plot_crossover(diag_dir, out_dir)

    print()
    print("Done.")


if __name__ == "__main__":
    main()