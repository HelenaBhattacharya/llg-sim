#!/usr/bin/env python3
"""
Publication-quality figures for §4.1: Antidot composite-MG benchmark.

Generates:
    fig_patch_map       — Fig 4a: AMR patch hierarchy
    fig_error_cmap      — Fig 4b,c: Absolute Bx error (cfft vs composite)
    fig5_crossover      — Fig 5: Runtime scaling + edge RMSE vs N
    fig6_diagnostic     — Fig 6: Bar chart, component-resolved Bz emphasis

Usage:
    python scripts/plot_antidot_benchmark.py \
        --dir out/bench_vcycle_diag \
        [--sweep-3nm crossover_sweep_3nm.csv] \
        [--sweep-20nm crossover_sweep_20nm.csv] \
        [--diag-nx 96]
"""

import argparse, os, csv as csv_mod, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


# ═══════════════════════════════════════════════════════════════════
#  Style
# ═══════════════════════════════════════════════════════════════════
def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 11, "mathtext.fontset": "cm",
        "axes.linewidth": 0.8, "axes.labelsize": 12, "axes.titlesize": 12,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2, "ytick.minor.size": 2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.top": True, "ytick.right": True,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7", "legend.fancybox": False,
        "legend.handlelength": 1.8, "lines.linewidth": 1.5,
        "figure.dpi": 200, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })

def _save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p); print(f"  Wrote {p}")
    plt.close(fig)

def read_csv(path):
    rows = []
    with open(path) as f:
        for row in csv_mod.DictReader(f):
            d = {}
            for k, v in row.items():
                k = k.strip()
                try: d[k] = float(v)
                except: d[k] = v
            rows.append(d)
    return rows

def _get_col(names, data, candidates):
    """Return data from the first matching column name."""
    for c in candidates:
        if c in names:
            return data[c]
    return None


# ═══════════════════════════════════════════════════════════════════
#  Figure 4a: AMR Patch Map
# ═══════════════════════════════════════════════════════════════════
def plot_patch_map(diag_dir, out_dir,
                   hole_r_nm=100.0, domain_nm=500.0, base_nx=256):
    csv_path = os.path.join(diag_dir, "patch_map.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP patch map: not found"); return

    data = np.atleast_1d(np.genfromtxt(csv_path, delimiter=",", names=True))
    dx_l0 = domain_nm / base_nx
    dh = domain_nm / 2.0

    level_style = {
        1: {"fc": (0.95, 0.75, 0.20, 0.18), "ec": (0.70, 0.55, 0.10),
            "lw": 0.8, "label": f"L1 (dx = {dx_l0/2:.2f} nm)"},
        2: {"fc": (0.40, 0.75, 0.40, 0.22), "ec": (0.15, 0.50, 0.15),
            "lw": 0.7, "label": f"L2 (dx = {dx_l0/4:.2f} nm)"},
        3: {"fc": (0.35, 0.60, 0.85, 0.25), "ec": (0.15, 0.35, 0.65),
            "lw": 0.6, "label": f"L3 (dx = {dx_l0/8:.3f} nm)"},
    }

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.add_patch(mpatches.Rectangle((-dh, -dh), domain_nm, domain_nm,
                 fc="0.96", ec="0.7", lw=0.5, zorder=0))
    lh = []
    for lvl in sorted(level_style.keys()):
        s = level_style[lvl]
        mask = data["level"].astype(int) == lvl
        drawn = False
        for row in data[mask]:
            x0 = float(row["coarse_i0"]) * dx_l0 - dh
            y0 = float(row["coarse_j0"]) * dx_l0 - dh
            w  = float(row["coarse_nx"]) * dx_l0
            ht = float(row["coarse_ny"]) * dx_l0
            ax.add_patch(mpatches.Rectangle((x0, y0), w, ht,
                         fc=s["fc"], ec=s["ec"], lw=float(s["lw"]),
                         zorder=lvl + 1))
            drawn = True
        if drawn:
            lh.append(mpatches.Patch(fc=s["fc"], ec=s["ec"],
                      lw=float(s["lw"]), label=str(s["label"])))

    ax.add_patch(Circle((0, 0), hole_r_nm, fc="white", ec="black",
                         lw=1.2, zorder=10))
    lh.append(Line2D([], [], color="black", lw=1.2, label="Hole boundary"))

    ax.set_xlim(-dh * 1.02, dh * 1.02)
    ax.set_ylim(-dh * 1.02, dh * 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (nm)"); ax.set_ylabel("$y$ (nm)")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.legend(handles=lh, loc="lower left", fontsize=8, frameon=True,
              framealpha=0.92, borderpad=0.5, handlelength=1.5, handleheight=1.2)
    fig.tight_layout()
    _save(fig, out_dir, "fig_patch_map")


# ═══════════════════════════════════════════════════════════════════
#  Figure 4b,c: Absolute Bx error colourmap
#  - Vacuum inside hole → white
#  - Zoomed to material side of boundary
#  - Sequential colourmap (YlOrRd)
# ═══════════════════════════════════════════════════════════════════
def plot_error_cmap(diag_dir, out_dir, hole_r_nm=100.0, domain_nm=500.0, base_nx=256):
    csv_l1 = os.path.join(diag_dir, "error_map_l1.csv")
    csv_l2 = os.path.join(diag_dir, "error_map_l2l3.csv")

    if not os.path.exists(csv_l2) and not os.path.exists(csv_l1):
        print(f"  SKIP error cmap: no error map CSVs found"); return

    cells = []

    if os.path.exists(csv_l1):
        d1 = np.atleast_1d(np.genfromtxt(csv_l1, delimiter=",", names=True))
        dx_l1 = domain_nm / base_nx / 2.0
        for k in range(len(d1)):
            cells.append((d1["x_nm"][k], d1["y_nm"][k], dx_l1,
                          d1["bx_fft"][k], d1["bx_cfft"][k], d1["bx_comp"][k]))
        print(f"  Error cmap: {len(d1)} L1 cells from {csv_l1}")

    if os.path.exists(csv_l2):
        d2 = np.atleast_1d(np.genfromtxt(csv_l2, delimiter=",", names=True))
        for k in range(len(d2)):
            cells.append((d2["x_nm"][k], d2["y_nm"][k], d2["dx_nm"][k],
                          d2["bx_fft"][k], d2["bx_cfft"][k], d2["bx_comp"][k]))
        print(f"  Error cmap: {len(d2)} L2+L3 cells from {csv_l2}")

    if not cells:
        print("  SKIP error cmap: no cells"); return

    x_nm   = np.array([c[0] for c in cells])
    y_nm   = np.array([c[1] for c in cells])
    dx_nm  = np.array([c[2] for c in cells])
    bx_fft = np.array([c[3] for c in cells])
    bx_cfft= np.array([c[4] for c in cells])
    bx_comp= np.array([c[5] for c in cells])

    bmax = np.max(np.abs(bx_fft))
    if bmax < 1e-20: bmax = 1.0

    abx_cfft = np.abs(bx_cfft - bx_fft) / bmax * 100.0
    abx_comp = np.abs(bx_comp - bx_fft) / bmax * 100.0

    r_cell = np.sqrt(x_nm**2 + y_nm**2)
    is_vacuum = r_cell < (hole_r_nm - 0.3)
    abx_cfft[is_vacuum] = np.nan
    abx_comp[is_vacuum] = np.nan

    x_lo = -(hole_r_nm + 15.0)
    x_hi = -(hole_r_nm - 5.0)
    y_lo, y_hi = -25.0, 25.0

    res = 0.10
    nxi = int(np.ceil((x_hi - x_lo) / res))
    nyi = int(np.ceil((y_hi - y_lo) / res))
    img_cfft = np.full((nyi, nxi), np.nan)
    img_comp = np.full((nyi, nxi), np.nan)

    for k in range(len(x_nm)):
        half = max(dx_nm[k] * 0.5, res * 0.5)
        ix0 = max(0, int((x_nm[k] - half - x_lo) / res))
        ix1 = min(nxi, int(np.ceil((x_nm[k] + half - x_lo) / res)))
        iy0 = max(0, int((y_nm[k] - half - y_lo) / res))
        iy1 = min(nyi, int(np.ceil((y_nm[k] + half - y_lo) / res)))
        if np.isfinite(abx_cfft[k]):
            img_cfft[iy0:iy1, ix0:ix1] = abx_cfft[k]
        if np.isfinite(abx_comp[k]):
            img_comp[iy0:iy1, ix0:ix1] = abx_comp[k]

    yy, xx = np.meshgrid(
        np.linspace(y_lo + res / 2, y_hi - res / 2, nyi),
        np.linspace(x_lo + res / 2, x_hi - res / 2, nxi),
        indexing="ij")
    hole_mask = np.sqrt(xx**2 + yy**2) < hole_r_nm
    img_cfft[hole_mask] = np.nan
    img_comp[hole_mask] = np.nan

    vmax = 20.0

    cmap = matplotlib.colormaps.get_cmap("YlOrRd").copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True,
                              gridspec_kw={"wspace": 0.08, "right": 0.87})

    for ax, title, img in zip(axes.flat,
                               ["Coarse-FFT", "Composite MG"],
                               [img_cfft, img_comp]):
        ax.imshow(img, origin="lower", extent=[x_lo, x_hi, y_lo, y_hi],
                  cmap=cmap, vmin=0, vmax=vmax,
                  aspect="equal", interpolation="nearest")

        theta = np.linspace(np.pi * 0.40, np.pi * 1.60, 600)
        arc_x = hole_r_nm * np.cos(theta)
        arc_y = hole_r_nm * np.sin(theta)
        vis = (arc_x >= x_lo) & (arc_x <= x_hi) & (arc_y >= y_lo) & (arc_y <= y_hi)
        if np.any(vis):
            ax.plot(arc_x[vis], arc_y[vis], "k-", lw=1.2, zorder=5)

        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("distance from centre (nm)")
        ax.set_title(title, fontsize=11, pad=4)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}"))
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    axes.flat[0].set_ylabel("$y$ (nm)")

    norm = Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cax = fig.add_axes((0.89, 0.15, 0.025, 0.70))
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$|\Delta B_x|/\mathrm{max}|B|$ (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=0.09, bottom=0.15, top=0.90)
    _save(fig, out_dir, "fig_error_cmap")


# ═══════════════════════════════════════════════════════════════════
#  Figure 4d: Azimuthally-averaged radial error profile
#
#  Shows error vs distance from hole boundary, averaged around the
#  full circumference.  Material side only (r > hole_r).
#  This is the clearest way to show that cfft error penetrates deep
#  into the material while composite error is confined to the
#  boundary.
#
#  Uses the same CSV data as plot_error_cmap.
# ═══════════════════════════════════════════════════════════════════
def plot_error_radial(diag_dir, out_dir,
                      hole_r_nm=100.0, domain_nm=500.0,
                      base_nx=256):
    csv_l1 = os.path.join(diag_dir, "error_map_l1.csv")
    csv_l2 = os.path.join(diag_dir, "error_map_l2l3.csv")

    if not os.path.exists(csv_l2) and not os.path.exists(csv_l1):
        print(f"  SKIP error radial: no error map CSVs found")
        return

    cells = []
    if os.path.exists(csv_l1):
        d1 = np.atleast_1d(
            np.genfromtxt(csv_l1, delimiter=",", names=True))
        dx_l1 = domain_nm / base_nx / 2.0
        for k in range(len(d1)):
            cells.append((d1["x_nm"][k], d1["y_nm"][k], dx_l1,
                          d1["bx_fft"][k], d1["bx_cfft"][k],
                          d1["bx_comp"][k]))

    if os.path.exists(csv_l2):
        d2 = np.atleast_1d(
            np.genfromtxt(csv_l2, delimiter=",", names=True))
        for k in range(len(d2)):
            cells.append((d2["x_nm"][k], d2["y_nm"][k],
                          d2["dx_nm"][k], d2["bx_fft"][k],
                          d2["bx_cfft"][k], d2["bx_comp"][k]))

    if not cells:
        print("  SKIP error radial: no cells"); return

    x_nm   = np.array([c[0] for c in cells])
    y_nm   = np.array([c[1] for c in cells])
    bx_fft = np.array([c[3] for c in cells])
    bx_cfft = np.array([c[4] for c in cells])
    bx_comp = np.array([c[5] for c in cells])

    bmax = np.max(np.abs(bx_fft))
    if bmax < 1e-20:
        bmax = 1.0

    # Absolute error as % of max|B|
    err_cfft = np.abs(bx_cfft - bx_fft) / bmax * 100.0
    err_comp = np.abs(bx_comp - bx_fft) / bmax * 100.0

    # Radial distance from hole centre
    r_cell = np.sqrt(x_nm**2 + y_nm**2)

    # Distance from hole boundary (positive = into material)
    dist_from_boundary = r_cell - hole_r_nm

    # Material cells only (outside hole)
    mat_mask = dist_from_boundary > 0.1  # small margin
    dist_mat = dist_from_boundary[mat_mask]
    err_cfft_mat = err_cfft[mat_mask]
    err_comp_mat = err_comp[mat_mask]

    if len(dist_mat) == 0:
        print("  SKIP error radial: no material cells"); return

    # ── Bin into radial shells ──────────────────────────────
    max_dist = 25.0  # nm from boundary
    bin_width = 0.5   # nm per bin
    n_bins = int(np.ceil(max_dist / bin_width))
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mean_cfft = np.zeros(n_bins)
    mean_comp = np.zeros(n_bins)
    p90_cfft  = np.zeros(n_bins)
    p90_comp  = np.zeros(n_bins)
    counts    = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = (dist_mat >= lo) & (dist_mat < hi)
        counts[b] = int(np.sum(in_bin))
        if counts[b] > 0:
            mean_cfft[b] = np.mean(err_cfft_mat[in_bin])
            mean_comp[b] = np.mean(err_comp_mat[in_bin])
            p90_cfft[b]  = np.percentile(err_cfft_mat[in_bin], 90)
            p90_comp[b]  = np.percentile(err_comp_mat[in_bin], 90)

    # Mask out empty bins
    valid = counts > 2
    bc = bin_centres[valid]
    mc = mean_cfft[valid]
    mk = mean_comp[valid]
    pc = p90_cfft[valid]
    pk = p90_comp[valid]

    # ── Plot ────────────────────────────────────────────────
    cc_col = "#2E7D32"  # cfft: green
    cm_col = "#C03030"  # composite: red

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.17,
                        top=0.92)

    # Shaded 90th percentile bands
    ax.fill_between(bc, 0, pc, color=cc_col, alpha=0.10,
                    zorder=1)
    ax.fill_between(bc, 0, pk, color=cm_col, alpha=0.10,
                    zorder=1)

    # Mean curves
    ax.plot(bc, mc, "-", color=cc_col, lw=1.8, ms=3,
            marker="s", label="Coarse-FFT (mean)", zorder=3)
    ax.plot(bc, mk, "-", color=cm_col, lw=1.8, ms=3,
            marker="^", label="Composite MG (mean)", zorder=3)

    # 90th percentile curves (thinner, dashed)
    ax.plot(bc, pc, "--", color=cc_col, lw=0.9, alpha=0.7,
            label="Coarse-FFT (90th %ile)", zorder=2)
    ax.plot(bc, pk, "--", color=cm_col, lw=0.9, alpha=0.7,
            label="Composite MG (90th %ile)", zorder=2)

    # Boundary marker
    ax.axvline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.text(0.3, ax.get_ylim()[1] * 0.92, "boundary",
            fontsize=8, color="0.3", fontstyle="italic",
            va="top")

    ax.set_xlabel(
        "Distance from hole boundary into material (nm)",
        fontsize=10)
    ax.set_ylabel(
        r"$|\Delta B_x|/\max|B|$ (%)", fontsize=10)
    ax.set_xlim(0, max_dist)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.grid(axis="both", alpha=0.12)

    _save(fig, out_dir, "fig_error_radial")
    print(f"  Radial profile: {len(dist_mat)} material cells, "
          f"{n_bins} bins")


# ═══════════════════════════════════════════════════════════════════
#  Figure 5: Cost–accuracy scaling (Problem A, dz=3nm)
#  (a) Runtime vs N — all 5 methods (fine FFT, cfft, MG, Newell, PPPM)
#  (b) Edge RMSE vs N — cfft vs composite MG-only
# ═══════════════════════════════════════════════════════════════════
def plot_crossover(csv_path, out_dir):
    if not os.path.exists(csv_path):
        print(f"  SKIP crossover: {csv_path} not found"); return

    raw = np.atleast_1d(np.genfromtxt(csv_path, delimiter=",", names=True))
    keep = (raw["base_nx"] >= 32) & (raw["base_nx"] <= 768)
    bad_sizes = {48, 384}
    for bs in bad_sizes:
        keep = keep & (raw["base_nx"] != bs)
    data = raw[keep]
    names = data.dtype.names or ()

    base_nx = data["base_nx"].astype(int)
    fine_nx = data["fine_nx"].astype(int)
    N_fine  = fine_nx.astype(float) ** 2
    t_fine  = data["t_fine_ms"] / 1000.0
    t_cfft  = data["t_cfft_ms"] / 1000.0

    t_mg = _get_col(names, data, ["t_mg_ms", "t_comp_ms"])
    t_mg = t_mg / 1000.0 if t_mg is not None else np.zeros_like(t_fine)

    e_comp = _get_col(names, data, ["e_mg_pct", "e_comp_pct"])
    if e_comp is None: e_comp = np.full_like(t_fine, np.nan)

    e_cfft = data["e_cfft_pct"] if "e_cfft_pct" in names else np.full_like(t_fine, np.nan)

    t_newell = data["t_newell_ms"] / 1000.0 if "t_newell_ms" in names else None
    t_pppm   = data["t_pppm_ms"]   / 1000.0 if "t_pppm_ms"   in names else None

    # Colours
    cf = "#1F4E9A"    # fine FFT: dark blue
    cc = "#2E7D32"    # cfft: green
    cm = "#C03030"    # MG-only: dark red
    cn = "#7B1FA2"    # Newell: purple
    cp = "#EF6C00"    # PPPM: orange

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(5.2, 7.0))
    fig.subplots_adjust(hspace=0.35, left=0.15, right=0.95, top=0.88, bottom=0.08)

    # ═══════════════════════════════════════════
    #  Top panel: Runtime vs N — all 5 methods
    # ═══════════════════════════════════════════
    mf = t_fine > 0
    if np.any(mf):
        ax_top.loglog(N_fine[mf], t_fine[mf], "o-", color=cf, ms=4,
                      label="Fine FFT (uniform)", zorder=3)
    mc = t_cfft > 0
    if np.any(mc):
        ax_top.loglog(N_fine[mc], t_cfft[mc], "s-", color=cc, ms=4,
                      label="Coarse-FFT + AMR", zorder=2)

    mmg = np.isfinite(t_mg) & (t_mg > 0)
    if np.any(mmg):
        ax_top.loglog(N_fine[mmg], t_mg[mmg], "^-", color=cm, ms=4,
                      label="Composite MG-only", zorder=2)

    if t_newell is not None:
        mn = np.isfinite(t_newell) & (t_newell > 0)
        if np.any(mn):
            ax_top.loglog(N_fine[mn], t_newell[mn], "D-", color=cn, ms=3.5,
                          label="Composite + Newell", zorder=2)

    if t_pppm is not None:
        mp = np.isfinite(t_pppm) & (t_pppm > 0)
        if np.any(mp):
            ax_top.loglog(N_fine[mp], t_pppm[mp], "v-", color=cp, ms=3.5,
                          label="Composite + PPPM", zorder=2)

    # Speedup annotations (MG-only vs fine FFT)
    if np.any(mf) and np.any(mmg):
        m = mf & mmg
        for ni, ti, tc in zip(N_fine[m], t_fine[m], t_mg[m]):
            sp = ti / tc
            if sp > 1.5:
                ax_top.annotate(f"{sp:.0f}" + r"$\times$",
                    (ni, tc), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=8.5, color=cm, fontstyle="italic")

    # N log N reference slope
    if np.any(mf) and int(np.sum(mf)) >= 2:
        Nr, tr = float(N_fine[mf][-1]), float(t_fine[mf][-1])
        Nl = np.logspace(np.log10(float(N_fine.min()) * 0.5),
                         np.log10(float(N_fine.max()) * 2.0), 60)
        c_fit = tr / (Nr * np.log(Nr))
        ax_top.plot(Nl, c_fit * Nl * np.log(Nl), "--",
                    color="0.6", lw=0.8, label=r"$\propto N\log N$", zorder=1)

    ax_top.set_ylabel("Wall-clock time (s)")
    ax_top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=8,
                  ncol=3, frameon=True, columnspacing=1.0, handletextpad=0.4)

    nmin = int(np.sqrt(float(N_fine.min())))
    nmax = int(np.sqrt(float(N_fine.max())))
    ax_top.set_xlabel(
        rf"$N = n^2$ fine-equivalent cells ($n = {nmin}$ to ${nmax}$)")

    # ═══════════════════════════════════════════
    #  Bottom panel: Edge RMSE vs N
    #  (cfft vs composite MG-only only)
    #  Threshold annotations to be added in Adobe
    # ═══════════════════════════════════════════
    me = np.isfinite(e_cfft)
    if np.any(me):
        ax_bot.semilogx(N_fine[me], e_cfft[me], "s-", color=cc, ms=4,
                        label="Coarse-FFT + AMR", zorder=2)
    mv2 = np.isfinite(e_comp)
    if np.any(mv2):
        ax_bot.semilogx(N_fine[mv2], e_comp[mv2], "^-", color=cm, ms=4,
                        label="Composite MG + AMR", zorder=2)

    ax_bot.set_xlabel(
        rf"$N = n^2$ fine-equivalent cells ($n = {nmin}$ to ${nmax}$)")
    ax_bot.set_ylabel("Edge RMSE (%)")
    ax_bot.set_ylim(bottom=0)
    ax_bot.legend(loc="upper right", fontsize=7.5)

    _save(fig, out_dir, "fig5_crossover")


# ═══════════════════════════════════════════════════════════════════
#  Figure 5b alt (i): N_eff / N_fine — AMR cell efficiency
#  Shows what fraction of the uniform fine grid the composite method
#  actually computes on.  Pairs with 5a to explain the speedup.
# ═══════════════════════════════════════════════════════════════════
def plot_neff_ratio(csv_path, out_dir,
                    hole_r_nm=100.0, domain_nm=500.0, amr_levels=3, ratio=2):
    if not os.path.exists(csv_path):
        print(f"  SKIP N_eff ratio: {csv_path} not found"); return

    raw = np.atleast_1d(np.genfromtxt(csv_path, delimiter=",", names=True))
    keep = (raw["base_nx"] >= 32) & (raw["base_nx"] <= 768)
    for bs in (48, 384):
        keep = keep & (raw["base_nx"] != bs)
    data = raw[keep]

    base_nx = data["base_nx"].astype(float)
    fine_nx = data["fine_nx"].astype(float)
    N_fine = fine_nx ** 2
    N_L0   = base_nx ** 2

    # Estimate N_eff = L0 cells + patch cells at all AMR levels.
    # Patches form a ring around the hole boundary at each level.
    # Ring width ≈ 8 cells at L1 resolution (ghost + indicator buffer),
    # narrowing at deeper levels due to tighter clustering.
    circumference_nm = 2.0 * np.pi * hole_r_nm
    N_patch = np.zeros_like(base_nx)
    for lev in range(1, amr_levels + 1):
        dx_lev = domain_nm / base_nx / (ratio ** lev)
        ring_width_cells = 8.0 / (ratio ** (lev - 1))  # tighter at deeper levels
        ring_width_nm = ring_width_cells * domain_nm / base_nx
        ring_area_nm2 = circumference_nm * ring_width_nm
        cells_at_lev = ring_area_nm2 / (dx_lev ** 2)
        N_patch += cells_at_lev

    N_eff = N_L0 + N_patch
    eff_ratio = N_eff / N_fine

    cm = "#C03030"

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.18, top=0.92)

    ax.semilogx(N_fine, eff_ratio * 100.0, "^-", color=cm, ms=5, lw=1.5,
                zorder=3)

    # Reference: base grid only (lower bound)
    base_ratio = N_L0 / N_fine * 100.0
    ax.semilogx(N_fine, base_ratio, ":", color="0.5", lw=1.0,
                label=rf"$L_0$ only ($1/{ratio**(2*amr_levels)}$ = "
                      rf"{100.0/ratio**(2*amr_levels):.1f}%)", zorder=1)

    ax.semilogx(N_fine, eff_ratio * 100.0, "^-", color=cm, ms=5, lw=1.5,
                label=r"$L_0$ + AMR patches ($N_{\mathrm{eff}}$)", zorder=3)

    # Annotate a representative point
    mid = len(N_fine) // 2
    ax.annotate(f"{eff_ratio[mid]*100:.1f}% of fine grid",
                (N_fine[mid], eff_ratio[mid] * 100.0),
                textcoords="offset points", xytext=(12, 8),
                fontsize=8, color=cm,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))

    nmin = int(np.sqrt(float(N_fine.min())))
    nmax = int(np.sqrt(float(N_fine.max())))
    ax.set_xlabel(
        rf"$N = n^2$ fine-equivalent cells ($n = {nmin}$ to ${nmax}$)")
    ax.set_ylabel(r"$N_{\mathrm{eff}} \,/\, N_{\mathrm{fine}}$ (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(axis="both", alpha=0.15)

    _save(fig, out_dir, "fig5b_neff_ratio")


# ═══════════════════════════════════════════════════════════════════
#  Figure 5b alt (ii): Edge RMSE vs N — fixed 4096² reference
#  Both fine FFT and composite compared to the SAME absolute reference
#  (the finest fine FFT in the sweep).
#
#  Requires: e_fine_abs_pct column in CSV (fine FFT at n² vs finest).
#  If not present, plots composite only and notes the missing curve.
# ═══════════════════════════════════════════════════════════════════
def plot_edge_vs_fixed_ref(csv_path, out_dir):
    if not os.path.exists(csv_path):
        print(f"  SKIP fixed-ref: {csv_path} not found"); return

    raw = np.atleast_1d(np.genfromtxt(csv_path, delimiter=",", names=True))
    keep = (raw["base_nx"] >= 32) & (raw["base_nx"] <= 768)
    for bs in (48, 384):
        keep = keep & (raw["base_nx"] != bs)
    data = raw[keep]
    names = data.dtype.names or ()

    fine_nx = data["fine_nx"].astype(float)
    N_fine = fine_nx ** 2

    # Composite edge RMSE vs local fine FFT reference
    # (approximately = vs absolute reference, since local fine FFT is
    # well-converged for this geometry)
    e_comp = _get_col(names, data, ["e_mg_pct", "e_comp_pct"])
    if e_comp is None:
        print("  SKIP fixed-ref: no composite accuracy column"); return

    # Fine FFT discretisation error vs fixed reference
    # This is e_fine_abs_pct if available (benchmark computes it when
    # a fixed reference file is provided via LLG_CV_FIXED_REF)
    e_fine_abs = data["e_fine_abs_pct"] if "e_fine_abs_pct" in names else None
    has_fine_abs = False
    if e_fine_abs is not None:
        has_fine_abs = bool(np.any(np.isfinite(e_fine_abs) & (e_fine_abs > 0)))

    cf = "#1F4E9A"
    cm = "#C03030"

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.18, top=0.90)

    valid_c = np.isfinite(e_comp) & (e_comp > 0)
    if np.any(valid_c):
        ax.semilogx(N_fine[valid_c], e_comp[valid_c], "^-", color=cm, ms=5,
                     label="Composite MG + AMR", zorder=3)

    if has_fine_abs and e_fine_abs is not None:
        valid_f = np.isfinite(e_fine_abs) & (e_fine_abs > 0)
        if np.any(valid_f):
            ax.semilogx(N_fine[valid_f], e_fine_abs[valid_f], "o-", color=cf,
                         ms=4, label="Fine FFT (uniform)", zorder=2)
    else:
        # No fine FFT absolute data — add a note
        ax.text(0.5, 0.92,
                "Fine FFT discretisation curve requires\n"
                "fixed-reference benchmark (LLG_CV_FIXED_REF)",
                transform=ax.transAxes, fontsize=7.5, color="0.4",
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          ec="0.7", lw=0.5))

    nmin = int(np.sqrt(float(N_fine.min())))
    nmax = int(np.sqrt(float(N_fine.max())))
    ax.set_xlabel(
        rf"$N = n^2$ fine-equivalent cells ($n = {nmin}$ to ${nmax}$)")
    ax.set_ylabel("Edge RMSE vs fixed ref. (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax.set_title("Convergence to absolute reference", fontsize=10, pad=6)

    _save(fig, out_dir, "fig5b_fixed_ref")
    if not has_fine_abs:
        print("  NOTE: fig5b_fixed_ref needs fine FFT absolute errors.")
        print("        Run sweep with LLG_CV_FIXED_REF=path/to/4096_ref.bin")


# ═══════════════════════════════════════════════════════════════════
#  Figure 5b: Decomposed N_eff / N_fine — AMR cell efficiency
#
#  Shows what fraction of the uniform fine grid the composite method
#  operates on, broken down by AMR level.  Pairs with 5(a) runtime
#  to explain the speedup mechanism.
#
#  Input: cell_count_sweep.csv from --cell-count-sweep benchmark mode.
#  Optional: crossover_sweep_3nm.csv for edge RMSE annotations.
# ═══════════════════════════════════════════════════════════════════
def plot_neff_decomposed(cells_csv, out_dir, acc_csv=None):
    if not cells_csv or not os.path.exists(cells_csv):
        print(f"  SKIP neff_decomposed: {cells_csv} not found")
        print(f"       Run: cargo run --release --bin bench_composite_vcycle -- --cell-count-sweep")
        return

    raw = np.atleast_1d(np.genfromtxt(cells_csv, delimiter=",", names=True))
    names = raw.dtype.names or ()

    base_nx  = raw["base_nx"].astype(int)
    fine_nx  = raw["fine_nx"].astype(int)
    N_fine   = raw["N_fine"].astype(float)
    cells_L0 = raw["cells_L0"].astype(float)
    cells_L1 = raw["cells_L1"].astype(float)
    cells_L2 = raw["cells_L2"].astype(float) if "cells_L2" in names else np.zeros_like(cells_L0)
    cells_L3 = raw["cells_L3"].astype(float) if "cells_L3" in names else np.zeros_like(cells_L0)
    N_eff    = raw["N_eff"].astype(float)

    # As fractions of N_fine (%)
    f_L0    = cells_L0 / N_fine * 100.0
    f_L1    = cells_L1 / N_fine * 100.0
    f_L2    = cells_L2 / N_fine * 100.0
    f_L3    = cells_L3 / N_fine * 100.0
    f_total = N_eff / N_fine * 100.0

    # Load accuracy data for annotations (optional)
    acc = {}
    if acc_csv and os.path.exists(acc_csv):
        acc_raw = np.atleast_1d(np.genfromtxt(acc_csv, delimiter=",", names=True))
        acc_names = acc_raw.dtype.names or ()
        for row in acc_raw:
            bnx = int(row["base_nx"])
            for col in ["e_mg_pct", "e_comp_pct"]:
                if col in acc_names:
                    val = float(row[col])
                    if np.isfinite(val) and val > 0:
                        acc[bnx] = val
                        break

    # Colours
    c_total = "#C03030"   # bold red — total N_eff
    c_L0    = "0.50"      # grey — base grid (constant)
    c_L1    = "#E8A040"   # warm yellow-orange — L1 patches
    c_L2    = "#3AA03A"   # green — L2 patches
    c_L3    = "#4080C8"   # blue — L3 patches

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    fig.subplots_adjust(left=0.14, right=0.95, bottom=0.17, top=0.85)

    # Per-level curves
    ax.semilogx(N_fine, f_L0, ":", color=c_L0, lw=1.2, zorder=1,
                label=r"$L_0$ base grid ($1/64$)")
    ax.semilogx(N_fine, f_L1, "-", color=c_L1, lw=1.2, marker="v", ms=4, zorder=2,
                label=r"$L_1$ patches")
    if np.any(f_L2 > 0):
        ax.semilogx(N_fine, f_L2, "-", color=c_L2, lw=1.0, marker="D", ms=3.5, zorder=2,
                    label=r"$L_2$ patches")
    if np.any(f_L3 > 0):
        ax.semilogx(N_fine, f_L3, "-", color=c_L3, lw=0.9, marker="s", ms=3, zorder=2,
                    label=r"$L_3$ patches")

    # Total N_eff (bold, on top)
    ax.semilogx(N_fine, f_total, "^-", color=c_total, lw=1.8, ms=6, zorder=5,
                label=r"Total $N_{\mathrm{eff}}$")

    # Annotations to be added in Adobe post-production.

    nmin = int(fine_nx.min())
    nmax = int(fine_nx.max())
    ax.set_xlabel(
        rf"$N = n^2$ fine-equivalent cells ($n = {nmin}$ to ${nmax}$)")
    ax.set_ylabel(r"Fraction of fine grid (%)")
    ax.set_ylim(bottom=0, top=max(f_total) * 1.3)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.grid(axis="both", alpha=0.12)

    ax.legend(loc="upper center", fontsize=8.5, ncol=3,
              columnspacing=1.0, handletextpad=0.4)

    _save(fig, out_dir, "fig5b_neff_decomposed")


# ═══════════════════════════════════════════════════════════════════
#  Figure 6: Component-resolved bar chart (Problem B)
#  Single panel. Bz bars wider + bolder. Annotated.
# ═══════════════════════════════════════════════════════════════════
def plot_diagnostic(csv_path_20nm, out_dir, target_nx=96):
    methods = ["MG-only", "Newell", "PPPM"]

    # Default values (will be overridden if data available)
    bx = [7.39, 7.35, 6.10]
    by = [5.28, 5.28, 3.55]
    bz = [8.86, 5.09, 8.58]
    timing_str = "3.0 s, 3.1 s, 3.1 s"

    # Try component_errors.csv first
    supp = os.path.join(os.path.dirname(csv_path_20nm), "component_errors.csv")
    if os.path.exists(supp):
        with open(supp) as f:
            rows = list(csv_mod.DictReader(f))
        if len(rows) >= 3:
            bx = [float(r["bx_pct"]) for r in rows]
            by = [float(r["by_pct"]) for r in rows]
            bz = [float(r["bz_pct"]) for r in rows]
            print(f"  Fig 6: component data from {supp}")
    elif os.path.exists(csv_path_20nm):
        rows = read_csv(csv_path_20nm)
        tr = [r for r in rows if int(r.get("base_nx", 0)) == target_nx]
        if tr:
            r = tr[0]
            bx_vals = [r.get(k, float("nan")) for k in
                       ("bx_mg_pct", "bx_newell_pct", "bx_pppm_pct")]
            if all(np.isfinite(v) for v in bx_vals):
                bx = bx_vals
            by_vals = [r.get(k, float("nan")) for k in
                       ("by_mg_pct", "by_newell_pct", "by_pppm_pct")]
            if all(np.isfinite(v) for v in by_vals):
                by = by_vals
            bz_vals = [r.get(k, float("nan")) for k in
                       ("bz_mg_pct", "bz_newell_pct", "bz_pppm_pct")]
            if all(np.isfinite(v) for v in bz_vals):
                bz = bz_vals
            tvals = [r.get(k, float("nan")) for k in
                     ("t_mg_ms", "t_newell_ms", "t_pppm_ms")]
            if all(np.isfinite(v) for v in tvals):
                timing_str = ", ".join(f"{v/1000:.1f} s" for v in tvals)
        print(f"  Fig 6: all components from sweep at L0={target_nx}")
    else:
        print(f"  Fig 6: using hardcoded defaults")

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    fig.subplots_adjust(left=0.13, right=0.95, bottom=0.20, top=0.88)

    x = np.arange(len(methods))
    w_xy = 0.17
    w_z  = 0.28

    c_bx = "#90CAF9"
    c_by = "#A5D6A7"
    c_bz = "#E53935"

    off_bx = -w_z / 2 - w_xy - 0.02
    off_by = -w_z / 2 - 0.01
    off_bz = w_xy / 2 + 0.02

    bars_bx = ax.bar(x + off_bx, bx, w_xy, label="$B_x$",
                     color=c_bx, edgecolor="0.6", lw=0.4, alpha=0.85)
    bars_by = ax.bar(x + off_by, by, w_xy, label="$B_y$",
                     color=c_by, edgecolor="0.6", lw=0.4, alpha=0.85)
    bars_bz = ax.bar(x + off_bz, bz, w_z,  label=r"$\mathbf{B_z}$",
                     color=c_bz, edgecolor="0.3", lw=0.7, zorder=3)

    for bars, fs, fw in [(bars_bx, 6.5, "normal"), (bars_by, 6.5, "normal"),
                          (bars_bz, 8.5, "bold")]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.12,
                        f"{h:.1f}", ha="center", va="bottom",
                        fontsize=fs, fontweight=fw)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel(r"RMSE (% of $|B|_{\mathrm{max}}$)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=3)
    ymax = max(max(bx), max(by), max(bz)) * 1.35
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.2)

    ax.text(0.5, -0.14,
            f"Runtime at $L_0 = {target_nx}$: {timing_str} (all equivalent)",
            transform=ax.transAxes, fontsize=8, color="0.4",
            ha="center", va="top")

    ax.set_title(
        r"Component-resolved accuracy"
        r" (300 nm domain, dz = 20 nm, $\mathbf{M}$ tilted 45$°$ $xz$)",
        fontsize=9.5, pad=8)

    _save(fig, out_dir, "fig6_diagnostic")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="§4.1 antidot benchmark figures")
    parser.add_argument("--dir", default="out/bench_vcycle_diag",
        help="Directory with single-run CSVs (patch_map, error_map)")
    parser.add_argument("--out", default=None,
        help="Output directory (default: same as --dir)")
    parser.add_argument("--sweep-3nm", default=None,
        help="CSV for dz=3nm MG-only sweep (Figure 5)")
    parser.add_argument("--sweep-20nm", default=None,
        help="CSV for dz=20nm three-method sweep (Figure 6)")
    parser.add_argument("--hole-r", type=float, default=100.0)
    parser.add_argument("--domain", type=float, default=500.0)
    parser.add_argument("--base-nx", type=int, default=256)
    parser.add_argument("--diag-nx", type=int, default=96,
        help="L0 for Figure 6 bar chart (default: 96)")
    parser.add_argument("--cell-count", default=None,
        help="CSV from --cell-count-sweep mode (Figure 5b decomposed)")
    args = parser.parse_args()

    diag_dir = args.dir
    out_dir = args.out or diag_dir
    os.makedirs(out_dir, exist_ok=True)
    setup_style()

    print(f"Single-run data: {diag_dir}/")
    print(f"Output: {out_dir}/")
    print()

    plot_patch_map(diag_dir, out_dir, args.hole_r, args.domain, args.base_nx)
    plot_error_cmap(diag_dir, out_dir, args.hole_r, args.domain, args.base_nx)
    plot_error_radial(diag_dir, out_dir, args.hole_r, args.domain, args.base_nx)

    sweep_3nm = args.sweep_3nm
    if sweep_3nm is None:
        for name in ("crossover_sweep_3nm.csv", "crossover_sweep.csv"):
            candidate = os.path.join(diag_dir, name)
            if os.path.exists(candidate):
                sweep_3nm = candidate; break
        if sweep_3nm is None:
            sweep_3nm = os.path.join(diag_dir, "crossover_sweep_3nm.csv")
    plot_crossover(sweep_3nm, out_dir)
    plot_neff_ratio(sweep_3nm, out_dir, args.hole_r, args.domain)
    plot_edge_vs_fixed_ref(sweep_3nm, out_dir)

    # Fig 5b decomposed: real per-level cell counts
    cells_csv = args.cell_count
    if cells_csv is None:
        cells_csv = os.path.join(diag_dir, "cell_count_sweep.csv")
    plot_neff_decomposed(cells_csv, out_dir, acc_csv=sweep_3nm)

    sweep_20nm = args.sweep_20nm or os.path.join(diag_dir, "crossover_sweep_20nm.csv")
    plot_diagnostic(sweep_20nm, out_dir, target_nx=args.diag_nx)

    print("\nDone.")

if __name__ == "__main__":
    main()