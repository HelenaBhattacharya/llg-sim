#!/usr/bin/env python3
"""
scripts/plot_dmi_transition.py

Generate thesis figures from bench_dmi_vortex_transition output.
Produces Zhang et al. JMMM 630 (2025) Fig 1/2 style plots.

Usage:
    python3 scripts/plot_dmi_transition.py --root out/bench_dmi_transition
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11, "mathtext.fontset": "cm",
        "axes.linewidth": 0.8, "axes.labelsize": 12,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })


def load_m_fine(path):
    """Load m_fine_D*.csv → dict with 2D arrays and physical coords."""
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if data.size == 0:
        return None
    i = data[:, 0].astype(int)
    j = data[:, 1].astype(int)
    x_nm = data[:, 2]
    y_nm = data[:, 3]
    mx = data[:, 4]
    my = data[:, 5]
    mz = data[:, 6]
    nx = i.max() + 1
    ny = j.max() + 1
    MX = np.full((ny, nx), np.nan)
    MY = np.full((ny, nx), np.nan)
    MZ = np.full((ny, nx), np.nan)
    for k in range(len(i)):
        MX[j[k], i[k]] = mx[k]
        MY[j[k], i[k]] = my[k]
        MZ[j[k], i[k]] = mz[k]
    # Physical extent
    dx = x_nm[1] - x_nm[0] if len(x_nm) > 1 else 2.0
    return {'mx': MX, 'my': MY, 'mz': MZ, 'nx': nx, 'ny': ny,
            'x_nm': x_nm, 'y_nm': y_nm, 'dx_nm': abs(dx)}


def load_summary(path):
    """Load summary.csv → list of dicts."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(',')
        for line in f:
            vals = line.strip().split(',')
            if len(vals) >= len(header):
                row = {}
                for h, v in zip(header, vals):
                    try:
                        row[h] = float(v)
                    except ValueError:
                        row[h] = v
                rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────
# Figure 1: mz colourmaps (Zhang et al. Fig 1 style)
# ─────────────────────────────────────────────────────────────────

def plot_mz_panels(root, summary):
    """Multi-panel mz colourmap — one panel per D value."""
    setup_style()
    d_values = [r['D_mJm2'] for r in summary]
    n = len(d_values)
    if n == 0:
        print("  No data in summary.csv")
        return

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n + 0.8, 3.4), squeeze=False)
    axes = axes[0]

    for idx, (d_val, row) in enumerate(zip(d_values, summary)):
        ax = axes[idx]
        d_str = f"{d_val:.1f}".replace('.', 'p')
        m_file = os.path.join(root, f'm_fine_D{d_str}.csv')
        m = load_m_fine(m_file)
        if m is None:
            ax.text(0.5, 0.5, f'No data\nD={d_val}', transform=ax.transAxes,
                    ha='center', va='center')
            continue

        mz = m['mz']
        nx, ny = m['nx'], m['ny']
        dx = m['dx_nm']
        half = nx * dx / 2

        # Disk mask
        x_arr = np.arange(nx) * dx - half + dx / 2
        y_arr = np.arange(ny) * dx - half + dx / 2
        X, Y = np.meshgrid(x_arr, y_arr)
        R = np.sqrt(X**2 + Y**2)
        R_disk = 100.0  # nm
        mz_plot = np.where(R <= R_disk, mz, np.nan)

        im = ax.imshow(mz_plot, origin='lower',
                       extent=(-half, half, -half, half),
                       cmap='RdBu_r', vmin=-1, vmax=1,
                       interpolation='bilinear')

        # Quiver overlay — sparse arrows showing in-plane direction
        skip = max(1, nx // 16)
        X_q = X[::skip, ::skip]
        Y_q = Y[::skip, ::skip]
        U_q = m['mx'][::skip, ::skip]
        V_q = m['my'][::skip, ::skip]
        R_q = np.sqrt(X_q**2 + Y_q**2)
        mag = np.sqrt(U_q**2 + V_q**2)
        draw = (R_q <= R_disk * 0.92) & (mag > 0.15) & ~np.isnan(U_q)
        ax.quiver(X_q[draw], Y_q[draw], U_q[draw], V_q[draw],
                  color='k', scale=22, width=0.004, headwidth=3,
                  alpha=0.7, pivot='mid')

        # Disk outline
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(R_disk * np.cos(theta), R_disk * np.sin(theta),
                'k-', lw=1.0)

        Q = row.get('Q', float('nan'))
        ax.set_title(f'$D = {d_val:.1f}$ mJ/m²\n$Q = {Q:+.1f}$', fontsize=10)

        # Only show axis labels on leftmost panel
        if idx == 0:
            ax.set_ylabel('$y$ (nm)')
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('$x$ (nm)')

        # Zoom to disk
        lim = R_disk * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')

    # Shared colourbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='RdBu_r', norm=Normalize(-1, 1)),
        ax=axes.tolist(), shrink=0.75, pad=0.04, aspect=25)
    cbar.set_label('$m_z$')

    fig.tight_layout(w_pad=0.5)
    out = os.path.join(root, 'fig_dmi_mz_panels.pdf')
    fig.savefig(out, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 2: Q vs D (Zhang et al. Fig 2 style)
# ─────────────────────────────────────────────────────────────────

def plot_q_vs_d(root, summary):
    """Topological charge Q vs DMI strength D."""
    setup_style()
    if not summary:
        print("  No summary data for Q vs D")
        return

    d_vals = [r['D_mJm2'] for r in summary]
    q_vals = [r['Q'] for r in summary]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.plot(d_vals, q_vals, 'o-', color='#D62728', ms=8, lw=1.5,
            label='This work (200 nm disk)')

    # Reference lines for expected Q values
    ax.axhline(0.5, color='grey', ls='--', lw=0.6, alpha=0.5)
    ax.axhline(-0.5, color='grey', ls='--', lw=0.6, alpha=0.5)
    ax.axhline(-1.5, color='grey', ls='--', lw=0.6, alpha=0.5)

    # Labels
    ax.text(max(d_vals) * 0.95, 0.55, 'Vortex ($Q = +\\frac{1}{2}$)',
            fontsize=8, ha='right', color='grey')
    ax.text(max(d_vals) * 0.95, -0.45, 'DMI vortex ($Q = -\\frac{1}{2}$)',
            fontsize=8, ha='right', color='grey')
    ax.text(max(d_vals) * 0.95, -1.45, '1-skyrmion vortex ($Q = -\\frac{3}{2}$)',
            fontsize=8, ha='right', color='grey')

    ax.set_xlabel('$D$ (mJ/m²)')
    ax.set_ylabel('Topological charge $Q$')
    ax.set_ylim(-2.5, 1.5)
    ax.legend(fontsize=9, loc='lower left')

    fig.tight_layout()
    out = os.path.join(root, 'fig_dmi_Q_vs_D.pdf')
    fig.savefig(out, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 3: Individual mz 3D surfaces (one per D)
# ─────────────────────────────────────────────────────────────────

def plot_mz_3d_per_d(root, summary):
    """3D mz surface for each D value — shows core expansion."""
    setup_style()

    for row in summary:
        d_val = row['D_mJm2']
        d_str = f"{d_val:.1f}".replace('.', 'p')
        m_file = os.path.join(root, f'm_fine_D{d_str}.csv')
        m = load_m_fine(m_file)
        if m is None:
            continue

        mz = m['mz']
        nx, ny = m['nx'], m['ny']
        dx = m['dx_nm']
        half = nx * dx / 2

        x_arr = np.arange(nx) * dx - half + dx / 2
        y_arr = np.arange(ny) * dx - half + dx / 2
        X, Y = np.meshgrid(x_arr, y_arr)
        R = np.sqrt(X**2 + Y**2)
        R_disk = 100.0

        # Subsample for wireframe visibility
        step = max(1, nx // 100)
        X_s = X[::step, ::step]
        Y_s = Y[::step, ::step]
        mz_s = mz[::step, ::step]
        R_s = np.sqrt(X_s**2 + Y_s**2)
        mz_plot = np.where(R_s <= R_disk * 0.88, mz_s, np.nan)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        assert isinstance(ax, Axes3D)

        norm = Normalize(vmin=-1, vmax=1)
        cmap = plt.colormaps['RdBu_r']
        colours = cmap(norm(np.nan_to_num(mz_plot, nan=0)))
        colours[np.isnan(mz_plot), 3] = 0.0

        ax.plot_surface(X_s, Y_s, mz_plot, facecolors=colours,
                        rstride=1, cstride=1,
                        linewidth=0.08, edgecolor='0.45',
                        alpha=0.92, antialiased=True, shade=False)

        Q = row.get('Q', float('nan'))
        ax.set_title(f'$D = {d_val:.1f}$ mJ/m², $Q = {Q:+.2f}$', fontsize=11)
        ax.set_xlabel('$x$ (nm)', fontsize=9, labelpad=4)
        ax.set_ylabel('$y$ (nm)', fontsize=9, labelpad=4)
        ax.set_zlabel('$m_z$', fontsize=9, labelpad=2)
        ax.view_init(elev=22, azim=-50)
        ax.set_zlim(-1.05, 1.05)
        lim = R_disk * 0.92
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.tick_params(labelsize=7)

        try:
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis.pane.fill = False
                axis.pane.set_edgecolor((0.92, 0.92, 0.92, 0.2))
        except AttributeError:
            pass
        ax.grid(True, color='0.93', alpha=0.25, linewidth=0.4)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)

        fig.tight_layout()
        out = os.path.join(root, f'fig_dmi_mz_3d_D{d_str}.pdf')
        fig.savefig(out, dpi=250, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot DMI transition results')
    parser.add_argument('--root', default='out/bench_dmi_transition',
                        help='Output directory from bench_dmi_vortex_transition')
    args = parser.parse_args()
    root = args.root

    setup_style()
    print(f"Plotting DMI transition from {root}/")

    summary = load_summary(os.path.join(root, 'summary.csv'))
    if not summary:
        print("  ERROR: summary.csv empty or not found")
        return

    print(f"  Found {len(summary)} D values: {[r['D_mJm2'] for r in summary]}")
    for r in summary:
        print(f"    D={r['D_mJm2']:.1f} mJ/m²  Q={r['Q']:+.3f}  <mz>={r['mz_avg']:.4f}  core_mz={r['mz_core']:.4f}")

    plot_mz_panels(root, summary)
    plot_q_vs_d(root, summary)
    plot_mz_3d_per_d(root, summary)

    print("\nDone.")


if __name__ == '__main__':
    main()