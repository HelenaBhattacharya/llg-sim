#!/usr/bin/env python3
"""
scripts/plot_vortex_gyration.py

Generate thesis-quality figures from bench_vortex_gyration output CSVs.

Usage:
    python scripts/plot_vortex_gyration.py --root out/bench_vortex_gyration --mode comp
    python scripts/plot_vortex_gyration.py --root out/bench_vortex_gyration --mode cfft
    python scripts/plot_vortex_gyration.py --root out/bench_vortex_gyration --mode both

Mode controls which solver's data to plot:
    comp  — composite MG only (comp_* files)
    cfft  — coarse-FFT only (non-prefixed files)
    both  — overlay both on trajectory/frequency plots; generate separate
            snapshot/patch/mz plots for each solver

Generates:
    fig_trajectory.pdf      — Core X/R vs Y/R spiral (Guslienko Fig 2 style)
    fig_core_xt.pdf         — Core x(t) damped oscillation with envelope + annotations
    fig_frequency.pdf       — Frequency comparison scatter vs Guslienko + Novosad
    fig_patch_map_*.pdf     — Patch maps with core marker (per solver)
    fig_mz_eq_*.pdf         — mz colourmap at equilibrium (per solver)
    fig_mz_gyr_*.pdf        — mz colourmaps during gyration (per solver)
    fig_mz_3d_*.pdf         — 3D mz surface (multi-view)
    fig_mesh_full.pdf       — Full-domain mesh showing multi-resolution grid + disk boundary
    fig_mz_xsec_*.pdf      — 1D mz cross-sections (Novosad Fig 4 style)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import hsv_to_rgb, Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# ─────────────────────────────────────────────────────────────────
# Publication style (matches antidot benchmark)
# ─────────────────────────────────────────────────────────────────

def setup_style():
    """Serif-font, ticks-in style matching plot_antidot_benchmark."""
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
        "legend.fontsize": 9, "legend.framealpha": 0.92,
        "legend.edgecolor": "0.7", "legend.fancybox": False,
        "legend.handlelength": 1.8, "lines.linewidth": 1.5,
        "figure.dpi": 200, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })

import matplotlib.ticker as mticker

# ─────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────

def load_grid_info(root):
    """Load grid metadata from grid_info.csv."""
    info = {}
    path = os.path.join(root, 'grid_info.csv')
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, using defaults")
        return {'base_nx': 80, 'base_ny': 80, 'fine_nx': 640, 'fine_ny': 640,
                'dx_m': 3.75e-9, 'dy_m': 3.75e-9, 'dz_m': 20e-9,
                'disk_r_m': 100e-9, 'domain_m': 300e-9, 'amr_levels': 3, 'ratio': 2}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row['param']
            val = row['value']
            try:
                info[key] = int(val)
            except ValueError:
                try:
                    info[key] = float(val)
                except ValueError:
                    info[key] = val
    return info


def load_core_csv(path):
    """Load core trajectory CSV → (t_ns, x_nm, y_nm, mz, core_level)."""
    t, x, y, mz, cl = [], [], [], [], []
    if not os.path.exists(path):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row['t_ns']))
            x.append(float(row['x_nm']))
            y.append(float(row['y_nm']))
            mz.append(float(row['mz']))
            cl.append(int(row['core_level']))
    return np.array(t), np.array(x), np.array(y), np.array(mz), np.array(cl)


def extract_phase3(t, x, y, mz, cl, t_cut_ns=None):
    """Extract Phase 3 data from a multi-phase core CSV.

    Phase boundaries are detected by step-number resets (time decreases).
    Phase 3 is the last segment.  If t_cut_ns is given, data beyond that
    time (relative to Phase 3 start) is discarded.

    Returns (t_p3, x_p3, y_p3, mz_p3, cl_p3) — all numpy arrays with
    Phase 3 time starting from 0.
    """
    if len(t) == 0:
        return t, x, y, mz, cl

    # Find phase boundaries (where t decreases = step reset)
    resets = [0]
    for i in range(1, len(t)):
        if t[i] < t[i - 1] - 1e-6:
            resets.append(i)

    # Phase 3 is the last segment
    p3_start = resets[-1]
    t3, x3, y3, mz3, cl3 = t[p3_start:], x[p3_start:], y[p3_start:], mz[p3_start:], cl[p3_start:]

    # Apply time cutoff
    if t_cut_ns is not None and len(t3) > 0:
        mask = t3 <= t_cut_ns
        t3, x3, y3, mz3, cl3 = t3[mask], x3[mask], y3[mask], mz3[mask], cl3[mask]

    return t3, x3, y3, mz3, cl3


def get_phase2_end(t, x, y):
    """Get the core position at the end of Phase 2 (start of gyration).

    Returns (x_nm, y_nm) at the last point before Phase 3.
    """
    if len(t) == 0:
        return 0.0, 0.0

    resets = []
    for i in range(1, len(t)):
        if t[i] < t[i - 1] - 1e-6:
            resets.append(i)

    if len(resets) >= 2:
        # Phase 2 ends just before the last reset
        idx = resets[-1] - 1
        return float(x[idx]), float(y[idx])
    elif len(resets) == 1:
        idx = resets[0] - 1
        return float(x[idx]), float(y[idx])
    else:
        return float(x[0]), float(y[0])


def load_patches_csv(path):
    """Load patch rectangles → list of (level, i0, j0, nx, ny)."""
    patches = []
    if not os.path.exists(path):
        return patches
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            patches.append((int(row['level']), int(row['i0']), int(row['j0']),
                            int(row['nx']), int(row['ny'])))
    return patches


def load_m_csv(path):
    """Load magnetisation CSV → (i, j, mx, my, mz) as structured arrays."""
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if data.size == 0:
        return None
    i = data[:, 0].astype(int)
    j = data[:, 1].astype(int)
    mx = data[:, 2]
    my = data[:, 3]
    mz_arr = data[:, 4]
    nx = i.max() + 1
    ny = j.max() + 1
    MX = np.zeros((ny, nx))
    MY = np.zeros((ny, nx))
    MZ = np.zeros((ny, nx))
    for k in range(len(i)):
        MX[j[k], i[k]] = mx[k]
        MY[j[k], i[k]] = my[k]
        MZ[j[k], i[k]] = mz_arr[k]
    return {'mx': MX, 'my': MY, 'mz': MZ, 'nx': nx, 'ny': ny}


def estimate_frequency(t_ns, x_nm):
    """Estimate gyration frequency via zero-crossing analysis.
    Works with 2+ crossings: consecutive crossings give half-periods."""
    crossings = []
    for i in range(1, len(x_nm)):
        if x_nm[i-1] * x_nm[i] < 0 and x_nm[i-1] != 0:
            frac = abs(x_nm[i-1]) / (abs(x_nm[i-1]) + abs(x_nm[i]))
            tc = t_ns[i-1] + frac * (t_ns[i] - t_ns[i-1])
            crossings.append(tc)
    if len(crossings) < 2:
        return 0.0, len(crossings)
    # Each consecutive pair of crossings is a half-period
    half_periods = [crossings[i+1] - crossings[i] for i in range(len(crossings) - 1)]
    avg_period = 2.0 * np.mean(half_periods)
    return 1.0 / avg_period, len(crossings)


def smooth_trajectory(x, window=51, order=3):
    """Savitzky-Golay smooth, handling short arrays gracefully."""
    if len(x) < window:
        return x.copy()
    return savgol_filter(x, window, order)


def fit_envelope(t_ns, x_nm):
    """Fit exponential decay envelope A(t) = A0 * exp(-t/tau) to peak amplitudes."""
    # Find local peaks (maxima of |x|)
    from scipy.signal import argrelextrema
    # Use smoothed signal for peak finding
    x_smooth = smooth_trajectory(x_nm, window=101, order=3) if len(x_nm) > 101 else x_nm
    abs_x = np.abs(x_smooth)

    # Find peaks of the absolute value
    peak_idx = argrelextrema(abs_x, np.greater, order=50)[0]
    if len(peak_idx) < 3:
        return None, None, None

    t_peaks = t_ns[peak_idx]
    a_peaks = abs_x[peak_idx]

    # Fit A0 * exp(-t/tau)
    def exp_decay(t, A0, tau):
        return A0 * np.exp(-t / tau)

    try:
        popt, _ = curve_fit(exp_decay, t_peaks, a_peaks, p0=[a_peaks[0], 5.0],
                            maxfev=5000)
        A0, tau = popt
        return A0, tau, exp_decay
    except Exception:
        return None, None, None


# ─────────────────────────────────────────────────────────────────
# Disk mask helper
# ─────────────────────────────────────────────────────────────────

def make_disk_mask(nx, ny, info, is_fine=False):
    """Create boolean mask: True inside disk, False outside."""
    if is_fine:
        ratio_total = info['ratio'] ** info['amr_levels']
        dx_plot = info['dx_m'] / ratio_total
    else:
        dx_plot = info['dx_m']

    cx = nx / 2.0
    cy = ny / 2.0
    R_pix = info['disk_r_m'] / dx_plot

    jj, ii = np.mgrid[0:ny, 0:nx]
    dist = np.sqrt((ii - cx + 0.5)**2 + (jj - cy + 0.5)**2)
    return dist <= R_pix


# ─────────────────────────────────────────────────────────────────
# Colour map: in-plane angle → HSV colour wheel (matching Rust)
# ─────────────────────────────────────────────────────────────────

def angle_colormap(mx, my, mz, disk_mask=None):
    """Convert (mx, my, mz) arrays to RGB image using HSV colour wheel.
    Cells outside disk_mask are set to white."""
    angle = np.arctan2(my, mx)  # -π to π
    hue = (angle + np.pi) / (2 * np.pi)  # 0 to 1

    sat = np.ones_like(hue)
    val = np.ones_like(hue)

    # Desaturate where |mz| is large (core region)
    high_mz = np.abs(mz) > 0.8
    sat[high_mz] = 1.0 - (np.abs(mz[high_mz]) - 0.8) / 0.2
    sat = np.clip(sat, 0, 1)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)

    # Override high-mz cells with blue-white-red diverging
    for idx in np.argwhere(high_mz):
        j, i = idx
        t = (mz[j, i] + 1) / 2  # 0 to 1
        if t < 0.5:
            a = t / 0.5
            rgb[j, i] = [a, a, 1.0]  # blue to white
        else:
            a = (t - 0.5) / 0.5
            rgb[j, i] = [1.0, 1-a, 1-a]  # white to red

    # Mask outside disk → white
    if disk_mask is not None:
        rgb[~disk_mask] = [1.0, 1.0, 1.0]

    return rgb


def draw_colour_wheel_inset(ax, pos=(0.78, 0.02), size=0.18):
    """Draw an HSV colour-wheel inset showing angle-to-colour mapping."""
    # Create inset axes
    inset = ax.inset_axes([pos[0], pos[1], size, size])
    n = 256
    theta = np.linspace(0, 2*np.pi, n)
    r = np.linspace(0, 1, n//2)
    T, R = np.meshgrid(theta, r)

    hue = (T + np.pi) / (2 * np.pi)
    hue = hue % 1.0
    sat = R
    val = np.ones_like(R)
    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = hsv_to_rgb(hsv)

    # Plot in polar-like manner using imshow on a circular mask
    x = R * np.cos(T)
    y = R * np.sin(T)

    # Simple approach: render onto a square grid
    grid_n = 128
    img = np.ones((grid_n, grid_n, 3))
    for gi in range(grid_n):
        for gj in range(grid_n):
            gx = (gj / (grid_n - 1)) * 2 - 1
            gy = (gi / (grid_n - 1)) * 2 - 1
            gr = np.sqrt(gx**2 + gy**2)
            if gr <= 1.0:
                ga = np.arctan2(gy, gx)
                gh = (ga + np.pi) / (2 * np.pi)
                gs = gr
                gv = 1.0
                h_rgb = hsv_to_rgb(np.array([[[gh, gs, gv]]]))[0, 0]
                img[grid_n - 1 - gi, gj] = h_rgb

    inset.imshow(img, extent=(-1, 1, -1, 1), interpolation='bilinear')
    inset.set_xlim(-1.15, 1.15)
    inset.set_ylim(-1.15, 1.15)

    # Add direction labels
    fs = 7
    inset.text(1.1, 0, '+x', ha='left', va='center', fontsize=fs, fontweight='bold')
    inset.text(-1.1, 0, '−x', ha='right', va='center', fontsize=fs, fontweight='bold')
    inset.text(0, 1.1, '+y', ha='center', va='bottom', fontsize=fs, fontweight='bold')
    inset.text(0, -1.1, '−y', ha='center', va='top', fontsize=fs, fontweight='bold')

    inset.set_aspect('equal')
    inset.axis('off')
    return inset


# ─────────────────────────────────────────────────────────────────
# Snapshot time helper
# ─────────────────────────────────────────────────────────────────

def get_snapshot_time(root, info, snap_idx):
    """Get the physical time for a gyration snapshot index.
    
    In the bench code, sn=0 is written at step=snap_every (the first multiple),
    so patches_000.csv corresponds to t=snap_every*dt, not t=0.
    Hence snap_idx → (snap_idx + 1) * snap_interval.
    """
    snap_every_s = 200e-12  # 200ps between snapshots
    t_ns = (snap_idx + 1) * snap_every_s * 1e9
    return t_ns


def get_core_at_time(root, t_target_ns, method='amr_cfft'):
    """Get core position at a specific Phase 3 time from the trajectory CSV."""
    csv_name = f'core_{method}.csv'
    t_raw, x_raw, y_raw, mz_raw, cl_raw = load_core_csv(os.path.join(root, csv_name))
    if len(t_raw) == 0:
        return None, None
    # Extract Phase 3 only
    t, x, y, _, _ = extract_phase3(t_raw, x_raw, y_raw, mz_raw, cl_raw)
    if len(t) == 0:
        return None, None
    idx = np.argmin(np.abs(t - t_target_ns))
    return x[idx], y[idx]


def get_core_at_snap(root, info, snap_idx, method='amr_cfft'):
    """Get core position for a given snapshot index."""
    t_ns = get_snapshot_time(root, info, snap_idx)
    return get_core_at_time(root, t_ns, method)


# ─────────────────────────────────────────────────────────────────
# Figure 1: Core Trajectory X/R vs Y/R (centred on orbit)
# ─────────────────────────────────────────────────────────────────

def plot_trajectory(root, info, methods, extra_core=None, t_cut_ns=None):
    """Core trajectory X/R vs Y/R with dot-series time evolution.
    Professional styling: dots at regular intervals coloured by time,
    connected by thin line. Disk boundary shown as circle."""
    setup_style()
    R_nm = info['disk_r_m'] * 1e9

    fig, ax = plt.subplots(figsize=(3.8, 4.2))

    # Disk boundary
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), '-', color='#CC6600', lw=1.0,
            alpha=0.5, zorder=1)

    # Extra core trajectories (fine/coarse) as thin background lines
    if extra_core:
        for csv_name, color, label in extra_core:
            t_raw, x_raw, y_raw, mz_raw, cl_raw = load_core_csv(os.path.join(root, csv_name))
            if len(t_raw) == 0:
                continue
            t, x, y, _, _ = extract_phase3(t_raw, x_raw, y_raw, mz_raw, cl_raw, t_cut_ns)
            if len(t) == 0:
                continue
            ax.plot(x / R_nm, y / R_nm, color=color, lw=0.6, alpha=0.4, label=label)

    # Plot each method with dot-series time colouring
    for core_csv, prefix, label, color in methods:
        t_raw, x_raw, y_raw, mz_raw, cl_raw = load_core_csv(os.path.join(root, core_csv))
        if len(t_raw) == 0:
            continue

        t, x, y, _, _ = extract_phase3(t_raw, x_raw, y_raw, mz_raw, cl_raw, t_cut_ns)
        if len(t) == 0:
            continue

        p2_x, p2_y = get_phase2_end(t_raw, x_raw, y_raw)
        xn, yn = x / R_nm, y / R_nm

        # Thin connecting line (faint)
        ax.plot(xn, yn, '-', color=color, lw=0.3, alpha=0.2, zorder=2)

        # Dots at regular intervals coloured by time
        dt_target = 0.015  # ns (~15ps between dots)
        if len(t) > 1:
            dt_data = np.median(np.diff(t))
            skip = max(1, int(dt_target / dt_data))
        else:
            skip = 1
        t_dots = t[::skip]
        x_dots = xn[::skip]
        y_dots = yn[::skip]

        sc = ax.scatter(x_dots, y_dots, c=t_dots, cmap='YlOrBr',
                        s=6, edgecolors='none', zorder=4,
                        vmin=t.min(), vmax=t.max())

        # Start marker (filled circle)
        p2_xn, p2_yn = p2_x / R_nm, p2_y / R_nm
        ax.plot(p2_xn, p2_yn, 'o', color='k', ms=5, zorder=6)
        ax.annotate(r'$t\!=\!0$', xy=(p2_xn, p2_yn), xytext=(6, 6),
                    textcoords='offset points', fontsize=8,
                    arrowprops=dict(arrowstyle='-', color='0.4', lw=0.5))

        # End marker (square)
        ax.plot(xn[-1], yn[-1], 's', color='0.3', ms=4, zorder=6)

        # Colourbar
        cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.02)
        cbar.set_label(r'$t$ (ns)', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    # Axes
    ax.set_xlabel(r'$X / R$')
    ax.set_ylabel(r'$Y / R$')
    ax.set_aspect('equal')
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # Auto-centre on trajectory data
    all_x, all_y = [], []
    for core_csv, prefix, label, colour in methods:
        t_raw, x_raw, y_raw, mz_raw, cl_raw = load_core_csv(os.path.join(root, core_csv))
        if len(t_raw) > 0:
            t, x, y, _, _ = extract_phase3(t_raw, x_raw, y_raw, mz_raw, cl_raw, t_cut_ns)
            if len(t) > 0:
                all_x.extend(x / R_nm); all_y.extend(y / R_nm)
    if all_x:
        cx = (min(all_x) + max(all_x)) / 2
        cy = (min(all_y) + max(all_y)) / 2
        hw = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2 * 1.4
        hw = max(hw, 0.18)
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(cy - hw, cy + hw)

    fig.tight_layout()
    out = os.path.join(root, 'fig_trajectory.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 2: Core x(t) — Damped Oscillation with envelope
# ─────────────────────────────────────────────────────────────────

def plot_core_xt(root, info, methods, extra_core=None, t_cut_ns=None):
    """Core x(t) with Savitzky-Golay smooth, exponential envelope, and annotations.
    Uses Phase 3 data only."""
    fig, ax = plt.subplots(figsize=(9, 4))

    # Build combined list of (csv_name, color, label)
    plot_list = []
    if extra_core:
        for csv_name, color, label in extra_core:
            plot_list.append((csv_name, color, label))
    for core_csv, prefix, label, color in methods:
        plot_list.append((core_csv, color, label))

    for name, color, label in plot_list:
        t_raw, x_raw, y_raw, mz_raw, cl_raw = load_core_csv(os.path.join(root, name))
        if len(t_raw) == 0:
            continue

        t, x, _, _, _ = extract_phase3(t_raw, x_raw, y_raw, mz_raw, cl_raw, t_cut_ns)
        if len(t) == 0:
            continue

        # Raw data as faint background
        ax.plot(t, x, color=color, lw=0.4, alpha=0.25)

        # Smoothed
        x_smooth = smooth_trajectory(x)
        ax.plot(t, x_smooth, color=color, lw=1.4, label=label)

        # Fit and plot exponential envelope
        A0, tau, env_func = fit_envelope(t, x)
        if A0 is not None and tau is not None and tau > 0 and env_func is not None:
            t_env = np.linspace(t.min(), t.max(), 200)
            env_upper = env_func(t_env, A0, tau)
            ax.plot(t_env, env_upper, '--', color=color, lw=1.0, alpha=0.7)
            ax.plot(t_env, -env_upper, '--', color=color, lw=1.0, alpha=0.7)

            # Frequency annotation with Thiele prediction
            f_sim, nc = estimate_frequency(t, x)
            if f_sim > 0:
                omega_0 = 2 * np.pi * f_sim * 1e9  # rad/s
                # Thiele: τ = 2/(α ω₀ [ln(R/a) + ½]), a ≈ 2 l_ex
                R_m = info.get('disk_r_m', 100e-9)
                dx_m = info.get('dx_m', 3.75e-9)
                Ms = 8.0e5  # A/m (Permalloy)
                A_ex = 1.3e-11  # J/m
                mu0 = 4 * np.pi * 1e-7
                l_ex = np.sqrt(2 * A_ex / (mu0 * Ms**2))
                a_core = 2 * l_ex  # core radius ≈ 2 l_ex ≈ 11 nm
                ln_term = np.log(R_m / a_core) + 0.5
                alpha_dyn = 0.01
                tau_thiele = 2.0 / (alpha_dyn * omega_0 * ln_term)
                tau_thiele_ns = tau_thiele * 1e9

                textstr = (f'f = {f_sim:.3f} GHz ({nc} crossings)\n'
                           f'τ_meas  = {tau:.2f} ns\n'
                           f'τ_Thiele = {tau_thiele_ns:.1f} ns\n'
                           f'  [ln(R/a)+½ = {ln_term:.2f}, a≈2l_ex]\n'
                           f'α = {alpha_dyn}')
                props = dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor=color, alpha=0.85)
                ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                        fontsize=8.5, verticalalignment='top', horizontalalignment='right',
                        bbox=props, color=color, family='monospace')

    ax.set_xlabel('t (ns)', fontsize=12)
    ax.set_ylabel('x$_{\\rm core}$ (nm)', fontsize=12)
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)

    out = os.path.join(root, 'fig_core_xt.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 3: Frequency Comparison
# ─────────────────────────────────────────────────────────────────

def plot_frequency(root, info, methods, extra_core=None, t_cut_ns=None):
    """Frequency vs aspect ratio with multi-source literature data.

    Sources:
      - Guslienko (2002) analytical: two-vortices curve b (digitised from Fig 3)
      - Novosad (2005) OOMMF micromagnetic (dotted line from Fig 5)
      - Guslienko (2002) OOMMF markers: 200 nm disks only (same Ms as us)
      - Novosad (2005) experimental: VNA microwave absorption
      - Park (2003) experimental: time-resolved Kerr
      - Guslienko (2006) XMCD-PEEM + calc (Ms=720 G, NOT 800)
    """
    setup_style()
    beta = info['dz_m'] / info['disk_r_m']

    # Our measured frequencies
    f_methods = []
    if extra_core:
        for csv_name, color, label in extra_core:
            t_raw, x_raw, y_raw, mz_raw, cl_raw = load_core_csv(os.path.join(root, csv_name))
            if len(t_raw) > 0:
                t, x, _, _, _ = extract_phase3(t_raw, x_raw, y_raw, mz_raw, cl_raw, t_cut_ns)
                if len(t) > 0:
                    f_sim, nc = estimate_frequency(t, x)
                    if f_sim > 0:
                        f_methods.append((label, f_sim, nc, color))
    for core_csv, prefix, label, color in methods:
        t_raw, x_raw, y_raw, mz_raw, cl_raw = load_core_csv(os.path.join(root, core_csv))
        if len(t_raw) > 0:
            t, x, _, _, _ = extract_phase3(t_raw, x_raw, y_raw, mz_raw, cl_raw, t_cut_ns)
            if len(t) > 0:
                f_sim, nc = estimate_frequency(t, x)
                if f_sim > 0:
                    f_methods.append((label, f_sim, nc, color))

    # ── Literature data ──

    # ── Guslienko (2002) analytical: two-vortices model (curve b) ──
    # Digitised from [G02] Fig 3, curve b.  Dense sampling for smooth
    # interpolation — the curve is nearly linear from (0.05, 240) to
    # (0.25, 1050) with slight sublinear flattening at higher β.
    from scipy.interpolate import PchipInterpolator
    _cB_beta = np.array([
        0.000, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070,
        0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140, 0.150,
        0.160, 0.170, 0.180, 0.190, 0.200, 0.210, 0.220, 0.230,
        0.240, 0.250, 0.260, 0.270, 0.280])
    _cB_f = np.array([
        0,     50,   100,  148,  195,  240,  288,  335,
        380,   420,  460,  500,  540,  578,  615,  650,
        688,   723,  755,  790,  825,  855,  890,  920,
        950,   985,  1015, 1045, 1075])  # MHz
    _cB_interp = PchipInterpolator(_cB_beta, _cB_f)
    beta_curve = np.linspace(0.005, 0.28, 200)
    f_analytical = _cB_interp(beta_curve)  # MHz

    # ── Novosad OOMMF micromagnetic line ──
    # Digitised from [N05] Fig 5, solid "micromagnetics" line.
    oommf_beta = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20])
    oommf_f    = np.array([75, 140, 215, 285, 350, 500, 680])  # MHz

    # ── Guslienko (2002) OOMMF: 200 nm disks only (■) ──
    # Ms = 800 kA/m — IDENTICAL to our simulation parameters.
    # Sits slightly below curve b.  No error bars (deterministic OOMMF).
    gus_oommf_beta = np.array([0.10, 0.20, 0.30, 0.40])
    gus_oommf_f    = np.array([400, 790, 1200, 1500])  # MHz

    # ── Novosad (2005) experimental — VNA microwave absorption ──
    # Exact values from [N05] abstract.  Bars = resonance FWHM.
    novosad_beta = np.array([0.020, 0.036, 0.073])
    novosad_f    = np.array([83.0, 162.0, 272.0])   # MHz
    novosad_delf = np.array([2.0, 11.0, 16.0])      # MHz FWHM

    # ── Park et al. (2003) — time-resolved Kerr microscopy ──
    # Digitised from [N05] Fig 5, open triangles.
    park_beta = np.array([0.12, 0.24])
    park_f    = np.array([420.0, 600.0])     # MHz
    park_df   = np.array([40.0, 90.0])       # MHz

    # ── Guslienko (2006 PRL) — XMCD-PEEM + micromagnetics ──
    # Ms = 720 G (NOT 800 G).  4 calc + 1 exp point, L = 30 nm.
    gus06_beta = np.array([0.0093, 0.0097, 0.0107, 0.0113, 0.0140])
    gus06_f    = np.array([43.0, 45.0, 53.0, 56.0, 63.0])   # MHz
    gus06_is_exp = np.array([False, False, False, False, True])
    gus06_df   = np.array([0, 0, 0, 0, 3.0])  # only exp has error

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Guslienko analytical (two-vortices, curve b)
    ax.plot(beta_curve, f_analytical, 'k-', lw=1.3, zorder=3,
            label=r'Analytical (two-vortices)')

    # Novosad OOMMF micromagnetics (dotted line)
    ax.plot(oommf_beta, oommf_f, 'k:', lw=1.0, zorder=2, alpha=0.7,
            label='OOMMF (Novosad)')

    # Guslienko 2002 OOMMF 200nm squares (same Ms, no error bars)
    ax.scatter(gus_oommf_beta, gus_oommf_f, s=18, c='none',
               edgecolors='0.4', linewidths=0.7, marker='o',
               zorder=4, label=r'OOMMF 200 nm (Guslienko)')

    # Novosad experimental (blue squares + linewidth bars)
    ax.errorbar(novosad_beta, novosad_f, yerr=novosad_delf,
                fmt='s', color='#2166AC', ms=4, capsize=2, capthick=0.7,
                elinewidth=0.7, markeredgecolor='k', markeredgewidth=0.4,
                zorder=5, label=r'Novosad (2005) expt')

    # Park experimental (green triangles + error bars)
    ax.errorbar(park_beta, park_f, yerr=park_df,
                fmt='^', color='#4DAF4A', ms=4.5, capsize=2, capthick=0.7,
                elinewidth=0.7, markeredgecolor='k', markeredgewidth=0.4,
                zorder=5, label=r'Park (2003) expt')

    # Guslienko 2006 PRL (grey diamonds — note Ms=720 G)
    calc_m = ~gus06_is_exp
    ax.scatter(gus06_beta[calc_m], gus06_f[calc_m], s=12,
               c='#777777', edgecolors='k', linewidths=0.3, marker='D',
               zorder=4, label=r'Guslienko (2006) calc')
    if gus06_is_exp.any():
        ax.errorbar(gus06_beta[gus06_is_exp], gus06_f[gus06_is_exp],
                    yerr=gus06_df[gus06_is_exp],
                    fmt='D', color='#777777', ms=4, capsize=1.5,
                    capthick=0.5, elinewidth=0.5,
                    markeredgecolor='k', markeredgewidth=0.3,
                    zorder=5, label=r'Guslienko (2006) expt')

    # Our simulation (filled red circle)
    for i, (lbl, f, nc, col) in enumerate(f_methods):
        ax.errorbar([beta], [f * 1000], yerr=[5.0],
                    fmt='o', color='#D6191B', ms=5.5,
                    markeredgecolor='k', markeredgewidth=0.6,
                    capsize=2, capthick=0.7, elinewidth=0.7,
                    zorder=7, label=f'This work ({f*1000:.0f} MHz)')

    # Axes
    ax.set_xlabel(r'Aspect ratio $\beta = L/R$')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_xlim(0, 0.28)
    ax.set_ylim(0, 1100)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # Legend — compact, positioned to avoid data overlap
    ax.legend(loc='upper left', fontsize=6.5, frameon=True,
              borderpad=0.3, handlelength=1.2, handletextpad=0.3,
              labelspacing=0.2, framealpha=0.92, edgecolor='0.7')

    fig.tight_layout()
    out = os.path.join(root, 'fig_frequency.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 4: Patch Map (with core position marker)
# ─────────────────────────────────────────────────────────────────

def plot_patch_map(root, info, patches_file, title, outname, snap_idx=None, core_method='amr_cfft'):
    """Patch map with disk outline and core position cross marker.
    García-Cervera Fig 4 style: physical nm axes, no title, consistent formatting."""
    patches = load_patches_csv(os.path.join(root, patches_file))
    if not patches:
        return

    base_nx = info['base_nx']
    base_ny = info['base_ny']
    dx_nm = info['dx_m'] * 1e9
    disk_r_cells = info['disk_r_m'] / info['dx_m']

    fig, ax = plt.subplots(figsize=(6, 6))

    # Disk outline
    cx, cy = base_nx / 2, base_ny / 2
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(cx + disk_r_cells * np.cos(theta),
            cy + disk_r_cells * np.sin(theta), 'k-', lw=1.8)

    # Patch rectangles — consistent colour scheme
    colours = {1: '#FFD700', 2: '#00CC00', 3: '#0077FF'}
    drawn_labels = set()

    for level, i0, j0, nx, ny in patches:
        c = colours.get(level, 'grey')
        rect = mpatches.Rectangle((i0, j0), nx, ny,
                                   linewidth=1.5, edgecolor=c,
                                   facecolor=c, alpha=0.15)
        ax.add_patch(rect)
        ax.add_patch(mpatches.Rectangle((i0, j0), nx, ny,
                                         linewidth=1.5, edgecolor=c,
                                         facecolor='none'))
        drawn_labels.add(level)

    # Core position marker
    core_x_nm, core_y_nm = None, None
    if snap_idx is not None:
        core_x_nm, core_y_nm = get_core_at_snap(root, info, snap_idx, method=core_method)
    else:
        core_x_nm, core_y_nm = 0.0, 0.0

    if core_x_nm is not None and core_y_nm is not None:
        core_i = core_x_nm / dx_nm + base_nx / 2
        core_j = core_y_nm / dx_nm + base_ny / 2
        ax.plot(core_i, core_j, 'x', color='white', ms=12, mew=3, zorder=10)
        ax.plot(core_i, core_j, 'x', color='black', ms=10, mew=2, zorder=11)

    # Manual legend — consistent with mesh zoom
    legend_patches = [mpatches.Patch(facecolor=colours[l], edgecolor=colours[l],
                                      alpha=0.4, label=f'L{l}') for l in sorted(colours)]
    ax.legend(handles=legend_patches, fontsize=10, loc='upper right',
              framealpha=0.95, edgecolor='0.7').set_zorder(50)

    ax.set_xlim(0, base_nx)
    ax.set_ylim(0, base_ny)
    ax.set_aspect('equal')

    # Physical nm axis labels — explicit ticks including 0 at disk centre
    R_nm = info['disk_r_m'] * 1e9
    half_domain_nm = base_nx * dx_nm / 2
    # Place ticks at round nm values centred on 0
    tick_spacing_nm = 50 if R_nm > 80 else 25
    nm_ticks = np.arange(-half_domain_nm, half_domain_nm + 1, tick_spacing_nm)
    nm_ticks = nm_ticks[np.abs(nm_ticks) <= half_domain_nm * 0.95]
    cell_ticks = nm_ticks / dx_nm + base_nx / 2
    ax.set_xticks(cell_ticks)
    ax.set_xticklabels([f'{int(v)}' for v in nm_ticks])
    ax.set_yticks(cell_ticks)
    ax.set_yticklabels([f'{int(v)}' for v in nm_ticks])
    ax.set_xlabel(r'$x$ (nm)', fontsize=11)
    ax.set_ylabel(r'$y$ (nm)', fontsize=11)
    ax.tick_params(labelsize=10)

    out = os.path.join(root, outname)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 5: mz Colourmap (with disk mask + colour wheel inset)
# ─────────────────────────────────────────────────────────────────

def plot_mz_colourmap(root, info, m_file, title, outname, snap_idx=None, core_method='amr_cfft'):
    """mz colourmap with disk outline, white masking outside disk, and colour wheel."""
    m = load_m_csv(os.path.join(root, m_file))
    if m is None:
        return

    nx, ny = m['nx'], m['ny']
    is_fine = 'fine' in m_file
    disk_mask = make_disk_mask(nx, ny, info, is_fine=is_fine)
    rgb = angle_colormap(m['mx'], m['my'], m['mz'], disk_mask=disk_mask)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb, origin='lower', interpolation='nearest')

    # Disk outline in pixel coordinates
    if is_fine:
        scale = nx / info['base_nx']
    else:
        scale = 1.0
    cx_pix, cy_pix = nx / 2, ny / 2
    r_pix = info['disk_r_m'] / info['dx_m'] * scale
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(cx_pix + r_pix * np.cos(theta), cy_pix + r_pix * np.sin(theta),
            'k-', lw=2)

    # Core marker
    core_x_nm, core_y_nm = None, None
    if snap_idx is not None:
        core_x_nm, core_y_nm = get_core_at_snap(root, info, snap_idx, method=core_method)
    if core_x_nm is not None and core_y_nm is not None:
        dx_nm = info['dx_m'] * 1e9
        core_i = core_x_nm / (dx_nm / scale) + nx / 2
        core_j = core_y_nm / (dx_nm / scale) + ny / 2
        ax.plot(core_i, core_j, '+', color='white', ms=14, mew=3, zorder=10)
        ax.plot(core_i, core_j, '+', color='black', ms=12, mew=1.5, zorder=11)

    # Add colour wheel inset
    draw_colour_wheel_inset(ax)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect('equal')
    ax.axis('off')

    out = os.path.join(root, outname)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 6: 3D mz Surface — Multi-view (perspective, side, bird's-eye)
# ─────────────────────────────────────────────────────────────────

def plot_mz_3d(root, info, m_file='m_fine_eq.csv', frame_label='Equilibrium'):
    """3D wireframe plot of mz — García-Cervera Fig 8/9 style.

    White background, dashed grid lines, z-axis fixed 0–1,
    wireframe mesh visible on the surface."""
    m = load_m_csv(os.path.join(root, m_file))
    if m is None:
        print(f"  Skipping 3D plot: {m_file} not found")
        return

    mz = m['mz']
    nx, ny = m['nx'], m['ny']

    # Subsample for visible wireframe — keep dense for good boundary resolution
    step = max(1, nx // 180)
    mz_sub = mz[::step, ::step]
    ny_s, nx_s = mz_sub.shape

    # Physical coordinates in nm
    dx_nm = info['dx_m'] * 1e9
    is_fine = 'fine' in m_file
    if is_fine:
        ratio_total = info['ratio'] ** info['amr_levels']
        dx_plot = dx_nm / ratio_total
    else:
        dx_plot = dx_nm

    x = np.arange(nx_s) * step * dx_plot
    y = np.arange(ny_s) * step * dx_plot
    X, Y = np.meshgrid(x, y)
    X -= X.mean()
    Y -= Y.mean()

    # Tight disk mask
    R_nm = info['disk_r_m'] * 1e9
    dist = np.sqrt(X**2 + Y**2)
    mz_plot = np.where(dist <= R_nm * 0.88, mz_sub, np.nan)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    assert isinstance(ax, Axes3D)
    # Orthographic projection: straight parallel axes like G-C's MATLAB plots
    try:
        ax.set_proj_type('ortho')
    except AttributeError:
        pass
    # Wide/flat box like G-C: x:y:z ≈ 1:1:0.25
    try:
        ax.set_box_aspect((1, 1, 0.25))
    except AttributeError:
        pass  # older matplotlib

    # García-Cervera style: coloured surface with thin wireframe edges
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.colormaps['RdYlBu_r']

    colours = cmap(norm(np.nan_to_num(mz_plot, nan=0)))
    colours[np.isnan(mz_plot), 3] = 0.0

    ax.plot_surface(X, Y, mz_plot, facecolors=colours,
                    rstride=1, cstride=1,
                    linewidth=0.08, edgecolor='0.45',
                    alpha=0.95, antialiased=True, shade=False)

    ax.set_xlabel('x (nm)', fontsize=11, labelpad=8)
    ax.set_ylabel('y (nm)', fontsize=11, labelpad=8)
    ax.set_zlabel(r'm$_z$', fontsize=11, labelpad=5)
    ax.view_init(elev=25, azim=-50)

    # Fixed z-axis: floor below 0 so the mz≈0 surface floats
    # above the base plane (García-Cervera style)
    ax.set_zlim(-0.12, 1.0)
    ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Tight axes so disk fills frame
    lim = R_nm * 0.92
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.tick_params(labelsize=9, pad=3)

    # García-Cervera style: white panes with very thin light edges
    try:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False  # type: ignore
            axis.pane.set_edgecolor((0.80, 0.80, 0.80, 0.15))  # type: ignore
            axis._axinfo['grid']['linewidth'] = 0.2  # type: ignore
            axis._axinfo['grid']['linestyle'] = ':'  # type: ignore
            axis._axinfo['grid']['color'] = (0.7, 0.7, 0.7)  # type: ignore
        # Make the 3D box frame lines thin and light
        for line in ax.xaxis.get_ticklines():
            line.set_linewidth(0.3)
        for line in ax.yaxis.get_ticklines():
            line.set_linewidth(0.3)
        for line in ax.zaxis.get_ticklines():
            line.set_linewidth(0.3)
    except (AttributeError, KeyError):
        pass
    ax.grid(True, color='0.80', alpha=0.20, linewidth=0.2, linestyle=':')
    # Thin the 3D box/spine lines
    try:
        for spine in ax.xaxis.line, ax.yaxis.line, ax.zaxis.line:
            spine.set_linewidth(0.3)  # type: ignore
            spine.set_color('0.6')  # type: ignore
    except AttributeError:
        pass

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)

    fig.tight_layout()
    out_base = os.path.splitext(m_file)[0]
    outname = f'fig_mz_3d_{out_base.replace("m_fine_", "").replace("m_coarse_", "c_")}.pdf'
    out = os.path.join(root, outname)
    fig.savefig(out, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 7: Mesh — Full domain with disk boundary
# ─────────────────────────────────────────────────────────────────

def plot_mesh_full(root, info, patches_file='patches_eq.csv',
                   m_file='m_fine_eq.csv', frame_label='Equilibrium'):
    """Full-domain view showing magnetisation with multi-resolution grid overlay."""
    m = load_m_csv(os.path.join(root, m_file))
    patches = load_patches_csv(os.path.join(root, patches_file))
    if m is None:
        print(f"  Skipping mesh plot: {m_file} not found")
        return

    nx, ny = m['nx'], m['ny']
    is_fine = 'fine' in m_file
    disk_mask = make_disk_mask(nx, ny, info, is_fine=is_fine)

    # Use mz-based colourmap matching the 3D plots
    cmap_mz = plt.colormaps['RdYlBu_r']
    norm_mz = Normalize(vmin=-0.02, vmax=1.0)
    rgb = cmap_mz(norm_mz(m['mz']))
    rgb[~disk_mask] = [1.0, 1.0, 1.0, 1.0]

    ratio_total = info['ratio'] ** info['amr_levels']
    base_nx = info['base_nx']

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, origin='lower', interpolation='nearest')

    # Draw coarse grid lines across full domain (L0)
    for i in range(base_nx + 1):
        xp = i * ratio_total
        ax.axvline(xp, color='grey', lw=0.3, alpha=0.4)
    for j in range(info['base_ny'] + 1):
        yp = j * ratio_total
        ax.axhline(yp, color='grey', lw=0.3, alpha=0.4)

    # Draw patches + internal grid lines for L1 and L2; just boxes for L3
    colours_p = {1: '#FFD700', 2: '#00CC00', 3: '#0077FF'}
    ratio = info['ratio']
    for level, i0, j0, pnx, pny in patches:
        scale = ratio_total
        fx0 = i0 * scale
        fy0 = j0 * scale
        fnx = pnx * scale
        fny = pny * scale
        c = colours_p.get(level, 'grey')

        # Patch boundary box
        lw = 2.5 if level == 1 else (2.0 if level == 2 else 1.5)
        ax.add_patch(mpatches.Rectangle((fx0, fy0), fnx, fny,
                                         linewidth=lw, edgecolor=c,
                                         facecolor='none', zorder=3))

        # Draw internal grid lines for L1 and L2
        if level <= 2:
            # Cell size at this level in fine pixels
            cell_fine = ratio_total // (ratio ** level)
            grid_lw = 0.15 if level == 1 else 0.1
            grid_alpha = 0.5 if level == 1 else 0.4
            # Vertical grid lines
            n_cells_x = pnx * (ratio ** level)  # number of cells at this level
            for ci in range(1, n_cells_x):
                xp = fx0 + ci * cell_fine
                ax.plot([xp, xp], [fy0, fy0 + fny],
                        color=c, lw=grid_lw, alpha=grid_alpha, zorder=2)
            # Horizontal grid lines
            n_cells_y = pny * (ratio ** level)
            for cj in range(1, n_cells_y):
                yp = fy0 + cj * cell_fine
                ax.plot([fx0, fx0 + fnx], [yp, yp],
                        color=c, lw=grid_lw, alpha=grid_alpha, zorder=2)

    # Disk outline
    cx_pix, cy_pix = nx / 2, ny / 2
    r_pix = info['disk_r_m'] / info['dx_m'] * (nx / info['base_nx'] if is_fine else 1)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(cx_pix + r_pix * np.cos(theta), cy_pix + r_pix * np.sin(theta),
            'k-', lw=2.5, zorder=4)

    # Legend — consistent with patch map and mesh zoom
    legend_handles = []
    for l in sorted(colours_p):
        legend_handles.append(mpatches.Patch(facecolor='none', edgecolor=colours_p[l],
                                              linewidth=2, label=f'L{l}'))
    legend_handles.append(Line2D([0], [0], color='k', lw=2, label='Disk boundary'))
    ax.legend(handles=legend_handles, fontsize=10, loc='upper right',
              framealpha=0.95, edgecolor='0.7').set_zorder(50)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect('equal')
    ax.axis('off')

    out = os.path.join(root, f'fig_mesh_full_{frame_label.split("(")[-1].replace(")","").replace(" ","_").lower() if "(" in frame_label else "eq"}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 8b: Zoomed Mesh — García-Cervera Fig 5 style
# ─────────────────────────────────────────────────────────────────

def plot_mesh_zoom(root, info, patches_file, m_file,
                   frame_label='Equilibrium', core_x_nm=None, core_y_nm=None,
                   zoom_base_cells=25):
    """Zoomed mesh view centred on the vortex core, showing cell-level resolution
    changes between AMR levels. Inspired by García-Cervera Fig 5."""
    m = load_m_csv(os.path.join(root, m_file))
    patches = load_patches_csv(os.path.join(root, patches_file))
    if m is None:
        print(f"  Skipping mesh zoom: {m_file} not found")
        return

    nx, ny = m['nx'], m['ny']
    is_fine = 'fine' in m_file
    disk_mask = make_disk_mask(nx, ny, info, is_fine=is_fine)

    # Use mz-based colourmap matching the 3D plots
    cmap_mz = plt.colormaps['RdYlBu_r']
    norm_mz = Normalize(vmin=-0.02, vmax=1.0)
    mz_img = cmap_mz(norm_mz(m['mz']))
    mz_img[~disk_mask] = [1.0, 1.0, 1.0, 1.0]

    ratio_total = info['ratio'] ** info['amr_levels']
    ratio = info['ratio']
    base_nx = info['base_nx']
    base_ny = info['base_ny']

    # Centre on core position (or disk centre)
    if core_x_nm is not None and core_y_nm is not None:
        dx_nm = info['dx_m'] * 1e9
        core_base_i = core_x_nm / dx_nm + base_nx / 2
        core_base_j = core_y_nm / dx_nm + base_ny / 2
    else:
        core_base_i = base_nx / 2
        core_base_j = base_ny / 2

    # Zoom window in base cells
    hw = zoom_base_cells
    i_lo = max(0, int(core_base_i - hw))
    i_hi = min(base_nx, int(core_base_i + hw))
    j_lo = max(0, int(core_base_j - hw))
    j_hi = min(base_ny, int(core_base_j + hw))

    # Convert to pixel indices in the ACTUAL image (which may be fine or coarse)
    pix_per_base = nx // base_nx  # fine: 8, coarse: 1
    fi_lo, fi_hi = i_lo * pix_per_base, i_hi * pix_per_base
    fj_lo, fj_hi = j_lo * pix_per_base, j_hi * pix_per_base

    # Clamp to image bounds
    fi_lo, fi_hi = max(0, fi_lo), min(nx, fi_hi)
    fj_lo, fj_hi = max(0, fj_lo), min(ny, fj_hi)

    if fi_hi <= fi_lo or fj_hi <= fj_lo:
        print(f"  Skipping mesh zoom: empty crop region")
        return

    # For grid line drawing, always use ratio_total (base-cell → fine-cell scale)
    # but the image extent must match the pixel scale
    # Map everything to "fine pixel" coordinates for consistent extent
    extent_lo_x = i_lo * ratio_total
    extent_hi_x = i_hi * ratio_total
    extent_lo_y = j_lo * ratio_total
    extent_hi_y = j_hi * ratio_total

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mz_img[fj_lo:fj_hi, fi_lo:fi_hi], origin='lower', interpolation='nearest',
              extent=(extent_lo_x, extent_hi_x, extent_lo_y, extent_hi_y))

    # L0 coarse grid lines (thick, dark grey)
    for i in range(i_lo, i_hi + 1):
        xp = i * ratio_total
        ax.plot([xp, xp], [extent_lo_y, extent_hi_y], color='0.35', lw=0.8, alpha=0.7, zorder=2)
    for j in range(j_lo, j_hi + 1):
        yp = j * ratio_total
        ax.plot([extent_lo_x, extent_hi_x], [yp, yp], color='0.35', lw=0.8, alpha=0.7, zorder=2)

    # Patch grid lines at each level's native resolution
    colours_p = {1: '#FFD700', 2: '#00CC00', 3: '#0077FF'}
    for level, pi0, pj0, pnx, pny in patches:
        c = colours_p.get(level, 'grey')
        fx0 = pi0 * ratio_total
        fy0 = pj0 * ratio_total
        fnx = pnx * ratio_total
        fny = pny * ratio_total

        # Skip if entirely outside zoom window
        if fx0 + fnx < extent_lo_x or fx0 > extent_hi_x or fy0 + fny < extent_lo_y or fy0 > extent_hi_y:
            continue

        # Cell size at this level in fine pixels
        cell_fine = ratio_total // (ratio ** level)
        lw_grid = 0.5 if level == 1 else (0.35 if level == 2 else 0.25)
        alpha_grid = 0.65 if level == 1 else (0.55 if level == 2 else 0.45)

        # Internal grid lines (clipped to zoom window)
        n_cells_x = pnx * (ratio ** level)
        for ci in range(n_cells_x + 1):
            xp = fx0 + ci * cell_fine
            if extent_lo_x <= xp <= extent_hi_x:
                y0c = max(fy0, extent_lo_y)
                y1c = min(fy0 + fny, extent_hi_y)
                ax.plot([xp, xp], [y0c, y1c], color=c, lw=lw_grid,
                        alpha=alpha_grid, zorder=2 + level)
        n_cells_y = pny * (ratio ** level)
        for cj in range(n_cells_y + 1):
            yp = fy0 + cj * cell_fine
            if extent_lo_y <= yp <= extent_hi_y:
                x0c = max(fx0, extent_lo_x)
                x1c = min(fx0 + fnx, extent_hi_x)
                ax.plot([x0c, x1c], [yp, yp], color=c, lw=lw_grid,
                        alpha=alpha_grid, zorder=2 + level)

        # Patch boundary (thicker)
        lw_box = 2.5 if level == 1 else (2.0 if level == 2 else 1.5)
        ax.add_patch(mpatches.Rectangle((fx0, fy0), fnx, fny,
                                         linewidth=lw_box, edgecolor=c,
                                         facecolor='none', zorder=5 + level))

    # Core marker (in fine-pixel coordinate space)
    if core_x_nm is not None and core_y_nm is not None:
        dx_nm = info['dx_m'] * 1e9
        ci_fine = core_x_nm / dx_nm * ratio_total + base_nx * ratio_total / 2
        cj_fine = core_y_nm / dx_nm * ratio_total + base_ny * ratio_total / 2
        ax.plot(ci_fine, cj_fine, 'x', color='white', ms=14, mew=3, zorder=20)
        ax.plot(ci_fine, cj_fine, 'x', color='black', ms=12, mew=2, zorder=21)

    # Legend — consistent with patch map
    legend_handles = [mpatches.Patch(facecolor='none', edgecolor=colours_p[l],
                                      linewidth=2, label=f'L{l}') for l in sorted(colours_p)]
    leg = ax.legend(handles=legend_handles, fontsize=10, loc='upper right',
                    framealpha=0.95, edgecolor='0.7')
    leg.set_zorder(50)

    ax.set_xlim(extent_lo_x, extent_hi_x)
    ax.set_ylim(extent_lo_y, extent_hi_y)
    ax.set_aspect('equal')

    # Convert tick labels from fine-pixel coordinates to nm — explicit ticks including 0
    dx_nm = info['dx_m'] * 1e9
    dx_fine_nm = dx_nm / ratio_total
    half_domain_nm = info['base_nx'] * dx_nm / 2

    # Compute nm range visible in the zoom window
    nm_lo_x = extent_lo_x * dx_fine_nm - half_domain_nm
    nm_hi_x = extent_hi_x * dx_fine_nm - half_domain_nm
    nm_lo_y = extent_lo_y * dx_fine_nm - half_domain_nm
    nm_hi_y = extent_hi_y * dx_fine_nm - half_domain_nm

    # Choose tick spacing based on zoom extent
    zoom_span = max(nm_hi_x - nm_lo_x, nm_hi_y - nm_lo_y)
    if zoom_span > 100:
        tick_sp = 50
    elif zoom_span > 40:
        tick_sp = 20
    else:
        tick_sp = 10

    # X ticks
    nm_xticks = np.arange(np.ceil(nm_lo_x / tick_sp) * tick_sp,
                          nm_hi_x + 0.1, tick_sp)
    fine_xticks = (nm_xticks + half_domain_nm) / dx_fine_nm
    ax.set_xticks(fine_xticks)
    ax.set_xticklabels([f'{int(v)}' for v in nm_xticks])

    # Y ticks
    nm_yticks = np.arange(np.ceil(nm_lo_y / tick_sp) * tick_sp,
                          nm_hi_y + 0.1, tick_sp)
    fine_yticks = (nm_yticks + half_domain_nm) / dx_fine_nm
    ax.set_yticks(fine_yticks)
    ax.set_yticklabels([f'{int(v)}' for v in nm_yticks])

    ax.set_xlabel(r'$x$ (nm)', fontsize=11)
    ax.set_ylabel(r'$y$ (nm)', fontsize=11)
    ax.tick_params(labelsize=10)

    tag = frame_label.split("(")[-1].replace(")", "").replace(" ", "_").lower() if "(" in frame_label else "eq"
    out = os.path.join(root, f'fig_mesh_zoom_{tag}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 9: Multi-Resolution 3D mz — García-Cervera Fig 9 style
# ─────────────────────────────────────────────────────────────────

def plot_mz_3d_multiresolution(root, info, m_fine_file, patches_file,
                                frame_label='Equilibrium',
                                core_x_nm=None, core_y_nm=None,
                                zoom_radius_nm=50):
    """Zoomed 3D plot centred on the vortex core — García-Cervera Fig 9 style.
    
    Zoomed to ±zoom_radius_nm around the core so L0/L1/L2/L3 cell size
    differences are clearly visible. Smooth coloured surface underneath,
    wireframe at each level's native resolution on top."""
    m = load_m_csv(os.path.join(root, m_fine_file))
    patches = load_patches_csv(os.path.join(root, patches_file))
    if m is None:
        print(f"  Skipping multi-res 3D: {m_fine_file} not found")
        return

    mz_full = m['mz']
    nx_fine, ny_fine = m['nx'], m['ny']
    ratio_total = info['ratio'] ** info['amr_levels']
    ratio = info['ratio']
    R_nm = info['disk_r_m'] * 1e9

    is_fine = 'fine' in m_fine_file
    dx_fine_nm = info['dx_m'] * 1e9 / ratio_total if is_fine else info['dx_m'] * 1e9

    # Core position — default to disk centre
    cx_nm = core_x_nm if core_x_nm is not None else 0.0
    cy_nm = core_y_nm if core_y_nm is not None else 0.0

    # Convert zoom window to fine pixel indices
    half_domain_nm = nx_fine * dx_fine_nm / 2
    fi_lo = max(0, int((cx_nm - zoom_radius_nm + half_domain_nm) / dx_fine_nm))
    fi_hi = min(nx_fine, int((cx_nm + zoom_radius_nm + half_domain_nm) / dx_fine_nm))
    fj_lo = max(0, int((cy_nm - zoom_radius_nm + half_domain_nm) / dx_fine_nm))
    fj_hi = min(ny_fine, int((cy_nm + zoom_radius_nm + half_domain_nm) / dx_fine_nm))

    # Crop mz and build coordinates
    mz_crop = mz_full[fj_lo:fj_hi, fi_lo:fi_hi].copy()
    ny_c, nx_c = mz_crop.shape
    x_lo_nm = fi_lo * dx_fine_nm - half_domain_nm
    y_lo_nm = fj_lo * dx_fine_nm - half_domain_nm
    x_arr = np.arange(nx_c) * dx_fine_nm + x_lo_nm
    y_arr = np.arange(ny_c) * dx_fine_nm + y_lo_nm
    X_full, Y_full = np.meshgrid(x_arr, y_arr)

    # Disk mask on cropped region
    dist_c = np.sqrt(X_full**2 + Y_full**2)
    mz_crop[dist_c > R_nm * 0.95] = np.nan

    # Surface at L1 stride for smoothness
    ss = max(1, ratio_total // ratio)
    X_s, Y_s = X_full[::ss, ::ss], Y_full[::ss, ::ss]
    mz_s = mz_crop[::ss, ::ss]

    cmap = plt.colormaps['RdYlBu_r']
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    assert isinstance(ax, Axes3D)
    try:
        ax.set_proj_type('ortho')
    except AttributeError:
        pass
    try:
        ax.set_box_aspect((1, 1, 0.25))
    except AttributeError:
        pass

    # ── 1. Smooth coloured surface ──
    colours = cmap(norm(np.nan_to_num(mz_s, nan=0)))
    colours[np.isnan(mz_s), 3] = 0.0
    ax.plot_surface(X_s, Y_s, mz_s, facecolors=colours,
                    rstride=1, cstride=1, linewidth=0,
                    alpha=0.88, antialiased=True, shade=False)

    # ── 2. Level map for cropped region ──
    lm_full = np.zeros((ny_fine, nx_fine), dtype=int)
    for level, i0, j0, pnx, pny in patches:
        pi0 = max(0, i0 * ratio_total)
        pj0 = max(0, j0 * ratio_total)
        pi1 = min(nx_fine, (i0 + pnx) * ratio_total)
        pj1 = min(ny_fine, (j0 + pny) * ratio_total)
        lm_full[pj0:pj1, pi0:pi1] = np.maximum(lm_full[pj0:pj1, pi0:pi1], level)
    lm_crop = lm_full[fj_lo:fj_hi, fi_lo:fi_hi]

    # ── 3. Wireframe at each level's density ──
    max_level = info['amr_levels']
    level_style = {
        0: {'color': '0.20', 'lw': 0.7, 'alpha': 0.7},
        1: {'color': '0.25', 'lw': 0.45, 'alpha': 0.6},
        2: {'color': '0.30', 'lw': 0.30, 'alpha': 0.55},
        3: {'color': '0.35', 'lw': 0.20, 'alpha': 0.5},
    }

    for lv in range(max_level + 1):
        stride = ratio_total // (ratio ** lv)
        stride = max(1, stride)
        sty = level_style.get(lv, {'color': '0.4', 'lw': 0.15, 'alpha': 0.4})

        # Horizontal lines
        for jj in range(0, ny_c, stride):
            xs, zs = [], []
            for ii in range(0, nx_c, stride):
                js = min(jj, ny_c - 1)
                is_ = min(ii, nx_c - 1)
                if lm_crop[js, is_] == lv and not np.isnan(mz_crop[js, is_]):
                    xs.append(x_arr[is_])
                    zs.append(mz_crop[js, is_])
                else:
                    if len(xs) > 1:
                        ax.plot(xs, np.full(len(xs), y_arr[js]), zs,
                                color=sty['color'], lw=sty['lw'],
                                alpha=sty['alpha'], zorder=1)
                    xs, zs = [], []
            if len(xs) > 1:
                ax.plot(xs, np.full(len(xs), y_arr[min(jj, ny_c-1)]), zs,
                        color=sty['color'], lw=sty['lw'],
                        alpha=sty['alpha'], zorder=1)

        # Vertical lines
        for ii in range(0, nx_c, stride):
            ys, zs = [], []
            for jj in range(0, ny_c, stride):
                js = min(jj, ny_c - 1)
                is_ = min(ii, nx_c - 1)
                if lm_crop[js, is_] == lv and not np.isnan(mz_crop[js, is_]):
                    ys.append(y_arr[js])
                    zs.append(mz_crop[js, is_])
                else:
                    if len(ys) > 1:
                        ax.plot(np.full(len(ys), x_arr[is_]), ys, zs,
                                color=sty['color'], lw=sty['lw'],
                                alpha=sty['alpha'], zorder=1)
                    ys, zs = [], []
            if len(ys) > 1:
                ax.plot(np.full(len(ys), x_arr[min(ii, nx_c-1)]), ys, zs,
                        color=sty['color'], lw=sty['lw'],
                        alpha=sty['alpha'], zorder=1)

    ax.set_xlabel('x (nm)', fontsize=11, labelpad=8)
    ax.set_ylabel('y (nm)', fontsize=11, labelpad=8)
    ax.set_zlabel(r'm$_z$', fontsize=11, labelpad=5)
    ax.view_init(elev=25, azim=-50)
    ax.set_zlim(-0.12, 1.0)
    ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=9, pad=3)

    try:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False  # type: ignore
            axis.pane.set_edgecolor((0.80, 0.80, 0.80, 0.15))  # type: ignore
            axis._axinfo['grid']['linewidth'] = 0.2  # type: ignore
            axis._axinfo['grid']['linestyle'] = ':'  # type: ignore
            axis._axinfo['grid']['color'] = (0.7, 0.7, 0.7)  # type: ignore
        for line in ax.xaxis.get_ticklines():
            line.set_linewidth(0.3)
        for line in ax.yaxis.get_ticklines():
            line.set_linewidth(0.3)
        for line in ax.zaxis.get_ticklines():
            line.set_linewidth(0.3)
    except (AttributeError, KeyError):
        pass
    ax.grid(True, color='0.80', alpha=0.20, linewidth=0.2, linestyle=':')
    try:
        for spine in ax.xaxis.line, ax.yaxis.line, ax.zaxis.line:
            spine.set_linewidth(0.3)  # type: ignore
            spine.set_color('0.6')  # type: ignore
    except AttributeError:
        pass

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)

    fig.tight_layout()
    tag = frame_label.split("(")[-1].replace(")", "").replace(" ", "_").lower() if "(" in frame_label else "eq"
    out = os.path.join(root, f'fig_mz_3d_multires_{tag}.pdf')
    fig.savefig(out, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 8: 1D mz Cross-Sections (Novosad Fig 4 style)
# ─────────────────────────────────────────────────────────────────

def plot_mz_cross_sections(root, info, m_files, labels, outname='fig_mz_cross_section.pdf'):
    """1D mz profile through the vortex core along and perpendicular to displacement.

    Reproduces Novosad et al. Fig 4: shows asymmetric core profile during motion.

    m_files: list of (filename, label) pairs to overlay
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for m_file, label in zip(m_files, labels):
        m = load_m_csv(os.path.join(root, m_file))
        if m is None:
            print(f"  Cross-section: skipping {m_file} (not found)")
            continue

        mz = m['mz']
        nx, ny = m['nx'], m['ny']
        is_fine = 'fine' in m_file

        # Physical pixel size in nm
        if is_fine:
            ratio_total = info['ratio'] ** info['amr_levels']
            dx_nm = info['dx_m'] * 1e9 / ratio_total
        else:
            dx_nm = info['dx_m'] * 1e9

        R_nm = info['disk_r_m'] * 1e9

        # Create disk mask to avoid boundary artefacts
        disk_mask = make_disk_mask(nx, ny, info, is_fine=is_fine)

        # Find vortex core: pixel with maximum mz inside disk
        mz_masked = np.where(disk_mask, mz, -999)
        core_jj, core_ii = np.unravel_index(np.argmax(mz_masked), mz.shape)

        # Disk centre in pixels
        cx = nx / 2.0
        cy = ny / 2.0

        # Displacement vector (pixels) from disk centre to core
        dx_disp = core_ii - cx
        dy_disp = core_jj - cy
        disp_mag = np.sqrt(dx_disp**2 + dy_disp**2)

        if disp_mag < 2:
            # Core is at centre (equilibrium): use x and y axes
            dir_par = np.array([1.0, 0.0])   # "along x" = XX'
            dir_perp = np.array([0.0, 1.0])   # "along y" = YY'
            par_label = "X—X' (horizontal)"
            perp_label = "Y—Y' (vertical)"
        else:
            # Dynamic: along displacement and perpendicular
            dir_par = np.array([dx_disp, dy_disp]) / disp_mag
            dir_perp = np.array([-dy_disp, dx_disp]) / disp_mag
            angle_deg = np.degrees(np.arctan2(dy_disp, dx_disp))
            par_label = f"A—A' (along disp, {angle_deg:.0f}°)"
            perp_label = f"B—B' (⊥ disp)"

        # Sample along each direction through the core
        half_len = int(R_nm / dx_nm * 1.0)  # sample out to R
        sample_pts = np.arange(-half_len, half_len + 1)
        r_nm = sample_pts * dx_nm  # physical distance from core in nm

        for ax_idx, (direction, dir_label) in enumerate([
            (dir_par, par_label), (dir_perp, perp_label)
        ]):
            mz_profile = []
            for s in sample_pts:
                pi = int(round(core_ii + s * direction[0]))
                pj = int(round(core_jj + s * direction[1]))
                if 0 <= pi < nx and 0 <= pj < ny and disk_mask[pj, pi]:
                    mz_profile.append(mz[pj, pi])
                else:
                    mz_profile.append(np.nan)

            mz_arr = np.array(mz_profile)
            axes[ax_idx].plot(r_nm, mz_arr, lw=1.5, label=f'{label}')
            axes[ax_idx].set_title(dir_label, fontsize=12)

    for ax in axes:
        ax.set_xlabel('Distance from core (nm)', fontsize=11)
        ax.set_ylabel('m$_z$', fontsize=11)
        ax.axhline(0, color='grey', lw=0.5, ls='--')
        ax.axvline(0, color='grey', lw=0.5, ls='--')
        ax.set_ylim(-0.3, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()

    out = os.path.join(root, outname)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure: Vector Field Glyphs (quiver plot of in-plane magnetisation)
# ─────────────────────────────────────────────────────────────────

def plot_vector_field(root, info, m_file, patches_file=None,
                      frame_label='Equilibrium', core_method='amr_comp',
                      snap_idx=None, zoom_radius_nm=None):
    """In-plane magnetisation quiver/glyph plot with mz background colour.

    Similar to Novosad Fig 4 / Zhang et al. Fig 1: arrows show in-plane
    direction, background colour shows mz. Optionally zoomed near core.
    Generates both full-disk and zoomed-core versions.
    """
    setup_style()
    m = load_m_csv(os.path.join(root, m_file))
    if m is None:
        print(f"  Skipping vector field: {m_file} not found")
        return

    mx, my, mz = m['mx'], m['my'], m['mz']
    nx, ny = m['nx'], m['ny']
    is_fine = 'fine' in m_file

    if is_fine:
        ratio_total = info['ratio'] ** info['amr_levels']
        dx_nm = info['dx_m'] * 1e9 / ratio_total
    else:
        dx_nm = info['dx_m'] * 1e9

    R_nm = info['disk_r_m'] * 1e9
    half_domain = nx * dx_nm / 2

    # Physical coordinate arrays
    x_nm_arr = np.arange(nx) * dx_nm - half_domain + dx_nm / 2
    y_nm_arr = np.arange(ny) * dx_nm - half_domain + dx_nm / 2
    X_nm, Y_nm = np.meshgrid(x_nm_arr, y_nm_arr)
    dist = np.sqrt(X_nm**2 + Y_nm**2)

    # Disk mask
    disk = dist <= R_nm

    # Core position
    core_x_nm: float = 0.0
    core_y_nm: float = 0.0
    if snap_idx is not None:
        cx, cy = get_core_at_snap(root, info, snap_idx, method=core_method)
        if cx is not None and cy is not None:
            core_x_nm, core_y_nm = float(cx), float(cy)

    # ── Generate two versions: full disk + zoomed near core ──
    for zoom_mode in ['full', 'zoom']:
        if zoom_mode == 'zoom':
            zr = zoom_radius_nm if zoom_radius_nm else 45
            x_lo, x_hi = core_x_nm - zr, core_x_nm + zr
            y_lo, y_hi = core_y_nm - zr, core_y_nm + zr
            # Arrow density: show every cell in zoom
            arrow_skip = max(1, int(2 / dx_nm))  # ~2 nm spacing
            figsize = (4.0, 4.0)
        else:
            x_lo, x_hi = -R_nm * 1.05, R_nm * 1.05
            y_lo, y_hi = -R_nm * 1.05, R_nm * 1.05
            # Arrow density: thin out for full disk
            arrow_skip = max(1, int(8 / dx_nm))  # ~8 nm spacing
            figsize = (4.5, 4.5)

        fig, ax = plt.subplots(figsize=figsize)

        # Background: mz colourmap (blue-white-red)
        mz_plot = np.where(disk, mz, np.nan)
        extent = (float(x_nm_arr[0] - dx_nm/2), float(x_nm_arr[-1] + dx_nm/2),
                  float(y_nm_arr[0] - dx_nm/2), float(y_nm_arr[-1] + dx_nm/2))
        ax.imshow(mz_plot, origin='lower', extent=extent,
                  cmap='RdBu_r', vmin=-1, vmax=1, alpha=0.4,
                  interpolation='bilinear', zorder=0)

        # Quiver arrows for in-plane magnetisation
        # Subsample for readability
        X_q = X_nm[::arrow_skip, ::arrow_skip]
        Y_q = Y_nm[::arrow_skip, ::arrow_skip]
        U_q = mx[::arrow_skip, ::arrow_skip]
        V_q = my[::arrow_skip, ::arrow_skip]
        mz_q = mz[::arrow_skip, ::arrow_skip]
        dist_q = np.sqrt(X_q**2 + Y_q**2)

        # Mask outside disk and in high-mz core region
        mask = (dist_q <= R_nm * 0.98)
        U_q = np.where(mask, U_q, 0)
        V_q = np.where(mask, V_q, 0)

        # Arrow colour: black in bulk, fade for high |mz|
        mag_inplane = np.sqrt(U_q**2 + V_q**2)
        alpha_arr = np.clip(mag_inplane / 0.5, 0.1, 1.0)

        # Only draw arrows where |m_inplane| > 0.1
        draw_mask = (mag_inplane > 0.1) & mask
        ax.quiver(X_q[draw_mask], Y_q[draw_mask],
                  U_q[draw_mask], V_q[draw_mask],
                  color='k', scale=25, width=0.004, headwidth=3.5,
                  headlength=3, headaxislength=2.5,
                  alpha=0.8, zorder=3, pivot='mid')

        # Core marker (for zoom mode, mark the mz peak)
        if zoom_mode == 'zoom':
            ax.plot(core_x_nm, core_y_nm, '+', color='red', ms=10,
                    mew=2, zorder=5)

        # Disk boundary
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(R_nm * np.cos(theta), R_nm * np.sin(theta),
                'k-', lw=1.5, zorder=4)

        # Patch outlines (if provided)
        if patches_file and zoom_mode == 'full':
            patches = load_patches_csv(os.path.join(root, patches_file))
            colours_p = {1: '#FFD700', 2: '#00CC00', 3: '#0077FF'}
            dx_base = info['dx_m'] * 1e9
            dh = info['base_nx'] * dx_base / 2
            for level, i0, j0, pnx, pny in patches:
                c = colours_p.get(level, 'grey')
                px = i0 * dx_base - dh
                py = j0 * dx_base - dh
                pw = pnx * dx_base
                ph = pny * dx_base
                ax.add_patch(mpatches.Rectangle(
                    (px, py), pw, ph, linewidth=0.8,
                    edgecolor=c, facecolor='none', zorder=4, alpha=0.6))

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$ (nm)')
        ax.set_ylabel(r'$y$ (nm)')
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

        # Colourbar for mz background
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=Normalize(-1, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(r'$m_z$', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        fig.tight_layout()
        tag = frame_label.split("(")[-1].replace(")", "").replace(" ", "_").lower() if "(" in frame_label else "eq"
        stem = os.path.splitext(m_file)[0].replace("m_fine_", "").replace("m_coarse_", "c_")
        out = os.path.join(root, f'fig_quiver_{zoom_mode}_{stem}_{tag}.pdf')
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 10: Simple Vortex Quiver Inset (for Illustrator compositing)
# ─────────────────────────────────────────────────────────────────

def plot_vortex_inset_quiver(root, info, m_file, snap_idx=None,
                             core_method='amr_comp'):
    """Minimal quiver plot showing vortex circulation — designed as a small
    standalone figure to be composited as an inset in Adobe Illustrator.

    Clean white background, no colourbar, no axis labels, thin disk outline,
    ~12×12 arrows, mz background for the core spot. Produces a square PDF.
    """
    setup_style()
    m = load_m_csv(os.path.join(root, m_file))
    if m is None:
        print(f"  Skipping vortex inset: {m_file} not found")
        return

    mx, my, mz = m['mx'], m['my'], m['mz']
    nx, ny = m['nx'], m['ny']
    is_fine = 'fine' in m_file

    if is_fine:
        ratio_total = info['ratio'] ** info['amr_levels']
        dx_nm = info['dx_m'] * 1e9 / ratio_total
    else:
        dx_nm = info['dx_m'] * 1e9

    R_nm = info['disk_r_m'] * 1e9

    # Physical coordinates
    half_domain = nx * dx_nm / 2
    x_arr = np.arange(nx) * dx_nm - half_domain + dx_nm / 2
    y_arr = np.arange(ny) * dx_nm - half_domain + dx_nm / 2
    X, Y = np.meshgrid(x_arr, y_arr)
    dist = np.sqrt(X**2 + Y**2)

    # Core position
    core_x, core_y = 0.0, 0.0
    if snap_idx is not None:
        cx, cy = get_core_at_snap(root, info, snap_idx, method=core_method)
        if cx is not None and cy is not None:
            core_x, core_y = float(cx), float(cy)

    # ── Small square figure, no axis labels ──
    fig, ax = plt.subplots(figsize=(3.0, 3.0))

    # Subtle mz background inside disk (just enough to show the core spot)
    disk_mask = dist <= R_nm
    mz_bg = np.where(disk_mask, mz, np.nan)
    extent = (float(x_arr[0] - dx_nm/2), float(x_arr[-1] + dx_nm/2),
              float(y_arr[0] - dx_nm/2), float(y_arr[-1] + dx_nm/2))
    ax.imshow(mz_bg, origin='lower', extent=extent,
              cmap='RdBu_r', vmin=-1, vmax=1, alpha=0.3,
              interpolation='bilinear', zorder=0)

    # Sparse arrows: ~20 across the disk diameter, thin shafts
    n_arrows = 20
    arrow_spacing_nm = 2 * R_nm / n_arrows
    skip = max(1, int(round(arrow_spacing_nm / dx_nm)))

    X_q = X[::skip, ::skip]
    Y_q = Y[::skip, ::skip]
    U_q = mx[::skip, ::skip]
    V_q = my[::skip, ::skip]
    dist_q = np.sqrt(X_q**2 + Y_q**2)
    mag_ip = np.sqrt(U_q**2 + V_q**2)

    # Only inside disk, only where in-plane component is significant
    draw = (dist_q <= R_nm * 0.92) & (mag_ip > 0.15)

    ax.quiver(X_q[draw], Y_q[draw], U_q[draw], V_q[draw],
              color='k', scale=28, width=0.0035, headwidth=3.2,
              headlength=2.8, headaxislength=2.2,
              alpha=0.8, zorder=3, pivot='mid')

    # Core marker
    ax.plot(core_x, core_y, 'o', color='red', ms=5, mew=0, zorder=5, alpha=0.9)

    # Thin disk boundary
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_nm * np.cos(theta), R_nm * np.sin(theta),
            'k-', lw=1.0, zorder=4)

    lim = R_nm * 1.08
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout(pad=0.1)
    stem = os.path.splitext(m_file)[0].replace("m_fine_", "").replace("m_coarse_", "c_")
    out = os.path.join(root, f'fig_vortex_inset_{stem}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot vortex gyration results')
    parser.add_argument('--root', default='out/bench_vortex_gyration',
                        help='Output directory from bench_vortex_gyration')
    parser.add_argument('--mode', choices=['cfft', 'comp', 'both'], default='both',
                        help='Which solver to plot: cfft, comp, or both (default: both)')
    parser.add_argument('--t_cut', type=float, default=None,
                        help='Time cutoff in ns for Phase 3 data (e.g. 1.5). '
                             'If not set, uses all available Phase 3 data.')
    parser.add_argument('--snap', type=int, default=None,
                        help='Snapshot index for patch map / mz plots. '
                             'If not set, auto-selects max displacement snapshot.')
    parser.add_argument('--report', action='store_true',
                        help='Generate only the key report figures (trajectory, '
                             'frequency, patch map at max disp, 3D mz eq, mesh).')
    args = parser.parse_args()
    root = args.root
    mode = args.mode
    t_cut = args.t_cut
    snap_choice = args.snap
    report_mode = args.report

    if not os.path.isdir(root):
        print(f"Error: {root} not found")
        sys.exit(1)

    setup_style()  # Apply publication-quality styling globally

    print(f"Plotting from {root}/ (mode={mode}, t_cut={t_cut}ns)" if t_cut else f"Plotting from {root}/ (mode={mode})")
    info = load_grid_info(root)
    print(f"  Grid: {info.get('base_nx','?')}² base, "
          f"{info.get('fine_nx','?')}² fine, "
          f"disk R={info.get('disk_r_m',0)*1e9:.0f} nm")

    # ── Build method list based on mode ──
    # Each entry: (core_csv, snapshot_prefix, label, colour)
    #   snapshot_prefix: '' for cfft files (patches_*.csv), 'comp_' for comp files
    all_methods = []
    if mode in ('cfft', 'both'):
        if os.path.exists(os.path.join(root, 'core_amr_cfft.csv')):
            all_methods.append(('core_amr_cfft.csv', '', 'AMR + cfft', '#00AA00'))
    if mode in ('comp', 'both'):
        if os.path.exists(os.path.join(root, 'core_amr_comp.csv')):
            all_methods.append(('core_amr_comp.csv', 'comp_', 'AMR + composite', '#DD6600'))

    # Also check for fine/coarse (always include if present and mode=both)
    extra_core = []
    if mode == 'both':
        if os.path.exists(os.path.join(root, 'core_fine.csv')):
            extra_core.append(('core_fine.csv', '#0000CC', 'Fine FFT'))
        if os.path.exists(os.path.join(root, 'core_coarse.csv')):
            extra_core.append(('core_coarse.csv', '#CC0000', 'Coarse FFT'))

    if not all_methods:
        print(f"  No data found for mode={mode}")
        sys.exit(1)

    # ── Auto-select snapshot at maximum displacement if --snap not given ──
    def find_max_disp_snap(root, prefix, core_method):
        """Find the gyration snapshot index closest to maximum core displacement."""
        core_csv = f'core_{core_method}.csv'
        t_raw, x_raw, y_raw, mz_raw, cl_raw = load_core_csv(os.path.join(root, core_csv))
        if len(t_raw) == 0:
            return 2  # fallback: snapshot 2 (t≈0.6ns)
        t, x, y, _, _ = extract_phase3(t_raw, x_raw, y_raw, mz_raw, cl_raw)
        if len(t) == 0:
            return 2
        # Find time of max displacement from disk centre
        r = np.sqrt(x**2 + y**2)
        t_max = t[np.argmax(r)]
        # Map to snapshot index: snap_idx N corresponds to t=(N+1)*0.2 ns
        snap_idx = max(0, int(round(t_max / 0.2)) - 1)
        return snap_idx

    # ── Core trajectory and frequency ──
    plot_trajectory(root, info, all_methods, extra_core, t_cut_ns=t_cut)
    plot_frequency(root, info, all_methods, extra_core, t_cut_ns=t_cut)

    if not report_mode:
        plot_core_xt(root, info, all_methods, extra_core, t_cut_ns=t_cut)

    # ── Per-method snapshot plots ──
    for core_csv, prefix, method_label, colour in all_methods:
        method_tag = 'cfft' if prefix == '' else 'comp'
        core_method = 'amr_cfft' if prefix == '' else 'amr_comp'

        # Determine which snapshot to use for the "key" patch map
        if snap_choice is not None:
            key_snap = snap_choice
        else:
            key_snap = find_max_disp_snap(root, prefix, core_method)
        print(f"  [{method_label}] Key snapshot index: {key_snap} (t≈{key_snap*0.2:.1f} ns)")

        # Equilibrium patch map
        eq_patches = f'{prefix}patches_eq.csv' if prefix else 'patches_eq.csv'
        if not os.path.exists(os.path.join(root, eq_patches)):
            eq_patches = 'patches_eq.csv'  # fall back to shared eq file

        # Equilibrium mesh (García-Cervera style)
        eq_mfine = f'{prefix}m_fine_eq.csv'
        if not os.path.exists(os.path.join(root, eq_mfine)):
            eq_mfine = 'm_fine_eq.csv'

        if os.path.exists(os.path.join(root, eq_patches)):
            if not report_mode:
                plot_patch_map(root, info, eq_patches,
                               f'AMR Patches — Equilibrium ({method_label})',
                               f'fig_patch_map_eq_{method_tag}.pdf',
                               snap_idx=None, core_method=core_method)

        # Key gyration patch map (at max displacement or user-selected snap)
        key_patches = f'{prefix}patches_{key_snap:03d}.csv'
        if os.path.exists(os.path.join(root, key_patches)):
            t_key = get_snapshot_time(root, info, key_snap)
            plot_patch_map(root, info, key_patches,
                           f'AMR Patches — {method_label} t={t_key:.1f} ns',
                           f'fig_patch_map_gyr_{method_tag}_{key_snap:03d}.pdf',
                           snap_idx=key_snap, core_method=core_method)

        # All gyration patch maps (skip in report mode)
        if not report_mode:
            for f in sorted(Path(root).glob(f'{prefix}patches_0*.csv')):
                idx_str = f.stem.replace(f'{prefix}patches_', '')
                idx = int(idx_str)
                if idx == key_snap:
                    continue  # already plotted above
                t_ns = get_snapshot_time(root, info, idx)
                plot_patch_map(root, info, f.name,
                               f'AMR Patches — {method_label} t={t_ns:.1f} ns',
                               f'fig_patch_map_gyr_{method_tag}_{idx_str}.pdf',
                               snap_idx=idx, core_method=core_method)

        # Equilibrium mz colourmap
        if os.path.exists(os.path.join(root, eq_mfine)):
            if not report_mode:
                plot_mz_colourmap(root, info, eq_mfine,
                                  f'Magnetisation — Equilibrium ({method_label})',
                                  f'fig_mz_eq_{method_tag}.pdf',
                                  snap_idx=None, core_method=core_method)

        # Gyration mz colourmaps (skip in report mode)
        if not report_mode:
            for f in sorted(Path(root).glob(f'{prefix}m_coarse_0*.csv')):
                stem = f.stem
                idx_str = stem.replace(f'{prefix}m_coarse_', '')
                idx = int(idx_str)
                t_ns = get_snapshot_time(root, info, idx)
                plot_mz_colourmap(root, info, f.name,
                                  f'{method_label} — t={t_ns:.1f} ns',
                                  f'fig_mz_gyr_{method_tag}_{idx_str}.pdf',
                                  snap_idx=idx, core_method=core_method)

            for f in sorted(Path(root).glob(f'{prefix}m_fine_0*.csv')):
                stem = f.stem
                idx_str = stem.replace(f'{prefix}m_fine_', '')
                idx = int(idx_str)
                t_ns = get_snapshot_time(root, info, idx)
                plot_mz_colourmap(root, info, f.name,
                                  f'{method_label} (fine) — t={t_ns:.1f} ns',
                                  f'fig_mz_fine_{method_tag}_{idx_str}.pdf',
                                  snap_idx=idx, core_method=core_method)

        # 3D mz — equilibrium (always generate)
        if os.path.exists(os.path.join(root, eq_mfine)):
            plot_mz_3d(root, info, eq_mfine, frame_label=f'Equilibrium ({method_label})')

        # 3D mz — key snapshot
        key_mfine = f'{prefix}m_fine_{key_snap:03d}.csv'
        if os.path.exists(os.path.join(root, key_mfine)):
            t_key = get_snapshot_time(root, info, key_snap)
            plot_mz_3d(root, info, key_mfine,
                       frame_label=f'{method_label} t={t_key:.1f} ns')

        # 3D mz — all fine snapshots (skip in report mode)
        if not report_mode:
            for f in sorted(Path(root).glob(f'{prefix}m_fine_0*.csv')):
                stem = f.stem
                idx_str = stem.replace(f'{prefix}m_fine_', '')
                idx = int(idx_str)
                if idx == key_snap:
                    continue
                t_ns = get_snapshot_time(root, info, idx)
                plot_mz_3d(root, info, f.name,
                           frame_label=f'{method_label} t={t_ns:.1f} ns')

        # Mesh full domain (García-Cervera style) — at equilibrium
        eq_patches_path = os.path.join(root, eq_patches)
        eq_mfine_path = os.path.join(root, eq_mfine)
        if os.path.exists(eq_patches_path) and os.path.exists(eq_mfine_path):
            plot_mesh_full(root, info, eq_patches, eq_mfine,
                           frame_label=f'Equilibrium ({method_label})')

        # Mesh full domain at key snapshot (same time as patch map)
        key_patches_file = f'{prefix}patches_{key_snap:03d}.csv'
        key_mfine_file = f'{prefix}m_fine_{key_snap:03d}.csv'
        # Fall back to coarse if fine not available at this snapshot
        if not os.path.exists(os.path.join(root, key_mfine_file)):
            key_mfine_file = f'{prefix}m_coarse_{key_snap:03d}.csv'
        if (os.path.exists(os.path.join(root, key_patches_file)) and
            os.path.exists(os.path.join(root, key_mfine_file))):
            t_key = get_snapshot_time(root, info, key_snap)
            plot_mesh_full(root, info, key_patches_file, key_mfine_file,
                           frame_label=f'{method_label} t={t_key:.1f} ns')

        # Zoomed mesh — García-Cervera Fig 5 style (at key snapshot)
        if (os.path.exists(os.path.join(root, key_patches_file)) and
            os.path.exists(os.path.join(root, key_mfine_file))):
            t_key = get_snapshot_time(root, info, key_snap)
            kx, ky = get_core_at_time(root, t_key, method=core_method)
            plot_mesh_zoom(root, info, key_patches_file, key_mfine_file,
                           frame_label=f'{method_label} t={t_key:.1f} ns',
                           core_x_nm=kx, core_y_nm=ky, zoom_base_cells=22)
        # Also at equilibrium
        if os.path.exists(eq_patches_path) and os.path.exists(eq_mfine_path):
            plot_mesh_zoom(root, info, eq_patches, eq_mfine,
                           frame_label=f'Equilibrium ({method_label})',
                           core_x_nm=0, core_y_nm=0, zoom_base_cells=22)

        # 1D mz cross-sections (skip in report mode)
        if not report_mode:
            if os.path.exists(os.path.join(root, eq_mfine)):
                plot_mz_cross_sections(root, info,
                                       [eq_mfine],
                                       [f'Equilibrium ({method_label})'],
                                       outname=f'fig_mz_xsec_eq_{method_tag}.pdf')

            fine_snaps = sorted(Path(root).glob(f'{prefix}m_fine_0*.csv'))
            if fine_snaps and os.path.exists(os.path.join(root, eq_mfine)):
                snap_indices = []
                for f in fine_snaps:
                    stem = f.stem
                    idx_str = stem.replace(f'{prefix}m_fine_', '')
                    snap_indices.append((int(idx_str), f.name))
                if snap_indices:
                    mid_target = max(si[0] for si in snap_indices) // 2
                    if mid_target == 0:
                        mid_target = max(si[0] for si in snap_indices)
                    best = min(snap_indices, key=lambda si: abs(si[0] - mid_target))
                    mid_idx, mid_file = best
                    t_mid = get_snapshot_time(root, info, mid_idx)
                    plot_mz_cross_sections(root, info,
                                           [eq_mfine, mid_file],
                                           [f'Equilibrium', f'{method_label} t={t_mid:.1f} ns'],
                                           outname=f'fig_mz_xsec_compare_{method_tag}.pdf')

        # Vector field quiver plots — equilibrium + key snapshot
        if os.path.exists(os.path.join(root, eq_mfine)):
            plot_vector_field(root, info, eq_mfine,
                              patches_file=eq_patches if os.path.exists(os.path.join(root, eq_patches)) else None,
                              frame_label=f'Equilibrium ({method_label})',
                              core_method=core_method, snap_idx=None)
            # Equilibrium inset quiver
            plot_vortex_inset_quiver(root, info, eq_mfine,
                                     snap_idx=None, core_method=core_method)
        key_fine_for_qv = f'{prefix}m_fine_{key_snap:03d}.csv'
        key_patches_for_qv = f'{prefix}patches_{key_snap:03d}.csv'
        if os.path.exists(os.path.join(root, key_fine_for_qv)):
            t_key = get_snapshot_time(root, info, key_snap)
            plot_vector_field(root, info, key_fine_for_qv,
                              patches_file=key_patches_for_qv if os.path.exists(os.path.join(root, key_patches_for_qv)) else None,
                              frame_label=f'{method_label} t={t_key:.1f} ns',
                              core_method=core_method, snap_idx=key_snap)

            # Small inset quiver (for Illustrator compositing onto 3D mz)
            plot_vortex_inset_quiver(root, info, key_fine_for_qv,
                                     snap_idx=key_snap, core_method=core_method)

    print("\nDone.")


if __name__ == '__main__':
    main()