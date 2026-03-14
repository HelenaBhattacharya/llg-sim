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
    """Estimate gyration frequency via zero-crossing analysis."""
    crossings = []
    for i in range(1, len(x_nm)):
        if x_nm[i-1] * x_nm[i] < 0 and x_nm[i-1] != 0:
            frac = abs(x_nm[i-1]) / (abs(x_nm[i-1]) + abs(x_nm[i]))
            tc = t_ns[i-1] + frac * (t_ns[i] - t_ns[i-1])
            crossings.append(tc)
    if len(crossings) < 3:
        return 0.0, len(crossings)
    periods = [crossings[i+2] - crossings[i] for i in range(len(crossings) - 2)]
    avg_period = np.mean(periods)
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
    snap_every=40000 steps, dt=5fs → 200ps per snapshot."""
    # Read from grid_info if available, otherwise estimate
    snap_every = 40000
    dt_s = 5e-15
    # Phase 3 starts after relax+field_relax
    t_ns = snap_idx * snap_every * dt_s * 1e9
    return t_ns


def get_core_at_time(root, t_target_ns, method='amr_cfft'):
    """Get core position at a specific time from the trajectory CSV."""
    t, x, y, _, _ = load_core_csv(os.path.join(root, f'core_{method}.csv'))
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

def plot_trajectory(root, info, methods, extra_core=None):
    """Guslienko Fig 2 style: damped spiral, centred on orbit, with time colouring."""
    R_nm = info['disk_r_m'] * 1e9

    fig, ax = plt.subplots(figsize=(6, 7))

    # Plot extra core trajectories (fine/coarse) if provided
    if extra_core:
        for csv_name, color, label in extra_core:
            t, x, y, _, _ = load_core_csv(os.path.join(root, csv_name))
            if len(t) == 0:
                continue
            xn, yn = x / R_nm, y / R_nm
            ax.plot(xn, yn, color=color, lw=0.8, alpha=0.5, label=label)

    # Plot each method with time colouring
    cmap_choices = {'AMR + cfft': 'Greens', 'AMR + composite': 'Oranges'}
    for core_csv, prefix, label, color in methods:
        t, x, y, _, _ = load_core_csv(os.path.join(root, core_csv))
        if len(t) == 0:
            continue

        xn = x / R_nm
        yn = y / R_nm

        # Smooth the trajectory
        xs = smooth_trajectory(xn)
        ys = smooth_trajectory(yn)

        # Raw data as faint background
        ax.plot(xn, yn, color=color, lw=0.4, alpha=0.25)

        # Time-coloured smooth trajectory
        points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(t.min(), t.max())
        cmap_name = cmap_choices.get(label, 'Blues')
        cmap_traj = matplotlib.colormaps[cmap_name]
        lc = LineCollection(segments, cmap=cmap_traj, norm=norm, linewidths=1.8)  # type: ignore[arg-type]
        lc.set_array(t[:-1])
        ax.add_collection(lc)

        # Start and end markers with coordinate labels
        ax.plot(xs[0], ys[0], 'o', color=color,
                ms=7, zorder=5, markeredgecolor='k', markeredgewidth=0.5)
        ax.plot(xs[-1], ys[-1], 's', color=color,
                ms=7, zorder=5, markeredgecolor='k', markeredgewidth=0.5)

        # Label start/end with coordinates
        ax.annotate(f'{label}\nStart ({x[0]:.0f}, {y[0]:.0f}) nm',
                     xy=(xs[0], ys[0]), xytext=(25, 15),
                     textcoords='offset points', fontsize=8.5,
                     arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8),
                     color='0.2', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
        ax.annotate(f'End ({x[-1]:.0f}, {y[-1]:.0f}) nm',
                     xy=(xs[-1], ys[-1]), xytext=(-70, -25),
                     textcoords='offset points', fontsize=8.5,
                     arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8),
                     color='0.2', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

        # Add colourbar for time
        cbar = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.02, label=f't (ns) — {label}')

    # Centre on orbit: compute bounding box of the trajectory data
    all_x, all_y = [], []
    for core_csv, prefix, label, colour in methods:
        t, x, y, _, _ = load_core_csv(os.path.join(root, core_csv))
        if len(t) > 0:
            all_x.extend(x / R_nm)
            all_y.extend(y / R_nm)
    if all_x:
        cx = (min(all_x) + max(all_x)) / 2
        cy = (min(all_y) + max(all_y)) / 2
        hw = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) / 2 * 1.3
        hw = max(hw, 0.15)  # minimum half-width
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(cy - hw, cy + hw)

    ax.set_xlabel('X / R', fontsize=13)
    ax.set_ylabel('Y / R', fontsize=13)
    ax.set_title('Vortex Core Trajectory', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    out = os.path.join(root, 'fig_trajectory.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 2: Core x(t) — Damped Oscillation with envelope
# ─────────────────────────────────────────────────────────────────

def plot_core_xt(root, info, methods, extra_core=None):
    """Core x(t) with Savitzky-Golay smooth, exponential envelope, and annotations."""
    fig, ax = plt.subplots(figsize=(9, 4))

    # Build combined list of (csv_name, color, label)
    plot_list = []
    if extra_core:
        for csv_name, color, label in extra_core:
            plot_list.append((csv_name, color, label))
    for core_csv, prefix, label, color in methods:
        plot_list.append((core_csv, color, label))

    for name, color, label in plot_list:
        t, x, y, _, _ = load_core_csv(os.path.join(root, name))
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
    ax.set_title('Core x(t) — Gyration', fontsize=13)
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

def plot_frequency(root, info, methods, extra_core=None):
    """Frequency vs Guslienko analytic + Novosad experimental data."""
    beta = info['dz_m'] / info['disk_r_m']

    # Our measured frequencies
    f_methods = []
    # Extra (fine/coarse)
    if extra_core:
        for csv_name, color, label in extra_core:
            t, x, _, _, _ = load_core_csv(os.path.join(root, csv_name))
            if len(t) > 0:
                f_sim, nc = estimate_frequency(t, x)
                if f_sim > 0:
                    f_methods.append((label, f_sim, nc, color))
    # Main methods
    for core_csv, prefix, label, color in methods:
        t, x, _, _, _ = load_core_csv(os.path.join(root, core_csv))
        if len(t) > 0:
            f_sim, nc = estimate_frequency(t, x)
            if f_sim > 0:
                f_methods.append((label, f_sim, nc, color))

    # Novosad experimental data
    novosad_beta = np.array([0.020, 0.036, 0.073])
    novosad_f = np.array([0.083, 0.162, 0.272])  # GHz

    # Guslienko empirical: f = 3.7 * beta GHz
    beta_curve = np.linspace(0.01, 0.25, 100)
    f_guslienko = 3.7 * beta_curve

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(beta_curve, f_guslienko * 1000, 'k-', lw=1.5,
            label='Guslienko empirical (3.7β)')

    ax.scatter(novosad_beta, np.array(novosad_f) * 1000, s=80, c='blue', marker='s',
               zorder=5, label='Novosad et al. (2005) expt')

    # Plot all available methods
    markers = ['*', 'D', '^', 'v', 'o', 's']
    for i, (lbl, f, nc, col) in enumerate(f_methods):
        ax.scatter(np.array([beta]), np.array([f * 1000]),
                   s=120, c=col,
                   marker=markers[i % len(markers)],
                   zorder=6, label=f'This work ({lbl}): {f:.3f} GHz ({nc} cr.)')

    ax.set_xlabel('Aspect ratio β = L/R', fontsize=12)
    ax.set_ylabel('Frequency (MHz)', fontsize=12)
    ax.set_title('Vortex Gyration Eigenfrequency vs Aspect Ratio', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 1000)

    out = os.path.join(root, 'fig_frequency.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────
# Figure 4: Patch Map (with core position marker)
# ─────────────────────────────────────────────────────────────────

def plot_patch_map(root, info, patches_file, title, outname, snap_idx=None, core_method='amr_cfft'):
    """Patch map with disk outline and core position cross marker."""
    patches = load_patches_csv(os.path.join(root, patches_file))
    if not patches:
        return

    base_nx = info['base_nx']
    base_ny = info['base_ny']
    disk_r_cells = info['disk_r_m'] / info['dx_m']

    fig, ax = plt.subplots(figsize=(7, 7))

    # Disk outline
    cx, cy = base_nx / 2, base_ny / 2
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(cx + disk_r_cells * np.cos(theta),
            cy + disk_r_cells * np.sin(theta), 'k-', lw=2)

    # Patch rectangles
    colours = {1: '#FFD700', 2: '#00CC00', 3: '#0077FF'}
    labels = {1: 'L1', 2: 'L2', 3: 'L3'}
    drawn_labels = set()

    for level, i0, j0, nx, ny in patches:
        c = colours.get(level, 'grey')
        label = labels.get(level) if level not in drawn_labels else None
        rect = mpatches.Rectangle((i0, j0), nx, ny,
                                   linewidth=1.5, edgecolor=c,
                                   facecolor=c, alpha=0.15)
        ax.add_patch(rect)
        ax.add_patch(mpatches.Rectangle((i0, j0), nx, ny,
                                         linewidth=1.5, edgecolor=c,
                                         facecolor='none'))
        if label:
            drawn_labels.add(level)

    # Core position marker
    core_x_nm, core_y_nm = None, None
    if snap_idx is not None:
        core_x_nm, core_y_nm = get_core_at_snap(root, info, snap_idx, method=core_method)
    else:
        # Equilibrium (B=0): vortex core sits at disk centre = (0, 0) nm
        core_x_nm, core_y_nm = 0.0, 0.0

    if core_x_nm is not None and core_y_nm is not None:
        # Convert nm to base-cell coordinates
        dx_nm = info['dx_m'] * 1e9
        core_i = core_x_nm / dx_nm + base_nx / 2
        core_j = core_y_nm / dx_nm + base_ny / 2
        ax.plot(core_i, core_j, 'x', color='white', ms=12, mew=3, zorder=10)
        ax.plot(core_i, core_j, 'x', color='black', ms=10, mew=2, zorder=11)

    # Manual legend
    legend_patches = [mpatches.Patch(facecolor=colours[l], edgecolor=colours[l],
                                      alpha=0.4, label=f'L{l}') for l in sorted(colours)]
    ax.legend(handles=legend_patches, fontsize=11, loc='upper right')

    ax.set_xlim(0, base_nx)
    ax.set_ylim(0, base_ny)
    ax.set_aspect('equal')
    ax.set_xlabel('i (base cells)', fontsize=11)
    ax.set_ylabel('j (base cells)', fontsize=11)
    ax.set_title(title, fontsize=13)

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

    ax.set_title(title, fontsize=13)
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
    """3D surface plot of mz with aggressive boundary masking, colourbar, 3 views."""
    m = load_m_csv(os.path.join(root, m_file))
    if m is None:
        print(f"  Skipping 3D plot: {m_file} not found")
        return

    mz = m['mz']
    nx, ny = m['nx'], m['ny']

    # Subsample for performance
    step = max(1, nx // 160)
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

    # Aggressive mask: R * 0.92 to clip boundary artefacts
    R_nm = info['disk_r_m'] * 1e9
    dist = np.sqrt(X**2 + Y**2)
    mz_plot = np.where(dist <= R_nm * 0.92, mz_sub, np.nan)

    norm = Normalize(vmin=-0.1, vmax=1.0)
    cmap = plt.colormaps['coolwarm']

    # Three views: perspective, side, bird's-eye
    views = [
        ('Perspective', 25, -60),
        ('Side view', 5, 0),
        ("Bird's eye", 90, 0),
    ]

    fig = plt.figure(figsize=(18, 6))

    for vi, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, vi + 1, projection='3d')
        assert isinstance(ax, Axes3D)

        colours = cmap(norm(np.nan_to_num(mz_plot, nan=0)))
        # Set NaN cells to transparent
        nan_mask = np.isnan(mz_plot)
        colours[nan_mask, 3] = 0.0

        surf = ax.plot_surface(X, Y, mz_plot, facecolors=colours,
                               rstride=1, cstride=1,
                               linewidth=0.1 if vi < 2 else 0,
                               edgecolor='k' if vi < 2 else 'none',
                               alpha=0.9, antialiased=True)

        # Floor contour for perspective view
        if vi == 0:
            z_floor = ax.get_zlim()[0]
            try:
                mz_floor = np.nan_to_num(mz_plot, nan=0)
                ax.contourf(X, Y, mz_floor, levels=20, zdir='z',
                            offset=z_floor, cmap='coolwarm', alpha=0.4)
            except Exception:
                pass

        ax.set_xlabel('x (nm)', fontsize=10)
        ax.set_ylabel('y (nm)', fontsize=10)
        if vi < 2:
            ax.set_zlabel('m$_z$', fontsize=10)
        ax.set_title(f'{view_name}', fontsize=11)
        ax.view_init(elev=elev, azim=azim)

    # Overall title with frame label
    fig.suptitle(f'Vortex Core — m$_z$ Profile ({frame_label})', fontsize=14, y=1.02)

    # Add colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.08, label='m$_z$')

    out_base = os.path.splitext(m_file)[0]
    outname = f'fig_mz_3d_{out_base.replace("m_fine_", "").replace("m_coarse_", "c_")}.pdf'
    out = os.path.join(root, outname)
    fig.savefig(out, dpi=200, bbox_inches='tight')
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
    rgb = angle_colormap(m['mx'], m['my'], m['mz'], disk_mask=disk_mask)

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

    # Legend
    legend_handles = []
    for l in sorted(colours_p):
        legend_handles.append(mpatches.Patch(facecolor='none', edgecolor=colours_p[l],
                                              linewidth=2, label=f'L{l}'))
    legend_handles.append(Line2D([0], [0], color='k', lw=2, label='Disk boundary'))
    ax.legend(handles=legend_handles, fontsize=10, loc='upper right')

    # Colour wheel inset
    draw_colour_wheel_inset(ax, pos=(0.78, 0.02), size=0.15)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect('equal')
    ax.set_title(f'Multi-Resolution Mesh — {frame_label}', fontsize=13)
    ax.axis('off')

    out = os.path.join(root, 'fig_mesh_full.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
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

    fig.suptitle('Vortex Core m$_z$ Cross-Section (cf. Novosad Fig 4)', fontsize=13)
    fig.tight_layout()

    out = os.path.join(root, outname)
    fig.savefig(out, dpi=300, bbox_inches='tight')
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
    args = parser.parse_args()
    root = args.root
    mode = args.mode

    if not os.path.isdir(root):
        print(f"Error: {root} not found")
        sys.exit(1)

    print(f"Plotting from {root}/ (mode={mode})")
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

    # ── Core trajectory and frequency (both solvers on same plot if available) ──
    plot_trajectory(root, info, all_methods, extra_core)
    plot_core_xt(root, info, all_methods, extra_core)
    plot_frequency(root, info, all_methods, extra_core)

    # ── Per-method snapshot plots ──
    for core_csv, prefix, method_label, colour in all_methods:
        method_tag = 'cfft' if prefix == '' else 'comp'
        core_method = 'amr_cfft' if prefix == '' else 'amr_comp'

        # Equilibrium patch map
        eq_patches = f'{prefix}patches_eq.csv' if prefix else 'patches_eq.csv'
        if not os.path.exists(os.path.join(root, eq_patches)):
            eq_patches = 'patches_eq.csv'  # fall back to shared eq file
        if os.path.exists(os.path.join(root, eq_patches)):
            plot_patch_map(root, info, eq_patches,
                           f'AMR Patches — Equilibrium ({method_label})',
                           f'fig_patch_map_eq_{method_tag}.pdf',
                           snap_idx=None, core_method=core_method)

        # Gyration patch maps
        for f in sorted(Path(root).glob(f'{prefix}patches_0*.csv')):
            idx_str = f.stem.replace(f'{prefix}patches_', '')
            idx = int(idx_str)
            t_ns = get_snapshot_time(root, info, idx)
            plot_patch_map(root, info, f.name,
                           f'AMR Patches — {method_label} t={t_ns:.1f} ns',
                           f'fig_patch_map_gyr_{method_tag}_{idx_str}.pdf',
                           snap_idx=idx, core_method=core_method)

        # Equilibrium mz colourmap
        eq_mfine = f'{prefix}m_fine_eq.csv'
        if not os.path.exists(os.path.join(root, eq_mfine)):
            eq_mfine = 'm_fine_eq.csv'
        if os.path.exists(os.path.join(root, eq_mfine)):
            plot_mz_colourmap(root, info, eq_mfine,
                              f'Magnetisation — Equilibrium ({method_label})',
                              f'fig_mz_eq_{method_tag}.pdf',
                              snap_idx=None, core_method=core_method)

        # Gyration mz colourmaps (coarse)
        for f in sorted(Path(root).glob(f'{prefix}m_coarse_0*.csv')):
            stem = f.stem
            idx_str = stem.replace(f'{prefix}m_coarse_', '')
            idx = int(idx_str)
            t_ns = get_snapshot_time(root, info, idx)
            plot_mz_colourmap(root, info, f.name,
                              f'{method_label} — t={t_ns:.1f} ns',
                              f'fig_mz_gyr_{method_tag}_{idx_str}.pdf',
                              snap_idx=idx, core_method=core_method)

        # Gyration mz colourmaps (fine, if available)
        for f in sorted(Path(root).glob(f'{prefix}m_fine_0*.csv')):
            stem = f.stem
            idx_str = stem.replace(f'{prefix}m_fine_', '')
            idx = int(idx_str)
            t_ns = get_snapshot_time(root, info, idx)
            plot_mz_colourmap(root, info, f.name,
                              f'{method_label} (fine) — t={t_ns:.1f} ns',
                              f'fig_mz_fine_{method_tag}_{idx_str}.pdf',
                              snap_idx=idx, core_method=core_method)

        # 3D mz (equilibrium)
        if os.path.exists(os.path.join(root, eq_mfine)):
            plot_mz_3d(root, info, eq_mfine, frame_label=f'Equilibrium ({method_label})')

        # 3D mz (fine snapshots)
        for f in sorted(Path(root).glob(f'{prefix}m_fine_0*.csv')):
            stem = f.stem
            idx_str = stem.replace(f'{prefix}m_fine_', '')
            idx = int(idx_str)
            t_ns = get_snapshot_time(root, info, idx)
            plot_mz_3d(root, info, f.name,
                       frame_label=f'{method_label} t={t_ns:.1f} ns')

        # Mesh full domain (first method only, using eq data)
        eq_patches_path = os.path.join(root, eq_patches)
        eq_mfine_path = os.path.join(root, eq_mfine)
        if os.path.exists(eq_patches_path) and os.path.exists(eq_mfine_path):
            plot_mesh_full(root, info, eq_patches, eq_mfine,
                           frame_label=f'Equilibrium ({method_label})')

        # 1D mz cross-sections
        if os.path.exists(os.path.join(root, eq_mfine)):
            plot_mz_cross_sections(root, info,
                                   [eq_mfine],
                                   [f'Equilibrium ({method_label})'],
                                   outname=f'fig_mz_xsec_eq_{method_tag}.pdf')

        # Compare eq vs mid-run snapshot
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

    print("\nDone.")


if __name__ == '__main__':
    main()