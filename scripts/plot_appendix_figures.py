#!/usr/bin/env python3
"""
scripts/plot_appendix_figures.py

  Figure A1 — Single-Spin Precession
  Figure A2 — Anisotropy Dynamics
  Figure A3 — DMI Chirality Selection
"""

import argparse, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

plt.rcParams.update({
    "font.family": "serif", "font.size": 14,
    "axes.labelsize": 15, "axes.titlesize": 15,
    "legend.fontsize": 13,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "lines.linewidth": 1.2,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ── I/O ──────────────────────────────────────────────────────

def load_rust_csv(p):
    return np.genfromtxt(p, delimiter=",", names=True, dtype=float)

def load_mumax_table(p):
    header = None
    with open(p) as f:
        for line in f:
            if line.startswith("# "): header = line.lstrip("# ").strip()
            if not line.startswith("#"): break
    if header is None: raise ValueError(f"No header in {p}")
    raw = header.split("\t")
    names, seen = [], {}
    for n in raw:
        n = re.sub(r"\s*\(.*?\)\s*", "", n).strip().replace(" ", "_")
        if not n: n = f"col{len(names)}"
        if n in seen: seen[n] += 1; n = f"{n}_{seen[n]}"
        else: seen[n] = 1
        names.append(n)
    data = np.loadtxt(p, comments="#")
    return {names[i]: data[:, i] for i in range(min(len(names), data.shape[1]))}

def load_conv(p):
    return np.genfromtxt(p, delimiter=",", names=True, dtype=float)

def _get(d, *keys):
    for k in keys:
        if k in d: return d[k]
    raise KeyError(f"None of {keys} found")


# ── Figure A1 ────────────────────────────────────────────────

def make_fig_a1(rust_path, mumax_path, conv_path, out_path,
                b0=1.0, alpha=0.01, theta_deg=5.0):

    gamma = 1.760_859_630_23e11
    omega = gamma * b0 / (1.0 + alpha**2)
    f_pred = gamma * b0 / (2 * np.pi)
    theta = np.radians(theta_deg)

    rust  = load_rust_csv(rust_path)
    mumax = load_mumax_table(mumax_path)
    conv  = load_conv(conv_path)

    t_r, my_r = rust["t"], rust["my"]
    t_m, my_m = _get(mumax, "t"), _get(mumax, "my")
    t_max = min(t_r[-1], t_m[-1])
    mr = t_r <= t_max * 1.001; mm = t_m <= t_max * 1.001
    t_r, my_r = t_r[mr], my_r[mr]
    t_m, my_m = t_m[mm], my_m[mm]

    # Analytical
    my_an = -np.sin(theta) * np.sin(omega * t_r) * np.exp(-alpha * omega * t_r)

    # RMSE vs MuMax
    my_m_i = np.interp(t_r, t_m, my_m)
    rmse_mumax = np.sqrt(np.mean((my_r - my_m_i)**2))
    max_err_mumax = np.max(np.abs(my_r - my_m_i))

    # FFT (both Rust and MuMax)
    dt_fft = np.median(np.diff(t_r))
    N_fft = len(my_r)
    win = np.hanning(N_fft)
    fft_r = np.abs(np.fft.rfft((my_r - my_r.mean()) * win))
    freqs = np.fft.rfftfreq(N_fft, d=dt_fft)
    i_peak = np.argmax(fft_r[1:]) + 1
    f_peak = freqs[i_peak]

    # MuMax FFT (interpolate to same uniform grid)
    my_m_uniform = np.interp(t_r, t_m, my_m)
    fft_m = np.abs(np.fft.rfft((my_m_uniform - my_m_uniform.mean()) * win))

    # ── Layout: 3 panels stacked vertically ──
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 10.0),
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.2]})

    # ═══ (a) Trajectory ═══
    ax = axes[0]

    # Analytical as thick transparent envelope
    ax.plot(t_r * 1e9, my_an, "-", color="#2ca02c", linewidth=4.0,
            alpha=0.3, zorder=1, label="Analytical")
    # Rust as thin solid line on top
    ax.plot(t_r * 1e9, my_r, "-", color="#1f77b4", linewidth=0.5,
            zorder=2, label="Rust (RK45)")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"$m_y$")
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_xlim(0, t_max * 1e9)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              borderaxespad=0.3, borderpad=0.4, handlelength=1.5)

    # Stats inside panel (a) — bottom-right where signal has decayed
    stat_text_a = (f"RMSE(Rust\u2013MuMax3) = {rmse_mumax:.1e}\n"
                   f"max|$\\Delta m_y$| = {max_err_mumax:.1e}")
    ax.text(0.97, 0.05, stat_text_a, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))

    # ═══ (b) FFT — Rust + MuMax + predicted ═══
    ax = axes[1]
    f_ghz = freqs * 1e-9

    ax.plot(f_ghz, fft_r / fft_r.max(), "-", color="#1f77b4", linewidth=1.2,
            label="Rust")
    ax.plot(f_ghz, fft_m / fft_m.max(), "--", color="#d62728", linewidth=1.0,
            alpha=0.7, label="MuMax3")
    ax.axvline(f_pred * 1e-9, color="#2ca02c", linestyle=":", linewidth=1.0,
               alpha=0.8, label="Analytical")

    ax.set_xlim(24, 32)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Normalised amplitude")
    ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              borderaxespad=0.3, borderpad=0.4, handlelength=1.5)

    # Stats inside panel (b) — bottom-left flat baseline region
    stat_text_b = (rf"$\gamma_0 B/2\pi$ = {f_pred*1e-9:.2f} GHz"
                   f"\nFFT peak = {f_peak*1e-9:.2f} GHz")
    ax.text(0.03, 0.95, stat_text_b, transform=ax.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))

    # ═══ (c) Convergence ═══
    ax = axes[2]
    dt_c = conv["dt"]

    methods = [
        ("Euler",  "error_euler",  "#1f77b4", "o", 1),
        ("RK23",   "error_rk23",   "#ff7f0e", "^", 3),
        ("RK4",    "error_rk4",    "#2ca02c", "s", 4),
        ("RK45",   "error_rk45",   "#9467bd", "D", 5),
    ]

    for name, col, color, marker, order in methods:
        if conv.dtype.names is None or col not in conv.dtype.names:
            print(f"  WARNING: column '{col}' not found in convergence CSV — skipping {name}")
            continue
        err = conv[col]
        valid = (err > 1e-16) & (err < 1.0)
        if np.sum(valid) < 4:
            print(f"  WARNING: only {np.sum(valid)} valid points for {name} — skipping")
            continue
        dv, ev = dt_c[valid], err[valid]

        ax.loglog(dv, ev, marker, color=color, markersize=5.5,
                  markeredgewidth=0, zorder=3, label=name)

        # Theoretical slope: anchor at centre of points above precision floor
        clean = ev > 1e-13
        if np.sum(clean) >= 4:
            dc, ec = dv[clean], ev[clean]
            mid = len(dc) // 2
            dt0, e0 = dc[mid], ec[mid]
            dt_line = np.logspace(np.log10(dc[0]) - 0.3,
                                  np.log10(dc[-1]) + 0.3, 80)
            e_line = e0 * (dt_line / dt0) ** order
            vis = (e_line > 5e-16) & (e_line < 5.0)
            ax.loglog(dt_line[vis], e_line[vis], "-", color=color,
                      linewidth=1.0, alpha=0.3, zorder=1)

            # Label at right end of clean data, offset right
            y_nudge = {"Euler": -14, "RK23": -12, "RK4": 0, "RK45": 0}.get(name, 0)
            ri = min(len(dc) - 1, int(len(dc) * 0.85))
            ax.annotate(f"$\\Delta t^{{{order}}}$",
                        xy=(dc[ri], ec[ri]), fontsize=13, color=color,
                        fontweight="bold",
                        xytext=(12, y_nudge), textcoords="offset points", va="center")

    ax.set_xlabel(r"Time step $\Delta t$ (s)")
    ax.set_ylabel("Absolute error after 1 period")
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylim(top=1.0)  # headroom for legend above Euler data
    ax.legend(loc="upper center", frameon=True, framealpha=0.95,
              ncol=4, columnspacing=0.3, handletextpad=0.15,
              borderaxespad=0.3,
              bbox_to_anchor=(0.35, 1.0))
    ax.yaxis.set_minor_locator(LogLocator(subs="auto", numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote Figure A1 → {out_path}")
    print(f"  RMSE = {rmse_mumax:.4e},  f_pred = {f_pred*1e-9:.4f} GHz,  f_peak = {f_peak*1e-9:.4f} GHz")


# ── Figure A2: dual y-axis ───────────────────────────────────

def make_fig_a2(rust_path, mumax_path, out_path):

    rust  = load_rust_csv(rust_path)
    mumax = load_mumax_table(mumax_path)
    t_r = rust["t"]; t_m = _get(mumax, "t")
    t_max = min(t_r[-1], t_m[-1])
    mr = t_r <= t_max * 1.001; mm = t_m <= t_max * 1.001
    t_r = t_r[mr]; t_m = t_m[mm]

    stride = max(1, len(t_m) // 200)

    fig, ax_l = plt.subplots(figsize=(6.5, 4.5))
    ax_r = ax_l.twinx()

    # ── Left axis: mx, my (large oscillation) ──
    c_mx, c_my = "#1f77b4", "#ff7f0e"

    # MuMax dots
    ax_l.plot(t_m[::stride]*1e9, _get(mumax,"mx")[mm][::stride], ".",
              color=c_mx, markersize=7.0, alpha=0.4, markeredgewidth=0, zorder=2)
    ax_l.plot(t_m[::stride]*1e9, _get(mumax,"my")[mm][::stride], ".",
              color=c_my, markersize=7.0, alpha=0.4, markeredgewidth=0, zorder=2,
              label="MuMax3 (dots)")

    # Rust lines
    ax_l.plot(t_r*1e9, rust["mx"][mr], "-", color=c_mx, linewidth=0.8,
              zorder=3, label=r"$m_x$")
    ax_l.plot(t_r*1e9, rust["my"][mr], "-", color=c_my, linewidth=0.8,
              zorder=3, label=r"$m_y$")

    ax_l.set_xlabel("Time (ns)")
    ax_l.set_ylabel(r"$m_x$, $m_y$", color="0.2")
    ax_l.set_xlim(0, t_max * 1e9)
    ax_l.tick_params(axis="y", labelcolor="0.2")

    # ── Right axis: mz (slow relaxation, zoomed scale) ──
    c_mz = "#2ca02c"

    ax_r.plot(t_m[::stride]*1e9, _get(mumax,"mz")[mm][::stride], ".",
              color=c_mz, markersize=7.0, alpha=0.45, markeredgewidth=0, zorder=2)
    ax_r.plot(t_r*1e9, rust["mz"][mr], "-", color=c_mz, linewidth=1.1,
              zorder=3, label=r"$m_z$")

    ax_r.set_ylabel(r"$m_z$", color=c_mz)
    ax_r.tick_params(axis="y", labelcolor=c_mz)

    # Combine legends — place ABOVE the plot as a horizontal strip, centred
    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    fig.legend(lines_l + lines_r, labels_l + labels_r,
               loc="upper center", ncol=4, frameon=True, framealpha=0.95,
               handletextpad=0.4, columnspacing=1.0,
               bbox_to_anchor=(0.5, 1.06))

    # RMSE below the plot
    rmse_strs = []
    for comp, lbl in [("mx", r"$m_x$"), ("my", r"$m_y$"), ("mz", r"$m_z$")]:
        m_interp = np.interp(t_r, t_m, _get(mumax, comp)[mm])
        rmse_c = np.sqrt(np.mean((rust[comp][mr] - m_interp)**2))
        rmse_strs.append(f"RMSE({lbl}) = {rmse_c:.1e}")

    fig.text(0.5, -0.02, "    ".join(rmse_strs),
             ha="center", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote Figure A2 → {out_path}")
    for s in rmse_strs: print(f"  {s}")


# ── OVF2 reader ──────────────────────────────────────────────

def load_ovf2(path):
    """Load MuMax3 OVF 2.0 binary file → (data[nz,ny,nx,3], nx, ny, dx)."""
    import struct
    header = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF in OVF header")
            s = line.decode("latin-1").strip()
            if s.startswith("# Begin: Data"):
                if "Binary 4" in s:
                    dt = np.float32; cb = 4
                elif "Binary 8" in s:
                    dt = np.float64; cb = 8
                else:
                    raise ValueError(f"Unknown format: {s}")
                break
            if ":" in s and s.startswith("#"):
                k, _, v = s.lstrip("# ").partition(":")
                header[k.strip().lower()] = v.strip()

        check = f.read(cb)
        if dt == np.float32:
            assert abs(struct.unpack("<f", check)[0] - 1234567.0) < 1
        else:
            assert abs(struct.unpack("<d", check)[0] - 123456789012345.0) < 1

        nx = int(header.get("xnodes", header.get("x nodes", 0)))
        ny = int(header.get("ynodes", header.get("y nodes", 0)))
        nz = int(header.get("znodes", header.get("z nodes", 0)))
        nc = int(header.get("valuedim", 3))
        raw = np.frombuffer(f.read(nx * ny * nz * nc * np.dtype(dt).itemsize), dtype=dt)
        data = raw.reshape((nz, ny, nx, nc))

    dx = float(header.get("xstepsize", header.get("x stepsize", 5e-9)))
    return data, nx, ny, dx


def ovf_midrow_slice(path):
    """Extract mid-row (j=ny/2) slice from OVF → (x, mx, my, mz)."""
    data, nx, ny, dx = load_ovf2(path)
    j = ny // 2
    slc = data[0, j, :, :]  # (nx, 3)
    x = (np.arange(nx) + 0.5) * dx
    return x, slc[:, 0], slc[:, 1], slc[:, 2]


def rust_midrow_slice(path):
    """Load Rust bloch_dmi CSV slice → (x, mx, my, mz)."""
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


# ── Figure A3: DMI chirality ──────────────────────────────────

def make_fig_a3(rust_dplus, rust_dminus, mumax_dplus, mumax_dminus, out_path):
    """
    Single-panel figure showing all 3 magnetisation components for ±D:
      mz: wall profile (solid) — identical for ±D
      mx: chirality component (dashed) — flips sign with D
      my: small residual (dotted) — nearly zero
    Rust = lines, MuMax3 = dots.  Blue = +D, red = −D.
    """
    from matplotlib.lines import Line2D

    # Load Rust slices
    xrp, mxrp, myrp, mzrp = rust_midrow_slice(rust_dplus)
    xrm, mxrm, myrm, mzrm = rust_midrow_slice(rust_dminus)

    # Load MuMax OVF slices
    xmp, mxmp, mymp, mzmp = ovf_midrow_slice(mumax_dplus)
    xmm, mxmm, mymm, mzmm = ovf_midrow_slice(mumax_dminus)

    # Centre on wall (minimum |mz|)
    def centre(x, mz):
        i0 = np.argmin(np.abs(mz))
        return (x - x[i0]) * 1e9  # nm

    xrp_c = centre(xrp, mzrp)
    xrm_c = centre(xrm, mzrm)
    xmp_c = centre(xmp, mzmp)
    xmm_c = centre(xmm, mzmm)

    # Colours: blue = +D, red = -D
    c_p, c_m = "#1f77b4", "#d62728"

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    stride = max(1, len(xmp_c) // 120)

    # ── MuMax dots (behind, all components) ──
    # +D MuMax
    ax.plot(xmp_c[::stride], mzmp[::stride], "o", color=c_p,
            markersize=5.5, alpha=0.45, markeredgewidth=0, zorder=2)
    ax.plot(xmp_c[::stride], mxmp[::stride], "o", color=c_p,
            markersize=5.5, alpha=0.45, markeredgewidth=0, zorder=2)
    ax.plot(xmp_c[::stride], mymp[::stride], "o", color=c_p,
            markersize=5.5, alpha=0.45, markeredgewidth=0, zorder=2)

    # -D MuMax
    ax.plot(xmm_c[::stride], mzmm[::stride], "s", color=c_m,
            markersize=5.0, alpha=0.45, markeredgewidth=0, zorder=2)
    ax.plot(xmm_c[::stride], mxmm[::stride], "s", color=c_m,
            markersize=5.0, alpha=0.45, markeredgewidth=0, zorder=2)
    ax.plot(xmm_c[::stride], mymm[::stride], "s", color=c_m,
            markersize=5.0, alpha=0.45, markeredgewidth=0, zorder=2)

    # ── Rust lines (on top), ordered by component for clean legend pairing ──
    # mz pair
    ax.plot(xrp_c, mzrp, "-",  color=c_p, linewidth=1.6, zorder=3,
            label=r"$+D\;m_z$")
    ax.plot(xrm_c, mzrm, "-",  color=c_m, linewidth=1.6, zorder=3,
            label=r"$-D\;m_z$")

    # mx pair
    ax.plot(xrp_c, mxrp, "--", color=c_p, linewidth=1.2, zorder=3,
            label=r"$+D\;m_x$")
    ax.plot(xrm_c, mxrm, "--", color=c_m, linewidth=1.2, zorder=3,
            label=r"$-D\;m_x$")

    # my pair
    ax.plot(xrp_c, myrp, ":",  color=c_p, linewidth=1.1, zorder=3,
            label=r"$+D\;m_y$")
    ax.plot(xrm_c, myrm, ":",  color=c_m, linewidth=1.1, zorder=3,
            label=r"$-D\;m_y$")

    ax.axhline(0, color="0.5", linewidth=0.4, zorder=1)
    ax.set_xlabel(r"$x - x_0$ (nm)")
    ax.set_ylabel("Magnetisation")
    ax.set_xlim(-500, 500)
    ax.set_ylim(-1.08, 1.08)

    # ── Collect legend handles ──
    mumax_handle = Line2D([], [], marker="o", color="0.55", markersize=7,
                          linestyle="None", alpha=0.5, markeredgewidth=0)

    handles, labels = ax.get_legend_handles_labels()
    all_handles = [mumax_handle] + handles
    all_labels  = ["MuMax3 (dots)"] + labels

    # ── RMSE: compute and simplify (±D values match by symmetry) ──
    rmse_vals = {}
    for comp_label, rust_p, rust_m, mu_p, mu_m in [
        ("m_z", mzrp, mzrm, mzmp, mzmm),
        ("m_x", mxrp, mxrm, mxmp, mxmm),
        ("m_y", myrp, myrm, mymp, mymm),
    ]:
        mu_p_i = np.interp(xrp_c, xmp_c, mu_p)
        mu_m_i = np.interp(xrm_c, xmm_c, mu_m)
        rmse_p = np.sqrt(np.mean((rust_p - mu_p_i)**2))
        rmse_m = np.sqrt(np.mean((rust_m - mu_m_i)**2))
        rmse_vals[comp_label] = 0.5 * (rmse_p + rmse_m)

    rmse_text = (f"RMSE(${list(rmse_vals.keys())[0]}$) = {list(rmse_vals.values())[0]:.1e}    "
                 f"RMSE(${list(rmse_vals.keys())[1]}$) = {list(rmse_vals.values())[1]:.1e}    "
                 f"RMSE(${list(rmse_vals.keys())[2]}$) = {list(rmse_vals.values())[2]:.1e}")

    fig.tight_layout(rect=(0, 0.05, 1, 0.92))

    # Now place legend and RMSE centred on the plot data window
    bb = ax.get_position()
    plot_centre_x = bb.x0 + bb.width / 2

    fig.legend(all_handles, all_labels,
               loc="upper center", ncol=4, frameon=True, framealpha=0.92,
               handletextpad=0.4, columnspacing=0.8,
               handlelength=2.0, borderpad=0.5,
               bbox_to_anchor=(plot_centre_x, 1.04))

    fig.text(plot_centre_x, 0.01, rmse_text,
             ha="center", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Wrote Figure A3 → {out_path}")
    for k, v in rmse_vals.items():
        print(f"  RMSE(${k}$) = {v:.1e}")


# ── CLI ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    p1 = sub.add_parser("fig_a1")
    p1.add_argument("--rust", required=True)
    p1.add_argument("--mumax", required=True)
    p1.add_argument("--convergence", required=True)
    p1.add_argument("--out", default="out/appendix/fig_A1_precession.png")
    p1.add_argument("--b0", type=float, default=1.0)
    p1.add_argument("--alpha", type=float, default=0.01)
    p1.add_argument("--theta", type=float, default=5.0)

    p2 = sub.add_parser("fig_a2")
    p2.add_argument("--rust", required=True)
    p2.add_argument("--mumax", required=True)
    p2.add_argument("--out", default="out/appendix/fig_A2_anisotropy.png")

    p3 = sub.add_parser("fig_a3")
    p3.add_argument("--rust-dplus", required=True,
                    help="Rust +D slice CSV (e.g. out/bloch_dmi/Dplus/rust_slice_final.csv)")
    p3.add_argument("--rust-dminus", required=True,
                    help="Rust -D slice CSV (e.g. out/bloch_dmi/Dminus/rust_slice_final.csv)")
    p3.add_argument("--mumax-dplus", required=True,
                    help="MuMax3 +D OVF file (e.g. mumax_outputs/bloch_dmi_plus/m_final_Dplus.ovf)")
    p3.add_argument("--mumax-dminus", required=True,
                    help="MuMax3 -D OVF file (e.g. mumax_outputs/bloch_dmi_minus/m_final_Dminus.ovf)")
    p3.add_argument("--out", default="out/appendix/fig_A3_dmi_chirality.png")

    args = p.parse_args()
    if args.cmd == "fig_a1":
        make_fig_a1(args.rust, args.mumax, args.convergence, args.out,
                     b0=args.b0, alpha=args.alpha, theta_deg=args.theta)
    elif args.cmd == "fig_a2":
        make_fig_a2(args.rust, args.mumax, args.out)
    elif args.cmd == "fig_a3":
        make_fig_a3(args.rust_dplus, args.rust_dminus,
                     args.mumax_dplus, args.mumax_dminus, args.out)
    else:
        p.print_help(); sys.exit(1)

if __name__ == "__main__":
    main()