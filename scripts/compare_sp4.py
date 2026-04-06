# scripts/compare_sp4.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray

# Style controls
MUMAX_MS = 2.0   # marker size for MuMax points
RUST_LW  = 1.6   # line width for Rust lines
VLINE_LW = 0.6   # vertical line width (optional)


def load_mumax_table(table_path: Path) -> Tuple[Array, Array, Array, Array]:
    """
    Load MuMax table.txt assuming first 4 numeric cols are: t, mx, my, mz.
    """
    data = np.loadtxt(table_path, comments="#")
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Expected >=4 columns in {table_path}, got shape {data.shape}")
    t = data[:, 0]  # seconds
    mx, my, mz = data[:, 1], data[:, 2], data[:, 3]
    return t, mx, my, mz


def load_rust_csv(csv_path: Path) -> Tuple[Array, Array, Array, Array]:
    """
    Load Rust CSV with a header. Expected columns: t_s,mx,my,mz
    (extra columns allowed).
    """
    raw = np.genfromtxt(csv_path, delimiter=",", names=True)
    names = raw.dtype.names or ()  # dtype.names can be None

    t_col = None
    for tkey in ("t_s", "t", "time_s"):
        if tkey in names:
            t_col = tkey
            break
    if t_col is None:
        raise ValueError(f"No time column found in {csv_path}. Header columns: {names}")

    for key in ("mx", "my", "mz"):
        if key not in names:
            raise ValueError(f"Missing '{key}' in {csv_path}. Header columns: {names}")

    t = raw[t_col]
    mx = raw["mx"]
    my = raw["my"]
    mz = raw["mz"]
    return t, mx, my, mz


def first_zero_crossing_time(t: Array, x: Array) -> Optional[float]:
    """
    Interpolated time (same units as t) where x first crosses 0.
    Returns None if no crossing.
    """
    s = np.sign(x).astype(float)
    s[s == 0.0] = 1.0
    idx = np.where(s[1:] * s[:-1] < 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])

    t0, t1 = float(t[i]), float(t[i + 1])
    x0, x1 = float(x[i]), float(x[i + 1])

    if x1 == x0:
        return None
    return t0 + (0.0 - x0) * (t1 - t0) / (x1 - x0)


def plot_triplet(
    ax,
    t_ns: Array,
    mx: Array,
    my: Array,
    mz: Array,
    *,
    marker: Optional[str],
    linestyle: Optional[str],
    prefix: str,
    ms: float = 2.0,
    lw: float = 1.6,
):
    """
    Plot mx,my,mz on the given axes with consistent style.
    - Use marker="o", linestyle="None" for points-only
    - Use marker=None, linestyle="-" for line-only
    """
    ax.plot(t_ns, mx, label=f"{prefix}mx", marker=marker, linestyle=linestyle, markersize=ms, linewidth=lw)
    ax.plot(t_ns, my, label=f"{prefix}my", marker=marker, linestyle=linestyle, markersize=ms, linewidth=lw)
    ax.plot(t_ns, mz, label=f"{prefix}mz", marker=marker, linestyle=linestyle, markersize=ms, linewidth=lw)


# ----------------------------
# Metrics helpers
# ----------------------------

def overlap_window(
    t1: Array,
    t2: Array,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
) -> Tuple[float, float]:
    lo = float(max(t1[0], t2[0]))
    hi = float(min(t1[-1], t2[-1]))
    if tmin is not None:
        lo = max(lo, float(tmin))
    if tmax is not None:
        hi = min(hi, float(tmax))
    if hi <= lo:
        raise ValueError(f"No overlap window: lo={lo}, hi={hi}")
    return lo, hi


def clip_time(t: Array, *ys: Array, lo: float, hi: float) -> Tuple[Array, ...]:
    mask = (t >= lo) & (t <= hi)
    out = (t[mask],)
    for y in ys:
        out += (y[mask],)
    return out


# New metrics helpers
def _sanitize_xy(t: Array, y: Array) -> Tuple[Array, Array]:
    """Remove NaN/inf, sort by time, and drop duplicate time entries."""
    mask = np.isfinite(t) & np.isfinite(y)
    t2 = np.asarray(t[mask], dtype=float)
    y2 = np.asarray(y[mask], dtype=float)

    if t2.size < 2:
        return t2, y2

    order = np.argsort(t2)
    t2 = t2[order]
    y2 = y2[order]

    # Drop duplicate times (keep first occurrence)
    _, idx = np.unique(t2, return_index=True)
    idx.sort()
    return t2[idx], y2[idx]


def metrics_on_grid(
    t_ref: Array,
    y_ref: Array,
    t_other: Array,
    y_other: Array,
) -> Tuple[float, float, float, float]:
    """
    Interpolate y_other(t_other) onto t_ref and compute:
      - RMSE
      - max |Δm|
      - p95 |Δm|
      - t_at_max (seconds)

    This is robust to NaNs/inf and non-monotonic/duplicate time stamps.
    """
    t_ref2, y_ref2 = _sanitize_xy(t_ref, y_ref)
    t_oth2, y_oth2 = _sanitize_xy(t_other, y_other)

    if t_ref2.size < 2 or t_oth2.size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    y_oth_i = np.interp(t_ref2, t_oth2, y_oth2)
    res = y_oth_i - y_ref2

    rmse_val = float(np.sqrt(np.mean(res * res)))
    abs_res = np.abs(res)
    max_abs = float(np.max(abs_res))
    p95_abs = float(np.quantile(abs_res, 0.95))
    i_max = int(np.argmax(abs_res))
    t_at_max = float(t_ref2[i_max])

    return rmse_val, max_abs, p95_abs, t_at_max


def _pct_full_scale(x: float) -> float:
    """Convert an absolute magnetisation error (|Δm|) to percent of full-scale (FS=1)."""
    return 100.0 * float(x)


def print_metrics_block(
    label: str,
    t_m: Array, mx_m: Array, my_m: Array, mz_m: Array,
    t_r: Array, mx_r: Array, my_r: Array, mz_r: Array,
    *,
    tmin: Optional[float],
    tmax: Optional[float],
    interp_to: str,
) -> None:
    lo, hi = overlap_window(t_m, t_r, tmin=tmin, tmax=tmax)

    # clip both to the same time window first
    t_m2, mx_m2, my_m2, mz_m2 = clip_time(t_m, mx_m, my_m, mz_m, lo=lo, hi=hi)
    t_r2, mx_r2, my_r2, mz_r2 = clip_time(t_r, mx_r, my_r, mz_r, lo=lo, hi=hi)

    if len(t_m2) < 2 or len(t_r2) < 2:
        print(f"[metrics] {label}: not enough points after clipping (lo={lo:.3e}, hi={hi:.3e})")
        return

    # choose reference grid
    if interp_to == "rust":
        tref = t_r2
        rm_mx, ma_mx, p95_mx, tmx = metrics_on_grid(tref, mx_r2, t_m2, mx_m2)
        rm_my, ma_my, p95_my, tmy = metrics_on_grid(tref, my_r2, t_m2, my_m2)
        rm_mz, ma_mz, p95_mz, tmz = metrics_on_grid(tref, mz_r2, t_m2, mz_m2)
    else:
        tref = t_m2
        rm_mx, ma_mx, p95_mx, tmx = metrics_on_grid(tref, mx_m2, t_r2, mx_r2)
        rm_my, ma_my, p95_my, tmy = metrics_on_grid(tref, my_m2, t_r2, my_r2)
        rm_mz, ma_mz, p95_mz, tmz = metrics_on_grid(tref, mz_m2, t_r2, mz_r2)

    print(f"\n[metrics] {label}  window: t in [{lo:.3e}, {hi:.3e}] s  (interp_to={interp_to})")
    print(
        f"  mx: RMSE={rm_mx:.6e}  max|Δm|={ma_mx:.3e} ({_pct_full_scale(ma_mx):.2f}%FS)  "
        f"p95|Δm|={p95_mx:.3e} ({_pct_full_scale(p95_mx):.2f}%FS)  t@max={tmx:.3e}s"
    )
    print(
        f"  my: RMSE={rm_my:.6e}  max|Δm|={ma_my:.3e} ({_pct_full_scale(ma_my):.2f}%FS)  "
        f"p95|Δm|={p95_my:.3e} ({_pct_full_scale(p95_my):.2f}%FS)  t@max={tmy:.3e}s"
    )
    print(
        f"  mz: RMSE={rm_mz:.6e}  max|Δm|={ma_mz:.3e} ({_pct_full_scale(ma_mz):.2f}%FS)  "
        f"p95|Δm|={p95_mz:.3e} ({_pct_full_scale(p95_mz):.2f}%FS)  t@max={tmz:.3e}s"
    )


# ----------------------------
# Dynamics diagnostics (NEW): frequency + phase drift
# ----------------------------

def _median_dt(t: Array) -> float:
    t2 = np.asarray(t, dtype=float)
    if t2.size < 3:
        return float("nan")
    d = np.diff(t2)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return float("nan")
    return float(np.median(d))


def _resample_uniform(t: Array, y: Array, *, t0: float, t1: float, dt: float) -> Tuple[Array, Array]:
    """Resample y(t) onto a uniform grid in [t0, t1] (inclusive-ish)."""
    if not np.isfinite(dt) or dt <= 0:
        return np.asarray([]), np.asarray([])

    # Avoid pathological point counts
    max_n = 200_000
    n = int(np.floor((t1 - t0) / dt)) + 1
    if n < 8:
        return np.asarray([]), np.asarray([])
    if n > max_n:
        dt = (t1 - t0) / float(max_n - 1)
        n = max_n

    tg = t0 + dt * np.arange(n, dtype=float)
    t2, y2 = _sanitize_xy(t, y)
    if t2.size < 2:
        return np.asarray([]), np.asarray([])

    yg = np.interp(tg, t2, y2)
    return tg, yg


def _hann(n: int) -> Array:
    if n <= 1:
        return np.ones((n,), dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=float) / float(n - 1))


def dominant_frequency_hz(t: Array, y: Array, *, fmin: float = 0.0, fmax: Optional[float] = None) -> float:
    """Estimate dominant frequency (Hz) in y(t) using FFT peak picking (after mean removal + Hann window)."""
    t2, y2 = _sanitize_xy(t, y)
    if t2.size < 8:
        return float("nan")

    dt = _median_dt(t2)
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")

    # uniform grid over the provided samples
    t0, t1 = float(t2[0]), float(t2[-1])
    tg, yg = _resample_uniform(t2, y2, t0=t0, t1=t1, dt=dt)
    if tg.size < 8:
        return float("nan")

    yg = yg - float(np.mean(yg))
    w = _hann(int(yg.size))
    x = yg * w

    # rFFT
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=dt)
    amp = np.abs(X)

    # ignore DC
    amp[0] = 0.0

    if fmin is not None and fmin > 0:
        amp[freqs < float(fmin)] = 0.0
    if fmax is not None and np.isfinite(fmax):
        amp[freqs > float(fmax)] = 0.0

    i = int(np.argmax(amp))
    if amp[i] <= 0:
        return float("nan")
    return float(freqs[i])


def _bandpass_fft(tg: Array, x: Array, *, f0: float, rel_bw: float) -> Array:
    """Crude FFT-domain bandpass around f0 with relative half-bandwidth rel_bw."""
    if tg.size < 8 or not np.isfinite(f0) or f0 <= 0:
        return x

    dt = float(tg[1] - tg[0])
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=dt)

    bw = float(rel_bw) * f0
    if not np.isfinite(bw) or bw <= 0:
        bw = 0.2 * f0

    f_lo = max(0.0, f0 - bw)
    f_hi = f0 + bw

    mask = (freqs >= f_lo) & (freqs <= f_hi)
    Xf = np.zeros_like(X)
    Xf[mask] = X[mask]
    xf = np.fft.irfft(Xf, n=x.size)
    return xf


def _analytic_signal(x: Array) -> Array:
    """Return analytic signal via Hilbert transform (FFT-based)."""
    n = int(x.size)
    if n < 8:
        return x.astype(complex)

    X = np.fft.fft(x)
    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1: n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1: (n + 1) // 2] = 2.0

    z = np.fft.ifft(X * h)
    return z


def phase_drift_slope(
    t: Array,
    y_ref: Array,
    y_other: Array,
    *,
    f0: float,
    rel_bw: float,
) -> Tuple[float, float]:
    """Estimate linear phase drift slope (rad/s) between y_other and y_ref, plus residual std (rad)."""
    if t.size < 16:
        return float("nan"), float("nan")

    # remove mean and (optionally) bandpass around f0
    x0 = np.asarray(y_ref, dtype=float) - float(np.mean(y_ref))
    x1 = np.asarray(y_other, dtype=float) - float(np.mean(y_other))

    x0 = _bandpass_fft(t, x0, f0=f0, rel_bw=rel_bw)
    x1 = _bandpass_fft(t, x1, f0=f0, rel_bw=rel_bw)

    z0 = _analytic_signal(x0)
    z1 = _analytic_signal(x1)

    ph0 = np.unwrap(np.angle(z0))
    ph1 = np.unwrap(np.angle(z1))

    dph = ph1 - ph0
    dph = dph - float(dph[0])  # remove constant offset

    # Fit dph(t) ~ a*t + b
    a, b = np.polyfit(t, dph, 1)
    fit = a * t + b
    resid = float(np.std(dph - fit))

    return float(a), resid


def _pick_component_for_dyn(mx: Array, my: Array, mz: Array, *, mode: str) -> Tuple[str, Array]:
    if mode in ("mx", "my", "mz"):
        return mode, {"mx": mx, "my": my, "mz": mz}[mode]

    # auto: choose largest variance component
    v = {
        "mx": float(np.nanstd(mx)),
        "my": float(np.nanstd(my)),
        "mz": float(np.nanstd(mz)),
    }
    # Pylance type-fix: v.get returns Optional[float], so use __getitem__ for total ordering.
    key = max(v.keys(), key=lambda k: v[k])
    return key, {"mx": mx, "my": my, "mz": mz}[key]


def print_dynamics_block(
    label: str,
    t_m: Array, mx_m: Array, my_m: Array, mz_m: Array,
    t_r: Array, mx_r: Array, my_r: Array, mz_r: Array,
    *,
    dyn_tmin: Optional[float],
    dyn_tmax: Optional[float],
    dyn_component: str,
    dyn_rel_bw: float,
) -> None:
    """Print frequency + phase drift diagnostics (Rust vs MuMax) on an overlap window."""
    lo, hi = overlap_window(t_m, t_r)

    # default to "second half" if user didn't set dyn_tmin
    if dyn_tmin is None:
        dyn_lo = lo + 0.5 * (hi - lo)
    else:
        dyn_lo = max(lo, float(dyn_tmin))

    dyn_hi = hi if dyn_tmax is None else min(hi, float(dyn_tmax))
    if dyn_hi <= dyn_lo:
        print(f"[dyn] {label}: no valid dynamics window (lo={dyn_lo:.3e}, hi={dyn_hi:.3e})")
        return

    # clip
    t_m2, mx_m2, my_m2, mz_m2 = clip_time(t_m, mx_m, my_m, mz_m, lo=dyn_lo, hi=dyn_hi)
    t_r2, mx_r2, my_r2, mz_r2 = clip_time(t_r, mx_r, my_r, mz_r, lo=dyn_lo, hi=dyn_hi)

    if t_m2.size < 16 or t_r2.size < 16:
        print(f"[dyn] {label}: not enough points after clipping (n_m={t_m2.size}, n_r={t_r2.size})")
        return

    # choose component (based on MuMax over the window)
    comp, y_m = _pick_component_for_dyn(mx_m2, my_m2, mz_m2, mode=dyn_component)
    _, y_r = _pick_component_for_dyn(mx_r2, my_r2, mz_r2, mode=comp)

    # choose uniform dt
    dt_m = _median_dt(np.asarray(t_m2, dtype=float))
    dt_r = _median_dt(np.asarray(t_r2, dtype=float))
    dt = min(dt_m, dt_r)
    if not np.isfinite(dt) or dt <= 0:
        print(f"[dyn] {label}: could not determine dt (dt_m={dt_m}, dt_r={dt_r})")
        return

    tg, ymg = _resample_uniform(t_m2, y_m, t0=dyn_lo, t1=dyn_hi, dt=dt)
    tg2, yrg = _resample_uniform(t_r2, y_r, t0=dyn_lo, t1=dyn_hi, dt=dt)
    if tg.size < 32 or tg2.size < 32:
        print(f"[dyn] {label}: resampling produced too few points")
        return

    # They should match but be safe:
    n = min(tg.size, tg2.size)
    tg = tg[:n]
    ymg = ymg[:n]
    yrg = yrg[:n]

    # frequency estimates
    f_m = dominant_frequency_hz(tg, ymg)
    f_r = dominant_frequency_hz(tg, yrg)

    # phase drift estimate around the midpoint frequency
    f0 = float(np.nanmean([f_m, f_r]))
    slope, resid = phase_drift_slope(tg, ymg, yrg, f0=f0, rel_bw=float(dyn_rel_bw))

    # conversions
    df_fft = f_r - f_m if (np.isfinite(f_r) and np.isfinite(f_m)) else float("nan")
    df_phase = slope / (2.0 * np.pi) if np.isfinite(slope) else float("nan")
    drift_cycles_per_ns = (df_phase * 1e-9) if np.isfinite(df_phase) else float("nan")
    drift_cycles_end = (df_phase * (dyn_hi - dyn_lo)) if np.isfinite(df_phase) else float("nan")

    print(f"\n[dyn] {label}  window: t in [{dyn_lo:.3e}, {dyn_hi:.3e}] s  component={comp}")
    print(f"  f_mumax={f_m:.6e} Hz   f_rust={f_r:.6e} Hz   Δf_fft={df_fft:.6e} Hz")
    print(f"  phase drift: slope={slope:.6e} rad/s  ⇒ Δf_phase={df_phase:.6e} Hz  ({drift_cycles_per_ns:.6e} cycles/ns)")
    print(f"  drift over window: {drift_cycles_end:.3e} cycles   phase-fit residual std={resid:.3e} rad")


# ----------------------------
# Residuals plot helpers (NEW)
# ----------------------------

def residual_on_grid(
    t_ref: Array,
    y_ref: Array,
    t_other: Array,
    y_other: Array,
    *,
    sign: str = "rust-minus-mumax",
) -> Array:
    """
    Return residuals on t_ref grid, using linear interpolation of y_other onto t_ref.

    sign:
      - "rust-minus-mumax": residual = y_other_interp - y_ref   (when ref is MuMax)
      - "mumax-minus-rust": residual = y_ref - y_other_interp
    """
    t_ref2, y_ref2 = _sanitize_xy(t_ref, y_ref)
    t_oth2, y_oth2 = _sanitize_xy(t_other, y_other)
    y_other_i = np.interp(t_ref2, t_oth2, y_oth2)
    if sign == "rust-minus-mumax":
        return y_other_i - y_ref2
    if sign == "mumax-minus-rust":
        return y_ref2 - y_other_i
    raise ValueError(f"Unknown sign='{sign}'")


def save_residuals_figure(
    out_path: Path,
    # SP4a
    t_a: Array, mx_a: Array, my_a: Array, mz_a: Array,
    t_ra: Array, mx_ra: Array, my_ra: Array, mz_ra: Array,
    # SP4b
    t_b: Array, mx_b: Array, my_b: Array, mz_b: Array,
    t_rb: Array, mx_rb: Array, my_rb: Array, mz_rb: Array,
    *,
    dpi: int = 200,
) -> None:
    """
    Save a 2-panel residual plot: Δm(t) = Rust(t) - MuMax(t) on the MuMax time grid.
    """
    # Overlap windows
    lo_a, hi_a = overlap_window(t_a, t_ra)
    lo_b, hi_b = overlap_window(t_b, t_rb)

    # Clip to overlap
    t_a2, mx_a2, my_a2, mz_a2 = clip_time(t_a, mx_a, my_a, mz_a, lo=lo_a, hi=hi_a)
    t_ra2, mx_ra2, my_ra2, mz_ra2 = clip_time(t_ra, mx_ra, my_ra, mz_ra, lo=lo_a, hi=hi_a)

    t_b2, mx_b2, my_b2, mz_b2 = clip_time(t_b, mx_b, my_b, mz_b, lo=lo_b, hi=hi_b)
    t_rb2, mx_rb2, my_rb2, mz_rb2 = clip_time(t_rb, mx_rb, my_rb, mz_rb, lo=lo_b, hi=hi_b)

    # Sanitize time and magnetisation arrays for plotting
    t_a2s, _ = _sanitize_xy(t_a2, mx_a2)
    t_b2s, _ = _sanitize_xy(t_b2, mx_b2)
    t_a_ns = t_a2s * 1e9
    t_b_ns = t_b2s * 1e9

    # Residuals on sanitized MuMax grids
    res_a_mx = residual_on_grid(t_a2s, mx_a2, t_ra2, mx_ra2, sign="rust-minus-mumax")
    res_a_my = residual_on_grid(t_a2s, my_a2, t_ra2, my_ra2, sign="rust-minus-mumax")
    res_a_mz = residual_on_grid(t_a2s, mz_a2, t_ra2, mz_ra2, sign="rust-minus-mumax")

    res_b_mx = residual_on_grid(t_b2s, mx_b2, t_rb2, mx_rb2, sign="rust-minus-mumax")
    res_b_my = residual_on_grid(t_b2s, my_b2, t_rb2, my_rb2, sign="rust-minus-mumax")
    res_b_mz = residual_on_grid(t_b2s, mz_b2, t_rb2, mz_rb2, sign="rust-minus-mumax")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1, ax2 = axes

    # SP4a residuals
    ax1.plot(t_a_ns, res_a_mx, label="Δmx", linewidth=RUST_LW)
    ax1.plot(t_a_ns, res_a_my, label="Δmy", linewidth=RUST_LW)
    ax1.plot(t_a_ns, res_a_mz, label="Δmz", linewidth=RUST_LW)
    ax1.axhline(0.0, linewidth=0.8)
    ax1.set_title("SP4a residuals (Rust − MuMax)")
    ax1.set_ylabel("Δm")
    ax1.legend(fontsize=7, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3)

    # SP4b residuals
    ax2.plot(t_b_ns, res_b_mx, label="Δmx", linewidth=RUST_LW)
    ax2.plot(t_b_ns, res_b_my, label="Δmy", linewidth=RUST_LW)
    ax2.plot(t_b_ns, res_b_mz, label="Δmz", linewidth=RUST_LW)
    ax2.axhline(0.0, linewidth=0.8)
    ax2.set_title("SP4b residuals (Rust − MuMax)")
    ax2.set_xlabel("t (ns)")
    ax2.set_ylabel("Δm")
    ax2.legend(fontsize=7, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _save_sp4a_residuals(
    out_path: Path,
    t_m: Array, mx_m: Array, my_m: Array, mz_m: Array,
    t_r: Array, mx_r: Array, my_r: Array, mz_r: Array,
    *,
    pub: bool = False,
    dpi: int = 200,
) -> None:
    """Save a single-panel SP4a residual plot."""
    lo, hi = overlap_window(t_m, t_r)
    t_m2, mx_m2, my_m2, mz_m2 = clip_time(t_m, mx_m, my_m, mz_m, lo=lo, hi=hi)
    t_r2, mx_r2, my_r2, mz_r2 = clip_time(t_r, mx_r, my_r, mz_r, lo=lo, hi=hi)

    t_m2s, _ = _sanitize_xy(t_m2, mx_m2)
    t_ns = t_m2s * 1e9

    res_mx = residual_on_grid(t_m2s, mx_m2, t_r2, mx_r2, sign="rust-minus-mumax")
    res_my = residual_on_grid(t_m2s, my_m2, t_r2, my_r2, sign="rust-minus-mumax")
    res_mz = residual_on_grid(t_m2s, mz_m2, t_r2, mz_r2, sign="rust-minus-mumax")

    if pub:
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.0))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    lw = 1.2 if pub else RUST_LW
    ax.plot(t_ns, res_mx, label=r"$\Delta m_x$", linewidth=lw, color="#d62728")
    ax.plot(t_ns, res_my, label=r"$\Delta m_y$", linewidth=lw, color="#1f77b4")
    ax.plot(t_ns, res_mz, label=r"$\Delta m_z$", linewidth=lw, color="#7f7f7f")
    ax.axhline(0.0, linewidth=0.5, color="0.6")

    ax.set_xlabel(r"Time (ns)")
    ax.set_ylabel(r"$\Delta m$ (Rust $-$ MuMax3)")
    ax.legend(fontsize=8 if pub else 7, frameon=True, framealpha=0.95, edgecolor="0.8",
              borderpad=0.3, labelspacing=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot Standard Problem 4 (a,b) MuMax outputs, optionally overlay Rust outputs."
    )
    ap.add_argument(
        "--mumax-root",
        type=Path,
        default=Path("mumax_outputs/st_problems/sp4"),
        help="Folder containing sp4a_out/table.txt and sp4b_out/table.txt",
    )
    ap.add_argument(
        "--rust-root",
        type=Path,
        default=None,
        help="Optional folder containing sp4a_rust/table.csv and sp4b_rust/table.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output PNG path (if omitted, shows interactive window).",
    )
    ap.add_argument(
        "--mark-mx-zero",
        action="store_true",
        help="Mark the first <mx>=0 crossing time for MuMax and Rust (if present).",
    )
    ap.add_argument(
        "--sp4a-only",
        action="store_true",
        help="Plot only SP4a (single panel) instead of both SP4a and SP4b.",
    )
    ap.add_argument(
        "--pub",
        action="store_true",
        help="Publication-quality styling: larger fonts, LaTeX labels, 'This work' labelling, PDF-ready.",
    )

    # metrics
    ap.add_argument(
        "--metrics",
        action="store_true",
        help="Print comparison metrics (RMSE, max|Δm|, p95|Δm|, t@max) for mx/my/mz (requires Rust outputs).",
    )
    ap.add_argument(
        "--metrics-tmin",
        type=float,
        default=None,
        help="Optional metrics window start time in seconds (e.g. 3e-9).",
    )
    ap.add_argument(
        "--metrics-tmax",
        type=float,
        default=None,
        help="Optional metrics window end time in seconds (e.g. 5e-9).",
    )
    ap.add_argument(
        "--metrics-interp",
        choices=["rust", "mumax"],
        default="rust",
        help="Interpolate the other dataset onto this time grid for metrics (default: rust).",
    )

    # dynamics diagnostics (frequency + phase drift)
    ap.add_argument(
        "--dyn-tmin",
        type=float,
        default=None,
        help=(
            "Optional dynamics window start time in seconds. If omitted, uses the second half of the overlap window."
        ),
    )
    ap.add_argument(
        "--dyn-tmax",
        type=float,
        default=None,
        help="Optional dynamics window end time in seconds.",
    )
    ap.add_argument(
        "--dyn-component",
        choices=["auto", "mx", "my", "mz"],
        default="auto",
        help="Component used for frequency/phase drift diagnostics (default: auto = pick largest-variance component).",
    )
    ap.add_argument(
        "--dyn-rel-bw",
        type=float,
        default=0.20,
        help="Relative half-bandwidth around dominant frequency for phase drift estimation (default: 0.20).",
    )

    args = ap.parse_args()

    # --- MuMax paths ---
    mumax_a = args.mumax_root / "sp4a_out" / "table.txt"
    t_a, mx_a, my_a, mz_a = load_mumax_table(mumax_a)
    t_a_ns = t_a * 1e9

    # SP4b data: bundled as optional tuple so Pylance can narrow with one check
    mumax_b: Optional[Tuple[Array, Array, Array, Array]] = None
    t_b_ns: Optional[Array] = None
    if not args.sp4a_only:
        mumax_b_path = args.mumax_root / "sp4b_out" / "table.txt"
        mumax_b = load_mumax_table(mumax_b_path)
        t_b_ns = mumax_b[0] * 1e9

    # --- Optional Rust ---
    rust_a = rust_b = None
    if args.rust_root is not None:
        ra = args.rust_root / "sp4a_rust" / "table.csv"
        if ra.exists():
            rust_a = load_rust_csv(ra)
        else:
            print(f"[warn] Rust SP4a table not found at: {ra}")

        if not args.sp4a_only:
            rb = args.rust_root / "sp4b_rust" / "table.csv"
            if rb.exists():
                rust_b = load_rust_csv(rb)
            else:
                print(f"[warn] Rust SP4b table not found at: {rb}")

    # --- metrics printout ---
    if args.metrics:
        if rust_a is None:
            print("[metrics] Rust SP4a outputs not available; metrics require --rust-root with sp4a_rust/table.csv")
        else:
            t_ra, mx_ra, my_ra, mz_ra = rust_a

            print_metrics_block(
                "SP4a",
                t_a, mx_a, my_a, mz_a,
                t_ra, mx_ra, my_ra, mz_ra,
                tmin=args.metrics_tmin,
                tmax=args.metrics_tmax,
                interp_to=args.metrics_interp,
            )

            # --- frequency + phase drift diagnostics ---
            print_dynamics_block(
                "SP4a",
                t_a, mx_a, my_a, mz_a,
                t_ra, mx_ra, my_ra, mz_ra,
                dyn_tmin=args.dyn_tmin,
                dyn_tmax=args.dyn_tmax,
                dyn_component=args.dyn_component,
                dyn_rel_bw=args.dyn_rel_bw,
            )

        if not args.sp4a_only and rust_b is not None and mumax_b is not None:
            t_b, mx_b, my_b, mz_b = mumax_b
            t_rb, mx_rb, my_rb, mz_rb = rust_b

            print_metrics_block(
                "SP4b",
                t_b, mx_b, my_b, mz_b,
                t_rb, mx_rb, my_rb, mz_rb,
                tmin=args.metrics_tmin,
                tmax=args.metrics_tmax,
                interp_to=args.metrics_interp,
            )

            print_dynamics_block(
                "SP4b",
                t_b, mx_b, my_b, mz_b,
                t_rb, mx_rb, my_rb, mz_rb,
                dyn_tmin=args.dyn_tmin,
                dyn_tmax=args.dyn_tmax,
                dyn_component=args.dyn_component,
                dyn_rel_bw=args.dyn_rel_bw,
            )

    # ======================================================================
    # Publication-quality style setup
    # ======================================================================
    if args.pub:
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 2.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        })
        # Publication labels
        _our_label = "Rust"
        _ref_label = "MuMax3"
        _mumax_ms = 3.0
        _rust_lw = 1.2
        _mumax_marker = "o"
        _legend_fs = 11
        _dpi = 300
    else:
        _our_label = "Rust"
        _ref_label = "MuMax"
        _mumax_ms = MUMAX_MS
        _rust_lw = RUST_LW
        _mumax_marker = "o"
        _legend_fs = 6
        _dpi = 200

    # ======================================================================
    # Colour definitions (greyscale-safe via line style distinction)
    # ======================================================================
    # MuMax3 uses markers only (no connecting line) — visually "dots"
    # Our solver uses solid lines — visually "lines"
    # Colour encodes component: red=mx, blue=my, grey=mz
    _cx = "#d62728"   # red
    _cy = "#1f77b4"   # blue
    _cz = "#7f7f7f"   # grey

    # ======================================================================
    # SP4a-only mode: single clean panel
    # ======================================================================
    if args.sp4a_only:
        if args.pub:
            # Full-width to align with the freeze-frame triptych below (2000×561 px @ 300 dpi)
            fig, ax = plt.subplots(1, 1, figsize=(6.667, 2.6))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6.667, 2.8))

        # MuMax3: hollow markers, no connecting line (no individual labels)
        ax.plot(t_a_ns, mx_a, marker=_mumax_marker, linestyle="None",
                color=_cx, markersize=_mumax_ms,
                markerfacecolor="none", markeredgewidth=0.8, zorder=1)
        ax.plot(t_a_ns, my_a, marker=_mumax_marker, linestyle="None",
                color=_cy, markersize=_mumax_ms,
                markerfacecolor="none", markeredgewidth=0.8, zorder=1)
        ax.plot(t_a_ns, mz_a, marker=_mumax_marker, linestyle="None",
                color=_cz, markersize=_mumax_ms,
                markerfacecolor="none", markeredgewidth=0.8, zorder=1)

        # Our solver: solid lines (no individual labels)
        if rust_a is not None:
            t_r, mx_r, my_r, mz_r = rust_a
            t_r_ns = t_r * 1e9
            ax.plot(t_r_ns, mx_r, linestyle="-", color=_cx, linewidth=_rust_lw, zorder=2)
            ax.plot(t_r_ns, my_r, linestyle="-", color=_cy, linewidth=_rust_lw, zorder=2)
            ax.plot(t_r_ns, mz_r, linestyle="-", color=_cz, linewidth=_rust_lw, zorder=2)

        ax.set_xlabel(r"Time (ns)")
        ax.set_ylabel(r"$\langle m_i \rangle$")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-1.05, 1.05)

        # Compact custom legend: single row ABOVE the axes so it can't overlap data
        from matplotlib.lines import Line2D
        _fs = 10 if args.pub else 8
        handles = [
            Line2D([], [], color="k", linestyle="-", linewidth=_rust_lw, label="Rust"),
            Line2D([], [], color="k", marker="o", linestyle="None", markersize=4,
                   markerfacecolor="none", markeredgewidth=0.8, label="MuMax3"),
            Line2D([], [], color=_cx, linestyle="-", linewidth=2.5, label=r"$m_x$"),
            Line2D([], [], color=_cy, linestyle="-", linewidth=2.5, label=r"$m_y$"),
            Line2D([], [], color=_cz, linestyle="-", linewidth=2.5, label=r"$m_z$"),
        ]
        ax.legend(
            handles=handles, ncol=5,
            loc="lower center", bbox_to_anchor=(0.5, 1.0),
            frameon=False,
            borderpad=0.1, labelspacing=0.2, handletextpad=0.4,
            columnspacing=1.0, handlelength=1.4, fontsize=_fs,
        )

        # Optional mx=0 crossing markers
        if args.mark_mx_zero:
            t0_ma = first_zero_crossing_time(t_a_ns, mx_a)
            if t0_ma is not None:
                ax.axvline(float(t0_ma), linestyle=":", linewidth=VLINE_LW, color="0.5")
            if rust_a is not None:
                t_r, mx_r, _, _ = rust_a
                t0_ra = first_zero_crossing_time(t_r * 1e9, mx_r)
                if t0_ra is not None:
                    ax.axvline(float(t0_ra), linestyle="--", linewidth=VLINE_LW, color="0.5")

        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.86)

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.out, dpi=_dpi)
            print(f"Wrote {args.out}")

            # Also save residuals for SP4a only
            if rust_a is not None:
                t_ra, mx_ra, my_ra, mz_ra = rust_a
                residual_path = args.out.parent / f"{args.out.stem}_residuals{args.out.suffix}"
                _save_sp4a_residuals(
                    residual_path, t_a, mx_a, my_a, mz_a,
                    t_ra, mx_ra, my_ra, mz_ra,
                    pub=args.pub, dpi=_dpi,
                )

            plt.close(fig)
        else:
            plt.show()
        return

    # ======================================================================
    # Original 2-panel mode (SP4a + SP4b)
    # ======================================================================
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax_a, ax_b = axes

    # ---------- SP4a ----------
    plot_triplet(ax_a, t_a_ns, mx_a, my_a, mz_a, marker="o", linestyle="None", prefix=f"{_ref_label} ", ms=_mumax_ms, lw=_rust_lw)
    ax_a.set_ylabel(r"$\langle m_i \rangle$")
    ax_a.set_title("SP4a")

    if rust_a is not None:
        t_r, mx_r, my_r, mz_r = rust_a
        plot_triplet(ax_a, t_r * 1e9, mx_r, my_r, mz_r, marker=None, linestyle="-", prefix=f"{_our_label} ", ms=_mumax_ms, lw=_rust_lw)

    ax_a.legend(fontsize=_legend_fs, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    # ---------- SP4b ----------
    if mumax_b is not None and t_b_ns is not None:
        t_b, mx_b, my_b, mz_b = mumax_b
        plot_triplet(ax_b, t_b_ns, mx_b, my_b, mz_b, marker="o", linestyle="None", prefix=f"{_ref_label} ", ms=_mumax_ms, lw=_rust_lw)
        ax_b.set_xlabel("t (ns)")
        ax_b.set_ylabel(r"$\langle m_i \rangle$")
        ax_b.set_title("SP4b")

        if rust_b is not None:
            t_r, mx_r, my_r, mz_r = rust_b
            plot_triplet(ax_b, t_r * 1e9, mx_r, my_r, mz_r, marker=None, linestyle="-", prefix=f"{_our_label} ", ms=_mumax_ms, lw=_rust_lw)

        ax_b.legend(fontsize=_legend_fs, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    # ---------- Optional mx=0 markers ----------
    if args.mark_mx_zero:
        t0_ma = first_zero_crossing_time(t_a_ns, mx_a)
        if t0_ma is not None:
            ax_a.axvline(float(t0_ma), linestyle=":", linewidth=VLINE_LW)

        if mumax_b is not None and t_b_ns is not None:
            t_b, mx_b, my_b, mz_b = mumax_b
            t0_mb = first_zero_crossing_time(t_b_ns, mx_b)
            if t0_mb is not None:
                ax_b.axvline(float(t0_mb), linestyle=":", linewidth=VLINE_LW)

        if rust_a is not None:
            t_r, mx_r, _, _ = rust_a
            t0_ra = first_zero_crossing_time(t_r * 1e9, mx_r)
            if t0_ra is not None:
                ax_a.axvline(float(t0_ra), linestyle="--", linewidth=VLINE_LW)

        if rust_b is not None:
            t_r, mx_r, _, _ = rust_b
            t0_rb = first_zero_crossing_time(t_r * 1e9, mx_r)
            if t0_rb is not None:
                ax_b.axvline(float(t0_rb), linestyle="--", linewidth=VLINE_LW)

    plt.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=_dpi)
        print(f"Wrote {args.out}")

        # --- Residuals plot saved next to overlay ---
        if rust_a is not None and rust_b is not None and mumax_b is not None:
            t_ra, mx_ra, my_ra, mz_ra = rust_a
            t_rb, mx_rb, my_rb, mz_rb = rust_b
            t_b, mx_b, my_b, mz_b = mumax_b

            residual_path = args.out.parent / f"{args.out.stem}_residuals{args.out.suffix}"
            save_residuals_figure(
                residual_path,
                t_a, mx_a, my_a, mz_a,
                t_ra, mx_ra, my_ra, mz_ra,
                t_b, mx_b, my_b, mz_b,
                t_rb, mx_rb, my_rb, mz_rb,
                dpi=_dpi,
            )
        else:
            print("[residuals] Rust outputs not available; residual plot requires --rust-root.")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()