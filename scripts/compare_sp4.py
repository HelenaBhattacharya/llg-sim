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

    # metrics
    ap.add_argument(
        "--metrics",
        action="store_true",
        help="Print comparison metrics (RMSE, max|Δm|, p95|Δm|, t@max) for mx/my/mz (requires Rust outputs).",
    )
    ap.add_argument(
        "--metrics-tmin",
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

    args = ap.parse_args()

    # --- MuMax paths ---
    mumax_a = args.mumax_root / "sp4a_out" / "table.txt"
    mumax_b = args.mumax_root / "sp4b_out" / "table.txt"

    t_a, mx_a, my_a, mz_a = load_mumax_table(mumax_a)
    t_b, mx_b, my_b, mz_b = load_mumax_table(mumax_b)

    # Convert to ns
    t_a_ns = t_a * 1e9
    t_b_ns = t_b * 1e9

    # --- Optional Rust ---
    rust_a = rust_b = None
    if args.rust_root is not None:
        ra = args.rust_root / "sp4a_rust" / "table.csv"
        rb = args.rust_root / "sp4b_rust" / "table.csv"
        if ra.exists() and rb.exists():
            rust_a = load_rust_csv(ra)
            rust_b = load_rust_csv(rb)
        else:
            print(f"[warn] Rust tables not found at:\n  {ra}\n  {rb}\nContinuing with MuMax only.")

    # --- metrics printout ---
    if args.metrics:
        if rust_a is None or rust_b is None:
            print("[metrics] Rust outputs not available; metrics require --rust-root with sp4a_rust/table.csv and sp4b_rust/table.csv")
        else:
            t_ra, mx_ra, my_ra, mz_ra = rust_a
            t_rb, mx_rb, my_rb, mz_rb = rust_b

            print_metrics_block(
                "SP4a",
                t_a, mx_a, my_a, mz_a,
                t_ra, mx_ra, my_ra, mz_ra,
                tmin=args.metrics_tmin,
                tmax=args.metrics_tmax,
                interp_to=args.metrics_interp,
            )
            print_metrics_block(
                "SP4b",
                t_b, mx_b, my_b, mz_b,
                t_rb, mx_rb, my_rb, mz_rb,
                tmin=args.metrics_tmin,
                tmax=args.metrics_tmax,
                interp_to=args.metrics_interp,
            )

    # --- Plot layout (UNCHANGED) ---
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax_a, ax_b = axes

    # ---------- SP4a ----------
    # MuMax: circles, points only
    plot_triplet(ax_a, t_a_ns, mx_a, my_a, mz_a, marker="o", linestyle="None", prefix="MuMax ", ms=MUMAX_MS, lw=RUST_LW)
    ax_a.set_ylabel("<m>")
    ax_a.set_title("SP4a")

    # Rust: solid line, no markers
    if rust_a is not None:
        t_r, mx_r, my_r, mz_r = rust_a
        plot_triplet(ax_a, t_r * 1e9, mx_r, my_r, mz_r, marker=None, linestyle="-", prefix="Rust ", ms=MUMAX_MS, lw=RUST_LW)

    ax_a.legend(fontsize=6, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    # ---------- SP4b ----------
    plot_triplet(ax_b, t_b_ns, mx_b, my_b, mz_b, marker="o", linestyle="None", prefix="MuMax ", ms=MUMAX_MS, lw=RUST_LW)
    ax_b.set_xlabel("t (ns)")
    ax_b.set_ylabel("<m>")
    ax_b.set_title("SP4b")

    if rust_b is not None:
        t_r, mx_r, my_r, mz_r = rust_b
        plot_triplet(ax_b, t_r * 1e9, mx_r, my_r, mz_r, marker=None, linestyle="-", prefix="Rust ", ms=MUMAX_MS, lw=RUST_LW)

    ax_b.legend(fontsize=6, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    # ---------- Optional mx=0 markers ----------
    if args.mark_mx_zero:
        # MuMax crossings
        t0_ma = first_zero_crossing_time(t_a_ns, mx_a)
        t0_mb = first_zero_crossing_time(t_b_ns, mx_b)
        if t0_ma is not None:
            ax_a.axvline(float(t0_ma), linestyle=":", linewidth=VLINE_LW)
        if t0_mb is not None:
            ax_b.axvline(float(t0_mb), linestyle=":", linewidth=VLINE_LW)

        # Rust crossings
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
        plt.savefig(args.out, dpi=200)
        print(f"Wrote {args.out}")

        # --- NEW: residuals plot saved next to overlay ---
        if rust_a is not None and rust_b is not None:
            t_ra, mx_ra, my_ra, mz_ra = rust_a
            t_rb, mx_rb, my_rb, mz_rb = rust_b

            residual_path = args.out.parent / f"{args.out.stem}_residuals{args.out.suffix}"
            save_residuals_figure(
                residual_path,
                # SP4a
                t_a, mx_a, my_a, mz_a,
                t_ra, mx_ra, my_ra, mz_ra,
                # SP4b
                t_b, mx_b, my_b, mz_b,
                t_rb, mx_rb, my_rb, mz_rb,
                dpi=200,
            )
        else:
            print("[residuals] Rust outputs not available; residual plot requires --rust-root.")
        plt.close(fig)
    else:
        # Interactive mode unchanged: only show the overlay window
        plt.show()


if __name__ == "__main__":
    main()