#!/usr/bin/env python3
# scripts/overlay_macrospin.py
#
# Overlay Rust vs MuMax table outputs and (optionally) compute FFT peak frequency.
#
# Features:
# - Robust MuMax header parsing (handles duplicate names by suffixing _2, _3, ...)
# - Optional clipping to overlap window or user-specified [tmin, tmax]
# - Optional quantitative metrics (RMSE and max|error|) on the overlap window
# - Optional FFT peak with Hann window + optional band-limit [fmin, fmax]
# - Derived columns:
#     --col m_parallel  => sqrt(mx^2 + my^2)
#     --col phi         => atan2(my, mx)  (optionally unwrap)
#
# Examples:
#   python3 scripts/overlay_macrospin.py \
#     out/uniform_film/rust_table_uniform_film.csv \
#     mumax_outputs/uniform_film_field/table.txt \
#     --col my --clip_overlap --metrics --do_fft --fft_tmin 1e-9 --fft_tmax 5e-9 --fmin 1e9 \
#     --out out/uniform_film/overlay_my.png
#
#   python3 scripts/overlay_macrospin.py \
#     out/uniform_film/rust_table_uniform_film.csv \
#     mumax_outputs/uniform_film_field/table.txt \
#     --col m_parallel --clip_overlap --metrics \
#     --out out/uniform_film/overlay_m_parallel.png

import argparse
import csv
import os
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

try:
    import numpy as np
except ImportError:
    np = None

MAG_COLS = {"mx", "my", "mz"}
DERIVED_COLS = {"m_parallel", "phi"}


# ----------------------------
# Parsing helpers
# ----------------------------

def read_rust_csv(path: str) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if v is None or v == "":
                    continue
                cols.setdefault(k, []).append(float(v))
    return cols


def _clean_header_token(tok: str) -> str:
    tok = tok.strip()
    tok = tok.strip("()")
    tok = tok.replace("/", "_")
    return tok


def make_unique(names: List[str]) -> List[str]:
    """Make duplicate names unique via suffix _2, _3, ..."""
    counts: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        if n in counts:
            counts[n] += 1
            out.append(f"{n}_{counts[n]}")
        else:
            counts[n] = 1
            out.append(n)
    return out


def read_mumax_table(path: str) -> Tuple[List[str], List[List[float]]]:
    header: List[str] = []
    data: List[List[float]] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # Header line containing t and mx/my/mz
                if "t" in line and ("mx" in line or "my" in line or "mz" in line):
                    toks = re.split(r"\s+", line.lstrip("#").strip())
                    cleaned: List[str] = []
                    for tok in toks:
                        # Drop unit tokens like "(s)" "(J)" "(T)" etc.
                        if tok.startswith("(") and tok.endswith(")"):
                            continue
                        cleaned.append(_clean_header_token(tok))
                    header = make_unique(cleaned)
                continue

            parts = re.split(r"\s+", line)
            try:
                row = [float(x) for x in parts]
            except ValueError:
                continue
            data.append(row)

    if not header:
        raise RuntimeError("Could not find MuMax header line (expected '# ... t mx my mz ...').")

    return header, data


def columns_from_table(header: List[str], data: List[List[float]]) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {h: [] for h in header}
    for row in data:
        for i, h in enumerate(header):
            if i < len(row):
                cols[h].append(row[i])
    return cols


# ----------------------------
# Analysis helpers
# ----------------------------

def mag_base(col: str) -> Optional[str]:
    """Return mx/my/mz if col is magnetization (supports my_2 etc)."""
    if col in MAG_COLS:
        return col
    base = col.split("_", 1)[0]
    return base if base in MAG_COLS else None


def clip_series(t: List[float], y: List[float], tmin: float, tmax: float) -> Tuple[List[float], List[float]]:
    out_t: List[float] = []
    out_y: List[float] = []
    for ti, yi in zip(t, y):
        if ti < tmin:
            continue
        if ti > tmax:
            break
        out_t.append(ti)
        out_y.append(yi)
    return out_t, out_y


def overlap_window(
    t_r: List[float],
    t_m: List[float],
    tmin: Optional[float],
    tmax: Optional[float],
) -> Tuple[float, float]:
    lo = max(t_r[0], t_m[0])
    hi = min(t_r[-1], t_m[-1])
    if tmin is not None:
        lo = max(lo, tmin)
    if tmax is not None:
        hi = min(hi, tmax)
    if hi <= lo:
        raise RuntimeError(f"No overlap window: lo={lo}, hi={hi}")
    return lo, hi


def interp_to(t_src: List[float], y_src: List[float], t_tgt: List[float]) -> List[float]:
    if np is None:
        raise RuntimeError("numpy required for interpolation/metrics")
    return list(np.interp(np.asarray(t_tgt), np.asarray(t_src), np.asarray(y_src)))


def rmse(a: List[float], b: List[float]) -> float:
    if np is None:
        raise RuntimeError("numpy required for metrics")
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    return float(np.sqrt(np.mean((a_arr - b_arr) ** 2)))


def max_abs_err(a: List[float], b: List[float]) -> float:
    if np is None:
        raise RuntimeError("numpy required for metrics")
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    return float(np.max(np.abs(a_arr - b_arr)))


def fft_peak_freq(
    t: List[float],
    y: List[float],
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> float:
    if np is None:
        raise RuntimeError("numpy not available; install numpy for FFT peak.")

    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(t_arr) < 8:
        raise RuntimeError("Not enough points for FFT")

    dt = float(np.median(np.diff(t_arr)))
    y_detrend = y_arr - float(np.mean(y_arr))

    # Hann window reduces spectral leakage
    window = np.hanning(len(y_detrend))
    y_win = y_detrend * window

    Y = np.fft.rfft(y_win)
    freqs = np.fft.rfftfreq(len(y_win), d=dt)
    mag = np.abs(Y)

    # Remove DC
    if len(mag) > 1:
        mag[0] = 0.0

    # Optional band-limiting
    if fmin is not None:
        mag[freqs < fmin] = 0.0
    if fmax is not None:
        mag[freqs > fmax] = 0.0

    k = int(np.argmax(mag))
    return float(freqs[k])


def compute_series(cols: Dict[str, List[float]], col: str, mag_suffix: str = "") -> List[float]:
    """
    Returns the data series for `col`, supporting derived columns:
      - m_parallel: sqrt(mx^2 + my^2)
      - phi: atan2(my, mx)
    For MuMax, you can pass mag_suffix="_2" to use mx_2/my_2/mz_2.
    """
    if col in DERIVED_COLS:
        mx_key = "mx" + mag_suffix
        my_key = "my" + mag_suffix

        if mx_key not in cols or my_key not in cols:
            raise RuntimeError(f"Missing required columns for {col}: need '{mx_key}' and '{my_key}'")

        mx = cols[mx_key]
        my = cols[my_key]

        if np is None:
            # small fallback (no numpy): do pure python math
            import math
            if col == "m_parallel":
                return [math.sqrt(a*a + b*b) for a, b in zip(mx, my)]
            if col == "phi":
                return [math.atan2(b, a) for a, b in zip(mx, my)]
        else:
            mx_a = np.asarray(mx)
            my_a = np.asarray(my)
            if col == "m_parallel":
                return list(np.sqrt(mx_a*mx_a + my_a*my_a))
            if col == "phi":
                return list(np.arctan2(my_a, mx_a))

        raise RuntimeError(f"Unhandled derived col: {col}")

    # Non-derived: direct column lookup
    key = col + mag_suffix if (mag_suffix and col in MAG_COLS) else col
    if key not in cols:
        raise RuntimeError(f"Missing column '{key}'. Available: {list(cols.keys())}")
    return cols[key]


def unwrap_phi(phi: List[float]) -> List[float]:
    if np is None:
        return phi
    return list(np.unwrap(np.asarray(phi)))


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rust_csv", help="Path to Rust CSV")
    ap.add_argument("mumax_table", help="Path to MuMax table.txt")

    ap.add_argument("--col", default="my",
                    help="Rust column to plot: mx,my,mz,E_total,... or derived: m_parallel, phi")
    ap.add_argument("--mumax_col", default=None,
                    help="Optional MuMax column name if different from --col (e.g. compare Rust my to MuMax my_2). "
                         "For derived cols, use --mumax_mag_suffix instead.")
    ap.add_argument("--mumax_mag_suffix", default="",
                    help="For MuMax magnetisation columns, use '' (default) or '_2' etc (only affects mx/my/mz used "
                         "by derived columns).")
    ap.add_argument("--unwrap_phi", action="store_true", help="If --col phi, unwrap phase for nicer plotting.")

    ap.add_argument("--out", default=None, help="Output plot filename")

    # Windowing
    ap.add_argument("--tmin", type=float, default=None, help="Min time (s) to include")
    ap.add_argument("--tmax", type=float, default=None, help="Max time (s) to include")
    ap.add_argument("--clip_overlap", action="store_true", help="Clip both datasets to overlapping time range")

    # Metrics
    ap.add_argument("--metrics", action="store_true", help="Print RMSE and max|err| (requires numpy)")
    ap.add_argument("--interp", choices=["rust", "mumax"], default="rust",
                    help="Interpolate the other dataset onto this grid for metrics (default: rust)")

    # FFT
    ap.add_argument("--do_fft", action="store_true", help="Compute FFT peak (mx/my/mz only; requires numpy)")
    ap.add_argument("--fft_tmin", type=float, default=None, help="FFT window start time (s)")
    ap.add_argument("--fft_tmax", type=float, default=None, help="FFT window end time (s)")
    ap.add_argument("--fmin", type=float, default=None, help="Ignore FFT freqs below fmin (Hz)")
    ap.add_argument("--fmax", type=float, default=None, help="Ignore FFT freqs above fmax (Hz)")

    args = ap.parse_args()

    rust_col = args.col
    mumax_col = args.mumax_col if args.mumax_col is not None else rust_col
    mag_suffix = args.mumax_mag_suffix

    rust = read_rust_csv(args.rust_csv)
    hdr, rows = read_mumax_table(args.mumax_table)
    mumax = columns_from_table(hdr, rows)

    # Time columns required
    if "t" not in rust:
        raise RuntimeError("Rust CSV is missing column 't'")
    if "t" not in mumax:
        print("MuMax columns found:", list(mumax.keys()))
        raise RuntimeError("MuMax table is missing column 't'")

    t_r = rust["t"]
    t_m = mumax["t"]

    # Build series
    # - Rust: derived cols use mx,my directly; no suffix support needed
    y_r = compute_series(rust, rust_col, mag_suffix="")

    # - MuMax: if caller explicitly set mumax_col, use it (only for non-derived).
    #   For derived cols, use mumax_mag_suffix to pick mx_2/my_2 etc.
    if rust_col in DERIVED_COLS:
        y_m = compute_series(mumax, rust_col, mag_suffix=mag_suffix)
    else:
        # non-derived: use mumax_col directly
        y_m = compute_series(mumax, mumax_col, mag_suffix="")

    if len(t_r) != len(y_r):
        raise RuntimeError(f"Rust length mismatch: len(t)={len(t_r)} vs len({rust_col})={len(y_r)}")
    if len(t_m) != len(y_m):
        raise RuntimeError(f"MuMax length mismatch: len(t)={len(t_m)} vs len({mumax_col})={len(y_m)}")

    # Optional unwrap
    if rust_col == "phi" and args.unwrap_phi:
        y_r = unwrap_phi(y_r)
        y_m = unwrap_phi(y_m)

    # Clip to overlap/time window
    if args.clip_overlap or args.tmin is not None or args.tmax is not None:
        lo, hi = overlap_window(t_r, t_m, args.tmin, args.tmax)
        t_r, y_r = clip_series(t_r, y_r, lo, hi)
        t_m, y_m = clip_series(t_m, y_m, lo, hi)
        print(f"[clip] using window t in [{lo:.3e}, {hi:.3e}] s")

    # Metrics
    if args.metrics:
        if np is None:
            print("numpy not available; skipping metrics.")
        else:
            if args.interp == "rust":
                y_m_i = interp_to(t_m, y_m, t_r)
                print(f"[metrics] RMSE({rust_col} vs {mumax_col}) = {rmse(y_r, y_m_i):.6e}")
                print(f"[metrics] max|err|({rust_col} vs {mumax_col}) = {max_abs_err(y_r, y_m_i):.6e}")
            else:
                y_r_i = interp_to(t_r, y_r, t_m)
                print(f"[metrics] RMSE({rust_col} vs {mumax_col}) = {rmse(y_r_i, y_m):.6e}")
                print(f"[metrics] max|err|({rust_col} vs {mumax_col}) = {max_abs_err(y_r_i, y_m):.6e}")

    # Output filename
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.rust_csv))[0]
        args.out = f"out/overlay_{base}_{rust_col}_vs_{mumax_col}.png"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Plot
    plt.figure()
    plt.plot(t_r, y_r, label=f"Rust {rust_col}(t)")
    label_m = mumax_col if rust_col not in DERIVED_COLS else f"{rust_col}{mag_suffix}"
    plt.plot(t_m, y_m, label=f"MuMax {label_m}(t)", linestyle="--")
    plt.xlabel("time (s)")

    if rust_col == "m_parallel":
        plt.ylabel("m_parallel (dimensionless)")
        plt.title("m_parallel(t): Rust vs MuMax")
    elif rust_col == "phi":
        plt.ylabel("phi (rad)")
        plt.title("phi(t): Rust vs MuMax")
    else:
        mb = mag_base(rust_col)
        if mb is not None:
            plt.ylabel(f"m_{mb[-1]} (dimensionless)")
            plt.title(f"{rust_col}(t): Rust vs MuMax")
        else:
            plt.ylabel(rust_col)
            plt.title(f"{rust_col} vs time: Rust vs MuMax")

    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")

    # FFT (only meaningful for magnetisation components)
    if args.do_fft:
        mb = mag_base(rust_col)
        if mb is None:
            print(f"Skipping FFT: --col {rust_col} is not magnetisation (mx/my/mz).")
            return
        if np is None:
            print("numpy not available; skipping FFT.")
            return

        # Use shared FFT window intersection
        lo_fft, hi_fft = overlap_window(t_r, t_m, args.fft_tmin, args.fft_tmax)
        t_r_fft, y_r_fft = clip_series(t_r, y_r, lo_fft, hi_fft)
        t_m_fft, y_m_fft = clip_series(t_m, y_m, lo_fft, hi_fft)

        print(f"[fft] using window t in [{lo_fft:.3e}, {hi_fft:.3e}] s")
        f_r = fft_peak_freq(t_r_fft, y_r_fft, fmin=args.fmin, fmax=args.fmax)
        f_m = fft_peak_freq(t_m_fft, y_m_fft, fmin=args.fmin, fmax=args.fmax)
        print(f"FFT peak frequency (Rust):  {f_r:.6e} Hz")
        print(f"FFT peak frequency (MuMax): {f_m:.6e} Hz")


if __name__ == "__main__":
    main()