#!/usr/bin/env python3
"""
compare_fmr.py

Overlay MuMax vs Rust for the FMR benchmark:
  (1) <my>(t) time series (aligned so both start at t=0 for the dynamic stage)
  (2) Psa vs f (GHz) where Psa = log10(|FFT|^2), normalized by peak power

Usage examples:

  python3 scripts/compare_fmr.py \
    --mumax mumax_outputs/st_problems/fmr/fmr_out/table.txt \
    --rust  runs/st_problems/fmr/fmr_rust/table.csv \
    --out   runs/st_problems/fmr/compare_fmr.png

Notes:
- MuMax table.txt usually has columns: t, mx, my, mz (plus others sometimes).
  Default assumes my is column index 2 (0-based).
- Rust CSV default assumes columns include 't_s' and 'my' (or 't' and 'my').
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_mumax_table(path: Path, my_col: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Load MuMax table.txt. Default: my is col 2 (0-based) assuming [t, mx, my, mz].
    """
    # Try pandas first (handles whitespace, comments)
    try:
        df = pd.read_csv(path, comment="#", delim_whitespace=True, header=None)
        arr = df.to_numpy(dtype=float)
    except Exception:
        arr = np.loadtxt(path, comments="#")

    if arr.ndim != 2 or arr.shape[1] <= my_col:
        raise ValueError(f"Unexpected MuMax table shape {arr.shape} for {path}")

    t = arr[:, 0].astype(float)
    my = arr[:, my_col].astype(float)
    return t, my


def load_rust_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Rust table.csv. Expected columns: t_s,my (or t,my).
    """
    df = pd.read_csv(path)
    if "t_s" in df.columns:
        t = df["t_s"].to_numpy(dtype=float)
    elif "t" in df.columns:
        t = df["t"].to_numpy(dtype=float)
    else:
        # fallback: first column
        t = df.iloc[:, 0].to_numpy(dtype=float)

    if "my" in df.columns:
        my = df["my"].to_numpy(dtype=float)
    else:
        # fallback: try third column (t,mx,my,mz)
        if df.shape[1] < 3:
            raise ValueError(f"Rust CSV does not contain 'my' and has too few columns: {df.columns}")
        my = df.iloc[:, 2].to_numpy(dtype=float)

    return t, my


def align_time_to_zero(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Shift time so it starts at 0 (dynamic stage alignment).
    Also removes any duplicate/unsorted time issues.
    """
    # sort by time just in case
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]

    # remove duplicate time points (keep first)
    _, unique_idx = np.unique(t, return_index=True)
    unique_idx.sort()
    t = t[unique_idx]
    y = y[unique_idx]

    t0 = t[0]
    return t - t0, y


def compute_psa(
    t: np.ndarray,
    y: np.ndarray,
    apply_hann: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return (f_GHz, psa, f_peak_GHz).

    psa = log10(P_norm + eps), where P_norm = |FFT|^2 / max(|FFT|^2)
    """
    # dt from median diff
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("Non-positive dt detected.")

    # remove DC
    y0 = y - float(np.mean(y))

    # window
    if apply_hann:
        w = np.hanning(len(y0))
        y0 = y0 * w

    Y = np.fft.rfft(y0)
    f = np.fft.rfftfreq(len(y0), d=dt)  # Hz
    P = np.abs(Y) ** 2

    # normalize
    Pmax = float(np.max(P)) if float(np.max(P)) > 0 else 1.0
    Pn = P / Pmax

    psa = np.log10(Pn + 1e-30)
    f_ghz = f / 1e9

    f_peak = float(f_ghz[np.argmax(P)])
    return f_ghz, psa, f_peak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mumax", type=str, required=True, help="MuMax table.txt path")
    ap.add_argument("--rust", type=str, required=True, help="Rust table.csv path")
    ap.add_argument("--out", type=str, required=True, help="Output PNG path")
    ap.add_argument("--mumax-my-col", type=int, default=2, help="0-based column index for my in MuMax table (default 2)")
    ap.add_argument("--tmax-ns", type=float, default=None, help="Optional max time (ns) for time-series plot")
    ap.add_argument("--fmin-ghz", type=float, default=6.0, help="Min frequency (GHz) for PSA plot (default 6)")
    ap.add_argument("--fmax-ghz", type=float, default=12.0, help="Max frequency (GHz) for PSA plot (default 12)")
    ap.add_argument("--no-hann", action="store_true", help="Disable Hann window before FFT")
    args = ap.parse_args()

    mumax_path = Path(args.mumax)
    rust_path = Path(args.rust)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load & align
    t_m, my_m = load_mumax_table(mumax_path, my_col=args.mumax_my_col)
    t_r, my_r = load_rust_csv(rust_path)

    t_m, my_m = align_time_to_zero(t_m, my_m)
    t_r, my_r = align_time_to_zero(t_r, my_r)

    # Convert to ns for plotting
    t_m_ns = t_m * 1e9
    t_r_ns = t_r * 1e9

    # Optional time crop
    if args.tmax_ns is not None:
        tm_mask = t_m_ns <= args.tmax_ns
        tr_mask = t_r_ns <= args.tmax_ns
        t_m_ns, my_m = t_m_ns[tm_mask], my_m[tm_mask]
        t_r_ns, my_r = t_r_ns[tr_mask], my_r[tr_mask]
        # back to seconds for FFT calc later
        t_m = t_m_ns * 1e-9
        t_r = t_r_ns * 1e-9

    # FFT/PSA
    f_m, psa_m, peak_m = compute_psa(t_m, my_m, apply_hann=not args.no_hann)
    f_r, psa_r, peak_r = compute_psa(t_r, my_r, apply_hann=not args.no_hann)

    print(f"Peak frequency (MuMax): {peak_m:.3f} GHz")
    print(f"Peak frequency (Rust) : {peak_r:.3f} GHz")

    # Plot: 2 panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    # Panel 1: my(t)
    ax1.plot(t_m_ns, my_m, label="MuMax ⟨my⟩(t)")
    ax1.plot(t_r_ns, my_r, label="Rust ⟨my⟩(t)")
    ax1.set_xlabel("t (ns)")
    ax1.set_ylabel("my average")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: PSA
    ax2.plot(f_m, psa_m, label="MuMax Psa=log10(|FFT|²)")
    ax2.plot(f_r, psa_r, label="Rust  Psa=log10(|FFT|²)")
    ax2.set_xlabel("f (GHz)")
    ax2.set_ylabel("Psa (a.u.)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(args.fmin_ghz, args.fmax_ghz)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())