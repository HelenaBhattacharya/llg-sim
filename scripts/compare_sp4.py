# scripts/compare_sp4.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


Array = np.ndarray
MUMAX_MS = 2.0   # marker size for MuMax squares
RUST_MS  = 2.0   # marker size for Rust circles
VLINE_LW = 0.6   # vertical line width (optional)

def load_mumax_table(table_path: Path) -> Tuple[Array, Array, Array, Array]:
    """
    Load MuMax table.txt assuming first 4 numeric cols are: t, mx, my, mz.
    This matches your original sp4_out.py behaviour.
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

    names = raw.dtype.names or ()  # fixes Pylance: dtype.names can be None

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


def plot_triplet(ax, t_ns: Array, mx: Array, my: Array, mz: Array, fmt: str, prefix: str, ms: float):
    ax.plot(t_ns, mx, fmt, label=f"{prefix}mx", markersize=ms)
    ax.plot(t_ns, my, fmt, label=f"{prefix}my", markersize=ms)
    ax.plot(t_ns, mz, fmt, label=f"{prefix}mz", markersize=ms)


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
    args = ap.parse_args()

    # --- MuMax paths (match your existing naming convention) ---
    mumax_a = args.mumax_root / "sp4a_out" / "table.txt"
    mumax_b = args.mumax_root / "sp4b_out" / "table.txt"

    t_a, mx_a, my_a, mz_a = load_mumax_table(mumax_a)
    t_b, mx_b, my_b, mz_b = load_mumax_table(mumax_b)

    # Convert to ns (match original)
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

    # --- Plot: SAME layout as your original sp4_out.py ---
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax_a, ax_b = axes

    # SP4a – top panel (MuMax markers "s" as original)
    plot_triplet(ax_a, t_a_ns, mx_a, my_a, mz_a, "s", prefix="", ms=MUMAX_MS)
    ax_a.set_ylabel("<m>")
    ax_a.set_title("SP4a")
    ax_a.legend(fontsize=6, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    # Overlay Rust if present (use a different marker for clarity)
    if rust_a is not None:
        t_r, mx_r, my_r, mz_r = rust_a
        plot_triplet(ax_a, t_r * 1e9, mx_r, my_r, mz_r, "o", prefix="Rust ", ms=RUST_MS)
        ax_a.legend(fontsize=6, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    # SP4b – bottom panel
    plot_triplet(ax_b, t_b_ns, mx_b, my_b, mz_b, "s", prefix="", ms=MUMAX_MS)
    ax_b.set_xlabel("t (ns)")
    ax_b.set_ylabel("<m>")
    ax_b.set_title("SP4b")
    ax_b.legend(fontsize=6, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    if rust_b is not None:
        t_r, mx_r, my_r, mz_r = rust_b
        plot_triplet(ax_b, t_r * 1e9, mx_r, my_r, mz_r, "o", prefix="Rust ", ms=RUST_MS)
        ax_b.legend(fontsize=6, frameon=True, framealpha=0.9, borderpad=0.3, labelspacing=0.3, handletextpad=0.4)

    if args.mark_mx_zero:
        # MuMax crossings
        t0_ma = first_zero_crossing_time(t_a_ns, mx_a)
        t0_mb = first_zero_crossing_time(t_b_ns, mx_b)
        if t0_ma is not None:
            ax_a.axvline(float(t0_ma), linestyle=":", linewidth=1)
        if t0_mb is not None:
            ax_b.axvline(float(t0_mb), linestyle=":", linewidth=1)

        # Rust crossings
        if rust_a is not None:
            t_r, mx_r, _, _ = rust_a
            t0_ra = first_zero_crossing_time(t_r * 1e9, mx_r)
            if t0_ra is not None:
                ax_a.axvline(float(t0_ra), linestyle="--", linewidth=1)
        if rust_b is not None:
            t_r, mx_r, _, _ = rust_b
            t0_rb = first_zero_crossing_time(t_r * 1e9, mx_r)
            if t0_rb is not None:
                ax_b.axvline(float(t0_rb), linestyle="--", linewidth=1)

    plt.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=200)
        print(f"Wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()