#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

Array = np.ndarray


def load_mumax_table(path: Path) -> Tuple[Array, Array, Array, Array]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Expected >=4 columns in {path}, got shape {data.shape}")
    d_lex = data[:, 0]
    mx = data[:, 1]
    my = data[:, 2]
    hc = data[:, 3]
    return d_lex, mx, my, hc


def load_rust_table(path: Path) -> Tuple[Array, Array, Array, Array]:
    raw = np.genfromtxt(path, delimiter=",", names=True)
    names = raw.dtype.names or ()
    required = ("d_lex", "mx_rem", "my_rem", "hc_over_ms")
    for k in required:
        if k not in names:
            raise ValueError(f"Missing column '{k}' in {path}. Found {names}")
    return raw["d_lex"], raw["mx_rem"], raw["my_rem"], raw["hc_over_ms"]


def sort_by_d(d, *cols):
    order = np.argsort(d)
    return (d[order],) + tuple(c[order] for c in cols)


def find_mumax_table_from_root(root: Path) -> Path:
    # Preferred path used in your MuMax outputs:
    candidate = root / "sp2_out" / "table.txt"
    if candidate.exists():
        return candidate

    # Fallback: search for table.txt inside root
    hits = list(root.rglob("table.txt"))
    if not hits:
        raise FileNotFoundError(f"No table.txt found under {root}")
    # Prefer one under sp2_out if present
    for h in hits:
        if "sp2_out" in str(h).replace("\\", "/"):
            return h
    return hits[0]


def find_rust_table_from_root(root: Path) -> Path:
    candidate = root / "table.csv"
    if candidate.exists():
        return candidate

    hits = list(root.rglob("table.csv"))
    if not hits:
        raise FileNotFoundError(f"No table.csv found under {root}")
    return hits[0]


def compute_metrics(d_m, mx_m, my_m, hc_m, d_r, mx_r, my_r, hc_r) -> str:
    # Compare on overlapping d values by nearest-match on integer d/lex
    dm_int = d_m.astype(int)
    dr_int = d_r.astype(int)

    common = np.intersect1d(dm_int, dr_int)
    if common.size == 0:
        return "No overlapping d/lex values found for metrics."

    def pick(arr_d_int, arr_y, dval):
        idx = np.where(arr_d_int == dval)[0]
        return arr_y[idx[0]]

    mx_err = []
    my_err = []
    hc_err = []

    for dval in common:
        mx_err.append(pick(dr_int, mx_r, dval) - pick(dm_int, mx_m, dval))
        my_err.append(pick(dr_int, my_r, dval) - pick(dm_int, my_m, dval))
        hc_err.append(pick(dr_int, hc_r, dval) - pick(dm_int, hc_m, dval))

    mx_err = np.array(mx_err)
    my_err = np.array(my_err)
    hc_err = np.array(hc_err)

    def rms(x):  # noqa: E741
        return float(np.sqrt(np.mean(x * x)))

    lines = [
        f"Metrics over {common.size} overlapping d/lex values (Rust - MuMax):",
        f"  mx:  RMS={rms(mx_err):.3e},  max|err|={float(np.max(np.abs(mx_err))):.3e}",
        f"  my:  RMS={rms(my_err):.3e},  max|err|={float(np.max(np.abs(my_err))):.3e}",
        f"  hc:  RMS={rms(hc_err):.3e},  max|err|={float(np.max(np.abs(hc_err))):.3e}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare MuMax vs Rust for Standard Problem #2 (SP2).")

    # Match SP4-style interface: root directories
    ap.add_argument("--mumax-root", type=Path, help="MuMax SP2 root (expects sp2_out/table.txt inside)")
    ap.add_argument("--rust-root", type=Path, help="Rust SP2 root (expects table.csv inside)")

    # Backwards-compatible: direct tables
    ap.add_argument("--mumax-table", type=Path, help="MuMax table.txt (d mx my Hc/Ms)")
    ap.add_argument("--rust-table", type=Path, help="Rust table.csv (headered)")

    ap.add_argument("--out", type=Path, default=Path("out/st_problems/sp2/sp2_overlay.png"),
                    help="Output PNG path for combined two-panel figure.")
    ap.add_argument("--paper-style", action="store_true", help="Match MuMax3 paper axis limits/ticks.")
    ap.add_argument("--metrics", action="store_true", help="Print basic error metrics (Rust vs MuMax).")
    ap.add_argument("--show", action="store_true", help="Show interactive window (also saves PNG).")
    ap.add_argument("--dpi", type=int, default=250, help="PNG DPI.")
    args = ap.parse_args()

    # Resolve input paths
    if args.mumax_table and args.rust_table:
        mumax_table = args.mumax_table
        rust_table = args.rust_table
    elif args.mumax_root and args.rust_root:
        mumax_table = find_mumax_table_from_root(args.mumax_root)
        rust_table = find_rust_table_from_root(args.rust_root)
    else:
        raise SystemExit(
            "Provide either (--mumax-table AND --rust-table) or (--mumax-root AND --rust-root)."
        )

    d_m, mx_m, my_m, hc_m = load_mumax_table(mumax_table)
    d_r, mx_r, my_r, hc_r = load_rust_table(rust_table)

    d_m, mx_m, my_m, hc_m = sort_by_d(d_m, mx_m, my_m, hc_m)
    d_r, mx_r, my_r, hc_r = sort_by_d(d_r, mx_r, my_r, hc_r)

    if args.metrics:
        print(compute_metrics(d_m, mx_m, my_m, hc_m, d_r, mx_r, my_r, hc_r))

    # ---------------------------
    # Combined figure (two panels)
    # ---------------------------
    fig, (ax_top, ax_bot) = plt.subplots(nrows=2, figsize=(7.2, 7.6))

    # ---- Top: Fig. 13 remanence style ----
    ax_top.plot(d_m, mx_m, "s", color="red", markersize=4, markeredgewidth=0.0)
    ax_top.plot(d_r, mx_r, "-", color="red", linewidth=1.2)

    ax_top.set_xlabel(r"$d/\ell_{ex}$")
    ax_top.set_ylabel(r"$<mx>$")

    ax_top_r = ax_top.twinx()
    ax_top_r.plot(d_m, my_m, "s", color="limegreen", markersize=4, markeredgewidth=0.0)
    ax_top_r.plot(d_r, my_r, "-", color="limegreen", linewidth=1.2)
    ax_top_r.set_ylabel(r"$<my>$")

    if args.paper_style:
        ax_top.set_xlim(0, 30)
        ax_top.set_ylim(0.9, 1.001)
        ax_top.set_xticks([0, 10, 20, 30])
        ax_top.set_yticks([0.9, 0.95, 1.0])

        ax_top_r.set_ylim(0.0, 0.1)
        ax_top_r.set_yticks([0.0, 0.05, 0.1])

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="None", color="red", markersize=6, label=r"$<mx>$"),
        Line2D([0], [0], marker="s", linestyle="None", color="limegreen", markersize=6, label=r"$<my>$"),
    ]
    ax_top.legend(handles=legend_handles, loc="lower right", frameon=True)
    ax_top.grid(False)
    ax_top_r.grid(False)

    # ---- Bottom: Fig. 14 coercivity style ----
    ax_bot.plot(d_m, hc_m, "s", color="red", markersize=4, markeredgewidth=0.0, label="mumax")
    ax_bot.plot(d_r, hc_r, "-", color="black", linewidth=1.2, label="rust")

    ax_bot.set_xlabel(r"$d/\ell_{ex}$")
    ax_bot.set_ylabel(r"$Hc/Msat$")

    if args.paper_style:
        ax_bot.set_xlim(0, 30)
        ax_bot.set_ylim(0.044, 0.064)
        ax_bot.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax_bot.set_yticks([0.044, 0.046, 0.048, 0.050, 0.052, 0.054, 0.056, 0.058, 0.060, 0.062, 0.064])

    ax_bot.grid(False)
    ax_bot.legend(loc="upper right", frameon=True)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Wrote: {args.out}")
    print(f"Using MuMax table: {mumax_table}")
    print(f"Using Rust table:  {rust_table}")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()