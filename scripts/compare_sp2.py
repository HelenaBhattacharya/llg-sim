#!/usr/bin/env python3
# Generate output plots to compare MuMax3 vs Rust for Standard Problem #2 (SP2):
# 1) Run MuMax3 visualisation:
# python3 scripts/mag_visualisation.py \
#   --input mumax_outputs/st_problems/sp2/sp2_out \
#   --output plots/sp2_mumax

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

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


def load_oommf_coercivity_table(path: Path) -> Tuple[Array, Array]:
    """Load OOMMF (Donahue/NIST) coercivity reference values.

    Expected CSV header (recommended):
        d_lex,hc_over_ms

    For robustness, we also accept a few common header variants.
    """
    raw = np.genfromtxt(path, delimiter=",", names=True)
    names = raw.dtype.names or ()

    # Accept a few variants for the d/l_ex column name
    d_candidates = ("d_lex", "d_over_lex", "d_l_ex", "dlex", "d")
    hc_candidates = ("hc_over_ms", "hc_ms", "hc", "Hc_Ms", "HcMs", "Hc_over_Ms", "Hc", "hcms")

    d_key = next((k for k in d_candidates if k in names), None)
    hc_key = next((k for k in hc_candidates if k in names), None)

    if d_key is None or hc_key is None:
        raise ValueError(
            f"Missing required columns in {path}. Found {names}. "
            f"Need one of {d_candidates} and one of {hc_candidates}."
        )

    return raw[d_key], raw[hc_key]


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


def find_oommf_table_from_root(root: Path) -> Path:
    """Find OOMMF reference table in the given SP2 run directory.

    By convention we look for a file placed next to table.csv:
        oommf_donahue.csv
    """
    preferred = root / "oommf_donahue.csv"
    if preferred.exists():
        return preferred

    # Fallback: any CSV containing 'oommf' or 'donahue' in the name
    hits = [
        p
        for p in root.rglob("*.csv")
        if any(tok in p.name.lower() for tok in ("oommf", "donahue"))
    ]
    if not hits:
        raise FileNotFoundError(
            f"No OOMMF reference CSV found under {root}. Expected {preferred.name} next to table.csv."
        )
    # Prefer the most specific-looking filename
    hits.sort(key=lambda p: ("donahue" not in p.name.lower(), len(p.name)))
    return hits[0]


def _sanitize_xy(x: Array, y: Array) -> Tuple[Array, Array]:
    """Remove NaN/inf, sort by x, and drop duplicate x entries."""
    mask = np.isfinite(x) & np.isfinite(y)
    x2 = np.asarray(x[mask], dtype=float)
    y2 = np.asarray(y[mask], dtype=float)

    if x2.size < 2:
        return x2, y2

    order = np.argsort(x2)
    x2 = x2[order]
    y2 = y2[order]

    # Drop duplicate x (keep first occurrence)
    _, idx = np.unique(x2, return_index=True)
    idx.sort()
    return x2[idx], y2[idx]


def match_on_common_d_lex(
    d_m: Array,
    mx_m: Array,
    my_m: Array,
    hc_m: Array,
    d_r: Array,
    mx_r: Array,
    my_r: Array,
    hc_r: Array,
) -> Tuple[Array, Array, Array, Array]:
    """
    Match MuMax and Rust results on common d/lex values.

    The SP2 tables usually report integer d/lex values. We match using rounded integers
    and return residuals (Rust - MuMax) on the common d/lex grid.
    """
    dm_int = np.rint(d_m).astype(int)
    dr_int = np.rint(d_r).astype(int)

    common = np.intersect1d(dm_int, dr_int)
    if common.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Build maps: d_int -> value (take first occurrence)
    mumax_map = {}
    for dval in common:
        idx = np.where(dm_int == dval)[0]
        if idx.size:
            i = int(idx[0])
            mumax_map[int(dval)] = (float(mx_m[i]), float(my_m[i]), float(hc_m[i]))

    rust_map = {}
    for dval in common:
        idx = np.where(dr_int == dval)[0]
        if idx.size:
            i = int(idx[0])
            rust_map[int(dval)] = (float(mx_r[i]), float(my_r[i]), float(hc_r[i]))

    # Only keep d where both exist
    d_common = []
    dmx = []
    dmy = []
    dhc = []

    for dval in sorted(set(mumax_map.keys()) & set(rust_map.keys())):
        mxm, mym, hcm = mumax_map[dval]
        mxr, myr, hcr = rust_map[dval]
        d_common.append(float(dval))
        dmx.append(mxr - mxm)
        dmy.append(myr - mym)
        dhc.append(hcr - hcm)

    return np.array(d_common), np.array(dmx), np.array(dmy), np.array(dhc)


def match_hc_rust_minus_avg_mumax_oommf(
    d_m: Array,
    hc_m: Array,
    d_o: Array,
    hc_o: Array,
    d_r: Array,
    hc_r: Array,
) -> Tuple[Array, Array]:
    """Return coercivity residuals: Rust − 0.5*(MuMax + OOMMF) on common integer d/lex."""
    dm_int = np.rint(d_m).astype(int)
    do_int = np.rint(d_o).astype(int)
    dr_int = np.rint(d_r).astype(int)

    common = np.intersect1d(np.intersect1d(dm_int, do_int), dr_int)
    if common.size == 0:
        return np.array([]), np.array([])

    mumax_map = {int(d): float(hc_m[np.where(dm_int == d)[0][0]]) for d in common if np.where(dm_int == d)[0].size}
    oommf_map = {int(d): float(hc_o[np.where(do_int == d)[0][0]]) for d in common if np.where(do_int == d)[0].size}
    rust_map = {int(d): float(hc_r[np.where(dr_int == d)[0][0]]) for d in common if np.where(dr_int == d)[0].size}

    d_common = []
    dhc = []
    for dval in sorted(set(mumax_map.keys()) & set(oommf_map.keys()) & set(rust_map.keys())):
        ref = 0.5 * (mumax_map[dval] + oommf_map[dval])
        d_common.append(float(dval))
        dhc.append(rust_map[dval] - ref)

    return np.array(d_common), np.array(dhc)


def compute_metrics_text(
    d_common_rem: Array,
    dmx: Array,
    dmy: Array,
    d_common_hc: Array,
    dhc: Array,
) -> str:
    """Format terminal metrics for SP2 residuals.

    Remanence residual definition:
        Δm = Rust − MuMax

    Coercivity residual definition:
        ΔHc = Rust − 0.5*(MuMax + OOMMF)
    """

    def rmse(x: Array) -> float:
        return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")

    def p95(x: Array) -> float:
        return float(np.quantile(np.abs(x), 0.95)) if x.size else float("nan")

    def max_abs(d: Array, x: Array) -> Tuple[float, float]:
        if x.size == 0:
            return float("nan"), float("nan")
        i = int(np.argmax(np.abs(x)))
        return float(np.abs(x[i])), float(d[i])

    lines = [
        f"[metrics] SP2 matched points: remanence={d_common_rem.size} (vs MuMax), coercivity={d_common_hc.size} (vs avg(MuMax, OOMMF))",
        "Remanence residual: Rust − MuMax",
    ]

    if d_common_rem.size:
        mx_max, mx_at = max_abs(d_common_rem, dmx)
        my_max, my_at = max_abs(d_common_rem, dmy)
        lines.extend(
            [
                f"  mx_rem: RMSE={rmse(dmx):.3e}  max|Δ|={mx_max:.3e}  p95|Δ|={p95(dmx):.3e}  d/lex@max={mx_at:.0f}",
                f"  my_rem: RMSE={rmse(dmy):.3e}  max|Δ|={my_max:.3e}  p95|Δ|={p95(dmy):.3e}  d/lex@max={my_at:.0f}",
            ]
        )
    else:
        lines.append("  (no overlapping d/lex values for remanence metrics)")

    lines.append("Coercivity residual: Rust − 0.5*(MuMax + OOMMF)")
    if d_common_hc.size:
        hc_max, hc_at = max_abs(d_common_hc, dhc)
        lines.append(
            f"  hc/Ms : RMSE={rmse(dhc):.3e}  max|Δ|={hc_max:.3e}  p95|Δ|={p95(dhc):.3e}  d/lex@max={hc_at:.0f}"
        )
    else:
        lines.append("  (no overlapping d/lex values for coercivity metrics)")

    return "\n".join(lines)


def save_residuals_figure(
    out_path: Path,
    d_common_rem: Array,
    dmx: Array,
    dmy: Array,
    d_common_hc: Array,
    dhc: Array,
    *,
    dpi: int = 250,
) -> None:
    """Save residuals plot for SP2.

    Top:  Remanence residuals  (Rust − MuMax)
    Bottom: Coercivity residuals (Rust − 0.5*(MuMax + OOMMF))
    """
    if d_common_rem.size == 0 and d_common_hc.size == 0:
        print("[residuals] No overlapping d/lex values; skipping residual plot.")
        return

    fig, (ax_top, ax_bot) = plt.subplots(nrows=2, figsize=(7.2, 7.6))

    # --- Top residuals: remanence components (Rust − MuMax) ---
    if d_common_rem.size:
        d_rem_x, dmx_p = _sanitize_xy(d_common_rem, dmx)
        d_rem_y, dmy_p = _sanitize_xy(d_common_rem, dmy)

        ax_top.axhline(0.0, linewidth=0.8)
        ax_top.plot(d_rem_x, dmx_p, "-o", color="red", markersize=3, linewidth=1.2, label="Δmx_rem")
        ax_top.plot(d_rem_y, dmy_p, "-o", color="limegreen", markersize=3, linewidth=1.2, label="Δmy_rem")
        ax_top.set_title("SP2 residuals: remanence (Rust − MuMax)")
        ax_top.legend(loc="best", frameon=True)
    else:
        ax_top.text(0.5, 0.5, "No overlap for remanence", ha="center", va="center")
        ax_top.set_title("SP2 residuals: remanence")

    ax_top.set_xlabel(r"$d/\ell_{ex}$")
    ax_top.set_ylabel("Residual")
    ax_top.grid(False)

    # --- Bottom residuals: coercivity (Rust − avg(MuMax, OOMMF)) ---
    if d_common_hc.size:
        d_hc, dhc_p = _sanitize_xy(d_common_hc, dhc)
        ax_bot.axhline(0.0, linewidth=0.8)
        ax_bot.plot(d_hc, dhc_p, "-o", color="black", markersize=3, linewidth=1.2, label="Δ(Hc/Ms)")
        ax_bot.set_title("SP2 residuals: coercivity (Rust − avg(MuMax, OOMMF))")
        ax_bot.legend(loc="best", frameon=True)
    else:
        ax_bot.text(0.5, 0.5, "No overlap for coercivity", ha="center", va="center")
        ax_bot.set_title("SP2 residuals: coercivity")

    ax_bot.set_xlabel(r"$d/\ell_{ex}$")
    ax_bot.set_ylabel("Residual")
    ax_bot.grid(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare MuMax vs Rust for Standard Problem #2 (SP2).")

    # Match SP4-style interface: root directories
    ap.add_argument("--mumax-root", type=Path, help="MuMax SP2 root (expects sp2_out/table.txt inside)")
    ap.add_argument("--rust-root", type=Path, help="Rust SP2 root (expects table.csv inside)")

    # Backwards-compatible: direct tables
    ap.add_argument("--mumax-table", type=Path, help="MuMax table.txt (d mx my Hc/Ms)")
    ap.add_argument("--rust-table", type=Path, help="Rust table.csv (headered)")

    # Optional OOMMF reference (Donahue/NIST) for coercivity only
    ap.add_argument(
        "--oommf-table",
        type=Path,
        help="OOMMF (Donahue/NIST) coercivity CSV. If omitted, we look for 'oommf_donahue.csv' next to Rust table.csv.",
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path for combined two-panel figure. Defaults to <rust-root>/sp2_overlay.png.",
    )
    ap.add_argument("--paper-style", action="store_true", help="Match MuMax3 paper axis limits/ticks.")
    ap.add_argument(
        "--metrics",
        action="store_true",
        help="Print metrics on matched d/lex grids. Remanence: Rust−MuMax. Coercivity: Rust−avg(MuMax, OOMMF).",
    )
    ap.add_argument("--show", action="store_true", help="Show interactive window (also saves PNG).")
    ap.add_argument("--dpi", type=int, default=250, help="PNG DPI.")
    args = ap.parse_args()

    # Resolve input paths
    if args.mumax_table and args.rust_table:
        mumax_table = args.mumax_table
        rust_table = args.rust_table
        rust_root: Optional[Path] = rust_table.parent
    elif args.mumax_root and args.rust_root:
        mumax_table = find_mumax_table_from_root(args.mumax_root)
        rust_table = find_rust_table_from_root(args.rust_root)
        rust_root = args.rust_root
    else:
        raise SystemExit(
            "Provide either (--mumax-table AND --rust-table) or (--mumax-root AND --rust-root)."
        )

    # Default output location: write next to the Rust outputs unless the user overrides --out.
    if args.out is None:
        if rust_root is not None:
            args.out = rust_root / "sp2_overlay.png"
        else:
            args.out = Path("sp2_overlay.png")

    # Resolve OOMMF reference table (coercivity only)
    oommf_table: Optional[Path] = None
    if args.oommf_table:
        oommf_table = args.oommf_table
    else:
        # Try to locate next to Rust table (or under rust root)
        try:
            if rust_root is not None:
                oommf_table = find_oommf_table_from_root(rust_root)
        except FileNotFoundError:
            oommf_table = None

    d_m, mx_m, my_m, hc_m = load_mumax_table(mumax_table)
    d_r, mx_r, my_r, hc_r = load_rust_table(rust_table)

    d_m, mx_m, my_m, hc_m = sort_by_d(d_m, mx_m, my_m, hc_m)
    d_r, mx_r, my_r, hc_r = sort_by_d(d_r, mx_r, my_r, hc_r)

    # Remanence residuals on matched d/lex grid (Rust − MuMax)
    d_common_rem, dmx, dmy, _dhc_unused = match_on_common_d_lex(d_m, mx_m, my_m, hc_m, d_r, mx_r, my_r, hc_r)

    # Coercivity residuals: Rust − avg(MuMax, OOMMF)
    d_common_hc = np.array([])
    dhc = np.array([])
    if oommf_table is not None and oommf_table.exists():
        d_o, hc_o = load_oommf_coercivity_table(oommf_table)
        d_o, hc_o = sort_by_d(d_o, hc_o)
        d_common_hc, dhc = match_hc_rust_minus_avg_mumax_oommf(d_m, hc_m, d_o, hc_o, d_r, hc_r)
    else:
        print("[warn] No OOMMF reference table found; coercivity metrics will fall back to Rust − MuMax and plot will omit OOMMF.")
        d_common_hc, dhc = match_hc_rust_minus_avg_mumax_oommf(d_m, hc_m, d_m, hc_m, d_r, hc_r)  # avg=Mumax

    if args.metrics:
        print(compute_metrics_text(d_common_rem, dmx, dmy, d_common_hc, dhc))

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
    # Plot points only for all three series (MuMax, Rust, OOMMF).
    ax_bot.plot(
        d_m,
        hc_m,
        linestyle="None",
        marker="s",
        color="red",
        markersize=4,
        markeredgewidth=0.0,
        label="mumax",
    )

    if oommf_table is not None and oommf_table.exists():
        ax_bot.plot(
            d_o,
            hc_o,
            linestyle="None",
            marker="^",
            color="tab:blue",
            markersize=3,
            markeredgewidth=0.0,
            label="oommf (Donahue)",
        )

    ax_bot.plot(
        d_r,
        hc_r,
        linestyle="None",
        marker="x",
        color="black",
        markersize=5,
        markeredgewidth=1.0,
        label="rust",
    )

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
    if oommf_table is not None and oommf_table.exists():
        print(f"Using OOMMF table: {oommf_table}")

    # Save residuals plot next to overlay
    residual_path = args.out.parent / f"{args.out.stem}_residuals{args.out.suffix}"
    save_residuals_figure(residual_path, d_common_rem, dmx, dmy, d_common_hc, dhc, dpi=args.dpi)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()

