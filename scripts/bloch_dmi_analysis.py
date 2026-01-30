import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================

def load_cfg_dmi(run_dir: Path):
    """
    Load DMI value from config.json if present.
    """
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return cfg.get("fields", {}).get("dmi", None)


def load_final_slice(run_dir: Path):
    """
    Load final Bloch DMI slice.

    Expected file:
      run_dir / rust_slice_final.csv

    Columns:
      x, mx, my, mz
    """
    slice_path = run_dir / "rust_slice_final.csv"
    if not slice_path.exists():
        raise FileNotFoundError(f"Missing {slice_path}")

    data = np.loadtxt(slice_path, delimiter=",", skiprows=1)

    if data.shape[1] != 4:
        raise ValueError(
            f"Expected 4 columns (x,mx,my,mz) in {slice_path}, got {data.shape[1]}"
        )

    x  = data[:, 0]   # meters
    mx = data[:, 1]
    my = data[:, 2]
    mz = data[:, 3]

    return x, mx, my, mz, slice_path


def wall_center(x, mz):
    """
    Wall center defined by minimum |mz|.
    """
    i0 = int(np.argmin(np.abs(mz)))
    return i0, x[i0]


def window_stats(mx, my, mz, i0, half_window):
    """
    Chirality statistics averaged in a window around the wall.
    """
    i_lo = max(0, i0 - half_window)
    i_hi = min(len(mx), i0 + half_window + 1)

    mx_w = mx[i_lo:i_hi]
    my_w = my[i_lo:i_hi]
    mz_w = mz[i_lo:i_hi]

    phi = np.arctan2(np.mean(my_w), np.mean(mx_w))

    return {
        "i_lo": i_lo,
        "i_hi": i_hi,
        "mx_mean": float(np.mean(mx_w)),
        "my_mean": float(np.mean(my_w)),
        "mz_mean": float(np.mean(mz_w)),
        "phi": float(phi),
    }


# ============================================================
# Main analysis
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dplus", type=Path, required=True,
                    help="Directory for +D run (e.g. out/bloch_dmi/Dplus)")
    ap.add_argument("--dminus", type=Path, required=True,
                    help="Directory for -D run (e.g. out/bloch_dmi/Dminus)")
    ap.add_argument("--half_window", type=int, default=10,
                    help="Half-width (cells) for chirality averaging")
    ap.add_argument("--out", type=Path,
                    default=Path("out/bloch_dmi/bloch_dmi_components.png"),
                    help="Output PNG path")
    args = ap.parse_args()

    # --------------------------------------------------------
    # Load slices
    # --------------------------------------------------------

    x_p, mx_p, my_p, mz_p, pfile = load_final_slice(args.dplus)
    x_m, mx_m, my_m, mz_m, mfile = load_final_slice(args.dminus)

    d_p = load_cfg_dmi(args.dplus)
    d_m = load_cfg_dmi(args.dminus)

    # --------------------------------------------------------
    # Center on wall
    # --------------------------------------------------------

    i0p, x0p = wall_center(x_p, mz_p)
    i0m, x0m = wall_center(x_m, mz_m)

    x_p_c = (x_p - x0p) * 1e9  # nm
    x_m_c = (x_m - x0m) * 1e9  # nm

    # --------------------------------------------------------
    # Chirality diagnostics (printed)
    # --------------------------------------------------------

    stats_p = window_stats(mx_p, my_p, mz_p, i0p, args.half_window)
    stats_m = window_stats(mx_m, my_m, mz_m, i0m, args.half_window)

    print("\n=== Bloch DMI chirality check ===\n")

    for label, dmi, i0, x0, stats in [
        ("+D", d_p, i0p, x0p, stats_p),
        ("-D", d_m, i0m, x0m, stats_m),
    ]:
        print(f"{label} case:")
        print(f"  DMI           = {dmi}")
        print(f"  wall index    = {i0}")
        print(f"  wall x0 (nm)  = {x0 * 1e9:.3f}")
        print(f"  <mx> window   = {stats['mx_mean']:+.6f}")
        print(f"  <my> window   = {stats['my_mean']:+.6f}")
        print(f"  phi (rad)     = {stats['phi']:+.6f}")
        print("")

    # --------------------------------------------------------
    # Decomposition plot (ONLY plot we keep)
    # --------------------------------------------------------

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.8))

    # mz: invariant under D
    plt.plot(x_p_c, mz_p, label=f"+D mz (D={d_p})", linewidth=1.6)
    plt.plot(x_m_c, mz_m, label=f"-D mz (D={d_m})", linewidth=1.6)

    # my: chirality-sensitive
    plt.plot(x_p_c, my_p, "--", label=f"+D my (D={d_p})", linewidth=1.4)
    plt.plot(x_m_c, my_m, "--", label=f"-D my (D={d_m})", linewidth=1.4)

    # mx: small, flips sign
    plt.plot(x_p_c, mx_p, ":", label=f"+D mx (D={d_p})", linewidth=1.2)
    plt.plot(x_m_c, mx_m, ":", label=f"-D mx (D={d_m})", linewidth=1.2)

    plt.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
    plt.xlabel(r"$x-x_0$ (nm)")
    plt.ylabel("magnetisation")
    plt.title("Bloch wall DMI chirality: component decomposition")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Wrote component chirality plot to {args.out}")
    print(f"+D slice: {pfile}")
    print(f"-D slice: {mfile}")


if __name__ == "__main__":
    main()