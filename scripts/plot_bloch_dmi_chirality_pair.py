import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def load_cfg_dmi(run_dir: Path):
    cfg_path = run_dir / "bloch_relax" / "config.json"
    if not cfg_path.exists():
        cfg_path = run_dir / "config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    return cfg.get("fields", {}).get("dmi", None)


def load_slice(run_dir: Path, tag: str):
    # Supports both layouts:
    #   out/bloch_relax_Dplus/bloch_relax/bloch_slices/rust_slice_t10ns.csv
    #   out/bloch_relax/bloch_slices/...
    candidates = [
        run_dir / "bloch_relax" / "bloch_slices" / f"rust_slice_{tag}.csv",
        run_dir / "bloch_slices" / f"rust_slice_{tag}.csv",
    ]
    for p in candidates:
        if p.exists():
            data = np.loadtxt(p, delimiter=",", skiprows=1)
            x = data[:, 0]  # meters
            if data.shape[1] == 4:
                mx, my, mz = data[:, 1], data[:, 2], data[:, 3]
            elif data.shape[1] == 3:
                mx, mz = data[:, 1], data[:, 2]
                my = np.zeros_like(mx)  # fallback; but you should aim for 4-col slices
            else:
                raise ValueError(f"Unexpected cols in {p}: {data.shape[1]}")
            return x, mx, my, mz, p
    raise FileNotFoundError(f"No slice file found for tag={tag} under {run_dir}")


def center_on_wall(x, mx, my, mz):
    # wall center index where mz closest to 0
    i0 = int(np.argmin(np.abs(mz)))
    x0 = x[i0]
    return (x - x0), i0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dplus", type=Path, required=True, help="Run dir for +D case (e.g. out/bloch_relax_Dplus)")
    ap.add_argument("--dminus", type=Path, required=True, help="Run dir for -D case (e.g. out/bloch_relax_Dminus)")
    ap.add_argument("--tag", default="t10ns", choices=["t0", "t5ns", "t10ns", "final"])
    ap.add_argument("--out", type=Path, default=None, help="Optional output PNG path")
    ap.add_argument("--plot", choices=["my", "all"], default="all", help="Plot only my(x) or all components")
    args = ap.parse_args()

    d_p = load_cfg_dmi(args.dplus)
    d_m = load_cfg_dmi(args.dminus)

    x_p, mx_p, my_p, mz_p, pfile = load_slice(args.dplus, args.tag)
    x_m, mx_m, my_m, mz_m, mfile = load_slice(args.dminus, args.tag)

    x_p_c, i0p = center_on_wall(x_p, mx_p, my_p, mz_p)
    x_m_c, i0m = center_on_wall(x_m, mx_m, my_m, mz_m)

    # convert to nm for plotting
    x_p_nm = x_p_c * 1e9
    x_m_nm = x_m_c * 1e9

    plt.figure(figsize=(7.5, 4.5))

    if args.plot == "my":
        plt.plot(x_p_nm, my_p, label=f"+D: my  (D={d_p})", linewidth=1.6)
        plt.plot(x_m_nm, my_m, label=f"-D: my  (D={d_m})", linewidth=1.6)
        plt.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
        plt.ylabel(r"$m_y$")
        plt.title(f"Bloch wall chirality flip: $m_y(x)$ at {args.tag}")
    else:
        # mz (should be similar), and my should flip sign
        plt.plot(x_p_nm, mz_p, label=f"+D: mz (D={d_p})", linewidth=1.6)
        plt.plot(x_m_nm, mz_m, label=f"-D: mz (D={d_m})", linewidth=1.6)

        plt.plot(x_p_nm, my_p, "--", label=f"+D: my (D={d_p})", linewidth=1.4)
        plt.plot(x_m_nm, my_m, "--", label=f"-D: my (D={d_m})", linewidth=1.4)

        plt.plot(x_p_nm, mx_p, ":", label=f"+D: mx (D={d_p})", linewidth=1.2)
        plt.plot(x_m_nm, mx_m, ":", label=f"-D: mx (D={d_m})", linewidth=1.2)

        plt.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
        plt.ylabel("magnetisation")
        plt.title(f"Bloch wall chirality: +D vs -D slices at {args.tag}")

    plt.xlabel(r"$x-x_0$ (nm)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=200)
        print(f"Wrote {args.out}")
    else:
        plt.show()

    print(f"Loaded:\n  +D slice: {pfile}\n  -D slice: {mfile}")
    print(f"Wall centers: i0(+D)={i0p}, i0(-D)={i0m}")


if __name__ == "__main__":
    main()