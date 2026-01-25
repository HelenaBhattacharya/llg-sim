import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_csv_with_header(path: Path) -> Dict[str, np.ndarray]:
    """
    Reads a CSV with a header row into a dict of numpy arrays.
    Robust to whitespace.
    """
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
    header = [h.strip() for h in header]
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(header):
        raise RuntimeError(f"{path} has {data.shape[1]} cols but header has {len(header)}: {header}")
    return {h: data[:, i] for i, h in enumerate(header)}


def load_run_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def fmt_dmi(dmi_val) -> str:
    if dmi_val is None:
        return "DMI=None"
    try:
        return f"DMI={float(dmi_val):.2e}"
    except Exception:
        return f"DMI={dmi_val}"


def fft_peak_freq(t: np.ndarray, y: np.ndarray) -> float:
    """
    FFT peak (Hz) after detrending (remove mean). Works for uniform time spacing.
    """
    if len(t) < 4:
        return 0.0
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return 0.0
    y0 = y - float(np.mean(y))
    Y = np.fft.rfft(y0)
    f = np.fft.rfftfreq(len(y0), d=dt)
    mag = np.abs(Y)
    if len(mag) > 1:
        mag[0] = 0.0
    k = int(np.argmax(mag)) if len(mag) else 0
    return float(f[k]) if k < len(f) else 0.0


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Run directories under runs/<id>/ (must contain config.json, avg_magnetisation.csv, energy_vs_time.csv)",
    )
    ap.add_argument("--out_dir", default="out/bloch_run_compare", help="Output directory for plots")
    ap.add_argument("--t_unit", default="ns", choices=["s", "ns"], help="Time unit on plots")
    ap.add_argument("--do_fft", action="store_true", help="Compute FFT peak of <my>(t) for each run")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    runs: List[Tuple[Path, Dict, Dict[str, np.ndarray], Dict[str, np.ndarray]]] = []
    labels: List[str] = []

    # Load everything
    for rd in args.run_dirs:
        run_dir = Path(rd)
        cfg = load_run_config(run_dir)

        mag_path = run_dir / "avg_magnetisation.csv"
        ene_path = run_dir / "energy_vs_time.csv"

        if not mag_path.exists():
            raise FileNotFoundError(f"Missing {mag_path}")
        if not ene_path.exists():
            raise FileNotFoundError(f"Missing {ene_path}")

        mag = read_csv_with_header(mag_path)
        ene = read_csv_with_header(ene_path)

        dmi_val = cfg.get("fields", {}).get("dmi", None)
        label = f"{run_dir.name} ({fmt_dmi(dmi_val)})"
        labels.append(label)
        runs.append((run_dir, cfg, mag, ene))

    # Helper: time axis conversion
    def convert_time(t: np.ndarray) -> np.ndarray:
        if args.t_unit == "s":
            return t
        # ns
        return t * 1e9

    t_label = "time (s)" if args.t_unit == "s" else "time (ns)"

    # -----------------------------
    # Plot avg magnetisation overlays
    # -----------------------------
    for comp, ylabel in [("mx_avg", r"$\langle m_x\rangle$"),
                         ("my_avg", r"$\langle m_y\rangle$"),
                         ("mz_avg", r"$\langle m_z\rangle$")]:
        plt.figure(figsize=(8, 4.5))
        for (run_dir, cfg, mag, ene), label in zip(runs, labels):
            if comp not in mag:
                continue
            t = convert_time(mag["t"])
            plt.plot(t, mag[comp], label=label)
        plt.xlabel(t_label)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs time (Bloch runs)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"m_{comp}_overlay.png", dpi=200)
        plt.close()

    # -----------------------------
    # Plot energy overlays
    # energy_vs_time.csv headers may differ across older runs.
    # We'll try to plot what exists.
    # -----------------------------
    energy_series = [
        ("E_ex", "Exchange energy (J)"),
        ("E_an", "Anisotropy energy (J)"),
        ("E_zee", "Zeeman energy (J)"),
        ("E_dmi", "DMI energy (J)"),
        ("E_tot", "Total energy (J)"),
    ]

    for key, title in energy_series:
        plt.figure(figsize=(8, 4.5))
        plotted_any = False
        for (run_dir, cfg, mag, ene), label in zip(runs, labels):
            if key not in ene:
                continue
            t = convert_time(ene["t"])
            plt.plot(t, ene[key], label=label)
            plotted_any = True
        if not plotted_any:
            plt.close()
            continue
        plt.xlabel(t_label)
        plt.ylabel("energy (J)")
        plt.title(title)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"energy_{key}_overlay.png", dpi=200)
        plt.close()

    # -----------------------------
    # Summary metrics
    # -----------------------------
    lines = []
    lines.append("Bloch run comparison summary\n")
    lines.append(f"Runs: {len(runs)}\n")
    lines.append(f"Time unit plotted: {args.t_unit}\n")
    lines.append("\n")

    for (run_dir, cfg, mag, ene), label in zip(runs, labels):
        dmi_val = cfg.get("fields", {}).get("dmi", None)
        lines.append(f"== {run_dir.name} ==\n")
        lines.append(f"label: {label}\n")
        lines.append(f"dmi: {dmi_val}\n")

        # final magnetisation averages
        for comp in ["mx_avg", "my_avg", "mz_avg"]:
            if comp in mag:
                lines.append(f"final {comp}: {mag[comp][-1]: .6e}\n")

        # final energies
        for k in ["E_ex", "E_an", "E_zee", "E_dmi", "E_tot"]:
            if k in ene:
                lines.append(f"final {k}: {ene[k][-1]: .6e}\n")

        # FFT peak
        if args.do_fft and ("t" in mag) and ("my_avg" in mag):
            fpk = fft_peak_freq(mag["t"], mag["my_avg"])
            lines.append(f"FFT peak freq of <my>(t): {fpk:.6e} Hz\n")

        lines.append("\n")

    with open(out_dir / "summary.txt", "w") as f:
        f.writelines(lines)

    print(f"Wrote plots + summary to: {out_dir}")
    print(f"Summary: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()