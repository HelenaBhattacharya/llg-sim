# scripts/overlay_macrospin.py
#
# Overlay Rust vs MuMax outputs and (optionally) compute FFT peak frequency.
#
# Examples:
# python3 scripts/overlay_macrospin.py \
#   out/macrospin_fmr/rust_table_macrospin_fmr.csv \
#   mumax_outputs/macrospin_fmr/table.txt \
#   --col my \
#   --do_fft \
#   --out out/macrospin_fmr/overlay_my_vs_time.png

# python3 scripts/overlay_macrospin.py \
#   out/macrospin_anisotropy/rust_table_macrospin_anisotropy.csv \
#   mumax_outputs/macrospin_anisotropy/table.txt \
#   --col mz \
#   --out out/macrospin_anisotropy/overlay_mz_vs_time.png

# python3 scripts/overlay_macrospin.py \
#   out/macrospin_anisotropy/rust_table_macrospin_anisotropy.csv \
#   mumax_outputs/macrospin_anisotropy/table.txt \
#   --col my \
#   --out out/macrospin_anisotropy/overlay_my_vs_time.png

# python3 scripts/overlay_macrospin.py \
#   out/uniform_film/rust_table_uniform_film.csv \
#   mumax_outputs/uniform_film_field/table.txt \
#   --col my \
#   --out out/uniform_film/overlay_my_vs_time.png

# Output:
#   out/<something>.png

import argparse
import csv
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

try:
    import numpy as np
except ImportError:
    np = None


MAG_COLS = {"mx", "my", "mz"}


def read_rust_csv(path: str) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols


def _clean_header_token(tok: str) -> str:
    tok = tok.strip()
    tok = tok.strip("()")
    tok = tok.replace("/", "_")
    return tok


def read_mumax_table(path: str) -> Tuple[List[str], List[List[float]]]:
    header: List[str] = []
    data: List[List[float]] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("#"):
                # header line
                if "t" in line and ("mx" in line or "my" in line or "mz" in line):
                    toks = re.split(r"\s+", line.lstrip("#").strip())
                    cleaned = []
                    for tok in toks:
                        # filter unit tokens like "(s)" "(J)"
                        if tok.startswith("(") and tok.endswith(")"):
                            continue
                        cleaned.append(_clean_header_token(tok))
                    header = cleaned
                continue

            parts = re.split(r"\s+", line)
            try:
                row = [float(x) for x in parts]
            except ValueError:
                continue
            data.append(row)

    if not header:
        raise RuntimeError("Could not find MuMax header line starting with '# ... t mx my mz ...'")

    return header, data


def columns_from_table(header: List[str], data: List[List[float]]) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {h: [] for h in header}
    for row in data:
        for i, h in enumerate(header):
            if i < len(row):
                cols[h].append(row[i])
    return cols


def fft_peak_freq(t: List[float], y: List[float]) -> float:
    if np is None:
        raise RuntimeError("numpy not available; install numpy for FFT peak.")
    t_arr = np.asarray(t)
    y_arr = np.asarray(y)

    dt = float(np.median(np.diff(t_arr)))
    y_detrend = y_arr - float(np.mean(y_arr))

    Y = np.fft.rfft(y_detrend)
    freqs = np.fft.rfftfreq(len(y_detrend), d=dt)

    mag = np.abs(Y)
    if len(mag) > 1:
        mag[0] = 0.0

    k = int(np.argmax(mag))
    return float(freqs[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rust_csv", help="Path to Rust CSV")
    ap.add_argument("mumax_table", help="Path to MuMax table.txt")
    ap.add_argument("--col", default="my", help="Column to plot (e.g. mx, my, mz, E_total, E_ex, ...)")
    ap.add_argument("--out", default=None, help="Output plot filename")
    ap.add_argument("--do_fft", action="store_true", help="Also compute FFT peak frequencies (mx/my/mz only; requires numpy)")
    args = ap.parse_args()

    col = args.col

    rust = read_rust_csv(args.rust_csv)
    hdr, rows = read_mumax_table(args.mumax_table)
    mumax = columns_from_table(hdr, rows)

    if "t" not in rust:
        raise RuntimeError("Rust CSV is missing column 't'")
    if "t" not in mumax:
        print("MuMax columns found:", list(mumax.keys()))
        raise RuntimeError("MuMax table is missing column 't'")

    if col not in rust:
        raise RuntimeError(f"Rust CSV missing column '{col}'. Available: {list(rust.keys())}")
    if col not in mumax:
        print("MuMax columns found:", list(mumax.keys()))
        raise RuntimeError(f"MuMax table missing column '{col}'")

    t_r = rust["t"]
    y_r = rust[col]
    t_m = mumax["t"]
    y_m = mumax[col]

    if args.out is None:
        # auto name
        base = os.path.splitext(os.path.basename(args.rust_csv))[0]
        args.out = f"out/overlay_{base}_{col}_vs_time.png"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure()
    plt.plot(t_r, y_r, label=f"Rust {col}(t)")
    plt.plot(t_m, y_m, label=f"MuMax {col}(t)", linestyle="--")
    plt.xlabel("time (s)")

    if col in MAG_COLS:
        plt.ylabel(f"m_{col[-1]} (dimensionless)")  # mx->m_x, etc.
        plt.title(f"{col}(t): Rust vs MuMax")
    else:
        plt.ylabel(col)
        plt.title(f"{col} vs time: Rust vs MuMax")

    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")

    if args.do_fft:
        if col not in MAG_COLS:
            print(f"Skipping FFT: --col {col} is not one of {sorted(MAG_COLS)}")
            return
        if np is None:
            print("numpy not available; skipping FFT.")
            return
        f_r = fft_peak_freq(t_r, y_r)
        f_m = fft_peak_freq(t_m, y_m)
        print(f"FFT peak frequency (Rust):  {f_r:.6e} Hz")
        print(f"FFT peak frequency (MuMax): {f_m:.6e} Hz")


if __name__ == "__main__":
    main()
