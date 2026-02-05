#!/usr/bin/env python3
"""
scripts/compare_demag.py

One comparison script for multiple demag-style tests.

------------------------------------------------------------
How to run
------------------------------------------------------------

1) demag_snapshot (full features)
  python3 scripts/compare_demag.py --test snapshot

  Optional extras:
    python3 scripts/compare_demag.py --test snapshot --tol 1e-6
    python3 scripts/compare_demag.py --test snapshot --plot out/demag_snapshot/demag_snapshot_abs.png --plot_mode abs
    python3 scripts/compare_demag.py --test snapshot --plot out/demag_snapshot/demag_snapshot_rel.png --plot_mode rel
    python3 scripts/compare_demag.py --test snapshot --rel_floor 1e-12

2) demagSmall (minimal output: 9 numbers + deltas)
  python3 scripts/compare_demag.py --test small

3) demag2D (minimal output: 9 numbers + deltas)
  python3 scripts/compare_demag.py --test 2d

4) demag2Dpbc (minimal output: two blocks Nx=2 and Nx=3, each has 9 numbers + deltas)
  python3 scripts/compare_demag.py --test pbc

Expected defaults:

snapshot:
  Rust:
    out/demag_snapshot/rust_demag_snapshot_x.csv
    out/demag_snapshot/rust_demag_snapshot_y.csv
    out/demag_snapshot/rust_demag_snapshot_z.csv
  MuMax:
    mumax_outputs/demag_snapshot_x/table.txt
    mumax_outputs/demag_snapshot_y/table.txt
    mumax_outputs/demag_snapshot_z/table.txt

small:
  Rust:
    out/demagSmall/rust_demagSmall.csv
  MuMax:
    mumax_outputs/demagSmall/table.txt

2d:
  Rust:
    out/demag2D/rust_demag2D.csv
  MuMax:
    mumax_outputs/demag2D/table.txt

pbc:
  Rust:
    out/demag2Dpbc/rust_demag2Dpbc.csv
      header: nx,ny,nz,pbc_x,pbc_y,case,t,mx,my,mz,B_demagx,B_demagy,B_demagz
      rows: Nx=2 cases then Nx=3 cases
  MuMax:
    mumax_outputs/demag2Dpbc/table.txt
      rows: Nx=2 cases then Nx=3 cases (written via TableSave() after each Run(0))
"""

import argparse
import csv
import math
import os
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

REL_FLOOR_DEFAULT = 1e-12  # snapshot mode only: values below this are treated as ~0 (abs-only)


# ----------------------------
# Shared helpers
# ----------------------------

def _clean_header_token(tok: str) -> str:
    tok = tok.strip()
    tok = tok.strip("()")
    tok = tok.replace("/", "_")
    return tok


def make_unique(names: List[str]) -> List[str]:
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
                if "t" in line and ("B_demag" in line or "B_eff" in line or "B_demagx" in line):
                    toks = re.split(r"\s+", line.lstrip("#").strip())
                    cleaned: List[str] = []
                    for tok in toks:
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
        raise RuntimeError(f"Could not find MuMax header line in {path}")
    if not data:
        raise RuntimeError(f"No numeric data rows found in MuMax table: {path}")

    return header, data


def mumax_rows_as_dicts(path: str) -> List[Dict[str, float]]:
    hdr, rows = read_mumax_table(path)
    out: List[Dict[str, float]] = []
    for r in rows:
        d: Dict[str, float] = {}
        for i, h in enumerate(hdr):
            if i < len(r):
                d[h] = r[i]
        out.append(d)
    return out


def rel_err_safe(delta: float, ref: float, rel_floor: float) -> Optional[float]:
    if abs(ref) < rel_floor:
        return None
    return abs(delta) / abs(ref)


def print_case_table(
    case_label: str,
    rust: Dict[str, float],
    mumax: Dict[str, float],
    keys: List[str],
    rel_floor: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    missing = []
    for k in keys:
        if k not in rust:
            missing.append(f"Rust missing '{k}'")
        if k not in mumax:
            missing.append(f"MuMax missing '{k}'")
    if missing:
        msg = "\n".join(missing)
        msg += "\n\nAvailable Rust cols: " + ", ".join(sorted(rust.keys()))
        msg += "\nAvailable MuMax cols: " + ", ".join(sorted(mumax.keys()))
        raise RuntimeError(msg)

    print(f"\n=== {case_label} ===\n")
    print(f"{'key':<10} {'rust':>16} {'mumax':>16} {'delta':>16} {'rel_err':>12}")
    print("-" * 76)

    deltas: Dict[str, float] = {}
    rels: Dict[str, float] = {}
    for k in keys:
        r = rust[k]
        m = mumax[k]
        d = r - m
        re = rel_err_safe(d, m, rel_floor)

        deltas[k] = d
        rels[k] = (re if re is not None else float("nan"))

        re_str = f"{re:>12.3e}" if re is not None else f"{'abs-only':>12}"
        print(f"{k:<10} {r:>16.9e} {m:>16.9e} {d:>16.9e} {re_str}")

    return deltas, rels


def read_rust_single_row_csv(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No data rows found in Rust CSV: {path}")

    out: Dict[str, float] = {}
    row0 = rows[0]
    for k, v in row0.items():
        if v is None or v == "":
            continue
        out[k.strip()] = float(v)
    return out


def tolerance_check_and_plot(
    all_rels: Dict[str, List[float]],
    all_deltas: Dict[str, List[float]],
    keys: List[str],
    tol: Optional[float],
    plot: Optional[str],
    plot_mode: str,
    title: str,
) -> None:
    if tol is not None:
        print("\n=== Tolerance check ===")
        ok = True
        for k in keys:
            vals = [v for v in all_rels[k] if not math.isnan(v)]
            worst = max(vals) if vals else 0.0
            if worst > tol:
                ok = False
                print(f"FAIL: {k} worst rel_err = {worst:.3e} > tol={tol:.3e}")
            else:
                print(f"PASS: {k} worst rel_err = {worst:.3e} <= tol={tol:.3e}")
        print("\nOVERALL:", "PASS" if ok else "FAIL")

    if plot:
        values = all_deltas if plot_mode == "abs" else all_rels
        x_labels = ["+x", "+y", "+z"]
        x = list(range(len(x_labels)))

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        for k in keys:
            ax.plot(x, values[k], marker="o", label=k)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("case")
        ax.set_ylabel("abs(delta)" if plot_mode == "abs" else "relative error")
        ax.set_title(title + f" ({plot_mode})")
        ax.legend(fontsize=8, ncol=3)
        plt.tight_layout()

        os.makedirs(os.path.dirname(plot), exist_ok=True)
        plt.savefig(plot, dpi=200)
        print(f"\nSaved plot: {plot}")


# ----------------------------
# snapshot (full features)
# ----------------------------

def run_test_snapshot(
    rust_dir: str,
    mumax_dir: str,
    tol: Optional[float],
    plot: Optional[str],
    plot_mode: str,
    rel_floor: float,
) -> None:
    keys = ["mx", "my", "mz", "B_demagx", "B_demagy", "B_demagz", "B_effx", "B_effy", "B_effz"]
    cases = [("uniform +x", "x"), ("uniform +y", "y"), ("uniform +z", "z")]

    all_deltas: Dict[str, List[float]] = {k: [] for k in keys}
    all_rels: Dict[str, List[float]] = {k: [] for k in keys}

    for label, suffix in cases:
        rust_csv = os.path.join(rust_dir, f"rust_demag_snapshot_{suffix}.csv")
        mumax_table = os.path.join(mumax_dir, f"demag_snapshot_{suffix}", "table.txt")

        rust = read_rust_single_row_csv(rust_csv)
        mumax = mumax_rows_as_dicts(mumax_table)[0]

        print(f"\nRust : {rust_csv}")
        print(f"MuMax: {mumax_table}")
        deltas, rels = print_case_table(label, rust, mumax, keys, rel_floor=rel_floor)

        for k in keys:
            all_deltas[k].append(deltas[k])
            all_rels[k].append(rels[k])

    tolerance_check_and_plot(all_rels, all_deltas, keys, tol, plot, plot_mode, title="Demag snapshot comparison")


# ----------------------------
# demagSmall / demag2D helpers (3-row CSV + 3-row MuMax table)
# ----------------------------

def _parse_rust_3row_csv(path: str, expected_filename: str) -> Dict[str, Dict[str, float]]:
    """
    Reads a 3-row CSV (x,y,z) and returns:
      {'mx': rowdict, 'my': rowdict, 'mz': rowdict}

    Accepts either:
      - a 'case' string column with mx/my/mz
      - OR 3 rows in order mx,row0; my,row1; mz,row2
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rows_raw: List[Dict[str, str]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_raw.append({k.strip(): (v.strip() if v is not None else "") for k, v in row.items()})

    if len(rows_raw) < 3:
        raise RuntimeError(f"Expected >=3 rows in {expected_filename}, got {len(rows_raw)}: {path}")

    def row_to_floats(r: Dict[str, str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in r.items():
            if k == "case":
                continue
            if v == "":
                continue
            out[k] = float(v)
        return out

    by_case: Dict[str, Dict[str, float]] = {}
    if "case" in rows_raw[0]:
        for r in rows_raw:
            c = r.get("case", "")
            if c in ("mx", "my", "mz"):
                by_case[c] = row_to_floats(r)

    if len(by_case) == 3:
        return by_case

    return {"mx": row_to_floats(rows_raw[0]), "my": row_to_floats(rows_raw[1]), "mz": row_to_floats(rows_raw[2])}


def _resolve_case_from_mumax_row(row: Dict[str, float], tol: float = 1e-6) -> Optional[str]:
    if "mx" not in row or "my" not in row or "mz" not in row:
        return None
    mx, my, mz = row["mx"], row["my"], row["mz"]
    if abs(mx - 1.0) < tol and abs(my) < tol and abs(mz) < tol:
        return "mx"
    if abs(my - 1.0) < tol and abs(mx) < tol and abs(mz) < tol:
        return "my"
    if abs(mz - 1.0) < tol and abs(mx) < tol and abs(my) < tol:
        return "mz"
    return None


def _load_mumax_xyz_cases_from_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    mumax_by_case: Dict[str, Dict[str, float]] = {}
    for r in rows:
        c = _resolve_case_from_mumax_row(r, tol=1e-6)
        if c and c not in mumax_by_case:
            mumax_by_case[c] = r
    if len(mumax_by_case) != 3:
        if len(rows) < 3:
            raise RuntimeError("Expected >=3 rows for mx/my/mz cases")
        mumax_by_case = {"mx": rows[0], "my": rows[1], "mz": rows[2]}
    return mumax_by_case


def _run_minimal_xyz(rust_csv: str, mumax_table: str, title: str) -> None:
    rust_by_case = _parse_rust_3row_csv(rust_csv, expected_filename=os.path.basename(rust_csv))
    mumax_rows = mumax_rows_as_dicts(mumax_table)
    mumax_by_case = _load_mumax_xyz_cases_from_rows(mumax_rows)

    keys = ["B_demagx", "B_demagy", "B_demagz"]

    print(f"\n[{title}] Rust : {rust_csv}")
    print(f"[{title}] MuMax: {mumax_table}")

    for label, case in [("uniform +x", "mx"), ("uniform +y", "my"), ("uniform +z", "mz")]:
        rust = rust_by_case[case]
        mumax = mumax_by_case[case]

        print(f"\n=== {label} ===")
        print(f"{'key':<10} {'rust':>16} {'mumax':>16} {'delta':>16}")
        print("-" * 60)

        for k in keys:
            if k not in rust:
                raise RuntimeError(f"Rust missing '{k}' in {rust_csv}. Found: {list(rust.keys())}")
            if k not in mumax:
                raise RuntimeError(f"MuMax missing '{k}' in {mumax_table}. Found: {list(mumax.keys())}")
            r = rust[k]
            m = mumax[k]
            d = r - m
            print(f"{k:<10} {r:>16.9e} {m:>16.9e} {d:>16.9e}")


def run_test_small(rust_dir: str, mumax_dir: str) -> None:
    rust_csv = os.path.join(rust_dir, "rust_demagSmall.csv")
    mumax_table = os.path.join(mumax_dir, "demagSmall", "table.txt")
    _run_minimal_xyz(rust_csv, mumax_table, title="demagSmall")


def run_test_2d(rust_dir: str, mumax_dir: str) -> None:
    rust_csv = os.path.join(rust_dir, "rust_demag2D.csv")
    mumax_table = os.path.join(mumax_dir, "demag2D", "table.txt")
    _run_minimal_xyz(rust_csv, mumax_table, title="demag2D")


# ----------------------------
# demag2Dpbc (two blocks: Nx=2 then Nx=3; each block has 3 rows)
# ----------------------------

def _parse_rust_demag2dpbc(path: str) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Parse out/demag2Dpbc/rust_demag2Dpbc.csv into:
      blocks[nx][case] = rowdict (floats), where case in {'mx','my','mz'}.

    Requires columns: nx, case, B_demagx, B_demagy, B_demagz (others ignored).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"No rows found in Rust PBC CSV: {path}")

    blocks: Dict[int, Dict[str, Dict[str, float]]] = {}
    for r in rows:
        nx_s = r.get("nx", "").strip()
        case = r.get("case", "").strip()
        if nx_s == "" or case == "":
            raise RuntimeError(f"Missing 'nx' or 'case' in Rust PBC CSV row: {r}")

        nx = int(nx_s)
        if case not in ("mx", "my", "mz"):
            continue

        rowf: Dict[str, float] = {}
        for k in ("B_demagx", "B_demagy", "B_demagz"):
            if k not in r or r[k] is None or r[k].strip() == "":
                raise RuntimeError(f"Missing '{k}' in Rust PBC CSV row: {r}")
            rowf[k] = float(r[k])

        blocks.setdefault(nx, {})[case] = rowf

    # Ensure required blocks exist
    for nx in (2, 3):
        if nx not in blocks:
            raise RuntimeError(f"Rust PBC CSV missing nx={nx} block: {path}")
        for case in ("mx", "my", "mz"):
            if case not in blocks[nx]:
                raise RuntimeError(f"Rust PBC CSV missing case={case} for nx={nx}: {path}")

    return blocks


def run_test_pbc(rust_dir: str, mumax_dir: str) -> None:
    rust_csv = os.path.join(rust_dir, "rust_demag2Dpbc.csv")
    mumax_table = os.path.join(mumax_dir, "demag2Dpbc", "table.txt")

    mumax_rows = mumax_rows_as_dicts(mumax_table)
    if len(mumax_rows) < 6:
        raise RuntimeError(f"Expected >=6 rows in MuMax demag2Dpbc table (Nx=2 then Nx=3), got {len(mumax_rows)}: {mumax_table}")

    # MuMax script is expected to write:
    # rows 0..2: Nx=2 cases
    # rows 3..5: Nx=3 cases
    mumax_block2 = _load_mumax_xyz_cases_from_rows(mumax_rows[0:3])
    mumax_block3 = _load_mumax_xyz_cases_from_rows(mumax_rows[3:6])

    rust_blocks = _parse_rust_demag2dpbc(rust_csv)

    keys = ["B_demagx", "B_demagy", "B_demagz"]

    print(f"\n[demag2Dpbc] Rust : {rust_csv}")
    print(f"[demag2Dpbc] MuMax: {mumax_table}")

    for nx, mumax_block in [(2, mumax_block2), (3, mumax_block3)]:
        print(f"\n--- Block nx={nx} ---")
        for label, case in [("uniform +x", "mx"), ("uniform +y", "my"), ("uniform +z", "mz")]:
            rust = rust_blocks[nx][case]
            mumax = mumax_block[case]

            print(f"\n=== {label} ===")
            print(f"{'key':<10} {'rust':>16} {'mumax':>16} {'delta':>16}")
            print("-" * 60)

            for k in keys:
                r = rust[k]
                m = mumax[k]
                d = r - m
                print(f"{k:<10} {r:>16.9e} {m:>16.9e} {d:>16.9e}")


# ----------------------------
# Main CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Compare MuMax vs Rust demag tests (snapshot, demagSmall, demag2D, demag2Dpbc)")
    ap.add_argument(
        "--test",
        choices=["snapshot", "small", "2d", "pbc"],
        default="snapshot",
        help="Which demag test to run: snapshot, small, 2d, or pbc (demag2Dpbc)",
    )
    ap.add_argument("--rust_dir", default=None, help="Rust output directory (defaults depend on --test)")
    ap.add_argument("--mumax_dir", default="mumax_outputs", help="MuMax outputs base directory")

    # Snapshot-only options
    ap.add_argument("--rel_floor", type=float, default=REL_FLOOR_DEFAULT,
                    help="(snapshot only) if |mumax| < rel_floor, suppress rel_err and report abs-only")
    ap.add_argument("--tol", type=float, default=None, help="(snapshot only) Optional relative error tolerance")
    ap.add_argument("--plot", default=None, help="(snapshot only) Optional output plot filename (png)")
    ap.add_argument("--plot_mode", choices=["abs", "rel"], default="abs",
                    help="(snapshot only) Plot abs(delta) or relative error")

    args = ap.parse_args()

    if args.test == "snapshot":
        rust_dir = args.rust_dir or "out/demag_snapshot"
        run_test_snapshot(rust_dir, args.mumax_dir, args.tol, args.plot, args.plot_mode, rel_floor=args.rel_floor)
    elif args.test == "small":
        rust_dir = args.rust_dir or "out/demagSmall"
        run_test_small(rust_dir, args.mumax_dir)
    elif args.test == "2d":
        rust_dir = args.rust_dir or "out/demag2D"
        run_test_2d(rust_dir, args.mumax_dir)
    else:  # pbc
        rust_dir = args.rust_dir or "out/demag2Dpbc"
        run_test_pbc(rust_dir, args.mumax_dir)


if __name__ == "__main__":
    main()