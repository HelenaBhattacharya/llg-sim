#!/usr/bin/env python3
"""ovf_stats.py — lightweight OVF (OOMMF OVF2) statistics + comparison tool.

Why this exists
---------------
When debugging Standard Problem 2 (SP2) / SP4 runs, plots can be misleading if:
  - the wrong OVF is being loaded,
  - components are swapped,
  - the colour scale hides the true dynamic range,
  - or the solver converges to a different metastable minimum.

This script reads OVF files directly (no discretisedfield dependency) and prints
basic, *ground-truth* stats:
  - avg/min/max for mx/my/mz
  - avg/min/max for projection onto an arbitrary direction
  - |m| consistency checks (mean/max deviation from 1)
  - optional residual metrics between two OVFs

It supports:
  - OVF2 "Data Binary 4" (float32)
  - OVF2 "Data Text"

Examples
--------
# One file
python3 scripts/ovf_stats.py runs/st_problems/sp2/ovf/d30/m_d30_hc.ovf

# Folder scan
python3 scripts/ovf_stats.py runs/st_problems/sp2/ovf/d30 --glob "*.ovf"

# Recursive scan + write CSV
python3 scripts/ovf_stats.py runs/st_problems/sp2/ovf --recursive --csv ovf_stats.csv

# Also report projection along SP2 field direction (-1,-1,-1)
python3 scripts/ovf_stats.py m_d30_hc.ovf --dir -1 -1 -1

# Compare two OVFs of identical shape
python3 scripts/ovf_stats.py a.ovf --compare b.ovf

Notes
-----
- The script assumes the OVF stores a *vector* field with valuedim=3.
- OVF ordering is assumed to be x-fastest, then y, then z (matching OOMMF/Rust writer).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------
# OVF reading
# ---------------------------


@dataclass
class OvfHeader:
    kv: Dict[str, str]

    @property
    def nx(self) -> int:
        return int(float(self.kv.get("xnodes", "1")))

    @property
    def ny(self) -> int:
        return int(float(self.kv.get("ynodes", "1")))

    @property
    def nz(self) -> int:
        return int(float(self.kv.get("znodes", "1")))

    @property
    def valuedim(self) -> int:
        return int(float(self.kv.get("valuedim", "3")))

    @property
    def desc(self) -> str:
        # There can be multiple Desc lines; keep the first if present.
        # (We store them while parsing.)
        return self.kv.get("desc", "")


def _parse_header_kv(raw: bytes) -> Tuple[OvfHeader, int]:
    """Parse OVF header key/values.

    Returns:
      (header, end_header_byte_index)

    end_header_byte_index points to the byte *after* the newline following
    '# End: Header'.
    """
    # Decode only the first chunk (header is ASCII).
    head = raw[: 256 * 1024].decode("utf-8", errors="ignore")
    end_tag = "# End: Header"
    end_pos = head.find(end_tag)
    if end_pos < 0:
        raise ValueError("OVF: missing '# End: Header'")

    # Find end-of-line after the end tag.
    end_line = head.find("\n", end_pos)
    if end_line < 0:
        raise ValueError("OVF: malformed header (no newline after '# End: Header')")

    header_text = head[:end_line]

    kv: Dict[str, str] = {}
    desc_lines: List[str] = []

    for line in header_text.splitlines():
        if not line.startswith("#"):
            continue
        s = line[1:].strip()
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k == "desc":
            desc_lines.append(v)
        # Keep last occurrence for normal keys.
        kv[k] = v

    if desc_lines:
        kv["desc"] = desc_lines[0]
        kv["desc_all"] = " | ".join(desc_lines)

    # Convert from character index -> byte index.
    # This is safe because we only searched in ASCII-ish header region.
    end_header_bytes = raw.find(b"# End: Header")
    end_header_bytes = raw.find(b"\n", end_header_bytes) + 1
    return OvfHeader(kv=kv), end_header_bytes


def _find_marker(raw: bytes, marker: bytes, start: int = 0) -> int:
    idx = raw.find(marker, start)
    if idx < 0:
        raise ValueError(f"OVF: missing marker {marker!r}")
    return idx


def _read_ovf2_binary4(raw: bytes, hdr: OvfHeader) -> np.ndarray:
    marker = b"# Begin: Data Binary 4"
    idx = _find_marker(raw, marker)
    start = raw.find(b"\n", idx) + 1

    # Data ends right before '# End: Data' line.
    end_idx = raw.find(b"# End: Data", start)
    if end_idx < 0:
        # Some writers omit it; fall back to end-of-file.
        end_idx = len(raw)

    data = raw[start:end_idx]

    # Many writers add a newline after binary block. Strip it.
    if data.endswith(b"\n"):
        data = data[:-1]

    if len(data) % 4 != 0:
        raise ValueError(f"OVF: binary4 payload length not divisible by 4 (len={len(data)})")

    floats = np.frombuffer(data, dtype="<f4")
    if floats.size == 0:
        raise ValueError("OVF: empty binary block")

    if not np.isfinite(float(floats[0])) or abs(float(floats[0]) - 1234567.0) > 1e-3:
        # Try big-endian.
        floats = np.frombuffer(data, dtype=">f4")
        if floats.size == 0 or abs(float(floats[0]) - 1234567.0) > 1e-3:
            raise ValueError(
                "OVF: endianness check failed (expected first float to be 1234567.0)"
            )

    vals = floats[1:]  # skip check value

    nx, ny, nz, vd = hdr.nx, hdr.ny, hdr.nz, hdr.valuedim
    expected = nx * ny * nz * vd
    if vals.size != expected:
        raise ValueError(f"OVF: unexpected data size {vals.size}, expected {expected}")

    # OVF ordering: x fastest, then y, then z.
    # Stored as (z, y, x, vd) in a flat stream.
    arr_zyx = vals.reshape((nz, ny, nx, vd))
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0, 3))
    return arr_xyz


def _read_ovf2_text(raw: bytes, hdr: OvfHeader) -> np.ndarray:
    marker = b"# Begin: Data Text"
    idx = _find_marker(raw, marker)

    # Read as text from marker onwards.
    tail = raw[idx:].decode("utf-8", errors="ignore")
    lines = tail.splitlines()

    # Find first data line after the Begin marker.
    start_i = None
    for i, line in enumerate(lines):
        if line.strip().startswith("# Begin: Data Text"):
            start_i = i + 1
            break
    if start_i is None:
        raise ValueError("OVF: could not locate text data start")

    # Collect until '# End: Data'
    data_vals: List[float] = []
    for line in lines[start_i:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("# End: Data"):
            break
        if s.startswith("#"):
            continue
        parts = s.split()
        for p in parts:
            try:
                data_vals.append(float(p))
            except ValueError:
                raise ValueError(f"OVF: failed to parse float in line: {line!r}")

    vals = np.asarray(data_vals, dtype=np.float32)
    nx, ny, nz, vd = hdr.nx, hdr.ny, hdr.nz, hdr.valuedim
    expected = nx * ny * nz * vd
    if vals.size != expected:
        raise ValueError(f"OVF: unexpected text data size {vals.size}, expected {expected}")

    arr_zyx = vals.reshape((nz, ny, nx, vd))
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0, 3))
    return arr_xyz


def read_ovf(path: Path) -> Tuple[np.ndarray, OvfHeader, str]:
    """Read an OVF file.

    Returns:
      (arr_xyz, header, mode)

    where arr_xyz has shape (nx, ny, nz, 3).
    """
    raw = path.read_bytes()
    hdr, _ = _parse_header_kv(raw)

    if hdr.valuedim != 3:
        raise ValueError(f"OVF: valuedim={hdr.valuedim} (expected 3)")

    if b"# Begin: Data Binary 4" in raw:
        arr = _read_ovf2_binary4(raw, hdr)
        return arr, hdr, "binary4"

    if b"# Begin: Data Text" in raw:
        arr = _read_ovf2_text(raw, hdr)
        return arr, hdr, "text"

    raise ValueError("OVF: unsupported data section (expected Binary 4 or Text)")


# ---------------------------
# Stats
# ---------------------------


@dataclass
class Stats:
    path: str
    nx: int
    ny: int
    nz: int
    mx_avg: float
    my_avg: float
    mz_avg: float
    msum_avg: float
    mx_min: float
    mx_max: float
    my_min: float
    my_max: float
    mz_min: float
    mz_max: float
    mag_mean: float
    mag_min: float
    mag_max: float
    mag_max_dev_from_1: float
    proj_dir: Optional[Tuple[float, float, float]] = None
    proj_avg: Optional[float] = None
    proj_min: Optional[float] = None
    proj_max: Optional[float] = None
    proj_perp_avg: Optional[float] = None


def _finite_minmax(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.min(x)), float(np.max(x)))


def compute_stats(arr_xyz: np.ndarray, path: Path, proj_dir: Optional[Sequence[float]] = None) -> Stats:
    if arr_xyz.ndim != 4 or arr_xyz.shape[-1] != 3:
        raise ValueError(f"Unexpected array shape {arr_xyz.shape}, expected (nx,ny,nz,3)")

    mx = arr_xyz[..., 0].astype(np.float64)
    my = arr_xyz[..., 1].astype(np.float64)
    mz = arr_xyz[..., 2].astype(np.float64)

    mx_avg = float(mx.mean())
    my_avg = float(my.mean())
    mz_avg = float(mz.mean())
    msum_avg = mx_avg + my_avg + mz_avg

    mx_min, mx_max = _finite_minmax(mx)
    my_min, my_max = _finite_minmax(my)
    mz_min, mz_max = _finite_minmax(mz)

    mag = np.sqrt(mx * mx + my * my + mz * mz)
    mag_mean = float(mag.mean())
    mag_min, mag_max = _finite_minmax(mag)
    mag_max_dev = float(np.max(np.abs(mag - 1.0)))

    s = Stats(
        path=str(path),
        nx=int(arr_xyz.shape[0]),
        ny=int(arr_xyz.shape[1]),
        nz=int(arr_xyz.shape[2]),
        mx_avg=mx_avg,
        my_avg=my_avg,
        mz_avg=mz_avg,
        msum_avg=msum_avg,
        mx_min=mx_min,
        mx_max=mx_max,
        my_min=my_min,
        my_max=my_max,
        mz_min=mz_min,
        mz_max=mz_max,
        mag_mean=mag_mean,
        mag_min=mag_min,
        mag_max=mag_max,
        mag_max_dev_from_1=mag_max_dev,
    )

    if proj_dir is not None:
        dx, dy, dz = [float(v) for v in proj_dir]
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm <= 0:
            raise ValueError("Projection direction must be non-zero")
        ux, uy, uz = dx / norm, dy / norm, dz / norm
        proj = mx * ux + my * uy + mz * uz
        proj_min, proj_max = _finite_minmax(proj)
        proj_avg = float(proj.mean())
        # magnitude perpendicular to direction: |m - (m·u)u|
        perp = np.sqrt(np.maximum(0.0, mag * mag - proj * proj))
        perp_avg = float(perp.mean())

        s.proj_dir = (ux, uy, uz)
        s.proj_avg = proj_avg
        s.proj_min = proj_min
        s.proj_max = proj_max
        s.proj_perp_avg = perp_avg

    return s


def compute_residual(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Residual metrics for two vector fields of identical shape."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    da = a.astype(np.float64) - b.astype(np.float64)
    dmx = da[..., 0]
    dmy = da[..., 1]
    dmz = da[..., 2]
    dmag = np.sqrt(dmx * dmx + dmy * dmy + dmz * dmz)
    return {
        "dmx_mean": float(dmx.mean()),
        "dmy_mean": float(dmy.mean()),
        "dmz_mean": float(dmz.mean()),
        "dmag_mean": float(dmag.mean()),
        "dmag_max": float(np.max(dmag)),
        "dmag_p99": float(np.quantile(dmag.reshape(-1), 0.99)),
    }


# ---------------------------
# CLI
# ---------------------------


def iter_ovf_files(inputs: Sequence[str], glob_pat: str, recursive: bool) -> List[Path]:
    out: List[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_file():
            out.append(p)
            continue
        if p.is_dir():
            it = p.rglob(glob_pat) if recursive else p.glob(glob_pat)
            out.extend([q for q in it if q.is_file()])
            continue
        raise FileNotFoundError(f"No such file or directory: {p}")

    # Stable order
    out = sorted(set(out), key=lambda x: str(x))
    return out


def print_stats_table(stats: List[Stats], show_proj: bool) -> None:
    if not stats:
        print("No OVF files found.")
        return

    # Header
    cols = [
        "file",
        "nx",
        "ny",
        "nz",
        "<mx>",
        "<my>",
        "<mz>",
        "<msum>",
        "mx[min,max]",
        "my[min,max]",
        "mz[min,max]",
        "|m|mean",
        "|m|maxdev",
    ]
    if show_proj:
        cols += ["<m·u>", "proj[min,max]", "<|m⊥|>"]

    # Print one per line (wide tables in terminals are painful)
    for st in stats:
        base = (
            f"{st.path} | "
            f"{st.nx}x{st.ny}x{st.nz} | "
            f"<m>=({st.mx_avg:+.6f}, {st.my_avg:+.6f}, {st.mz_avg:+.6f}) | "
            f"msum={st.msum_avg:+.6f} | "
            f"mx=[{st.mx_min:+.6f},{st.mx_max:+.6f}] "
            f"my=[{st.my_min:+.6f},{st.my_max:+.6f}] "
            f"mz=[{st.mz_min:+.6f},{st.mz_max:+.6f}] | "
            f"|m|mean={st.mag_mean:.6f} maxdev={st.mag_max_dev_from_1:.3e}"
        )

        if show_proj and st.proj_avg is not None:
            base += (
                f" | <m·u>={st.proj_avg:+.6f} "
                f"proj=[{st.proj_min:+.6f},{st.proj_max:+.6f}] "
                f"<|m⊥|>={st.proj_perp_avg:.6f}"
            )

        print(base)


def write_csv(stats: List[Stats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "nx",
        "ny",
        "nz",
        "mx_avg",
        "my_avg",
        "mz_avg",
        "msum_avg",
        "mx_min",
        "mx_max",
        "my_min",
        "my_max",
        "mz_min",
        "mz_max",
        "mag_mean",
        "mag_min",
        "mag_max",
        "mag_max_dev_from_1",
        "proj_ux",
        "proj_uy",
        "proj_uz",
        "proj_avg",
        "proj_min",
        "proj_max",
        "proj_perp_avg",
    ]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for st in stats:
            row = {
                "path": st.path,
                "nx": st.nx,
                "ny": st.ny,
                "nz": st.nz,
                "mx_avg": st.mx_avg,
                "my_avg": st.my_avg,
                "mz_avg": st.mz_avg,
                "msum_avg": st.msum_avg,
                "mx_min": st.mx_min,
                "mx_max": st.mx_max,
                "my_min": st.my_min,
                "my_max": st.my_max,
                "mz_min": st.mz_min,
                "mz_max": st.mz_max,
                "mag_mean": st.mag_mean,
                "mag_min": st.mag_min,
                "mag_max": st.mag_max,
                "mag_max_dev_from_1": st.mag_max_dev_from_1,
                "proj_ux": st.proj_dir[0] if st.proj_dir else "",
                "proj_uy": st.proj_dir[1] if st.proj_dir else "",
                "proj_uz": st.proj_dir[2] if st.proj_dir else "",
                "proj_avg": st.proj_avg if st.proj_avg is not None else "",
                "proj_min": st.proj_min if st.proj_min is not None else "",
                "proj_max": st.proj_max if st.proj_max is not None else "",
                "proj_perp_avg": st.proj_perp_avg if st.proj_perp_avg is not None else "",
            }
            w.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute basic stats for OVF files (OVF2 binary4/text).")
    ap.add_argument("inputs", nargs="+", help="OVF files or directories")
    ap.add_argument("--glob", default="*.ovf", help="Glob used when an input is a directory (default: *.ovf)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders when scanning a directory")
    ap.add_argument(
        "--dir",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        help="Also compute projection m·u onto this direction vector (will be normalised)",
    )
    ap.add_argument("--csv", type=str, default=None, help="Write per-file stats to CSV")
    ap.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Compare FIRST resolved OVF against this OVF (prints residual metrics)",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    files = iter_ovf_files(args.inputs, args.glob, args.recursive)
    if not files:
        print("No OVF files found.")
        return 2

    proj_dir = args.dir

    stats: List[Stats] = []
    for p in files:
        try:
            arr, hdr, mode = read_ovf(p)
            st = compute_stats(arr, p, proj_dir=proj_dir)
            stats.append(st)
        except Exception as e:
            print(f"ERROR: {p}: {e}", file=sys.stderr)

    show_proj = proj_dir is not None
    print_stats_table(stats, show_proj=show_proj)

    if args.csv:
        write_csv(stats, Path(args.csv))
        print(f"\nWrote CSV: {args.csv}")

    if args.compare:
        a_path = files[0]
        b_path = Path(args.compare)
        print("\nResidual comparison")
        print(f"  A: {a_path}")
        print(f"  B: {b_path}")
        a_arr, _, _ = read_ovf(a_path)
        b_arr, _, _ = read_ovf(b_path)
        res = compute_residual(a_arr, b_arr)
        for k, v in res.items():
            print(f"  {k}: {v:+.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
