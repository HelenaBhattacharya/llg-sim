"""ovf_utils.py — helpers for OVF discovery + loading for llg-sim visualisation scripts.

This module is shared by:
  - scripts/mag_viewer.py        (interactive PyVista viewer)
  - scripts/mag_visualisation.py (batch PNG/movie exporter)

Scope (for now):
  - Standard Problem 2 (SP2)
  - Standard Problem 4 (SP4)

Design goals:
  - Be tolerant of both MuMax-style and Rust-style directory layouts.
  - Be tolerant of older Rust OVFs missing xmin/xmax/... fields.
  - Keep problem-specific logic in one place (series discovery), so the viewer/exporter
    can stay clean.

Notes:
  - We use `discretisedfield` to read OVF (text or binary4).
  - We intentionally do NOT import PyVista here, so batch export can run without PyVista.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import discretisedfield as df
import numpy as np


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OvfSeries:
    """A discoverable ordered set of OVF frames."""

    series_id: str
    problem: str  # "sp2" | "sp4"
    source: str  # "rust" | "mumax" | "custom"
    kind: str  # e.g. "sp4a", "sp2_sweep", "sp2_d30_hc"
    stage: Optional[str] = None  # "rem" | "hc" | None
    d_lex: Optional[int] = None
    tag: Optional[str] = None
    frames: List[Path] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# OVF header helpers
# -----------------------------------------------------------------------------


# Time stamps:
# - Rust OVFs include: "Total simulation time: <t> s"
# - MuMax3 OVFs typically include: "Time: <t> s" (or similar)
_TIME_RE_RUST = re.compile(r"Total simulation time:\\s*([+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\s*s")
_TIME_RE_MUMAX = re.compile(r"\\bTime:\\s*([+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\s*s\\b")
_TIME_RE_MUMAX2 = re.compile(r"\\bTime\\s*\\(s\\)\\s*:\\s*([+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\b")
_TIME_RE_GENERIC = re.compile(r"\\bt\\s*=\\s*([+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\s*s\\b")
_TIME_RES = (_TIME_RE_RUST, _TIME_RE_MUMAX, _TIME_RE_MUMAX2, _TIME_RE_GENERIC)


def try_parse_time_seconds_from_ovf(path: Path) -> Optional[float]:
    """Best-effort parse of a physical time stamp from an OVF header.

    Supported patterns:
      * Rust:   "Total simulation time: <t> s"
      * MuMax3: "Time: <t> s" (or variants)
    """
    try:
        raw = path.read_bytes()[:64 * 1024]
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return None

    for rx in _TIME_RES:
        m = rx.search(text)
        if not m:
            continue
        try:
            return float(m.group(1))
        except Exception:
            continue
    return None


def try_parse_time_ns_from_ovf(path: Path) -> Optional[float]:
    t_s = try_parse_time_seconds_from_ovf(path)
    if t_s is None:
        return None
    return t_s * 1e9


# -----------------------------------------------------------------------------
# OVF compatibility (older Rust OVFs may omit xmin/xmax/ymin/ymax/zmin/zmax)
# -----------------------------------------------------------------------------


def _ovf_header_kv(text: str) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        s = line[1:].strip()
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        kv[k.strip().lower()] = v.strip()
    return kv


def _patch_ovf_add_minmax(raw: bytes) -> bytes:
    """Patch OVF header to include xmin/xmax/ymin/ymax/zmin/zmax if missing.

    discretisedfield expects these, but some older Rust outputs didn't include them.
    We can synthesise them from xbase/xstepsize/xnodes, etc.
    """
    text = raw.decode("utf-8", errors="ignore")

    hdr_start = text.find("# Begin: Header")
    hdr_end = text.find("# End: Header")
    if hdr_start == -1 or hdr_end == -1 or hdr_end <= hdr_start:
        return raw

    header_text = text[hdr_start:hdr_end]
    kv = _ovf_header_kv(header_text)

    if all(k in kv for k in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")):
        return raw

    def get_f(key: str, default: float = 0.0) -> float:
        try:
            return float(kv.get(key, default))
        except Exception:
            return default

    def get_i(key: str, default: int = 1) -> int:
        try:
            return int(float(kv.get(key, default)))
        except Exception:
            return default

    xbase = get_f("xbase", 0.0)
    ybase = get_f("ybase", 0.0)
    zbase = get_f("zbase", 0.0)

    dx = get_f("xstepsize", 0.0)
    dy = get_f("ystepsize", 0.0)
    dz = get_f("zstepsize", 0.0)

    nx = get_i("xnodes", 1)
    ny = get_i("ynodes", 1)
    nz = get_i("znodes", 1)

    # In OVF, *base* is the cell-center coordinate for index 0.
    # Min/max refer to mesh bounds (cell edges), so subtract/add half a cell.
    xmin = xbase - 0.5 * dx
    ymin = ybase - 0.5 * dy
    zmin = zbase - 0.5 * dz
    xmax = xmin + nx * dx
    ymax = ymin + ny * dy
    zmax = zmin + nz * dz

    insert = (
        f"# xmin: {xmin}\n"
        f"# ymin: {ymin}\n"
        f"# zmin: {zmin}\n"
        f"# xmax: {xmax}\n"
        f"# ymax: {ymax}\n"
        f"# zmax: {zmax}\n"
    )

    patched = text[:hdr_end] + insert + text[hdr_end:]
    return patched.encode("utf-8")


def load_field_from_ovf_compat(path: Path) -> df.Field:
    """Load OVF via discretisedfield, patching the header if required."""
    try:
        return df.Field.from_file(str(path))
    except KeyError as e:
        key = str(e).strip("'")
        if key not in {"xmin", "ymin", "zmin", "xmax", "ymax", "zmax"}:
            raise

        raw = path.read_bytes()
        patched = _patch_ovf_add_minmax(raw)

        tmp = tempfile.NamedTemporaryFile(prefix="ovf_patch_", suffix=".ovf", delete=False)
        try:
            tmp.write(patched)
            tmp.flush()
            tmp.close()
            return df.Field.from_file(tmp.name)
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except Exception:
                pass


def load_slice_field(path: Path) -> df.Field:
    """Load OVF and return a 2D-ish slice (z plane for thin films)."""
    field = load_field_from_ovf_compat(path)
    edges = field.mesh.region.edges

    # Thin-film heuristic: if z extent is much smaller than x extent, take z slice
    if edges[2] < edges[0] / 10:
        return field.sel("z")

    # Otherwise, take middle slice
    z_center = field.mesh.region.center[2]
    return field.sel(z=z_center)


def field_vector_array_2d(m: df.Field) -> np.ndarray:
    """Return vector array shaped (nx, ny, 3) for a 2D slice Field."""
    arr: Optional[np.ndarray] = None

    if hasattr(m, "array"):
        try:
            arr = np.asarray(m.array)
        except Exception:
            arr = None

    if arr is None:
        # discretisedfield fallback
        arr = np.asarray(getattr(m, "asarray")())

    # Possible shapes: (nx, ny, 3) or (nx, ny, 1, 3)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        arr = arr[:, :, 0, :]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected vector array shape: {arr.shape}")

    return arr


# -----------------------------------------------------------------------------
# Source/problem inference
# -----------------------------------------------------------------------------


def infer_source(input_path: Path) -> str:
    """Infer whether the input data is from MuMax or Rust."""
    name = input_path.name.lower()
    s = str(input_path).lower()

    if name.endswith("_out") or "mumax" in s:
        return "mumax"
    if name.endswith("_rust") or ("runs" in s and "mumax" not in s):
        return "rust"

    # Common subfolders
    if (input_path / "sp4a_out").exists() or (input_path / "sp4b_out").exists():
        return "mumax"
    if (input_path / "sp4a_rust").exists() or (input_path / "sp4b_rust").exists():
        return "rust"

    return "custom"


def detect_problem(input_path: Path) -> str:
    """Best-effort problem detection (sp2 vs sp4)."""
    s = str(input_path).lower()
    if "sp2" in s:
        return "sp2"
    if "sp4" in s:
        return "sp4"

    # Look for SP4 case subfolders
    if (input_path / "sp4a_out").exists() or (input_path / "sp4b_out").exists() or (input_path / "sp4a_rust").exists() or (input_path / "sp4b_rust").exists():
        return "sp4"

    # Look for SP2 ovf layout
    sp2_root = find_sp2_ovf_root(input_path)
    if sp2_root is not None:
        # at least one dXX folder or m_d?? file exists
        if any(p.is_dir() and _SP2_DDIR_RE.match(p.name) for p in sp2_root.iterdir()):
            return "sp2"
        if list(sp2_root.glob("m_d*_rem.ovf")) or list(sp2_root.glob("m_d*_hc.ovf")):
            return "sp2"

    # Fall back: if any directory contains m*.ovf assume SP4-like
    if list(input_path.glob("m*.ovf")):
        return "sp4"

    return "sp4"  # safest default


# -----------------------------------------------------------------------------
# Sorting / naming helpers
# -----------------------------------------------------------------------------


_M_RE = re.compile(r"^m(\d+)\.ovf$")
_F_RE = re.compile(r"_f(\d+)\.ovf$")
_SP2_LEGACY_TAG_RE = re.compile(r"^(?P<tag>.+?)_t_.*_f\d+\.ovf$")
_SP2_TLABEL_RE = re.compile(r"_t_(?P<t>[^_]+)_f\d+\.ovf$")
# Legacy SP2 frame naming: <tag>_t_<time>_f<frame>.ovf
# Used as a *search* filter when discovering evolution frames.
_SP2_TAG_TIME_RE = re.compile(r"_t_[^_]+_f\d+\.ovf$")
_SP2_D_FROM_FINAL_RE = re.compile(r"^m_d(\d+)_")


def _sort_key_ovf(p: Path) -> Tuple[int, int, str]:
    name = p.name
    m = _M_RE.match(name)
    if m:
        return (0, int(m.group(1)), name)
    m = _F_RE.search(name)
    if m:
        return (1, int(m.group(1)), name)
    return (2, 0, name)


def sort_ovf_frames(paths: Iterable[Path]) -> List[Path]:
    return sorted(list(paths), key=_sort_key_ovf)


def try_parse_sp2_time_label_seconds_from_name(path: Path) -> Optional[float]:
    """Parse SP2 legacy filename segment `_t_..._fNNNNNN.ovf`.

    Returns time in seconds if it looks like:
      - 000010ns  -> 10e-9
      - 1p000000e-09s (legacy fallback) -> float("1.000000e-09")
    """
    m = _SP2_TLABEL_RE.search(path.name)
    if not m:
        return None
    tl = m.group("t")

    # 000010ns
    if tl.endswith("ns"):
        try:
            ns = float(tl[:-2])
            return ns * 1e-9
        except Exception:
            return None

    # fallback: Rust replaced '.' with 'p' for scientific notation labels
    if tl.endswith("s"):
        t = tl[:-1].replace("p", ".")
        try:
            return float(t)
        except Exception:
            return None

    return None


def try_parse_sp2_tag_from_name(path: Path) -> Optional[str]:
    m = _SP2_LEGACY_TAG_RE.match(path.name)
    if m:
        return m.group("tag")
    return None


# -----------------------------------------------------------------------------
# SP4 discovery
# -----------------------------------------------------------------------------


def discover_sp4_series(input_path: Path, source: str = "auto") -> List[OvfSeries]:
    """Discover SP4 case folders and return one series per case."""
    if source == "auto":
        source = infer_source(input_path)

    # If input is already a case folder
    if list(input_path.glob("m*.ovf")):
        case = _case_from_dirname(input_path.name)
        frames = sort_ovf_frames(input_path.glob("m*.ovf"))
        return [
            OvfSeries(
                series_id=f"sp4:{case}:{source}",
                problem="sp4",
                source=source,
                kind=case,
                frames=frames,
            )
        ]

    # Otherwise treat as root with case subfolders
    cases: Dict[str, Path] = {}
    for sub in ["sp4a_out", "sp4b_out", "sp4a_rust", "sp4b_rust"]:
        d = input_path / sub
        if d.exists() and list(d.glob("m*.ovf")):
            cases[_case_from_dirname(d.name)] = d

    if not cases:
        # Fall back: any child folder with m*.ovf
        for d in sorted([p for p in input_path.iterdir() if p.is_dir()]):
            if list(d.glob("m*.ovf")):
                cases[_case_from_dirname(d.name)] = d

    series: List[OvfSeries] = []
    for case, d in cases.items():
        frames = sort_ovf_frames(d.glob("m*.ovf"))
        if not frames:
            continue
        series.append(
            OvfSeries(
                series_id=f"sp4:{case}:{source}",
                problem="sp4",
                source=source,
                kind=case,
                frames=frames,
                meta={"case_dir": str(d)},
            )
        )

    # Stable ordering (sp4a then sp4b)
    series.sort(key=lambda s: s.kind)
    return series


def _case_from_dirname(dirname: str) -> str:
    d = dirname.lower()
    if d.startswith("sp4a"):
        return "sp4a"
    if d.startswith("sp4b"):
        return "sp4b"
    return dirname


# -----------------------------------------------------------------------------
# SP2 discovery
# -----------------------------------------------------------------------------


_SP2_DDIR_RE = re.compile(r"^d(\d{1,3})$")


def find_sp2_ovf_root(input_path: Path) -> Optional[Path]:
    """Try to find the SP2 ovf root folder.

    Accepts any of:
      - runs/st_problems/sp2
      - runs/st_problems/sp2/ovf
      - runs/st_problems/sp2/ovf/d30
      - .../ovf/d30/hc
    """
    p = input_path

    # If user points at '.../ovf', use it
    if p.name.lower() == "ovf" and p.exists():
        return p

    # If user points at a dXX folder, use its parent
    if _SP2_DDIR_RE.match(p.name.lower()):
        return p.parent

    # If user points at rem/hc folder, dXX is its parent, root is parent of dXX
    if p.name.lower() in {"rem", "hc"} and _SP2_DDIR_RE.match(p.parent.name.lower()):
        return p.parent.parent

    # If input contains an 'ovf' folder, use that
    if (p / "ovf").exists() and (p / "ovf").is_dir():
        return p / "ovf"

    # Otherwise, maybe input is already the ovf root
    if any(child.is_dir() and _SP2_DDIR_RE.match(child.name) for child in p.iterdir()):
        return p

    # Or maybe it's a folder containing m_dXX files directly
    if list(p.glob("m_d*_rem.ovf")) or list(p.glob("m_d*_hc.ovf")):
        return p

    return None


def discover_sp2_series(input_path: Path, source: str = "auto") -> List[OvfSeries]:
    """Discover SP2 series.

    Returns:
      - one sweep series per stage (rem/hc) if final files exist
      - per-d evolution series for rem
      - per-d evolution series for hc (possibly multiple tags)
    """
    if source == "auto":
        source = infer_source(input_path)

    root = find_sp2_ovf_root(input_path)
    if root is None:
        return []

    d_dirs = _list_sp2_d_dirs(root)

    series: List[OvfSeries] = []

    # Sweep series (final states)
    sweep_rem: List[Tuple[int, Path]] = []
    sweep_hc: List[Tuple[int, Path]] = []

    for d, d_dir in d_dirs:
        finals = _find_sp2_finals_in_d_dir(d_dir)
        if "rem" in finals:
            sweep_rem.append((d, finals["rem"]))
        if "hc" in finals:
            sweep_hc.append((d, finals["hc"]))

        # Evolution: rem
        rem_dir = d_dir / "rem"
        if rem_dir.exists() and rem_dir.is_dir():
            rem_files = list(rem_dir.glob("*.ovf"))
            if rem_files:
                # If MuMax-style m0000000.ovf frames are present, ignore non-frame OVFs
                # (e.g. a final state saved into the same folder).
                if any(_M_RE.match(p.name) for p in rem_files):
                    rem_files = [p for p in rem_files if _M_RE.match(p.name)]
                elif any(_SP2_TAG_TIME_RE.search(p.name) for p in rem_files):
                    rem_files = [p for p in rem_files if _SP2_TAG_TIME_RE.search(p.name)]
                frames = sort_ovf_frames(rem_files)

                final_rem = finals.get("rem")
                if final_rem is not None and final_rem not in frames:
                    frames = list(frames) + [final_rem]

                meta = {"d_dir": str(d_dir)}
                if final_rem is not None:
                    meta["final_frame"] = str(final_rem)

                series.append(
                    OvfSeries(
                        series_id=f"sp2:d{d:02d}:rem:{source}",
                        problem="sp2",
                        source=source,
                        kind=f"d{d:02d}_rem_evolution",
                        stage="rem",
                        d_lex=d,
                        tag="rem",
                        frames=frames,
                        meta=meta,
                    )
                )

        # Evolution: hc
        hc_dir = d_dir / "hc"
        if hc_dir.exists() and hc_dir.is_dir():
            hc_series = _discover_sp2_hc_series_in_hc_dir(d, hc_dir, source)

            seed_low = finals.get("seed_low")
            seed_high = finals.get("seed_high")
            final_hc_pos = finals.get("hc_pos")
            final_hc_best = finals.get("hc_best")
            final_hc = finals.get("hc")

            if hc_series:
                hc_series2: List[OvfSeries] = []
                for s in hc_series:
                    frames_new = list(s.frames)
                    # Append diagnostics at end in order: seed_low, seed_high, pos, best, final
                    for fp in (seed_low, seed_high, final_hc_pos, final_hc_best, final_hc):
                        if fp is not None and fp not in frames_new:
                            frames_new.append(fp)

                    meta = dict(s.meta)
                    if final_hc is not None:
                        meta["final_frame_hc"] = str(final_hc)
                    if seed_low is not None:
                        meta["final_seed_low"] = str(seed_low)
                    if seed_high is not None:
                        meta["final_seed_high"] = str(seed_high)
                    if final_hc_pos is not None:
                        meta["final_frame_hc_pos"] = str(final_hc_pos)
                    if final_hc_best is not None:
                        meta["final_frame_hc_best"] = str(final_hc_best)

                    hc_series2.append(
                        OvfSeries(
                            series_id=s.series_id,
                            problem=s.problem,
                            source=s.source,
                            kind=s.kind,
                            stage=s.stage,
                            d_lex=s.d_lex,
                            tag=s.tag,
                            frames=frames_new,
                            meta=meta,
                        )
                    )
                hc_series = hc_series2

            series.extend(hc_series)

    # Build sweeps (sorted by d desc)
    if sweep_rem:
        sweep_rem.sort(key=lambda x: x[0], reverse=True)
        series.append(
            OvfSeries(
                series_id=f"sp2:sweep:rem:{source}",
                problem="sp2",
                source=source,
                kind="sweep_rem",
                stage="rem",
                frames=[p for (_d, p) in sweep_rem],
                meta={"d_values": [d for (d, _p) in sweep_rem]},
            )
        )

    if sweep_hc:
        sweep_hc.sort(key=lambda x: x[0], reverse=True)
        series.append(
            OvfSeries(
                series_id=f"sp2:sweep:hc:{source}",
                problem="sp2",
                source=source,
                kind="sweep_hc",
                stage="hc",
                frames=[p for (_d, p) in sweep_hc],
                meta={"d_values": [d for (d, _p) in sweep_hc]},
            )
        )

    # Provide stable ordering: sweeps first, then evolutions by d desc
    def order_key(s: OvfSeries) -> Tuple[int, int, str]:
        if s.kind.startswith("sweep"):
            return (0, 0, s.kind)
        if s.d_lex is not None:
            return (1, -int(s.d_lex), s.kind)
        return (2, 0, s.kind)

    series.sort(key=order_key)
    return series


def _list_sp2_d_dirs(root: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for ddir in sorted([p for p in root.iterdir() if p.is_dir()]):
        m = _SP2_DDIR_RE.match(ddir.name.lower())
        if not m:
            continue
        d = int(m.group(1))
        out.append((d, ddir))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def _find_sp2_finals_in_d_dir(d_dir: Path) -> Dict[str, Path]:
    finals: Dict[str, Path] = {}
    for p in d_dir.glob("m_d*_*.ovf"):
        name = p.name
        if name.endswith("_rem.ovf"):
            finals["rem"] = p
        elif name.endswith("_hc_pos.ovf"):
            finals["hc_pos"] = p
        elif name.endswith("_hc_best.ovf"):
            finals["hc_best"] = p
        elif name.endswith("_hc.ovf"):
            finals["hc"] = p
    # MuMax bracket seeds (stored at d_dir root)
    seed_low = d_dir / "_seed_low.ovf"
    seed_high = d_dir / "_seed_high.ovf"
    if seed_low.exists():
        finals["seed_low"] = seed_low
    if seed_high.exists():
        finals["seed_high"] = seed_high
    return finals


def _discover_sp2_hc_series_in_hc_dir(d: int, hc_dir: Path, source: str) -> List[OvfSeries]:
    """Discover coercivity-evolution series inside ovf/dXX/hc/.

    Handles both:
      - legacy: many `*_t_..._fNNNNNN.ovf` in the same folder
      - mumax naming: subfolders per tag, with m0000000.ovf naming
    """
    series: List[OvfSeries] = []

    # If subfolders exist, treat each as separate series
    subdirs = [p for p in hc_dir.iterdir() if p.is_dir()]
    if subdirs:
        for sd in sorted(subdirs):
            frames = sort_ovf_frames(sd.glob("*.ovf"))
            if not frames:
                continue
            tag = sd.name
            series.append(
                OvfSeries(
                    series_id=f"sp2:d{d:02d}:hc:{tag}:{source}",
                    problem="sp2",
                    source=source,
                    kind=f"d{d:02d}_hc_evolution_{tag}",
                    stage="hc",
                    d_lex=d,
                    tag=tag,
                    frames=frames,
                    meta={"hc_dir": str(hc_dir), "tag_dir": str(sd)},
                )
            )
        return series

    # Otherwise group files in hc_dir by tag prefix before `_t_` if present
    files = list(hc_dir.glob("*.ovf"))
    if not files:
        return []

    # If MuMax-style frames exist directly in hc_dir, keep only those frames.
    if any(_M_RE.match(p.name) for p in files):
        files = [p for p in files if _M_RE.match(p.name)]

    groups: Dict[str, List[Path]] = {}
    for p in files:
        # If MuMax naming appears directly in hc_dir
        if _M_RE.match(p.name):
            groups.setdefault("hc", []).append(p)
            continue

        tag = try_parse_sp2_tag_from_name(p) or "hc"
        groups.setdefault(tag, []).append(p)

    for tag, paths in sorted(groups.items()):
        frames = sort_ovf_frames(paths)
        if not frames:
            continue
        series.append(
            OvfSeries(
                series_id=f"sp2:d{d:02d}:hc:{tag}:{source}",
                problem="sp2",
                source=source,
                kind=f"d{d:02d}_hc_evolution_{tag}",
                stage="hc",
                d_lex=d,
                tag=tag,
                frames=frames,
                meta={"hc_dir": str(hc_dir)},
            )
        )

    return series


# -----------------------------------------------------------------------------
# Convenience selectors (used by scripts)
# -----------------------------------------------------------------------------


def pick_sp2_sweep_series(series: List[OvfSeries], stage: str) -> Optional[OvfSeries]:
    stage = stage.lower().strip()
    want = "sweep_rem" if stage in {"rem", "remanence"} else "sweep_hc"
    for s in series:
        if s.problem == "sp2" and s.kind == want:
            return s
    return None


def pick_sp2_evolution_series(
    series: List[OvfSeries], d_lex: int, stage: str, tag: Optional[str] = None
) -> Optional[OvfSeries]:
    stage = stage.lower().strip()
    stage_norm = "rem" if stage in {"rem", "remanence"} else "hc"

    cands = [
        s
        for s in series
        if s.problem == "sp2" and s.d_lex == d_lex and s.stage == stage_norm and len(s.frames) > 0
    ]
    if not cands:
        return None

    if stage_norm == "rem":
        # only one expected
        return cands[0]

    # hc: choose tag if provided
    if tag:
        for s in cands:
            if s.tag == tag or s.kind.endswith(f"_{tag}"):
                return s

    # hc with no tag: return ALL frames across tags, time-sorted.
    # This is useful for diagnostics (bc0_strict, anchor_strict, bisect_strict, bracket_coarse, etc.).

    # Concatenate all frames across candidates.
    all_frames: List[Path] = []
    for s in cands:
        all_frames.extend(list(s.frames))

    # De-duplicate (by path) while preserving first occurrence.
    seen: set[str] = set()
    uniq_frames: List[Path] = []
    for p in all_frames:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq_frames.append(p)

    # Sort by time label embedded in filename if available, else fall back to OVF header time,
    # else fall back to name ordering.
    def _t_key(p: Path) -> Tuple[bool, float, str]:
        t_s = try_parse_sp2_time_label_seconds_from_name(p)
        if t_s is None:
            t_s = try_parse_time_seconds_from_ovf(p)
        if t_s is None:
            return (True, 0.0, p.name)
        return (False, float(t_s), p.name)

    sorted_frames = sorted(uniq_frames, key=_t_key)
    
    # Ensure diagnostic finals are last, in order: hc_pos, hc_best, hc_final (if present).
    name_seed_low = "_seed_low.ovf"
    name_seed_high = "_seed_high.ovf"
    name_pos = f"m_d{d_lex}_hc_pos.ovf"
    name_best = f"m_d{d_lex}_hc_best.ovf"
    name_final = f"m_d{d_lex}_hc.ovf"

    labels: Dict[str, str] = {}

    for nm, lab in (
        (name_seed_low, "seed_low"),
        (name_seed_high, "seed_high"),
        (name_pos, "hc_pos_strict"),
        (name_best, "hc_best_strict"),
        (name_final, "hc_final"),
    ):
        hits = [p for p in uniq_frames if p.name == nm]
        if hits:
            p0 = hits[0]
            sorted_frames = [p for p in sorted_frames if p != p0] + [p0]
            labels[str(p0)] = lab
            
    # Build a synthetic series descriptor.
    base = cands[0]
    tags = sorted({(s.tag or "") for s in cands})

    meta: Dict[str, Any] = {
        "combined_tags": tags,
        "combined_series_ids": [s.series_id for s in cands],
    }
    if labels:
        meta["frame_labels"] = labels

    return OvfSeries(
        series_id=f"sp2:d{d_lex:02d}:hc:all:{base.source}",
        problem="sp2",
        source=base.source,
        kind=f"d{d_lex:02d}_hc_evolution_all",
        stage="hc",
        d_lex=d_lex,
        tag=None,
        frames=sorted_frames,
        meta=meta,
    )