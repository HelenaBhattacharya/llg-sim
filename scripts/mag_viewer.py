#!/usr/bin/env python3
"""mag_viewer.py — interactive PyVista OVF viewer for SP2 + SP4.

This viewer is intended for quick interactive inspection of OVF snapshots produced
by either:
  - Rust (llg-sim) runs/...
  - MuMax mumax_outputs/...

Supported (current focus)
-------------------------
SP4
  - time evolution snapshots m0000000.ovf ... (MuMax-style naming)

SP2
  - sweep mode: final states across d/lex (remanence or coercivity)
  - evolve mode: time-series for one d/lex (remanence relax or coercivity relax)

Compare mode (optional)
-----------------------
If you provide --input-b, the viewer loads A and B series in lock-step so you can
cycle between:
  A, B, Δfixed, Δauto, |Δ|.

Controls (keyboard)
-------------------
  n / p        next / previous frame
  1 / 2 / 3    color by mx / my / mz
  g            toggle glyph arrows
  [ / ]        denser / sparser glyphs
  space        play / pause
  - / + / =    slower / faster playback
  v            cycle compare mode (A -> B -> Δfixed -> Δauto -> |Δ|)
  w            toggle warp-by-scalar ("altitude" view)
  u / j        increase / decrease warp scale
  s            screenshot to plots/viewer/...
  r            reset camera (top-down or isometric depending on warp)
  q / Esc      quit

Notes on "warp / altitude"
--------------------------
Warp displaces the mesh in +z by `warp_scale * scalar_value`.

- By default, the scalar value is the SAME quantity you are colouring by (mx/my/mz
  or |Δ| in compare mode). This is standard in the literature (e.g. surface plots
  of m3/out-of-plane magnetisation).

- In a strict top-down view you won't *see* height. When warp is enabled we switch
  to an isometric view automatically so the height becomes visible. You can still
  rotate/zoom with the mouse as usual.

If your m_z is ~0 (common for in-plane problems), the m_z colourmap will look flat
and warping by m_z will also look flat. In that case try mx/my or use compare mode
(|Δ|) where small differences become visible.

Examples
--------
SP4 (Rust):
  python3 scripts/mag_viewer.py --input runs/st_problems/sp4 --problem sp4 --case sp4a

SP2 sweep (final remanence across d/lex):
  python3 scripts/mag_viewer.py --input runs/st_problems/sp2 --problem sp2 --stage rem

SP2 evolve (coercivity relaxation series for d=30):
  python3 scripts/mag_viewer.py --input runs/st_problems/sp2 --problem sp2 --d 30 --stage hc

Compare (SP4 Rust vs MuMax):
  python3 scripts/mag_viewer.py --input runs/st_problems/sp4 --input-b mumax_outputs/st_problems/sp4 --problem sp4 --case sp4a
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import discretisedfield as df
import pyvista as pv

# Optional Qt backend for smoother interaction
try:
    from pyvistaqt import BackgroundPlotter  # type: ignore

    _HAVE_PYVISTAQT = True
except Exception:
    BackgroundPlotter = None  # type: ignore
    _HAVE_PYVISTAQT = False


# -----------------------------------------------------------------------------
# Import shared helpers (src/ovf_utils.py)
# -----------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_REPO_ROOT = next(
    (
        c
        for c in (_HERE.parent, _HERE.parent.parent)
        if (c / "src").exists() and (c / "src").is_dir()
    ),
    _HERE.parent,
)
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    import ovf_utils
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Could not import ovf_utils. Expected at src/ovf_utils.py.\n"
        "Make sure you copied ovf_utils.py into the repo's src/ folder.\n"
        f"Import error: {e}"
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _field_signature(m: df.Field) -> Tuple[int, int, float, float]:
    """Signature used to decide if we must rebuild the PyVista geometry.

    We intentionally IGNORE absolute origin (xmin/ymin) because:
      - many OVFs are logically identical and should share geometry
      - older Rust OVFs may be missing xmin/xmax and require patching
        (small header differences should not cause persistent actor stacking)

    Signature: (nx, ny, dx, dy)
    """
    n = m.mesh.n
    if len(n) == 3:
        nx, ny, _ = n
    else:
        nx, ny = n

    cell = m.mesh.cell
    if len(cell) == 3:
        dx, dy, _ = cell
    else:
        dx, dy = cell

    return int(nx), int(ny), float(dx), float(dy)


def _coords_for_structured_grid(m: df.Field, units: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X3, Y3, Z3 arrays for pv.StructuredGrid.

    Coordinates are cell-centres (x-fastest), matching OVF conventions.
    """
    n = m.mesh.n
    if len(n) == 3:
        nx, ny, _ = n
    else:
        nx, ny = n

    cell = m.mesh.cell
    if len(cell) == 3:
        dx, dy, _ = cell
    else:
        dx, dy = cell

    # Robust origin. For most of our OVFs this is 0 anyway.
    try:
        xmin, ymin, _ = m.mesh.region.pmin
    except Exception:
        xmin, ymin = 0.0, 0.0

    scale = 1e9 if units == "nm" else 1.0
    xs = (xmin + (np.arange(nx) + 0.5) * dx) * scale
    ys = (ymin + (np.arange(ny) + 0.5) * dy) * scale

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.zeros_like(X)

    # StructuredGrid expects 3D arrays (nx, ny, nz). We use nz=1.
    return X[:, :, None], Y[:, :, None], Z[:, :, None]


def _flatten_fortran(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).ravel(order="F")


def _parse_sp2_d_from_path(p: Path) -> Optional[int]:
    m = re.search(r"m_d(\d+)_", p.name)
    if m:
        return int(m.group(1))
    return None


def _finite_minmax(a: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(a, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.min()), float(arr.max())


@dataclass
class FrameInfo:
    path: Path
    label: str
    t_s: Optional[float] = None
    d_lex: Optional[int] = None


# -----------------------------------------------------------------------------
# Series selection + alignment
# -----------------------------------------------------------------------------


def _select_sp4_series(series: List[ovf_utils.OvfSeries], case: Optional[str]) -> ovf_utils.OvfSeries:
    if not series:
        raise ValueError("No SP4 series found")

    if case:
        case2 = case.lower().strip()
        for s in series:
            if s.kind.lower() == case2:
                return s
        raise ValueError(f"Requested case '{case}' not found. Available: {[s.kind for s in series]}")

    for s in series:
        if s.kind.lower() == "sp4a":
            return s
    return series[0]


def _select_sp2_series(
    series: List[ovf_utils.OvfSeries],
    d_lex: Optional[int],
    stage: str,
    tag: Optional[str],
) -> ovf_utils.OvfSeries:
    if not series:
        raise ValueError("No SP2 series found")

    stage2 = stage.lower().strip()
    stage2 = "rem" if stage2 in {"rem", "remanence"} else "hc"

    if d_lex is None:
        s = ovf_utils.pick_sp2_sweep_series(series, stage2)
        if s is None:
            raise ValueError(f"No SP2 sweep series for stage={stage2}")
        return s

    s = ovf_utils.pick_sp2_evolution_series(series, int(d_lex), stage2, tag=tag)
    if s is None:
        cands = [x for x in series if x.problem == "sp2" and x.d_lex == int(d_lex) and x.stage == stage2]
        tags = sorted({x.tag for x in cands if x.tag})
        if stage2 == "hc" and tags:
            raise ValueError(
                f"No SP2 evolution series for d={d_lex} stage={stage2} tag={tag}. Available tags: {tags}"
            )
        raise ValueError(f"No SP2 evolution series for d={d_lex} stage={stage2}")
    return s


def _build_frame_list(series: ovf_utils.OvfSeries) -> List[FrameInfo]:
    out: List[FrameInfo] = []
    for p in series.frames:
        if series.problem == "sp4":
            t_s = ovf_utils.try_parse_time_seconds_from_ovf(p)
            label = f"t={t_s * 1e9:.3f} ns" if t_s is not None else p.name
            out.append(FrameInfo(path=p, label=label, t_s=t_s))
        else:
            if series.kind.startswith("sweep"):
                d = _parse_sp2_d_from_path(p)
                label = f"d/lex={d}" if d is not None else p.name
                out.append(FrameInfo(path=p, label=label, d_lex=d))
            else:
                # Prefer explicit frame labels injected by ovf_utils (e.g. hc_pos_strict/hc_best_strict/hc_final)
                frame_labels = series.meta.get("frame_labels")
                if isinstance(frame_labels, dict):
                    lab = frame_labels.get(str(p))
                    if isinstance(lab, str) and lab:
                        out.append(FrameInfo(path=p, label=lab, t_s=None, d_lex=series.d_lex))
                        continue

                t_s = ovf_utils.try_parse_sp2_time_label_seconds_from_name(p)
                if t_s is None:
                    t_s = ovf_utils.try_parse_time_seconds_from_ovf(p)

                tag = ovf_utils.try_parse_sp2_tag_from_name(p)
                if t_s is not None:
                    if tag:
                        label = f"{tag} | t={t_s * 1e9:.3f} ns"
                    else:
                        label = f"t={t_s * 1e9:.3f} ns"
                else:
                    label = tag if tag else p.name

                out.append(FrameInfo(path=p, label=label, t_s=t_s, d_lex=series.d_lex))
    return out


def _downsample_frames_by_ns(frames: List[FrameInfo], sample_ns: Optional[float]) -> List[FrameInfo]:
    """Downsample a time-series to ~one frame per `sample_ns` of simulated time."""
    if sample_ns is None or sample_ns <= 0:
        return frames

    bins: Dict[int, FrameInfo] = {}
    for f in frames:
        if f.t_s is None:
            continue
        b = int(round((f.t_s * 1e9) / float(sample_ns)))
        bins.setdefault(b, f)
    if not bins:
        return frames
    return [bins[k] for k in sorted(bins.keys())]


def _downsample_paired_frames_by_ns(
    frames_a: List[FrameInfo], frames_b: List[FrameInfo], sample_ns: Optional[float]
) -> Tuple[List[FrameInfo], List[FrameInfo]]:
    if sample_ns is None or sample_ns <= 0:
        return frames_a, frames_b

    bins: Dict[int, int] = {}
    for i, fa in enumerate(frames_a):
        if fa.t_s is None:
            continue
        b = int(round((fa.t_s * 1e9) / float(sample_ns)))
        bins.setdefault(b, i)
    if not bins:
        return frames_a, frames_b

    idxs = [bins[k] for k in sorted(bins.keys())]
    return [frames_a[i] for i in idxs], [frames_b[i] for i in idxs]


def _align_for_compare(
    series_a: ovf_utils.OvfSeries, series_b: ovf_utils.OvfSeries
) -> Tuple[List[FrameInfo], List[FrameInfo]]:
    fa = _build_frame_list(series_a)
    fb = _build_frame_list(series_b)

    # SP2 sweep: align by d/lex
    if series_a.problem == "sp2" and series_a.kind.startswith("sweep") and series_b.kind.startswith("sweep"):
        map_a: Dict[int, FrameInfo] = {f.d_lex: f for f in fa if f.d_lex is not None}
        map_b: Dict[int, FrameInfo] = {f.d_lex: f for f in fb if f.d_lex is not None}
        common = sorted(set(map_a.keys()) & set(map_b.keys()), reverse=True)
        if not common:
            raise ValueError("No common d/lex values found between A and B sweeps")
        aa = [map_a[d] for d in common]
        bb = [map_b[d] for d in common]
        return aa, bb

    # Default: align by index
    n = min(len(fa), len(fb))
    return fa[:n], fb[:n]


# -----------------------------------------------------------------------------
# Viewer
# -----------------------------------------------------------------------------


class MagViewer:
    def __init__(
        self,
        frames_a: List[FrameInfo],
        series_a: ovf_utils.OvfSeries,
        frames_b: Optional[List[FrameInfo]] = None,
        series_b: Optional[ovf_utils.OvfSeries] = None,
        component: str = "mx",
        glyphs_on: bool = True,
        glyph_stride: int = 0,
        glyph_scale: float = 0.0,
        units: str = "nm",
        use_qt: bool = False,
        screenshot_root: Path = Path("plots") / "viewer",
    ):
        self.series_a = series_a
        self.series_b = series_b
        self.frames_a = frames_a
        self.frames_b = frames_b
        self.compare = frames_b is not None and series_b is not None

        self.idx = 0
        self.component = component
        self.units = units

        self.glyphs_on = glyphs_on
        self.glyph_stride = int(glyph_stride)
        self.glyph_scale = float(glyph_scale)

        # Compare view mode:
        #   a | b | delta_fixed | delta_auto
        self.view_mode = "delta_auto" if self.compare else "a"

        # Warp (altitude) view
        self.warp_on = False
        self.warp_scale = 0.0  # auto if <=0

        # Playback
        self.playing = False
        self.fps = 5.0
        self._last_play_step = time.perf_counter()

        self.screenshot_root = screenshot_root

        # Plotter selection
        self.use_qt = bool(use_qt) and _HAVE_PYVISTAQT and BackgroundPlotter is not None
        if self.use_qt:
            self.plotter: Any = BackgroundPlotter(show=True)  # type: ignore
        else:
            self.plotter = pv.Plotter()

        # Theme / background (helps reduce macOS transient magenta frames)
        try:
            pv.set_plot_theme("document")
        except Exception:
            pass
        try:
            self.plotter.set_background("white")
        except Exception:
            pass

        # Prevent PyVista pick callback from erroring on new attribute creation (PyVista>=0.45)
        # Prefer the supported API; if unavailable, just ignore (warning is non-fatal).
        try:
            set_new_attr = getattr(pv, "set_new_attribute", None)
            if callable(set_new_attr):
                set_new_attr(self.plotter, "pickpoint", None)
        except Exception:
            pass

        # Geometry cache
        self._grid: Optional[pv.StructuredGrid] = None
        self._grid_sig: Optional[Tuple[int, int, float, float]] = None
        self._base_points: Optional[np.ndarray] = None  # used for warp-in-place

        # Data cache for current frame
        self._cur_loaded_idx: Optional[int] = None
        self._cur_vec_a: Optional[np.ndarray] = None  # (nx, ny, 3)
        self._cur_vec_b: Optional[np.ndarray] = None

        # Actors
        self._mesh_actor: Any = None
        self._glyph_actor: Any = None

        # Text overlay
        self._status_name = "status"

        # VTK timer
        self._timer_created = False
        self._timer_observer_id: Optional[int] = None
        self._timer_id: Optional[int] = None
        self._timer_callback: Optional[Any] = None  # keep strong ref to VTK observer callable

        # Keep last scalar stats for overlay
        self._last_scalar_minmax: Tuple[float, float] = (0.0, 0.0)
        # Metrics for overlay: (mx_mean, my_mean, mz_mean, msum, mpar)
        self._last_metrics: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Loading / geometry
    # ------------------------------------------------------------------

    def _remove_actor(self, actor: Any) -> None:
        if actor is None:
            return
        try:
            self.plotter.remove_actor(actor)
        except Exception:
            pass

    def _clear_scene(self) -> None:
        """Remove mesh/glyph actors and scalar bar.

        This is critical when geometry changes: otherwise actors can stack up and
        you end up seeing multiple frames "superimposed".
        """
        self._remove_actor(self._mesh_actor)
        self._remove_actor(self._glyph_actor)
        self._mesh_actor = None
        self._glyph_actor = None
        try:
            self.plotter.remove_scalar_bar()
        except Exception:
            pass

    def _load_current(self) -> None:
        if self._cur_loaded_idx == self.idx:
            return

        info_a = self.frames_a[self.idx]
        m_a = ovf_utils.load_slice_field(info_a.path)
        vec_a = ovf_utils.field_vector_array_2d(m_a)

        vec_b: Optional[np.ndarray] = None
        if self.compare and self.frames_b is not None:
            info_b = self.frames_b[self.idx]
            m_b = ovf_utils.load_slice_field(info_b.path)
            vec_b = ovf_utils.field_vector_array_2d(m_b)
            if vec_b.shape != vec_a.shape:
                raise ValueError(f"A/B frame shape mismatch: {vec_a.shape} vs {vec_b.shape}")

        # Update caches
        self._cur_loaded_idx = self.idx
        self._cur_vec_a = vec_a
        self._cur_vec_b = vec_b

        # Ensure geometry
        sig = _field_signature(m_a)
        if self._grid is None or self._grid_sig != sig:
            X3, Y3, Z3 = _coords_for_structured_grid(m_a, units=self.units)
            self._grid = pv.StructuredGrid(X3, Y3, Z3)
            self._grid_sig = sig
            self._base_points = np.asarray(self._grid.points).copy()

            # Allocate arrays once (we update them in-place)
            npts = self._grid.n_points
            self._grid.point_data["scalars"] = np.zeros(npts, dtype=np.float64)
            self._grid.point_data["vec"] = np.zeros((npts, 3), dtype=np.float64)
            self._grid.point_data["vec_mag"] = np.zeros(npts, dtype=np.float64)

            # When geometry changes, clear previous actors to avoid stacking
            self._clear_scene()

    # ------------------------------------------------------------------
    # Compute view arrays + ranges
    # ------------------------------------------------------------------

    def _current_vec_for_view(self) -> Tuple[np.ndarray, str]:
        """Return (vec, mode_label) where vec is (nx, ny, 3) for the current view."""
        assert self._cur_vec_a is not None
        vec_a = self._cur_vec_a
        vec_b = self._cur_vec_b

        if not self.compare or self.view_mode == "a":
            return vec_a, "A"
        if self.view_mode == "b":
            assert vec_b is not None
            return vec_b, "B"
        if self.view_mode in {"delta_fixed", "delta_auto"}:
            assert vec_b is not None
            return (vec_a - vec_b), "Δ"
        return vec_a, "A"

    def _compute_view_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (scalar_vals_1d, vec_1d, mag_1d).

        - scalar_vals_1d is the quantity used for colourmap and for warp (if enabled)
        - vec_1d/mag_1d are used for glyph arrows (always in-plane)
        """
        vec, _ = self._current_vec_for_view()

        mx = vec[:, :, 0]
        my = vec[:, :, 1]
        mz = vec[:, :, 2]

        # Scalars for colour/warp
        if self.compare and self.view_mode == "abs":
            scal = np.sqrt(mx * mx + my * my + mz * mz)
            scalar_kind = "|Δ|"
        else:            
            if self.component == "mx":
                scal = mx
                scalar_kind = "mx"
            elif self.component == "my":
                scal = my
                scalar_kind = "my"
            elif self.component in {"mpar", "m_parallel", "mproj"}:
                # Projection onto SP2 coercivity field direction ĥ = (-1,-1,-1)/sqrt(3)
                # m_parallel = m · ĥ = -(mx + my + mz)/sqrt(3)
                scal = -(mx + my + mz) / np.sqrt(3.0)
                scalar_kind = "mpar"
            else:
                scal = mz
                scalar_kind = "mz"

        scal_1d = _flatten_fortran(scal)

        # Glyphs: always use in-plane components of the SAME vec we are visualising
        vec_inplane = np.stack([mx, my, np.zeros_like(mx)], axis=-1)
        vec_1d = vec_inplane.reshape((-1, 3), order="F")
        mag_1d = np.linalg.norm(vec_1d, axis=1)

        # Scalar stats for overlay
        self._last_scalar_minmax = _finite_minmax(scal_1d)

        # Metrics for status overlay (based on the currently viewed vec field)
        mx_mean = float(np.mean(mx))
        my_mean = float(np.mean(my))
        mz_mean = float(np.mean(mz))
        msum = mx_mean + my_mean + mz_mean
        mpar = -msum / np.sqrt(3.0)
        self._last_metrics = (mx_mean, my_mean, mz_mean, msum, mpar)

        # Remember scalar_kind for clim logic
        self._last_scalar_kind = scalar_kind
        return scal_1d, vec_1d, mag_1d

    def _clim_for_current_view(self, scalars: np.ndarray) -> Tuple[float, float]:
        """Colour limits.

        Goals:
          - A/B: fixed [-1, 1] for mx/my/mz (stable colour bar, avoids tiny ranges)
          - Δfixed: fixed [-1, 1] (so small differences look flat)
          - Δauto: symmetric around 0 using max|Δ| (so differences are visible)
          - |Δ|: [0, max] (visible magnitude)

        This also helps reduce the macOS/VTK "magenta" glitch that often appears
        when scalar ranges collapse to ~0.
        """
        if not self.compare:
            # Single series
            if self._last_scalar_kind in {"mx", "my", "mz"}:
                return (-1.0, 1.0)
            vmin, vmax = _finite_minmax(scalars)
            if vmin == vmax:
                eps = 1e-12
                return (vmin - eps, vmax + eps)
            return (vmin, vmax)

        # Compare mode
        if self.view_mode in {"a", "b"}:
            return (-1.0, 1.0)

        if self.view_mode == "delta_fixed":
            return (-1.0, 1.0)

        if self.view_mode == "delta_auto":
            arr = np.asarray(scalars, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            maxabs = float(np.max(np.abs(arr))) if arr.size else 0.0
            maxabs = max(maxabs, 1e-6)
            return (-maxabs, maxabs)
    
        vmin, vmax = _finite_minmax(scalars)
        if vmin == vmax:
            eps = 1e-12
            return (vmin - eps, vmax + eps)
        return (vmin, vmax)


    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _status_text(self) -> str:
        info_a = self.frames_a[self.idx]

        if self.series_a.problem == "sp4":
            head = f"SP4 {self.series_a.kind} ({self.series_a.source})"
        else:
            if self.series_a.kind.startswith("sweep"):
                head = f"SP2 sweep {self.series_a.stage} ({self.series_a.source})"
            else:
                head = f"SP2 d={self.series_a.d_lex} {self.series_a.stage} ({self.series_a.source})"

        if not self.compare:
            view = "A"
        else:
            view = {
                "a": "A",
                "b": "B",
                "delta_fixed": "Δfixed",
                "delta_auto": "Δauto",
            }.get(self.view_mode, self.view_mode)

        comp = self.component
        vmin, vmax = self._last_scalar_minmax

        line1 = f"{head}  |  {info_a.label}  |  frame {self.idx+1}/{len(self.frames_a)}"
        if self.compare and self.frames_b is not None:
            info_b = self.frames_b[self.idx]
            line1 += f"\nCompare: {view}  |  B: {info_b.path.name}"
        else:
            line1 += f"\nView: {view}"

        warp_txt = "ON" if self.warp_on else "OFF"
        line2 = (
            f"Color: {comp}  |  Scalars: [{vmin:+.3e}, {vmax:+.3e}]  |  "
            f"Glyphs: {'ON' if self.glyphs_on else 'OFF'}  |  Warp: {warp_txt}  |  Play: {'ON' if self.playing else 'OFF'} (fps={self.fps:.1f})"
        )
        if self.warp_on:
            line2 += f" (scale={self.warp_scale:.3g} {self.units})"

        # Add always-on metrics display
        mx_mean, my_mean, mz_mean, msum, mpar = self._last_metrics
        line2 += (
            f"\nmsum={msum:+.6f}  <m>=({mx_mean:+.3f},{my_mean:+.3f},{mz_mean:+.3f})  mpar={mpar:+.6f}"
        )

        line3 = (
            "Keys: n/p next/prev, 1/2/3/4 mx/my/mz/mpar, g glyphs, [/] density, space play, f hc finals, "
            "v compare, w warp, u/j warp scale, s screenshot, r reset, q quit"
        )
        return line1 + "\n" + line2 + "\n" + line3
    def cycle_hc_diagnostics(self) -> None:
        """Cycle among SP2 hc diagnostic final frames (hc_pos_strict -> hc_best_strict -> hc_final) if present."""
        if self.series_a.problem != "sp2" or (self.series_a.stage or "") != "hc":
            return

        # Build a list of indices for the diagnostic frames by filename suffix.
        names = [f.path.name for f in self.frames_a]
        idx_pos = [i for i, n in enumerate(names) if n.endswith("_hc_pos.ovf")]
        idx_best = [i for i, n in enumerate(names) if n.endswith("_hc_best.ovf")]
        idx_final = [i for i, n in enumerate(names) if n.endswith("_hc.ovf") and not n.endswith("_hc_pos.ovf") and not n.endswith("_hc_best.ovf")]

        order: List[int] = []
        if idx_pos:
            order.append(idx_pos[-1])
        if idx_best:
            order.append(idx_best[-1])
        if idx_final:
            order.append(idx_final[-1])

        if len(order) < 2:
            return

        # Find next in cycle relative to current index.
        if self.idx in order:
            j = order.index(self.idx)
            self.idx = order[(j + 1) % len(order)]
        else:
            self.idx = order[0]

        self._update_mesh(reset_camera=False)

    def _ensure_mesh_actor(self, clim: Tuple[float, float]) -> None:
        assert self._grid is not None

        if self._mesh_actor is not None:
            return

        # Remove any existing scalar bar to avoid accumulation
        try:
            self.plotter.remove_scalar_bar()
        except Exception:
            pass

        self._mesh_actor = self.plotter.add_mesh(
            self._grid,
            scalars="scalars",
            cmap="viridis",
            show_edges=False,
            nan_color="white",
            clim=clim,
        )

        self._apply_default_view(reset_camera=True)

    def _apply_default_view(self, reset_camera: bool) -> None:
        """Choose a sensible default view depending on warp state."""
        try:
            if self.warp_on:
                # 3D-ish view
                if hasattr(self.plotter, "disable_parallel_projection"):
                    self.plotter.disable_parallel_projection()
                self.plotter.view_isometric()
            else:
                # 2D top-down view
                self.plotter.view_xy()
                if hasattr(self.plotter, "enable_parallel_projection"):
                    self.plotter.enable_parallel_projection()
            if reset_camera:
                self.plotter.reset_camera()
        except Exception:
            pass

    def _apply_warp_in_place(self, scalars_1d: np.ndarray) -> None:
        """Warp the current grid points in-place using current scalar array."""
        if self._grid is None or self._base_points is None:
            return

        pts = np.asarray(self._grid.points)
        base = np.asarray(self._base_points)

        if not self.warp_on:
            # Restore
            pts[:, 2] = base[:, 2]
            self._grid.points = pts
            return

        # Auto warp scale based on physical size (so it is actually visible).
        if self.warp_scale <= 0.0:
            try:
                xmin, xmax, ymin, ymax, _zmin, _zmax = self._grid.bounds
                extent = min(abs(xmax - xmin), abs(ymax - ymin))
                # 10% of the smallest extent is a good "visible but not insane" default.
                self.warp_scale = max(1e-9, 0.10 * float(extent))
            except Exception:
                self.warp_scale = 1.0

        # Displace in z
        z = base[:, 2] + np.nan_to_num(scalars_1d, nan=0.0, posinf=0.0, neginf=0.0) * float(self.warp_scale)
        pts[:, 2] = z
        self._grid.points = pts

    def _update_mesh(self, reset_camera: bool = False) -> None:
        self._load_current()
        assert self._grid is not None

        scal_1d, vec_1d, mag_1d = self._compute_view_arrays()
        scal_1d = np.nan_to_num(scal_1d, nan=0.0, posinf=0.0, neginf=0.0)
        vec_1d = np.nan_to_num(vec_1d, nan=0.0, posinf=0.0, neginf=0.0)
        mag_1d = np.nan_to_num(mag_1d, nan=0.0, posinf=0.0, neginf=0.0)

        # Update arrays in-place (reduces renderer churn)
        np.copyto(np.asarray(self._grid.point_data["scalars"]), scal_1d)
        np.copyto(np.asarray(self._grid.point_data["vec"]), vec_1d)
        np.copyto(np.asarray(self._grid.point_data["vec_mag"]), mag_1d)

        clim = self._clim_for_current_view(scal_1d)

        self._ensure_mesh_actor(clim=clim)

        # Update scalar range
        try:
            actor = self._mesh_actor
            mapper = getattr(actor, "mapper", None)
            if mapper is not None and hasattr(mapper, "SetScalarRange"):
                mapper.SetScalarRange(float(clim[0]), float(clim[1]))
        except Exception:
            pass

        # Warp (altitude) view
        self._apply_warp_in_place(scal_1d)

        # Glyphs
        if self._glyph_actor is not None:
            self._remove_actor(self._glyph_actor)
            self._glyph_actor = None

        if self.glyphs_on:
            glyphs = self._make_glyphs(self._grid)
            self._glyph_actor = self.plotter.add_mesh(
                glyphs,
                color="white",
                opacity=0.95,
                lighting=False,
                nan_color="white",
            )

        # Status text
        try:
            self.plotter.add_text(self._status_text(), name=self._status_name, font_size=11)
        except Exception:
            try:
                self.plotter.add_text(self._status_text(), position="upper_left", font_size=11)
            except Exception:
                pass

        if reset_camera:
            self._apply_default_view(reset_camera=True)

        try:
            self.plotter.render()
        except Exception:
            pass

    def _make_glyphs(self, grid: pv.StructuredGrid) -> pv.PolyData:
        stride = self.glyph_stride
        dims = grid.dimensions
        nx = int(dims[0])
        ny = int(dims[1])
        _nz = int(dims[2]) if len(dims) > 2 else 1
        if stride <= 0:
            stride = max(1, int(round(min(nx, ny) / 12.0)))
        stride = max(1, int(stride))

        ii = np.arange(0, nx, stride)
        jj = np.arange(0, ny, stride)
        ids = np.array([i + nx * j for j in jj for i in ii], dtype=int)

        pts = grid.points[ids]
        poly = pv.PolyData(pts)
        vec = np.asarray(grid.point_data["vec"])[ids]
        mag = np.asarray(grid.point_data["vec_mag"])[ids]

        poly["vec"] = vec
        poly["mag"] = mag

        # Glyph size
        if self.glyph_scale <= 0.0:
            try:
                dx_est = float(np.linalg.norm(grid.points[1] - grid.points[0])) if grid.n_points > 1 else 1.0
            except Exception:
                dx_est = 1.0
            scale = 0.8 * dx_est * float(stride)
        else:
            scale = float(self.glyph_scale)

        return cast(pv.PolyData, poly.glyph(orient="vec", scale="mag", factor=scale))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def next_frame(self) -> None:
        self.idx = (self.idx + 1) % len(self.frames_a)
        self._update_mesh(reset_camera=False)

    def prev_frame(self) -> None:
        self.idx = (self.idx - 1) % len(self.frames_a)
        self._update_mesh(reset_camera=False)

    def set_component(self, comp: str) -> None:
        self.component = comp
        self._update_mesh(reset_camera=False)

    def toggle_glyphs(self) -> None:
        self.glyphs_on = not self.glyphs_on
        self._update_mesh(reset_camera=False)

    def more_dense(self) -> None:
        self.glyph_stride = max(1, self.glyph_stride // 2) if self.glyph_stride > 1 else 1
        self._update_mesh(reset_camera=False)

    def less_dense(self) -> None:
        self.glyph_stride = 2 if self.glyph_stride <= 1 else min(4096, self.glyph_stride * 2)
        self._update_mesh(reset_camera=False)

    def reset_camera(self) -> None:
        self._update_mesh(reset_camera=True)
    
    def toggle_play(self) -> None:
        self.playing = not self.playing
        self._last_play_step = time.perf_counter()
        # Create the timer on-demand. Some backends only have `iren` after the window is shown.
        if self.playing:
            self._ensure_timer()
            if not self._timer_created:
                print("[playback] Play enabled but timer not created; try running with --qt")
        # Refresh overlay so Play status changes immediately
        self._update_mesh(reset_camera=False)

    def slower(self) -> None:
        self.fps = max(0.5, self.fps * 0.8)

    def faster(self) -> None:
        self.fps = min(60.0, self.fps * 1.25)

    def cycle_view_mode(self) -> None:
        if not self.compare:
            return
        order = ["a", "b", "delta_fixed", "delta_auto"]
        i = order.index(self.view_mode) if self.view_mode in order else 0
        self.view_mode = order[(i + 1) % len(order)]
        self._update_mesh(reset_camera=False)

    def toggle_warp(self) -> None:
        self.warp_on = not self.warp_on
        # When turning warp on, switch to a 3D-ish view so height is visible.
        self._apply_default_view(reset_camera=True)
        self._update_mesh(reset_camera=False)

    def warp_more(self) -> None:
        if self.warp_scale <= 0.0:
            # initialise from auto
            self.warp_on = True
            self._update_mesh(reset_camera=False)
        self.warp_scale *= 1.25
        self._update_mesh(reset_camera=False)

    def warp_less(self) -> None:
        if self.warp_scale <= 0.0:
            self.warp_on = True
            self._update_mesh(reset_camera=False)
        self.warp_scale *= 0.8
        self._update_mesh(reset_camera=False)

    def screenshot(self) -> None:
        problem = self.series_a.problem
        source = self.series_a.source
        kind = self.series_a.kind
        stage = self.series_a.stage or ""
        tag = self.series_a.tag or ""

        parts = [problem, source, kind]
        if stage:
            parts.append(stage)
        if tag and tag not in parts:
            parts.append(tag)
        out_dir = self.screenshot_root.joinpath(*parts)
        out_dir.mkdir(parents=True, exist_ok=True)

        mode = self.view_mode if self.compare else "a"
        comp = self.component
        out_path = out_dir / f"shot_{self.idx:06d}_{mode}_{comp}.png"
        try:
            self.plotter.screenshot(str(out_path))
            print(f"[screenshot] wrote {out_path}")
        except Exception as e:
            print(f"[screenshot] failed: {e}")

    # ------------------------------------------------------------------
    # Timer / run loop
    # ------------------------------------------------------------------

    def _ensure_timer(self) -> None:
        """Create a repeating VTK timer that drives playback.

        Note: On some backends the real VTK interactor is available as
        `plotter.iren.interactor` or `plotter.iren._iren`.

        We keep a strong reference to the callback to avoid it being garbage-collected.
        """
        if self._timer_created:
            return

        iren = getattr(self.plotter, "iren", None)
        if iren is None:
            return

        # Prefer the underlying vtkRenderWindowInteractor when available.
        vtk_iren = getattr(iren, "interactor", None) or getattr(iren, "_iren", None) or iren

        def _on_timer(_obj: Any = None, _event: str = "") -> None:
            if not self.playing:
                return
            now = time.perf_counter()
            dt = now - self._last_play_step
            if dt >= (1.0 / max(self.fps, 0.1)):
                self._last_play_step = now
                self.next_frame()

        # Keep a strong reference to the callback; some VTK/Python bindings require this.
        self._timer_callback = _on_timer

        try:
            if hasattr(vtk_iren, "Initialize"):
                try:
                    vtk_iren.Initialize()
                except Exception:
                    pass

            if hasattr(vtk_iren, "AddObserver") and hasattr(vtk_iren, "CreateRepeatingTimer"):
                self._timer_observer_id = vtk_iren.AddObserver("TimerEvent", self._timer_callback)
                self._timer_id = vtk_iren.CreateRepeatingTimer(50)  # ms
                self._timer_created = True
                try:
                    self.plotter.render()
                except Exception:
                    pass
                return
        except Exception:
            pass

        # Fallback: PyVista callback API (Qt backend often supports this)
        try:
            add_cb = getattr(self.plotter, "add_callback", None)
            if callable(add_cb):
                add_cb(lambda: self._timer_callback(), interval=50)  # type: ignore[misc]
                self._timer_created = True
        except Exception:
            pass

    def run(self) -> None:
        """Start the interactive viewer."""
        self._update_mesh(reset_camera=True)

        self.plotter.add_key_event("n", lambda: self.next_frame())
        self.plotter.add_key_event("p", lambda: self.prev_frame())

        self.plotter.add_key_event("1", lambda: self.set_component("mx"))
        self.plotter.add_key_event("2", lambda: self.set_component("my"))
        self.plotter.add_key_event("3", lambda: self.set_component("mz"))
        self.plotter.add_key_event("4", lambda: self.set_component("mpar"))

        self.plotter.add_key_event("g", lambda: self.toggle_glyphs())
        self.plotter.add_key_event("[", lambda: self.more_dense())
        self.plotter.add_key_event("]", lambda: self.less_dense())

        self.plotter.add_key_event("r", lambda: self.reset_camera())
        self.plotter.add_key_event("f", lambda: self.cycle_hc_diagnostics())

        # Spacebar key symbol varies slightly across VTK/PyVista backends
        self.plotter.add_key_event("space", lambda: self.toggle_play())
        self.plotter.add_key_event("Space", lambda: self.toggle_play())
        self.plotter.add_key_event(" ", lambda: self.toggle_play())

        self.plotter.add_key_event("-", lambda: self.slower())
        self.plotter.add_key_event("plus", lambda: self.faster())
        self.plotter.add_key_event("=", lambda: self.faster())

        self.plotter.add_key_event("v", lambda: self.cycle_view_mode())
        self.plotter.add_key_event("w", lambda: self.toggle_warp())
        self.plotter.add_key_event("u", lambda: self.warp_more())
        self.plotter.add_key_event("j", lambda: self.warp_less())
        self.plotter.add_key_event("s", lambda: self.screenshot())

        # Quit (more stable than plotter.close() from inside callbacks)
        def request_quit() -> None:
            try:
                iren = getattr(self.plotter, "iren", None)
                if iren is not None and hasattr(iren, "TerminateApp"):
                    iren.TerminateApp()
                app = getattr(self.plotter, "app", None)
                if app is not None:
                    try:
                        app.quit()
                    except Exception:
                        pass
            except Exception:
                pass

        self.plotter.add_key_event("q", request_quit)
        self.plotter.add_key_event("Escape", request_quit)

        # Show window
        try:
            self.plotter.show()
            if self.use_qt:
                app = getattr(self.plotter, "app", None)
                if app is not None:
                    if hasattr(app, "exec"):
                        app.exec()
                    elif hasattr(app, "exec_"):
                        app.exec_()
        finally:
            try:
                self.plotter.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Interactive OVF viewer (PyVista) for SP2 + SP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qt", action="store_true", help="Use pyvistaqt BackgroundPlotter (if installed).")
    p.add_argument("--input", required=True, help="Input root (runs/... or mumax_outputs/...).")
    p.add_argument("--input-b", default=None, help="Optional second input root for compare mode.")

    p.add_argument(
        "--problem",
        choices=["auto", "sp2", "sp4"],
        default="auto",
        help="Problem preset (auto tries to detect).",
    )
    p.add_argument(
        "--source",
        choices=["auto", "rust", "mumax", "custom"],
        default="auto",
        help="Override source detection (labels only).",
    )

    # SP4
    p.add_argument("--case", choices=["sp4a", "sp4b"], default=None, help="SP4 case.")

    # SP2
    p.add_argument("--d", type=int, default=None, help="SP2: d/lex for evolution mode (omit for sweep).")
    p.add_argument("--stage", choices=["rem", "hc"], default="rem", help="SP2: stage selection.")
    p.add_argument("--tag", type=str, default=None, help="SP2: choose a coercivity series tag (optional).")

    # Viewer options
    p.add_argument(
    "--component",
    choices=["mx", "my", "mz", "mpar"],
    default="mx",
    help="Scalar to color by (mx/my/mz) or mpar = m·(-1,-1,-1)/sqrt(3) for SP2.",
    )
    p.add_argument("--no-glyphs", action="store_true", help="Start with glyphs off.")
    p.add_argument("--glyph-stride", type=int, default=0, help="Glyph stride (0=auto).")
    p.add_argument("--glyph-scale", type=float, default=0.0, help="Glyph scale (0=auto).")
    p.add_argument("--units", choices=["nm", "m"], default="nm")
    p.add_argument(
        "--sample-ns",
        type=float,
        default=None,
        help="Optional time-based downsampling for evolution series (ignored for sweeps).",
    )

    args = p.parse_args()

    input_a = Path(args.input)
    if not input_a.exists():
        raise SystemExit(f"Input path does not exist: {input_a}")

    problem = ovf_utils.detect_problem(input_a) if args.problem == "auto" else args.problem
    source_a = ovf_utils.infer_source(input_a) if args.source == "auto" else args.source

    if problem == "sp4":
        series_a_all = ovf_utils.discover_sp4_series(input_a, source=source_a)
        series_a = _select_sp4_series(series_a_all, args.case)
    else:
        series_a_all = ovf_utils.discover_sp2_series(input_a, source=source_a)
        series_a = _select_sp2_series(series_a_all, args.d, args.stage, args.tag)

    frames_a = _downsample_frames_by_ns(_build_frame_list(series_a), args.sample_ns)
    if not frames_a:
        raise SystemExit("No OVF frames found for the selected series")

    # Optional compare mode
    frames_b: Optional[List[FrameInfo]] = None
    series_b: Optional[ovf_utils.OvfSeries] = None
    if args.input_b:
        input_b = Path(args.input_b)
        if not input_b.exists():
            raise SystemExit(f"Input-B path does not exist: {input_b}")

        source_b = ovf_utils.infer_source(input_b) if args.source == "auto" else args.source

        if problem == "sp4":
            series_b_all = ovf_utils.discover_sp4_series(input_b, source=source_b)
            series_b = _select_sp4_series(series_b_all, args.case)
        else:
            series_b_all = ovf_utils.discover_sp2_series(input_b, source=source_b)
            series_b = _select_sp2_series(series_b_all, args.d, args.stage, args.tag)

        frames_a, frames_b = _align_for_compare(series_a, series_b)
        frames_a, frames_b = _downsample_paired_frames_by_ns(frames_a, frames_b, args.sample_ns)

    print("=" * 80)
    print("Mag Viewer")
    print("=" * 80)
    print(f"Problem:   {problem}")
    print(f"Series A:  {series_a.series_id}  ({len(frames_a)} frames)")
    if series_b is not None and frames_b is not None:
        print(f"Series B:  {series_b.series_id}  ({len(frames_b)} frames)")
        print("Mode:      compare (press 'v' to cycle A/B/Δfixed/Δauto)")
    else:
        print("Mode:      single")
    print(f"Units:     {args.units}")
    print("=" * 80)

    viewer = MagViewer(
        frames_a=frames_a,
        series_a=series_a,
        frames_b=frames_b,
        series_b=series_b,
        component=args.component,
        glyphs_on=(not args.no_glyphs),
        glyph_stride=args.glyph_stride,
        glyph_scale=args.glyph_scale,
        units=args.units,
        use_qt=args.qt,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

