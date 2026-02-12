#!/usr/bin/env python3
# ===============================
# Example commands (SP4) — interactive PyVista viewer
# ===============================
#
# 1) MuMax SP4 root (choose case with --case)
#    python3 scripts/mag_viewer.py \
#      --input mumax_outputs/st_problems/sp4 \
#      --case sp4a
#
# 2) Rust SP4 root (choose case with --case)
#    python3 scripts/mag_viewer.py \
#      --input runs/st_problems/sp4 \
#      --case sp4b
#
# 3) Direct case folder (MuMax or Rust)
#    python3 scripts/mag_viewer.py \
#      --input mumax_outputs/st_problems/sp4/sp4a_out
#
#    python3 scripts/mag_viewer.py \
#      --input runs/st_problems/sp4/sp4b_rust
#
# 4) Force source override (rarely needed)
#    python3 scripts/mag_viewer.py \
#      --input runs/st_problems/sp4 \
#      --case sp4a \
#      --source rust
#
# Controls (in the viewer window):
#   n / p       : next / previous snapshot
#   1 / 2 / 3   : color by mx / my / mz
#   g           : toggle glyph arrows
#   [ / ]       : increase / decrease glyph density (stride)
#   r           : reset camera
#   q / Esc     : quit


"""
mag_viewer.py — Interactive OVF magnetisation viewer (PyVista)

Goal:
  - Load OVF snapshots (m*.ovf) from MuMax3 or Rust SP4 outputs
  - Interactively step through time and inspect the magnetisation field:
      • scalar colormap of mx/my/mz
      • optional in-plane glyph arrows (mx,my)

Notes:
  - This is intentionally separate from mag_visualisation.py (batch PNG exporter).
  - Requires: pyvista, numpy, discretisedfield
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import discretisedfield as df


try:
    import pyvista as pv
except Exception as e:
    print("Error: PyVista is required for mag_viewer.py.")
    print("Install with: pip install pyvista")
    print(f"Import error: {e}")
    sys.exit(1)

# Optional Qt backend (more stable windowing on macOS for some setups)
try:
    from pyvistaqt import BackgroundPlotter  # type: ignore
    _HAVE_PYVISTAQT = True
except Exception:
    BackgroundPlotter = None  # type: ignore
    _HAVE_PYVISTAQT = False


# ----------------------------
# Path / folder discovery utils
# ----------------------------

def infer_source(input_path: Path) -> str:
    """Infer whether the input data is from MuMax or Rust."""
    s = str(input_path).lower()
    name = input_path.name.lower()

    if name.endswith("_out") or "mumax" in s:
        return "mumax"
    if name.endswith("_rust") or ("runs" in s and "mumax" not in s):
        return "rust"

    if (input_path / "sp4a_out").exists() or (input_path / "sp4b_out").exists():
        return "mumax"
    if (input_path / "sp4a_rust").exists() or (input_path / "sp4b_rust").exists():
        return "rust"

    return "custom"


def case_label_from_dirname(dirname: str) -> str:
    """Map input directory names to stable case labels."""
    d = dirname.lower()
    if d.startswith("sp4a"):
        return "sp4a"
    if d.startswith("sp4b"):
        return "sp4b"
    return dirname


def find_ovf_case_dirs(input_path: Path) -> List[Path]:
    """Return directories that contain m*.ovf files."""
    if list(input_path.glob("m*.ovf")):
        return [input_path]

    preferred: List[Path] = []
    for sub in ["sp4a_out", "sp4b_out", "sp4a_rust", "sp4b_rust"]:
        d = input_path / sub
        if d.exists() and list(d.glob("m*.ovf")):
            preferred.append(d)
    if preferred:
        return preferred

    found: List[Path] = []
    for d in sorted([p for p in input_path.iterdir() if p.is_dir()]):
        if list(d.glob("m*.ovf")):
            found.append(d)
    return found



def pick_case_dir(case_dirs: List[Path], case: Optional[str]) -> Path:
    """Select a case directory from discovered case dirs."""
    if not case_dirs:
        raise ValueError("No case directories found.")

    # If only one directory has OVFs, no need to specify --case
    if len(case_dirs) == 1:
        return case_dirs[0]

    want = (case or "sp4a").lower().strip()
    if want in ("a", "sp4a"):
        want = "sp4a"
    if want in ("b", "sp4b"):
        want = "sp4b"

    for d in case_dirs:
        if case_label_from_dirname(d.name) == want:
            return d

    # Also allow passing folder name directly (sp4a_out, sp4a_rust, etc.)
    for d in case_dirs:
        if d.name.lower() == want:
            return d

    raise ValueError(
        f"Requested case '{case}' not found. Available: "
        f"{[case_label_from_dirname(d.name) for d in case_dirs]}"
    )


# ----------------------------
# SP2 helpers (virtual cases: remanence vs coercivity)
# ----------------------------

_D_RE = re.compile(r"m_d(\d+)_")

def is_sp2_folder(dir_path: Path) -> bool:
    """Detect SP2-style naming: m_dXX_rem.ovf and/or m_dXX_hc.ovf."""
    return bool(list(dir_path.glob("m_d*_rem.ovf")) or list(dir_path.glob("m_d*_hc.ovf")))

def d_from_sp2_name(p: Path) -> int:
    m = _D_RE.search(p.name)
    return int(m.group(1)) if m else 0

def filter_sp2_case(files: List[Path], case: str) -> List[Path]:
    """Filter SP2 OVFs by virtual case: remanence or coercivity."""
    c = (case or "remanence").lower().strip()
    if c in ("rem", "remanence"):
        keep = [p for p in files if p.name.endswith("_rem.ovf")]
        return sorted(keep, key=d_from_sp2_name, reverse=True)
    if c in ("hc", "coercivity"):
        keep = [p for p in files if p.name.endswith("_hc.ovf")]
        return sorted(keep, key=d_from_sp2_name, reverse=True)
    # If user asks for all, just sort by d if possible
    if c in ("all", "both"):
        return sorted(files, key=d_from_sp2_name, reverse=True)
    return files


# ----------------------------
# OVF time parsing (optional)
# ----------------------------

_TIME_RE = re.compile(r"Total simulation time:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*s")


def try_parse_time_seconds_from_ovf(path: Path) -> Optional[float]:
    """Try to parse 'Total simulation time' from OVF header."""
    try:
        with open(path, "rb") as f:
            raw = f.read(64 * 1024)
        text = raw.decode("utf-8", errors="ignore")
        # Only scan header-ish region
        cut = text.find("# Begin:")
        if cut != -1:
            text = text[:cut]
        m = _TIME_RE.search(text)
        if m:
            return float(m.group(1))
    except Exception:
        return None
    return None


# ----------------------------
# Field -> PyVista conversion
# ----------------------------

def load_df_field_slice(ovf_path: Path) -> df.Field:
    """Load OVF and return a 2D-ish slice suitable for plotting."""
    field = df.Field.from_file(str(ovf_path))

    # Thin film heuristic: if z extent is much smaller than x extent, take z slice
    mesh_dims = field.mesh.region.edges
    if mesh_dims[2] < mesh_dims[0] / 10:
        return field.sel("z")

    z_center = field.mesh.region.center[2]
    return field.sel(z=z_center)


def _extract_vector_array(m: df.Field, nx: int, ny: int) -> np.ndarray:
    """Get vector array with shape (nx, ny, 3)."""
    # discretisedfield commonly exposes .array; keep fallbacks
    arr = None
    if hasattr(m, "array"):
        arr = np.asarray(m.array)
    if arr is None:
        try:
            arr = np.asarray(cast(Any, m).asarray())
        except Exception as e:
            raise RuntimeError(f"Could not extract numpy array from discretisedfield Field: {e}")

    # Shapes we might see:
    #   (nx, ny, 3)
    #   (nx, ny, 1, 3)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        arr = arr[:, :, 0, :]  # drop z=0
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Unexpected field array shape: {arr.shape}")

    if arr.shape[0] != nx or arr.shape[1] != ny:
        raise ValueError(f"Array shape {arr.shape} does not match mesh (nx, ny)=({nx},{ny})")

    return arr


def df_to_structured_grid(m: df.Field, units: str = "nm") -> Tuple[pv.StructuredGrid, dict]:
    """Convert a discretisedfield 2D slice to a PyVista StructuredGrid.

    We create points at cell centers and attach point_data:
      mx, my, mz, m_mag, m_vec (in-plane vector)
    """
    n = m.mesh.n
    if len(n) == 3:
        nx, ny, _ = n
    elif len(n) == 2:
        nx, ny = n
    else:
        raise ValueError(f"Unexpected mesh.n: {n}")

    cell = m.mesh.cell
    if len(cell) == 3:
        dx, dy, dz = cell
    elif len(cell) == 2:
        dx, dy = cell
        dz = 0.0
    else:
        raise ValueError(f"Unexpected mesh.cell length: {len(cell)} (value: {cell})")

    try:
        xmin, ymin, _ = m.mesh.region.pmin
    except Exception:
        xmin, ymin = 0.0, 0.0

    scale = 1e9 if units == "nm" else 1.0

    xs = (xmin + (np.arange(nx) + 0.5) * dx) * scale
    ys = (ymin + (np.arange(ny) + 0.5) * dy) * scale
    X, Y = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)
    Z = np.zeros_like(X)  # planar

    # PyVista stubs expect coordinates passed into the constructor.
    # Build 3D arrays of shape (nx, ny, 1) for a single z-slice.
    X3 = X[:, :, None]
    Y3 = Y[:, :, None]
    Z3 = Z[:, :, None]

    grid = pv.StructuredGrid(X3, Y3, Z3)

    vec = _extract_vector_array(m, nx, ny)  # (nx, ny, 3)

    mx = vec[:, :, 0].ravel(order="F")
    my = vec[:, :, 1].ravel(order="F")
    mz = vec[:, :, 2].ravel(order="F")
    m_mag = np.sqrt(mx * mx + my * my + mz * mz)

    m_vec = np.column_stack([mx, my, np.zeros_like(mx)])  # in-plane arrows

    grid.point_data["mx"] = mx
    grid.point_data["my"] = my
    grid.point_data["mz"] = mz
    grid.point_data["m_mag"] = m_mag
    grid.point_data["m_vec"] = m_vec

    meta = {"nx": nx, "ny": ny, "dx": dx, "dy": dy, "dz": dz, "units": units}
    return grid, meta


def make_glyphs(grid: pv.StructuredGrid, stride: int, scale: float) -> pv.PolyData:
    """Create downsampled glyph arrows from grid.point_data['m_vec']."""
    stride = max(1, int(stride))

    nx, ny, nz = grid.dimensions
    if nz != 1:
        # For SP4 we expect a single plane, but keep safe
        pass

    ii = np.arange(0, nx, stride)
    jj = np.arange(0, ny, stride)

    # Flatten index into Fortran-ordered points (i-fastest)
    ids = []
    for j in jj:
        for i in ii:
            ids.append(i + nx * j)
    ids = np.array(ids, dtype=int)

    # Construct PolyData with points in the constructor (Pylance-friendly)
    pts = grid.points[ids]
    poly = pv.PolyData(pts)

    m_vec = np.asarray(grid.point_data["m_vec"])[ids]
    m_mag = np.asarray(grid.point_data["m_mag"])[ids]

    # Attach arrays as point data
    poly["m_vec"] = m_vec
    poly["m_mag"] = m_mag

    glyphs = cast(pv.PolyData, poly.glyph(orient="m_vec", scale="m_mag", factor=float(scale)))
    return glyphs


# ----------------------------
# Interactive viewer
# ----------------------------

class MagViewer:
    def __init__(
        self,
        ovf_files: List[Path],
        component: str = "mx",
        glyphs_on: bool = True,
        glyph_stride: int = 0,
        glyph_scale: float = 0.0,
        units: str = "nm",
        use_qt: bool = False,
    ):
        self.ovf_files = ovf_files
        self.idx = 0

        self.component = component
        self.glyphs_on = glyphs_on
        # glyph_stride=0 => auto based on mesh size
        self.glyph_stride = int(glyph_stride)
        # glyph_scale<=0 => auto based on mesh spacing and stride
        self.glyph_scale = float(glyph_scale)
        self.units = units

        self.use_qt = bool(use_qt)
        if self.use_qt and _HAVE_PYVISTAQT and BackgroundPlotter is not None:
            # show=False so we keep a similar lifecycle to pv.Plotter.show()
            self.plotter = BackgroundPlotter(show=False)  # type: ignore
        else:
            self.use_qt = False
            self.plotter = pv.Plotter()
        # Set an explicit background each time to avoid occasional magenta fallback rendering
        # that can appear on macOS/VTK during transient OpenGL context issues.
        try:
            pv.set_plot_theme("document")
        except Exception:
            pass
        try:
            plotter = cast(Any, self.plotter)
            plotter.set_background("white")
        except Exception:
            pass

        # Track actors so we can remove them without calling plotter.clear().
        # (plotter.clear() can detach the renderer from the interactor on macOS/VTK,
        #  and is correlated with transient full-window magenta frames.)
        self._mesh_actor: Any = None
        self._glyph_actor: Any = None
        self._text_actor: Any = None

    def _status_text(self, path: Path) -> str:
        t = try_parse_time_seconds_from_ovf(path)
        t_str = f"{t:.3e} s" if t is not None else "unknown t"
        return (
            f"{path.name}  |  t = {t_str}\n"
            f"Color: {self.component}  |  Glyphs: {'ON' if self.glyphs_on else 'OFF'}\n"
            f"Stride: {self.glyph_stride}  |  Glyph scale: {self.glyph_scale}  |  Units: {self.units}\n"
            "Keys: n/p next/prev, 1/2/3 mx/my/mz, g glyphs, [/] density, r reset, q quit"
        )

    def _load_grid(self, path: Path) -> pv.StructuredGrid:
        m = load_df_field_slice(path)
        grid, _meta = df_to_structured_grid(m, units=self.units)
        return grid

    def _render(self, reset_camera: bool = False):
        path = self.ovf_files[self.idx]
        grid = self._load_grid(path)

        # Remove previous actors instead of calling plotter.clear().
        # This keeps the renderer attached to the interactor and is more stable on macOS/VTK.
        if self._mesh_actor is not None:
            try:
                self.plotter.remove_actor(self._mesh_actor)
            except Exception:
                pass
            self._mesh_actor = None

        if self._glyph_actor is not None:
            try:
                self.plotter.remove_actor(self._glyph_actor)
            except Exception:
                pass
            self._glyph_actor = None

        if self._text_actor is not None:
            try:
                self.plotter.remove_actor(self._text_actor)
            except Exception:
                pass
            self._text_actor = None

        # Best-effort: remove any existing scalar bar (prevents accumulation across refreshes)
        try:
            cast(Any, self.plotter).remove_scalar_bar()
        except Exception:
            pass

        # Scalar mesh
        scalars = self.component
        if scalars not in grid.point_data:
            scalars = "mx"
            self.component = "mx"

        # Robust colour limits.
        # For mz in thin films, values are often extremely close to 0; a near-degenerate
        # scalar range can trigger rendering glitches on some macOS/VTK/OpenGL paths.
        try:
            arr = np.asarray(grid.point_data[scalars], dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            if scalars == "mz":
                # Use symmetric limits around zero, but enforce a small floor so the
                # range never collapses to ~0.
                maxabs = float(np.max(np.abs(arr)))
                floor = 1e-6  # unit magnetisation floor; preserves real mz patterns when present
                maxabs = max(maxabs, floor)
                clim = (-maxabs, maxabs)
            else:
                vmin = float(arr.min())
                vmax = float(arr.max())
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    vmin, vmax = -1.0, 1.0
                if vmin == vmax:
                    eps = 1e-12
                    vmin -= eps
                    vmax += eps
                clim = (vmin, vmax)
        except Exception:
            clim = None

        self._mesh_actor = self.plotter.add_mesh(
            grid,
            scalars=scalars,
            cmap="viridis",
            show_edges=False,
            nan_color="white",
            clim=clim,
        )

        # Ensure the interactor style has a current renderer (reduces
        # "no current renderer" warnings that correlate with magenta frames).
        try:
            iren = cast(Any, self.plotter).iren
            style = getattr(iren, "style", None)
            ren = getattr(self.plotter, "renderer", None)
            if style is not None and ren is not None:
                style.SetDefaultRenderer(ren)
                style.SetCurrentRenderer(ren)
        except Exception:
            pass

        # Auto glyph stride/scale (keeps glyphs visible for small grids like FMR)
        nx, ny, _ = grid.dimensions
        if self.glyph_stride <= 0:
            # Aim for ~10–15 arrows across the smaller dimension
            stride_eff = max(1, int(round(min(nx, ny) / 12.0)))
        else:
            stride_eff = max(1, int(self.glyph_stride))

        if self.glyph_scale <= 0.0:
            # Estimate spacing from adjacent points (units already applied: nm or m)
            try:
                pts = grid.points
                dx_est = float(np.linalg.norm(pts[1] - pts[0])) if pts.shape[0] > 1 else 1.0
            except Exception:
                dx_est = 1.0
            scale_eff = 0.8 * dx_est * float(stride_eff)
        else:
            scale_eff = float(self.glyph_scale)

        # Glyph overlay
        if self.glyphs_on:
            glyphs = make_glyphs(grid, stride=stride_eff, scale=scale_eff)
            self._glyph_actor = self.plotter.add_mesh(
                glyphs,
                color="white",
                opacity=0.95,
                lighting=False,
                nan_color="white",
            )
        else:
            self._glyph_actor = None

        # Text overlay
        self._text_actor = self.plotter.add_text(
            self._status_text(path),
            font_size=11,
            position="upper_left",
        )

        # Camera
        if reset_camera:
            plotter = cast(Any, self.plotter)
            plotter.view_xy()
            plotter.reset_camera()

        self.plotter.render()

    # --- Key handlers (wrapped with lambdas in add_key_event) ---

    def next(self):
        self.idx = (self.idx + 1) % len(self.ovf_files)
        self._render(reset_camera=False)

    def prev(self):
        self.idx = (self.idx - 1) % len(self.ovf_files)
        self._render(reset_camera=False)

    def set_component(self, comp: str):
        self.component = comp
        self._render(reset_camera=False)

    def toggle_glyphs(self):
        self.glyphs_on = not self.glyphs_on
        self._render(reset_camera=False)

    def more_dense(self):
        # '[' -> smaller stride -> more arrows
        self.glyph_stride = max(1, self.glyph_stride // 2)
        self._render(reset_camera=False)

    def less_dense(self):
        # ']' -> larger stride -> fewer arrows
        self.glyph_stride = min(2048, self.glyph_stride * 2)
        self._render(reset_camera=False)

    def reset_camera(self):
        self._render(reset_camera=True)

    def run(self):
        # Initial render
        self._render(reset_camera=True)

        # Bind keys.
        # PyVista's type stubs can confuse Pylance here, so cast once to Any.
        add_key_event = cast(Any, self.plotter).add_key_event

        add_key_event("n", lambda: self.next())
        add_key_event("p", lambda: self.prev())

        add_key_event("1", lambda: self.set_component("mx"))
        add_key_event("2", lambda: self.set_component("my"))
        add_key_event("3", lambda: self.set_component("mz"))

        add_key_event("g", lambda: self.toggle_glyphs())
        add_key_event("[", lambda: self.more_dense())
        add_key_event("]", lambda: self.less_dense())

        add_key_event("r", lambda: self.reset_camera())

        # More stable quit handling on macOS/VTK:
        # Avoid calling plotter.close() directly from inside the VTK event loop.
        # Instead, request the interactor to terminate; then close in finally.
        def request_quit() -> None:
            try:
                iren = cast(Any, self.plotter).iren
                if iren is not None:
                    iren.TerminateApp()
                app = getattr(self.plotter, "app", None)
                if app is not None:
                    try:
                        app.quit()
                    except Exception:
                        pass
            except Exception:
                pass

        add_key_event("q", request_quit)
        add_key_event("Escape", request_quit)

        # Show the window. If using the Qt backend, run the Qt event loop.
        try:
            self.plotter.show()
            if self.use_qt:
                app = getattr(self.plotter, "app", None)
                if app is not None:
                    # Qt6 uses exec(); older bindings sometimes provide exec_()
                    if hasattr(app, "exec"):
                        app.exec()
                    elif hasattr(app, "exec_"):
                        app.exec_()
                    return
        except AttributeError as e:
            if "IsCurrent" in str(e) and "NoneType" in str(e):
                return
            raise
        finally:
            try:
                self.plotter.close()
            except Exception:
                pass


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive OVF viewer (PyVista) for SP4 (MuMax3 or Rust).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MuMax SP4 root
  python3 scripts/mag_viewer.py --input mumax_outputs/st_problems/sp4 --case sp4a

  # Rust SP4 root
  python3 scripts/mag_viewer.py --input runs/st_problems/sp4 --case sp4b

  # Direct case folder
  python3 scripts/mag_viewer.py --input mumax_outputs/st_problems/sp4/sp4a_out
        """,
    )
    parser.add_argument(
        "--qt",
        action="store_true",
        help="Use the Qt backend via pyvistaqt.BackgroundPlotter (requires pyvistaqt).",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="SP4 root folder or direct case folder containing m*.ovf.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Case selection. For SP4: sp4a/sp4b. For SP2: remanence/coercivity (or rem/hc).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["auto", "mumax", "rust"],
        default="auto",
        help="Override source detection (default: auto).",
    )

    parser.add_argument(
        "--component",
        type=str,
        choices=["mx", "my", "mz"],
        default="mx",
        help="Initial component to color by (default: mx).",
    )
    parser.add_argument(
        "--no-glyphs",
        action="store_true",
        help="Start with glyph arrows disabled.",
    )
    parser.add_argument(
        "--glyph-stride",
        type=int,
        default=0,
        help="Glyph downsample stride. 0 = auto (recommended). Bigger = fewer arrows.",
    )
    parser.add_argument(
        "--glyph-scale",
        type=float,
        default=0.0,
        help="Glyph scale factor (arrow size). 0 = auto (recommended).",
    )
    parser.add_argument(
        "--units",
        type=str,
        choices=["nm", "m"],
        default="nm",
        help="Axis units (default: nm).",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}")
        return 1

    # Determine source (informational / parity with mag_visualisation)
    if args.source == "auto":
        source = infer_source(input_path)
    else:
        source = args.source

    case_dirs = find_ovf_case_dirs(input_path)
    if not case_dirs:
        print(f"Error: no OVF files found in input path or its known subdirectories: {input_path}")
        return 1

    try:
        case_dir = pick_case_dir(case_dirs, args.case)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    ovf_files = sorted(case_dir.glob("m*.ovf"))
    # SP2 virtual cases: treat remanence vs coercivity as separate sequences
    if is_sp2_folder(case_dir):
        want_case = args.case or "remanence"
        ovf_files = filter_sp2_case(ovf_files, want_case)
    if not ovf_files:
        if is_sp2_folder(case_dir):
            print(f"Error: no OVFs found for requested SP2 case '{args.case}'. Try --case remanence or --case coercivity.")
        else:
            print(f"Error: no m*.ovf files found in: {case_dir}")
        return 1

    print("=" * 70)
    print(f"Mag Viewer - Standard Problem 4 ({source})")
    print("=" * 70)
    print(f"Input:      {input_path}")
    print(f"Case dir:   {case_label_from_dirname(case_dir.name)}  ({case_dir})")
    print(f"Snapshots:  {len(ovf_files)}")
    print(f"Units:      {args.units}")
    if is_sp2_folder(case_dir):
        print(f"SP2 view:   {args.case or 'remanence'}")
    print("=" * 70)

    viewer = MagViewer(
        ovf_files=ovf_files,
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
