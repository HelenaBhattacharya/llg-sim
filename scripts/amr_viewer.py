#!/usr/bin/env python3
"""
amr_viewer.py — lightweight OVF post-processor for AMR benchmarks.

Reads OVF 2.0 ASCII files written by `amr_vortex_relax.rs` (and similar),
and generates:

1) In-plane angle map (HSV wheel) for coarse / fine / amr
2) m_z heatmap for coarse / fine / amr
3) Optional 3D warped surface (PyVista preferred; matplotlib fallback)

Expected folder layout (from `--root out/amr_vortex_relax`):
  out/amr_vortex_relax/
    ovf_coarse/m0000300.ovf
    ovf_fine/m0000300.ovf
    ovf_amr/m0000300.ovf

Run examples:
  # generate OVFs from rust
  #   LLG_AMR_MAX_LEVEL=2 cargo run --release --bin amr_vortex_relax -- --ovf
  # then:
  python3 scripts/amr_viewer.py --root out/amr_vortex_relax --step 300

  # latest step automatically
  python3 scripts/amr_viewer.py --root out/amr_vortex_relax --step latest

  # also make 3D warp (offscreen screenshot)
  python3 scripts/amr_viewer.py --root out/amr_vortex_relax --step 300 --warp

  # interactive PyVista window (if available)
  python3 scripts/amr_viewer.py --root out/amr_vortex_relax --step 300 --warp --show

  # 3D warp with mesh overlay from regrid_levels.csv
  python3 scripts/amr_viewer.py --root out/amr_vortex_relax --step 300 --warp --mesh-overlay
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, Literal, cast, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# For type checking imshow origin
Origin = Literal["lower", "upper"]


# ----------------------------
# OVF loading
# ----------------------------

@dataclass(frozen=True)
class OvfMeta:
    nx: int
    ny: int
    dx: float
    dy: float
    dz: float
    xbase: float
    ybase: float
    zbase: float
    title: str


@dataclass
class OvfField:
    meta: OvfMeta
    m: np.ndarray  # shape (ny, nx, 3), float64


_OVF_KV_RE = re.compile(r"^\#\s*([^:]+)\s*:\s*(.*)$")


def _parse_header_kv(lines: List[str]) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for ln in lines:
        m = _OVF_KV_RE.match(ln)
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        kv[key] = val
    return kv


def load_ovf_text(path: Path) -> OvfField:
    """
    Minimal OVF 2.0 ASCII reader for the files produced by our Rust writer.
    Assumes:
      - '# Begin: Data Text' ... '# End: Data Text'
      - 3 floats per line
      - ordering x fastest, then y (matching our writer)
    """
    txt = path.read_text().splitlines()

    # Split header / data
    try:
        i_data0 = next(i for i, ln in enumerate(txt) if ln.strip() == "# Begin: Data Text")
        i_data1 = next(i for i, ln in enumerate(txt) if ln.strip() == "# End: Data Text")
    except StopIteration as e:
        raise ValueError(f"{path}: couldn't find '# Begin: Data Text' / '# End: Data Text'") from e

    header_lines = txt[: i_data0]
    data_lines = txt[i_data0 + 1 : i_data1]

    kv = _parse_header_kv(header_lines)

    def get_int(k: str) -> int:
        return int(kv[k])

    def get_float(k: str) -> float:
        return float(kv[k])

    nx = get_int("xnodes")
    ny = get_int("ynodes")
    dx = get_float("xstepsize")
    dy = get_float("ystepsize")
    dz = get_float("zstepsize")
    xbase = get_float("xbase")
    ybase = get_float("ybase")
    zbase = get_float("zbase")
    title = kv.get("title", path.name)

    floats: List[float] = []
    for ln in data_lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        floats.extend([float(parts[0]), float(parts[1]), float(parts[2])])

    arr = np.asarray(floats, dtype=np.float64)
    expected = nx * ny * 3
    if arr.size != expected:
        raise ValueError(f"{path}: expected {expected} floats, got {arr.size}")

    m = arr.reshape((ny, nx, 3))
    meta = OvfMeta(nx=nx, ny=ny, dx=dx, dy=dy, dz=dz, xbase=xbase, ybase=ybase, zbase=zbase, title=title)
    return OvfField(meta=meta, m=m)


def find_latest_step(folder: Path) -> int:
    """
    Finds the largest step number from files named mXXXXXXX.ovf in folder.
    """
    steps = []
    for p in folder.glob("m*.ovf"):
        m = re.match(r"m(\d+)\.ovf$", p.name)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        raise FileNotFoundError(f"No OVF files found in {folder}")
    return max(steps)


@dataclass(frozen=True)
class Rect:
    i0: int
    j0: int
    nx: int
    ny: int

    @property
    def i1(self) -> int:
        return self.i0 + self.nx

    @property
    def j1(self) -> int:
        return self.j0 + self.ny


def load_regrid_levels_csv(path: Path) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    if not path.exists():
        return rows
    lines = path.read_text().splitlines()
    if not lines:
        return rows
    header = lines[0].strip().split(",")
    idx = {k: i for i, k in enumerate(header)}

    def get_int(parts: List[str], key: str, default: int = 0) -> int:
        if key not in idx:
            return default
        try:
            return int(parts[idx[key]])
        except Exception:
            return default

    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        parts = s.split(",")
        if len(parts) < len(header):
            continue
        row = {"step": get_int(parts, "step", 0)}
        # L1 union
        row.update({
            "l1_i0": get_int(parts, "l1_i0", 0),
            "l1_j0": get_int(parts, "l1_j0", 0),
            "l1_nx": get_int(parts, "l1_nx", 0),
            "l1_ny": get_int(parts, "l1_ny", 0),
            "l2_i0": get_int(parts, "l2_i0", 0),
            "l2_j0": get_int(parts, "l2_j0", 0),
            "l2_nx": get_int(parts, "l2_nx", 0),
            "l2_ny": get_int(parts, "l2_ny", 0),
            "l3_i0": get_int(parts, "l3_i0", 0),
            "l3_j0": get_int(parts, "l3_j0", 0),
            "l3_nx": get_int(parts, "l3_nx", 0),
            "l3_ny": get_int(parts, "l3_ny", 0),
        })
        rows.append(row)

    rows.sort(key=lambda r: r["step"])
    return rows


def rects_for_step(rows: List[Dict[str, int]], step: int) -> Tuple[Optional[Rect], Optional[Rect], Optional[Rect]]:
    """Return (L1_union_rect, L2_union_rect, L3_union_rect) for the latest row with row.step <= step."""
    if not rows:
        return None, None, None
    chosen = rows[0]
    for r in rows:
        if r["step"] <= step:
            chosen = r
        else:
            break

    def mk(prefix: str) -> Optional[Rect]:
        nx = chosen.get(f"{prefix}_nx", 0)
        ny = chosen.get(f"{prefix}_ny", 0)
        if nx <= 0 or ny <= 0:
            return None
        return Rect(
            i0=chosen.get(f"{prefix}_i0", 0),
            j0=chosen.get(f"{prefix}_j0", 0),
            nx=nx,
            ny=ny,
        )

    return mk("l1"), mk("l2"), mk("l3")

def load_regrid_patches_csv(path: Path) -> List[Dict[str, int]]:
    """
    step,level,patch_id,i0,j0,nx,ny
    """
    rows: List[Dict[str, int]] = []
    if not path.exists():
        return rows
    lines = path.read_text().splitlines()
    if not lines:
        return rows

    header = lines[0].strip().split(",")
    idx = {k: i for i, k in enumerate(header)}

    def get_int(parts: List[str], key: str, default: int = 0) -> int:
        if key not in idx:
            return default
        try:
            return int(parts[idx[key]])
        except Exception:
            return default

    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        parts = s.split(",")
        if len(parts) < len(header):
            continue
        rows.append({
            "step": get_int(parts, "step", 0),
            "level": get_int(parts, "level", 0),
            "patch_id": get_int(parts, "patch_id", 0),
            "i0": get_int(parts, "i0", 0),
            "j0": get_int(parts, "j0", 0),
            "nx": get_int(parts, "nx", 0),
            "ny": get_int(parts, "ny", 0),
        })

    rows.sort(key=lambda r: (r["step"], r["level"], r["patch_id"]))
    return rows


def patch_rects_for_step(rows: List[Dict[str, int]], step: int, max_level: int = 3) -> Dict[int, List[Rect]]:
    """
    Return level -> list of Rect for the latest step <= requested.
    """
    out: Dict[int, List[Rect]] = {lvl: [] for lvl in range(1, max_level + 1)}
    if not rows:
        return out

    # Find the latest logged step <= requested
    latest = None
    for r in rows:
        if r["step"] <= step:
            latest = r["step"]
        else:
            break
    if latest is None:
        latest = rows[0]["step"]

    for r in rows:
        if r["step"] != latest:
            continue
        lvl = r["level"]
        if lvl < 1 or lvl > max_level:
            continue
        nx = r["nx"]
        ny = r["ny"]
        if nx <= 0 or ny <= 0:
            continue
        out[lvl].append(Rect(i0=r["i0"], j0=r["j0"], nx=nx, ny=ny))

    return out

# ----------------------------
# Plotting utilities
# ----------------------------

def angle_rgb(m: np.ndarray) -> np.ndarray:
    """
    m: (ny,nx,3). Returns RGB image (ny,nx,3) mapping in-plane angle to hue.
    """
    mx = m[..., 0]
    my = m[..., 1]
    phi = np.arctan2(my, mx)  # [-pi,pi]
    h = (phi + np.pi) / (2.0 * np.pi)  # [0,1)
    s = np.ones_like(h)
    v = np.ones_like(h)
    hsv = np.stack([h, s, v], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return rgb


def save_angle_map(path: Path, field: OvfField, origin: Origin = "lower") -> None:
    rgb = angle_rgb(field.m)
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb, origin=origin)
    plt.title(f"{field.meta.title} — in-plane angle")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_mz_map(path: Path, field: OvfField, origin: Origin = "lower", vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    mz = field.m[..., 2]
    plt.figure(figsize=(6, 6))
    im = plt.imshow(mz, origin=origin, vmin=vmin, vmax=vmax)
    plt.title(f"{field.meta.title} — m_z")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_diff_map(path: Path, a: OvfField, b: OvfField, origin: Origin = "lower") -> None:
    """
    Saves |Δm| map between fields a and b (must have same grid).
    """
    if (a.meta.nx, a.meta.ny, a.meta.dx, a.meta.dy) != (b.meta.nx, b.meta.ny, b.meta.dx, b.meta.dy):
        raise ValueError("Diff requires same grid (nx,ny,dx,dy)")
    dm = np.linalg.norm(a.m - b.m, axis=-1)
    plt.figure(figsize=(6, 6))
    im = plt.imshow(dm, origin=origin)
    plt.title(f"|Δm|: {a.meta.title} vs {b.meta.title}")
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ----------------------------
# 3D warp (PyVista preferred)
# ----------------------------

def try_import_pyvista():
    try:
        import pyvista as pv  # type: ignore
        return pv
    except Exception:
        return None


def warp_surface_pyvista(
    out_png: Path,
    field: OvfField,
    warp_scalar: str = "mz",
    color_by: str = "angle",
    warp_scale: float = 50.0,
    show: bool = False,
    offscreen: bool = True,
    mesh_overlay: bool = False,
    coarse_field: Optional[OvfField] = None,
    l1_rects: Optional[List[Rect]] = None,
    l2_rects: Optional[List[Rect]] = None,
    l3_rects: Optional[List[Rect]] = None,
    args_local: Any = None,
) -> None:
    pv = try_import_pyvista()
    if pv is None:
        raise RuntimeError("pyvista not available. Install pyvista or use --warp-mpl.")

    ny, nx = field.meta.ny, field.meta.nx
    dx, dy = field.meta.dx, field.meta.dy
    x0, y0 = field.meta.xbase, field.meta.ybase

    # Build a uniform grid in *normalized* coordinates (x/L, y/L) to avoid camera/scale issues
    # when x,y are in meters and extremely small.
    #
    # We store scalars on points (nx*ny points, znodes=1).
    pv_any: Any = pv
    grid = pv_any.ImageData(
        dimensions=(nx, ny, 1),
        spacing=(1.0 / max(nx, 1), 1.0 / max(ny, 1), 1.0),
        origin=(0.0, 0.0, 0.0),
    )

    # Update coordinate parameters to match the normalized grid for overlay helpers.
    x0, y0 = 0.0, 0.0
    dx, dy = 1.0 / max(nx, 1), 1.0 / max(ny, 1)

    m = field.m
    mx, my, mz = m[..., 0], m[..., 1], m[..., 2]

    if warp_scalar == "mz":
        scalar = mz
    elif warp_scalar == "mxy":
        scalar = np.sqrt(mx * mx + my * my)
    elif warp_scalar == "mdotz":
        scalar = mz  # alias
    else:
        raise ValueError(f"Unknown warp_scalar: {warp_scalar}")

    if color_by == "mz":
        color = mz
    elif color_by == "mxy":
        color = np.sqrt(mx * mx + my * my)
    elif color_by == "angle":
        phi = np.arctan2(my, mx)
        color = (phi + np.pi) / (2.0 * np.pi)  # hue scalar
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    # Attach scalars to points
    grid["warp"] = scalar.ravel(order="C")
    grid["color"] = color.ravel(order="C")

    warped = grid.warp_by_scalar("warp", factor=warp_scale)
    warped_any: Any = warped

    pl = pv_any.Plotter(off_screen=(offscreen and not show), window_size=[1000, 800])
    surface_actor: Any = pl.add_mesh(
        warped_any,
        scalars="color",
        cmap="hsv" if color_by == "angle" else "viridis",
        show_scalar_bar=True,
        scalar_bar_args={"title": color_by},
    )
    axes_actor: Any = pl.add_axes()

    # Best-effort scalar bar actor handle (depends on PyVista version)
    scalar_bar_actor: Any = getattr(pl, "scalar_bar", None)
    if scalar_bar_actor is None:
        try:
            sb = getattr(pl, "scalar_bars", None)
            if isinstance(sb, dict) and len(sb) > 0:
                scalar_bar_actor = list(sb.values())[0]
        except Exception:
            scalar_bar_actor = None

    # Robust camera setup: fit to data.
    pl.view_isometric()
    pl.reset_camera()

    def z_at_xy(x: float, y: float) -> float:
        # Using normalized coordinates: x,y in [0,1].
        ii = int(np.clip(round((x - x0) / dx), 0, nx - 1))
        jj = int(np.clip(round((y - y0) / dy), 0, ny - 1))
        return float(warp_scale * scalar[jj, ii])

    def add_polyline(points_xy: List[Tuple[float, float]], color_name: str, width: int = 2, z_eps: float = 0.01):
        pts = []
        for (px, py) in points_xy:
            pz = z_at_xy(px, py) + z_eps
            pts.append([px, py, pz])
        pts_np = np.asarray(pts, dtype=float)
        poly = pv_any.PolyData(pts_np)
        # One polyline cell
        npts = pts_np.shape[0]
        poly.lines = np.hstack([[npts], np.arange(npts)]).astype(np.int64)
        return pl.add_mesh(poly, color=color_name, line_width=width)

    def add_rect_outline(rect: Rect, color_name: str, width: int = 3):
        # Rect in base-grid indices; convert to boundary coordinates in normalized units
        if coarse_field is None:
            return
        nx0 = coarse_field.meta.nx
        ratio_total = field.meta.nx // nx0
        if field.meta.nx % nx0 != 0: return None
        # Boundary indices in finest grid
        fi0 = rect.i0 * ratio_total
        fj0 = rect.j0 * ratio_total
        fi1 = rect.i1 * ratio_total
        fj1 = rect.j1 * ratio_total
        # Boundary coords (normalized)
        xb0 = (fi0) / field.meta.nx
        yb0 = (fj0) / field.meta.ny
        xb1 = (fi1) / field.meta.nx
        yb1 = (fj1) / field.meta.ny
        pts = [(xb0, yb0), (xb1, yb0), (xb1, yb1), (xb0, yb1), (xb0, yb0)]
        return add_polyline(pts, color_name, width=width)

    def add_grid_in_rect(rect: Rect, level: int, color_name: str, width: int = 1, stride_min: int = 1):
        # Draw grid lines for level=1 (L1) or level=2 (L2) inside rect.
        if coarse_field is None:
            return []
        nx0 = coarse_field.meta.nx
        ratio_total = field.meta.nx // nx0
        base_ratio = 2
        # spacing in finest indices: s = ratio_total / (base_ratio^level)
        s = max(ratio_total // (base_ratio ** level), stride_min)

        fi0 = rect.i0 * ratio_total
        fj0 = rect.j0 * ratio_total
        fi1 = rect.i1 * ratio_total
        fj1 = rect.j1 * ratio_total

        xb0 = (fi0) / field.meta.nx
        yb0 = (fj0) / field.meta.ny
        xb1 = (fi1) / field.meta.nx
        yb1 = (fj1) / field.meta.ny

        actors: List[Any] = []
        # Vertical lines
        k = fi0
        while k <= fi1:
            xk = k / field.meta.nx
            actors.append(add_polyline([(xk, yb0), (xk, yb1)], color_name, width=width))
            k += s
        # Horizontal lines
        k = fj0
        while k <= fj1:
            yk = k / field.meta.ny
            actors.append(add_polyline([(xb0, yk), (xb1, yk)], color_name, width=width))
            k += s
        return actors

    overlay_actors: List[Any] = []
    glyph_actor: Optional[Any] = None

    if mesh_overlay and coarse_field is not None:
        # Draw L0 (coarse) grid across the full domain, plus L1/L2 grids inside their rects.
        # For L0, we intentionally sub-sample (every ~8 coarse cells) to keep the 3D view readable.

        nx0 = coarse_field.meta.nx
        ny0 = coarse_field.meta.ny
        ratio_total = field.meta.nx // max(nx0, 1)
        # draw every N coarse cells
        l0_stride_coarse = 8
        l0_stride_fine = max(1, ratio_total * l0_stride_coarse)

        full_rect = Rect(i0=0, j0=0, nx=nx0, ny=ny0)
        overlay_actors.extend(
            add_grid_in_rect(full_rect, level=0, color_name="lightgray", width=2, stride_min=l0_stride_fine)
        )

        # L1/L2/L3 patches (per-patch lists)
        if l1_rects:
            for r in l1_rects:
                a = add_rect_outline(r, "black", width=3)
                if a is not None:
                    overlay_actors.append(a)
                overlay_actors.extend(add_grid_in_rect(r, level=1, color_name="black", width=1))

        if l2_rects:
            for r in l2_rects:
                a = add_rect_outline(r, "red", width=3)
                if a is not None:
                    overlay_actors.append(a)
                overlay_actors.extend(add_grid_in_rect(r, level=2, color_name="red", width=1))

        if l3_rects:
            for r in l3_rects:
                a = add_rect_outline(r, "dodgerblue", width=3)
                if a is not None:
                    overlay_actors.append(a)
                overlay_actors.extend(add_grid_in_rect(r, level=3, color_name="dodgerblue", width=1))
    # (Glyphs are handled lazily in interactive toggles below)

    # Interactive toggles (only meaningful for --show)
    if show:
        # Help text (mag_viewer-style)
        help_text = (
            "Keys: m=mesh  g=glyphs  a=axes  b=scalarbar  w=wireframe  r=reset  h=help  q=quit"
        )
        help_actor: Any = pl.add_text(help_text, position="upper_left", font_size=12, color="black")

        # Robust visibility helpers (PyVista actors vary by version)
        def _set_visible_one(act: Any, vis: bool) -> None:
            if act is None:
                return
            if hasattr(act, "visibility"):
                try:
                    act.visibility = vis
                    return
                except Exception:
                    pass
            if hasattr(act, "SetVisibility"):
                try:
                    act.SetVisibility(vis)
                    return
                except Exception:
                    pass
            if hasattr(act, "actor") and hasattr(act.actor, "SetVisibility"):
                try:
                    act.actor.SetVisibility(vis)
                    return
                except Exception:
                    pass

        def _set_visible_many(acts: List[Any], vis: bool) -> None:
            for a in acts:
                _set_visible_one(a, vis)

        # Glyphs are created lazily on first toggle (default OFF)
        glyph_actor: Optional[Any] = None

        def ensure_glyph_actor() -> Any:
            nonlocal glyph_actor
            if glyph_actor is not None:
                return glyph_actor

            stride = int(getattr(args_local, "glyph_stride", 16) if args_local is not None else 16)
            scale = float(getattr(args_local, "glyph_scale", 0.02) if args_local is not None else 0.02)

            pts: List[List[float]] = []
            vecs: List[List[float]] = []
            for jj in range(0, ny, stride):
                for ii in range(0, nx, stride):
                    pts.append([ii / nx, jj / ny, float(warp_scale * scalar[jj, ii]) + 0.02])
                    vecs.append([float(mx[jj, ii]), float(my[jj, ii]), float(mz[jj, ii])])

            pts_np = np.asarray(pts, dtype=float)
            vec_np = np.asarray(vecs, dtype=float)
            cloud = pv_any.PolyData(pts_np)
            cloud["vec"] = vec_np
            # Slim arrow geometry for clearer dense plots
            arrow = pv_any.Arrow(tip_length=0.3, tip_radius=0.06, shaft_radius=0.02)
            glyphs = cloud.glyph(orient="vec", scale=False, factor=scale, geom=arrow)

            glyph_actor = pl.add_mesh(glyphs, color="white")
            _set_visible_one(glyph_actor, False)  # default OFF
            return glyph_actor

        state = {
            "mesh": True,
            "glyph": False,
            "axes": True,
            "bar": True,
            "help": True,
            "wire": False,
        }

        def toggle_mesh() -> None:
            state["mesh"] = not state["mesh"]
            _set_visible_many(overlay_actors, state["mesh"])

        def toggle_glyph() -> None:
            ga = ensure_glyph_actor()
            state["glyph"] = not state["glyph"]
            _set_visible_one(ga, state["glyph"])

        def toggle_axes() -> None:
            state["axes"] = not state["axes"]
            _set_visible_one(axes_actor, state["axes"])

        def toggle_bar() -> None:
            state["bar"] = not state["bar"]
            _set_visible_one(scalar_bar_actor, state["bar"])

        def toggle_help() -> None:
            state["help"] = not state["help"]
            _set_visible_one(help_actor, state["help"])

        def toggle_wireframe() -> None:
            state["wire"] = not state["wire"]
            try:
                prop = surface_actor.GetProperty()
                if state["wire"]:
                    prop.SetRepresentationToWireframe()
                else:
                    prop.SetRepresentationToSurface()
            except Exception:
                pass

        def reset_cam() -> None:
            pl.reset_camera()

        # defaults
        _set_visible_many(overlay_actors, True)

        pl.add_key_event("m", toggle_mesh)
        pl.add_key_event("g", toggle_glyph)
        pl.add_key_event("a", toggle_axes)
        pl.add_key_event("b", toggle_bar)
        pl.add_key_event("h", toggle_help)
        pl.add_key_event("w", toggle_wireframe)
        pl.add_key_event("r", reset_cam)

    if show:
        pl.show()
    else:
        pl.screenshot(str(out_png))
        pl.close()


def warp_surface_mpl(out_png: Path, field: OvfField, warp_scale: float = 50.0) -> None:
    """
    Fallback 3D surface using matplotlib (slower, less interactive).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    ny, nx = field.meta.ny, field.meta.nx
    dx, dy = field.meta.dx, field.meta.dy
    x0, y0 = field.meta.xbase, field.meta.ybase

    xs = x0 + np.arange(nx) * dx
    ys = y0 + np.arange(ny) * dy
    X, Y = np.meshgrid(xs, ys)
    Z = warp_scale * field.m[..., 2]

    fig = plt.figure(figsize=(8, 6))
    ax: Any = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_title(f"{field.meta.title} — warped by m_z")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Benchmark output root, e.g. out/amr_vortex_relax")
    ap.add_argument("--step", type=str, default="latest", help="Step index (e.g. 0, 100, 300) or 'latest'")
    ap.add_argument("--out", type=str, default=None, help="Output folder for plots (default: <root>/plots)")
    ap.add_argument("--origin", type=str, default="lower", choices=["lower", "upper"], help="imshow origin")
    ap.add_argument("--warp", action="store_true", help="Generate 3D warp plot (PyVista if available)")
    ap.add_argument("--warp-mpl", action="store_true", help="Force matplotlib 3D warp (no PyVista)")
    ap.add_argument("--warp-scale", type=float, default=0.5, help="Warp scale factor for 3D plot")
    ap.add_argument("--show", action="store_true", help="Show interactive PyVista window (implies not offscreen)")
    ap.add_argument("--mesh-overlay", action="store_true", help="Overlay L1/L2 grid lines on the 3D warp (uses regrid_levels.csv)")
    ap.add_argument("--mesh-log", type=str, default=None, help="Path to regrid_levels.csv (default: <root>/regrid_levels.csv)")
    ap.add_argument("--glyph-stride", type=int, default=16, help="Downsample stride for glyphs (larger => fewer arrows)")
    ap.add_argument("--glyph-scale", type=float, default=0.02, help="Glyph arrow size in normalized units")
    args = ap.parse_args()
    origin: Origin = cast(Origin, args.origin)

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else (root / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    folders = {
        "coarse": root / "ovf_coarse",
        "fine": root / "ovf_fine",
        "amr": root / "ovf_amr",
    }
    for k, p in folders.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k} folder: {p}")

    if args.step == "latest":
        step = find_latest_step(folders["amr"])
    else:
        step = int(args.step)

    fname = f"m{step:07d}.ovf"
    f_coarse = folders["coarse"] / fname
    f_fine = folders["fine"] / fname
    f_amr = folders["amr"] / fname

    if not f_coarse.exists() or not f_fine.exists() or not f_amr.exists():
        raise FileNotFoundError(f"Missing one or more OVFs for step={step}: {fname}")

    coarse = load_ovf_text(f_coarse)
    fine = load_ovf_text(f_fine)
    amr = load_ovf_text(f_amr)

    l1_rects: Optional[List[Rect]] = None
    l2_rects: Optional[List[Rect]] = None
    l3_rects: Optional[List[Rect]] = None

    if args.mesh_overlay:
        patches_csv = root / "regrid_patches.csv"
        if patches_csv.exists():
            rows_p = load_regrid_patches_csv(patches_csv)
            d = patch_rects_for_step(rows_p, step, max_level=3)
            l1_rects = d.get(1, [])
            l2_rects = d.get(2, [])
            l3_rects = d.get(3, [])
        else:
            mesh_log = Path(args.mesh_log) if args.mesh_log else (root / "regrid_levels.csv")
            rows = load_regrid_levels_csv(mesh_log)
            l1, l2, l3 = rects_for_step(rows, step)
            l1_rects = [l1] if l1 is not None else []
            l2_rects = [l2] if l2 is not None else []
            l3_rects = [l3] if l3 is not None else []

    # 2D plots
    save_angle_map(out_dir / f"angle_coarse_{step:07d}.png", coarse, origin=origin)
    save_angle_map(out_dir / f"angle_fine_{step:07d}.png", fine, origin=origin)
    save_angle_map(out_dir / f"angle_amr_{step:07d}.png", amr, origin=origin)

    # Use same color limits for mz comparisons
    mz_all = np.concatenate([coarse.m[..., 2].ravel(), fine.m[..., 2].ravel(), amr.m[..., 2].ravel()])
    vmin = float(np.min(mz_all))
    vmax = float(np.max(mz_all))

    save_mz_map(out_dir / f"mz_coarse_{step:07d}.png", coarse, origin=origin, vmin=vmin, vmax=vmax)
    save_mz_map(out_dir / f"mz_fine_{step:07d}.png", fine, origin=origin, vmin=vmin, vmax=vmax)
    save_mz_map(out_dir / f"mz_amr_{step:07d}.png", amr, origin=origin, vmin=vmin, vmax=vmax)

    # Diff maps where grids match
    if (fine.meta.nx, fine.meta.ny) == (amr.meta.nx, amr.meta.ny) and (fine.meta.dx, fine.meta.dy) == (amr.meta.dx, amr.meta.dy):
        save_diff_map(out_dir / f"dm_amr_vs_fine_{step:07d}.png", amr, fine, origin=origin)

    # 3D warp plots
    if args.warp or args.warp_mpl:
        if args.warp_mpl:
            warp_surface_mpl(out_dir / f"warp_mz_amr_{step:07d}.png", amr, warp_scale=args.warp_scale)
        else:
            # Prefer PyVista
            try:
                warp_surface_pyvista(
                    out_png=out_dir / f"warp_mz_amr_{step:07d}.png",
                    field=amr,
                    warp_scalar="mz",
                    color_by="angle",
                    warp_scale=args.warp_scale,
                    show=args.show,
                    offscreen=not args.show,
                    mesh_overlay=args.mesh_overlay,
                    coarse_field=coarse,
                    l1_rects=l1_rects,
                    l2_rects=l2_rects,
                    l3_rects=l3_rects,
                    args_local=args,
                )
            except Exception as e:
                # Fallback
                print(f"[warn] PyVista warp failed ({e}); falling back to matplotlib 3D warp.")
                warp_surface_mpl(out_dir / f"warp_mz_amr_{step:07d}.png", amr, warp_scale=args.warp_scale)

    print(f"[amr_viewer] wrote plots to: {out_dir}")
    print(f"[amr_viewer] step = {step}, file = {fname}")


if __name__ == "__main__":
    main()