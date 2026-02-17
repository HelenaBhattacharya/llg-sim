#!/usr/bin/env python3
"""mag_visualisation.py — batch OVF visualisation exporter for SP2 + SP4.

This script focuses on:
  - Standard Problem 4 (SP4): time evolution (m0000000.ovf ...)
  - Standard Problem 2 (SP2):
      • sweep across d/lex (final rem + final hc)
      • single-d evolution (rem relax series and coercivity relax series)

Outputs are written under the `plots/` folder by default.

Key design goals:
  - Work with both Rust and MuMax folder layouts.
  - Keep plotting code modular and reusable.
  - Produce frame PNGs and (optionally) MP4 movies.

Dependencies:
  - numpy
  - matplotlib
  - discretisedfield
  - (optional for movie) ffmpeg in PATH

Examples
--------

SP4 (export both cases, all frames; also make mz movies):
  python3 scripts/mag_visualisation.py --input runs/st_problems/sp4 --problem sp4 --movie

SP4 (MuMax):
  python3 scripts/mag_visualisation.py --input mumax_outputs/st_problems/sp4 --problem sp4 --movie

SP2 sweep (final states across d/lex, rem + hc):
  python3 scripts/mag_visualisation.py --input runs/st_problems/sp2 --problem sp2

SP2 single d/lex evolution (coercivity relaxation frames for d=30):
  python3 scripts/mag_visualisation.py --input runs/st_problems/sp2 --problem sp2 --d 30 --stage hc --movie
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

import discretisedfield as df


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
        "Could not import ovf_utils. Expected at src/ovf_utils.py. "
        "Make sure you copied ovf_utils.py into the repo's src/ folder.\n"
        f"Import error: {e}"
    )


# -----------------------------------------------------------------------------
# Plot primitives
# -----------------------------------------------------------------------------


@dataclass
class PlotStyle:
    """Visual style options for a frame."""

    cmap: str = "viridis"
    arrow_color: str = "white"
    arrow_scale: float = 10.0
    arrow_width: float = 0.0018
    show_arrows: bool = True
    show_axes: bool = True
    show_colorbar: bool = True
    clim: Optional[Tuple[float, float]] = (-1.0, 1.0)


def _field_vector_array_2d(m: df.Field) -> np.ndarray:
    return ovf_utils.field_vector_array_2d(m)


def _edges_nm(m: df.Field) -> Tuple[float, float]:
    edges = m.mesh.region.edges
    return float(edges[0]) * 1e9, float(edges[1]) * 1e9


def _auto_arrow_stride(nx: int, ny: int, target: int = 10) -> int:
    """Choose a quiver stride that yields ~target arrows across the smaller dimension."""
    return max(1, int(round(min(nx, ny) / float(target))))


def plot_component_with_quiver(
    m: df.Field,
    comp: str,
    out_path: Path,
    title: str,
    style: PlotStyle,
    arrow_stride: Optional[int] = None,
    units: str = "nm",
) -> None:
    """PyVista-like frame: background scalar + in-plane quiver arrows."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vec = _field_vector_array_2d(m)
    mx = vec[:, :, 0]
    my = vec[:, :, 1]
    mz = vec[:, :, 2]

    if comp == "mx":
        bg = mx
        cbar_label = "m_x"
    elif comp == "my":
        bg = my
        cbar_label = "m_y"
    elif comp == "mz":
        bg = mz
        cbar_label = "m_z"
    elif comp in {"mpar", "m_parallel", "mproj"}:
        # Projection onto SP2 coercivity field direction ĥ = (-1,-1,-1)/sqrt(3)
        # m_parallel = m · ĥ = -(mx + my + mz)/sqrt(3)
        bg = -(mx + my + mz) / np.sqrt(3.0)
        cbar_label = "m_parallel"
    else:
        raise ValueError(f"Unknown component: {comp}")

    nx, ny = bg.shape
    if arrow_stride is None:
        arrow_stride = _auto_arrow_stride(nx, ny)
    s = max(1, int(arrow_stride))

    if units == "nm":
        lx, ly = _edges_nm(m)
        xlabel, ylabel = "x (nm)", "y (nm)"
    else:
        edges = m.mesh.region.edges
        lx, ly = float(edges[0]), float(edges[1])
        xlabel, ylabel = "x (m)", "y (m)"

    xs = (np.arange(nx) + 0.5) * (lx / nx)
    ys = (np.arange(ny) + 0.5) * (ly / ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    Xs = X[::s, ::s]
    Ys = Y[::s, ::s]

    # Scale arrows into plot units (nm or m) so they are actually visible.
    # Interpret style.arrow_scale as an arrow length measured in “number of cells”.
    dx = lx / nx
    dy = ly / ny
    Us = mx[::s, ::s] * dx * style.arrow_scale
    Vs = my[::s, ::s] * dy * style.arrow_scale

    # Figure size roughly matches aspect ratio
    aspect = (lx / ly) if (ly > 0) else 1.0
    aspect = max(0.25, min(4.0, float(aspect)))
    fig_w = min(12.0, 6.5 * aspect) if aspect >= 1.0 else 6.5
    fig_h = 6.5 if aspect >= 1.0 else min(12.0, 6.5 / aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        bg.T,
        origin="lower",
        extent=(0, lx, 0, ly),
        cmap=style.cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=style.clim[0] if style.clim else None,
        vmax=style.clim[1] if style.clim else None,
    )

    if style.show_arrows:
        ax.quiver(
            Xs.T,
            Ys.T,
            Us.T,
            Vs.T,
            color=style.arrow_color,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=style.arrow_width,
            pivot="mid",
            headwidth=3.0,
            headlength=3.0,
            headaxislength=3.0,
            minlength=0.0,
        )

    if style.show_axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title)

    # --- Metrics overlay (always-on) ---
    mx_mean = float(mx.mean())
    my_mean = float(my.mean())
    mz_mean = float(mz.mean())
    msum = mx_mean + my_mean + mz_mean
    mpar = -msum / np.sqrt(3.0)
    ax.text(
        0.02,
        0.98,
        f"msum={msum:+.6f}\n<m>=({mx_mean:+.3f},{my_mean:+.3f},{mz_mean:+.3f})\nmpar={mpar:+.6f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6, edgecolor="none"),
    )

    if style.show_colorbar:
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05)
        cbar.set_label(cbar_label)
    plt.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_residual_with_quiver(
    a: df.Field,
    b: df.Field,
    mode: str,
    out_path: Path,
    title: str,
    style: PlotStyle,
    arrow_stride: Optional[int] = None,
    units: str = "nm",
) -> None:
    """Residual plot (A - B).

    mode:
      - "dx" / "dy" / "dz" : background is Δm component
      - "dmag"              : background is |Δm|

    Arrows show in-plane Δm = (Δmx, Δmy).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    va = _field_vector_array_2d(a)
    vb = _field_vector_array_2d(b)
    if va.shape != vb.shape:
        raise ValueError(f"Residual requires matching shapes, got {va.shape} vs {vb.shape}")

    dm = va - vb
    dmx = dm[:, :, 0]
    dmy = dm[:, :, 1]
    dmz = dm[:, :, 2]
    dmag = np.sqrt(dmx * dmx + dmy * dmy + dmz * dmz)

    if mode == "dx":
        bg = dmx
        cbar_label = "Δm_x"
        clim = (-1e-2, 1e-2)
    elif mode == "dy":
        bg = dmy
        cbar_label = "Δm_y"
        clim = (-1e-2, 1e-2)
    elif mode == "dz":
        bg = dmz
        cbar_label = "Δm_z"
        clim = (-1e-2, 1e-2)
    elif mode == "dmag":
        bg = dmag
        cbar_label = "|Δm|"
        clim = (0.0, 5e-2)
    else:
        raise ValueError(f"Unknown residual mode: {mode}")

    # Copy style but set tighter clim for residuals
    style2 = PlotStyle(
        cmap=style.cmap,
        arrow_color=style.arrow_color,
        arrow_scale=style.arrow_scale,
        arrow_width=style.arrow_width,
        show_arrows=style.show_arrows,
        show_axes=style.show_axes,
        show_colorbar=style.show_colorbar,
        clim=clim,
    )

    nx, ny = bg.shape
    if arrow_stride is None:
        arrow_stride = _auto_arrow_stride(nx, ny)
    s = max(1, int(arrow_stride))

    if units == "nm":
        lx, ly = _edges_nm(a)
        xlabel, ylabel = "x (nm)", "y (nm)"
    else:
        edges = a.mesh.region.edges
        lx, ly = float(edges[0]), float(edges[1])
        xlabel, ylabel = "x (m)", "y (m)"

    xs = (np.arange(nx) + 0.5) * (lx / nx)
    ys = (np.arange(ny) + 0.5) * (ly / ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    Xs = X[::s, ::s]
    Ys = Y[::s, ::s]

    dx = lx / nx
    dy = ly / ny
    Us = dmx[::s, ::s] * dx * style2.arrow_scale
    Vs = dmy[::s, ::s] * dy * style2.arrow_scale

    aspect = (lx / ly) if (ly > 0) else 1.0
    aspect = max(0.25, min(4.0, float(aspect)))
    fig_w = min(12.0, 6.5 * aspect) if aspect >= 1.0 else 6.5
    fig_h = 6.5 if aspect >= 1.0 else min(12.0, 6.5 / aspect)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        bg.T,
        origin="lower",
        extent=(0, lx, 0, ly),
        cmap=style2.cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=style2.clim[0] if style2.clim else None,
        vmax=style2.clim[1] if style2.clim else None,
    )

    if style2.show_arrows:
        ax.quiver(
            Xs.T,
            Ys.T,
            Us.T,
            Vs.T,
            color=style2.arrow_color,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=style2.arrow_width,
            pivot="mid",
            headwidth=3.0,
            headlength=3.0,
            headaxislength=3.0,
            minlength=0.0,
        )

    if style2.show_axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title)

    # --- Residual metrics overlay (always-on) ---
    dmag_rms = float(np.sqrt(np.mean(dmag * dmag)))
    dmag_max = float(np.max(dmag))
    ax.text(
        0.02,
        0.98,
        f"|Δm|_rms={dmag_rms:.3e}\n|Δm|_max={dmag_max:.3e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6, edgecolor="none"),
    )

    if style2.show_colorbar:
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05)
        cbar.set_label(cbar_label)
    plt.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Movie helper
# -----------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def write_mp4_from_frames(
    frames_dir: Path,
    out_mp4: Path,
    fps: float,
    pattern: str = "frame_%06d.png",
) -> bool:
    """Create an MP4 using ffmpeg. Returns True on success.

    Common failure cause: matplotlib often writes PNGs with odd pixel widths/heights
    (especially with bbox_inches='tight'). H.264/yuv420p encoders require even
    dimensions, so we always apply a scale filter to force even sizes.

    On some macOS ffmpeg builds, libx264 may be unavailable. We fall back to the
    hardware encoder (h264_videotoolbox) and then to mpeg4.
    """
    if not _ffmpeg_available():
        return False

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    input_pat = str(frames_dir / pattern)

    codecs = ["libx264", "h264_videotoolbox", "mpeg4"]
    last_err = ""

    for codec in codecs:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(float(fps)),
            "-start_number",
            "0",
            "-i",
            input_pat,
            # Force even dimensions for encoder compatibility
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            return True

        last_err = (res.stderr or "").strip()

    print("\n[movie] ffmpeg failed to build mp4")
    print(f"  frames_dir: {frames_dir}")
    print(f"  out_mp4:    {out_mp4}")
    if last_err:
        print("  ffmpeg stderr:\n" + last_err)
    else:
        print("  (no stderr captured)")

    print("\n  Try manually:")
    print(
        f"    ffmpeg -y -framerate {fps} -start_number 0 -i {frames_dir}/{pattern} "
        f"-vf scale=trunc(iw/2)*2:trunc(ih/2)*2 -c:v libx264 -pix_fmt yuv420p "
        f"-movflags +faststart {out_mp4}"
    )

    return False


# -----------------------------------------------------------------------------
# Export logic
# -----------------------------------------------------------------------------


def _normalise_components(comps: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in comps:
        c2 = c.strip().lower()
        if not c2:
            continue
        if c2 in {"all", "*"}:
            return ["mx", "my", "mz"]
        if c2 in {"mx", "my", "mz", "mpar", "m_parallel", "mproj"}:
            # keep aliases; downstream plotting handles them
            out.append(c2)
        else:
            raise ValueError(f"Unknown component: {c}")
    return out or ["mx", "my", "mz"]


def _frame_label_sp2(series: ovf_utils.OvfSeries, idx: int) -> str:
    """Best-effort frame label for SP2."""
    p = series.frames[idx]
    # If ovf_utils provided explicit frame labels (e.g. hc_pos_strict/hc_best_strict/hc_final), use them.
    frame_labels = series.meta.get("frame_labels")
    if isinstance(frame_labels, dict):
        lab = frame_labels.get(str(p))
        if isinstance(lab, str) and lab:
            return lab

    if series.kind.startswith("sweep"):
        # use d/lex from filename
        m = re.search(r"m_d(\d+)_", p.name)
        if m:
            return f"d/lex={int(m.group(1))}"
        return p.stem

    # evolution: if this is the appended final snapshot, label it explicitly
    final_frame = series.meta.get("final_frame")
    if isinstance(final_frame, str):
        try:
            if Path(final_frame).resolve() == p.resolve():
                return "t=final"
        except Exception:
            pass

    # evolution: include tag prefix when available (e.g. bisect_strict_bc0p046156)
    tag = ovf_utils.try_parse_sp2_tag_from_name(p)

    # evolution: try time from filename first
    t_s = ovf_utils.try_parse_sp2_time_label_seconds_from_name(p)
    if t_s is None:
        t_s = ovf_utils.try_parse_time_seconds_from_ovf(p)
    if t_s is not None:
        if tag:
            return f"{tag} | t={t_s * 1e9:.3f} ns"
        return f"t={t_s * 1e9:.3f} ns"
    if tag:
        return tag
    return p.stem


def _frame_label_sp4(series: ovf_utils.OvfSeries, idx: int) -> str:
    p = series.frames[idx]
    t_s = ovf_utils.try_parse_time_seconds_from_ovf(p)
    if t_s is not None:
        return f"t={t_s * 1e9:.3f} ns"
    return p.stem


def export_series_frames(
    series: ovf_utils.OvfSeries,
    out_root: Path,
    components: Sequence[str],
    units: str,
    movie: bool,
    movie_component: str,
    fps: float,
    sample_dt_ns: Optional[float] = None,
    max_frames: Optional[int] = None,
    clim: Optional[Tuple[float, float]] = (-1.0, 1.0),
    vectors: bool = False,
    verbose: bool = True,
) -> None:
    """Export one series as PNG frames (and optional MP4 movie)."""

    comps = _normalise_components(components)
    if movie_component not in comps:
        comps = list(comps) + [movie_component]

    frames = list(series.frames)

    # Optional time-based downsampling for evolution series.
    # Useful when the raw OVF dump cadence is finer than desired (e.g. SP2 per-ps dumps).
    if sample_dt_ns is not None and sample_dt_ns > 0 and not series.kind.startswith("sweep"):
        bins: Dict[int, Path] = {}
        for p in frames:
            t_s = ovf_utils.try_parse_sp2_time_label_seconds_from_name(p)
            if t_s is None:
                t_s = ovf_utils.try_parse_time_seconds_from_ovf(p)
            if t_s is None:
                continue
            b = int(round((t_s * 1e9) / float(sample_dt_ns)))
            bins.setdefault(b, p)
        if bins:
            frames = [bins[k] for k in sorted(bins.keys())]

    if max_frames is not None:
        frames = frames[: int(max_frames)]

    if not frames:
        return

    # Output layout
    #   SP4: plots/sp4/<source>/<case>/frames/<comp>/frame_000000.png
    #   SP2 evolve: plots/sp2/<source>/d30/hc/<tag>/frames/<comp>/frame_000000.png
    #   SP2 sweep:  plots/sp2/<source>/sweep/<stage>/<comp>/d30.png
    if series.problem == "sp4":
        base_dir = out_root / "sp4" / series.source / series.kind
        label_fn = _frame_label_sp4
    else:
        if series.kind.startswith("sweep"):
            base_dir = out_root / "sp2" / series.source / "sweep" / (series.stage or "unknown")
        else:
            dpart = f"d{series.d_lex:02d}" if series.d_lex is not None else "d??"
            if series.stage == "hc" and series.tag:
                base_dir = out_root / "sp2" / series.source / dpart / "hc" / series.tag
            else:
                base_dir = out_root / "sp2" / series.source / dpart / (series.stage or "unknown")
        label_fn = _frame_label_sp2

    style = PlotStyle(cmap="viridis", arrow_color="white", show_arrows=False, clim=clim)
    style_vec = PlotStyle(
        cmap=style.cmap,
        arrow_color="white",
        arrow_scale=10.0,
        arrow_width=0.0018,
        show_arrows=True,
        show_axes=style.show_axes,
        show_colorbar=False,
        clim=clim,
    )

    if series.kind.startswith("sweep"):
        # For sweeps: we only export one image per d value (not frame_%06d)
        comp_dirs = {comp: (base_dir / comp) for comp in comps}
        for d in comp_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        for p in frames:
            m = ovf_utils.load_slice_field(p)
            dlab = re.search(r"m_d(\d+)_", p.name)
            dval = int(dlab.group(1)) if dlab else 0
            for comp in comps:
                title = f"SP2 {series.stage or ''} {series.source} d/lex={dval} ({comp})"
                out_png = comp_dirs[comp] / f"d{dval:02d}.png"
                plot_component_with_quiver(m, comp, out_png, title, style, units=units)

        # Optional: build a sweep movie for movie_component
        if movie:
            comp_dir = comp_dirs[movie_component]
            frames_dir = comp_dir / "_frames_tmp"
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)

            # Sort by d descending in the series meta if available
            d_vals = series.meta.get("d_values")
            ordered: List[Tuple[int, Path]] = []
            if isinstance(d_vals, list) and len(d_vals) == len(frames):
                for dv, fp in zip(d_vals, frames):
                    ordered.append((int(dv), fp))
                ordered.sort(key=lambda x: x[0], reverse=True)
            else:
                for fp in frames:
                    m2 = re.search(r"m_d(\d+)_", fp.name)
                    dv = int(m2.group(1)) if m2 else 0
                    ordered.append((dv, fp))
                ordered.sort(key=lambda x: x[0], reverse=True)

            for i, (dv, _fp) in enumerate(ordered):
                src = comp_dir / f"d{dv:02d}.png"
                if src.exists():
                    shutil.copyfile(src, frames_dir / f"frame_{i:06d}.png")

            out_mp4 = base_dir / "movies" / f"sweep_{series.stage}_{movie_component}.mp4"
            ok = write_mp4_from_frames(frames_dir, out_mp4, fps=fps)
            shutil.rmtree(frames_dir, ignore_errors=True)
            if verbose:
                if ok:
                    print(f"  [movie] wrote {out_mp4}")
                else:
                    if _ffmpeg_available():
                        print(f"  [movie] ffmpeg failed for {out_mp4}")
                    else:
                        print("  [movie] ffmpeg not found; sweep frames exported only")
        return

    # Evolution: export sequential frames
    frame_dirs = {comp: (base_dir / "frames" / comp) for comp in comps}
    for d in frame_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    vector_frame_dirs = {comp: (base_dir / "frames_vectors" / comp) for comp in comps} if vectors else {}
    for ddir in vector_frame_dirs.values():
        ddir.mkdir(parents=True, exist_ok=True)

    for i, pth in enumerate(frames):
        m = ovf_utils.load_slice_field(pth)
        flabel = label_fn(series, i)
        for comp in comps:
            title = f"{series.problem.upper()} {series.kind} {series.source}  {flabel}  ({comp})"
            out_png = frame_dirs[comp] / f"frame_{i:06d}.png"
            plot_component_with_quiver(m, comp, out_png, title, style, units=units)

            if vectors:
                out_png_v = vector_frame_dirs[comp] / f"frame_{i:06d}.png"
                plot_component_with_quiver(m, comp, out_png_v, title, style_vec, units=units)

    if movie:
        frames_dir = frame_dirs[movie_component]
        out_mp4 = base_dir / "movies" / f"{series.kind}_{movie_component}.mp4"
        ok = write_mp4_from_frames(frames_dir, out_mp4, fps=fps)
        if verbose:
            if ok:
                print(f"  [movie] wrote {out_mp4}")
            else:
                if _ffmpeg_available():
                    print(f"  [movie] ffmpeg failed for {out_mp4} (run manually to see output)")
                else:
                    print(
                        "  [movie] ffmpeg not found; frames are ready. To build mp4:\n"
                        f"    ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%06d.png "
                        f"-c:v libx264 -pix_fmt yuv420p {out_mp4}"
                    )


def export_residual_series(
    series_a: ovf_utils.OvfSeries,
    series_b: ovf_utils.OvfSeries,
    out_root: Path,
    units: str,
    modes: Sequence[str] = ("dmag", "dz"),
    movie: bool = False,
    fps: float = 10.0,
    max_frames: Optional[int] = None,
) -> None:
    """Export residual (A-B) frames for aligned series."""
    if series_a.problem != series_b.problem:
        raise ValueError("Residual export requires matching problems")

    style = PlotStyle(cmap="magma", arrow_color="white", clim=None)

    if series_a.problem == "sp4":
        n = min(len(series_a.frames), len(series_b.frames))
        if max_frames is not None:
            n = min(n, int(max_frames))
        base_dir = out_root / "compare" / "sp4" / f"{series_a.source}_vs_{series_b.source}" / series_a.kind

        frame_dirs = {mode: (base_dir / "frames" / mode) for mode in modes}
        for d in frame_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            a = ovf_utils.load_slice_field(series_a.frames[i])
            b = ovf_utils.load_slice_field(series_b.frames[i])
            flabel = _frame_label_sp4(series_a, i)
            for mode in modes:
                title = f"SP4 {series_a.kind} Δ ({series_a.source} - {series_b.source}) {flabel} ({mode})"
                out_png = frame_dirs[mode] / f"frame_{i:06d}.png"
                plot_residual_with_quiver(a, b, mode, out_png, title, style, units=units)

        if movie and "dmag" in modes:
            out_mp4 = base_dir / "movies" / f"{series_a.kind}_dmag.mp4"
            write_mp4_from_frames(frame_dirs["dmag"], out_mp4, fps=fps)
        return

    # SP2 sweep alignment by d value
    if series_a.kind.startswith("sweep") and series_b.kind.startswith("sweep"):
        def d_from(p: Path) -> Optional[int]:
            m = re.search(r"m_d(\d+)_", p.name)
            return int(m.group(1)) if m else None

        map_a: Dict[int, Path] = {}
        for p in series_a.frames:
            d = d_from(p)
            if d is not None:
                map_a[int(d)] = p

        map_b: Dict[int, Path] = {}
        for p in series_b.frames:
            d = d_from(p)
            if d is not None:
                map_b[int(d)] = p

        common = sorted(set(map_a.keys()) & set(map_b.keys()), reverse=True)
        if not common:
            print("  [compare] No common d values between series")
            return

        base_dir = (
            out_root
            / "compare"
            / "sp2"
            / f"{series_a.source}_vs_{series_b.source}"
            / "sweep"
            / (series_a.stage or "unknown")
        )
        out_dirs = {mode: (base_dir / mode) for mode in modes}
        for ddir in out_dirs.values():
            ddir.mkdir(parents=True, exist_ok=True)

        for d in common:
            a = ovf_utils.load_slice_field(map_a[d])
            b = ovf_utils.load_slice_field(map_b[d])
            for mode in modes:
                title = f"SP2 {series_a.stage} Δ ({series_a.source} - {series_b.source}) d/lex={d} ({mode})"
                out_png = out_dirs[mode] / f"d{int(d):02d}.png"
                plot_residual_with_quiver(a, b, mode, out_png, title, style, units=units)
        return

    print("  [compare] SP2 evolution compare not enabled (requires aligned time-series).")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Batch export OVF plots/movies for SP2 + SP4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--vectors",
        action="store_true",
        help=(
            "Also export an extra set of frames with emphasized vector glyphs (no colorbar) "
            "under frames_vectors/."
        ),
    )
    p.add_argument("--input", required=True, help="Input folder (runs/... or mumax_outputs/...).")
    p.add_argument(
        "--problem",
        choices=["auto", "sp2", "sp4"],
        default="auto",
        help="Problem preset (auto tries to detect).",
    )
    p.add_argument(
        "--source",
        choices=["auto", "rust", "mumax"],
        default="auto",
        help="Override source detection.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output root directory. Default: plots/",
    )
    p.add_argument(
        "--units",
        choices=["nm", "m"],
        default="nm",
        help="Axis units.",
    )
    p.add_argument(
        "--components",
        nargs="+",
        default=["mx", "my", "mz"],
        help="Components to export (mx my mz mpar) or 'all'. Use mpar to plot m·(-1,-1,-1)/sqrt(3).",
    )
    p.add_argument(
        "--movie",
        action="store_true",
        help="Also build MP4 movies (requires ffmpeg).",
    )
    p.add_argument(
        "--movie-component",
        choices=["mx", "my", "mz", "mpar"],
        default="mx",
        help="Which component to use for the movie (frames still exported for --components).",
    )
    p.add_argument("--fps", type=float, default=10.0, help="Movie frames-per-second.")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the number of frames exported per evolution series (debugging).",
    )
    p.add_argument(
        "--sample-ns",
        type=float,
        default=None,
        help=(
            "Optional time-based downsampling for evolution series. "
            "Example: --sample-ns 1.0 selects ~one frame per simulated nanosecond. "
            "(Ignored for sweeps.)"
        ),
    )

    p.add_argument(
        "--clim",
        nargs="+",
        default=None,
        help=(
            "Colour limits for component plots: either 'auto' or two numbers (e.g. -1 1). "
            "Default is fixed [-1, 1] for direct comparability."
        ),
    )

    # SP4 selection
    p.add_argument(
        "--case",
        choices=["sp4a", "sp4b", "both"],
        default="both",
        help="SP4 case selection when input is a root folder.",
    )

    # SP2 selection
    p.add_argument(
        "--d",
        type=int,
        default=None,
        help="SP2: if set, export evolution series for this d/lex. If unset, export sweep finals.",
    )
    p.add_argument(
        "--stage",
        choices=["rem", "hc", "both"],
        default="both",
        help="SP2: stage selection (remanence or coercivity).",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="SP2: for coercivity evolution, choose a specific series tag (optional).",
    )

    # Compare/residual
    p.add_argument(
        "--input-b",
        default=None,
        help="Optional second input root to export residuals (A-B).",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Export residuals between --input (A) and --input-b (B).",
    )

    args = p.parse_args()

    input_a = Path(args.input)
    if not input_a.exists():
        raise SystemExit(f"Input path does not exist: {input_a}")

    out_root = Path(args.output) if args.output else Path("plots")
    out_root.mkdir(parents=True, exist_ok=True)

    # Colour scale handling for mx/my/mz component plots.
    # Default is fixed [-1, 1] to keep plots comparable between codes.
    clim: Optional[Tuple[float, float]] = (-1.0, 1.0)
    if args.clim is not None:
        if len(args.clim) == 1 and str(args.clim[0]).lower() == "auto":
            clim = None  # matplotlib auto-scale per plot
        elif len(args.clim) == 2:
            try:
                clim = (float(args.clim[0]), float(args.clim[1]))
            except Exception:
                p.error("--clim expects either 'auto' or two numbers, e.g. --clim -1 1")
        else:
            p.error("--clim expects either 'auto' or two numbers, e.g. --clim -1 1")

    source_a = ovf_utils.infer_source(input_a) if args.source == "auto" else args.source
    problem = ovf_utils.detect_problem(input_a) if args.problem == "auto" else args.problem

    if args.compare:
        if not args.input_b:
            raise SystemExit("--compare requires --input-b")
        input_b = Path(args.input_b)
        if not input_b.exists():
            raise SystemExit(f"Input-B path does not exist: {input_b}")
        source_b = ovf_utils.infer_source(input_b) if args.source == "auto" else args.source

        if problem == "sp4":
            a_series = ovf_utils.discover_sp4_series(input_a, source=source_a)
            b_series = ovf_utils.discover_sp4_series(input_b, source=source_b)
            b_by_kind = {s.kind: s for s in b_series}
            for s in a_series:
                if args.case != "both" and s.kind != args.case:
                    continue
                if s.kind not in b_by_kind:
                    print(f"[compare] Missing B series for {s.kind}; skipping")
                    continue
                export_residual_series(
                    s,
                    b_by_kind[s.kind],
                    out_root,
                    units=args.units,
                    movie=args.movie,
                    fps=args.fps,
                    max_frames=args.max_frames,
                )
            print("Done (compare).")
            return 0

        if problem == "sp2":
            a_series = ovf_utils.discover_sp2_series(input_a, source=source_a)
            b_series = ovf_utils.discover_sp2_series(input_b, source=source_b)
            stages = [args.stage] if args.stage != "both" else ["rem", "hc"]
            for st in stages:
                sa = ovf_utils.pick_sp2_sweep_series(a_series, st)
                sb = ovf_utils.pick_sp2_sweep_series(b_series, st)
                if sa is None or sb is None:
                    print(f"[compare] Missing sweep series for stage={st}; skipping")
                    continue
                export_residual_series(sa, sb, out_root, units=args.units, movie=False)
            print("Done (compare).")
            return 0

        raise SystemExit(f"Unknown problem for compare: {problem}")

    # Non-compare export
    if problem == "sp4":
        series = ovf_utils.discover_sp4_series(input_a, source=source_a)
        if not series:
            raise SystemExit("No SP4 OVF series found")
        for s in series:
            if args.case != "both" and s.kind != args.case:
                continue
            print(f"[sp4] exporting {s.kind} ({len(s.frames)} frames)")
            export_series_frames(
                s,
                out_root=out_root,
                components=args.components,
                units=args.units,
                movie=args.movie,
                movie_component=args.movie_component,
                fps=args.fps,
                sample_dt_ns=args.sample_ns,
                max_frames=args.max_frames,
                clim=clim,
                vectors=args.vectors,
            )
        print("Done.")
        return 0

    if problem == "sp2":
        series = ovf_utils.discover_sp2_series(input_a, source=source_a)
        if not series:
            raise SystemExit("No SP2 OVF series found")

        if args.d is None:
            # Sweep mode
            stages = [args.stage] if args.stage != "both" else ["rem", "hc"]
            for st in stages:
                s = ovf_utils.pick_sp2_sweep_series(series, st)
                if s is None:
                    print(f"[sp2 sweep] no series for stage={st}; skipping")
                    continue
                print(f"[sp2 sweep] exporting {st} ({len(s.frames)} d-values)")
                export_series_frames(
                    s,
                    out_root=out_root,
                    components=args.components,
                    units=args.units,
                    movie=args.movie,
                    movie_component=args.movie_component,
                    fps=args.fps,
                    sample_dt_ns=args.sample_ns,
                    max_frames=args.max_frames,
                    clim=clim,
                    vectors=args.vectors,
                )
            print("Done.")
            return 0

        # Single-d evolution mode
        stages = [args.stage] if args.stage != "both" else ["rem", "hc"]
        for st in stages:
            s = ovf_utils.pick_sp2_evolution_series(series, args.d, st, tag=args.tag)
            if s is None:
                print(f"[sp2 evolve] no series for d={args.d} stage={st} tag={args.tag}; skipping")
                continue
            print(f"[sp2 evolve] exporting d={args.d} stage={st} tag={s.tag} ({len(s.frames)} frames)")
            export_series_frames(
                s,
                out_root=out_root,
                components=args.components,
                units=args.units,
                movie=args.movie,
                movie_component=args.movie_component,
                fps=args.fps,
                sample_dt_ns=args.sample_ns,
                max_frames=args.max_frames,
                clim=clim,
                vectors=args.vectors,
            )
        print("Done.")
        return 0

    raise SystemExit(f"Unknown problem: {problem}")


if __name__ == "__main__":
    raise SystemExit(main())


