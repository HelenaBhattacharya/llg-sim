#!/usr/bin/env python3
"""OVF magnetisation visualisation (MuMax3 or Rust).

This script is a small visualisation runner with *problem-specific* presets.

Supported presets:
  - sp4 : long thin strip (Standard Problem 4) -> component maps + dense vectors + strip-friendly resample
  - fmr : square film (FMR benchmark) -> Ubermag-like resample (10,10) and mz background option
  - sp2 : many OVFs across d/lex -> grouped remanence vs coercivity outputs
  - generic : fallback (safe defaults)

Typical commands:

  # SP4 (auto-detect source, output defaults to plots/sp4_<source>/...)
  python3 scripts/mag_visualisation.py --input mumax_outputs/st_problems/sp4
  python3 scripts/mag_visualisation.py --input runs/st_problems/sp4

  # FMR (force preset + explicit output)
  python3 scripts/mag_visualisation.py --input runs/st_problems/fmr/fmr_rust --problem fmr --output plots/fmr_rust

  # SP2 (group remanence/coercivity)
  python3 scripts/mag_visualisation.py --input mumax_outputs/st_problems/sp2/sp2_out --problem sp2 --output plots/sp2_mumax

Notes:
  - OVF files are read via discretisedfield.
  - For thin films, we plot the z-slice.
  - Figure sizes are chosen automatically from sample aspect ratio (unless overridden in code).
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import discretisedfield as df
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# OVF header helpers
# ---------------------------

_TIME_RE = re.compile(r"Total simulation time:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*s")

def try_parse_time_ns_from_ovf(path: Path):
    """Parse 'Total simulation time' from OVF header and return time in ns (float).

    Returns None if not found.
    """
    try:
        raw = path.read_bytes()[:64 * 1024]
        text = raw.decode("utf-8", errors="ignore")
        m = _TIME_RE.search(text)
        if m:
            t_s = float(m.group(1))
            return t_s * 1e9
    except Exception:
        return None
    return None

# ---------------------------
# OVF compatibility (Rust OVF writer may omit xmin/xmax/ymin/ymax/zmin/zmax)
# ---------------------------

def _ovf_header_kv(text: str):
    """Parse OVF header key/value pairs of the form '# key: value'."""
    kv = {}
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
    """If OVF header lacks xmin/xmax/... fields, synthesize them from xbase/xstepsize/xnodes, etc."""
    text = raw.decode("utf-8", errors="ignore")

    hdr_start = text.find("# Begin: Header")
    hdr_end = text.find("# End: Header")
    if hdr_start == -1 or hdr_end == -1 or hdr_end <= hdr_start:
        return raw

    header_text = text[hdr_start:hdr_end]
    kv = _ovf_header_kv(header_text)

    # If already present, do nothing
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

    # Discretisedfield expects min/max box bounds
    xmin = xbase
    ymin = ybase
    zmin = zbase
    xmax = xbase + nx * dx
    ymax = ybase + ny * dy
    zmax = zbase + nz * dz

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


def _load_field_from_ovf_compat(path: Path) -> df.Field:
    """Load OVF via discretisedfield, patching headers if needed (for Rust OVFs missing xmin/xmax)."""
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

# ---------------------------
# Helpers: input detection
# ---------------------------

def infer_source(input_path: Path) -> str:
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


def find_ovf_case_dirs(input_path: Path):
    """Return a list of directories that contain m*.ovf files.

    Supports:
      - root folders containing case subfolders (sp4a_out/sp4b_out, sp4a_rust/sp4b_rust)
      - direct case folders
      - any folder where OVFs are present directly
    """

    if list(input_path.glob("m*.ovf")):
        return [input_path]

    preferred = []
    for sub in ["sp4a_out", "sp4b_out", "sp4a_rust", "sp4b_rust"]:
        d = input_path / sub
        if d.exists() and list(d.glob("m*.ovf")):
            preferred.append(d)

    if preferred:
        return preferred

    found = []
    for d in sorted([p for p in input_path.iterdir() if p.is_dir()]):
        if list(d.glob("m*.ovf")):
            found.append(d)

    return found


def detect_problem(input_path: Path, case_dirs) -> str:
    """Heuristic to choose a plotting preset."""

    # If explicit SP4 structure exists
    if (input_path / "sp4a_out").exists() or (input_path / "sp4b_out").exists() or (input_path / "sp4a_rust").exists() or (input_path / "sp4b_rust").exists():
        return "sp4"

    # Scan filenames
    ovf_files = []
    for d in case_dirs:
        ovf_files.extend(sorted(d.glob("m*.ovf")))

    names = [p.name for p in ovf_files]

    # SP2 naming: m_dXX_rem / m_dXX_hc
    if any(re.match(r"m_d\d+_rem\.ovf$", n) for n in names) or any(re.match(r"m_d\d+_hc\.ovf$", n) for n in names):
        return "sp2"

    # FMR naming: m_relaxed / m_dyn etc
    if any(n.startswith("m_relaxed") for n in names) or any(n.startswith("m_dyn") for n in names):
        return "fmr"
    
    # Skyrmion folders commonly named sk*, skyrmion*, sk1*, etc.
    s2 = str(input_path).lower()
    if "sk1" in s2 or "skyrm" in s2 or "/sk" in s2 or input_path.name.lower().startswith("sk"):
        return "sk"

    return "generic"


# ---------------------------
# Helpers: plotting
# ---------------------------

def load_slice_field(ovf_path: Path) -> df.Field:
    field = _load_field_from_ovf_compat(ovf_path)

    # For thin films select z plane
    mesh_dims = field.mesh.region.edges
    if mesh_dims[2] < mesh_dims[0] / 10:
        return field.sel("z")

    # For 3D, middle slice
    z_center = field.mesh.region.center[2]
    return field.sel(z=z_center)


def resample_for_aspect(m: df.Field, target_nx: int = 25, min_ny: int = 5, max_ny: int = 25):
    """Choose a resample (nx, ny) that roughly respects aspect ratio."""
    edges = m.mesh.region.edges
    lx = float(edges[0])
    ly = float(edges[1])
    if lx <= 0 or ly <= 0:
        return (target_nx, min_ny)
    ratio = ly / lx
    ny = int(round(target_nx * ratio))
    ny = max(min_ny, min(max_ny, ny))
    return (target_nx, ny)


def auto_figsize(m: df.Field, max_w: float = 12.0, max_h: float = 7.0, base: float = 6.0):
    """Pick a sensible figure size from the sample aspect ratio.

    - Long thin samples get wide figures.
    - Near-square samples get square-ish figures.
    """
    edges = m.mesh.region.edges
    lx = float(edges[0])
    ly = float(edges[1])
    if lx <= 0 or ly <= 0:
        return (base, base)

    aspect = lx / ly

    # Clamp extreme aspects
    aspect = max(0.25, min(4.0, aspect))

    if aspect >= 1.0:
        w = min(max_w, base * aspect)
        h = min(max_h, base)
    else:
        w = min(max_w, base)
        h = min(max_h, base / aspect)

    return (w, h)


def plot_scalar(m_scalar: df.Field, title: str, cbar: str, out_path: Path, figsize=None):
    if figsize is None:
        figsize = auto_figsize(m_scalar)
    fig, ax = plt.subplots(figsize=figsize)
    m_scalar.mpl.scalar(ax=ax, colorbar_label=cbar)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vectors(m_vec: df.Field, title: str, out_path: Path, figsize=None):
    if figsize is None:
        figsize = auto_figsize(m_vec)
    fig, ax = plt.subplots(figsize=figsize)
    m_vec.mpl.vector(ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vectors_colored(
    m_vec: df.Field,
    title: str,
    out_path: Path,
    bg_scalar: df.Field,
    bg_label: str,
    cmap: str = "magma",
    alpha: float = 0.7,
    figsize=None,
    vector_color=None,
):
    if figsize is None:
        figsize = auto_figsize(bg_scalar)
    fig, ax = plt.subplots(figsize=figsize)
    # Best-effort arrow color (discretisedfield API may not accept color kwarg in all versions)
    if vector_color is None:
        m_vec.mpl.vector(ax=ax)
    else:
        try:
            m_vec.mpl.vector(ax=ax, color=vector_color)
        except TypeError:
            m_vec.mpl.vector(ax=ax)
    bg_scalar.mpl.scalar(ax=ax, cmap=cmap, alpha=alpha, colorbar_label=bg_label)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# ---------------------------
# Matplotlib helper for "PyVista-like" frames
# ---------------------------

def _field_vector_array_2d(m: df.Field) -> np.ndarray:
    """Return vector array shaped (nx, ny, 3) for a 2D slice Field."""
    arr = None
    if hasattr(m, "array"):
        try:
            arr = np.asarray(m.array)
        except Exception:
            arr = None
    if arr is None:
        # discretisedfield fallback
        arr = np.asarray(cast(Any, m).asarray())

    # Possible shapes: (nx, ny, 3) or (nx, ny, 1, 3)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        arr = arr[:, :, 0, :]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected vector array shape: {arr.shape}")
    return arr


def plot_scalar_with_quiver(
    m: df.Field,
    comp: str,
    title: str,
    out_path: Path,
    arrow_stride: int = 2,
    cmap: str = "viridis",
):
    """Plot a full-res scalar background with in-plane arrows overlaid.

    This is designed to visually match the PyVista view in mag_viewer:
      - background: chosen component (mx/my/mz)
      - arrows: in-plane (mx,my)
      - single horizontal colorbar
      - axes in nm
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vec = _field_vector_array_2d(m)  # (nx, ny, 3)
    mx = vec[:, :, 0]
    my = vec[:, :, 1]
    mz = vec[:, :, 2]

    if comp == "mx":
        bg = mx
        cbar_label = "mx"
    elif comp == "my":
        bg = my
        cbar_label = "my"
    elif comp == "mz":
        bg = mz
        cbar_label = "mz"
    else:
        raise ValueError(f"Unknown component: {comp}")

    # Geometry in nm
    edges = m.mesh.region.edges
    lx_nm = float(edges[0]) * 1e9
    ly_nm = float(edges[1]) * 1e9

    nx, ny = bg.shape

    # Cell-center coordinates for quiver
    xs = (np.arange(nx) + 0.5) * (lx_nm / nx)
    ys = (np.arange(ny) + 0.5) * (ly_nm / ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # Subsample arrows
    s = max(1, int(arrow_stride))
    Xs = X[::s, ::s]
    Ys = Y[::s, ::s]
    Us = mx[::s, ::s]
    Vs = my[::s, ::s]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    im = ax.imshow(
        bg.T,
        origin="lower",
        extent=(0, lx_nm, 0, ly_nm),
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )

    # White arrows, similar feel to the PyVista frames
    ax.quiver(
        Xs.T,
        Ys.T,
        Us.T,
        Vs.T,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=35.0,
        width=0.0022,
        pivot="mid",
        headwidth=3.0,
        headlength=3.6,
        headaxislength=3.0,
        minlength=0.0,
    )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, fraction=0.05)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_skyrmion_mz_with_quiver(
    m: df.Field,
    title: str,
    out_path: Path,
    arrow_stride: int = 4,
    cmap: str = "viridis",
    msat: float | None = None,
):
    """Skyrmion-style frame: mz colormap + in-plane arrows, axes centered at (0,0) in nm.

    If msat is provided, background is scaled to Mz in A/m (mz * msat).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vec = _field_vector_array_2d(m)  # (nx, ny, 3)
    mx = vec[:, :, 0]
    my = vec[:, :, 1]
    mz = vec[:, :, 2]

    # Geometry in nm
    edges = m.mesh.region.edges
    lx_nm = float(edges[0]) * 1e9
    ly_nm = float(edges[1]) * 1e9

    nx, ny = mz.shape

    # Centered coordinates for display
    xs = (np.arange(nx) + 0.5) * (lx_nm / nx) - 0.5 * lx_nm
    ys = (np.arange(ny) + 0.5) * (ly_nm / ny) - 0.5 * ly_nm
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # Background scaling
    if msat is None:
        bg = mz
        cbar_label = "mz"
        vmin, vmax = -1.0, 1.0
    else:
        bg = mz * float(msat)
        cbar_label = "z-component (A/m)"
        vmin, vmax = -float(msat), float(msat)

    # Subsample arrows
    s = max(1, int(arrow_stride))
    Xs = X[::s, ::s]
    Ys = Y[::s, ::s]
    Us = mx[::s, ::s]
    Vs = my[::s, ::s]

    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    im = ax.imshow(
        bg.T,
        origin="lower",
        extent=(-0.5 * lx_nm, 0.5 * lx_nm, -0.5 * ly_nm, 0.5 * ly_nm),
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    ax.quiver(
        Xs.T,
        Ys.T,
        Us.T,
        Vs.T,
        color="black",
        angles="xy",
        scale_units="xy",
        scale=35.0,
        width=0.0022,
        pivot="mid",
        headwidth=3.0,
        headlength=3.6,
        headaxislength=3.0,
        minlength=0.0,
    )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------------------------
# Presets
# ---------------------------

def preset_sp4(ovf_path: Path, out_dir: Path):
    """SP4: long thin strip."""
    out_dir.mkdir(parents=True, exist_ok=True)

    m = load_slice_field(ovf_path)
    base = ovf_path.stem

    # Component maps (strip-friendly aspect)
    plot_scalar(m.x, f"{base} - X component", "mx", out_dir / f"{base}_mx.png", figsize=None)
    plot_scalar(m.y, f"{base} - Y component", "my", out_dir / f"{base}_my.png", figsize=None)
    plot_scalar(m.z, f"{base} - Z component", "mz", out_dir / f"{base}_mz.png", figsize=None)

    # Vector plots
    plot_vectors(m, f"{base} - Vector field", out_dir / f"{base}_vectors.png", figsize=None)

    # Colored vectors (SP4-friendly resample)
    m_small = m.resample((20, 5))
    plot_vectors_colored(
        m_small,
        f"{base} - Colored vector field",
        out_dir / f"{base}_vectors_colored.png",
        bg_scalar=m.x,
        bg_label="mx",
        cmap="magma",
        alpha=0.7,
        figsize=None,
    )


def preset_fmr(ovf_path: Path, out_dir: Path, msat: float = 8e5):
    """FMR: square film.

    For each snapshot, write three PyVista-like frames (scalar bg + arrows):
      - mx background + arrows
      - my background + arrows
      - mz background + arrows

    No extra "ubermag" plot and no scalar-only maps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    m = load_slice_field(ovf_path)
    base = ovf_path.stem

    # Time label from OVF header (ns)
    t_ns = try_parse_time_ns_from_ovf(ovf_path)
    if t_ns is None:
        t_label = "t=? ns"
    else:
        t_label = f"t={t_ns:.3f} ns"

    # Use full-res background (24x24) and a modest arrow subsample
    arrow_stride = 2

    plot_scalar_with_quiver(
        m,
        "mx",
        f"{base} ({t_label}) - mx bg + arrows",
        out_dir / f"{base}_mx.png",
        arrow_stride=arrow_stride,
        cmap="viridis",
    )

    plot_scalar_with_quiver(
        m,
        "my",
        f"{base} ({t_label}) - my bg + arrows",
        out_dir / f"{base}_my.png",
        arrow_stride=arrow_stride,
        cmap="viridis",
    )

    plot_scalar_with_quiver(
        m,
        "mz",
        f"{base} ({t_label}) - mz bg + arrows",
        out_dir / f"{base}_mz.png",
        arrow_stride=arrow_stride,
        cmap="viridis",
    )


def preset_sk(ovf_path: Path, out_dir: Path, msat: float = 5.8e5):
    """SK: skyrmion snapshots (thin film).

    Produces one PNG per OVF: mz colormap (scaled by Msat) + in-plane arrows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    m = load_slice_field(ovf_path)
    base = ovf_path.stem

    # Time label from OVF header (ns)
    t_ns = try_parse_time_ns_from_ovf(ovf_path)
    if t_ns is None:
        t_label = "t=? ns"
    else:
        t_label = f"t={t_ns:.3f} ns"

    plot_skyrmion_mz_with_quiver(
        m,
        title=f"{base} ({t_label}) - skyrmion (mz + arrows)",
        out_path=out_dir / f"{base}_sk.png",
        arrow_stride=4,
        cmap="viridis",
        msat=msat,
    )

def preset_sp2_group(input_dir: Path, out_dir: Path):
    """SP2: group OVFs by remanence vs coercivity across d/lex."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ovfs = sorted(input_dir.glob("m_d*_*.ovf"))
    if not ovfs:
        # fallback to any m*.ovf
        ovfs = sorted(input_dir.glob("m*.ovf"))

    rem = []
    hc = []
    for p in ovfs:
        if p.name.endswith("_rem.ovf"):
            rem.append(p)
        elif p.name.endswith("_hc.ovf"):
            hc.append(p)

    def d_from_name(p: Path) -> int:
        m = re.search(r"m_d(\d+)_", p.name)
        return int(m.group(1)) if m else 0

    rem = sorted(rem, key=d_from_name, reverse=True)
    hc = sorted(hc, key=d_from_name, reverse=True)

    rem_dir = out_dir / "remanence"
    hc_dir = out_dir / "coercivity"
    rem_dir.mkdir(parents=True, exist_ok=True)
    hc_dir.mkdir(parents=True, exist_ok=True)

    # For each OVF: make 2 summary plots (mx scalar + colored vectors on mx)
    def process_list(lst, target_dir: Path, tag: str):
        for p in lst:
            m = load_slice_field(p)
            dval = d_from_name(p)
            # Aspect-aware resample for vectors
            rs = resample_for_aspect(m, target_nx=25, min_ny=5, max_ny=25)
            m_small = m.resample(rs)

            base = f"d{dval:02d}_{tag}"

            # Component scalars (mx, my, mz)
            plot_scalar(m.x, f"{base} - mx", "mx", target_dir / f"{base}_mx.png", figsize=None)
            plot_scalar(m.y, f"{base} - my", "my", target_dir / f"{base}_my.png", figsize=None)
            plot_scalar(m.z, f"{base} - mz", "mz", target_dir / f"{base}_mz.png", figsize=None)

            # Colored vectors on mx (mx is most informative for SP2 switching patterns)
            plot_vectors_colored(
                m_small,
                f"{base} - vectors (mx bg)",
                target_dir / f"{base}_vectors_colored.png",
                bg_scalar=m.x,
                bg_label="mx",
                cmap="magma",
                alpha=0.7,
                figsize=None,
            )

    if rem:
        process_list(rem, rem_dir, "rem")
    if hc:
        process_list(hc, hc_dir, "hc")

    # If we didn't match naming, fall back to generic per-file plots
    if not rem and not hc:
        generic_dir = out_dir / "generic"
        generic_dir.mkdir(parents=True, exist_ok=True)
        for p in ovfs:
            preset_generic(p, generic_dir)


def preset_generic(ovf_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    m = load_slice_field(ovf_path)
    base = ovf_path.stem

    plot_scalar(m.x, f"{base} - mx", "mx", out_dir / f"{base}_mx.png", figsize=(6, 4))
    plot_scalar(m.y, f"{base} - my", "my", out_dir / f"{base}_my.png", figsize=(6, 4))
    plot_scalar(m.z, f"{base} - mz", "mz", out_dir / f"{base}_mz.png", figsize=(6, 4))

    rs = resample_for_aspect(m, target_nx=20, min_ny=5, max_ny=20)
    m_small = m.resample(rs)
    plot_vectors(m_small, f"{base} - vectors", out_dir / f"{base}_vectors.png", figsize=(6, 4))


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise OVF magnetisation snapshots with problem-specific presets (sp4/sp2/fmr).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=str, required=True, help="Path to folder containing OVFs (or SP4 root with case subfolders).")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots.")
    parser.add_argument("--source", type=str, choices=["auto", "mumax", "rust"], default="auto", help="Override source detection.")
    parser.add_argument("--problem", type=str, choices=["auto", "sp4", "sp2", "fmr", "sk", "generic"], default="auto", help="Select plotting preset.")
    parser.add_argument("--msat", type=float, default=8e5, help="Msat (A/m) used for FMR mz scaling in Ubermag-style plot.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}")
        return 1

    case_dirs = find_ovf_case_dirs(input_path)
    if not case_dirs:
        print(f"Error: No OVF files found in input path or its known subdirectories: {input_path}")
        return 1

    # Determine source
    if args.source == "auto":
        source = infer_source(input_path)
    else:
        source = args.source

    # Determine problem preset
    if args.problem == "auto":
        problem = detect_problem(input_path, case_dirs)
    else:
        problem = args.problem

    # Determine output path
    if args.output is not None:
        out_root = Path(args.output)
    else:
        out_root = Path("plots") / f"{problem}_{source}"

    print("=" * 70)
    print(f"Mag Visualisation ({problem}, {source})")
    print("=" * 70)
    print(f"Input:  {input_path}")
    print(f"Output: {out_root}")
    print("=" * 70)

    # Special handling for SP2: we expect a single directory with many OVFs
    if problem == "sp2":
        if len(case_dirs) != 1:
            # If we got multiple dirs, process each as a group
            for d in case_dirs:
                preset_sp2_group(d, out_root / d.name)
        else:
            preset_sp2_group(case_dirs[0], out_root)
        return 0

    # SP4: preserve case subfolders if present
    if problem == "sp4" and len(case_dirs) > 1:
        for d in case_dirs:
            case_label = d.name
            out_dir = out_root / case_label
            for ovf in sorted(d.glob("m*.ovf")):
                preset_sp4(ovf, out_dir)
        return 0

    # For FMR/generic or single directory: process all OVFs in each case_dir
    for d in case_dirs:
        out_dir = out_root
        if len(case_dirs) > 1:
            out_dir = out_root / d.name

        ovfs = sorted(d.glob("m*.ovf"))
        if not ovfs:
            continue

        for ovf in ovfs:
            if problem == "fmr":
                preset_fmr(ovf, out_dir, msat=args.msat)
            elif problem == "sk":
                preset_sk(ovf, out_dir, msat=args.msat)
            elif problem == "sp4":
                preset_sp4(ovf, out_dir)
            else:
                preset_generic(ovf, out_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
