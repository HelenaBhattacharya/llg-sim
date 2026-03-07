#!/usr/bin/env python3
"""
amr_viewer.py — AMR benchmark visualiser with multi-angle 3D plots.

Reads OVF 2.0 ASCII files written by amr_vortex_relax (and sibling benchmarks)
and generates a plots/ folder with:

  1. 2D top-down mz map          (with AMR patch mesh overlaid)
  2. 2D top-down in-plane angle  (with AMR patch mesh overlaid)
  3. 2D AMR vs fine diff map
  4. 3D warped surface — isometric perspective  (mesh overlaid)
  5. 3D warped surface — top-down (elev=90)    (mesh overlaid)
  6. 3D warped surface — side view (elev=5)     (mesh overlaid)
  7. 3D warped surface — front view (elev=5, azim=0)
  8. AMR patch-level layout diagram
  9. 2x2 summary panel (best for presentations)

All images written to:  <root>/plots/step_XXXXXXX_<tag>.png

Step argument accepts an integer (e.g. 100, 300) or the string "latest".

Examples:
  python amr_viewer.py --root out/amr_vortex_relax --step latest
  python amr_viewer.py --root out/amr_vortex_relax --step 0
  python amr_viewer.py --root out/amr_vortex_relax --step 300 --warp-scale 0.8
  python amr_viewer.py --root out/amr_vortex_relax --step 300 --no-mesh
  python amr_viewer.py --root out/amr_vortex_relax --step latest --summary-only
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OvfMeta:
    nx: int; ny: int
    dx: float; dy: float; dz: float
    xbase: float; ybase: float; zbase: float
    title: str

@dataclass
class OvfField:
    meta: OvfMeta
    m: np.ndarray          # (ny, nx, 3) float64

@dataclass(frozen=True)
class Rect:
    """Patch rectangle in coarse-grid index space."""
    i0: int; j0: int; nx: int; ny: int
    @property
    def i1(self): return self.i0 + self.nx
    @property
    def j1(self): return self.j0 + self.ny


# ──────────────────────────────────────────────────────────────────────────────
# OVF loading
# ──────────────────────────────────────────────────────────────────────────────

_KV_RE = re.compile(r"^\#\s*([^:]+)\s*:\s*(.*)$")

def _kv(lines):
    d = {}
    for ln in lines:
        m = _KV_RE.match(ln)
        if m: d[m.group(1).strip().lower()] = m.group(2).strip()
    return d

def load_ovf_text(path: Path) -> OvfField:
    txt = path.read_text().splitlines()
    try:
        i0 = next(i for i,ln in enumerate(txt) if ln.strip() == "# Begin: Data Text")
        i1 = next(i for i,ln in enumerate(txt) if ln.strip() == "# End: Data Text")
    except StopIteration as e:
        raise ValueError(f"{path}: missing Begin/End Data Text") from e
    kv = _kv(txt[:i0])
    nx, ny = int(kv["xnodes"]), int(kv["ynodes"])
    dx, dy, dz = float(kv["xstepsize"]), float(kv["ystepsize"]), float(kv["zstepsize"])
    xb = float(kv.get("xbase","0")); yb = float(kv.get("ybase","0")); zb = float(kv.get("zbase","0"))
    title = kv.get("title", path.name)
    floats = []
    for ln in txt[i0+1:i1]:
        s = ln.strip()
        if not s or s.startswith("#"): continue
        parts = s.split()
        if len(parts) >= 3: floats.extend(float(p) for p in parts[:3])
    arr = np.array(floats, dtype=np.float64)
    if arr.size != nx*ny*3:
        raise ValueError(f"{path}: expected {nx*ny*3} floats, got {arr.size}")
    meta = OvfMeta(nx=nx, ny=ny, dx=dx, dy=dy, dz=dz, xbase=xb, ybase=yb, zbase=zb, title=title)
    return OvfField(meta=meta, m=arr.reshape((ny, nx, 3)))

def find_latest_step(folder: Path) -> int:
    steps = [int(m.group(1)) for p in folder.glob("m*.ovf")
             if (m := re.match(r"m(\d+)\.ovf$", p.name))]
    if not steps: raise FileNotFoundError(f"No OVF files in {folder}")
    return max(steps)


# ──────────────────────────────────────────────────────────────────────────────
# Patch CSV  (regrid_patches.csv  — step,level,patch_id,i0,j0,nx,ny)
# ──────────────────────────────────────────────────────────────────────────────

def load_regrid_patches_csv(path: Path) -> List[Dict]:
    if not path.exists(): return []
    lines = path.read_text().splitlines()
    if len(lines) < 2: return []
    header = lines[0].strip().split(",")
    idx = {k: i for i,k in enumerate(header)}
    rows = []
    for ln in lines[1:]:
        s = ln.strip()
        if not s: continue
        parts = s.split(",")
        if len(parts) < len(header): continue
        rows.append({k: int(parts[idx[k]]) for k in header if k in idx})
    rows.sort(key=lambda r: (r["step"], r["level"], r["patch_id"]))
    return rows

def patch_rects_for_step(rows: List[Dict], step: int,
                          max_level: int = 3) -> Dict[int, List[Rect]]:
    out: Dict[int, List[Rect]] = {l: [] for l in range(1, max_level+1)}
    if not rows: return out
    valid_steps = [r["step"] for r in rows if r["step"] <= step]
    latest = max(valid_steps) if valid_steps else rows[0]["step"]
    for r in rows:
        if r["step"] != latest: continue
        lvl = r.get("level", 0)
        if 1 <= lvl <= max_level:
            out[lvl].append(Rect(i0=r["i0"], j0=r["j0"], nx=r["nx"], ny=r["ny"]))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────────────────────────────────────

def angle_rgb(m: np.ndarray) -> np.ndarray:
    phi = np.arctan2(m[...,1], m[...,0])
    h = (phi + np.pi) / (2*np.pi)
    return hsv_to_rgb(np.stack([h, np.ones_like(h), np.ones_like(h)], axis=-1))

def angle_scalar(m: np.ndarray) -> np.ndarray:
    return (np.arctan2(m[...,1], m[...,0]) + np.pi) / (2*np.pi)


# ──────────────────────────────────────────────────────────────────────────────
# Mesh overlay
# ──────────────────────────────────────────────────────────────────────────────

_LCOL = {1: "black",    2: "crimson",   3: "dodgerblue"}
_LLBL = {1: "L1 patch", 2: "L2 patch",  3: "L3 patch"}
_LLW  = {1: 2.0,        2: 2.0,         3: 2.0}

def _norm(rect: Rect, cnx: int, cny: int):
    return rect.i0/cnx, rect.j0/cny, rect.i1/cnx, rect.j1/cny

def draw_patches_2d(ax, level_rects, coarse_nx, coarse_ny, field_nx, field_ny):
    sx = field_nx / coarse_nx;  sy = field_ny / coarse_ny
    for lvl, rects in sorted(level_rects.items()):
        col = _LCOL.get(lvl, "gray");  lw = _LLW.get(lvl, 1.5)
        for r in rects:
            ax.add_patch(mpatches.Rectangle(
                (r.i0*sx - 0.5, r.j0*sy - 0.5), r.nx*sx, r.ny*sy,
                linewidth=lw, edgecolor=col, facecolor="none", zorder=5))

def draw_patches_3d(ax, level_rects, coarse_nx, coarse_ny, mz_field, warp_scale, n=60):
    ny_f, nx_f = mz_field.shape
    def z_at(xn, yn):
        ii = int(np.clip(xn*nx_f, 0, nx_f-1));  jj = int(np.clip(yn*ny_f, 0, ny_f-1))
        return float(warp_scale * mz_field[jj, ii])
    for lvl, rects in sorted(level_rects.items()):
        col = _LCOL.get(lvl, "gray");  lw = 1.5 + (lvl==2)*0.5
        for r in rects:
            x0,y0,x1,y1 = _norm(r, coarse_nx, coarse_ny)
            edges = [
                (np.linspace(x0,x1,n), np.full(n,y0)),
                (np.linspace(x0,x1,n), np.full(n,y1)),
                (np.full(n,x0), np.linspace(y0,y1,n)),
                (np.full(n,x1), np.linspace(y0,y1,n)),
            ]
            for ex,ey in edges:
                ez = np.array([z_at(xx,yy)+0.01 for xx,yy in zip(ex,ey)])
                ax.plot(ex, ey, ez, color=col, linewidth=lw, zorder=10)

def _patch_legend(ax, level_rects):
    handles = [mpatches.Patch(edgecolor=_LCOL.get(l,"gray"), facecolor="none",
                              linewidth=1.5, label=_LLBL.get(l,f"L{l}"))
               for l in sorted(level_rects) if level_rects[l]]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.85)


# ──────────────────────────────────────────────────────────────────────────────
# 2D plots
# ──────────────────────────────────────────────────────────────────────────────

def save_2d_mz(out, field, step, lr, cnx, cny, vmin=None, vmax=None):
    ny, nx = field.meta.ny, field.meta.nx
    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(field.m[...,2], origin="lower", cmap="RdBu_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("$m_z$", fontsize=12)
    draw_patches_2d(ax, lr, cnx, cny, nx, ny);  _patch_legend(ax, lr)
    ax.set_title(f"AMR  $m_z$  (step {step})", fontsize=13, fontweight="bold")
    ax.set_xlabel("x  [cells]", fontsize=11);  ax.set_ylabel("y  [cells]", fontsize=11)
    fig.tight_layout();  fig.savefig(out, dpi=180, bbox_inches="tight");  plt.close(fig)
    print(f"  saved: {Path(out).name}")

def save_2d_angle(out, field, step, lr, cnx, cny):
    ny, nx = field.meta.ny, field.meta.nx
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.imshow(angle_rgb(field.m), origin="lower", interpolation="nearest")
    draw_patches_2d(ax, lr, cnx, cny, nx, ny);  _patch_legend(ax, lr)
    ax.set_title(f"AMR  in-plane angle  (step {step})", fontsize=13, fontweight="bold")
    ax.set_xlabel("x  [cells]", fontsize=11);  ax.set_ylabel("y  [cells]", fontsize=11)
    fig.tight_layout();  fig.savefig(out, dpi=180, bbox_inches="tight");  plt.close(fig)
    print(f"  saved: {Path(out).name}")

def save_2d_diff(out, amr, fine, step):
    if (amr.meta.nx, amr.meta.ny) != (fine.meta.nx, fine.meta.ny):
        print(f"  [skip diff] grid mismatch"); return
    dm = np.linalg.norm(amr.m - fine.m, axis=-1)
    rmse = float(np.sqrt(np.mean(dm**2)))
    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(dm, origin="lower", cmap="hot_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(r"$|\Delta\mathbf{m}|$", fontsize=12)
    ax.set_title(f"|AMR − fine|   step {step}   RMSE = {rmse:.3e}", fontsize=12, fontweight="bold")
    ax.set_xlabel("x  [cells]", fontsize=11);  ax.set_ylabel("y  [cells]", fontsize=11)
    fig.tight_layout();  fig.savefig(out, dpi=180, bbox_inches="tight");  plt.close(fig)
    print(f"  saved: {Path(out).name}")

def save_2d_patch_layout(out, lr, cnx, cny, step):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(0,1);  ax.set_ylim(0,1);  ax.set_aspect("equal");  ax.set_facecolor("#f5f5f5")
    fill = {1:"#FFD700", 2:"#32CD32", 3:"#1E90FF"}
    for lvl in sorted(lr):
        for r in lr[lvl]:
            x0,y0,x1,y1 = _norm(r, cnx, cny)
            ax.fill([x0,x1,x1,x0], [y0,y0,y1,y1], color=fill.get(lvl,"#aaa"), alpha=0.55, zorder=lvl)
            ax.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], color=_LCOL.get(lvl,"gray"), linewidth=2, zorder=lvl+10)
    _patch_legend(ax, lr)
    ax.set_title(f"AMR patch layout  (step {step})", fontsize=13, fontweight="bold")
    ax.set_xlabel("x/L", fontsize=11);  ax.set_ylabel("y/L", fontsize=11);  ax.grid(True, alpha=0.3)
    fig.tight_layout();  fig.savefig(out, dpi=180, bbox_inches="tight");  plt.close(fig)
    print(f"  saved: {Path(out).name}")


# ──────────────────────────────────────────────────────────────────────────────
# 3D warped-surface plots  (four camera angles)
# ──────────────────────────────────────────────────────────────────────────────

_CAMERAS = [
    # (file_label, elev, azim, title_suffix)
    ("perspective", 28, -55, "perspective"),
    ("top",         88, -90, "top-down"),
    ("side",         8, -90, "side"),
    ("front",        8,   0, "front"),
]

def save_3d_views(out_dir, stem, field, lr, cnx, cny, step,
                  warp_scale=0.5, show_mesh=True, color_by="mz"):
    ny, nx = field.meta.ny, field.meta.nx
    mz = field.m[...,2]
    xs = np.linspace(0, 1, nx);  ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys);  Z = warp_scale * mz
    stride = max(1, min(nx,ny) // 80)

    if color_by == "angle":
        C = angle_scalar(field.m);  cmap = "hsv";  clabel = "angle"
    else:
        C = mz;  cmap = "RdBu_r";  clabel = "$m_z$"

    norm = mcolors.Normalize(vmin=float(C.min()), vmax=float(C.max()))

    for (label, elev, azim, title_sfx) in _CAMERAS:
        fig = plt.figure(figsize=(9, 7))
        ax: Any = fig.add_subplot(111, projection="3d")

        ax.plot_surface(X, Y, Z,
            facecolors=plt.get_cmap(cmap)(norm(C)),
            rstride=stride, cstride=stride,
            linewidth=0, antialiased=True, shade=True, alpha=0.92)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.08, aspect=18).set_label(clabel, fontsize=10)

        if show_mesh:
            draw_patches_3d(ax, lr, cnx, cny, mz, warp_scale)

        ax.set_xlabel("x/L", fontsize=9, labelpad=6)
        ax.set_ylabel("y/L", fontsize=9, labelpad=6)
        ax.set_zlabel(f"$m_z$ × {warp_scale:.1f}", fontsize=9, labelpad=6)
        ax.set_title(f"AMR $m_z$ warp — {title_sfx}  (step {step})",
                     fontsize=11, fontweight="bold", pad=10)
        ax.view_init(elev=elev, azim=azim)
        ax.set_facecolor("#f8f8f8");  fig.patch.set_facecolor("#f8f8f8")

        out = Path(out_dir) / f"{stem}_3d_{label}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight");  plt.close(fig)
        print(f"  saved: {out.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary panel  (2 × 2, best for presentations)
# ──────────────────────────────────────────────────────────────────────────────

def save_summary_panel(out, amr, fine, lr, cnx, cny, step, warp_scale=0.5):
    ny, nx = amr.meta.ny, amr.meta.nx
    mz = amr.m[...,2]
    xs = np.linspace(0,1,nx);  ys = np.linspace(0,1,ny)
    X, Y = np.meshgrid(xs, ys);  Z = warp_scale * mz
    stride = max(1, min(nx,ny) // 60)

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(f"AMR Vortex Relaxation — step {step}",
                 fontsize=15, fontweight="bold", y=0.98)

    # A: mz top-down + mesh
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(mz, origin="lower", cmap="RdBu_r", interpolation="nearest")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label("$m_z$", fontsize=10)
    draw_patches_2d(ax1, lr, cnx, cny, nx, ny);  _patch_legend(ax1, lr)
    ax1.set_title("$m_z$  (top-down + mesh)", fontsize=11)
    ax1.set_xlabel("x [cells]", fontsize=9);  ax1.set_ylabel("y [cells]", fontsize=9)

    # B: in-plane angle + mesh
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(angle_rgb(amr.m), origin="lower", interpolation="nearest")
    draw_patches_2d(ax2, lr, cnx, cny, nx, ny);  _patch_legend(ax2, lr)
    ax2.set_title("In-plane angle  (top-down + mesh)", fontsize=11)
    ax2.set_xlabel("x [cells]", fontsize=9);  ax2.set_ylabel("y [cells]", fontsize=9)

    # C: 3D perspective + mesh
    ax3: Any = fig.add_subplot(2, 2, 3, projection="3d")
    norm3 = mcolors.Normalize(vmin=float(mz.min()), vmax=float(mz.max()))
    ax3.plot_surface(X, Y, Z,
        facecolors=plt.get_cmap("RdBu_r")(norm3(mz)),
        rstride=stride, cstride=stride, linewidth=0, antialiased=True, shade=True, alpha=0.92)
    draw_patches_3d(ax3, lr, cnx, cny, mz, warp_scale)
    ax3.view_init(elev=28, azim=-55)
    ax3.set_xlabel("x/L", fontsize=8);  ax3.set_ylabel("y/L", fontsize=8)
    ax3.set_zlabel(f"$m_z$×{warp_scale}", fontsize=8)
    ax3.set_title("3D warp — perspective (+ mesh)", fontsize=11)

    # D: diff or patch layout
    ax4 = fig.add_subplot(2, 2, 4)
    if fine is not None and (fine.meta.nx, fine.meta.ny) == (nx, ny):
        dm = np.linalg.norm(amr.m - fine.m, axis=-1)
        rmse = float(np.sqrt(np.mean(dm**2)))
        im4 = ax4.imshow(dm, origin="lower", cmap="hot_r", interpolation="nearest")
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).set_label(r"$|\Delta\mathbf{m}|$", fontsize=10)
        ax4.set_title(f"|AMR − fine|   RMSE = {rmse:.3e}", fontsize=11)
        ax4.set_xlabel("x [cells]", fontsize=9);  ax4.set_ylabel("y [cells]", fontsize=9)
    else:
        ax4.set_xlim(0,1);  ax4.set_ylim(0,1);  ax4.set_aspect("equal")
        ax4.set_facecolor("#f0f0f0")
        fill = {1:"#FFD700",2:"#32CD32",3:"#1E90FF"}
        for lvl in sorted(lr):
            for r in lr[lvl]:
                x0,y0,x1,y1 = _norm(r, cnx, cny)
                ax4.fill([x0,x1,x1,x0],[y0,y0,y1,y1], color=fill.get(lvl,"#aaa"), alpha=0.6)
                ax4.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0], color=_LCOL.get(lvl,"gray"), linewidth=2)
        _patch_legend(ax4, lr)
        ax4.set_title("Patch layout", fontsize=11)
        ax4.set_xlabel("x/L", fontsize=9);  ax4.set_ylabel("y/L", fontsize=9)
        ax4.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=160, bbox_inches="tight");  plt.close(fig)
    print(f"  saved: {Path(out).name}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="amr_viewer — multi-angle plots for AMR benchmarks")
    ap.add_argument("--root",        required=True, help="Benchmark output root, e.g. out/amr_vortex_relax")
    ap.add_argument("--step",        default="latest", help="Step number or 'latest'")
    ap.add_argument("--out",         default=None,  help="Output folder (default: <root>/plots)")
    ap.add_argument("--warp-scale",  type=float, default=0.5, help="3D warp height scale (default 0.5)")
    ap.add_argument("--color-by",    default="mz", choices=["mz","angle"], help="3D surface colour")
    ap.add_argument("--no-mesh",     action="store_true", help="Disable patch mesh overlay")
    ap.add_argument("--no-3d",       action="store_true", help="Skip 3D plots (faster)")
    ap.add_argument("--no-panel",    action="store_true", help="Skip summary panel")
    ap.add_argument("--summary-only",action="store_true", help="Only generate summary panel")
    args = ap.parse_args()

    root    = Path(args.root)
    out_dir = Path(args.out) if args.out else (root / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    show_mesh = not args.no_mesh

    folders = {k: root / f"ovf_{k}" for k in ("coarse","fine","amr")}
    for k,p in folders.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k} OVF folder: {p}")

    step = find_latest_step(folders["amr"]) if args.step == "latest" else int(args.step)
    fname = f"m{step:07d}.ovf"
    print(f"[amr_viewer] root={root}  step={step}  file={fname}")
    print(f"[amr_viewer] writing plots to {out_dir}")

    missing = [str(folders[k]/fname) for k in ("coarse","fine","amr") if not (folders[k]/fname).exists()]
    if missing:
        raise FileNotFoundError(f"Missing OVF(s): {', '.join(missing)}")

    coarse = load_ovf_text(folders["coarse"] / fname)
    fine   = load_ovf_text(folders["fine"]   / fname)
    amr    = load_ovf_text(folders["amr"]    / fname)
    cnx, cny = coarse.meta.nx, coarse.meta.ny
    anx, any_ = amr.meta.nx, amr.meta.ny
    print(f"  coarse: {cnx}×{cny}   fine: {fine.meta.nx}×{fine.meta.ny}   AMR: {anx}×{any_}")

    # Patch data
    lr: Dict[int, List[Rect]] = {1:[], 2:[], 3:[]}
    patches_csv = root / "regrid_patches.csv"
    if patches_csv.exists() and show_mesh:
        rows = load_regrid_patches_csv(patches_csv)
        lr   = patch_rects_for_step(rows, step, max_level=3)
        total = sum(len(v) for v in lr.values())
        print(f"  patches: {total} total | " +
              " | ".join(f"L{l}: {len(lr[l])}" for l in sorted(lr) if lr[l]))
    else:
        show_mesh = False
        print("  [mesh overlay disabled]")

    mz_all = np.concatenate([coarse.m[...,2].ravel(), fine.m[...,2].ravel(), amr.m[...,2].ravel()])
    vmin, vmax = float(mz_all.min()), float(mz_all.max())
    stem = f"step_{step:07d}"

    if not args.summary_only:
        print("\n[amr_viewer] 2D plots...")
        save_2d_mz(out_dir/f"{stem}_2d_mz.png",    amr, step, lr if show_mesh else {}, cnx, cny, vmin, vmax)
        save_2d_angle(out_dir/f"{stem}_2d_angle.png", amr, step, lr if show_mesh else {}, cnx, cny)
        save_2d_diff(out_dir/f"{stem}_2d_diff.png",   amr, fine, step)
        save_2d_patch_layout(out_dir/f"{stem}_patch_layout.png", lr, cnx, cny, step)

        if not args.no_3d:
            print("\n[amr_viewer] 3D views (4 camera angles)...")
            save_3d_views(out_dir, stem, amr, lr if show_mesh else {},
                          cnx, cny, step, args.warp_scale, show_mesh, args.color_by)

    if not args.no_panel:
        print("\n[amr_viewer] summary panel...")
        save_summary_panel(out_dir/f"{stem}_summary.png",
                           amr, fine, lr if show_mesh else {},
                           cnx, cny, step, args.warp_scale)

    print(f"\n[amr_viewer] done — {out_dir}")


if __name__ == "__main__":
    main()