#!/usr/bin/env python3
"""compare_sk1.py — visualise FFT vs Poisson-MG demag on sk1 outputs.

Reads OVFs written by sk1.rs under:
  runs/st_problems/sk1/<case>_rust/

Expected files:
  m.ovf
  b_fft.ovf
  b_mg_env.ovf
  b_diff_env.ovf   (optional; if missing we compute diff)

Outputs PNGs into:
  runs/st_problems/sk1/<case>_rust/plots/

Usage:
  python3 scripts/compare_sk1.py --case sk1a
  python3 scripts/compare_sk1.py --case sk1a --root runs/st_problems/sk1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


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


def load_ovf_text(path: Path) -> Tuple[Dict[str, str], np.ndarray]:
    """Minimal OVF 2.0 ASCII reader.

    Returns: (meta_kv, arr) where arr shape = (ny, nx, 3)
    Assumes x-fastest then y ordering.
    """
    txt = path.read_text().splitlines()
    try:
        i0 = next(i for i, ln in enumerate(txt) if ln.strip() == "# Begin: Data Text")
        i1 = next(i for i, ln in enumerate(txt) if ln.strip() == "# End: Data Text")
    except StopIteration as e:
        raise ValueError(f"{path}: missing '# Begin: Data Text' / '# End: Data Text'") from e

    kv = _parse_header_kv(txt[:i0])

    def get_int(k: str) -> int:
        return int(kv[k])

    nx = get_int("xnodes")
    ny = get_int("ynodes")

    floats: List[float] = []
    for ln in txt[i0 + 1 : i1]:
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
    return kv, arr.reshape((ny, nx, 3))


def save_mz_with_glyphs(path: Path, m: np.ndarray, title: str, stride: int = 10) -> None:
    """Plot m_z as a heatmap and overlay light-grey in-plane glyph arrows (mx,my)."""
    mz = m[..., 2]
    mx = m[..., 0]
    my = m[..., 1]

    ny, nx = mz.shape
    xs = np.arange(0, nx, stride)
    ys = np.arange(0, ny, stride)
    X, Y = np.meshgrid(xs, ys)

    U = mx[ys[:, None], xs[None, :]]
    V = my[ys[:, None], xs[None, :]]

    plt.figure(figsize=(6, 6))
    im = plt.imshow(mz, origin="lower", cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # Normalise in-plane vectors for visual clarity
    mag = np.sqrt(U**2 + V**2)
    mag[mag == 0.0] = 1.0
    U_plot = U / mag
    V_plot = V / mag

    plt.quiver(
        X,
        Y,
        U_plot,
        V_plot,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=0.35,
        width=0.004,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=3.5,
        minlength=0.0,
        pivot="mid",
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_angle_with_glyphs(path: Path, m: np.ndarray, title: str, stride: int = 10) -> None:
    """Plot in-plane angle as HSV hue and overlay light-grey in-plane glyph arrows (mx,my)."""
    mx = m[..., 0]
    my = m[..., 1]

    phi = np.arctan2(my, mx)  # [-pi, pi]
    h = (phi + np.pi) / (2.0 * np.pi)  # [0,1)

    ny, nx = h.shape
    xs = np.arange(0, nx, stride)
    ys = np.arange(0, ny, stride)
    X, Y = np.meshgrid(xs, ys)

    U = mx[ys[:, None], xs[None, :]]
    V = my[ys[:, None], xs[None, :]]

    cmap = plt.get_cmap("hsv")
    rgb = cmap(h)[..., :3]

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb, origin="lower")

    # Normalise in-plane vectors for visual clarity
    mag = np.sqrt(U**2 + V**2)
    mag[mag == 0.0] = 1.0
    U_plot = U / mag
    V_plot = V / mag

    plt.quiver(
        X,
        Y,
        U_plot,
        V_plot,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=0.35,
        width=0.004,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=3.5,
        minlength=0.0,
        pivot="mid",
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_im(
    path: Path,
    img: np.ndarray,
    title: str,
    cmap: str | None = None,
    vmin=None,
    vmax=None,
) -> None:
    plt.figure(figsize=(6, 6))
    if img.ndim == 3:
        plt.imshow(img, origin="upper")
    else:
        plt.imshow(img, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_hist(path: Path, data: np.ndarray, title: str, bins: int = 200) -> None:
    x = data[np.isfinite(data)].ravel()
    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case",
        required=True,
        help="sk1 case tag, e.g. sk1a / sk1f (without _rust)",
    )
    ap.add_argument(
        "--root",
        default="runs/st_problems/sk1",
        help="root folder containing <case>_rust directories",
    )
    args = ap.parse_args()

    case_dir = Path(args.root) / f"{args.case}_rust"
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing: {case_dir}")

    plots = case_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    _, m = load_ovf_text(case_dir / "m.ovf")
    _, b_fft = load_ovf_text(case_dir / "b_fft.ovf")
    _, b_mg = load_ovf_text(case_dir / "b_mg_env.ovf")

    diff_path = case_dir / "b_diff_env.ovf"
    if diff_path.exists():
        _, b_diff = load_ovf_text(diff_path)
    else:
        b_diff = b_mg - b_fft

    db_mag = np.linalg.norm(b_diff, axis=-1)

    # Magnetisation visuals
    save_mz_with_glyphs(plots / "m_mz.png", m, f"{args.case}: m_z with in-plane glyphs")
    save_angle_with_glyphs(plots / "m_angle.png", m, f"{args.case}: in-plane angle with glyphs")

    # Field component maps (FFT vs MG)
    for comp, name in [(0, "Bx"), (1, "By"), (2, "Bz")]:
        vmin = np.nanmin(np.concatenate([b_fft[..., comp].ravel(), b_mg[..., comp].ravel()]))
        vmax = np.nanmax(np.concatenate([b_fft[..., comp].ravel(), b_mg[..., comp].ravel()]))

        save_im(
            plots / f"fft_{name}.png",
            b_fft[..., comp],
            f"{args.case}: FFT {name}",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        save_im(
            plots / f"mg_{name}.png",
            b_mg[..., comp],
            f"{args.case}: MG {name}",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        save_im(
            plots / f"diff_{name}.png",
            b_diff[..., comp],
            f"{args.case}: (MG-FFT) {name}",
            cmap="coolwarm",
        )

    # Difference magnitude + histograms
    save_im(plots / "diff_B_mag.png", db_mag, f"{args.case}: |ΔB| (MG-FFT)", cmap="magma")
    save_hist(plots / "hist_diff_B_mag.png", db_mag, f"{args.case}: histogram |ΔB|")

    # Midline lineouts (j = ny//2)
    j = db_mag.shape[0] // 2
    x = np.arange(db_mag.shape[1])

    plt.figure(figsize=(8, 4))
    plt.plot(x, b_fft[j, :, 2], label="FFT Bz")
    plt.plot(x, b_mg[j, :, 2], label="MG Bz")
    plt.plot(x, b_diff[j, :, 2], label="ΔBz (MG-FFT)")
    plt.title(f"{args.case}: midline (y=ny//2) Bz comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "lineout_mid_Bz.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, db_mag[j, :], label="|ΔB|")
    plt.title(f"{args.case}: midline |ΔB|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "lineout_mid_diff_B_mag.png", dpi=200)
    plt.close()

    print(f"[compare_sk1] wrote plots to {plots}")


if __name__ == "__main__":
    main()