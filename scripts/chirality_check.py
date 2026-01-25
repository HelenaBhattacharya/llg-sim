import numpy as np
from pathlib import Path
import argparse

def load_slice(run_dir: str, tag: str):
    f = Path(run_dir) / "bloch_relax" / "bloch_slices" / f"rust_slice_{tag}.csv"
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    x = d[:, 0]
    mx, my, mz = d[:, 1], d[:, 2], d[:, 3]
    return x, mx, my, mz

def wall_center_index(mz: np.ndarray) -> int:
    return int(np.argmin(np.abs(mz)))  # closest to mz=0

def window_stats(mx, my, mz, i0: int, half_window: int):
    n = len(mx)
    i_lo = max(0, i0 - half_window)
    i_hi = min(n, i0 + half_window + 1)

    mx_w = mx[i_lo:i_hi]
    my_w = my[i_lo:i_hi]
    mz_w = mz[i_lo:i_hi]

    mx_mean = float(np.mean(mx_w))
    my_mean = float(np.mean(my_w))
    mz_mean = float(np.mean(mz_w))

    phi_mean = float(np.arctan2(my_mean, mx_mean))
    return (i_lo, i_hi, mx_mean, my_mean, mz_mean, phi_mean)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="out/bloch_relax")
    ap.add_argument("--tag", default="final", choices=["t0","t5ns","t10ns","final"])
    ap.add_argument("--half_window", type=int, default=10, help="window half-width in cells")
    args = ap.parse_args()

    x, mx, my, mz = load_slice(args.run_dir, args.tag)
    i0 = wall_center_index(mz)

    phi_point = float(np.arctan2(my[i0], mx[i0]))
    i_lo, i_hi, mx_mean, my_mean, mz_mean, phi_mean = window_stats(mx, my, mz, i0, args.half_window)

    print(f"run_dir: {args.run_dir}")
    print(f"tag: {args.tag}")
    print(f"wall index i0={i0}, x={x[i0]*1e9:.3f} nm, mz(i0)={mz[i0]:+.6f}")
    print(f"point: mx={mx[i0]:+.6f}, my={my[i0]:+.6f}, phi=atan2(my,mx)={phi_point:+.6f} rad")
    print(f"window: [{i_lo}:{i_hi}] (N={i_hi-i_lo})")
    print(f"  <mx>={mx_mean:+.6f}, <my>={my_mean:+.6f}, <mz>={mz_mean:+.6f}")
    print(f"  phi_window=atan2(<my>,<mx>)={phi_mean:+.6f} rad")
    print(f"max|mx|={np.max(np.abs(mx)):.6f}, max|my|={np.max(np.abs(my)):.6f}")

if __name__ == "__main__":
    main()