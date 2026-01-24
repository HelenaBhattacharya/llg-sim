import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
RUN_DIR = Path("out/bloch_relax")
SLICES_DIR = RUN_DIR / "bloch_slices"

# Choose which slice to plot:
# options: "t0", "t5ns", "t10ns", "final"
TAG = "t10ns"

SLICE_FILE = SLICES_DIR / f"rust_slice_{TAG}.csv"

# -------------------------------------------------
# Load config.json to get DMI value
# -------------------------------------------------
with open(RUN_DIR / "config.json", "r") as f:
    cfg = json.load(f)

dmi = cfg["fields"]["dmi"]

# -------------------------------------------------
# Load slice data
# Supports:
#   - x,mx,mz
#   - x,mx,my,mz
# -------------------------------------------------
data = np.loadtxt(SLICE_FILE, delimiter=",", skiprows=1)

x_nm = data[:, 0] * 1e9  # convert to nm

if data.shape[1] == 3:
    # x, mx, mz
    mx = data[:, 1]
    mz = data[:, 2]

    # Reconstruct my from |m|=1 (sign ambiguous)
    my = np.sqrt(np.maximum(0.0, 1.0 - mx**2 - mz**2))

    # Choose a consistent sign using the centre cell (crude but OK for quick checks)
    mid = len(my) // 2
    my *= np.sign(mx[mid]) if mx[mid] != 0 else 1.0

elif data.shape[1] == 4:
    # x, mx, my, mz
    mx = data[:, 1]
    my = data[:, 2]
    mz = data[:, 3]
else:
    raise ValueError(f"Unexpected number of columns in slice file: {data.shape[1]}")

# Optional: centre x-axis so wall centre is near 0
x_nm = x_nm - np.mean(x_nm)

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(x_nm, mz, label=r"$m_z$", linewidth=2)
plt.plot(x_nm, mx, label=r"$m_x$", linewidth=2)
plt.plot(x_nm, my, label=r"$m_y$", linewidth=2)

plt.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
plt.xlabel("x − x₀ (nm)")
plt.ylabel("magnetisation")
plt.title(f"Bloch / Néel wall slice ({TAG}) (DMI = {dmi} J/m$^2$)")
plt.legend()
plt.tight_layout()
plt.show()