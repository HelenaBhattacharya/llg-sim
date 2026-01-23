import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
RUN_DIR = Path("out/bloch_relax")
SLICE_FILE = RUN_DIR / "bloch_slices" / "rust_slice_final.csv"

# -------------------------------------------------
# Load config.json to get DMI value
# -------------------------------------------------
with open(RUN_DIR / "config.json", "r") as f:
    cfg = json.load(f)

dmi = cfg["fields"]["dmi"]

# -------------------------------------------------
# Load slice data
# -------------------------------------------------
data = np.loadtxt(SLICE_FILE, delimiter=",", skiprows=1)

x = data[:, 0] * 1e9  # convert to nm
mx = data[:, 1]
mz = data[:, 2]

# NOTE:
# rust_slice files currently store x, mx, mz
# We reconstruct my using normalization if needed
my = np.sqrt(np.maximum(0.0, 1.0 - mx**2 - mz**2))

# Restore sign using central region
mid = len(my) // 2
my *= np.sign(mx[mid])  # crude but effective for chirality

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure(figsize=(7, 4))

plt.plot(x, mz, label=r"$m_z$", linewidth=2)
plt.plot(x, mx, label=r"$m_x$", linewidth=2)
plt.plot(x, my, label=r"$m_y$", linewidth=2)

plt.axhline(0.0, color="k", linestyle=":", linewidth=0.8)
plt.xlabel("x (nm)")
plt.ylabel("magnetisation")
plt.title(f"Bloch / Néel wall slice (DMI = {dmi} J/m$^2$)")
plt.legend()
plt.tight_layout()
plt.show()