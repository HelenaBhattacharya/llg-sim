# llg-sim - DRAFT

A modular micromagnetic solver for the Landau–Lifshitz–Gilbert (LLG) equation, written in Rust.

This repository is developed as part of the CMP11 MPhys project **“Building the Next-Generation Micromagnetic Simulator”**.
The focus is on clarity, modularity, and extensibility (e.g. additional energy terms / adaptive meshing), while benchmarking against **MuMax3**.

---

## Features

### Physics / effective fields
- Thin-film (2D) micromagnetics (Nx×Ny×1) with full 3D magnetisation vectors
- Exchange, uniaxial anisotropy, Zeeman field
- **Demagnetising field (demag)** (`src/effective_field/demag.rs`):
  - treats each finite difference cell as a uniformly magnetised rectangular prism
  - builds a demag kernel K_ij once per geometry and computes B_demag = K * M via FFT convolution
  - kernel is cached on disk (out/demag_cache/) to avoid recomputation across runs
- Interfacial DMI (Néel-type) for thin films

### Time integration and relaxation
- Integrators for LLG dynamics:
  - Explicit Euler
  - RK4 (frozen field)
  - RK4 (recompute field at sub-steps)
  - Adaptive RK45 (Dormand-Prince 5(4)) with recompute field
  - **Relaxation controller** (`src/relax.rs`):
    - precession suppressed (damping-only RHS)
    - adaptive **RK23** (Bogacki–Shampine 3(2)) relax stepper
    - energy-descent phase → torque-descent phase
    - tolerance tightening
    - Configurable torque-check stride

### Validation and comparison workflow
- Reproducible benchmark binaries under `src/bin/*`
- Python plotting/overlay scripts under `scripts/`
- MuMax3 reference scripts under `mumax/`
- Generated outputs are written to `out/` and `runs/` and `mumax_outputs/` (all ignored by git)

---

## Quickstart

### Prerequisites:
- Rust toolchain (main code): `rustc + cargo`
- Python3 (for plotting)
- Python packages: `numpy, matplotlib`
- Mumax3 on GPU machine/cluster (used for reference outputs)

### Clone and build
```bash 
git clone https://github.com/HelenaBhattacharya/llg-sim.git
cd llg-sim
cargo build --release
```
---

### Run tests
```bash 
cargo test
# or just the integration-style tests:
cargo test --test validation
```
---

### Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```
---
## Benchmarks and how to reproduce

### Standard Problem 4
Run Rust:
```bash
cargo run --release --bin st_problems -- sp4 a
cargo run --release --bin st_problems -- sp4 b
```

Run MuMax3 code on GPU and compare outputs:
```bash
python3 scripts/compare_sp4.py \
  --mumax-root mumax_outputs/st_problems/sp4 \
  --rust-root  runs/st_problems/sp4
```

### Uniform film field test (rk45 solver + demag ON)
Run Rust:
```bash
cargo run --release --bin uniform_film_field_rk45 -- demag=on
```
Example overlays (m_y and m_z):
```bash
python3 scripts/overlay_macrospin.py \
  out/uniform_film_rk45_demag_on/rust_table_uniform_film.csv \
  mumax_outputs/uniform_film_field_demag_on/table.txt \
  --col my --clip_overlap --metrics
```
---
```bash
python3 scripts/overlay_macrospin.py \
  out/uniform_film_rk45_demag_on/rust_table_uniform_film.csv \
  mumax_outputs/uniform_film_field_demag_on/table.txt \
  --col mz --clip_overlap --metrics
```
> ⚠ Current status: m_z(t) matches closely; m_y(t) shows remaining discrepancy under RK45+demag. This is an active debugging item.

### DMI chirality check
Run Rust (two runs with opposite DMI sign):
```bash
cargo run --release --bin bloch_dmi -- dmi=1e-4
cargo run --release --bin bloch_dmi -- dmi=-1e-4
```
---
Analyse:
```bash
python3 scripts/bloch_dmi_analysis.py \
  --dplus  out/bloch_relax_Dplus \
  --dminus out/bloch_relax_Dminus
```
---
### Exploratory CLI (not a benchmark)
Outputs go to `runs/”`.
```bash
cargo run --release -- tilt mumaxlike steps=20000 integrator=rk45 movie
```
---
> Additional reproducible benchmarks are available under `src/bin/`. Each benchmark file contains the canonical run command(s), expected output folder(s), and the recommended post-processing (Python) commands in the header comment.

## Outputs and folder conventions
- This repo uses three output roots (all gitignored):
  - `out/`
Canonical outputs from Rust benchmark binaries (plots, CSVs, configs).
Also contains out/demag_cache/: cached Fourier-space demag kernels keyed by geometry.
  - `runs/`
Exploratory/ad-hoc runs (e.g. command-line driver runs, intermediate experiments).
  - `mumax_outputs/`
MuMax3 outputs copied back from SCARF (or another machine) for direct overlays with Rust results.

## Repository structure

```text
llg-sim/
├── src/                  # solver core + utilities
│   ├── effective_field/  # exchange / anisotropy / zeeman / dmi / demag
│   ├── llg.rs            # LLG RHS + integrators (RK23 + RK45 included)
│   ├── relax.rs          # MuMax-like Relax controller (energy→torque)
│   └── bin/              # reproducible benchmark binaries
├── scripts/              # python overlays / plots (numpy, matplotlib)
├── mumax/                # MuMax3 reference scripts (.mx3)
├── tests/                # cargo test integration-style checks
├── out/                  # Rust benchmark outputs (gitignored)
├── runs/                 # exploratory outputs (gitignored)
├── mumax_outputs/        # MuMax3 outputs copied back for overlays (gitignored)
├── Cargo.toml
└── README.md
```

## License
MIT License (see `License`)
