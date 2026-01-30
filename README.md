# llg-sim - DRAFT

A modular micromagnetic solver for the Landau–Lifshitz–Gilbert (LLG) equation, written in Rust.

This repository is developed as part of the CMP11 MPhys project **“Building the Next-Generation Micromagnetic Simulator”**.  
The focus is on clarity, modularity, and extensibility (e.g. adaptive meshing and additional energy terms), while benchmarking against **MuMax3**.

---

## Features

### Physics / effective fields
- Thin-film (2D) micromagnetics (Nx×Ny×1) with full 3D magnetisation vectors
- Exchange, uniaxial anisotropy, Zeeman field
- Demagnetising field via FFT-accelerated convolution (cached kernel)
- Interfacial DMI (Néel-type) for thin films (MuMax-style)

### Time integration and relaxation
- Explicit Euler / RK4 / RK4 (recompute-field)
- Adaptive RK45 (Dormand–Prince) with recompute-field
- **MuMax-like relaxation controller** (`src/relax.rs`):
  - precession suppressed (damping-only RHS)
  - adaptive **RK23** (Bogacki–Shampine 3(2)) relax stepper
  - energy-descent phase → torque-descent phase
  - tolerance tightening (MuMax-style)

### Validation and comparison workflow
- Reproducible benchmark binaries under `src/bin/*`
- Python plotting scripts under `scripts/`
- MuMax3 reference scripts under `mumax/`
- Outputs are written to `out/` and `runs/` and `mumax_outputs/` (ignored by git)

---

## Repository structure

```text
llg-sim/
├── src/
│   ├── bin/
│   │   ├── st_problems/
│   │   │   ├── main.rs                 # entrypoint for μMAG standard problems
│   │   │   └── sp4.rs                  # Standard Problem 4 (SP4a/SP4b)
│   │   ├── bloch_dmi.rs                # DMI chirality validation (+D vs −D)
│   │   ├── demag_cube.rs               # quick demag factor sanity check
│   │   ├── macrospin_anisotropy.rs     # macrospin anisotropy benchmark
│   │   ├── macrospin_fmr.rs            # macrospin FMR ringdown benchmark
│   │   ├── relax_uniform_noisy.rs      # relaxation validation vs MuMax Relax()
│   │   ├── uniform_film_field.rs       # uniform film benchmark (RK4 recompute)
│   │   └── uniform_film_field_rk45.rs  # uniform film benchmark (adaptive RK45)
│   ├── effective_field/                # exchange/anisotropy/zeeman/dmi/demag field terms
│   ├── energy.rs  
│   ├── main.rs                       
│   ├── grid.rs                         
│   ├── llg.rs                          # LLG RHS + integrators (incl RK23 + RK45)
│   ├── relax.rs                        # relaxation regime controller
│   ├── params.rs                       # material + solver parameters
│   ├── vector_field.rs
│   ├── vec3.rs 
│   ├── visualisation.rs                # plotting helpers for main.rs runs
│   └── config.rs                       # config.json writer for outputs
├── scripts/
│   ├── overlay_macrospin.py            # MuMax vs Rust overlays for macrospin/uniform film
│   ├── compare_sp4.py                  # SP4 MuMax vs Rust comparison (two-panel)
│   ├── overlay_relax.py                # relaxation validation (MuMax vs Rust final state)
│   ├── bloch_dmi_analysis.py           # DMI chirality printout + plots (+D vs −D)
├── mumax/
│   ├── st_problems/                    # μMAG standard problem mx3 scripts (SCARF)
│   ├── macrospin_fmr.mx3
│   ├── macrospin_anisotropy.mx3
│   ├── relax_uniform_noisy.mx3        
│   ├── uniform_film_field_demag_on.mx3
│   └── uniform_film_field_demag_off.mx3
├── tests/
│   └── validation.rs                   # integration-style physics sanity checks
├── out/                                # benchmark outputs (gitignored)
├── runs/                               # exploratory outputs (gitignored)
├── mumax_outputs/                      # MuMax outputs pulled from SCARF (gitignored)
├── Cargo.toml
└── README.md