<<<<<<< HEAD
# llg-sim

A modular micromagnetic solver for the Landau–Lifshitz–Gilbert (LLG) equation, written in Rust.

This project is developed as part of the CMP11 MPhys project *“Building the Next-Generation Micromagnetic Simulator”*.  
The focus is on clarity, modularity, and extensibility rather than raw GPU performance, with the long-term goal of enabling research into adaptive meshing and advanced micromagnetic models.

The solver currently supports:
- Thin-film (2D) micromagnetics with full 3D magnetisation vectors
- Exchange, uniaxial anisotropy, and Zeeman fields
- Multiple time integrators (Euler, RK4, RK4 with field recomputation)
- Reproducible benchmark workflows with MuMax3 comparison

---

## Repository structure

```text
llg-sim/
├── src/                    # Rust solver source code
│   ├── bin/                # Benchmark binaries
│   │   ├── macrospin_fmr.rs
│   │   ├── macrospin_anisotropy.rs
│   │   ├── uniform_film_field.rs
│   │   └── bloch_relax.rs
│   ├── effective_field/    # Zeeman, exchange, anisotropy fields
│   ├── llg.rs
│   ├── grid.rs
│   ├── energy.rs
│   ├── params.rs
│   └── config.rs           # Per-run metadata (config.json)
├── scripts/                # Python analysis and plotting
│   ├── overlay_macrospin.py
│   └── overlay_bloch_slice.py
├── mumax/                  # MuMax3 reference scripts (.mx3)
├── out/                    # Rust benchmark outputs (gitignored)
├── runs/                   # Exploratory / movie runs (gitignored)
├── mumax_outputs/          # MuMax3 outputs pulled from SCARF (gitignored)
├── tests/                  # Unit tests
├── Cargo.toml
└── README.md
=======
# llg-sim

A modular micromagnetic solver for the Landau–Lifshitz–Gilbert (LLG) equation, written in Rust.

This project is developed as part of the CMP11 MPhys project *“Building the Next-Generation Micromagnetic Simulator”*.  
The focus is on clarity, modularity, and extensibility rather than raw GPU performance, with the long-term goal of enabling research into adaptive meshing and advanced micromagnetic models.

The solver currently supports:
- Thin-film (2D) micromagnetics with full 3D magnetisation vectors
- Exchange, uniaxial anisotropy, and Zeeman fields
- Multiple time integrators (Euler, RK4, RK4 with field recomputation)
- Reproducible benchmark workflows with MuMax3 comparison

---

## Repository structure

```text
llg-sim/
├── src/                    # Rust solver source code
│   ├── bin/                # Benchmark binaries
│   │   ├── macrospin_fmr.rs
│   │   ├── macrospin_anisotropy.rs
│   │   ├── uniform_film_field.rs
│   │   └── bloch_relax.rs
│   ├── effective_field/    # Zeeman, exchange, anisotropy fields
│   ├── llg.rs
│   ├── grid.rs
│   ├── energy.rs
│   ├── params.rs
│   └── config.rs           # Per-run metadata (config.json)
├── scripts/                # Python analysis and plotting
│   ├── overlay_macrospin.py
│   └── overlay_bloch_slice.py
├── mumax/                  # MuMax3 reference scripts (.mx3)
├── out/                    # Rust benchmark outputs (gitignored)
├── runs/                   # Exploratory / movie runs (gitignored)
├── mumax_outputs/          # MuMax3 outputs pulled from SCARF (gitignored)
├── tests/                  # Unit tests
├── Cargo.toml
└── README.md
>>>>>>> ae3cad2 (first attempt at introducing dmi)
