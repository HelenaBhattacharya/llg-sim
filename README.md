# llg-sim

A two-dimensional micromagnetic solver implementing the Landau–Lifshitz–Gilbert
equation with adaptive mesh refinement (AMR) and composite multigrid
demagnetising-field computation. Written in Rust.

Developed as part of MPhys project CMP11.

---

## Features

- Full LLG dynamics with Dormand–Prince (RK45) adaptive integration and
  Bogacki–Shampine (RK23) damping-only relaxation
- Effective fields: exchange, uniaxial anisotropy, interfacial DMI
  (Néel-type), Zeeman, demagnetising field (FFT convolution)
- Block-structured AMR with temporal subcycling and dynamic regridding
- Composite multigrid demagnetising field with configurable near-field
  correction (MG-only, direct Newell, PPPM)
- Validated against MuMax3 and µMAG Standard Problems 2 and 4

---

## Prerequisites

- Rust toolchain: `rustc` + `cargo`
- Python 3 with `numpy`, `matplotlib` (for plotting)
- MuMax3 (for generating reference data; not required to run the solver)

---

## Build and test
```bash
cargo build --release
cargo test
```

---

## Reproducing report results

### Standard Problem 4 (Section 3.1)
```bash
cargo run --release --bin st_problems -- sp4 a
```

Compare against MuMax3:
```bash
python3 scripts/compare_sp4.py \
  --mumax-root mumax_outputs/st_problems/sp4 \
  --rust-root runs/st_problems/sp4
```

### Standard Problem 2 (Section 3.2)
```bash
cargo run --release --bin st_problems -- sp2
```

Compare against MuMax3:
```bash
python3 scripts/compare_sp2.py
```

### Anti-dot benchmark (Section 4.1)

Single run at L0 = 256 with composite V-cycle:
```bash
LLG_DEMAG_COMPOSITE_VCYCLE=1 \
  cargo run --release --bin bench_composite_vcycle -- --plots
```

Crossover sweep across grid sizes:
```bash
LLG_DEMAG_COMPOSITE_VCYCLE=1 \
  cargo run --release --bin bench_composite_vcycle -- --sweep --plots
```

### Vortex gyration (Section 4.2)

Full three-phase simulation with AMR and direct Newell correction:
```bash
LLG_AMR_MAX_LEVEL=3 LLG_NEWELL_DIRECT=1 \
  cargo run --release --bin bench_vortex_gyration -- --plots
```

### Component-level validation (Appendix A)
```bash
cargo run --release --bin macrospin_fmr
cargo run --release --bin macrospin_anisotropy
cargo run --release --bin bloch_dmi -- dmi=1e-4
cargo run --release --bin bloch_dmi -- dmi=-1e-4
```

---

## Repository structure
```text
llg-sim/
├── src/
│   ├── lib.rs
│   ├── llg.rs, relax.rs, minimize.rs, equilibrate.rs
│   ├── params.rs, grid.rs, vector_field.rs
│   ├── geometry_mask.rs, initial_states.rs, energy.rs, ovf.rs
│   ├── effective_field/
│   │   ├── exchange.rs, anisotropy.rs, dmi.rs, zeeman.rs
│   │   ├── demag_fft_uniform.rs, coarse_fft_demag.rs
│   │   ├── demag_poisson_mg.rs, mg_composite.rs
│   │   ├── mg_kernels.rs, mg_treecode.rs, mg_diagnostics.rs
│   ├── amr/
│   │   ├── hierarchy.rs, patch.rs, indicator.rs
│   │   ├── clustering.rs, interp.rs, regrid.rs, stepper.rs
│   └── bin/
│       ├── st_problems/ (sp4.rs, sp2.rs)
│       ├── macrospin_fmr.rs, macrospin_anisotropy.rs
│       ├── bloch_dmi.rs
│       ├── bench_composite_vcycle.rs
│       └── bench_vortex_gyration.rs
├── tests/
│   ├── validation.rs, test_demag_methods.rs, subcycling_tests.rs
├── scripts/
│   ├── compare_sp4.py, compare_sp2.py
│   ├── plot_antidot_benchmark.py, plot_vortex_gyration.py
│   ├── plot_appendix_figures.py, ovf_utils.py
├── mumax/                # MuMax3 reference scripts
├── Cargo.toml
├── LICENSE
└── README.md
```

---

## License

MIT