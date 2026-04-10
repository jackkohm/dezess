# CLAUDE.md

## Project: dezess

**DE-MCz Slice Sampler** — a fully GPU-parallel ensemble slice sampling library built on JAX.

### What this project does

dezess is a Bayesian posterior sampler combining:
- **DE-MCz Z-matrix proposals** — all walkers update simultaneously
- **Fixed-iteration slice sampling** — guaranteed acceptance, no while-loops, fast JIT
- **Pluggable strategy modules** — 13 directions × 5 widths × 6 slices × 3 Z-matrices × 3 ensembles

### Architecture

```
dezess/
├── api.py              # High-level dezess.sample() interface
├── core/
│   ├── loop.py         # Central orchestrator — resolves strategies, runs warmup/production
│   ├── slice_sample.py # Core slice sampling primitives
│   └── types.py        # VariantConfig, SliceState, WalkerAux, SamplerState
├── directions/         # 13 direction strategies (de_mcz, snooker, momentum, pca, ...)
├── width/              # 5 width strategies (scalar, scale_aware, stochastic, ...)
├── slice/              # 6 slice execution strategies (fixed, nurs, overrelaxed, ...)
├── zmatrix/            # 3 Z-matrix strategies (circular, hierarchical, live)
├── ensemble/           # 3 ensemble strategies (standard, parallel_tempering, block_gibbs)
├── benchmark/          # Benchmarking infrastructure + named variant registry
└── tests/              # Gaussian moments, convergence, Geweke, SBC, performance
```

### Current branch: `claude/implement-nurs-sampler-P6ZL5`

**PR #7** — Implements the **NURS (No-Underrun Sampler)** from [arXiv:2501.18548](https://arxiv.org/abs/2501.18548) as a new slice execution strategy.

NURS replaces stepping-out + shrinking with orbit-based categorical sampling:
1. **Metropolis-adjusted shift** — randomises lattice alignment for reversibility
2. **Orbit doubling** — extends forward/backward with streaming categorical selection
3. **No-underrun stopping** — stops when endpoint density is negligible vs orbit total
4. **Categorical selection** — samples proportional to density (not uniform on slice)

Key files added/changed:
- `dezess/slice/nurs.py` — NURS implementation (190 lines)
- `dezess/core/loop.py` — Registered `nurs` in SLICE_STRATEGIES
- `dezess/benchmark/registry.py` — Added `nurs`, `nurs_deep`, `nurs_scalar` variants
- `dezess/tests/test_gaussian_moments.py` — 3 NURS tests (all passing)
- `bench_nurs.py` — Head-to-head benchmark script

### Running tests

```bash
pip install -e ".[test]"
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/ -v -s
```

### Running the NURS benchmark

```bash
python bench_nurs.py
```

### Default sampler

`scale_aware` — DE-MCz directions + scale-aware width + fixed slice. Set in `dezess/core/loop.py:DEFAULT_CONFIG`.

### Key design constraints

- **No while-loops** — use `lax.fori_loop` for JIT compatibility
- **All arrays fixed-shape** — NamedTuples with pre-allocated JAX arrays
- **vmap across walkers** — each walker updated in parallel
- **float64 throughout** — `jax.config.update("jax_enable_x64", True)`
