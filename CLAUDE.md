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

**`ensemble_kwargs["complementary_prob"]`** (0.0–1.0) enables the zeus-style
complementary-pair fallback in **both** standard and block-Gibbs ensembles.
With probability `complementary_prob` per step, the configured direction is
replaced by a pair drawn from the OTHER half of the current walker positions
(bypassing the Z-matrix). Use cp≈0.5 if you suspect biased Z-matrix from
short warmup + biased init. cp=0.0 (default) is byte-identical to the legacy
behavior.

### Streaming-write + resume

Long runs can stream samples + sampler state to disk during the run
(one JIT compile, one execution) instead of saving at the end. Pass
`stream_path="path/to/dir"` to `run_variant` (or `dezess.sample`).
Samples accumulate in `path/to/dir/chunk_NNN/{samples,log_probs}.npy`
as numpy memmaps; sampler state is overwritten atomically in
`path/to/dir/state/`.

Read while the run is in progress (or post-hoc) with
`dezess.read_streaming(path)` — concatenates all chunks and returns
`{samples, log_probs, n_steps_total, n_walkers, n_dim, config_name}`.

Resume after a kill (or for more samples) with
`dezess.resume_streaming(path, log_prob_fn, n_more_steps=N)` —
reads the state, calls `run_variant` with `n_warmup=0` + the saved
mu / z_matrix / positions, writes to a new chunk. The JIT compile
hits the persistent cache (`JAX_COMPILATION_CACHE_DIR`) so resume
is cheap.

### γ-jump for block-Gibbs MH+DR mode-jumping

`ensemble_kwargs["gamma_jump_prob"]` (0.0–1.0) — Braak (2006) γ=1
mode-jumping for the block-Gibbs MH+DR path. With probability
`gamma_jump_prob` per step, the proposal magnitude is overridden to
land directly on another walker (`x_prop = x + (z_j - z_k)` instead of
the diffusion-optimal `x + (mu_b/sqrt(bsize/2)) * (z_j - z_k)`). MH
accept/reject decides if it lands in another mode. Composes with
`complementary_prob` (gamma jump applies to whichever direction was
selected). Stage-1 only — DR's stage-2 fallback is unaffected. Typical
production values 0.05–0.10. gamma_jump_prob=0.0 default is byte-identical
to legacy behavior.

Slice-based ensembles (standard, parallel_tempering) ignore this knob —
they shrink the slice to find a valid point, which can't cross
low-density barriers regardless of bracket width.

### Snooker (Braak 2008) for block-Gibbs MH+DR

`ensemble_kwargs["snooker_prob"]` (0.0–1.0) — Braak (2008) snooker
proposal as a mixture-of-moves option for the block-Gibbs MH+DR path
(`block_mh_step` and `block_conditional_step`). With probability
`snooker_prob` per step, the proposal is constructed in the block
subspace as `x + γ_snooker * (proj_p(z_p) - proj_p(z_q)) * p` where
`p = (x - z_a) / ‖.‖` is the projection direction through anchor
`z_a` and `γ_snooker = 1.7` (Braak optimum, 1D radial line).

The MH acceptance includes the snooker Jacobian
`(bsize - 1) * log(‖x_new - z_a‖ / ‖x - z_a‖)` using the BLOCK
dimension — critical for correctness in block-Gibbs.

Mixture semantics (single dice roll, partitioned):
- `u < complementary_prob` → complementary direction (DE-MCz proposal)
- `complementary_prob ≤ u < complementary_prob + snooker_prob` → snooker
- `u ≥ complementary_prob + snooker_prob` → plain DE-MCz

`complementary_prob + snooker_prob ≤ 1.0` is enforced; otherwise
ValueError. Stage-1 only — DR's stage-2 stays as DE-MCz fallback.
Snooker is the right tool when block-Gibbs mixing is slow due to
**cross-block correlations** (a cluster of correlated dims spread
across blocks). For your bg_MH+DR Sanders setup, try `snooker_prob=0.1`
when `log_10 M ↔ stream-nuisance` correlations exceed ~0.5.

`snooker_prob=0.0` (default) is byte-identical to legacy behavior.

### Key design constraints

- **No while-loops** — use `lax.fori_loop` for JIT compatibility
- **All arrays fixed-shape** — NamedTuples with pre-allocated JAX arrays
- **vmap across walkers** — each walker updated in parallel
- **float64 throughout** — `jax.config.update("jax_enable_x64", True)`
