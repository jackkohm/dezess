# Hybrid Complementary + Z-Matrix Direction Source Design

## Goal

Add a hybrid direction source for block-Gibbs MH (`bg_MH+DR`) that mixes Z-matrix-based DE directions with snapshot-based complementary-half DE directions. Default behavior (when the new flag is unset) is byte-identical to current bg_MH+DR.

## Motivation

The Z-matrix is frozen at end of warmup. We documented that it can be biased toward warmup geometry (mu collapsed to 0.144 instead of staying near 1 in our previous Sanders run). Complementary-ensemble directions self-correct to current walker geometry but lose the long-range diversity from accumulated warmup history. A hybrid combines both: archive diversity AND current local geometry.

This is purely additive: existing bg_MH+DR users see no change unless they opt in via the new `complementary_prob` flag.

## API

One new `ensemble_kwargs` flag:

```python
ensemble_kwargs={
    "block_sizes": [7, 14],
    "use_mh": True,
    "delayed_rejection": True,
    "complementary_prob": 0.5,    # NEW: probability per walker per block update
                                  # of using complementary direction instead of Z-matrix.
                                  # Default 0.0 = current bg_MH+DR (no change).
}
```

Valid range: `0.0 <= complementary_prob <= 1.0`.
- `0.0`: pure Z-matrix (current behavior)
- `0.5`: balanced hybrid (recommended starting point)
- `1.0`: pure complementary (zeus-like)

## Architecture

The change is local to the four block-Gibbs JIT functions in `loop.py`: `block_sweep_step`, `block_mh_step`, `block_mh_cov_step`, `block_conditional_step`. Each of these calls `_mh_walker` (or equivalent) per walker via vmap.

For each block update:

1. **Snapshot positions at block start** (free — JAX arrays are immutable references)

2. **Multi-GPU only**: replicate the snapshot via `with_sharding_constraint(positions, replicated)` so every walker can read from the complementary half across GPUs. One all-gather per block.

3. **Per-walker random split**: assign each walker to half A or half B. Done implicitly via walker index modulo 2 OR via a per-block random permutation (random per-block split is preferred for fresh diversity).

4. **Per-walker direction selection** inside `_mh_walker`:
   - Compute Z-matrix direction `d_zmat` (existing logic — pick pair from Z-matrix).
   - Compute complementary direction `d_comp` (new logic — pick pair from snapshot's complementary half).
   - Stochastic selection: `use_comp = uniform(key) < complementary_prob`; `d_block = where(use_comp, d_comp, d_zmat)`.
   - Both directions are always computed (cheap), then `jnp.where` selects.

5. **Standard MH+DR accept/reject** with the chosen direction. Unchanged.

When `complementary_prob == 0.0` (default), the entire complementary-direction code path is bypassed at JIT trace time via a Python-level `if`, so byte-identical behavior to current bg_MH+DR is preserved.

## Complementary direction generation

For walker `i` in half H_i (A or B), with snapshot `pos_snapshot`:

```python
# Sample j, k from the COMPLEMENTARY half (the half walker i is NOT in)
key, k1, k2 = jax.random.split(key, 3)
half_size = n_walkers // 2

# Determine my half (deterministic split: i < half_size = A, else B)
my_half = walker_idx >= half_size

# Complementary half index range
comp_offset = jnp.where(my_half, 0, half_size)

# Sample j from complementary half
j_off = jax.random.randint(k1, (), 0, half_size)
# Sample k from complementary half, ensuring k != j
k_off = jax.random.randint(k2, (), 0, half_size)
k_off = jnp.where(k_off == j_off, (k_off + 1) % half_size, k_off)

j_idx = comp_offset + j_off
k_idx = comp_offset + k_off

# DE direction in block subspace, scale-aware width applied normally
diff_comp = pos_snapshot[j_idx, b_idx] - pos_snapshot[k_idx, b_idx]
norm_comp = jnp.linalg.norm(diff_comp)
d_block_comp = diff_comp / jnp.maximum(norm_comp, 1e-30)
mu_eff_comp = mu_b * norm_comp / jnp.maximum(dim_corr, 1e-30)
```

Note: `walker_idx` (0 to n_walkers-1) is passed in to `_mh_walker` as an additional vmap axis (`in_axes=0` over `jnp.arange(n_walkers)`). This gives each vmapped walker access to its own index for the half-assignment. Half-assignment is deterministic: walkers `[0, n_walkers/2)` are half A, walkers `[n_walkers/2, n_walkers)` are half B. The split is static per JIT-compile (no random permutation needed — diversity comes from the random `j, k` pair selection within the complementary half).

## Detailed balance argument

Per Section 1 of our brainstorming discussion: the joint target factorizes as `π(X) = ∏ π(x_i)` (each walker independently samples the same posterior). Each walker's MH update uses ONLY the snapshot positions (not other walkers' new positions in the same block call). Each walker's MH preserves its marginal `π(x_i)`. Independent valid updates on each marginal preserve the joint. Therefore the snapshot-based parallel update is a valid kernel.

The hybrid choice (Z-matrix vs complementary, per walker, stochastic) is also valid because it's a mixture of two valid kernels.

When `complementary_prob > 0`, this is technically adaptive MCMC (the proposal distribution depends on current chain state via the snapshot). Falls under standard adaptive MCMC theory (Roberts & Rosenthal 2007) which is asymptotically valid under containment + diminishing adaptation. The complementary contribution doesn't formally satisfy diminishing adaptation, but the same is true for emcee/zeus and is universally accepted in practice.

## Multi-GPU implications

The snapshot replication adds one all-gather per block:
- 64 walkers × 21 dims × 8 bytes = 11 KB per gather
- 5 blocks per sweep × 1 gather/block = 5 gathers per sweep
- All-gather latency: microseconds at this size
- Negligible vs the per-block log_prob cost

When `complementary_prob == 0`, no snapshot, no all-gather, no overhead.

## Tests

1. **Backwards-compat**: `complementary_prob=0.0` produces identical samples to current bg_MH+DR (same seed, same key, same target). Bit-exact reproducibility check.

2. **Standard targets**: `complementary_prob=0.5` recovers correct variance on:
   - 21D Gaussian (mean_var ≈ 1.0)
   - Block-coupled 63D (var_ratio ≈ 1.0)

3. **Stale-Z stress test** (the motivating use case): construct a biased Z-matrix (e.g., samples from a single mode of a bimodal target). Run two configs:
   - `complementary_prob=0.0`: should converge slowly or fail
   - `complementary_prob=0.5`: should converge faster (validates the hybrid actually mitigates stale-Z bias)

4. **Range validation**: `complementary_prob` outside `[0.0, 1.0]` raises ValueError.

## Files Changed

- Modify: `dezess/core/loop.py` — add complementary direction logic to the 4 block-Gibbs step functions, parse `complementary_prob` from `ensemble_kwargs`
- Create: `dezess/tests/test_complementary_hybrid.py` — backwards-compat, convergence, stale-Z tests
- Modify: `dezess/benchmark/registry.py` — add `bg_mh_dr_hybrid` variant config

## Out of scope (deliberately deferred)

- Two-half pass approach (Option A from brainstorming) — heavier, only marginally better mixing per step
- Per-block tuning of `complementary_prob` (single global value sufficient)
- Tracking which directions came from where for diagnostics (could add later if useful)
- Hybrid for non-block-Gibbs samplers (default ensemble already gets benefit from Z-matrix)
