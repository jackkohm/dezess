# Complementary direction for standard ensemble — design spec

**Date:** 2026-04-17
**Status:** Approved (awaiting plan)

## Goal

Make `complementary_prob` a first-class option for the **standard** ensemble (currently it only works for `block_gibbs`). When set, it works orthogonally on top of *any* of the 13 direction strategies, by replacing the configured direction with a complementary-pair direction at probability `cp`.

This eliminates the gap exposed in our 2026-04-16 stale-Z stress test: bg_MH+DR can use the hybrid, but the simpler `scale_aware` default cannot.

## Motivation

The 2026-04-16 stale-Z benchmark showed that with short warmup + biased initialization, the Z-matrix gets frozen in an unrepresentative state and DE-MCz pairs draw from a poor empirical distribution. The complementary-pair pattern (zeus/emcee-style two-half snapshot) bypasses the Z-matrix when triggered, sampling directly from the *current* walker positions instead. We made it work for `block_gibbs` already; this spec generalizes.

## Non-goals

- **Parallel-tempering ensemble.** Walkers are temperature-stratified, so the "halves" partition doesn't apply. Document and skip.
- **Refactoring `block_gibbs` to share the helper.** Block-Gibbs uses block-aware indices. Could share later, out of scope here.
- **New direction strategies.** No new directions added; we wrap existing ones.

## Architecture

### New module: `dezess/directions/complementary.py`

Single helper:

```python
def sample_complementary_pair(
    x_i: Array,
    positions_snapshot: Array,   # (n_walkers, n_dim), replicated across GPUs
    walker_idx: int,             # this walker's index in the ensemble
    key: Array,
    aux: WalkerAux,
) -> tuple[Array, Array, WalkerAux]:
    """Pick j, k from the OTHER half of positions_snapshot.

    Half determined by walker_idx >= n_walkers // 2.
    Returns (direction, key, aux) — same signature as direction strategies.
    aux.direction_scale set to ‖positions[j] - positions[k]‖ for scale_aware width.
    """
```

This is the same pattern used inside `block_mh_step` etc., extracted into a reusable helper.

### Modified: `dezess/core/loop.py`

Inside `_make_step_fn(parallel_step)`:

1. **Read config** (after existing `dir_kwargs`/`width_kwargs` resolution):
   ```python
   complementary_prob = float(ens_kwargs.get("complementary_prob", 0.0))
   if not (0.0 <= complementary_prob <= 1.0):
       raise ValueError(...)
   use_complementary = complementary_prob > 0.0
   ```
2. **Update sharding annotations** (when `sharding_info is not None and use_complementary`):
   - Add a replicated `pos_snapshot` arg to `parallel_step` signature.
   - Add `walker_indices` (walker-sharded) arg.
3. **Snapshot replication** at step entry:
   ```python
   if use_complementary and sharding_info is not None:
       pos_snapshot = jax.lax.with_sharding_constraint(positions, repl_sh)
   else:
       pos_snapshot = positions
   ```
4. **Branch in `update_one`**:
   ```python
   def update_one(x_i, lp_i, w_key, prev_d, bw, d_anchor, d_scale, temp, walker_idx):
       w_aux = WalkerAux(...)
       if use_complementary:
           w_key, kc = jax.random.split(w_key)
           use_comp = jax.random.uniform(kc, dtype=jnp.float64) < complementary_prob
           # both branches must run under vmap (lax.cond becomes lax.select)
           d_z, w_key, aux_z = dir_mod.sample_direction(...)   # unchanged path
           d_c, w_key, aux_c = sample_complementary_pair(
               x_i, pos_snapshot, walker_idx, w_key, w_aux,
           )
           d = jnp.where(use_comp, d_c, d_z)
           # Pick the matching direction_scale; other aux fields are direction-strategy-specific
           # so we honor the configured strategy's anchor/etc. and only swap the scale.
           direction_scale = jnp.where(use_comp, aux_c.direction_scale, aux_z.direction_scale)
           w_aux = aux_z._replace(direction_scale=direction_scale)
       else:
           d, w_key, w_aux = dir_mod.sample_direction(...)   # current behavior, byte-identical
       # ... width / slice / rest unchanged ...
   ```
5. **vmap signature** updated to thread `walker_idx`:
   ```python
   results = jax.vmap(update_one)(
       positions, log_probs, walker_keys,
       walker_aux_prev_dir, walker_aux_bw, walker_aux_d_anchor,
       walker_aux_d_scale, temperatures,
       jnp.arange(n_walkers),
   )
   ```

### Wrap-up

`anchor`, `prev_direction`, and other direction-strategy-specific aux fields are kept from the configured strategy's output (`aux_z`). Only `direction_scale` is swapped, because that's what `width=scale_aware` reads.

For `direction="snooker"`, the snooker Jacobian `(d-1) log ‖x - z_anchor‖` is part of the slice log-density. When the complementary branch fires the proposal direction has no anchor, so the Jacobian must be suppressed. Implementation: gate the Jacobian term with a scalar multiplier:

```python
# Right after the use_comp roll (or default to 1.0 when use_complementary is False):
snooker_active = jnp.where(use_comp, 0.0, 1.0) if use_complementary else 1.0

if config.direction == "snooker":
    z_anchor = w_aux.direction_anchor      # only set when zmat branch fires
    def lp_fn(x):
        base = lp_eval(log_prob_fn, x) / temp
        dist = jnp.linalg.norm(x - z_anchor)
        return base + snooker_active * (ndim_j - 1) * jnp.log(jnp.maximum(dist, 1e-30))
    dist_0 = jnp.linalg.norm(x_i - z_anchor)
    lp_x_slice = lp_i / temp + snooker_active * (ndim_j - 1) * jnp.log(jnp.maximum(dist_0, 1e-30))
```

The existing post-slice re-eval `lp_new_untempered = lp_eval(log_prob_fn, x_new)` for snooker stays — it strips whichever Jacobian was (or wasn't) baked in, giving us the raw log-prob for storage.

This snooker special-case is the only direction-specific complication.

## Sharding

When `n_gpus > 1` and `use_complementary`:
- `pos_snapshot` replicated via `with_sharding_constraint(positions, repl_sh)` — one all-gather per step.
- `walker_indices` (`jnp.arange(n_walkers)`) is walker-sharded, automatic from concat.

Cost: one all-gather of `(n_walkers, n_dim)` doubles per step. For 64 walkers × 21 dims = 10 KB. Negligible vs. log-prob eval cost.

When `n_gpus == 1`: `pos_snapshot = positions` directly. No copy, no overhead.

When `cp == 0`: Python-level `if use_complementary` skips snapshot creation entirely. JIT graph is byte-identical to current.

## Public-facing config

```python
config = VariantConfig(
    name="scale_aware_hybrid",
    direction="de_mcz",
    width="scale_aware",
    slice_fn="fixed",
    zmatrix="circular",
    ensemble="standard",
    width_kwargs={"scale_factor": 1.0},
    ensemble_kwargs={"complementary_prob": 0.5},
)
```

`ensemble_kwargs["complementary_prob"]` is the single knob, identical key/semantics to the existing block-Gibbs option.

## Tests

Add to `dezess/tests/test_complementary_standard.py` (new file, mirrors `test_complementary_hybrid.py`):

1. **`test_cp_zero_matches_baseline`** — cp=0 produces samples byte-identical to plain `scale_aware`. Regression guard against accidental graph divergence.
2. **`test_cp_half_recovers_gaussian_variance`** — cp=0.5 on 21D anisotropic Gaussian: per-dim variance within 5% of true diagonal.
3. **`test_cp_one_recovers_gaussian_variance`** — cp=1.0 same check.
4. **`test_cp_works_with_snooker`** — cp=0.5 + `direction="snooker"`: posterior mean within 0.1 of true zero, no NaNs.
5. **`test_cp_works_with_weighted_pair`** — cp=0.5 + `direction="weighted_pair"`: same check.
6. **`test_cp_multi_gpu_matches_single_gpu`** — cp=0.5, 2 GPUs vs 1 GPU: per-dim variance within 10% (Monte-Carlo noise threshold).

## Documentation

Add a one-line note to `dezess/CLAUDE.md` under "Default sampler":

> `ensemble_kwargs["complementary_prob"]` (0.0–1.0) enables the zeus-style complementary-pair fallback in **both** standard and block-Gibbs ensembles. Use cp≈0.5 if you suspect biased Z-matrix from short warmup + biased init.

## Risks & mitigations

- **Snooker Jacobian when complementary fires** — addressed in design above with `lax.cond` on the Jacobian term.
- **Width-strategy assumptions** — only `scale_aware` and `per_direction` read `direction_scale`. Both are unaffected by which branch produced the scale (it's a pair-norm in both cases).
- **vmap branch overhead** — both `dir_mod.sample_direction` and `sample_complementary_pair` evaluate every step under vmap. The configured direction is already cheap; the complementary helper is cheaper. Combined overhead ~2 extra `gather + sub + norm` per walker per step. Negligible.
- **Reproducibility** — adding `walker_idx` arg and an extra `jax.random.split` shifts the PRNG stream when `cp > 0`. cp=0 path takes the `else` branch with no extra splits, preserving byte-identical reproducibility.

## Acceptance criteria

- All 6 tests pass on H200.
- New script `bench_complementary_standard_stale.py` (mirrors `bench_complementary_stale.py` but configs are `scale_aware` with cp ∈ {0, 0.25, 0.5, 1.0}) shows cp=0.5 has better worst-dim ESS than cp=0 on the stale-Z target.
- No regression on existing `test_complementary_hybrid.py` tests (block_gibbs path unchanged).
- No regression on `test_gaussian_moments.py` for plain `scale_aware` (cp=0 path).
