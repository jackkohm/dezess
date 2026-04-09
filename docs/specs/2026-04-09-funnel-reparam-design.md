# Funnel Reparameterization & Block-Gibbs Design

**Date:** 2026-04-09
**Status:** Approved
**Goal:** Make dezess handle 63D funnel geometry (R-hat < 1.2, log-width variance within 50% of true)

## Problem

The 63D stream-fitting posterior has four pathologies:

1. **Funnel geometry** — width params control scale of nuisance by orders of magnitude
2. **Block structure** — 7 potential + 4x14 nuisance, blocks conditionally independent given potential
3. **Banana ridges** — curved degeneracies in potential params
4. **Possible multimodality** — discrete modes in orientation angles

Current baseline: ALL variants fail on funnel_63d (R-hat 3.2-127, log-width variance 3-12 vs expected 9).

## GPU Baseline (RTX 4070 Super, 128 walkers, 8k production)

| Variant | funnel_63d R-hat max | lw_var (expect 9) |
|---|---|---|
| overrelaxed | 3.18 | 3.27 |
| scale_aware | 3.73 | 6.01 |
| baseline | 126.7 | 7.02 |
| snooker | 47.4 | 12.4 |

## Approach

Three independent strategies, tested separately then combined:

1. **Non-centered reparameterization** via internal bijector transform
2. **Block-Gibbs updates** with per-block mu tuning
3. **Combined** — transform + block-Gibbs composed

## Design

### 1. Transform / Bijector System

**New module: `dezess/transforms.py`**

Core type:

```python
class Transform(NamedTuple):
    forward: Callable    # z (unconstrained) -> x (original)
    inverse: Callable    # x (original) -> z (unconstrained)
    log_det_jac: Callable  # z -> scalar log|det(df/dz)|
```

Integration into `run_variant` and `sample()`:

- Accept optional `transform` argument
- `inverse(init_positions)` maps to unconstrained space at start
- Sampler works in z-space with wrapped log-prob: `lp(z) = log_prob(forward(z)) + log_det_jac(z)`
- Returned samples mapped back via `forward(z_samples)`

Built-in transforms:

- `Identity()` — no-op default
- `NonCenteredFunnel(width_idx, offset_indices)` — single funnel block
  - Forward: `x[offset_i] = z[offset_i] * exp(z[width_idx] / 2)`, identity elsewhere
  - Inverse: `z[offset_i] = x[offset_i] / exp(x[width_idx] / 2)`, identity elsewhere
  - Log-det-Jacobian: `len(offset_indices) * z[width_idx] / 2`
- `compose(*transforms)` — chain transforms sequentially
- `block_transform(transforms, index_lists)` — apply different transforms to different param blocks

Convenience helper for the stream posterior:

```python
transform = dezess.transforms.multi_funnel(
    n_potential=7,
    funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)],
    # (start_idx, block_size) — first dim is log-width, rest are offsets
)
```

Builds a `block_transform` with `NonCenteredFunnel` per stream block and `Identity` for potential.

### 2. Block-Gibbs Updates

**New module: `dezess/ensemble/block_gibbs.py`**

Block specification via `ensemble_kwargs`:

```python
ensemble_kwargs={
    "block_sizes": [7, 14, 14, 14, 14],  # contiguous shorthand
    # OR
    "blocks": [[0,...,6], [7,...,20], ...],  # explicit indices
}
```

Per-step algorithm:

1. Randomly permute block order (avoids systematic bias)
2. For each block `b` with indices `idx_b`:
   - Extract sub-Z-matrix: `z_b = z[:, idx_b]`
   - Draw direction `d_b` from sub-Z-matrix (DE-MCz pair difference in block subspace)
   - Use block-specific `mu_b` (tuned independently during warmup)
   - Slice-sample along `d_b`, evaluating **full** `log_prob(x)` but only moving block coords
   - Update `x[idx_b]` with result
3. All walkers updated in parallel within each block (vmap over walkers)

Key properties:

- Full log-prob eval ensures correct Gibbs conditional
- Per-block mu adapts to each block's scale
- Random sweep order for ergodicity
- `n_blocks` log-prob evals per walker per step (vs 1 for standard)

Implementation: new `block_sweep_step` function in `loop.py` replaces `parallel_step` when `ensemble="block_gibbs"`. Uses `lax.fori_loop` over blocks with `vmap` over walkers within each block.

Per-block mu tuning during warmup: each block tracks its own bracket ratio and tunes independently. Stored as `mu_blocks` array of shape `(n_blocks,)` inside `block_gibbs.py`'s warmup state, passed to `block_sweep_step` alongside the global `mu` (which is unused when block-Gibbs is active). The `run_variant` return dict includes `"mu_blocks"` for inspection.

### 3. Combined Mode

Transform maps to unconstrained space, block-Gibbs sweeps over blocks in that space. Block boundaries match transform boundaries.

```python
result = dezess.sample(
    log_prob_fn, init,
    transform=dezess.transforms.multi_funnel(
        n_potential=7,
        funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)],
    ),
    variant="block_gibbs_scale_aware",
    ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14]},
)
```

### 4. New Registered Variants

| Name | Ensemble | Width | Purpose |
|---|---|---|---|
| `block_gibbs` | block_gibbs | scalar | Test block structure alone |
| `block_gibbs_scale_aware` | block_gibbs | scale_aware | Block + scale-aware |
| `transform_scale_aware` | standard | scale_aware | Test transform alone (user provides transform) |
| `block_gibbs_transform` | block_gibbs | scale_aware | Combined (user provides transform) |

## Testing

**New file: `dezess/tests/test_funnel_reparam.py`**

| Test | Target | What it validates |
|---|---|---|
| `test_transform_roundtrip` | N/A | `forward(inverse(x)) == x` and `inverse(forward(z)) == z` |
| `test_geweke_with_transform` | neals_funnel(10) | One-step invariance with NonCenteredFunnel — validates Jacobian |
| `test_ncp_funnel_10d` | neals_funnel(10) | Transform alone converges: R-hat < 1.05, lw_var in [7.2, 10.8] |
| `test_block_gibbs_block_coupled` | block_coupled_gaussian | Block-Gibbs matches/beats standard: R-hat < 1.1 |
| `test_block_gibbs_funnel_63d` | funnel_63d | Block-Gibbs alone on funnel: expect improvement over baseline |
| `test_combined_funnel_63d` | funnel_63d | Transform + block-Gibbs: R-hat < 1.2, lw_var in [4.5, 13.5] |

**New file: `bench_funnel_reparam.py`**

Runs all combinations on funnel_63d, prints comparison table vs baseline.

## Success Criteria

On funnel_63d with 128 walkers and 8000 production steps:

- R-hat max < 1.2
- Log-width variance within 50% of true (4.5 - 13.5) for all 4 funnel blocks
- No regression on block_coupled_gaussian or banana_63d

## Files

| File | Action | ~Lines |
|---|---|---|
| `dezess/transforms.py` | New | 150 |
| `dezess/ensemble/block_gibbs.py` | New | 200 |
| `dezess/core/loop.py` | Modify | +30 |
| `dezess/api.py` | Modify | +10 |
| `dezess/benchmark/registry.py` | Modify | +20 |
| `dezess/tests/test_funnel_reparam.py` | New | 200 |
| `bench_funnel_reparam.py` | New | 100 |
