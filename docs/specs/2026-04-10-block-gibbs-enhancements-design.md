# Block-Gibbs MH Enhancements Design

## Goal

Three composable enhancements to block-Gibbs MH that maximize ESS per sequential log_prob eval under GPU saturation (64 walkers fill the H200). Target: the 63D Sanders stream-fitting posterior with funnel geometry, bounded priors, no gradients.

## Architecture

All three enhancements modify the block-Gibbs MH sweep in `loop.py`. They are independently toggleable via `ensemble_kwargs` flags and compose naturally:

```
ensemble_kwargs={
    "block_sizes": [7, 14, 14, 14, 14],
    "use_mh": True,
    "conditional_log_prob": my_cond_fn,   # Enhancement 1
    "delayed_rejection": True,             # Enhancement 2
    "use_block_cov": True,                 # Enhancement 3
}
```

## Enhancement 1: Conditional Independence Parallelism

### Problem

Block-Gibbs MH does 5 sequential full log_prob evals per sweep (1 per block). But the 4 stream blocks are conditionally independent given potential — updating stream 1 doesn't affect the MH ratio for stream 2.

### Design

Replace the sequential scan over all 5 blocks with a 2-round structure:

**Round 1** — Potential block (block 0): Standard MH using `full_log_prob`. 1 eval batch of 64 walkers.

**Round 2** — All stream blocks in parallel: For each walker, propose new params for all 4 streams independently. Evaluate using `conditional_log_prob(params, stream_idx)` via nested vmap (walkers x streams). Accept/reject each stream independently per walker.

The `conditional_log_prob` function is provided by the user. It evaluates a single stream's contribution:

```python
def conditional_log_prob(params, block_idx):
    """block_idx is 0-indexed among the conditional blocks (streams 0-3)."""
    pot = params[:7]
    nuis = params[7 + block_idx*14 : 7 + (block_idx+1)*14]
    return stream_log_likelihood(data[block_idx], pot, nuis) + nuisance_prior(nuis)
```

### MH Acceptance

For stream block s, the MH ratio uses conditional log_probs:

```
log alpha = cond_lp(params_new, s) - cond_lp(params_old, s)
```

This is correct because the other streams' contributions cancel in the ratio (conditional independence given potential).

### Stored Log Probs

After a conditional sweep, the stored `log_probs` per walker must be reconstructed. Two options:

**A) Reconstruct from conditionals**: `full_lp = pot_lp + sum(cond_lps)`. Requires the conditional to return the same terms that sum to the full log_prob. Fragile if there are shared terms (e.g., potential prior counted in both).

**B) Re-evaluate full log_prob after conditional sweep**: 1 extra full eval per sweep. Guarantees consistency. Cost: 3 rounds instead of 2 (pot update, parallel stream updates, full re-eval). Still 3x better than 5 sequential.

Recommend **B** for correctness. The full re-eval also updates the Z-matrix with correct full log_probs.

### Cost

| Approach | Sequential eval batches | Notes |
|---|---|---|
| Current bg_MH | 5 | 1 per block |
| Conditional (option A) | 2 | pot + parallel streams |
| Conditional (option B) | 3 | pot + parallel streams + re-eval |

Option B is 40% fewer evals than current bg_MH. Each stream conditional is ~4x cheaper than a full eval, so the parallel stream batch uses similar GPU resources as 1 full eval batch.

### Implementation

The `block_mh_step` function in `loop.py` is restructured:

1. MH update potential block using `full_log_prob` (existing code, 1 block only)
2. Generate proposals for all stream blocks: for each walker, draw DE directions and propose for all 4 streams
3. Evaluate all proposals via `vmap(vmap(conditional_log_prob))` over (walkers, streams) — 1 batch
4. Accept/reject per (walker, stream) independently
5. Scatter accepted stream params back into full position vectors
6. Re-evaluate `full_log_prob` for all walkers — 1 batch

## Enhancement 2: Delayed Rejection (Simple Mixture)

### Problem

Block MH has ~50% acceptance in 14D blocks. Half the evals are wasted on rejections where the walker stays put.

### Design

When a block's first proposal is rejected, try a second proposal with a smaller step size. This is a valid mixture kernel: with probability ~50% (first accept), apply kernel K1 (large step); with probability ~50% (first reject), apply kernel K2 (small step).

```
Stage 1: y1 = x + mu * d1    → eval, accept with α1 = min(1, π(y1)/π(x))
Stage 2 (if rejected):
          y2 = x + (mu/3) * d2  → eval, accept with α2 = min(1, π(y2)/π(x))
```

Stage 2 uses an independent DE direction d2 and a step size scaled by 1/3. Both stages individually satisfy detailed balance, so the mixture is valid.

### Why not full Green-Mira

Full delayed rejection (Green & Mira 2001) corrects stage 2 for the stage 1 rejection, giving ~85% stage-2 acceptance vs ~70% for the simple version. But it requires evaluating a reference point y1* (1 extra eval). Under GPU saturation:

- Simple: 1.5 evals/block avg, 85% overall acceptance, **0.57 moves/eval**
- Green-Mira: 2.0 evals/block avg, 92% acceptance, **0.46 moves/eval**

The simple version wins on ESS/eval.

### Implementation

In `_mh_walker` (inside `block_mh_step`), after the first MH test:

```python
# Stage 1
x_prop1 = x + mu_eff * d1
lp_prop1 = log_prob(x_prop1)
accept1 = log_u1 < (lp_prop1 - lp_x)

# Stage 2 (only matters if stage 1 rejected)
x_prop2 = x + (mu_eff / 3.0) * d2   # new direction, smaller step
lp_prop2 = log_prob(x_prop2)
accept2 = log_u2 < (lp_prop2 - lp_x)

# Combine: take stage 1 if accepted, else stage 2 if accepted, else stay
x_new = jnp.where(accept1, x_prop1,
        jnp.where(accept2, x_prop2, x))
lp_new = jnp.where(accept1, lp_prop1,
         jnp.where(accept2, lp_prop2, lp_x))
```

Both proposals are always generated from x (not from the accepted state), so they can be computed upfront. However, both require separate log_prob evaluations. Under GPU saturation at 64 walkers, each eval is a sequential batch. Implement as 2 sequential evals per block. The `jnp.where` selection handles the accept/reject logic without data-dependent branching.

### Cost

Average per block: 2 evals (both always evaluated for JIT simplicity).
Overall acceptance: ~85% (vs ~50% for plain MH).

## Enhancement 3: Block Covariance Proposals

### Problem

DE-MCz proposals are 1D lines through the block's parameter space. In a 14D block with correlated parameters (3 sub-funnels, each with different scales), a random 1D direction misses most of the structure.

### Design

After warmup, extract per-block covariance from the Z-matrix and use it to generate covariance-adapted proposals:

```
Current:   y = x + mu * d_hat                      (1D, random DE direction)
Proposed:  y = x + mu * L_block @ z / sqrt(d/2)    (full-dimensional, z ~ N(0,I))
```

where `L_block = cholesky(cov_block)` is precomputed from Z-matrix entries.

### Precomputation (after warmup, before production)

```python
L_blocks = []
for block_idx, block_indices in enumerate(blocks):
    z_block = z_matrix[:z_count, block_indices]   # (n_z, block_dim)
    cov = jnp.cov(z_block.T)                      # (block_dim, block_dim)
    # Regularize for numerical stability
    cov = cov + 1e-8 * jnp.eye(block_dim)
    L_blocks.append(jnp.linalg.cholesky(cov))     # (block_dim, block_dim)
```

This happens once in the post-warmup setup section of `run_variant`, alongside PCA/flow/whitening computation.

### Proposal generation

```python
# In _mh_walker:
z = jax.random.normal(key, (block_dim,))
delta_block = L_block @ z                    # covariance-scaled direction
norm = jnp.sqrt(jnp.sum(delta_block**2))
d_block = delta_block / jnp.maximum(norm, 1e-30)

# Embed into full space
d_full = jnp.zeros(ndim)
d_full = d_full.at[block_indices].set(d_block)

# Step size: use norm as natural scale (like scale_aware)
dim_corr = jnp.sqrt(block_dim / 2.0)
mu_eff = mu_block * norm / dim_corr

x_prop = x + mu_eff * d_full
```

The covariance provides both the direction (via L @ z) AND the natural scale (via the norm of the resulting vector). This replaces both the DE direction and the scale-aware width computation.

### Fallback

If the Z-matrix has too few entries for a reliable covariance estimate (n_z < 2 * block_dim), fall back to standard DE-MCz directions for that block. This handles early warmup gracefully.

### JAX compatibility

`L_blocks` is a list of fixed-shape arrays, precomputed after warmup. They're captured as closure variables in `block_mh_step`. No dynamic shapes — the block sizes are known at compile time.

Since different blocks have different sizes (7 vs 14), the L matrices have different shapes. They can't be stacked into a single array for scan. Options:

**A) Separate L matrices per block**: The potential block (7x7) and stream blocks (14x14) have different sizes. Store as a padded array `(n_blocks, max_block_size, max_block_size)` with masking.

**B) Same L for all stream blocks**: Since the 4 stream blocks have identical structure, compute one shared covariance from all stream entries (pooled). Potential block gets its own. Only 2 L matrices needed.

Recommend **A** — per-block covariance captures differences between streams (e.g., GD1 may have tighter constraints than Pal 5).

### Cost

Same as plain MH: 1 eval per block. The matrix multiply `L @ z` is negligible.

## Composition

All three enhancements compose:

```
Sweep with all 3 enabled:
  Round 1: MH update potential block using full_log_prob
           - Block covariance proposal (7D Gaussian)
           - Delayed rejection if rejected (mu/3, new direction)
           → 2 full evals (stage 1 + stage 2)

  Round 2: Parallel MH update all 4 stream blocks using conditional_log_prob
           - Block covariance proposals (14D Gaussian each)
           - Delayed rejection per stream if rejected
           → 2 batches of 4 conditional evals (stage 1 + stage 2)

  Round 3: Re-evaluate full_log_prob for Z-matrix consistency
           → 1 full eval batch

Total: 5 sequential batches (2 full + 2 conditional + 1 re-eval)
```

Compare to current bg_MH: 5 sequential full eval batches. With conditionals being ~4x cheaper, the effective cost is ~3.5 full-eval-equivalents, with much better acceptance (~85% vs ~50%) and covariance-adapted proposals.

## Testing Strategy

1. **Unit tests**: Round-trip block covariance (generate from known cov, recover it). Delayed rejection acceptance rate in known Gaussian. Conditional log_prob consistency (sum of conditionals ≈ full log_prob).

2. **Convergence tests**: Run on `funnel_63d` and `block_coupled_gaussian` targets with decomposed conditional log_probs. Compare R-hat, ESS, lw_var against plain bg_MH.

3. **Correctness test**: Geweke-style test — verify that the stationary distribution of the enhanced sampler matches the target (at least on a tractable Gaussian).

## Files Changed

- `dezess/core/loop.py` — Restructure `block_mh_step` for conditional parallelism, add delayed rejection logic, add block covariance precomputation and proposal generation
- `dezess/ensemble/block_gibbs.py` — Add `parse_conditional` helper for validating conditional_log_prob
- `dezess/benchmark/registry.py` — Add variant configs for the enhanced block-Gibbs
- `dezess/tests/test_block_gibbs_enhanced.py` — New test file for all 3 enhancements
