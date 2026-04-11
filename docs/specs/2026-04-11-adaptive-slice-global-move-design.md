# Adaptive Slice Sampling + Global Move Design

## Goal

Two new pluggable strategies for dezess: (1) an adaptive slice execution strategy using `while_loop` for both stepping-out and shrinking, enabling mode-jumping along DE directions; (2) a Global Move direction strategy that fits a GMM to the Z-matrix and proposes jumps between mixture components.

Together: adaptive slice extends brackets far enough to reach distant modes, and global move provides directions that explicitly connect modes.

## Motivation

The GD1 stream-fitting posterior shows bimodality in potential parameters. The current fixed-iteration slice (`fori_loop` with `n_expand=2`) can't extend brackets far enough to bridge modes. Block-Gibbs MH can't jump modes because it updates one block at a time (no coordinated full-dimensional jump). Default slice with adaptive stepping-out can bridge modes if the DE direction connects them — and global move ensures directions DO connect modes.

## Constraint

All new code must work within dezess's existing pluggable architecture. No changes to `loop.py`'s step function structure. The adaptive slice is a new file in `dezess/slice/`, the global move is a new file in `dezess/directions/`, and the GMM fitter is a utility in `dezess/directions/`.

---

## Feature 1: Adaptive Slice Strategy

### File: `dezess/slice/adaptive.py`

Same `execute()` interface as all other slice strategies:

```python
def execute(log_prob_fn, x, d, lp_x, mu, key,
            n_expand=50, n_shrink=50, **kwargs):
    """Adaptive slice sampling with while_loop for both phases.
    
    n_expand: maximum stepping-out iterations (safety cap)
    n_shrink: maximum shrinking iterations (safety cap)
    """
```

### Stepping-out (while_loop)

Expands bracket `[L, R]` outward along direction `d` until `log_prob` drops below the slice threshold on both sides, or `n_expand` iterations reached:

```python
def _expand_cond(state):
    L, R, exp_L, exp_R, n_iter = state
    return (exp_L | exp_R) & (n_iter < n_expand)

def _expand_body(state):
    L, R, exp_L, exp_R, n_iter = state
    lp_L = lp_eval(log_prob_fn, x + L * d)
    lp_R = lp_eval(log_prob_fn, x + R * d)
    still_L = exp_L & (lp_L > log_u)
    still_R = exp_R & (lp_R > log_u)
    L = jnp.where(still_L, L - mu, L)
    R = jnp.where(still_R, R + mu, R)
    return (L, R, still_L, still_R, n_iter + 1)
```

Under vmap, all walkers run until the slowest one finishes. Converged walkers are masked (ops execute but results discarded). This is the same behavior as `zeus_jax.py` lines 129-149.

### Shrinking (while_loop)

Proposes points within the bracket until one is accepted, or `n_shrink` iterations reached:

```python
def _shrink_cond(state):
    _, _, _, found, _, n_iter = state
    return (~found) & (n_iter < n_shrink)
```

Same as `early_stop.py`'s shrinking, which already uses while_loop.

### Registration

Add to `SLICE_STRATEGIES` in `loop.py`:
```python
from dezess.slice import adaptive as adaptive_slice
SLICE_STRATEGIES["adaptive"] = adaptive_slice
```

### Usage

```python
config = VariantConfig(
    slice_fn="adaptive",
    slice_kwargs={"n_expand": 50, "n_shrink": 50},
)
```

Compatible with all direction strategies, width strategies, and ensemble strategies (including block_gibbs).

---

## Feature 2: Global Move Direction Strategy

### File: `dezess/directions/global_move.py`

Direction strategy that proposes jumps between GMM components fitted to the Z-matrix. Only active during production (after warmup).

```python
def sample_direction(x_i, z_matrix, z_count, z_log_probs, key, aux,
                     gmm_means=None, gmm_covs=None, gmm_weights=None,
                     gmm_chols=None, **kwargs):
    """Propose direction connecting x_i to a different GMM component."""
```

### Algorithm

1. Identify which component `x_i` belongs to (highest responsibility)
2. Select a target component ≠ current (sample proportional to weights, excluding current)
3. Sample a target point from the target component: `target = mean_k + L_k @ z`, `z ~ N(0,I)`
4. Direction = `target - x_i` (unnormalized — magnitude encodes the inter-mode distance)

The direction module returns the normalized direction and stores the raw magnitude in `aux.direction_scale` for scale-aware width computation.

### Fallback

When `gmm_means is None` (during warmup), fall back to standard DE-MCz direction:

```python
if gmm_means is None:
    return de_mcz.sample_direction(x_i, z_matrix, z_count, z_log_probs, key, aux)
```

### Post-warmup setup

In `loop.py`'s post-warmup section, add GMM fitting alongside existing PCA/flow/whitening:

```python
if config.direction == "global_move" and n_warmup > 0:
    n_components = dir_kwargs.get("n_components", 2)
    gmm_means, gmm_covs, gmm_weights, gmm_chols = fit_gmm(
        z_padded, z_count, n_components=n_components)
    dir_kwargs["gmm_means"] = gmm_means
    dir_kwargs["gmm_covs"] = gmm_covs
    dir_kwargs["gmm_weights"] = gmm_weights
    dir_kwargs["gmm_chols"] = gmm_chols
```

---

## Feature 3: JAX GMM via EM

### File: `dezess/directions/gmm.py`

Pure JAX implementation of Gaussian Mixture Model fitting via Expectation-Maximization.

```python
def fit_gmm(z_matrix, z_count, n_components=2, n_iter=100, key=None, reg=1e-6):
    """Fit GMM to Z-matrix samples via EM.
    
    Returns (means, covs, weights, chols):
        means: (K, ndim)
        covs: (K, ndim, ndim)  
        weights: (K,)
        chols: (K, ndim, ndim) — precomputed Cholesky for sampling
    """
```

### Initialization

K-means++ style: pick first center randomly from Z-matrix, subsequent centers with probability proportional to distance from nearest existing center. This gives well-separated initial components.

```python
def _kmeans_pp_init(z_matrix, z_count, n_components, key):
    """K-means++ initialization for GMM means."""
```

### EM Algorithm

Fixed `n_iter` iterations via `lax.fori_loop` (JIT-friendly, convergence is fast with thousands of samples):

**E-step:** Compute log-responsibilities for each sample under each component. Use `jax.scipy.special.logsumexp` for numerical stability.

```python
# log p(x_n | k) = -0.5 * (x - mu_k)^T Sigma_k^{-1} (x - mu_k) - 0.5 * log|Sigma_k| + log(w_k)
# responsibilities[n, k] = softmax over k of log p(x_n | k)
```

**M-step:** Update means, covariances, weights from responsibilities.

```python
# N_k = sum_n r[n,k]
# mu_k = (1/N_k) * sum_n r[n,k] * x_n
# Sigma_k = (1/N_k) * sum_n r[n,k] * (x_n - mu_k)(x_n - mu_k)^T + reg * I
# w_k = N_k / N
```

All operations are vmapped over components and use batched matrix operations. No Python loops.

### Regularization

Add `reg * I` to each covariance after M-step to prevent singularity. Default `reg=1e-6`.

---

## Mixing Local + Global Directions

For multimodal targets, use the existing `n_slices_per_step` mechanism:

```python
config = VariantConfig(
    direction="global_move",
    slice_fn="adaptive",
    n_slices_per_step=2,    # first slice: DE-MCz (local), second: global_move
    direction_kwargs={"n_components": 2},
    slice_kwargs={"n_expand": 50, "n_shrink": 50},
)
```

Wait — `n_slices_per_step` currently uses the SAME direction module for all slices, just with Gram-Schmidt orthogonalization. To mix DE-MCz + global_move, we need the global_move module itself to randomly choose between local (DE-MCz) and global (GMM) directions:

```python
def sample_direction(x_i, z_matrix, z_count, z_log_probs, key, aux,
                     gmm_means=None, global_prob=0.1, **kwargs):
    key, k_choice = jax.random.split(key)
    use_global = jax.random.uniform(k_choice) < global_prob
    # Python-level if won't work (traced). Use lax.cond:
    d_global, key_g, aux_g = _global_direction(...)
    d_local, key_l, aux_l = _local_direction(...)  # DE-MCz fallback
    d = jnp.where(use_global, d_global, d_local)
    ...
```

Default `global_prob=0.1` — 10% of steps propose mode jumps, 90% do local exploration. Configurable via `direction_kwargs={"global_prob": 0.2}`.

Since `jnp.where` evaluates both branches (both directions are computed), the cost is 2× direction computation per step. But direction computation is negligible compared to log_prob evaluation, so this is fine.

---

## Testing Strategy

1. **Adaptive slice unit test:** Run on standard Gaussian, verify variance ≈ 1.0. Compare ESS against fixed slice.

2. **Mode-jumping test:** Create a bimodal Gaussian (2 well-separated modes). Run with adaptive slice + global_move. Verify walkers visit both modes (both means recovered in samples). Run with default slice — verify it fails to jump.

3. **GMM fitting test:** Fit GMM to known 2-component Gaussian mixture. Verify recovered means/covs/weights match ground truth within tolerance.

4. **Convergence test:** Run on `block_coupled_gaussian` and `funnel_63d` targets. Verify R-hat and ESS are comparable to or better than existing strategies.

---

## Files

- Create: `dezess/slice/adaptive.py` — adaptive while_loop slice strategy
- Create: `dezess/directions/global_move.py` — GMM-based global direction strategy
- Create: `dezess/directions/gmm.py` — JAX EM GMM fitter
- Modify: `dezess/core/loop.py` — register adaptive in SLICE_STRATEGIES, register global_move in DIRECTION_STRATEGIES, add GMM fitting to post-warmup setup
- Modify: `dezess/benchmark/registry.py` — add variant configs
- Create: `dezess/tests/test_adaptive_global.py` — tests for both features
