# Adaptive Slice + Global Move Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add adaptive while_loop slice sampling and GMM-based global move directions to dezess for mode-jumping in multimodal posteriors.

**Architecture:** Three new files: `dezess/slice/adaptive.py` (while_loop slice), `dezess/directions/gmm.py` (JAX EM fitter), `dezess/directions/global_move.py` (GMM direction strategy). Each plugs into the existing strategy registries. GMM fitting happens post-warmup alongside PCA/flow.

**Tech Stack:** JAX, jax.numpy, jax.lax.while_loop, jax.scipy.special.logsumexp

---

### Task 1: Adaptive Slice Strategy — Test

**Files:**
- Create: `dezess/tests/test_adaptive_global.py`

- [ ] **Step 1: Write the test**

```python
"""Tests for adaptive slice sampling and global move directions."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig

jax.config.update("jax_enable_x64", True)


def test_adaptive_slice_gaussian_variance():
    """Adaptive slice on 20D Gaussian: variance approx 1.0."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    key = jax.random.PRNGKey(42)
    init = jax.random.normal(key, (64, 20)) * 0.1

    config = VariantConfig(
        name="adaptive_test",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="adaptive",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        slice_kwargs={"n_expand": 50, "n_shrink": 50},
    )

    result = run_variant(log_prob, init, n_steps=5000, config=config,
                         n_warmup=2000, verbose=False)
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 20)
    mean_var = float(np.var(flat, axis=0).mean())

    assert 0.8 < mean_var < 1.2, f"mean_var={mean_var:.4f}, expected ~1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_adaptive_global.py::test_adaptive_slice_gaussian_variance -v -s`
Expected: FAIL — `"adaptive"` not in SLICE_STRATEGIES

- [ ] **Step 3: Commit**

```bash
git add dezess/tests/test_adaptive_global.py
git commit -m "test: add adaptive slice variance test"
```

---

### Task 2: Adaptive Slice Strategy — Implementation

**Files:**
- Create: `dezess/slice/adaptive.py`
- Modify: `dezess/core/loop.py:35,70-78` (import + register)

- [ ] **Step 1: Create `dezess/slice/adaptive.py`**

```python
"""Adaptive slice sampling with while_loop for both stepping-out and shrinking.

Like zeus (Karamanis & Beutler 2021), both phases use lax.while_loop
with variable iteration counts. Under vmap, all walkers run until the
slowest one converges — converged walkers are masked but still execute.

The key advantage over fixed-iteration slice: the bracket can extend
arbitrarily far along a DE direction, enabling mode-jumping when the
direction connects two separated modes.

n_expand and n_shrink are safety caps (maximum iterations), not fixed counts.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from dezess.core.slice_sample import safe_log_prob

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def execute(
    log_prob_fn: Callable,
    x: Array,
    d: Array,
    lp_x: Array,
    mu: Array,
    key: Array,
    n_expand: int = 50,
    n_shrink: int = 50,
    **kwargs,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Adaptive slice sampling along direction d.

    Parameters
    ----------
    n_expand : int
        Maximum stepping-out iterations per side (safety cap).
    n_shrink : int
        Maximum shrinking iterations (safety cap).

    Returns ``(x_new, lp_new, key, found, L, R)``.
    """
    key, k_u, k_bracket = jax.random.split(key, 3)
    log_u = lp_x + jnp.log(jax.random.uniform(k_u, dtype=jnp.float64) + 1e-30)

    r0 = jax.random.uniform(k_bracket, dtype=jnp.float64)
    L_init = -r0 * mu
    R_init = L_init + mu

    # --- Stepping-out: while_loop, expands until log_prob < threshold ---
    def _expand_cond(state):
        _, _, exp_L, exp_R, n_iter = state
        return (exp_L | exp_R) & (n_iter < n_expand)

    def _expand_body(state):
        L, R, exp_L, exp_R, n_iter = state
        lp_L = safe_log_prob(log_prob_fn, x + L * d)
        lp_R = safe_log_prob(log_prob_fn, x + R * d)
        still_L = exp_L & (lp_L > log_u)
        still_R = exp_R & (lp_R > log_u)
        L = jnp.where(still_L, L - mu, L)
        R = jnp.where(still_R, R + mu, R)
        return (L, R, still_L, still_R, n_iter + 1)

    L, R, _, _, _ = lax.while_loop(
        _expand_cond, _expand_body,
        (L_init, R_init, jnp.bool_(True), jnp.bool_(True), jnp.int32(0))
    )

    # --- Shrinking: while_loop, stops when acceptable point found ---
    def _shrink_cond(state):
        _, _, _, found, _, n_iter = state
        return (~found) & (n_iter < n_shrink)

    def _shrink_body(state):
        L, R, x_best, found, key, n_iter = state
        key, k_prop = jax.random.split(key)
        t = L + jax.random.uniform(k_prop, dtype=jnp.float64) * (R - L)
        x_prop = x + t * d
        lp_prop = safe_log_prob(log_prob_fn, x_prop)
        accept = lp_prop > log_u
        x_best = jnp.where(accept, x_prop, x_best)
        L = jnp.where(~accept & (t < 0.0), t, L)
        R = jnp.where(~accept & (t >= 0.0), t, R)
        return (L, R, x_best, accept, key, n_iter + 1)

    key, k_shrink = jax.random.split(key)
    _, _, x_new, found, _, _ = lax.while_loop(
        _shrink_cond, _shrink_body,
        (L, R, x, jnp.bool_(False), k_shrink, jnp.int32(0))
    )

    x_new = jnp.where(found, x_new, x)
    lp_new = jnp.where(found, safe_log_prob(log_prob_fn, x_new), lp_x)

    return x_new, lp_new, key, found, L, R
```

- [ ] **Step 2: Register in loop.py**

Add the import at line 35 (after the other slice imports):
```python
from dezess.slice import fixed as fixed_slice, adaptive_budget, delayed_rejection, early_stop, overrelaxed, nurs as nurs_slice, multi_try as multi_try_slice, adaptive as adaptive_slice
```

Add to `SLICE_STRATEGIES` dict (after `"multi_try"` entry, around line 77):
```python
    "adaptive": adaptive_slice,
```

- [ ] **Step 3: Run test**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_adaptive_global.py::test_adaptive_slice_gaussian_variance -v -s`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add dezess/slice/adaptive.py dezess/core/loop.py
git commit -m "feat: add adaptive while_loop slice strategy"
```

---

### Task 3: GMM Fitter — Test

**Files:**
- Modify: `dezess/tests/test_adaptive_global.py`

- [ ] **Step 1: Add GMM fitting test**

Append to `dezess/tests/test_adaptive_global.py`:

```python
def test_gmm_fit_recovers_means():
    """GMM EM on a known 2-component Gaussian recovers means."""
    from dezess.directions.gmm import fit_gmm

    ndim = 5
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    # Generate 2-component mixture
    mu1 = jnp.zeros(ndim).at[0].set(-3.0)
    mu2 = jnp.zeros(ndim).at[0].set(3.0)
    samples1 = jax.random.normal(k1, (500, ndim)) + mu1
    samples2 = jax.random.normal(k2, (500, ndim)) + mu2
    samples = jnp.concatenate([samples1, samples2], axis=0)

    means, covs, weights, chols = fit_gmm(
        samples, jnp.int32(1000), n_components=2, n_iter=100,
        key=jax.random.PRNGKey(42))

    # Sort by first coordinate to match
    order = jnp.argsort(means[:, 0])
    means = means[order]

    assert jnp.abs(means[0, 0] - (-3.0)) < 0.5, f"mean1[0]={means[0,0]:.2f}, expected -3.0"
    assert jnp.abs(means[1, 0] - 3.0) < 0.5, f"mean2[0]={means[1,0]:.2f}, expected 3.0"
    assert jnp.abs(weights[order[0]] - 0.5) < 0.1, f"weight1={weights[order[0]]:.2f}, expected 0.5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_adaptive_global.py::test_gmm_fit_recovers_means -v -s`
Expected: FAIL — `dezess.directions.gmm` doesn't exist

- [ ] **Step 3: Commit**

```bash
git add dezess/tests/test_adaptive_global.py
git commit -m "test: add GMM fitting test"
```

---

### Task 4: GMM Fitter — Implementation

**Files:**
- Create: `dezess/directions/gmm.py`

- [ ] **Step 1: Create `dezess/directions/gmm.py`**

```python
"""Gaussian Mixture Model fitting via EM in pure JAX.

Used by the global_move direction strategy to identify modes
in the Z-matrix and propose inter-mode jumps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def _kmeans_pp_init(data: Array, n_data: int, n_components: int, key: Array) -> Array:
    """K-means++ initialization for GMM means."""
    ndim = data.shape[1]
    centers = jnp.zeros((n_components, ndim), dtype=jnp.float64)

    # First center: random sample
    key, k0 = jax.random.split(key)
    idx = jax.random.randint(k0, (), 0, n_data)
    centers = centers.at[0].set(data[idx])

    def _pick_next(i, state):
        centers, key = state
        # Distance from each point to nearest existing center
        dists = jax.vmap(lambda x: jnp.min(
            jnp.sum((centers[:i] - x) ** 2, axis=1)
        ))(data[:n_data])
        # Sample proportional to distance squared
        key, k = jax.random.split(key)
        probs = dists / jnp.maximum(jnp.sum(dists), 1e-30)
        idx = jax.random.choice(k, n_data, p=probs)
        centers = centers.at[i].set(data[idx])
        return centers, key

    centers, _ = lax.fori_loop(1, n_components, _pick_next, (centers, key))
    return centers


def fit_gmm(
    data: Array,
    n_data: Array,
    n_components: int = 2,
    n_iter: int = 100,
    key: Array = None,
    reg: float = 1e-6,
) -> tuple[Array, Array, Array, Array]:
    """Fit GMM via EM algorithm.

    Parameters
    ----------
    data : (max_n, ndim)
        Data array (padded; only first n_data rows used).
    n_data : int scalar
        Number of valid data points.
    n_components : int
        Number of mixture components.
    n_iter : int
        Number of EM iterations (fixed for JIT).
    key : PRNGKey
        For initialization.
    reg : float
        Covariance regularization.

    Returns
    -------
    means : (K, ndim)
    covs : (K, ndim, ndim)
    weights : (K,)
    chols : (K, ndim, ndim)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n = int(n_data)
    ndim = data.shape[1]
    K = n_components
    X = data[:n]  # (n, ndim)

    # Initialize means via k-means++
    means = _kmeans_pp_init(data, n, K, key)

    # Initialize covariances as identity
    covs = jnp.tile(jnp.eye(ndim, dtype=jnp.float64), (K, 1, 1))

    # Initialize weights as uniform
    weights = jnp.ones(K, dtype=jnp.float64) / K

    def _em_step(_, state):
        means, covs, weights = state

        # E-step: compute log responsibilities
        # log r[n, k] = log w_k + log N(x_n | mu_k, Sigma_k)
        def _log_component(k):
            diff = X - means[k]  # (n, ndim)
            prec = jnp.linalg.inv(covs[k])  # (ndim, ndim)
            sign, log_det = jnp.linalg.slogdet(covs[k])
            mahal = jnp.sum((diff @ prec) * diff, axis=1)  # (n,)
            return -0.5 * (mahal + log_det + ndim * jnp.log(2 * jnp.pi)) + jnp.log(weights[k])

        log_resp = jax.vmap(_log_component)(jnp.arange(K))  # (K, n)
        # Normalize: r[n,k] = exp(log_resp[k,n]) / sum_j exp(log_resp[j,n])
        log_resp_T = log_resp.T  # (n, K)
        log_norm = jax.scipy.special.logsumexp(log_resp_T, axis=1, keepdims=True)
        resp = jnp.exp(log_resp_T - log_norm)  # (n, K)

        # M-step
        N_k = jnp.sum(resp, axis=0)  # (K,)
        N_k_safe = jnp.maximum(N_k, 1e-10)

        # New means
        new_means = (resp.T @ X) / N_k_safe[:, None]  # (K, ndim)

        # New covariances
        def _update_cov(k):
            diff = X - new_means[k]  # (n, ndim)
            weighted = diff * resp[:, k:k+1]  # (n, ndim)
            cov_k = (weighted.T @ diff) / N_k_safe[k]  # (ndim, ndim)
            return cov_k + reg * jnp.eye(ndim, dtype=jnp.float64)

        new_covs = jax.vmap(_update_cov)(jnp.arange(K))  # (K, ndim, ndim)

        # New weights
        new_weights = N_k_safe / jnp.float64(n)

        return new_means, new_covs, new_weights

    means, covs, weights = lax.fori_loop(0, n_iter, _em_step, (means, covs, weights))

    # Precompute Cholesky factors for sampling
    chols = jax.vmap(jnp.linalg.cholesky)(covs)  # (K, ndim, ndim)

    return means, covs, weights, chols
```

- [ ] **Step 2: Run test**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_adaptive_global.py::test_gmm_fit_recovers_means -v -s`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add dezess/directions/gmm.py
git commit -m "feat: add JAX GMM fitter via EM"
```

---

### Task 5: Global Move Direction — Test

**Files:**
- Modify: `dezess/tests/test_adaptive_global.py`

- [ ] **Step 1: Add mode-jumping test**

Append to `dezess/tests/test_adaptive_global.py`:

```python
def test_global_move_finds_both_modes():
    """Global move + adaptive slice finds both modes of a bimodal Gaussian."""
    from dezess.targets import gaussian_mixture

    target = gaussian_mixture(ndim=5, separation=6.0)
    key = jax.random.PRNGKey(42)
    # Initialize walkers in BOTH modes
    init = target.sample(key, 64)

    config = VariantConfig(
        name="global_move_test",
        direction="global_move",
        width="scale_aware",
        slice_fn="adaptive",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        direction_kwargs={"n_components": 2, "global_prob": 0.1},
        slice_kwargs={"n_expand": 50, "n_shrink": 50},
    )

    result = run_variant(target.log_prob, init, n_steps=5000, config=config,
                         n_warmup=2000, verbose=False)
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 5)

    # Check that samples span both modes (mean of dim 0 near 0 if both visited)
    x0 = flat[:, 0]
    frac_left = float((x0 < 0).mean())
    frac_right = float((x0 > 0).mean())

    assert frac_left > 0.1, f"Only {frac_left:.1%} in left mode, expected >10%"
    assert frac_right > 0.1, f"Only {frac_right:.1%} in right mode, expected >10%"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_adaptive_global.py::test_global_move_finds_both_modes -v -s`
Expected: FAIL — `"global_move"` not in DIRECTION_STRATEGIES

- [ ] **Step 3: Commit**

```bash
git add dezess/tests/test_adaptive_global.py
git commit -m "test: add global move mode-jumping test"
```

---

### Task 6: Global Move Direction — Implementation

**Files:**
- Create: `dezess/directions/global_move.py`
- Modify: `dezess/core/loop.py:30,47-60,825+` (import, register, post-warmup GMM fit)

- [ ] **Step 1: Create `dezess/directions/global_move.py`**

```python
"""Global Move direction: GMM-based inter-mode jumps.

Fits a Gaussian Mixture Model to the Z-matrix after warmup, then
proposes directions connecting the current walker to a different
mixture component. This enables mode-jumping in multimodal posteriors.

Mixes local DE-MCz directions (probability 1-global_prob) with
global GMM directions (probability global_prob). Both directions
are computed every step; jnp.where selects which one to use.

Based on zeus's GlobalMove (Karamanis et al. 2021).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dezess.directions import de_mcz

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def sample_direction(
    x_i: Array,
    z_matrix: Array,
    z_count: Array,
    z_log_probs: Array,
    key: Array,
    aux: Array,
    gmm_means: Array = None,
    gmm_covs: Array = None,
    gmm_weights: Array = None,
    gmm_chols: Array = None,
    global_prob: float = 0.1,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample a direction — local DE-MCz or global GMM jump.

    During warmup (gmm_means=None), always uses DE-MCz.
    During production, uses GMM jump with probability global_prob.
    """
    # Always compute local direction (DE-MCz fallback)
    key, k_local = jax.random.split(key)
    d_local, k_local_out, aux_local = de_mcz.sample_direction(
        x_i, z_matrix, z_count, z_log_probs, k_local, aux, **kwargs)

    if gmm_means is None:
        return d_local, key, aux_local

    K = gmm_means.shape[0]
    ndim = x_i.shape[0]

    # Compute global direction
    key, k_global, k_choice, k_target, k_select = jax.random.split(key, 5)

    # Identify which component x_i belongs to (highest responsibility)
    def _log_resp(k):
        diff = x_i - gmm_means[k]
        prec = jnp.linalg.inv(gmm_covs[k])
        mahal = diff @ prec @ diff
        sign, log_det = jnp.linalg.slogdet(gmm_covs[k])
        return -0.5 * (mahal + log_det) + jnp.log(gmm_weights[k])

    log_resps = jax.vmap(_log_resp)(jnp.arange(K))
    current_component = jnp.argmax(log_resps)

    # Select target component != current (weighted by gmm_weights)
    # Zero out current component's weight, renormalize
    target_weights = gmm_weights.at[current_component].set(0.0)
    target_weights = target_weights / jnp.maximum(jnp.sum(target_weights), 1e-30)
    target_component = jax.random.choice(k_target, K, p=target_weights)

    # Sample target point from target component
    z = jax.random.normal(k_global, (ndim,), dtype=jnp.float64)
    target_point = gmm_means[target_component] + gmm_chols[target_component] @ z

    # Direction = target - current position (unnormalized)
    diff_global = target_point - x_i
    norm_global = jnp.linalg.norm(diff_global)
    d_global = diff_global / jnp.maximum(norm_global, 1e-30)

    # Choose local or global
    use_global = jax.random.uniform(k_select, dtype=jnp.float64) < global_prob
    d = jnp.where(use_global, d_global, d_local)
    norm = jnp.where(use_global, norm_global, aux_local.direction_scale)

    aux_out = aux._replace(direction_scale=norm)
    return d, key, aux_out
```

- [ ] **Step 2: Register in loop.py and add post-warmup GMM fitting**

Add import at line 30 (with other direction imports):
```python
from dezess.directions import de_mcz, snooker, pca, weighted_pair, momentum, riemannian, flow, whitened, gradient, coordinate, local_pair, kde_direction, global_move
```

Add to `DIRECTION_STRATEGIES` dict (after `"kde"`, around line 59):
```python
    "global_move": global_move,
```

Add GMM fitting in the post-warmup section (after the KDE bandwidth block, around line 870). Follow the exact pattern of the existing PCA/flow/whitening blocks:

```python
    # Fit GMM if using global_move directions
    if config.direction == "global_move" and n_warmup > 0:
        from dezess.directions.gmm import fit_gmm
        n_comp = dir_kwargs.get("n_components", 2)
        if verbose:
            print(f"  [{config.name}] Fitting GMM ({n_comp} components)...", flush=True)
        key, k_gmm = jax.random.split(key)
        gmm_means, gmm_covs, gmm_weights, gmm_chols = fit_gmm(
            z_padded, z_count, n_components=n_comp,
            n_iter=dir_kwargs.get("gmm_n_iter", 100),
            key=k_gmm,
        )
        dir_kwargs["gmm_means"] = gmm_means
        dir_kwargs["gmm_covs"] = gmm_covs
        dir_kwargs["gmm_weights"] = gmm_weights
        dir_kwargs["gmm_chols"] = gmm_chols
        if verbose:
            print(f"  [{config.name}] GMM fitted: weights={np.array(gmm_weights).round(3)}", flush=True)
```

Also add `global_move` to the re-JIT condition (find the line with `config.direction in ("pca", "flow", "whitened", "kde")` and add `"global_move"`):
```python
    if not use_block_gibbs and (needs_rejit or config.direction in ("pca", "flow", "whitened", "kde", "global_move")):
```

- [ ] **Step 3: Run test**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_adaptive_global.py::test_global_move_finds_both_modes -v -s`
Expected: PASS — walkers visit both modes

- [ ] **Step 4: Commit**

```bash
git add dezess/directions/global_move.py dezess/core/loop.py
git commit -m "feat: add global move direction with GMM-based mode jumping"
```

---

### Task 7: Registry Configs + Full Test Suite

**Files:**
- Modify: `dezess/benchmark/registry.py`
- Modify: `dezess/tests/test_adaptive_global.py`

- [ ] **Step 1: Add variant configs to registry**

Add to the VARIANTS dict in `dezess/benchmark/registry.py` (before the closing `}`):

```python
    # --- Adaptive slice + Global move variants ---
    "adaptive_slice": VariantConfig(
        name="adaptive_slice",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="adaptive",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        slice_kwargs={"n_expand": 50, "n_shrink": 50},
    ),
    "global_move": VariantConfig(
        name="global_move",
        direction="global_move",
        width="scale_aware",
        slice_fn="adaptive",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        direction_kwargs={"n_components": 2, "global_prob": 0.1},
        slice_kwargs={"n_expand": 50, "n_shrink": 50},
    ),
```

- [ ] **Step 2: Add combined test verifying existing tests still pass**

Append to `dezess/tests/test_adaptive_global.py`:

```python
def test_adaptive_slice_block_gibbs():
    """Adaptive slice works with block-Gibbs MH."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    key = jax.random.PRNGKey(42)
    init = jax.random.normal(key, (32, 21)) * 0.1

    config = VariantConfig(
        name="adaptive_bg",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="adaptive",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14]},
        slice_kwargs={"n_expand": 20, "n_shrink": 20},
    )

    result = run_variant(log_prob, init, n_steps=3000, config=config,
                         n_warmup=1000, verbose=False)
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 21)
    mean_var = float(np.var(flat, axis=0).mean())

    assert 0.8 < mean_var < 1.2, f"mean_var={mean_var:.4f}, expected ~1.0"
```

- [ ] **Step 3: Run full test suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -v -s`
Expected: All tests PASS (68 existing + 4 new = 72)

- [ ] **Step 4: Commit**

```bash
git add dezess/benchmark/registry.py dezess/tests/test_adaptive_global.py
git commit -m "feat: add adaptive slice + global move registry configs and tests"
```

---

### Task 8: Mode-Jumping Benchmark

**Files:**
- Create: `bench_mode_jumping.py`

- [ ] **Step 1: Create benchmark script**

```python
#!/usr/bin/env python
"""Benchmark: mode-jumping on bimodal Gaussian mixture."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time, gc
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.benchmark.metrics import compute_ess, compute_rhat
from dezess.targets import gaussian_mixture

print(f"JAX devices: {jax.devices()}")

target = gaussian_mixture(ndim=10, separation=8.0)

configs = {
    "default (fixed slice)": VariantConfig(
        name="scale_aware", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    ),
    "adaptive slice": VariantConfig(
        name="adaptive", direction="de_mcz", width="scale_aware",
        slice_fn="adaptive", zmatrix="circular", ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        slice_kwargs={"n_expand": 50, "n_shrink": 50},
    ),
    "global_move + adaptive": VariantConfig(
        name="global_move", direction="global_move", width="scale_aware",
        slice_fn="adaptive", zmatrix="circular", ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        direction_kwargs={"n_components": 2, "global_prob": 0.1},
        slice_kwargs={"n_expand": 50, "n_shrink": 50},
    ),
}

key = jax.random.PRNGKey(42)
init = target.sample(key, 64)  # walkers in both modes

hdr = f"{'Strategy':<30s} | {'R-hat':>6s} | {'ESS min':>8s} | {'Wall':>6s} | {'Left %':>7s} | {'Right %':>7s}"
print(f"\n{hdr}")
print("-" * len(hdr))

for name, config in configs.items():
    t0 = time.time()
    result = run_variant(target.log_prob, init, n_steps=7000, config=config,
                         n_warmup=2000, verbose=False)
    wall = time.time() - t0
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 10)
    ess = compute_ess(samples)
    rhat = compute_rhat(samples)
    frac_left = float((flat[:, 0] < 0).mean())
    frac_right = float((flat[:, 0] > 0).mean())
    print(f"{name:<30s} | {float(rhat.max()):6.3f} | {float(ess.min()):8.1f} | "
          f"{wall:5.1f}s | {frac_left:6.1%} | {frac_right:6.1%}", flush=True)
    gc.collect()
```

- [ ] **Step 2: Run benchmark**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro python bench_mode_jumping.py`

Expected: global_move + adaptive shows walkers in both modes (Left % and Right % both > 10%). Default fixed slice may fail to jump.

- [ ] **Step 3: Commit**

```bash
git add bench_mode_jumping.py
git commit -m "bench: add mode-jumping benchmark on bimodal Gaussian"
```
