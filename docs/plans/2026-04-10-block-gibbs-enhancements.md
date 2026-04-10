# Block-Gibbs MH Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three composable enhancements to block-Gibbs MH — delayed rejection, block covariance proposals, and conditional independence parallelism — to maximize ESS per sequential log_prob eval under GPU saturation.

**Architecture:** Each enhancement modifies the `block_mh_step` JIT function in `loop.py`, toggled via `ensemble_kwargs` flags. Delayed rejection adds a second MH proposal at smaller scale. Block covariance replaces 1D DE directions with full-dimensional Gaussian proposals from precomputed per-block Cholesky factors. Conditional independence restructures the sweep into 2 rounds (potential + parallel streams) using a user-provided conditional log_prob.

**Tech Stack:** JAX, jax.numpy, jax.vmap, jax.lax.scan, numpy (for covariance precomputation)

---

### Task 1: Delayed Rejection — Test

**Files:**
- Create: `dezess/tests/test_block_gibbs_enhanced.py`

- [ ] **Step 1: Write the failing test for delayed rejection**

```python
"""Tests for block-Gibbs MH enhancements."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig

jax.config.update("jax_enable_x64", True)


def test_delayed_rejection_variance():
    """Delayed rejection on 63D Gaussian: variance ≈ 1.0, better acceptance than plain MH."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    key = jax.random.PRNGKey(42)
    init = jax.random.normal(key, (32, 63)) * 0.1

    config = VariantConfig(
        name="bg_mh_dr",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14, 14, 14, 14],
            "use_mh": True,
            "delayed_rejection": True,
        },
    )

    result = run_variant(log_prob, init, n_steps=5000, config=config,
                         n_warmup=2000, verbose=False)
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 63)
    mean_var = float(np.var(flat, axis=0).mean())

    assert 0.8 < mean_var < 1.2, f"mean_var={mean_var:.4f}, expected ~1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_block_gibbs_enhanced.py::test_delayed_rejection_variance -v -s`
Expected: FAIL — `delayed_rejection` kwarg not yet handled

- [ ] **Step 3: Commit test**

```bash
git add dezess/tests/test_block_gibbs_enhanced.py
git commit -m "test: add delayed rejection variance test for block-Gibbs MH"
```

---

### Task 2: Delayed Rejection — Implementation

**Files:**
- Modify: `dezess/core/loop.py:499-561` (the `block_mh_step` function)

- [ ] **Step 1: Add `delayed_rejection` flag parsing in block-Gibbs setup**

In `loop.py`, after the `use_block_mh` line (around line 427):

```python
    use_block_mh = use_block_gibbs and ens_kwargs.get("use_mh", False)
    use_delayed_rejection = use_block_mh and ens_kwargs.get("delayed_rejection", False)
```

- [ ] **Step 2: Modify `_mh_walker` in `block_mh_step` to add delayed rejection**

Replace the `_mh_walker` function body inside `block_mh_step` (lines 510-545) with:

```python
                def _mh_walker(x_full, lp, wk):
                    wk, k1, k2, k_accept1 = jax.random.split(wk, 4)
                    idx1 = jax.random.randint(k1, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jax.random.randint(k2, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

                    # DE direction in block subspace
                    z1_block = z_padded[idx1][b_idx]
                    z2_block = z_padded[idx2][b_idx]
                    diff = (z1_block - z2_block) * mask
                    norm = jnp.sqrt(jnp.sum(diff**2))

                    # Scale-aware step
                    dim_corr = jnp.sqrt(jnp.float64(bsize) / 2.0)
                    mu_eff = mu_b * norm / jnp.maximum(dim_corr, 1e-30)

                    # Full-space direction
                    d_block = diff / jnp.maximum(norm, 1e-30)
                    d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                    d_full = d_full.at[b_idx].add(d_block * mask)

                    # Stage 1: full-scale proposal
                    x_prop1 = x_full + mu_eff * d_full
                    lp_prop1 = safe_log_prob(log_prob_fn, x_prop1)
                    log_u1 = jnp.log(jax.random.uniform(k_accept1, dtype=jnp.float64) + 1e-30)
                    accept1 = log_u1 < (lp_prop1 - lp)

                    if use_delayed_rejection:
                        # Stage 2: smaller step, new direction
                        wk, k3, k4, k_accept2 = jax.random.split(wk, 4)
                        idx3 = jax.random.randint(k3, (), 0, z_padded.shape[0]) % z_count
                        idx4 = jax.random.randint(k4, (), 0, z_padded.shape[0]) % z_count
                        idx4 = jnp.where(idx3 == idx4, (idx3 + 1) % z_count, idx4)

                        z3_block = z_padded[idx3][b_idx]
                        z4_block = z_padded[idx4][b_idx]
                        diff2 = (z3_block - z4_block) * mask
                        norm2 = jnp.sqrt(jnp.sum(diff2**2))
                        mu_eff2 = mu_b * norm2 / jnp.maximum(dim_corr, 1e-30) / 3.0

                        d_block2 = diff2 / jnp.maximum(norm2, 1e-30)
                        d_full2 = jnp.zeros(n_dim, dtype=jnp.float64)
                        d_full2 = d_full2.at[b_idx].add(d_block2 * mask)

                        x_prop2 = x_full + mu_eff2 * d_full2
                        lp_prop2 = safe_log_prob(log_prob_fn, x_prop2)
                        log_u2 = jnp.log(jax.random.uniform(k_accept2, dtype=jnp.float64) + 1e-30)
                        accept2 = log_u2 < (lp_prop2 - lp)

                        # Stage 1 wins if accepted, else stage 2, else stay
                        x_new = jnp.where(accept1, x_prop1,
                                jnp.where(accept2, x_prop2, x_full))
                        lp_new = jnp.where(accept1, lp_prop1,
                                 jnp.where(accept2, lp_prop2, lp))
                        either_accepted = accept1 | accept2
                    else:
                        x_new = jnp.where(accept1, x_prop1, x_full)
                        lp_new = jnp.where(accept1, lp_prop1, lp)
                        either_accepted = accept1

                    target = jnp.float64(n_expand + 1)
                    br = jnp.where(either_accepted, 2.0 * target - 1.0, 1.0)
                    return x_new, lp_new, wk, br
```

Note: The `if use_delayed_rejection` is a Python-level check on a bool constant, resolved at JIT trace time — no data-dependent branching.

- [ ] **Step 3: Run test to verify it passes**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_block_gibbs_enhanced.py::test_delayed_rejection_variance -v -s`
Expected: PASS with mean_var ≈ 1.0

- [ ] **Step 4: Commit**

```bash
git add dezess/core/loop.py
git commit -m "feat: add delayed rejection to block-Gibbs MH"
```

---

### Task 3: Block Covariance Proposals — Test

**Files:**
- Modify: `dezess/tests/test_block_gibbs_enhanced.py`

- [ ] **Step 1: Add test for block covariance proposals**

Append to `dezess/tests/test_block_gibbs_enhanced.py`:

```python
def test_block_covariance_variance():
    """Block covariance proposals on 63D Gaussian: variance ≈ 1.0."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    key = jax.random.PRNGKey(42)
    init = jax.random.normal(key, (32, 63)) * 0.1

    config = VariantConfig(
        name="bg_mh_cov",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14, 14, 14, 14],
            "use_mh": True,
            "use_block_cov": True,
        },
    )

    result = run_variant(log_prob, init, n_steps=5000, config=config,
                         n_warmup=2000, verbose=False)
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 63)
    mean_var = float(np.var(flat, axis=0).mean())

    assert 0.8 < mean_var < 1.2, f"mean_var={mean_var:.4f}, expected ~1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_block_gibbs_enhanced.py::test_block_covariance_variance -v -s`
Expected: FAIL — `use_block_cov` not yet handled

- [ ] **Step 3: Commit test**

```bash
git add dezess/tests/test_block_gibbs_enhanced.py
git commit -m "test: add block covariance proposal test"
```

---

### Task 4: Block Covariance Proposals — Implementation

**Files:**
- Modify: `dezess/core/loop.py` (post-warmup setup + `block_mh_step`)

- [ ] **Step 1: Add `use_block_cov` flag parsing**

In `loop.py`, after the `use_delayed_rejection` line:

```python
    use_block_cov = use_block_mh and ens_kwargs.get("use_block_cov", False)
```

- [ ] **Step 2: Add block covariance precomputation in post-warmup setup**

In the `# --- Post-warmup setup ---` section (after line 791), add a new block for covariance computation. Place it after the existing PCA/flow/whitening blocks:

```python
    # Compute per-block Cholesky factors if using block covariance proposals
    # Padded to (n_blocks, max_block_size, max_block_size) for scan compatibility
    if use_block_cov and n_warmup > 0:
        if verbose:
            print(f"  [{config.name}] Computing per-block covariance...", flush=True)
        block_L_padded = np.zeros((n_blocks_val, max_block_size, max_block_size),
                                  dtype=np.float64)
        z_np = np.array(z_padded[:int(z_count)])
        for bi, b in enumerate(blocks):
            b_np = np.array(b)
            bsz = len(b_np)
            z_block = z_np[:, b_np[:bsz]]  # (n_z, bsz)
            if z_block.shape[0] > 2 * bsz:
                cov_b = np.cov(z_block.T)
                cov_b += 1e-8 * np.eye(bsz)
                L_b = np.linalg.cholesky(cov_b)
                block_L_padded[bi, :bsz, :bsz] = L_b
            else:
                # Not enough samples — use identity (fallback to DE directions)
                block_L_padded[bi, :bsz, :bsz] = np.eye(bsz)
        block_L_padded = jnp.array(block_L_padded, dtype=jnp.float64)
    elif use_block_cov:
        # No warmup — identity fallback
        block_L_padded = jnp.zeros((n_blocks_val, max_block_size, max_block_size),
                                    dtype=jnp.float64)
        for bi, b in enumerate(blocks):
            bsz = len(b)
            block_L_padded = block_L_padded.at[bi, :bsz, :bsz].set(jnp.eye(bsz))
    else:
        block_L_padded = None
```

- [ ] **Step 3: Modify `_mh_walker` to use block covariance when enabled**

Replace the DE direction generation section of `_mh_walker` (the lines from `idx1 = ...` through `d_full = ...`) with a version that conditionally uses block covariance. The `if use_block_cov` is a Python-level constant, resolved at trace time:

```python
                def _mh_walker(x_full, lp, wk):
                    wk, k1, k2, k_accept1 = jax.random.split(wk, 4)

                    if use_block_cov:
                        # Covariance-adapted proposal: L @ z
                        L_b = block_L_padded[block_idx]  # (max_block_size, max_block_size)
                        z_rand = jax.random.normal(k1, (max_block_size,), dtype=jnp.float64)
                        z_rand = z_rand * mask  # zero out padding dims
                        delta = L_b @ z_rand    # (max_block_size,)
                        delta = delta * mask
                        norm = jnp.sqrt(jnp.sum(delta**2))
                        d_block = delta / jnp.maximum(norm, 1e-30)
                    else:
                        # Standard DE direction
                        idx1 = jax.random.randint(k1, (), 0, z_padded.shape[0]) % z_count
                        idx2 = jax.random.randint(k2, (), 0, z_padded.shape[0]) % z_count
                        idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
                        z1_block = z_padded[idx1][b_idx]
                        z2_block = z_padded[idx2][b_idx]
                        diff = (z1_block - z2_block) * mask
                        norm = jnp.sqrt(jnp.sum(diff**2))
                        d_block = diff / jnp.maximum(norm, 1e-30)

                    dim_corr = jnp.sqrt(jnp.float64(bsize) / 2.0)
                    mu_eff = mu_b * norm / jnp.maximum(dim_corr, 1e-30)

                    d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                    d_full = d_full.at[b_idx].add(d_block * mask)

                    # ... rest of MH (stage 1 + optional stage 2) unchanged ...
```

Important: `block_idx` is the ORIGINAL block index (from `block_data`), not the permuted scan index. In the current `_one_block_mh`, the `block_data` tuple contains `(block_idx, bsize, mu_b)` where `block_idx` is the perm'd index. We need to use this to index into `block_L_padded`. This already works because `block_L_padded[block_idx]` accesses the correct block's Cholesky factor.

Also important: for delayed rejection stage 2 with block covariance, generate a SECOND random `z_rand2` for the stage 2 direction (it should be independent).

- [ ] **Step 4: Pass `block_L_padded` into the JIT function and rebuild if needed**

The `block_mh_step` function needs access to `block_L_padded`. Since it's defined inside `if use_block_gibbs:`, `block_L_padded` is a closure variable. However, `block_L_padded` is computed AFTER `block_mh_step` is defined.

Fix: move `block_mh_step` definition to after the post-warmup setup, OR make `block_L_padded` a parameter. The cleanest approach: define a NEW JIT function `block_mh_cov_step` in the post-warmup section that captures `block_L_padded`, and set `_block_step_fn = block_mh_cov_step`. This follows the pattern already used for re-JIT after PCA/flow.

```python
    if use_block_cov and n_warmup > 0:
        # ... covariance computation from Step 2 ...

        # Rebuild block_mh_step with covariance captured
        @jax.jit
        def block_mh_cov_step(positions, log_probs, z_padded, z_count,
                              mu_blocks_arg, key):
            # ... same structure as block_mh_step but _mh_walker uses block_L_padded ...
            # (block_L_padded is captured as closure variable)
        
        _block_step_fn = block_mh_cov_step

        if verbose:
            print(f"  [{config.name}] Re-JIT compiling with block covariance...", flush=True)
        t_rejit = time.time()
        positions, log_probs, key, _ = _block_step_fn(
            positions, log_probs, z_padded, z_count, mu_blocks, key,
        )
        positions.block_until_ready()
        if verbose:
            print(f"  [{config.name}] Re-JIT: {time.time()-t_rejit:.1f}s", flush=True)
```

The full `block_mh_cov_step` function body is the same as `block_mh_step` but with the covariance proposal logic from Step 3 inside `_mh_walker`.

- [ ] **Step 5: Run test to verify it passes**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_block_gibbs_enhanced.py::test_block_covariance_variance -v -s`
Expected: PASS with mean_var ≈ 1.0

- [ ] **Step 6: Commit**

```bash
git add dezess/core/loop.py
git commit -m "feat: add block covariance proposals to block-Gibbs MH"
```

---

### Task 5: Conditional Independence — Test

**Files:**
- Modify: `dezess/tests/test_block_gibbs_enhanced.py`

- [ ] **Step 1: Add a block-coupled Gaussian with conditional log_prob**

Append to `dezess/tests/test_block_gibbs_enhanced.py`:

```python
def _make_block_coupled_conditional():
    """Build a block-coupled 63D Gaussian with decomposed conditional log_prob."""
    from dezess.targets_stream import block_coupled_gaussian

    n_pot, n_streams, n_nuis = 7, 4, 14
    ndim = n_pot + n_streams * n_nuis
    target = block_coupled_gaussian(n_potential=n_pot, n_streams=n_streams,
                                     n_nuisance_per_stream=n_nuis)

    # Extract precision matrix for conditional decomposition
    prec = np.linalg.inv(np.array(target.cov))
    prec_jax = jnp.array(prec, dtype=jnp.float64)

    # Full log_prob (already on target)
    full_lp = target.log_prob

    # Conditional log_prob for stream block_idx (0-indexed among streams)
    # For a Gaussian: log p(x_s | x_pot, x_rest) ∝ -0.5 * x^T P x
    # Since streams are conditionally independent given pot, the conditional
    # only depends on pot and stream_s params. But for simplicity (and
    # correctness of the MH ratio), we can use the FULL log_prob as the
    # conditional — the other terms cancel in the acceptance ratio.
    # The point of the conditional is to be CHEAPER, not different.
    #
    # For testing, use the full log_prob as a stand-in (no speedup, but
    # verifies the parallel sweep logic is correct).
    def conditional_log_prob(params, block_idx):
        return full_lp(params)

    return target, conditional_log_prob


def test_conditional_independence_variance():
    """Conditional independence parallelism: variance ≈ 1.0 on block-coupled Gaussian."""
    target, cond_lp = _make_block_coupled_conditional()
    key = jax.random.PRNGKey(42)
    init = target.sample(key, 32)

    config = VariantConfig(
        name="bg_mh_cond",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14, 14, 14, 14],
            "use_mh": True,
            "conditional_log_prob": cond_lp,
        },
    )

    result = run_variant(target.log_prob, init, n_steps=5000, config=config,
                         n_warmup=2000, verbose=False)
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, target.ndim)
    mean_var = float(np.var(flat, axis=0).mean())

    # Check against known true variance
    true_var = np.diag(np.array(target.cov))
    sample_var = np.var(flat, axis=0)
    var_ratio = float(np.mean(sample_var / true_var))

    assert 0.7 < var_ratio < 1.3, f"var_ratio={var_ratio:.4f}, expected ~1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_block_gibbs_enhanced.py::test_conditional_independence_variance -v -s`
Expected: FAIL — `conditional_log_prob` kwarg not yet handled

- [ ] **Step 3: Commit test**

```bash
git add dezess/tests/test_block_gibbs_enhanced.py
git commit -m "test: add conditional independence parallelism test"
```

---

### Task 6: Conditional Independence — Implementation

**Files:**
- Modify: `dezess/core/loop.py` (restructure `block_mh_step` for conditional parallelism)

This is the most complex task. The sweep is restructured from sequential scan over all blocks to: (1) potential block update, (2) parallel stream updates via conditional, (3) full log_prob re-eval.

- [ ] **Step 1: Parse conditional_log_prob and identify conditional blocks**

In the block-Gibbs setup section (after `use_block_cov`):

```python
    conditional_log_prob_fn = ens_kwargs.get("conditional_log_prob", None)
    use_conditional = conditional_log_prob_fn is not None and use_block_mh
    if use_conditional:
        # Block 0 is the "joint" block (potential), updated with full_log_prob.
        # Blocks 1..N are conditional blocks (streams), updated in parallel.
        n_conditional_blocks = n_blocks_val - 1
```

- [ ] **Step 2: Build the conditional block MH step function**

Add a new JIT function `block_conditional_step` alongside the existing `block_mh_step`. This function does 3 rounds instead of scanning over all blocks:

```python
        if use_conditional:
            @jax.jit
            def block_conditional_step(positions, log_probs, z_padded, z_count,
                                       mu_blocks_arg, key):
                # ---- Round 1: Update potential block (block 0) with full_log_prob ----
                pot_block_idx = jnp.int32(0)
                pot_bsize = block_sizes_arr[0]
                pot_mu = mu_blocks_arg[0]
                pot_b_idx = padded_blocks[0]
                pot_mask = (jnp.arange(max_block_size) < pot_bsize).astype(jnp.float64)

                def _mh_pot_walker(x_full, lp, wk):
                    wk, k1, k2, k_accept = jax.random.split(wk, 4)
                    idx1 = jax.random.randint(k1, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jax.random.randint(k2, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

                    z1_b = z_padded[idx1][pot_b_idx]
                    z2_b = z_padded[idx2][pot_b_idx]
                    diff = (z1_b - z2_b) * pot_mask
                    norm = jnp.sqrt(jnp.sum(diff**2))
                    dim_corr = jnp.sqrt(jnp.float64(pot_bsize) / 2.0)
                    mu_eff = pot_mu * norm / jnp.maximum(dim_corr, 1e-30)

                    d_block = diff / jnp.maximum(norm, 1e-30)
                    d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                    d_full = d_full.at[pot_b_idx].add(d_block * pot_mask)

                    x_prop = x_full + mu_eff * d_full
                    lp_prop = safe_log_prob(log_prob_fn, x_prop)
                    log_u = jnp.log(jax.random.uniform(k_accept, dtype=jnp.float64) + 1e-30)
                    accept = log_u < (lp_prop - lp)
                    x_new = jnp.where(accept, x_prop, x_full)
                    lp_new = jnp.where(accept, lp_prop, lp)
                    return x_new, lp_new, wk

                key, k_pot = jax.random.split(key)
                pot_keys = jax.random.split(k_pot, n_walkers)
                positions, log_probs, _ = jax.vmap(_mh_pot_walker)(
                    positions, log_probs, pot_keys)

                # ---- Round 2: Update all stream blocks in parallel ----
                # For each walker, propose for all n_conditional_blocks streams
                # Evaluate conditional_log_prob for each (walker, stream) pair

                cond_block_indices = jnp.arange(n_conditional_blocks, dtype=jnp.int32)
                # Conditional block sizes (blocks 1..N)
                cond_bsizes = block_sizes_arr[1:]
                cond_mus = mu_blocks_arg[1:]
                cond_b_idxs = padded_blocks[1:]  # (n_cond, max_block_size)

                def _mh_one_stream(x_full, lp, wk, stream_idx):
                    """MH update for one stream block of one walker."""
                    wk, k1, k2, k_accept = jax.random.split(wk, 4)
                    s_b_idx = cond_b_idxs[stream_idx]
                    s_bsize = cond_bsizes[stream_idx]
                    s_mu = cond_mus[stream_idx]
                    s_mask = (jnp.arange(max_block_size) < s_bsize).astype(jnp.float64)

                    idx1 = jax.random.randint(k1, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jax.random.randint(k2, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

                    z1_b = z_padded[idx1][s_b_idx]
                    z2_b = z_padded[idx2][s_b_idx]
                    diff = (z1_b - z2_b) * s_mask
                    norm = jnp.sqrt(jnp.sum(diff**2))
                    dim_corr = jnp.sqrt(jnp.float64(s_bsize) / 2.0)
                    mu_eff = s_mu * norm / jnp.maximum(dim_corr, 1e-30)

                    d_block = diff / jnp.maximum(norm, 1e-30)
                    d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                    d_full = d_full.at[s_b_idx].add(d_block * s_mask)

                    x_prop = x_full + mu_eff * d_full

                    # Conditional log_prob for MH ratio
                    cond_lp_old = conditional_log_prob_fn(x_full, stream_idx)
                    cond_lp_new = conditional_log_prob_fn(x_prop, stream_idx)

                    log_u = jnp.log(jax.random.uniform(k_accept, dtype=jnp.float64) + 1e-30)
                    accept = log_u < (cond_lp_new - cond_lp_old)
                    x_new = jnp.where(accept, x_prop, x_full)
                    return x_new, accept

                def _update_all_streams(x_full, lp, wk):
                    """Update all stream blocks for one walker."""
                    wk_streams = jax.random.split(wk, n_conditional_blocks)
                    # vmap over stream indices
                    x_proposals, accepts = jax.vmap(
                        _mh_one_stream, in_axes=(None, None, 0, 0)
                    )(x_full, lp, wk_streams, cond_block_indices)
                    # x_proposals: (n_cond, ndim) — each has its own block changed
                    # Merge: start from x_full, overwrite each accepted stream's block
                    x_merged = x_full
                    for s in range(n_conditional_blocks):
                        s_b_idx_s = cond_b_idxs[s]
                        s_mask_s = (jnp.arange(max_block_size) < cond_bsizes[s]).astype(jnp.float64)
                        # Extract the proposed block values
                        proposed_block = x_proposals[s][s_b_idx_s]
                        current_block = x_merged[s_b_idx_s]
                        new_block = jnp.where(accepts[s], proposed_block, current_block)
                        x_merged = x_merged.at[s_b_idx_s].set(
                            new_block * s_mask_s + x_merged[s_b_idx_s] * (1.0 - s_mask_s))
                    return x_merged

                key, k_streams = jax.random.split(key)
                stream_keys = jax.random.split(k_streams, n_walkers)
                positions = jax.vmap(_update_all_streams)(
                    positions, log_probs, stream_keys)

                # ---- Round 3: Re-evaluate full log_prob for consistency ----
                log_probs = jax.vmap(
                    lambda x: safe_log_prob(log_prob_fn, x))(positions)

                # Bracket ratio proxy (average acceptance)
                mean_br = jnp.float64(n_expand + 1)  # neutral — let ESJD tune
                return positions, log_probs, key, mean_br

            _block_step_fn = block_conditional_step
```

Note: The Python `for s in range(n_conditional_blocks)` loop in `_update_all_streams` is unrolled at trace time since `n_conditional_blocks` is a Python int constant (4). This is fine for small numbers of blocks.

- [ ] **Step 3: Run test to verify it passes**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_block_gibbs_enhanced.py::test_conditional_independence_variance -v -s`
Expected: PASS with var_ratio ≈ 1.0

- [ ] **Step 4: Commit**

```bash
git add dezess/core/loop.py
git commit -m "feat: add conditional independence parallelism to block-Gibbs MH"
```

---

### Task 7: Combined Test + Registry

**Files:**
- Modify: `dezess/tests/test_block_gibbs_enhanced.py`
- Modify: `dezess/benchmark/registry.py`

- [ ] **Step 1: Add combined test (all 3 enhancements)**

Append to `dezess/tests/test_block_gibbs_enhanced.py`:

```python
def test_all_enhancements_combined():
    """All 3 enhancements on block-coupled Gaussian: variance ≈ 1.0."""
    target, cond_lp = _make_block_coupled_conditional()
    key = jax.random.PRNGKey(42)
    init = target.sample(key, 32)

    config = VariantConfig(
        name="bg_mh_full",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14, 14, 14, 14],
            "use_mh": True,
            "conditional_log_prob": cond_lp,
            "delayed_rejection": True,
            "use_block_cov": True,
        },
    )

    result = run_variant(target.log_prob, init, n_steps=5000, config=config,
                         n_warmup=2000, verbose=False)
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, target.ndim)
    true_var = np.diag(np.array(target.cov))
    sample_var = np.var(flat, axis=0)
    var_ratio = float(np.mean(sample_var / true_var))

    assert 0.7 < var_ratio < 1.3, f"var_ratio={var_ratio:.4f}, expected ~1.0"
```

- [ ] **Step 2: Add variant configs to registry**

Append to the variants dict in `dezess/benchmark/registry.py`, before the closing `}`:

```python
    # --- Enhanced Block-Gibbs MH variants ---
    "block_gibbs_mh_dr": VariantConfig(
        name="block_gibbs_mh_dr",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14, 14, 14, 14],
            "use_mh": True,
            "delayed_rejection": True,
        },
    ),
    "block_gibbs_mh_cov": VariantConfig(
        name="block_gibbs_mh_cov",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14, 14, 14, 14],
            "use_mh": True,
            "use_block_cov": True,
        },
    ),
    "block_gibbs_mh_full": VariantConfig(
        name="block_gibbs_mh_full",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14, 14, 14, 14],
            "use_mh": True,
            "delayed_rejection": True,
            "use_block_cov": True,
        },
    ),
```

- [ ] **Step 3: Run all tests**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_block_gibbs_enhanced.py -v -s`
Expected: All 4 tests PASS

- [ ] **Step 4: Commit**

```bash
git add dezess/tests/test_block_gibbs_enhanced.py dezess/benchmark/registry.py
git commit -m "feat: add combined test and registry configs for enhanced block-Gibbs"
```

---

### Task 8: Benchmark on Funnel Targets

**Files:**
- Modify: `bench_nurs_vmap.py`

- [ ] **Step 1: Update benchmark to compare all enhanced variants**

Replace the `CONFIGS` dict and run parameters in `bench_nurs_vmap.py`:

```python
CONFIGS = {
    "bg_MH (5 ev)": VariantConfig(
        name="block_gibbs_mh",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True},
    ),
    "bg_MH+DR (10 ev)": VariantConfig(
        name="block_gibbs_mh_dr",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True,
                         "delayed_rejection": True},
    ),
    "bg_MH+cov (5 ev)": VariantConfig(
        name="block_gibbs_mh_cov",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True,
                         "use_block_cov": True},
    ),
    "bg_MH+DR+cov (10 ev)": VariantConfig(
        name="block_gibbs_mh_full",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True,
                         "delayed_rejection": True, "use_block_cov": True},
    ),
}
```

- [ ] **Step 2: Run benchmark**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro python bench_nurs_vmap.py`

Compare R-hat, ESS, lw_var, and wall time across variants.

- [ ] **Step 3: Commit benchmark**

```bash
git add bench_nurs_vmap.py
git commit -m "bench: compare enhanced block-Gibbs variants on funnel targets"
```
