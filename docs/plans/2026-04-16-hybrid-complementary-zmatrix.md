# Hybrid Complementary + Z-Matrix Direction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `complementary_prob` flag to bg_MH+DR that mixes Z-matrix DE directions with snapshot-based complementary-half DE directions, addressing the documented Z-matrix staleness problem.

**Architecture:** All changes localized to `block_mh_step` (and its 3 variants) in `loop.py`. When `complementary_prob > 0`, each walker has stochastic per-block choice between Z-matrix and complementary direction. Walker index passed via vmap. Default `0.0` is byte-identical to current bg_MH+DR.

**Tech Stack:** JAX, jax.vmap, jax.lax.with_sharding_constraint

---

### Task 1: Backwards-compat test (TDD)

**Files:**
- Create: `dezess/tests/test_complementary_hybrid.py`

- [ ] **Step 1: Write the test that proves complementary_prob=0.0 is identical to current behavior**

```python
"""Tests for hybrid complementary + Z-matrix direction source."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig

jax.config.update("jax_enable_x64", True)


def _make_bg_mh_dr_config(complementary_prob=0.0):
    """Helper: build a bg_MH+DR config with optional complementary_prob."""
    ens_kwargs = {
        "block_sizes": [7, 14],
        "use_mh": True,
        "delayed_rejection": True,
    }
    if complementary_prob > 0.0:
        ens_kwargs["complementary_prob"] = complementary_prob
    return VariantConfig(
        name=f"bg_mh_dr_cp{complementary_prob}",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )


def test_complementary_prob_zero_matches_current_bg_mh_dr():
    """complementary_prob=0.0 (or unset) produces identical samples to current bg_MH+DR."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 21)) * 0.1

    # Run 1: explicitly complementary_prob=0.0
    result_explicit = run_variant(
        log_prob, init, n_steps=300,
        config=_make_bg_mh_dr_config(complementary_prob=0.0),
        n_warmup=100, key=jax.random.PRNGKey(0), verbose=False,
    )

    # Run 2: complementary_prob unset (default = 0.0 = current behavior)
    result_default = run_variant(
        log_prob, init, n_steps=300,
        config=_make_bg_mh_dr_config(),  # no complementary_prob
        n_warmup=100, key=jax.random.PRNGKey(0), verbose=False,
    )

    np.testing.assert_array_equal(
        np.array(result_explicit["samples"]),
        np.array(result_default["samples"]),
        err_msg="complementary_prob=0.0 must be byte-identical to default bg_MH+DR",
    )
```

- [ ] **Step 2: Run test to verify it PASSES on current code (no changes yet)**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_complementary_hybrid.py::test_complementary_prob_zero_matches_current_bg_mh_dr -v -s`
Expected: PASS (current code ignores `complementary_prob` so both runs produce identical samples)

- [ ] **Step 3: Commit**

```bash
git add dezess/tests/test_complementary_hybrid.py
git commit -m "test: add backwards-compat test for complementary_prob=0.0"
```

---

### Task 2: Add `complementary_prob` parsing and pass walker_idx via vmap

**Files:**
- Modify: `dezess/core/loop.py:480` (add flag parsing alongside other ensemble flags)
- Modify: `dezess/core/loop.py:678,1147` (pass walker_idx into _mh_walker via vmap)

- [ ] **Step 1: Parse `complementary_prob` from ensemble_kwargs**

In `dezess/core/loop.py`, find the line `use_block_cov = use_block_mh and ens_kwargs.get("use_block_cov", False)` (around line 480). After this line, add:

```python
    use_block_cov = use_block_mh and ens_kwargs.get("use_block_cov", False)
    complementary_prob = float(ens_kwargs.get("complementary_prob", 0.0))
    if not (0.0 <= complementary_prob <= 1.0):
        raise ValueError(
            f"complementary_prob must be in [0.0, 1.0], got {complementary_prob}"
        )
    use_complementary = complementary_prob > 0.0
```

- [ ] **Step 2: Update vmap calls to pass walker_idx**

Find the vmap call `new_pos, new_lp, _, brs = jax.vmap(_mh_walker)(pos, lps, wkeys)` at line 678 (inside `block_mh_step`). Replace with:

```python
                k, k_walkers = jax.random.split(k)
                wkeys = jax.random.split(k_walkers, n_walkers)
                walker_indices = jnp.arange(n_walkers, dtype=jnp.int32)
                new_pos, new_lp, _, brs = jax.vmap(_mh_walker)(pos, lps, wkeys, walker_indices)
                mean_br = jnp.mean(brs)
                return (new_pos, new_lp, k), mean_br
```

Find the same pattern at line 1147 (inside `block_mh_cov_step`). Replace identically:

```python
                k, k_walkers = jax.random.split(k)
                wkeys = jax.random.split(k_walkers, n_walkers)
                walker_indices = jnp.arange(n_walkers, dtype=jnp.int32)
                new_pos, new_lp, _, brs = jax.vmap(_mh_walker)(pos, lps, wkeys, walker_indices)
                mean_br = jnp.mean(brs)
                return (new_pos, new_lp, k), mean_br
```

For `block_sweep_step` (slice-based) and `block_conditional_step`, also find their vmap calls and add `walker_indices`. Use grep to find them:

Run: `grep -n "vmap(_update_walker\|vmap(_mh_walker\|vmap(_mh_pot_walker\|vmap(_one_stream\|vmap(_walker" dezess/core/loop.py`

For each call, follow the same pattern: add `walker_indices = jnp.arange(n_walkers, dtype=jnp.int32)` on the line before, append `walker_indices` to the vmap args, and (in the next task) update the function signature to accept it.

- [ ] **Step 3: Update _mh_walker signatures to accept walker_idx**

For each `_mh_walker` (and equivalents like `_update_walker`, `_mh_pot_walker`, `_mh_one_stream`), change the signature from:
```python
def _mh_walker(x_full, lp, wk):
```
to:
```python
def _mh_walker(x_full, lp, wk, walker_idx):
```

The body doesn't need to use `walker_idx` yet (Task 3 will add usage). For now, just accept it. This is a pure interface change — behavior unchanged.

Use grep to find the function definitions:
Run: `grep -n "def _mh_walker\|def _update_walker\|def _mh_pot_walker\|def _mh_one_stream" dezess/core/loop.py`

For each one, add `walker_idx` to the parameter list.

- [ ] **Step 4: Run backwards-compat test**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_complementary_hybrid.py -v -s`
Expected: PASS — adding an unused param doesn't change behavior

- [ ] **Step 5: Run a quick smoke check on existing tests**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_block_gibbs_enhanced.py::test_delayed_rejection_variance -v --tb=short`
Expected: PASS — bg_MH+DR still works correctly

- [ ] **Step 6: Commit**

```bash
git add dezess/core/loop.py dezess/tests/test_complementary_hybrid.py
git commit -m "feat: parse complementary_prob and thread walker_idx through vmap"
```

---

### Task 3: Implement complementary direction logic in block_mh_step

**Files:**
- Modify: `dezess/core/loop.py:601-680` (the `block_mh_step` JIT function)

- [ ] **Step 1: Add convergence test for complementary_prob=0.5**

Append to `dezess/tests/test_complementary_hybrid.py`:

```python
def test_complementary_prob_half_recovers_gaussian_variance():
    """complementary_prob=0.5 on 21D Gaussian: variance approx 1.0."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 21)) * 0.1

    config = _make_bg_mh_dr_config(complementary_prob=0.5)
    result = run_variant(
        log_prob, init, n_steps=3000, config=config,
        n_warmup=1500, verbose=False,
    )
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 21)
    mean_var = float(np.var(flat, axis=0).mean())
    assert 0.7 < mean_var < 1.3, f"mean_var={mean_var:.4f}, expected ~1.0"


def test_complementary_prob_one_recovers_gaussian_variance():
    """complementary_prob=1.0 (pure complementary) on 21D Gaussian."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 21)) * 0.1

    config = _make_bg_mh_dr_config(complementary_prob=1.0)
    result = run_variant(
        log_prob, init, n_steps=3000, config=config,
        n_warmup=1500, verbose=False,
    )
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 21)
    mean_var = float(np.var(flat, axis=0).mean())
    assert 0.7 < mean_var < 1.3, f"mean_var={mean_var:.4f}, expected ~1.0"
```

- [ ] **Step 2: Run tests to verify they FAIL (no implementation yet)**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_complementary_hybrid.py -v -s 2>&1 | tail -10`
Expected: 1 PASS (backwards-compat), 2 FAIL (variance assertions fail because complementary_prob is ignored)

- [ ] **Step 3: Implement complementary direction in block_mh_step**

In `dezess/core/loop.py`, the `block_mh_step` function (around line 601) needs to:
1. Accept the `positions` snapshot from the OUTER scope (not just per-walker `x_full`). The full `positions` array is already in scope inside `_one_block_mh` as the `pos` argument.
2. Compute complementary direction inside `_mh_walker` when `use_complementary` is True.
3. Stochastically choose between Z-matrix and complementary directions per walker.

Find the `_mh_walker` function inside `block_mh_step` (lines ~612-674). The current body computes `diff` from Z-matrix indices `idx1`, `idx2`. We need to ALSO compute a complementary `diff_comp` from the snapshot, then select.

Replace the body of `_mh_walker` (the function defined at line ~612) with this new version. Key changes marked with NEW comments:

```python
                def _mh_walker(x_full, lp, wk, walker_idx):
                    wk, k1, k2, k_accept1 = jax.random.split(wk, 4)

                    # --- Z-matrix direction (always computed) ---
                    idx1 = jax.random.randint(k1, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jax.random.randint(k2, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

                    z1_block = z_padded[idx1, b_idx]
                    z2_block = z_padded[idx2, b_idx]
                    diff_zmat = (z1_block - z2_block) * mask
                    norm_zmat = jnp.linalg.norm(diff_zmat)

                    if use_complementary:
                        # --- NEW: Complementary direction from snapshot ---
                        wk, kc1, kc2, kc_choice = jax.random.split(wk, 4)
                        half_size = n_walkers // 2

                        # My half: 0 if walker_idx < half_size (half A), else 1 (half B)
                        my_half = walker_idx >= half_size
                        # Complementary half offset: if I'm in B, comp is at [0, half_size); if I'm in A, comp is at [half_size, n_walkers)
                        comp_offset = jnp.where(my_half, 0, half_size)

                        # Sample j, k from complementary half
                        j_off = jax.random.randint(kc1, (), 0, half_size)
                        k_off = jax.random.randint(kc2, (), 0, half_size)
                        # Ensure k != j (rotate if collision)
                        k_off = jnp.where(k_off == j_off, (k_off + 1) % half_size, k_off)
                        j_idx = comp_offset + j_off
                        k_idx = comp_offset + k_off

                        # Read from snapshot 'pos' (which is the block-start positions)
                        c1_block = pos[j_idx, b_idx]
                        c2_block = pos[k_idx, b_idx]
                        diff_comp = (c1_block - c2_block) * mask
                        norm_comp = jnp.linalg.norm(diff_comp)

                        # Stochastic choice
                        use_comp = jax.random.uniform(kc_choice, dtype=jnp.float64) < complementary_prob
                        diff = jnp.where(use_comp, diff_comp, diff_zmat)
                        norm = jnp.where(use_comp, norm_comp, norm_zmat)
                    else:
                        # Original: Z-matrix only
                        diff = diff_zmat
                        norm = norm_zmat

                    # Scale-aware step (unchanged from here on)
                    dim_corr = jnp.sqrt(jnp.float64(bsize) / 2.0)
                    mu_eff = mu_b * norm / jnp.maximum(dim_corr, 1e-30)

                    # Full-space direction
                    d_block = diff / jnp.maximum(norm, 1e-30)
                    d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                    d_full = d_full.at[b_idx].add(d_block * mask)

                    # Stage 1: full-scale proposal
                    x_prop1 = x_full + mu_eff * d_full
                    lp_prop1 = lp_eval(log_prob_fn, x_prop1)
                    log_u1 = jnp.log(jax.random.uniform(k_accept1, dtype=jnp.float64) + 1e-30)
                    accept1 = log_u1 < (lp_prop1 - lp)

                    if use_delayed_rejection:
                        # Stage 2: smaller step, new DE direction (always Z-matrix for stage 2)
                        wk, k3, k4, k_accept2 = jax.random.split(wk, 4)
                        idx3 = jax.random.randint(k3, (), 0, z_padded.shape[0]) % z_count
                        idx4 = jax.random.randint(k4, (), 0, z_padded.shape[0]) % z_count
                        idx4 = jnp.where(idx3 == idx4, (idx3 + 1) % z_count, idx4)

                        z3_block = z_padded[idx3, b_idx]
                        z4_block = z_padded[idx4, b_idx]
                        diff2 = (z3_block - z4_block) * mask
                        norm2 = jnp.linalg.norm(diff2)
                        mu_eff2 = mu_b * norm2 / jnp.maximum(dim_corr, 1e-30) / 3.0

                        d_block2 = diff2 / jnp.maximum(norm2, 1e-30)
                        d_full2 = jnp.zeros(n_dim, dtype=jnp.float64)
                        d_full2 = d_full2.at[b_idx].add(d_block2 * mask)

                        x_prop2 = x_full + mu_eff2 * d_full2
                        lp_prop2 = lp_eval(log_prob_fn, x_prop2)
                        log_u2 = jnp.log(jax.random.uniform(k_accept2, dtype=jnp.float64) + 1e-30)
                        accept2 = log_u2 < (lp_prop2 - lp)

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

The key invariants:
- Snapshot `pos` is read by walkers across the vmap — JAX broadcasts it (same array seen by every vmapped walker)
- For multi-GPU: `pos` is walker-sharded. We need ALL walkers to see ALL positions. Add a `with_sharding_constraint` to replicate `pos` before passing it into `_one_block_mh`. See Step 4.

- [ ] **Step 4: Replicate snapshot for multi-GPU**

Find the start of `_one_block_mh` inside `block_mh_step` (around line 606):

```python
            def _one_block_mh(carry, block_data):
                pos, lps, k = carry
                block_idx, bsize, mu_b = block_data
                b_idx = padded_blocks[block_idx]
                mask = (jnp.arange(max_block_size) < bsize).astype(jnp.float64)
```

Right after `mask = ...`, add the snapshot replication for multi-GPU:

```python
                # Replicate snapshot for complementary direction reads (multi-GPU only)
                if use_complementary and sharding_info is not None:
                    pos = jax.lax.with_sharding_constraint(
                        pos, sharding_info["replicated"]
                    )
```

This makes `pos` (the snapshot) available on every GPU so any walker can read any other walker's position. Single-GPU runs skip this (no-op).

- [ ] **Step 5: Run tests**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_complementary_hybrid.py -v -s`
Expected: All 3 tests PASS (backwards-compat + 2 variance tests)

- [ ] **Step 6: Run existing bg_MH+DR test to confirm no regression**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_block_gibbs_enhanced.py::test_delayed_rejection_variance -v --tb=short`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add dezess/core/loop.py dezess/tests/test_complementary_hybrid.py
git commit -m "feat: complementary direction in block_mh_step with snapshot"
```

---

### Task 4: Apply same change to block_mh_cov_step

**Files:**
- Modify: `dezess/core/loop.py` — find `block_mh_cov_step` and apply identical pattern

The `block_mh_cov_step` is structurally identical to `block_mh_step` but uses Cholesky-decomposed Gaussian proposals from precomputed block covariances instead of DE directions. The complementary direction option still applies — when chosen, use complementary DE pair from snapshot instead of cov-Cholesky.

- [ ] **Step 1: Find block_mh_cov_step**

Run: `grep -n "def block_mh_cov_step" dezess/core/loop.py`

It's defined inside `if use_block_cov and n_warmup > 0:` (around line 1071).

- [ ] **Step 2: Apply the same pattern**

Read the current `block_mh_cov_step` (similar structure to `block_mh_step`, ~150 lines). Find its `_mh_walker` (inside `_one_block_mh`). The current logic uses `block_L_padded[block_idx]` for the Cholesky proposal:

```python
def _mh_walker(x_full, lp, wk, walker_idx):
    wk, k1, k2, k_accept1 = jax.random.split(wk, 4)

    # Covariance-adapted proposal: L @ z
    L_b = block_L_padded[block_idx]
    z_rand = jax.random.normal(k1, (max_block_size,), dtype=jnp.float64)
    z_rand = z_rand * mask
    delta_cov = L_b @ z_rand
    delta_cov = delta_cov * mask
    norm_cov = jnp.linalg.norm(delta_cov)
    diff_cov = delta_cov  # alias for clarity

    if use_complementary:
        # Complementary direction from snapshot (same as block_mh_step pattern)
        wk, kc1, kc2, kc_choice = jax.random.split(wk, 4)
        half_size = n_walkers // 2
        my_half = walker_idx >= half_size
        comp_offset = jnp.where(my_half, 0, half_size)
        j_off = jax.random.randint(kc1, (), 0, half_size)
        k_off = jax.random.randint(kc2, (), 0, half_size)
        k_off = jnp.where(k_off == j_off, (k_off + 1) % half_size, k_off)
        j_idx = comp_offset + j_off
        k_idx = comp_offset + k_off

        c1_block = pos[j_idx, b_idx]
        c2_block = pos[k_idx, b_idx]
        diff_comp = (c1_block - c2_block) * mask
        norm_comp = jnp.linalg.norm(diff_comp)

        use_comp = jax.random.uniform(kc_choice, dtype=jnp.float64) < complementary_prob
        diff = jnp.where(use_comp, diff_comp, diff_cov)
        norm = jnp.where(use_comp, norm_comp, norm_cov)
    else:
        diff = diff_cov
        norm = norm_cov

    # ... rest of the function unchanged (scale-aware step, MH+DR) ...
```

Also add the snapshot replication at the top of `_one_block_mh` in this function (same as Task 3 Step 4).

- [ ] **Step 3: Run all complementary tests + bg_mh_cov tests**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_complementary_hybrid.py dezess/tests/test_block_gibbs_enhanced.py -v --tb=short 2>&1 | tail -15`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add dezess/core/loop.py
git commit -m "feat: complementary direction in block_mh_cov_step"
```

---

### Task 5: Apply to block_sweep_step (slice-based) and block_conditional_step

**Files:**
- Modify: `dezess/core/loop.py` — `block_sweep_step` and `block_conditional_step`

These two also have `_mh_walker` (or equivalent) functions. The pattern is the same.

- [ ] **Step 1: Find both functions**

Run: `grep -n "def block_sweep_step\|def block_conditional_step" dezess/core/loop.py`

`block_sweep_step` is the slice-based block-Gibbs (uses `slice_sample_fixed` instead of MH). The walker function inside is `_update_walker`.

`block_conditional_step` is for conditional independence parallelism. Its walker functions are `_mh_pot_walker` and `_mh_one_stream`.

- [ ] **Step 2: Update each walker function with complementary logic**

For `block_sweep_step._update_walker`:
- Apply same pattern: compute Z-matrix `diff_zmat`, compute complementary `diff_comp` if `use_complementary`, select via `jnp.where`
- The proposal uses `slice_sample_fixed` after computing the direction. Direction selection is BEFORE the slice call, so the integration is the same.

For `block_conditional_step._mh_pot_walker` and `_mh_one_stream`:
- Same pattern. Both are MH-based like `_mh_walker`.
- For `_mh_one_stream`, the complementary half logic uses the same walker_idx-based half assignment

In both functions, also update the vmap call sites to pass `walker_indices` (already done in Task 2 Step 2). Verify that's been applied to all vmap call sites.

Also add the snapshot replication at the top of each `_one_block` (or equivalent) function (same `with_sharding_constraint` pattern).

- [ ] **Step 3: Run all tests**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_complementary_hybrid.py dezess/tests/test_block_gibbs_enhanced.py -v --tb=short 2>&1 | tail -15`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add dezess/core/loop.py
git commit -m "feat: complementary direction in block_sweep_step and block_conditional_step"
```

---

### Task 6: Stale-Z stress test + benchmark on H200

**Files:**
- Modify: `dezess/tests/test_complementary_hybrid.py` (add stale-Z test)
- Create: `bench_complementary.py` (H200 benchmark)
- Create: `run_bench_complementary.sh` (Slurm script)

- [ ] **Step 1: Add stale-Z stress test**

Append to `dezess/tests/test_complementary_hybrid.py`:

```python
def test_complementary_helps_with_biased_warmup():
    """When warmup produces a biased Z-matrix, complementary_prob>0 should help."""
    # Use a 10D correlated Gaussian. Initialize walkers far from the mean
    # so warmup builds a biased Z-matrix.
    ndim = 10
    rng = np.random.default_rng(42)
    A = rng.standard_normal((ndim, ndim))
    Q, _ = np.linalg.qr(A)
    evals = np.linspace(1.0, 10.0, ndim)
    cov = Q @ np.diag(evals) @ Q.T
    cov = (cov + cov.T) / 2
    prec = jnp.array(np.linalg.inv(cov), dtype=jnp.float64)

    @jax.jit
    def log_prob(x):
        return -0.5 * x @ prec @ x

    # Initialize FAR from the mean to bias the Z-matrix toward climb history
    init = jax.random.normal(jax.random.PRNGKey(42), (32, ndim)) * 0.01 + 5.0

    # Pure Z-matrix (current bg_MH+DR)
    config_zmat = _make_bg_mh_dr_config(complementary_prob=0.0)
    config_zmat = config_zmat._replace(
        ensemble_kwargs={**config_zmat.ensemble_kwargs,
                         "block_sizes": [5, 5]}
    )
    result_zmat = run_variant(
        log_prob, init, n_steps=2000, config=config_zmat,
        n_warmup=500, key=jax.random.PRNGKey(0), verbose=False,
    )

    # Hybrid (50/50)
    config_hybrid = _make_bg_mh_dr_config(complementary_prob=0.5)
    config_hybrid = config_hybrid._replace(
        ensemble_kwargs={**config_hybrid.ensemble_kwargs,
                         "block_sizes": [5, 5]}
    )
    result_hybrid = run_variant(
        log_prob, init, n_steps=2000, config=config_hybrid,
        n_warmup=500, key=jax.random.PRNGKey(0), verbose=False,
    )

    samples_zmat = np.array(result_zmat["samples"]).reshape(-1, ndim)
    samples_hybrid = np.array(result_hybrid["samples"]).reshape(-1, ndim)

    # Both should at least move from init region (>1 unit from x=5)
    final_zmat_mean = np.linalg.norm(samples_zmat[-100:].mean(axis=0))
    final_hybrid_mean = np.linalg.norm(samples_hybrid[-100:].mean(axis=0))

    # Hybrid should converge closer to true mean (0)
    assert final_hybrid_mean < final_zmat_mean + 1.0, \
        f"hybrid={final_hybrid_mean:.2f}, zmat={final_zmat_mean:.2f}"
```

This test isn't a strict assertion of "hybrid is faster" (which would be flaky), just that hybrid doesn't make things worse. Real validation comes from the H200 benchmark.

- [ ] **Step 2: Run stale-Z test**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_complementary_hybrid.py::test_complementary_helps_with_biased_warmup -v -s`
Expected: PASS

- [ ] **Step 3: Create H200 benchmark**

Create `bench_complementary.py`:

```python
#!/usr/bin/env python
"""Benchmark: pure Z-matrix vs hybrid (Z+complementary) for bg_MH+DR.

Compares ESS_min and convergence on a biased-init scenario that produces
stale Z-matrices.
"""
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
from dezess.targets_stream import block_coupled_gaussian

print(f"JAX devices: {jax.devices()}")

target = block_coupled_gaussian()
NDIM = target.ndim
N_WARMUP = 2000
N_PROD = 5000


def make_config(complementary_prob):
    return VariantConfig(
        name=f"bg_mh_dr_cp{complementary_prob}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14, 14, 14, 14],
            "use_mh": True,
            "delayed_rejection": True,
            "complementary_prob": complementary_prob,
        },
    )


configs = [
    ("Pure Z-matrix (cp=0.0)", 0.0),
    ("Hybrid 25/75 (cp=0.25)", 0.25),
    ("Hybrid 50/50 (cp=0.5)", 0.5),
    ("Hybrid 75/25 (cp=0.75)", 0.75),
    ("Pure complementary (cp=1.0)", 1.0),
]

key = jax.random.PRNGKey(42)
init = target.sample(key, 64)

hdr = (f"  {'Setup':<35s} | {'R-hat':>6s} | {'ESS min':>8s} | "
       f"{'ESS mean':>9s} | {'Wall':>6s} | {'ESS/sec':>8s}")
print(f"\n{hdr}")
print("  " + "-" * (len(hdr) - 2))

for label, cp in configs:
    config = make_config(cp)
    t0 = time.time()
    result = run_variant(
        target.log_prob, init, n_steps=N_WARMUP + N_PROD,
        config=config, n_warmup=N_WARMUP, verbose=False,
    )
    wall = time.time() - t0
    samples = np.array(result["samples"])
    ess = compute_ess(samples)
    rhat = compute_rhat(samples)
    print(f"  {label:<35s} | {float(rhat.max()):6.3f} | "
          f"{float(ess.min()):8.1f} | {float(ess.mean()):9.1f} | "
          f"{wall:5.1f}s | {float(ess.min())/wall:8.2f}",
          flush=True)
    gc.collect()
```

- [ ] **Step 4: Create Slurm wrapper**

Create `run_bench_complementary.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=bench_comp
#SBATCH --output=slurm_bench_comp_%j.out
#SBATCH --error=slurm_bench_comp_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
module load conda/Miniforge3-25.3.1-3
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/gpfs/scrubbed/jackkohm/.jax_cache
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
else
  cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi

CONDA_ENV="/gpfs/scrubbed/jackkohm/conda-envs/Astro"
echo "=== bench_complementary.py — $(date) ==="

cd GIVEMEPotential/third_party/dezess
conda run --no-capture-output -p "$CONDA_ENV" python -u bench_complementary.py
```

- [ ] **Step 5: Submit benchmark to H200**

Run: `cd /home/jaks/Documents/jek284/ondemand/data/sys/myjobs/projects/default/givemepotential && scripts/hyak_sync_submit.sh GIVEMEPotential/third_party/dezess/run_bench_complementary.sh`
Note the JOB_ID printed.

- [ ] **Step 6: Wait for results, pull and report**

Run `scripts/hyak_status.sh <JOB_ID>` periodically until COMPLETED.
Then: `scripts/hyak_pull.sh "slurm_bench_comp_<JOB_ID>.out"` and `cat slurm_bench_comp_<JOB_ID>.out`.

Compare ESS_min across `complementary_prob` values 0.0, 0.25, 0.5, 0.75, 1.0.

- [ ] **Step 7: Commit benchmark scripts**

```bash
git add bench_complementary.py run_bench_complementary.sh dezess/tests/test_complementary_hybrid.py
git commit -m "bench: complementary direction sweep on H200"
```
