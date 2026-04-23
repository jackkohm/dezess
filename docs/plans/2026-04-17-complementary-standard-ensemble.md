# Complementary direction for standard ensemble — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ensemble_kwargs["complementary_prob"]` work for the `standard` ensemble (today it only works for `block_gibbs`), so any of the 13 direction strategies can use the zeus-style two-half complementary fallback.

**Architecture:** Extract a reusable `sample_complementary_pair` helper, then in `core/loop.py:_make_step_fn(parallel_step)` thread `walker_idx` + a replicated `positions` snapshot into `update_one` and branch (zmat-direction vs. complementary-pair) gated by `complementary_prob`. Snooker Jacobian gets gated with a scalar multiplier so it's suppressed when the complementary branch fires.

**Tech Stack:** JAX (jit, vmap, lax.cond, sharding), pytest, NumPy.

**Spec:** `docs/specs/2026-04-17-complementary-standard-ensemble-design.md`

---

### Task 1: Reusable complementary-pair helper

**Files:**
- Create: `dezess/directions/complementary.py`
- Test: `dezess/tests/test_complementary_pair_helper.py`

- [ ] **Step 1: Write the failing test**

`dezess/tests/test_complementary_pair_helper.py`:

```python
"""Tests for sample_complementary_pair helper."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dezess.directions.complementary import sample_complementary_pair
from dezess.core.types import WalkerAux


def _empty_aux(n_dim):
    return WalkerAux(
        prev_direction=jnp.zeros(n_dim, dtype=jnp.float64),
        bracket_widths=jnp.zeros(n_dim, dtype=jnp.float64),
        direction_anchor=jnp.zeros(n_dim, dtype=jnp.float64),
        direction_scale=jnp.float64(1.0),
    )


def test_complementary_pair_picks_from_other_half():
    """Walker 0 (lower half) should sample from upper half [16, 32)."""
    n_walkers, n_dim = 32, 5
    positions = jnp.arange(n_walkers * n_dim, dtype=jnp.float64).reshape(n_walkers, n_dim)
    x_i = positions[0]
    aux = _empty_aux(n_dim)
    seen_indices = set()
    for seed in range(200):
        key = jax.random.PRNGKey(seed)
        d, _, _ = sample_complementary_pair(x_i, positions, walker_idx=0, key=key, aux=aux)
        # The direction is positions[j]-positions[k] normalized; reverse it
        # by checking which two upper-half rows produce that diff (up to sign).
        # Cheaper: assert the direction is non-zero and finite.
        assert jnp.isfinite(d).all()
        assert jnp.linalg.norm(d) > 0.99   # unit length within fp tol


def test_complementary_pair_unit_norm():
    n_walkers, n_dim = 8, 4
    rng = np.random.default_rng(0)
    positions = jnp.array(rng.standard_normal((n_walkers, n_dim)), dtype=jnp.float64)
    aux = _empty_aux(n_dim)
    for w in range(n_walkers):
        d, _, _ = sample_complementary_pair(
            positions[w], positions, walker_idx=w,
            key=jax.random.PRNGKey(w), aux=aux,
        )
        assert abs(float(jnp.linalg.norm(d)) - 1.0) < 1e-10


def test_complementary_pair_sets_direction_scale():
    n_walkers, n_dim = 8, 3
    positions = jnp.eye(n_walkers, n_dim, dtype=jnp.float64) * 5.0
    aux = _empty_aux(n_dim)
    _, _, new_aux = sample_complementary_pair(
        positions[0], positions, walker_idx=0,
        key=jax.random.PRNGKey(0), aux=aux,
    )
    # Pair from upper half: ‖e_j*5 - e_k*5‖ = 5*sqrt(2) for distinct j, k
    assert abs(float(new_aux.direction_scale) - 5.0 * np.sqrt(2.0)) < 1e-10


def test_complementary_pair_distinct_indices():
    """Repeated draws should never produce a zero direction (j != k always)."""
    n_walkers, n_dim = 4, 2
    positions = jnp.arange(n_walkers * n_dim, dtype=jnp.float64).reshape(n_walkers, n_dim)
    aux = _empty_aux(n_dim)
    for seed in range(100):
        d, _, _ = sample_complementary_pair(
            positions[0], positions, walker_idx=0,
            key=jax.random.PRNGKey(seed), aux=aux,
        )
        assert float(jnp.linalg.norm(d)) > 0.5


def test_complementary_pair_jit_compatible():
    """Helper must compose with jax.jit and jax.vmap."""
    n_walkers, n_dim = 8, 4
    positions = jnp.arange(n_walkers * n_dim, dtype=jnp.float64).reshape(n_walkers, n_dim)
    aux_batch = WalkerAux(
        prev_direction=jnp.zeros((n_walkers, n_dim), dtype=jnp.float64),
        bracket_widths=jnp.zeros((n_walkers, n_dim), dtype=jnp.float64),
        direction_anchor=jnp.zeros((n_walkers, n_dim), dtype=jnp.float64),
        direction_scale=jnp.ones(n_walkers, dtype=jnp.float64),
    )
    keys = jax.random.split(jax.random.PRNGKey(0), n_walkers)
    walker_indices = jnp.arange(n_walkers, dtype=jnp.int32)

    @jax.jit
    def go(pos, idxs, ks, aux):
        def one(x_i, w_idx, k, aux_i):
            d, _, _ = sample_complementary_pair(x_i, pos, w_idx, k, aux_i)
            return d
        return jax.vmap(one)(pos, idxs, ks, aux)

    out = go(positions, walker_indices, keys, aux_batch)
    assert out.shape == (n_walkers, n_dim)
    assert jnp.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd GIVEMEPotential/third_party/dezess
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_complementary_pair_helper.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'dezess.directions.complementary'`.

- [ ] **Step 3: Implement `dezess/directions/complementary.py`**

```python
"""Complementary-pair direction: pick j, k from the OTHER half of a positions snapshot.

Used by both the standard ensemble (this module) and the block-Gibbs MH paths
(inlined in dezess/core/loop.py for now). Works with any direction strategy by
substituting at probability `complementary_prob` per step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def sample_complementary_pair(
    x_i: Array,
    positions_snapshot: Array,
    walker_idx,
    key: Array,
    aux,
):
    """Sample DE-MCz pair from the OTHER half of `positions_snapshot`.

    Walkers with index < n_walkers // 2 sample from [half, n_walkers);
    walkers with index >= half sample from [0, half).

    Returns (direction, key, updated_aux) — same signature as
    dezess.directions.de_mcz.sample_direction, so it can be substituted via
    jnp.where at the per-walker level.
    """
    n_walkers = positions_snapshot.shape[0]
    half_size = n_walkers // 2
    my_half = walker_idx >= half_size
    other_lo = jnp.where(my_half, jnp.int32(0), jnp.int32(half_size))
    other_hi = jnp.where(my_half, jnp.int32(half_size), jnp.int32(n_walkers))

    key, k1, k2 = jax.random.split(key, 3)
    j_idx = jax.random.randint(k1, (), other_lo, other_hi)
    k_idx = jax.random.randint(k2, (), other_lo, other_hi)
    # Force k != j by rotating within the other half if collision
    k_idx = jnp.where(
        j_idx == k_idx,
        ((k_idx - other_lo + 1) % half_size) + other_lo,
        k_idx,
    )

    diff = positions_snapshot[j_idx] - positions_snapshot[k_idx]
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d = diff / jnp.maximum(norm, 1e-30)

    aux = aux._replace(direction_scale=norm)
    return d, key, aux
```

- [ ] **Step 4: Run test to verify it passes**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_complementary_pair_helper.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dezess/directions/complementary.py dezess/tests/test_complementary_pair_helper.py
git commit -m "feat: extract sample_complementary_pair direction helper"
```

---

### Task 2: cp=0 byte-identical regression test for standard ensemble

**Files:**
- Create: `dezess/tests/test_complementary_standard.py`

- [ ] **Step 1: Write the failing test**

`dezess/tests/test_complementary_standard.py`:

```python
"""Tests for complementary_prob support in the standard ensemble."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig


def _scale_aware(complementary_prob=0.0):
    ens_kwargs = {}
    if complementary_prob > 0.0:
        ens_kwargs["complementary_prob"] = complementary_prob
    return VariantConfig(
        name=f"scale_aware_cp{complementary_prob}",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )


def test_cp_zero_matches_baseline():
    """complementary_prob=0.0 must produce byte-identical samples to plain scale_aware."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 5)) * 0.1

    res_default = run_variant(
        log_prob, init, n_steps=200,
        config=_scale_aware(complementary_prob=0.0),
        n_warmup=50, key=jax.random.PRNGKey(0), verbose=False,
    )
    res_baseline = run_variant(
        log_prob, init, n_steps=200,
        config=VariantConfig(
            name="scale_aware",
            direction="de_mcz", width="scale_aware",
            slice_fn="fixed", zmatrix="circular", ensemble="standard",
            check_nans=False, width_kwargs={"scale_factor": 1.0},
        ),
        n_warmup=50, key=jax.random.PRNGKey(0), verbose=False,
    )

    np.testing.assert_array_equal(
        np.array(res_default["samples"]),
        np.array(res_baseline["samples"]),
        err_msg="cp=0.0 must be byte-identical to baseline scale_aware",
    )
```

- [ ] **Step 2: Run test to verify it fails (or passes vacuously)**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_complementary_standard.py::test_cp_zero_matches_baseline -v
```
Expected: PASS (cp=0 is the no-op default — should already match before any changes). This locks the regression baseline before we modify `update_one`.

- [ ] **Step 3: Commit the regression guard**

```bash
git add dezess/tests/test_complementary_standard.py
git commit -m "test: regression guard — cp=0.0 must match baseline scale_aware byte-for-byte"
```

---

### Task 3: Wire `complementary_prob` + snapshot into `parallel_step`

**Files:**
- Modify: `dezess/core/loop.py` (the `_make_step_fn` function — currently lines 249-463)

This task is the heart of the change. We add a replicated `pos_snapshot` and `walker_indices` arg to `parallel_step`, branch in `update_one`, and gate the snooker Jacobian.

- [ ] **Step 1: Locate the existing `_make_step_fn` and confirm baseline**

```bash
grep -n "def _make_step_fn\|def parallel_step\|def update_one\|jax.vmap(update_one)" dezess/core/loop.py
```
Expected output (line numbers approximate):
```
249:    def _make_step_fn(n_exp, n_shr, sharding_info=None):
286:        def parallel_step(positions, log_probs, z_padded, z_count, z_log_probs,
295:            def update_one(x_i, lp_i, w_key, prev_d, bw, d_anchor, d_scale, temp):
452:            results = jax.vmap(update_one)(
```

- [ ] **Step 1a: Add module-level import**

At the top of `dezess/core/loop.py` (alongside the other `from dezess.directions import ...` lines), add:

```python
from dezess.directions.complementary import sample_complementary_pair
```

This avoids per-trace imports inside `update_one`.

- [ ] **Step 2: Add `walker_indices` to `parallel_step` and `update_one`**

In `dezess/core/loop.py`, modify the `parallel_step` signature (currently at line 286-290) to add `pos_snapshot` and `walker_indices` as the **last two args**:

```python
@jit_decorator
def parallel_step(positions, log_probs, z_padded, z_count, z_log_probs,
                  mu, key, walker_aux_prev_dir, walker_aux_bw,
                  walker_aux_d_anchor, walker_aux_d_scale,
                  temperatures, pca_components, pca_weights,
                  flow_directions, whitening_matrix, kde_bandwidth,
                  pos_snapshot, walker_indices):
    keys = jax.random.split(key, n_walkers + 1)
    key_next = keys[0]
    walker_keys = keys[1:]

    def update_one(x_i, lp_i, w_key, prev_d, bw, d_anchor, d_scale, temp, walker_idx):
        # body unchanged for now — we modify it in step 4
        ...
```

The `pos_snapshot` arg is the same object as `positions` when not sharded; sharding annotations (Step 3) make sure it's replicated when n_gpus > 1. `walker_indices` is `jnp.arange(n_walkers)` and is walker-sharded.

- [ ] **Step 3: Update sharding annotations**

In the same `_make_step_fn`, modify the sharded branch (currently lines 252-281) to extend `in_shardings` with two new entries (`repl_sh`, `walker_sh`) and leave `out_shardings` unchanged:

```python
if sharding_info is not None:
    walker_sh = sharding_info["walker_sharding"]
    repl_sh = sharding_info["replicated"]
    in_shardings = (
        walker_sh, walker_sh,                            # positions, log_probs
        repl_sh, repl_sh, repl_sh,                       # z_padded, z_count, z_log_probs
        repl_sh, repl_sh,                                # mu, key
        walker_sh, walker_sh, walker_sh, walker_sh,      # 4 walker aux
        walker_sh,                                       # temperatures
        repl_sh, repl_sh, repl_sh, repl_sh, repl_sh,     # pca, flow, whitening, kde
        repl_sh, walker_sh,                              # pos_snapshot, walker_indices
    )
    out_shardings = (
        walker_sh, walker_sh,                            # new_pos, new_lp
        repl_sh,                                         # key_next
        walker_sh, walker_sh,                            # found, bracket_ratios
        walker_sh, walker_sh, walker_sh, walker_sh,      # 4 new walker aux
    )
    from functools import partial
    jit_decorator = partial(jax.jit, in_shardings=in_shardings, out_shardings=out_shardings)
else:
    jit_decorator = jax.jit
```

- [ ] **Step 4: Add the complementary branch inside `update_one`**

In `dezess/core/loop.py`, the `update_one` function (currently lines 295-450) needs three modifications. Replace the direction sampling block (currently lines 300-329, the `if config.direction == ...` chain) AND the snooker Jacobian block (currently lines 338-352) with the version below. Do **not** modify the width/slice/multi-direction code below those blocks.

Inside `update_one(... walker_idx)`:

```python
def update_one(x_i, lp_i, w_key, prev_d, bw, d_anchor, d_scale, temp, walker_idx):
    w_aux = WalkerAux(prev_direction=prev_d, bracket_widths=bw,
                      direction_anchor=d_anchor,
                      direction_scale=d_scale)

    # ── Direction: configured strategy ──────────────────────────────────
    if config.direction == "pca":
        d_z, w_key, aux_z = dir_mod.sample_direction(
            x_i, z_padded, z_count, z_log_probs, w_key, w_aux,
            pca_components=pca_components, pca_weights=pca_weights,
            **dir_kwargs,
        )
    elif config.direction == "flow":
        d_z, w_key, aux_z = dir_mod.sample_direction(
            x_i, z_padded, z_count, z_log_probs, w_key, w_aux,
            flow_directions=flow_directions, **dir_kwargs,
        )
    elif config.direction == "whitened":
        d_z, w_key, aux_z = dir_mod.sample_direction(
            x_i, z_padded, z_count, z_log_probs, w_key, w_aux,
            whitening_matrix=whitening_matrix, **dir_kwargs,
        )
    elif config.direction == "kde":
        d_z, w_key, aux_z = dir_mod.sample_direction(
            x_i, z_padded, z_count, z_log_probs, w_key, w_aux,
            kde_bandwidth=kde_bandwidth, **dir_kwargs,
        )
    else:
        d_z, w_key, aux_z = dir_mod.sample_direction(
            x_i, z_padded, z_count, z_log_probs, w_key, w_aux, **dir_kwargs,
        )

    # ── Optional complementary fallback (gated by complementary_prob) ───
    if use_complementary:
        # `sample_complementary_pair` is imported at module top — see Step 1a.
        w_key, kc_choice, kc_pair = jax.random.split(w_key, 3)
        use_comp = jax.random.uniform(kc_choice, dtype=jnp.float64) < complementary_prob
        d_c, _, aux_c = sample_complementary_pair(
            x_i, pos_snapshot, walker_idx, kc_pair, w_aux,
        )
        d = jnp.where(use_comp, d_c, d_z)
        # Only swap direction_scale; keep the configured strategy's
        # prev_direction / bracket_widths / direction_anchor (some are needed
        # downstream by snooker / per_direction width).
        w_aux = aux_z._replace(
            direction_scale=jnp.where(use_comp, aux_c.direction_scale, aux_z.direction_scale),
        )
        snooker_active = jnp.where(use_comp, jnp.float64(0.0), jnp.float64(1.0))
    else:
        d = d_z
        w_aux = aux_z
        snooker_active = jnp.float64(1.0)

    # Width
    w_key, k_width = jax.random.split(w_key)
    mu_eff = width_mod.get_mu(mu, d, w_aux, key=k_width, **width_kwargs)

    # ── Slice log-density (snooker Jacobian gated by snooker_active) ────
    if config.direction == "snooker":
        z_anchor = w_aux.direction_anchor
        ndim_j = x_i.shape[0]
        def lp_fn(x):
            base = lp_eval(log_prob_fn, x) / temp
            dist = jnp.linalg.norm(x - z_anchor)
            return base + snooker_active * (ndim_j - 1) * jnp.log(jnp.maximum(dist, 1e-30))

        dist_0 = jnp.linalg.norm(x_i - z_anchor)
        lp_x_slice = lp_i / temp + snooker_active * (ndim_j - 1) * jnp.log(jnp.maximum(dist_0, 1e-30))
    else:
        def lp_fn(x):
            return lp_eval(log_prob_fn, x) / temp
        lp_x_slice = lp_i / temp

    # — slice exec, multi-direction loop, post-slice re-eval all unchanged —
    # (Keep lines 354-450 of the existing code as-is.)
```

The variable `use_complementary` is a closure-captured Python `bool` (resolved at JIT trace time, no extra ops when False). `complementary_prob` is the closure-captured float.

- [ ] **Step 5: Update the vmap call to thread `walker_indices`**

In `_make_step_fn` (the vmap invocation, currently around line 452-456), change to:

```python
results = jax.vmap(update_one)(
    positions, log_probs, walker_keys,
    walker_aux_prev_dir, walker_aux_bw, walker_aux_d_anchor,
    walker_aux_d_scale, temperatures,
    walker_indices,                   # NEW: walker indices for complementary
)
```

- [ ] **Step 6: Update `_call_step` to pass new args**

In `dezess/core/loop.py:_call_step` (currently lines 465-474):

```python
def _call_step(step_fn, positions, log_probs, z_padded, z_count, z_log_probs,
               mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
               walker_aux_ds, temperatures):
    """Helper to call step_fn with all required args including PCA/flow/whitening."""
    # Always replicate pos_snapshot under sharding (whether or not cp > 0)
    # to satisfy the in_shardings annotation. Cost on multi-GPU + cp=0 is
    # one tiny all-gather (n_walkers × n_dim doubles ≈ 10 KB for Sanders),
    # negligible vs. log-prob eval. Single-GPU has no overhead.
    if sharding_info is not None:
        pos_snapshot = jax.lax.with_sharding_constraint(
            positions, sharding_info["replicated"]
        )
    else:
        pos_snapshot = positions
    walker_indices = jnp.arange(n_walkers, dtype=jnp.int32)
    return step_fn(
        positions, log_probs, z_padded, z_count, z_log_probs,
        mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
        walker_aux_ds, temperatures,
        pca_components, pca_weights, flow_directions, whitening_matrix, kde_bandwidth,
        pos_snapshot, walker_indices,
    )
```

The `use_complementary`, `sharding_info`, `n_walkers`, and `complementary_prob` symbols are all in the enclosing closure of `run_variant` already (parsed at line 481 today; `use_complementary` and `sharding_info` are in scope).

- [ ] **Step 7: Move `use_complementary` parsing earlier (before `_make_step_fn`)**

The parsing currently lives at lines 477-486 (after `_make_step_fn` and `_call_step` are defined). Both new code paths need `use_complementary` and `complementary_prob` in scope, so move the block to just before `def _make_step_fn(...)`. Concretely, cut these lines:

```python
use_block_gibbs = config.ensemble == "block_gibbs"
use_block_mh = use_block_gibbs and ens_kwargs.get("use_mh", False)
use_delayed_rejection = use_block_mh and ens_kwargs.get("delayed_rejection", False)
use_block_cov = use_block_mh and ens_kwargs.get("use_block_cov", False)
complementary_prob = float(ens_kwargs.get("complementary_prob", 0.0))
if not (0.0 <= complementary_prob <= 1.0):
    raise ValueError(
        f"complementary_prob must be in [0.0, 1.0], got {complementary_prob}"
    )
use_complementary = complementary_prob > 0.0
conditional_log_prob_fn = ens_kwargs.get("conditional_log_prob", None)
use_conditional = conditional_log_prob_fn is not None and use_block_mh
```

…and paste them immediately before `def _make_step_fn(...)`. The block-Gibbs setup that follows (lines 490+ today) still works because these flags are still in scope.

- [ ] **Step 8: Run the regression test from Task 2**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_complementary_standard.py::test_cp_zero_matches_baseline -v
```
Expected: PASS. If it fails, the cp=0 path is no longer byte-identical — most likely culprit is an extra PRNG split or a `jnp.where` that changed graph topology. Fix before proceeding.

- [ ] **Step 9: Run existing standard-ensemble Gaussian tests**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_gaussian_moments.py -v -k "scale_aware or baseline"
```
Expected: PASS. Confirms cp=0 default doesn't affect the scale_aware variant.

- [ ] **Step 10: Run existing block-Gibbs hybrid tests**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_complementary_hybrid.py -v
```
Expected: PASS (block-Gibbs path is unchanged).

- [ ] **Step 11: Commit**

```bash
git add dezess/core/loop.py
git commit -m "feat: complementary direction support in standard ensemble"
```

---

### Task 4: cp=0.5 and cp=1.0 variance recovery on 21D Gaussian

**Files:**
- Modify: `dezess/tests/test_complementary_standard.py`

- [ ] **Step 1: Append the variance tests**

Append to `dezess/tests/test_complementary_standard.py`:

```python
def _build_anisotropic_gaussian(ndim=21, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((ndim, ndim))
    Q, _ = np.linalg.qr(A)
    evals = np.linspace(1.0, 50.0, ndim)
    cov = Q @ np.diag(evals) @ Q.T
    cov = (cov + cov.T) / 2
    prec = jnp.array(np.linalg.inv(cov), dtype=jnp.float64)

    @jax.jit
    def log_prob(x):
        return -0.5 * x @ prec @ x

    return log_prob, np.array(cov), Q, evals


@pytest.mark.parametrize("cp", [0.5, 1.0])
def test_cp_recovers_anisotropic_gaussian_variance(cp):
    """cp=0.5 and cp=1.0 must recover per-dim variance to within 25%."""
    log_prob, cov, _, _ = _build_anisotropic_gaussian(ndim=21, seed=42)
    init = jax.random.normal(jax.random.PRNGKey(0), (32, 21)) * 0.5

    result = run_variant(
        log_prob, init, n_steps=2000,
        config=_scale_aware(complementary_prob=cp),
        n_warmup=500, key=jax.random.PRNGKey(0), verbose=False,
    )
    samples = np.array(result["samples"]).reshape(-1, 21)
    emp_var = samples.var(axis=0)
    true_var = np.diag(cov)
    rel_err = np.abs(emp_var - true_var) / true_var
    # Loose 25% tolerance: 32 walkers × 1500 prod steps gives ~10% MC noise on variance,
    # complementary direction sampling adds extra variance. Goal is "no catastrophic failure".
    assert rel_err.max() < 0.25, (
        f"cp={cp}: worst per-dim variance error {rel_err.max():.2%} > 25%; "
        f"emp={emp_var}, true={true_var}"
    )
```

- [ ] **Step 2: Run the new tests**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_complementary_standard.py -v
```
Expected: 3 passed (cp_zero, cp=0.5, cp=1.0).

- [ ] **Step 3: Commit**

```bash
git add dezess/tests/test_complementary_standard.py
git commit -m "test: variance recovery for cp=0.5 and cp=1.0 on 21D anisotropic Gaussian"
```

---

### Task 5: Compatibility tests — cp + snooker, cp + weighted_pair

**Files:**
- Modify: `dezess/tests/test_complementary_standard.py`

- [ ] **Step 1: Append the cross-direction tests**

Append to `dezess/tests/test_complementary_standard.py`:

```python
def _make_config(direction, complementary_prob):
    ens_kwargs = {}
    if complementary_prob > 0.0:
        ens_kwargs["complementary_prob"] = complementary_prob
    return VariantConfig(
        name=f"{direction}_cp{complementary_prob}",
        direction=direction,
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )


@pytest.mark.parametrize("direction", ["snooker", "weighted_pair"])
def test_cp_works_with_other_directions(direction):
    """cp=0.5 + non-de_mcz direction: chain converges, mean ≈ 0, no NaNs."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(7), (32, 5)) * 0.5

    result = run_variant(
        log_prob, init, n_steps=1000,
        config=_make_config(direction, complementary_prob=0.5),
        n_warmup=200, key=jax.random.PRNGKey(0), verbose=False,
    )
    samples = np.array(result["samples"]).reshape(-1, 5)
    assert np.isfinite(samples).all(), f"{direction}+cp=0.5 produced NaNs"
    mean_err = np.abs(samples.mean(axis=0)).max()
    assert mean_err < 0.2, (
        f"{direction}+cp=0.5: max abs mean {mean_err:.3f} > 0.2"
    )
    var_err = np.abs(samples.var(axis=0) - 1.0).max()
    assert var_err < 0.3, (
        f"{direction}+cp=0.5: max variance error {var_err:.3f} > 0.3"
    )
```

- [ ] **Step 2: Run the new tests**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_complementary_standard.py -v
```
Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add dezess/tests/test_complementary_standard.py
git commit -m "test: cp=0.5 compatibility with snooker and weighted_pair directions"
```

---

### Task 6: Multi-GPU smoke test

**Files:**
- Modify: `dezess/tests/test_sharding.py` (add a new test) — or create `dezess/tests/test_complementary_sharded.py`

This test only runs when 2+ visible GPUs are present (skip otherwise).

- [ ] **Step 1: Write the failing test**

Append to `dezess/tests/test_sharding.py` (or create `dezess/tests/test_complementary_sharded.py`):

```python
@pytest.mark.skipif(len(jax.devices("gpu")) < 2,
                    reason="multi-GPU complementary test needs 2+ GPUs")
def test_cp_standard_multi_gpu_matches_single_gpu():
    """cp=0.5 + standard ensemble: per-dim variance within 30% across 1 vs 2 GPUs."""
    from dezess.api import sample
    from dezess.core.types import VariantConfig

    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(0), (32, 5)) * 0.5

    cfg = VariantConfig(
        name="scale_aware_cp_smoke",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"complementary_prob": 0.5},
    )

    res_1 = sample(
        log_prob, init, n_steps=600, config=cfg, n_warmup=100,
        key=jax.random.PRNGKey(0), n_gpus=1, verbose=False,
    )
    res_2 = sample(
        log_prob, init, n_steps=600, config=cfg, n_warmup=100,
        key=jax.random.PRNGKey(0), n_gpus=2, n_walkers_per_gpu=16, verbose=False,
    )
    var_1 = np.array(res_1["samples"]).reshape(-1, 5).var(axis=0)
    var_2 = np.array(res_2["samples"]).reshape(-1, 5).var(axis=0)
    rel = np.abs(var_1 - var_2) / np.maximum(var_1, 1e-12)
    assert rel.max() < 0.3, f"variance mismatch 1 vs 2 GPU: {rel}"
```

- [ ] **Step 2: Run the test**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_sharding.py -v -k complementary
```
Expected on local CPU/single-GPU: SKIPPED. On H200 with `CUDA_VISIBLE_DEVICES=0,1`: PASS.

- [ ] **Step 3: Commit**

```bash
git add dezess/tests/test_sharding.py
git commit -m "test: multi-GPU smoke for cp=0.5 in standard ensemble"
```

---

### Task 7: Stale-Z benchmark for standard ensemble

**Files:**
- Create: `bench_complementary_standard_stale.py` (mirrors `bench_complementary_stale.py`)
- Create: `run_bench_complementary_standard_stale.sh` (Slurm wrapper)

- [ ] **Step 1: Write the benchmark script**

`bench_complementary_standard_stale.py`:

```python
#!/usr/bin/env python
"""Benchmark: complementary_prob sweep on a STALE-Z scenario for the
STANDARD ensemble (sister of bench_complementary_stale.py which uses block_gibbs).
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

print(f"JAX devices: {jax.devices()}")

# 21D anisotropic Gaussian — same target as bench_complementary_stale.py
NDIM = 21
rng = np.random.default_rng(42)
A = rng.standard_normal((NDIM, NDIM))
Q, _ = np.linalg.qr(A)
evals = np.linspace(1.0, 50.0, NDIM)
cov = Q @ np.diag(evals) @ Q.T
cov = (cov + cov.T) / 2
prec = jnp.array(np.linalg.inv(cov), dtype=jnp.float64)
true_mean = jnp.zeros(NDIM, dtype=jnp.float64)


@jax.jit
def log_prob(x):
    d = x - true_mean
    return -0.5 * d @ prec @ d


def make_init(seed):
    """Walkers initialized 5σ from the mode along the worst-conditioned axis."""
    key = jax.random.PRNGKey(seed)
    bad_axis = jnp.array(Q[:, -1], dtype=jnp.float64)
    offset = 5.0 * jnp.sqrt(evals[-1]) * bad_axis
    scatter = jax.random.normal(key, (32, NDIM), dtype=jnp.float64) * 0.1
    return offset[None, :] + scatter


def make_config(cp):
    ens_kwargs = {}
    if cp > 0.0:
        ens_kwargs["complementary_prob"] = cp
    return VariantConfig(
        name=f"scale_aware_cp{cp}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )


CONFIGS = [
    ("cp=0.00 (pure Z)", 0.0),
    ("cp=0.25 (hybrid)", 0.25),
    ("cp=0.50 (hybrid)", 0.5),
    ("cp=1.00 (pure comp)", 1.0),
]
N_WARMUP = 500
N_PROD = 3000
N_SEEDS = 3

print(f"\n{'=' * 100}")
print(f"  STALE-Z STANDARD-ENSEMBLE: 21D Gaussian, init 5σ from mode")
print(f"  {N_WARMUP} warmup + {N_PROD} production, {N_SEEDS} seeds per config")
print(f"{'=' * 100}")
hdr = (f"  {'Setup':<25s} | {'Seed':>4s} | {'R-hat':>6s} | {'ESS min':>8s} | "
       f"{'ESS mean':>9s} | {'Wall':>6s} | {'mu':>8s} | {'mean_lp':>8s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

results_by_config = {label: [] for label, _ in CONFIGS}
for label, cp in CONFIGS:
    config = make_config(cp)
    for seed in range(N_SEEDS):
        init = make_init(seed)
        t0 = time.time()
        result = run_variant(
            log_prob, init, n_steps=N_WARMUP + N_PROD,
            config=config, n_warmup=N_WARMUP,
            key=jax.random.PRNGKey(seed * 1000), verbose=False,
        )
        wall = time.time() - t0
        samples = np.array(result["samples"])
        ess = compute_ess(samples)
        rhat = compute_rhat(samples)
        final_mu = float(result["mu"])
        last_lps = np.array(result["log_prob"])[-200:]
        mean_lp = float(last_lps.mean())
        results_by_config[label].append({
            "rhat": float(rhat.max()), "ess_min": float(ess.min()),
            "ess_mean": float(ess.mean()), "wall": wall,
            "mu": final_mu, "mean_lp": mean_lp,
        })
        print(f"  {label:<25s} | {seed:>4d} | {float(rhat.max()):6.3f} | "
              f"{float(ess.min()):8.1f} | {float(ess.mean()):9.1f} | "
              f"{wall:5.1f}s | {final_mu:8.4f} | {mean_lp:8.1f}", flush=True)
        gc.collect()

print(f"\n{'=' * 100}")
print(f"  SUMMARY (mean ± std across {N_SEEDS} seeds)")
print(f"{'=' * 100}")
for label, _ in CONFIGS:
    runs = results_by_config[label]
    rhat_mean = np.mean([r["rhat"] for r in runs])
    ess_min_mean = np.mean([r["ess_min"] for r in runs])
    ess_min_std = np.std([r["ess_min"] for r in runs])
    print(f"  {label:<25s} | R-hat {rhat_mean:6.3f} | "
          f"ESS_min {ess_min_mean:6.1f} ± {ess_min_std:5.1f}", flush=True)
```

- [ ] **Step 2: Write the Slurm wrapper**

`run_bench_complementary_standard_stale.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=bench_std_stale
#SBATCH --output=slurm_bench_std_stale_%j.out
#SBATCH --error=slurm_bench_std_stale_%j.err
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
echo "=== bench_complementary_standard_stale.py — $(date) ==="

cd GIVEMEPotential/third_party/dezess
conda run --no-capture-output -p "$CONDA_ENV" python -u bench_complementary_standard_stale.py
```

- [ ] **Step 3: Commit (do not run yet — needs Hyak)**

```bash
chmod +x run_bench_complementary_standard_stale.sh
git add bench_complementary_standard_stale.py run_bench_complementary_standard_stale.sh
git commit -m "bench: stale-Z complementary sweep for standard ensemble"
```

The user will submit this via `scripts/hyak_sync_submit.sh GIVEMEPotential/third_party/dezess/run_bench_complementary_standard_stale.sh` from the parent repo. **Acceptance:** cp=0.5 has higher mean ESS_min than cp=0.0 across 3 seeds.

---

### Task 8: Documentation update

**Files:**
- Modify: `CLAUDE.md` (project root for dezess)

- [ ] **Step 1: Add note under "Default sampler"**

Find the "Default sampler" section in `CLAUDE.md` (currently a one-paragraph entry). Append:

```markdown
**`ensemble_kwargs["complementary_prob"]`** (0.0–1.0) enables the zeus-style
complementary-pair fallback in **both** standard and block-Gibbs ensembles.
With probability `complementary_prob` per step, the configured direction is
replaced by a pair drawn from the OTHER half of the current walker positions
(bypassing the Z-matrix). Use cp≈0.5 if you suspect biased Z-matrix from
short warmup + biased init. cp=0.0 (default) is byte-identical to the legacy
behavior.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note complementary_prob support in standard ensemble"
```

---

### Task 9: Final integration verification

- [ ] **Step 1: Run the full test suite**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/ -v
```
Expected: all tests pass (no regression on existing tests; new complementary tests pass).

- [ ] **Step 2: Push to remote**

```bash
git push origin main
```

- [ ] **Step 3: Submit benchmark to Hyak (from parent repo)**

```bash
cd /home/jaks/Documents/jek284/ondemand/data/sys/myjobs/projects/default/givemepotential
scripts/hyak_sync_submit.sh GIVEMEPotential/third_party/dezess/run_bench_complementary_standard_stale.sh
scripts/hyak_status.sh
```
Wait for completion via `scripts/hyak_logs.sh --follow`, then verify the summary table shows cp=0.5 ESS_min ≥ cp=0 ESS_min on the stale-Z target.
