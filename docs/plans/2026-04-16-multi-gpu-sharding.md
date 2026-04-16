# Multi-GPU Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-GPU sharding to dezess via `jax.sharding`, with replicated Z-matrix and K=100 sample batching, while preserving zero-overhead single-GPU fallback.

**Architecture:** New `dezess/core/sharding.py` provides Mesh/sharding setup. `loop.py` accepts `n_gpus` and `n_walkers_per_gpu` parameters; when `n_gpus > 1`, the JIT step functions are recompiled with sharding annotations and the production scan gathers samples to host every K=100 steps. The Z-matrix is replicated across all GPUs.

**Tech Stack:** JAX, jax.sharding (Mesh, NamedSharding, PartitionSpec), jax.lax.with_sharding_constraint

---

### Task 1: Sharding utility module — TDD

**Files:**
- Create: `dezess/core/sharding.py`
- Create: `dezess/tests/test_sharding.py`

- [ ] **Step 1: Write the failing test for `setup_sharding`**

```python
"""Tests for multi-GPU sharding support."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_setup_sharding_single_gpu():
    """n_gpus=1 returns None mesh — no sharding."""
    from dezess.core.sharding import setup_sharding

    result = setup_sharding(n_gpus=1, n_walkers_total=64)
    assert result is None, "Single-GPU should return None (no sharding)"


def test_setup_sharding_multi_gpu_validates_walker_count():
    """Multi-GPU raises if walker count doesn't divide n_gpus evenly."""
    from dezess.core.sharding import setup_sharding

    if len(jax.devices()) < 2:
        pytest.skip("Need >= 2 devices")

    with pytest.raises(ValueError, match="must be divisible"):
        setup_sharding(n_gpus=2, n_walkers_total=63)  # 63 not divisible by 2


def test_setup_sharding_multi_gpu_returns_shardings():
    """Multi-GPU returns mesh + walker_sharding + replicated_sharding."""
    from dezess.core.sharding import setup_sharding

    if len(jax.devices()) < 2:
        pytest.skip("Need >= 2 devices")

    result = setup_sharding(n_gpus=2, n_walkers_total=64)
    assert result is not None
    assert "mesh" in result
    assert "walker_sharding" in result
    assert "replicated" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_setup_sharding_single_gpu -v -s`
Expected: FAIL — `dezess.core.sharding` module doesn't exist

- [ ] **Step 3: Create `dezess/core/sharding.py`**

```python
"""Multi-GPU sharding utilities for dezess.

Provides Mesh setup and sharding specs for distributing walkers across
multiple GPUs while keeping the Z-matrix and tuning state replicated.

When n_gpus == 1, all helpers return None and the caller falls back to
the single-GPU code path (zero overhead).
"""

from __future__ import annotations

from typing import Optional

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def setup_sharding(
    n_gpus: int,
    n_walkers_total: int,
) -> Optional[dict]:
    """Build sharding specs for multi-GPU runs.

    Parameters
    ----------
    n_gpus : int
        Number of GPUs to use. If 1, returns None.
    n_walkers_total : int
        Total walker count across all GPUs. Must be divisible by n_gpus.

    Returns
    -------
    None if n_gpus == 1.
    Otherwise dict with:
        - "mesh": jax.sharding.Mesh
        - "walker_sharding": NamedSharding sharding the first axis
        - "replicated": NamedSharding replicating across all GPUs
    """
    if n_gpus == 1:
        return None

    if n_walkers_total % n_gpus != 0:
        raise ValueError(
            f"n_walkers_total ({n_walkers_total}) must be divisible by "
            f"n_gpus ({n_gpus})"
        )

    devices = jax.devices()[:n_gpus]
    if len(devices) < n_gpus:
        raise ValueError(
            f"Requested n_gpus={n_gpus} but only {len(devices)} devices available"
        )

    mesh = Mesh(devices, ('walkers',))
    walker_sharding = NamedSharding(mesh, P('walkers'))
    replicated = NamedSharding(mesh, P())

    return {
        "mesh": mesh,
        "walker_sharding": walker_sharding,
        "replicated": replicated,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py -v -s`
Expected: All 3 tests PASS (multi-GPU tests skip on single-GPU machine)

- [ ] **Step 5: Commit**

```bash
git add dezess/core/sharding.py dezess/tests/test_sharding.py
git commit -m "feat: add Mesh setup utility for multi-GPU sharding"
```

---

### Task 2: Add `n_gpus` and `n_walkers_per_gpu` to run_variant signature

**Files:**
- Modify: `dezess/core/loop.py:106-135` (function signature + initial validation)
- Modify: `dezess/tests/test_sharding.py` (add backwards-compat test)

- [ ] **Step 1: Add backwards-compatibility test**

Append to `dezess/tests/test_sharding.py`:

```python
def test_run_variant_default_single_gpu_unchanged():
    """run_variant with defaults (no n_gpus) should behave identically to before."""
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig

    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 5)) * 0.1

    config = VariantConfig(
        name="single_gpu",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
    )

    # Call without n_gpus param (default = 1)
    result = run_variant(log_prob, init, n_steps=500, config=config,
                         n_warmup=200, verbose=False)
    samples = np.array(result["samples"])
    assert samples.shape == (300, 32, 5)
```

- [ ] **Step 2: Run test to verify behavior is unchanged before any changes**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_run_variant_default_single_gpu_unchanged -v -s`
Expected: PASS (current code, no changes yet)

- [ ] **Step 3: Add params to run_variant in loop.py**

In `dezess/core/loop.py`, modify the `run_variant` function signature (around line 106-121). Add `n_gpus` and `n_walkers_per_gpu` after `transform`:

```python
def run_variant(
    log_prob_fn: Callable[[Array], Array],
    init_positions: Array,
    n_steps: int,
    config: Optional[VariantConfig] = None,
    n_warmup: int = 0,
    key: Optional[Array] = None,
    mu: float = 1.0,
    tune: bool = True,
    z_max_size: int = 50000,
    z_initial: Optional[Array] = None,
    target_ess: Optional[float] = None,
    progress_fn: Optional[Callable] = None,
    verbose: bool = True,
    transform: Optional[Transform] = None,
    n_gpus: int = 1,                    # NEW
    n_walkers_per_gpu: Optional[int] = None,  # NEW
) -> dict:
```

After the line `n_walkers, n_dim = init_positions.shape` (around line 135), add validation:

```python
    n_walkers, n_dim = init_positions.shape

    # --- Multi-GPU sharding setup ---
    from dezess.core.sharding import setup_sharding
    if n_gpus > 1:
        if n_walkers_per_gpu is None:
            n_walkers_per_gpu = n_walkers // n_gpus
        expected = n_gpus * n_walkers_per_gpu
        if n_walkers != expected:
            raise ValueError(
                f"init_positions has {n_walkers} walkers but n_gpus={n_gpus} "
                f"× n_walkers_per_gpu={n_walkers_per_gpu} = {expected} expected"
            )
    sharding_info = setup_sharding(n_gpus, n_walkers)
```

- [ ] **Step 4: Run backwards-compat test again**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_run_variant_default_single_gpu_unchanged -v -s`
Expected: PASS — single-GPU path is unchanged

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -v --tb=no 2>&1 | tail -3`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add dezess/core/loop.py dezess/tests/test_sharding.py
git commit -m "feat: add n_gpus and n_walkers_per_gpu params to run_variant"
```

---

### Task 3: Shard initial positions onto mesh

**Files:**
- Modify: `dezess/core/loop.py` (after sharding setup, around line 145)

- [ ] **Step 1: Add shard-init test**

Append to `dezess/tests/test_sharding.py`:

```python
def test_init_positions_sharded_onto_mesh():
    """When n_gpus > 1, init_positions get distributed onto the mesh."""
    from dezess.core.sharding import setup_sharding

    if len(jax.devices()) < 2:
        pytest.skip("Need >= 2 devices")

    sharding_info = setup_sharding(n_gpus=2, n_walkers_total=64)
    init = jax.random.normal(jax.random.PRNGKey(42), (64, 5))

    sharded = jax.device_put(init, sharding_info["walker_sharding"])
    assert sharded.sharding == sharding_info["walker_sharding"]
    # Each device should hold half the walkers
    addressable = sharded.addressable_data(0).shape
    assert addressable[0] == 32  # 64 / 2 = 32 per device
```

- [ ] **Step 2: Run test (skipped on single-GPU)**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_init_positions_sharded_onto_mesh -v -s`
Expected: SKIP (single-GPU machine) or PASS (multi-GPU)

- [ ] **Step 3: Add init sharding to run_variant**

In `dezess/core/loop.py`, find the line `positions = jnp.array(init_positions, dtype=jnp.float64)` (around line 605 — inside the `# --- Initial log-probs ---` section). Replace with:

```python
    positions = jnp.array(init_positions, dtype=jnp.float64)
    if sharding_info is not None:
        positions = jax.device_put(positions, sharding_info["walker_sharding"])
    log_probs = jax.jit(jax.vmap(lambda x: lp_eval(log_prob_fn, x)))(positions)
    log_probs.block_until_ready()
```

The `lp_eval` vmap automatically inherits the walker sharding from `positions` — JAX figures it out.

- [ ] **Step 4: Run backwards-compat test again**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_run_variant_default_single_gpu_unchanged -v -s`
Expected: PASS — single-GPU path still works (sharding_info is None, no device_put)

- [ ] **Step 5: Commit**

```bash
git add dezess/core/loop.py dezess/tests/test_sharding.py
git commit -m "feat: shard initial positions onto mesh when n_gpus > 1"
```

---

### Task 4: Sharded JIT for the standard step function

**Files:**
- Modify: `dezess/core/loop.py:236-405` (parallel_step JIT)

This is the core change. The `parallel_step` JIT function needs to declare shardings when `n_gpus > 1`.

- [ ] **Step 1: Add the sharded-step end-to-end test**

Append to `dezess/tests/test_sharding.py`:

```python
def test_run_variant_n_gpus_2_recovers_gaussian_variance():
    """End-to-end: 2-GPU run on 10D Gaussian recovers variance ~ 1.0."""
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig

    if len(jax.devices()) < 2:
        pytest.skip("Need >= 2 devices")

    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (64, 10)) * 0.1

    config = VariantConfig(
        name="multi_gpu_test",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
    )

    result = run_variant(log_prob, init, n_steps=2000, config=config,
                         n_warmup=1000, verbose=False,
                         n_gpus=2, n_walkers_per_gpu=32)
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 10)
    mean_var = float(np.var(flat, axis=0).mean())
    assert 0.7 < mean_var < 1.3, f"mean_var={mean_var:.4f}, expected ~1.0"
```

- [ ] **Step 2: Run test (skipped on single-GPU)**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_run_variant_n_gpus_2_recovers_gaussian_variance -v -s`
Expected: SKIP locally (single-GPU). On H200 nodes with multiple GPUs, will FAIL until Task 4 is implemented.

- [ ] **Step 3: Modify `_make_step_fn` to accept sharding info**

In `dezess/core/loop.py`, find `def _make_step_fn(n_exp, n_shr):` (around line 232). Modify it to optionally take sharding specs:

```python
    def _make_step_fn(n_exp, n_shr, sharding_info=None):
        """Create a JIT-compiled step function with given budget."""

        if sharding_info is not None:
            walker_sh = sharding_info["walker_sharding"]
            repl_sh = sharding_info["replicated"]
            in_shardings = (
                walker_sh, walker_sh, repl_sh, repl_sh, repl_sh,  # positions, log_probs, z_padded, z_count, z_log_probs
                repl_sh, repl_sh,                                   # mu, key
                walker_sh, walker_sh, walker_sh, walker_sh,         # walker aux (4 fields)
                walker_sh,                                          # temperatures
                repl_sh, repl_sh, repl_sh, repl_sh, repl_sh,        # pca/flow/whitening/kde
            )
            out_shardings = (
                walker_sh, walker_sh, repl_sh, walker_sh, walker_sh,  # positions, log_probs, key, found, br
                walker_sh, walker_sh, walker_sh, walker_sh,            # walker aux
            )
            jit_decorator = jax.jit(static_argnums=(), 
                                     in_shardings=in_shardings,
                                     out_shardings=out_shardings)
        else:
            jit_decorator = jax.jit

        @jit_decorator
        def parallel_step(positions, log_probs, z_padded, z_count, z_log_probs,
                          mu, key, walker_aux_prev_dir, walker_aux_bw,
                          walker_aux_d_anchor, walker_aux_d_scale,
                          temperatures, pca_components, pca_weights,
                          flow_directions, whitening_matrix, kde_bandwidth):
            # ... existing function body unchanged ...
```

NOTE: Do NOT modify the function body. Only the decorator setup and the function signature stays the same. The vmap inside the function automatically inherits sharding from the input arrays.

Then find where `_make_step_fn` is called (search for `step_fn = _make_step_fn`) and pass `sharding_info`:

```python
    step_fn = _make_step_fn(n_expand, n_shrink, sharding_info=sharding_info)
```

- [ ] **Step 4: Run backwards-compat test**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_run_variant_default_single_gpu_unchanged -v -s`
Expected: PASS — when sharding_info is None, decorator is plain `jax.jit`

- [ ] **Step 5: Run full test suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -v --tb=no 2>&1 | tail -3`
Expected: All existing tests pass

- [ ] **Step 6: Commit**

```bash
git add dezess/core/loop.py
git commit -m "feat: shard parallel_step JIT when n_gpus > 1"
```

---

### Task 5: Z-matrix all-gather during warmup

**Files:**
- Modify: `dezess/core/loop.py` (warmup loop, around line 670 — after step + before z-matrix append)

- [ ] **Step 1: Find the Z-matrix append in warmup**

In `dezess/core/loop.py`, locate the warmup loop (search for `# Append to Z-matrix`). It looks like:

```python
        # Append to Z-matrix (every 5 steps to reduce overhead)
        if step % 5 == 0 or step == n_warmup - 1:
            z_padded, z_count, z_log_probs = circular_zmatrix.append(
                z_padded, z_count, z_log_probs, positions, log_probs, z_max_size,
            )
```

- [ ] **Step 2: Replace with sharding-aware version**

Replace that block with:

```python
        # Append to Z-matrix (every 5 steps to reduce overhead)
        if step % 5 == 0 or step == n_warmup - 1:
            if sharding_info is not None:
                # All-gather walker positions to every GPU before append
                pos_full = jax.lax.with_sharding_constraint(
                    positions, sharding_info["replicated"])
                lp_full = jax.lax.with_sharding_constraint(
                    log_probs, sharding_info["replicated"])
            else:
                pos_full = positions
                lp_full = log_probs
            z_padded, z_count, z_log_probs = circular_zmatrix.append(
                z_padded, z_count, z_log_probs, pos_full, lp_full, z_max_size,
            )
```

- [ ] **Step 3: Run multi-GPU test (will only run on multi-GPU machine)**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_run_variant_n_gpus_2_recovers_gaussian_variance -v -s`
Expected: SKIP (single-GPU) or PASS (multi-GPU). The all-gather makes the Z-matrix consistent across GPUs.

- [ ] **Step 4: Run backwards-compat test**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_run_variant_default_single_gpu_unchanged -v -s`
Expected: PASS — when sharding_info is None, no change to behavior

- [ ] **Step 5: Commit**

```bash
git add dezess/core/loop.py
git commit -m "feat: all-gather positions to Z-matrix during multi-GPU warmup"
```

---

### Task 6: Sharded production scan with K=100 sample batching

**Files:**
- Modify: `dezess/core/loop.py:1225-1300` (production loop with `_scan_steps`)

The existing production loop already uses `lax.scan` with `SCAN_BATCH=50`. We need to:
1. Make `SCAN_BATCH=100` when sharded
2. Add sharding annotations to `_scan_steps` JIT
3. Gather samples to host after each scan batch

- [ ] **Step 1: Locate the production scan in loop.py**

In `dezess/core/loop.py`, find `SCAN_BATCH = 50` (around line 1226). The block above it sets `use_scan`.

- [ ] **Step 2: Replace SCAN_BATCH and _scan_steps with sharding-aware versions**

Replace the block from `SCAN_BATCH = 50` through the end of `_run_batch` with:

```python
    # Scan batch size: K=100 for sharded runs (matches checkpoint cadence),
    # 50 for single-GPU (current behavior, lower memory in flight).
    SCAN_BATCH = 100 if sharding_info is not None else 50
    use_scan = (config.ensemble not in ("parallel_tempering", "block_gibbs")) and not live_z

    if use_scan:
        if sharding_info is not None:
            walker_sh = sharding_info["walker_sharding"]
            repl_sh = sharding_info["replicated"]
            scan_jit = jax.jit  # scan inherits shardings from carry
        else:
            scan_jit = jax.jit

        @scan_jit
        def _scan_steps(carry, _):
            (pos, lps, k, pd, bw, da, ds) = carry
            (pos, lps, k, found, br, pd, bw, da, ds) = step_fn(
                pos, lps, z_frozen, z_count_frozen, z_lp_frozen,
                mu, k, pd, bw, da, ds, temperatures,
                pca_components, pca_weights, flow_directions, whitening_matrix, kde_bandwidth,
            )
            return (pos, lps, k, pd, bw, da, ds), (pos, lps, found, br)

        def _run_batch(positions, log_probs, key, walker_aux_pd, walker_aux_bw,
                       walker_aux_da, walker_aux_ds, n_batch):
            carry = (positions, log_probs, key, walker_aux_pd, walker_aux_bw,
                     walker_aux_da, walker_aux_ds)
            carry, outputs = jax.lax.scan(_scan_steps, carry, None, length=n_batch)
            (positions, log_probs, key, walker_aux_pd, walker_aux_bw,
             walker_aux_da, walker_aux_ds) = carry
            batch_samples, batch_lps, batch_found, batch_br = outputs

            # Gather sharded samples to host (one collective + D2H per K steps)
            if sharding_info is not None:
                batch_samples = jax.lax.with_sharding_constraint(
                    batch_samples, sharding_info["replicated"])
                batch_lps = jax.lax.with_sharding_constraint(
                    batch_lps, sharding_info["replicated"])

            return (positions, log_probs, key, walker_aux_pd, walker_aux_bw,
                    walker_aux_da, walker_aux_ds,
                    batch_samples, batch_lps, batch_found, batch_br)
```

- [ ] **Step 3: Run multi-GPU test**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_run_variant_n_gpus_2_recovers_gaussian_variance -v -s`
Expected: SKIP (single-GPU) or PASS (multi-GPU)

- [ ] **Step 4: Run full test suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -v --tb=no 2>&1 | tail -3`
Expected: All existing tests pass (single-GPU SCAN_BATCH=50 unchanged)

- [ ] **Step 5: Commit**

```bash
git add dezess/core/loop.py
git commit -m "feat: K=100 batched sample gather for multi-GPU production scan"
```

---

### Task 7: API pass-through

**Files:**
- Modify: `dezess/api.py` (add params, pass through to run_variant)

- [ ] **Step 1: Find sample function in api.py**

Run: `grep -n "def sample" dezess/api.py | head -5`

The `sample()` function wraps `run_variant`. Add the new params.

- [ ] **Step 2: Add params to api.py sample()**

In `dezess/api.py`, find the `def sample(` signature. Add `n_gpus: int = 1` and `n_walkers_per_gpu: Optional[int] = None` to the signature (alongside existing optional kwargs). Then in the body where `run_variant` is called, pass them through:

```python
    result = run_variant(
        log_prob_fn=log_prob,
        init_positions=init,
        n_steps=n_steps,
        config=config,
        n_warmup=n_warmup,
        # ... existing kwargs ...
        n_gpus=n_gpus,
        n_walkers_per_gpu=n_walkers_per_gpu,
    )
```

- [ ] **Step 3: Run api tests to confirm no regression**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_api.py -v --tb=no 2>&1 | tail -3`
Expected: All api tests still pass (no caller passes n_gpus, default = 1)

- [ ] **Step 4: Commit**

```bash
git add dezess/api.py
git commit -m "feat: expose n_gpus and n_walkers_per_gpu in dezess.sample API"
```

---

### Task 8: Checkpoint gather

**Files:**
- Modify: `dezess/checkpoint.py`

- [ ] **Step 1: Add gather to save_checkpoint**

In `dezess/checkpoint.py`, locate `save_checkpoint`. The existing code does:

```python
    samples = np.asarray(result["samples"]) if "samples" in result else np.asarray(result.samples)
```

This already calls `np.asarray` which gathers from devices. The fix is making sure the gather happens correctly for sharded arrays. Add an explicit `jax.device_get` before `np.asarray`:

Find:
```python
    samples = np.asarray(result["samples"]) if "samples" in result else np.asarray(result.samples)
    log_prob = np.asarray(result["log_prob"]) if "log_prob" in result else np.asarray(result.log_prob)
```

Replace with:

```python
    # Gather sharded arrays to host before converting to numpy
    samples_raw = result["samples"] if "samples" in result else result.samples
    log_prob_raw = result["log_prob"] if "log_prob" in result else result.log_prob
    samples = np.asarray(jax.device_get(samples_raw))
    log_prob = np.asarray(jax.device_get(log_prob_raw))
```

Also add `import jax` at the top of the file if not present.

- [ ] **Step 2: Run checkpoint tests**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -k checkpoint -v --tb=no 2>&1 | tail -3`
Expected: All checkpoint tests pass

- [ ] **Step 3: Commit**

```bash
git add dezess/checkpoint.py
git commit -m "feat: explicit device_get for sharded arrays in checkpoint save"
```

---

### Task 9: Block-Gibbs sharding (the most-used config)

**Files:**
- Modify: `dezess/core/loop.py:430-595` (block_mh_step JIT)

Block-Gibbs is the user's recommended config (bg_MH+DR). It needs sharding annotations on its dedicated step function `block_mh_step`.

- [ ] **Step 1: Add block-Gibbs sharding test**

Append to `dezess/tests/test_sharding.py`:

```python
def test_block_gibbs_mh_dr_sharded():
    """bg_MH+DR works with multi-GPU sharding."""
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig

    if len(jax.devices()) < 2:
        pytest.skip("Need >= 2 devices")

    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (64, 21)) * 0.1

    config = VariantConfig(
        name="bg_mh_dr_sharded",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14], "use_mh": True,
                         "delayed_rejection": True},
    )

    result = run_variant(log_prob, init, n_steps=1500, config=config,
                         n_warmup=500, verbose=False,
                         n_gpus=2, n_walkers_per_gpu=32)
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 21)
    mean_var = float(np.var(flat, axis=0).mean())
    assert 0.7 < mean_var < 1.3, f"mean_var={mean_var:.4f}, expected ~1.0"
```

- [ ] **Step 2: Run test (skipped on single-GPU)**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_block_gibbs_mh_dr_sharded -v -s`
Expected: SKIP (single-GPU). On multi-GPU machines, will FAIL until block-Gibbs JIT is sharded.

- [ ] **Step 3: Add sharding to block_mh_step**

In `dezess/core/loop.py`, find `def block_mh_step(positions, log_probs, ...)` (search for `block_mh_step`). The current decorator is `@jax.jit`. Wrap it with conditional sharding:

```python
        if sharding_info is not None:
            walker_sh = sharding_info["walker_sharding"]
            repl_sh = sharding_info["replicated"]
            block_mh_jit = jax.jit(
                in_shardings=(walker_sh, walker_sh, repl_sh, repl_sh, repl_sh, repl_sh),
                out_shardings=(walker_sh, walker_sh, repl_sh, repl_sh),
            )
        else:
            block_mh_jit = jax.jit

        @block_mh_jit
        def block_mh_step(positions, log_probs, z_padded, z_count,
                          mu_blocks_arg, key):
            # ... existing body unchanged ...
```

Apply the same pattern to `block_sweep_step` (and any other block step variants you find — `block_mh_cov_step`, `block_conditional_step` if they're built post-warmup, leave those for later if not in scope).

- [ ] **Step 4: Run multi-GPU block-Gibbs test**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/test_sharding.py::test_block_gibbs_mh_dr_sharded -v -s`
Expected: SKIP (single-GPU) or PASS (multi-GPU)

- [ ] **Step 5: Run full test suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -v --tb=no 2>&1 | tail -3`
Expected: All existing tests still pass (sharding_info is None for all of them)

- [ ] **Step 6: Commit**

```bash
git add dezess/core/loop.py
git commit -m "feat: shard block_mh_step and block_sweep_step JITs"
```

---

### Task 10: Multi-GPU benchmark on H200

**Files:**
- Create: `bench_multi_gpu.py`
- Create: `run_bench_multi_gpu.sh`

- [ ] **Step 1: Create benchmark script**

Create `bench_multi_gpu.py`:

```python
#!/usr/bin/env python
"""Benchmark: single-GPU vs multi-GPU (sharded ensemble) on bg_MH+DR.

Compares total ESS and ESS/sec for 64 walkers (single GPU) vs
256 walkers across 4 GPUs (sharded ensemble).
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
n_devices = len(jax.devices())
print(f"Detected {n_devices} devices")

target = block_coupled_gaussian()
NDIM = target.ndim
N_WARMUP = 2000
N_PROD = 5000

def make_config(block_sizes):
    return VariantConfig(
        name="bg_mh_dr",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": block_sizes, "use_mh": True,
                         "delayed_rejection": True},
    )

configs = [
    ("1 GPU x 64 walkers", 1, 64),
    ("2 GPUs x 64 walkers (128 total)", 2, 64),
    ("4 GPUs x 64 walkers (256 total)", 4, 64),
]

key = jax.random.PRNGKey(42)

hdr = f"  {'Setup':<40s} | {'R-hat':>6s} | {'ESS min':>8s} | {'ESS mean':>9s} | {'Wall':>6s} | {'ESS/sec':>8s}"
print(f"\n{hdr}")
print("  " + "-" * (len(hdr) - 2))

for label, n_gpus, n_walkers_per_gpu in configs:
    if n_gpus > n_devices:
        print(f"  {label:<40s} | SKIP (only {n_devices} devices)")
        continue
    n_total = n_gpus * n_walkers_per_gpu
    init = target.sample(key, n_total)
    config = make_config([7, 14, 14, 14, 14])

    t0 = time.time()
    result = run_variant(
        target.log_prob, init, n_steps=N_WARMUP + N_PROD,
        config=config, n_warmup=N_WARMUP, verbose=False,
        n_gpus=n_gpus, n_walkers_per_gpu=n_walkers_per_gpu,
    )
    wall = time.time() - t0
    samples = np.array(result["samples"])
    ess = compute_ess(samples)
    rhat = compute_rhat(samples)
    print(f"  {label:<40s} | {float(rhat.max()):6.3f} | {float(ess.min()):8.1f} | "
          f"{float(ess.mean()):9.1f} | {wall:5.1f}s | {float(ess.min())/wall:8.2f}",
          flush=True)
    gc.collect()
```

- [ ] **Step 2: Create Slurm script**

Create `run_bench_multi_gpu.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=bench_mgpu
#SBATCH --output=slurm_bench_mgpu_%j.out
#SBATCH --error=slurm_bench_mgpu_%j.err
#SBATCH --time=0:30:00
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=64G
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
echo "=== bench_multi_gpu.py — $(date) ==="

cd GIVEMEPotential/third_party/dezess
conda run --no-capture-output -p "$CONDA_ENV" python -u bench_multi_gpu.py
```

- [ ] **Step 3: Submit benchmark to H200**

Run: `cd /home/jaks/Documents/jek284/ondemand/data/sys/myjobs/projects/default/givemepotential && scripts/hyak_sync_submit.sh GIVEMEPotential/third_party/dezess/run_bench_multi_gpu.sh`
Expected: Submitted batch job <ID>

- [ ] **Step 4: Wait for results, pull, and report**

Run: `scripts/hyak_status.sh <JOB_ID>` — wait for COMPLETED
Then: `scripts/hyak_pull.sh "slurm_bench_mgpu_<JOB_ID>.out"` and `cat slurm_bench_mgpu_<JOB_ID>.out`

Expected output: 1-GPU, 2-GPU, 4-GPU comparison showing ESS_min scaling roughly linearly with number of GPUs (since 4× walkers yield ~4× more independent samples).

- [ ] **Step 5: Commit benchmark scripts**

```bash
git add bench_multi_gpu.py run_bench_multi_gpu.sh
git commit -m "bench: add multi-GPU sharding benchmark on H200"
```
