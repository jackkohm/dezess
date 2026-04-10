# JAX Optimization Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize dezess hot-path code for XLA fusion, memory reuse, and reduced redundancy without changing numerical precision or sampler behavior.

**Architecture:** Behavior-preserving refactoring across 6 files. Each task targets one optimization category, applies it globally, runs the full test suite to verify no regressions, and commits. Tasks are independent — order doesn't matter.

**Tech Stack:** JAX, jax.numpy, jax.lax, XLA

---

### Task 1: XLA Fusion — Double Indexing → Single Fancy Index

**Files:**
- Modify: `dezess/core/loop.py` (lines 466-467, 524-525, 551-552, 1041-1042, 1083-1084)

All instances of `z_padded[idx1][b_idx]` create an intermediate array. Replace with `z_padded[idx1, b_idx]` for a single fused gather.

- [ ] **Step 1: Replace all double-indexing in loop.py**

Find and replace every instance. There are 10 occurrences across 5 functions:

```python
# In block_sweep_step (_update_walker):
z1_block = z_padded[idx1, b_idx]   # was z_padded[idx1][b_idx]
z2_block = z_padded[idx2, b_idx]

# In block_mh_step (_mh_walker) — stage 1:
z1_block = z_padded[idx1, b_idx]
z2_block = z_padded[idx2, b_idx]

# In block_mh_step (_mh_walker) — stage 2 (delayed rejection):
z3_block = z_padded[idx3, b_idx]
z4_block = z_padded[idx4, b_idx]

# In block_conditional_step (_mh_pot_walker):
z1_b = z_padded[idx1, pot_b_idx]
z2_b = z_padded[idx2, pot_b_idx]

# In block_conditional_step (_mh_one_stream):
z1_b = z_padded[idx1, s_b_idx]
z2_b = z_padded[idx2, s_b_idx]
```

Also check `block_mh_cov_step` — it may not have this pattern since it uses covariance proposals, but verify.

- [ ] **Step 2: Run full test suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -v -s`
Expected: All 68 tests PASS

- [ ] **Step 3: Commit**

```bash
git add dezess/core/loop.py
git commit -m "perf: replace double indexing with single fancy index for XLA fusion"
```

---

### Task 2: XLA Fusion — `jnp.linalg.norm` + `.at[].set()` + `lax.clamp`

**Files:**
- Modify: `dezess/core/loop.py` (13 sqrt/sum locations, 7 .add locations)
- Modify: `dezess/slice/multi_try.py` (lines 75, 111)

- [ ] **Step 1: Replace manual sqrt(sum(x**2)) with linalg.norm in loop.py**

Replace all instances of:
```python
norm = jnp.sqrt(jnp.sum(diff**2))
```
with:
```python
norm = jnp.linalg.norm(diff)
```

Locations in loop.py (13 instances): lines 289, 292, 366, 367, 469, 527, 554, 949, 971, 1044, 1086, and any in block_mh_cov_step.

For the snooker distance (lines 289, 292), replace:
```python
dist = jnp.sqrt(jnp.sum((x - z_anchor)**2))
# becomes:
dist = jnp.linalg.norm(x - z_anchor)
```

For the Gram-Schmidt norms (lines 366-367):
```python
norm_ortho = jnp.linalg.norm(d_ortho)
norm_raw = jnp.linalg.norm(diff)
```

- [ ] **Step 2: Replace linalg.norm in multi_try.py**

In `dezess/slice/multi_try.py`, replace:
```python
# Line 75:
norms = jnp.sqrt(jnp.sum(diffs ** 2, axis=1))
# becomes:
norms = jnp.linalg.norm(diffs, axis=1)

# Line 111:
ref_norms = jnp.sqrt(jnp.sum(ref_diffs ** 2, axis=1))
# becomes:
ref_norms = jnp.linalg.norm(ref_diffs, axis=1)
```

- [ ] **Step 3: Replace .at[].add() with .at[].set() on zero arrays in loop.py**

Replace all 7 instances. Each follows the pattern:
```python
# Before:
d_full = jnp.zeros(n_dim, dtype=jnp.float64)
d_full = d_full.at[b_idx].add(d_block * mask)

# After (avoids read-modify-write on zeros):
d_full = jnp.zeros(n_dim, dtype=jnp.float64)
d_full = d_full.at[b_idx].set(d_block * mask)
```

Locations: lines 474, 536, 559, 956, 976, 1050, 1092.

- [ ] **Step 4: Run full test suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -v -s`
Expected: All 68 tests PASS (minor float64 differences from norm implementation are acceptable)

- [ ] **Step 5: Commit**

```bash
git add dezess/core/loop.py dezess/slice/multi_try.py
git commit -m "perf: use linalg.norm, .at[].set(), for better XLA fusion"
```

---

### Task 3: Memory — donate_argnums

**Files:**
- Modify: `dezess/core/loop.py`

- [ ] **Step 1: Add donate_argnums to all JIT step functions**

Find each `@jax.jit` decorated step function and add `donate_argnums=(0, 1)` to donate the `positions` and `log_probs` buffers:

```python
# parallel_step (in _make_step_fn):
@jax.jit(donate_argnums=(0, 1))
def parallel_step(positions, log_probs, z_padded, z_count, z_log_probs, ...):

# block_sweep_step:
@jax.jit(donate_argnums=(0, 1))
def block_sweep_step(positions, log_probs, z_padded, z_count, ...):

# block_mh_step:
@jax.jit(donate_argnums=(0, 1))
def block_mh_step(positions, log_probs, z_padded, z_count, ...):

# block_mh_cov_step (inside post-warmup):
@jax.jit(donate_argnums=(0, 1))
def block_mh_cov_step(positions, log_probs, z_padded, z_count, ...):

# block_conditional_step (inside post-warmup):
@jax.jit(donate_argnums=(0, 1))
def block_conditional_step(positions, log_probs, z_padded, z_count, ...):
```

Also add to the production scan:
```python
# _scan_steps:
@jax.jit(donate_argnums=(0,))
def _scan_steps(carry, _):
```

- [ ] **Step 2: Add inline comment explaining donation**

At the first `donate_argnums` usage, add:
```python
# donate_argnums=(0, 1): positions and log_probs are consumed each step.
# Donating avoids allocating a new buffer — XLA reuses the input buffer
# for the output. The donated buffer becomes invalid after the call.
```

- [ ] **Step 3: Run full test suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -v -s`
Expected: All 68 tests PASS

Note: if any test reads old position arrays after calling the step function, donation will cause a RuntimeError ("buffer was donated"). This would indicate a bug in the test, not the optimization. Fix by copying the input before the step call in the test.

- [ ] **Step 4: Commit**

```bash
git add dezess/core/loop.py
git commit -m "perf: add donate_argnums for buffer reuse in step functions"
```

---

### Task 4: Pre-computed Constants

**Files:**
- Modify: `dezess/core/loop.py`
- Modify: `dezess/core/slice_sample.py`
- Modify: `dezess/core/types.py`

- [ ] **Step 1: Add `check_nans` field to VariantConfig**

In `dezess/core/types.py`, add a new field to `VariantConfig`:
```python
class VariantConfig(NamedTuple):
    """Configuration selecting which strategies to compose."""
    name: str
    direction: str = "de_mcz"
    width: str = "scalar"
    slice_fn: str = "fixed"
    zmatrix: str = "circular"
    ensemble: str = "standard"
    tune_method: str = "bracket"
    n_slices_per_step: int = 1
    check_nans: bool = True          # NEW: set False to skip isfinite checks
    direction_kwargs: dict = {}
    width_kwargs: dict = {}
    slice_kwargs: dict = {}
    zmatrix_kwargs: dict = {}
    ensemble_kwargs: dict = {}
```

- [ ] **Step 2: Add unsafe_log_prob to slice_sample.py**

In `dezess/core/slice_sample.py`, after `safe_log_prob`:
```python
def unsafe_log_prob(log_prob_fn: Callable, x: Array) -> Array:
    """Evaluate log_prob without NaN/Inf checking. Use after warmup verification."""
    return log_prob_fn(x)
```

- [ ] **Step 3: Select lp_eval based on config in loop.py**

In `run_variant`, after the config validation block (around line 160), add:
```python
    # Select log_prob evaluator based on NaN checking preference
    lp_eval = safe_log_prob if config.check_nans else unsafe_log_prob
```

Add the import at the top of loop.py:
```python
from dezess.core.slice_sample import safe_log_prob, unsafe_log_prob
```

Then replace all `safe_log_prob` calls in the step functions with `lp_eval`. Since `lp_eval` is a Python variable resolved at trace time, this adds no overhead — JAX traces through whichever function was selected.

Note: do NOT replace calls in the initial log_prob computation (line ~567: `jax.vmap(lambda x: safe_log_prob(log_prob_fn, x))`) — always use safe there for warmup safety.

- [ ] **Step 4: Pre-compute bracket ratio constants in block-Gibbs setup**

In the block-Gibbs setup section (after `use_block_cov` line), add:
```python
    # Pre-compute bracket ratio targets (avoids creating JAX scalars every step)
    br_accept = jnp.float64(2.0 * (n_expand + 1) - 1.0)
    br_reject = jnp.float64(1.0)
```

Then in `_mh_walker`, replace:
```python
    target = jnp.float64(n_expand + 1)
    br = jnp.where(either_accepted, 2.0 * target - 1.0, 1.0)
```
with:
```python
    br = jnp.where(either_accepted, br_accept, br_reject)
```

- [ ] **Step 5: Run full test suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -v -s`
Expected: All 68 tests PASS

- [ ] **Step 6: Commit**

```bash
git add dezess/core/types.py dezess/core/slice_sample.py dezess/core/loop.py
git commit -m "perf: add check_nans config, pre-compute bracket ratio constants"
```

---

### Task 5: Gram-Schmidt Optimization

**Files:**
- Modify: `dezess/core/loop.py` (lines 353-364, the multi-slice orthogonalization)

- [ ] **Step 1: Replace scan with fori_loop for Gram-Schmidt**

Find the Gram-Schmidt section (around line 353):
```python
                        def _gs_step(d_raw, j):
                            active = j <= i  # only remove for dirs already set
                            proj = jnp.dot(d_raw, pdirs[j]) * pdirs[j]
                            d_raw = jnp.where(active, d_raw - proj, d_raw)
                            return d_raw, None
                        d_ortho, _ = jax.lax.scan(
                            _gs_step, diff, jnp.arange(n_max_slices),
                        )
```

Replace with:
```python
                        def _ortho_one(j, d_raw):
                            proj = jnp.dot(d_raw, pdirs[j]) * pdirs[j]
                            return d_raw - proj
                        # Only iterate over directions already set (i+1),
                        # not the full n_max_slices. i is a traced value,
                        # which is valid for fori_loop bounds.
                        d_ortho = jax.lax.fori_loop(0, i + 1, _ortho_one, diff)
```

This eliminates masked no-ops for unused slots. With `n_slices_per_step=3`, the first iteration does 1 projection instead of 3; the second does 2 instead of 3.

- [ ] **Step 2: Run full test suite**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro pytest dezess/tests/ -v -s`
Expected: All 68 tests PASS

- [ ] **Step 3: Commit**

```bash
git add dezess/core/loop.py
git commit -m "perf: optimize Gram-Schmidt to iterate only over set directions"
```

---

### Task 6: Benchmark Before/After

**Files:**
- Modify: `bench_nurs_vmap.py` (restore comparison configs)

- [ ] **Step 1: Create before/after benchmark script**

Create `bench_optimization.py`:

```python
#!/usr/bin/env python
"""Benchmark before/after JAX optimization pass.

Runs bg_MH+DR on funnel_63d and block_coupled_63d, measures wall time.
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
from dezess.targets_stream import funnel_63d, block_coupled_gaussian

print(f"JAX devices: {jax.devices()}")

config = VariantConfig(
    name="bg_mh_dr",
    direction="de_mcz",
    width="scale_aware",
    slice_fn="fixed",
    zmatrix="circular",
    ensemble="block_gibbs",
    width_kwargs={"scale_factor": 1.0},
    ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True,
                     "delayed_rejection": True},
)

# Also test with check_nans=False
config_unsafe = config._replace(check_nans=False)

targets = [
    ("Funnel 63D", funnel_63d()),
    ("Block-Coupled 63D", block_coupled_gaussian()),
]

def run_one(cfg, target, n_walkers=64, n_warmup=2000, n_prod=5000):
    key = jax.random.PRNGKey(42)
    init = target.sample(key, n_walkers) if target.sample else jax.random.normal(key, (n_walkers, target.ndim)) * 0.1
    t0 = time.time()
    result = run_variant(target.log_prob, init, n_steps=n_warmup + n_prod,
                         config=cfg, n_warmup=n_warmup, verbose=False)
    wall = time.time() - t0
    samples = np.array(result["samples"])
    ess = compute_ess(samples)
    rhat = compute_rhat(samples)
    return {"rhat": float(rhat.max()), "ess_min": float(ess.min()),
            "wall": wall}

hdr = f"{'Target':<22s} | {'Config':<20s} | {'R-hat':>6s} | {'ESS min':>8s} | {'Wall':>6s}"
print(f"\n{hdr}")
print("-" * len(hdr))

for tname, target in targets:
    for label, cfg in [("safe (default)", config), ("unsafe (no NaN)", config_unsafe)]:
        r = run_one(cfg, target)
        print(f"{tname:<22s} | {label:<20s} | {r['rhat']:6.3f} | {r['ess_min']:8.1f} | {r['wall']:5.1f}s",
              flush=True)
        gc.collect()
    print()
```

- [ ] **Step 2: Run benchmark**

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n Astro python bench_optimization.py`

Compare wall times. The optimizations should show:
- Slightly faster wall time (fusion + donation)
- Identical ESS/R-hat (behavior-preserving)
- `check_nans=False` should be marginally faster

- [ ] **Step 3: Commit**

```bash
git add bench_optimization.py
git commit -m "bench: add before/after JAX optimization benchmark"
```
