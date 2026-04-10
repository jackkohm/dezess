# JAX Optimization Pass Design

## Goal

Optimize dezess hot-path code for speed, memory, and XLA fusion without changing numerical precision or sampler behavior. All 64+ existing tests must continue to pass with identical results (modulo floating-point ordering effects from `linalg.norm` vs manual sqrt).

## Scope

Only hot-path files called every MCMC step. NOT benchmarks, test files, or one-time setup code.

**Files in scope:**
- `dezess/core/loop.py` — main orchestrator, JIT step functions
- `dezess/core/slice_sample.py` — safe_log_prob wrapper
- `dezess/width/scale_aware.py` — scale-aware width computation
- `dezess/slice/nurs.py` — NURS orbit sampler
- `dezess/slice/multi_try.py` — multi-try DE-MCz

## Constraint

Do NOT change float64 precision. Do NOT change sampler semantics. Every optimization must be behavior-preserving. Tests are the correctness gate.

---

## Section 1: XLA Fusion Fixes

### 1a. Double indexing → single fancy index

Replace all `z_padded[idx1][b_idx]` with `z_padded[idx1, b_idx]`. This eliminates an intermediate array allocation and allows XLA to fuse the two gathers into one.

Locations in `loop.py`: lines 466-467, 524-525, 551-552, 1041-1042, 1083-1084, and all equivalent lines in `block_mh_cov_step` and `block_conditional_step`.

### 1b. `jnp.linalg.norm` instead of manual `sqrt(sum(x**2))`

Replace:
```python
norm = jnp.sqrt(jnp.sum(diff**2))
```
with:
```python
norm = jnp.linalg.norm(diff)
```

XLA has a fused norm kernel. Applies everywhere a vector norm is computed (direction normalization, scale-aware width).

### 1c. `.at[].set()` instead of `.at[].add()` on zero arrays

When scattering into a freshly created zero array:
```python
d_full = jnp.zeros(n_dim, dtype=jnp.float64)
d_full = d_full.at[b_idx].add(d_block * mask)  # read-modify-write on zeros
```
Replace `.add()` with `.set()`:
```python
d_full = jnp.zeros(n_dim, dtype=jnp.float64)
d_full = d_full.at[b_idx].set(d_block * mask)  # direct write, no read needed
```

### 1d. `lax.clamp` for clamping patterns

Replace:
```python
mu = jnp.clip(mu * adjustment, MU_MIN, MU_MAX)
```
with:
```python
mu = jax.lax.clamp(jnp.float64(MU_MIN), mu * adjustment, jnp.float64(MU_MAX))
```
Single XLA op instead of two separate min/max ops. Apply in the mu tuning loop.

---

## Section 2: Memory — donate_argnums

### 2a. `parallel_step` donation

```python
@jax.jit(donate_argnums=(0, 1))
def parallel_step(positions, log_probs, z_padded, z_count, ...):
```

Donates `positions` and `log_probs` input buffers — they're replaced every step.

Do NOT donate `z_padded`, `z_count`, or `z_log_probs` — these are reused (Z-matrix is frozen in production, but shared during warmup).

### 2b. `block_mh_step` and variants

Same pattern — donate positions and log_probs (args 0 and 1):
```python
@jax.jit(donate_argnums=(0, 1))
def block_mh_step(positions, log_probs, ...):

@jax.jit(donate_argnums=(0, 1))
def block_mh_cov_step(positions, log_probs, ...):

@jax.jit(donate_argnums=(0, 1))
def block_conditional_step(positions, log_probs, ...):
```

### 2c. Production scan

The `_scan_steps` JIT function also takes positions/log_probs as part of its carry. Donate the carry tuple's first elements:
```python
@jax.jit(donate_argnums=(0,))  # donate the carry tuple
def _scan_steps(carry, _):
```

Note: when donating a pytree (tuple), JAX donates all leaf buffers within it. The carry tuple contains `(positions, log_probs, key, ...)` — all leaves get donated. This is fine since `_scan_steps` produces a new carry each iteration.

---

## Section 3: Pre-computed Constants

### 3a. Dimension correction factor

In `scale_aware.py`, `jnp.sqrt(ndim / 2.0)` is computed per call. Pre-compute once:

```python
# In run_variant setup:
dim_corr_factor = jnp.sqrt(jnp.float64(n_dim) / 2.0)
```

Pass to `get_mu` via `width_kwargs` or capture in the step function closure.

### 3b. Bracket ratio target

In `_mh_walker`:
```python
target = jnp.float64(n_expand + 1)
```
This creates a new JAX scalar every call. Pre-compute outside:
```python
# In block-Gibbs setup:
br_target = jnp.float64(n_expand + 1)
br_accept = 2.0 * br_target - 1.0
br_reject = jnp.float64(1.0)
```
Capture as closure constants.

### 3c. Optional unsafe log_prob

Add config flag `check_nans` (default True). When False, skip the `isfinite` + `where` in `safe_log_prob`:

```python
# In slice_sample.py:
def safe_log_prob(log_prob_fn, x):
    lp = log_prob_fn(x)
    return jnp.where(jnp.isfinite(lp), lp, jnp.float64(-1e30))

def unsafe_log_prob(log_prob_fn, x):
    return log_prob_fn(x)
```

In `loop.py`, select based on config:
```python
lp_eval = safe_log_prob if config.check_nans else unsafe_log_prob
```

Add `check_nans: bool = True` field to `VariantConfig`.

### 3d. Gram-Schmidt optimization

In the `n_slices_per_step > 1` path, replace:
```python
d_ortho, _ = jax.lax.scan(_gs_step, diff, jnp.arange(n_max_slices))
```
with:
```python
d_ortho = jax.lax.fori_loop(0, i + 1, _ortho_one, diff)
```
Only iterates over directions already set, not the full padded array.

Note: `i` is the scan iteration index (traced), so `fori_loop(0, i+1, ...)` uses a traced bound — this is valid for `fori_loop` (unlike `scan` which needs static length).

---

## Section 4: Code Cleanliness

### 4a. Extract shared MH proposal logic

Create a helper used by all block step functions:

```python
def _de_proposal(x_full, z_padded, z_count, b_idx, mask, bsize, mu_b, key):
    """Generate a DE-MCz MH proposal for one block. Returns (x_prop, lp_direction_norm, d_full, mu_eff)."""
    ...

def _cov_proposal(x_full, L_block, b_idx, mask, bsize, mu_b, key):
    """Generate a covariance-adapted MH proposal for one block."""
    ...
```

These are pure functions called within the JIT boundary — no separate `@jax.jit` needed. They deduplicate the proposal logic across `block_mh_step`, `block_mh_cov_step`, and `block_conditional_step`.

### 4b. Extract MH accept/reject

```python
def _mh_accept_reject(x_full, lp, x_prop, lp_prop, key):
    """Single-stage MH accept/reject. Returns (x_new, lp_new, accepted)."""
    ...

def _mh_delayed_reject(x_full, lp, x_prop1, lp_prop1, x_prop2, lp_prop2, key):
    """Two-stage delayed rejection. Returns (x_new, lp_new, either_accepted)."""
    ...
```

### 4c. Comments

Add inline comments for non-obvious optimization choices:
- Why `donate_argnums=(0, 1)` — buffer reuse
- Why `.set` not `.add` — avoids read-modify-write
- Why `linalg.norm` — XLA fused kernel
- Why `lax.clamp` — single op

No docstrings on internal helpers. No type annotations on closure-internal functions (JAX traces them anyway).

---

## Testing Strategy

1. Run full test suite before and after — all 68 tests must pass
2. Compare numerical output on a fixed seed: sample mean/variance on 10D Gaussian should match to float64 precision (small differences from `linalg.norm` vs manual sqrt are acceptable)
3. Benchmark before/after: measure wall time on `bench_nurs_vmap.py` targets

## Files Changed

- `dezess/core/loop.py` — fusion fixes, donate_argnums, constants, extract helpers
- `dezess/core/slice_sample.py` — add `unsafe_log_prob`
- `dezess/core/types.py` — add `check_nans` field to `VariantConfig`
- `dezess/width/scale_aware.py` — pre-compute dim correction
- `dezess/slice/nurs.py` — `linalg.norm`, fusion fixes
- `dezess/slice/multi_try.py` — `linalg.norm`, fusion fixes
