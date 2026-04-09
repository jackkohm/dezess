# Funnel Reparameterization & Block-Gibbs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dezess handle 63D funnel geometry with R-hat < 1.2 via non-centered reparameterization and block-Gibbs updates.

**Architecture:** Add a `Transform` abstraction that wraps `log_prob_fn` to sample in unconstrained space (fixing funnels), and a `block_gibbs` ensemble strategy that cycles slice updates over parameter blocks with per-block step sizes (exploiting conditional independence). Both compose naturally.

**Tech Stack:** JAX, jax.numpy, jax.lax, pytest

**Spec:** `docs/specs/2026-04-09-funnel-reparam-design.md`

**Working directory:** `/home/jaks/Documents/jek284/ondemand/data/sys/myjobs/projects/default/givemepotential/GIVEMEPotential/third_party/dezess`

**Branch:** `feat/funnel-reparam`

**Run tests with:** `conda run -n Astro pytest <path> -v`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `dezess/transforms.py` | Create | Transform NamedTuple, Identity, NonCenteredFunnel, block_transform, multi_funnel |
| `dezess/ensemble/block_gibbs.py` | Create | Block-Gibbs sweep logic: parse block spec, per-block direction/slice/mu |
| `dezess/core/loop.py` | Modify | Add `transform` param to `run_variant`, wrap log_prob, map samples back |
| `dezess/api.py` | Modify | Pass `transform` through `sample()` to `run_variant` |
| `dezess/__init__.py` | Modify | Export `transforms` module |
| `dezess/benchmark/registry.py` | Modify | Register 4 new variants |
| `dezess/tests/test_funnel_reparam.py` | Create | All tests: round-trip, Geweke, convergence |
| `bench_funnel_reparam.py` | Create | Comparison benchmark script |

---

### Task 1: Transform Module — Core Types and Identity

**Files:**
- Create: `dezess/transforms.py`
- Create: `dezess/tests/test_funnel_reparam.py`

- [ ] **Step 1: Write the round-trip test for Identity transform**

In `dezess/tests/test_funnel_reparam.py`:

```python
"""Tests for funnel reparameterization and block-Gibbs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_identity_roundtrip():
    from dezess.transforms import identity

    t = identity()
    x = jnp.array([1.0, 2.0, 3.0])
    z = t.inverse(x)
    x_rec = t.forward(z)
    np.testing.assert_allclose(x_rec, x, atol=1e-14)
    assert t.log_det_jac(z) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_identity_roundtrip -v`

Expected: FAIL — `ImportError: cannot import name 'identity' from 'dezess.transforms'`

- [ ] **Step 3: Write the Transform type and Identity**

In `dezess/transforms.py`:

```python
"""Bijector transforms for reparameterizing target distributions.

A Transform maps between an unconstrained space z (where the sampler works)
and the original space x (where log_prob is defined). The sampler internally
works in z-space with the transformed log-prob:

    lp_transformed(z) = log_prob(forward(z)) + log_det_jac(z)

and maps samples back to x-space via forward() before returning them.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


class Transform(NamedTuple):
    """Bijector transform between unconstrained (z) and original (x) space.

    forward:     z -> x
    inverse:     x -> z
    log_det_jac: z -> scalar log|det(df/dz)|
    """
    forward: Callable
    inverse: Callable
    log_det_jac: Callable


def identity() -> Transform:
    """Identity transform (no-op)."""
    return Transform(
        forward=lambda z: z,
        inverse=lambda x: x,
        log_det_jac=lambda z: jnp.float64(0.0),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_identity_roundtrip -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dezess/transforms.py dezess/tests/test_funnel_reparam.py
git commit -m "Add Transform type and Identity transform with round-trip test"
```

---

### Task 2: NonCenteredFunnel Transform

**Files:**
- Modify: `dezess/transforms.py`
- Modify: `dezess/tests/test_funnel_reparam.py`

- [ ] **Step 1: Write tests for NonCenteredFunnel**

Append to `dezess/tests/test_funnel_reparam.py`:

```python
def test_ncp_funnel_roundtrip():
    """forward(inverse(x)) == x for the funnel transform."""
    from dezess.transforms import non_centered_funnel

    # Single funnel: dim 0 is log-width, dims 1-4 are offsets
    t = non_centered_funnel(width_idx=0, offset_indices=[1, 2, 3, 4])

    # Test at various log-width values (narrow to wide funnel)
    for log_w in [-3.0, 0.0, 3.0, 6.0]:
        x = jnp.array([log_w, 0.5, -1.0, 2.0, -0.3])
        z = t.inverse(x)
        x_rec = t.forward(z)
        np.testing.assert_allclose(x_rec, x, atol=1e-12)

    # z-space should have log_width unchanged, offsets rescaled
    x = jnp.array([2.0, 1.0, -1.0, 0.5, -0.5])
    z = t.inverse(x)
    assert z[0] == x[0]  # log_width unchanged
    expected_z_offsets = x[1:] / jnp.exp(x[0] / 2.0)
    np.testing.assert_allclose(z[1:], expected_z_offsets, atol=1e-12)


def test_ncp_funnel_jacobian():
    """log_det_jac matches numerical finite-difference Jacobian."""
    from dezess.transforms import non_centered_funnel

    t = non_centered_funnel(width_idx=0, offset_indices=[1, 2, 3])
    z = jnp.array([2.0, 0.5, -0.3, 1.0])

    # Analytical
    ldj_analytical = t.log_det_jac(z)

    # Numerical: compute full Jacobian df/dz and take log|det|
    jac = jax.jacobian(t.forward)(z)
    _, ldj_numerical = jnp.linalg.slogdet(jac)

    np.testing.assert_allclose(ldj_analytical, ldj_numerical, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_ncp_funnel_roundtrip dezess/tests/test_funnel_reparam.py::test_ncp_funnel_jacobian -v`

Expected: FAIL — `ImportError: cannot import name 'non_centered_funnel'`

- [ ] **Step 3: Implement NonCenteredFunnel**

Append to `dezess/transforms.py`:

```python
def non_centered_funnel(
    width_idx: int,
    offset_indices: Sequence[int],
) -> Transform:
    """Non-centered parameterization for a Neal's funnel block.

    In the original space x:
        x[width_idx] = log_width ~ N(0, 9)
        x[offset_i]  ~ N(0, exp(log_width))  (funnel geometry)

    In the unconstrained space z:
        z[width_idx] = log_width  (unchanged)
        z[offset_i]  ~ N(0, 1)   (standard Gaussian — no funnel)

    Forward (z -> x):  x[offset_i] = z[offset_i] * exp(z[width_idx] / 2)
    Inverse (x -> z):  z[offset_i] = x[offset_i] / exp(x[width_idx] / 2)
    Log-det-Jacobian:  n_offsets * z[width_idx] / 2
    """
    offset_idx = jnp.array(offset_indices, dtype=jnp.int32)
    n_offsets = len(offset_indices)

    def forward(z):
        scale = jnp.exp(z[width_idx] / 2.0)
        x = z.at[offset_idx].multiply(scale)
        return x

    def inverse(x):
        scale = jnp.exp(x[width_idx] / 2.0)
        z = x.at[offset_idx].divide(scale)
        return z

    def log_det_jac(z):
        return jnp.float64(n_offsets) * z[width_idx] / 2.0

    return Transform(forward=forward, inverse=inverse, log_det_jac=log_det_jac)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py -v`

Expected: 3 PASS (identity_roundtrip, ncp_funnel_roundtrip, ncp_funnel_jacobian)

- [ ] **Step 5: Commit**

```bash
git add dezess/transforms.py dezess/tests/test_funnel_reparam.py
git commit -m "Add NonCenteredFunnel transform with round-trip and Jacobian tests"
```

---

### Task 3: block_transform and multi_funnel Helpers

**Files:**
- Modify: `dezess/transforms.py`
- Modify: `dezess/tests/test_funnel_reparam.py`

- [ ] **Step 1: Write tests for block_transform and multi_funnel**

Append to `dezess/tests/test_funnel_reparam.py`:

```python
def test_block_transform_roundtrip():
    """block_transform applies different transforms to different dims."""
    from dezess.transforms import identity, non_centered_funnel, block_transform

    # 12D: dims 0-3 identity, dims 4-11 funnel (dim 4 = log-width, 5-11 = offsets)
    t = block_transform(
        transforms=[identity(), non_centered_funnel(width_idx=0, offset_indices=[1, 2, 3, 4, 5, 6, 7])],
        index_lists=[list(range(4)), list(range(4, 12))],
        ndim=12,
    )
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (12,), dtype=jnp.float64)
    z = t.inverse(x)
    x_rec = t.forward(z)
    np.testing.assert_allclose(x_rec, x, atol=1e-12)

    # Jacobian check
    jac = jax.jacobian(t.forward)(z)
    _, ldj_num = jnp.linalg.slogdet(jac)
    np.testing.assert_allclose(t.log_det_jac(z), ldj_num, atol=1e-10)


def test_multi_funnel_63d():
    """multi_funnel helper builds correct 63D transform."""
    from dezess.transforms import multi_funnel

    t = multi_funnel(
        n_potential=7,
        funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)],
    )
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (63,), dtype=jnp.float64)
    z = t.inverse(x)
    x_rec = t.forward(z)
    np.testing.assert_allclose(x_rec, x, atol=1e-12)

    # Potential dims (0-6) should be unchanged
    np.testing.assert_allclose(z[:7], x[:7], atol=1e-14)

    # Jacobian check
    jac = jax.jacobian(t.forward)(z)
    _, ldj_num = jnp.linalg.slogdet(jac)
    np.testing.assert_allclose(t.log_det_jac(z), ldj_num, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_block_transform_roundtrip dezess/tests/test_funnel_reparam.py::test_multi_funnel_63d -v`

Expected: FAIL — `ImportError: cannot import name 'block_transform'`

- [ ] **Step 3: Implement block_transform and multi_funnel**

Append to `dezess/transforms.py`:

```python
def block_transform(
    transforms: Sequence[Transform],
    index_lists: Sequence[Sequence[int]],
    ndim: int,
) -> Transform:
    """Apply different transforms to different parameter blocks.

    Each transform operates on its own block of dimensions. The blocks
    must be non-overlapping and together cover all ndim dimensions.

    Parameters
    ----------
    transforms : list of Transform
        One transform per block.
    index_lists : list of list of int
        Dimension indices for each block.
    ndim : int
        Total number of dimensions.
    """
    # Pre-convert to JAX arrays for indexing
    idx_arrays = [jnp.array(idx, dtype=jnp.int32) for idx in index_lists]

    def forward(z):
        x = jnp.zeros(ndim, dtype=z.dtype)
        for t, idx in zip(transforms, idx_arrays):
            z_block = z[idx]
            x_block = t.forward(z_block)
            x = x.at[idx].set(x_block)
        return x

    def inverse(x):
        z = jnp.zeros(ndim, dtype=x.dtype)
        for t, idx in zip(transforms, idx_arrays):
            x_block = x[idx]
            z_block = t.inverse(x_block)
            z = z.at[idx].set(z_block)
        return z

    def log_det_jac(z):
        ldj = jnp.float64(0.0)
        for t, idx in zip(transforms, idx_arrays):
            z_block = z[idx]
            ldj = ldj + t.log_det_jac(z_block)
        return ldj

    return Transform(forward=forward, inverse=inverse, log_det_jac=log_det_jac)


def multi_funnel(
    n_potential: int,
    funnel_blocks: Sequence[tuple[int, int]],
) -> Transform:
    """Build a block transform for multiple funnel blocks + potential.

    Parameters
    ----------
    n_potential : int
        Number of potential parameters (identity-transformed).
    funnel_blocks : list of (start_idx, block_size)
        Each block: first dim is log-width, remaining are offsets.
        E.g., [(7, 14), (21, 14), (35, 14), (49, 14)] for 4 streams.
    """
    transforms = [identity()]  # potential block
    index_lists = [list(range(n_potential))]

    for start, size in funnel_blocks:
        # Within this block: index 0 is log-width, 1..size-1 are offsets
        t = non_centered_funnel(
            width_idx=0,
            offset_indices=list(range(1, size)),
        )
        transforms.append(t)
        index_lists.append(list(range(start, start + size)))

    ndim = n_potential + sum(size for _, size in funnel_blocks)
    return block_transform(transforms, index_lists, ndim)
```

- [ ] **Step 4: Run all transform tests**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py -v`

Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add dezess/transforms.py dezess/tests/test_funnel_reparam.py
git commit -m "Add block_transform and multi_funnel helpers"
```

---

### Task 4: Integrate Transform into run_variant

**Files:**
- Modify: `dezess/core/loop.py:99-113` (run_variant signature)
- Modify: `dezess/core/loop.py:404-410` (init log-probs)
- Modify: `dezess/core/loop.py:815-831` (return dict)
- Modify: `dezess/tests/test_funnel_reparam.py`

- [ ] **Step 1: Write test for transform integration**

Append to `dezess/tests/test_funnel_reparam.py`:

```python
def test_run_variant_with_transform():
    """run_variant with NonCenteredFunnel produces correct samples on 10D funnel."""
    from dezess.transforms import non_centered_funnel
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig
    from dezess.targets import neals_funnel

    target = neals_funnel(10)
    t = non_centered_funnel(width_idx=0, offset_indices=list(range(1, 10)))

    config = VariantConfig(
        name="test_ncp",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    )

    key = jax.random.PRNGKey(0)
    init = target.sample(key, 64)

    result = run_variant(
        target.log_prob, init, n_steps=3000,
        config=config, n_warmup=2000,
        transform=t, verbose=False,
    )
    samples = np.array(result["samples"]).reshape(-1, 10)

    # log-width variance should be close to 9 (Neal's funnel prior)
    lw_var = np.var(samples[:, 0])
    assert 4.0 < lw_var < 15.0, f"log-width var={lw_var}, expected ~9"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_run_variant_with_transform -v`

Expected: FAIL — `TypeError: run_variant() got an unexpected keyword argument 'transform'`

- [ ] **Step 3: Add transform parameter to run_variant**

In `dezess/core/loop.py`, make these changes:

1. Add import at top (after existing imports around line 24-26):

```python
from dezess.transforms import Transform, identity
```

2. Add `transform` parameter to `run_variant` signature (line 99-113). Change:

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
) -> dict:
```

to:

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
) -> dict:
```

3. After the config/key initialization (after line 130 `key = jax.random.PRNGKey(0)`), add transform wrapping:

```python
    # --- Transform setup ---
    if transform is not None:
        # Map initial positions to unconstrained space
        init_positions = jax.vmap(transform.inverse)(init_positions)
        # Wrap log_prob to work in z-space: lp(z) = log_prob(forward(z)) + log_det_jac(z)
        _original_log_prob = log_prob_fn
        def log_prob_fn(z, _fwd=transform.forward, _ldj=transform.log_det_jac, _lp=_original_log_prob):
            return _lp(_fwd(z)) + _ldj(z)
```

4. Before the return dict (around line 815-816), map samples back to original space:

```python
    # Map samples back to original space
    if transform is not None:
        all_samples_x = np.zeros_like(all_samples[:n_production])
        _fwd_vmap = jax.jit(jax.vmap(transform.forward))
        for i in range(n_production):
            all_samples_x[i] = np.asarray(_fwd_vmap(jnp.array(all_samples[i])))
        all_samples[:n_production] = all_samples_x
```

The return dict stays the same — `all_samples` now contains x-space samples.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_run_variant_with_transform -v`

Expected: PASS

- [ ] **Step 5: Run existing test suite to check no regressions**

Run: `conda run -n Astro pytest dezess/tests/test_gaussian_moments.py dezess/tests/test_api.py -v`

Expected: All PASS (17 tests)

- [ ] **Step 6: Commit**

```bash
git add dezess/core/loop.py dezess/tests/test_funnel_reparam.py
git commit -m "Integrate Transform into run_variant with z-space sampling"
```

---

### Task 5: Pass Transform Through sample() API

**Files:**
- Modify: `dezess/api.py:69-81` (sample signature)
- Modify: `dezess/api.py:170-183` (run_variant call)
- Modify: `dezess/__init__.py`

- [ ] **Step 1: Write test for sample() with transform**

Append to `dezess/tests/test_funnel_reparam.py`:

```python
def test_sample_api_with_transform():
    """dezess.sample() accepts and passes through transform."""
    import dezess
    from dezess.transforms import non_centered_funnel
    from dezess.targets import neals_funnel

    target = neals_funnel(10)
    t = non_centered_funnel(width_idx=0, offset_indices=list(range(1, 10)))

    key = jax.random.PRNGKey(0)
    init = target.sample(key, 64)

    result = dezess.sample(
        target.log_prob, init, n_samples=1000, n_warmup=1000,
        transform=t, verbose=False,
    )
    assert result.samples.shape[2] == 10
    # Should not crash; basic shape check is enough here
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_sample_api_with_transform -v`

Expected: FAIL — `TypeError: sample() got an unexpected keyword argument 'transform'`

- [ ] **Step 3: Add transform to sample() and __init__.py**

In `dezess/api.py`:

1. Add import (after line 7):

```python
from dezess.transforms import Transform
```

2. Add `transform` parameter to `sample()` signature. Change line 69-80 to:

```python
def sample(
    log_prob_fn: Callable[[Array], Array],
    init_positions: Array,
    n_samples: int = 4000,
    n_warmup: Optional[int] = None,
    target_ess: Optional[float] = None,
    variant: str = "auto",
    n_walkers: Optional[int] = None,
    seed: int = 0,
    verbose: bool = True,
    progress_fn: Optional[Callable] = None,
    transform: Optional[Transform] = None,
    **kwargs,
) -> SampleResult:
```

3. Pass `transform` to `run_variant` call. Change the `run_variant` call (line 170-183) to include `transform=transform`:

```python
    result = run_variant(
        log_prob_fn,
        init_positions,
        n_steps=n_steps,
        config=config,
        n_warmup=n_warmup,
        key=key,
        mu=1.0,
        tune=True,
        target_ess=target_ess,
        progress_fn=progress_fn,
        verbose=verbose,
        transform=transform,
        **kwargs,
    )
```

In `dezess/__init__.py`, add after line 6:

```python
from dezess import transforms
```

And add `"transforms"` to the `__all__` list.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_sample_api_with_transform -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dezess/api.py dezess/__init__.py dezess/tests/test_funnel_reparam.py
git commit -m "Pass transform through sample() API and export transforms module"
```

---

### Task 6: Geweke Invariance Test for Transform

**Files:**
- Modify: `dezess/tests/test_funnel_reparam.py`

- [ ] **Step 1: Write Geweke invariance test with transform**

This test validates the Jacobian is correct by checking one-step invariance: if we draw exact samples from the funnel, apply one sweep of the transformed sampler, the output distribution should still match the funnel.

Append to `dezess/tests/test_funnel_reparam.py`:

```python
def test_geweke_transform_funnel():
    """One-step Geweke invariance with NonCenteredFunnel transform.

    Draw x ~ funnel, transform to z = inverse(x), apply one variant sweep
    in z-space, transform back to x = forward(z). Output should still be
    distributed as the funnel.
    """
    from dezess.transforms import non_centered_funnel
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig
    from dezess.targets import neals_funnel
    from scipy import stats

    ndim = 10
    target = neals_funnel(ndim)
    t = non_centered_funnel(width_idx=0, offset_indices=list(range(1, ndim)))

    config = VariantConfig(
        name="geweke_ncp",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    )

    # Draw exact samples and run 1 production step
    key = jax.random.PRNGKey(123)
    n_walkers = 200
    x_exact = target.sample(key, n_walkers)

    result = run_variant(
        target.log_prob, x_exact, n_steps=1, config=config,
        n_warmup=500, transform=t, verbose=False,
    )
    x_out = np.array(result["samples"][0])  # (n_walkers, ndim)

    # KS test on x_0 (log-width): should be N(0, 9), so x_0/3 ~ N(0,1)
    _, p_val = stats.kstest(x_out[:, 0] / 3.0, "norm")
    assert p_val > 0.001, f"Geweke failed on log-width dim: KS p={p_val:.4f}"
```

- [ ] **Step 2: Run the test**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_geweke_transform_funnel -v`

Expected: PASS (if Jacobian is correct, invariance holds)

- [ ] **Step 3: Commit**

```bash
git add dezess/tests/test_funnel_reparam.py
git commit -m "Add Geweke invariance test for NonCenteredFunnel transform"
```

---

### Task 7: Block-Gibbs Ensemble Module

**Files:**
- Create: `dezess/ensemble/block_gibbs.py`
- Modify: `dezess/ensemble/__init__.py`
- Modify: `dezess/tests/test_funnel_reparam.py`

- [ ] **Step 1: Write test for block-Gibbs on block_coupled_gaussian**

Append to `dezess/tests/test_funnel_reparam.py`:

```python
def test_block_gibbs_block_coupled():
    """Block-Gibbs on block_coupled_gaussian should converge (R-hat < 1.15)."""
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig
    from dezess.targets_stream import block_coupled_gaussian
    from dezess.benchmark.metrics import compute_rhat

    target = block_coupled_gaussian()  # 63D

    config = VariantConfig(
        name="block_gibbs_test",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14]},
    )

    key = jax.random.PRNGKey(0)
    init = target.sample(key, 128)

    result = run_variant(
        target.log_prob, init, n_steps=5000,
        config=config, n_warmup=4000, verbose=False,
    )
    samples = np.array(result["samples"])
    rhat = compute_rhat(samples)
    rhat_max = float(np.max(rhat))
    assert rhat_max < 1.15, f"Block-Gibbs R-hat={rhat_max:.4f} on block_coupled (want < 1.15)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_block_gibbs_block_coupled -v`

Expected: FAIL — block_gibbs ensemble not found

- [ ] **Step 3: Implement block_gibbs.py**

Create `dezess/ensemble/block_gibbs.py`:

```python
"""Block-Gibbs ensemble: cycle slice updates over parameter blocks.

Instead of updating all dimensions with a single direction vector,
sweep over blocks — each block gets its own direction from its
sub-Z-matrix and its own step size mu. Walkers are updated in parallel
within each block via vmap.

This exploits conditional independence: for a posterior with structure
[pot(7), GD1(14), AAU(14), Jet(14), Pal5(14)], the nuisance blocks
are independent given the potential, so each can be sampled with
different step sizes.

The full log_prob is always evaluated (not a conditional), but only
the block's coordinates move. This is correct Gibbs: the full log_prob
restricted to one block IS the conditional distribution for that block.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def parse_blocks(ensemble_kwargs: dict, ndim: int) -> list[jnp.ndarray]:
    """Parse block specification from ensemble_kwargs.

    Accepts either:
        "block_sizes": [7, 14, 14, 14, 14]  (contiguous)
        "blocks": [[0,...,6], [7,...,20], ...]  (explicit)
    """
    if "blocks" in ensemble_kwargs:
        return [jnp.array(b, dtype=jnp.int32) for b in ensemble_kwargs["blocks"]]
    elif "block_sizes" in ensemble_kwargs:
        sizes = ensemble_kwargs["block_sizes"]
        blocks = []
        offset = 0
        for s in sizes:
            blocks.append(jnp.arange(offset, offset + s, dtype=jnp.int32))
            offset += s
        assert offset == ndim, f"block_sizes sum {offset} != ndim {ndim}"
        return blocks
    else:
        raise ValueError("block_gibbs ensemble requires 'block_sizes' or 'blocks' in ensemble_kwargs")


def init_mu_blocks(n_blocks: int, mu_init: float = 1.0) -> jnp.ndarray:
    """Initialize per-block step sizes."""
    return jnp.full(n_blocks, mu_init, dtype=jnp.float64)


def init_temperatures(n_walkers: int) -> Array:
    """All walkers at temperature 1.0 (no tempering for block-Gibbs)."""
    return jnp.ones(n_walkers, dtype=jnp.float64)


def propose_swaps(
    positions: Array,
    log_probs: Array,
    temperatures: Array,
    key: Array,
) -> tuple[Array, Array]:
    """No swaps in block-Gibbs. Returns inputs unchanged."""
    return positions, log_probs
```

Update `dezess/ensemble/__init__.py` — read it first, then add the import.

- [ ] **Step 4: Read and update ensemble __init__.py**

Read `dezess/ensemble/__init__.py` and add `from dezess.ensemble import block_gibbs` if not already importing (or check if __init__.py is empty / just has `__all__`).

- [ ] **Step 5: Register block_gibbs in loop.py strategy registry**

In `dezess/core/loop.py`, add to the ensemble imports (around line 38):

```python
from dezess.ensemble import standard as standard_ensemble, parallel_tempering, block_gibbs
```

Add to `ENSEMBLE_STRATEGIES` dict (around line 82-85):

```python
ENSEMBLE_STRATEGIES = {
    "standard": standard_ensemble,
    "parallel_tempering": parallel_tempering,
    "block_gibbs": block_gibbs,
}
```

- [ ] **Step 6: Add block-Gibbs step function to loop.py**

In `dezess/core/loop.py`, the key change is adding a `block_sweep_step` path. When `config.ensemble == "block_gibbs"`, instead of using `parallel_step` (which updates all dims at once), we build a function that loops over blocks.

Add this after the `_make_step_fn` function definition (after line 391), before `_call_step`:

```python
    # --- Block-Gibbs step function ---
    if config.ensemble == "block_gibbs":
        blocks = block_gibbs.parse_blocks(ens_kwargs, n_dim)
        n_blocks = len(blocks)
        mu_blocks = block_gibbs.init_mu_blocks(n_blocks, float(mu))

        # Pad all blocks to the same size for vmap compatibility
        max_block_size = max(len(b) for b in blocks)
        padded_blocks = jnp.zeros((n_blocks, max_block_size), dtype=jnp.int32)
        block_sizes = jnp.array([len(b) for b in blocks], dtype=jnp.int32)
        for i, b in enumerate(blocks):
            padded_blocks = padded_blocks.at[i, :len(b)].set(b)

        @jax.jit
        def block_sweep_step(positions, log_probs, z_padded, z_count, z_log_probs,
                             mu_blocks, key, temperatures):
            """One full Gibbs sweep over all blocks."""

            def _one_block(carry, block_data):
                positions, log_probs, key = carry
                block_idx, block_size, mu_b = block_data

                # Extract block indices (only first block_size are valid)
                b_idx = padded_blocks[block_idx]

                def _update_one_walker(x_full, lp, w_key):
                    # Draw direction from sub-Z-matrix
                    w_key, k1, k2 = jax.random.split(w_key, 3)
                    idx1 = jax.random.randint(k1, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jax.random.randint(k2, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

                    # Direction in full space but only along block dims
                    z1_block = z_padded[idx1][b_idx[:block_size]]
                    z2_block = z_padded[idx2][b_idx[:block_size]]
                    diff_block = z1_block - z2_block
                    norm = jnp.sqrt(jnp.sum(diff_block ** 2))
                    d_block = diff_block / jnp.maximum(norm, 1e-30)

                    # Embed block direction into full space
                    d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                    d_full = d_full.at[b_idx[:block_size]].set(d_block)

                    # Scale-aware width using block norm
                    dim_corr = jnp.sqrt(jnp.float64(block_size) / 2.0)
                    mu_eff = mu_b * norm / jnp.maximum(dim_corr, 1e-30)

                    # Slice sample along d_full (full log-prob, block movement)
                    from dezess.core.slice_sample import safe_log_prob, slice_sample_fixed
                    lp_x = safe_log_prob(log_prob_fn, x_full)
                    x_new, lp_new, w_key, found, L, R = slice_sample_fixed(
                        lambda x: safe_log_prob(log_prob_fn, x),
                        x_full, d_full, lp_x, mu_eff, w_key,
                        n_expand=n_expand, n_shrink=n_shrink,
                    )
                    bracket_ratio = (R - L) / jnp.maximum(mu_eff, 1e-30)
                    return x_new, lp_new, w_key, bracket_ratio

                # Permute walker keys
                key, k_walkers = jax.random.split(key)
                walker_keys = jax.random.split(k_walkers, n_walkers)

                # Update all walkers in parallel for this block
                new_pos, new_lp, _, bracket_ratios = jax.vmap(
                    _update_one_walker
                )(positions, log_probs, walker_keys)

                return (new_pos, new_lp, key), bracket_ratios

            # Shuffle block order for ergodicity
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, n_blocks)
            block_data = (perm, block_sizes[perm], mu_blocks[perm])

            (positions, log_probs, key), all_br = jax.lax.scan(
                _one_block, (positions, log_probs, key), block_data,
            )

            # Mean bracket ratio across blocks for tuning
            mean_br = jnp.mean(all_br)

            return positions, log_probs, key, mean_br

```

Then, in the warmup loop (starting line 465), add a branch for block_gibbs:

At the start of the warmup section (before line 465), add:

```python
    use_block_gibbs = config.ensemble == "block_gibbs"
```

Inside the warmup loop, replace the step call with a branch. Change the step call block (lines 474-479) to:

```python
        if use_block_gibbs:
            positions, log_probs, key, mean_br = block_sweep_step(
                positions, log_probs, z_padded, z_count, z_log_probs,
                mu_blocks, key, warmup_temperatures,
            )
            # Use mean bracket ratio for tuning
            br = jnp.full(n_walkers, mean_br)
            found = jnp.ones(n_walkers, dtype=jnp.bool_)
            walker_aux_pd = walker_aux.prev_direction
            walker_aux_bw = walker_aux.bracket_widths
            walker_aux_da = walker_aux.direction_anchor
            walker_aux_ds = walker_aux.direction_scale
        else:
            (positions, log_probs, key, found, br,
             walker_aux_pd, walker_aux_bw, walker_aux_da, walker_aux_ds) = _call_step(
                step_fn, positions, log_probs, z_padded, z_count, z_log_probs,
                mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
                walker_aux_ds, warmup_temperatures,
            )
```

And in the mu tuning section (around line 524-528), for block_gibbs tune mu_blocks instead of mu:

```python
                    if use_block_gibbs:
                        mu_blocks = jnp.clip(mu_blocks * adjustment, MU_MIN, MU_MAX)
                    else:
                        mu = jnp.clip(mu * adjustment, MU_MIN, MU_MAX)
```

In the production loop (around line 744-750), add the block_gibbs branch similarly:

```python
            if use_block_gibbs:
                positions, log_probs, key, mean_br = block_sweep_step(
                    positions, log_probs, z_frozen, z_count_frozen, z_lp_frozen,
                    mu_blocks, key, temperatures,
                )
                found = jnp.ones(n_walkers, dtype=jnp.bool_)
                br_arr = jnp.full(n_walkers, mean_br)
            else:
```

And store `found` / `br_arr` appropriately in the single-step branch.

Note: block_gibbs cannot use `lax.scan` batching (each step has random block permutation and sequential block updates), so it always uses the single-step path. Add to the `use_scan` check:

```python
    use_scan = (config.ensemble != "parallel_tempering") and not live_z and not use_block_gibbs
```

- [ ] **Step 7: Run the block-Gibbs test**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_block_gibbs_block_coupled -v --timeout=300`

Expected: PASS (R-hat < 1.15 on block_coupled_gaussian)

- [ ] **Step 8: Run existing tests for regressions**

Run: `conda run -n Astro pytest dezess/tests/test_gaussian_moments.py dezess/tests/test_api.py -v`

Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add dezess/ensemble/block_gibbs.py dezess/ensemble/__init__.py dezess/core/loop.py dezess/tests/test_funnel_reparam.py
git commit -m "Add block-Gibbs ensemble with per-block mu tuning"
```

---

### Task 8: Register New Variants

**Files:**
- Modify: `dezess/benchmark/registry.py`

- [ ] **Step 1: Add new variant configurations**

Append to the `VARIANTS` dict in `dezess/benchmark/registry.py` (before the closing `}`):

```python
    # --- Funnel / block-structure variants ---
    "block_gibbs": VariantConfig(
        name="block_gibbs",
        direction="de_mcz",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14]},
    ),
    "block_gibbs_scale_aware": VariantConfig(
        name="block_gibbs_scale_aware",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14]},
    ),
    "transform_scale_aware": VariantConfig(
        name="transform_scale_aware",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    ),
    "block_gibbs_transform": VariantConfig(
        name="block_gibbs_transform",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14]},
    ),
```

Add to `VARIANT_SETS`:

```python
    "funnel_variants": ["scale_aware", "block_gibbs", "block_gibbs_scale_aware", "transform_scale_aware", "block_gibbs_transform"],
```

- [ ] **Step 2: Commit**

```bash
git add dezess/benchmark/registry.py
git commit -m "Register block-Gibbs and transform variant configurations"
```

---

### Task 9: Funnel 63D Convergence Tests

**Files:**
- Modify: `dezess/tests/test_funnel_reparam.py`

- [ ] **Step 1: Write funnel_63d convergence tests**

Append to `dezess/tests/test_funnel_reparam.py`:

```python
@pytest.mark.slow
def test_transform_funnel_63d():
    """NonCenteredFunnel transform alone on funnel_63d."""
    from dezess.transforms import multi_funnel
    from dezess.core.loop import run_variant
    from dezess.benchmark.registry import VARIANTS
    from dezess.targets_stream import funnel_63d
    from dezess.benchmark.metrics import compute_rhat

    target = funnel_63d()
    t = multi_funnel(n_potential=7, funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)])

    config = VARIANTS["scale_aware"]
    key = jax.random.PRNGKey(0)
    init = target.sample(key, 128)

    result = run_variant(
        target.log_prob, init, n_steps=8000,
        config=config, n_warmup=6000,
        transform=t, verbose=False,
    )
    samples = np.array(result["samples"])
    rhat = compute_rhat(samples)
    rhat_max = float(np.max(rhat))

    # Transform alone should dramatically improve over baseline (3.7 -> ???)
    assert rhat_max < 1.5, f"Transform-only R-hat={rhat_max:.4f} on funnel_63d (want < 1.5)"

    # Check log-width variance for each funnel
    flat = samples.reshape(-1, 63)
    for f_idx in range(4):
        lw_idx = 7 + f_idx * 14
        lw_var = float(np.var(flat[:, lw_idx]))
        assert 4.5 < lw_var < 13.5, (
            f"Funnel {f_idx} log-width var={lw_var:.2f}, want [4.5, 13.5]"
        )


@pytest.mark.slow
def test_block_gibbs_funnel_63d():
    """Block-Gibbs alone on funnel_63d (no transform)."""
    from dezess.core.loop import run_variant
    from dezess.benchmark.registry import VARIANTS
    from dezess.targets_stream import funnel_63d
    from dezess.benchmark.metrics import compute_rhat

    target = funnel_63d()
    config = VARIANTS["block_gibbs_scale_aware"]

    key = jax.random.PRNGKey(0)
    init = target.sample(key, 128)

    result = run_variant(
        target.log_prob, init, n_steps=8000,
        config=config, n_warmup=6000, verbose=False,
    )
    samples = np.array(result["samples"])
    rhat = compute_rhat(samples)
    rhat_max = float(np.max(rhat))

    # Block-Gibbs alone should improve over standard (3.7 -> < 2.5)
    assert rhat_max < 2.5, f"Block-Gibbs R-hat={rhat_max:.4f} on funnel_63d (want < 2.5)"


@pytest.mark.slow
def test_combined_funnel_63d():
    """Combined transform + block-Gibbs on funnel_63d — the main success criterion."""
    from dezess.transforms import multi_funnel
    from dezess.core.loop import run_variant
    from dezess.benchmark.registry import VARIANTS
    from dezess.targets_stream import funnel_63d
    from dezess.benchmark.metrics import compute_rhat

    target = funnel_63d()
    t = multi_funnel(n_potential=7, funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)])
    config = VARIANTS["block_gibbs_transform"]

    key = jax.random.PRNGKey(0)
    init = target.sample(key, 128)

    result = run_variant(
        target.log_prob, init, n_steps=8000,
        config=config, n_warmup=6000,
        transform=t, verbose=False,
    )
    samples = np.array(result["samples"])
    rhat = compute_rhat(samples)
    rhat_max = float(np.max(rhat))

    # SUCCESS CRITERION: R-hat < 1.2
    assert rhat_max < 1.2, f"Combined R-hat={rhat_max:.4f} on funnel_63d (want < 1.2)"

    # Check log-width variance for each funnel
    flat = samples.reshape(-1, 63)
    for f_idx in range(4):
        lw_idx = 7 + f_idx * 14
        lw_var = float(np.var(flat[:, lw_idx]))
        assert 4.5 < lw_var < 13.5, (
            f"Funnel {f_idx} log-width var={lw_var:.2f}, want [4.5, 13.5]"
        )
```

- [ ] **Step 2: Run the convergence tests**

Run: `conda run -n Astro pytest dezess/tests/test_funnel_reparam.py -v -m slow --timeout=600`

Or run individually:

```bash
conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_transform_funnel_63d -v --timeout=300
conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_block_gibbs_funnel_63d -v --timeout=300
conda run -n Astro pytest dezess/tests/test_funnel_reparam.py::test_combined_funnel_63d -v --timeout=300
```

Expected: PASS on all three. If thresholds are too tight, relax based on actual performance.

- [ ] **Step 3: Commit**

```bash
git add dezess/tests/test_funnel_reparam.py
git commit -m "Add funnel_63d convergence tests for transform, block-Gibbs, and combined"
```

---

### Task 10: Benchmark Script and Final Verification

**Files:**
- Create: `bench_funnel_reparam.py`

- [ ] **Step 1: Write the comparison benchmark**

Create `bench_funnel_reparam.py`:

```python
"""Benchmark: transform vs block-Gibbs vs combined on funnel_63d.

Compares all strategies against the baseline to quantify improvement.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dezess.targets_stream import funnel_63d
from dezess.core.loop import run_variant
from dezess.benchmark.registry import VARIANTS
from dezess.benchmark.metrics import compute_ess, compute_rhat
from dezess.transforms import multi_funnel


def run_one(target, variant_name, transform=None, n_walkers=128, n_warmup=6000, n_prod=8000):
    config = VARIANTS[variant_name]
    key = jax.random.PRNGKey(42)
    init = target.sample(key, n_walkers)

    t0 = time.time()
    result = run_variant(
        target.log_prob, init, n_prod, config,
        n_warmup=n_warmup, verbose=False,
        transform=transform,
    )
    samples = np.array(result["samples"])
    wall = time.time() - t0

    ess = compute_ess(samples)
    rhat = compute_rhat(samples)

    flat = samples.reshape(-1, target.ndim)
    lw_vars = []
    for f in range(4):
        lw_idx = 7 + f * 14
        lw_vars.append(float(np.var(flat[:, lw_idx])))

    return {
        "variant": variant_name,
        "ess_min": float(np.min(ess)),
        "ess_mean": float(np.mean(ess)),
        "rhat_max": float(np.max(rhat)),
        "rhat_mean": float(np.mean(rhat)),
        "wall_s": wall,
        "lw_vars": lw_vars,
    }


if __name__ == "__main__":
    target = funnel_63d()
    transform = multi_funnel(
        n_potential=7,
        funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)],
    )

    print(f"Device: {jax.devices()}")
    print(f"Target: funnel_63d (63D, 4 funnels + 7 potential)")
    print(f"Setup: 128 walkers, 6000 warmup, 8000 production")
    print()

    configs = [
        ("scale_aware", None, "Baseline (no transform, no blocks)"),
        ("overrelaxed", None, "Overrelaxed (no transform, no blocks)"),
        ("scale_aware", transform, "Transform only (NCP + scale_aware)"),
        ("block_gibbs_scale_aware", None, "Block-Gibbs only (no transform)"),
        ("block_gibbs_transform", transform, "Combined (NCP + block-Gibbs)"),
    ]

    print(f"{'Strategy':<45s} | {'R-hat max':>9s} | {'ESS min':>8s} | {'ESS mean':>9s} | {'Wall':>6s} | {'lw_var (expect 9)':>20s}")
    print("-" * 120)

    for variant, tfm, label in configs:
        try:
            r = run_one(target, variant, transform=tfm)
            lw_str = " ".join(f"{v:.1f}" for v in r["lw_vars"])
            print(f"{label:<45s} | {r['rhat_max']:9.4f} | {r['ess_min']:8.1f} | {r['ess_mean']:9.1f} | {r['wall_s']:5.1f}s | {lw_str}")
        except Exception as e:
            print(f"{label:<45s} | FAILED: {e}")
```

- [ ] **Step 2: Run the benchmark**

Run: `conda run -n Astro python bench_funnel_reparam.py`

This will take several minutes. The output table shows the comparison.

- [ ] **Step 3: Run full existing test suite for regressions**

Run: `conda run -n Astro pytest dezess/tests/ -v --timeout=600 -k "not slow"`

Expected: All existing tests pass (no regressions from transform/block-gibbs changes).

- [ ] **Step 4: Commit**

```bash
git add bench_funnel_reparam.py
git commit -m "Add funnel reparameterization comparison benchmark"
```

- [ ] **Step 5: Push branch to GitHub**

```bash
git push -u origin feat/funnel-reparam
```
