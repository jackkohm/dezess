"""Gaussian moment sanity tests for all sampler variants.

The simplest correctness check: sample from a standard Gaussian and verify
per-dimension variance is close to 1.0. This catches the class of bugs where
the sampler contracts (variance << 1) or explodes (variance >> 1).

The original snooker direction bug (pre-Jacobian fix) produced variance ~0.07
in 20D. This test would have caught it immediately.

Pass criteria:
  - Mean per-dimension variance in [0.8, 1.2] for 20D standard Gaussian
"""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dezess.core.loop import run_variant, DEFAULT_CONFIG
from dezess.core.types import VariantConfig
from dezess.sampler import run_demcz_slice

jax.config.update("jax_enable_x64", True)

NDIM = 20
N_WALKERS = 64
N_STEPS = 5000
N_WARMUP = 2000


@jax.jit
def _std_gaussian_log_prob(x):
    return -0.5 * jnp.sum(x ** 2)


def _run_and_check_variance(config, label="", atol=0.2):
    """Run a variant on a standard Gaussian and check per-dim variance."""
    init = jax.random.normal(jax.random.PRNGKey(42), (N_WALKERS, NDIM)) * 0.1
    result = run_variant(
        _std_gaussian_log_prob,
        init,
        n_steps=N_STEPS,
        config=config,
        n_warmup=N_WARMUP,
        key=jax.random.PRNGKey(43),
        mu=1.0,
        tune=True,
        verbose=False,
    )
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, NDIM)
    var_per_dim = np.var(flat, axis=0)
    mean_var = float(np.mean(var_per_dim))

    print(f"\n  {label}: mean_var={mean_var:.4f} "
          f"(min={np.min(var_per_dim):.4f}, max={np.max(var_per_dim):.4f})")

    assert mean_var > 1.0 - atol, (
        f"{label}: mean variance {mean_var:.4f} < {1.0 - atol} "
        f"(sampler is contracting!)"
    )
    assert mean_var < 1.0 + atol, (
        f"{label}: mean variance {mean_var:.4f} > {1.0 + atol} "
        f"(sampler is exploding!)"
    )
    return mean_var


def test_default_config():
    """The default config must pass the Gaussian moment test."""
    _run_and_check_variance(DEFAULT_CONFIG, "default")


def test_baseline_de_mcz():
    """Baseline DE-MCz with scalar width."""
    config = VariantConfig(
        name="baseline",
        direction="de_mcz",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
    )
    _run_and_check_variance(config, "baseline")


def test_snooker_stochastic():
    """Snooker with Jacobian correction + stochastic width."""
    config = VariantConfig(
        name="snooker_stochastic",
        direction="snooker",
        width="stochastic",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"sigma": 0.5},
    )
    _run_and_check_variance(config, "snooker_stochastic")


def test_scale_aware():
    """Scale-aware width using |z_r1 - z_r2|."""
    config = VariantConfig(
        name="scale_aware",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    )
    _run_and_check_variance(config, "scale_aware")


def test_zeus_gamma():
    """Zeus-style LogNormal-randomized scale-aware width."""
    config = VariantConfig(
        name="zeus_gamma",
        direction="de_mcz",
        width="zeus_gamma",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"sigma": 1.0},
    )
    _run_and_check_variance(config, "zeus_gamma")


def test_local_pair():
    """Local pair direction with scale-aware width."""
    config = VariantConfig(
        name="local_pair_scale",
        direction="local_pair",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"n_candidates": 10, "local_mix": 0.3},
        width_kwargs={"scale_factor": 1.0},
    )
    _run_and_check_variance(config, "local_pair_scale")


def test_nurs():
    """NURS: full algorithm with doubling, shift, and no-underrun stopping."""
    config = VariantConfig(
        name="nurs",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="nurs",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        slice_kwargs={"n_expand": 3, "density_threshold": 0.001},
    )
    _run_and_check_variance(config, "nurs")


def test_nurs_deep():
    """NURS with deeper orbit (max 32 points)."""
    config = VariantConfig(
        name="nurs_deep",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="nurs",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        slice_kwargs={"n_expand": 5, "density_threshold": 0.001},
    )
    _run_and_check_variance(config, "nurs_deep")


def test_nurs_scalar():
    """NURS with scalar (non-scale-aware) width."""
    config = VariantConfig(
        name="nurs_scalar",
        direction="de_mcz",
        width="scalar",
        slice_fn="nurs",
        zmatrix="circular",
        ensemble="standard",
        slice_kwargs={"n_expand": 3, "density_threshold": 0.001},
    )
    _run_and_check_variance(config, "nurs_scalar")


def test_original_sampler():
    """The original run_demcz_slice path must also pass."""
    init = jax.random.normal(jax.random.PRNGKey(42), (N_WALKERS, NDIM)) * 0.1
    result = run_demcz_slice(
        _std_gaussian_log_prob,
        init,
        n_steps=N_STEPS,
        n_warmup=N_WARMUP,
        key=jax.random.PRNGKey(43),
        mu=1.0,
        tune=True,
        verbose=False,
    )
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, NDIM)
    mean_var = float(np.mean(np.var(flat, axis=0)))

    print(f"\n  original_sampler: mean_var={mean_var:.4f}")
    assert mean_var > 0.8, f"Original sampler contracting: {mean_var:.4f}"
    assert mean_var < 1.2, f"Original sampler exploding: {mean_var:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
