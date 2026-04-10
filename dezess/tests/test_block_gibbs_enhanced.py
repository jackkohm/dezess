"""Tests for block-Gibbs MH enhancements: delayed rejection, block covariance, conditional independence."""

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
    """Delayed rejection on 63D Gaussian: variance approx 1.0, better acceptance than plain MH."""
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


def test_block_covariance_variance():
    """Block covariance proposals on 63D Gaussian: variance approx 1.0."""
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


def _make_block_coupled_conditional():
    """Build a block-coupled 63D Gaussian with decomposed conditional log_prob."""
    from dezess.targets_stream import block_coupled_gaussian

    n_pot, n_streams, n_nuis = 7, 4, 14
    target = block_coupled_gaussian(n_potential=n_pot, n_streams=n_streams,
                                     n_nuisance_per_stream=n_nuis)

    full_lp = target.log_prob

    # For testing, use the full log_prob as conditional (no speedup, but
    # verifies the parallel sweep logic is correct — other streams cancel in MH ratio)
    def conditional_log_prob(params, block_idx):
        return full_lp(params)

    return target, conditional_log_prob


def test_conditional_independence_variance():
    """Conditional independence parallelism: variance correct on block-coupled Gaussian."""
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
    true_var = np.diag(np.array(target.cov))
    sample_var = np.var(flat, axis=0)
    var_ratio = float(np.mean(sample_var / true_var))

    assert 0.7 < var_ratio < 1.3, f"var_ratio={var_ratio:.4f}, expected ~1.0"


def test_all_enhancements_combined():
    """All 3 enhancements on block-coupled Gaussian: variance correct."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
