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


def test_gmm_fit_recovers_means():
    """GMM EM on a known 2-component Gaussian recovers means."""
    from dezess.directions.gmm import fit_gmm

    ndim = 5
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    mu1 = jnp.zeros(ndim).at[0].set(-3.0)
    mu2 = jnp.zeros(ndim).at[0].set(3.0)
    samples1 = jax.random.normal(k1, (500, ndim)) + mu1
    samples2 = jax.random.normal(k2, (500, ndim)) + mu2
    samples = jnp.concatenate([samples1, samples2], axis=0)

    means, covs, weights, chols = fit_gmm(
        samples, jnp.int32(1000), n_components=2, n_iter=100,
        key=jax.random.PRNGKey(42))

    order = jnp.argsort(means[:, 0])
    means = means[order]

    assert jnp.abs(means[0, 0] - (-3.0)) < 0.5, f"mean1[0]={means[0,0]:.2f}, expected -3.0"
    assert jnp.abs(means[1, 0] - 3.0) < 0.5, f"mean2[0]={means[1,0]:.2f}, expected 3.0"
    assert jnp.abs(weights[order[0]] - 0.5) < 0.1, f"weight1={weights[order[0]]:.2f}, expected 0.5"


def test_global_move_finds_both_modes():
    """Global move + adaptive slice finds both modes of a bimodal Gaussian."""
    from dezess.targets import gaussian_mixture

    target = gaussian_mixture(ndim=5, separation=6.0)
    key = jax.random.PRNGKey(42)
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

    x0 = flat[:, 0]
    frac_left = float((x0 < 0).mean())
    frac_right = float((x0 > 0).mean())

    assert frac_left > 0.1, f"Only {frac_left:.1%} in left mode, expected >10%"
    assert frac_right > 0.1, f"Only {frac_right:.1%} in right mode, expected >10%"


def test_adaptive_slice_block_gibbs():
    """Adaptive slice works with block-Gibbs."""
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
