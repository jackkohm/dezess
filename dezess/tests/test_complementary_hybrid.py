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

    result_explicit = run_variant(
        log_prob, init, n_steps=300,
        config=_make_bg_mh_dr_config(complementary_prob=0.0),
        n_warmup=100, key=jax.random.PRNGKey(0), verbose=False,
    )

    result_default = run_variant(
        log_prob, init, n_steps=300,
        config=_make_bg_mh_dr_config(),
        n_warmup=100, key=jax.random.PRNGKey(0), verbose=False,
    )

    np.testing.assert_array_equal(
        np.array(result_explicit["samples"]),
        np.array(result_default["samples"]),
        err_msg="complementary_prob=0.0 must be byte-identical to default bg_MH+DR",
    )


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


def test_complementary_helps_with_biased_warmup():
    """When warmup produces a biased Z-matrix, complementary_prob>0 shouldn't hurt."""
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

    config_zmat = _make_bg_mh_dr_config(complementary_prob=0.0)
    config_zmat = config_zmat._replace(
        ensemble_kwargs={**config_zmat.ensemble_kwargs,
                         "block_sizes": [5, 5]}
    )
    result_zmat = run_variant(
        log_prob, init, n_steps=2000, config=config_zmat,
        n_warmup=500, key=jax.random.PRNGKey(0), verbose=False,
    )

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

    final_zmat_mean = np.linalg.norm(samples_zmat[-100:].mean(axis=0))
    final_hybrid_mean = np.linalg.norm(samples_hybrid[-100:].mean(axis=0))

    # Hybrid should not be dramatically worse than pure Z-matrix
    assert final_hybrid_mean < final_zmat_mean + 1.0, \
        f"hybrid={final_hybrid_mean:.2f}, zmat={final_zmat_mean:.2f}"
