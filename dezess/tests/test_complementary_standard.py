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


def _build_isotropic_gaussian(ndim=21):
    """21D standard Gaussian: variance 1.0 in every dimension."""
    @jax.jit
    def log_prob(x):
        return -0.5 * jnp.sum(x ** 2)
    return log_prob


@pytest.mark.parametrize("cp", [0.5, 1.0])
def test_cp_recovers_isotropic_gaussian_variance(cp):
    """cp=0.5 and cp=1.0 must recover per-dim variance to within 20%
    on an isotropic 21D Gaussian.

    64 walkers chosen deliberately: each half has 32 walkers, so
    complementary differences span up to 31 dimensions — enough for
    a 21D target. With 32 walkers (16 per half, 15D span) cp=1.0
    is rank-deficient and under-covers ~6 dimensions.
    """
    log_prob = _build_isotropic_gaussian(ndim=21)
    init = jax.random.normal(jax.random.PRNGKey(0), (64, 21)) * 0.5

    result = run_variant(
        log_prob, init, n_steps=2000,
        config=_scale_aware(complementary_prob=cp),
        n_warmup=500, key=jax.random.PRNGKey(0), verbose=False,
    )
    samples = np.array(result["samples"]).reshape(-1, 21)
    emp_var = samples.var(axis=0)
    true_var = np.ones(21)
    rel_err = np.abs(emp_var - true_var) / true_var
    assert rel_err.max() < 0.20, (
        f"cp={cp}: worst per-dim variance error {rel_err.max():.2%} > 20%; "
        f"emp_var={emp_var}"
    )
