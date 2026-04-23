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
