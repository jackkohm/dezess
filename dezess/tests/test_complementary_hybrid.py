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
