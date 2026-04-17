"""Tests for multi-GPU sharding support."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_setup_sharding_single_gpu():
    """n_gpus=1 returns None mesh — no sharding."""
    from dezess.core.sharding import setup_sharding

    result = setup_sharding(n_gpus=1, n_walkers_total=64)
    assert result is None, "Single-GPU should return None (no sharding)"


def test_setup_sharding_multi_gpu_validates_walker_count():
    """Multi-GPU raises if walker count doesn't divide n_gpus evenly."""
    from dezess.core.sharding import setup_sharding

    if len(jax.devices()) < 2:
        pytest.skip("Need >= 2 devices")

    with pytest.raises(ValueError, match="must be divisible"):
        setup_sharding(n_gpus=2, n_walkers_total=63)


def test_setup_sharding_multi_gpu_returns_shardings():
    """Multi-GPU returns mesh + walker_sharding + replicated_sharding."""
    from dezess.core.sharding import setup_sharding

    if len(jax.devices()) < 2:
        pytest.skip("Need >= 2 devices")

    result = setup_sharding(n_gpus=2, n_walkers_total=64)
    assert result is not None
    assert "mesh" in result
    assert "walker_sharding" in result
    assert "replicated" in result


def test_run_variant_default_single_gpu_unchanged():
    """run_variant with defaults (no n_gpus) should behave identically to before."""
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig

    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 5)) * 0.1

    config = VariantConfig(
        name="single_gpu",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
    )

    result = run_variant(log_prob, init, n_steps=500, config=config,
                         n_warmup=200, verbose=False)
    samples = np.array(result["samples"])
    assert samples.shape == (300, 32, 5)


def test_init_positions_sharded_onto_mesh():
    """When n_gpus > 1, init_positions get distributed onto the mesh."""
    from dezess.core.sharding import setup_sharding

    if len(jax.devices()) < 2:
        pytest.skip("Need >= 2 devices")

    sharding_info = setup_sharding(n_gpus=2, n_walkers_total=64)
    init = jax.random.normal(jax.random.PRNGKey(42), (64, 5))

    sharded = jax.device_put(init, sharding_info["walker_sharding"])
    assert sharded.sharding == sharding_info["walker_sharding"]
    addressable = sharded.addressable_data(0).shape
    assert addressable[0] == 32  # 64 / 2 = 32 per device


def test_run_variant_n_gpus_2_recovers_gaussian_variance():
    """End-to-end: 2-GPU run on 10D Gaussian recovers variance ~ 1.0."""
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig

    if len(jax.devices()) < 2:
        pytest.skip("Need >= 2 devices")

    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (64, 10)) * 0.1

    config = VariantConfig(
        name="multi_gpu_test",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
    )

    result = run_variant(log_prob, init, n_steps=2000, config=config,
                         n_warmup=1000, verbose=False,
                         n_gpus=2, n_walkers_per_gpu=32)
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 10)
    mean_var = float(np.var(flat, axis=0).mean())
    assert 0.7 < mean_var < 1.3, f"mean_var={mean_var:.4f}, expected ~1.0"
