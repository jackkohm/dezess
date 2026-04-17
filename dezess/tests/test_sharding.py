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
