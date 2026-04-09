"""Tests for the high-level dezess API."""

from __future__ import annotations

import os
import tempfile

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dezess

jax.config.update("jax_enable_x64", True)


@jax.jit
def _gaussian_log_prob(x):
    return -0.5 * jnp.sum(x ** 2)


def test_sample_basic():
    """Basic dezess.sample() call."""
    result = dezess.sample(
        _gaussian_log_prob,
        jnp.zeros((16, 5)) + 0.1,
        n_samples=200,
        seed=42,
        verbose=False,
    )
    assert result.samples.shape[1] == 16  # walkers
    assert result.samples.shape[2] == 5   # dim
    assert result.ess_min > 0
    assert result.rhat_max < 100


def test_sample_with_target_ess():
    """Early stopping with target_ess."""
    result = dezess.sample(
        _gaussian_log_prob,
        jnp.zeros((16, 5)) + 0.1,
        n_samples=5000,
        target_ess=200,
        seed=42,
        verbose=False,
    )
    # Should stop early
    assert result.n_steps < 5000


def test_sample_variants():
    """Test named variant presets."""
    for variant in ["auto", "fast", "thorough"]:
        result = dezess.sample(
            _gaussian_log_prob,
            jnp.zeros((16, 5)) + 0.1,
            n_samples=100,
            variant=variant,
            seed=42,
            verbose=False,
        )
        assert result.samples.ndim == 3


def test_sample_bad_input():
    """Validate input errors."""
    with pytest.raises(ValueError, match="2D"):
        dezess.sample(_gaussian_log_prob, jnp.zeros(5), n_samples=10)
    with pytest.raises(ValueError, match="at least 2"):
        dezess.sample(_gaussian_log_prob, jnp.zeros((1, 5)), n_samples=10)


def test_init_walkers():
    """Test walker initialization helper."""
    init = dezess.init_walkers(32, 10, center=np.ones(10), scale=0.5)
    assert init.shape == (32, 10)
    assert np.abs(np.mean(np.array(init)) - 1.0) < 0.5


def test_find_map():
    """Test MAP estimation."""
    mu = jnp.array([1.0, 2.0, -1.0])
    x_map = dezess.find_map(
        lambda x: -0.5 * jnp.sum((x - mu) ** 2),
        jnp.zeros(3),
        n_steps=500,
    )
    assert float(jnp.max(jnp.abs(x_map - mu))) < 0.1


def test_flatten_and_thin():
    """Test post-processing utilities."""
    result = dezess.sample(
        _gaussian_log_prob,
        jnp.zeros((16, 5)) + 0.1,
        n_samples=200,
        seed=42,
        verbose=False,
    )
    flat = dezess.flatten_samples(result.samples)
    assert flat.ndim == 2
    assert flat.shape[1] == 5

    thinned = dezess.thin_samples(result.samples)
    assert thinned.ndim == 3


def test_summary_stats():
    """Test summary statistics."""
    result = dezess.sample(
        _gaussian_log_prob,
        jnp.zeros((16, 5)) + 0.1,
        n_samples=200,
        seed=42,
        verbose=False,
    )
    stats = dezess.summary_stats(result.samples, param_names=["a", "b", "c", "d", "e"])
    assert len(stats["mean"]) == 5
    assert len(stats["param_names"]) == 5


def test_checkpoint_roundtrip():
    """Test save/load checkpoint."""
    result_dict = {
        "samples": np.random.randn(100, 16, 5),
        "log_prob": np.random.randn(100, 16),
        "mu": 1.0,
        "z_matrix": np.random.randn(100, 5),
        "config": None,
    }
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        dezess.save_checkpoint(path, result_dict)
        loaded = dezess.load_checkpoint(path)
        assert loaded["final_positions"].shape == (16, 5)
        assert loaded["mu"] == 1.0
    finally:
        os.unlink(path)


def test_autocorrelation():
    """Test ACF computation."""
    result = dezess.sample(
        _gaussian_log_prob,
        jnp.zeros((16, 5)) + 0.1,
        n_samples=200,
        seed=42,
        verbose=False,
    )
    acf = dezess.autocorrelation(result.samples, max_lag=10, dim=0)
    assert len(acf) == 10

    iat = dezess.integrated_autocorr_time(result.samples, dim=0)
    assert iat >= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
