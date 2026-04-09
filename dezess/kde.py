"""JAX-native Gaussian Kernel Density Estimation for direction sampling.

Provides fast KDE sampling without scipy dependency. Used by both
zeus_jax (complementary ensemble) and dezess (Z-matrix) for
KDE-smoothed direction generation.

The KDE sampling trick:
  1. Pick a random data point from the reference set
  2. Add Gaussian noise scaled by bandwidth h
  3. Repeat for a second point
  4. Direction = point1 - point2

This gives smoother, more diverse directions than raw pair differences
because the KDE fills gaps between discrete reference points.

Bandwidth uses Scott's rule: h = n^{-1/(d+4)} * std_per_dim
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def compute_bandwidth(data: Array, n_active: Array) -> Array:
    """Compute KDE bandwidth per dimension using Scott's rule.

    h_d = n^{-1/(d+4)} * std_d

    Parameters
    ----------
    data : (n_max, n_dim) padded array
    n_active : scalar, number of active entries

    Returns
    -------
    bandwidth : (n_dim,) per-dimension bandwidth
    """
    n_dim = data.shape[1]
    # Scott's factor
    scott = jnp.power(jnp.maximum(n_active, 2.0).astype(jnp.float64),
                       -1.0 / (n_dim + 4))

    # Per-dimension std (robust: use only active entries via weighted mean)
    # Since data is padded with zeros, compute std from first n_active entries
    # Use a safe approach: compute mean and variance with the full array
    # but weight by active mask
    mask = jnp.arange(data.shape[0]) < n_active
    weights = mask.astype(jnp.float64)
    w_sum = jnp.sum(weights)

    mean = jnp.sum(data * weights[:, None], axis=0) / jnp.maximum(w_sum, 1.0)
    var = jnp.sum((data - mean) ** 2 * weights[:, None], axis=0) / jnp.maximum(w_sum - 1, 1.0)
    std = jnp.sqrt(jnp.maximum(var, 1e-30))

    return scott * std


def sample_kde_directions(
    data: Array,
    n_active: Array,
    n_directions: int,
    key: Array,
    bandwidth: Array = None,
    mu: float = 1.0,
) -> Array:
    """Sample direction vectors from KDE of the reference set.

    Each direction = 2 * mu * (kde_sample_1 - kde_sample_2).

    Parameters
    ----------
    data : (n_max, n_dim) reference points (padded)
    n_active : scalar, number of active entries
    n_directions : int, how many directions to generate
    key : PRNG key
    bandwidth : (n_dim,) or None. If None, computed via Scott's rule.
    mu : scale factor

    Returns
    -------
    directions : (n_directions, n_dim)
    """
    if bandwidth is None:
        bandwidth = compute_bandwidth(data, n_active)

    n_dim = data.shape[1]
    key, k1, k2, k_noise1, k_noise2 = jax.random.split(key, 5)

    # Pick random data points
    idx1 = jax.random.randint(k1, (n_directions,), 0, data.shape[0])
    idx1 = idx1 % jnp.maximum(n_active, 1).astype(jnp.int32)
    idx2 = jax.random.randint(k2, (n_directions,), 0, data.shape[0])
    idx2 = idx2 % jnp.maximum(n_active, 1).astype(jnp.int32)

    points1 = data[idx1]  # (n_directions, n_dim)
    points2 = data[idx2]  # (n_directions, n_dim)

    # Add bandwidth-scaled Gaussian noise
    noise1 = jax.random.normal(k_noise1, (n_directions, n_dim), dtype=jnp.float64) * bandwidth
    noise2 = jax.random.normal(k_noise2, (n_directions, n_dim), dtype=jnp.float64) * bandwidth

    samples1 = points1 + noise1
    samples2 = points2 + noise2

    # Direction = difference of KDE samples
    directions = 2.0 * mu * (samples1 - samples2)

    return directions


def sample_one_kde_direction(
    data: Array,
    n_active: Array,
    key: Array,
    bandwidth: Array,
    mu: float = 1.0,
) -> Array:
    """Sample a single KDE direction. Suitable for vmap over walkers.

    Returns (n_dim,) direction vector.
    """
    n_dim = data.shape[1]
    key, k1, k2, k_n1, k_n2 = jax.random.split(key, 5)

    idx1 = jax.random.randint(k1, (), 0, data.shape[0]) % jnp.maximum(n_active, 1).astype(jnp.int32)
    idx2 = jax.random.randint(k2, (), 0, data.shape[0]) % jnp.maximum(n_active, 1).astype(jnp.int32)

    p1 = data[idx1] + jax.random.normal(k_n1, (n_dim,), dtype=jnp.float64) * bandwidth
    p2 = data[idx2] + jax.random.normal(k_n2, (n_dim,), dtype=jnp.float64) * bandwidth

    return 2.0 * mu * (p1 - p2)
