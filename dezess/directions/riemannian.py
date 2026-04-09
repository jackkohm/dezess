"""Gradient-free Riemannian directions.

Estimates a local metric tensor from k-nearest Z-matrix neighbors,
then samples directions that respect local geometry. This approximates
Riemannian HMC without needing gradients of the target.

Particularly powerful for funnel-type targets where geometry changes
drastically across the space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def sample_direction(
    x_i: Array,
    z_matrix: Array,
    z_count: Array,
    z_log_probs: Array,
    key: Array,
    aux: Array,
    k_neighbors: int = 30,
    riemannian_mix: float = 0.5,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample direction using local covariance of k-nearest Z neighbors.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors to use for local covariance.
    riemannian_mix : float
        Probability of using Riemannian direction vs DE-MCz fallback.
    """
    n_dim = x_i.shape[0]
    key, k_choice, k_normal, k_sign, k_idx1, k_idx2 = jax.random.split(key, 6)

    # Compute distances to all Z-matrix entries
    diffs = z_matrix - x_i[None, :]  # (z_max, n_dim)
    dists = jnp.sum(diffs ** 2, axis=1)  # (z_max,)
    # Mask inactive entries with large distance
    active_mask = jnp.arange(z_matrix.shape[0]) < z_count
    dists = jnp.where(active_mask, dists, jnp.float64(1e30))

    # k-nearest neighbors via top_k on negative distances (O(n) vs O(n log n) argsort)
    k_eff = jnp.minimum(k_neighbors, z_count)
    neg_dists = -dists
    _, neighbor_idx = jax.lax.top_k(neg_dists, k_neighbors)

    # Local covariance from neighbors
    neighbors = z_matrix[neighbor_idx]  # (k, n_dim)
    neighbor_mean = jnp.mean(neighbors, axis=0)
    centered = neighbors - neighbor_mean
    local_cov = centered.T @ centered / jnp.maximum(k_eff - 1, 1).astype(jnp.float64)

    # Add regularization for numerical stability
    local_cov = local_cov + 1e-6 * jnp.eye(n_dim)

    # Cholesky of local covariance for sampling
    L = jnp.linalg.cholesky(local_cov)

    # Sample from local Gaussian: d = L @ z, then normalize
    z_normal = jax.random.normal(k_normal, (n_dim,), dtype=jnp.float64)
    d_riem = L @ z_normal
    d_riem_norm = jnp.sqrt(jnp.sum(d_riem ** 2))
    d_riem = d_riem / jnp.maximum(d_riem_norm, 1e-30)

    # Random sign for symmetry
    sign = 2.0 * jax.random.bernoulli(k_sign).astype(jnp.float64) - 1.0
    d_riem = d_riem * sign

    # DE-MCz fallback
    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
    diff_de = z_matrix[idx1] - z_matrix[idx2]
    norm_de = jnp.sqrt(jnp.sum(diff_de ** 2))
    d_demcz = diff_de / jnp.maximum(norm_de, 1e-30)

    # Mix
    use_riem = jax.random.bernoulli(k_choice, riemannian_mix)
    d = jnp.where(use_riem, d_riem, d_demcz)

    # Store DE-MCz pair distance for scale_aware width compatibility
    aux = aux._replace(direction_scale=norm_de)

    return d, key, aux
