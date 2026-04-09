"""Affine-invariant whitened direction strategy.

Estimates the covariance from the Z-matrix, computes the Cholesky factor,
and draws directions in the whitened space. This makes the sampler
affine-invariant -- it performs identically on any linearly-transformed
target.

The whitening matrix (inverse Cholesky of covariance) is computed once
at the start of production (on the frozen Z-matrix) and cached. During
warmup, falls back to standard DE-MCz.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def compute_whitening_matrix(z_padded: Array, z_count: Array) -> Array:
    """Compute L_inv from the Z-matrix covariance. Returns (n_dim, n_dim) array.

    Steps:
      1. Extract active Z-matrix entries and center them.
      2. Estimate covariance with regularization (1e-6 * I) for stability.
      3. Compute Cholesky factor L such that Cov = L @ L^T.
      4. Invert L to obtain the whitening matrix L_inv.
    """
    active = z_padded[:z_count]
    mean = jnp.mean(active, axis=0)
    centered = active - mean

    n_dim = z_padded.shape[1]
    cov = (centered.T @ centered) / jnp.maximum(z_count - 1, 1)
    cov = cov + 1e-6 * jnp.eye(n_dim, dtype=jnp.float64)

    L = jnp.linalg.cholesky(cov)
    L_inv = jnp.linalg.inv(L)

    return L_inv


def sample_direction(
    x_i: Array,
    z_matrix: Array,
    z_count: Array,
    z_log_probs: Array,
    key: Array,
    aux: Array,
    whitening_matrix: Optional[Array] = None,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample a whitened DE-MCz direction from two random Z-matrix entries.

    If whitening_matrix (L_inv) is provided, computes the whitened difference:
        diff_w = L_inv @ (z_r1 - z_r2)
        d = normalize(diff_w)

    If whitening_matrix is None, falls back to standard DE-MCz
    (normalize raw diff).

    Returns (direction, key, updated_aux).
    """
    key, k_idx1, k_idx2 = jax.random.split(key, 3)

    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

    diff = z_matrix[idx1] - z_matrix[idx2]

    if whitening_matrix is not None:
        diff_w = whitening_matrix @ diff
    else:
        diff_w = diff

    norm = jnp.sqrt(jnp.sum(diff_w ** 2))
    d = diff_w / jnp.maximum(norm, 1e-30)

    # Store raw (unwhitened) scale for scale_aware width compatibility
    raw_norm = jnp.sqrt(jnp.sum(diff ** 2))
    aux = aux._replace(direction_scale=raw_norm)

    return d, key, aux
