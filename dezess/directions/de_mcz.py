"""Baseline DE-MCz direction: uniform random pair from Z-matrix."""

from __future__ import annotations

from typing import Callable

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
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample a DE-MCz direction from two random Z-matrix entries.

    Returns (direction, key, updated_aux).
    """
    key, k_idx1, k_idx2 = jax.random.split(key, 3)

    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

    diff = z_matrix[idx1] - z_matrix[idx2]
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d = diff / jnp.maximum(norm, 1e-30)

    aux = aux._replace(direction_scale=norm)

    return d, key, aux
