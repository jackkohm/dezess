"""Snooker direction (ter Braak & Vrugt 2008).

Projects the DE-MCz difference vector through the current point,
concentrating proposals toward posterior mass. The direction points
from a random Z-matrix entry through x_i, which biases moves toward
high-density regions while preserving detailed balance via the
Jacobian correction handled in the slice sampling step.
"""

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
    """Snooker direction: project z_r1 through x_i using z_r2 as anchor.

    d = x_i - z_r1 (direction from a random past state toward current position),
    then normalized. This concentrates proposals along the line connecting
    the current position to past states.
    """
    key, k_idx1, k_idx2 = jax.random.split(key, 3)

    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

    z_r1 = z_matrix[idx1]
    z_r2 = z_matrix[idx2]

    # Snooker: direction from z_r1 through x_i
    # This is the line connecting z_r1 to x_i, extended beyond x_i
    diff = x_i - z_r1
    norm = jnp.sqrt(jnp.sum(diff ** 2))

    # Fallback to standard DE-MCz if x_i == z_r1
    fallback = z_r1 - z_r2
    fallback_norm = jnp.sqrt(jnp.sum(fallback ** 2))

    use_snooker = norm > 1e-30
    d_raw = jnp.where(use_snooker, diff, fallback)
    d_norm = jnp.where(use_snooker, norm, fallback_norm)
    d = d_raw / jnp.maximum(d_norm, 1e-30)

    return d, key, aux
