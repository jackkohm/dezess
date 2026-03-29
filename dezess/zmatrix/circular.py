"""Baseline circular buffer Z-matrix management."""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def append(
    z_padded: Array,
    z_count: Array,
    z_log_probs: Array,
    new_positions: Array,
    new_log_probs: Array,
    z_max_size: int,
) -> tuple[Array, Array, Array]:
    """Append new positions to Z-matrix circular buffer.

    Returns (z_padded_new, z_count_new, z_log_probs_new).
    """
    n_walkers = new_positions.shape[0]
    start_idx = z_count % z_max_size
    indices = (jnp.arange(n_walkers) + start_idx) % z_max_size
    z_padded_new = z_padded.at[indices].set(new_positions)
    z_log_probs_new = z_log_probs.at[indices].set(new_log_probs)
    z_count_new = jnp.minimum(z_count + n_walkers, jnp.int32(z_max_size))
    return z_padded_new, z_count_new, z_log_probs_new
