"""Weighted pair selection for Z-matrix directions.

Prefers Z-matrix entries with higher log-probability, concentrating
direction sampling on the high-density region of the posterior.
Uses the Gumbel-max trick for efficient weighted sampling without
the overhead of jax.random.choice's cumulative-sum approach.
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
    temperature: float = 1.0,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample direction from Z-matrix pairs weighted by log-probability.

    Uses the Gumbel-max trick: argmax(log_weight + Gumbel noise)
    gives a sample from the categorical distribution proportional
    to exp(log_weight). This is O(n) and vmap-friendly.

    Parameters
    ----------
    temperature : float
        Controls sharpness: lower = more concentrated on high-prob entries.
    """
    key, k_g1, k_g2 = jax.random.split(key, 3)

    z_max = z_matrix.shape[0]
    active_mask = jnp.arange(z_max) < z_count

    # Log-weights scaled by temperature
    log_weights = z_log_probs / jnp.maximum(temperature, 1e-10)
    # Mask inactive entries
    log_weights = jnp.where(active_mask, log_weights, jnp.float64(-1e30))

    # Gumbel-max trick: argmax(log_w + gumbel_noise) ~ Categorical(softmax(log_w))
    gumbel1 = -jnp.log(-jnp.log(jax.random.uniform(k_g1, (z_max,), dtype=jnp.float64) + 1e-30) + 1e-30)
    gumbel2 = -jnp.log(-jnp.log(jax.random.uniform(k_g2, (z_max,), dtype=jnp.float64) + 1e-30) + 1e-30)

    idx1 = jnp.argmax(log_weights + gumbel1)
    idx2 = jnp.argmax(log_weights + gumbel2)
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

    diff = z_matrix[idx1] - z_matrix[idx2]
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d = diff / jnp.maximum(norm, 1e-30)

    return d, key, aux
