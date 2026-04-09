"""Momentum-augmented directions.

Maintains a per-walker momentum vector that biases the next direction
toward the previous accepted direction. This creates persistent motion
through narrow corridors (like funnel necks) without needing gradients.

d_new = alpha * d_prev + (1-alpha) * d_random, then re-normalized.

The direction symmetry is maintained because:
1. d_random is symmetric (DE-MCz pair difference)
2. The momentum decays, so after a few steps the direction
   distribution is dominated by the symmetric random component
3. The frozen Z-matrix makes d_random independent of x
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
    alpha: float = 0.3,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Momentum-biased direction: blend previous direction with DE-MCz.

    Parameters
    ----------
    alpha : float
        Momentum coefficient in [0, 1). Higher = more persistent.
        alpha=0 recovers standard DE-MCz.
    """
    key, k_idx1, k_idx2 = jax.random.split(key, 3)

    # Standard DE-MCz random direction
    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
    diff = z_matrix[idx1] - z_matrix[idx2]
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d_random = diff / jnp.maximum(norm, 1e-30)

    # Blend with previous direction (stored in aux.prev_direction)
    prev_d = aux.prev_direction
    prev_norm = jnp.sqrt(jnp.sum(prev_d ** 2))
    has_prev = prev_norm > 1e-20

    d_blended = alpha * prev_d + (1.0 - alpha) * d_random
    d_blended_norm = jnp.sqrt(jnp.sum(d_blended ** 2))
    d_blended = d_blended / jnp.maximum(d_blended_norm, 1e-30)

    # Use blended if we have a valid previous direction
    d = jnp.where(has_prev, d_blended, d_random)

    # Update aux with the new direction for next step
    from dezess.core.types import WalkerAux
    new_aux = WalkerAux(
        prev_direction=d,
        bracket_widths=aux.bracket_widths,
        direction_anchor=aux.direction_anchor,
        direction_scale=aux.direction_scale,
    )

    return d, key, new_aux
