"""Complementary-pair direction: pick j, k from the OTHER half of a positions snapshot.

Used by both the standard ensemble (this module) and the block-Gibbs MH paths
(inlined in dezess/core/loop.py for now). Works with any direction strategy by
substituting at probability `complementary_prob` per step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def sample_complementary_pair(
    x_i: Array,
    positions_snapshot: Array,
    walker_idx,
    key: Array,
    aux,
):
    """Sample DE-MCz pair from the OTHER half of `positions_snapshot`.

    Walkers with index < n_walkers // 2 sample from [half, n_walkers);
    walkers with index >= half sample from [0, half).

    Returns (direction, key, updated_aux) — same signature as
    dezess.directions.de_mcz.sample_direction, so it can be substituted via
    jnp.where at the per-walker level.
    """
    n_walkers = positions_snapshot.shape[0]
    half_size = n_walkers // 2
    my_half = walker_idx >= half_size
    other_lo = jnp.where(my_half, jnp.int32(0), jnp.int32(half_size))
    other_hi = jnp.where(my_half, jnp.int32(half_size), jnp.int32(n_walkers))

    key, k1, k2 = jax.random.split(key, 3)
    j_idx = jax.random.randint(k1, (), other_lo, other_hi)
    k_idx = jax.random.randint(k2, (), other_lo, other_hi)
    # Force k != j by rotating within the other half if collision
    k_idx = jnp.where(
        j_idx == k_idx,
        ((k_idx - other_lo + 1) % half_size) + other_lo,
        k_idx,
    )

    diff = positions_snapshot[j_idx] - positions_snapshot[k_idx]
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d = diff / jnp.maximum(norm, 1e-30)

    aux = aux._replace(direction_scale=norm)
    return d, key, aux
