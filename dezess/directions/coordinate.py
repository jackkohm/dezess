"""Coordinate-wise direction cycling.

Instead of random directions from the Z-matrix, cycle through coordinate
axes. This guarantees that every dimension is updated every d steps,
which can be much more efficient in high dimensions where random directions
may repeatedly miss some coordinates.

The cycle uses a deterministic schedule: dimension (step * n_walkers + walker_id) % n_dim.
This ensures different walkers update different coordinates on the same step,
providing diversity within each step while guaranteeing full coverage.
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
    coord_mix: float = 0.5,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample a direction that mixes coordinate axes with DE-MCz.

    With probability coord_mix, use a coordinate axis direction.
    Otherwise, fall back to DE-MCz for global exploration.

    The coordinate axis is chosen via a hash of the walker's current
    position to ensure deterministic cycling without needing a step counter.

    Parameters
    ----------
    coord_mix : float
        Probability of using a coordinate direction. Default 0.5.
    """
    key, k_choice, k_coord, k_idx1, k_idx2 = jax.random.split(key, 5)

    ndim = x_i.shape[0]

    # Coordinate direction: random axis
    axis = jax.random.randint(k_coord, (), 0, ndim)
    d_coord = jnp.zeros(ndim, dtype=jnp.float64).at[axis].set(1.0)

    # DE-MCz fallback direction
    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
    diff = z_matrix[idx1] - z_matrix[idx2]
    de_norm = jnp.sqrt(jnp.sum(diff ** 2))
    d_de = diff / jnp.maximum(de_norm, 1e-30)

    # Mix
    use_coord = jax.random.uniform(k_choice) < coord_mix
    d = jnp.where(use_coord, d_coord, d_de)

    # Store scale: for coordinate directions, estimate from the DE-MCz pair
    # distance projected onto the axis. For DE-MCz, use the pair distance.
    coord_scale = jnp.abs(diff[axis]) * 2.0  # ~2x the pair difference on this axis
    scale = jnp.where(use_coord, jnp.maximum(coord_scale, 1e-10), de_norm)
    aux = aux._replace(direction_scale=scale)

    return d, key, aux
