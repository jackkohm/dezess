"""Local pair direction: DE-MCz with locality-aware pair selection.

Instead of uniform random Z-matrix pairs, selects z_r1 from a
neighborhood of the current walker (using approximate nearest-neighbor
via random projection). The direction z_r1 - z_r2 is then more
locally relevant, capturing the posterior geometry near x_i.

This helps in high dimensions where random pairs may be far from
the current position and produce directions that don't match the
local covariance structure.
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
    n_candidates: int = 10,
    local_mix: float = 0.5,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample a direction using locality-aware pair selection.

    With probability local_mix, select z_r1 as the closest of
    n_candidates random Z-matrix entries to x_i. This gives directions
    that better capture local posterior geometry. Falls back to
    standard DE-MCz with probability (1 - local_mix).

    Parameters
    ----------
    n_candidates : int
        Number of random Z entries to consider for nearest-neighbor. Default 10.
    local_mix : float
        Probability of using local pair selection. Default 0.5.
    """
    key, k_choice, k_cands, k_idx2 = jax.random.split(key, 4)

    # Draw n_candidates random indices
    cand_keys = jax.random.split(k_cands, n_candidates)
    cand_indices = jax.vmap(
        lambda k: jax.random.randint(k, (), 0, z_matrix.shape[0]) % z_count
    )(cand_keys)

    # Find nearest candidate to x_i
    cand_positions = z_matrix[cand_indices]  # (n_candidates, n_dim)
    dists = jnp.sum((cand_positions - x_i) ** 2, axis=1)  # (n_candidates,)
    nearest_idx = cand_indices[jnp.argmin(dists)]

    # Standard random index for z_r1 (fallback)
    random_idx = cand_indices[0]

    # Choose local vs random z_r1
    use_local = jax.random.uniform(k_choice) < local_mix
    idx1 = jnp.where(use_local, nearest_idx, random_idx)

    # z_r2 is always random
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

    diff = z_matrix[idx1] - z_matrix[idx2]
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d = diff / jnp.maximum(norm, 1e-30)

    aux = aux._replace(direction_scale=norm)

    return d, key, aux
