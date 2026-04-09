"""KDE-smoothed directions from the Z-matrix.

Instead of raw pair differences (DE-MCz), samples directions from a
Gaussian KDE fitted to the Z-matrix. This fills gaps between discrete
archive points, giving smoother and more diverse directions.

The KDE bandwidth is computed once per production run (after warmup)
and passed as a kwarg. Direction generation is O(1) per walker
(just two random index lookups + noise), same as DE-MCz.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dezess.kde import compute_bandwidth

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def compute_kde_bandwidth(z_padded: Array, z_count: Array) -> Array:
    """Compute KDE bandwidth from Z-matrix. Call once after warmup."""
    return compute_bandwidth(z_padded, z_count)


def sample_direction(
    x_i: Array,
    z_matrix: Array,
    z_count: Array,
    z_log_probs: Array,
    key: Array,
    aux: Array,
    kde_bandwidth: Array = None,
    kde_mix: float = 1.0,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample a KDE-smoothed direction from the Z-matrix.

    Draws two points from the Z-matrix KDE (data point + bandwidth noise)
    and returns their normalized difference.

    Parameters
    ----------
    kde_bandwidth : (n_dim,) or None
        Per-dimension KDE bandwidth. If None, falls back to raw DE-MCz.
    kde_mix : float
        Probability of using KDE vs raw DE-MCz. Default 1.0 (always KDE).
    """
    key, k_choice, k_idx1, k_idx2, k_n1, k_n2 = jax.random.split(key, 6)
    n_dim = x_i.shape[0]

    # Raw DE-MCz pair
    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

    p1_raw = z_matrix[idx1]
    p2_raw = z_matrix[idx2]

    if kde_bandwidth is not None:
        # KDE-smoothed: add bandwidth noise
        p1_kde = p1_raw + jax.random.normal(k_n1, (n_dim,), dtype=jnp.float64) * kde_bandwidth
        p2_kde = p2_raw + jax.random.normal(k_n2, (n_dim,), dtype=jnp.float64) * kde_bandwidth

        use_kde = jax.random.uniform(k_choice) < kde_mix
        p1 = jnp.where(use_kde, p1_kde, p1_raw)
        p2 = jnp.where(use_kde, p2_kde, p2_raw)
    else:
        p1 = p1_raw
        p2 = p2_raw

    diff = p1 - p2
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d = diff / jnp.maximum(norm, 1e-30)

    aux = aux._replace(direction_scale=norm)

    return d, key, aux
