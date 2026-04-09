"""Snooker direction (ter Braak & Vrugt 2008).

Slice sampling along the line connecting z_r1 (a random archived state)
and the current walker x_i.  Because the line is parameterised radially
from z_r1, the d-dimensional volume element picks up a Jacobian factor
|x - z_r1|^{d-1}.  The caller (loop.py) adds this correction to the
slice log-density so the kernel samples from the correct target.

Requires the Jacobian correction: the anchor z_r1 is returned via
aux.direction_anchor so the caller can compute it.
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
    """Snooker direction: line from z_r1 through x_i.

    d = normalize(x_i - z_r1).  The anchor z_r1 is stored in
    aux.direction_anchor so the caller can apply the Jacobian
    correction (ndim-1)*log|x - z_r1| to the slice density.
    """
    key, k_idx1, k_idx2 = jax.random.split(key, 3)

    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

    z_r1 = z_matrix[idx1]
    z_r2 = z_matrix[idx2]

    # Snooker: direction from z_r1 through x_i
    diff = x_i - z_r1
    norm = jnp.sqrt(jnp.sum(diff ** 2))

    # Fallback to standard DE-MCz if x_i == z_r1
    fallback = z_r1 - z_r2
    fallback_norm = jnp.sqrt(jnp.sum(fallback ** 2))

    use_snooker = norm > 1e-30
    d_raw = jnp.where(use_snooker, diff, fallback)
    d_norm = jnp.where(use_snooker, norm, fallback_norm)
    d = d_raw / jnp.maximum(d_norm, 1e-30)

    # Store anchor for Jacobian correction; fallback anchor is midpoint
    # of z_r1 and z_r2 (safe — Jacobian correction is ~0 when unused)
    anchor = jnp.where(use_snooker, z_r1, (z_r1 + z_r2) / 2.0)
    aux = aux._replace(direction_anchor=anchor)

    return d, key, aux
