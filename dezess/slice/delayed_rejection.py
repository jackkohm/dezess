"""Delayed rejection slice sampling.

When the shrinking loop exhausts its budget (cap-hit / stay-put),
instead of returning x unchanged, tries a second slice along a
different random direction with a smaller mu. This converts expensive
stay-puts into useful moves.

Since the first attempt already evaluated log_prob at several points,
the delayed rejection framework (Tierney & Mira 1999) gives the
correct acceptance probability for the second attempt.

For slice sampling, the second attempt is simply another slice sample
with independent randomness. Because slice sampling has guaranteed
acceptance (within its bracket), we just need to ensure the combined
kernel preserves the target. The key insight: if attempt 1 fails
(found=False), we try attempt 2 with a fresh direction and smaller mu.
The combined kernel is:
  K = p_found_1 * K_1 + (1-p_found_1) * K_2
Both K_1 and K_2 are pi-invariant (Theorem 3 from the proof),
so the mixture is also pi-invariant.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

from dezess.core.slice_sample import slice_sample_fixed, safe_log_prob

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def execute(
    log_prob_fn: Callable,
    x: Array,
    d: Array,
    lp_x: Array,
    mu: Array,
    key: Array,
    n_expand: int = 3,
    n_shrink: int = 12,
    retry_mu_factor: float = 0.3,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Slice sample with delayed rejection on cap-hit.

    Parameters
    ----------
    retry_mu_factor : float
        Scale factor for mu on retry (typically < 1 for smaller steps).
    """
    # First attempt
    x1, lp1, key, found1, L1, R1 = slice_sample_fixed(
        log_prob_fn, x, d, lp_x, mu, key,
        n_expand=n_expand, n_shrink=n_shrink,
    )

    # Second attempt with smaller mu and different direction
    key, k_retry_dir = jax.random.split(key)
    # Rotate direction by ~90 degrees: d2 = d_perp component
    n_dim = x.shape[0]
    noise = jax.random.normal(k_retry_dir, (n_dim,), dtype=jnp.float64)
    # Gram-Schmidt: remove d component from noise
    d2 = noise - jnp.dot(noise, d) * d
    d2_norm = jnp.sqrt(jnp.sum(d2 ** 2))
    d2 = d2 / jnp.maximum(d2_norm, 1e-30)

    mu_retry = mu * retry_mu_factor

    x2, lp2, key, found2, L2, R2 = slice_sample_fixed(
        log_prob_fn, x, d2, lp_x, mu_retry, key,
        n_expand=n_expand, n_shrink=n_shrink,
    )

    # Use retry result only if first attempt failed
    retry_success = (~found1) & found2
    x_out = jnp.where(found1, x1, jnp.where(retry_success, x2, x))
    lp_out = jnp.where(found1, lp1, jnp.where(retry_success, lp2, lp_x))
    found_out = found1 | found2
    L_out = jnp.where(found1, L1, L2)
    R_out = jnp.where(found1, R1, R2)

    return x_out, lp_out, key, found_out, L_out, R_out
