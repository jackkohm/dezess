"""Scale-aware slice width: uses direction_scale from DE-MCz.

Instead of a single global mu, the effective width is proportional to
the norm of (z_r1 - z_r2), which is a natural estimate of the posterior
width along the sampled direction.  Falls back to the tuned global mu
when direction_scale is near zero (e.g. when a non-DE-MCz direction
module is used).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def get_mu(mu: Array, d: Array, aux: Array, scale_factor: float = 1.0,
           min_fraction: float = 0.1, dim_correct: bool = True, **kwargs) -> Array:
    """Return scale_factor * aux.direction_scale, with dimension correction.

    In d dimensions, |z_r1 - z_r2| ~ sqrt(2*d) * sigma, where sigma is the
    per-coordinate posterior scale. But the slice width should be O(sigma),
    not O(sqrt(d) * sigma). When dim_correct=True (default), we divide by
    sqrt(d) to compensate for this distance concentration effect.

    Parameters
    ----------
    scale_factor : float
        Multiplicative factor applied to the (corrected) direction scale.
        Default 1.0.
    min_fraction : float
        Minimum bracket width as a fraction of the tuned global mu.
        Prevents tiny brackets during early warmup. Default 0.1.
    dim_correct : bool
        If True, divide direction_scale by sqrt(n_dim) to correct for
        high-dimensional distance concentration. Default True.
    """
    ds = aux.direction_scale
    if dim_correct:
        ndim = d.shape[0]
        # Correct for concentration of distances: in d dimensions,
        # |z_r1 - z_r2| ~ sqrt(2*d) * sigma_per_coord.
        # We want O(sigma_per_coord), so divide by sqrt(2*d).
        # But the bracket-ratio tuning will also adapt, so use a
        # gentler correction: sqrt(d) / sqrt(2) = sqrt(d/2).
        ds = ds / jnp.sqrt(jnp.maximum(ndim / 2.0, 1.0))
    mu_eff = scale_factor * ds
    mu_floor = min_fraction * mu
    # Use direction_scale when available, floor it to avoid tiny brackets,
    # fall back to global mu when direction_scale is zero
    return jnp.where(aux.direction_scale > 1e-30, jnp.maximum(mu_eff, mu_floor), mu)


def tune_mu(mu: Array, bracket_ratios: Array, aux: Array,
            target_ratio: float = 4.0, **kwargs) -> tuple:
    """Tune the fallback mu (same as scalar). Scale-aware width is applied at get_mu time."""
    med_ratio = jnp.median(bracket_ratios)
    ratio_ema = kwargs.get("ratio_ema", target_ratio)
    ratio_ema = 0.3 * med_ratio + 0.7 * ratio_ema
    adjustment = (ratio_ema / target_ratio) ** 0.5
    new_mu = jnp.clip(mu * adjustment, 1e-8, 1e6)
    return new_mu, aux, ratio_ema
