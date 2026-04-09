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
           min_fraction: float = 0.1, **kwargs) -> Array:
    """Return scale_factor * aux.direction_scale, with a floor of min_fraction * mu.

    Parameters
    ----------
    scale_factor : float
        Multiplicative factor applied to the direction scale.
        Default 1.0 (use the raw norm).
    min_fraction : float
        Minimum bracket width as a fraction of the tuned global mu.
        Prevents tiny brackets during early warmup when the Z-matrix
        is still clustered near initialization. Default 0.1.
    """
    ds = aux.direction_scale
    mu_eff = scale_factor * ds
    mu_floor = min_fraction * mu
    # Use direction_scale when available, floor it to avoid tiny brackets,
    # fall back to global mu when direction_scale is zero
    return jnp.where(ds > 1e-30, jnp.maximum(mu_eff, mu_floor), mu)


def tune_mu(mu: Array, bracket_ratios: Array, aux: Array,
            target_ratio: float = 4.0, **kwargs) -> tuple[Array, Array]:
    """Tune the fallback mu (same as scalar). Scale-aware width is applied at get_mu time."""
    med_ratio = jnp.median(bracket_ratios)
    ratio_ema = kwargs.get("ratio_ema", target_ratio)
    ratio_ema = 0.3 * med_ratio + 0.7 * ratio_ema
    adjustment = (ratio_ema / target_ratio) ** 0.5
    new_mu = jnp.clip(mu * adjustment, 1e-8, 1e6)
    return new_mu, aux, ratio_ema
