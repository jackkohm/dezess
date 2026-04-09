"""Zeus-style stochastic slice width using direction_scale.

Inspired by the zeus sampler (Karamanis & Beutler 2021), this draws
a random scale factor gamma ~ LogNormal(0, sigma) and multiplies it
by aux.direction_scale.  The stochastic scaling provides robustness
to mis-estimated widths and helps with multi-scale targets.

Falls back to the global mu when direction_scale is near zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def get_mu(mu: Array, d: Array, aux: Array, key: Array = None,
           sigma: float = 1.0, **kwargs) -> Array:
    """Draw gamma ~ LogNormal(0, sigma) and return gamma * direction_scale.

    Parameters
    ----------
    sigma : float
        Standard deviation of the log-normal distribution for gamma.
        Default 1.0.
    key : Array
        JAX PRNG key for drawing the random scale.
    """
    ds = aux.direction_scale

    if key is None:
        # No randomness available; fall back to deterministic scale
        return jnp.where(ds > 1e-30, ds, mu)

    # gamma ~ LogNormal(0, sigma)  i.e.  log(gamma) ~ Normal(0, sigma)
    noise = jax.random.normal(key, dtype=jnp.float64) * sigma
    gamma = jnp.exp(noise)

    mu_eff = gamma * ds
    # Fall back to global mu when direction_scale is near zero
    return jnp.where(ds > 1e-30, mu_eff, mu)


def tune_mu(mu: Array, bracket_ratios: Array, aux: Array,
            target_ratio: float = 4.0, **kwargs) -> tuple:
    """Tune the fallback mu (same as scalar). Zeus gamma width is applied at get_mu time."""
    med_ratio = jnp.median(bracket_ratios)
    ratio_ema = kwargs.get("ratio_ema", target_ratio)
    ratio_ema = 0.3 * med_ratio + 0.7 * ratio_ema
    adjustment = (ratio_ema / target_ratio) ** 0.5
    new_mu = jnp.clip(mu * adjustment, 1e-8, 1e6)
    return new_mu, aux, ratio_ema
