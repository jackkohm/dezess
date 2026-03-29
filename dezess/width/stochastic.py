"""Stochastic slice width: mu ~ LogNormal(log(mu_tuned), sigma).

Draws a fresh mu for each walker at each step, providing robustness
to mis-tuned scale and naturally handling multi-scale problems.

Preserves detailed balance because mu is drawn fresh each step,
independent of x.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def get_mu(mu: Array, d: Array, aux: Array, key: Array = None,
           sigma: float = 0.5, **kwargs) -> Array:
    """Draw mu ~ LogNormal(log(mu_base), sigma).

    Parameters
    ----------
    sigma : float
        Log-scale standard deviation. Higher = more variability.
        sigma=0 recovers deterministic scalar mu.
    """
    if key is None:
        return mu
    log_mu = jnp.log(jnp.maximum(mu, 1e-30))
    noise = jax.random.normal(key, dtype=jnp.float64) * sigma
    return jnp.exp(log_mu + noise)


def tune_mu(mu: Array, bracket_ratios: Array, aux: Array,
            target_ratio: float = 4.0, **kwargs) -> tuple[Array, Array]:
    """Tune the base mu (same as scalar). Stochasticity is applied at get_mu time."""
    med_ratio = jnp.median(bracket_ratios)
    ratio_ema = kwargs.get("ratio_ema", target_ratio)
    ratio_ema = 0.3 * med_ratio + 0.7 * ratio_ema
    adjustment = (ratio_ema / target_ratio) ** 0.5
    new_mu = jnp.clip(mu * adjustment, 1e-8, 1e6)
    return new_mu, aux, ratio_ema
