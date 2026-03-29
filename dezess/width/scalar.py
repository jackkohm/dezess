"""Baseline scalar slice width: single mu for all directions."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def get_mu(mu: Array, d: Array, aux: Array, **kwargs) -> Array:
    """Return the scalar slice width (identity function)."""
    return mu


def tune_mu(mu: Array, bracket_ratios: Array, aux: Array,
            target_ratio: float = 4.0, **kwargs) -> tuple[Array, Array]:
    """Tune scalar mu based on median bracket ratio.

    Returns (new_mu, updated_aux).
    """
    med_ratio = jnp.median(bracket_ratios)
    ratio_ema = kwargs.get("ratio_ema", target_ratio)
    ratio_ema = 0.3 * med_ratio + 0.7 * ratio_ema
    adjustment = (ratio_ema / target_ratio) ** 0.5
    new_mu = jnp.clip(mu * adjustment, 1e-8, 1e6)
    return new_mu, aux, ratio_ema
