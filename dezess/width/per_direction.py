"""Per-direction adaptive slice width.

Maintains an EMA of accepted bracket widths per dimension, providing
a diagonal scale estimate. When slice sampling along direction d,
the effective mu is the projection of the scale vector onto d.

This handles anisotropic targets where different directions need
different scales.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def get_mu(mu: Array, d: Array, aux: Array, **kwargs) -> Array:
    """Compute direction-dependent mu from per-dimension bracket widths.

    mu_eff = sqrt(d^T @ diag(bracket_widths^2) @ d)
    Falls back to scalar mu if bracket_widths haven't been calibrated yet.
    """
    bw = aux.bracket_widths
    bw_valid = jnp.sum(bw) > 1e-20
    # Project per-dimension scale onto direction
    mu_dir = jnp.sqrt(jnp.sum((d * bw) ** 2))
    mu_dir = jnp.maximum(mu_dir, 1e-8)
    return jnp.where(bw_valid, mu_dir, mu)


def update_bracket_widths(
    aux: Array,
    d: Array,
    L: Array,
    R: Array,
    found: Array,
    ema_alpha: float = 0.1,
) -> Array:
    """Update per-dimension bracket width EMA after a slice sample.

    Projects the accepted bracket width back onto each dimension
    and updates the running average.

    Parameters
    ----------
    ema_alpha : float
        EMA coefficient. Smaller = slower adaptation.
    """
    from dezess.core.types import WalkerAux

    bracket_width = jnp.abs(R - L)
    # Per-dimension contribution: |d_i| * bracket_width
    per_dim = jnp.abs(d) * bracket_width

    old_bw = aux.bracket_widths
    has_old = jnp.sum(old_bw) > 1e-20
    # EMA update, only for dimensions where |d_i| > 0
    new_bw = jnp.where(
        found & has_old,
        ema_alpha * per_dim + (1.0 - ema_alpha) * old_bw,
        jnp.where(found & ~has_old, per_dim, old_bw)
    )

    return WalkerAux(
        prev_direction=aux.prev_direction,
        bracket_widths=new_bw,
    )


def tune_mu(mu: Array, bracket_ratios: Array, aux: Array,
            target_ratio: float = 4.0, **kwargs) -> tuple[Array, Array]:
    """For per-direction width, scalar mu is a fallback. Tune it normally."""
    med_ratio = jnp.median(bracket_ratios)
    ratio_ema = kwargs.get("ratio_ema", target_ratio)
    ratio_ema = 0.3 * med_ratio + 0.7 * ratio_ema
    adjustment = (ratio_ema / target_ratio) ** 0.5
    new_mu = jnp.clip(mu * adjustment, 1e-8, 1e6)
    return new_mu, aux, ratio_ema
