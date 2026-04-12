"""Pure Metropolis-Hastings step along a direction.

Proposes x_new = x + t*d where t ~ Uniform(-mu/2, mu/2) and accepts
with the MH ratio. Exactly 1 log-prob evaluation per step.

This is the minimum possible eval count for a valid MCMC move.
When combined with scale_aware width (mu ≈ natural scale along d),
acceptance is typically 40-60%, near the optimal for random-walk MH.

Compared with fixed slice (2*n_expand + n_shrink evals): each eval
counts more — a single well-calibrated proposal rather than bracketing
and shrinking. Works best when the width adapter is accurate.

n_expand and n_shrink kwargs are accepted but ignored (interface compat).
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from dezess.core.slice_sample import safe_log_prob

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def execute(
    log_prob_fn: Callable,
    x: Array,
    d: Array,
    lp_x: Array,
    mu: Array,
    key: Array,
    n_expand: int = 1,
    n_shrink: int = 1,
    **kwargs,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Pure MH step along direction d. Exactly 1 log-prob eval per call.

    Parameters
    ----------
    mu : scalar
        Proposal half-width. With scale_aware width, this is calibrated
        to the natural scale of the target along d.
    n_expand, n_shrink : ignored (kept for interface compatibility).

    Returns (x_new, lp_new, key, accepted, L, R)
    where L=-mu/2, R=mu/2 are the proposal bounds (for diagnostics).
    """
    key, k_prop, k_accept = jax.random.split(key, 3)

    # Gaussian proposal N(0, mu^2) along direction d.
    # Symmetric proposal cancels from MH ratio (standard random-walk MH).
    # Optimal acceptance for 1-D Gaussian MH ≈ 44% (Roberts & Rosenthal 1998).
    t = jax.random.normal(k_prop, dtype=jnp.float64) * mu
    x_prop = x + t * d
    lp_prop = safe_log_prob(log_prob_fn, x_prop)

    # Metropolis accept/reject (proposal is symmetric → no Hastings correction)
    log_u = jnp.log(jax.random.uniform(k_accept, dtype=jnp.float64) + 1e-30)
    accept = log_u < (lp_prop - lp_x)

    x_new  = jnp.where(accept, x_prop, x)
    lp_new = jnp.where(accept, lp_prop, lp_x)

    return x_new, lp_new, key, accept, -mu * jnp.float64(2.0), mu * jnp.float64(2.0)
