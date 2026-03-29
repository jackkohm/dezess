"""Core slice sampling primitives shared across strategies."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def safe_log_prob(log_prob_fn: Callable, x: Array) -> Array:
    """Evaluate log_prob, mapping NaN/Inf to -1e30."""
    lp = log_prob_fn(x)
    return jnp.where(jnp.isfinite(lp), lp, jnp.float64(-1e30))


def slice_sample_fixed(
    log_prob_fn: Callable,
    x: Array,
    d: Array,
    lp_x: Array,
    mu: Array,
    key: Array,
    n_expand: int = 3,
    n_shrink: int = 12,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Slice sample along direction d using fixed-iteration loops.

    Parameters
    ----------
    n_expand : int
        Max stepping-out iterations per side.
    n_shrink : int
        Max shrinking iterations.

    Returns (x_new, lp_new, key, found, L, R).
    """
    key, k_u, k_bracket = jax.random.split(key, 3)
    log_u = lp_x + jnp.log(jax.random.uniform(k_u, dtype=jnp.float64) + 1e-30)

    r0 = jax.random.uniform(k_bracket, dtype=jnp.float64)
    L_init = -r0 * mu
    R_init = L_init + mu

    def _expand_both(i, state):
        L, R, exp_L, exp_R = state
        lp_L = safe_log_prob(log_prob_fn, x + L * d)
        lp_R = safe_log_prob(log_prob_fn, x + R * d)
        should_L = exp_L & (lp_L > log_u)
        should_R = exp_R & (lp_R > log_u)
        L = jnp.where(should_L, L - mu, L)
        R = jnp.where(should_R, R + mu, R)
        return (L, R, exp_L & should_L, exp_R & should_R)

    L, R, _, _ = lax.fori_loop(
        0, n_expand, _expand_both,
        (L_init, R_init, jnp.bool_(True), jnp.bool_(True))
    )

    def _shrink_step(i, state):
        L, R, x_best, lp_best, found, key = state
        key, k_prop = jax.random.split(key)
        t = L + jax.random.uniform(k_prop, dtype=jnp.float64) * (R - L)
        x_prop = x + t * d
        lp_prop = safe_log_prob(log_prob_fn, x_prop)

        good = (lp_prop > log_u) & (~found)
        x_best = jnp.where(good, x_prop, x_best)
        lp_best = jnp.where(good, lp_prop, lp_best)
        found = found | (lp_prop > log_u)

        still_shrinking = ~found
        L = jnp.where(still_shrinking & (t < 0.0), t, L)
        R = jnp.where(still_shrinking & (t >= 0.0), t, R)

        return (L, R, x_best, lp_best, found, key)

    key, k_shrink = jax.random.split(key)
    init_shrink = (L, R, x, lp_x, jnp.bool_(False), k_shrink)
    _, _, x_new, lp_new, found, _ = lax.fori_loop(
        0, n_shrink, _shrink_step, init_shrink
    )

    x_new = jnp.where(found, x_new, x)
    lp_new = jnp.where(found, lp_new, lp_x)

    return x_new, lp_new, key, found, L, R
