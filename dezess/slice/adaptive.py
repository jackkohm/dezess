"""Adaptive slice sampling with while_loop for both stepping-out and shrinking.

Like zeus (Karamanis & Beutler 2021), both phases use lax.while_loop
with variable iteration counts. Under vmap, all walkers run until the
slowest one converges — converged walkers are masked but still execute.

The key advantage over fixed-iteration slice: the bracket can extend
arbitrarily far along a DE direction, enabling mode-jumping when the
direction connects two separated modes.

n_expand and n_shrink are safety caps (maximum iterations), not fixed counts.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

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
    n_expand: int = 50,
    n_shrink: int = 50,
    **kwargs,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Adaptive slice sampling along direction d.

    Parameters
    ----------
    n_expand : int
        Maximum stepping-out iterations per side (safety cap).
    n_shrink : int
        Maximum shrinking iterations (safety cap).

    Returns ``(x_new, lp_new, key, found, L, R)``.
    """
    key, k_u, k_bracket = jax.random.split(key, 3)
    log_u = lp_x + jnp.log(jax.random.uniform(k_u, dtype=jnp.float64) + 1e-30)

    r0 = jax.random.uniform(k_bracket, dtype=jnp.float64)
    L_init = -r0 * mu
    R_init = L_init + mu

    # --- Stepping-out: while_loop, expands until log_prob < threshold ---
    def _expand_cond(state):
        _, _, exp_L, exp_R, n_iter = state
        return (exp_L | exp_R) & (n_iter < n_expand)

    def _expand_body(state):
        L, R, exp_L, exp_R, n_iter = state
        lp_L = safe_log_prob(log_prob_fn, x + L * d)
        lp_R = safe_log_prob(log_prob_fn, x + R * d)
        still_L = exp_L & (lp_L > log_u)
        still_R = exp_R & (lp_R > log_u)
        L = jnp.where(still_L, L - mu, L)
        R = jnp.where(still_R, R + mu, R)
        return (L, R, still_L, still_R, n_iter + 1)

    L, R, _, _, _ = lax.while_loop(
        _expand_cond, _expand_body,
        (L_init, R_init, jnp.bool_(True), jnp.bool_(True), jnp.int32(0))
    )

    # --- Shrinking: while_loop, stops when acceptable point found ---
    def _shrink_cond(state):
        _, _, _, found, _, n_iter = state
        return (~found) & (n_iter < n_shrink)

    def _shrink_body(state):
        L, R, x_best, found, key, n_iter = state
        key, k_prop = jax.random.split(key)
        t = L + jax.random.uniform(k_prop, dtype=jnp.float64) * (R - L)
        x_prop = x + t * d
        lp_prop = safe_log_prob(log_prob_fn, x_prop)
        accept = lp_prop > log_u
        x_best = jnp.where(accept, x_prop, x_best)
        L = jnp.where(~accept & (t < 0.0), t, L)
        R = jnp.where(~accept & (t >= 0.0), t, R)
        return (L, R, x_best, accept, key, n_iter + 1)

    key, k_shrink = jax.random.split(key)
    _, _, x_new, found, _, _ = lax.while_loop(
        _shrink_cond, _shrink_body,
        (L, R, x, jnp.bool_(False), k_shrink, jnp.int32(0))
    )

    x_new = jnp.where(found, x_new, x)
    lp_new = jnp.where(found, safe_log_prob(log_prob_fn, x_new), lp_x)

    return x_new, lp_new, key, found, L, R
