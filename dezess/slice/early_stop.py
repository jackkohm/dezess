"""While-loop slice execution that stops early when the slice is found.

Unlike the fixed fori_loop implementation, this uses jax.lax.while_loop
to stop shrinking as soon as an acceptable point is found. This avoids
wasted log-prob evaluations for walkers that find the slice quickly.

Trade-off: while_loop has slower JIT compilation and can't be as easily
optimized by XLA. Best for expensive log-probs where saving evaluations
matters more than JIT overhead.
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
    n_expand: int = 3,
    n_shrink: int = 12,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Slice sample with early-stopping while_loop for shrinking.

    Stepping-out uses fori_loop (fixed budget, usually fast).
    Shrinking uses while_loop (stops as soon as an acceptable point is found).

    Returns (x_new, lp_new, key, found, L, R).
    """
    key, k_u, k_bracket = jax.random.split(key, 3)
    log_u = lp_x + jnp.log(jax.random.uniform(k_u, dtype=jnp.float64) + 1e-30)

    r0 = jax.random.uniform(k_bracket, dtype=jnp.float64)
    L_init = -r0 * mu
    R_init = L_init + mu

    # Stepping-out: fixed budget (usually 1-2 iterations with scale_aware)
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

    # Shrinking: while_loop that stops as soon as found
    key, k_shrink = jax.random.split(key)

    def _shrink_cond(state):
        _, _, _, _, found, _, n_iter = state
        return (~found) & (n_iter < n_shrink)

    def _shrink_body(state):
        L, R, x_best, lp_best, found, key, n_iter = state
        key, k_prop = jax.random.split(key)
        t = L + jax.random.uniform(k_prop, dtype=jnp.float64) * (R - L)
        x_prop = x + t * d
        lp_prop = safe_log_prob(log_prob_fn, x_prop)

        good = lp_prop > log_u
        x_best = jnp.where(good, x_prop, x_best)
        lp_best = jnp.where(good, lp_prop, lp_best)
        found = good

        # Shrink bracket if not found
        L = jnp.where(~good & (t < 0.0), t, L)
        R = jnp.where(~good & (t >= 0.0), t, R)

        return (L, R, x_best, lp_best, found, key, n_iter + 1)

    init_state = (L, R, x, lp_x, jnp.bool_(False), k_shrink, jnp.int32(0))
    _, _, x_new, lp_new, found, _, _ = lax.while_loop(
        _shrink_cond, _shrink_body, init_state
    )

    x_new = jnp.where(found, x_new, x)
    lp_new = jnp.where(found, lp_new, lp_x)

    return x_new, lp_new, key, found, L, R
