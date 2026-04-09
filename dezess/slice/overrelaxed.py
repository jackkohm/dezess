"""Overrelaxed slice sampling (Neal 2003, Section 5.2).

Instead of drawing a random point uniformly from the slice, overrelaxation
places the new point at the reflection of the current point through the
slice midpoint:

    t_new = (L + R) / 2 + ((L + R) / 2 - t_current) = L + R - t_current

Then if t_new is outside the slice, shrink toward the midpoint. This
reduces random walk behavior and can dramatically improve ESS per step
because consecutive samples traverse the slice rather than random walking.

With probability (1 - overrelax_prob), falls back to standard uniform
sampling for ergodicity (pure overrelaxation is deterministic and not
ergodic on its own).
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
    n_expand: int = 2,
    n_shrink: int = 8,
    overrelax_prob: float = 0.8,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Overrelaxed slice sample along direction d.

    With probability overrelax_prob, proposes the reflected point.
    Otherwise does standard uniform slice sampling for ergodicity.

    Returns (x_new, lp_new, key, found, L, R).
    """
    key, k_u, k_bracket, k_choice = jax.random.split(key, 4)
    log_u = lp_x + jnp.log(jax.random.uniform(k_u, dtype=jnp.float64) + 1e-30)

    r0 = jax.random.uniform(k_bracket, dtype=jnp.float64)
    L_init = -r0 * mu
    R_init = L_init + mu

    # Stepping-out (same as fixed)
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

    # Current position in slice coordinates: t_current = 0 (since x = x + 0*d)
    t_current = jnp.float64(0.0)

    # Overrelaxation: reflect through slice midpoint
    use_overrelax = jax.random.uniform(k_choice) < overrelax_prob
    t_mid = (L + R) / 2.0
    t_reflected = L + R - t_current  # = L + R since t_current = 0

    # Shrinking: if reflected point is outside slice, shrink toward midpoint.
    # If not overrelaxing, do standard uniform shrinking.
    def _shrink_step(i, state):
        L, R, x_best, lp_best, found, key = state
        key, k_prop = jax.random.split(key)

        # Overrelaxed: try reflected point first, then shrink toward midpoint
        # Standard: uniform random in [L, R]
        t_overrelax = jnp.where(
            i == 0,
            t_reflected,  # first try: reflected point
            t_mid + (jax.random.uniform(k_prop, dtype=jnp.float64) - 0.5) * (R - L) * 0.5
        )
        t_standard = L + jax.random.uniform(k_prop, dtype=jnp.float64) * (R - L)
        t = jnp.where(use_overrelax, t_overrelax, t_standard)

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
