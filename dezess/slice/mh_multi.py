"""Multi-try MH: draw k proposals along the direction, pick the best accepted.

Each call draws k uniform proposals along direction d, evaluates them all,
then selects the best accepted one (highest log_prob > threshold). Falls
back to current position if none accepted.

Total evals: k per step (vectorised via vmap).
Best used with k=2 or k=3 — gives higher ESS per step than single MH
when the proposal distribution has significant probability outside the slice.

Interface note: k is passed via n_expand (k = n_expand + 1, so n_expand=1 → k=2).
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
    n_expand: int = 2,
    n_shrink: int = 1,
    **kwargs,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Multi-try MH: k = n_expand proposals, pick best accepted. k evals/step.

    Proposes k points x + t_i*d with t_i ~ Uniform(-mu/2, mu/2).
    Selects the proposal with highest log_prob that exceeds log_u (MH threshold).
    Falls back to x if none accepted.

    Evals per step: n_expand (vectorised).
    """
    k = n_expand  # number of proposals
    key, k_props, k_u = jax.random.split(key, 3)

    # Draw k uniform offsets in [-mu/2, mu/2]
    ts = (jax.random.uniform(k_props, shape=(k,), dtype=jnp.float64) - 0.5) * mu
    # Propose k points
    props = x[None, :] + ts[:, None] * d[None, :]   # (k, n_dim)

    # Evaluate all k proposals (vectorised)
    lp_props = jax.vmap(lambda xp: safe_log_prob(log_prob_fn, xp))(props)  # (k,)

    # MH threshold: log u, shared across all proposals for correctness
    log_u = jnp.log(jax.random.uniform(k_u, dtype=jnp.float64) + 1e-30) + lp_x

    # Mask: proposal accepted if lp_prop > log_u
    accepted_mask = lp_props > log_u  # (k,)

    # Among accepted proposals, pick the one with highest log_prob
    masked_lps = jnp.where(accepted_mask, lp_props, jnp.float64(-jnp.inf))
    best_idx = jnp.argmax(masked_lps)
    any_accepted = jnp.any(accepted_mask)

    x_best = props[best_idx]
    lp_best = lp_props[best_idx]

    x_new  = jnp.where(any_accepted, x_best, x)
    lp_new = jnp.where(any_accepted, lp_best, lp_x)

    return x_new, lp_new, key, any_accepted, -mu * jnp.float64(0.5), mu * jnp.float64(0.5)
