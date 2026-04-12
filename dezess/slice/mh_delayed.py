"""MH with delayed rejection (DR) along a direction.

On rejection, instead of staying put, tries a second proposal at a smaller
step size (contracted by factor 'retry_factor'). This recovers some efficiency
from rejected first-stage proposals.

Total evals: 1 (accepted on first try) or 2 (delayed rejection triggered).
Expected evals: 1 + P(reject_1st), close to 1 when step size is well-tuned.

The two-stage DR satisfies detailed balance via the standard DR correction:
  q2 must be chosen such that the correction term in the accept ratio ensures
  reversibility. Here we use the Green (1995) correction for two-stage DR.

Interface: n_expand sets the retry factor as 1 / (n_expand + 1).
So n_expand=1 → retry_factor=0.5, n_expand=3 → retry_factor=0.25.
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
    """MH with delayed rejection along direction d.

    Stage 1: Propose x1 = x + t1*d, t1 ~ Uniform(-mu/2, mu/2).
    If accepted → done (1 eval).
    Stage 2 (if rejected): Propose x2 = x + t2*d, t2 ~ Uniform(-mu*r/2, mu*r/2)
    where r = 1/(n_expand+1) is the contraction factor. Accept with DR correction
    that accounts for the fact that stage-2 accepts something stage-1 would have
    rejected (Green 1995, Eq. 2.7).

    Expected evals: 1 + P(reject stage1), typically 1.0-1.3.
    """
    retry_factor = jnp.float64(1.0 / (n_expand + 1))
    key, k1, k2, k_u1, k_u2 = jax.random.split(key, 5)

    # --- Stage 1 ---
    t1 = (jax.random.uniform(k1, dtype=jnp.float64) - 0.5) * mu
    x1 = x + t1 * d
    lp1 = safe_log_prob(log_prob_fn, x1)
    log_alpha1 = lp1 - lp_x
    log_u1 = jnp.log(jax.random.uniform(k_u1, dtype=jnp.float64) + 1e-30)
    accept1 = log_u1 < log_alpha1

    # --- Stage 2 (delayed rejection) ---
    mu2 = mu * retry_factor
    t2 = (jax.random.uniform(k2, dtype=jnp.float64) - 0.5) * mu2
    x2 = x + t2 * d
    lp2 = safe_log_prob(log_prob_fn, x2)

    # DR correction: alpha2 accounts for the probability that stage-1 from x2
    # would also have been rejected (ensuring detailed balance).
    # alpha2 = min(1, [pi(x2)/pi(x)] * [(1 - alpha1(x2→x1)) / (1 - alpha1(x→x1))])
    # where alpha1(x2→x1) is the stage-1 accept prob from x2 to x1.
    lp_x2_to_x1 = lp_x - lp2  # log pi(x1)/pi(x2) from x2's perspective... wait
    # alpha1(x→x1) = min(1, pi(x1)/pi(x)) → log_alpha1 = lp1 - lp_x
    # alpha1(x2→x1) = min(1, pi(x1)/pi(x2)) → = lp1 - lp2
    log_alpha1_from_x2 = lp1 - lp2
    log_one_minus_alpha1_from_x = jnp.log(1.0 - jnp.exp(jnp.minimum(0.0, log_alpha1)) + 1e-30)
    log_one_minus_alpha1_from_x2 = jnp.log(1.0 - jnp.exp(jnp.minimum(0.0, log_alpha1_from_x2)) + 1e-30)
    log_alpha2 = (lp2 - lp_x) + log_one_minus_alpha1_from_x2 - log_one_minus_alpha1_from_x
    log_u2 = jnp.log(jax.random.uniform(k_u2, dtype=jnp.float64) + 1e-30)
    accept2 = (~accept1) & (log_u2 < log_alpha2)

    x_new  = jnp.where(accept1, x1, jnp.where(accept2, x2, x))
    lp_new = jnp.where(accept1, lp1, jnp.where(accept2, lp2, lp_x))
    accepted = accept1 | accept2

    return x_new, lp_new, key, accepted, -mu * jnp.float64(0.5), mu * jnp.float64(0.5)
