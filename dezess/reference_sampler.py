"""Uncapped reference slice sampler for correctness verification.

This is a slow but provably correct implementation of the same algorithm
as sampler.py. It uses Python while-loops (via lax.while_loop) with no
iteration budget -- stepping-out and shrinking run until they naturally
terminate.

The sole purpose of this module is to produce ground-truth outputs for
comparison against the fixed-iteration GPU kernel. It is NOT meant for
production use.

Key differences from sampler.py:
  - Stepping-out: lax.while_loop (unlimited, safety cap 1000)
  - Shrinking: lax.while_loop (unlimited, safety cap 1000)
  - Returns diagnostic info (n_expand_L, n_expand_R, n_shrink, found)
  - Same Z-matrix direction selection, same PRNG key consumption order
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray

SAFETY_CAP = 1000


def _safe_log_prob(log_prob_fn: Callable, x: Array) -> Array:
    """Evaluate log_prob, mapping NaN/Inf to -Inf."""
    lp = log_prob_fn(x)
    return jnp.where(jnp.isfinite(lp), lp, jnp.float64(-1e30))


def reference_slice_sample(
    log_prob_fn: Callable,
    x: Array,           # (n_dim,)
    d: Array,           # (n_dim,) unit direction
    lp_x: Array,        # scalar log_prob(x)
    mu: Array,           # scalar slice width
    key: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Reference (uncapped) slice sample along direction d.

    Uses the same PRNG key consumption order as _slice_sample_fixed in
    sampler.py so that outputs can be compared directly.

    Returns (x_new, lp_new, key, found, n_expand_L, n_expand_R, n_shrink).
    """
    # Slice height -- same key split as sampler.py
    key, k_u, k_bracket = jax.random.split(key, 3)
    log_u = lp_x + jnp.log(jax.random.uniform(k_u, dtype=jnp.float64) + 1e-30)

    # Initial bracket -- same as sampler.py
    r0 = jax.random.uniform(k_bracket, dtype=jnp.float64)
    L_init = -r0 * mu
    R_init = L_init + mu

    # --- Stepping out (interleaved L and R, matching sampler.py) ---
    # sampler.py does fori_loop over _expand_both which evaluates BOTH
    # boundaries each iteration. We replicate this exactly: each iteration
    # evaluates L then R, and conditionally expands.

    def _expand_cond(state):
        L, R, exp_L, exp_R, n_iters = state
        return (exp_L | exp_R) & (n_iters < SAFETY_CAP)

    def _expand_body(state):
        L, R, exp_L, exp_R, n_iters = state
        lp_L = _safe_log_prob(log_prob_fn, x + L * d)
        lp_R = _safe_log_prob(log_prob_fn, x + R * d)
        should_L = exp_L & (lp_L > log_u)
        should_R = exp_R & (lp_R > log_u)
        L = jnp.where(should_L, L - mu, L)
        R = jnp.where(should_R, R + mu, R)
        return (L, R, exp_L & should_L, exp_R & should_R, n_iters + 1)

    init_expand = (L_init, R_init, jnp.bool_(True), jnp.bool_(True), jnp.int32(0))
    L, R, _, _, n_expand_iters = lax.while_loop(_expand_cond, _expand_body, init_expand)

    # --- Shrinking (unlimited, matching sampler.py's key consumption) ---
    # sampler.py's _shrink_step splits key each iteration. We must consume
    # keys in the same order. The key fed into shrinking comes from:
    #   key, k_shrink = jax.random.split(key)
    # Then each iteration does: key, k_prop = jax.random.split(key)

    key, k_shrink = jax.random.split(key)

    def _shrink_cond(state):
        L, R, x_best, lp_best, found, key, n_shrink = state
        return (~found) & (n_shrink < SAFETY_CAP)

    def _shrink_body(state):
        L, R, x_best, lp_best, found, key, n_shrink = state
        key, k_prop = jax.random.split(key)
        t = L + jax.random.uniform(k_prop, dtype=jnp.float64) * (R - L)
        x_prop = x + t * d
        lp_prop = _safe_log_prob(log_prob_fn, x_prop)

        good = (lp_prop > log_u) & (~found)
        x_best = jnp.where(good, x_prop, x_best)
        lp_best = jnp.where(good, lp_prop, lp_best)
        found = found | (lp_prop > log_u)

        # Shrink bracket
        still_shrinking = ~found
        L = jnp.where(still_shrinking & (t < 0.0), t, L)
        R = jnp.where(still_shrinking & (t >= 0.0), t, R)

        return (L, R, x_best, lp_best, found, key, n_shrink + 1)

    init_shrink = (L, R, x, lp_x, jnp.bool_(False), k_shrink, jnp.int32(0))
    _, _, x_new, lp_new, found, _, n_shrink = lax.while_loop(
        _shrink_cond, _shrink_body, init_shrink
    )

    # Stay put if not found (safety cap hit -- should never happen in practice)
    x_new = jnp.where(found, x_new, x)
    lp_new = jnp.where(found, lp_new, lp_x)

    return x_new, lp_new, key, found, n_expand_iters, n_shrink, L, R


def reference_update_one_walker(
    log_prob_fn: Callable,
    x_i: Array,
    lp_i: Array,
    z_matrix: Array,
    z_count: int,
    mu: Array,
    key: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Reference single-walker update matching sampler._update_one_walker.

    Returns (x_new, lp_new, key, found, n_expand_iters, n_shrink).
    """
    # Same key splits as sampler.py
    key, k_idx1, k_idx2 = jax.random.split(key, 3)

    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0])
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0])
    idx1 = idx1 % z_count
    idx2 = idx2 % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
    z_r1 = z_matrix[idx1]
    z_r2 = z_matrix[idx2]

    diff = z_r1 - z_r2
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d = diff / jnp.maximum(norm, 1e-30)

    x_new, lp_new, key, found, n_expand, n_shrink, L, R = reference_slice_sample(
        log_prob_fn, x_i, d, lp_i, mu, key
    )

    return x_new, lp_new, key, found, n_expand, n_shrink
