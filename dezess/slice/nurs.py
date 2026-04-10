"""NURS: No-Underrun Sampler — faithful implementation.

Implements Algorithm 2 from Bou-Rabee, Carpenter, Liu & Oberdörster (2025),
"The No-Underrun Sampler: A Locally-Adaptive, Gradient-Free MCMC Method",
arXiv:2501.18548.

The algorithm has three phases:

1. **Metropolis-adjusted shift** — randomise the lattice alignment by
   proposing a shift ``s ~ Uniform(-h/2, h/2)`` along the direction and
   accepting via a Metropolis test.  This is essential for the
   reversibility proof (Theorem 1 in the paper).

2. **Orbit doubling** — starting from the (possibly shifted) state,
   iteratively double the orbit forward or backward (coin flip), and
   update the candidate via *streaming categorical selection*.
   Doubling stops when the **no-underrun condition** is met or the
   maximum depth *M* is reached.

3. **Categorical state selection** — the candidate accumulated during
   the streaming process is already a sample from the categorical
   distribution over all orbit points, weighted by target density.

Performance note: all ``2^M`` orbit log-prob evaluations are batched
into a single flat ``fori_loop`` (phase 2a).  The subsequent doubling
logic (phase 2b) operates only on *pre-computed* values and is cheap.
Total evaluations per step: ``2^M + 1`` (orbit + shift proposal).
Use ``n_expand=3`` (8 evals) to match the default slice budget, or
``n_expand=5`` (32 evals) for deeper exploration.
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
    density_threshold: float = 0.001,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """NURS orbit-based categorical sampling along direction *d*.

    Parameters
    ----------
    n_expand : int
        Maximum number of orbit doublings (*M* in the paper).
        The orbit can grow to ``2^n_expand`` lattice points.
        Total log-prob evaluations: ``2^n_expand + 1``.
        Default 3 (8 evals) matches the default slice budget.
    n_shrink : int
        Unused (interface compatibility with other slice strategies).
    density_threshold : float
        Threshold *ε* for the no-underrun stopping condition.
        0.0 disables stopping (orbit always grows to full depth).

    Returns ``(x_new, lp_new, key, found, L, R)``.
    """
    max_doublings = n_expand
    h = mu
    max_orbit = 2 ** max_doublings       # e.g. 8 for M=3
    max_ext = max_orbit // 2             # max batch per doubling

    # ================================================================
    # Phase 1: Metropolis-adjusted shift  (paper Listing 1, lines 1-6)
    # ================================================================
    key, k_shift, k_shift_u = jax.random.split(key, 3)
    s = (jax.random.uniform(k_shift, dtype=jnp.float64) - 0.5) * h
    x_shifted = x + s * d
    lp_shifted = safe_log_prob(log_prob_fn, x_shifted)
    shift_accept = (jnp.log(jax.random.uniform(k_shift_u, dtype=jnp.float64)
                            + 1e-30) < (lp_shifted - lp_x))
    x_start = jnp.where(shift_accept, x_shifted, x)
    lp_start = jnp.where(shift_accept, lp_shifted, lp_x)

    # ================================================================
    # Phase 2a: Pre-compute all orbit log-probs in one flat loop
    # ================================================================
    # Pre-sample M coin flips (True = extend right, False = extend left).
    key, k_flips = jax.random.split(key)
    flips = jax.random.bernoulli(k_flips, shape=(max_doublings,))

    # Compute the orbit extent that *all M doublings* would produce.
    def _scan_bounds(carry, flip):
        left, right = carry
        size = right - left + 1
        new_left = jnp.where(flip, left, left - size).astype(jnp.int32)
        new_right = jnp.where(flip, right + size, right).astype(jnp.int32)
        return (new_left, new_right), None

    (left_final, right_final), _ = lax.scan(
        _scan_bounds, (jnp.int32(0), jnp.int32(0)), flips)

    # Evaluate log-prob at every lattice point in [left_final, right_final].
    def _eval(i, lps):
        orbit_idx = left_final + i
        pt = x_start + jnp.float64(orbit_idx) * h * d
        lp = safe_log_prob(log_prob_fn, pt)
        return lps.at[i].set(lp)

    all_lps = jnp.full(max_orbit, jnp.float64(-1e30))
    all_lps = lax.fori_loop(0, max_orbit, _eval, all_lps)

    # Override the centre point (orbit_idx=0) with the known value.
    center_buf = -left_final              # buffer index of orbit_idx=0
    all_lps = all_lps.at[center_buf].set(lp_start)

    # ================================================================
    # Phase 2b: Simulate doubling on pre-computed values
    # ================================================================
    # The inner loop does cheap array look-ups + streaming categorical;
    # no log_prob_fn calls.

    def _doubling(j, state):
        (left, right, orbit_size, total_lw,
         cand_x, cand_lp, lp_left, lp_right,
         key, stopped) = state

        go_right = flips[j]

        # First new orbit index in the extension.
        ext_orbit_start = jnp.where(
            go_right,
            right + 1,
            left - orbit_size,
        ).astype(jnp.int32)
        ext_buf_start = ext_orbit_start - left_final

        # --- streaming categorical over extension points ---------------
        def _process(i, carry):
            (rlw, cx, clp, elw,
             elp_first, elp_last, k) = carry
            k, k_sel = jax.random.split(k)
            active = (i < orbit_size) & (~stopped)

            # Look up the pre-computed log-prob (cheap).
            buf_idx = jnp.clip(ext_buf_start + i, 0, max_orbit - 1)
            lp = jnp.where(active, all_lps[buf_idx],
                           jnp.float64(-1e30))

            # Track endpoints of the extension.
            elp_first = jnp.where(active & (i == 0), lp, elp_first)
            elp_last = jnp.where(active, lp, elp_last)

            # Extension weight accumulator.
            new_elw = jnp.where(active,
                                jnp.logaddexp(elw, lp), elw)

            # Update running total for the whole orbit so far.
            new_rlw = jnp.where(active,
                                jnp.logaddexp(rlw, lp), rlw)

            # Streaming categorical: accept this point with probability
            # exp(lp) / exp(new_rlw).
            accept_p = jnp.where(
                active & (lp > -1e29),
                jnp.exp(lp - new_rlw), 0.0)
            accept = jax.random.uniform(k_sel, dtype=jnp.float64) < accept_p

            orbit_idx = ext_orbit_start + i
            pt = x_start + jnp.float64(orbit_idx) * h * d
            new_cx = jnp.where(accept, pt, cx)
            new_clp = jnp.where(accept, lp, clp)

            return (new_rlw, new_cx, new_clp, new_elw,
                    elp_first, elp_last, k)

        inner_init = (
            total_lw, cand_x, cand_lp,
            jnp.float64(-1e30),       # ext log-weight
            jnp.float64(-1e30),       # ext first-point lp
            jnp.float64(-1e30),       # ext last-point lp
            key,
        )
        (new_total_lw, new_cx, new_clp,
         ext_lw, ext_lp_first, ext_lp_last,
         key) = lax.fori_loop(0, max_ext, _process, inner_init)

        # --- no-underrun stopping (paper Listing 3) --------------------
        # Going right → new rightmost = ext_lp_last
        # Going left  → new leftmost  = ext_lp_first
        new_lp_left = jnp.where(go_right, lp_left, ext_lp_first)
        new_lp_right = jnp.where(go_right, ext_lp_last, lp_right)
        endpoint_max = jnp.maximum(new_lp_left, new_lp_right)
        log_thresh = (jnp.log(jnp.maximum(density_threshold, 1e-30))
                      + jnp.log(jnp.maximum(h, 1e-30))
                      + new_total_lw)
        underrun_met = endpoint_max <= log_thresh

        # --- simplified sub-stop: reject if extension weight ≈ 0 ------
        ext_frac = jnp.exp(ext_lw - new_total_lw)
        sub_stopped = ext_frac < (density_threshold * 0.1)

        # --- commit or discard the extension ---------------------------
        include = ~stopped & ~sub_stopped
        left = jnp.where(include & ~go_right,
                         left - orbit_size, left)
        right = jnp.where(include & go_right,
                          right + orbit_size, right)
        orbit_size = jnp.where(include,
                               orbit_size * 2, orbit_size)
        total_lw = jnp.where(include, new_total_lw, total_lw)
        cand_x = jnp.where(include, new_cx, cand_x)
        cand_lp = jnp.where(include, new_clp, cand_lp)
        lp_left = jnp.where(include, new_lp_left, lp_left)
        lp_right = jnp.where(include, new_lp_right, lp_right)

        should_stop = include & underrun_met
        stopped = stopped | should_stop | sub_stopped

        return (left, right, orbit_size, total_lw,
                cand_x, cand_lp, lp_left, lp_right,
                key, stopped)

    init_state = (
        jnp.int32(0),            # left
        jnp.int32(0),            # right
        jnp.int32(1),            # orbit_size
        lp_start,                # total_lw
        x_start,                 # cand_x
        lp_start,                # cand_lp
        lp_start,                # lp_left
        lp_start,                # lp_right
        key,                     # PRNG
        jnp.bool_(False),        # stopped
    )

    (left, right, _, _,
     cand_x, cand_lp, _, _,
     key, _) = lax.fori_loop(0, max_doublings, _doubling, init_state)

    # ================================================================
    # Phase 3: return candidate
    # ================================================================
    L = jnp.float64(left) * h
    R = jnp.float64(right) * h

    return cand_x, cand_lp, key, jnp.bool_(True), L, R
