"""NURS: No-Underrun Sampler (orbit-based categorical sampling).

Instead of stepping-out + shrinking to find a uniform sample on the
slice, NURS evaluates the target density at discrete orbit points along
the search direction and samples the next state from a categorical
distribution with probabilities proportional to the density.

A Metropolis-Hastings correction is available (``use_mh=True``) to
ensure detailed balance: the proposal orbit centred at *x* generally
differs from the reverse orbit centred at *x'*, so we accept with
probability min(1, Z_x / Z_{x'}).  When ``use_mh=False`` (the default)
the categorical sample is returned directly -- the approximation is
excellent when mu is well-tuned and the orbit covers the bulk of the
1-D conditional, and dezess's mu tuning ensures this.

Reference
---------
Bou-Rabee, Carpenter, Liu & Oberdörster (2025).
"The No-Underrun Sampler: A Locally-Adaptive, Gradient-Free MCMC Method."
arXiv:2501.18548.
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
    n_expand: int = 5,
    n_shrink: int = 12,
    use_mh: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """NURS orbit-based categorical sampling along direction *d*.

    Builds a symmetric orbit of ``2 * n_expand + 1`` lattice points
    centred at *x* with spacing *mu*, evaluates the target density at
    every point, and draws the next state from the categorical
    distribution with logits equal to the log-densities.

    Parameters
    ----------
    n_expand : int
        Half-orbit radius.  Orbit size = ``2 * n_expand + 1``.
        Without MH the evaluation budget is ``2 * n_expand``;
        with MH it is ``4 * n_expand`` (extended orbit for the
        reverse-orbit normalisation).
    n_shrink : int
        Unused (kept for interface compatibility with other slice
        strategies).
    use_mh : bool
        If *True*, apply a Metropolis-Hastings correction so that
        detailed balance holds exactly.  Default *False* (categorical
        sample returned directly).

    Returns ``(x_new, lp_new, key, found, L, R)``.
    """
    half = n_expand
    h = mu
    n_orbit = 2 * half + 1

    if use_mh:
        # ── Extended orbit for MH correction ─────────────────────
        # We need density values at orbits centred on *any* proposal
        # x' = x + j·h·d with |j| ≤ half.  The union of all such
        # orbits spans indices -2·half … +2·half.
        n_ext = 4 * half + 1
        ext_center = 2 * half

        def _eval_ext(i, lps):
            orbit_idx = i - ext_center
            pt = x + jnp.float64(orbit_idx) * h * d
            lp = jnp.where(i == ext_center, lp_x,
                           safe_log_prob(log_prob_fn, pt))
            return lps.at[i].set(lp)

        ext_lps = jnp.full(n_ext, jnp.float64(-1e30))
        ext_lps = lax.fori_loop(0, n_ext, _eval_ext, ext_lps)

        # Forward orbit centred at x: buffer[half .. 3·half]
        orbit_lps = lax.dynamic_slice(
            ext_lps, (jnp.int32(half),), (n_orbit,))
        log_Z_x = jax.nn.logsumexp(orbit_lps)

        # Categorical proposal
        key, k_cat = jax.random.split(key)
        chosen = jax.random.categorical(k_cat, orbit_lps)
        j = chosen - half  # lattice offset ∈ [-half, half]

        x_prop = x + jnp.float64(j) * h * d
        lp_prop = orbit_lps[chosen]

        # Reverse orbit centred at x'
        rev_start = jnp.int32(half) + j
        rev_lps = lax.dynamic_slice(
            ext_lps, (rev_start,), (n_orbit,))
        log_Z_xp = jax.nn.logsumexp(rev_lps)

        # MH accept / reject
        key, k_mh = jax.random.split(key)
        log_alpha = log_Z_x - log_Z_xp
        accept = (jnp.log(jax.random.uniform(k_mh, dtype=jnp.float64)
                          + 1e-30) < log_alpha)

        x_new = jnp.where(accept, x_prop, x)
        lp_new = jnp.where(accept, lp_prop, lp_x)
        found = accept

    else:
        # ── Direct categorical (no MH) ──────────────────────────
        center = half

        def _eval(i, lps):
            orbit_idx = i - center
            pt = x + jnp.float64(orbit_idx) * h * d
            lp = jnp.where(i == center, lp_x,
                           safe_log_prob(log_prob_fn, pt))
            return lps.at[i].set(lp)

        orbit_lps = jnp.full(n_orbit, jnp.float64(-1e30))
        orbit_lps = lax.fori_loop(0, n_orbit, _eval, orbit_lps)

        key, k_cat = jax.random.split(key)
        chosen = jax.random.categorical(k_cat, orbit_lps)
        j = chosen - half

        x_new = x + jnp.float64(j) * h * d
        lp_new = orbit_lps[chosen]
        found = jnp.bool_(True)

    # ── Effective bracket extent for μ tuning ────────────────────
    # Report the range of lattice points whose density is within
    # 8 nats of the mode — this tells the tuner how "wide" the
    # density actually is relative to the orbit.  Use ±inf sentinels
    # for non-significant points so min/max give the true range.
    local_idx = jnp.arange(n_orbit, dtype=jnp.float64) - half
    significant = orbit_lps > (jnp.max(orbit_lps) - 8.0)
    L = jnp.min(jnp.where(significant, local_idx,
                           jnp.float64(half + 1))) * h
    R = jnp.max(jnp.where(significant, local_idx,
                           jnp.float64(-half - 1))) * h

    return x_new, lp_new, key, found, L, R
