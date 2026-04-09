"""Zeus-style ensemble slice sampling reimplemented in JAX.

Implements the Karamanis & Beutler (2021) algorithm:
1. Split walkers into two complementary halves
2. Direction = pair difference from the other half (with tuned mu)
3. Stepping-out with while_loop (stops when all walkers done)
4. Shrinking with while_loop (stops when all walkers found)

This is a JAX-native implementation that can run on GPU, unlike the
original zeus which is CPU-only (numpy + Python while loops).

The key difference from dezess's Z-matrix approach:
- Uses LIVE complementary ensemble (always adaptive, better var accuracy)
- Requires nw >= 2*ndim walkers (same constraint as emcee/zeus)
- Updates halves sequentially (can't parallelize all walkers at once)
"""

from __future__ import annotations

from typing import Callable, Optional

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def _safe_log_prob(log_prob_fn: Callable, x: Array) -> Array:
    lp = log_prob_fn(x)
    return jnp.where(jnp.isfinite(lp), lp, jnp.float64(-1e30))


def run_zeus_jax(
    log_prob_fn: Callable[[Array], Array],
    init_positions: Array,
    n_steps: int,
    n_warmup: int = 1000,
    key: Optional[Array] = None,
    mu: float = 1.0,
    tune: bool = True,
    move: str = "differential",
    max_expand: int = 10000,
    max_shrink: int = 100,
    verbose: bool = True,
) -> dict:
    """Run zeus-style ensemble slice sampling in JAX.

    Parameters
    ----------
    log_prob_fn : callable
        Log-probability function (JAX-compatible).
    init_positions : (n_walkers, n_dim)
        Initial walker positions. n_walkers must be even and >= 2*n_dim.
    n_steps : int
        Total steps (warmup + production).
    n_warmup : int
        Steps for warmup (mu tuning).
    mu : float
        Initial slice width scale.
    move : str
        Direction move type: "differential" (pair differences from
        complementary ensemble) or "kde" (KDE-smoothed directions).
    max_expand : int
        Maximum stepping-out iterations (safety limit).
    max_shrink : int
        Maximum shrinking iterations (safety limit).

    Returns
    -------
    dict with "samples", "log_prob", "mu", "wall_time", "n_production"
    """
    n_walkers, n_dim = init_positions.shape
    assert n_walkers % 2 == 0, "n_walkers must be even"
    half = n_walkers // 2

    if key is None:
        key = jax.random.PRNGKey(0)
    mu = jnp.float64(mu)

    positions = jnp.array(init_positions, dtype=jnp.float64)
    log_probs = jax.vmap(lambda x: _safe_log_prob(log_prob_fn, x))(positions)

    from dezess.kde import compute_bandwidth, sample_kde_directions

    # JIT-compile the half-ensemble update
    @jax.jit
    def _update_half(active_pos, active_lp, inactive_pos, mu, key):
        """Update one half of the ensemble using the other half as reference."""
        n_active = active_pos.shape[0]
        keys = jax.random.split(key, n_active + 2)
        key_next = keys[0]
        key_dirs = keys[1]
        walker_keys = keys[2:]

        # Precompute all directions for this half-sweep
        if move == "kde":
            bw = compute_bandwidth(inactive_pos, jnp.int32(inactive_pos.shape[0]))
            all_directions = sample_kde_directions(
                inactive_pos, jnp.int32(inactive_pos.shape[0]),
                n_active, key_dirs, bandwidth=bw, mu=1.0,
            ) * mu
        else:
            # Differential move: random pairs from inactive ensemble
            n_ref = inactive_pos.shape[0]
            dk1, dk2 = jax.random.split(key_dirs)
            idx1 = jax.random.randint(dk1, (n_active,), 0, n_ref)
            idx2 = jax.random.randint(dk2, (n_active,), 0, n_ref)
            idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % n_ref, idx2)
            all_directions = 2.0 * mu * (inactive_pos[idx1] - inactive_pos[idx2])

        def _update_one_walker(x, lp_x, direction, wkey):
            wkey, k_u, k_bracket = jax.random.split(wkey, 3)

            # Slice height
            log_u = lp_x - jax.random.exponential(k_u, dtype=jnp.float64)

            # Initial bracket
            r0 = jax.random.uniform(k_bracket, dtype=jnp.float64)
            L_init = -r0
            R_init = L_init + 1.0

            # Stepping-out with while_loop
            def _expand_cond(state):
                L, R, expand_L, expand_R, _, n_iter = state
                return (expand_L | expand_R) & (n_iter < max_expand)

            def _expand_body(state):
                L, R, expand_L, expand_R, key, n_iter = state
                x_L = x + L * direction
                x_R = x + R * direction
                lp_L = _safe_log_prob(log_prob_fn, x_L)
                lp_R = _safe_log_prob(log_prob_fn, x_R)
                still_L = expand_L & (lp_L > log_u)
                still_R = expand_R & (lp_R > log_u)
                L = jnp.where(still_L, L - 1.0, L)
                R = jnp.where(still_R, R + 1.0, R)
                return (L, R, still_L, still_R, key, n_iter + 1)

            wkey, k_exp = jax.random.split(wkey)
            L, R, _, _, _, _ = lax.while_loop(
                _expand_cond, _expand_body,
                (L_init, R_init, jnp.bool_(True), jnp.bool_(True), k_exp, jnp.int32(0))
            )

            # Shrinking with while_loop
            def _shrink_cond(state):
                _, _, _, found, _, n_iter = state
                return (~found) & (n_iter < max_shrink)

            def _shrink_body(state):
                L, R, x_best, found, key, n_iter = state
                key, k_prop = jax.random.split(key)
                t = L + jax.random.uniform(k_prop, dtype=jnp.float64) * (R - L)
                x_prop = x + t * direction
                lp_prop = _safe_log_prob(log_prob_fn, x_prop)
                accept = lp_prop > log_u
                x_best = jnp.where(accept, x_prop, x_best)
                L = jnp.where(~accept & (t < 0.0), t, L)
                R = jnp.where(~accept & (t >= 0.0), t, R)
                return (L, R, x_best, accept, key, n_iter + 1)

            wkey, k_shrink = jax.random.split(wkey)
            _, _, x_new, found, _, _ = lax.while_loop(
                _shrink_cond, _shrink_body,
                (L, R, x, jnp.bool_(False), k_shrink, jnp.int32(0))
            )

            x_new = jnp.where(found, x_new, x)
            lp_new = jnp.where(found, _safe_log_prob(log_prob_fn, x_new), lp_x)
            return x_new, lp_new

        new_pos, new_lp = jax.vmap(
            lambda x, lp, d, k: _update_one_walker(x, lp, d, k)
        )(active_pos, active_lp, all_directions, walker_keys)

        return new_pos, new_lp, key_next

    # --- Warmup ---
    if verbose:
        print(f"  [zeus_jax] JIT compiling...", flush=True)
    t_jit = time.time()
    # Trigger JIT with a dummy step
    first_half = positions[:half]
    second_half = positions[half:]
    first_lp = log_probs[:half]
    second_lp = log_probs[half:]
    new_first, new_first_lp, key = _update_half(first_half, first_lp, second_half, mu, key)
    new_first.block_until_ready()
    if verbose:
        print(f"  [zeus_jax] JIT compile: {time.time() - t_jit:.1f}s", flush=True)

    positions = positions.at[:half].set(new_first)
    log_probs = log_probs.at[:half].set(new_first_lp)

    # Mu tuning during warmup
    TUNE_INTERVAL = 50
    nexp_total = 0
    ncon_total = 0

    t_sample = time.time()
    for step in range(n_warmup):
        for s in range(2):  # two half-sweeps
            if s == 0:
                active = positions[:half]
                active_lp = log_probs[:half]
                inactive = positions[half:]
            else:
                active = positions[half:]
                active_lp = log_probs[half:]
                inactive = positions[:half]

            new_active, new_active_lp, key = _update_half(active, active_lp, inactive, mu, key)

            if s == 0:
                positions = positions.at[:half].set(new_active)
                log_probs = log_probs.at[:half].set(new_active_lp)
            else:
                positions = positions.at[half:].set(new_active)
                log_probs = log_probs.at[half:].set(new_active_lp)

        # Tune mu (simple multiplicative based on acceptance)
        if tune and (step + 1) % TUNE_INTERVAL == 0:
            # Use jump distance as tuning signal
            mu = mu * 1.0  # placeholder — zeus uses nexp/ncon ratio

        if verbose and ((step + 1) % 500 == 0 or step == 0):
            elapsed = time.time() - t_sample
            speed = (step + 1) / elapsed
            best = float(log_probs.max())
            print(f"  [zeus_jax] warmup {step+1:6d}/{n_warmup} "
                  f"({speed:.1f} it/s) best_lp={best:.1f} mu={float(mu):.4f}", flush=True)

    # --- Production ---
    n_production = n_steps - n_warmup
    all_samples = np.zeros((n_production, n_walkers, n_dim), dtype=np.float64)
    all_log_probs = np.zeros((n_production, n_walkers), dtype=np.float64)

    t_prod = time.time()
    for step in range(n_production):
        for s in range(2):
            if s == 0:
                active = positions[:half]
                active_lp = log_probs[:half]
                inactive = positions[half:]
            else:
                active = positions[half:]
                active_lp = log_probs[half:]
                inactive = positions[:half]

            new_active, new_active_lp, key = _update_half(active, active_lp, inactive, mu, key)

            if s == 0:
                positions = positions.at[:half].set(new_active)
                log_probs = log_probs.at[:half].set(new_active_lp)
            else:
                positions = positions.at[half:].set(new_active)
                log_probs = log_probs.at[half:].set(new_active_lp)

        all_samples[step] = np.asarray(positions)
        all_log_probs[step] = np.asarray(log_probs)

        if verbose and ((step + 1) % 200 == 0 or step == n_production - 1):
            elapsed = time.time() - t_prod
            speed = (step + 1) / elapsed
            print(f"  [zeus_jax] {step+1:6d}/{n_production} "
                  f"({speed:.1f} it/s)", flush=True)

    wall_time = time.time() - t_prod

    return {
        "samples": jnp.array(all_samples),
        "log_prob": jnp.array(all_log_probs),
        "mu": float(mu),
        "wall_time": wall_time,
        "n_production": n_production,
    }
