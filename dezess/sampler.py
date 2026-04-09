"""
DE-MCz Slice Sampler: fully GPU-parallel ensemble slice sampling.

Combines:
  - DE-MCz Z-matrix directions (ter Braak & Vrugt 2008): ALL walkers
    update simultaneously, no complementary set splitting
  - Slice sampling along the direction (Karamanis & Beutler 2021):
    guaranteed acceptance, good mixing
  - Fixed-iteration fori_loop (no while_loop): fast JIT compilation

Each step, for ALL walkers in parallel:
  1. Draw two past states from the Z-matrix -> direction d
  2. Slice sample along d using fixed-iteration stepping-out + shrinking
  3. (warmup only) Append new positions to the Z-matrix

No while_loops anywhere -- stepping-out uses fori_loop(0, N_EXPAND, ...),
shrinking uses fori_loop(0, N_SHRINK, ...). Both gate on a `done` flag
so extra iterations are no-ops after convergence.

Theory (DE-MCz path, this file):
  The Z-matrix is frozen after warmup. Conditioned on a fixed Z,
  each walker update uses d = normalize(z_r1 - z_r2), a direction
  independent of the walker being updated, so the fully parallel
  sweep leaves the product target invariant.

Theory (snooker path, dezess/directions/snooker.py):
  The snooker direction d = normalize(x_i - z_r1) IS state-dependent.
  Correctness follows from a different argument: the line through z_r1
  and x_i is parameterised radially from z_r1, so the 1D conditional
  includes a Jacobian factor |x - z_r1|^{d-1}. This correction is
  applied to the slice log-density in core/loop.py. Without it the
  sampler contracts severely (variance ~0.07 instead of ~1.0).

Fixed-iteration slice: exact when the slice is found within budget;
otherwise the chain performs a self-transition (the shrinkage procedure
is symmetric so the truncated kernel preserves detailed balance).

This gives: slice sampling quality + DE-MCz parallelism + fast compilation.

References:
  - ter Braak & Vrugt (2008), "Differential Evolution Markov Chain
    with snooker updater and fewer chains"
  - Karamanis & Beutler (2021), "Ensemble Slice Sampling: Parallel,
    black-box and gradient-free inference for correlated & multimodal
    distributions", arXiv:2002.06212
  - Neal (2003), "Slice Sampling", Annals of Statistics 31(3):705-767
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray

# Fixed iteration counts -- upper bounds. Actual work stops early via
# conditional masking. With well-tuned mu, most walkers need 0-2
# expansions and find an acceptable point in 2-5 shrink iterations.
# Every iteration evaluates log_prob regardless (JAX can't short-circuit),
# so keeping these small directly controls cost per step.
N_EXPAND = 3     # max stepping-out iterations per side (3 * mu expansion)
N_SHRINK = 12    # max shrinking iterations (12 halvings -> precision ~ mu * 2^-12)


def _safe_log_prob(log_prob_fn: Callable, x: Array) -> Array:
    """Evaluate log_prob, mapping NaN/Inf to -Inf."""
    lp = log_prob_fn(x)
    return jnp.where(jnp.isfinite(lp), lp, jnp.float64(-1e30))


def _slice_sample_fixed(
    log_prob_fn: Callable,
    x: Array,           # (n_dim,) current position
    d: Array,           # (n_dim,) unit direction
    lp_x: Array,        # scalar log_prob(x)
    mu: Array,           # scalar slice width
    key: Array,
) -> tuple[Array, Array, Array, Array]:
    """Slice sample along direction d using fixed-iteration loops.

    Returns (x_new, lp_new, key, found).
    """
    # Slice height
    key, k_u, k_bracket = jax.random.split(key, 3)
    log_u = lp_x + jnp.log(jax.random.uniform(k_u, dtype=jnp.float64) + 1e-30)

    # Initial bracket
    r0 = jax.random.uniform(k_bracket, dtype=jnp.float64)
    L_init = -r0 * mu
    R_init = L_init + mu

    # --- Stepping out: fixed N_EXPAND iterations per side ---
    # Combined left+right expansion in a single fori_loop.
    def _expand_both(i, state):
        L, R, exp_L, exp_R = state
        lp_L = _safe_log_prob(log_prob_fn, x + L * d)
        lp_R = _safe_log_prob(log_prob_fn, x + R * d)
        should_L = exp_L & (lp_L > log_u)
        should_R = exp_R & (lp_R > log_u)
        L = jnp.where(should_L, L - mu, L)
        R = jnp.where(should_R, R + mu, R)
        return (L, R, exp_L & should_L, exp_R & should_R)

    L, R, _, _ = lax.fori_loop(0, N_EXPAND, _expand_both,
                                (L_init, R_init, jnp.bool_(True), jnp.bool_(True)))

    # --- Shrinking: fixed N_SHRINK iterations ---
    # Propose uniformly in [L, R], accept if above slice, else shrink bracket.
    # Once accepted, `found` flag gates all further iterations as no-ops.
    def _shrink_step(i, state):
        L, R, x_best, lp_best, found, key = state
        key, k_prop = jax.random.split(key)
        t = L + jax.random.uniform(k_prop, dtype=jnp.float64) * (R - L)
        x_prop = x + t * d
        lp_prop = _safe_log_prob(log_prob_fn, x_prop)

        good = (lp_prop > log_u) & (~found)
        x_best = jnp.where(good, x_prop, x_best)
        lp_best = jnp.where(good, lp_prop, lp_best)
        found = found | (lp_prop > log_u)

        # Shrink bracket (only if not yet found)
        still_shrinking = ~found
        L = jnp.where(still_shrinking & (t < 0.0), t, L)
        R = jnp.where(still_shrinking & (t >= 0.0), t, R)

        return (L, R, x_best, lp_best, found, key)

    key, k_shrink = jax.random.split(key)
    init_shrink = (L, R, x, lp_x, jnp.bool_(False), k_shrink)
    _, _, x_new, lp_new, found, _ = lax.fori_loop(0, N_SHRINK, _shrink_step, init_shrink)

    # If not found after N_SHRINK iterations (extremely rare), stay put
    x_new = jnp.where(found, x_new, x)
    lp_new = jnp.where(found, lp_new, lp_x)

    return x_new, lp_new, key, found, L, R


def _update_one_walker(
    log_prob_fn: Callable,
    x_i: Array,
    lp_i: Array,
    z_matrix: Array,
    z_count: int,
    mu: Array,
    key: Array,
) -> tuple[Array, Array, Array, Array]:
    """Update one walker: pick Z-matrix direction, slice sample along it.

    Returns (x_new, lp_new, key, found).
    """
    key, k_idx1, k_idx2 = jax.random.split(key, 3)

    # Draw two distinct past states from Z-matrix
    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0])
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0])
    idx1 = idx1 % z_count
    idx2 = idx2 % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
    z_r1 = z_matrix[idx1]
    z_r2 = z_matrix[idx2]

    # Direction from Z-matrix pair (normalized)
    diff = z_r1 - z_r2
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d = diff / jnp.maximum(norm, 1e-30)

    # Slice sample along this direction
    x_new, lp_new, key, found, L, R = _slice_sample_fixed(log_prob_fn, x_i, d, lp_i, mu, key)

    # Bracket ratio: how many mu-widths the bracket spans.
    # 1 = no expansion, 2*N_EXPAND+1 = maxed out.
    bracket_ratio = (R - L) / jnp.maximum(mu, 1e-30)

    return x_new, lp_new, key, found, bracket_ratio


def run_demcz_slice(
    log_prob_fn: Callable[[Array], Array],
    init_positions: Array,
    n_steps: int,
    n_warmup: int = 0,
    key: Optional[Array] = None,
    mu: float = 1.0,
    tune: bool = True,
    z_max_size: int = 50000,
    z_initial: Optional[Array] = None,
    verbose: bool = True,
) -> dict:
    """Run the DE-MCz Slice Sampler.

    Parameters
    ----------
    log_prob_fn : callable
        Log-probability: (n_dim,) -> scalar. Must be JAX-compatible.
    init_positions : (n_walkers, n_dim)
        Initial walker positions.
    n_steps : int
        Total steps (warmup + production).
    n_warmup : int
        Steps to discard as burn-in. Z-matrix grows during warmup,
        then is frozen for production.
    key : JAX PRNG key
    mu : float
        Slice width scale factor.
    tune : bool
        Auto-tune mu during warmup based on bracket expansion ratio.
        Targets ~1-2 expansions per side (bracket_ratio ~ N_EXPAND+1).
    z_max_size : int
        Maximum Z-matrix size (circular buffer).
    z_initial : optional array (M, n_dim)
        Pre-populate Z-matrix (e.g., from a previous run).
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        "samples": (n_production, n_walkers, n_dim) array
        "log_prob": (n_production, n_walkers) array
        "mu": float
        "z_matrix": frozen Z-matrix used during production
    """
    n_walkers, n_dim = init_positions.shape
    if key is None:
        key = jax.random.PRNGKey(0)
    mu = jnp.float64(mu)

    # Initialize Z-matrix
    z_padded = jnp.zeros((z_max_size, n_dim), dtype=jnp.float64)
    if z_initial is not None:
        n_z = min(z_initial.shape[0], z_max_size)
        z_padded = z_padded.at[:n_z].set(z_initial[:n_z])
        z_count = jnp.int32(n_z)
        if verbose:
            print(f"  [dezess] Z-matrix pre-populated with {n_z} samples", flush=True)
    else:
        z_padded = z_padded.at[:n_walkers].set(init_positions)
        z_count = jnp.int32(n_walkers)

    # --- JIT-compile the parallel step ---
    @jax.jit
    def parallel_step(positions, log_probs, z_padded, z_count, mu, key):
        """One step: update all walkers in parallel via Z-matrix slice sampling."""
        keys = jax.random.split(key, n_walkers + 1)
        key_next = keys[0]
        walker_keys = keys[1:]

        new_positions, new_log_probs, _, found, bracket_ratios = jax.vmap(
            lambda x, lp, k: _update_one_walker(
                log_prob_fn, x, lp, z_padded, z_count, mu, k
            )
        )(positions, log_probs, walker_keys)

        # Batch-append new positions to Z-matrix (circular buffer)
        start_idx = z_count % z_max_size
        indices = (jnp.arange(n_walkers) + start_idx) % z_max_size
        z_padded_new = z_padded.at[indices].set(new_positions)
        z_count_new = jnp.minimum(z_count + n_walkers, jnp.int32(z_max_size))

        return new_positions, new_log_probs, z_padded_new, z_count_new, key_next, found, bracket_ratios

    # --- Initial log-probs ---
    if verbose:
        print("  [dezess] Computing initial log-probs...", flush=True)
    positions = jnp.array(init_positions, dtype=jnp.float64)
    log_probs = jax.jit(jax.vmap(lambda x: _safe_log_prob(log_prob_fn, x)))(positions)
    log_probs.block_until_ready()

    # --- JIT compile ---
    if verbose:
        print("  [dezess] JIT compiling...", flush=True)
    t_jit = time.time()
    positions, log_probs, z_padded, z_count, key, _, _ = parallel_step(
        positions, log_probs, z_padded, z_count, mu, key)
    positions.block_until_ready()
    t_jit = time.time() - t_jit
    if verbose:
        print(f"  [dezess] JIT compile: {t_jit:.1f}s", flush=True)

    # --- Warmup (Z-matrix grows, mu tuning) ---
    # Tune mu so bracket width is ~3-5x mu (1-2 expansions per side).
    # bracket_ratio = (R-L)/mu: 1 = no expansion, 2*N_EXPAND+1 = maxed out.
    # Target ratio ~ N_EXPAND+1: use ~half the expansion budget on average,
    # leaving headroom for harder directions.
    TUNE_INTERVAL = 50
    TARGET_RATIO = float(N_EXPAND + 1)  # 4.0 with N_EXPAND=3
    MU_MIN, MU_MAX = 1e-8, 1e6
    ratio_ema = TARGET_RATIO  # exponential moving average of median bracket ratio

    t_sample = time.time()
    for step in range(n_warmup):
        positions, log_probs, z_padded, z_count, key, _, br = parallel_step(
            positions, log_probs, z_padded, z_count, mu, key)

        if tune and (step + 1) % TUNE_INTERVAL == 0:
            # Median bracket ratio across walkers this step
            med_ratio = float(jnp.median(br))
            # EMA with short memory (alpha=0.3) for responsiveness
            ratio_ema = 0.3 * med_ratio + 0.7 * ratio_ema
            # Multiplicative update: scale mu so observed ratio → target
            # Damped to prevent oscillation: move 50% of the way in log-space
            if ratio_ema > 0:
                adjustment = (ratio_ema / TARGET_RATIO) ** 0.5
                mu = jnp.clip(mu * adjustment, MU_MIN, MU_MAX)

        if verbose and ((step + 1) % 500 == 0 or step == 0):
            elapsed = time.time() - t_sample
            speed = (step + 1) / elapsed
            best = float(log_probs.max())
            med_br = float(jnp.median(br))
            print(f"  [dezess] warmup {step+1:6d}/{n_warmup} "
                  f"({speed:.1f} it/s) best_lp={best:.1f} z_count={int(z_count)} "
                  f"mu={float(mu):.4f} bracket_ratio={med_br:.1f}", flush=True)

    # --- Production (Z-matrix frozen) ---
    n_production = n_steps - n_warmup
    all_samples = np.zeros((n_production, n_walkers, n_dim), dtype=np.float64)
    all_log_probs = np.zeros((n_production, n_walkers), dtype=np.float64)
    z_frozen = z_padded
    z_count_frozen = z_count
    if verbose:
        print(f"  [dezess] Z-matrix frozen at {int(z_count_frozen)} entries for production",
              flush=True)

    # First production step
    positions, log_probs, _, _, key, _, _ = parallel_step(
        positions, log_probs, z_frozen, z_count_frozen, mu, key)
    all_samples[0] = np.asarray(positions)
    all_log_probs[0] = np.asarray(log_probs)

    t_prod = time.time()
    for step in range(1, n_production):
        positions, log_probs, _, _, key, _, _ = parallel_step(
            positions, log_probs, z_frozen, z_count_frozen, mu, key)

        all_samples[step] = np.asarray(positions)
        all_log_probs[step] = np.asarray(log_probs)

        if verbose and ((step + 1) % 200 == 0 or step == n_production - 1):
            elapsed = time.time() - t_prod
            speed = (step + 1) / elapsed
            eta = (n_production - step - 1) / speed if speed > 0 else 0
            best = float(all_log_probs[:step+1].max())
            mean_lp = float(all_log_probs[max(0,step-199):step+1].mean())
            print(f"  [dezess] {step+1:6d}/{n_production} "
                  f"({speed:.1f} it/s, ETA {eta:.0f}s) "
                  f"best_lp={best:.1f} mean_lp={mean_lp:.1f}", flush=True)

    return {
        "samples": jnp.array(all_samples),
        "log_prob": jnp.array(all_log_probs),
        "mu": float(mu),
        "z_matrix": z_frozen[:int(z_count_frozen)],
    }
