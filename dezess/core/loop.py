"""Generic variant runner that composes strategy modules.

This is the central orchestrator. Given a VariantConfig, it:
1. Resolves strategy modules (direction, width, slice, zmatrix, ensemble)
2. Builds a JIT-compiled parallel_step function
3. Runs warmup (Z-matrix growth, mu tuning, adaptive budget)
4. Runs production (frozen Z, frozen budgets)
5. Returns samples + diagnostics

The original sampler.py is untouched; this module provides the
same interface with pluggable strategies.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from dezess.core.types import VariantConfig, WalkerAux, SamplerState
from dezess.core.slice_sample import safe_log_prob

# Direction modules
from dezess.directions import de_mcz, snooker, pca, weighted_pair, momentum, riemannian, flow, whitened
# Width modules
from dezess.width import scalar as scalar_width, stochastic as stochastic_width, per_direction
from dezess.width import scale_aware as scale_aware_width, zeus_gamma as zeus_gamma_width
# Slice modules
from dezess.slice import fixed as fixed_slice, adaptive_budget, delayed_rejection
# Z-matrix modules
from dezess.zmatrix import circular as circular_zmatrix, hierarchical as hierarchical_zmatrix
# Ensemble modules
from dezess.ensemble import standard as standard_ensemble, parallel_tempering

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray

# Strategy registries
DIRECTION_STRATEGIES = {
    "de_mcz": de_mcz,
    "snooker": snooker,
    "pca": pca,
    "weighted_pair": weighted_pair,
    "momentum": momentum,
    "riemannian": riemannian,
    "flow": flow,
    "whitened": whitened,
}

WIDTH_STRATEGIES = {
    "scalar": scalar_width,
    "stochastic": stochastic_width,
    "per_direction": per_direction,
    "scale_aware": scale_aware_width,
    "zeus_gamma": zeus_gamma_width,
}

SLICE_STRATEGIES = {
    "fixed": fixed_slice,
    "adaptive_budget": adaptive_budget,
    "delayed_rejection": delayed_rejection,
}

ZMATRIX_STRATEGIES = {
    "circular": circular_zmatrix,
    "hierarchical": hierarchical_zmatrix,
}

ENSEMBLE_STRATEGIES = {
    "standard": standard_ensemble,
    "parallel_tempering": parallel_tempering,
}


DEFAULT_CONFIG = VariantConfig(
    name="scale_aware",
    direction="de_mcz",
    width="scale_aware",
    slice_fn="fixed",
    zmatrix="circular",
    ensemble="standard",
    width_kwargs={"scale_factor": 1.0},
)


def run_variant(
    log_prob_fn: Callable[[Array], Array],
    init_positions: Array,
    n_steps: int,
    config: Optional[VariantConfig] = None,
    n_warmup: int = 0,
    key: Optional[Array] = None,
    mu: float = 1.0,
    tune: bool = True,
    z_max_size: int = 50000,
    z_initial: Optional[Array] = None,
    verbose: bool = True,
) -> dict:
    """Run a sampler variant composed from the given config.

    Same interface as run_demcz_slice but with pluggable strategies.

    Returns dict with:
        "samples": (n_production, n_walkers, n_dim)
        "log_prob": (n_production, n_walkers)
        "mu": float
        "z_matrix": frozen Z-matrix
        "diagnostics": dict with per-step metrics
    """
    if config is None:
        config = DEFAULT_CONFIG
    n_walkers, n_dim = init_positions.shape
    if key is None:
        key = jax.random.PRNGKey(0)
    mu = jnp.float64(mu)

    # Resolve strategies
    dir_mod = DIRECTION_STRATEGIES[config.direction]
    width_mod = WIDTH_STRATEGIES[config.width]
    slice_mod = SLICE_STRATEGIES[config.slice_fn]
    zmat_mod = ZMATRIX_STRATEGIES[config.zmatrix]
    ens_mod = ENSEMBLE_STRATEGIES[config.ensemble]

    dir_kwargs = dict(config.direction_kwargs)
    width_kwargs = dict(config.width_kwargs)
    slice_kwargs = dict(config.slice_kwargs)
    zmat_kwargs = dict(config.zmatrix_kwargs)
    ens_kwargs = dict(config.ensemble_kwargs)

    # Slice budget (may be tuned during warmup for adaptive_budget)
    n_expand = slice_kwargs.pop("n_expand", 3)
    n_shrink = slice_kwargs.pop("n_shrink", 12)

    # Initialize Z-matrix
    z_padded = jnp.zeros((z_max_size, n_dim), dtype=jnp.float64)
    z_log_probs = jnp.full(z_max_size, -1e30, dtype=jnp.float64)

    if z_initial is not None:
        n_z = min(z_initial.shape[0], z_max_size)
        z_padded = z_padded.at[:n_z].set(z_initial[:n_z])
        z_count = jnp.int32(n_z)
    else:
        z_padded = z_padded.at[:n_walkers].set(init_positions)
        z_count = jnp.int32(n_walkers)

    # Initialize walker auxiliary state
    walker_aux = WalkerAux(
        prev_direction=jnp.zeros((n_walkers, n_dim), dtype=jnp.float64),
        bracket_widths=jnp.zeros((n_walkers, n_dim), dtype=jnp.float64),
        direction_anchor=jnp.zeros((n_walkers, n_dim), dtype=jnp.float64),
        direction_scale=jnp.ones(n_walkers, dtype=jnp.float64),
    )

    # Initialize temperatures for parallel tempering
    temperatures = jnp.ones(n_walkers, dtype=jnp.float64)
    if config.ensemble == "parallel_tempering":
        n_temps = ens_kwargs.get("n_temps", 4)
        t_max = ens_kwargs.get("t_max", 10.0)
        temperatures = parallel_tempering.init_temperatures(n_walkers, n_temps, t_max)

    # PCA components: pre-initialize with identity (updated after warmup)
    pca_components = jnp.eye(n_dim, dtype=jnp.float64)
    pca_weights = jnp.ones(n_dim, dtype=jnp.float64) / n_dim

    # Flow directions: pre-initialize with zeros (updated after warmup)
    # During warmup, flow_directions is a dummy; the step_fn will use DE-MCz fallback
    flow_directions = jnp.zeros((500, n_dim), dtype=jnp.float64)

    # Whitening matrix: pre-initialize with identity (updated after warmup)
    whitening_matrix = jnp.eye(n_dim, dtype=jnp.float64)

    # --- Build JIT-compiled step function ---
    # All strategy-specific kwargs are captured as closure variables at trace time.
    # Python if-checks on config.direction/width/etc are resolved at trace time
    # (they are string constants), so no data-dependent branching inside JIT.
    def _make_step_fn(n_exp, n_shr):
        """Create a JIT-compiled step function with given budget."""

        @jax.jit
        def parallel_step(positions, log_probs, z_padded, z_count, z_log_probs,
                          mu, key, walker_aux_prev_dir, walker_aux_bw,
                          walker_aux_d_anchor, walker_aux_d_scale,
                          temperatures, pca_components, pca_weights,
                          flow_directions, whitening_matrix):
            keys = jax.random.split(key, n_walkers + 1)
            key_next = keys[0]
            walker_keys = keys[1:]

            def update_one(x_i, lp_i, w_key, prev_d, bw, d_anchor, d_scale, temp):
                w_aux = WalkerAux(prev_direction=prev_d, bracket_widths=bw,
                                  direction_anchor=d_anchor,
                                  direction_scale=d_scale)

                # Direction — strategy-specific kwargs resolved at trace time
                if config.direction == "pca":
                    d, w_key, w_aux = dir_mod.sample_direction(
                        x_i, z_padded, z_count, z_log_probs, w_key, w_aux,
                        pca_components=pca_components, pca_weights=pca_weights,
                        **dir_kwargs,
                    )
                elif config.direction == "flow":
                    d, w_key, w_aux = dir_mod.sample_direction(
                        x_i, z_padded, z_count, z_log_probs, w_key, w_aux,
                        flow_directions=flow_directions,
                        **dir_kwargs,
                    )
                elif config.direction == "whitened":
                    d, w_key, w_aux = dir_mod.sample_direction(
                        x_i, z_padded, z_count, z_log_probs, w_key, w_aux,
                        whitening_matrix=whitening_matrix,
                        **dir_kwargs,
                    )
                else:
                    d, w_key, w_aux = dir_mod.sample_direction(
                        x_i, z_padded, z_count, z_log_probs, w_key, w_aux,
                        **dir_kwargs,
                    )

                # Width
                w_key, k_width = jax.random.split(w_key)
                mu_eff = width_mod.get_mu(mu, d, w_aux, key=k_width, **width_kwargs)

                # Build slice log-density.
                # For snooker: include Jacobian |x - anchor|^{d-1} so
                # the 1D conditional in radial coords is correct.
                if config.direction == "snooker":
                    z_anchor = w_aux.direction_anchor
                    ndim_j = x_i.shape[0]
                    def lp_fn(x):
                        base = safe_log_prob(log_prob_fn, x) / temp
                        dist = jnp.sqrt(jnp.sum((x - z_anchor)**2))
                        return base + (ndim_j - 1) * jnp.log(jnp.maximum(dist, 1e-30))

                    dist_0 = jnp.sqrt(jnp.sum((x_i - z_anchor)**2))
                    lp_x_slice = lp_i / temp + (ndim_j - 1) * jnp.log(jnp.maximum(dist_0, 1e-30))
                else:
                    def lp_fn(x):
                        return safe_log_prob(log_prob_fn, x) / temp

                    lp_x_slice = lp_i / temp

                # Slice sample
                x_new, lp_new, w_key, found, L, R = slice_mod.execute(
                    lp_fn, x_i, d, lp_x_slice, mu_eff, w_key,
                    n_expand=n_exp, n_shrink=n_shr,
                    **slice_kwargs,
                )

                # Un-temper the log_prob for storage (no Jacobian)
                lp_new_untempered = safe_log_prob(log_prob_fn, x_new)

                # Update per-direction bracket widths if using that strategy
                if config.width == "per_direction":
                    w_aux = per_direction.update_bracket_widths(
                        w_aux, d, L, R, found,
                        ema_alpha=width_kwargs.get("ema_alpha", 0.1),
                    )

                # Multi-direction: additional sequential slices along fresh DE-MCz directions
                if config.n_slices_per_step > 1:
                    def _one_slice(i, carry):
                        x_c, lp_c, w_key_c, found_c, L_c, R_c = carry
                        w_key_c, k_dir = jax.random.split(w_key_c)
                        k_idx1, k_idx2 = jax.random.split(k_dir)
                        idx1 = jax.random.randint(k_idx1, (), 0, z_padded.shape[0]) % z_count
                        idx2 = jax.random.randint(k_idx2, (), 0, z_padded.shape[0]) % z_count
                        idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
                        diff = z_padded[idx1] - z_padded[idx2]
                        norm = jnp.sqrt(jnp.sum(diff ** 2))
                        d_new = diff / jnp.maximum(norm, 1e-30)
                        w_key_c, k_w = jax.random.split(w_key_c)
                        mu_eff_c = width_mod.get_mu(mu, d_new, w_aux, key=k_w, **width_kwargs)
                        def lp_fn_multi(x):
                            return safe_log_prob(log_prob_fn, x) / temp
                        x_c, _, w_key_c, found_new, L_new, R_new = slice_mod.execute(
                            lp_fn_multi, x_c, d_new, lp_c / temp, mu_eff_c, w_key_c,
                            n_expand=n_exp, n_shrink=n_shr,
                        )
                        lp_c = safe_log_prob(log_prob_fn, x_c)
                        return (x_c, lp_c, w_key_c, found_c & found_new, L_new, R_new)

                    init_carry = (x_new, lp_new_untempered, w_key, found, L, R)
                    x_new, lp_new_untempered, w_key, found, L, R = jax.lax.fori_loop(
                        0, config.n_slices_per_step - 1, _one_slice, init_carry
                    )

                bracket_ratio = (R - L) / jnp.maximum(mu_eff, 1e-30)

                return (x_new, lp_new_untempered, w_key, found, bracket_ratio,
                        w_aux.prev_direction, w_aux.bracket_widths,
                        w_aux.direction_anchor, w_aux.direction_scale)

            results = jax.vmap(update_one)(
                positions, log_probs, walker_keys,
                walker_aux_prev_dir, walker_aux_bw, walker_aux_d_anchor,
                walker_aux_d_scale, temperatures,
            )
            (new_pos, new_lp, _, found, bracket_ratios,
             new_prev_dirs, new_bws, new_d_anchors, new_d_scales) = results

            return (new_pos, new_lp, key_next, found, bracket_ratios,
                    new_prev_dirs, new_bws, new_d_anchors, new_d_scales)

        return parallel_step

    def _call_step(step_fn, positions, log_probs, z_padded, z_count, z_log_probs,
                   mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
                   walker_aux_ds, temperatures):
        """Helper to call step_fn with all required args including PCA/flow/whitening."""
        return step_fn(
            positions, log_probs, z_padded, z_count, z_log_probs,
            mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
            walker_aux_ds, temperatures,
            pca_components, pca_weights, flow_directions, whitening_matrix,
        )

    # --- Initial log-probs ---
    if verbose:
        print(f"  [{config.name}] Computing initial log-probs...", flush=True)
    positions = jnp.array(init_positions, dtype=jnp.float64)
    log_probs = jax.jit(jax.vmap(lambda x: safe_log_prob(log_prob_fn, x)))(positions)
    log_probs.block_until_ready()

    # Populate initial Z log probs
    z_log_probs = z_log_probs.at[:n_walkers].set(log_probs)

    # --- JIT compile ---
    step_fn = _make_step_fn(n_expand, n_shrink)
    if verbose:
        print(f"  [{config.name}] JIT compiling...", flush=True)
    t_jit = time.time()
    (positions, log_probs, key, _, _,
     walker_aux_pd, walker_aux_bw, walker_aux_da, walker_aux_ds) = _call_step(
        step_fn, positions, log_probs, z_padded, z_count, z_log_probs,
        mu, key, walker_aux.prev_direction, walker_aux.bracket_widths,
        walker_aux.direction_anchor, walker_aux.direction_scale, temperatures,
    )
    positions.block_until_ready()
    t_jit = time.time() - t_jit
    if verbose:
        print(f"  [{config.name}] JIT compile: {t_jit:.1f}s", flush=True)

    # Update Z after JIT warmup step
    z_padded, z_count, z_log_probs = circular_zmatrix.append(
        z_padded, z_count, z_log_probs, positions, log_probs, z_max_size,
    )

    # --- Warmup ---
    TUNE_INTERVAL = 50
    TARGET_RATIO = float(n_expand + 1)
    MU_MIN, MU_MAX = 1e-8, 1e6
    ratio_ema = TARGET_RATIO
    cap_hit_counts = 0
    total_samples_warmup = 0

    # ESJD tuning state
    esjd_ema = 0.0
    best_esjd = 0.0
    best_mu = mu
    prev_positions = positions

    t_sample = time.time()
    for step in range(n_warmup):
        (positions, log_probs, key, found, br,
         walker_aux_pd, walker_aux_bw, walker_aux_da, walker_aux_ds) = _call_step(
            step_fn, positions, log_probs, z_padded, z_count, z_log_probs,
            mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
            walker_aux_ds, temperatures,
        )

        # Append to Z-matrix
        z_padded, z_count, z_log_probs = circular_zmatrix.append(
            z_padded, z_count, z_log_probs, positions, log_probs, z_max_size,
        )

        # Track cap-hits for adaptive budget
        cap_hit_counts += int(jnp.sum(~found))
        total_samples_warmup += n_walkers

        # Track ESJD
        esjd = float(jnp.mean(jnp.sum((positions - prev_positions)**2, axis=-1)))
        esjd_ema = 0.3 * esjd + 0.7 * esjd_ema
        prev_positions = positions

        # Tune mu
        if tune and (step + 1) % TUNE_INTERVAL == 0:
            if config.tune_method == "esjd":
                if esjd_ema > best_esjd:
                    best_esjd = esjd_ema
                    best_mu = mu
                if (step + 1) % (TUNE_INTERVAL * 2) == 0:
                    mu = jnp.clip(best_mu * 1.2, MU_MIN, MU_MAX)
                else:
                    mu = jnp.clip(best_mu * 0.83, MU_MIN, MU_MAX)
            else:
                med_ratio = float(jnp.median(br))
                ratio_ema = 0.3 * med_ratio + 0.7 * ratio_ema
                if ratio_ema > 0:
                    adjustment = (ratio_ema / TARGET_RATIO) ** 0.5
                    mu = jnp.clip(mu * adjustment, MU_MIN, MU_MAX)

        # Replica exchange swaps
        if config.ensemble == "parallel_tempering":
            key, k_swap = jax.random.split(key)
            positions, log_probs = parallel_tempering.propose_swaps(
                positions, log_probs, temperatures, k_swap,
                n_temps=ens_kwargs.get("n_temps", 4),
            )

        if verbose and ((step + 1) % 500 == 0 or step == 0):
            elapsed = time.time() - t_sample
            speed = (step + 1) / elapsed
            best = float(log_probs.max())
            med_br = float(jnp.median(br))
            print(f"  [{config.name}] warmup {step+1:6d}/{n_warmup} "
                  f"({speed:.1f} it/s) best_lp={best:.1f} z_count={int(z_count)} "
                  f"mu={float(mu):.4f} bracket_ratio={med_br:.1f}", flush=True)

    # Finalize ESJD tuning
    if config.tune_method == "esjd" and tune and n_warmup > 0:
        mu = best_mu

    # --- Post-warmup setup ---

    # Train flow and precompute direction bank
    if config.direction == "flow" and n_warmup > 0:
        if verbose:
            print(f"  [{config.name}] Training normalizing flow...", flush=True)
        # Cap training data to avoid slow training on large Z-matrices
        n_train = min(int(z_count), 2000)
        z_for_training = z_padded[:n_train]
        key, k_flow = jax.random.split(key)
        flow_params = flow.train_flow(
            k_flow, z_for_training, n_dim,
            n_epochs=dir_kwargs.get("flow_epochs", 200),
            n_layers=dir_kwargs.get("flow_layers", 3),
            hidden=dir_kwargs.get("flow_hidden", 64),
        )
        if verbose:
            print(f"  [{config.name}] Precomputing flow direction bank...", flush=True)
        key, k_dirs = jax.random.split(key)
        flow_directions = flow.precompute_flow_directions(
            flow_params, z_padded, z_count,
            n_directions=500, key=k_dirs,
        )
        # Remove flow_params from dir_kwargs (we use precomputed directions now)
        dir_kwargs.pop("flow_params", None)
        if verbose:
            print(f"  [{config.name}] Flow directions ready ({flow_directions.shape[0]} directions)", flush=True)

    # Compute PCA if using PCA directions
    if config.direction == "pca" and n_warmup > 0:
        if verbose:
            print(f"  [{config.name}] Computing PCA components...", flush=True)
        pca_components, pca_weights = pca.compute_pca_components(z_padded, z_count)

    # Compute whitening matrix if using whitened directions
    if config.direction == "whitened" and n_warmup > 0:
        if verbose:
            print(f"  [{config.name}] Computing whitening matrix...", flush=True)
        whitening_matrix = whitened.compute_whitening_matrix(z_padded, z_count)

    # Adaptive budget: tune N_EXPAND/N_SHRINK and re-JIT
    needs_rejit = False
    if config.slice_fn == "adaptive_budget" and total_samples_warmup > 0:
        cap_hit_rate = cap_hit_counts / total_samples_warmup
        n_expand, n_shrink = adaptive_budget.select_budget(cap_hit_rate)
        if verbose:
            print(f"  [{config.name}] Adaptive budget: cap_hit_rate={cap_hit_rate:.4f} "
                  f"-> n_expand={n_expand}, n_shrink={n_shrink}", flush=True)
        needs_rejit = True

    # Re-JIT if budget changed or PCA/flow directions updated
    if needs_rejit or config.direction in ("pca", "flow", "whitened"):
        step_fn = _make_step_fn(n_expand, n_shrink)
        if verbose:
            print(f"  [{config.name}] Re-JIT compiling for production...", flush=True)
        t_rejit = time.time()
        (positions, log_probs, key, _, _,
         walker_aux_pd, walker_aux_bw, walker_aux_da, walker_aux_ds) = _call_step(
            step_fn, positions, log_probs, z_padded, z_count, z_log_probs,
            mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
            walker_aux_ds, temperatures,
        )
        positions.block_until_ready()
        if verbose:
            print(f"  [{config.name}] Re-JIT: {time.time()-t_rejit:.1f}s", flush=True)

    # --- Production (Z-matrix frozen) ---
    n_production = n_steps - n_warmup
    all_samples = np.zeros((n_production, n_walkers, n_dim), dtype=np.float64)
    all_log_probs = np.zeros((n_production, n_walkers), dtype=np.float64)
    all_found = np.zeros((n_production, n_walkers), dtype=np.bool_)
    all_bracket_ratios = np.zeros((n_production, n_walkers), dtype=np.float64)

    z_frozen = z_padded
    z_count_frozen = z_count
    z_lp_frozen = z_log_probs

    if verbose:
        print(f"  [{config.name}] Z-matrix frozen at {int(z_count_frozen)} entries",
              flush=True)

    t_prod = time.time()
    for step in range(n_production):
        (positions, log_probs, key, found, br,
         walker_aux_pd, walker_aux_bw, walker_aux_da, walker_aux_ds) = step_fn(
            positions, log_probs, z_frozen, z_count_frozen, z_lp_frozen,
            mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
            walker_aux_ds, temperatures,
            pca_components, pca_weights, flow_directions, whitening_matrix,
        )

        # Replica exchange swaps
        if config.ensemble == "parallel_tempering":
            key, k_swap = jax.random.split(key)
            positions, log_probs = parallel_tempering.propose_swaps(
                positions, log_probs, temperatures, k_swap,
                n_temps=ens_kwargs.get("n_temps", 4),
            )

        all_samples[step] = np.asarray(positions)
        all_log_probs[step] = np.asarray(log_probs)
        all_found[step] = np.asarray(found)
        all_bracket_ratios[step] = np.asarray(br)

        if verbose and ((step + 1) % 200 == 0 or step == n_production - 1):
            elapsed = time.time() - t_prod
            speed = (step + 1) / elapsed
            eta = (n_production - step - 1) / speed if speed > 0 else 0
            best = float(all_log_probs[:step+1].max())
            mean_lp = float(all_log_probs[max(0,step-199):step+1].mean())
            zero_rate = 1.0 - all_found[max(0,step-199):step+1].mean()
            print(f"  [{config.name}] {step+1:6d}/{n_production} "
                  f"({speed:.1f} it/s, ETA {eta:.0f}s) "
                  f"best_lp={best:.1f} mean_lp={mean_lp:.1f} "
                  f"zero_move={zero_rate:.4f}", flush=True)

    wall_time = time.time() - t_prod

    return {
        "samples": jnp.array(all_samples),
        "log_prob": jnp.array(all_log_probs),
        "mu": float(mu),
        "z_matrix": z_frozen[:int(z_count_frozen)],
        "config": config,
        "wall_time": wall_time,
        "n_production": n_production,
        "diagnostics": {
            "found": all_found,
            "bracket_ratios": all_bracket_ratios,
            "n_expand": n_expand,
            "n_shrink": n_shrink,
            "cap_hit_rate": 1.0 - all_found.mean(),
        },
    }
