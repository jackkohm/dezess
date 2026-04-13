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
from dezess.core.slice_sample import safe_log_prob, unsafe_log_prob
from dezess.diagnostics import StreamingDiagnostics
from dezess.transforms import Transform, identity

# Direction modules
from dezess.directions import de_mcz, snooker, pca, weighted_pair, momentum, riemannian, flow, whitened, gradient, coordinate, local_pair, kde_direction, global_move, random_gaussian
# Width modules
from dezess.width import scalar as scalar_width, stochastic as stochastic_width, per_direction
from dezess.width import scale_aware as scale_aware_width, zeus_gamma as zeus_gamma_width
from dezess.width import cov_aware as cov_aware_width
# Slice modules
from dezess.slice import fixed as fixed_slice, adaptive_budget, delayed_rejection, early_stop, overrelaxed, nurs as nurs_slice, multi_try as multi_try_slice, adaptive as adaptive_slice
from dezess.slice import mh as mh_slice
from dezess.slice import mh_multi as mh_multi_slice
from dezess.slice import mh_adaptive as mh_adaptive_slice
# Z-matrix modules
from dezess.zmatrix import circular as circular_zmatrix, hierarchical as hierarchical_zmatrix, live as live_zmatrix
# Ensemble modules
from dezess.ensemble import standard as standard_ensemble, parallel_tempering, block_gibbs
from dezess.core.slice_sample import slice_sample_fixed

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
    "gradient": gradient,
    "coordinate": coordinate,
    "local_pair": local_pair,
    "kde": kde_direction,
    "global_move": global_move,
    "random_gaussian": random_gaussian,
}

WIDTH_STRATEGIES = {
    "scalar": scalar_width,
    "stochastic": stochastic_width,
    "per_direction": per_direction,
    "scale_aware": scale_aware_width,
    "zeus_gamma": zeus_gamma_width,
    "cov_aware": cov_aware_width,
}

SLICE_STRATEGIES = {
    "fixed": fixed_slice,
    "adaptive_budget": adaptive_budget,
    "delayed_rejection": delayed_rejection,
    "early_stop": early_stop,
    "overrelaxed": overrelaxed,
    "nurs": nurs_slice,
    "multi_try": multi_try_slice,
    "adaptive": adaptive_slice,
    "mh": mh_slice,
    "mh_multi": mh_multi_slice,
    "mh_adaptive": mh_adaptive_slice,
}

ZMATRIX_STRATEGIES = {
    "circular": circular_zmatrix,
    "hierarchical": hierarchical_zmatrix,
    "live": live_zmatrix,
}

ENSEMBLE_STRATEGIES = {
    "standard": standard_ensemble,
    "parallel_tempering": parallel_tempering,
    "block_gibbs": block_gibbs,
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
    target_ess: Optional[float] = None,
    progress_fn: Optional[Callable] = None,
    verbose: bool = True,
    transform: Optional[Transform] = None,
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

    # --- Transform setup ---
    if transform is not None:
        init_positions = jax.vmap(transform.inverse)(init_positions)
        _original_log_prob = log_prob_fn
        def log_prob_fn(z, _fwd=transform.forward, _ldj=transform.log_det_jac, _lp=_original_log_prob):
            return _lp(_fwd(z)) + _ldj(z)

    # Resolve strategies (validate names early for clear error messages)
    for name, registry, label in [
        (config.direction, DIRECTION_STRATEGIES, "direction"),
        (config.width, WIDTH_STRATEGIES, "width"),
        (config.slice_fn, SLICE_STRATEGIES, "slice_fn"),
        (config.zmatrix, ZMATRIX_STRATEGIES, "zmatrix"),
        (config.ensemble, ENSEMBLE_STRATEGIES, "ensemble"),
    ]:
        if name not in registry:
            raise ValueError(
                f"Unknown {label} strategy '{name}'. "
                f"Available: {sorted(registry.keys())}"
            )
    dir_mod = DIRECTION_STRATEGIES[config.direction]
    width_mod = WIDTH_STRATEGIES[config.width]
    slice_mod = SLICE_STRATEGIES[config.slice_fn]
    zmat_mod = ZMATRIX_STRATEGIES[config.zmatrix]
    ens_mod = ENSEMBLE_STRATEGIES[config.ensemble]

    # Select log_prob evaluator: skip NaN checks when config.check_nans=False
    lp_eval = safe_log_prob if config.check_nans else unsafe_log_prob

    dir_kwargs = dict(config.direction_kwargs)
    # Pop meta-parameter: threshold for automatic PCA direction switch after warmup.
    # If empirical condition number > cov_pca_threshold and direction=="de_mcz",
    # switch to PCA for production (requires width="cov_aware" so cov_chol is available).
    cov_pca_threshold = float(dir_kwargs.pop("cov_pca_threshold", 0))
    # For gradient directions, compute grad_fn from log_prob
    if config.direction == "gradient":
        dir_kwargs["grad_fn"] = jax.grad(lambda x: safe_log_prob(log_prob_fn, x))
    width_kwargs = dict(config.width_kwargs)
    slice_kwargs = dict(config.slice_kwargs)
    zmat_kwargs = dict(config.zmatrix_kwargs)
    ens_kwargs = dict(config.ensemble_kwargs)

    # Slice budget (may be tuned during warmup for adaptive_budget).
    # scale_aware width produces well-calibrated brackets, so fewer
    # expand/shrink iterations are needed (2/8 vs 3/12 default).
    if config.width == "scale_aware" and "n_expand" not in slice_kwargs and "n_shrink" not in slice_kwargs:
        n_expand = slice_kwargs.pop("n_expand", 2)
        n_shrink = slice_kwargs.pop("n_shrink", 8)
    else:
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

    # Covariance Cholesky: pre-initialize with identity (updated after warmup for mh_adaptive)
    cov_chol = jnp.eye(n_dim, dtype=jnp.float64)

    # KDE bandwidth: pre-initialize with ones (updated after warmup)
    kde_bandwidth = jnp.ones(n_dim, dtype=jnp.float64)

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
                          flow_directions, whitening_matrix, kde_bandwidth,
                          cov_chol):
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
                elif config.direction == "kde":
                    d, w_key, w_aux = dir_mod.sample_direction(
                        x_i, z_padded, z_count, z_log_probs, w_key, w_aux,
                        kde_bandwidth=kde_bandwidth,
                        **dir_kwargs,
                    )
                else:
                    d, w_key, w_aux = dir_mod.sample_direction(
                        x_i, z_padded, z_count, z_log_probs, w_key, w_aux,
                        **dir_kwargs,
                    )

                # Width
                w_key, k_width = jax.random.split(w_key)
                if config.width == "cov_aware":
                    mu_eff = width_mod.get_mu(mu, d, w_aux, key=k_width,
                                              cov_chol=cov_chol, **width_kwargs)
                else:
                    mu_eff = width_mod.get_mu(mu, d, w_aux, key=k_width, **width_kwargs)

                # Build slice log-density.
                # For snooker: include Jacobian |x - anchor|^{d-1} so
                # the 1D conditional in radial coords is correct.
                if config.direction == "snooker":
                    z_anchor = w_aux.direction_anchor
                    ndim_j = x_i.shape[0]
                    def lp_fn(x):
                        base = lp_eval(log_prob_fn, x) / temp
                        dist = jnp.linalg.norm(x - z_anchor)
                        return base + (ndim_j - 1) * jnp.log(jnp.maximum(dist, 1e-30))

                    dist_0 = jnp.linalg.norm(x_i - z_anchor)
                    lp_x_slice = lp_i / temp + (ndim_j - 1) * jnp.log(jnp.maximum(dist_0, 1e-30))
                else:
                    def lp_fn(x):
                        return lp_eval(log_prob_fn, x) / temp

                    lp_x_slice = lp_i / temp

                # Slice sample
                if config.slice_fn == "multi_try":
                    x_new, lp_new, w_key, found, L, R = slice_mod.execute(
                        lp_fn, x_i, d, lp_x_slice, mu_eff, w_key,
                        n_expand=n_exp, n_shrink=n_shr,
                        z_matrix=z_padded, z_count=z_count, base_mu=mu,
                        **slice_kwargs,
                    )
                elif config.slice_fn == "mh_adaptive":
                    x_new, lp_new, w_key, found, L, R = slice_mod.execute(
                        lp_fn, x_i, d, lp_x_slice, mu_eff, w_key,
                        n_expand=n_exp, n_shrink=n_shr,
                        cov_chol=cov_chol,
                        **slice_kwargs,
                    )
                else:
                    x_new, lp_new, w_key, found, L, R = slice_mod.execute(
                        lp_fn, x_i, d, lp_x_slice, mu_eff, w_key,
                        n_expand=n_exp, n_shrink=n_shr,
                        **slice_kwargs,
                    )

                # Un-temper the log_prob for storage.
                # For standard ensemble at T=1 (no snooker Jacobian), lp_new
                # from the slice sampler IS the raw log_prob — skip the
                # redundant evaluation. But during warmup tempering (temp > 1)
                # or with snooker/tempering, we must re-evaluate.
                if config.direction == "snooker" or config.ensemble == "parallel_tempering":
                    lp_new_untempered = lp_eval(log_prob_fn, x_new)
                else:
                    # temp=1 in production; during warmup tempering this will
                    # be re-evaluated in the warmup loop when needed
                    lp_new_untempered = lp_new

                # Update per-direction bracket widths if using that strategy
                if config.width == "per_direction":
                    w_aux = per_direction.update_bracket_widths(
                        w_aux, d, L, R, found,
                        ema_alpha=width_kwargs.get("ema_alpha", 0.1),
                    )

                # Multi-direction: additional sequential slices along
                # orthogonalized DE-MCz directions (Gibbs-like coordinate cycling).
                # Each direction is made orthogonal to all previous ones via
                # Gram-Schmidt, ensuring each slice explores a different subspace.
                if config.n_slices_per_step > 1:
                    # prev_dirs stores directions used so far for orthogonalization
                    # Shape: (max_slices, n_dim) — padded with zeros
                    n_max_slices = config.n_slices_per_step
                    ndim_local = x_i.shape[0]
                    prev_dirs = jnp.zeros((n_max_slices, ndim_local), dtype=jnp.float64)
                    prev_dirs = prev_dirs.at[0].set(d)  # first direction already used

                    def _one_slice(i, carry):
                        x_c, lp_c, w_key_c, found_c, L_c, R_c, pdirs = carry
                        w_key_c, k_dir = jax.random.split(w_key_c)
                        k_idx1, k_idx2 = jax.random.split(k_dir)
                        idx1 = jax.random.randint(k_idx1, (), 0, z_padded.shape[0]) % z_count
                        idx2 = jax.random.randint(k_idx2, (), 0, z_padded.shape[0]) % z_count
                        idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
                        diff = z_padded[idx1] - z_padded[idx2]

                        # Gram-Schmidt: remove components along previous directions.
                        # Only iterate over directions already set (i+1 is traced,
                        # valid for fori_loop bounds unlike scan which needs static length).
                        def _ortho_one(j, d_raw):
                            proj = jnp.dot(d_raw, pdirs[j]) * pdirs[j]
                            return d_raw - proj
                        d_ortho = jax.lax.fori_loop(0, i + 1, _ortho_one, diff)

                        norm_ortho = jnp.linalg.norm(d_ortho)
                        norm_raw = jnp.linalg.norm(diff)
                        # Fall back to raw direction if orthogonal component is tiny
                        use_ortho = norm_ortho > 1e-10 * norm_raw
                        d_final = jnp.where(use_ortho, d_ortho, diff)
                        d_norm = jnp.where(use_ortho, norm_ortho, norm_raw)
                        d_new = d_final / jnp.maximum(d_norm, 1e-30)

                        # Store for next iteration's orthogonalization
                        pdirs = pdirs.at[i + 1].set(d_new)

                        # Width for additional slice — pass cov_chol if width=cov_aware
                        w_aux_c = w_aux._replace(direction_scale=norm_raw)
                        w_key_c, k_w = jax.random.split(w_key_c)
                        if config.width == "cov_aware":
                            mu_eff_c = width_mod.get_mu(mu, d_new, w_aux_c, key=k_w,
                                                        cov_chol=cov_chol, **width_kwargs)
                        else:
                            mu_eff_c = width_mod.get_mu(mu, d_new, w_aux_c, key=k_w, **width_kwargs)
                        def lp_fn_multi(x):
                            return lp_eval(log_prob_fn, x) / temp
                        x_c, _, w_key_c, found_new, L_new, R_new = slice_mod.execute(
                            lp_fn_multi, x_c, d_new, lp_c / temp, mu_eff_c, w_key_c,
                            n_expand=n_exp, n_shrink=n_shr,
                        )
                        lp_c = lp_eval(log_prob_fn, x_c)
                        return (x_c, lp_c, w_key_c, found_c & found_new, L_new, R_new, pdirs)

                    init_carry = (x_new, lp_new_untempered, w_key, found, L, R, prev_dirs)
                    x_new, lp_new_untempered, w_key, found, L, R, _ = jax.lax.fori_loop(
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
        """Helper to call step_fn with all required args including PCA/flow/whitening/cov."""
        return step_fn(
            positions, log_probs, z_padded, z_count, z_log_probs,
            mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
            walker_aux_ds, temperatures,
            pca_components, pca_weights, flow_directions, whitening_matrix, kde_bandwidth,
            cov_chol,
        )

    # --- Block-Gibbs setup ---
    use_block_gibbs = config.ensemble == "block_gibbs"
    use_block_mh = use_block_gibbs and ens_kwargs.get("use_mh", False)
    use_delayed_rejection = use_block_mh and ens_kwargs.get("delayed_rejection", False)
    use_block_cov = use_block_mh and ens_kwargs.get("use_block_cov", False)
    conditional_log_prob_fn = ens_kwargs.get("conditional_log_prob", None)
    use_conditional = conditional_log_prob_fn is not None and use_block_mh

    if use_block_gibbs:
        blocks = block_gibbs.parse_blocks(ens_kwargs, n_dim)
        n_blocks_val = len(blocks)
        mu_blocks = block_gibbs.init_mu_blocks(n_blocks_val, float(mu))

        # Pad blocks to same size for scan compatibility
        max_block_size = max(len(b) for b in blocks)
        padded_blocks = jnp.zeros((n_blocks_val, max_block_size), dtype=jnp.int32)
        block_sizes_arr = jnp.array([len(b) for b in blocks], dtype=jnp.int32)
        for i, b in enumerate(blocks):
            padded_blocks = padded_blocks.at[i, :len(b)].set(b)

        if use_conditional:
            n_conditional_blocks = n_blocks_val - 1  # blocks 1..N are conditional

        @jax.jit
        def block_sweep_step(positions, log_probs, z_padded, z_count,
                             mu_blocks_arg, key):

            def _one_block(carry, block_data):
                pos, lps, k = carry
                block_idx, bsize, mu_b = block_data
                b_idx = padded_blocks[block_idx]  # (max_block_size,)
                # Mask: 1 for valid block indices, 0 for padding
                mask = (jnp.arange(max_block_size) < bsize).astype(jnp.float64)

                def _update_walker(x_full, lp, wk):
                    wk, k1, k2 = jax.random.split(wk, 3)
                    idx1 = jax.random.randint(k1, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jax.random.randint(k2, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

                    # Direction in block subspace (full padded size)
                    z1_block = z_padded[idx1, b_idx]  # (max_block_size,)
                    z2_block = z_padded[idx2, b_idx]
                    diff = (z1_block - z2_block) * mask  # zero out padding
                    norm = jnp.linalg.norm(diff)
                    d_block = diff / jnp.maximum(norm, 1e-30)

                    # Embed into full space: scatter block direction
                    d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                    d_full = d_full.at[b_idx].add(d_block * mask)

                    # Scale-aware width
                    dim_corr = jnp.sqrt(jnp.float64(bsize) / 2.0)
                    mu_eff = mu_b * norm / jnp.maximum(dim_corr, 1e-30)

                    # Slice sample
                    x_new, lp_new, wk, found, L, R = slice_sample_fixed(
                        lambda x: lp_eval(log_prob_fn, x),
                        x_full, d_full, lp, mu_eff, wk,
                        n_expand=n_expand, n_shrink=n_shrink,
                    )
                    br = (R - L) / jnp.maximum(mu_eff, 1e-30)
                    return x_new, lp_new, wk, br

                k, k_walkers = jax.random.split(k)
                wkeys = jax.random.split(k_walkers, n_walkers)
                new_pos, new_lp, _, brs = jax.vmap(_update_walker)(pos, lps, wkeys)
                mean_br = jnp.mean(brs)
                return (new_pos, new_lp, k), mean_br

            # Shuffle block order
            k, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, n_blocks_val)
            block_data = (perm, block_sizes_arr[perm], mu_blocks_arg[perm])

            (pos_out, lps_out, k_out), all_br = jax.lax.scan(
                _one_block, (positions, log_probs, k), block_data,
            )
            mean_br = jnp.mean(all_br)
            return pos_out, lps_out, k_out, mean_br

        # --- Block-Gibbs MH: 1 eval per block instead of ~10 for slice ---
        @jax.jit
        def block_mh_step(positions, log_probs, z_padded, z_count,
                          mu_blocks_arg, key):

            def _one_block_mh(carry, block_data):
                pos, lps, k = carry
                block_idx, bsize, mu_b = block_data
                b_idx = padded_blocks[block_idx]
                mask = (jnp.arange(max_block_size) < bsize).astype(jnp.float64)

                def _mh_walker(x_full, lp, wk):
                    wk, k1, k2, k_accept1 = jax.random.split(wk, 4)
                    idx1 = jax.random.randint(k1, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jax.random.randint(k2, (), 0, z_padded.shape[0]) % z_count
                    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

                    # DE direction in block subspace
                    z1_block = z_padded[idx1, b_idx]
                    z2_block = z_padded[idx2, b_idx]
                    diff = (z1_block - z2_block) * mask
                    norm = jnp.linalg.norm(diff)

                    # Scale-aware step
                    dim_corr = jnp.sqrt(jnp.float64(bsize) / 2.0)
                    mu_eff = mu_b * norm / jnp.maximum(dim_corr, 1e-30)

                    # Full-space direction
                    d_block = diff / jnp.maximum(norm, 1e-30)
                    d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                    d_full = d_full.at[b_idx].add(d_block * mask)

                    # Stage 1: full-scale proposal
                    x_prop1 = x_full + mu_eff * d_full
                    lp_prop1 = lp_eval(log_prob_fn, x_prop1)
                    log_u1 = jnp.log(jax.random.uniform(k_accept1, dtype=jnp.float64) + 1e-30)
                    accept1 = log_u1 < (lp_prop1 - lp)

                    if use_delayed_rejection:
                        # Stage 2: smaller step, new DE direction
                        wk, k3, k4, k_accept2 = jax.random.split(wk, 4)
                        idx3 = jax.random.randint(k3, (), 0, z_padded.shape[0]) % z_count
                        idx4 = jax.random.randint(k4, (), 0, z_padded.shape[0]) % z_count
                        idx4 = jnp.where(idx3 == idx4, (idx3 + 1) % z_count, idx4)

                        z3_block = z_padded[idx3, b_idx]
                        z4_block = z_padded[idx4, b_idx]
                        diff2 = (z3_block - z4_block) * mask
                        norm2 = jnp.linalg.norm(diff2)
                        mu_eff2 = mu_b * norm2 / jnp.maximum(dim_corr, 1e-30) / 3.0

                        d_block2 = diff2 / jnp.maximum(norm2, 1e-30)
                        d_full2 = jnp.zeros(n_dim, dtype=jnp.float64)
                        d_full2 = d_full2.at[b_idx].add(d_block2 * mask)

                        x_prop2 = x_full + mu_eff2 * d_full2
                        lp_prop2 = lp_eval(log_prob_fn, x_prop2)
                        log_u2 = jnp.log(jax.random.uniform(k_accept2, dtype=jnp.float64) + 1e-30)
                        accept2 = log_u2 < (lp_prop2 - lp)

                        # Stage 1 wins if accepted, else stage 2, else stay
                        x_new = jnp.where(accept1, x_prop1,
                                jnp.where(accept2, x_prop2, x_full))
                        lp_new = jnp.where(accept1, lp_prop1,
                                 jnp.where(accept2, lp_prop2, lp))
                        either_accepted = accept1 | accept2
                    else:
                        x_new = jnp.where(accept1, x_prop1, x_full)
                        lp_new = jnp.where(accept1, lp_prop1, lp)
                        either_accepted = accept1

                    target = jnp.float64(n_expand + 1)
                    br = jnp.where(either_accepted, 2.0 * target - 1.0, 1.0)
                    return x_new, lp_new, wk, br

                k, k_walkers = jax.random.split(k)
                wkeys = jax.random.split(k_walkers, n_walkers)
                new_pos, new_lp, _, brs = jax.vmap(_mh_walker)(pos, lps, wkeys)
                mean_br = jnp.mean(brs)
                return (new_pos, new_lp, k), mean_br

            k, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, n_blocks_val)
            block_data = (perm, block_sizes_arr[perm], mu_blocks_arg[perm])

            (pos_out, lps_out, k_out), all_br = jax.lax.scan(
                _one_block_mh, (positions, log_probs, k), block_data,
            )
            mean_br = jnp.mean(all_br)
            return pos_out, lps_out, k_out, mean_br

    # --- Initial log-probs ---
    if verbose:
        print(f"  [{config.name}] Computing initial log-probs...", flush=True)
    positions = jnp.array(init_positions, dtype=jnp.float64)
    log_probs = jax.jit(jax.vmap(lambda x: lp_eval(log_prob_fn, x)))(positions)
    log_probs.block_until_ready()

    # Populate initial Z log probs
    z_log_probs = z_log_probs.at[:n_walkers].set(log_probs)

    # --- JIT compile ---
    _block_step_fn = block_mh_step if use_block_mh else block_sweep_step if use_block_gibbs else None
    if use_block_gibbs:
        # JIT compile block step
        step_label = "block_mh_step" if use_block_mh else "block_sweep_step"
        if verbose:
            print(f"  [{config.name}] JIT compiling {step_label}...", flush=True)
        t_jit = time.time()
        positions, log_probs, key, _ = _block_step_fn(
            positions, log_probs, z_padded, z_count, mu_blocks, key,
        )
        positions.block_until_ready()
        t_jit = time.time() - t_jit
        if verbose:
            print(f"  [{config.name}] JIT compile: {t_jit:.1f}s", flush=True)
        # Dummy aux values (not used by block-Gibbs but needed for variable scope)
        walker_aux_pd = walker_aux.prev_direction
        walker_aux_bw = walker_aux.bracket_widths
        walker_aux_da = walker_aux.direction_anchor
        walker_aux_ds = walker_aux.direction_scale
    else:
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

    # Dual averaging state (Nesterov 2009 / NUTS-style)
    # Works in log-space: log_mu_bar converges to optimal log(mu)
    log_mu = float(jnp.log(mu))
    log_mu_bar = log_mu  # running average of log(mu)
    H_bar = 0.0  # running average of adaptation statistic
    da_gamma = 0.05  # shrinkage toward initial mu
    da_t0 = 10  # stabilization offset
    da_kappa = 0.75  # forgetting rate

    # Warmup temperature schedule: start warm (T>1) for broader exploration,
    # linearly cool to T=1 over warmup. Disabled by default because the
    # extra log-prob re-evaluation per step adds overhead. Enable via
    # ens_kwargs["warmup_t_start"] for hard targets.
    warmup_t_start = float(ens_kwargs.get("warmup_t_start", 1.0))

    t_sample = time.time()
    for step in range(n_warmup):
        if use_block_gibbs:
            # Block-Gibbs warmup step
            positions, log_probs, key, br_mean = _block_step_fn(
                positions, log_probs, z_padded, z_count, mu_blocks, key,
            )
            # Fake found/br for compatibility with downstream code
            found = jnp.ones(n_walkers, dtype=jnp.bool_)
            br = jnp.full(n_walkers, br_mean, dtype=jnp.float64)
        else:
            # Linear temperature schedule
            if warmup_t_start > 1.0 and n_warmup > 0:
                frac = step / max(n_warmup - 1, 1)
                warmup_temp = warmup_t_start * (1.0 - frac) + 1.0 * frac
                warmup_temperatures = jnp.full(n_walkers, warmup_temp, dtype=jnp.float64)
            else:
                warmup_temperatures = temperatures

            (positions, log_probs, key, found, br,
             walker_aux_pd, walker_aux_bw, walker_aux_da, walker_aux_ds) = _call_step(
                step_fn, positions, log_probs, z_padded, z_count, z_log_probs,
                mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
                walker_aux_ds, warmup_temperatures,
            )

            # During warmup tempering, the stored lp_new is tempered (log_prob / temp).
            # Re-evaluate at T=1 so the next step sees the correct untempered value.
            if warmup_t_start > 1.0 and warmup_temp > 1.001:
                log_probs = jax.jit(jax.vmap(lambda x: lp_eval(log_prob_fn, x)))(positions)

        # Append to Z-matrix (every 5 steps to reduce overhead)
        if step % 5 == 0 or step == n_warmup - 1:
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

        # Tune mu (or mu_blocks for block-Gibbs)
        if tune and (step + 1) % TUNE_INTERVAL == 0:
            if use_block_gibbs:
                # Scale all mu_blocks by the same bracket-ratio adjustment
                med_ratio = float(jnp.median(br))
                ratio_ema = 0.3 * med_ratio + 0.7 * ratio_ema
                if ratio_ema > 0:
                    adjustment = (ratio_ema / TARGET_RATIO) ** 0.5
                    mu_blocks = jnp.clip(mu_blocks * adjustment, MU_MIN, MU_MAX)
                    mu = jnp.float64(jnp.mean(mu_blocks))
            elif config.tune_method == "esjd":
                if esjd_ema > best_esjd:
                    best_esjd = esjd_ema
                    best_mu = mu
                if (step + 1) % (TUNE_INTERVAL * 2) == 0:
                    mu = jnp.clip(best_mu * 1.2, MU_MIN, MU_MAX)
                else:
                    mu = jnp.clip(best_mu * 0.83, MU_MIN, MU_MAX)
            elif config.tune_method == "dual_avg":
                # Dual averaging (Nesterov 2009 / NUTS-style) in log-space.
                # Targets bracket_ratio ≈ TARGET_RATIO. The adaptation
                # statistic H = 1 - ratio/target is zero at the target.
                med_ratio = float(jnp.median(br))
                m_adapt = (step + 1) // TUNE_INTERVAL
                w = 1.0 / (m_adapt + da_t0)
                H_bar = (1.0 - w) * H_bar + w * (1.0 - med_ratio / TARGET_RATIO)
                log_mu = float(jnp.log(mu)) - (jnp.sqrt(m_adapt) / da_gamma) * H_bar
                eta = m_adapt ** (-da_kappa)
                log_mu_bar = eta * log_mu + (1.0 - eta) * log_mu_bar
                mu = jnp.clip(jnp.exp(log_mu), MU_MIN, MU_MAX)
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

        if verbose:
            elapsed = time.time() - t_sample
            speed = (step + 1) / elapsed
            best = float(log_probs.max())
            med_br = float(jnp.median(br))
            print(f"  [{config.name}] warmup {step+1:6d}/{n_warmup} "
                  f"({speed:.1f} it/s) best_lp={best:.1f} z_count={int(z_count)} "
                  f"mu={float(mu):.4f} bracket_ratio={med_br:.1f}", flush=True)

    # Warmup auto-extension: if ESJD is still changing rapidly or mu
    # hasn't stabilized, extend warmup by up to 2x the original length.
    # (Not used for block-Gibbs which has its own mu_blocks tuning.)
    if tune and n_warmup > 100 and esjd_ema > 0 and not use_block_gibbs:
        # Check: is mu still changing significantly between tune intervals?
        # Use the last bracket_ratio as a proxy — if it's far from target,
        # warmup hasn't converged.
        last_br = float(jnp.median(br)) if n_warmup > 0 else TARGET_RATIO
        mu_unstable = abs(last_br - TARGET_RATIO) > TARGET_RATIO * 0.5
        max_extra = n_warmup  # at most double the warmup
        extra_done = 0

        if mu_unstable and config.tune_method == "bracket":
            if verbose:
                print(f"  [{config.name}] Warmup auto-extending (bracket_ratio={last_br:.1f}, "
                      f"target={TARGET_RATIO:.1f})...", flush=True)
            for extra_step in range(max_extra):
                (positions, log_probs, key, found, br,
                 walker_aux_pd, walker_aux_bw, walker_aux_da, walker_aux_ds) = _call_step(
                    step_fn, positions, log_probs, z_padded, z_count, z_log_probs,
                    mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
                    walker_aux_ds, temperatures,
                )
                z_padded, z_count, z_log_probs = circular_zmatrix.append(
                    z_padded, z_count, z_log_probs, positions, log_probs, z_max_size,
                )
                extra_done += 1
                if tune and (extra_step + 1) % TUNE_INTERVAL == 0:
                    med_ratio = float(jnp.median(br))
                    ratio_ema = 0.3 * med_ratio + 0.7 * ratio_ema
                    if ratio_ema > 0:
                        adjustment = (ratio_ema / TARGET_RATIO) ** 0.5
                        mu = jnp.clip(mu * adjustment, MU_MIN, MU_MAX)
                    # Check convergence
                    if abs(med_ratio - TARGET_RATIO) <= TARGET_RATIO * 0.3:
                        break
            if verbose:
                print(f"  [{config.name}] Extended warmup by {extra_done} steps "
                      f"(bracket_ratio now {float(jnp.median(br)):.1f})", flush=True)

    # Finalize adaptive tuning
    if config.tune_method == "esjd" and tune and n_warmup > 0:
        mu = best_mu
    elif config.tune_method == "dual_avg" and tune and n_warmup > 0:
        mu = jnp.clip(jnp.exp(log_mu_bar), MU_MIN, MU_MAX)
        if verbose:
            print(f"  [{config.name}] Dual-avg final mu: {float(mu):.4f}", flush=True)

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

    # Compute empirical covariance Cholesky if using mh_adaptive or cov_aware width
    cov_chol_updated = False
    if (config.slice_fn == "mh_adaptive" or config.width == "cov_aware") and n_warmup > 0:
        if verbose:
            print(f"  [{config.name}] Computing empirical covariance Cholesky...", flush=True)
        cov_chol = mh_adaptive_slice.compute_cov_chol(z_padded, z_count)
        cov_chol_updated = True
        if verbose:
            diag_std = float(jnp.sqrt(jnp.diag(cov_chol @ cov_chol.T).mean()))
            print(f"  [{config.name}] cov_chol ready: mean_std={diag_std:.3f}", flush=True)

    # Adaptive direction switching: measure empirical condition number and switch
    # de_mcz → pca when the target appears ill-conditioned (unimodal, high aspect ratio).
    # Unimodal ill-conditioned targets benefit 2x from PCA directions (clean eigenvector
    # proposals vs noisy DE-MCz mixture directions). Multimodal targets (low condition
    # number, ~10) correctly stay with de_mcz for inter-mode exploration.
    if (cov_pca_threshold > 0
            and config.direction == "de_mcz"
            and cov_chol_updated):
        Sigma = cov_chol @ cov_chol.T
        eigvals = jnp.linalg.eigvalsh(Sigma)
        empirical_condition = float(eigvals[-1] / jnp.maximum(eigvals[0], jnp.float64(1e-30)))
        if empirical_condition > cov_pca_threshold:
            config = config._replace(direction="pca")
            dir_mod = DIRECTION_STRATEGIES["pca"]   # update module to match new config
            pca_components, pca_weights = pca.compute_pca_components(z_padded, z_count)
            if verbose:
                print(f"  [{config.name}] Auto-PCA: empirical_condition={empirical_condition:.1f} > threshold={cov_pca_threshold:.1f}", flush=True)

    # Compute KDE bandwidth if using KDE directions
    if config.direction == "kde" and n_warmup > 0:
        if verbose:
            print(f"  [{config.name}] Computing KDE bandwidth...", flush=True)
        kde_bandwidth = kde_direction.compute_kde_bandwidth(z_padded, z_count)

    # Fit GMM if using global_move directions
    if config.direction == "global_move" and n_warmup > 0:
        from dezess.directions.gmm import fit_gmm
        n_comp = dir_kwargs.get("n_components", 2)
        if verbose:
            print(f"  [{config.name}] Fitting GMM ({n_comp} components)...", flush=True)
        key, k_gmm = jax.random.split(key)
        gmm_means, gmm_covs, gmm_weights, gmm_chols = fit_gmm(
            z_padded, z_count, n_components=n_comp,
            n_iter=dir_kwargs.get("gmm_n_iter", 100),
            key=k_gmm,
        )
        dir_kwargs["gmm_means"] = gmm_means
        dir_kwargs["gmm_covs"] = gmm_covs
        dir_kwargs["gmm_weights"] = gmm_weights
        dir_kwargs["gmm_chols"] = gmm_chols
        if verbose:
            print(f"  [{config.name}] GMM fitted: weights={np.array(gmm_weights).round(3)}", flush=True)

    # Compute per-block Cholesky factors for covariance-adapted proposals
    if use_block_cov and n_warmup > 0:
        if verbose:
            print(f"  [{config.name}] Computing per-block covariance...", flush=True)
        block_L_padded = np.zeros((n_blocks_val, max_block_size, max_block_size),
                                  dtype=np.float64)
        z_np = np.array(z_padded[:int(z_count)])
        for bi, b in enumerate(blocks):
            b_np = np.array(b)
            bsz = len(b_np)
            z_block = z_np[:, b_np[:bsz]]
            if z_block.shape[0] > 2 * bsz:
                cov_b = np.cov(z_block.T)
                cov_b += 1e-8 * np.eye(bsz)
                L_b = np.linalg.cholesky(cov_b)
                block_L_padded[bi, :bsz, :bsz] = L_b
            else:
                block_L_padded[bi, :bsz, :bsz] = np.eye(bsz)
        block_L_padded = jnp.array(block_L_padded, dtype=jnp.float64)
        if verbose:
            print(f"  [{config.name}] Block covariance ready ({n_blocks_val} blocks)", flush=True)
    elif use_block_cov:
        block_L_padded = jnp.zeros((n_blocks_val, max_block_size, max_block_size),
                                    dtype=jnp.float64)
        for bi, b in enumerate(blocks):
            bsz = len(b)
            block_L_padded = block_L_padded.at[bi, :bsz, :bsz].set(jnp.eye(bsz))
    else:
        block_L_padded = None

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
    # (skip for block-Gibbs which uses its own JIT-compiled step)
    if not use_block_gibbs and (needs_rejit or config.direction in ("pca", "flow", "whitened", "kde", "global_move") or config.slice_fn == "mh_adaptive" or config.width == "cov_aware"):
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

    # Re-JIT block step if block covariance was computed
    if use_block_cov and n_warmup > 0:
        @jax.jit
        def block_mh_cov_step(positions, log_probs, z_padded, z_count,
                              mu_blocks_arg, key):

            def _one_block_mh(carry, block_data):
                pos, lps, k = carry
                block_idx, bsize, mu_b = block_data
                b_idx = padded_blocks[block_idx]
                mask = (jnp.arange(max_block_size) < bsize).astype(jnp.float64)

                def _mh_walker(x_full, lp, wk):
                    wk, k1, k2, k_accept1 = jax.random.split(wk, 4)

                    # Covariance-adapted proposal: L @ z
                    L_b = block_L_padded[block_idx]  # (max_block_size, max_block_size)
                    z_rand = jax.random.normal(k1, (max_block_size,), dtype=jnp.float64)
                    z_rand = z_rand * mask  # zero out padding dims
                    delta = L_b @ z_rand
                    delta = delta * mask
                    norm = jnp.linalg.norm(delta)
                    d_block = delta / jnp.maximum(norm, 1e-30)

                    dim_corr = jnp.sqrt(jnp.float64(bsize) / 2.0)
                    mu_eff = mu_b * norm / jnp.maximum(dim_corr, 1e-30)

                    d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                    d_full = d_full.at[b_idx].add(d_block * mask)

                    # Stage 1
                    x_prop1 = x_full + mu_eff * d_full
                    lp_prop1 = lp_eval(log_prob_fn, x_prop1)
                    log_u1 = jnp.log(jax.random.uniform(k_accept1, dtype=jnp.float64) + 1e-30)
                    accept1 = log_u1 < (lp_prop1 - lp)

                    if use_delayed_rejection:
                        # Stage 2: smaller step, new random direction from covariance
                        wk, k3, k_accept2 = jax.random.split(wk, 3)
                        z_rand2 = jax.random.normal(k3, (max_block_size,), dtype=jnp.float64)
                        z_rand2 = z_rand2 * mask
                        delta2 = L_b @ z_rand2
                        delta2 = delta2 * mask
                        norm2 = jnp.linalg.norm(delta2)
                        d_block2 = delta2 / jnp.maximum(norm2, 1e-30)
                        mu_eff2 = mu_b * norm2 / jnp.maximum(dim_corr, 1e-30) / 3.0

                        d_full2 = jnp.zeros(n_dim, dtype=jnp.float64)
                        d_full2 = d_full2.at[b_idx].add(d_block2 * mask)

                        x_prop2 = x_full + mu_eff2 * d_full2
                        lp_prop2 = lp_eval(log_prob_fn, x_prop2)
                        log_u2 = jnp.log(jax.random.uniform(k_accept2, dtype=jnp.float64) + 1e-30)
                        accept2 = log_u2 < (lp_prop2 - lp)

                        x_new = jnp.where(accept1, x_prop1,
                                jnp.where(accept2, x_prop2, x_full))
                        lp_new = jnp.where(accept1, lp_prop1,
                                 jnp.where(accept2, lp_prop2, lp))
                        either_accepted = accept1 | accept2
                    else:
                        x_new = jnp.where(accept1, x_prop1, x_full)
                        lp_new = jnp.where(accept1, lp_prop1, lp)
                        either_accepted = accept1

                    target = jnp.float64(n_expand + 1)
                    br = jnp.where(either_accepted, 2.0 * target - 1.0, 1.0)
                    return x_new, lp_new, wk, br

                k, k_walkers = jax.random.split(k)
                wkeys = jax.random.split(k_walkers, n_walkers)
                new_pos, new_lp, _, brs = jax.vmap(_mh_walker)(pos, lps, wkeys)
                mean_br = jnp.mean(brs)
                return (new_pos, new_lp, k), mean_br

            k, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, n_blocks_val)
            block_data = (perm, block_sizes_arr[perm], mu_blocks_arg[perm])

            (pos_out, lps_out, k_out), all_br = jax.lax.scan(
                _one_block_mh, (positions, log_probs, k), block_data,
            )
            mean_br = jnp.mean(all_br)
            return pos_out, lps_out, k_out, mean_br

        _block_step_fn = block_mh_cov_step

        if verbose:
            print(f"  [{config.name}] Re-JIT compiling with block covariance...", flush=True)
        t_rejit = time.time()
        positions, log_probs, key, _ = _block_step_fn(
            positions, log_probs, z_padded, z_count, mu_blocks, key,
        )
        positions.block_until_ready()
        if verbose:
            print(f"  [{config.name}] Re-JIT: {time.time()-t_rejit:.1f}s", flush=True)

    if use_conditional:
        @jax.jit
        def block_conditional_step(positions, log_probs, z_padded, z_count,
                                   mu_blocks_arg, key):
            # ---- Round 1: Update potential block (block 0) with full_log_prob ----
            pot_b_idx = padded_blocks[0]
            pot_bsize = block_sizes_arr[0]
            pot_mu = mu_blocks_arg[0]
            pot_mask = (jnp.arange(max_block_size) < pot_bsize).astype(jnp.float64)

            def _mh_pot_walker(x_full, lp, wk):
                wk, k1, k2, k_accept = jax.random.split(wk, 4)
                idx1 = jax.random.randint(k1, (), 0, z_padded.shape[0]) % z_count
                idx2 = jax.random.randint(k2, (), 0, z_padded.shape[0]) % z_count
                idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

                z1_b = z_padded[idx1, pot_b_idx]
                z2_b = z_padded[idx2, pot_b_idx]
                diff = (z1_b - z2_b) * pot_mask
                norm = jnp.linalg.norm(diff)
                dim_corr = jnp.sqrt(jnp.float64(pot_bsize) / 2.0)
                mu_eff = pot_mu * norm / jnp.maximum(dim_corr, 1e-30)

                d_block = diff / jnp.maximum(norm, 1e-30)
                d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                d_full = d_full.at[pot_b_idx].add(d_block * pot_mask)

                x_prop = x_full + mu_eff * d_full
                lp_prop = lp_eval(log_prob_fn, x_prop)
                log_u = jnp.log(jax.random.uniform(k_accept, dtype=jnp.float64) + 1e-30)
                accept = log_u < (lp_prop - lp)
                x_new = jnp.where(accept, x_prop, x_full)
                lp_new = jnp.where(accept, lp_prop, lp)
                return x_new, lp_new, wk

            key, k_pot = jax.random.split(key)
            pot_keys = jax.random.split(k_pot, n_walkers)
            positions, log_probs, _ = jax.vmap(_mh_pot_walker)(
                positions, log_probs, pot_keys)

            # ---- Round 2: Update all stream blocks in parallel ----
            cond_block_indices = jnp.arange(n_conditional_blocks, dtype=jnp.int32)
            cond_bsizes = block_sizes_arr[1:]         # (n_cond,)
            cond_mus = mu_blocks_arg[1:]               # (n_cond,)
            cond_b_idxs = padded_blocks[1:]            # (n_cond, max_block_size)

            def _mh_one_stream(x_full, wk, stream_idx):
                """MH update for one stream block. Returns proposed position and accept flag."""
                wk, k1, k2, k_accept = jax.random.split(wk, 4)
                s_b_idx = cond_b_idxs[stream_idx]
                s_bsize = cond_bsizes[stream_idx]
                s_mu = cond_mus[stream_idx]
                s_mask = (jnp.arange(max_block_size) < s_bsize).astype(jnp.float64)

                idx1 = jax.random.randint(k1, (), 0, z_padded.shape[0]) % z_count
                idx2 = jax.random.randint(k2, (), 0, z_padded.shape[0]) % z_count
                idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)

                z1_b = z_padded[idx1, s_b_idx]
                z2_b = z_padded[idx2, s_b_idx]
                diff = (z1_b - z2_b) * s_mask
                norm = jnp.linalg.norm(diff)
                dim_corr = jnp.sqrt(jnp.float64(s_bsize) / 2.0)
                mu_eff = s_mu * norm / jnp.maximum(dim_corr, 1e-30)

                d_block = diff / jnp.maximum(norm, 1e-30)
                d_full = jnp.zeros(n_dim, dtype=jnp.float64)
                d_full = d_full.at[s_b_idx].add(d_block * s_mask)

                x_prop = x_full + mu_eff * d_full

                # Conditional log_prob for MH ratio
                cond_lp_old = conditional_log_prob_fn(x_full, stream_idx)
                cond_lp_new = conditional_log_prob_fn(x_prop, stream_idx)

                log_u = jnp.log(jax.random.uniform(k_accept, dtype=jnp.float64) + 1e-30)
                accept = log_u < (cond_lp_new - cond_lp_old)
                return x_prop, accept

            def _update_all_streams_one_walker(x_full, wk):
                """Update all stream blocks for one walker, return merged position."""
                wk_streams = jax.random.split(wk, n_conditional_blocks)

                # vmap over stream indices: propose for each stream independently
                x_proposals, accepts = jax.vmap(
                    lambda wk_s, s_idx: _mh_one_stream(x_full, wk_s, s_idx),
                    in_axes=(0, 0)
                )(wk_streams, cond_block_indices)
                # x_proposals: (n_cond, ndim), accepts: (n_cond,)

                # Merge accepted blocks back into x_full
                def _merge_one_stream(carry, stream_data):
                    x_merged = carry
                    s_idx, x_prop_s, accepted_s = stream_data
                    s_b_idx_s = cond_b_idxs[s_idx]
                    s_bsize_s = cond_bsizes[s_idx]
                    s_mask_s = (jnp.arange(max_block_size) < s_bsize_s).astype(jnp.float64)

                    proposed_block = x_prop_s[s_b_idx_s]
                    current_block = x_merged[s_b_idx_s]
                    new_block = jnp.where(accepted_s, proposed_block, current_block)
                    # Only overwrite valid indices (mask out padding)
                    final_block = new_block * s_mask_s + x_merged[s_b_idx_s] * (1.0 - s_mask_s)
                    x_merged = x_merged.at[s_b_idx_s].set(final_block)
                    return x_merged, None

                x_merged, _ = jax.lax.scan(
                    _merge_one_stream, x_full,
                    (cond_block_indices, x_proposals, accepts),
                )
                return x_merged

            key, k_streams = jax.random.split(key)
            stream_keys = jax.random.split(k_streams, n_walkers)
            positions = jax.vmap(_update_all_streams_one_walker)(positions, stream_keys)

            # ---- Round 3: Re-evaluate full log_prob for consistency ----
            log_probs = jax.vmap(lambda x: lp_eval(log_prob_fn, x))(positions)

            mean_br = jnp.float64(n_expand + 1)  # neutral bracket ratio
            return positions, log_probs, key, mean_br

        _block_step_fn = block_conditional_step

        if verbose:
            print(f"  [{config.name}] JIT compiling conditional step...", flush=True)
        t_rejit = time.time()
        positions, log_probs, key, _ = _block_step_fn(
            positions, log_probs, z_padded, z_count, mu_blocks, key,
        )
        positions.block_until_ready()
        if verbose:
            print(f"  [{config.name}] Conditional JIT: {time.time()-t_rejit:.1f}s", flush=True)

    # --- Production (Z-matrix frozen) ---
    n_production = n_steps - n_warmup
    all_samples = np.zeros((n_production, n_walkers, n_dim), dtype=np.float64)
    all_log_probs = np.zeros((n_production, n_walkers), dtype=np.float64)
    all_found = np.zeros((n_production, n_walkers), dtype=np.bool_)
    all_bracket_ratios = np.zeros((n_production, n_walkers), dtype=np.float64)

    z_frozen = z_padded
    z_count_frozen = z_count
    z_lp_frozen = z_log_probs
    live_z = config.zmatrix == "live"
    live_update_rate = float(zmat_kwargs.get("update_rate", 0.01))

    if verbose:
        mode = "live" if live_z else "frozen"
        print(f"  [{config.name}] Z-matrix {mode} at {int(z_count_frozen)} entries",
              flush=True)

    # Streaming diagnostics for live ESS/R-hat monitoring
    stream_diag = StreamingDiagnostics(n_walkers, n_dim)

    # Build a batched scan step for reduced Python overhead.
    # Batches SCAN_BATCH steps into a single JIT call via lax.scan.
    SCAN_BATCH = 50
    # scan doesn't support swaps, live Z-matrix, or block-Gibbs
    # (block-Gibbs permutes blocks each step so can't batch)
    use_scan = (config.ensemble not in ("parallel_tempering", "block_gibbs")) and not live_z

    if use_scan:
        @jax.jit
        def _scan_steps(carry, _):
            (pos, lps, k, pd, bw, da, ds) = carry
            (pos, lps, k, found, br, pd, bw, da, ds) = step_fn(
                pos, lps, z_frozen, z_count_frozen, z_lp_frozen,
                mu, k, pd, bw, da, ds, temperatures,
                pca_components, pca_weights, flow_directions, whitening_matrix, kde_bandwidth,
                cov_chol,
            )
            return (pos, lps, k, pd, bw, da, ds), (pos, lps, found, br)

        def _run_batch(positions, log_probs, key, walker_aux_pd, walker_aux_bw,
                       walker_aux_da, walker_aux_ds, n_batch):
            carry = (positions, log_probs, key, walker_aux_pd, walker_aux_bw,
                     walker_aux_da, walker_aux_ds)
            carry, outputs = jax.lax.scan(_scan_steps, carry, None, length=n_batch)
            (positions, log_probs, key, walker_aux_pd, walker_aux_bw,
             walker_aux_da, walker_aux_ds) = carry
            batch_samples, batch_lps, batch_found, batch_br = outputs
            return (positions, log_probs, key, walker_aux_pd, walker_aux_bw,
                    walker_aux_da, walker_aux_ds,
                    batch_samples, batch_lps, batch_found, batch_br)

    t_prod = time.time()
    step_idx = 0

    while step_idx < n_production:
        # Determine batch size
        remaining = n_production - step_idx
        batch_sz = min(SCAN_BATCH, remaining) if use_scan else 1

        if use_scan and batch_sz > 1:
            # Batched scan
            (positions, log_probs, key, walker_aux_pd, walker_aux_bw,
             walker_aux_da, walker_aux_ds,
             batch_samples, batch_lps, batch_found, batch_br) = _run_batch(
                positions, log_probs, key, walker_aux_pd, walker_aux_bw,
                walker_aux_da, walker_aux_ds, batch_sz,
            )
            b_samples = np.asarray(batch_samples)
            b_lps = np.asarray(batch_lps)
            b_found = np.asarray(batch_found)
            b_br = np.asarray(batch_br)
            for i in range(batch_sz):
                all_samples[step_idx + i] = b_samples[i]
                all_log_probs[step_idx + i] = b_lps[i]
                all_found[step_idx + i] = b_found[i]
                all_bracket_ratios[step_idx + i] = b_br[i]
                stream_diag.update(b_samples[i], b_lps[i])
            step_idx += batch_sz
        else:
            # Single step (fallback for parallel tempering, block-Gibbs, or last batch)
            if use_block_gibbs:
                positions, log_probs, key, br_mean = _block_step_fn(
                    positions, log_probs, z_frozen, z_count_frozen,
                    mu_blocks, key,
                )
                found = jnp.ones(n_walkers, dtype=jnp.bool_)
                br = jnp.full(n_walkers, br_mean, dtype=jnp.float64)
            else:
                (positions, log_probs, key, found, br,
                 walker_aux_pd, walker_aux_bw, walker_aux_da, walker_aux_ds) = step_fn(
                    positions, log_probs, z_frozen, z_count_frozen, z_lp_frozen,
                    mu, key, walker_aux_pd, walker_aux_bw, walker_aux_da,
                    walker_aux_ds, temperatures,
                    pca_components, pca_weights, flow_directions, whitening_matrix, kde_bandwidth,
                    cov_chol,
                )
            if config.ensemble == "parallel_tempering":
                key, k_swap = jax.random.split(key)
                positions, log_probs = parallel_tempering.propose_swaps(
                    positions, log_probs, temperatures, k_swap,
                    n_temps=ens_kwargs.get("n_temps", 4),
                )
            pos_np = np.asarray(positions)
            all_samples[step_idx] = pos_np
            all_log_probs[step_idx] = np.asarray(log_probs)
            all_found[step_idx] = np.asarray(found)
            all_bracket_ratios[step_idx] = np.asarray(br)
            stream_diag.update(pos_np, np.asarray(log_probs))

            # Live Z-matrix: update archive during production
            if live_z:
                key, k_live = jax.random.split(key)
                z_frozen, z_count_frozen, z_lp_frozen = live_zmatrix.append(
                    z_frozen, z_count_frozen, z_lp_frozen,
                    positions, log_probs, z_max_size,
                    update_rate=live_update_rate, key=k_live,
                )

            step_idx += 1

        if verbose:
            elapsed = time.time() - t_prod
            speed = step_idx / elapsed if elapsed > 0 else 0
            eta = (n_production - step_idx) / speed if speed > 0 else 0
            best = float(all_log_probs[:step_idx].max())
            mean_lp = float(all_log_probs[max(0,step_idx-200):step_idx].mean())
            zero_rate = 1.0 - all_found[max(0,step_idx-200):step_idx].mean()
            diag = stream_diag.summary()
            print(f"  [{config.name}] {step_idx:6d}/{n_production} "
                  f"({speed:.1f} it/s, ETA {eta:.0f}s) "
                  f"best_lp={best:.1f} mean_lp={mean_lp:.1f} "
                  f"zero_move={zero_rate:.4f} "
                  f"ESS={diag['ess_min']:.0f} rhat={diag['rhat_max']:.3f}",
                  flush=True)

        # Progress callback
        if progress_fn is not None and step_idx % 50 < batch_sz:
            elapsed = time.time() - t_prod
            progress_fn({
                "step": step_idx,
                "n_steps": n_production,
                "ess_min": stream_diag.ess_min(),
                "rhat_max": stream_diag.rhat_max(),
                "speed": step_idx / elapsed if elapsed > 0 else 0,
            })

        # Early stopping: stop when target ESS is reached
        if target_ess is not None and step_idx >= 100:
            current_ess = stream_diag.ess_min()
            if current_ess >= target_ess:
                if verbose:
                    print(f"  [{config.name}] Early stop: ESS={current_ess:.0f} >= "
                          f"target={target_ess:.0f} at step {step_idx}/{n_production}",
                          flush=True)
                n_production = step_idx
                break

    wall_time = time.time() - t_prod

    # --- Map samples back to x-space ---
    if transform is not None:
        all_samples_x = np.zeros_like(all_samples[:n_production])
        _fwd_vmap = jax.jit(jax.vmap(transform.forward))
        for i in range(n_production):
            all_samples_x[i] = np.asarray(_fwd_vmap(jnp.array(all_samples[i])))
        all_samples[:n_production] = all_samples_x

    result = {
        "samples": jnp.array(all_samples[:n_production]),
        "log_prob": jnp.array(all_log_probs[:n_production]),
        "mu": float(mu),
        "z_matrix": z_frozen[:int(z_count_frozen)],
        "config": config,
        "wall_time": wall_time,
        "n_production": n_production,
        "diagnostics": {
            "found": all_found[:n_production],
            "bracket_ratios": all_bracket_ratios[:n_production],
            "n_expand": n_expand,
            "n_shrink": n_shrink,
            "cap_hit_rate": 1.0 - all_found[:n_production].mean(),
            "streaming": stream_diag.summary(),
        },
    }
    if use_block_gibbs:
        result["mu_blocks"] = np.array(mu_blocks)
    return result
