"""Multi-Try DE-MCz: K parallel proposals with MTM acceptance.

Instead of sequential slice stepping along one direction, generate K
proposals along K different DE-MCz directions and evaluate all in parallel.
Select the best via categorical weighting, accept via Multiple-Try
Metropolis (Liu, Liang & Wong 2000).

Cost: 2K evaluations per step (K forward + K reverse), all parallel.
Explores K directions simultaneously — in 63D, this is far more
informative than K points along a single direction.

The proposal q is symmetric (DE-MCz with frozen Z-matrix, random pair
selection), so the MTM weight function simplifies to w(y) = pi(y).
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
    n_expand: int = 8,
    n_shrink: int = 12,
    z_matrix: Array = None,
    z_count: Array = None,
    base_mu: Array = None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Multi-try DE-MCz with MTM acceptance.

    Parameters
    ----------
    n_expand : int
        Number of proposals K. Total evals: 2K (K forward + K reverse).
        All evals within each batch are parallel via vmap.
    n_shrink : int
        Unused (interface compatibility).
    z_matrix : array (z_max_size, ndim)
        Frozen Z-matrix for drawing DE direction pairs.
    z_count : array (int32 scalar)
        Number of valid entries in z_matrix.
    base_mu : array (float64 scalar)
        Base step size (before scale-aware adjustment).

    Returns ``(x_new, lp_new, key, accepted, L, R)``.
    """
    K = n_expand
    ndim = x.shape[0]
    dim_corr = jnp.sqrt(jnp.float64(ndim) / 2.0)
    mu_base = base_mu

    # ================================================================
    # Phase 1: Generate K DE-MCz directions from Z-matrix
    # ================================================================
    key, k1, k2 = jax.random.split(key, 3)
    idx1s = jax.random.randint(k1, (K,), 0, z_matrix.shape[0]) % z_count
    idx2s = jax.random.randint(k2, (K,), 0, z_matrix.shape[0]) % z_count
    idx2s = jnp.where(idx1s == idx2s, (idx2s + 1) % z_count, idx2s)

    diffs = z_matrix[idx1s] - z_matrix[idx2s]        # (K, ndim)
    norms = jnp.linalg.norm(diffs, axis=1)    # (K,)
    dirs = diffs / jnp.maximum(norms, 1e-30)[:, None] # (K, ndim)

    # Scale-aware width per direction
    mu_effs = mu_base * norms / jnp.maximum(dim_corr, 1e-30)  # (K,)

    # ================================================================
    # Phase 2: Generate K proposals and evaluate in parallel
    # ================================================================
    proposals = x[None, :] + mu_effs[:, None] * dirs  # (K, ndim)
    lp_proposals = jax.vmap(
        lambda pt: safe_log_prob(log_prob_fn, pt)
    )(proposals)  # (K,)

    # ================================================================
    # Phase 3: Select proposal j via Gumbel-max (categorical ~ pi)
    # ================================================================
    key, k_sel = jax.random.split(key)
    gumbel = -jnp.log(-jnp.log(
        jax.random.uniform(k_sel, (K,), dtype=jnp.float64) + 1e-30
    ) + 1e-30)
    j = jnp.argmax(lp_proposals + gumbel)
    y = proposals[j]
    lp_y = lp_proposals[j]

    # ================================================================
    # Phase 4: Generate K reference points from y and evaluate
    # ================================================================
    key, k3, k4 = jax.random.split(key, 3)
    ref_idx1s = jax.random.randint(k3, (K,), 0, z_matrix.shape[0]) % z_count
    ref_idx2s = jax.random.randint(k4, (K,), 0, z_matrix.shape[0]) % z_count
    ref_idx2s = jnp.where(
        ref_idx1s == ref_idx2s, (ref_idx1s + 1) % z_count, ref_idx2s
    )

    ref_diffs = z_matrix[ref_idx1s] - z_matrix[ref_idx2s]
    ref_norms = jnp.linalg.norm(ref_diffs, axis=1)
    ref_dirs = ref_diffs / jnp.maximum(ref_norms, 1e-30)[:, None]
    ref_mu_effs = mu_base * ref_norms / jnp.maximum(dim_corr, 1e-30)

    references = y[None, :] + ref_mu_effs[:, None] * ref_dirs  # (K, ndim)
    # Current state x takes the j-th reference slot
    references = references.at[j].set(x)

    lp_refs = jax.vmap(
        lambda pt: safe_log_prob(log_prob_fn, pt)
    )(references)  # (K,)
    # Use known lp_x for the current-state reference
    lp_refs = lp_refs.at[j].set(lp_x)

    # ================================================================
    # Phase 5: MTM accept/reject
    # ================================================================
    W_forward = jax.scipy.special.logsumexp(lp_proposals)
    W_reverse = jax.scipy.special.logsumexp(lp_refs)

    key, k_accept = jax.random.split(key)
    log_alpha = jnp.minimum(jnp.float64(0.0), W_forward - W_reverse)
    accept = (jnp.log(jax.random.uniform(k_accept, dtype=jnp.float64) + 1e-30)
              < log_alpha)

    x_new = jnp.where(accept, y, x)
    lp_new = jnp.where(accept, lp_y, lp_x)

    # Return bracket-like values for mu tuning compatibility.
    # Use mean proposal displacement / mu as a proxy for bracket ratio.
    mean_disp = jnp.mean(mu_effs)
    L = -mean_disp
    R = mean_disp

    return x_new, lp_new, key, accept, L, R
