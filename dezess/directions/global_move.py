"""Global Move direction: GMM-based inter-mode jumps.

Fits a Gaussian Mixture Model to the Z-matrix after warmup, then
proposes directions connecting the current walker to a different
mixture component. This enables mode-jumping in multimodal posteriors.

Mixes local DE-MCz directions (probability 1-global_prob) with
global GMM directions (probability global_prob). Both directions
are computed every step; jnp.where selects which one to use.

Based on zeus's GlobalMove (Karamanis et al. 2021).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dezess.directions import de_mcz

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def sample_direction(
    x_i: Array,
    z_matrix: Array,
    z_count: Array,
    z_log_probs: Array,
    key: Array,
    aux: Array,
    gmm_means: Array = None,
    gmm_covs: Array = None,
    gmm_weights: Array = None,
    gmm_chols: Array = None,
    global_prob: float = 0.1,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample a direction — local DE-MCz or global GMM jump.

    During warmup (gmm_means=None), always uses DE-MCz.
    During production, uses GMM jump with probability global_prob.
    """
    # Always compute local direction (DE-MCz fallback)
    key, k_local = jax.random.split(key)
    d_local, k_local_out, aux_local = de_mcz.sample_direction(
        x_i, z_matrix, z_count, z_log_probs, k_local, aux)

    if gmm_means is None:
        return d_local, key, aux_local

    K = gmm_means.shape[0]
    ndim = x_i.shape[0]

    # Compute global direction
    key, k_global, k_target, k_select = jax.random.split(key, 4)

    # Identify which component x_i belongs to (highest responsibility)
    def _log_resp(k):
        diff = x_i - gmm_means[k]
        prec = jnp.linalg.inv(gmm_covs[k])
        mahal = diff @ prec @ diff
        _, log_det = jnp.linalg.slogdet(gmm_covs[k])
        return -0.5 * (mahal + log_det) + jnp.log(gmm_weights[k])

    log_resps = jax.vmap(_log_resp)(jnp.arange(K))
    current_component = jnp.argmax(log_resps)

    # Select target component != current (weighted by gmm_weights)
    target_weights = gmm_weights.at[current_component].set(0.0)
    target_weights = target_weights / jnp.maximum(jnp.sum(target_weights), 1e-30)
    target_component = jax.random.choice(k_target, K, p=target_weights)

    # Sample target point from target component
    z = jax.random.normal(k_global, (ndim,), dtype=jnp.float64)
    target_point = gmm_means[target_component] + gmm_chols[target_component] @ z

    # Direction = target - current position (unnormalized)
    diff_global = target_point - x_i
    norm_global = jnp.linalg.norm(diff_global)
    d_global = diff_global / jnp.maximum(norm_global, 1e-30)

    # Choose local or global
    use_global = jax.random.uniform(k_select, dtype=jnp.float64) < global_prob
    d = jnp.where(use_global, d_global, d_local)
    norm = jnp.where(use_global, norm_global, aux_local.direction_scale)

    aux_out = aux_local._replace(direction_scale=norm)
    return d, key, aux_out
