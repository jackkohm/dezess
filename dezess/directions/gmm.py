"""Gaussian Mixture Model fitting via EM in pure JAX.

Used by the global_move direction strategy to identify modes
in the Z-matrix and propose inter-mode jumps.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def _kmeans_pp_init(data: Array, n_data: int, n_components: int, key: Array) -> Array:
    """K-means++ initialization for GMM means."""
    ndim = data.shape[1]
    centers = jnp.zeros((n_components, ndim), dtype=jnp.float64)

    # First center: random sample
    key, k0 = jax.random.split(key)
    idx = jax.random.randint(k0, (), 0, n_data)
    centers = centers.at[0].set(data[idx])

    def _pick_next(i, state):
        centers, key = state
        # Mask out centers that haven't been set yet (indices >= i).
        # This avoids dynamic slicing (centers[:i]) inside fori_loop.
        mask = jnp.arange(n_components) < i  # (K,) bool
        def _dist_to_nearest(x):
            sq_dists = jnp.sum((centers - x) ** 2, axis=1)  # (K,)
            # Replace uninitialized centers with +inf so they don't win
            sq_dists = jnp.where(mask, sq_dists, jnp.inf)
            return jnp.min(sq_dists)
        dists = jax.vmap(_dist_to_nearest)(data[:n_data])
        key, k = jax.random.split(key)
        probs = dists / jnp.maximum(jnp.sum(dists), 1e-30)
        idx = jax.random.choice(k, n_data, p=probs)
        centers = centers.at[i].set(data[idx])
        return centers, key

    centers, _ = lax.fori_loop(1, n_components, _pick_next, (centers, key))
    return centers


def fit_gmm(
    data: Array,
    n_data: Array,
    n_components: int = 2,
    n_iter: int = 100,
    key: Array = None,
    reg: float = 1e-6,
) -> tuple[Array, Array, Array, Array]:
    """Fit GMM via EM algorithm.

    Parameters
    ----------
    data : (max_n, ndim)
        Data array (padded; only first n_data rows used).
    n_data : int scalar
        Number of valid data points.
    n_components : int
        Number of mixture components.
    n_iter : int
        Number of EM iterations (fixed for JIT).
    key : PRNGKey
        For initialization.
    reg : float
        Covariance regularization.

    Returns
    -------
    means : (K, ndim)
    covs : (K, ndim, ndim)
    weights : (K,)
    chols : (K, ndim, ndim)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n = int(n_data)
    ndim = data.shape[1]
    K = n_components
    X = data[:n]

    # Initialize means via k-means++
    means = _kmeans_pp_init(data, n, K, key)

    # Initialize covariances as identity
    covs = jnp.tile(jnp.eye(ndim, dtype=jnp.float64), (K, 1, 1))

    # Initialize weights as uniform
    weights = jnp.ones(K, dtype=jnp.float64) / K

    def _em_step(_, state):
        means, covs, weights = state

        # E-step: compute log responsibilities
        def _log_component(k):
            diff = X - means[k]
            prec = jnp.linalg.inv(covs[k])
            sign, log_det = jnp.linalg.slogdet(covs[k])
            mahal = jnp.sum((diff @ prec) * diff, axis=1)
            return -0.5 * (mahal + log_det + ndim * jnp.log(2 * jnp.pi)) + jnp.log(weights[k])

        log_resp = jax.vmap(_log_component)(jnp.arange(K))
        log_resp_T = log_resp.T
        log_norm = jax.scipy.special.logsumexp(log_resp_T, axis=1, keepdims=True)
        resp = jnp.exp(log_resp_T - log_norm)

        # M-step
        N_k = jnp.sum(resp, axis=0)
        N_k_safe = jnp.maximum(N_k, 1e-10)

        new_means = (resp.T @ X) / N_k_safe[:, None]

        def _update_cov(k):
            diff = X - new_means[k]
            r_k = resp[:, k]  # (n,) — integer indexing, fine under vmap
            cov_k = (diff * r_k[:, None]).T @ diff / N_k_safe[k]
            return cov_k + reg * jnp.eye(ndim, dtype=jnp.float64)

        new_covs = jax.vmap(_update_cov)(jnp.arange(K))
        new_weights = N_k_safe / jnp.float64(n)

        return new_means, new_covs, new_weights

    means, covs, weights = lax.fori_loop(0, n_iter, _em_step, (means, covs, weights))

    # Precompute Cholesky factors for sampling
    chols = jax.vmap(jnp.linalg.cholesky)(covs)

    return means, covs, weights, chols
