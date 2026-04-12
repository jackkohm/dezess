"""Full-covariance adaptive MH proposal from Z-matrix empirical covariance.

After warmup, uses the empirical covariance Cholesky from the Z-matrix
to construct an optimal Gaussian proposal in all d dimensions simultaneously.

Theory (Roberts-Gelman-Gilks 1997 optimal scaling):
    x_prop = x + (scale_factor / sqrt(d)) * cov_chol @ N(0, I)
    Optimal scale_factor = 2.38 for multivariate normal targets.
    Theoretical acceptance rate ≈ 0.234 at optimum.

During warmup, cov_chol = I, giving isotropic Gaussian proposals.
During production, cov_chol comes from the frozen Z-matrix posterior samples.

This is fundamentally different from 1D random-walk MH (mh_multi):
- mh_multi: proposes along a 1D direction, mixing requires O(d) steps
- mh_adaptive: proposes in all d dimensions simultaneously, mixing ~ O(1) steps
"""

import jax
import jax.numpy as jnp

from dezess.core.slice_sample import safe_log_prob


def execute(log_prob_fn, x, d, lp_x, mu, key, n_expand=1, n_shrink=1,
            cov_chol=None, scale_factor=2.38, **kwargs):
    """Full-covariance Gaussian MH proposal.

    Args:
        log_prob_fn: Log-probability function.
        x: Current position (n_dim,).
        d: Direction vector (n_dim,) — used only for bracket L/R bookkeeping.
        lp_x: Current log-probability.
        mu: Current step size (used for dummy bracket L/R return).
        key: JAX random key.
        n_expand: Ignored (kept for API compatibility).
        n_shrink: Ignored (kept for API compatibility).
        cov_chol: Cholesky factor of empirical covariance (n_dim, n_dim).
                  During warmup: identity matrix → isotropic proposal.
                  During production: Z-matrix empirical covariance Cholesky.
        scale_factor: Scales the proposal (default 2.38 = RGG optimal).
                      Tunable just like scale_factor in scale_aware width.

    Returns:
        x_new, lp_new, key, accepted, L, R
        L/R are dummy bracket bounds (±mu/2) so bracket_ratio=1 → no mu tuning.
    """
    ndim = x.shape[0]
    key, k_prop, k_u = jax.random.split(key, 3)

    # Propose: x + (scale / sqrt(d)) * cov_chol @ N(0, I)
    z = jax.random.normal(k_prop, shape=(ndim,), dtype=jnp.float64)
    scale = jnp.float64(scale_factor) / jnp.sqrt(jnp.float64(ndim))
    step = scale * (cov_chol @ z)

    x_prop = x + step
    lp_prop = safe_log_prob(log_prob_fn, x_prop)

    log_u = jnp.log(jax.random.uniform(k_u, dtype=jnp.float64) + 1e-30)
    accept = log_u < (lp_prop - lp_x)

    x_new = jnp.where(accept, x_prop, x)
    lp_new = jnp.where(accept, lp_prop, lp_x)

    # Dummy bracket bounds: L=-mu/2, R=mu/2 → bracket_ratio=1 = TARGET_RATIO
    # This tells the mu-tuner "no adjustment needed", keeping mu stable.
    return x_new, lp_new, key, accept, -mu * jnp.float64(0.5), mu * jnp.float64(0.5)


def compute_cov_chol(z_padded, z_count, reg=1e-6):
    """Compute Cholesky of empirical covariance from Z-matrix samples.

    Args:
        z_padded: (z_max_size, n_dim) padded Z-matrix.
        z_count: Number of valid entries.
        reg: Diagonal regularization for numerical stability.

    Returns:
        cov_chol: (n_dim, n_dim) lower-triangular Cholesky factor.
    """
    import numpy as np
    n_z = int(z_count)
    z_samples = np.array(z_padded[:n_z])
    n_dim = z_samples.shape[1]

    mean = z_samples.mean(axis=0)
    z_c = z_samples - mean

    # Numerically stable covariance: divide by (n-1) for unbiased estimator
    cov = (z_c.T @ z_c) / max(n_z - 1, 1)
    cov += reg * np.eye(n_dim)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Fallback: increase regularization
        cov += (1e-3 * np.trace(cov) / n_dim) * np.eye(n_dim)
        L = np.linalg.cholesky(cov)

    return jnp.array(L, dtype=jnp.float64)
