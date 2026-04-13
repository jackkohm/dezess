"""Covariance-aware width: correct 1D slice calibration via inverse formula.

Two calibration modes selectable via ``use_inverse``:

FORWARD (use_inverse=False, legacy):
    sigma_d = ||L^T @ d_unit||  where  L L^T = Sigma_empirical
    This equals sqrt(d_unit^T Sigma d_unit) — the marginal std in direction d.
    Correct for pure eigenvector proposals (PCA), but OVERESTIMATES for mixed
    directions in ill-conditioned space (ratio can exceed 10x), causing high
    rejection rates.

INVERSE (use_inverse=True, default):
    sigma_1d = 1 / ||L^{-1} @ d_unit||  = 1 / sqrt(d_unit^T Sigma^{-1} d_unit)
    This is the true 1D slice width along d in the posterior geometry:
    the set of points x + t*d satisfying the Gaussian log-prob constraint.
    For ANY direction (pure or mixed), this gives the SAME MH acceptance rate:
    P(accept) ≈ 1 / scale_factor.
    Correct for all direction strategies including DE-MCz with ill-conditioned targets.

For pure eigenvector directions both formulas agree: sigma = sqrt(lambda_i).
For mixed directions in ill-conditioned space the inverse formula is much smaller,
ensuring well-calibrated proposals and high acceptance across all directions.

Proposal width:  mu_eff = scale_factor * sigma_1d

During warmup, cov_chol = I, so sigma_1d = 1.0 (inverse) or 1.0 (forward).
After warmup re-JIT, cov_chol is the empirical covariance Cholesky.

Optimal scale_factor:
  - Inverse formula: scale_factor = 2-4 (acceptance ~25-50% for uniform U)
  - Forward formula: scale_factor = 4-6 (compensates for overestimate)
"""

import jax.numpy as jnp


def get_mu(mu, d, aux, key=None, cov_chol=None, scale_factor=1.0,
           use_inverse=True, **kwargs):
    """Compute calibrated slice width along direction d.

    Parameters
    ----------
    mu : scalar
        Global scale (used as floor, irrelevant here).
    d : Array (n_dim,)
        Direction vector (unnormalized, from DE-MCz or any direction module).
    aux : WalkerAux
        Walker auxiliary state.
    key : ignored.
    cov_chol : Array (n_dim, n_dim)
        Lower-triangular Cholesky of empirical covariance L: Sigma = L L^T.
        During warmup: identity → sigma = 1.0.
    scale_factor : float
        Scales the calibrated width. Default 1.0.
    use_inverse : bool
        If True (default): use inverse formula sigma = 1/||L^{-1} d_unit||.
        If False (legacy): use forward formula sigma = ||L^T d_unit||.

    Returns
    -------
    mu_eff : scalar
    """
    d_norm = jnp.linalg.norm(d)
    d_unit = d / jnp.maximum(d_norm, 1e-30)

    if use_inverse:
        # Correct 1D slice width: sigma = 1 / sqrt(d^T Sigma^{-1} d)
        # = 1 / ||L^{-1} d_unit||  via triangular solve
        z = jnp.linalg.solve(cov_chol, d_unit)   # L^{-1} d_unit  (O(d^2))
        sigma_1d = 1.0 / jnp.maximum(jnp.linalg.norm(z), jnp.float64(1e-30))
    else:
        # Legacy forward: sqrt(d^T Sigma d) = ||L^T d_unit||
        sigma_1d = jnp.linalg.norm(cov_chol.T @ d_unit)

    mu_eff = scale_factor * sigma_1d
    return jnp.maximum(mu_eff, jnp.float64(1e-8))
