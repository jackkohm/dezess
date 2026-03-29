"""Analytical target distributions for testing.

Each target is a NamedTuple with:
  - log_prob(x): JAX-jittable (n_dim,) -> scalar
  - sample(key, n): draw n exact i.i.d. samples -> (n, n_dim)
  - ndim: int
  - name: str
  - mean: (n_dim,) true mean (where defined)
  - cov: (n_dim, n_dim) true covariance (where defined)
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


class Target(NamedTuple):
    log_prob: Callable
    sample: Callable
    ndim: int
    name: str
    mean: Optional[jnp.ndarray] = None
    cov: Optional[jnp.ndarray] = None


def isotropic_gaussian(ndim: int = 10) -> Target:
    """Standard isotropic Gaussian N(0, I)."""
    def log_prob(x):
        return -0.5 * jnp.sum(x ** 2)

    def sample(key, n):
        return jax.random.normal(key, (n, ndim), dtype=jnp.float64)

    return Target(
        log_prob=log_prob,
        sample=sample,
        ndim=ndim,
        name=f"isotropic_gaussian_d{ndim}",
        mean=jnp.zeros(ndim),
        cov=jnp.eye(ndim),
    )


def correlated_gaussian(ndim: int = 10, seed: int = 42) -> Target:
    """Correlated Gaussian with condition number ~50."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((ndim, ndim))
    cov = A @ A.T / ndim + np.eye(ndim)
    # Inflate condition number
    evals, evecs = np.linalg.eigh(cov)
    evals = np.linspace(1.0, 50.0, ndim)
    cov = evecs @ np.diag(evals) @ evecs.T
    cov = (cov + cov.T) / 2  # ensure symmetry

    mu = rng.standard_normal(ndim)
    prec = np.linalg.inv(cov)
    L = np.linalg.cholesky(cov)

    mu_jnp = jnp.array(mu, dtype=jnp.float64)
    prec_jnp = jnp.array(prec, dtype=jnp.float64)
    L_jnp = jnp.array(L, dtype=jnp.float64)

    def log_prob(x):
        d = x - mu_jnp
        return -0.5 * d @ prec_jnp @ d

    def sample(key, n):
        z = jax.random.normal(key, (n, ndim), dtype=jnp.float64)
        return z @ L_jnp.T + mu_jnp

    return Target(
        log_prob=log_prob,
        sample=sample,
        ndim=ndim,
        name=f"correlated_gaussian_d{ndim}",
        mean=jnp.array(mu),
        cov=jnp.array(cov),
    )


def student_t(ndim: int = 10, df: float = 5.0, seed: int = 42) -> Target:
    """Multivariate Student-t: heavier tails than Gaussian.

    x = mu + L @ z / sqrt(v / df), where z ~ N(0,I), v ~ chi2(df).
    """
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(ndim) * 0.5
    A = rng.standard_normal((ndim, ndim))
    scale = A @ A.T / ndim + np.eye(ndim)
    L = np.linalg.cholesky(scale)
    prec = np.linalg.inv(scale)

    mu_jnp = jnp.array(mu, dtype=jnp.float64)
    prec_jnp = jnp.array(prec, dtype=jnp.float64)
    L_jnp = jnp.array(L, dtype=jnp.float64)
    log_det = jnp.linalg.slogdet(jnp.array(scale))[1]

    def log_prob(x):
        d = x - mu_jnp
        maha = d @ prec_jnp @ d
        return -(df + ndim) / 2.0 * jnp.log(1.0 + maha / df)

    def sample(key, n):
        k1, k2 = jax.random.split(key)
        z = jax.random.normal(k1, (n, ndim), dtype=jnp.float64)
        # chi2(df) = gamma(df/2, 2)
        v = jax.random.gamma(k2, df / 2.0, (n,), dtype=jnp.float64) * 2.0
        scale_factor = jnp.sqrt(df / v)[:, None]
        return z @ L_jnp.T * scale_factor + mu_jnp

    # Mean = mu (for df > 1), Cov = scale * df/(df-2) (for df > 2)
    cov_true = jnp.array(scale) * df / (df - 2.0) if df > 2 else None

    return Target(
        log_prob=log_prob,
        sample=sample,
        ndim=ndim,
        name=f"student_t_d{ndim}_df{df:.0f}",
        mean=jnp.array(mu),
        cov=cov_true,
    )


def gaussian_mixture(ndim: int = 5, separation: float = 6.0, seed: int = 42) -> Target:
    """Mixture of 2 equal-weight Gaussians, separated along first axis."""
    rng = np.random.default_rng(seed)
    mu1 = np.zeros(ndim)
    mu1[0] = -separation / 2
    mu2 = np.zeros(ndim)
    mu2[0] = separation / 2

    # Slightly different covariances
    A1 = rng.standard_normal((ndim, ndim))
    cov1 = A1 @ A1.T / ndim + np.eye(ndim)
    A2 = rng.standard_normal((ndim, ndim))
    cov2 = A2 @ A2.T / ndim + np.eye(ndim)

    prec1 = np.linalg.inv(cov1)
    prec2 = np.linalg.inv(cov2)
    L1 = np.linalg.cholesky(cov1)
    L2 = np.linalg.cholesky(cov2)
    log_det1 = np.linalg.slogdet(cov1)[1]
    log_det2 = np.linalg.slogdet(cov2)[1]

    mu1_j = jnp.array(mu1, dtype=jnp.float64)
    mu2_j = jnp.array(mu2, dtype=jnp.float64)
    prec1_j = jnp.array(prec1, dtype=jnp.float64)
    prec2_j = jnp.array(prec2, dtype=jnp.float64)
    L1_j = jnp.array(L1, dtype=jnp.float64)
    L2_j = jnp.array(L2, dtype=jnp.float64)
    ld1 = jnp.float64(log_det1)
    ld2 = jnp.float64(log_det2)

    def _log_component(x, mu, prec, log_det):
        d = x - mu
        return -0.5 * (d @ prec @ d + log_det + ndim * jnp.log(2 * jnp.pi))

    def log_prob(x):
        lp1 = _log_component(x, mu1_j, prec1_j, ld1)
        lp2 = _log_component(x, mu2_j, prec2_j, ld2)
        return jnp.logaddexp(lp1, lp2) - jnp.log(2.0)

    def sample(key, n):
        k1, k2, k3 = jax.random.split(key, 3)
        # Choose component
        which = jax.random.bernoulli(k1, 0.5, (n,))
        z = jax.random.normal(k2, (n, ndim), dtype=jnp.float64)
        s1 = z @ L1_j.T + mu1_j
        s2 = z @ L2_j.T + mu2_j
        # Note: using same z for both is fine since we select one per sample
        # But for exact sampling, draw separate noise
        z2 = jax.random.normal(k3, (n, ndim), dtype=jnp.float64)
        s2 = z2 @ L2_j.T + mu2_j
        return jnp.where(which[:, None], s2, s1)

    # Mixture mean and covariance
    mix_mean = jnp.array((mu1 + mu2) / 2)
    mix_cov = jnp.array(
        0.5 * (cov1 + cov2)
        + 0.25 * np.outer(mu1 - mu2, mu1 - mu2)
    )

    return Target(
        log_prob=log_prob,
        sample=sample,
        ndim=ndim,
        name=f"gaussian_mixture_d{ndim}",
        mean=mix_mean,
        cov=mix_cov,
    )


def neals_funnel(ndim: int = 10) -> Target:
    """Neal's funnel: x_0 ~ N(0, 9), x_i|x_0 ~ N(0, exp(x_0)) for i=1..d-1.

    Challenging for fixed-mu samplers due to the scale varying by orders
    of magnitude depending on x_0.
    """
    def log_prob(x):
        # x[0] ~ N(0, 9)
        lp = -0.5 * x[0] ** 2 / 9.0 - 0.5 * jnp.log(9.0 * 2 * jnp.pi)
        # x[1:] | x[0] ~ N(0, exp(x[0]))
        var = jnp.exp(x[0])
        lp = lp + jnp.sum(
            -0.5 * x[1:] ** 2 / var
            - 0.5 * (x[0] + jnp.log(2 * jnp.pi))
        )
        return lp

    def sample(key, n):
        k1, k2 = jax.random.split(key)
        x0 = jax.random.normal(k1, (n,), dtype=jnp.float64) * 3.0  # std=3
        z = jax.random.normal(k2, (n, ndim - 1), dtype=jnp.float64)
        xi = z * jnp.exp(x0 / 2.0)[:, None]
        return jnp.concatenate([x0[:, None], xi], axis=1)

    return Target(
        log_prob=log_prob,
        sample=sample,
        ndim=ndim,
        name=f"neals_funnel_d{ndim}",
        mean=jnp.zeros(ndim),
        cov=None,  # non-trivial: Var(x_i) = E[exp(x_0)] = exp(9/2)
    )


# Registry for easy iteration in tests
ALL_TARGETS = {
    "isotropic_10": lambda: isotropic_gaussian(10),
    "isotropic_30": lambda: isotropic_gaussian(30),
    "correlated_10": lambda: correlated_gaussian(10),
    "student_t_10": lambda: student_t(10),
    "mixture_5": lambda: gaussian_mixture(5),
    "funnel_10": lambda: neals_funnel(10),
}
