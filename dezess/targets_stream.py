"""Test targets matching the GD1/stream fitting posterior structure.

These targets replicate the key pathologies of the 63D stream fitting
problem: block-coupled structure, funnel geometry, banana ridges,
and possible multimodality.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def block_coupled_gaussian(
    n_potential: int = 7,
    n_streams: int = 4,
    n_nuisance_per_stream: int = 14,
    coupling_strength: float = 0.5,
    condition_number: float = 50.0,
    seed: int = 42,
):
    """Block-coupled Gaussian matching the stream fitting structure.

    pot    GD1    AAU    Jet    Pal5
    [dense  corr   corr   corr   corr ]
    [corr   dense   0      0      0   ]
    [corr    0    dense    0      0   ]
    [corr    0      0    dense    0   ]
    [corr    0      0      0    dense ]

    The nuisance blocks are conditionally independent given potential params.
    """
    from dezess.targets import Target

    ndim = n_potential + n_streams * n_nuisance_per_stream
    rng = np.random.RandomState(seed)

    # Build the block covariance matrix
    cov = np.zeros((ndim, ndim))

    # Potential block (dense, ill-conditioned)
    A_pot = rng.randn(n_potential, n_potential)
    cov_pot = A_pot @ A_pot.T / n_potential
    evals = np.linspace(1.0, condition_number, n_potential)
    evecs = np.linalg.qr(rng.randn(n_potential, n_potential))[0]
    cov_pot = evecs @ np.diag(evals) @ evecs.T
    cov[:n_potential, :n_potential] = cov_pot

    # Nuisance blocks (each dense, independent of each other)
    for s in range(n_streams):
        start = n_potential + s * n_nuisance_per_stream
        end = start + n_nuisance_per_stream
        A_nuis = rng.randn(n_nuisance_per_stream, n_nuisance_per_stream)
        cov_nuis = A_nuis @ A_nuis.T / n_nuisance_per_stream + np.eye(n_nuisance_per_stream)
        cov[start:end, start:end] = cov_nuis

        # Coupling between potential and this nuisance block
        coupling = rng.randn(n_potential, n_nuisance_per_stream) * coupling_strength
        cov[:n_potential, start:end] = coupling
        cov[start:end, :n_potential] = coupling.T

    # Ensure positive definite
    cov = (cov + cov.T) / 2
    min_eig = np.linalg.eigvalsh(cov).min()
    if min_eig < 0.01:
        cov += (0.01 - min_eig + 0.1) * np.eye(ndim)

    prec = np.linalg.inv(cov)
    L = np.linalg.cholesky(cov)
    mu = np.zeros(ndim)

    prec_jax = jnp.array(prec, dtype=jnp.float64)
    L_jax = jnp.array(L, dtype=jnp.float64)
    mu_jax = jnp.array(mu, dtype=jnp.float64)

    def log_prob(x):
        d = x - mu_jax
        return -0.5 * d @ prec_jax @ d

    def sample(key, n):
        z = jax.random.normal(key, (n, ndim), dtype=jnp.float64)
        return z @ L_jax.T + mu_jax

    return Target(
        log_prob=log_prob, sample=sample, ndim=ndim,
        name=f"block_coupled_{ndim}d",
        mean=jnp.array(mu), cov=jnp.array(cov),
    )


def funnel_63d(n_funnels: int = 4, n_per_funnel: int = 14, n_potential: int = 7):
    """63D target with multiple Neal's funnels.

    Each stream has a width parameter that controls the scale of its
    nuisance parameters, creating funnel geometry. The potential
    parameters are standard Gaussian.
    """
    from dezess.targets import Target

    ndim = n_potential + n_funnels * n_per_funnel
    assert ndim == 63, f"Expected 63D, got {ndim}D"

    def log_prob(x):
        # Potential params: standard Gaussian
        lp = -0.5 * jnp.sum(x[:n_potential] ** 2)

        # Each funnel: x[0] of each block is log-width,
        # remaining params are N(0, exp(log_width))
        for f in range(n_funnels):
            start = n_potential + f * n_per_funnel
            log_width = x[start]  # log-width parameter
            offsets = x[start + 1:start + n_per_funnel]

            # log_width ~ N(0, 9) (like Neal's funnel)
            lp = lp - 0.5 * log_width ** 2 / 9.0

            # offsets | log_width ~ N(0, exp(log_width))
            var = jnp.exp(log_width)
            lp = lp - 0.5 * jnp.sum(offsets ** 2) / var
            lp = lp - 0.5 * (n_per_funnel - 1) * log_width  # log-det

        return lp

    def sample(key, n):
        keys = jax.random.split(key, 2 + n_funnels)
        # Potential params
        pot = jax.random.normal(keys[0], (n, n_potential), dtype=jnp.float64)
        parts = [pot]
        for f in range(n_funnels):
            k = keys[2 + f]
            k1, k2 = jax.random.split(k)
            log_w = jax.random.normal(k1, (n,), dtype=jnp.float64) * 3.0
            offsets = jax.random.normal(k2, (n, n_per_funnel - 1), dtype=jnp.float64)
            offsets = offsets * jnp.exp(log_w / 2.0)[:, None]
            parts.append(log_w[:, None])
            parts.append(offsets)
        return jnp.concatenate(parts, axis=1)

    return Target(
        log_prob=log_prob, sample=sample, ndim=ndim,
        name="funnel_63d", mean=jnp.zeros(ndim), cov=None,
    )


def banana_63d(n_potential: int = 7, n_nuisance: int = 56, b: float = 10.0):
    """63D target with banana-shaped potential params + Gaussian nuisance.

    First 7 params have Rosenbrock-like banana ridges.
    Remaining 56 are standard Gaussian (nuisance).
    """
    from dezess.targets import Target

    ndim = n_potential + n_nuisance

    def log_prob(x):
        pot = x[:n_potential]
        nuis = x[n_potential:]

        # Banana in potential params
        lp_pot = -jnp.sum(
            b * (pot[1:] - pot[:-1] ** 2) ** 2 + (1.0 - pot[:-1]) ** 2
        )

        # Standard Gaussian nuisance
        lp_nuis = -0.5 * jnp.sum(nuis ** 2)

        return lp_pot + lp_nuis

    def sample(key, n):
        k1, k2 = jax.random.split(key)
        pot = jax.random.normal(k1, (n, n_potential), dtype=jnp.float64) * 0.5 + 1.0
        nuis = jax.random.normal(k2, (n, n_nuisance), dtype=jnp.float64)
        return jnp.concatenate([pot, nuis], axis=1)

    return Target(
        log_prob=log_prob, sample=sample, ndim=ndim,
        name="banana_63d", mean=None, cov=None,
    )
