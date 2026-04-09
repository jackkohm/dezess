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


def sanders_funnel_63d(n_potential: int = 7, n_streams: int = 4, n_nuisance: int = 14):
    """63D target mimicking the Sanders stream posterior funnel structure.

    Matches the real Sanders layout:
        pot(7) + 4 x [phi, psi, gamma_0..2, omega_0..2, log_u, log_w, log_w0,
                       log_Omega_s, log_tmax, logit_epsilon]

    Three sub-funnels per stream:
        log_u  (idx 8)  -> gamma_1 (3), gamma_2 (4)
        log_w  (idx 9)  -> omega_1 (6), omega_2 (7)
        log_w0 (idx 10) -> gamma_0 (2), omega_0 (5)

    Non-funnel params (phi, psi, log_Omega_s, log_tmax, logit_epsilon) are
    standard Gaussian. Potential params have mild correlation (condition ~20).
    """
    from dezess.targets import Target

    ndim = n_potential + n_streams * n_nuisance
    assert ndim == 63, f"Expected 63D, got {ndim}D"

    # Correlated potential params
    rng = np.random.RandomState(42)
    evecs = np.linalg.qr(rng.randn(n_potential, n_potential))[0]
    evals = np.linspace(1.0, 20.0, n_potential)
    cov_pot = evecs @ np.diag(evals) @ evecs.T
    prec_pot = np.linalg.inv(cov_pot)
    L_pot = np.linalg.cholesky(cov_pot)
    prec_pot_jax = jnp.array(prec_pot, dtype=jnp.float64)
    L_pot_jax = jnp.array(L_pot, dtype=jnp.float64)

    def log_prob(x):
        # Potential: correlated Gaussian
        pot = x[:n_potential]
        lp = -0.5 * pot @ prec_pot_jax @ pot

        for s in range(n_streams):
            start = n_potential + s * n_nuisance
            nuis = x[start:start + n_nuisance]

            # Direction angles: phi(0), psi(1) ~ N(0, 1)
            lp = lp - 0.5 * (nuis[0] ** 2 + nuis[1] ** 2)

            # Sub-funnel 1: log_u(8) -> gamma_1(3), gamma_2(4)
            log_u = nuis[8]
            lp = lp - 0.5 * log_u ** 2 / 9.0  # log_u ~ N(0, 9)
            var_u = jnp.exp(log_u)
            lp = lp - 0.5 * (nuis[3] ** 2 + nuis[4] ** 2) / var_u
            lp = lp - log_u  # log-det for 2 offsets

            # Sub-funnel 2: log_w(9) -> omega_1(6), omega_2(7)
            log_w = nuis[9]
            lp = lp - 0.5 * log_w ** 2 / 9.0
            var_w = jnp.exp(log_w)
            lp = lp - 0.5 * (nuis[6] ** 2 + nuis[7] ** 2) / var_w
            lp = lp - log_w

            # Sub-funnel 3: log_w0(10) -> gamma_0(2), omega_0(5)
            log_w0 = nuis[10]
            lp = lp - 0.5 * log_w0 ** 2 / 9.0
            var_w0 = jnp.exp(log_w0)
            lp = lp - 0.5 * (nuis[2] ** 2 + nuis[5] ** 2) / var_w0
            lp = lp - log_w0

            # Remaining params: standard Gaussian
            # log_Omega_s(11), log_tmax(12), logit_epsilon(13)
            lp = lp - 0.5 * (nuis[11] ** 2 + nuis[12] ** 2 + nuis[13] ** 2)

        return lp

    def sample(key, n):
        keys = jax.random.split(key, 2 + n_streams * 4)
        # Potential
        pot = jax.random.normal(keys[0], (n, n_potential), dtype=jnp.float64) @ L_pot_jax.T
        parts = [pot]

        for s in range(n_streams):
            k_base = 2 + s * 4
            k_dir, k_widths, k_offsets, k_rest = keys[k_base:k_base + 4]

            block = jnp.zeros((n, n_nuisance), dtype=jnp.float64)
            # Direction angles
            dirs = jax.random.normal(k_dir, (n, 2), dtype=jnp.float64)
            block = block.at[:, 0].set(dirs[:, 0])
            block = block.at[:, 1].set(dirs[:, 1])

            # Width params ~ N(0, 9)
            widths = jax.random.normal(k_widths, (n, 3), dtype=jnp.float64) * 3.0
            log_u, log_w, log_w0 = widths[:, 0], widths[:, 1], widths[:, 2]
            block = block.at[:, 8].set(log_u)
            block = block.at[:, 9].set(log_w)
            block = block.at[:, 10].set(log_w0)

            # Offsets scaled by widths
            z_off = jax.random.normal(k_offsets, (n, 6), dtype=jnp.float64)
            block = block.at[:, 3].set(z_off[:, 0] * jnp.exp(log_u / 2.0))   # gamma_1
            block = block.at[:, 4].set(z_off[:, 1] * jnp.exp(log_u / 2.0))   # gamma_2
            block = block.at[:, 6].set(z_off[:, 2] * jnp.exp(log_w / 2.0))   # omega_1
            block = block.at[:, 7].set(z_off[:, 3] * jnp.exp(log_w / 2.0))   # omega_2
            block = block.at[:, 2].set(z_off[:, 4] * jnp.exp(log_w0 / 2.0))  # gamma_0
            block = block.at[:, 5].set(z_off[:, 5] * jnp.exp(log_w0 / 2.0))  # omega_0

            # Remaining standard Gaussian
            rest = jax.random.normal(k_rest, (n, 3), dtype=jnp.float64)
            block = block.at[:, 11].set(rest[:, 0])
            block = block.at[:, 12].set(rest[:, 1])
            block = block.at[:, 13].set(rest[:, 2])

            parts.append(block)

        return jnp.concatenate(parts, axis=1)

    return Target(
        log_prob=log_prob, sample=sample, ndim=ndim,
        name="sanders_funnel_63d", mean=jnp.zeros(ndim), cov=None,
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
