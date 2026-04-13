"""Random Gaussian direction: isotropic full-space exploration.

Samples a uniformly random direction from the n_dim-dimensional unit sphere
(equivalently: d = randn(n_dim) / ||randn(n_dim)||).

Unlike DE-MCz (biased toward high-variance eigendirections) or PCA
(explicit eigenvalue-proportional weighting), random Gaussian directions are
statistically isotropic. When combined with cov_aware inverse-formula width:

    sigma_1d = 1 / ||L^{-1} d|| = 1 / sqrt(d^T Sigma^{-1} d)

the acceptance rate is identical for ALL directions regardless of condition
number, and every eigenvector of the posterior is explored equally often.
This eliminates the condition-number bottleneck that limits DE-MCz on
correlated or moderately ill-conditioned targets.

Key advantages over DE-MCz:
  - Isotropic: no bias toward high-variance directions
  - Full-space: covers ALL n_dim dimensions including the null space of
    a rank-deficient Z-matrix (when n_walkers < n_dim)
  - With cov_aware (inverse): uniform acceptance rate across all directions

Key advantages over PCA:
  - No eigendecomposition needed (cheaper at trace time)
  - Uniform by construction (no eigenvalue-weighting bias)
  - Not restricted to pre-computed eigenvectors

Can be mixed with DE-MCz via `de_mix` parameter to retain Z-matrix adaptivity
for multimodal targets (DE-MCz directions naturally point between modes).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def sample_direction(
    x_i: Array,
    z_matrix: Array,
    z_count: Array,
    z_log_probs: Array,
    key: Array,
    aux: Array,
    de_mix: float = 0.0,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample a random Gaussian direction, optionally mixed with DE-MCz.

    Parameters
    ----------
    de_mix : float
        Probability of using DE-MCz direction instead of random Gaussian.
        Default 0.0 (pure random Gaussian).
        Set > 0 to retain DE-MCz's inter-mode bridging for multimodal targets.
    """
    key, k_rand, k_choice, k_idx1, k_idx2 = jax.random.split(key, 5)

    # Random Gaussian direction (uniformly distributed on unit sphere)
    d_raw = jax.random.normal(k_rand, x_i.shape, dtype=jnp.float64)
    norm_rand = jnp.linalg.norm(d_raw)
    d_rand = d_raw / jnp.maximum(norm_rand, 1e-30)

    # DE-MCz direction (from Z-matrix pair)
    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
    diff = z_matrix[idx1] - z_matrix[idx2]
    norm_de = jnp.linalg.norm(diff)
    d_de = diff / jnp.maximum(norm_de, 1e-30)

    # Mix: use DE-MCz with probability de_mix, random Gaussian otherwise
    use_de = jax.random.uniform(k_choice) < de_mix
    d = jnp.where(use_de, d_de, d_rand)

    # Store DE-MCz scale for scale_aware width fallback compatibility
    aux = aux._replace(direction_scale=norm_de)

    return d, key, aux
