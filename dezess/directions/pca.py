"""PCA-informed directions from the Z-matrix.

Computes principal components of the Z-matrix and samples directions
along them with probability proportional to eigenvalue magnitude.
This helps in strongly correlated targets by aligning proposals
with the posterior's principal axes.

The SVD is computed once at the start of production (on the frozen
Z-matrix) and cached. During warmup, falls back to standard DE-MCz.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def compute_pca_components(z_matrix: Array, z_count: Array) -> tuple[Array, Array]:
    """Compute PCA components and weights from Z-matrix.

    Returns (components, weights) where:
      components: (n_dim, n_dim) - principal component directions (rows)
      weights: (n_dim,) - normalized eigenvalues as sampling weights
    """
    # Center the active portion
    active = z_matrix[:z_count]
    mean = jnp.mean(active, axis=0)
    centered = active - mean

    # SVD: U @ diag(S) @ Vt = centered
    # Vt rows are the principal directions
    _, S, Vt = jnp.linalg.svd(centered, full_matrices=False)

    # Use squared singular values as weights (proportional to variance)
    weights = S ** 2
    weights = weights / jnp.maximum(jnp.sum(weights), 1e-30)

    return Vt, weights


def sample_direction(
    x_i: Array,
    z_matrix: Array,
    z_count: Array,
    z_log_probs: Array,
    key: Array,
    aux: Array,
    pca_components: Array = None,
    pca_weights: Array = None,
    pca_mix: float = 0.5,
    pca_power: float = 1.0,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample direction: mix of PCA component and DE-MCz.

    With probability pca_mix, sample a principal component. Otherwise,
    use standard DE-MCz pair difference.

    Parameters
    ----------
    pca_components : (n_dim, n_dim) or None
        Principal components (rows). If None, uses pure DE-MCz.
    pca_weights : (n_dim,) or None
        Eigenvalue-proportional sampling weights.
    pca_mix : float
        Probability of using PCA direction vs DE-MCz.
    pca_power : float
        Controls eigenvalue weighting of PCA component selection.
        1.0 = proportional to eigenvalue (default, favours high-variance axes).
        0.0 = uniform across all components (equal mixing time per eigenvector).
        Intermediate values interpolate. For maximising ESS_min on
        ill-conditioned targets, pca_power=0.0 is optimal: each eigenvector
        is visited equally often and cov_aware calibrates the step size, so
        ALL directions mix at the same rate regardless of condition number.
    """
    key, k_choice, k_comp, k_sign, k_idx1, k_idx2 = jax.random.split(key, 6)

    # PCA direction
    n_dim = x_i.shape[0]
    # Always provide defaults at call site; this is a static check at trace time
    if pca_components is None:
        pca_components = jnp.eye(n_dim)
    if pca_weights is None:
        pca_weights = jnp.ones(n_dim) / n_dim

    # Gumbel-max trick for weighted sampling.
    # pca_power=1.0: eigenvalue-proportional (original behaviour).
    # pca_power=0.0: uniform (each eigenvector equally likely) — optimal for ESS_min.
    effective_weights = pca_weights ** pca_power
    effective_weights = effective_weights / jnp.maximum(jnp.sum(effective_weights), 1e-30)
    gumbel = -jnp.log(-jnp.log(jax.random.uniform(k_comp, (n_dim,), dtype=jnp.float64) + 1e-30) + 1e-30)
    log_weights = jnp.log(effective_weights + 1e-30)
    comp_idx = jnp.argmax(log_weights + gumbel)
    d_pca = pca_components[comp_idx]
    # Random sign for symmetry
    sign = 2.0 * jax.random.bernoulli(k_sign).astype(jnp.float64) - 1.0
    d_pca = d_pca * sign

    # DE-MCz fallback
    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
    diff = z_matrix[idx1] - z_matrix[idx2]
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d_demcz = diff / jnp.maximum(norm, 1e-30)

    # Mix
    use_pca = jax.random.bernoulli(k_choice, pca_mix)
    d = jnp.where(use_pca, d_pca, d_demcz)

    # Store DE-MCz pair distance for scale_aware width compatibility
    aux = aux._replace(direction_scale=norm)

    return d, key, aux
