"""Gradient-assisted directions for when autodiff is available.

WARNING: The pure gradient direction d = normalize(grad(log_prob)(x)) is
state-dependent and radially biased toward the mode, causing the same
contraction problem as the uncorrected snooker direction. Even at 10%
mixing with DE-MCz, the sampler underestimates variance by 3x.

The random_ortho mode (directions orthogonal to the gradient) does not
have this bias but empirically does not improve ESS/eval over DE-MCz.

EXPERIMENTAL: Use only for targets where you have verified correctness
via moment tests. The gradient direction needs a Jacobian correction
analogous to snooker's, but the "anchor point" (the mode) is unknown,
making this correction impractical in general.
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
    grad_fn: object = None,
    grad_mix: float = 0.5,
    grad_mode: str = "gradient",
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample a direction, mixing gradient info with DE-MCz.

    Parameters
    ----------
    grad_fn : callable
        Gradient of log_prob: x -> grad(log_prob)(x). Must be JAX-compatible.
        If None, falls back to pure DE-MCz.
    grad_mix : float
        Probability of using gradient direction vs DE-MCz. Default 0.5.
    grad_mode : str
        "gradient" = slice along gradient direction.
        "random_ortho" = random direction orthogonal to gradient
        (explores ridges/contours).
    """
    key, k_choice, k_idx1, k_idx2, k_ortho = jax.random.split(key, 5)

    # DE-MCz fallback direction
    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
    diff = z_matrix[idx1] - z_matrix[idx2]
    de_norm = jnp.sqrt(jnp.sum(diff ** 2))
    d_de = diff / jnp.maximum(de_norm, 1e-30)

    # Store direction scale from DE-MCz
    aux = aux._replace(direction_scale=de_norm)

    if grad_fn is None:
        return d_de, key, aux

    # Gradient direction
    g = grad_fn(x_i)
    g_norm = jnp.sqrt(jnp.sum(g ** 2))
    g_unit = g / jnp.maximum(g_norm, 1e-30)
    g_valid = g_norm > 1e-30

    if grad_mode == "random_ortho":
        # Random direction orthogonal to gradient — explores ridges
        # Generate random vector, subtract gradient component, normalize
        v = jax.random.normal(k_ortho, shape=x_i.shape, dtype=jnp.float64)
        v = v - jnp.dot(v, g_unit) * g_unit
        v_norm = jnp.sqrt(jnp.sum(v ** 2))
        d_grad = v / jnp.maximum(v_norm, 1e-30)
        # If v is parallel to g (extremely rare), fall back to g
        d_grad = jnp.where(v_norm > 1e-30, d_grad, g_unit)
    else:
        # Slice along gradient direction
        d_grad = g_unit

    # Mix: use gradient direction with probability grad_mix
    use_grad = jax.random.uniform(k_choice) < grad_mix
    d = jnp.where(use_grad & g_valid, d_grad, d_de)

    return d, key, aux
