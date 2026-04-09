"""Bijector transforms for reparameterizing target distributions.

A Transform maps between an unconstrained space z (where the sampler works)
and the original space x (where log_prob is defined). The sampler internally
works in z-space with the transformed log-prob:

    lp_transformed(z) = log_prob(forward(z)) + log_det_jac(z)

and maps samples back to x-space via forward() before returning them.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


class Transform(NamedTuple):
    """Bijector transform between unconstrained (z) and original (x) space.

    forward:     z -> x
    inverse:     x -> z
    log_det_jac: z -> scalar log|det(df/dz)|
    """
    forward: Callable
    inverse: Callable
    log_det_jac: Callable


def identity() -> Transform:
    """Identity transform (no-op)."""
    return Transform(
        forward=lambda z: z,
        inverse=lambda x: x,
        log_det_jac=lambda z: jnp.float64(0.0),
    )


def non_centered_funnel(
    width_idx: int,
    offset_indices: Sequence[int],
) -> Transform:
    """Non-centered parameterization for a Neal's funnel block.

    In the original space x:
        x[width_idx] = log_width ~ N(0, 9)
        x[offset_i]  ~ N(0, exp(log_width))  (funnel geometry)

    In the unconstrained space z:
        z[width_idx] = log_width  (unchanged)
        z[offset_i]  ~ N(0, 1)   (standard Gaussian — no funnel)

    Forward (z -> x):  x[offset_i] = z[offset_i] * exp(z[width_idx] / 2)
    Inverse (x -> z):  z[offset_i] = x[offset_i] / exp(x[width_idx] / 2)
    Log-det-Jacobian:  n_offsets * z[width_idx] / 2
    """
    offset_idx = jnp.array(offset_indices, dtype=jnp.int32)
    n_offsets = len(offset_indices)

    def forward(z):
        scale = jnp.exp(z[width_idx] / 2.0)
        x = z.at[offset_idx].set(z[offset_idx] * scale)
        return x

    def inverse(x):
        scale = jnp.exp(x[width_idx] / 2.0)
        z = x.at[offset_idx].set(x[offset_idx] / scale)
        return z

    def log_det_jac(z):
        return jnp.float64(n_offsets) * z[width_idx] / 2.0

    return Transform(forward=forward, inverse=inverse, log_det_jac=log_det_jac)
