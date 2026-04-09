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
