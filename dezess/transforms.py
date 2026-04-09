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


def block_transform(
    transforms: Sequence[Transform],
    index_lists: Sequence[Sequence[int]],
    ndim: int,
) -> Transform:
    """Apply different transforms to different parameter blocks.

    Each transform operates on its own block of dimensions. The blocks
    must be non-overlapping and together cover all ndim dimensions.
    """
    idx_arrays = [jnp.array(idx, dtype=jnp.int32) for idx in index_lists]

    def forward(z):
        x = jnp.zeros(ndim, dtype=z.dtype)
        for t, idx in zip(transforms, idx_arrays):
            z_block = z[idx]
            x_block = t.forward(z_block)
            x = x.at[idx].set(x_block)
        return x

    def inverse(x):
        z = jnp.zeros(ndim, dtype=x.dtype)
        for t, idx in zip(transforms, idx_arrays):
            x_block = x[idx]
            z_block = t.inverse(x_block)
            z = z.at[idx].set(z_block)
        return z

    def log_det_jac(z):
        ldj = jnp.float64(0.0)
        for t, idx in zip(transforms, idx_arrays):
            z_block = z[idx]
            ldj = ldj + t.log_det_jac(z_block)
        return ldj

    return Transform(forward=forward, inverse=inverse, log_det_jac=log_det_jac)


def multi_funnel(
    n_potential: int,
    funnel_blocks: Sequence[tuple[int, int]],
) -> Transform:
    """Build a block transform for multiple funnel blocks + potential.

    Parameters
    ----------
    n_potential : int
        Number of potential parameters (identity-transformed).
    funnel_blocks : list of (start_idx, block_size)
        Each block: first dim is log-width, remaining are offsets.
    """
    transforms = [identity()]
    index_lists = [list(range(n_potential))]

    for start, size in funnel_blocks:
        t = non_centered_funnel(
            width_idx=0,
            offset_indices=list(range(1, size)),
        )
        transforms.append(t)
        index_lists.append(list(range(start, start + size)))

    ndim = n_potential + sum(size for _, size in funnel_blocks)
    return block_transform(transforms, index_lists, ndim)


def multi_width_funnel(
    width_offset_pairs: Sequence[tuple[int, Sequence[int]]],
    ndim: int,
) -> Transform:
    """Non-centered parameterization with multiple width params in one block.

    Each (width_idx, offset_indices) pair defines a sub-funnel:
    the width param controls the scale of its associated offsets.
    All other dimensions are identity-transformed.

    Parameters
    ----------
    width_offset_pairs : list of (width_idx, offset_indices)
        Each pair: width_idx is the log-scale param, offset_indices
        are the params whose scale it controls.
    ndim : int
        Total dimensions in this block.

    Example (Sanders stream block, 14 params):
        multi_width_funnel([
            (8, [3, 4]),   # log_u -> gamma_1, gamma_2
            (9, [6, 7]),   # log_w -> omega_1, omega_2
            (10, [2, 5]),  # log_w0 -> gamma_0, omega_0
        ], ndim=14)
    """
    # Precompute all width/offset arrays
    pairs = [(w, jnp.array(offs, dtype=jnp.int32), len(offs))
             for w, offs in width_offset_pairs]

    def forward(z):
        x = z
        for w_idx, o_idx, _ in pairs:
            scale = jnp.exp(z[w_idx] / 2.0)
            x = x.at[o_idx].set(z[o_idx] * scale)
        return x

    def inverse(x):
        z = x
        for w_idx, o_idx, _ in pairs:
            scale = jnp.exp(x[w_idx] / 2.0)
            z = z.at[o_idx].set(x[o_idx] / scale)
        return z

    def log_det_jac(z):
        ldj = jnp.float64(0.0)
        for w_idx, _, n_offs in pairs:
            ldj = ldj + jnp.float64(n_offs) * z[w_idx] / 2.0
        return ldj

    return Transform(forward=forward, inverse=inverse, log_det_jac=log_det_jac)


def sanders_funnel(
    n_potential: int = 7,
    n_streams: int = 4,
    n_nuisance: int = 14,
) -> Transform:
    """Non-centered transform matching the Sanders stream posterior.

    Per-stream nuisance layout (14 params):
        0: phi, 1: psi           — direction angles (unchanged)
        2: gamma_0               — along-n angle offset (scaled by log_w0)
        3: gamma_1, 4: gamma_2   — perp angle offsets (scaled by log_u)
        5: omega_0               — along-n freq offset (scaled by log_w0)
        6: omega_1, 7: omega_2   — perp freq offsets (scaled by log_w)
        8: log_u                 — perp angle width
        9: log_w                 — perp freq width
        10: log_w0               — along-n width
        11: log_Omega_s, 12: log_tmax, 13: logit_epsilon — unchanged

    Three sub-funnels per stream:
        log_u  (8)  -> gamma_1 (3), gamma_2 (4)
        log_w  (9)  -> omega_1 (6), omega_2 (7)
        log_w0 (10) -> gamma_0 (2), omega_0 (5)
    """
    ndim = n_potential + n_streams * n_nuisance

    stream_transform = multi_width_funnel(
        width_offset_pairs=[
            (8, [3, 4]),   # log_u -> gamma_1, gamma_2
            (9, [6, 7]),   # log_w -> omega_1, omega_2
            (10, [2, 5]),  # log_w0 -> gamma_0, omega_0
        ],
        ndim=n_nuisance,
    )

    transforms = [identity()]  # potential block
    index_lists = [list(range(n_potential))]

    for s in range(n_streams):
        start = n_potential + s * n_nuisance
        transforms.append(stream_transform)
        index_lists.append(list(range(start, start + n_nuisance)))

    return block_transform(transforms, index_lists, ndim)
