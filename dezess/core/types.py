"""Shared types for the variant sampler framework.

Strategy functions are plain callables with documented signatures.
State is carried in NamedTuples with fixed-shape JAX arrays so
everything flows through jit/vmap without recompilation.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import jax.numpy as jnp

Array = jnp.ndarray


class VariantConfig(NamedTuple):
    """Configuration selecting which strategies to compose."""
    name: str
    direction: str = "de_mcz"
    width: str = "scalar"
    slice_fn: str = "fixed"
    zmatrix: str = "circular"
    ensemble: str = "standard"
    # Strategy-specific kwargs (e.g. momentum alpha, stochastic sigma)
    direction_kwargs: dict = {}
    width_kwargs: dict = {}
    slice_kwargs: dict = {}
    zmatrix_kwargs: dict = {}
    ensemble_kwargs: dict = {}


class SliceState(NamedTuple):
    """Per-walker state returned by a slice execution strategy."""
    x_new: Array       # (n_dim,)
    lp_new: Array      # scalar
    key: Array          # PRNG key
    found: Array        # bool scalar
    L: Array            # scalar - final left bracket
    R: Array            # scalar - final right bracket


class WalkerAux(NamedTuple):
    """Per-walker auxiliary state carried between steps.

    Fields are optional (zeros when unused). Fixed shapes are set at init
    so JAX traces through without recompilation.
    """
    prev_direction: Array   # (n_dim,) for momentum directions
    bracket_widths: Array   # (n_dim,) EMA of bracket widths per dim (per-direction width)
    direction_anchor: Array  # (n_dim,) snooker anchor point (z_r1) for Jacobian correction


class SamplerState(NamedTuple):
    """Full sampler state passed through the step function."""
    positions: Array      # (n_walkers, n_dim)
    log_probs: Array      # (n_walkers,)
    z_padded: Array       # (z_max_size, n_dim)
    z_count: Array        # int32 scalar
    z_log_probs: Array    # (z_max_size,) log probs for weighted pair selection
    mu: Array             # scalar or (n_dim,) for per-direction
    key: Array            # PRNG key
    walker_aux: Any       # WalkerAux with (n_walkers, ...) arrays
    step: Array           # int32 scalar - current step index
