"""Baseline standard ensemble: all walkers update in parallel, no tempering."""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def init_temperatures(n_walkers: int) -> Array:
    """All walkers at temperature 1.0 (no tempering)."""
    return jnp.ones(n_walkers, dtype=jnp.float64)


def propose_swaps(
    positions: Array,
    log_probs: Array,
    temperatures: Array,
    key: Array,
) -> tuple[Array, Array]:
    """No swaps in standard ensemble. Returns inputs unchanged."""
    return positions, log_probs
