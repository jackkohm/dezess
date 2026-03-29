"""Adaptive iteration budget for slice sampling.

Tracks the cap-hit rate during warmup and adjusts N_EXPAND and N_SHRINK
for production. Easy targets get smaller budgets (faster), hard targets
get larger budgets (more robust).

The budgets are determined during warmup and frozen for production,
so the production kernel is re-JIT-compiled with the tuned values.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from dezess.core.slice_sample import slice_sample_fixed

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray

# Budget presets
BUDGETS = {
    "minimal": (2, 8),
    "standard": (3, 12),
    "generous": (4, 16),
    "heavy": (5, 20),
}


def select_budget(cap_hit_rate: float) -> tuple[int, int]:
    """Select N_EXPAND, N_SHRINK based on observed cap-hit rate.

    Parameters
    ----------
    cap_hit_rate : float
        Fraction of walker-steps that hit the shrinking cap (0 to 1).

    Returns
    -------
    (n_expand, n_shrink) : tuple[int, int]
    """
    if cap_hit_rate < 0.001:
        return BUDGETS["minimal"]
    elif cap_hit_rate < 0.01:
        return BUDGETS["standard"]
    elif cap_hit_rate < 0.05:
        return BUDGETS["generous"]
    else:
        return BUDGETS["heavy"]


def execute(
    log_prob_fn: Callable,
    x: Array,
    d: Array,
    lp_x: Array,
    mu: Array,
    key: Array,
    n_expand: int = 3,
    n_shrink: int = 12,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Slice sample with configurable budget. Same interface as fixed."""
    return slice_sample_fixed(
        log_prob_fn, x, d, lp_x, mu, key,
        n_expand=n_expand, n_shrink=n_shrink,
    )
