"""Block-Gibbs ensemble: cycle slice updates over parameter blocks.

Instead of updating all dimensions with a single direction vector,
sweep over blocks -- each block gets its own direction from its
sub-Z-matrix and its own step size mu. Walkers are updated in parallel
within each block via vmap.

The full log_prob is always evaluated (not a conditional), but only
the block's coordinates move. This is correct Gibbs: the full log_prob
restricted to one block IS the conditional distribution for that block.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def parse_blocks(ensemble_kwargs: dict, ndim: int) -> list[jnp.ndarray]:
    """Parse block specification from ensemble_kwargs.

    Accepts either:
        "block_sizes": [7, 14, 14, 14, 14]  (contiguous)
        "blocks": [[0,...,6], [7,...,20], ...]  (explicit)
    """
    if "blocks" in ensemble_kwargs:
        return [jnp.array(b, dtype=jnp.int32) for b in ensemble_kwargs["blocks"]]
    elif "block_sizes" in ensemble_kwargs:
        sizes = ensemble_kwargs["block_sizes"]
        blocks = []
        offset = 0
        for s in sizes:
            blocks.append(jnp.arange(offset, offset + s, dtype=jnp.int32))
            offset += s
        assert offset == ndim, f"block_sizes sum {offset} != ndim {ndim}"
        return blocks
    else:
        raise ValueError(
            "block_gibbs ensemble requires 'block_sizes' or 'blocks' "
            "in ensemble_kwargs"
        )


def init_mu_blocks(n_blocks: int, mu_init: float = 1.0) -> jnp.ndarray:
    """Initialize per-block step sizes."""
    return jnp.full(n_blocks, mu_init, dtype=jnp.float64)


def init_temperatures(n_walkers: int) -> Array:
    """All walkers at temperature 1.0 (no tempering for block-Gibbs)."""
    return jnp.ones(n_walkers, dtype=jnp.float64)


def propose_swaps(positions, log_probs, temperatures, key):
    """No swaps in block-Gibbs. Returns inputs unchanged."""
    return positions, log_probs
