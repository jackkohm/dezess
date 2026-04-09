"""Live Z-matrix: slowly update archive during production.

Unlike the frozen circular buffer, the live Z-matrix replaces a small
fraction of entries each production step. This provides adaptivity
(the archive tracks the evolving ensemble) without requiring
complementary ensemble splitting.

The update rate controls the trade-off:
- rate=0: fully frozen (standard dezess, clean invariance proof)
- rate=0.01: replace 1% per step (slowly adaptive)
- rate=1.0: replace all entries every step (fully live, like complementary split)

For rate > 0, the kernel is no longer exactly invariant (it's adaptive),
but with a small rate the bias is negligible and the improved archive
quality can significantly improve mixing in high dimensions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def append(
    z_padded: Array,
    z_count: Array,
    z_log_probs: Array,
    new_positions: Array,
    new_log_probs: Array,
    z_max_size: int,
    update_rate: float = 0.01,
    key: Array = None,
) -> tuple[Array, Array, Array]:
    """Update Z-matrix by replacing a fraction of entries.

    Randomly selects entries to replace (proportional to update_rate).

    Parameters
    ----------
    update_rate : float
        Fraction of Z-matrix to replace per call. Default 0.01 (1%).
    key : JAX PRNG key
        Required for random replacement selection.
    """
    n_walkers = new_positions.shape[0]

    if key is None:
        # Deterministic: replace oldest entries (circular buffer behavior)
        start_idx = z_count % z_max_size
        n_replace = jnp.minimum(n_walkers, jnp.int32(z_max_size * update_rate) + 1)
        indices = (jnp.arange(n_replace) + start_idx) % z_max_size
        z_padded_new = z_padded.at[indices].set(new_positions[:n_replace])
        z_log_probs_new = z_log_probs.at[indices].set(new_log_probs[:n_replace])
        z_count_new = jnp.maximum(z_count, n_replace)
    else:
        # Random replacement: select random entries to overwrite
        n_replace = jnp.int32(jnp.maximum(z_count * update_rate, 1.0))
        n_replace = jnp.minimum(n_replace, n_walkers)
        replace_indices = jax.random.randint(key, (n_walkers,), 0, z_count)
        # Only replace first n_replace walkers
        mask = jnp.arange(n_walkers) < n_replace
        z_padded_new = z_padded.at[replace_indices].set(
            jnp.where(mask[:, None], new_positions, z_padded[replace_indices])
        )
        z_log_probs_new = z_log_probs.at[replace_indices].set(
            jnp.where(mask, new_log_probs, z_log_probs[replace_indices])
        )
        z_count_new = z_count

    return z_padded_new, z_count_new, z_log_probs_new
