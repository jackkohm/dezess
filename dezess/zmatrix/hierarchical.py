"""Hierarchical Z-matrix management.

Maintains three levels:
  Level 0 (recent):  Last ~500 entries — current posterior geometry
  Level 1 (thinned): Every 10th from older entries — global structure
  Level 2 (extreme): Entries with highest/lowest log-prob — tail coverage

Direction pairs are sampled with probability proportional to level weights,
giving both local precision and global exploration.

Implementation: a single padded array with sections for each level.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray

# Level sizes
LEVEL0_SIZE = 500    # recent
LEVEL1_SIZE = 450    # thinned (every 10th from older)
LEVEL2_SIZE = 50     # extreme (highest/lowest log-prob)
TOTAL_SIZE = LEVEL0_SIZE + LEVEL1_SIZE + LEVEL2_SIZE


def init_hierarchical(n_dim: int) -> tuple[Array, Array, Array]:
    """Initialize hierarchical Z-matrix storage.

    Returns (z_padded, z_counts_per_level, z_log_probs).
    z_counts_per_level: (3,) int32 array [count_level0, count_level1, count_level2]
    """
    z_padded = jnp.zeros((TOTAL_SIZE, n_dim), dtype=jnp.float64)
    z_counts = jnp.zeros(3, dtype=jnp.int32)
    z_log_probs = jnp.full(TOTAL_SIZE, -1e30, dtype=jnp.float64)
    return z_padded, z_counts, z_log_probs


def append(
    z_padded: Array,
    z_count: Array,
    z_log_probs: Array,
    new_positions: Array,
    new_log_probs: Array,
    z_max_size: int,
    step: int = 0,
) -> tuple[Array, Array, Array]:
    """Append new positions to hierarchical Z-matrix.

    Level 0 (recent): always receives new entries (circular)
    Level 1 (thinned): receives every 10th batch
    Level 2 (extreme): maintains top/bottom entries by log-prob
    """
    n_walkers = new_positions.shape[0]

    # For simplicity in the hierarchical case, we use the standard
    # circular buffer but organize it into sections.
    # Level 0: indices [0, LEVEL0_SIZE)
    l0_start = z_count % LEVEL0_SIZE
    l0_indices = (jnp.arange(n_walkers) + l0_start) % LEVEL0_SIZE
    z_padded = z_padded.at[l0_indices].set(new_positions)
    z_log_probs = z_log_probs.at[l0_indices].set(new_log_probs)

    # Level 1: indices [LEVEL0_SIZE, LEVEL0_SIZE + LEVEL1_SIZE)
    # Add every 10th batch
    is_thinned_step = (step % 10) == 0
    l1_offset = LEVEL0_SIZE
    l1_count = jnp.minimum(z_count // 10, LEVEL1_SIZE)
    l1_start = l1_count % LEVEL1_SIZE
    l1_indices = (jnp.arange(n_walkers) + l1_start) % LEVEL1_SIZE + l1_offset
    z_padded = jnp.where(
        is_thinned_step,
        z_padded.at[l1_indices].set(new_positions),
        z_padded,
    )
    z_log_probs = jnp.where(
        is_thinned_step,
        z_log_probs.at[l1_indices].set(new_log_probs),
        z_log_probs,
    )

    # Level 2: indices [LEVEL0_SIZE + LEVEL1_SIZE, TOTAL_SIZE)
    # Keep extreme entries (highest and lowest log-prob)
    l2_offset = LEVEL0_SIZE + LEVEL1_SIZE
    # Find current min/max log-prob in level 2
    l2_lps = z_log_probs[l2_offset:l2_offset + LEVEL2_SIZE]
    # Replace the entry closest to median (least extreme) with new extreme
    for_extremes = jnp.concatenate([new_log_probs])
    max_new = jnp.max(for_extremes)
    min_new = jnp.min(for_extremes)
    max_idx = jnp.argmax(new_log_probs)
    min_idx = jnp.argmin(new_log_probs)

    # Simple: put max in first half, min in second half of level 2
    half2 = LEVEL2_SIZE // 2
    l2_max_slot = l2_offset + (z_count % half2)
    l2_min_slot = l2_offset + half2 + (z_count % half2)

    z_padded = z_padded.at[l2_max_slot].set(new_positions[max_idx])
    z_log_probs = z_log_probs.at[l2_max_slot].set(max_new)
    z_padded = z_padded.at[l2_min_slot].set(new_positions[min_idx])
    z_log_probs = z_log_probs.at[l2_min_slot].set(min_new)

    z_count_new = jnp.minimum(z_count + n_walkers, jnp.int32(z_max_size))

    return z_padded, z_count_new, z_log_probs


def sample_indices(
    key: Array,
    z_count: Array,
    z_log_probs: Array,
    level_weights: tuple = (0.5, 0.3, 0.2),
) -> tuple[Array, Array]:
    """Sample two distinct indices from the hierarchical Z-matrix.

    Samples a level first (weighted), then an index within that level.
    """
    key, k_level, k_idx1, k_idx2 = jax.random.split(key, 4)

    weights = jnp.array(level_weights, dtype=jnp.float64)
    weights = weights / jnp.sum(weights)

    level = jax.random.choice(k_level, 3, p=weights)

    # Level offsets and sizes
    offsets = jnp.array([0, LEVEL0_SIZE, LEVEL0_SIZE + LEVEL1_SIZE], dtype=jnp.int32)
    sizes = jnp.array([LEVEL0_SIZE, LEVEL1_SIZE, LEVEL2_SIZE], dtype=jnp.int32)

    offset = offsets[level]
    size = sizes[level]
    active_size = jnp.minimum(size, z_count)
    active_size = jnp.maximum(active_size, 2)  # need at least 2

    idx1 = offset + jax.random.randint(k_idx1, (), 0, size) % active_size
    idx2 = offset + jax.random.randint(k_idx2, (), 0, size) % active_size
    idx2 = jnp.where(idx1 == idx2, offset + (idx2 - offset + 1) % active_size, idx2)

    return idx1, idx2
