"""Multi-GPU sharding utilities for dezess.

Provides Mesh setup and sharding specs for distributing walkers across
multiple GPUs while keeping the Z-matrix and tuning state replicated.

When n_gpus == 1, all helpers return None and the caller falls back to
the single-GPU code path (zero overhead).
"""

from __future__ import annotations

from typing import Optional

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def setup_sharding(
    n_gpus: int,
    n_walkers_total: int,
) -> Optional[dict]:
    """Build sharding specs for multi-GPU runs.

    Parameters
    ----------
    n_gpus : int
        Number of GPUs to use. If 1, returns None.
    n_walkers_total : int
        Total walker count across all GPUs. Must be divisible by n_gpus.

    Returns
    -------
    None if n_gpus == 1.
    Otherwise dict with:
        - "mesh": jax.sharding.Mesh
        - "walker_sharding": NamedSharding sharding the first axis
        - "replicated": NamedSharding replicating across all GPUs
    """
    if n_gpus == 1:
        return None

    if n_walkers_total % n_gpus != 0:
        raise ValueError(
            f"n_walkers_total ({n_walkers_total}) must be divisible by "
            f"n_gpus ({n_gpus})"
        )

    devices = jax.devices()[:n_gpus]
    if len(devices) < n_gpus:
        raise ValueError(
            f"Requested n_gpus={n_gpus} but only {len(devices)} devices available"
        )

    mesh = Mesh(devices, ('walkers',))
    walker_sharding = NamedSharding(mesh, P('walkers'))
    replicated = NamedSharding(mesh, P())

    return {
        "mesh": mesh,
        "walker_sharding": walker_sharding,
        "replicated": replicated,
    }
