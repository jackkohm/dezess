"""Parallel tempering (replica exchange) ensemble.

Runs walkers at K different temperatures in a geometric ladder.
After each step, proposes swaps between adjacent temperature pairs.
The cold chain (T=1) gives posterior samples.

Swap acceptance: min(1, exp((1/T_i - 1/T_j) * (lp_j - lp_i)))
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def init_temperatures(
    n_walkers: int,
    n_temps: int = 4,
    t_max: float = 10.0,
) -> Array:
    """Create geometric temperature ladder.

    Parameters
    ----------
    n_walkers : int
        Total number of walkers (must be divisible by n_temps).
    n_temps : int
        Number of temperature levels.
    t_max : float
        Maximum temperature.

    Returns
    -------
    temperatures : (n_walkers,)
    """
    walkers_per_temp = n_walkers // n_temps
    temps = jnp.geomspace(1.0, t_max, n_temps)
    temperatures = jnp.repeat(temps, walkers_per_temp)
    if len(temperatures) < n_walkers:
        temperatures = jnp.concatenate([
            temperatures,
            jnp.ones(n_walkers - len(temperatures))
        ])
    return temperatures


def propose_swaps(
    positions: Array,
    log_probs: Array,
    temperatures: Array,
    key: Array,
    n_temps: int = 4,
    swap_every: int = 1,
) -> tuple[Array, Array]:
    """Propose replica exchange swaps between adjacent temperature levels.

    Uses lax.fori_loop for JAX-compatible vectorized swaps.
    """
    n_walkers = positions.shape[0]
    walkers_per_temp = n_walkers // n_temps

    unique_temps = jnp.geomspace(temperatures[0], temperatures[-1], n_temps)

    # Pre-split all keys
    keys = jax.random.split(key, 2 * (n_temps - 1))

    def swap_level(i, state):
        positions, log_probs = state

        k_pair = keys[2 * i]
        k_accept = keys[2 * i + 1]

        idx_lo = jnp.arange(walkers_per_temp) + i * walkers_per_temp
        idx_hi = jnp.arange(walkers_per_temp) + (i + 1) * walkers_per_temp

        perm = jax.random.permutation(k_pair, walkers_per_temp)
        idx_hi_shuffled = idx_hi[perm]

        T_lo = unique_temps[i]
        T_hi = unique_temps[i + 1]

        beta_diff = 1.0 / T_lo - 1.0 / T_hi
        lp_lo = log_probs[idx_lo]
        lp_hi = log_probs[idx_hi_shuffled]
        log_alpha = beta_diff * (lp_hi - lp_lo)

        u = jax.random.uniform(k_accept, (walkers_per_temp,), dtype=jnp.float64)
        accept = jnp.log(u) < log_alpha

        pos_lo = positions[idx_lo]
        pos_hi = positions[idx_hi_shuffled]
        new_pos_lo = jnp.where(accept[:, None], pos_hi, pos_lo)
        new_pos_hi = jnp.where(accept[:, None], pos_lo, pos_hi)

        new_lp_lo = jnp.where(accept, lp_hi, lp_lo)
        new_lp_hi = jnp.where(accept, lp_lo, lp_hi)

        positions = positions.at[idx_lo].set(new_pos_lo)
        positions = positions.at[idx_hi_shuffled].set(new_pos_hi)
        log_probs = log_probs.at[idx_lo].set(new_lp_lo)
        log_probs = log_probs.at[idx_hi_shuffled].set(new_lp_hi)

        return positions, log_probs

    positions, log_probs = lax.fori_loop(
        0, n_temps - 1, swap_level, (positions, log_probs)
    )

    return positions, log_probs
