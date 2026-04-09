"""Automatic tuning utilities for dezess.

Provides functions to recommend walker count and warmup length based
on short pilot runs.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.api import _resolve_variant

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def recommend_walkers(
    log_prob_fn: Callable[[Array], Array],
    n_dim: int,
    seed: int = 0,
    variant: str = "auto",
    verbose: bool = True,
) -> int:
    """Recommend the number of walkers based on a short pilot run.

    Runs quick trials with different walker counts and picks the one
    that maximizes ESS per log-prob evaluation.

    Parameters
    ----------
    log_prob_fn : callable
        Log-probability function.
    n_dim : int
        Number of dimensions.
    seed : int
        Random seed.
    variant : str
        Variant to use for the pilot.
    verbose : bool
        Print pilot results.

    Returns
    -------
    int : recommended number of walkers.
    """
    config = _resolve_variant(variant, n_dim)

    # Test a range of walker counts
    # Minimum is 2*n_dim for DE-MCz to have diverse directions,
    # but scale_aware can work with fewer
    candidates = []
    min_walkers = max(8, n_dim)
    for factor in [1.0, 1.5, 2.0, 3.0, 4.0]:
        nw = max(min_walkers, int(n_dim * factor))
        # Round up to even for symmetry
        nw = nw + (nw % 2)
        if nw not in [c[0] for c in candidates]:
            candidates.append((nw, factor))

    pilot_steps = 500
    pilot_warmup = 200

    if verbose:
        print(f"dezess.recommend_walkers: {n_dim}D, testing {len(candidates)} walker counts")

    best_nw = candidates[0][0]
    best_ess_per_eval = 0.0
    results = []

    for nw, factor in candidates:
        try:
            key = jax.random.PRNGKey(seed)
            init = jax.random.normal(key, (nw, n_dim), dtype=jnp.float64) * 0.1

            result = run_variant(
                log_prob_fn, init,
                n_steps=pilot_steps + pilot_warmup,
                config=config,
                n_warmup=pilot_warmup,
                key=jax.random.PRNGKey(seed + 1),
                mu=1.0,
                tune=True,
                verbose=False,
            )

            streaming = result["diagnostics"].get("streaming", {})
            ess_min = streaming.get("ess_min", 0.0)
            wall_time = result["wall_time"]

            # ESS per evaluation: ess / (n_steps * n_walkers * evals_per_step)
            evals_per_step = nw * 18  # 2*n_expand + n_shrink = 18
            total_evals = pilot_steps * evals_per_step
            ess_per_eval = ess_min / total_evals if total_evals > 0 else 0

            ess_per_sec = ess_min / wall_time if wall_time > 0 else 0

            results.append((nw, ess_per_eval, ess_per_sec, ess_min))

            if verbose:
                print(f"  nw={nw:4d}: ESS/eval={ess_per_eval:.6f}  "
                      f"ESS/s={ess_per_sec:.1f}  ESS_min={ess_min:.0f}")

            if ess_per_eval > best_ess_per_eval:
                best_ess_per_eval = ess_per_eval
                best_nw = nw

        except Exception as e:
            if verbose:
                print(f"  nw={nw:4d}: ERROR {str(e)[:50]}")

        # Clear JIT caches to avoid OOM
        import gc
        gc.collect()

    if verbose:
        print(f"  Recommendation: {best_nw} walkers "
              f"(ESS/eval={best_ess_per_eval:.6f})")

    return best_nw


def estimate_n_steps(
    log_prob_fn: Callable[[Array], Array],
    init_positions: Array,
    target_ess: float = 1000,
    variant: str = "auto",
    seed: int = 0,
    verbose: bool = True,
) -> int:
    """Estimate the number of production steps needed to reach target ESS.

    Runs a short pilot, measures ESS growth rate, and extrapolates.

    Parameters
    ----------
    log_prob_fn : callable
    init_positions : array (n_walkers, n_dim)
    target_ess : float
        Target minimum ESS.
    variant : str
    seed : int
    verbose : bool

    Returns
    -------
    int : estimated number of production steps needed.
    """
    n_walkers, n_dim = init_positions.shape
    config = _resolve_variant(variant, n_dim)

    pilot_steps = 300
    pilot_warmup = 200

    result = run_variant(
        log_prob_fn, init_positions,
        n_steps=pilot_steps + pilot_warmup,
        config=config,
        n_warmup=pilot_warmup,
        key=jax.random.PRNGKey(seed),
        mu=1.0,
        tune=True,
        verbose=False,
    )

    streaming = result["diagnostics"].get("streaming", {})
    pilot_ess = streaming.get("ess_min", 0.0)

    if pilot_ess <= 0:
        estimate = pilot_steps * 10  # conservative fallback
    else:
        # ESS grows roughly linearly with steps
        ess_per_step = pilot_ess / pilot_steps
        estimate = int(target_ess / ess_per_step) + 1

    # Add safety margin
    estimate = int(estimate * 1.3)
    estimate = max(estimate, 500)

    if verbose:
        print(f"dezess.estimate_n_steps: pilot ESS={pilot_ess:.0f} in {pilot_steps} steps "
              f"-> estimate {estimate} steps for target ESS={target_ess:.0f}")

    return estimate
