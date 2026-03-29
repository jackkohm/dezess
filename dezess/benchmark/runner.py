"""Benchmark runner: executes variants on targets and collects metrics.

Supports running specific variant×target combinations, repeated trials,
and produces structured output for comparison. Clears GPU memory between
runs to prevent OOM on long benchmark sweeps.
"""

from __future__ import annotations

import gc
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from dezess.core.loop import run_variant
from dezess.benchmark.metrics import compute_diagnostics
from dezess.benchmark.registry import VARIANTS, TARGET_SETS, VARIANT_SETS
from dezess.targets import ALL_TARGETS

jax.config.update("jax_enable_x64", True)


def _clear_gpu_memory():
    """Release unreferenced JAX arrays and run garbage collection.

    We do NOT call jax.clear_caches() every run because that forces
    expensive re-JIT compilation. Instead we just gc.collect() to
    release array references, and only clear caches every N runs.
    """
    gc.collect()


def _clear_gpu_memory_full():
    """Full GPU memory clear including JIT caches. Use sparingly."""
    jax.clear_caches()
    gc.collect()


def run_single(
    variant_name: str,
    target_name: str,
    n_walkers: int = 64,
    n_steps: int = 5000,
    n_warmup: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run a single variant on a single target and return metrics.

    Returns dict with variant_name, target_name, and all metrics.
    """
    config = VARIANTS[variant_name]
    target = ALL_TARGETS[target_name]()

    key = jax.random.PRNGKey(seed)
    init = target.sample(key, n_walkers)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Running: {variant_name} on {target_name}")
        print(f"  {n_walkers} walkers, {n_steps} steps ({n_warmup} warmup)")
        print(f"{'='*60}")

    result = run_variant(
        target.log_prob, init,
        n_steps=n_steps,
        config=config,
        n_warmup=n_warmup,
        key=jax.random.PRNGKey(seed + 1),
        mu=1.0,
        tune=True,
        verbose=verbose,
    )

    metrics = compute_diagnostics(result, target)
    metrics["variant"] = variant_name
    metrics["target"] = target_name
    metrics["seed"] = seed

    if verbose:
        _print_metrics(metrics)

    return metrics


def run_comparison(
    variant_names: Optional[list] = None,
    target_names: Optional[list] = None,
    variant_set: Optional[str] = None,
    target_set: Optional[str] = None,
    n_walkers: int = 64,
    n_steps: int = 5000,
    n_warmup: int = 1000,
    n_trials: int = 1,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """Run multiple variants on multiple targets and collect all metrics.

    Clears GPU memory between runs to prevent OOM accumulation.

    Parameters
    ----------
    variant_names : list or None
        Specific variants to run. If None, uses variant_set.
    target_names : list or None
        Specific targets. If None, uses target_set.
    variant_set : str or None
        Named set from VARIANT_SETS (e.g., "width_variants").
    target_set : str or None
        Named set from TARGET_SETS (e.g., "standard").
    n_trials : int
        Number of repeated runs per combination (different seeds).

    Returns
    -------
    list of metric dicts, one per (variant, target, trial).
    """
    if variant_names is None:
        variant_names = VARIANT_SETS.get(variant_set, ["baseline"])
    if target_names is None:
        target_names = TARGET_SETS.get(target_set, ["isotropic_10"])

    all_metrics = []
    total = len(variant_names) * len(target_names) * n_trials
    idx = 0

    for variant_name in variant_names:
        for target_name in target_names:
            for trial in range(n_trials):
                idx += 1
                trial_seed = seed + trial * 1000
                if verbose:
                    print(f"\n[{idx}/{total}] {variant_name} x {target_name} "
                          f"(trial {trial+1}/{n_trials})")

                try:
                    metrics = run_single(
                        variant_name, target_name,
                        n_walkers=n_walkers,
                        n_steps=n_steps,
                        n_warmup=n_warmup,
                        seed=trial_seed,
                        verbose=verbose,
                    )
                    metrics["trial"] = trial
                    all_metrics.append(metrics)
                except Exception as e:
                    print(f"  ERROR: {variant_name} x {target_name}: {e}")
                    all_metrics.append({
                        "variant": variant_name,
                        "target": target_name,
                        "trial": trial,
                        "error": str(e),
                    })

                # Light cleanup between runs (gc only, keeps JIT caches)
                _clear_gpu_memory()
                # Full cache clear every 10 runs to prevent OOM buildup
                if idx % 10 == 0:
                    _clear_gpu_memory_full()

    return all_metrics


def _print_metrics(m: dict):
    """Pretty-print a metrics dict."""
    print(f"\n  Results for {m.get('variant', '?')} on {m.get('target', '?')}:")
    print(f"    Wall time:      {m.get('wall_time', 0):.1f}s")
    print(f"    Speed:          {m.get('speed_its', 0):.0f} it/s")
    print(f"    Final mu:       {m.get('final_mu', 0):.4f}")
    print(f"    ESS:            min={m.get('ess_min', 0):.0f} "
          f"mean={m.get('ess_mean', 0):.0f} max={m.get('ess_max', 0):.0f}")
    print(f"    ESS/s:          min={m.get('ess_per_sec_min', 0):.1f} "
          f"mean={m.get('ess_per_sec_mean', 0):.1f}")
    print(f"    ESS/eval:       {m.get('ess_per_eval', 0):.6f}")
    print(f"    R-hat:          max={m.get('rhat_max', 0):.4f}")
    print(f"    MCSE:           max={m.get('mcse_max', 0):.4f}")
    print(f"    Jump distance:  {m.get('mean_jump', 0):.4f}")
    print(f"    Zero-move rate: {m.get('zero_move_rate', 0):.4f}")
    print(f"    Cap-hit rate:   {m.get('cap_hit_rate', 0):.4f}")
    print(f"    Z-matrix SV:    {m.get('sv_ratio', 0):.4f}")
    if "bias_max" in m:
        print(f"    Bias (max):     {m.get('bias_max', 0):.4f}")
