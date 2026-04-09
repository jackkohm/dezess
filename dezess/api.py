"""High-level API for dezess.

Provides a simple `sample()` function that handles variant selection,
warmup, production, and diagnostics automatically. Users just provide
a log-prob function and initial positions.

Usage:
    import dezess

    result = dezess.sample(log_prob_fn, init_positions, n_samples=2000)
    samples = result.samples          # (n_samples, n_walkers, n_dim)
    print(result.ess_min)             # minimum ESS across dimensions
    print(result.rhat_max)            # maximum R-hat across dimensions
"""

from __future__ import annotations

from typing import Callable, Optional, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.transforms import Transform

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


class SampleResult(NamedTuple):
    """Result from dezess.sample().

    Attributes
    ----------
    samples : ndarray, shape (n_steps, n_walkers, n_dim)
        Production samples. Flatten with samples.reshape(-1, n_dim) for
        a single array of draws.
    log_prob : ndarray, shape (n_steps, n_walkers)
        Log-probability at each sample.
    ess_min : float
        Minimum effective sample size across dimensions.
    ess_mean : float
        Mean effective sample size across dimensions.
    rhat_max : float
        Maximum R-hat across dimensions (< 1.05 is good).
    n_steps : int
        Number of production steps actually run (may be less than
        requested if target_ess triggered early stopping).
    wall_time : float
        Wall-clock time for production phase in seconds.
    mu : float
        Final tuned step size.
    variant : str
        Name of the variant that was used.
    """
    samples: np.ndarray
    log_prob: np.ndarray
    ess_min: float
    ess_mean: float
    rhat_max: float
    n_steps: int
    wall_time: float
    mu: float
    variant: str


def sample(
    log_prob_fn: Callable[[Array], Array],
    init_positions: Array,
    n_samples: int = 4000,
    n_warmup: Optional[int] = None,
    target_ess: Optional[float] = None,
    variant: str = "auto",
    n_walkers: Optional[int] = None,
    seed: int = 0,
    verbose: bool = True,
    progress_fn: Optional[Callable] = None,
    transform: Optional[Transform] = None,
    **kwargs,
) -> SampleResult:
    """Draw posterior samples using dezess.

    This is the recommended entry point. It handles variant selection,
    warmup length, and step size tuning automatically.

    Parameters
    ----------
    log_prob_fn : callable
        Log-probability function: (n_dim,) -> scalar. Must be JAX-compatible.
    init_positions : array, shape (n_walkers, n_dim)
        Initial walker positions. If n_walkers is specified and
        init_positions has a different number of walkers, it will be
        resampled.
    n_samples : int
        Number of production steps to run. Total samples =
        n_samples * n_walkers. Default 4000.
    n_warmup : int or None
        Warmup steps. If None, auto-set to max(500, n_samples // 4).
    target_ess : float or None
        If set, stop production early when minimum ESS across all
        dimensions reaches this value. Useful for expensive models.
    variant : str
        Sampling variant to use. Options:
        - "auto" (default): auto-select based on problem properties
        - "fast": scale_aware (best ESS/eval for expensive log-probs)
        - "thorough": multi3_scale (best ESS/s for cheap log-probs)
        - Any registered variant name (e.g., "baseline", "snooker_stochastic")
    seed : int
        Random seed.
    verbose : bool
        Print progress during sampling.
    **kwargs
    progress_fn : callable or None
        Callback function called every 50 production steps with a dict:
        {"step": int, "n_steps": int, "ess_min": float, "rhat_max": float,
         "speed": float (it/s)}. Useful for progress bars or notebook widgets.
    **kwargs
        Additional arguments passed to run_variant (e.g., z_max_size).

    Returns
    -------
    SampleResult
        Named tuple with samples, diagnostics, and metadata.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> result = dezess.sample(
    ...     lambda x: -0.5 * jnp.sum(x**2),
    ...     jnp.zeros((32, 5)),
    ...     n_samples=2000,
    ... )
    >>> print(f"ESS: {result.ess_min:.0f}, R-hat: {result.rhat_max:.4f}")
    """
    init_positions = jnp.array(init_positions, dtype=jnp.float64)
    if init_positions.ndim != 2:
        raise ValueError(
            f"init_positions must be 2D (n_walkers, n_dim), got shape {init_positions.shape}"
        )
    n_walkers_init, n_dim = init_positions.shape
    if n_walkers_init < 2:
        raise ValueError(f"Need at least 2 walkers, got {n_walkers_init}")
    if n_dim < 1:
        raise ValueError(f"Need at least 1 dimension, got {n_dim}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    # Auto-set warmup: ensure enough steps to fill the Z-matrix buffer.
    # Formula: z_max_size * append_interval / n_walkers, with a minimum.
    if n_warmup is None:
        z_max = kwargs.get("z_max_size", 50000)
        append_interval = 5  # Z-matrix appends every 5 warmup steps
        warmup_to_fill = int(z_max * append_interval / n_walkers_init)
        base_warmup = max(500, n_samples // 4)
        n_warmup = max(base_warmup, warmup_to_fill)

    # Auto-select variant
    config = _resolve_variant(variant, n_dim)

    n_steps = n_samples + n_warmup

    if verbose:
        print(f"dezess: {n_dim}D, {n_walkers_init} walkers, "
              f"{n_warmup} warmup + {n_samples} production, "
              f"variant={config.name}", flush=True)

    key = jax.random.PRNGKey(seed)

    result = run_variant(
        log_prob_fn,
        init_positions,
        n_steps=n_steps,
        config=config,
        n_warmup=n_warmup,
        key=key,
        mu=1.0,
        tune=True,
        target_ess=target_ess,
        progress_fn=progress_fn,
        verbose=verbose,
        transform=transform,
        **kwargs,
    )

    # Extract streaming diagnostics
    streaming = result["diagnostics"].get("streaming", {})

    # Convergence warnings
    rhat = streaming.get("rhat_max", float("inf"))
    ess = streaming.get("ess_min", 0.0)
    n_div = streaming.get("n_divergent", 0)
    if verbose:
        # Clean summary line
        wall = result["wall_time"]
        speed = result["n_production"] / wall if wall > 0 else 0
        print(f"dezess: done — {result['n_production']} steps in {wall:.1f}s "
              f"({speed:.0f} it/s), ESS={ess:.0f}, R-hat={rhat:.3f}", flush=True)
        if rhat > 1.1:
            print(f"  WARNING: R-hat={rhat:.3f} > 1.1 — chains may not have converged. "
                  f"Consider more warmup or production steps.", flush=True)
        if ess < 100:
            print(f"  WARNING: ESS={ess:.0f} < 100 — insufficient effective samples. "
                  f"Consider more production steps or target_ess.", flush=True)
        if n_div > 0:
            print(f"  WARNING: {n_div} divergent steps detected — check model for "
                  f"numerical issues.", flush=True)

    return SampleResult(
        samples=np.asarray(result["samples"]),
        log_prob=np.asarray(result["log_prob"]),
        ess_min=streaming.get("ess_min", 0.0),
        ess_mean=streaming.get("ess_mean", 0.0),
        rhat_max=streaming.get("rhat_max", float("inf")),
        n_steps=result["n_production"],
        wall_time=result["wall_time"],
        mu=result["mu"],
        variant=config.name,
    )


def sample_until(
    log_prob_fn: Callable[[Array], Array],
    init_positions: Array,
    target_ess: float = 400,
    max_rhat: float = 1.05,
    max_steps: int = 50000,
    batch_size: int = 1000,
    variant: str = "auto",
    seed: int = 0,
    verbose: bool = True,
    **kwargs,
) -> SampleResult:
    """Sample until convergence criteria are met.

    Runs production in batches, checking ESS and R-hat after each batch.
    Stops when both criteria are satisfied or max_steps is reached.

    Parameters
    ----------
    target_ess : float
        Minimum ESS to achieve. Default 400.
    max_rhat : float
        Maximum acceptable R-hat. Default 1.05.
    max_steps : int
        Maximum total production steps. Default 50000.
    batch_size : int
        Steps per batch. Default 1000.
    """
    n_walkers, n_dim = jnp.array(init_positions).shape

    # Auto warmup
    n_warmup = kwargs.pop("n_warmup", None)
    if n_warmup is None:
        base_warmup = max(500, batch_size // 2)
        dim_factor = max(1.0, n_dim / 20.0)
        n_warmup = int(base_warmup * dim_factor)

    all_samples = []
    all_lps = []
    total_steps = 0
    current_init = jnp.array(init_positions, dtype=jnp.float64)
    mu = 1.0
    z_matrix = None

    while total_steps < max_steps:
        steps_this_batch = min(batch_size, max_steps - total_steps)

        result = run_variant(
            log_prob_fn,
            current_init,
            n_steps=steps_this_batch + (n_warmup if total_steps == 0 else 0),
            config=_resolve_variant(variant, n_dim),
            n_warmup=n_warmup if total_steps == 0 else 0,
            key=jax.random.PRNGKey(seed + total_steps),
            mu=mu,
            tune=total_steps == 0,
            z_initial=z_matrix,
            verbose=verbose,
            **kwargs,
        )

        all_samples.append(np.asarray(result["samples"]))
        all_lps.append(np.asarray(result["log_prob"]))
        total_steps += result["n_production"]
        mu = result["mu"]
        z_matrix = np.asarray(result["z_matrix"])
        current_init = jnp.array(np.asarray(result["samples"])[-1])

        # Check convergence
        combined = np.concatenate(all_samples, axis=0)
        combined_lps = np.concatenate(all_lps, axis=0)
        from dezess.diagnostics import StreamingDiagnostics
        stream = StreamingDiagnostics(n_walkers, n_dim)
        for i in range(combined.shape[0]):
            stream.update(combined[i], combined_lps[i])
        diag = stream.summary()

        ess = diag["ess_min"]
        rhat = diag["rhat_max"]

        if verbose:
            print(f"dezess.sample_until: {total_steps} steps, "
                  f"ESS={ess:.0f}/{target_ess:.0f}, "
                  f"R-hat={rhat:.3f}/{max_rhat:.3f}", flush=True)

        if ess >= target_ess and rhat <= max_rhat:
            if verbose:
                print(f"dezess.sample_until: converged!", flush=True)
            break

    combined = np.concatenate(all_samples, axis=0)
    combined_lps = np.concatenate(all_lps, axis=0)

    return SampleResult(
        samples=combined,
        log_prob=combined_lps,
        ess_min=ess,
        ess_mean=diag["ess_mean"],
        rhat_max=rhat,
        n_steps=total_steps,
        wall_time=0.0,
        mu=mu,
        variant=variant,
    )


def run_chains(
    log_prob_fn: Callable[[Array], Array],
    init_positions: Array,
    n_chains: int = 4,
    n_samples: int = 2000,
    n_warmup: Optional[int] = None,
    variant: str = "auto",
    seed: int = 0,
    verbose: bool = True,
    **kwargs,
) -> SampleResult:
    """Run multiple independent chains and combine results.

    Each chain starts from a different random perturbation of init_positions.
    This provides proper between-chain R-hat validation. The returned
    SampleResult treats all chains' walkers as independent chains.

    Parameters
    ----------
    n_chains : int
        Number of independent chains to run. Default 4.
    """
    all_samples = []
    all_lps = []
    n_walkers, n_dim = init_positions.shape

    for chain_idx in range(n_chains):
        chain_key = jax.random.PRNGKey(seed + chain_idx * 1000)
        # Perturb initialization for each chain
        noise = jax.random.normal(chain_key, init_positions.shape, dtype=jnp.float64) * 0.01
        chain_init = init_positions + noise

        if verbose:
            print(f"  Chain {chain_idx + 1}/{n_chains}:", flush=True)

        chain_result = sample(
            log_prob_fn, chain_init,
            n_samples=n_samples,
            n_warmup=n_warmup,
            variant=variant,
            seed=seed + chain_idx * 1000 + 1,
            verbose=verbose,
            **kwargs,
        )
        all_samples.append(chain_result.samples)
        all_lps.append(chain_result.log_prob)

    # Combine: concatenate walker dimension across chains
    # Each chain has (n_steps, n_walkers, n_dim) -> combined (n_steps, n_chains*n_walkers, n_dim)
    min_steps = min(s.shape[0] for s in all_samples)
    combined_samples = np.concatenate([s[:min_steps] for s in all_samples], axis=1)
    combined_lps = np.concatenate([lp[:min_steps] for lp in all_lps], axis=1)

    # Re-compute streaming diagnostics on combined samples
    from dezess.diagnostics import StreamingDiagnostics
    stream = StreamingDiagnostics(combined_samples.shape[1], n_dim)
    for step in range(min_steps):
        stream.update(combined_samples[step], combined_lps[step])
    diag = stream.summary()

    if verbose:
        print(f"\n  Combined: {n_chains} chains x {n_walkers} walkers = "
              f"{n_chains * n_walkers} total chains, {min_steps} steps")
        print(f"  ESS min: {diag['ess_min']:.0f}, R-hat max: {diag['rhat_max']:.4f}")

    return SampleResult(
        samples=combined_samples,
        log_prob=combined_lps,
        ess_min=diag["ess_min"],
        ess_mean=diag["ess_mean"],
        rhat_max=diag["rhat_max"],
        n_steps=min_steps,
        wall_time=sum(1 for _ in all_samples),  # approximate
        mu=0.0,  # not meaningful for combined
        variant=variant,
    )


def diagnose(result: SampleResult) -> None:
    """Print a diagnostic summary of the sampling result.

    Checks for common issues and provides actionable recommendations.
    """
    print(f"\n{'='*60}")
    print(f"  dezess diagnostic summary")
    print(f"{'='*60}")

    n_steps = result.n_steps
    n_walkers = result.samples.shape[1] if result.samples.ndim == 3 else 0
    n_dim = result.samples.shape[2] if result.samples.ndim == 3 else result.samples.shape[1]
    n_total = n_steps * n_walkers if n_walkers > 0 else n_steps

    print(f"  Steps: {n_steps}, Walkers: {n_walkers}, Dim: {n_dim}")
    print(f"  Total draws: {n_total:,}")
    print(f"  Wall time: {result.wall_time:.1f}s")
    print(f"  Variant: {result.variant}")
    print()

    # ESS check
    ess = result.ess_min
    print(f"  ESS (min): {ess:.0f}")
    if ess >= 400:
        print(f"    OK — sufficient for most posterior summaries")
    elif ess >= 100:
        print(f"    MARGINAL — consider more production steps")
    else:
        print(f"    LOW — need significantly more steps or better tuning")

    # R-hat check
    rhat = result.rhat_max
    print(f"  R-hat (max): {rhat:.4f}")
    if rhat < 1.01:
        print(f"    EXCELLENT — chains well-converged")
    elif rhat < 1.05:
        print(f"    GOOD — chains likely converged")
    elif rhat < 1.1:
        print(f"    MARGINAL — consider more warmup")
    else:
        print(f"    POOR — chains have not converged. Try:")
        print(f"      - More warmup steps")
        print(f"      - More walkers")
        print(f"      - Different initialization")

    # Efficiency
    if result.wall_time > 0:
        ess_per_sec = ess / result.wall_time
        print(f"  ESS/s: {ess_per_sec:.1f}")

    print(f"  Tuned mu: {result.mu:.6f}")
    print(f"{'='*60}\n")


def init_walkers(
    n_walkers: int,
    n_dim: int,
    center: Optional[np.ndarray] = None,
    scale: float = 0.1,
    seed: int = 0,
) -> jnp.ndarray:
    """Generate initial walker positions.

    Parameters
    ----------
    n_walkers : int
        Number of walkers.
    n_dim : int
        Number of dimensions.
    center : array or None
        Center point for initialization. If None, uses zeros.
    scale : float
        Standard deviation of the Gaussian ball around center.
        Default 0.1 (tight ball, good for well-initialized runs).
    seed : int
        Random seed.

    Returns
    -------
    array (n_walkers, n_dim)
    """
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, (n_walkers, n_dim), dtype=jnp.float64) * scale
    if center is not None:
        center = jnp.array(center, dtype=jnp.float64)
        return noise + center
    return noise


def find_map(
    log_prob_fn: Callable[[Array], Array],
    init: Array,
    n_steps: int = 1000,
    learning_rate: float = 0.01,
) -> jnp.ndarray:
    """Find the MAP (maximum a posteriori) estimate via gradient ascent.

    Useful for finding a good starting point for sampling. Requires
    the log-prob to be differentiable via JAX autodiff.

    Parameters
    ----------
    log_prob_fn : callable
        Log-probability function (must support jax.grad).
    init : array (n_dim,)
        Starting point for optimization.
    n_steps : int
        Number of gradient ascent steps.
    learning_rate : float
        Step size for gradient ascent.

    Returns
    -------
    map_estimate : array (n_dim,)
        Approximate MAP point.
    """
    x = jnp.array(init, dtype=jnp.float64)
    grad_fn = jax.grad(log_prob_fn)

    @jax.jit
    def step(x):
        g = grad_fn(x)
        return x + learning_rate * g

    for _ in range(n_steps):
        x = step(x)

    return x


def _resolve_variant(variant: str, n_dim: int) -> VariantConfig:
    """Resolve a variant name to a VariantConfig."""
    if variant == "auto":
        # scale_aware is the best all-rounder across dimensions.
        # Tested: momentum, PCA, local_pair all underperform at 30D+.
        return VariantConfig(
            name="scale_aware",
            direction="de_mcz",
            width="scale_aware",
            slice_fn="fixed",
            zmatrix="circular",
            ensemble="standard",
            width_kwargs={"scale_factor": 1.0},
        )
    elif variant == "lean":
        # Maximum ESS per evaluation (very expensive log-probs).
        # Uses minimal slice budget (5 evals/step). ~3% cap-hit rate
        # on easy targets, higher on funnels. Best when each log-prob
        # call costs seconds.
        return VariantConfig(
            name="scale_aware_lean",
            direction="de_mcz",
            width="scale_aware",
            slice_fn="fixed",
            zmatrix="circular",
            ensemble="standard",
            width_kwargs={"scale_factor": 1.0},
            slice_kwargs={"n_expand": 1, "n_shrink": 3},
        )
    elif variant == "fast":
        # Best ESS per evaluation (for expensive log-probs)
        return VariantConfig(
            name="scale_aware",
            direction="de_mcz",
            width="scale_aware",
            slice_fn="fixed",
            zmatrix="circular",
            ensemble="standard",
            width_kwargs={"scale_factor": 1.0},
        )
    elif variant == "thorough":
        # Best wall-clock ESS/s (for cheap log-probs)
        return VariantConfig(
            name="multi3_scale",
            direction="de_mcz",
            width="scale_aware",
            slice_fn="fixed",
            zmatrix="circular",
            ensemble="standard",
            n_slices_per_step=3,
            width_kwargs={"scale_factor": 1.0},
        )
    else:
        # Try to look up from registry
        from dezess.benchmark.registry import VARIANTS
        if variant in VARIANTS:
            return VARIANTS[variant]
        raise ValueError(
            f"Unknown variant '{variant}'. Use 'auto', 'fast', 'thorough', "
            f"or a registered name: {sorted(VARIANTS.keys())}"
        )
