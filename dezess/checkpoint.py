"""Checkpoint save/resume for dezess sampler runs.

Saves the frozen Z-matrix, final walker positions, tuned mu, and config
so a run can be resumed for more production samples without re-doing warmup.

Usage:
    # Save after a run
    result = dezess.sample(log_prob, init, n_samples=1000)
    dezess.save_checkpoint("run1.npz", result)

    # Resume for more samples
    result2 = dezess.resume("run1.npz", log_prob, n_samples=2000)

    # Resume with a transform (must pass same transform used originally)
    result3 = dezess.resume("run1.npz", log_prob, n_samples=2000,
                            transform=my_transform)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def save_checkpoint(path: str, result: dict) -> None:
    """Save sampler state to a .npz file for later resumption.

    Parameters
    ----------
    path : str
        File path (should end in .npz).
    result : dict
        Result dict from run_variant or the raw dict underlying SampleResult.
        Must contain: samples, log_prob, mu, z_matrix.
    """
    samples = np.asarray(result["samples"]) if "samples" in result else np.asarray(result.samples)
    log_prob = np.asarray(result["log_prob"]) if "log_prob" in result else np.asarray(result.log_prob)

    # Get final walker positions (last production step)
    final_positions = samples[-1]  # (n_walkers, n_dim)
    final_lps = log_prob[-1]  # (n_walkers,)

    mu = result["mu"] if "mu" in result else result.mu
    z_matrix = np.asarray(result.get("z_matrix", result["z_matrix"]))

    # Save config name if available
    config_name = ""
    if "config" in result and result["config"] is not None:
        config_name = result["config"].name

    # Save mu_blocks if block-Gibbs was used
    mu_blocks = result.get("mu_blocks", None)

    save_dict = dict(
        final_positions=final_positions,
        final_log_probs=final_lps,
        mu=np.float64(mu),
        z_matrix=z_matrix,
        config_name=np.array(config_name),
        samples=samples,
        log_prob=log_prob,
    )
    if mu_blocks is not None:
        save_dict["mu_blocks"] = np.asarray(mu_blocks)

    np.savez(path, **save_dict)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint file.

    Returns dict with: final_positions, final_log_probs, mu, z_matrix,
    config_name, samples, log_prob, and optionally mu_blocks.
    """
    data = np.load(path, allow_pickle=False)
    result = {
        "final_positions": data["final_positions"],
        "final_log_probs": data["final_log_probs"],
        "mu": float(data["mu"]),
        "z_matrix": data["z_matrix"],
        "config_name": str(data["config_name"]),
        "samples": data["samples"],
        "log_prob": data["log_prob"],
    }
    if "mu_blocks" in data:
        result["mu_blocks"] = data["mu_blocks"]
    return result


def resume(
    path: str,
    log_prob_fn: Callable[[Array], Array],
    n_samples: int = 2000,
    target_ess: Optional[float] = None,
    variant: Optional[str] = None,
    transform=None,
    seed: int = 1,
    verbose: bool = True,
) -> dict:
    """Resume sampling from a checkpoint.

    Loads the frozen Z-matrix and final walker positions, then runs
    more production steps without re-doing warmup.

    Parameters
    ----------
    path : str
        Path to checkpoint .npz file.
    log_prob_fn : callable
        Same log-prob function used in the original run.
    n_samples : int
        Number of additional production steps to run.
    target_ess : float or None
        Early stopping target (ESS across new + old samples).
    variant : str or None
        Variant to use. If None, uses the variant from the checkpoint.
    transform : Transform or None
        Same transform used in the original run. Required if the original
        run used a transform — checkpoint stores x-space samples but
        the sampler works in z-space.
    seed : int
        Random seed for the resumed run.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with new samples and combined diagnostics.
    """
    from dezess.core.loop import run_variant
    from dezess.api import _resolve_variant

    ckpt = load_checkpoint(path)

    # Resolve variant
    if variant is None:
        variant = ckpt["config_name"] or "auto"
    n_dim = ckpt["final_positions"].shape[1]
    config = _resolve_variant(variant, n_dim)

    init = jnp.array(ckpt["final_positions"], dtype=jnp.float64)
    z_initial = jnp.array(ckpt["z_matrix"], dtype=jnp.float64)

    if verbose:
        print(f"dezess: Resuming from {path} ({ckpt['z_matrix'].shape[0]} Z entries, "
              f"mu={ckpt['mu']:.4f}, variant={config.name})", flush=True)

    result = run_variant(
        log_prob_fn,
        init,
        n_steps=n_samples,  # all production, no warmup
        config=config,
        n_warmup=0,
        key=jax.random.PRNGKey(seed),
        mu=ckpt["mu"],
        tune=False,  # already tuned
        z_initial=z_initial,
        target_ess=target_ess,
        transform=transform,
        verbose=verbose,
    )

    # Combine old and new samples
    old_samples = ckpt["samples"]
    new_samples = np.asarray(result["samples"])
    combined_samples = np.concatenate([old_samples, new_samples], axis=0)

    old_lp = ckpt["log_prob"]
    new_lp = np.asarray(result["log_prob"])
    combined_lp = np.concatenate([old_lp, new_lp], axis=0)

    result["samples"] = jnp.array(combined_samples)
    result["log_prob"] = jnp.array(combined_lp)
    result["n_production"] = combined_samples.shape[0]

    if verbose:
        print(f"dezess: Combined {old_samples.shape[0]} + {new_samples.shape[0]} = "
              f"{combined_samples.shape[0]} production steps", flush=True)

    return result
