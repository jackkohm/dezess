"""Metrics computation for sampler benchmarking.

Computes ESS, R-hat, MCSE, jump distance, zero-move rate, and Z-matrix span
from sampler output.
"""

from __future__ import annotations

import numpy as np


def ess_fft(chain: np.ndarray) -> float:
    """FFT-based effective sample size for a 1D chain."""
    n = len(chain)
    if n < 10:
        return 0.0
    x = chain - chain.mean()
    var = np.var(x)
    if var < 1e-30:
        return 0.0
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real / (n * var)

    tau = 1.0
    for lag in range(1, n):
        tau += 2 * acf[lag]
        if lag >= 5 * tau:
            break
    return n / max(tau, 1.0)


def compute_ess(samples_3d: np.ndarray) -> np.ndarray:
    """Compute ESS per dimension from (n_steps, n_walkers, n_dim) samples.

    Uses the mean chain across walkers.
    """
    mean_chain = samples_3d.mean(axis=1)
    ndim = mean_chain.shape[1]
    return np.array([ess_fft(mean_chain[:, d]) for d in range(ndim)])


def compute_rhat(samples_3d: np.ndarray) -> np.ndarray:
    """Rank-normalized split R-hat (Vehtari et al. 2021).

    Treats each walker as an independent chain, splits each in half.
    Returns R-hat per dimension.
    """
    n_steps, n_walkers, ndim = samples_3d.shape
    half = n_steps // 2
    # Split each walker chain in half -> 2*n_walkers chains
    chains = np.concatenate([
        samples_3d[:half],   # first half: (half, n_walkers, ndim)
        samples_3d[half:2*half],  # second half
    ], axis=1)  # (half, 2*n_walkers, ndim)

    m = chains.shape[1]  # number of chains
    n = chains.shape[0]  # length of each chain

    rhat = np.zeros(ndim)
    for d in range(ndim):
        chain_means = chains[:, :, d].mean(axis=0)  # (m,)
        chain_vars = chains[:, :, d].var(axis=0, ddof=1)  # (m,)
        W = chain_vars.mean()
        B = n * chain_means.var(ddof=1)
        var_hat = (n - 1) / n * W + B / n
        rhat[d] = np.sqrt(var_hat / max(W, 1e-30))

    return rhat


def compute_diagnostics(result: dict, target=None) -> dict:
    """Compute full diagnostics from a sampler result dict.

    Parameters
    ----------
    result : dict
        Output from run_variant or run_demcz_slice.
    target : Target or None
        If provided, computes bias against true mean.

    Returns
    -------
    dict with all diagnostic metrics.
    """
    samples_3d = np.asarray(result["samples"])
    n_prod, n_walkers, ndim = samples_3d.shape
    wall_time = result.get("wall_time", 1.0)

    # ESS
    ess = compute_ess(samples_3d)

    # R-hat
    rhat = compute_rhat(samples_3d)

    # MCSE
    flat = samples_3d.reshape(-1, ndim)
    sample_mean = flat.mean(0)
    sample_std = flat.std(0)
    mcse = sample_std / np.sqrt(np.maximum(ess * n_walkers, 1))

    # Jump distance
    jumps = np.diff(samples_3d, axis=0)
    jump_dist = np.sqrt((jumps ** 2).sum(axis=-1))
    norm_jump = jump_dist / (sample_std.sum() + 1e-30)
    mean_jump = norm_jump.mean()

    # Zero-move rate
    zero_moves = (jump_dist == 0.0).mean()

    # Cap-hit rate from diagnostics if available
    diag = result.get("diagnostics", {})
    cap_hit_rate = diag.get("cap_hit_rate", zero_moves)

    # Z-matrix span
    z_mat = np.asarray(result.get("z_matrix", np.zeros((1, ndim))))
    z_centered = z_mat - z_mat.mean(0)
    svd_vals = np.linalg.svd(z_centered, compute_uv=False)
    sv_ratio = svd_vals[-1] / (svd_vals[0] + 1e-30) if len(svd_vals) > 0 else 0

    # Speed
    speed = n_prod / wall_time

    # ESS per log_prob evaluation
    n_expand = diag.get("n_expand", 3)
    n_shrink = diag.get("n_shrink", 12)
    evals_per_step = n_walkers * (2 * n_expand + n_shrink)
    total_evals = evals_per_step * n_prod
    ess_per_eval = ess.min() * n_walkers / max(total_evals, 1)

    metrics = {
        "wall_time": wall_time,
        "speed_its": speed,
        "ess_min": ess.min(),
        "ess_max": ess.max(),
        "ess_mean": ess.mean(),
        "ess_per_sec_min": ess.min() / wall_time,
        "ess_per_sec_mean": ess.mean() / wall_time,
        "ess_per_eval": ess_per_eval,
        "rhat_max": rhat.max(),
        "rhat_mean": rhat.mean(),
        "mcse_max": mcse.max(),
        "mcse_mean": mcse.mean(),
        "mean_jump": mean_jump,
        "zero_move_rate": zero_moves,
        "cap_hit_rate": cap_hit_rate,
        "sv_ratio": sv_ratio,
        "final_mu": result.get("mu", 0),
        "ndim": ndim,
        "n_walkers": n_walkers,
        "n_production": n_prod,
        "n_expand": n_expand,
        "n_shrink": n_shrink,
    }

    # Bias if target has known mean
    if target is not None and target.mean is not None:
        true_mean = np.asarray(target.mean)
        bias = np.abs(sample_mean - true_mean).max()
        rel_bias = bias / (np.abs(true_mean).max() + 1e-30)
        metrics["bias_max"] = bias
        metrics["rel_bias_max"] = rel_bias

    return metrics
