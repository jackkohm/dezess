"""Utility functions for working with dezess samples.

Common post-processing operations: flattening walker dimensions,
thinning correlated samples, computing summary statistics, and
extracting traces for diagnostics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def flatten_samples(samples: np.ndarray) -> np.ndarray:
    """Flatten (n_steps, n_walkers, n_dim) -> (n_total, n_dim).

    Merges the step and walker dimensions into a single draw dimension.
    """
    if samples.ndim == 3:
        n_steps, n_walkers, n_dim = samples.shape
        return samples.reshape(-1, n_dim)
    return samples


def thin_samples(
    samples: np.ndarray,
    thin: Optional[int] = None,
    target_ess_ratio: float = 0.5,
) -> np.ndarray:
    """Thin samples to reduce autocorrelation.

    Parameters
    ----------
    samples : (n_steps, n_walkers, n_dim) or (n_total, n_dim)
        Input samples.
    thin : int or None
        Thinning interval. If None, auto-compute from lag-1 autocorrelation
        to achieve approximately target_ess_ratio * n_samples effective
        independent draws.
    target_ess_ratio : float
        Target ratio of ESS to total samples when auto-thinning.
        Default 0.5 (keep half the samples as approximately independent).

    Returns
    -------
    Thinned samples with same number of dimensions but fewer draws.
    """
    if samples.ndim == 3:
        n_steps, n_walkers, n_dim = samples.shape
        if thin is None:
            # Estimate thinning from lag-1 autocorrelation of the mean chain
            mean_chain = samples.mean(axis=1)  # (n_steps, n_dim)
            thin = _auto_thin(mean_chain, target_ess_ratio)
        return samples[::thin]
    elif samples.ndim == 2:
        if thin is None:
            thin = _auto_thin(samples, target_ess_ratio)
        return samples[::thin]
    return samples


def _auto_thin(chain: np.ndarray, target_ratio: float = 0.5) -> int:
    """Estimate thinning interval from lag-1 autocorrelation."""
    n = chain.shape[0]
    if n < 10:
        return 1

    # Compute lag-1 autocorrelation across all dimensions
    rho1_vals = []
    for d in range(chain.shape[1]):
        x = chain[:, d]
        x_centered = x - x.mean()
        var = np.var(x_centered)
        if var < 1e-30:
            continue
        acf1 = np.sum(x_centered[:-1] * x_centered[1:]) / ((n - 1) * var)
        rho1_vals.append(acf1)

    if not rho1_vals:
        return 1

    # Max lag-1 autocorrelation across dims
    rho1 = np.max(np.abs(rho1_vals))

    # IAT ≈ (1 + rho1) / (1 - rho1)
    if rho1 >= 0.99:
        return max(1, n // 10)  # very correlated, thin aggressively
    iat = max(1.0, (1 + rho1) / (1 - rho1))

    # Thin to achieve target ESS ratio
    thin = max(1, int(iat / target_ratio))
    return min(thin, n // 2)  # don't thin more than half


def summary_stats(
    samples: np.ndarray,
    param_names: Optional[list] = None,
    quantiles: tuple = (0.025, 0.25, 0.5, 0.75, 0.975),
) -> dict:
    """Compute summary statistics for each parameter.

    Parameters
    ----------
    samples : (n_steps, n_walkers, n_dim) or (n_total, n_dim)
    param_names : list of str or None
        Names for each dimension. Auto-generated if None.
    quantiles : tuple of float
        Quantiles to compute.

    Returns
    -------
    dict with keys: "mean", "std", "quantiles", "param_names"
    """
    flat = flatten_samples(samples)
    n_dim = flat.shape[1]

    if param_names is None:
        param_names = [f"x[{i}]" for i in range(n_dim)]

    means = np.mean(flat, axis=0)
    stds = np.std(flat, axis=0)
    qs = np.quantile(flat, quantiles, axis=0)  # (n_quantiles, n_dim)

    return {
        "mean": means,
        "std": stds,
        "quantiles": {f"{q:.1%}": qs[i] for i, q in enumerate(quantiles)},
        "param_names": param_names,
    }


def autocorrelation(
    samples: np.ndarray,
    max_lag: int = 100,
    dim: int = 0,
) -> np.ndarray:
    """Compute the autocorrelation function for a given dimension.

    Parameters
    ----------
    samples : (n_steps, n_walkers, n_dim) or (n_total, n_dim)
    max_lag : int
        Maximum lag to compute. Default 100.
    dim : int
        Which dimension to compute ACF for.

    Returns
    -------
    acf : (max_lag,) array of autocorrelation values at lags 0..max_lag-1.
    """
    if samples.ndim == 3:
        # Use mean chain across walkers
        chain = samples[:, :, dim].mean(axis=1)
    elif samples.ndim == 2:
        chain = samples[:, dim]
    else:
        chain = samples

    n = len(chain)
    max_lag = min(max_lag, n - 1)
    x = chain - chain.mean()
    var = np.var(x)
    if var < 1e-30:
        return np.zeros(max_lag)

    # FFT-based autocorrelation
    fft_x = np.fft.fft(x, n=2 * n)
    acf_full = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real / (n * var)
    return acf_full[:max_lag]


def integrated_autocorr_time(
    samples: np.ndarray,
    dim: int = 0,
) -> float:
    """Estimate the integrated autocorrelation time (IAT) for a dimension.

    Uses Geyer's initial positive sequence estimator (conservative).

    Parameters
    ----------
    samples : (n_steps, n_walkers, n_dim) or (n_total, n_dim)
    dim : int
        Dimension to compute IAT for.

    Returns
    -------
    float : estimated IAT. ESS ≈ n_samples / IAT.
    """
    acf = autocorrelation(samples, max_lag=len(samples) if samples.ndim < 3 else samples.shape[0], dim=dim)
    n = len(acf)

    # Geyer's initial positive sequence
    tau = 1.0
    for lag in range(1, n):
        tau += 2 * acf[lag]
        if lag >= 5 * tau:
            break
    return max(tau, 1.0)


def print_summary(
    samples: np.ndarray,
    param_names: Optional[list] = None,
    max_params: int = 20,
) -> None:
    """Print a formatted summary table of parameter estimates.

    Parameters
    ----------
    samples : (n_steps, n_walkers, n_dim) or (n_total, n_dim)
    param_names : list of str or None
    max_params : int
        Maximum number of parameters to display.
    """
    stats = summary_stats(samples, param_names)
    flat = flatten_samples(samples)
    n_total, n_dim = flat.shape

    print(f"\nSummary ({n_total:,} draws, {n_dim} parameters):")
    print(f"{'param':>12s} {'mean':>10s} {'std':>10s} "
          f"{'2.5%':>10s} {'50%':>10s} {'97.5%':>10s}")
    print("-" * 65)

    names = stats["param_names"]
    for i in range(min(n_dim, max_params)):
        print(f"{names[i]:>12s} "
              f"{stats['mean'][i]:10.4f} "
              f"{stats['std'][i]:10.4f} "
              f"{stats['quantiles']['2.5%'][i]:10.4f} "
              f"{stats['quantiles']['50.0%'][i]:10.4f} "
              f"{stats['quantiles']['97.5%'][i]:10.4f}")

    if n_dim > max_params:
        print(f"  ... ({n_dim - max_params} more parameters)")
