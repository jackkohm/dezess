"""ArviZ integration for dezess.

Converts dezess samples to ArviZ InferenceData objects for standard
Bayesian diagnostics, trace plots, and posterior analysis.

Usage:
    result = dezess.sample(log_prob, init)
    idata = dezess.to_arviz(result)
    az.plot_trace(idata)
    az.summary(idata)

Requires: pip install arviz
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def to_inference_data(
    result,
    param_names: Optional[list] = None,
    coords: Optional[dict] = None,
    dims: Optional[dict] = None,
):
    """Convert dezess result to ArviZ InferenceData.

    Parameters
    ----------
    result : SampleResult or dict
        Output from dezess.sample() or run_variant().
    param_names : list of str or None
        Names for each dimension. Auto-generated if None.
    coords : dict or None
        ArviZ coordinates for labeled dimensions.
    dims : dict or None
        ArviZ dimension mapping.

    Returns
    -------
    arviz.InferenceData
    """
    try:
        import arviz as az
    except ImportError:
        raise ImportError(
            "arviz is required for to_inference_data(). "
            "Install with: pip install arviz"
        )

    # Extract samples array
    if hasattr(result, "samples"):
        # SampleResult named tuple
        samples = np.asarray(result.samples)
        log_prob = np.asarray(result.log_prob)
    elif isinstance(result, dict):
        samples = np.asarray(result["samples"])
        log_prob = np.asarray(result.get("log_prob", result.get("log_probs")))
    else:
        raise TypeError(f"Unexpected result type: {type(result)}")

    # samples shape: (n_steps, n_walkers, n_dim)
    n_steps, n_walkers, n_dim = samples.shape

    if param_names is None:
        param_names = [f"x{i}" for i in range(n_dim)]

    # ArviZ expects: dict of {param_name: (n_chains, n_draws)} arrays
    # We treat each walker as a chain
    posterior_dict = {}
    for i, name in enumerate(param_names):
        # (n_walkers, n_steps) — ArviZ convention: (chains, draws)
        posterior_dict[name] = samples[:, :, i].T  # (n_walkers, n_steps)

    # Log-likelihood (ArviZ uses this for LOO-CV etc.)
    sample_stats = {
        "lp": log_prob.T,  # (n_walkers, n_steps)
    }

    return az.from_dict(
        posterior=posterior_dict,
        sample_stats=sample_stats,
        coords=coords,
        dims=dims,
    )
