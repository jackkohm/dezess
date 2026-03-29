"""Visualization utilities for benchmark results.

Generates plots comparing variant performance across targets.
Uses matplotlib if available, falls back to text-based output.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _check_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def heatmap(
    all_metrics: list[dict],
    metric: str = "ess_per_sec_min",
    output_path: str = "benchmark_heatmap.png",
    title: Optional[str] = None,
) -> Optional[str]:
    """Create a heatmap of metric values across variants and targets.

    Returns the output path if matplotlib is available, None otherwise.
    """
    plt = _check_matplotlib()
    if plt is None:
        print("matplotlib not available; skipping heatmap")
        return None

    valid = [m for m in all_metrics if "error" not in m and metric in m]
    targets = sorted(set(m["target"] for m in valid))
    variants = sorted(set(m["variant"] for m in valid))

    # Build matrix
    data = np.full((len(variants), len(targets)), np.nan)
    for i, variant in enumerate(variants):
        for j, target in enumerate(targets):
            vals = [m[metric] for m in valid
                    if m["variant"] == variant and m["target"] == target]
            if vals:
                data[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(8, len(targets) * 1.5), max(6, len(variants) * 0.5)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=8)

    # Annotate cells
    for i in range(len(variants)):
        for j in range(len(targets)):
            val = data[i, j]
            if not np.isnan(val):
                text = f"{val:.2f}" if val < 100 else f"{val:.0f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, label=metric)
    ax.set_title(title or f"Benchmark: {metric}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Heatmap saved to {output_path}")
    return output_path


def convergence_plot(
    result: dict,
    output_path: str = "convergence.png",
    dims: Optional[list] = None,
) -> Optional[str]:
    """Plot trace and running mean for selected dimensions."""
    plt = _check_matplotlib()
    if plt is None:
        return None

    samples = np.asarray(result["samples"])
    n_steps, n_walkers, ndim = samples.shape

    if dims is None:
        dims = list(range(min(4, ndim)))

    fig, axes = plt.subplots(len(dims), 2, figsize=(12, 3 * len(dims)))
    if len(dims) == 1:
        axes = axes[None, :]

    for i, d in enumerate(dims):
        # Trace plot (first 4 walkers)
        for w in range(min(4, n_walkers)):
            axes[i, 0].plot(samples[:, w, d], alpha=0.5, linewidth=0.5)
        axes[i, 0].set_ylabel(f"dim {d}")
        axes[i, 0].set_title("Trace" if i == 0 else "")

        # Running mean
        mean_chain = samples[:, :, d].mean(axis=1)
        running_mean = np.cumsum(mean_chain) / (np.arange(n_steps) + 1)
        axes[i, 1].plot(running_mean)
        axes[i, 1].set_title("Running mean" if i == 0 else "")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def scaling_plot(
    all_metrics: list[dict],
    metric: str = "ess_per_sec_min",
    output_path: str = "scaling.png",
) -> Optional[str]:
    """Plot metric vs dimension for each variant."""
    plt = _check_matplotlib()
    if plt is None:
        return None

    valid = [m for m in all_metrics if "error" not in m and metric in m]
    variants = sorted(set(m["variant"] for m in valid))

    fig, ax = plt.subplots(figsize=(10, 6))

    for variant in variants:
        vr = [m for m in valid if m["variant"] == variant]
        dims = sorted(set(m["ndim"] for m in vr))
        vals = []
        for d in dims:
            dv = [m[metric] for m in vr if m["ndim"] == d]
            vals.append(np.mean(dv))
        ax.plot(dims, vals, "o-", label=variant, markersize=4)

    ax.set_xlabel("Dimension")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs Dimension")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
