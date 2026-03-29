"""Comparison tables from benchmark results.

Produces formatted output showing how variants compare on each target,
with both absolute metrics and relative-to-baseline normalization.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Key metrics for comparison (higher is better for first group, lower for second)
HIGHER_IS_BETTER = {"ess_min", "ess_mean", "ess_per_sec_min", "ess_per_sec_mean",
                     "ess_per_eval", "mean_jump", "sv_ratio", "speed_its"}
LOWER_IS_BETTER = {"zero_move_rate", "cap_hit_rate", "rhat_max", "mcse_max",
                    "bias_max", "rel_bias_max", "wall_time"}


def results_to_table(
    all_metrics: list[dict],
    metric_keys: Optional[list] = None,
    sort_by: str = "ess_per_sec_min",
) -> str:
    """Format benchmark results as a readable comparison table.

    Parameters
    ----------
    all_metrics : list of dicts from runner.run_comparison
    metric_keys : which metrics to show (default: key subset)
    sort_by : metric to sort variants by (within each target)

    Returns
    -------
    Formatted string table.
    """
    if metric_keys is None:
        metric_keys = [
            "ess_min", "ess_per_sec_min", "ess_per_eval",
            "zero_move_rate", "rhat_max", "speed_its", "wall_time",
        ]

    # Filter out errors
    valid = [m for m in all_metrics if "error" not in m]
    if not valid:
        return "No valid results to compare."

    # Group by target
    targets = sorted(set(m["target"] for m in valid))
    variants = sorted(set(m["variant"] for m in valid))

    lines = []
    for target in targets:
        lines.append(f"\n{'='*80}")
        lines.append(f"  TARGET: {target}")
        lines.append(f"{'='*80}")

        target_results = [m for m in valid if m["target"] == target]

        # Average across trials for each variant
        variant_avg = {}
        for variant in variants:
            vr = [m for m in target_results if m["variant"] == variant]
            if not vr:
                continue
            avg = {}
            for k in metric_keys:
                vals = [m[k] for m in vr if k in m]
                avg[k] = np.mean(vals) if vals else float("nan")
            variant_avg[variant] = avg

        if not variant_avg:
            continue

        # Sort by sort_by metric
        reverse = sort_by in HIGHER_IS_BETTER
        sorted_variants = sorted(
            variant_avg.keys(),
            key=lambda v: variant_avg[v].get(sort_by, 0),
            reverse=reverse,
        )

        # Header
        header = f"  {'Variant':<25s}"
        for k in metric_keys:
            header += f" {k:>15s}"
        lines.append(header)
        lines.append("  " + "-" * (25 + 16 * len(metric_keys)))

        # Baseline values for relative comparison
        baseline = variant_avg.get("baseline", {})

        for variant in sorted_variants:
            row = f"  {variant:<25s}"
            for k in metric_keys:
                val = variant_avg[variant].get(k, float("nan"))
                if np.isnan(val):
                    row += f" {'N/A':>15s}"
                elif val < 0.001 and val > 0:
                    row += f" {val:>15.6f}"
                elif abs(val) > 1000:
                    row += f" {val:>15.0f}"
                else:
                    row += f" {val:>15.4f}"
            lines.append(row)

        # Relative to baseline
        if baseline:
            lines.append("")
            lines.append(f"  {'Relative to baseline:':<25s}")
            lines.append("  " + "-" * (25 + 16 * len(metric_keys)))
            for variant in sorted_variants:
                if variant == "baseline":
                    continue
                row = f"  {variant:<25s}"
                for k in metric_keys:
                    val = variant_avg[variant].get(k, float("nan"))
                    base_val = baseline.get(k, float("nan"))
                    if np.isnan(val) or np.isnan(base_val) or abs(base_val) < 1e-30:
                        row += f" {'N/A':>15s}"
                    else:
                        ratio = val / base_val
                        # Show as improvement/regression
                        if k in HIGHER_IS_BETTER:
                            marker = "+" if ratio > 1 else ""
                            pct = (ratio - 1) * 100
                        else:
                            marker = "+" if ratio < 1 else ""
                            pct = (1 - ratio) * 100
                        row += f" {marker}{pct:>13.1f}%"
                lines.append(row)

    return "\n".join(lines)


def find_best_variant(
    all_metrics: list[dict],
    metric: str = "ess_per_sec_min",
) -> dict:
    """Find the best variant for each target by the given metric.

    Returns dict mapping target -> (best_variant, best_value).
    """
    valid = [m for m in all_metrics if "error" not in m and metric in m]
    targets = set(m["target"] for m in valid)

    best = {}
    higher = metric in HIGHER_IS_BETTER

    for target in sorted(targets):
        target_results = [m for m in valid if m["target"] == target]
        if not target_results:
            continue

        # Average across trials
        variants = set(m["variant"] for m in target_results)
        best_val = -np.inf if higher else np.inf
        best_var = None

        for variant in variants:
            vals = [m[metric] for m in target_results if m["variant"] == variant]
            avg = np.mean(vals)
            if (higher and avg > best_val) or (not higher and avg < best_val):
                best_val = avg
                best_var = variant

        best[target] = (best_var, best_val)

    return best


def summary_report(all_metrics: list[dict]) -> str:
    """Generate a concise summary of what works best where.

    Returns formatted string with recommendations.
    """
    lines = ["\n" + "=" * 60, "  SUMMARY: Best Variant per Target", "=" * 60]

    for metric, label in [
        ("ess_per_sec_min", "ESS/sec (min)"),
        ("ess_per_eval", "ESS/eval"),
        ("zero_move_rate", "Zero-move rate"),
    ]:
        best = find_best_variant(all_metrics, metric)
        lines.append(f"\n  By {label}:")
        for target, (variant, value) in sorted(best.items()):
            lines.append(f"    {target:<25s} -> {variant:<25s} ({value:.4f})")

    return "\n".join(lines)
