"""Hyperparameter benchmark: measures ESS per total log-prob eval.

Metric = ESS_min / (n_walkers * N_SAMPLES * evals_per_step)

This is pure sampler efficiency: information gained per likelihood call,
independent of parallelism. n_walkers appears in the denominator so adding
walkers only wins if ESS scales proportionally.

Targets: ill_conditioned_60, funnel_30, correlated_60
(high-dimensional, hard — tests regime where n_walkers << n_dim)

Outputs 'METRIC: <float>' as last line for autoresearch verify command.
"""
import json
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.targets import ill_conditioned_gaussian, neals_funnel, correlated_gaussian

# --- Load config ---
with open("hparam_config.json") as f:
    cfg = json.load(f)

n_walkers       = int(cfg["n_walkers"])
n_expand        = int(cfg["n_expand"])
n_shrink        = int(cfg["n_shrink"])
n_slices        = int(cfg.get("n_slices_per_step", 1))
direction       = cfg.get("direction", "de_mcz")
direction_kwargs = cfg.get("direction_kwargs", {})
width           = cfg.get("width", "scale_aware")
width_kwargs    = cfg.get("width_kwargs", {"scale_factor": 1.0})
slice_fn        = cfg.get("slice_fn", "fixed")
slice_kwargs    = cfg.get("slice_kwargs", {})

# Inject n_expand/n_shrink into slice_kwargs so they're respected
if slice_fn == "fixed":
    slice_kwargs = dict(slice_kwargs)
    slice_kwargs.setdefault("n_expand", n_expand)
    slice_kwargs.setdefault("n_shrink", n_shrink)

print(f"Config: n_walkers={n_walkers}, direction={direction}, width={width}, "
      f"slice={slice_fn}, n_expand={n_expand}, n_shrink={n_shrink}, "
      f"n_slices={n_slices}")

config = VariantConfig(
    name="autoresearch_candidate",
    direction=direction,
    width=width,
    slice_fn=slice_fn,
    zmatrix="circular",
    ensemble="standard",
    n_slices_per_step=n_slices,
    direction_kwargs=direction_kwargs,
    width_kwargs=width_kwargs,
    slice_kwargs=slice_kwargs,
)

# Hard targets: high-dimensional, n_walkers << n_dim
TARGETS = [
    ("ill_conditioned_60", ill_conditioned_gaussian(60, condition=1000.0)),
    ("funnel_30",          neals_funnel(30)),
    ("correlated_60",      correlated_gaussian(60)),
]

N_SAMPLES = 2000
N_WARMUP  = 1500

# Evals per walker per step (times n_slices for multi-direction)
# fixed/early_stop: 2*n_expand (expand checks both L+R) + n_shrink (shrink proposals)
# nurs:             2^n_expand + 1  (orbit of size 2^n_expand + 1 shift-proposal eval)
if slice_fn == "nurs":
    evals_per_step = n_slices * (2 ** n_expand + 1)
else:
    evals_per_step = n_slices * (2 * n_expand + n_shrink)

metric_values = []

for name, target in TARGETS:
    init = target.sample(jax.random.PRNGKey(42), n_walkers)

    try:
        result = run_variant(
            target.log_prob,
            init,
            n_steps=N_SAMPLES + N_WARMUP,
            config=config,
            n_warmup=N_WARMUP,
            key=jax.random.PRNGKey(0),
            mu=1.0,
            tune=True,
            verbose=False,
        )

        streaming = result["diagnostics"].get("streaming", {})
        ess  = float(streaming.get("ess_min", 0.0))
        wall = float(result["wall_time"])

        total_evals = n_walkers * N_SAMPLES * evals_per_step
        ess_per_eval = ess / total_evals if total_evals > 0 else 0.0

        metric_values.append(ess_per_eval)
        print(f"  {name}: ESS={ess:.1f}, ESS/eval={ess_per_eval:.6f}, wall={wall:.2f}s")

    except Exception as e:
        print(f"  {name}: FAILED — {e}")
        metric_values.append(0.0)

mean_metric = sum(metric_values) / len(metric_values) if metric_values else 0.0
print(f"METRIC: {mean_metric:.6f}")
