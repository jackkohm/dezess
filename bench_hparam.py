"""Hyperparameter benchmark: hard-regime efficiency.

Metric = ESS_min / (n_walkers * N_SAMPLES * evals_per_step)

Same ESS-per-total-eval metric as before, but on harder targets where
n_dim is comparable to or larger than n_walkers.

Targets (hard — n_dim >= n_walkers):
  - ill_cond_100: 100D, condition=1000  (empirical cond ~1575 with 128W)
  - corr_100:     100D, condition=50    (empirical cond ~77  with 128W)
  - mix_20:       20D Gaussian mixture  (empirical cond ~22  with 128W)

These test the regime where the sampler must work much harder than the
previous 60D targets with 256 walkers.

Outputs 'METRIC: <float>' as last line for autoresearch verify command.
"""
import json
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.targets import ill_conditioned_gaussian, correlated_gaussian, gaussian_mixture

# --- Load config ---
with open("hparam_config.json") as f:
    cfg = json.load(f)

n_walkers        = int(cfg["n_walkers"])
n_expand         = int(cfg["n_expand"])
n_shrink         = int(cfg["n_shrink"])
n_slices         = int(cfg.get("n_slices_per_step", 1))
N_WARMUP_CFG     = int(cfg.get("n_warmup", 1500))
direction        = cfg.get("direction", "de_mcz")
direction_kwargs = cfg.get("direction_kwargs", {})
width            = cfg.get("width", "scale_aware")
width_kwargs     = cfg.get("width_kwargs", {"scale_factor": 1.0})
slice_fn         = cfg.get("slice_fn", "fixed")
slice_kwargs     = cfg.get("slice_kwargs", {})

if slice_fn == "fixed":
    slice_kwargs = dict(slice_kwargs)
    slice_kwargs.setdefault("n_expand", n_expand)
    slice_kwargs.setdefault("n_shrink", n_shrink)
elif slice_fn == "mh_multi":
    # Ensure slice_kwargs has n_expand so evals_per_step counts correctly.
    # The loop defaults n_expand=3 for cov_aware width (vs 2 for scale_aware),
    # so we must explicitly set it here to match the config file value.
    slice_kwargs = dict(slice_kwargs)
    slice_kwargs.setdefault("n_expand", n_expand)

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

TARGETS = [
    ("ill_cond_100",  ill_conditioned_gaussian(100, condition=1000.0)),
    ("corr_100",      correlated_gaussian(100)),
    ("mix_20",        gaussian_mixture(20)),
]

N_SAMPLES = 2000
N_WARMUP  = N_WARMUP_CFG

if slice_fn == "nurs":
    evals_per_step = n_slices * (2 ** n_expand + 1)
elif slice_fn == "mh":
    evals_per_step = n_slices * 1
elif slice_fn == "mh_multi":
    evals_per_step = n_slices * n_expand
elif slice_fn == "mh_adaptive":
    evals_per_step = n_slices * 1
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
        rhat = float(streaming.get("rhat_max", float("inf")))
        ess_per_eval = ess / total_evals if total_evals > 0 else 0.0

        if rhat > 1.1:
            print(f"  {name}: ESS={ess:.1f}, R-hat={rhat:.4f} FAIL, ESS/eval=0")
            metric_values.append(0.0)
        else:
            metric_values.append(ess_per_eval)
            print(f"  {name}: ESS={ess:.1f}, R-hat={rhat:.4f}, ESS/eval={ess_per_eval:.6f}, wall={wall:.2f}s")

    except Exception as e:
        print(f"  {name}: FAILED — {e}")
        metric_values.append(0.0)

mean_metric = sum(metric_values) / len(metric_values) if metric_values else 0.0
print(f"METRIC: {mean_metric:.6f}")
