"""Hyperparameter benchmark: measures mean ESS/sec across three challenging targets.

Reads n_walkers, n_expand, n_shrink from hparam_config.json.
Outputs 'METRIC: <float>' as last line — used by autoresearch verify command.

Targets: ill_conditioned_30, funnel_10, correlated_30
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

n_walkers = int(cfg["n_walkers"])
n_expand  = int(cfg["n_expand"])
n_shrink  = int(cfg["n_shrink"])

print(f"Config: n_walkers={n_walkers}, n_expand={n_expand}, n_shrink={n_shrink}")

config = VariantConfig(
    name="scale_aware_tuned",
    direction="de_mcz",
    width="scale_aware",
    slice_fn="fixed",
    zmatrix="circular",
    ensemble="standard",
    width_kwargs={"scale_factor": 1.0},
    slice_kwargs={"n_expand": n_expand, "n_shrink": n_shrink},
)

TARGETS = [
    ("ill_conditioned_30", ill_conditioned_gaussian(30)),
    ("funnel_10",          neals_funnel(10)),
    ("correlated_30",      correlated_gaussian(30)),
]

N_SAMPLES = 1500
N_WARMUP  = 800

ess_per_sec_list = []

for name, target in TARGETS:
    init = target.sample(jax.random.PRNGKey(42), n_walkers)

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
    ess_s = ess / wall if wall > 0 else 0.0

    ess_per_sec_list.append(ess_s)
    print(f"  {name}: ESS={ess:.1f}, wall={wall:.2f}s, ESS/s={ess_s:.2f}")

mean_ess_per_sec = sum(ess_per_sec_list) / len(ess_per_sec_list)
print(f"METRIC: {mean_ess_per_sec:.4f}")
