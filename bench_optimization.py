#!/usr/bin/env python
"""Benchmark JAX optimization pass: wall time + correctness."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time, gc
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.benchmark.metrics import compute_ess, compute_rhat
from dezess.targets_stream import funnel_63d, block_coupled_gaussian

print(f"JAX devices: {jax.devices()}")

configs = {
    "bg_MH+DR (safe)": VariantConfig(
        name="bg_mh_dr", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True,
                         "delayed_rejection": True},
    ),
    "bg_MH+DR (no NaN check)": VariantConfig(
        name="bg_mh_dr_unsafe", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True,
                         "delayed_rejection": True},
    ),
    "default (scale_aware)": VariantConfig(
        name="scale_aware", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    ),
    "default (no NaN check)": VariantConfig(
        name="scale_aware_unsafe", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
    ),
}

targets = [
    ("Funnel 63D", funnel_63d()),
    ("Block-Coupled 63D", block_coupled_gaussian()),
]


def run_one(cfg, target, n_walkers=64, n_warmup=2000, n_prod=5000):
    key = jax.random.PRNGKey(42)
    init = target.sample(key, n_walkers) if target.sample else jax.random.normal(key, (n_walkers, target.ndim)) * 0.1
    t0 = time.time()
    result = run_variant(target.log_prob, init, n_steps=n_warmup + n_prod,
                         config=cfg, n_warmup=n_warmup, verbose=False)
    wall = time.time() - t0
    samples = np.array(result["samples"])
    ess = compute_ess(samples)
    rhat = compute_rhat(samples)
    return {"rhat": float(rhat.max()), "ess_min": float(ess.min()),
            "ess_mean": float(ess.mean()), "wall": wall,
            "it_per_sec": n_prod / wall}


hdr = f"{'Target':<22s} | {'Config':<25s} | {'R-hat':>6s} | {'ESS min':>8s} | {'Wall':>6s} | {'it/s':>8s}"
print(f"\n{hdr}")
print("-" * len(hdr))

for tname, target in targets:
    for label, cfg in configs.items():
        r = run_one(cfg, target)
        print(f"{tname:<22s} | {label:<25s} | {r['rhat']:6.3f} | "
              f"{r['ess_min']:8.1f} | {r['wall']:5.1f}s | {r['it_per_sec']:7.0f}",
              flush=True)
        gc.collect()
    print()
