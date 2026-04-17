#!/usr/bin/env python
"""Benchmark: pure Z-matrix vs hybrid (Z+complementary) for bg_MH+DR.

Sweeps complementary_prob over {0.0, 0.25, 0.5, 0.75, 1.0} on the
block-coupled 63D Gaussian target.
"""
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
from dezess.targets_stream import block_coupled_gaussian

print(f"JAX devices: {jax.devices()}")

target = block_coupled_gaussian()
NDIM = target.ndim
N_WARMUP = 2000
N_PROD = 5000


def make_config(complementary_prob):
    return VariantConfig(
        name=f"bg_mh_dr_cp{complementary_prob}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14, 14, 14, 14],
            "use_mh": True,
            "delayed_rejection": True,
            "complementary_prob": complementary_prob,
        },
    )


configs = [
    ("Pure Z-matrix (cp=0.0)", 0.0),
    ("Hybrid 25/75 (cp=0.25)", 0.25),
    ("Hybrid 50/50 (cp=0.5)", 0.5),
    ("Hybrid 75/25 (cp=0.75)", 0.75),
    ("Pure complementary (cp=1.0)", 1.0),
]

key = jax.random.PRNGKey(42)
init = target.sample(key, 64)

hdr = (f"  {'Setup':<35s} | {'R-hat':>6s} | {'ESS min':>8s} | "
       f"{'ESS mean':>9s} | {'Wall':>6s} | {'ESS/sec':>8s}")
print(f"\n{hdr}")
print("  " + "-" * (len(hdr) - 2))

for label, cp in configs:
    config = make_config(cp)
    t0 = time.time()
    result = run_variant(
        target.log_prob, init, n_steps=N_WARMUP + N_PROD,
        config=config, n_warmup=N_WARMUP, verbose=False,
    )
    wall = time.time() - t0
    samples = np.array(result["samples"])
    ess = compute_ess(samples)
    rhat = compute_rhat(samples)
    print(f"  {label:<35s} | {float(rhat.max()):6.3f} | "
          f"{float(ess.min()):8.1f} | {float(ess.mean()):9.1f} | "
          f"{wall:5.1f}s | {float(ess.min())/wall:8.2f}",
          flush=True)
    gc.collect()
