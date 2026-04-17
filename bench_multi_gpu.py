#!/usr/bin/env python
"""Benchmark: single-GPU vs multi-GPU (sharded ensemble) on bg_MH+DR.

Compares total ESS and ESS/sec for 64 walkers (single GPU) vs
128/256 walkers across 2/4 GPUs (sharded ensemble).
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
n_devices = len(jax.devices())
print(f"Detected {n_devices} devices")

target = block_coupled_gaussian()
NDIM = target.ndim
N_WARMUP = 2000
N_PROD = 5000


def make_config(block_sizes):
    return VariantConfig(
        name="bg_mh_dr",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": block_sizes, "use_mh": True,
                         "delayed_rejection": True},
    )


configs = [
    ("1 GPU x 64 walkers", 1, 64),
    ("2 GPUs x 64 walkers (128 total)", 2, 64),
    ("4 GPUs x 64 walkers (256 total)", 4, 64),
]

key = jax.random.PRNGKey(42)

hdr = (f"  {'Setup':<40s} | {'R-hat':>6s} | {'ESS min':>8s} | "
       f"{'ESS mean':>9s} | {'Wall':>6s} | {'ESS/sec':>8s}")
print(f"\n{hdr}")
print("  " + "-" * (len(hdr) - 2))

for label, n_gpus, n_walkers_per_gpu in configs:
    if n_gpus > n_devices:
        print(f"  {label:<40s} | SKIP (only {n_devices} devices)")
        continue
    n_total = n_gpus * n_walkers_per_gpu
    init = target.sample(key, n_total)
    config = make_config([7, 14, 14, 14, 14])

    t0 = time.time()
    result = run_variant(
        target.log_prob, init, n_steps=N_WARMUP + N_PROD,
        config=config, n_warmup=N_WARMUP, verbose=False,
        n_gpus=n_gpus, n_walkers_per_gpu=n_walkers_per_gpu,
    )
    wall = time.time() - t0
    samples = np.array(result["samples"])
    ess = compute_ess(samples)
    rhat = compute_rhat(samples)
    print(f"  {label:<40s} | {float(rhat.max()):6.3f} | "
          f"{float(ess.min()):8.1f} | {float(ess.mean()):9.1f} | "
          f"{wall:5.1f}s | {float(ess.min())/wall:8.2f}",
          flush=True)
    gc.collect()
