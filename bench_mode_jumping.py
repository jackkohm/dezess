#!/usr/bin/env python
"""Benchmark: mode-jumping on bimodal Gaussian mixture."""
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
from dezess.targets import gaussian_mixture

print(f"JAX devices: {jax.devices()}")

target = gaussian_mixture(ndim=10, separation=8.0)

configs = {
    "default (fixed slice)": VariantConfig(
        name="scale_aware", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    ),
    "adaptive slice": VariantConfig(
        name="adaptive", direction="de_mcz", width="scale_aware",
        slice_fn="adaptive", zmatrix="circular", ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        slice_kwargs={"n_expand": 50, "n_shrink": 50},
    ),
    "global_move + adaptive": VariantConfig(
        name="global_move", direction="global_move", width="scale_aware",
        slice_fn="adaptive", zmatrix="circular", ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        direction_kwargs={"n_components": 2, "global_prob": 0.1},
        slice_kwargs={"n_expand": 50, "n_shrink": 50},
    ),
}

key = jax.random.PRNGKey(42)
init = target.sample(key, 64)

hdr = f"{'Strategy':<30s} | {'R-hat':>6s} | {'ESS min':>8s} | {'Wall':>6s} | {'Left %':>7s} | {'Right %':>7s}"
print(f"\n{hdr}")
print("-" * len(hdr))

for name, config in configs.items():
    t0 = time.time()
    result = run_variant(target.log_prob, init, n_steps=7000, config=config,
                         n_warmup=2000, verbose=False)
    wall = time.time() - t0
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 10)
    ess = compute_ess(samples)
    rhat = compute_rhat(samples)
    frac_left = float((flat[:, 0] < 0).mean())
    frac_right = float((flat[:, 0] > 0).mean())
    print(f"{name:<30s} | {float(rhat.max()):6.3f} | {float(ess.min()):8.1f} | "
          f"{wall:5.1f}s | {frac_left:6.1%} | {frac_right:6.1%}", flush=True)
    gc.collect()
