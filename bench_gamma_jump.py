#!/usr/bin/env python
"""Benchmark: does gamma_jump help bg_MH+DR find modes on a multimodal target?

Same target as bench_multimodal.py: 21D 3-mode mixture at ±6σ, init at mode 0.
Sweeps gamma_jump_prob ∈ {0.0, 0.05, 0.1, 0.3} for bg_MH+DR + cp=0.5.
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

print(f"JAX devices: {jax.devices()}")

NDIM = 21
N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 5000
N_SEEDS = 3
SEPARATION = 6.0

MODE_CENTERS = jnp.array([
    [+SEPARATION] + [0.0] * (NDIM - 1),
    [-SEPARATION] + [0.0] * (NDIM - 1),
    [0.0, +SEPARATION] + [0.0] * (NDIM - 2),
], dtype=jnp.float64)
LOG_WEIGHTS = jnp.full((3,), jnp.log(1.0 / 3.0), dtype=jnp.float64)


@jax.jit
def log_prob(x):
    diffs = x[None, :] - MODE_CENTERS
    log_comps = LOG_WEIGHTS - 0.5 * jnp.sum(diffs ** 2, axis=1)
    return jax.scipy.special.logsumexp(log_comps)


def make_init(seed):
    """All walkers near mode 0 — chain must DISCOVER the others."""
    key = jax.random.PRNGKey(seed)
    return MODE_CENTERS[0][None, :] + 0.3 * jax.random.normal(
        key, (N_WALKERS, NDIM), dtype=jnp.float64
    )


def assign_modes(samples_flat):
    centers_np = np.array(MODE_CENTERS)
    d2 = ((samples_flat[:, None, :] - centers_np[None, :, :]) ** 2).sum(-1)
    nearest = np.argmin(d2, axis=1)
    return np.array([float((nearest == k).mean()) for k in range(3)])


def make_cfg(gj):
    return VariantConfig(
        name=f"bg_mh_dr_cp0.5_gj{gj}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14],
            "use_mh": True,
            "delayed_rejection": True,
            "complementary_prob": 0.5,
            **({"gamma_jump_prob": gj} if gj > 0.0 else {}),
        },
    )


CONFIGS = [
    ("gj=0.00", 0.00),
    ("gj=0.05", 0.05),
    ("gj=0.10", 0.10),
    ("gj=0.20", 0.20),
]

UNIFORM = 1.0 / 3.0
print(f"\n{'=' * 100}")
print(f"  GAMMA-JUMP MULTIMODAL: 21D, 3 modes at ±{SEPARATION}σ, init at mode 0")
print(f"  bg_MH+DR + cp=0.5, sweeping gamma_jump_prob ∈ {{0, 0.05, 0.1, 0.3}}")
print(f"  {N_WALKERS} walkers, {N_WARMUP} warmup + {N_PROD} prod, {N_SEEDS} seeds")
print(f"{'=' * 100}")
hdr = (f"  {'Setup':<10s} | {'seed':>4s} | {'mode_0':>6s} | {'mode_1':>6s} | "
       f"{'mode_2':>6s} | {'cov_err':>7s} | {'mean_lp':>7s} | {'wall':>6s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

results = {label: [] for label, _ in CONFIGS}
for label, gj in CONFIGS:
    cfg = make_cfg(gj)
    for seed in range(N_SEEDS):
        init = make_init(seed)
        t0 = time.time()
        result = run_variant(
            log_prob, init, n_steps=N_WARMUP + N_PROD,
            config=cfg, n_warmup=N_WARMUP,
            key=jax.random.PRNGKey(seed * 1000), verbose=False,
        )
        wall = time.time() - t0
        samples = np.array(result["samples"]).reshape(-1, NDIM)
        fracs = assign_modes(samples)
        cov_err = float(np.max(np.abs(fracs - UNIFORM)))
        mean_lp = float(np.array(result["log_prob"])[-500:].mean())
        results[label].append({"fracs": fracs, "cov_err": cov_err, "mean_lp": mean_lp, "wall": wall})
        print(f"  {label:<10s} | {seed:>4d} | {fracs[0]:6.3f} | {fracs[1]:6.3f} | "
              f"{fracs[2]:6.3f} | {cov_err:7.3f} | {mean_lp:7.2f} | {wall:5.1f}s",
              flush=True)
        gc.collect()

print(f"\n{'=' * 100}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"  cov_err = max|frac[k] - 1/3| ; 0 = uniform across modes ; 0.667 = stuck in 1 mode")
print(f"{'=' * 100}")
hdr2 = (f"  {'Setup':<10s} | {'mode_0':>6s} | {'mode_1':>6s} | {'mode_2':>6s} | "
        f"{'cov_err':>7s} | {'mean_lp':>8s} | {'wall':>7s}")
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))
for label, _ in CONFIGS:
    runs = results[label]
    fracs_m = np.mean([r["fracs"] for r in runs], axis=0)
    cov = np.mean([r["cov_err"] for r in runs])
    lp = np.mean([r["mean_lp"] for r in runs])
    w = np.mean([r["wall"] for r in runs])
    print(f"  {label:<10s} | {fracs_m[0]:6.3f} | {fracs_m[1]:6.3f} | "
          f"{fracs_m[2]:6.3f} | {cov:7.3f} | {lp:8.2f} | {w:6.1f}s", flush=True)
