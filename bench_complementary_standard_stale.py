#!/usr/bin/env python
"""Benchmark: complementary_prob sweep on a STALE-Z scenario for the
STANDARD ensemble (sister of bench_complementary_stale.py which uses block_gibbs).
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

print(f"JAX devices: {jax.devices()}")

# 21D anisotropic Gaussian — same target as bench_complementary_stale.py
NDIM = 21
rng = np.random.default_rng(42)
A = rng.standard_normal((NDIM, NDIM))
Q, _ = np.linalg.qr(A)
evals = np.linspace(1.0, 50.0, NDIM)
cov = Q @ np.diag(evals) @ Q.T
cov = (cov + cov.T) / 2
prec = jnp.array(np.linalg.inv(cov), dtype=jnp.float64)
true_mean = jnp.zeros(NDIM, dtype=jnp.float64)


@jax.jit
def log_prob(x):
    d = x - true_mean
    return -0.5 * d @ prec @ d


def make_init(seed):
    """Walkers initialized 5σ from the mode along the worst-conditioned axis."""
    key = jax.random.PRNGKey(seed)
    bad_axis = jnp.array(Q[:, -1], dtype=jnp.float64)
    offset = 5.0 * jnp.sqrt(evals[-1]) * bad_axis
    scatter = jax.random.normal(key, (32, NDIM), dtype=jnp.float64) * 0.1
    return offset[None, :] + scatter


def make_config(cp):
    ens_kwargs = {}
    if cp > 0.0:
        ens_kwargs["complementary_prob"] = cp
    return VariantConfig(
        name=f"scale_aware_cp{cp}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )


CONFIGS = [
    ("cp=0.00 (pure Z)", 0.0),
    ("cp=0.25 (hybrid)", 0.25),
    ("cp=0.50 (hybrid)", 0.5),
    ("cp=1.00 (pure comp)", 1.0),
]
N_WARMUP = 500
N_PROD = 3000
N_SEEDS = 3

print(f"\n{'=' * 100}")
print(f"  STALE-Z STANDARD-ENSEMBLE: 21D Gaussian, init 5σ from mode")
print(f"  {N_WARMUP} warmup + {N_PROD} production, {N_SEEDS} seeds per config")
print(f"{'=' * 100}")
hdr = (f"  {'Setup':<25s} | {'Seed':>4s} | {'R-hat':>6s} | {'ESS min':>8s} | "
       f"{'ESS mean':>9s} | {'Wall':>6s} | {'mu':>8s} | {'mean_lp':>8s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

results_by_config = {label: [] for label, _ in CONFIGS}
for label, cp in CONFIGS:
    config = make_config(cp)
    for seed in range(N_SEEDS):
        init = make_init(seed)
        t0 = time.time()
        result = run_variant(
            log_prob, init, n_steps=N_WARMUP + N_PROD,
            config=config, n_warmup=N_WARMUP,
            key=jax.random.PRNGKey(seed * 1000), verbose=False,
        )
        wall = time.time() - t0
        samples = np.array(result["samples"])
        ess = compute_ess(samples)
        rhat = compute_rhat(samples)
        final_mu = float(result["mu"])
        last_lps = np.array(result["log_prob"])[-200:]
        mean_lp = float(last_lps.mean())
        results_by_config[label].append({
            "rhat": float(rhat.max()), "ess_min": float(ess.min()),
            "ess_mean": float(ess.mean()), "wall": wall,
            "mu": final_mu, "mean_lp": mean_lp,
        })
        print(f"  {label:<25s} | {seed:>4d} | {float(rhat.max()):6.3f} | "
              f"{float(ess.min()):8.1f} | {float(ess.mean()):9.1f} | "
              f"{wall:5.1f}s | {final_mu:8.4f} | {mean_lp:8.1f}", flush=True)
        gc.collect()

print(f"\n{'=' * 100}")
print(f"  SUMMARY (mean ± std across {N_SEEDS} seeds)")
print(f"{'=' * 100}")
for label, _ in CONFIGS:
    runs = results_by_config[label]
    rhat_mean = np.mean([r["rhat"] for r in runs])
    ess_min_mean = np.mean([r["ess_min"] for r in runs])
    ess_min_std = np.std([r["ess_min"] for r in runs])
    print(f"  {label:<25s} | R-hat {rhat_mean:6.3f} | "
          f"ESS_min {ess_min_mean:6.1f} ± {ess_min_std:5.1f}", flush=True)
