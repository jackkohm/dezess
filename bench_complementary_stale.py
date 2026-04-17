#!/usr/bin/env python
"""Benchmark: complementary_prob sweep on a STALE-Z scenario.

Mimics the user's real Sanders setup:
- 21D correlated Gaussian (matches 1-stream Sanders dim)
- Block structure [7, 14] (potential + nuisance)
- Walkers initialized FAR from mode (mimics CMA-ES climb)
- Short warmup (forces biased Z-matrix to be frozen)

Multiple seeds per config to bound variance.
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

# ── Build target: 21D anisotropic Gaussian (mimics 1-stream Sanders) ────

NDIM = 21
rng = np.random.default_rng(42)

# Strong anisotropy: condition number ~50 (typical for stream posteriors)
A = rng.standard_normal((NDIM, NDIM))
Q, _ = np.linalg.qr(A)
evals = np.linspace(1.0, 50.0, NDIM)
cov = Q @ np.diag(evals) @ Q.T
cov = (cov + cov.T) / 2
prec = jnp.array(np.linalg.inv(cov), dtype=jnp.float64)
L = np.linalg.cholesky(cov)
true_mean = jnp.zeros(NDIM, dtype=jnp.float64)


@jax.jit
def log_prob(x):
    d = x - true_mean
    return -0.5 * d @ prec @ d


# ── Initialization: far from mode (mimics CMA-ES scatter into "climb") ──

def make_init(seed):
    """Walkers initialized FAR from posterior mode (5σ along principal axes)."""
    key = jax.random.PRNGKey(seed)
    # Initialize at offset = 5*sqrt(max_evals) along the worst-conditioned direction
    bad_axis = jnp.array(Q[:, -1], dtype=jnp.float64)  # most stretched direction
    offset = 5.0 * jnp.sqrt(evals[-1]) * bad_axis
    # Tiny scatter around offset (mimics CMA scatter at end of optimization)
    scatter = jax.random.normal(key, (32, NDIM), dtype=jnp.float64) * 0.1
    return offset[None, :] + scatter


# ── Configs ─────────────────────────────────────────────────────────────

def make_config(complementary_prob):
    return VariantConfig(
        name=f"bg_mh_dr_cp{complementary_prob}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14],   # matches user's 1-stream Sanders
            "use_mh": True,
            "delayed_rejection": True,
            "complementary_prob": complementary_prob,
        },
    )


CONFIGS = [
    ("cp=0.00 (pure Z)", 0.0),
    ("cp=0.25 (hybrid)", 0.25),
    ("cp=0.50 (hybrid)", 0.5),
    ("cp=1.00 (pure comp)", 1.0),
]

N_WARMUP = 500    # SHORT warmup — forces biased Z-matrix to be frozen
N_PROD = 3000
N_SEEDS = 3       # 3 seeds per config to bound variance


# ── Run sweep ───────────────────────────────────────────────────────────

print(f"\n{'=' * 100}")
print(f"  STALE-Z SCENARIO: 21D correlated Gaussian, init 5σ from mode, blocks=[7,14]")
print(f"  {N_WARMUP} warmup + {N_PROD} production, {N_SEEDS} seeds per config")
print(f"  Target: condition number ~50, walkers must 'climb' to mode")
print(f"{'=' * 100}")
hdr = (f"  {'Setup':<25s} | {'Seed':>4s} | {'R-hat':>6s} | {'ESS min':>8s} | "
       f"{'ESS mean':>9s} | {'Wall':>6s} | {'mu':>8s} | {'mean_lp':>8s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

# Collect for summary
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
        flat = samples.reshape(-1, NDIM)
        ess = compute_ess(samples)
        rhat = compute_rhat(samples)
        final_mu = float(result["mu"])
        # Mean log_prob in last 200 steps (closer to 0 = closer to mode)
        last_lps = np.array(result["log_prob"])[-200:]
        mean_lp = float(last_lps.mean())

        results_by_config[label].append({
            "rhat": float(rhat.max()),
            "ess_min": float(ess.min()),
            "ess_mean": float(ess.mean()),
            "wall": wall,
            "mu": final_mu,
            "mean_lp": mean_lp,
        })
        print(f"  {label:<25s} | {seed:>4d} | {float(rhat.max()):6.3f} | "
              f"{float(ess.min()):8.1f} | {float(ess.mean()):9.1f} | "
              f"{wall:5.1f}s | {final_mu:8.4f} | {mean_lp:8.1f}",
              flush=True)
        gc.collect()

# ── Summary: mean ± std across seeds ────────────────────────────────────

print(f"\n{'=' * 100}")
print(f"  SUMMARY (mean ± std across {N_SEEDS} seeds)")
print(f"{'=' * 100}")
hdr2 = (f"  {'Setup':<25s} | {'R-hat (mean)':>13s} | {'ESS min (mean ± std)':>22s} | "
        f"{'mu':>10s} | {'mean_lp':>10s}")
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))

for label, _ in CONFIGS:
    runs = results_by_config[label]
    rhat_mean = np.mean([r["rhat"] for r in runs])
    ess_min_mean = np.mean([r["ess_min"] for r in runs])
    ess_min_std = np.std([r["ess_min"] for r in runs])
    mu_mean = np.mean([r["mu"] for r in runs])
    lp_mean = np.mean([r["mean_lp"] for r in runs])
    print(f"  {label:<25s} | {rhat_mean:13.3f} | {ess_min_mean:8.1f} ± {ess_min_std:6.1f}        | "
          f"{mu_mean:10.4f} | {lp_mean:10.1f}", flush=True)

print()
print(f"  Interpretation:")
print(f"    - R-hat near 1.0 = converged. R-hat > 1.5 = biased Z hurt convergence.")
print(f"    - ESS_min higher = more effective samples in the slowest direction.")
print(f"    - mu_final near 1.0 = healthy tuning. mu << 1 = biased Z forced shrinkage.")
print(f"    - mean_lp closer to 0 = closer to true mode (true posterior peaks at lp~-10.5).")
