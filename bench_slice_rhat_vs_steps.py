#!/usr/bin/env python
"""Slice cp=0 long run: measure split-R-hat and ESS at checkpoints up to 50k steps.

Same target/init as bench_slice_vs_mh_whitened.py (51D isotropic unit Gaussian,
64 walkers, sigma=1.0 init). One config (the default), one seed, long run.

Reports at each checkpoint: max R-hat across dims, min R-hat across dims,
median R-hat, ESS_min, wall-time. Tells us when slice cp=0 actually reaches
R-hat <= 1.05 and <= 1.01 on this target.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.benchmark.metrics import compute_ess

print(f"JAX devices: {jax.devices()}")

NDIM = 51
N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 50000
CHECKPOINTS = [1000, 2000, 5000, 10000, 20000, 50000]
SEED = 0


@jax.jit
def log_prob(x):
    return -0.5 * jnp.sum(x * x)


def make_init(seed):
    return jax.random.normal(jax.random.PRNGKey(seed), (N_WALKERS, NDIM), dtype=jnp.float64)


cfg = VariantConfig(
    name="slice_cp0_long",
    direction="de_mcz", width="scale_aware",
    slice_fn="fixed", zmatrix="circular", ensemble="standard",
    check_nans=False, width_kwargs={"scale_factor": 1.0},
    ensemble_kwargs={},
)


def split_rhat(samples_3d):
    """Standard split-R-hat across walkers (each walker split in half).

    samples_3d: (n_steps, n_walkers, n_dim) — production samples only.
    Returns rhat per dim, shape (n_dim,).
    """
    n_steps, n_walkers, n_dim = samples_3d.shape
    half = n_steps // 2
    # Stack: (2*n_walkers, half, n_dim)
    chains = np.stack([samples_3d[:half], samples_3d[half:2*half]], axis=0)
    chains = chains.reshape(2 * n_walkers, half, n_dim)
    chain_means = chains.mean(axis=1)                  # (2N, ndim)
    chain_vars = chains.var(axis=1, ddof=1)            # (2N, ndim)
    W = chain_vars.mean(axis=0)                        # within-chain var (ndim,)
    B = half * chain_means.var(axis=0, ddof=1)         # between-chain var * n (ndim,)
    V_hat = (half - 1) / half * W + B / half
    rhat = np.sqrt(np.maximum(V_hat / np.maximum(W, 1e-30), 0.0))
    return rhat


init = make_init(SEED)
print(f"\n{'=' * 100}")
print(f"  slice cp=0 LONG RUN — {NDIM}D isotropic unit Gaussian, well-init (sigma=1.0)")
print(f"  {N_WALKERS} walkers, {N_WARMUP} warmup + {N_PROD} prod, seed={SEED}")
print(f"{'=' * 100}\n", flush=True)

t0 = time.time()
result = run_variant(
    log_prob, init,
    n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
    config=cfg, key=jax.random.PRNGKey(SEED * 1000), verbose=False,
)
wall_total = time.time() - t0
samples_prod = np.array(result["samples"])    # (N_PROD, N_WALKERS, NDIM)
print(f"Total run: {wall_total:.1f}s ({N_PROD} prod steps × {N_WALKERS} walkers, "
      f"{N_PROD/wall_total:.0f} steps/s)\n", flush=True)

hdr = (f"  {'n_prod':>7s} | {'wall':>7s} | "
       f"{'Rhat_max':>9s} | {'Rhat_med':>9s} | {'Rhat_min':>9s} | "
       f"{'ESS_min':>8s} | {'ESS_med':>8s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for n in CHECKPOINTS:
    if n > N_PROD:
        continue
    sub = samples_prod[:n]
    rhat = split_rhat(sub)
    ess = compute_ess(sub)
    wall_proportional = wall_total * (n / N_PROD)
    print(f"  {n:>7d} | {wall_proportional:>6.1f}s | "
          f"{float(rhat.max()):>9.4f} | {float(np.median(rhat)):>9.4f} | "
          f"{float(rhat.min()):>9.4f} | {float(ess.min()):>8.1f} | "
          f"{float(np.median(ess)):>8.1f}", flush=True)

print(f"\n{'=' * 100}")
print(f"  TARGETS:  R-hat ≤ 1.05 (loose) / 1.01 (strict)")
print(f"  ESS targets:  ≥ 400 (strict) / ≥ 100 (loose) per worst dim")
print(f"{'=' * 100}", flush=True)
