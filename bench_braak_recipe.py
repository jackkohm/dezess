#!/usr/bin/env python
"""Head-to-head: Braak (2008) recipe vs other sp/gj combos on the Sanders-like surrogate.

Same 51D target as bench_sanders_surrogate.py (cross-block ρ=0.6).
Configs (all with cp=0.5):
  baseline:        sp=0,    gj=0
  pure snooker:    sp=0.10, gj=0
  pure gamma:      sp=0,    gj=0.10
  Braak recipe:    sp=0.10, gj=0.10   (Braak 2008's 90/10 split + 10% γ=1)
  prev sweet spot: sp=0.20, gj=0
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
from dezess.benchmark.metrics import compute_ess

print(f"JAX devices: {jax.devices()}")

# Same target as bench_sanders_surrogate.py
N_STREAMS = 4
POT_DIM = 7
NUIS_PER_STREAM = 11
NDIM = POT_DIM + N_STREAMS * NUIS_PER_STREAM
ALPHA = 0.6
COUPLED_NUIS_IDX = [POT_DIM + NUIS_PER_STREAM * i for i in range(N_STREAMS)]
M = np.eye(NDIM, dtype=np.float64)
for j in COUPLED_NUIS_IDX:
    M[j, 0] = ALPHA
    M[j, j] = np.sqrt(1.0 - ALPHA ** 2)
true_cov = M @ M.T
prec_jax = jnp.array(np.linalg.inv(true_cov), dtype=jnp.float64)


@jax.jit
def log_prob(x):
    return -0.5 * x @ prec_jax @ x


N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 5000
N_SEEDS = 3
LAGS = [50, 100, 200]


def make_init(seed):
    return jax.random.normal(jax.random.PRNGKey(seed), (N_WALKERS, NDIM), dtype=jnp.float64) * 0.3


def make_cfg(sp, gj):
    ens = {
        "block_sizes": [POT_DIM, N_STREAMS * NUIS_PER_STREAM],
        "use_mh": True,
        "delayed_rejection": True,
        "complementary_prob": 0.5,
    }
    if sp > 0.0:
        ens["snooker_prob"] = sp
    if gj > 0.0:
        ens["gamma_jump_prob"] = gj
    return VariantConfig(
        name=f"sp{sp}_gj{gj}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


def autocorr_at_lags(samples_3d, lags):
    n_steps, _, n_dim = samples_3d.shape
    chain = samples_3d.mean(axis=1)   # (n_steps, n_dim)
    chain = chain - chain.mean(axis=0, keepdims=True)
    var = np.maximum(chain.var(axis=0), 1e-30)
    out = []
    for L in lags:
        c = (chain[:-L] * chain[L:]).mean(axis=0) / var
        out.append(c)
    return np.array(out)   # (n_lags, n_dim)


CONFIGS = [
    ("baseline cp=0.5",        0.00, 0.00),
    ("pure snooker sp=0.10",   0.10, 0.00),
    ("pure gj=0.10",           0.00, 0.10),
    ("Braak sp=0.10+gj=0.10",  0.10, 0.10),
    ("sweet spot sp=0.20",     0.20, 0.00),
]

print(f"\n{'=' * 100}")
print(f"  BRAAK RECIPE COMPARISON: Sanders-like 51D, cross-block ρ=0.6")
print(f"  bg_MH+DR + cp=0.5, {N_WALKERS} walkers, {N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds")
print(f"{'=' * 100}")
hdr = (f"  {'Setup':<24s} | {'seed':>4s} | {'ESS_min':>8s} | "
       f"{'ACF50':>6s} | {'ACF100':>6s} | {'ACF200':>6s} | {'Fro%':>5s} | {'wall':>6s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

results = {label: [] for label, _, _ in CONFIGS}
for label, sp, gj in CONFIGS:
    cfg = make_cfg(sp, gj)
    for seed in range(N_SEEDS):
        init = make_init(seed)
        t0 = time.time()
        result = run_variant(
            log_prob, init, n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
            config=cfg, key=jax.random.PRNGKey(seed * 1000), verbose=False,
        )
        wall = time.time() - t0
        samples = np.array(result["samples"])
        flat = samples.reshape(-1, NDIM)
        ess = compute_ess(samples)
        emp_cov = np.cov(flat, rowvar=False)
        fro_rel = np.linalg.norm(emp_cov - true_cov) / np.linalg.norm(true_cov)
        acf = autocorr_at_lags(samples, LAGS)
        acf_x0 = acf[:, 0]
        results[label].append({
            "ess_min": float(ess.min()),
            "acf": [float(v) for v in acf_x0],
            "fro_rel": float(fro_rel),
            "wall": wall,
        })
        print(f"  {label:<24s} | {seed:>4d} | {float(ess.min()):>8.1f} | "
              f"{acf_x0[0]:>6.3f} | {acf_x0[1]:>6.3f} | {acf_x0[2]:>6.3f} | "
              f"{fro_rel*100:>4.1f}% | {wall:>5.1f}s", flush=True)
        gc.collect()


print(f"\n{'=' * 100}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"{'=' * 100}")
hdr2 = (f"  {'Setup':<24s} | {'ESS_min':>8s} | "
        f"{'ACF50':>6s} | {'ACF100':>6s} | {'ACF200':>6s} | {'Fro%':>5s} | {'wall':>7s}")
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))
for label, _, _ in CONFIGS:
    runs = results[label]
    em_min = np.mean([r["ess_min"] for r in runs])
    a50 = np.mean([r["acf"][0] for r in runs])
    a100 = np.mean([r["acf"][1] for r in runs])
    a200 = np.mean([r["acf"][2] for r in runs])
    fro = np.mean([r["fro_rel"] for r in runs])
    w = np.mean([r["wall"] for r in runs])
    print(f"  {label:<24s} | {em_min:>8.1f} | {a50:>6.3f} | {a100:>6.3f} | "
          f"{a200:>6.3f} | {fro*100:>4.1f}% | {w:>6.1f}s", flush=True)
