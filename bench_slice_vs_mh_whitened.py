#!/usr/bin/env python
"""Slice (scale_aware+fixed) vs bg_MH+DR on a fully whitened, well-init posterior.

Target: 51D isotropic unit Gaussian (the "best case" for either sampler — no
cross-block correlation, no nonlinear degeneracies, no flat floor). Walkers
init at N(0, I) so the ensemble starts matched to the target (no biased Z).

This isolates the question: when nothing is exploitable by snooker / blocking,
does slice's "guaranteed acceptance, no MH rejection waste" beat MH+DR per
wall-second? Or does bg_MH+DR's cheaper-per-step still win?

Configs:
  slice cp=0    — scale_aware + fixed slice (the default baseline)
  slice cp=0.5  — same + complementary direction
  bg_MH+DR 1blk cp=0.5         — one block of 51 dims (degenerates to full-space MH)
  bg_MH+DR 2blk cp=0.5         — Sanders-style [7, 44] split (does blocking hurt on isotropic?)
  bg_MH+DR 1blk Braak          — cp=0.5 + sp=0.10 + gj=0.10 (do the extras hurt on isotropic?)
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

# 51D isotropic unit Gaussian (the whitened target)
NDIM = 51
true_cov = np.eye(NDIM, dtype=np.float64)


@jax.jit
def log_prob(x):
    return -0.5 * jnp.sum(x * x)


N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 5000
N_SEEDS = 3
LAGS = [50, 100, 200]


def make_init(seed):
    # sigma=1.0 — matched to the target's unit variance ("well-initialized")
    return jax.random.normal(jax.random.PRNGKey(seed), (N_WALKERS, NDIM), dtype=jnp.float64)


def cfg_slice(cp):
    ens_kwargs = {}
    if cp > 0.0:
        ens_kwargs["complementary_prob"] = cp
    return VariantConfig(
        name=f"slice_cp{cp}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )


def cfg_bgmhdr(block_sizes, cp, sp=0.0, gj=0.0, name=None):
    ens = {
        "block_sizes": list(block_sizes),
        "use_mh": True,
        "delayed_rejection": True,
        "complementary_prob": cp,
    }
    if sp > 0.0:
        ens["snooker_prob"] = sp
    if gj > 0.0:
        ens["gamma_jump_prob"] = gj
    return VariantConfig(
        name=name or f"bgmh_bs{block_sizes}_cp{cp}_sp{sp}_gj{gj}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


def autocorr_at_lags(samples_3d, lags):
    chain = samples_3d.mean(axis=1)
    chain = chain - chain.mean(axis=0, keepdims=True)
    var = np.maximum(chain.var(axis=0), 1e-30)
    out = []
    for L in lags:
        c = (chain[:-L] * chain[L:]).mean(axis=0) / var
        out.append(c)
    return np.array(out)


CONFIGS = [
    ("slice cp=0",              cfg_slice(0.0)),
    ("slice cp=0.5",            cfg_slice(0.5)),
    ("bg_MH+DR 1blk cp=0.5",    cfg_bgmhdr([NDIM], 0.5, name="bgmh_1blk_cp05")),
    ("bg_MH+DR 2blk cp=0.5",    cfg_bgmhdr([7, NDIM - 7], 0.5, name="bgmh_2blk_cp05")),
    ("bg_MH+DR 1blk Braak",     cfg_bgmhdr([NDIM], 0.5, sp=0.10, gj=0.10,
                                            name="bgmh_1blk_braak")),
]

print(f"\n{'=' * 110}")
print(f"  SLICE vs bg_MH+DR — {NDIM}D isotropic unit Gaussian, well-init (sigma=1.0)")
print(f"  {N_WALKERS} walkers, {N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds")
print(f"{'=' * 110}")
hdr = (f"  {'Setup':<24s} | {'seed':>4s} | {'ESS_min':>8s} | {'ESS/s':>7s} | "
       f"{'ACF50':>6s} | {'ACF100':>6s} | {'ACF200':>6s} | {'Fro%':>5s} | {'wall':>6s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

results = {label: [] for label, _ in CONFIGS}
for label, cfg in CONFIGS:
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
        ess_min = float(ess.min())
        ess_per_s = ess_min / max(wall, 1e-9)
        emp_cov = np.cov(flat, rowvar=False)
        fro_rel = np.linalg.norm(emp_cov - true_cov) / np.linalg.norm(true_cov)
        acf = autocorr_at_lags(samples, LAGS)
        acf_x0 = acf[:, 0]
        results[label].append({
            "ess_min": ess_min,
            "ess_per_s": ess_per_s,
            "acf": [float(v) for v in acf_x0],
            "fro_rel": float(fro_rel),
            "wall": wall,
        })
        print(f"  {label:<24s} | {seed:>4d} | {ess_min:>8.1f} | {ess_per_s:>7.1f} | "
              f"{acf_x0[0]:>6.3f} | {acf_x0[1]:>6.3f} | {acf_x0[2]:>6.3f} | "
              f"{fro_rel*100:>4.1f}% | {wall:>5.1f}s", flush=True)
        gc.collect()

print(f"\n{'=' * 110}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"{'=' * 110}")
hdr2 = (f"  {'Setup':<24s} | {'ESS_min':>8s} | {'ESS/s':>7s} | "
        f"{'ACF50':>6s} | {'ACF100':>6s} | {'ACF200':>6s} | {'Fro%':>5s} | {'wall':>7s}")
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))
for label, _ in CONFIGS:
    runs = results[label]
    em_min = np.mean([r["ess_min"] for r in runs])
    eps   = np.mean([r["ess_per_s"] for r in runs])
    a50 = np.mean([r["acf"][0] for r in runs])
    a100 = np.mean([r["acf"][1] for r in runs])
    a200 = np.mean([r["acf"][2] for r in runs])
    fro = np.mean([r["fro_rel"] for r in runs])
    w = np.mean([r["wall"] for r in runs])
    print(f"  {label:<24s} | {em_min:>8.1f} | {eps:>7.1f} | "
          f"{a50:>6.3f} | {a100:>6.3f} | {a200:>6.3f} | {fro*100:>4.1f}% | {w:>6.1f}s",
          flush=True)
