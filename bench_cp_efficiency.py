#!/usr/bin/env python
"""Complementary-prob sweep — what's the right cp value?

Same 51D [7,44] Sanders-style target as bench_dr_efficiency.py.
Matched init (the "good init" case where Z is good from step 0).

Configs (each at cp ∈ {0.0, 0.5, 1.0}):
  slice          — affine-invariant ensemble slice with scale_aware width
  bg_MH 1blk     — single-block MH, no DR (per-eval winner from prior bench)

Theory predicts:
  cp=0   (Z-matrix only):  best when Z is clean (good init, long warmup)
  cp=0.5 (mixed):          robust hedge
  cp=1   (live pair only): zeus-style, best when Z is biased (bad init)

With matched init, cp=0 should win; with biased init, cp=1 should win.
This bench is the cp=0-friendly regime.
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

POT_DIM = 7
NUIS_DIM = 44
NDIM = POT_DIM + NUIS_DIM
N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 20000
N_SEEDS = 3
LAGS = [50, 200, 1000]
ALPHAS = [0.0, 0.6, 0.9]
CPS = [0.0, 0.5, 1.0]


def build_target(alpha):
    M = np.eye(NDIM, dtype=np.float64)
    for j in range(NUIS_DIM):
        pot_idx = j % POT_DIM
        row = POT_DIM + j
        M[row, pot_idx] = alpha
        M[row, row]     = np.sqrt(1.0 - alpha ** 2)
    cov = M @ M.T
    return cov, M


def make_init(seed, M):
    key = jax.random.PRNGKey(seed)
    z = jax.random.normal(key, (N_WALKERS, NDIM), dtype=jnp.float64)
    return z @ jnp.asarray(M).T


def make_log_prob(prec_jax):
    @jax.jit
    def log_prob(x):
        return -0.5 * x @ prec_jax @ x
    return log_prob


def cfg_slice(cp):
    ens = {}
    if cp > 0.0:
        ens["complementary_prob"] = cp
    return VariantConfig(
        name=f"slice_cp{cp}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


def cfg_bgmh_1blk(cp):
    ens = {
        "block_sizes": [NDIM],
        "use_mh": True,
        "delayed_rejection": False,    # no-DR (per-eval winner)
        "complementary_prob": cp,
        "lp_dead": -1e6,
    }
    return VariantConfig(
        name=f"bgmh_1blk_cp{cp}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


# (label, config, evals_per_step_per_walker)
CONFIGS = []
for cp in CPS:
    CONFIGS.append((f"slice cp={cp}",         cfg_slice(cp),       4))
for cp in CPS:
    CONFIGS.append((f"bg_MH 1blk cp={cp}",    cfg_bgmh_1blk(cp),   1))


def autocorr_at_lags(samples_3d, lags):
    chain = samples_3d.mean(axis=1)
    chain = chain - chain.mean(axis=0, keepdims=True)
    var = np.maximum(chain.var(axis=0), 1e-30)
    out = []
    for L in lags:
        if L >= chain.shape[0]:
            out.append(np.full(chain.shape[1], np.nan))
            continue
        c = (chain[:-L] * chain[L:]).mean(axis=0) / var
        out.append(c)
    return np.array(out)


all_results = {}
for alpha in ALPHAS:
    true_cov, M = build_target(alpha)
    prec = np.linalg.inv(true_cov)
    log_prob = make_log_prob(jnp.asarray(prec))
    sd = np.sqrt(np.diag(true_cov))
    rho_emp = np.abs(true_cov[:POT_DIM, POT_DIM:] / np.outer(sd[:POT_DIM], sd[POT_DIM:])).max()

    print(f"\n{'=' * 120}")
    print(f"  CP SWEEP — α = {alpha}  (max |cross-block corr|: {rho_emp:.3f})")
    print(f"  {NDIM}D, {N_WALKERS} walkers, {N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds, MATCHED init")
    print(f"{'=' * 120}")
    hdr = (f"  {'Setup':<22s} | {'seed':>4s} | {'ESS_min':>8s} | {'ESS/eval':>9s} | {'ESS/s':>6s} | "
           f"{'ACF50':>6s} | {'ACF200':>6s} | {'Fro%':>5s} | {'wall':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results = {label: [] for label, _, _ in CONFIGS}
    for label, cfg, evals_per_step in CONFIGS:
        for seed in range(N_SEEDS):
            init = make_init(seed, M)
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
            total_evals = N_PROD * evals_per_step * N_WALKERS
            ess_per_eval = ess_min / total_evals * 1e6
            emp_cov = np.cov(flat, rowvar=False)
            fro_rel = np.linalg.norm(emp_cov - true_cov) / np.linalg.norm(true_cov)
            acf = autocorr_at_lags(samples, LAGS)
            acf_x0 = acf[:, 0]
            results[label].append({
                "ess_min": ess_min, "ess_per_eval": ess_per_eval, "ess_per_s": ess_per_s,
                "acf": [float(v) for v in acf_x0],
                "fro_rel": float(fro_rel), "wall": wall,
            })
            print(f"  {label:<22s} | {seed:>4d} | {ess_min:>8.1f} | {ess_per_eval:>9.2f} | "
                  f"{ess_per_s:>6.1f} | {acf_x0[0]:>6.3f} | {acf_x0[1]:>6.3f} | "
                  f"{fro_rel*100:>4.1f}% | {wall:>5.1f}s", flush=True)
            gc.collect()
    all_results[alpha] = results


print(f"\n\n{'#' * 120}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"{'#' * 120}")
for alpha in ALPHAS:
    print(f"\n  α = {alpha}")
    hdr2 = (f"  {'Setup':<22s} | {'ESS_min':>8s} | {'ESS/eval':>9s} | {'ESS/s':>6s} | "
            f"{'ACF50':>6s} | {'ACF200':>6s} | {'Fro%':>5s}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for label, _, _ in CONFIGS:
        runs = all_results[alpha][label]
        em_min = np.mean([r["ess_min"] for r in runs])
        epe   = np.mean([r["ess_per_eval"] for r in runs])
        eps   = np.mean([r["ess_per_s"] for r in runs])
        a50   = np.mean([r["acf"][0] for r in runs])
        a200  = np.mean([r["acf"][1] for r in runs])
        fro   = np.mean([r["fro_rel"] for r in runs])
        print(f"  {label:<22s} | {em_min:>8.1f} | {epe:>9.2f} | {eps:>6.1f} | "
              f"{a50:>6.3f} | {a200:>6.3f} | {fro*100:>4.1f}%", flush=True)

# Headline cp comparison: for each (sampler family × α), which cp wins on ESS/eval?
print(f"\n\n{'#' * 120}")
print(f"  HEADLINE: best cp per sampler family × α  (matched init — cp=0 favored by Z geometry)")
print(f"{'#' * 120}")
print(f"  {'α':>6s} | {'slice best cp':>16s} {'(ESS/eval)':>11s} | {'1blk MH best cp':>16s} {'(ESS/eval)':>11s}")
for alpha in ALPHAS:
    slice_scores = {cp: np.mean([r["ess_per_eval"] for r in all_results[alpha][f"slice cp={cp}"]])
                    for cp in CPS}
    mh_scores    = {cp: np.mean([r["ess_per_eval"] for r in all_results[alpha][f"bg_MH 1blk cp={cp}"]])
                    for cp in CPS}
    best_slice_cp = max(slice_scores, key=slice_scores.get)
    best_mh_cp    = max(mh_scores,    key=mh_scores.get)
    print(f"  {alpha:>6.1f} | {f'cp={best_slice_cp}':>16s} {slice_scores[best_slice_cp]:>11.2f}  | "
          f"{f'cp={best_mh_cp}':>16s} {mh_scores[best_mh_cp]:>11.2f}")

# Per-cp ratio relative to cp=0 — shows whether moving away from Z helps or hurts.
print(f"\n  RELATIVE ESS/eval (vs cp=0 baseline) per α — higher = cp setting beats pure Z")
print(f"  {'α':>6s} | {'slice cp=0.5':>13s} {'cp=1':>8s} | {'1blk cp=0.5':>13s} {'cp=1':>8s}")
for alpha in ALPHAS:
    s = all_results[alpha]
    s0 = np.mean([r["ess_per_eval"] for r in s["slice cp=0.0"]])
    s5 = np.mean([r["ess_per_eval"] for r in s["slice cp=0.5"]])
    s1 = np.mean([r["ess_per_eval"] for r in s["slice cp=1.0"]])
    m0 = np.mean([r["ess_per_eval"] for r in s["bg_MH 1blk cp=0.0"]])
    m5 = np.mean([r["ess_per_eval"] for r in s["bg_MH 1blk cp=0.5"]])
    m1 = np.mean([r["ess_per_eval"] for r in s["bg_MH 1blk cp=1.0"]])
    print(f"  {alpha:>6.1f} | {s5/max(s0,1e-9):>13.2f}x {s1/max(s0,1e-9):>7.2f}x | "
          f"{m5/max(m0,1e-9):>13.2f}x {m1/max(m0,1e-9):>7.2f}x")
