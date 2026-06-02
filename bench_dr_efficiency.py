#!/usr/bin/env python
"""DR on vs off — is delayed rejection worth the 2x eval cost?

Same 51D [7, 44] Sanders-style target as bench_correlation_sweep_v2.py,
init matched to target. Three regimes:
  ρ=0.0  — clean isotropic, well-tuned                  (DR theory says: loses)
  ρ=0.6  — moderate cross-block correlation             (DR theory says: mixed)
  ρ=0.9  — strong cross-block where 2blk struggles      (DR theory says: maybe rescues)

Compare ESS_min / wall-second across:
  slice cp=0                — reference baseline (no DR involved)
  bg_MH+DR 1blk cp=0.5      — single-block MH WITH delayed rejection
  bg_MH    1blk cp=0.5      — same but DR OFF
  bg_MH+DR 2blk cp=0.5      — two-block MH WITH delayed rejection
  bg_MH    2blk cp=0.5      — same but DR OFF
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
    M_jax = jnp.asarray(M)
    return z @ M_jax.T


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


def cfg_bgmh(block_sizes, cp, dr, name):
    ens = {
        "block_sizes": list(block_sizes),
        "use_mh": True,
        "delayed_rejection": dr,
        "complementary_prob": cp,
        "lp_dead": -1e6,   # explicit for cache safety (see test_lp_dead_guard)
    }
    return VariantConfig(
        name=name,
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


CONFIGS = [
    ("slice cp=0 (ref)",        cfg_slice(0.0)),
    ("bg_MH+DR 1blk cp=0.5",    cfg_bgmh([NDIM], 0.5, True,  "bgmh_1blk_dr")),
    ("bg_MH    1blk cp=0.5",    cfg_bgmh([NDIM], 0.5, False, "bgmh_1blk_nodr")),
    ("bg_MH+DR 2blk cp=0.5",    cfg_bgmh([POT_DIM, NUIS_DIM], 0.5, True,  "bgmh_2blk_dr")),
    ("bg_MH    2blk cp=0.5",    cfg_bgmh([POT_DIM, NUIS_DIM], 0.5, False, "bgmh_2blk_nodr")),
]


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

    print(f"\n{'=' * 110}")
    print(f"  DR on/off — α = {alpha}  (max |cross-block corr| in target: {rho_emp:.3f})")
    print(f"  {NDIM}D, blocks=[{POT_DIM}, {NUIS_DIM}], {N_WALKERS} walkers, "
          f"{N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds")
    print(f"{'=' * 110}")
    hdr = (f"  {'Setup':<24s} | {'seed':>4s} | {'ESS_min':>8s} | {'ESS/s':>6s} | "
           f"{'ACF50':>6s} | {'ACF200':>6s} | {'Fro%':>5s} | {'wall':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results = {label: [] for label, _ in CONFIGS}
    for label, cfg in CONFIGS:
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
            emp_cov = np.cov(flat, rowvar=False)
            fro_rel = np.linalg.norm(emp_cov - true_cov) / np.linalg.norm(true_cov)
            acf = autocorr_at_lags(samples, LAGS)
            acf_x0 = acf[:, 0]
            results[label].append({
                "ess_min": ess_min, "ess_per_s": ess_per_s,
                "acf": [float(v) for v in acf_x0],
                "fro_rel": float(fro_rel), "wall": wall,
            })
            print(f"  {label:<24s} | {seed:>4d} | {ess_min:>8.1f} | {ess_per_s:>6.1f} | "
                  f"{acf_x0[0]:>6.3f} | {acf_x0[1]:>6.3f} | "
                  f"{fro_rel*100:>4.1f}% | {wall:>5.1f}s", flush=True)
            gc.collect()
    all_results[alpha] = results


print(f"\n\n{'#' * 110}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds) — DR pays its 2x cost iff ESS/s improves")
print(f"{'#' * 110}")
for alpha in ALPHAS:
    print(f"\n  cross-block α = {alpha}")
    hdr2 = (f"  {'Setup':<24s} | {'ESS_min':>8s} | {'ESS/s':>6s} | "
            f"{'ACF50':>6s} | {'ACF200':>6s} | {'Fro%':>5s} | {'wall':>7s}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for label, _ in CONFIGS:
        runs = all_results[alpha][label]
        em_min = np.mean([r["ess_min"] for r in runs])
        eps   = np.mean([r["ess_per_s"] for r in runs])
        a50   = np.mean([r["acf"][0] for r in runs])
        a200  = np.mean([r["acf"][1] for r in runs])
        fro   = np.mean([r["fro_rel"] for r in runs])
        w     = np.mean([r["wall"] for r in runs])
        print(f"  {label:<24s} | {em_min:>8.1f} | {eps:>6.1f} | "
              f"{a50:>6.3f} | {a200:>6.3f} | {fro*100:>4.1f}% | {w:>6.1f}s", flush=True)

# Compute the DR-vs-no-DR ratio per ρ for the headline answer
print(f"\n\n{'#' * 110}")
print(f"  DR / no-DR ESS/s RATIO  (>1 means DR pays its cost, <1 means it costs more than it gives)")
print(f"{'#' * 110}")
print(f"  {'α':>6s} | {'1blk DR/noDR':>14s} | {'2blk DR/noDR':>14s}")
for alpha in ALPHAS:
    r1_dr = np.mean([r["ess_per_s"] for r in all_results[alpha]["bg_MH+DR 1blk cp=0.5"]])
    r1_no = np.mean([r["ess_per_s"] for r in all_results[alpha]["bg_MH    1blk cp=0.5"]])
    r2_dr = np.mean([r["ess_per_s"] for r in all_results[alpha]["bg_MH+DR 2blk cp=0.5"]])
    r2_no = np.mean([r["ess_per_s"] for r in all_results[alpha]["bg_MH    2blk cp=0.5"]])
    ratio_1 = r1_dr / max(r1_no, 1e-9)
    ratio_2 = r2_dr / max(r2_no, 1e-9)
    print(f"  {alpha:>6.1f} | {ratio_1:>14.2f}x | {ratio_2:>14.2f}x")
