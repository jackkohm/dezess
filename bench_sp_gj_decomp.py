#!/usr/bin/env python
"""sp/gj decomposition — is snooker alone net positive when separated from gj?

Tests 5 configs on bg_MH+DR 1blk cp=0.5 (the production setup) across
α ∈ {0.0, 0.6, 0.9}:
  A. clean              — no sp, no gj
  B. sp=0.05 only       — conservative snooker
  C. sp=0.10 only       — Braak snooker rate
  D. gj=0.10 only       — Braak gamma jump rate
  E. sp=0.10 + gj=0.10  — full Braak recipe

All configs include lp_dead=-1e6 (the new guard active). Same 51D [7,44]
Sanders-style anisotropic target as prior benches. Matched init.

Primary metric: ESS/eval (eval cost is 2 per step for all configs since DR
adds 1 extra eval per block, and 1 block × 2 = 2). Differences in ESS_min
directly translate to ESS/eval ratios.

Headline answer: per α, which sp/gj setting wins?
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time, gc
import jax, jax.numpy as jnp
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
EVALS_PER_STEP = 2   # 1 block * (1 stage1 + 1 stage2 DR) = 2 evals per walker per step


def build_target(alpha):
    M = np.eye(NDIM, dtype=np.float64)
    for j in range(NUIS_DIM):
        pot_idx = j % POT_DIM
        row = POT_DIM + j
        M[row, pot_idx] = alpha
        M[row, row]     = np.sqrt(1.0 - alpha ** 2)
    return M @ M.T, M


def make_init(seed, M):
    key = jax.random.PRNGKey(seed)
    z = jax.random.normal(key, (N_WALKERS, NDIM), dtype=jnp.float64)
    return z @ jnp.asarray(M).T


def make_log_prob(prec_jax):
    @jax.jit
    def log_prob(x):
        return -0.5 * x @ prec_jax @ x
    return log_prob


def cfg(sp, gj, name):
    ens = {
        "block_sizes": [NDIM],
        "use_mh": True,
        "delayed_rejection": True,
        "complementary_prob": 0.5,
        "lp_dead": -1e6,
    }
    if sp > 0:
        ens["snooker_prob"] = sp
    if gj > 0:
        ens["gamma_jump_prob"] = gj
    return VariantConfig(
        name=name,
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


CONFIGS = [
    ("A. clean             ", cfg(sp=0.00, gj=0.00, name="bgmh_clean")),
    ("B. sp=0.05 only      ", cfg(sp=0.05, gj=0.00, name="bgmh_sp05")),
    ("C. sp=0.10 only      ", cfg(sp=0.10, gj=0.00, name="bgmh_sp10")),
    ("D. gj=0.10 only      ", cfg(sp=0.00, gj=0.10, name="bgmh_gj10")),
    ("E. sp=0.10 + gj=0.10 ", cfg(sp=0.10, gj=0.10, name="bgmh_braak")),
]


def autocorr_at_lags(samples_3d, lags):
    chain = samples_3d.mean(axis=1)
    chain = chain - chain.mean(axis=0, keepdims=True)
    var = np.maximum(chain.var(axis=0), 1e-30)
    out = []
    for L in lags:
        if L >= chain.shape[0]:
            out.append(np.full(chain.shape[1], np.nan)); continue
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
    print(f"  sp/gj DECOMPOSITION — α = {alpha}  (max |cross-block corr|: {rho_emp:.3f})")
    print(f"  {NDIM}D, blocks=[{NDIM}], {N_WALKERS} walkers, {N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds")
    print(f"{'=' * 120}")
    hdr = (f"  {'Setup':<24s} | {'seed':>4s} | {'ESS_min':>8s} | {'ESS/eval':>9s} | "
           f"{'ACF50':>6s} | {'ACF200':>6s} | {'Fro%':>5s} | {'wall':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results = {label: [] for label, _ in CONFIGS}
    for label, config in CONFIGS:
        for seed in range(N_SEEDS):
            init = make_init(seed, M)
            t0 = time.time()
            result = run_variant(
                log_prob, init, n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
                config=config, key=jax.random.PRNGKey(seed * 1000), verbose=False,
            )
            wall = time.time() - t0
            samples = np.array(result["samples"])
            flat = samples.reshape(-1, NDIM)
            ess = compute_ess(samples)
            ess_min = float(ess.min())
            total_evals = N_PROD * EVALS_PER_STEP * N_WALKERS
            ess_per_eval = ess_min / total_evals * 1e6
            emp_cov = np.cov(flat, rowvar=False)
            fro_rel = np.linalg.norm(emp_cov - true_cov) / np.linalg.norm(true_cov)
            acf = autocorr_at_lags(samples, LAGS)
            acf_x0 = acf[:, 0]
            results[label].append({
                "ess_min": ess_min, "ess_per_eval": ess_per_eval,
                "acf": [float(v) for v in acf_x0],
                "fro_rel": float(fro_rel), "wall": wall,
            })
            print(f"  {label:<24s} | {seed:>4d} | {ess_min:>8.1f} | {ess_per_eval:>9.2f} | "
                  f"{acf_x0[0]:>6.3f} | {acf_x0[1]:>6.3f} | "
                  f"{fro_rel*100:>4.1f}% | {wall:>5.1f}s", flush=True)
            gc.collect()
    all_results[alpha] = results


print(f"\n\n{'#' * 120}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"{'#' * 120}")
for alpha in ALPHAS:
    print(f"\n  α = {alpha}")
    hdr2 = (f"  {'Setup':<24s} | {'ESS_min':>8s} | {'ESS/eval':>9s} | "
            f"{'ACF50':>6s} | {'ACF200':>6s} | {'Fro%':>5s}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for label, _ in CONFIGS:
        runs = all_results[alpha][label]
        em_min = np.mean([r["ess_min"] for r in runs])
        epe   = np.mean([r["ess_per_eval"] for r in runs])
        a50   = np.mean([r["acf"][0] for r in runs])
        a200  = np.mean([r["acf"][1] for r in runs])
        fro   = np.mean([r["fro_rel"] for r in runs])
        print(f"  {label:<24s} | {em_min:>8.1f} | {epe:>9.2f} | "
              f"{a50:>6.3f} | {a200:>6.3f} | {fro*100:>4.1f}%", flush=True)

print(f"\n\n{'#' * 120}")
print(f"  HEADLINE: best config per α  (ESS/eval)  + ratio vs clean baseline")
print(f"{'#' * 120}")
print(f"  {'α':>6s} | {'WINNER':<22s} | {'ESS/eval':>9s} | {'vs clean':>10s} | {'sp ratio':>10s} {'gj ratio':>10s} {'braak ratio':>11s}")
for alpha in ALPHAS:
    scores = {label.strip(): np.mean([r["ess_per_eval"] for r in all_results[alpha][label]])
              for label, _ in CONFIGS}
    winner = max(scores, key=scores.get)
    clean = scores["A. clean"]
    print(f"  {alpha:>6.1f} | {winner:<22s} | {scores[winner]:>9.2f} | "
          f"{scores[winner]/max(clean,1e-9):>9.2f}x | "
          f"{scores['C. sp=0.10 only']/max(clean,1e-9):>9.2f}x "
          f"{scores['D. gj=0.10 only']/max(clean,1e-9):>9.2f}x "
          f"{scores['E. sp=0.10 + gj=0.10']/max(clean,1e-9):>10.2f}x")
