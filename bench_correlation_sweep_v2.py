#!/usr/bin/env python
"""Cross-block correlation sweep — STRONG correlation construction.

Fixes the v1 bug where dividing by POT_DIM produced cross-block correlations
of only {0.04, 0.10, 0.23} for ρ_factor ∈ {0.3, 0.6, 0.9}.

Construction: each nuisance dim j is correlated with EXACTLY ONE potential
dim (j mod POT_DIM) at level ALPHA:
    theta_pot[i]  = z[i]
    theta_nuis[j] = ALPHA * z[j mod POT_DIM] + sqrt(1-ALPHA²) * z[POT_DIM + j]

Gives:
    corr(theta_pot[i], theta_nuis[j]) = ALPHA  if j mod POT_DIM == i, else 0
    max |cross-block corr| = ALPHA exactly
    var(everything) = 1

This is the regime the user worries about: whitening missing strong residual
cross-block correlation. Cleanest possible structure for diagnosing where
the (1-ρ²) Gibbs penalty starts to dominate.
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
ALPHAS = [0.3, 0.6, 0.9]


def build_target(alpha):
    """Each nuis dim j couples to pot dim (j mod POT_DIM) at correlation alpha."""
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


CONFIGS = [
    ("slice cp=0",             cfg_slice(0.0)),
    ("slice cp=0.5",           cfg_slice(0.5)),
    ("bg_MH+DR 1blk cp=0.5",   cfg_bgmhdr([NDIM], 0.5, name="bgmh_1blk_cp05")),
    ("bg_MH+DR 2blk cp=0.5",   cfg_bgmhdr([POT_DIM, NUIS_DIM], 0.5,
                                           name="bgmh_2blk_cp05")),
    ("bg_MH+DR 2blk Braak",    cfg_bgmhdr([POT_DIM, NUIS_DIM], 0.5,
                                           sp=0.10, gj=0.10,
                                           name="bgmh_2blk_braak")),
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
    prec_jax = jnp.asarray(prec)
    log_prob = make_log_prob(prec_jax)

    sd = np.sqrt(np.diag(true_cov))
    rho_emp = np.abs(true_cov[:POT_DIM, POT_DIM:] / np.outer(sd[:POT_DIM], sd[POT_DIM:])).max()

    print(f"\n{'=' * 110}")
    print(f"  CROSS-BLOCK α = {alpha}  (max |cross-block correlation| in target: {rho_emp:.3f})")
    print(f"  {NDIM}D, blocks=[{POT_DIM}, {NUIS_DIM}], {N_WALKERS} walkers, "
          f"{N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds")
    print(f"{'=' * 110}")
    hdr = (f"  {'Setup':<24s} | {'seed':>4s} | {'ESS_min':>8s} | {'ESS/s':>6s} | "
           f"{'ACF50':>6s} | {'ACF200':>6s} | {'ACF1k':>6s} | {'Fro%':>5s} | {'wall':>6s}")
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
                "ess_min": ess_min,
                "ess_per_s": ess_per_s,
                "acf": [float(v) for v in acf_x0],
                "fro_rel": float(fro_rel),
                "wall": wall,
            })
            print(f"  {label:<24s} | {seed:>4d} | {ess_min:>8.1f} | {ess_per_s:>6.1f} | "
                  f"{acf_x0[0]:>6.3f} | {acf_x0[1]:>6.3f} | {acf_x0[2]:>6.3f} | "
                  f"{fro_rel*100:>4.1f}% | {wall:>5.1f}s", flush=True)
            gc.collect()
    all_results[alpha] = results


print(f"\n\n{'#' * 110}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"{'#' * 110}")
for alpha in ALPHAS:
    print(f"\n  cross-block α (= max |corr|) = {alpha}")
    hdr2 = (f"  {'Setup':<24s} | {'ESS_min':>8s} | {'ESS/s':>6s} | "
            f"{'ACF50':>6s} | {'ACF200':>6s} | {'ACF1k':>6s} | {'Fro%':>5s} | {'wall':>7s}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for label, _ in CONFIGS:
        runs = all_results[alpha][label]
        em_min = np.mean([r["ess_min"] for r in runs])
        eps = np.mean([r["ess_per_s"] for r in runs])
        a50 = np.mean([r["acf"][0] for r in runs])
        a200 = np.mean([r["acf"][1] for r in runs])
        a1k = np.mean([r["acf"][2] for r in runs])
        fro = np.mean([r["fro_rel"] for r in runs])
        w = np.mean([r["wall"] for r in runs])
        print(f"  {label:<24s} | {em_min:>8.1f} | {eps:>6.1f} | "
              f"{a50:>6.3f} | {a200:>6.3f} | {a1k:>6.3f} | {fro*100:>4.1f}% | "
              f"{w:>6.1f}s", flush=True)
