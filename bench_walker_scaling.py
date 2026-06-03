"""How much does adding walkers help mixing? — slice & bg_MH+DR on 61D.

Sweeps n_walkers in {64, 128, 256, 512} on a 61D correlated Gaussian (the
user's real parameter-space size; [5,56] latent-factor cross-block structure,
representing residual post-whitening correlation).

The decisive question: does adding walkers REDUCE per-chain autocorrelation tau
(genuine mixing help -> super-linear ESS, better ESS/eval) or just add
independent chains at constant tau (pure throughput -> linear ESS, flat
ESS/eval)? Theory: cp>0 needs n_walkers >= 2*d+2 = 124 for the half-ensembles
to span 61 dims, so 128 is marginal and 256 should help super-linearly.

Reports per (config, n_walkers): ESS_min, per-chain ESS (ESS_min/n_walkers,
proportional to 1/tau), tau_est, ESS/1k-eval, cov-truth, wall. Then the
ESS_min doubling ratio (ESS(2N)/ESS(N)): >2 = super-linear (mixing improves),
~2 = linear (throughput only), <2 = diminishing returns.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys, time
import jax, jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

import dezess
from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.benchmark.metrics import compute_ess

print(f"dezess: {dezess.__file__}")

POT, NUIS = 5, 56
NDIM = POT + NUIS   # 61
ALPHA = 0.5
N_WARMUP = 1000
N_PROD = 3000
N_SEEDS = 2
WALKERS = [64, 128, 256, 512]
N_EXP, N_SHR = 2, 8


def build_target():
    M = np.eye(NDIM)
    for j in range(NUIS):
        M[POT + j, j % POT] = ALPHA
        M[POT + j, POT + j] = np.sqrt(1 - ALPHA**2)
    cov = M @ M.T
    prec = jnp.asarray(np.linalg.inv(cov))
    @jax.jit
    def lp(x):
        return -0.5 * x @ prec @ x
    return lp, cov


def cfg_slice(cp):
    ens = {"complementary_prob": cp} if cp > 0 else {}
    return VariantConfig(name=f"slice_cp{cp}", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard", check_nans=False,
        width_kwargs={"scale_factor": 1.0}, ensemble_kwargs=ens)


def cfg_mh(cp):
    return VariantConfig(name=f"bgmh_cp{cp}", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs", check_nans=False,
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [NDIM], "use_mh": True,
                         "delayed_rejection": True, "complementary_prob": cp,
                         "lp_dead": -1e6})


def evals_per_step(name, nw):
    if name.startswith("slice"):
        return nw * (N_EXP + N_SHR + 1)
    return nw * 2   # bg_MH+DR 1blk: stage1 + DR


CONFIGS = [("slice cp=0.5", cfg_slice(0.5)),
           ("bgmhdr cp=0.5", cfg_mh(0.5))]

log_prob, cov = build_target()

print(f"\n{'=' * 96}")
print(f"  WALKER SCALING — {NDIM}D correlated (alpha={ALPHA}), {N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds")
print(f"  cp>0 rank threshold: n_walkers >= 2*d+2 = {2*NDIM+2}")
print(f"{'=' * 96}")

results = {}
for cname, cfgfn in CONFIGS:
    print(f"\n  --- {cname} ---")
    hdr = (f"  {'walkers':>7s} | {'ESS_min':>8s} | {'ESS/chain':>9s} | {'tau_est':>8s} | "
           f"{'ESS/1k-ev':>9s} | {'cov-truth':>9s} | {'wall':>6s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    results[cname] = {}
    for nw in WALKERS:
        ess_l, fro_l, wall_l = [], [], []
        for seed in range(N_SEEDS):
            init = jax.random.normal(jax.random.PRNGKey(seed), (nw, NDIM), dtype=jnp.float64)
            t0 = time.time()
            res = run_variant(log_prob, init, n_steps=N_WARMUP + N_PROD,
                              n_warmup=N_WARMUP, config=cfgfn,
                              key=jax.random.PRNGKey(seed * 1000), verbose=False)
            wall = time.time() - t0
            samples = np.array(res["samples"])
            ess_l.append(float(compute_ess(samples).min()))
            flat = samples.reshape(-1, NDIM)
            fro_l.append(np.linalg.norm(np.cov(flat, rowvar=False) - cov) / np.linalg.norm(cov))
            wall_l.append(wall)
        ess_m = np.mean(ess_l); fro_m = np.mean(fro_l); wall_m = np.mean(wall_l)
        ess_per_chain = ess_m / nw
        tau_est = N_PROD / max(ess_per_chain, 1e-9)
        eveq = N_PROD * evals_per_step(cname, nw)
        ess_per_1k = ess_m / eveq * 1000
        results[cname][nw] = ess_m
        print(f"  {nw:>7d} | {ess_m:>8.0f} | {ess_per_chain:>9.2f} | {tau_est:>8.0f} | "
              f"{ess_per_1k:>9.3f} | {fro_m*100:>8.1f}% | {wall_m:>5.1f}s", flush=True)

print(f"\n{'=' * 96}")
print(f"  ESS_min DOUBLING RATIO  ESS(2N)/ESS(N)  — >2 super-linear (mixing improves),")
print(f"  ~2 linear (throughput only), <2 diminishing returns")
print(f"{'=' * 96}")
print(f"  {'config':<16s} | {'64->128':>8s} | {'128->256':>9s} | {'256->512':>9s}")
for cname, _ in CONFIGS:
    r = results[cname]
    r1 = r[128] / r[64]; r2 = r[256] / r[128]; r3 = r[512] / r[256]
    print(f"  {cname:<16s} | {r1:>7.2f}x | {r2:>8.2f}x | {r3:>8.2f}x")
print(f"\n  On GPU adding walkers is ~free in wall-time until saturation, so even")
print(f"  linear (2x) ESS gain is worth it there; the EVAL cost (expensive target)")
print(f"  scales with walkers, so super-linear (>2x) is what makes more walkers")
print(f"  pay off per-eval. Crossing 2*d+2={2*NDIM+2} should show super-linear 128->256.")
