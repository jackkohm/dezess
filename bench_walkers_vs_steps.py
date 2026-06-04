"""Walkers vs steps at FIXED sample budget — what's more valuable?

Fixed production budget n_walkers * n_prod = 256000, swept across the split:
  (512, 500), (256, 1000), (128, 2000), (64, 4000), (32, 8000)
Same warmup (1000) for all. 61D correlated target.

Question: at fixed total samples, is it better to spend the budget on more
walkers (parallel bulk samples) or more steps (decorrelate the slow modes)?

Prediction: ESS_min (slowest collective mode) rises with MORE STEPS / fewer
walkers (only time decorrelates the slow mode); cov-truth (bulk) is roughly
split-independent at fixed total samples; too-few walkers hurt via the cp
rank floor. Includes slice cp=0 (no rank floor -> clean walkers-vs-steps)
alongside the cp=0.5 production configs.
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
NDIM = POT + NUIS
ALPHA = 0.5
N_WARMUP = 1000
BUDGET = 256000           # fixed production samples = n_walkers * n_prod
SPLITS = [(512, 500), (256, 1000), (128, 2000), (64, 4000), (32, 8000)]
N_SEEDS = 2


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


CONFIGS = [("slice cp=0", cfg_slice(0.0)),
           ("slice cp=0.5", cfg_slice(0.5)),
           ("bgmhdr cp=0.5", cfg_mh(0.5))]

log_prob, cov = build_target()

print(f"\n{'=' * 92}")
print(f"  WALKERS vs STEPS — {NDIM}D correlated, FIXED {BUDGET} prod samples, warmup={N_WARMUP}, {N_SEEDS} seeds")
print(f"{'=' * 92}")

for cname, cfg in CONFIGS:
    print(f"\n  --- {cname} ---")
    hdr = (f"  {'walkers':>7s} x {'steps':<6s} | {'ESS_min':>8s} | {'cov-truth':>9s} | "
           f"{'warmup-overhead':>15s} | {'wall':>6s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    for nw, nprod in SPLITS:
        ess_l, fro_l, wall_l = [], [], []
        for seed in range(N_SEEDS):
            init = jax.random.normal(jax.random.PRNGKey(seed), (nw, NDIM), dtype=jnp.float64)
            t0 = time.time()
            res = run_variant(log_prob, init, n_steps=N_WARMUP + nprod,
                              n_warmup=N_WARMUP, config=cfg,
                              key=jax.random.PRNGKey(seed * 1000), verbose=False)
            wall = time.time() - t0
            samples = np.array(res["samples"])
            ess_l.append(float(compute_ess(samples).min()))
            flat = samples.reshape(-1, NDIM)
            fro_l.append(np.linalg.norm(np.cov(flat, rowvar=False) - cov) / np.linalg.norm(cov))
            wall_l.append(wall)
        ess_m = np.mean(ess_l); fro_m = np.mean(fro_l); wall_m = np.mean(wall_l)
        # warmup overhead = warmup samples / production samples (wasted fraction)
        overhead = (nw * N_WARMUP) / BUDGET
        rows.append((nw, nprod, ess_m, fro_m, overhead, wall_m))
        print(f"  {nw:>7d} x {nprod:<6d} | {ess_m:>8.0f} | {fro_m*100:>8.1f}% | "
              f"{overhead*100:>13.0f}% | {wall_m:>5.1f}s", flush=True)
    # best split per metric
    best_ess = max(rows, key=lambda r: r[2])
    best_cov = min(rows, key=lambda r: r[3])
    print(f"    best ESS_min (slow mode): {best_ess[0]}x{best_ess[1]} (ESS={best_ess[2]:.0f})")
    print(f"    best cov-truth (bulk):    {best_cov[0]}x{best_cov[1]} (cov={best_cov[3]*100:.1f}%)")

print(f"\n{'=' * 92}")
print(f"  At fixed sample budget: ESS_min (slow mode) favors MORE STEPS; cov-truth")
print(f"  (bulk) is ~split-independent; too-few walkers hurt cp>0 via the rank floor.")
print(f"  warmup-overhead = wasted warmup evals as a fraction of the production budget")
print(f"  (grows with walkers -> another reason not to over-allocate walkers).")
