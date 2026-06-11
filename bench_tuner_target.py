"""A/B the retargeted MH tuner: target_accept in {0.50 (legacy), 0.35, 0.234, 0.15}.

61D correlated Gaussian (alpha=0.5 latent factor, the user's param-space
scale), bg_MH 1blk no-DR cp=0.5, 128 walkers, 1000+3000 steps, 3 seeds.
sp=0 for the main sweep (clean tuner measurement); one extra config at the
new default with sp=0.1 to confirm the production mixture still tunes sanely.

Measured per config:
  realized acceptance  — fraction of steps each walker actually moved
  tuned mu             — mu_blocks[0] after warmup
  ESS_min / ESS_med    — sampler quality
  ESS/1k-eval          — the target metric (1 eval/step/walker, no DR)
  cov-truth Fro        — correctness guard

Expected: realized acceptance lands near each config's target; ESS/eval
peaks around 0.234 (the ESJD optimum), with 0.5 at ~70% of it.
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
N_WALKERS = 128
N_WARMUP = 1000
N_PROD = 3000
N_SEEDS = 3


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


def cfg(ta, sp=0.0):
    ens = {"block_sizes": [NDIM], "use_mh": True, "delayed_rejection": False,
           "complementary_prob": 0.5, "lp_dead": -1e6, "target_accept": ta}
    if sp > 0:
        ens["snooker_prob"] = sp
    return VariantConfig(name=f"bgmh_ta{ta}_sp{sp}", direction="de_mcz",
        width="scale_aware", slice_fn="fixed", zmatrix="circular",
        ensemble="block_gibbs", check_nans=False,
        width_kwargs={"scale_factor": 1.0}, ensemble_kwargs=ens)


CONFIGS = [
    ("ta=0.50 (legacy)", cfg(0.50)),
    ("ta=0.35",          cfg(0.35)),
    ("ta=0.234 (new)",   cfg(0.234)),
    ("ta=0.15",          cfg(0.15)),
    ("ta=0.234 sp=0.1",  cfg(0.234, sp=0.1)),
]

log_prob, cov = build_target()

print(f"\n{'=' * 104}")
print(f"  MH TUNER TARGET A/B — {NDIM}D corr (a={ALPHA}), 1blk no-DR cp=0.5, "
      f"{N_WALKERS}w, {N_WARMUP}+{N_PROD}, {N_SEEDS} seeds")
print(f"{'=' * 104}")
hdr = (f"  {'config':<18s} | {'accept':>7s} | {'mu_tuned':>8s} | {'ESS_min':>8s} | "
       f"{'ESS_med':>8s} | {'ESS/1k-ev':>9s} | {'cov-truth':>9s} | {'wall':>6s}")
print(hdr); print("  " + "-" * (len(hdr) - 2))

for label, c in CONFIGS:
    acc_l, mu_l, essm_l, essd_l, fro_l, wall_l = [], [], [], [], [], []
    for seed in range(N_SEEDS):
        init = jax.random.normal(jax.random.PRNGKey(seed), (N_WALKERS, NDIM),
                                 dtype=jnp.float64)
        t0 = time.time()
        res = run_variant(log_prob, init, n_steps=N_WARMUP + N_PROD,
                          n_warmup=N_WARMUP, config=c,
                          key=jax.random.PRNGKey(seed * 1000), verbose=False)
        wall = time.time() - t0
        samples = np.array(res["samples"])
        # realized acceptance = fraction of steps each walker moved
        moved = np.any(np.diff(samples, axis=0) != 0.0, axis=-1)  # (T-1, W)
        acc = float(moved.mean())
        ess = compute_ess(samples)
        flat = samples.reshape(-1, NDIM)
        fro = np.linalg.norm(np.cov(flat, rowvar=False) - cov) / np.linalg.norm(cov)
        mu_t = float(np.asarray(res.get("mu_blocks", [res["mu"]])).ravel()[0])
        acc_l.append(acc); mu_l.append(mu_t)
        essm_l.append(float(ess.min())); essd_l.append(float(np.median(ess)))
        fro_l.append(fro); wall_l.append(wall)
    n_ev = N_PROD * N_WALKERS
    print(f"  {label:<18s} | {np.mean(acc_l):>7.3f} | {np.mean(mu_l):>8.4f} | "
          f"{np.mean(essm_l):>8.1f} | {np.mean(essd_l):>8.1f} | "
          f"{np.mean(essm_l)/n_ev*1000:>9.3f} | {np.mean(fro_l)*100:>8.1f}% | "
          f"{np.mean(wall_l):>5.1f}s", flush=True)

print("\n  Check: realized acceptance should track each target; ESS/eval and")
print("  ESS_med should peak near ta=0.234; cov-truth must stay flat across")
print("  configs (any drift would indicate a correctness problem, not tuning).")
