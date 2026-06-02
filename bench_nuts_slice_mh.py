"""Three-way: dezess NUTS vs slice vs bg_MH+DR on the same targets.

Same 64 walkers, 1000 warmup + 3000 prod, 2 seeds, on two targets:
  T1: 51D isotropic unit Gaussian
  T2: 51D [7,44] cross-block correlated Gaussian (alpha=0.6, Sanders-like)

Eval accounting (the metric that matters for expensive targets):
  NUTS  : gradient evals = total leapfrog steps (measured exactly)
  slice : function evals  = n_prod * n_walkers * (n_expand + n_shrink + 1)
  bg_MH+DR: function evals = n_prod * n_walkers * (n_blocks * 2)   [stage1 + DR]
A reverse-mode gradient costs ~2-3x a function eval, so we also report an
'eval-equivalent' column with grad = 2.5x func to put them on common footing.
Caveat: on the REAL Sanders target the gradient is NaN, so NUTS does not apply
there regardless of these numbers — this compares the three where grads work.
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

POT_DIM, NUIS_DIM = 7, 44
NDIM = POT_DIM + NUIS_DIM   # 51
N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 3000
N_SEEDS = 2
GRAD_COST = 2.5    # reverse-mode grad ~ 2.5x a function eval (rough)
N_EXP, N_SHR = 2, 8   # scale_aware fixed-slice budget


def iso_target():
    cov = np.eye(NDIM)
    prec = jnp.asarray(np.linalg.inv(cov))
    @jax.jit
    def lp(x):
        return -0.5 * x @ prec @ x
    return lp, cov


def corr_target(alpha=0.6):
    M = np.eye(NDIM)
    for j in range(NUIS_DIM):
        M[POT_DIM + j, j % POT_DIM] = alpha
        M[POT_DIM + j, POT_DIM + j] = np.sqrt(1 - alpha**2)
    cov = M @ M.T
    prec = jnp.asarray(np.linalg.inv(cov))
    @jax.jit
    def lp(x):
        return -0.5 * x @ prec @ x
    return lp, cov


def cfg_nuts():
    return VariantConfig(name="nuts", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="nuts", check_nans=False,
        ensemble_kwargs={"mass_type": "dense", "max_tree_depth": 10,
                         "target_accept": 0.8, "step_size0": 0.5})


def cfg_slice():
    return VariantConfig(name="slice", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard", check_nans=False,
        width_kwargs={"scale_factor": 1.0}, ensemble_kwargs={})


def cfg_mh():
    return VariantConfig(name="bgmhdr", direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs", check_nans=False,
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [NDIM], "use_mh": True,
                         "delayed_rejection": True, "complementary_prob": 0.5,
                         "lp_dead": -1e6})


def evals_for(sampler, res):
    """(grad_evals, func_evals, eval_equiv) for the run."""
    if sampler == "nuts":
        ge = res["diagnostics"]["grad_evals"]
        return ge, 0, ge * GRAD_COST
    if sampler == "slice":
        fe = N_PROD * N_WALKERS * (N_EXP + N_SHR + 1)
        return 0, fe, fe
    # bg_MH+DR 1 block, stage1 + DR
    fe = N_PROD * N_WALKERS * (1 * 2)
    return 0, fe, fe


SAMPLERS = [("nuts", cfg_nuts), ("slice", cfg_slice), ("bgmhdr", cfg_mh)]
TARGETS = [("isotropic", iso_target), ("correlated", corr_target)]

print(f"\n{'=' * 104}")
print(f"  NUTS vs SLICE vs bg_MH+DR — {NDIM}D, {N_WALKERS} walkers, "
      f"{N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds")
print(f"{'=' * 104}")
hdr = (f"  {'target':<11s} | {'sampler':<8s} | {'cov-truth':>9s} | {'ESS_min':>8s} | "
       f"{'ESS/s':>7s} | {'ESS/1k-eveq':>11s} | {'wall':>6s}")
print(hdr); print("  " + "-" * (len(hdr) - 2))

for tname, tfn in TARGETS:
    log_prob, cov = tfn()
    for sname, cfgfn in SAMPLERS:
        ess_l, fro_l, eveq_l, wall_l, esps_l = [], [], [], [], []
        for seed in range(N_SEEDS):
            init = jax.random.normal(jax.random.PRNGKey(seed), (N_WALKERS, NDIM),
                                     dtype=jnp.float64)
            t0 = time.time()
            res = run_variant(log_prob, init, n_steps=N_WARMUP + N_PROD,
                              n_warmup=N_WARMUP, config=cfgfn(),
                              key=jax.random.PRNGKey(seed * 1000), verbose=False)
            wall = time.time() - t0
            samples = np.array(res["samples"])
            ess = float(compute_ess(samples).min())
            flat = samples.reshape(-1, NDIM)
            fro = np.linalg.norm(np.cov(flat, rowvar=False) - cov) / np.linalg.norm(cov)
            _, _, eveq = evals_for(sname, res)
            ess_l.append(ess); fro_l.append(fro); eveq_l.append(eveq)
            wall_l.append(wall); esps_l.append(ess / wall)
        ess_m = np.mean(ess_l); fro_m = np.mean(fro_l)
        eveq_m = np.mean(eveq_l); wall_m = np.mean(wall_l); esps_m = np.mean(esps_l)
        ess_per_1k_eveq = ess_m / eveq_m * 1000
        print(f"  {tname:<11s} | {sname:<8s} | {fro_m*100:>8.1f}% | {ess_m:>8.0f} | "
              f"{esps_m:>7.0f} | {ess_per_1k_eveq:>11.2f} | {wall_m:>5.1f}s", flush=True)
    print("  " + "-" * (len(hdr) - 2))

print("\n  ESS/1k-eveq = ESS per 1000 eval-equivalents (grad counted as "
      f"{GRAD_COST}x a function eval). This is the cost metric that extrapolates")
print("  to expensive targets. ESS/s is the practical GPU metric on cheap targets.")
print("  NB: on the real Sanders target the gradient is NaN -> NUTS N/A there.")
