"""NUTS (1 chain) vs slice vs bg_MH+DR — matched TOTAL SAMPLES.

NUTS is single-chain by nature (independent walkers, no ensemble coupling);
slice and bg_MH+DR REQUIRE the ensemble. Running NUTS as many short chains
penalizes its ESS estimate (layout artifact). So here NUTS runs as 1 long
chain and the ensemble samplers as 64 walkers, all matched to the SAME total
sample count so cov-vs-truth (layout-robust) is directly comparable.

  NUTS  : 1 chain  x 64000
  slice : 64 walkers x 1000
  bg_MH+DR: 64 walkers x 1000           (all = 64000 total samples)

cov-vs-truth at matched samples is the trustworthy metric (ESS is layout-
confounded across single-chain vs ensemble). ESS/eval also reported.
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
NDIM = POT_DIM + NUIS_DIM
TOTAL = 64000
GRAD_COST = 2.5
N_EXP, N_SHR = 2, 8
N_SEEDS = 2


def iso_target():
    cov = np.eye(NDIM)
    prec = jnp.asarray(np.linalg.inv(cov))
    @jax.jit
    def lp(x): return -0.5 * x @ prec @ x
    return lp, cov


def corr_target(alpha=0.6):
    M = np.eye(NDIM)
    for j in range(NUIS_DIM):
        M[POT_DIM + j, j % POT_DIM] = alpha
        M[POT_DIM + j, POT_DIM + j] = np.sqrt(1 - alpha**2)
    cov = M @ M.T
    prec = jnp.asarray(np.linalg.inv(cov))
    @jax.jit
    def lp(x): return -0.5 * x @ prec @ x
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


# (label, cfg, n_walkers, n_prod)
SAMPLERS = [
    ("nuts(1ch)", cfg_nuts, 1, TOTAL),
    ("slice(64)", cfg_slice, 64, TOTAL // 64),
    ("bgmhdr(64)", cfg_mh, 64, TOTAL // 64),
]
TARGETS = [("isotropic", iso_target), ("correlated", corr_target)]


def eveq_for(name, res, n_walkers, n_prod):
    if name.startswith("nuts"):
        ge = res["diagnostics"]["grad_evals"]
        return ge * GRAD_COST
    if name.startswith("slice"):
        return n_prod * n_walkers * (N_EXP + N_SHR + 1)
    return n_prod * n_walkers * (1 * 2)


print(f"\n{'=' * 100}")
print(f"  NUTS(1 chain) vs SLICE(64) vs bg_MH+DR(64) — {NDIM}D, matched {TOTAL} samples, {N_SEEDS} seeds")
print(f"{'=' * 100}")
hdr = (f"  {'target':<11s} | {'sampler':<11s} | {'cov-truth':>9s} | {'ESS_min':>8s} | "
       f"{'ESS/1k-eveq':>11s} | {'wall':>7s}")
print(hdr); print("  " + "-" * (len(hdr) - 2))

for tname, tfn in TARGETS:
    log_prob, cov = tfn()
    for sname, cfgfn, nw, npr in SAMPLERS:
        fro_l, ess_l, eveq_l, wall_l = [], [], [], []
        for seed in range(N_SEEDS):
            init = jax.random.normal(jax.random.PRNGKey(seed), (nw, NDIM), dtype=jnp.float64)
            t0 = time.time()
            res = run_variant(log_prob, init, n_steps=1000 + npr, n_warmup=1000,
                              config=cfgfn(), key=jax.random.PRNGKey(seed * 1000),
                              verbose=False)
            wall = time.time() - t0
            samples = np.array(res["samples"])
            # ESS: a single chain is undefined for the multi-chain estimator,
            # so split a 1-chain run into pseudo-chains just for the ESS number.
            if samples.shape[1] == 1:
                n = samples.shape[0]
                npc = 8
                seg = n // npc
                ess_in = samples[:seg * npc, 0, :].reshape(npc, seg, NDIM).transpose(1, 0, 2)
            else:
                ess_in = samples
            ess = float(compute_ess(ess_in).min())
            flat = samples.reshape(-1, NDIM)
            fro = np.linalg.norm(np.cov(flat, rowvar=False) - cov) / np.linalg.norm(cov)
            fro_l.append(fro); ess_l.append(ess)
            eveq_l.append(eveq_for(sname, res, nw, npr)); wall_l.append(wall)
        fro_m = np.mean(fro_l); ess_m = np.mean(ess_l)
        eveq_m = np.mean(eveq_l); wall_m = np.mean(wall_l)
        print(f"  {tname:<11s} | {sname:<11s} | {fro_m*100:>8.1f}% | {ess_m:>8.0f} | "
              f"{ess_m/eveq_m*1000:>11.2f} | {wall_m:>5.1f}s", flush=True)
    print("  " + "-" * (len(hdr) - 2))

print("\n  cov-vs-truth at matched samples is the layout-robust metric (ESS differs")
print("  across single-chain NUTS vs 64-walker ensembles). NB: real Sanders grad")
print("  is NaN -> NUTS N/A there; for Sanders it's slice vs bg_MH+DR only.")
