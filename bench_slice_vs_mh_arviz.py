#!/usr/bin/env python
"""Slice vs bg_MH+DR with arviz diagnostics — rank R-hat, bulk ESS, tail ESS.

Same target/init as the original bench_slice_vs_mh_whitened.py (51D
isotropic unit Gaussian, sigma=1.0 init, 64 walkers, 1k warmup + 5k prod,
3 seeds). Five configs:
  slice cp=0
  slice cp=0.5
  bg_MH+DR 1blk cp=0.5
  bg_MH+DR 2blk cp=0.5
  bg_MH+DR 2blk Braak (cp=0.5 + sp=0.10 + gj=0.10)

Replaces dezess.compute_ess with arviz:
  - az.rhat(method="rank") — rank-normalized split R-hat (Vehtari et al. 2021)
  - az.ess(method="bulk") — bulk ESS for mean/variance estimation
  - az.ess(method="tail") — tail ESS for quantile estimation (min over 5%/95%)

The treatment of each walker as a separate chain is standard for ensemble
MCMC and matches what Stan/PyMC do.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time, gc, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import jax, jax.numpy as jnp
import numpy as np
import arviz as az
jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig

print(f"JAX devices: {jax.devices()}")
print(f"arviz version: {az.__version__}\n")

NDIM = 51
N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 5000
N_SEEDS = 3
EVALS = {"slice": 4, "bgmh_dr": 2, "bgmh_dr_2blk": 4}   # for ESS/eval calc


@jax.jit
def log_prob(x):
    return -0.5 * jnp.sum(x * x)


def make_init(seed):
    return jax.random.normal(jax.random.PRNGKey(seed), (N_WALKERS, NDIM), dtype=jnp.float64)


def cfg_slice(cp):
    ens = {"complementary_prob": cp} if cp > 0 else {}
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
        "use_mh": True, "delayed_rejection": True,
        "complementary_prob": cp, "lp_dead": -1e6,
    }
    if sp > 0: ens["snooker_prob"] = sp
    if gj > 0: ens["gamma_jump_prob"] = gj
    return VariantConfig(
        name=name or f"bgmh_{block_sizes}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


# (label, config, evals_per_step_per_walker)
CONFIGS = [
    ("slice cp=0",              cfg_slice(0.0),                                                   4),
    ("slice cp=0.5",            cfg_slice(0.5),                                                   4),
    ("bg_MH+DR 1blk cp=0.5",    cfg_bgmhdr([NDIM], 0.5, name="bgmh_1blk"),                        2),
    ("bg_MH+DR 2blk cp=0.5",    cfg_bgmhdr([7, NDIM - 7], 0.5, name="bgmh_2blk"),                 4),
    ("bg_MH+DR 2blk Braak",     cfg_bgmhdr([7, NDIM - 7], 0.5, sp=0.10, gj=0.10, name="braak"),   4),
]


def arviz_diagnostics(samples_3d):
    """samples_3d: (n_steps, n_walkers, n_dim). Returns dict of arviz metrics."""
    chains = np.array(samples_3d).transpose((1, 0, 2))   # (n_walkers, n_steps, n_dim) = (chains, draws, param)
    idata = az.from_dict(posterior={"theta": chains})

    rhat = az.rhat(idata, method="rank")["theta"].values  # (n_dim,)
    ess_b = az.ess(idata, method="bulk")["theta"].values  # (n_dim,)
    ess_t = az.ess(idata, method="tail")["theta"].values  # (n_dim,)

    return {
        "rhat_max": float(rhat.max()),
        "rhat_med": float(np.median(rhat)),
        "ess_bulk_min": float(ess_b.min()),
        "ess_bulk_med": float(np.median(ess_b)),
        "ess_tail_min": float(ess_t.min()),
        "ess_tail_med": float(np.median(ess_t)),
    }


print(f"{'=' * 130}")
print(f"  SLICE vs bg_MH+DR — 51D isotropic, matched init (sigma=1.0)")
print(f"  arviz rank R-hat + bulk/tail ESS  |  {N_WALKERS} walkers, {N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds")
print(f"{'=' * 130}")
hdr = (f"  {'Setup':<24s} | {'seed':>4s} | {'Rhat_max':>8s} | {'Rhat_med':>8s} | "
       f"{'ESS_b_min':>9s} | {'ESS_b_med':>9s} | {'ESS_t_min':>9s} | {'ESS_t_med':>9s} | {'wall':>5s}")
print(hdr); print("  " + "-" * (len(hdr) - 2))

results = {label: [] for label, _, _ in CONFIGS}
for label, cfg, evals_per_step in CONFIGS:
    for seed in range(N_SEEDS):
        init = make_init(seed)
        t0 = time.time()
        result = run_variant(
            log_prob, init, n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
            config=cfg, key=jax.random.PRNGKey(seed * 1000), verbose=False,
        )
        wall = time.time() - t0
        samples = np.array(result["samples"])
        d = arviz_diagnostics(samples)
        d["wall"] = wall
        d["evals_per_step"] = evals_per_step
        results[label].append(d)
        print(f"  {label:<24s} | {seed:>4d} | {d['rhat_max']:>8.4f} | {d['rhat_med']:>8.4f} | "
              f"{d['ess_bulk_min']:>9.1f} | {d['ess_bulk_med']:>9.1f} | "
              f"{d['ess_tail_min']:>9.1f} | {d['ess_tail_med']:>9.1f} | {wall:>4.1f}s", flush=True)
        gc.collect()


print(f"\n\n{'#' * 130}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"{'#' * 130}")
hdr2 = (f"  {'Setup':<24s} | {'Rhat_max':>8s} | {'Rhat_med':>8s} | "
        f"{'ESS_b_min':>9s} | {'ESS_b/eval':>10s} | {'ESS_t_min':>9s} | {'ESS_t/eval':>10s} | {'wall':>6s}")
print(hdr2); print("  " + "-" * (len(hdr2) - 2))
for label, _, _ in CONFIGS:
    rs = results[label]
    rhat_max = np.mean([r["rhat_max"] for r in rs])
    rhat_med = np.mean([r["rhat_med"] for r in rs])
    eb_min = np.mean([r["ess_bulk_min"] for r in rs])
    eb_med = np.mean([r["ess_bulk_med"] for r in rs])
    et_min = np.mean([r["ess_tail_min"] for r in rs])
    et_med = np.mean([r["ess_tail_med"] for r in rs])
    eps = rs[0]["evals_per_step"]
    total_evals = N_PROD * eps * N_WALKERS
    eb_per_eval = eb_min / total_evals * 1e6
    et_per_eval = et_min / total_evals * 1e6
    w = np.mean([r["wall"] for r in rs])
    print(f"  {label:<24s} | {rhat_max:>8.4f} | {rhat_med:>8.4f} | "
          f"{eb_min:>9.1f} | {eb_per_eval:>10.2f} | "
          f"{et_min:>9.1f} | {et_per_eval:>10.2f} | {w:>5.1f}s", flush=True)

print(f"\n  Stan convention: Rhat < 1.01 = converged. ESS_min ≥ 400 recommended.")
print(f"  ESS/eval in millionths (10⁻⁶) — multiply by 1e6 total evals to get raw ESS.")
