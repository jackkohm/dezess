#!/usr/bin/env python
"""Slice vs bg_MH+DR with arviz diagnostics + visual artifacts.

Same 51D isotropic, matched init, 5-config setup. Extends the arviz bench
with:
  - Corner plots (first 6 dims, seed 0 only)
  - Trace plots (first 4 dims, all walkers, seed 0 only)
  - Explicit B (between-walker) / W (within-walker) variance decomposition
    per config — exposes whether walkers AGREE on the posterior (small B/W)
    or DISAGREE (large B/W → high R-hat).

All PNG artifacts saved to artifacts/arviz_bench/ and listed at end so
they're easy to pull back.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time, gc, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax, jax.numpy as jnp
import numpy as np
import arviz as az
jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig

print(f"JAX devices: {jax.devices()}")
print(f"arviz version: {az.__version__}")

NDIM = 51
N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 5000
N_SEEDS = 3

ARTIFACTS_DIR = "artifacts/arviz_bench"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
print(f"Saving plots to {ARTIFACTS_DIR}/\n")

CORNER_DIMS = [0, 1, 7, 8, 30, 50]   # spread across blocks
TRACE_DIMS = [0, 7, 30, 50]
TRACE_WALKERS = list(range(8))


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


CONFIGS = [
    ("slice_cp0",       "slice cp=0",           cfg_slice(0.0),                                                   4),
    ("slice_cp05",      "slice cp=0.5",         cfg_slice(0.5),                                                   4),
    ("bgmh_1blk",       "bg_MH+DR 1blk cp=0.5", cfg_bgmhdr([NDIM], 0.5, name="bgmh_1blk"),                        2),
    ("bgmh_2blk",       "bg_MH+DR 2blk cp=0.5", cfg_bgmhdr([7, NDIM - 7], 0.5, name="bgmh_2blk"),                 4),
    ("bgmh_2blk_braak", "bg_MH+DR 2blk Braak",  cfg_bgmhdr([7, NDIM - 7], 0.5, sp=0.10, gj=0.10, name="braak"),   4),
]


def diagnose(samples_3d):
    """samples_3d: (n_steps, n_walkers, n_dim). Returns full diagnostic dict."""
    chains = np.array(samples_3d).transpose((1, 0, 2))         # (walkers, steps, dim)
    idata = az.from_dict({"theta": chains[None]})

    rhat = az.rhat(idata, method="rank")["theta"].values
    ess_b = az.ess(idata, method="bulk")["theta"].values
    ess_t = az.ess(idata, method="tail")["theta"].values

    # B / W decomposition per dim (Gelman-Rubin original)
    n_walkers, n_steps, n_dim = chains.shape
    chain_means = chains.mean(axis=1)          # (walkers, dim)
    chain_vars = chains.var(axis=1, ddof=1)    # (walkers, dim)
    W = chain_vars.mean(axis=0)                # within-chain var, per dim
    B = n_steps * chain_means.var(axis=0, ddof=1)   # between-chain × n, per dim

    return {
        "rhat_max": float(rhat.max()), "rhat_med": float(np.median(rhat)),
        "ess_bulk_min": float(ess_b.min()), "ess_bulk_med": float(np.median(ess_b)),
        "ess_tail_min": float(ess_t.min()), "ess_tail_med": float(np.median(ess_t)),
        "W_med": float(np.median(W)), "W_max": float(W.max()),
        "B_med": float(np.median(B)), "B_max": float(B.max()),
        "BW_ratio_med": float(np.median(B / np.maximum(W, 1e-30))),
        "BW_ratio_max": float((B / np.maximum(W, 1e-30)).max()),
    }


def plot_corner(samples_3d, dims, title, fname):
    """Manual corner plot — scatter + hexbin for the upper triangle."""
    flat = np.array(samples_3d).reshape(-1, samples_3d.shape[-1])
    n = len(dims)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    for i, di in enumerate(dims):
        for j, dj in enumerate(dims):
            ax = axes[i, j]
            if i == j:
                ax.hist(flat[:, di], bins=50, density=True, color="steelblue", alpha=0.7)
                ax.set_ylabel("")
            elif i > j:
                # 2D hexbin to spot speckle
                ax.hexbin(flat[:, dj], flat[:, di], gridsize=40, cmap="Blues", mincnt=1)
            else:
                # scatter (smaller sample) to ALSO see speckle directly
                idx = np.random.RandomState(0).choice(flat.shape[0], size=min(3000, flat.shape[0]), replace=False)
                ax.scatter(flat[idx, dj], flat[idx, di], s=0.5, alpha=0.3, color="black")
            if i == n - 1:
                ax.set_xlabel(f"d{dj}")
            if j == 0:
                ax.set_ylabel(f"d{di}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=80)
    plt.close(fig)


def plot_traces(samples_3d, dims, walkers, title, fname):
    """Trace plot — per-dim, several walkers overlaid."""
    arr = np.array(samples_3d)   # (steps, walkers, dim)
    n_steps = arr.shape[0]
    n_dim = len(dims)
    fig, axes = plt.subplots(n_dim, 1, figsize=(11, 2.0 * n_dim))
    if n_dim == 1: axes = [axes]
    for ax, d in zip(axes, dims):
        for w in walkers:
            ax.plot(arr[:, w, d], lw=0.4, alpha=0.6)
        ax.set_ylabel(f"d{d}")
        ax.set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=80)
    plt.close(fig)


# --- Run benches ---
print(f"\n{'=' * 130}")
print(f"  SLICE vs bg_MH+DR — arviz diagnostics + B/W decomposition + corner/trace plots")
print(f"  {NDIM}D isotropic, matched init, {N_WALKERS} walkers, {N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds")
print(f"{'=' * 130}")
hdr = (f"  {'Setup':<24s} | {'seed':>4s} | {'Rhat_max':>8s} | "
       f"{'ESS_b_min':>9s} | {'ESS_t_min':>9s} | "
       f"{'B/W_med':>8s} | {'B/W_max':>8s} | {'wall':>5s}")
print(hdr); print("  " + "-" * (len(hdr) - 2))

results = {label: [] for _, label, _, _ in CONFIGS}
for slug, label, cfg, evals in CONFIGS:
    for seed in range(N_SEEDS):
        init = make_init(seed)
        t0 = time.time()
        result = run_variant(
            log_prob, init, n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
            config=cfg, key=jax.random.PRNGKey(seed * 1000), verbose=False,
        )
        wall = time.time() - t0
        samples = np.array(result["samples"])
        d = diagnose(samples)
        d["wall"] = wall; d["evals_per_step"] = evals
        results[label].append(d)
        print(f"  {label:<24s} | {seed:>4d} | {d['rhat_max']:>8.4f} | "
              f"{d['ess_bulk_min']:>9.1f} | {d['ess_tail_min']:>9.1f} | "
              f"{d['BW_ratio_med']:>8.4f} | {d['BW_ratio_max']:>8.4f} | {wall:>4.1f}s",
              flush=True)

        # On seed 0 only, generate the visual artifacts
        if seed == 0:
            plot_corner(samples, CORNER_DIMS,
                        f"{label} — corner (dims {CORNER_DIMS})",
                        f"{ARTIFACTS_DIR}/corner_{slug}.png")
            plot_traces(samples, TRACE_DIMS, TRACE_WALKERS,
                        f"{label} — traces (dims {TRACE_DIMS}, walkers 0-7)",
                        f"{ARTIFACTS_DIR}/trace_{slug}.png")
            print(f"    → {ARTIFACTS_DIR}/corner_{slug}.png")
            print(f"    → {ARTIFACTS_DIR}/trace_{slug}.png", flush=True)
        gc.collect()


print(f"\n\n{'#' * 130}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"{'#' * 130}")
hdr2 = (f"  {'Setup':<24s} | {'Rhat_max':>8s} | {'Rhat_med':>8s} | "
        f"{'ESS_b_min':>9s} | {'ESS_b/eval':>10s} | {'ESS_t_min':>9s} | {'ESS_t/eval':>10s} | "
        f"{'B/W_med':>8s} | {'W_med':>7s} | {'B_med':>7s}")
print(hdr2); print("  " + "-" * (len(hdr2) - 2))
for _, label, _, _ in CONFIGS:
    rs = results[label]
    rhat_max = np.mean([r["rhat_max"] for r in rs])
    rhat_med = np.mean([r["rhat_med"] for r in rs])
    eb_min = np.mean([r["ess_bulk_min"] for r in rs])
    eb_med = np.mean([r["ess_bulk_med"] for r in rs])
    et_min = np.mean([r["ess_tail_min"] for r in rs])
    et_med = np.mean([r["ess_tail_med"] for r in rs])
    W_med = np.mean([r["W_med"] for r in rs])
    B_med = np.mean([r["B_med"] for r in rs])
    bw_med = np.mean([r["BW_ratio_med"] for r in rs])
    eps = rs[0]["evals_per_step"]
    total_evals = N_PROD * eps * N_WALKERS
    eb_per_eval = eb_min / total_evals * 1e6
    et_per_eval = et_min / total_evals * 1e6
    print(f"  {label:<24s} | {rhat_max:>8.4f} | {rhat_med:>8.4f} | "
          f"{eb_min:>9.1f} | {eb_per_eval:>10.2f} | "
          f"{et_min:>9.1f} | {et_per_eval:>10.2f} | "
          f"{bw_med:>8.4f} | {W_med:>7.3f} | {B_med:>7.3f}")

print(f"\n  Stan: Rhat<1.01, ESS_min≥400. ESS/eval in ×10⁻⁶ units.")
print(f"  B/W ratio: small → walkers agree on posterior; large → walkers disagree → high Rhat.")
print(f"  W ≈ 1.0 expected (unit Gaussian within-walker variance). B should be small (chains agree on mean).")
print(f"\n  Saved {len(CONFIGS) * 2} PNGs to {ARTIFACTS_DIR}/")
