#!/usr/bin/env python
"""Diagnostic: trace plots for cp=0.0, 0.5, 1.0 on stale-Z scenario.

Saves samples to .npy and generates 4 diagnostic plots:
  1. Log-prob traces (per walker) over time
  2. Worst-dim position traces (per walker)
  3. Per-dim ESS bar chart
  4. Autocorrelation for worst dim
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.benchmark.metrics import compute_ess

print(f"JAX devices: {jax.devices()}")

# Same target as bench_complementary_stale.py
NDIM = 21
rng = np.random.default_rng(42)
A = rng.standard_normal((NDIM, NDIM))
Q, _ = np.linalg.qr(A)
evals = np.linspace(1.0, 50.0, NDIM)
cov = Q @ np.diag(evals) @ Q.T
cov = (cov + cov.T) / 2
prec = jnp.array(np.linalg.inv(cov), dtype=jnp.float64)
true_mean = jnp.zeros(NDIM, dtype=jnp.float64)


@jax.jit
def log_prob(x):
    d = x - true_mean
    return -0.5 * d @ prec @ d


def make_init(seed=0):
    """Walkers initialized FAR from posterior mode."""
    key = jax.random.PRNGKey(seed)
    bad_axis = jnp.array(Q[:, -1], dtype=jnp.float64)
    offset = 5.0 * jnp.sqrt(evals[-1]) * bad_axis
    scatter = jax.random.normal(key, (32, NDIM), dtype=jnp.float64) * 0.1
    return offset[None, :] + scatter


def make_config(cp):
    return VariantConfig(
        name=f"bg_mh_dr_cp{cp}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={
            "block_sizes": [7, 14],
            "use_mh": True,
            "delayed_rejection": True,
            "complementary_prob": cp,
        },
    )


CPs = [0.0, 0.5, 1.0]
N_WARMUP = 500
N_PROD = 3000
SEED = 0

results = {}
for cp in CPs:
    print(f"Running cp={cp}...")
    config = make_config(cp)
    init = make_init(SEED)
    t0 = time.time()
    result = run_variant(
        log_prob, init, n_steps=N_WARMUP + N_PROD,
        config=config, n_warmup=N_WARMUP,
        key=jax.random.PRNGKey(SEED * 1000), verbose=False,
    )
    wall = time.time() - t0
    samples = np.array(result["samples"])      # (n_prod, n_walkers, ndim)
    log_probs = np.array(result["log_prob"])   # (n_prod, n_walkers)
    ess_per_dim = compute_ess(samples)
    results[cp] = {
        "samples": samples, "log_probs": log_probs, "wall": wall,
        "ess_per_dim": ess_per_dim, "mu": float(result["mu"]),
    }
    print(f"  cp={cp}: wall={wall:.1f}s, mu={float(result['mu']):.3f}, "
          f"ess_min={ess_per_dim.min():.1f}, ess_mean={ess_per_dim.mean():.1f}")


# ── Plot 1: log-prob traces (one panel per cp) ──────────────────────────

fig, axes = plt.subplots(len(CPs), 1, figsize=(12, 3 * len(CPs)), sharex=True)
for ax, cp in zip(axes, CPs):
    lp = results[cp]["log_probs"]   # (n_prod, n_walkers)
    for w in range(lp.shape[1]):
        ax.plot(lp[:, w], alpha=0.3, lw=0.5)
    ax.axhline(-NDIM/2, color="red", ls="--", lw=1, label="true posterior mean lp")
    ax.set_ylabel(f"log_prob\n(cp={cp})")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, N_PROD)
axes[-1].set_xlabel("Production step")
fig.suptitle("Log-prob traces per walker (production)")
fig.tight_layout()
fig.savefig("diag_logprob_traces.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print("Saved diag_logprob_traces.png")


# ── Plot 2: worst-dim position traces ───────────────────────────────────

# Find worst dim (smallest ESS) for cp=0.0 — same dim across all cps
worst_dim = int(results[0.0]["ess_per_dim"].argmin())
print(f"Worst-mixing dim (for cp=0.0): {worst_dim}")

fig, axes = plt.subplots(len(CPs), 1, figsize=(12, 3 * len(CPs)), sharex=True, sharey=True)
for ax, cp in zip(axes, CPs):
    samples = results[cp]["samples"]     # (n_prod, n_walkers, ndim)
    for w in range(samples.shape[1]):
        ax.plot(samples[:, w, worst_dim], alpha=0.3, lw=0.5)
    ess_d = results[cp]["ess_per_dim"][worst_dim]
    ax.set_ylabel(f"x[{worst_dim}]\n(cp={cp}, ESS={ess_d:.1f})")
    ax.axhline(0.0, color="red", ls="--", lw=1, label="true mean")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, N_PROD)
axes[-1].set_xlabel("Production step")
fig.suptitle(f"Position traces for worst-mixing dim (dim {worst_dim})")
fig.tight_layout()
fig.savefig("diag_worstdim_traces.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print("Saved diag_worstdim_traces.png")


# ── Plot 3: per-dim ESS bar chart ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
width = 0.25
x_pos = np.arange(NDIM)
for i, cp in enumerate(CPs):
    ess = results[cp]["ess_per_dim"]
    ax.bar(x_pos + i * width, ess, width, label=f"cp={cp}", alpha=0.8)
ax.set_xlabel("Dimension")
ax.set_ylabel("ESS")
ax.set_title("Per-dimension ESS — lower = slower mixing")
ax.legend()
ax.axhline(100, color="grey", ls="--", lw=0.5, alpha=0.5, label="ESS=100 (publication threshold)")
ax.set_xticks(x_pos + width)
ax.set_xticklabels([str(i) for i in range(NDIM)])
fig.tight_layout()
fig.savefig("diag_per_dim_ess.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print("Saved diag_per_dim_ess.png")


# ── Plot 4: autocorrelation for worst dim ──────────────────────────────

def autocorr(x, max_lag=200):
    """Autocorrelation function via FFT."""
    n = len(x)
    x = x - x.mean()
    if x.var() < 1e-30:
        return np.zeros(max_lag)
    fx = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fx * np.conj(fx))[:n].real / (n * x.var())
    return acf[:max_lag]


fig, ax = plt.subplots(figsize=(12, 5))
for cp in CPs:
    samples = results[cp]["samples"]
    # ACF of mean-walker chain at worst_dim
    chain = samples[:, :, worst_dim].mean(axis=1)
    acf = autocorr(chain, max_lag=300)
    ax.plot(acf, label=f"cp={cp}")
ax.axhline(0.0, color="grey", lw=0.5)
ax.axhline(0.05, color="red", ls="--", lw=0.5, alpha=0.5, label="ACF=0.05")
ax.set_xlabel("Lag (steps)")
ax.set_ylabel("ACF")
ax.set_title(f"Autocorrelation of mean-walker chain at dim {worst_dim} (worst-mixing)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("diag_autocorr.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print("Saved diag_autocorr.png")

print("\nAll plots saved.")
