#!/usr/bin/env python
"""Benchmark: snooker_prob sweep on a SANDERS-LIKE surrogate target.

Constructs a 51D Gaussian (7 potential + 4 streams × 11 nuisance) with the
key structural features observed in the real Sanders posterior xcorr plot:

  - log_10 M (x_0) couples to the FIRST nuisance coord of EACH stream
    (x_7, x_18, x_29, x_40) at Pearson ρ = 0.6 — matches the strong
    log_10 M ↔ ω_0/γ_0 correlations seen in production.
  - Other potential coords (x_1..x_6) have ZERO nuisance coupling —
    matches "a_out, e_out, a_in, k look pinned" observation.
  - All marginals are unit Gaussian.
  - Block partition [7, 44] matches user's bg_MH+DR config.

Latent factor construction (Σ = M Mᵀ where z ~ N(0, I_51)):
  x_0 = z_0
  x_j = z_j           for j ∈ {1..6}
  x_{7+11i} = α z_0 + sqrt(1-α²) z_{7+11i}   for i ∈ {0..3}, α = 0.6
  x_{7+11i+j} = z_{7+11i+j}                  for i ∈ {0..3}, j ∈ {1..10}

Sweeps snooker_prob ∈ {0, 0.05, 0.1, 0.2, 0.3} with cp=0.5 fixed (matches user's
production setup). Reports ESS, autocorrelation at lag {50, 100, 200} for the
cross-block-coupled coord x_0, Frobenius cov error, recovered ρ(x_0, x_7),
and lp drift.

Goal: tune snooker_prob and verify it actually helps cross-block mixing on
a target that mimics the real bottleneck.
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

# ── Target construction ─────────────────────────────────────────────────

N_STREAMS = 4
POT_DIM = 7
NUIS_PER_STREAM = 11
NDIM = POT_DIM + N_STREAMS * NUIS_PER_STREAM   # 51
ALPHA = 0.6                                     # cross-block correlation strength

# Indices of the "ω_0-like" coord in each stream (couples to x_0)
COUPLED_NUIS_IDX = [POT_DIM + NUIS_PER_STREAM * i for i in range(N_STREAMS)]
# = [7, 18, 29, 40] for 4 streams

# Build linear transform M such that x = M z, z ~ N(0, I)
M = np.eye(NDIM, dtype=np.float64)
for j in COUPLED_NUIS_IDX:
    M[j, 0] = ALPHA
    M[j, j] = np.sqrt(1.0 - ALPHA ** 2)

# True covariance and precision
true_cov = M @ M.T
prec = np.linalg.inv(true_cov)
prec_jax = jnp.array(prec, dtype=jnp.float64)

# Verify expected correlations
true_corr = true_cov / np.sqrt(np.outer(np.diag(true_cov), np.diag(true_cov)))
print(f"\nTrue correlations Pearson(x_0, x_j) for j in {COUPLED_NUIS_IDX}:")
for j in COUPLED_NUIS_IDX:
    print(f"  ρ(x_0, x_{j}) = {true_corr[0, j]:.4f}  (target = {ALPHA})")
print(f"True Pearson(x_0, x_1) = {true_corr[0, 1]:.4f}  (target = 0)")

@jax.jit
def log_prob(x):
    return -0.5 * x @ prec_jax @ x


def make_init(seed):
    """Tight scatter near origin — chain converges in warmup, then we measure mixing."""
    return jax.random.normal(jax.random.PRNGKey(seed), (N_WALKERS, NDIM), dtype=jnp.float64) * 0.3


def make_cfg(sp):
    ens = {
        "block_sizes": [POT_DIM, N_STREAMS * NUIS_PER_STREAM],
        "use_mh": True,
        "delayed_rejection": True,
        "complementary_prob": 0.5,
    }
    if sp > 0.0:
        ens["snooker_prob"] = sp
    return VariantConfig(
        name=f"bg_mh_dr_cp0.5_sp{sp}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


def autocorr_at_lag(samples_2d, max_lag):
    """ACF of mean-walker chain for each dim, returned at specific lags.

    samples_2d: (n_steps, n_walkers, n_dim) → averaged over walkers.
    Returns ACF[lag, dim] for lag in [1, max_lag].
    """
    n_steps, n_walkers, n_dim = samples_2d.shape
    chain = samples_2d.mean(axis=1)   # (n_steps, n_dim)
    chain = chain - chain.mean(axis=0, keepdims=True)
    var = chain.var(axis=0)
    var = np.where(var < 1e-30, 1e-30, var)
    acf = np.zeros((max_lag + 1, n_dim))
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        c = (chain[:-lag] * chain[lag:]).mean(axis=0) / var
        acf[lag] = c
    return acf


# ── Run sweep ───────────────────────────────────────────────────────────

N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 5000
N_SEEDS = 3
LAGS_REPORT = [50, 100, 200]
SP_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]

print(f"\n{'=' * 100}")
print(f"  SANDERS-LIKE SURROGATE: 51D Gaussian with cross-block ρ={ALPHA}")
print(f"  block partition [{POT_DIM}, {N_STREAMS * NUIS_PER_STREAM}], "
      f"{N_WALKERS} walkers, {N_WARMUP} warmup + {N_PROD} prod, {N_SEEDS} seeds")
print(f"  bg_MH+DR + cp=0.5 sweeping snooker_prob ∈ {SP_VALUES}")
print(f"{'=' * 100}")
hdr = (f"  {'sp':>5s} | {'seed':>4s} | {'ESS_min':>8s} | {'ESS_mn':>7s} | "
       f"{'ACF50':>6s} | {'ACF100':>6s} | {'ACF200':>6s} | "
       f"{'Fro%':>5s} | {'ρ̂':>6s} | {'Δlp':>6s} | {'wall':>6s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

results = {sp: [] for sp in SP_VALUES}
for sp in SP_VALUES:
    cfg = make_cfg(sp)
    for seed in range(N_SEEDS):
        init = make_init(seed)
        t0 = time.time()
        result = run_variant(
            log_prob, init, n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
            config=cfg, key=jax.random.PRNGKey(seed * 1000), verbose=False,
        )
        wall = time.time() - t0
        samples = np.array(result["samples"])               # (n_prod, n_walkers, ndim)
        flat = samples.reshape(-1, NDIM)
        ess = compute_ess(samples)
        emp_cov = np.cov(flat, rowvar=False)
        fro_rel = np.linalg.norm(emp_cov - true_cov) / np.linalg.norm(true_cov)
        emp_std = np.sqrt(np.diag(emp_cov))
        rho_hat = emp_cov[0, COUPLED_NUIS_IDX[0]] / (emp_std[0] * emp_std[COUPLED_NUIS_IDX[0]])
        # ACF for x_0 (cross-block-coupled dim, where snooker should matter most)
        acf = autocorr_at_lag(samples, max_lag=max(LAGS_REPORT))
        acf_x0 = acf[:, 0]
        # lp drift: slope over last half of production
        lp_arr = np.array(result["log_prob"])               # (n_prod, n_walkers)
        lp_mean = lp_arr.mean(axis=1)
        half = len(lp_mean) // 2
        slope = np.polyfit(np.arange(half), lp_mean[half:], 1)[0]
        # Drift across full prod range, scaled
        delta_lp = lp_mean[-100:].mean() - lp_mean[:100].mean()

        results[sp].append({
            "ess_min": float(ess.min()), "ess_mean": float(ess.mean()),
            "acf": [float(acf_x0[L]) for L in LAGS_REPORT],
            "fro_rel": float(fro_rel), "rho_hat": float(rho_hat),
            "delta_lp": float(delta_lp), "wall": wall,
        })
        print(f"  {sp:>5.2f} | {seed:>4d} | {float(ess.min()):>8.1f} | "
              f"{float(ess.mean()):>7.1f} | "
              f"{acf_x0[LAGS_REPORT[0]]:>6.3f} | "
              f"{acf_x0[LAGS_REPORT[1]]:>6.3f} | "
              f"{acf_x0[LAGS_REPORT[2]]:>6.3f} | "
              f"{fro_rel*100:>4.1f}% | {rho_hat:>6.3f} | "
              f"{delta_lp:>6.2f} | {wall:>5.1f}s", flush=True)
        gc.collect()


# ── Summary ─────────────────────────────────────────────────────────────

print(f"\n{'=' * 100}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"{'=' * 100}")
hdr2 = (f"  {'sp':>5s} | {'ESS_min':>8s} | {'ESS_mn':>7s} | "
        f"{'ACF50':>6s} | {'ACF100':>6s} | {'ACF200':>6s} | "
        f"{'Fro%':>5s} | {'ρ̂':>6s} | {'wall':>7s}")
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))
for sp in SP_VALUES:
    runs = results[sp]
    em_min = np.mean([r["ess_min"] for r in runs])
    em_mn = np.mean([r["ess_mean"] for r in runs])
    acf_50 = np.mean([r["acf"][0] for r in runs])
    acf_100 = np.mean([r["acf"][1] for r in runs])
    acf_200 = np.mean([r["acf"][2] for r in runs])
    fro = np.mean([r["fro_rel"] for r in runs])
    rho = np.mean([r["rho_hat"] for r in runs])
    w = np.mean([r["wall"] for r in runs])
    print(f"  {sp:>5.2f} | {em_min:>8.1f} | {em_mn:>7.1f} | "
          f"{acf_50:>6.3f} | {acf_100:>6.3f} | {acf_200:>6.3f} | "
          f"{fro*100:>4.1f}% | {rho:>6.3f} | {w:>6.1f}s", flush=True)

print()
print(f"  Read the table:")
print(f"    - ESS_min higher = better mixing in worst dim")
print(f"    - ACF at lag L closer to 0 = faster autocorrelation decay (faster mixing)")
print(f"    - Fro% = posterior recovery error (lower = correct)")
print(f"    - ρ̂ = recovered cross-block correlation (target {ALPHA})")
print(f"    - If snooker helps: expect lower ACF, possibly higher ESS, similar Fro/ρ̂")
print(f"    - If snooker doesn't help: ACF/ESS roughly unchanged or worse")
