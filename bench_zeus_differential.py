#!/usr/bin/env python
"""zeus DifferentialMove vs dezess slice on the cross-block correlation sweep.

Same 51D target as bench_correlation_sweep_v2.py: [7, 44] block structure,
each nuis dim j correlated with pot dim (j mod POT_DIM) at level ALPHA.
Init matched to target (N(0, Σ)).

Reports per-step ESS, ACF50/200/1000, cov err, wall — so we can compare to
the dezess slice results from v2 on per-step mixing terms (wall is unfair:
zeus is CPU-driven, dezess is GPU-vmap'd).
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time, gc
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

import zeus

print(f"zeus {zeus.__version__}")
print(f"JAX devices: {jax.devices()}")

POT_DIM = 7
NUIS_DIM = 44
NDIM = POT_DIM + NUIS_DIM
N_WALKERS = 128   # zeus requires ≥ 2*ndim walkers (= 102 for 51D)
N_WARMUP = 1000
N_PROD = 20000
N_SEEDS = 3
LAGS = [50, 200, 1000]
ALPHAS = [0.3, 0.6, 0.9]


def build_target(alpha):
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
    return np.array(z @ jnp.asarray(M).T)


def make_log_prob(prec):
    prec_jax = jnp.asarray(prec)

    @jax.jit
    def log_prob_vec(x):
        # x: (nwalkers, ndim) — vectorized
        return -0.5 * jnp.sum((x @ prec_jax) * x, axis=-1)

    def log_prob_np(x):
        return np.asarray(log_prob_vec(jnp.asarray(x)))

    return log_prob_np


def autocorr_at_lags(samples_3d, lags):
    """samples_3d: (n_steps, n_walkers, n_dim)"""
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


def split_rhat(samples_3d):
    n_steps, n_walkers, n_dim = samples_3d.shape
    half = n_steps // 2
    chains = np.stack([samples_3d[:half], samples_3d[half:2*half]], axis=0)
    chains = chains.reshape(2 * n_walkers, half, n_dim)
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    W = chain_vars.mean(axis=0)
    B = half * chain_means.var(axis=0, ddof=1)
    V_hat = (half - 1) / half * W + B / half
    return np.sqrt(np.maximum(V_hat / np.maximum(W, 1e-30), 0.0))


def compute_ess_split(samples_3d):
    """Standard ESS using split-chain autocorrelation; matches dezess metric."""
    n_steps, n_walkers, n_dim = samples_3d.shape
    # Per-walker integrated ACT, then ESS = n_steps * n_walkers / mean_tau
    # Simple approach: ESS per dim = n_steps * n_walkers / (1 + 2 * sum positive ACF)
    ess = np.zeros(n_dim)
    for d in range(n_dim):
        for w in range(n_walkers):
            chain = samples_3d[:, w, d]
            chain = chain - chain.mean()
            var = chain.var() + 1e-30
            tau = 1.0
            for lag in range(1, min(200, n_steps // 4)):
                rho = (chain[:-lag] * chain[lag:]).mean() / var
                if rho < 0.05:
                    break
                tau += 2 * rho
            ess[d] += n_steps / tau
    return ess


all_results = {}
for alpha in ALPHAS:
    true_cov, M = build_target(alpha)
    prec = np.linalg.inv(true_cov)
    log_prob_fn = make_log_prob(prec)

    sd = np.sqrt(np.diag(true_cov))
    rho_emp = np.abs(true_cov[:POT_DIM, POT_DIM:] / np.outer(sd[:POT_DIM], sd[POT_DIM:])).max()

    print(f"\n{'=' * 110}")
    print(f"  ZEUS DifferentialMove — α = {alpha}  (max |cross-block corr|: {rho_emp:.3f})")
    print(f"  {NDIM}D, blocks=[{POT_DIM}, {NUIS_DIM}], {N_WALKERS} walkers, "
          f"{N_WARMUP}+{N_PROD} steps, {N_SEEDS} seeds")
    print(f"{'=' * 110}")
    hdr = (f"  {'seed':>4s} | {'ESS_min':>8s} | {'ESS/s':>6s} | "
           f"{'ACF50':>6s} | {'ACF200':>6s} | {'ACF1k':>6s} | "
           f"{'Rhat_max':>8s} | {'Fro%':>5s} | {'wall':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    runs = []
    for seed in range(N_SEEDS):
        init = make_init(seed, M)
        np.random.seed(seed * 1000)

        # zeus EnsembleSampler with DifferentialMove only
        sampler = zeus.EnsembleSampler(
            nwalkers=N_WALKERS, ndim=NDIM, logprob_fn=log_prob_fn,
            moves=zeus.moves.DifferentialMove(),
            vectorize=True, verbose=False, light_mode=True,
        )

        t0 = time.time()
        sampler.run_mcmc(init, N_WARMUP + N_PROD, progress=False)
        wall = time.time() - t0

        # Discard warmup
        chain = sampler.get_chain(discard=N_WARMUP)   # (N_PROD, N_WALKERS, NDIM)
        flat = chain.reshape(-1, NDIM)
        emp_cov = np.cov(flat, rowvar=False)
        fro_rel = np.linalg.norm(emp_cov - true_cov) / np.linalg.norm(true_cov)
        acf = autocorr_at_lags(chain, LAGS)
        acf_x0 = acf[:, 0]
        rhat = split_rhat(chain)
        ess = compute_ess_split(chain)
        ess_min = float(ess.min())
        ess_per_s = ess_min / max(wall, 1e-9)

        runs.append({
            "ess_min": ess_min,
            "ess_per_s": ess_per_s,
            "acf": [float(v) for v in acf_x0],
            "fro_rel": float(fro_rel),
            "rhat_max": float(rhat.max()),
            "wall": wall,
        })
        print(f"  {seed:>4d} | {ess_min:>8.1f} | {ess_per_s:>6.2f} | "
              f"{acf_x0[0]:>6.3f} | {acf_x0[1]:>6.3f} | {acf_x0[2]:>6.3f} | "
              f"{float(rhat.max()):>8.4f} | "
              f"{fro_rel*100:>4.1f}% | {wall:>5.1f}s", flush=True)
        gc.collect()

    all_results[alpha] = runs


print(f"\n\n{'#' * 110}")
print(f"  ZEUS SUMMARY (mean across {N_SEEDS} seeds)")
print(f"{'#' * 110}")
hdr2 = (f"  {'α':>4s} | {'ESS_min':>8s} | {'ESS/s':>6s} | "
        f"{'ACF50':>6s} | {'ACF200':>6s} | {'ACF1k':>6s} | "
        f"{'Rhat_max':>8s} | {'Fro%':>5s} | {'wall':>7s}")
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))
for alpha in ALPHAS:
    runs = all_results[alpha]
    em_min = np.mean([r["ess_min"] for r in runs])
    eps = np.mean([r["ess_per_s"] for r in runs])
    a50 = np.mean([r["acf"][0] for r in runs])
    a200 = np.mean([r["acf"][1] for r in runs])
    a1k = np.mean([r["acf"][2] for r in runs])
    rh = np.mean([r["rhat_max"] for r in runs])
    fro = np.mean([r["fro_rel"] for r in runs])
    w = np.mean([r["wall"] for r in runs])
    print(f"  {alpha:>4.1f} | {em_min:>8.1f} | {eps:>6.2f} | "
          f"{a50:>6.3f} | {a200:>6.3f} | {a1k:>6.3f} | "
          f"{rh:>8.4f} | {fro*100:>4.1f}% | {w:>6.1f}s", flush=True)

print(f"\n  Compare these per-step ACF/ESS_min/Fro values against dezess slice cp=0")
print(f"  from bench_correlation_sweep_v2 (job 127201) on the same targets.")
