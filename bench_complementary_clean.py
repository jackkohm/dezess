#!/usr/bin/env python
"""Head-to-head: scale_aware (standard ensemble + de_mcz + scale_aware width
+ fixed slice + circular zmatrix) at cp ∈ {0.0, 0.5, 1.0} on CLEAN targets
(no stale-Z stress — well-initialized, standard warmup).

Answers: is cp=0.5 actually a win on regular targets, or only when Z-matrix is biased?

5 seeds × 3 cp values × 3 targets = 45 runs, ~15 min on H200.
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
from dezess.benchmark.metrics import compute_ess, compute_rhat

print(f"JAX devices: {jax.devices()}")

N_WALKERS = 64        # 32/half spans 31D — enough for 21D target at cp=1.0
N_WARMUP = 500
N_PROD = 3000
N_SEEDS = 5
CP_VALUES = [0.0, 0.5, 1.0]

# ── Targets ─────────────────────────────────────────────────────────────

def build_isotropic(ndim):
    @jax.jit
    def lp(x):
        return -0.5 * jnp.sum(x ** 2)
    true_var = np.ones(ndim)
    return lp, true_var, ndim


def build_anisotropic(ndim, seed=42, cond=50.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((ndim, ndim))
    Q, _ = np.linalg.qr(A)
    evals = np.linspace(1.0, cond, ndim)
    cov = Q @ np.diag(evals) @ Q.T
    cov = (cov + cov.T) / 2
    prec = jnp.array(np.linalg.inv(cov), dtype=jnp.float64)

    @jax.jit
    def lp(x):
        return -0.5 * x @ prec @ x

    return lp, np.diag(cov), ndim


TARGETS = [
    ("21D isotropic",           build_isotropic(21)),
    ("21D aniso cond=50",       build_anisotropic(21)),
    ("5D aniso cond=50",        build_anisotropic(5)),
]


# ── Config ──────────────────────────────────────────────────────────────

def make_config(cp):
    ens_kwargs = {}
    if cp > 0.0:
        ens_kwargs["complementary_prob"] = cp
    return VariantConfig(
        name=f"scale_aware_cp{cp}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )


# ── Init: tight scatter near mode (clean — no stale-Z) ─────────────────

def make_init(seed, ndim):
    """Tight Gaussian scatter near origin. Warmup converges quickly; this is
    the "chain mixing" regime, not the "chain climbing" regime.
    """
    return jax.random.normal(jax.random.PRNGKey(seed), (N_WALKERS, ndim)) * 0.5


# ── Run sweep ───────────────────────────────────────────────────────────

print(f"\n{'=' * 100}")
print(f"  CLEAN TARGET SWEEP — scale_aware × cp ∈ {{0.0, 0.5, 1.0}}")
print(f"  {N_WALKERS} walkers, {N_WARMUP} warmup + {N_PROD} prod, {N_SEEDS} seeds × 3 cp × 3 targets = {N_SEEDS*3*3} runs")
print(f"{'=' * 100}")

# Per-target results: results[target_name][cp] = list of seed dicts
results = {tname: {cp: [] for cp in CP_VALUES} for tname, _ in TARGETS}

for tname, (lp, true_var, ndim) in TARGETS:
    print(f"\n── Target: {tname} ──")
    hdr = f"  {'cp':>5s} | {'seed':>4s} | {'R-hat':>6s} | {'ESS min':>8s} | {'ESS mean':>9s} | {'wall':>6s} | {'var_err':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cp in CP_VALUES:
        cfg = make_config(cp)
        for seed in range(N_SEEDS):
            init = make_init(seed, ndim)
            t0 = time.time()
            result = run_variant(
                lp, init, n_steps=N_WARMUP + N_PROD,
                config=cfg, n_warmup=N_WARMUP,
                key=jax.random.PRNGKey(seed * 1000), verbose=False,
            )
            wall = time.time() - t0
            samples = np.array(result["samples"])  # (n_prod, n_walkers, ndim)
            ess = compute_ess(samples)
            rhat = compute_rhat(samples)
            emp_var = samples.reshape(-1, ndim).var(axis=0)
            var_err = float(np.abs(emp_var - true_var).max() / true_var.max())

            results[tname][cp].append({
                "rhat": float(rhat.max()),
                "ess_min": float(ess.min()),
                "ess_mean": float(ess.mean()),
                "wall": wall,
                "var_err": var_err,
            })
            print(f"  {cp:>5.2f} | {seed:>4d} | {float(rhat.max()):6.3f} | "
                  f"{float(ess.min()):8.1f} | {float(ess.mean()):9.1f} | "
                  f"{wall:5.1f}s | {var_err:7.2%}", flush=True)
            gc.collect()


# ── Summary tables ──────────────────────────────────────────────────────

print(f"\n{'=' * 100}")
print(f"  SUMMARY (mean ± std across {N_SEEDS} seeds)")
print(f"{'=' * 100}")

for tname, (_, _, ndim) in TARGETS:
    print(f"\n── {tname} ──")
    hdr2 = f"  {'cp':>5s} | {'R-hat (mean)':>13s} | {'ESS_min (mean ± std)':>22s} | {'ESS_mean':>10s} | {'var_err':>9s}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for cp in CP_VALUES:
        runs = results[tname][cp]
        rhat_m = np.mean([r["rhat"] for r in runs])
        em_m = np.mean([r["ess_min"] for r in runs])
        em_s = np.std([r["ess_min"] for r in runs])
        ea_m = np.mean([r["ess_mean"] for r in runs])
        ve_m = np.mean([r["var_err"] for r in runs])
        print(f"  {cp:>5.2f} | {rhat_m:13.3f} | {em_m:8.1f} ± {em_s:6.1f}        | "
              f"{ea_m:10.1f} | {ve_m:8.2%}", flush=True)


print(f"\n{'=' * 100}")
print(f"  INTERPRETATION")
print(f"{'=' * 100}")
print(f"  - Higher ESS_min = better mixing in slowest dim. The headline metric.")
print(f"  - Lower var_err = closer to true variance. Bounded by MC noise (~5-10% with these settings).")
print(f"  - R-hat near 1.0 = converged.")
print(f"  - If cp=0.0 wins on every target → use cp=0.0 as default; cp=0.5 is insurance for stale-Z only.")
print(f"  - If cp=0.5 wins or ties → cp=0.5 is the safe default everywhere.")
print(f"  - If cp=1.0 wins → unexpected; investigate.")
