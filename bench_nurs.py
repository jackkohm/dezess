#!/usr/bin/env python
"""Head-to-head benchmark: NURS variants vs default scale_aware sampler.

Compares ESS/step and ESS/sec on standard Gaussian targets in 10D and 20D,
a correlated Gaussian (condition ~50), and Rosenbrock (banana).
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import gc
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

print(f"JAX devices: {jax.devices()}", flush=True)

# ── Metrics ──────────────────────────────────────────────────────────────

def ess_fft(chain):
    n = len(chain)
    if n < 10:
        return 0.0
    x = chain - chain.mean()
    var = np.var(x)
    if var < 1e-30:
        return 0.0
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real / (n * var)
    tau = 1.0
    for lag in range(1, n):
        tau += 2 * acf[lag]
        if lag >= 5 * tau:
            break
    return n / max(tau, 1.0)


def compute_ess(samples_3d):
    """(n_steps, n_walkers, n_dim) -> ESS per dim."""
    mean_chain = samples_3d.mean(axis=1)
    return np.array([ess_fft(mean_chain[:, d]) for d in range(mean_chain.shape[1])])


# ── Targets ──────────────────────────────────────────────────────────────

def make_targets():
    targets = {}

    # Standard Gaussian 10D
    for ndim in [10, 20]:
        def _lp(x, _nd=ndim):
            return -0.5 * jnp.sum(x**2)
        targets[f"gaussian_{ndim}d"] = {
            "log_prob": jax.jit(_lp),
            "ndim": ndim,
            "init_scale": 0.1,
        }

    # Correlated Gaussian 10D (condition ~50)
    ndim = 10
    rng = np.random.default_rng(42)
    A = rng.standard_normal((ndim, ndim))
    Q, _ = np.linalg.qr(A)
    evals = np.linspace(1.0, 50.0, ndim)
    cov = Q @ np.diag(evals) @ Q.T
    cov = (cov + cov.T) / 2
    prec = np.linalg.inv(cov)
    prec_j = jnp.array(prec, dtype=jnp.float64)
    L_chol = np.linalg.cholesky(cov)

    @jax.jit
    def corr_lp(x):
        return -0.5 * x @ prec_j @ x

    targets["correlated_10d"] = {
        "log_prob": corr_lp,
        "ndim": ndim,
        "init_scale": None,
        "L_chol": L_chol,
    }

    # Rosenbrock (banana) 4D
    ndim = 4
    @jax.jit
    def rosen_lp(x):
        return -jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

    targets["rosenbrock_4d"] = {
        "log_prob": rosen_lp,
        "ndim": ndim,
        "init_scale": 0.5,
        "init_center": 1.0,
    }

    return targets


# ── Variant Configs ──────────────────────────────────────────────────────

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig

VARIANTS = {
    "default (scale_aware)": VariantConfig(
        name="scale_aware",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    ),
    "NURS": VariantConfig(
        name="nurs",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="nurs",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        slice_kwargs={"n_expand": 5, "use_mh": False},
    ),
    "NURS (MH)": VariantConfig(
        name="nurs_mh",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="nurs",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        slice_kwargs={"n_expand": 3, "use_mh": True},
    ),
    "NURS (scalar width)": VariantConfig(
        name="nurs_scalar",
        direction="de_mcz",
        width="scalar",
        slice_fn="nurs",
        zmatrix="circular",
        ensemble="standard",
        slice_kwargs={"n_expand": 5, "use_mh": False},
    ),
    "baseline (fixed/scalar)": VariantConfig(
        name="baseline",
        direction="de_mcz",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
    ),
}


# ── Runner ───────────────────────────────────────────────────────────────

def run_one(config, target, n_walkers=64, n_steps=5000, n_warmup=2000):
    ndim = target["ndim"]
    key = jax.random.PRNGKey(42)

    if "L_chol" in target:
        L_chol = target["L_chol"]
        init = jax.random.normal(key, (n_walkers, ndim)) @ jnp.array(L_chol).T
    elif target.get("init_center") is not None:
        init = jax.random.normal(key, (n_walkers, ndim)) * target["init_scale"] + target["init_center"]
    else:
        init = jax.random.normal(key, (n_walkers, ndim)) * target["init_scale"]

    t0 = time.time()
    result = run_variant(
        target["log_prob"], init,
        n_steps=n_steps, n_warmup=n_warmup,
        config=config,
        key=jax.random.PRNGKey(1),
        verbose=False,
    )
    wall = time.time() - t0

    samples = np.array(result["samples"])
    n_prod = samples.shape[0]
    ess = compute_ess(samples)
    flat = samples.reshape(-1, ndim)
    mean_var = float(np.mean(np.var(flat, axis=0)))

    return {
        "ess_min": float(ess.min()),
        "ess_mean": float(ess.mean()),
        "ess_per_sec": float(ess.min() / wall),
        "wall_time": wall,
        "mean_var": mean_var,
        "mu_final": float(result["mu"]),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    targets = make_targets()

    n_steps = 5000
    n_warmup = 2000

    results = []
    total = len(VARIANTS) * len(targets)
    idx = 0

    for tname, target in targets.items():
        for vname, config in VARIANTS.items():
            idx += 1
            print(f"[{idx}/{total}] {vname:25s} x {tname:20s} ... ",
                  end="", flush=True)
            try:
                r = run_one(config, target, n_steps=n_steps, n_warmup=n_warmup)
                r["variant"] = vname
                r["target"] = tname
                results.append(r)
                print(f"ESS/s={r['ess_per_sec']:8.1f}  ESS_min={r['ess_min']:8.1f}  "
                      f"var={r['mean_var']:.4f}  wall={r['wall_time']:6.1f}s",
                      flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                import traceback; traceback.print_exc()
                results.append({"variant": vname, "target": tname, "error": str(e)})
            gc.collect()

    # ── Print comparison table ───────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"  {'NURS vs DEFAULT SAMPLER COMPARISON':^96s}")
    print("=" * 100)

    for tname in targets:
        print(f"\n  TARGET: {tname}")
        print(f"  {'Variant':<25s} {'ESS_min':>10s} {'ESS/sec':>10s} "
              f"{'Wall(s)':>10s} {'Variance':>10s} {'mu_final':>12s}")
        print("  " + "-" * 82)

        target_results = sorted(
            [r for r in results if r.get("target") == tname and "error" not in r],
            key=lambda r: r.get("ess_per_sec", 0),
            reverse=True,
        )

        for r in target_results:
            marker = " <-- best" if r == target_results[0] else ""
            print(f"  {r['variant']:<25s} {r['ess_min']:>10.1f} "
                  f"{r['ess_per_sec']:>10.1f} {r['wall_time']:>10.1f} "
                  f"{r['mean_var']:>10.4f} {r['mu_final']:>12.6f}{marker}")

        for r in results:
            if r.get("target") == tname and "error" in r:
                print(f"  {r['variant']:<25s} {'ERROR':>10s}: {r['error'][:50]}")

    # Summary
    print("\n" + "=" * 100)
    print("  PER-TARGET WINNERS (ESS/sec)")
    print("=" * 100)
    for tname in targets:
        valid = [r for r in results if r.get("target") == tname and "error" not in r]
        if valid:
            best = max(valid, key=lambda r: r.get("ess_per_sec", 0))
            print(f"  {tname:<25s} {best['variant']:<25s} ({best['ess_per_sec']:.1f} ESS/s)")


if __name__ == "__main__":
    main()
