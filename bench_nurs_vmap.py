#!/usr/bin/env python
"""Benchmark: NURS (vmap) speedup + quality on funnel targets.

Tests:
1. Wall-time speedup on standard Gaussian (vmap vs fori_loop baseline)
2. NURS vs default on sanders_funnel_63d (funnel quality in x-space)
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
import time
import gc

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.core.slice_sample import safe_log_prob
from dezess.benchmark.metrics import compute_ess, compute_rhat
from dezess.targets_stream import funnel_63d, block_coupled_gaussian

print(f"JAX devices: {jax.devices()}", flush=True)

# ── Part 1: vmap vs fori_loop microbenchmark ─────────────────────────────

def bench_orbit_eval():
    """Time the orbit evaluation phase alone: vmap vs fori_loop."""
    ndim = 63
    log_prob_fn = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (ndim,), dtype=jnp.float64)
    d = jax.random.normal(key, (ndim,), dtype=jnp.float64)
    d = d / jnp.linalg.norm(d)
    h = 1.0

    for M in [3, 5, 7]:
        max_orbit = 2 ** M
        orbit_indices = jnp.arange(max_orbit, dtype=jnp.float64)

        # --- fori_loop version ---
        def fori_version(x, d, h):
            def _eval(i, lps):
                pt = x + jnp.float64(i) * h * d
                lp = safe_log_prob(log_prob_fn, pt)
                return lps.at[i].set(lp)
            return lax.fori_loop(0, max_orbit, _eval,
                                 jnp.full(max_orbit, -1e30))

        fori_jit = jax.jit(fori_version)
        _ = fori_jit(x, d, h).block_until_ready()

        t0 = time.time()
        for _ in range(200):
            fori_jit(x, d, h).block_until_ready()
        t_fori = (time.time() - t0) / 200

        # --- vmap version ---
        def vmap_version(x, d, h):
            pts = x + orbit_indices[:, None] * h * d
            return jax.vmap(lambda pt: safe_log_prob(log_prob_fn, pt))(pts)

        vmap_jit = jax.jit(vmap_version)
        _ = vmap_jit(x, d, h).block_until_ready()

        t0 = time.time()
        for _ in range(200):
            vmap_jit(x, d, h).block_until_ready()
        t_vmap = (time.time() - t0) / 200

        speedup = t_fori / t_vmap
        print(f"  M={M} ({max_orbit:3d} pts): fori={t_fori*1e3:.2f}ms  "
              f"vmap={t_vmap*1e3:.2f}ms  speedup={speedup:.1f}x")


print("\n" + "=" * 80)
print("PART 1: Orbit evaluation microbenchmark (63D Gaussian)")
print("=" * 80)
bench_orbit_eval()

# ── Part 2: Full sampler comparison on funnel targets ────────────────────

CONFIGS = {
    "bg_MH (5 ev)": VariantConfig(
        name="block_gibbs_mh",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True},
    ),
    "bg_MH+DR (10 ev)": VariantConfig(
        name="block_gibbs_mh_dr",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True,
                         "delayed_rejection": True},
    ),
    "bg_MH+cov (5 ev)": VariantConfig(
        name="block_gibbs_mh_cov",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True,
                         "use_block_cov": True},
    ),
    "bg_MH+DR+cov (10 ev)": VariantConfig(
        name="block_gibbs_mh_full",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14], "use_mh": True,
                         "delayed_rejection": True, "use_block_cov": True},
    ),
}


def run_one(config, target, n_walkers=64, n_warmup=4000, n_prod=10000):
    key = jax.random.PRNGKey(42)
    if target.sample is not None:
        init = target.sample(key, n_walkers)
    else:
        init = jax.random.normal(key, (n_walkers, target.ndim)) * 0.1

    t0 = time.time()
    result = run_variant(
        target.log_prob, init,
        n_steps=n_warmup + n_prod,
        config=config,
        n_warmup=n_warmup,
        key=jax.random.PRNGKey(1),
        verbose=False,
    )
    wall = time.time() - t0
    samples = np.array(result["samples"])
    ess = compute_ess(samples)
    rhat = compute_rhat(samples)

    # Funnel log-width variance check (expect ~9 for width params)
    flat = samples.reshape(-1, target.ndim)
    lw_vars = []
    if "funnel" in target.name:
        for f in range(4):
            lw_vars.append(float(np.var(flat[:, 7 + f * 14])))

    return {
        "ess_min": float(ess.min()),
        "ess_mean": float(ess.mean()),
        "rhat_max": float(rhat.max()),
        "rhat_mean": float(rhat.mean()),
        "wall": wall,
        "lw_vars": lw_vars,
        "mu": float(result["mu"]),
    }


targets = [
    ("Funnel 63D", funnel_63d()),
    ("Block-Coupled 63D", block_coupled_gaussian()),
]

print("\n" + "=" * 80)
print("PART 2: Sampler comparison on funnel targets (no transform)")
print("=" * 80)

for tname, target in targets:
    print(f"\n{'─' * 80}")
    print(f"TARGET: {tname}  (ndim={target.ndim})")
    print(f"{'─' * 80}")
    hdr = (f"  {'Strategy':<25s} | {'R-hat max':>9s} | {'R-hat mean':>10s} | "
           f"{'ESS min':>8s} | {'ESS mean':>9s} | {'Wall':>6s} | {'mu':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for vname, config in CONFIGS.items():
        try:
            r = run_one(config, target)
            line = (f"  {vname:<25s} | {r['rhat_max']:9.4f} | {r['rhat_mean']:10.4f} | "
                    f"{r['ess_min']:8.1f} | {r['ess_mean']:9.1f} | {r['wall']:5.1f}s | "
                    f"{r['mu']:8.4f}")
            if r["lw_vars"]:
                lw_str = " ".join(f"{v:.1f}" for v in r["lw_vars"])
                line += f"  lw_var=[{lw_str}]"
                if "funnel" in tname.lower():
                    line += " (expect ~9)"
            print(line, flush=True)
        except Exception as e:
            print(f"  {vname:<25s} | FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()
        gc.collect()
