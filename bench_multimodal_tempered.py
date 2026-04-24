#!/usr/bin/env python
"""Multimodal + tempered warmup: does warmup_t_start rescue global_move?

Same target as bench_multimodal.py: 21D 3-mode mixture, init at mode 0.
Sweeps warmup_t_start ∈ {1, 10, 50} for two configs:
  - global_move cp=0.0  (the previous failure — 100% stuck in mode 0)
  - scale_aware cp=0.5  (the previous near-failure — 94% stuck)

Plus a single warm-start scale_aware cp=0.0 baseline for reference.

Hypothesis: tempered warmup spreads walkers across modes during warmup,
so the GMM (fit at end of warmup) captures all modes and global_move
can finally jump between them in production.
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

NDIM = 21
N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 5000
N_SEEDS = 3
SEPARATION = 6.0

MODE_CENTERS = jnp.array([
    [+SEPARATION] + [0.0] * (NDIM - 1),
    [-SEPARATION] + [0.0] * (NDIM - 1),
    [0.0, +SEPARATION] + [0.0] * (NDIM - 2),
], dtype=jnp.float64)
LOG_WEIGHTS = jnp.full((3,), jnp.log(1.0 / 3.0), dtype=jnp.float64)


@jax.jit
def log_prob(x):
    diffs = x[None, :] - MODE_CENTERS
    log_comps = LOG_WEIGHTS - 0.5 * jnp.sum(diffs ** 2, axis=1)
    return jax.scipy.special.logsumexp(log_comps)


def make_init(seed):
    """All walkers near MODE 0 — chain must DISCOVER the others."""
    key = jax.random.PRNGKey(seed)
    return MODE_CENTERS[0][None, :] + 0.3 * jax.random.normal(
        key, (N_WALKERS, NDIM), dtype=jnp.float64
    )


def assign_modes(samples_flat):
    centers_np = np.array(MODE_CENTERS)
    d2 = ((samples_flat[:, None, :] - centers_np[None, :, :]) ** 2).sum(-1)
    nearest = np.argmin(d2, axis=1)
    return np.array([float((nearest == k).mean()) for k in range(3)])


# ── Configs ─────────────────────────────────────────────────────────────

def make_config(direction, cp, t_start):
    ens_kwargs = {}
    if cp > 0.0:
        ens_kwargs["complementary_prob"] = cp
    if t_start > 1.0:
        ens_kwargs["warmup_t_start"] = t_start

    base = dict(
        name=f"{direction}_cp{cp}_t{t_start}",
        direction=direction, width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )
    if direction == "global_move":
        base["direction_kwargs"] = {"n_components": 3, "global_prob": 0.1}
    return VariantConfig(**base)


CONFIGS = [
    # Reference: zero-cp baseline (matches yesterday's run)
    ("scale_aware cp=0.0 t=1",     make_config("de_mcz",      0.0, 1.0)),

    # Hypothesis test: does tempering save global_move?
    ("global_move cp=0.0 t=1",     make_config("global_move", 0.0, 1.0)),
    ("global_move cp=0.0 t=10",    make_config("global_move", 0.0, 10.0)),
    ("global_move cp=0.0 t=50",    make_config("global_move", 0.0, 50.0)),

    # Side question: does tempering alone help cp=0.5?
    ("scale_aware cp=0.5 t=10",    make_config("de_mcz",      0.5, 10.0)),
    ("scale_aware cp=0.5 t=50",    make_config("de_mcz",      0.5, 50.0)),

    # Combo: tempering + cp + global_move
    ("global_move cp=0.5 t=50",    make_config("global_move", 0.5, 50.0)),
]


# ── Run ─────────────────────────────────────────────────────────────────

print(f"\n{'=' * 110}")
print(f"  TEMPERED-WARMUP MULTIMODAL: 21D, 3 modes at ±{SEPARATION}σ, init at mode 0")
print(f"  {N_WALKERS} walkers, {N_WARMUP} warmup + {N_PROD} prod, {N_SEEDS} seeds × {len(CONFIGS)} configs = {N_SEEDS * len(CONFIGS)} runs")
print(f"{'=' * 110}")

UNIFORM = 1.0 / 3.0
hdr = (f"  {'Setup':<28s} | {'seed':>4s} | {'mode_0':>6s} | {'mode_1':>6s} | "
       f"{'mode_2':>6s} | {'cov_err':>7s} | {'mean_lp':>7s} | {'wall':>6s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

results = {label: [] for label, _ in CONFIGS}

for label, cfg in CONFIGS:
    for seed in range(N_SEEDS):
        init = make_init(seed)
        t0 = time.time()
        result = run_variant(
            log_prob, init, n_steps=N_WARMUP + N_PROD,
            config=cfg, n_warmup=N_WARMUP,
            key=jax.random.PRNGKey(seed * 1000), verbose=False,
        )
        wall = time.time() - t0
        samples = np.array(result["samples"]).reshape(-1, NDIM)
        fracs = assign_modes(samples)
        cov_err = float(np.max(np.abs(fracs - UNIFORM)))
        mean_lp = float(np.array(result["log_prob"])[-500:].mean())

        results[label].append({
            "fracs": fracs, "cov_err": cov_err, "mean_lp": mean_lp, "wall": wall,
        })
        print(f"  {label:<28s} | {seed:>4d} | {fracs[0]:6.3f} | {fracs[1]:6.3f} | "
              f"{fracs[2]:6.3f} | {cov_err:7.3f} | {mean_lp:7.2f} | {wall:5.1f}s",
              flush=True)
        gc.collect()


# ── Summary ─────────────────────────────────────────────────────────────

print(f"\n{'=' * 110}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"  cov_err = max|frac[k] - 1/3| ; 0 = uniform across modes ; 0.667 = stuck in 1 mode")
print(f"{'=' * 110}")
hdr2 = (f"  {'Setup':<28s} | {'mode_0':>6s} | {'mode_1':>6s} | {'mode_2':>6s} | "
        f"{'cov_err':>7s} | {'mean_lp':>8s} | {'wall':>7s}")
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))
for label, _ in CONFIGS:
    runs = results[label]
    fracs_m = np.mean([r["fracs"] for r in runs], axis=0)
    cov = np.mean([r["cov_err"] for r in runs])
    lp = np.mean([r["mean_lp"] for r in runs])
    w = np.mean([r["wall"] for r in runs])
    print(f"  {label:<28s} | {fracs_m[0]:6.3f} | {fracs_m[1]:6.3f} | "
          f"{fracs_m[2]:6.3f} | {cov:7.3f} | {lp:8.2f} | {w:6.1f}s", flush=True)

print()
print(f"  Decisive outcomes:")
print(f"    cov_err < 0.05  → all 3 modes equally populated. WIN.")
print(f"    cov_err in 0.1-0.4 → discovered some modes, uneven.")
print(f"    cov_err > 0.5  → stuck (fail).")
