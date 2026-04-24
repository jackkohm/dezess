#!/usr/bin/env python
"""Multimodal head-to-head: 21D 3-mode mixture, init at ONE mode.

Tests whether the sampler can discover and populate the other two modes.

Configs:
  - scale_aware cp=0.0  (baseline, no insurance)
  - scale_aware cp=0.5  (recommended hedge)
  - scale_aware cp=1.0  (pure complementary)
  - global_move (cp=0)  (GMM-based mode jumping)
  - global_move + cp=0.5 (combo)

Metrics:
  - mode_coverage_err = max|frac[k] - 1/3| (uniform = 0; one-mode-stuck = 0.667)
  - mean_lp / lp_min  (chain finds high-lp regions)
  - ESS (mixing within a mode)
  - wall_time
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
SEPARATION = 6.0     # mode centers ±6σ apart along their axes

# ── Target: 3 well-separated unit Gaussians ─────────────────────────────

# Mode centers: along axis 0 (+SEP, -SEP) and axis 1 (+SEP).
# Equal mixture weights.
MODE_CENTERS = jnp.array([
    [+SEPARATION] + [0.0] * (NDIM - 1),
    [-SEPARATION] + [0.0] * (NDIM - 1),
    [0.0, +SEPARATION] + [0.0] * (NDIM - 2),
], dtype=jnp.float64)
LOG_WEIGHTS = jnp.full((3,), jnp.log(1.0 / 3.0), dtype=jnp.float64)


@jax.jit
def log_prob(x):
    """logsumexp over 3 unit-Gaussian components (equal weights)."""
    diffs = x[None, :] - MODE_CENTERS                  # (3, ndim)
    log_comps = LOG_WEIGHTS - 0.5 * jnp.sum(diffs ** 2, axis=1)
    return jax.scipy.special.logsumexp(log_comps)


def make_init(seed):
    """All walkers near MODE 0 — chain must DISCOVER the others."""
    key = jax.random.PRNGKey(seed)
    return MODE_CENTERS[0][None, :] + 0.3 * jax.random.normal(
        key, (N_WALKERS, NDIM), dtype=jnp.float64
    )


def assign_modes(samples_flat):
    """Assign each sample to nearest mode, return per-mode fraction."""
    centers_np = np.array(MODE_CENTERS)            # (3, ndim)
    d2 = ((samples_flat[:, None, :] - centers_np[None, :, :]) ** 2).sum(-1)
    nearest = np.argmin(d2, axis=1)                # (n_samples,)
    fracs = np.array([float((nearest == k).mean()) for k in range(3)])
    return fracs


# ── Configs ─────────────────────────────────────────────────────────────

def make_scale_aware(cp):
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


def make_global_move(cp, global_prob=0.1):
    ens_kwargs = {}
    if cp > 0.0:
        ens_kwargs["complementary_prob"] = cp
    return VariantConfig(
        name=f"global_move_cp{cp}",
        direction="global_move", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        direction_kwargs={"n_components": 3, "global_prob": global_prob},
        ensemble_kwargs=ens_kwargs,
    )


CONFIGS = [
    ("scale_aware cp=0.0",   make_scale_aware(0.0)),
    ("scale_aware cp=0.5",   make_scale_aware(0.5)),
    ("scale_aware cp=1.0",   make_scale_aware(1.0)),
    ("global_move cp=0.0",   make_global_move(0.0)),
    ("global_move cp=0.5",   make_global_move(0.5)),
]


# ── Run sweep ───────────────────────────────────────────────────────────

print(f"\n{'=' * 110}")
print(f"  MULTIMODAL: 21D, 3 unit-Gaussian modes, separation={SEPARATION}σ")
print(f"  {N_WALKERS} walkers, {N_WARMUP} warmup + {N_PROD} prod, init at mode 0")
print(f"  {N_SEEDS} seeds × {len(CONFIGS)} configs = {N_SEEDS * len(CONFIGS)} runs")
print(f"{'=' * 110}")

UNIFORM_FRAC = 1.0 / 3.0
hdr = (f"  {'Setup':<22s} | {'seed':>4s} | {'mode_0':>6s} | {'mode_1':>6s} | "
       f"{'mode_2':>6s} | {'cov_err':>7s} | {'mean_lp':>7s} | {'ESS_min':>7s} | {'wall':>6s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

results_by_config = {label: [] for label, _ in CONFIGS}

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
        samples = np.array(result["samples"])           # (n_prod, n_walkers, ndim)
        flat = samples.reshape(-1, NDIM)
        fracs = assign_modes(flat)
        cov_err = float(np.max(np.abs(fracs - UNIFORM_FRAC)))
        mean_lp = float(np.array(result["log_prob"])[-500:].mean())
        ess = compute_ess(samples)
        ess_min = float(ess.min())

        results_by_config[label].append({
            "fracs": fracs, "cov_err": cov_err,
            "mean_lp": mean_lp, "ess_min": ess_min, "wall": wall,
        })
        print(f"  {label:<22s} | {seed:>4d} | {fracs[0]:6.3f} | {fracs[1]:6.3f} | "
              f"{fracs[2]:6.3f} | {cov_err:7.3f} | {mean_lp:7.2f} | "
              f"{ess_min:7.1f} | {wall:5.1f}s", flush=True)
        gc.collect()


# ── Summary ─────────────────────────────────────────────────────────────

print(f"\n{'=' * 110}")
print(f"  SUMMARY (mean across {N_SEEDS} seeds)")
print(f"  cov_err = max|frac[k] - 1/3| ; 0 = perfectly uniform across modes ; 0.667 = stuck in 1 mode")
print(f"{'=' * 110}")

hdr2 = (f"  {'Setup':<22s} | {'mode_0':>6s} | {'mode_1':>6s} | {'mode_2':>6s} | "
        f"{'cov_err':>7s} | {'mean_lp':>8s} | {'ESS_min':>8s} | {'wall':>7s}")
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))

for label, _ in CONFIGS:
    runs = results_by_config[label]
    fracs_mean = np.mean([r["fracs"] for r in runs], axis=0)
    cov_err = np.mean([r["cov_err"] for r in runs])
    mean_lp = np.mean([r["mean_lp"] for r in runs])
    ess_min = np.mean([r["ess_min"] for r in runs])
    wall = np.mean([r["wall"] for r in runs])
    print(f"  {label:<22s} | {fracs_mean[0]:6.3f} | {fracs_mean[1]:6.3f} | "
          f"{fracs_mean[2]:6.3f} | {cov_err:7.3f} | {mean_lp:8.2f} | "
          f"{ess_min:8.1f} | {wall:6.1f}s", flush=True)

print()
print(f"  Read the table:")
print(f"    - cov_err < 0.1: chain found and populated ALL 3 modes well")
print(f"    - cov_err in [0.1, 0.4]: discovered some modes but coverage uneven")
print(f"    - cov_err > 0.5: stuck near init mode (mode 0)")
