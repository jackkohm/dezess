#!/usr/bin/env python
"""Speckle diagnostic — does sp+gj produce tail-speckle vs clean cp=0.5?

Tests the hypothesis that snooker_prob + gamma_jump_prob create scattered
tail samples (the 'speckle' fingerprint) on a uni-modal anisotropic target.

Target: 51D Gaussian with α=0.6 cross-block correlation. Bg_MH 1blk
(not 2blk — 1blk is the per-eval winner from prior benches). Single seed,
6000 steps. Diagnostic only — not a full benchmark.

Reports for each config:
  - lp distribution: mean, median, percentiles, min, max
  - tail fraction: frac of samples with lp < MAP - {50, 100, 200}
  - sample norm spread: max ‖z‖ vs typical ‖z‖
  - Per-dim sample range (max - min) for the dim with largest range
  - ESS_min for context
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
import jax, jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig
from dezess.benchmark.metrics import compute_ess

print(f"JAX devices: {jax.devices()}")

POT_DIM = 7
NUIS_DIM = 44
NDIM = POT_DIM + NUIS_DIM
ALPHA = 0.6
N_WALKERS = 64
N_WARMUP = 1000
N_PROD = 5000
SEED = 0


def build_target():
    M = np.eye(NDIM, dtype=np.float64)
    for j in range(NUIS_DIM):
        pot_idx = j % POT_DIM
        row = POT_DIM + j
        M[row, pot_idx] = ALPHA
        M[row, row]     = np.sqrt(1.0 - ALPHA ** 2)
    return M @ M.T, M


true_cov, M = build_target()
prec = np.linalg.inv(true_cov)
prec_jax = jnp.asarray(prec)


@jax.jit
def log_prob(x):
    return -0.5 * x @ prec_jax @ x


# Sample MAP lp (= 0 at origin for unit-mean Gaussian)
MAP_LP = 0.0
# Expected typical-set log-prob: -d/2
TYPICAL_LP = -NDIM / 2.0
print(f"Target: {NDIM}D Gaussian, cross-block α={ALPHA}")
print(f"MAP lp = {MAP_LP}, typical-set lp ≈ {TYPICAL_LP:.1f} (= -d/2)\n")


def make_init(seed):
    key = jax.random.PRNGKey(seed)
    z = jax.random.normal(key, (N_WALKERS, NDIM), dtype=jnp.float64)
    return z @ jnp.asarray(M).T


def make_cfg(sp, gj, name):
    ens = {
        "block_sizes": [NDIM],
        "use_mh": True,
        "delayed_rejection": True,
        "complementary_prob": 0.5,
        "lp_dead": -1e6,
    }
    if sp > 0:
        ens["snooker_prob"] = sp
    if gj > 0:
        ens["gamma_jump_prob"] = gj
    return VariantConfig(
        name=name,
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


CONFIGS = [
    ("Clean (cp=0.5)",            make_cfg(sp=0.0, gj=0.0, name="bgmh_clean")),
    ("Braak (cp=0.5,sp=.1,gj=.1)", make_cfg(sp=0.1, gj=0.1, name="bgmh_braak")),
]


def diagnose(label, result):
    samples = np.array(result["samples"])      # (N_PROD, N_WALKERS, NDIM)
    lps     = np.array(result["log_prob"])     # (N_PROD, N_WALKERS)
    flat    = samples.reshape(-1, NDIM)
    lp_flat = lps.flatten()

    norms = np.linalg.norm(flat, axis=-1)
    # Expected typical-set radius ≈ √d ≈ 7.14 for 51D unit-equivalent

    # lp distribution stats
    lp_med = np.median(lp_flat)
    lp_max = lp_flat.max()
    lp_min = lp_flat.min()
    lp_p01 = np.percentile(lp_flat, 1)
    lp_p99 = np.percentile(lp_flat, 99)

    # Tail fractions
    tail_50  = (lp_flat < MAP_LP - 50).mean()
    tail_100 = (lp_flat < MAP_LP - 100).mean()
    tail_200 = (lp_flat < MAP_LP - 200).mean()
    tail_500 = (lp_flat < MAP_LP - 500).mean()

    # Per-dim range (max - min) for the most extreme dim
    per_dim_range = flat.max(axis=0) - flat.min(axis=0)
    max_range_dim = per_dim_range.argmax()

    ess = compute_ess(samples)

    print(f"\n=== {label} ===")
    print(f"  lp:    max={lp_max:.1f}  median={lp_med:.1f}  min={lp_min:.1f}  p1={lp_p01:.1f}  p99={lp_p99:.1f}")
    print(f"  lp span (max-min): {lp_max - lp_min:.1f}  (typical-set span ~{2*np.sqrt(NDIM/2):.0f})")
    print(f"  TAIL FRACTIONS (frac of samples with lp < MAP_lp − K):")
    print(f"    K=50:  {tail_50*100:>6.2f}%")
    print(f"    K=100: {tail_100*100:>6.2f}%")
    print(f"    K=200: {tail_200*100:>6.2f}%")
    print(f"    K=500: {tail_500*100:>6.2f}%")
    print(f"  ‖z‖ stats: max={norms.max():.2f}  p99={np.percentile(norms, 99):.2f}  "
          f"median={np.median(norms):.2f}  (typical {np.sqrt(NDIM):.2f})")
    print(f"  Most extreme dim {max_range_dim}: range={per_dim_range[max_range_dim]:.2f}  "
          f"(typical 4-sigma ≈ {4*np.sqrt(true_cov[max_range_dim, max_range_dim]):.2f})")
    print(f"  ESS_min = {ess.min():.1f}  median = {np.median(ess):.1f}")


for label, cfg in CONFIGS:
    init = make_init(SEED)
    result = run_variant(
        log_prob, init,
        n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
        config=cfg, key=jax.random.PRNGKey(SEED * 1000), verbose=False,
    )
    diagnose(label, result)

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("  - If Braak's tail fractions (K=200, K=500) are much higher than Clean's,")
print("    sp/gj are producing the speckle by accepting tail proposals.")
print("  - If Braak's max ‖z‖ is much higher than Clean's, snooker outward pump")
print("    is pushing walkers into the tails (within the feasible region).")
print("  - If ESS_min is similar but tails differ, sp/gj are producing tail")
print("    samples without buying mixing benefit — net negative on Sanders-like targets.")
