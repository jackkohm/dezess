"""Benchmark: transform vs block-Gibbs vs combined on funnel_63d.

Compares all strategies against the baseline to quantify improvement.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dezess.targets_stream import funnel_63d
from dezess.core.loop import run_variant
from dezess.benchmark.registry import VARIANTS
from dezess.benchmark.metrics import compute_ess, compute_rhat
from dezess.transforms import multi_funnel


def run_one(target, variant_name, transform=None, n_walkers=128, n_warmup=6000, n_prod=8000):
    config = VARIANTS[variant_name]
    key = jax.random.PRNGKey(42)
    init = target.sample(key, n_walkers)

    t0 = time.time()
    result = run_variant(
        target.log_prob, init, n_prod, config,
        n_warmup=n_warmup, verbose=False,
        transform=transform,
    )
    samples = np.array(result["samples"])
    wall = time.time() - t0

    ess = compute_ess(samples)
    rhat = compute_rhat(samples)

    flat = samples.reshape(-1, target.ndim)
    lw_vars = []
    for f in range(4):
        lw_idx = 7 + f * 14
        lw_vars.append(float(np.var(flat[:, lw_idx])))

    return {
        "variant": variant_name,
        "ess_min": float(np.min(ess)),
        "ess_mean": float(np.mean(ess)),
        "rhat_max": float(np.max(rhat)),
        "rhat_mean": float(np.mean(rhat)),
        "wall_s": wall,
        "lw_vars": lw_vars,
    }


if __name__ == "__main__":
    target = funnel_63d()
    transform = multi_funnel(
        n_potential=7,
        funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)],
    )

    print(f"Device: {jax.devices()}")
    print(f"Target: funnel_63d (63D, 4 funnels + 7 potential)")
    print(f"Setup: 128 walkers, 6000 warmup, 8000 production")
    print()

    configs = [
        ("scale_aware", None, "Baseline (no transform, no blocks)"),
        ("overrelaxed", None, "Overrelaxed (no transform, no blocks)"),
        ("scale_aware", transform, "Transform only (NCP + scale_aware)"),
        ("block_gibbs_scale_aware", None, "Block-Gibbs only (no transform)"),
        ("block_gibbs_transform", transform, "Combined (NCP + block-Gibbs)"),
    ]

    print(f"{'Strategy':<45s} | {'R-hat max':>9s} | {'ESS min':>8s} | {'ESS mean':>9s} | {'Wall':>6s} | {'lw_var (expect 9)':>20s}")
    print("-" * 120)

    for variant, tfm, label in configs:
        try:
            r = run_one(target, variant, transform=tfm)
            lw_str = " ".join(f"{v:.1f}" for v in r["lw_vars"])
            print(f"{label:<45s} | {r['rhat_max']:9.4f} | {r['ess_min']:8.1f} | {r['ess_mean']:9.1f} | {r['wall_s']:5.1f}s | {lw_str}")
        except Exception as e:
            print(f"{label:<45s} | FAILED: {e}")
