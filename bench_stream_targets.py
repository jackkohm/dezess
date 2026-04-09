"""Benchmark dezess on stream-fitting targets (block_coupled, funnel_63d, banana_63d).

Run on local GPU to establish baseline performance numbers.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dezess.targets_stream import block_coupled_gaussian, funnel_63d, banana_63d
from dezess.targets import neals_funnel
from dezess.core.loop import run_variant
from dezess.benchmark.registry import VARIANTS
from dezess.benchmark.metrics import compute_ess, compute_rhat


def run_one(target, variant_name, n_walkers=128, n_warmup=4000, n_prod=5000):
    """Run a single variant on a target and report diagnostics."""
    config = VARIANTS[variant_name]
    key = jax.random.PRNGKey(42)

    # Initialize walkers
    if target.sample is not None:
        init = target.sample(key, n_walkers)
    else:
        init = jax.random.normal(key, (n_walkers, target.ndim)) * 0.1

    # Warmup: JIT compile
    t0 = time.time()
    result = run_variant(
        target.log_prob, init, n_prod, config,
        n_warmup=n_warmup, verbose=False,
    )
    # Block until done
    samples = np.array(result["samples"])
    wall = time.time() - t0

    # Diagnostics
    ess = compute_ess(samples)
    rhat = compute_rhat(samples)

    # Variance accuracy (if true cov known)
    var_acc = None
    if target.cov is not None:
        true_var = np.diag(np.array(target.cov))
        sample_var = np.var(samples.reshape(-1, target.ndim), axis=0)
        var_acc = np.mean(sample_var / true_var)

    # For funnel: check log-width marginal variance (should be ~9)
    funnel_diag = None
    if "funnel" in target.name:
        flat = samples.reshape(-1, target.ndim)
        # Check first funnel's log-width (index 7 for funnel_63d, 0 for neals_funnel)
        if target.ndim == 63:
            lw_idx = 7  # first funnel log-width
            lw_samples = flat[:, lw_idx]
        else:
            lw_idx = 0
            lw_samples = flat[:, lw_idx]
        funnel_diag = {
            "log_width_mean": float(np.mean(lw_samples)),
            "log_width_var": float(np.var(lw_samples)),  # should be ~9
            "log_width_expected_var": 9.0,
        }

    return {
        "variant": variant_name,
        "target": target.name,
        "ndim": target.ndim,
        "n_walkers": n_walkers,
        "wall_s": wall,
        "ess_min": float(np.min(ess)),
        "ess_mean": float(np.mean(ess)),
        "ess_max": float(np.max(ess)),
        "rhat_max": float(np.max(rhat)),
        "rhat_mean": float(np.mean(rhat)),
        "var_acc": var_acc,
        "funnel_diag": funnel_diag,
    }


def print_result(r):
    print(f"\n  {r['variant']:30s}  |  ESS min={r['ess_min']:6.1f}  mean={r['ess_mean']:6.1f}"
          f"  |  R-hat max={r['rhat_max']:.4f}  mean={r['rhat_mean']:.4f}"
          f"  |  {r['wall_s']:.1f}s", end="")
    if r["var_acc"] is not None:
        print(f"  |  var_acc={r['var_acc']:.3f}", end="")
    if r["funnel_diag"] is not None:
        fd = r["funnel_diag"]
        print(f"  |  lw_var={fd['log_width_var']:.2f} (expect 9.0)", end="")
    print()


if __name__ == "__main__":
    variants = ["scale_aware", "baseline", "snooker_stochastic", "zeus_gamma"]

    # Also check if overrelaxed and KDE are available
    for v in ["overrelaxed", "kde_directions", "whitened"]:
        if v in VARIANTS:
            variants.append(v)

    print(f"Available variants to test: {variants}")
    print(f"Device: {jax.devices()}")

    # 1. Neal's funnel 10D (sanity check — should work)
    print("\n" + "=" * 80)
    print("TARGET: Neal's Funnel 10D (sanity check)")
    print("=" * 80)
    target = neals_funnel(10)
    for v in variants:
        try:
            r = run_one(target, v, n_walkers=64, n_warmup=2000, n_prod=3000)
            print_result(r)
        except Exception as e:
            print(f"\n  {v:30s}  |  FAILED: {e}")

    # 2. Block-coupled 63D
    print("\n" + "=" * 80)
    print("TARGET: Block-Coupled Gaussian 63D")
    print("=" * 80)
    target = block_coupled_gaussian()
    for v in variants:
        try:
            r = run_one(target, v, n_walkers=128, n_warmup=4000, n_prod=5000)
            print_result(r)
        except Exception as e:
            print(f"\n  {v:30s}  |  FAILED: {e}")

    # 3. Funnel 63D (the hard one)
    print("\n" + "=" * 80)
    print("TARGET: Funnel 63D (4 funnels + 7 potential)")
    print("=" * 80)
    target = funnel_63d()
    for v in variants:
        try:
            r = run_one(target, v, n_walkers=128, n_warmup=6000, n_prod=8000)
            print_result(r)
        except Exception as e:
            print(f"\n  {v:30s}  |  FAILED: {e}")

    # 4. Banana 63D
    print("\n" + "=" * 80)
    print("TARGET: Banana 63D")
    print("=" * 80)
    target = banana_63d()
    for v in variants:
        try:
            r = run_one(target, v, n_walkers=128, n_warmup=4000, n_prod=5000)
            print_result(r)
        except Exception as e:
            print(f"\n  {v:30s}  |  FAILED: {e}")
