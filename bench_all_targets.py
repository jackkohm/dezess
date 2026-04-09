"""Benchmark all strategies on all three stream-fitting targets."""
import time
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dezess.targets_stream import block_coupled_gaussian, funnel_63d, banana_63d
from dezess.core.loop import run_variant
from dezess.benchmark.registry import VARIANTS
from dezess.benchmark.metrics import compute_ess, compute_rhat
from dezess.transforms import multi_funnel

transform = multi_funnel(n_potential=7, funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)])


def run_one(target, variant_name, tfm=None, n_walkers=128, n_warmup=4000, n_prod=5000):
    config = VARIANTS[variant_name]
    key = jax.random.PRNGKey(42)
    if target.sample is not None:
        init = target.sample(key, n_walkers)
    else:
        init = jax.random.normal(key, (n_walkers, target.ndim)) * 0.1
    t0 = time.time()
    r = run_variant(target.log_prob, init, n_prod, config, n_warmup=n_warmup, verbose=False, transform=tfm)
    s = np.array(r["samples"])
    wall = time.time() - t0
    ess = compute_ess(s)
    rhat = compute_rhat(s)
    var_acc = None
    if target.cov is not None:
        true_var = np.diag(np.array(target.cov))
        sample_var = np.var(s.reshape(-1, target.ndim), axis=0)
        var_acc = float(np.mean(sample_var / true_var))

    # Funnel log-width variance check
    lw_vars = None
    if "funnel" in target.name:
        flat = s.reshape(-1, target.ndim)
        lw_vars = [float(np.var(flat[:, 7 + f * 14])) for f in range(4)]

    return {
        "rhat_max": float(np.max(rhat)),
        "rhat_mean": float(np.mean(rhat)),
        "ess_min": float(np.min(ess)),
        "ess_mean": float(np.mean(ess)),
        "wall": wall,
        "var_acc": var_acc,
        "lw_vars": lw_vars,
    }


strategies = [
    ("scale_aware", None, "Baseline"),
    ("block_gibbs_scale_aware", None, "Block-Gibbs"),
    ("scale_aware", transform, "Transform"),
    ("block_gibbs_transform", transform, "Combined"),
]

targets = [
    ("Block-Coupled 63D", block_coupled_gaussian()),
    ("Funnel 63D", funnel_63d()),
    ("Banana 63D", banana_63d()),
]

print(f"Device: {jax.devices()}")

for tname, target in targets:
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"TARGET: {tname}")
    print(sep)
    hdr = f"{'Strategy':<20s} | {'R-hat max':>9s} | {'R-hat mean':>10s} | {'ESS min':>8s} | {'ESS mean':>9s} | {'Wall':>6s}"
    if target.cov is not None:
        hdr += f" | {'var_acc':>8s}"
    print(hdr)
    print("-" * len(hdr))

    for vname, tfm, label in strategies:
        try:
            r = run_one(target, vname, tfm)
            line = (
                f"{label:<20s} | {r['rhat_max']:9.4f} | {r['rhat_mean']:10.4f} "
                f"| {r['ess_min']:8.1f} | {r['ess_mean']:9.1f} | {r['wall']:5.1f}s"
            )
            if r["var_acc"] is not None:
                line += f" | {r['var_acc']:8.3f}"
            if r["lw_vars"] is not None:
                lw_str = " ".join(f"{v:.1f}" for v in r["lw_vars"])
                line += f"  lw_var=[{lw_str}] (expect 9)"
            print(line)
        except Exception as e:
            print(f"{label:<20s} | FAILED: {e}")
