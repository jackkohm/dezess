#!/usr/bin/env python
"""Diagnostic for the lp_dead guard test — prints what's actually happening."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig

NDIM = 30
N_WALKERS = 64
FLOOR = -1e8
FEASIBLE_R = 3.0


@jax.jit
def floored_lp(x):
    r2 = jnp.sum(x * x)
    return jnp.where(r2 < FEASIBLE_R ** 2, -0.5 * r2, jnp.float64(FLOOR))


def make_cfg(lp_dead):
    ens = {
        "block_sizes": [NDIM], "use_mh": True, "delayed_rejection": True,
        "complementary_prob": 0.5, "snooker_prob": 0.10, "gamma_jump_prob": 0.05,
        "lp_dead": lp_dead,
    }
    return VariantConfig(
        name=f"diag_lp{lp_dead}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


rng = np.random.default_rng(0)
init_good = 0.3 * rng.standard_normal((N_WALKERS - 8, NDIM))
init_bad = rng.standard_normal((8, NDIM))
init_bad = init_bad * (5.0 / np.linalg.norm(init_bad, axis=1, keepdims=True))
init = jnp.asarray(np.concatenate([init_good, init_bad], axis=0), dtype=jnp.float64)

# Confirm init norms
init_norms = np.linalg.norm(np.array(init), axis=1)
init_lps = np.array(jax.vmap(floored_lp)(init))
print(f"Init norms — 8 bad walkers: {init_norms[-8:].round(2)}")
print(f"Init lp    — 8 bad walkers: {init_lps[-8:].round(1)}")
print(f"Init norms — 8 good walkers (sample): {init_norms[:8].round(2)}")
print(f"Init lp    — 8 good walkers (sample): {init_lps[:8].round(2)}")
print()

for lp_dead in [-1e6, -1e10]:
    label = "guarded" if lp_dead == -1e6 else "UNGUARDED"
    print(f"--- {label} (lp_dead={lp_dead}) ---")
    result = run_variant(
        floored_lp, init, n_steps=600, n_warmup=200,
        config=make_cfg(lp_dead), key=jax.random.PRNGKey(0), verbose=False,
    )
    samples = np.array(result["samples"])      # (400, 64, 30)
    lps = np.array(result["log_prob"])         # (400, 64)
    norms = np.linalg.norm(samples, axis=-1)   # (400, 64)

    # Per-walker max norm across production
    walker_max_norm = norms.max(axis=0)
    # Per-walker mean lp
    walker_mean_lp = lps.mean(axis=0)

    print(f"  bad walker (idx 56-63) max norms: {walker_max_norm[-8:].round(1)}")
    print(f"  bad walker (idx 56-63) mean lps:  {walker_mean_lp[-8:].round(0)}")
    print(f"  good walker (idx 0-7)  max norms: {walker_max_norm[:8].round(2)}")
    print(f"  good walker (idx 0-7)  mean lps:  {walker_mean_lp[:8].round(1)}")
    print(f"  max norm overall: {norms.max():.1f}")
    print(f"  frac samples on floor: {(lps < -1e6).mean()*100:.1f}%")
    print(f"  norm of walker with max norm  — over time, first/mid/last 5 steps:")
    bad_idx = int(walker_max_norm.argmax())
    print(f"    walker {bad_idx}: first {norms[:5, bad_idx].round(2)}")
    print(f"    walker {bad_idx}: mid   {norms[195:200, bad_idx].round(2)}")
    print(f"    walker {bad_idx}: last  {norms[-5:, bad_idx].round(2)}")
    print()
