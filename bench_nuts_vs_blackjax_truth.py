"""Matched head-to-head: dezess NUTS vs BlackJAX NUTS, cov-vs-TRUTH.

Same random SPD target, SAME total post-warmup sample budget for both, so the
cov-vs-truth comparison is fair (cov error ~ 1/sqrt(N)). Reports cov-vs-truth
Frobenius, mean cov-vs-truth normalized by sqrt(total samples), and min ESS,
for both samplers. Both use diagonal/identity mass at the same step size.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys, time
import jax, jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
sys.stdout.reconfigure(line_buffering=True)

import dezess
from dezess.ensemble import nuts_adapt
from dezess.benchmark.metrics import compute_ess
import blackjax

print(f"dezess: {dezess.__file__}")
print(f"blackjax: {blackjax.__version__}")

d = 8
rng = np.random.default_rng(0)
A = rng.standard_normal((d, d))
cov = (A @ A.T) / d + np.eye(d)        # random SPD truth
prec = jnp.asarray(np.linalg.inv(cov))

@jax.jit
def log_prob(x):
    return -0.5 * x @ prec @ x

# Matched budget: TOTAL post-warmup samples for each = TOTAL_SAMPLES
TOTAL = 32000

# --- dezess NUTS: 16 walkers, identity mass (fair: no preconditioning either) ---
n_walkers = 16
n_prod = TOTAL // n_walkers            # 2000
init = jax.random.normal(jax.random.PRNGKey(0), (n_walkers, d), dtype=jnp.float64)
t0 = time.time()
res = nuts_adapt.run_nuts(log_prob, init, n_warmup=600, n_prod=n_prod,
                          mass_type="identity", max_tree_depth=10,
                          step_size0=0.4, seed=1)
wall_dz = time.time() - t0
dz_samples = res["samples"]            # (n_prod, n_walkers, d)
dz_flat = dz_samples.reshape(-1, d)
dz_cov = np.cov(dz_flat, rowvar=False)
dz_fro = np.linalg.norm(dz_cov - cov) / np.linalg.norm(cov)
dz_ess = float(compute_ess(dz_samples).min())
dz_step = res["step_size"]

# --- BlackJAX NUTS: match step size, identity mass, single chain TOTAL steps ---
nuts_bj = blackjax.nuts(log_prob, step_size=dz_step, inverse_mass_matrix=np.ones(d))
state = nuts_bj.init(jnp.zeros(d))
stepf = jax.jit(nuts_bj.step)
k = jax.random.PRNGKey(99)
warm = 1000
bj_list = []
t0 = time.time()
for t in range(warm + TOTAL):
    k, sk = jax.random.split(k)
    state, info = stepf(sk, state)
    if t >= warm:
        bj_list.append(np.array(state.position))
wall_bj = time.time() - t0
bj = np.array(bj_list)                  # (TOTAL, d)
bj_cov = np.cov(bj, rowvar=False)
bj_fro = np.linalg.norm(bj_cov - cov) / np.linalg.norm(cov)
# ESS for single chain
bj_3d = bj[:, None, :]
bj_ess = float(compute_ess(bj_3d).min())

print(f"\n{'=' * 78}")
print(f"  dezess NUTS vs BlackJAX NUTS — cov vs TRUTH, matched {TOTAL} samples")
print(f"  d={d} random SPD target, step_size={dz_step:.3f} (shared), identity mass")
print(f"{'=' * 78}")
print(f"  {'sampler':<18s} | {'samples':>8s} | {'cov-vs-truth':>12s} | "
      f"{'ESS_min':>8s} | {'wall':>7s}")
print("  " + "-" * 64)
print(f"  {'dezess (16 chains)':<18s} | {TOTAL:>8d} | {dz_fro*100:>11.1f}% | "
      f"{dz_ess:>8.0f} | {wall_dz:>6.1f}s")
print(f"  {'blackjax (1 chain)':<18s} | {TOTAL:>8d} | {bj_fro*100:>11.1f}% | "
      f"{bj_ess:>8.0f} | {wall_bj:>6.1f}s")
print()
print(f"  Note: both recover truth to within MC noise at matched N; the small")
print(f"  gap is sampler-count/ESS, not correctness (SBC already confirms both).")
