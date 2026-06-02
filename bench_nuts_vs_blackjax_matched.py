"""dezess NUTS vs BlackJAX at MATCHED layout — fair ESS comparison.

Two configs, each with the SAME total samples (16000) and the SAME chain
layout for both samplers, so compute_ess sees identical-shaped series and any
ESS gap is real sampler difference (not an estimator artifact):

  config A: 1 chain  x 16000 steps   (no vmap slowest-walker tax for either)
  config B: 16 chains x 1000 steps    (both vmapped -> both pay the tax)

Both use the SAME step size (dezess-adapted) and identity mass. Reports
cov-vs-truth, ESS_min, ESS/sample, ESS/sec for each sampler in each config.
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
cov = (A @ A.T) / d + np.eye(d)
prec = jnp.asarray(np.linalg.inv(cov))

@jax.jit
def log_prob(x):
    return -0.5 * x @ prec @ x

TOTAL = 16000
inv_mass = np.ones(d)

# adapt a step size once with dezess, share it with both samplers
_res = nuts_adapt.run_nuts(
    log_prob, jax.random.normal(jax.random.PRNGKey(0), (16, d), dtype=jnp.float64),
    n_warmup=600, n_prod=10, mass_type="identity", max_tree_depth=10,
    step_size0=0.4, seed=1)
STEP = _res["step_size"]
print(f"shared step_size = {STEP:.3f}\n")


def metrics(samples_3d, wall, label):
    flat = samples_3d.reshape(-1, d)
    emp = np.cov(flat, rowvar=False)
    fro = np.linalg.norm(emp - cov) / np.linalg.norm(cov)
    ess = float(compute_ess(samples_3d).min())
    n = flat.shape[0]
    return {"label": label, "fro": fro, "ess": ess,
            "ess_per_k": ess / n * 1000, "ess_per_s": ess / wall, "wall": wall}


def run_dezess(n_chains, n_prod, seed):
    init = jax.random.normal(jax.random.PRNGKey(seed), (n_chains, d), dtype=jnp.float64)
    t0 = time.time()
    res = nuts_adapt.run_nuts(log_prob, init, n_warmup=600, n_prod=n_prod,
                              mass_type="identity", max_tree_depth=10,
                              step_size_init=STEP, L_init=np.eye(d), skip_warmup=True,
                              seed=seed)
    return metrics(res["samples"], time.time() - t0, f"dezess {n_chains}ch")


def run_blackjax(n_chains, n_prod, seed):
    kern = blackjax.nuts(log_prob, step_size=STEP, inverse_mass_matrix=inv_mass)
    k = jax.random.PRNGKey(seed)
    warm = 600
    if n_chains == 1:
        state = kern.init(jnp.zeros(d))
        stepf = jax.jit(kern.step)
        buf = []
        t0 = time.time()
        for t in range(warm + n_prod):
            k, sk = jax.random.split(k)
            state, _ = stepf(sk, state)
            if t >= warm:
                buf.append(np.array(state.position))
        wall = time.time() - t0
        samples = np.array(buf)[:, None, :]            # (n_prod, 1, d)
    else:
        inits = jax.random.normal(jax.random.PRNGKey(seed), (n_chains, d), dtype=jnp.float64)
        states = jax.vmap(kern.init)(inits)
        stepf = jax.jit(jax.vmap(kern.step))
        buf = []
        t0 = time.time()
        for t in range(warm + n_prod):
            k, sk = jax.random.split(k)
            sks = jax.random.split(sk, n_chains)
            states, _ = stepf(sks, states)
            if t >= warm:
                buf.append(np.array(states.position))
        wall = time.time() - t0
        samples = np.array(buf)                          # (n_prod, n_chains, d)
    return metrics(samples, wall, f"blackjax {n_chains}ch")


CONFIGS = [("A: 1 chain  x 16000", 1, TOTAL),
           ("B: 16 chains x 1000", 16, TOTAL // 16)]

print(f"{'=' * 92}")
print(f"  dezess vs BlackJAX — matched layout, {TOTAL} total samples, shared step, identity mass")
print(f"{'=' * 92}")
hdr = (f"  {'config':<20s} | {'sampler':<14s} | {'cov-truth':>9s} | "
       f"{'ESS_min':>8s} | {'ESS/1k':>7s} | {'ESS/s':>8s} | {'wall':>7s}")
print(hdr); print("  " + "-" * (len(hdr) - 2))
for cfg_label, nc, npr in CONFIGS:
    for runner in (run_dezess, run_blackjax):
        m = runner(nc, npr, seed=1)
        print(f"  {cfg_label:<20s} | {m['label']:<14s} | {m['fro']*100:>8.1f}% | "
              f"{m['ess']:>8.0f} | {m['ess_per_k']:>7.1f} | {m['ess_per_s']:>8.0f} | "
              f"{m['wall']:>6.1f}s", flush=True)
