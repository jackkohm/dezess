#!/usr/bin/env python
"""Compare dezess vs emcee on a correlated Gaussian.

Runs both samplers on the same target and reports ESS, R-hat, and timing.

Requires: pip install emcee
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import dezess

jax.config.update("jax_enable_x64", True)

# --- Target: 10D correlated Gaussian ---
ndim = 10
rng = np.random.RandomState(42)
A = rng.randn(ndim, ndim)
cov = A @ A.T / ndim + np.eye(ndim)
prec = np.linalg.inv(cov)
mu = rng.randn(ndim)

prec_jax = jnp.array(prec)
mu_jax = jnp.array(mu)


@jax.jit
def log_prob_jax(x):
    d = x - mu_jax
    return -0.5 * d @ prec_jax @ d


def log_prob_numpy(x):
    d = x - mu
    return -0.5 * d @ prec @ d


n_walkers = 64
n_production = 4000
n_warmup = 1000

# --- dezess ---
print("Running dezess...")
init_d = jax.random.normal(jax.random.PRNGKey(42), (n_walkers, ndim)) * 0.1 + mu_jax
t0 = time.time()
result_d = dezess.sample(
    log_prob_jax, init_d,
    n_samples=n_production, seed=42, verbose=False,
)
t_dezess = time.time() - t0

# --- emcee ---
try:
    import emcee
    print("Running emcee...")
    init_e = np.array(init_d)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob_numpy)
    t0 = time.time()
    sampler.run_mcmc(init_e, n_warmup, progress=False)
    sampler.reset()
    sampler.run_mcmc(sampler._previous_state, n_production, progress=False)
    t_emcee = time.time() - t0

    # ESS
    from dezess.benchmark.metrics import compute_ess, compute_rhat
    ess_e = compute_ess(sampler.get_chain())
    rhat_e = compute_rhat(sampler.get_chain())
    has_emcee = True
except ImportError:
    print("emcee not installed, skipping comparison")
    has_emcee = False

# --- Results ---
print(f"\n{'='*60}")
print(f"  dezess vs emcee — {ndim}D correlated Gaussian")
print(f"  {n_walkers} walkers, {n_production} production steps")
print(f"{'='*60}")
print(f"{'':>15s} {'dezess':>12s}", end="")
if has_emcee:
    print(f" {'emcee':>12s}", end="")
print()

print(f"{'Wall time':>15s} {t_dezess:12.2f}s", end="")
if has_emcee:
    print(f" {t_emcee:12.2f}s", end="")
print()

print(f"{'ESS min':>15s} {result_d.ess_min:12.0f}", end="")
if has_emcee:
    print(f" {np.min(ess_e):12.0f}", end="")
print()

print(f"{'R-hat max':>15s} {result_d.rhat_max:12.4f}", end="")
if has_emcee:
    print(f" {np.max(rhat_e):12.4f}", end="")
print()

print(f"{'ESS/s':>15s} {result_d.ess_min/t_dezess:12.1f}", end="")
if has_emcee:
    print(f" {np.min(ess_e)/t_emcee:12.1f}", end="")
print()

if has_emcee:
    ratio = result_d.ess_min / max(np.min(ess_e), 1)
    print(f"\ndezess produces {ratio:.1f}x more effective samples than emcee")
