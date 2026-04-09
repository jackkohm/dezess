#!/usr/bin/env python
"""Basic example: sample from a 10D correlated Gaussian.

Demonstrates the simplest dezess workflow:
1. Define a log-probability function
2. Call dezess.sample()
3. Inspect results with dezess.diagnose() and dezess.print_summary()
"""

import jax
import jax.numpy as jnp
import numpy as np
import dezess

# --- Define the target distribution ---
# A 10D Gaussian with non-trivial covariance
ndim = 10
rng = np.random.RandomState(42)
A = rng.randn(ndim, ndim)
cov = A @ A.T / ndim + np.eye(ndim)
prec = jnp.array(np.linalg.inv(cov))
mu = jnp.array(rng.randn(ndim))


@jax.jit
def log_prob(x):
    d = x - mu
    return -0.5 * d @ prec @ d


# --- Sample ---
# Initialize walkers near zero (dezess will explore from here)
init = dezess.init_walkers(32, ndim, center=np.array(mu), scale=0.5)

result = dezess.sample(
    log_prob,
    init,
    n_samples=2000,
    seed=42,
)

# --- Diagnose ---
dezess.diagnose(result)

# --- Summary statistics ---
param_names = [f"x{i}" for i in range(ndim)]
dezess.print_summary(result.samples, param_names=param_names)

# --- Verify against known truth ---
flat = dezess.flatten_samples(result.samples)
print("\nVerification against known truth:")
print(f"  Mean error: {np.max(np.abs(np.mean(flat, axis=0) - np.array(mu))):.4f}")
print(f"  Var error:  {np.max(np.abs(np.var(flat, axis=0) - np.diag(np.array(cov)))):.4f}")
