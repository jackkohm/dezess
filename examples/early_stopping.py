#!/usr/bin/env python
"""Example: early stopping with target ESS.

For expensive models, you want the minimum number of samples.
Set target_ess to stop as soon as you have enough effective samples.
"""

import jax
import jax.numpy as jnp
import dezess

# Simple target (pretend it's expensive)
@jax.jit
def log_prob(x):
    return -0.5 * jnp.sum(x ** 2)

init = dezess.init_walkers(64, 20, scale=0.1)

# Without early stopping: runs all 10000 steps
result_full = dezess.sample(
    log_prob, init, n_samples=10000, seed=42, verbose=False,
)
print(f"Full run: {result_full.n_steps} steps, ESS={result_full.ess_min:.0f}")

# With early stopping: stops when ESS reaches 500
result_early = dezess.sample(
    log_prob, init, n_samples=10000, target_ess=500, seed=42, verbose=False,
)
print(f"Early stop: {result_early.n_steps} steps, ESS={result_early.ess_min:.0f}")
print(f"Speedup: {result_full.n_steps / result_early.n_steps:.0f}x fewer steps")
