#!/usr/bin/env python
"""Run full benchmark comparison of all sampler variants."""

import os
import sys
import gc
import time

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Limit GPU memory growth
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

print(f"JAX devices: {jax.devices()}", flush=True)

from dezess.benchmark.runner import run_single
from dezess.benchmark.compare import results_to_table, summary_report

VARIANTS = [
    'baseline', 'snooker', 'snooker_stochastic',
    'per_direction_width', 'stochastic_width',
    'momentum_03', 'pca_directions', 'riemannian',
    'flow_directions', 'delayed_rejection', 'adaptive_budget',
    'parallel_tempering', 'full_kitchen_sink',
]

TARGETS = [
    'isotropic_10', 'correlated_10', 'student_t_10', 'mixture_5',
    'rosenbrock_4', 'ill_conditioned_10', 'ring_2', 'funnel_10',
]

all_metrics = []
total = len(VARIANTS) * len(TARGETS)
idx = 0
t_start = time.time()

for vname in VARIANTS:
    for tname in TARGETS:
        idx += 1
        t0 = time.time()
        try:
            m = run_single(vname, tname, n_walkers=64, n_steps=3000,
                          n_warmup=500, seed=42, verbose=False)
            m['trial'] = 0
            all_metrics.append(m)
            dt = time.time() - t0
            print(f'[{idx}/{total}] {vname:25s} x {tname:25s}: '
                  f'ESS/s={m["ess_per_sec_min"]:8.1f}  ({dt:.1f}s)', flush=True)
        except Exception as e:
            dt = time.time() - t0
            print(f'[{idx}/{total}] {vname:25s} x {tname:25s}: '
                  f'ERROR {e}  ({dt:.1f}s)', flush=True)
            all_metrics.append({
                'variant': vname, 'target': tname, 'trial': 0, 'error': str(e)
            })

        # Cleanup between runs
        gc.collect()

print(f'\nTotal benchmark time: {time.time() - t_start:.0f}s\n', flush=True)
print(results_to_table(all_metrics), flush=True)
print(summary_report(all_metrics), flush=True)
