# dezess

**DE-MCz Slice Sampler**: fully GPU-parallel ensemble slice sampling in JAX.

Combines DE-MCz Z-matrix proposals (ter Braak & Vrugt 2008) with fixed-iteration slice sampling (Karamanis & Beutler 2021). All walkers update simultaneously — no complementary sets, no while-loops, fast JIT compilation.

## Install

```bash
pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
import dezess

result = dezess.sample(
    lambda x: -0.5 * jnp.sum(x**2),  # log-probability function
    jnp.zeros((32, 10)),               # initial positions (n_walkers, n_dim)
    n_samples=2000,
)

print(f"ESS: {result.ess_min:.0f}, R-hat: {result.rhat_max:.4f}")
dezess.print_summary(result.samples)
```

## Features

- **Scale-aware slice width**: automatically uses `|z_r1 - z_r2|` as the bracket width instead of a single global scalar. 1.3-1.5x better ESS per log-prob evaluation with zero overhead.
- **Early stopping**: set `target_ess=1000` to stop production as soon as sufficient ESS is reached.
- **Streaming diagnostics**: live ESS and R-hat during sampling.
- **Multi-direction steps**: Gibbs-like coordinate cycling with orthogonalized directions for better mixing.
- **Checkpoint/resume**: save and resume long-running jobs without re-doing warmup.
- **ArviZ integration**: convert samples to InferenceData for standard Bayesian diagnostics.
- **Automatic tuning**: `recommend_walkers()` and `estimate_n_steps()` for automatic hyperparameter selection.
- **Progress callback**: hook into the sampling loop for custom monitoring or progress bars.

## Variant Selection

```python
# For expensive log-probs: best ESS per evaluation (default)
result = dezess.sample(log_prob, init, variant="fast")

# For cheap log-probs: best wall-clock ESS/s
result = dezess.sample(log_prob, init, variant="thorough")

# Full control
from dezess import VariantConfig
config = VariantConfig(
    name="custom",
    direction="de_mcz",
    width="scale_aware",
    slice_fn="fixed",
    n_slices_per_step=3,
)
result = dezess.sample(log_prob, init, variant="custom")  # or pass config to run_variant
```

## Early Stopping

```python
# Stop when ESS reaches 500 (saves compute for expensive models)
result = dezess.sample(log_prob, init, n_samples=10000, target_ess=500)
# Might stop at 150 steps instead of 10000
```

## Checkpoint/Resume

```python
# Save after a run
result = dezess.run_variant(log_prob, init, n_steps=5000, n_warmup=1000)
dezess.save_checkpoint("run.npz", result)

# Resume for more samples later
result2 = dezess.resume("run.npz", log_prob, n_samples=5000)
```

## ArviZ Integration

```python
import arviz as az

result = dezess.sample(log_prob, init, n_samples=2000)
idata = dezess.to_inference_data(result, param_names=["mu", "sigma", "beta"])
az.plot_trace(idata)
az.summary(idata)
```

## Automatic Tuning

```python
# Recommend walker count
n_walkers = dezess.recommend_walkers(log_prob, n_dim=20)

# Estimate steps needed for target ESS
n_steps = dezess.estimate_n_steps(log_prob, init, target_ess=1000)
```

## Available Variants

| Variant | Direction | Width | Best For |
|---------|-----------|-------|----------|
| `scale_aware` (default) | DE-MCz | scale-aware | Expensive log-probs |
| `baseline` | DE-MCz | scalar | Simplicity |
| `zeus_gamma` | DE-MCz | zeus-style | Multi-scale targets |
| `snooker_stochastic` | Snooker | stochastic | Funnels, bananas |
| `multi_direction_3` | DE-MCz | scalar | Cheap log-probs (3 slices/step) |
| `whitened` | Whitened DE-MCz | scalar | Strongly correlated |

## Performance vs emcee

On standard benchmarks (64 walkers, 4000 production steps, CPU):

| Target | dezess ESS | emcee ESS | dezess R-hat |
|--------|:---:|:---:|:---:|
| isotropic_10 | **97** | 16 | 1.005 |
| correlated_10 | **98** | 28 | 1.006 |
| mixture_5 | **283** | 37 | 1.005 |

dezess produces 2-7x more effective samples with better convergence diagnostics.

## Tests

```bash
pip install -e ".[test]"
pytest dezess/tests/
```

19 tests covering Gaussian moment correctness, reference matching, and performance.

## References

- ter Braak & Vrugt (2008), "Differential Evolution Markov Chain with snooker updater and fewer chains"
- Karamanis & Beutler (2021), "Ensemble Slice Sampling", arXiv:2002.06212
- Neal (2003), "Slice Sampling", Annals of Statistics 31(3):705-767
