# dezess

**DE-MCz Slice Sampler**: fully GPU-parallel ensemble slice sampling in JAX.

Combines DE-MCz Z-matrix proposals (ter Braak & Vrugt 2008) with fixed-iteration slice sampling (Karamanis & Beutler 2021). All walkers update simultaneously — no complementary sets, no while-loops, fast JIT compilation.

## Install

```bash
pip install -e .
```

## Usage

```python
import jax
import jax.numpy as jnp
from dezess import run_demcz_slice

def log_prob(x):
    return -0.5 * jnp.sum(x ** 2)

init = jax.random.normal(jax.random.PRNGKey(0), (64, 10))
result = run_demcz_slice(log_prob, init, n_steps=3000, n_warmup=500, mu=1.0)
samples = result["samples"]  # (2500, 64, 10)
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

## References

- ter Braak & Vrugt (2008), "Differential Evolution Markov Chain with snooker updater and fewer chains"
- Karamanis & Beutler (2021), "Ensemble Slice Sampling", arXiv:2002.06212
- Neal (2003), "Slice Sampling", Annals of Statistics 31(3):705-767
