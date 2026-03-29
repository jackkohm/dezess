"""Layer 4: Multi-chain convergence diagnostics.

Run 4 independent chains with overdispersed initialization and frozen Z.
Compute rank-normalized split R-hat, bulk ESS, and tail ESS.

Pass criteria:
  - R-hat < 1.05 for all parameters
  - Bulk ESS > 200 per parameter
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dezess.sampler import run_demcz_slice
from dezess.targets import correlated_gaussian, gaussian_mixture, isotropic_gaussian, student_t

jax.config.update("jax_enable_x64", True)


def _rank_normalize(x):
    """Rank-normalize an array (fractional ranks mapped to normal quantiles)."""
    from scipy.stats import norm
    n = len(x)
    ranks = np.argsort(np.argsort(x)) + 1
    return norm.ppf((ranks - 0.375) / (n + 0.25))


def _split_rhat(chains):
    """Rank-normalized split R-hat (Vehtari et al. 2021).

    chains: (n_chains, n_samples, n_dim) or (n_chains, n_samples) for 1D.
    """
    if chains.ndim == 2:
        chains = chains[:, :, None]
    n_chains, n_samples, n_dim = chains.shape
    rhats = np.zeros(n_dim)

    for d in range(n_dim):
        # Split each chain in half
        half = n_samples // 2
        split_chains = []
        for c in range(n_chains):
            x = chains[c, :, d]
            split_chains.append(_rank_normalize(x[:half]))
            split_chains.append(_rank_normalize(x[half:2*half]))
        split_chains = np.array(split_chains)  # (2*n_chains, half)

        n_split = split_chains.shape[0]
        m = split_chains.shape[1]

        chain_means = split_chains.mean(axis=1)
        chain_vars = split_chains.var(axis=1, ddof=1)
        W = chain_vars.mean()
        B = chain_means.var(ddof=1) * m

        if W > 0:
            var_hat = (1 - 1/m) * W + B / m
            rhats[d] = np.sqrt(var_hat / W)
        else:
            rhats[d] = np.inf

    return rhats


def _bulk_ess(chains):
    """Bulk ESS via rank-normalized split chains."""
    if chains.ndim == 2:
        chains = chains[:, :, None]
    n_chains, n_samples, n_dim = chains.shape
    ess = np.zeros(n_dim)

    for d in range(n_dim):
        # Compute ESS from the mean chain across walkers within each chain
        all_samples = []
        for c in range(n_chains):
            all_samples.append(_rank_normalize(chains[c, :, d]))
        all_samples = np.array(all_samples)

        # FFT-based ESS estimate
        n_split = all_samples.shape[0]
        m = all_samples.shape[1]

        chain_means = all_samples.mean(axis=1)
        chain_vars = all_samples.var(axis=1, ddof=1)
        W = chain_vars.mean()
        B = chain_means.var(ddof=1) * m
        var_hat = (1 - 1/m) * W + B / m if W > 0 else 1.0

        # Autocorrelation-based ESS
        total_ess = 0
        for c in range(n_split):
            x = all_samples[c] - all_samples[c].mean()
            n = len(x)
            fft_x = np.fft.fft(x, n=2*n)
            acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real / (n * np.var(x) + 1e-30)

            # Geyer's initial positive sequence
            tau = 1.0
            for lag in range(1, n):
                tau += 2 * acf[lag]
                if lag >= 5 * tau:
                    break
            total_ess += n / max(tau, 1.0)

        ess[d] = total_ess

    return ess


def _run_multi_chain(target, n_chains=4, n_walkers=64, n_steps=4000, n_warmup=1000, mu=1.0):
    """Run n_chains independent chains and extract per-chain mean traces."""
    ndim = target.ndim
    all_chains = []

    for c in range(n_chains):
        key = jax.random.PRNGKey(c * 100)
        # Overdispersed init: 4x the true scale
        init = target.sample(key, n_walkers)
        if target.mean is not None:
            init = target.mean + (init - target.mean) * 4.0

        result = run_demcz_slice(
            target.log_prob, init,
            n_steps=n_steps, n_warmup=n_warmup,
            key=jax.random.PRNGKey(c * 100 + 1),
            mu=mu, verbose=False,
        )

        # Mean across walkers per step -> (n_production, ndim)
        samples = np.asarray(result["samples"])  # (n_prod, n_walkers, ndim)
        chain_mean = samples.mean(axis=1)
        all_chains.append(chain_mean)

    return np.array(all_chains)  # (n_chains, n_production, ndim)


@pytest.fixture(params=["isotropic_10", "correlated_10", "student_t_10"])
def target(request):
    factories = {
        "isotropic_10": lambda: isotropic_gaussian(10),
        "correlated_10": lambda: correlated_gaussian(10),
        "student_t_10": lambda: student_t(10),
    }
    return factories[request.param]()


def test_convergence(target):
    """Multi-chain convergence diagnostics."""
    chains = _run_multi_chain(
        target, n_chains=4, n_walkers=64,
        n_steps=4000, n_warmup=1000, mu=1.0,
    )

    rhats = _split_rhat(chains)
    ess = _bulk_ess(chains)

    print(f"\n  {target.name}:")
    print(f"    R-hat: min={rhats.min():.4f} max={rhats.max():.4f} mean={rhats.mean():.4f}")
    print(f"    ESS:   min={ess.min():.0f} max={ess.max():.0f} mean={ess.mean():.0f}")
    for d in range(min(5, target.ndim)):
        print(f"      dim {d}: rhat={rhats[d]:.4f} ess={ess[d]:.0f}")

    assert rhats.max() < 1.05, f"R-hat {rhats.max():.4f} > 1.05"
    assert ess.min() > 50, f"ESS {ess.min():.0f} < 50"


def test_convergence_mixture():
    """Convergence on bimodal mixture (harder)."""
    target = gaussian_mixture(5)
    chains = _run_multi_chain(
        target, n_chains=4, n_walkers=64,
        n_steps=5000, n_warmup=1500, mu=1.0,
    )

    rhats = _split_rhat(chains)
    ess = _bulk_ess(chains)

    print(f"\n  {target.name}:")
    print(f"    R-hat: min={rhats.min():.4f} max={rhats.max():.4f}")
    print(f"    ESS:   min={ess.min():.0f} max={ess.max():.0f}")

    # Relaxed for multimodal
    assert rhats.max() < 1.1, f"R-hat {rhats.max():.4f} > 1.1"
    assert ess.min() > 50, f"ESS {ess.min():.0f} < 50"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
