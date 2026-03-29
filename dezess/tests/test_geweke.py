"""Layer 3: One-step Geweke invariance tests.

If x_0 ~ pi and the kernel preserves pi, then x_1 ~ pi after one sweep.
We draw exact samples from pi, apply one production sweep with a frozen
Z-matrix, and verify the output distribution matches pi using KS tests.

Tests multiple targets and multiple Z-matrix conditions (well-calibrated,
overdispersed, underdispersed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from dezess.sampler import _safe_log_prob, _slice_sample_fixed, _update_one_walker
from dezess.targets import (
    ALL_TARGETS,
    correlated_gaussian,
    gaussian_mixture,
    isotropic_gaussian,
    neals_funnel,
    student_t,
)

jax.config.update("jax_enable_x64", True)


def _one_sweep_slice(target, z_matrix, z_count, x_all, mu, key):
    """Apply one production sweep to all walkers with frozen Z."""
    z_max_size = z_matrix.shape[0]

    def _update(x, k):
        lp = target.log_prob(x)
        return _update_one_walker(target.log_prob, x, lp, z_matrix, z_count, mu, k)

    keys = jax.random.split(key, x_all.shape[0])
    x_new, _, _, _, _ = jax.jit(jax.vmap(_update))(x_all, keys)
    return x_new


@pytest.fixture(
    params=["isotropic_10", "correlated_10", "student_t_10", "mixture_5"],
)
def target(request):
    factories = {
        "isotropic_10": lambda: isotropic_gaussian(10),
        "correlated_10": lambda: correlated_gaussian(10),
        "student_t_10": lambda: student_t(10),
        "mixture_5": lambda: gaussian_mixture(5),
    }
    return factories[request.param]()


@pytest.fixture(params=["calibrated", "overdispersed", "underdispersed"])
def z_type(request):
    return request.param


def _build_z(target, z_type, key, n_z=5000, z_max_size=10000):
    """Build a frozen Z-matrix of the specified type."""
    samples = target.sample(key, n_z)
    if z_type == "overdispersed":
        # Inflate around the mean
        if target.mean is not None:
            samples = target.mean + (samples - target.mean) * 3.0
        else:
            samples = samples * 3.0
    elif z_type == "underdispersed":
        if target.mean is not None:
            samples = target.mean + (samples - target.mean) * 0.3
        else:
            samples = samples * 0.3

    z_padded = jnp.zeros((z_max_size, target.ndim), dtype=jnp.float64)
    z_padded = z_padded.at[:n_z].set(samples)
    z_count = jnp.int32(n_z)
    return z_padded, z_count


def test_geweke_one_step(target, z_type):
    """One production sweep preserves the target distribution."""
    n_samples = 5000
    key = jax.random.PRNGKey(42)
    k_z, k_x0, k_sweep = jax.random.split(key, 3)

    z_padded, z_count = _build_z(target, z_type, k_z)
    x0 = target.sample(k_x0, n_samples)
    mu = jnp.float64(1.0)

    x1 = _one_sweep_slice(target, z_padded, z_count, x0, mu, k_sweep)

    x0_np = np.asarray(x0)
    x1_np = np.asarray(x1)

    # Per-dimension KS test
    n_dims = target.ndim
    p_values = []
    for j in range(n_dims):
        stat, p = stats.ks_2samp(x0_np[:, j], x1_np[:, j])
        p_values.append(p)

    p_values = np.array(p_values)
    # Bonferroni correction: reject if any p < alpha/n_dims
    alpha = 0.01
    bonf_threshold = alpha / n_dims
    n_reject = (p_values < bonf_threshold).sum()
    min_p = p_values.min()

    # Also check marginal means and stds
    mean_diff = np.max(np.abs(x0_np.mean(0) - x1_np.mean(0)))
    std_ratio = np.max(np.abs(x0_np.std(0) / x1_np.std(0) - 1.0))

    print(f"\n  {target.name} (Z={z_type}):")
    print(f"    KS min_p={min_p:.4f} (Bonferroni threshold={bonf_threshold:.4f})")
    print(f"    KS rejections: {n_reject}/{n_dims}")
    print(f"    max |mean diff|: {mean_diff:.4f}")
    print(f"    max |std ratio - 1|: {std_ratio:.4f}")

    assert n_reject == 0, (
        f"KS rejected {n_reject}/{n_dims} dimensions "
        f"(min_p={min_p:.6f}, threshold={bonf_threshold:.6f})"
    )


def test_geweke_funnel():
    """Separate test for Neal's funnel (harder, more forgiving threshold)."""
    target = neals_funnel(10)
    n_samples = 5000
    key = jax.random.PRNGKey(99)
    k_z, k_x0, k_sweep = jax.random.split(key, 3)

    # Well-calibrated Z only (funnel is hard enough)
    z_padded, z_count = _build_z(target, "calibrated", k_z)
    x0 = target.sample(k_x0, n_samples)
    mu = jnp.float64(1.0)

    x1 = _one_sweep_slice(target, z_padded, z_count, x0, mu, k_sweep)

    x0_np = np.asarray(x0)
    x1_np = np.asarray(x1)

    # Just check the top dimension (x_0) which has the widest range
    stat, p = stats.ks_2samp(x0_np[:, 0], x1_np[:, 0])
    print(f"\n  Funnel x_0: KS p={p:.4f}")

    # More forgiving: funnel is hard for fixed-mu samplers
    assert p > 0.001, f"Funnel KS test failed: p={p:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
