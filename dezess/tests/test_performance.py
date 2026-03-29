"""Layer 6: Performance diagnostics and benchmarks.

Reports: ESS/second, MCSE, cap-hit rate, jump distance, Z-matrix span,
and wall-clock timing. All on analytical targets.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dezess.sampler import N_EXPAND, N_SHRINK, run_demcz_slice
from dezess.targets import correlated_gaussian, gaussian_mixture, isotropic_gaussian, student_t

jax.config.update("jax_enable_x64", True)


def _ess_fft(chain):
    """FFT-based effective sample size for a 1D chain."""
    n = len(chain)
    x = chain - chain.mean()
    var = np.var(x)
    if var < 1e-30:
        return 0.0
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real / (n * var)

    tau = 1.0
    for lag in range(1, n):
        tau += 2 * acf[lag]
        if lag >= 5 * tau:
            break
    return n / max(tau, 1.0)


def _run_and_diagnose(target, n_walkers=64, n_steps=5000, n_warmup=1000, mu=1.0):
    """Run sampler and compute diagnostics."""
    key = jax.random.PRNGKey(0)
    init = target.sample(key, n_walkers)

    t0 = time.time()
    result = run_demcz_slice(
        target.log_prob, init,
        n_steps=n_steps, n_warmup=n_warmup,
        key=jax.random.PRNGKey(1), mu=mu, tune=True,
        verbose=False,
    )
    wall_time = time.time() - t0

    samples_3d = np.asarray(result["samples"])  # (n_prod, n_walkers, ndim)
    n_prod, nw, ndim = samples_3d.shape

    # Mean chain across walkers per step
    mean_chain = samples_3d.mean(axis=1)  # (n_prod, ndim)

    # ESS per dimension (from the mean chain)
    ess = np.array([_ess_fft(mean_chain[:, d]) for d in range(ndim)])

    # MCSE: Monte Carlo standard error of the mean
    flat = samples_3d.reshape(-1, ndim)
    sample_mean = flat.mean(0)
    sample_std = flat.std(0)
    mcse = sample_std / np.sqrt(np.maximum(ess * nw, 1))

    # Jump distance: mean ||x_new - x_old|| per step, normalized by std
    jumps = np.diff(samples_3d, axis=0)  # (n_prod-1, n_walkers, ndim)
    jump_dist = np.sqrt((jumps ** 2).sum(axis=-1))  # (n_prod-1, n_walkers)
    norm_jump = jump_dist / (sample_std.sum() + 1e-30)
    mean_jump = norm_jump.mean()

    # Zero-move rate: fraction of walker-steps where position didn't change
    zero_moves = (jump_dist == 0.0).mean()

    # Z-matrix span: SVD of the frozen Z
    z_mat = np.asarray(result["z_matrix"])
    z_centered = z_mat - z_mat.mean(0)
    svd_vals = np.linalg.svd(z_centered, compute_uv=False)
    # Ratio of smallest to largest singular value
    sv_ratio = svd_vals[-1] / (svd_vals[0] + 1e-30) if len(svd_vals) > 0 else 0

    n_production = n_steps - n_warmup
    speed = n_production / wall_time

    return {
        "wall_time": wall_time,
        "speed_its": speed,
        "ess_min": ess.min(),
        "ess_max": ess.max(),
        "ess_mean": ess.mean(),
        "ess_per_sec_min": ess.min() / wall_time,
        "ess_per_sec_mean": ess.mean() / wall_time,
        "mcse_max": mcse.max(),
        "mcse_mean": mcse.mean(),
        "mean_jump": mean_jump,
        "zero_move_rate": zero_moves,
        "sv_ratio": sv_ratio,
        "final_mu": result["mu"],
        "ndim": ndim,
        "n_walkers": n_walkers,
        "n_production": n_production,
    }


@pytest.fixture(params=["isotropic_10", "correlated_10", "student_t_10", "mixture_5"])
def target(request):
    factories = {
        "isotropic_10": lambda: isotropic_gaussian(10),
        "correlated_10": lambda: correlated_gaussian(10),
        "student_t_10": lambda: student_t(10),
        "mixture_5": lambda: gaussian_mixture(5),
    }
    return factories[request.param]()


def test_performance(target):
    """Performance diagnostics for a single target."""
    diag = _run_and_diagnose(target, n_walkers=64, n_steps=5000, n_warmup=1000)

    print(f"\n  {target.name} (d={diag['ndim']}):")
    print(f"    Wall time:      {diag['wall_time']:.1f}s")
    print(f"    Speed:          {diag['speed_its']:.0f} it/s")
    print(f"    Final mu:       {diag['final_mu']:.4f}")
    print(f"    ESS:            min={diag['ess_min']:.0f} mean={diag['ess_mean']:.0f} max={diag['ess_max']:.0f}")
    print(f"    ESS/s:          min={diag['ess_per_sec_min']:.0f} mean={diag['ess_per_sec_mean']:.0f}")
    print(f"    MCSE:           max={diag['mcse_max']:.4f} mean={diag['mcse_mean']:.4f}")
    print(f"    Jump distance:  {diag['mean_jump']:.4f} (normalized)")
    print(f"    Zero-move rate: {diag['zero_move_rate']:.4f}")
    print(f"    Z-matrix SV ratio: {diag['sv_ratio']:.4f}")

    # Basic quality checks
    assert diag["zero_move_rate"] < 0.1, f"Zero-move rate {diag['zero_move_rate']:.2%} > 10%"
    assert diag["sv_ratio"] > 0.001, f"Z-matrix SV ratio {diag['sv_ratio']:.4f} < 0.001 (poor span)"
    assert diag["ess_min"] > 10, f"ESS min {diag['ess_min']:.0f} < 10"


def test_high_dim():
    """Performance on a 30-D isotropic Gaussian."""
    target = isotropic_gaussian(30)
    diag = _run_and_diagnose(target, n_walkers=128, n_steps=5000, n_warmup=1000)

    print(f"\n  {target.name} (d=30, 128 walkers):")
    print(f"    Speed:     {diag['speed_its']:.0f} it/s")
    print(f"    Final mu:  {diag['final_mu']:.4f}")
    print(f"    ESS:       min={diag['ess_min']:.0f} mean={diag['ess_mean']:.0f}")
    print(f"    ESS/s:     min={diag['ess_per_sec_min']:.0f}")
    print(f"    Zero-move: {diag['zero_move_rate']:.4f}")

    assert diag["zero_move_rate"] < 0.1
    assert diag["ess_min"] > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
