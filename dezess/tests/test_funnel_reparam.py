"""Tests for funnel reparameterization and block-Gibbs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def test_identity_roundtrip():
    from dezess.transforms import identity

    t = identity()
    x = jnp.array([1.0, 2.0, 3.0])
    z = t.inverse(x)
    x_rec = t.forward(z)
    np.testing.assert_allclose(x_rec, x, atol=1e-14)
    assert t.log_det_jac(z) == 0.0


def test_ncp_funnel_roundtrip():
    """forward(inverse(x)) == x for the funnel transform."""
    from dezess.transforms import non_centered_funnel

    t = non_centered_funnel(width_idx=0, offset_indices=[1, 2, 3, 4])

    for log_w in [-3.0, 0.0, 3.0, 6.0]:
        x = jnp.array([log_w, 0.5, -1.0, 2.0, -0.3])
        z = t.inverse(x)
        x_rec = t.forward(z)
        np.testing.assert_allclose(x_rec, x, atol=1e-12)

    x = jnp.array([2.0, 1.0, -1.0, 0.5, -0.5])
    z = t.inverse(x)
    assert z[0] == x[0]
    expected_z_offsets = x[1:] / jnp.exp(x[0] / 2.0)
    np.testing.assert_allclose(z[1:], expected_z_offsets, atol=1e-12)


def test_ncp_funnel_jacobian():
    """log_det_jac matches numerical finite-difference Jacobian."""
    from dezess.transforms import non_centered_funnel

    t = non_centered_funnel(width_idx=0, offset_indices=[1, 2, 3])
    z = jnp.array([2.0, 0.5, -0.3, 1.0])

    ldj_analytical = t.log_det_jac(z)

    jac = jax.jacobian(t.forward)(z)
    _, ldj_numerical = jnp.linalg.slogdet(jac)

    np.testing.assert_allclose(ldj_analytical, ldj_numerical, atol=1e-10)


def test_block_transform_roundtrip():
    """block_transform applies different transforms to different dims."""
    from dezess.transforms import identity, non_centered_funnel, block_transform

    t = block_transform(
        transforms=[identity(), non_centered_funnel(width_idx=0, offset_indices=[1, 2, 3, 4, 5, 6, 7])],
        index_lists=[list(range(4)), list(range(4, 12))],
        ndim=12,
    )
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (12,), dtype=jnp.float64)
    z = t.inverse(x)
    x_rec = t.forward(z)
    np.testing.assert_allclose(x_rec, x, atol=1e-12)

    jac = jax.jacobian(t.forward)(z)
    _, ldj_num = jnp.linalg.slogdet(jac)
    np.testing.assert_allclose(t.log_det_jac(z), ldj_num, atol=1e-10)


def test_multi_funnel_63d():
    """multi_funnel helper builds correct 63D transform."""
    from dezess.transforms import multi_funnel

    t = multi_funnel(
        n_potential=7,
        funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)],
    )
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (63,), dtype=jnp.float64)
    z = t.inverse(x)
    x_rec = t.forward(z)
    np.testing.assert_allclose(x_rec, x, atol=1e-12)

    np.testing.assert_allclose(z[:7], x[:7], atol=1e-14)

    jac = jax.jacobian(t.forward)(z)
    _, ldj_num = jnp.linalg.slogdet(jac)
    np.testing.assert_allclose(t.log_det_jac(z), ldj_num, atol=1e-10)


def test_sample_api_with_transform():
    """dezess.sample() accepts and passes through transform."""
    import dezess
    from dezess.transforms import non_centered_funnel
    from dezess.targets import neals_funnel

    target = neals_funnel(10)
    t = non_centered_funnel(width_idx=0, offset_indices=list(range(1, 10)))

    key = jax.random.PRNGKey(0)
    init = target.sample(key, 64)

    result = dezess.sample(
        target.log_prob, init, n_samples=1000, n_warmup=1000,
        transform=t, verbose=False,
    )
    assert result.samples.shape[2] == 10


def test_run_variant_with_transform():
    """run_variant with NonCenteredFunnel produces correct samples on 10D funnel."""
    from dezess.transforms import non_centered_funnel
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig
    from dezess.targets import neals_funnel

    target = neals_funnel(10)
    t = non_centered_funnel(width_idx=0, offset_indices=list(range(1, 10)))

    config = VariantConfig(
        name="test_ncp",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    )

    key = jax.random.PRNGKey(0)
    init = target.sample(key, 64)

    result = run_variant(
        target.log_prob, init, n_steps=3000,
        config=config, n_warmup=2000,
        transform=t, verbose=False,
    )
    samples = np.array(result["samples"]).reshape(-1, 10)

    # log-width variance should be close to 9 (Neal's funnel prior)
    lw_var = np.var(samples[:, 0])
    assert 4.0 < lw_var < 15.0, f"log-width var={lw_var}, expected ~9"


def test_geweke_transform_funnel():
    """One-step Geweke invariance with NonCenteredFunnel transform.

    Draw x ~ funnel, transform to z = inverse(x), apply one variant sweep
    in z-space, transform back to x = forward(z). Output should still be
    distributed as the funnel.
    """
    from dezess.transforms import non_centered_funnel
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig
    from dezess.targets import neals_funnel
    from scipy import stats

    ndim = 10
    target = neals_funnel(ndim)
    t = non_centered_funnel(width_idx=0, offset_indices=list(range(1, ndim)))

    config = VariantConfig(
        name="geweke_ncp",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    )

    key = jax.random.PRNGKey(123)
    n_walkers = 200
    x_exact = target.sample(key, n_walkers)

    n_warmup_steps = 500
    result = run_variant(
        target.log_prob, x_exact, n_steps=n_warmup_steps + 1, config=config,
        n_warmup=n_warmup_steps, transform=t, verbose=False,
    )
    x_out = np.array(result["samples"][0])

    _, p_val = stats.kstest(x_out[:, 0] / 3.0, "norm")
    assert p_val > 0.001, f"Geweke failed on log-width dim: KS p={p_val:.4f}"


def test_block_gibbs_block_coupled():
    """Block-Gibbs on block_coupled_gaussian should converge (R-hat < 1.2)."""
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig
    from dezess.targets_stream import block_coupled_gaussian
    from dezess.benchmark.metrics import compute_rhat

    target = block_coupled_gaussian()  # 63D

    config = VariantConfig(
        name="block_gibbs_test",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs={"block_sizes": [7, 14, 14, 14, 14]},
    )

    key = jax.random.PRNGKey(0)
    init = target.sample(key, 128)

    result = run_variant(
        target.log_prob, init, n_steps=5000,
        config=config, n_warmup=4000, verbose=False,
    )
    samples = np.array(result["samples"])
    rhat = compute_rhat(samples)
    rhat_max = float(np.max(rhat))
    assert rhat_max < 1.2, f"Block-Gibbs R-hat={rhat_max:.4f} on block_coupled (want < 1.2)"


@pytest.mark.slow
def test_transform_funnel_63d():
    """NonCenteredFunnel transform alone on funnel_63d."""
    from dezess.transforms import multi_funnel
    from dezess.core.loop import run_variant
    from dezess.benchmark.registry import VARIANTS
    from dezess.targets_stream import funnel_63d
    from dezess.benchmark.metrics import compute_rhat

    target = funnel_63d()
    t = multi_funnel(n_potential=7, funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)])

    config = VARIANTS["scale_aware"]
    key = jax.random.PRNGKey(42)
    init = target.sample(key, 128)

    result = run_variant(
        target.log_prob, init, n_steps=8000,
        config=config, n_warmup=6000,
        transform=t, verbose=False,
    )
    samples = np.array(result["samples"])
    rhat = compute_rhat(samples)
    rhat_max = float(np.max(rhat))

    # Transform alone: observed R-hat ~1.07
    assert rhat_max < 1.2, f"Transform-only R-hat={rhat_max:.4f} on funnel_63d (want < 1.2)"

    # Check log-width variance for each funnel (observed ~8.2-8.9)
    flat = samples.reshape(-1, 63)
    for f_idx in range(4):
        lw_idx = 7 + f_idx * 14
        lw_var = float(np.var(flat[:, lw_idx]))
        assert 4.5 < lw_var < 13.5, (
            f"Funnel {f_idx} log-width var={lw_var:.2f}, want [4.5, 13.5]"
        )


@pytest.mark.slow
def test_block_gibbs_funnel_63d():
    """Block-Gibbs alone on funnel_63d (no transform)."""
    from dezess.core.loop import run_variant
    from dezess.benchmark.registry import VARIANTS
    from dezess.targets_stream import funnel_63d
    from dezess.benchmark.metrics import compute_rhat

    target = funnel_63d()
    config = VARIANTS["block_gibbs_scale_aware"]

    key = jax.random.PRNGKey(42)
    init = target.sample(key, 128)

    result = run_variant(
        target.log_prob, init, n_steps=8000,
        config=config, n_warmup=6000, verbose=False,
    )
    samples = np.array(result["samples"])
    rhat = compute_rhat(samples)
    rhat_max = float(np.max(rhat))

    # Block-Gibbs alone: observed R-hat ~2.8. Should improve over baseline (3.7)
    assert rhat_max < 3.5, f"Block-Gibbs R-hat={rhat_max:.4f} on funnel_63d (want < 3.5)"


@pytest.mark.slow
def test_combined_funnel_63d():
    """Combined transform + block-Gibbs on funnel_63d — the main success criterion."""
    from dezess.transforms import multi_funnel
    from dezess.core.loop import run_variant
    from dezess.benchmark.registry import VARIANTS
    from dezess.targets_stream import funnel_63d
    from dezess.benchmark.metrics import compute_rhat

    target = funnel_63d()
    t = multi_funnel(n_potential=7, funnel_blocks=[(7, 14), (21, 14), (35, 14), (49, 14)])
    config = VARIANTS["block_gibbs_transform"]

    key = jax.random.PRNGKey(42)
    init = target.sample(key, 128)

    result = run_variant(
        target.log_prob, init, n_steps=8000,
        config=config, n_warmup=6000,
        transform=t, verbose=False,
    )
    samples = np.array(result["samples"])
    rhat = compute_rhat(samples)
    rhat_max = float(np.max(rhat))

    # SUCCESS CRITERION: R-hat < 1.2 (observed ~1.02)
    assert rhat_max < 1.2, f"Combined R-hat={rhat_max:.4f} on funnel_63d (want < 1.2)"

    # Check log-width variance for each funnel (observed ~8.2-8.4)
    flat = samples.reshape(-1, 63)
    for f_idx in range(4):
        lw_idx = 7 + f_idx * 14
        lw_var = float(np.var(flat[:, lw_idx]))
        assert 4.5 < lw_var < 13.5, (
            f"Funnel {f_idx} log-width var={lw_var:.2f}, want [4.5, 13.5]"
        )
