"""Tests for hybrid complementary + Z-matrix direction source."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig

jax.config.update("jax_enable_x64", True)


def _make_bg_mh_dr_config(complementary_prob=0.0):
    """Helper: build a bg_MH+DR config with optional complementary_prob."""
    ens_kwargs = {
        "block_sizes": [7, 14],
        "use_mh": True,
        "delayed_rejection": True,
    }
    if complementary_prob > 0.0:
        ens_kwargs["complementary_prob"] = complementary_prob
    return VariantConfig(
        name=f"bg_mh_dr_cp{complementary_prob}",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="block_gibbs",
        check_nans=False,
        width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )


def test_complementary_prob_zero_matches_current_bg_mh_dr():
    """complementary_prob=0.0 (or unset) produces identical samples to current bg_MH+DR."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 21)) * 0.1

    result_explicit = run_variant(
        log_prob, init, n_steps=300,
        config=_make_bg_mh_dr_config(complementary_prob=0.0),
        n_warmup=100, key=jax.random.PRNGKey(0), verbose=False,
    )

    result_default = run_variant(
        log_prob, init, n_steps=300,
        config=_make_bg_mh_dr_config(),
        n_warmup=100, key=jax.random.PRNGKey(0), verbose=False,
    )

    np.testing.assert_array_equal(
        np.array(result_explicit["samples"]),
        np.array(result_default["samples"]),
        err_msg="complementary_prob=0.0 must be byte-identical to default bg_MH+DR",
    )


def test_complementary_prob_half_recovers_gaussian_variance():
    """complementary_prob=0.5 on 21D Gaussian: variance approx 1.0."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 21)) * 0.1

    config = _make_bg_mh_dr_config(complementary_prob=0.5)
    result = run_variant(
        log_prob, init, n_steps=3000, config=config,
        n_warmup=1500, verbose=False,
    )
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 21)
    mean_var = float(np.var(flat, axis=0).mean())
    assert 0.7 < mean_var < 1.3, f"mean_var={mean_var:.4f}, expected ~1.0"


def test_complementary_prob_one_recovers_gaussian_variance():
    """complementary_prob=1.0 (pure complementary) on 21D Gaussian."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x**2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 21)) * 0.1

    config = _make_bg_mh_dr_config(complementary_prob=1.0)
    result = run_variant(
        log_prob, init, n_steps=3000, config=config,
        n_warmup=1500, verbose=False,
    )
    samples = np.array(result["samples"])
    flat = samples.reshape(-1, 21)
    mean_var = float(np.var(flat, axis=0).mean())
    assert 0.7 < mean_var < 1.3, f"mean_var={mean_var:.4f}, expected ~1.0"


def test_complementary_helps_with_biased_warmup():
    """When warmup produces a biased Z-matrix, complementary_prob>0 shouldn't hurt."""
    ndim = 10
    rng = np.random.default_rng(42)
    A = rng.standard_normal((ndim, ndim))
    Q, _ = np.linalg.qr(A)
    evals = np.linspace(1.0, 10.0, ndim)
    cov = Q @ np.diag(evals) @ Q.T
    cov = (cov + cov.T) / 2
    prec = jnp.array(np.linalg.inv(cov), dtype=jnp.float64)

    @jax.jit
    def log_prob(x):
        return -0.5 * x @ prec @ x

    # Initialize FAR from the mean to bias the Z-matrix toward climb history
    init = jax.random.normal(jax.random.PRNGKey(42), (32, ndim)) * 0.01 + 5.0

    config_zmat = _make_bg_mh_dr_config(complementary_prob=0.0)
    config_zmat = config_zmat._replace(
        ensemble_kwargs={**config_zmat.ensemble_kwargs,
                         "block_sizes": [5, 5]}
    )
    result_zmat = run_variant(
        log_prob, init, n_steps=2000, config=config_zmat,
        n_warmup=500, key=jax.random.PRNGKey(0), verbose=False,
    )

    config_hybrid = _make_bg_mh_dr_config(complementary_prob=0.5)
    config_hybrid = config_hybrid._replace(
        ensemble_kwargs={**config_hybrid.ensemble_kwargs,
                         "block_sizes": [5, 5]}
    )
    result_hybrid = run_variant(
        log_prob, init, n_steps=2000, config=config_hybrid,
        n_warmup=500, key=jax.random.PRNGKey(0), verbose=False,
    )

    samples_zmat = np.array(result_zmat["samples"]).reshape(-1, ndim)
    samples_hybrid = np.array(result_hybrid["samples"]).reshape(-1, ndim)

    final_zmat_mean = np.linalg.norm(samples_zmat[-100:].mean(axis=0))
    final_hybrid_mean = np.linalg.norm(samples_hybrid[-100:].mean(axis=0))

    # Hybrid should not be dramatically worse than pure Z-matrix
    assert final_hybrid_mean < final_zmat_mean + 1.0, \
        f"hybrid={final_hybrid_mean:.2f}, zmat={final_zmat_mean:.2f}"


def _make_bg_mh_dr_gamma_config(complementary_prob=0.0, gamma_jump_prob=0.0):
    """bg_MH+DR config with optional cp + gamma_jump."""
    ens_kwargs = {
        "block_sizes": [7, 14],
        "use_mh": True,
        "delayed_rejection": True,
    }
    if complementary_prob > 0.0:
        ens_kwargs["complementary_prob"] = complementary_prob
    if gamma_jump_prob > 0.0:
        ens_kwargs["gamma_jump_prob"] = gamma_jump_prob
    return VariantConfig(
        name=f"bg_mh_dr_cp{complementary_prob}_gj{gamma_jump_prob}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )


def test_gamma_jump_zero_matches_baseline_bg_mh_dr():
    """gamma_jump_prob=0.0 must produce byte-identical samples to current bg_MH+DR."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 21)) * 0.1

    res_explicit = run_variant(
        log_prob, init, n_steps=300,
        config=_make_bg_mh_dr_gamma_config(complementary_prob=0.0, gamma_jump_prob=0.0),
        n_warmup=100, key=jax.random.PRNGKey(0), verbose=False,
    )
    res_default = run_variant(
        log_prob, init, n_steps=300,
        config=_make_bg_mh_dr_config(complementary_prob=0.0),  # uses original helper
        n_warmup=100, key=jax.random.PRNGKey(0), verbose=False,
    )

    np.testing.assert_array_equal(
        np.array(res_explicit["samples"]),
        np.array(res_default["samples"]),
        err_msg="gamma_jump_prob=0.0 must be byte-identical to baseline bg_MH+DR",
    )


@pytest.mark.parametrize("gj", [0.05, 0.10])
def test_gamma_jump_recovers_gaussian_variance(gj):
    """gamma_jump_prob > 0 with bg_MH+DR must still recover unit Gaussian variance."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(0), (64, 21)) * 0.5

    result = run_variant(
        log_prob, init, n_steps=2000,
        config=_make_bg_mh_dr_gamma_config(complementary_prob=0.5, gamma_jump_prob=gj),
        n_warmup=500, key=jax.random.PRNGKey(0), verbose=False,
    )
    samples = np.array(result["samples"]).reshape(-1, 21)
    emp_var = samples.var(axis=0)
    rel_err = np.abs(emp_var - 1.0)
    # Tolerance reflects production-relevant gj values (0.05–0.10). Higher gj
    # would inject more rejected proposals and need a looser bound.
    assert rel_err.max() < 0.25, (
        f"gj={gj}: worst per-dim variance error {rel_err.max():.2%} > 25%; "
        f"emp_var={emp_var}"
    )


# ────────────────────────────────────────────────────────────────────────
# Snooker (Braak 2008) MH proposal regression tests
# ────────────────────────────────────────────────────────────────────────


def _make_bg_mh_dr_snooker_config(
    complementary_prob=0.0, snooker_prob=0.0, gamma_jump_prob=0.0,
    block_sizes=(7, 14),
):
    """bg_MH+DR config with optional cp + sp + gj."""
    ens_kwargs = {
        "block_sizes": list(block_sizes),
        "use_mh": True,
        "delayed_rejection": True,
    }
    if complementary_prob > 0.0:
        ens_kwargs["complementary_prob"] = complementary_prob
    if snooker_prob > 0.0:
        ens_kwargs["snooker_prob"] = snooker_prob
    if gamma_jump_prob > 0.0:
        ens_kwargs["gamma_jump_prob"] = gamma_jump_prob
    return VariantConfig(
        name=f"bg_mh_dr_cp{complementary_prob}_sp{snooker_prob}_gj{gamma_jump_prob}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens_kwargs,
    )


def test_snooker_zero_matches_baseline_bg_mh_dr():
    """snooker_prob=0.0 + cp=0.0 must produce byte-identical samples to current bg_MH+DR."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(42), (32, 21)) * 0.1

    res_explicit = run_variant(
        log_prob, init, n_steps=300,
        config=_make_bg_mh_dr_snooker_config(snooker_prob=0.0, complementary_prob=0.0),
        n_warmup=100, key=jax.random.PRNGKey(0), verbose=False,
    )
    res_default = run_variant(
        log_prob, init, n_steps=300,
        config=_make_bg_mh_dr_config(complementary_prob=0.0),
        n_warmup=100, key=jax.random.PRNGKey(0), verbose=False,
    )
    np.testing.assert_array_equal(
        np.array(res_explicit["samples"]),
        np.array(res_default["samples"]),
        err_msg="snooker_prob=0.0 + cp=0.0 must be byte-identical to baseline bg_MH+DR",
    )


def test_snooker_invalid_prob_raises():
    """cp + sp > 1.0 must raise ValueError; sp out of [0,1] must raise."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(0), (32, 21)) * 0.5

    cfg = _make_bg_mh_dr_snooker_config(complementary_prob=0.7, snooker_prob=0.5)
    with pytest.raises(ValueError, match="must be ≤ 1.0"):
        run_variant(log_prob, init, n_steps=10, config=cfg, n_warmup=5,
                    key=jax.random.PRNGKey(0), verbose=False)

    cfg = _make_bg_mh_dr_snooker_config(snooker_prob=1.5)
    with pytest.raises(ValueError, match="snooker_prob must be"):
        run_variant(log_prob, init, n_steps=10, config=cfg, n_warmup=5,
                    key=jax.random.PRNGKey(0), verbose=False)


@pytest.mark.parametrize("sp", [0.5, 1.0])
def test_snooker_recovers_unit_gaussian_variance(sp):
    """Pure or heavy snooker must recover unit Gaussian per-dim variance.

    This is the LOAD-BEARING test for the (bsize-1) Jacobian. A wrong sign,
    wrong dimension, or omitted Jacobian produces silently biased variance.
    Run on Hyak GPU for tight error bounds.
    """
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(0), (64, 21)) * 0.5

    result = run_variant(
        log_prob, init, n_steps=4000,
        config=_make_bg_mh_dr_snooker_config(snooker_prob=sp),
        n_warmup=1000, key=jax.random.PRNGKey(0), verbose=False,
    )
    samples = np.array(result["samples"]).reshape(-1, 21)
    emp_var = samples.var(axis=0)
    rel_err = np.abs(emp_var - 1.0)
    assert rel_err.max() < 0.20, (
        f"sp={sp}: worst per-dim variance error {rel_err.max():.2%} > 20%; "
        f"emp_var={emp_var}"
    )


def test_snooker_recovers_anisotropic_gaussian_covariance():
    """Pure snooker on an anisotropic Gaussian must recover the OFF-DIAGONAL covariance.

    A subtle Jacobian error (wrong sign, wrong dim) might pass the diagonal
    variance test but corrupt off-diagonal correlations. This catches that.
    """
    ndim = 21
    rng = np.random.default_rng(42)
    A = rng.standard_normal((ndim, ndim))
    Q, _ = np.linalg.qr(A)
    evals = np.linspace(1.0, 10.0, ndim)   # cond=10, mild anisotropy
    cov = Q @ np.diag(evals) @ Q.T
    cov = (cov + cov.T) / 2
    prec = jnp.array(np.linalg.inv(cov), dtype=jnp.float64)
    true_cov = np.array(cov)

    @jax.jit
    def log_prob(x):
        return -0.5 * x @ prec @ x

    init = jax.random.normal(jax.random.PRNGKey(0), (64, ndim)) * 0.5
    result = run_variant(
        log_prob, init, n_steps=4000,
        config=_make_bg_mh_dr_snooker_config(snooker_prob=1.0),
        n_warmup=1000, key=jax.random.PRNGKey(0), verbose=False,
    )
    samples = np.array(result["samples"]).reshape(-1, ndim)
    emp_cov = np.cov(samples, rowvar=False)
    # Compare full covariance — use Frobenius-norm relative error
    diff = emp_cov - true_cov
    rel_err = np.linalg.norm(diff) / np.linalg.norm(true_cov)
    assert rel_err < 0.20, (
        f"pure snooker covariance recovery failed: ‖emp-true‖/‖true‖ = "
        f"{rel_err:.2%} > 20% on cond=10 anisotropic 21D Gaussian"
    )


def test_snooker_mixed_with_complementary_and_gamma_jump():
    """All three knobs together (cp + sp + gj) compose without crashing,
    and the chain still recovers Gaussian moments."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(0), (64, 21)) * 0.5

    # cp + sp = 0.8 ≤ 1.0; gj is independent
    result = run_variant(
        log_prob, init, n_steps=2000,
        config=_make_bg_mh_dr_snooker_config(
            complementary_prob=0.5, snooker_prob=0.3, gamma_jump_prob=0.05,
        ),
        n_warmup=500, key=jax.random.PRNGKey(0), verbose=False,
    )
    samples = np.array(result["samples"]).reshape(-1, 21)
    assert np.isfinite(samples).all()
    emp_var = samples.var(axis=0)
    rel_err = np.abs(emp_var - 1.0)
    assert rel_err.max() < 0.30, (
        f"mixed cp=0.5 + sp=0.3 + gj=0.05: worst variance err {rel_err.max():.2%}"
    )
