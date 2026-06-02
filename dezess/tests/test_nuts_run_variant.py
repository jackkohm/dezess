"""Phase 4 — end-to-end NUTS through run_variant (ensemble='nuts').

Validates the integration: VariantConfig(ensemble='nuts') dispatches to the
NUTS path, returns the standard run_variant result dict, and recovers a
correlated Gaussian. Also checks mass_type passthrough via ensemble_kwargs.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

import dezess
print(f"dezess imported from: {dezess.__file__}")
from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig


def _cfg(mass_type, max_tree_depth=10):
    return VariantConfig(
        name=f"nuts_{mass_type}",
        direction="de_mcz", width="scale_aware",       # ignored by NUTS path
        slice_fn="fixed", zmatrix="circular",
        ensemble="nuts", check_nans=False,
        ensemble_kwargs={"mass_type": mass_type,
                         "max_tree_depth": max_tree_depth,
                         "target_accept": 0.8,
                         "step_size0": 0.3},
    )


def test_nuts_run_variant_correlated_gaussian():
    d = 6
    rng = np.random.default_rng(0)
    A = rng.standard_normal((d, d))
    cov = (A @ A.T) / d + 0.5 * np.eye(d)
    prec = jnp.asarray(np.linalg.inv(cov))

    def log_prob(x):
        return -0.5 * x @ prec @ x

    init = jax.random.normal(jax.random.PRNGKey(0), (16, d), dtype=jnp.float64)
    res = run_variant(log_prob, init, n_steps=1400, n_warmup=600,
                      config=_cfg("dense"), key=jax.random.PRNGKey(7),
                      verbose=False)

    # standard result-dict keys present
    for k in ["samples", "log_prob", "mu", "diagnostics"]:
        assert k in res, f"missing result key {k!r}"
    assert res["samples"].shape == (800, 16, d), f"bad shape {res['samples'].shape}"
    assert "mean_tree_depth" in res["diagnostics"]

    flat = np.array(res["samples"]).reshape(-1, d)
    emp_cov = np.cov(flat, rowvar=False)
    fro = np.linalg.norm(emp_cov - cov) / np.linalg.norm(cov)
    print(f"  run_variant nuts dense: cov Fro err={fro*100:.1f}%  "
          f"mean_depth={res['diagnostics']['mean_tree_depth']:.2f}  "
          f"div_rate={res['diagnostics']['divergence_rate']:.3f}  "
          f"step_size={res['mu']:.3f}")
    assert fro < 0.15, f"cov error {fro*100:.1f}% too high"
    assert res["diagnostics"]["divergence_rate"] < 0.02


def test_nuts_run_variant_mass_type_passthrough():
    """All three mass types run end-to-end through run_variant and are unbiased."""
    d = 4
    def log_prob(x):
        return -0.5 * jnp.sum(x * x)
    init = jax.random.normal(jax.random.PRNGKey(1), (16, d), dtype=jnp.float64)
    for mt in ["identity", "diag", "dense"]:
        res = run_variant(log_prob, init, n_steps=900, n_warmup=400,
                          config=_cfg(mt), key=jax.random.PRNGKey(11),
                          verbose=False)
        flat = np.array(res["samples"]).reshape(-1, d)
        mean = np.abs(flat.mean(axis=0)).max()
        var = flat.var(axis=0)
        print(f"  mass_type={mt:8s}: max|mean|={mean:.3f}  var={var.round(2)}  "
              f"depth={res['diagnostics']['mean_tree_depth']:.2f}")
        assert mean < 0.15, f"{mt}: mean biased ({mean:.3f})"
        assert np.abs(var - 1.0).max() < 0.2, f"{mt}: var off ({var.round(3)})"


if __name__ == "__main__":
    test_nuts_run_variant_correlated_gaussian()
    print("PASS: nuts via run_variant recovers correlated Gaussian")
    test_nuts_run_variant_mass_type_passthrough()
    print("PASS: all mass types pass through run_variant unbiased")
    print("\nPhase 4 (NUTS integrated into run_variant) all green.")
