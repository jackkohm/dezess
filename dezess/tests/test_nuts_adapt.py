"""Phase 3 tests — NUTS mass-matrix adaptation (identity / diag / dense).

On an ill-conditioned, rotated Gaussian (so diagonal preconditioning can't
fully help but dense can), validate:
  1. dense adaptation recovers the true covariance
  2. dense gives LOWER mean tree depth than identity (preconditioning works:
     fewer leapfrog steps to traverse the whitened isotropic target)
  3. all three mass types are divergence-free and unbiased in the mean
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

import dezess
print(f"dezess imported from: {dezess.__file__}")
from dezess.ensemble import nuts_adapt


def _ill_conditioned_target(d=10, cond=100.0, seed=0):
    """Rotated Gaussian with condition number `cond` (not axis-aligned)."""
    rng = np.random.default_rng(seed)
    # eigenvalues log-spaced from 1 to cond
    eig = np.exp(np.linspace(0.0, np.log(cond), d))
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))   # random rotation
    cov = Q @ np.diag(eig) @ Q.T
    cov = 0.5 * (cov + cov.T)
    prec = jnp.asarray(np.linalg.inv(cov))

    def log_prob(x):
        return -0.5 * x @ prec @ x

    return log_prob, cov


def test_dense_recovers_covariance():
    d = 8
    log_prob, cov = _ill_conditioned_target(d=d, cond=80.0, seed=1)
    init = jax.random.normal(jax.random.PRNGKey(0), (16, d), dtype=jnp.float64)

    res = nuts_adapt.run_nuts(log_prob, init, n_warmup=600, n_prod=800,
                              mass_type="dense", max_tree_depth=10,
                              step_size0=0.3, seed=3, verbose=True)
    flat = res["samples"].reshape(-1, d)
    emp_cov = np.cov(flat, rowvar=False)
    fro = np.linalg.norm(emp_cov - cov) / np.linalg.norm(cov)
    print(f"  dense: cov Fro err={fro*100:.1f}%  mean_depth={res['mean_depth']:.2f}  "
          f"div_rate={res['div_rate']:.3f}")
    assert res["div_rate"] < 0.02, f"too many divergences: {res['div_rate']:.3f}"
    assert fro < 0.18, f"dense NUTS cov error {fro*100:.1f}% too high"
    assert np.abs(flat.mean(axis=0)).max() < 0.5, "mean biased"


def test_dense_beats_identity_depth():
    """On an ill-conditioned target, dense preconditioning should need
    shallower trees than identity mass (the whole point of adaptation)."""
    d = 8
    log_prob, cov = _ill_conditioned_target(d=d, cond=100.0, seed=2)
    init = jax.random.normal(jax.random.PRNGKey(1), (16, d), dtype=jnp.float64)

    res_id = nuts_adapt.run_nuts(log_prob, init, n_warmup=600, n_prod=500,
                                 mass_type="identity", max_tree_depth=10,
                                 step_size0=0.1, seed=4)
    res_de = nuts_adapt.run_nuts(log_prob, init, n_warmup=600, n_prod=500,
                                 mass_type="dense", max_tree_depth=10,
                                 step_size0=0.3, seed=4)
    print(f"  identity mean_depth={res_id['mean_depth']:.2f}  "
          f"dense mean_depth={res_de['mean_depth']:.2f}")
    assert res_de["mean_depth"] < res_id["mean_depth"], (
        f"dense ({res_de['mean_depth']:.2f}) should need shallower trees than "
        f"identity ({res_id['mean_depth']:.2f}) on an ill-conditioned target")


def test_diag_runs_and_unbiased():
    d = 6
    log_prob, cov = _ill_conditioned_target(d=d, cond=30.0, seed=5)
    init = jax.random.normal(jax.random.PRNGKey(2), (16, d), dtype=jnp.float64)
    res = nuts_adapt.run_nuts(log_prob, init, n_warmup=500, n_prod=600,
                              mass_type="diag", max_tree_depth=10,
                              step_size0=0.2, seed=6)
    flat = res["samples"].reshape(-1, d)
    assert res["div_rate"] < 0.02, f"diag divergences {res['div_rate']:.3f}"
    assert np.abs(flat.mean(axis=0)).max() < 0.6, "diag mean biased"


if __name__ == "__main__":
    test_dense_recovers_covariance()
    print("PASS: dense recovers covariance")
    test_dense_beats_identity_depth()
    print("PASS: dense beats identity on tree depth")
    test_diag_runs_and_unbiased()
    print("PASS: diag runs unbiased")
    print("\nPhase 3 (mass-matrix adaptation) all green.")
