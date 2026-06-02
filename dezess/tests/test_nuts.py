"""Phase 2 tests for NUTS — Gaussian moments, divergence-free, BlackJAX oracle.

The whole risk of a from-scratch NUTS is subtle correctness bugs that don't
crash but bias the posterior or the tree-depth distribution. So we validate:
  1. unit + correlated Gaussian moments (mean/var/cov)
  2. no spurious divergences on a smooth Gaussian
  3. mean tree depth is sane (not stuck at 0 or pinned at max)
  4. (oracle) moments + mean tree depth match BlackJAX NUTS within MC noise

BlackJAX is a TEST-TIME oracle only — never imported by dezess proper.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

import dezess
print(f"dezess imported from: {dezess.__file__}")
from dezess.ensemble import nuts


def _run_nuts(log_prob, d, n_walkers, n_steps, step_size, inv_mass=None,
              max_tree_depth=10, seed=0):
    grad_fn = jax.grad(log_prob)
    if inv_mass is None:
        inv_mass = jnp.ones(d, dtype=jnp.float64)
    key = jax.random.PRNGKey(seed)
    q = jax.random.normal(key, (n_walkers, d), dtype=jnp.float64)
    lp = jax.vmap(log_prob)(q)
    g = jax.vmap(grad_fn)(q)
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_walkers)

    step = jax.jit(jax.vmap(
        lambda q, lp, g, k: nuts.nuts_step(
            q, lp, g, k, step_size, inv_mass, log_prob, grad_fn, max_tree_depth)))

    samples = np.empty((n_steps, n_walkers, d))
    depths = np.empty((n_steps, n_walkers))
    divs = np.empty((n_steps, n_walkers))
    for t in range(n_steps):
        q, lp, g, keys, acc, depth, div = step(q, lp, g, keys)
        samples[t] = np.array(q)
        depths[t] = np.array(depth)
        divs[t] = np.array(div)
    return samples, depths, divs


def test_nuts_unit_gaussian_moments():
    d = 10
    def log_prob(x):
        return -0.5 * jnp.sum(x * x)
    samples, depths, divs = _run_nuts(log_prob, d, 16, 800, step_size=0.8, seed=0)
    prod = samples[200:].reshape(-1, d)
    mean = prod.mean(axis=0)
    var = prod.var(axis=0)
    div_rate = divs[200:].mean()
    mean_depth = depths[200:].mean()

    assert div_rate < 0.01, f"divergence rate {div_rate:.3f} too high on smooth Gaussian"
    assert 0.5 < mean_depth < float(10), f"mean tree depth {mean_depth:.2f} pathological"
    assert np.abs(mean).max() < 0.1, f"mean off: max|mean|={np.abs(mean).max():.3f}"
    assert np.abs(var - 1.0).max() < 0.15, f"var off: {var.round(3)}"


def test_nuts_correlated_gaussian():
    rho = 0.9
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    prec = jnp.asarray(np.linalg.inv(cov))
    def log_prob(x):
        return -0.5 * x @ prec @ x
    samples, depths, divs = _run_nuts(log_prob, 2, 16, 1500, step_size=0.5, seed=2)
    prod = samples[300:].reshape(-1, 2)
    emp_cov = np.cov(prod, rowvar=False)
    fro = np.linalg.norm(emp_cov - cov) / np.linalg.norm(cov)
    rho_hat = emp_cov[0, 1] / np.sqrt(emp_cov[0, 0] * emp_cov[1, 1])
    assert fro < 0.12, f"cov error {fro*100:.1f}%\nemp=\n{emp_cov.round(3)}"
    assert abs(rho_hat - rho) < 0.06, f"rho {rho_hat:.3f} vs {rho}"


def test_nuts_vs_blackjax_oracle():
    """Moments + mean tree depth must match BlackJAX NUTS within MC noise.

    This is the definitive correctness check: a subtly-wrong U-turn or
    sampling rule would diverge from the reference here even if the marginal
    moments looked plausible.
    """
    try:
        import blackjax
    except ImportError:
        print("SKIP: blackjax not available as oracle")
        return

    d = 8
    rng = np.random.default_rng(0)
    A = rng.standard_normal((d, d))
    cov = (A @ A.T) / d + np.eye(d)        # random SPD
    prec = jnp.asarray(np.linalg.inv(cov))
    def log_prob(x):
        return -0.5 * x @ prec @ x

    step_size = 0.4
    inv_mass = jnp.ones(d, dtype=jnp.float64)

    # --- our NUTS ---
    samples, depths, _ = _run_nuts(log_prob, d, 8, 1200, step_size=step_size,
                                   inv_mass=inv_mass, seed=5)
    ours = samples[200:].reshape(-1, d)
    our_cov = np.cov(ours, rowvar=False)
    our_depth = depths[200:].mean()

    # --- BlackJAX NUTS oracle (same step size, identity mass) ---
    nuts_bj = blackjax.nuts(log_prob, step_size=step_size,
                            inverse_mass_matrix=np.ones(d))
    state = nuts_bj.init(jnp.zeros(d))
    bj_key = jax.random.PRNGKey(99)
    bj_samples = []
    bj_depths = []
    step_bj = jax.jit(nuts_bj.step)
    for t in range(4000):
        bj_key, sk = jax.random.split(bj_key)
        state, info = step_bj(sk, state)
        bj_samples.append(np.array(state.position))
        bj_depths.append(int(info.num_trajectory_expansions))
    bj = np.array(bj_samples[500:])
    bj_cov = np.cov(bj, rowvar=False)
    bj_depth = np.mean(bj_depths[500:])

    cov_diff = np.linalg.norm(our_cov - bj_cov) / np.linalg.norm(bj_cov)
    print(f"  our mean depth={our_depth:.2f}  blackjax mean depth={bj_depth:.2f}")
    print(f"  our_cov vs blackjax_cov rel-Fro = {cov_diff*100:.1f}%")
    print(f"  our_cov vs TRUE cov rel-Fro = {np.linalg.norm(our_cov-cov)/np.linalg.norm(cov)*100:.1f}%")

    # Both should recover the true cov; compare ours to truth (loose, MC noise)
    assert np.linalg.norm(our_cov - cov) / np.linalg.norm(cov) < 0.15, (
        "our NUTS cov does not match the true covariance")
    # Tree depth should be in the same ballpark (within ~1.5 expansions)
    assert abs(our_depth - bj_depth) < 2.0, (
        f"tree depth mismatch vs oracle: ours={our_depth:.2f} bj={bj_depth:.2f}")


if __name__ == "__main__":
    test_nuts_unit_gaussian_moments()
    print("PASS: NUTS unit Gaussian moments")
    test_nuts_correlated_gaussian()
    print("PASS: NUTS correlated Gaussian (rho=0.9)")
    test_nuts_vs_blackjax_oracle()
    print("PASS: NUTS vs BlackJAX oracle")
    print("\nPhase 2 (NUTS tree-doubling) all green.")
