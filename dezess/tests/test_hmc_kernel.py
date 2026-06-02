"""Phase 1 tests for the HMC kernel — energy conservation + Gaussian moments.

Validates leapfrog integration, the Metropolis-on-Hamiltonian accept, and the
diagonal mass machinery IN ISOLATION (running the kernel directly, not through
run_variant). If these pass, the foundation is correct before NUTS tree logic
is added on top.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from dezess.ensemble import hmc


def test_leapfrog_energy_conservation():
    """With a small step size, leapfrog conserves the Hamiltonian to O(eps^2).

    Unit Gaussian in 5D, long trajectory, tiny step -> energy drift must be
    tiny. This isolates the integrator from the Metropolis layer.
    """
    d = 5
    prec = jnp.eye(d, dtype=jnp.float64)

    def log_prob(x):
        return -0.5 * x @ prec @ x

    grad_fn = jax.grad(log_prob)
    inv_mass = jnp.ones(d, dtype=jnp.float64)

    key = jax.random.PRNGKey(0)
    q0 = jax.random.normal(key, (d,), dtype=jnp.float64)
    p0 = hmc.sample_momentum(jax.random.PRNGKey(1), inv_mass)
    g0 = grad_fn(q0)

    H0 = -log_prob(q0) + hmc.kinetic_energy(p0, inv_mass)

    step_size = 0.01
    q1, p1, g1 = hmc.leapfrog(q0, p0, g0, step_size, inv_mass, grad_fn, 200)
    H1 = -log_prob(q1) + hmc.kinetic_energy(p1, inv_mass)

    rel_drift = abs(float(H1 - H0)) / abs(float(H0))
    assert rel_drift < 1e-3, (
        f"Energy drift {rel_drift:.2e} too large — leapfrog should conserve H "
        f"to O(eps^2)={step_size**2:.0e}.")


def test_leapfrog_reversibility():
    """Leapfrog is time-reversible: integrate forward then flip momentum and
    integrate again -> return to start."""
    d = 4
    def log_prob(x):
        return -0.5 * jnp.sum(x * x)
    grad_fn = jax.grad(log_prob)
    inv_mass = jnp.ones(d, dtype=jnp.float64)

    q0 = jax.random.normal(jax.random.PRNGKey(3), (d,), dtype=jnp.float64)
    p0 = hmc.sample_momentum(jax.random.PRNGKey(4), inv_mass)
    g0 = grad_fn(q0)

    q1, p1, g1 = hmc.leapfrog(q0, p0, g0, 0.1, inv_mass, grad_fn, 20)
    # Reverse: flip momentum, integrate same number of steps
    q2, p2, g2 = hmc.leapfrog(q1, -p1, g1, 0.1, inv_mass, grad_fn, 20)

    assert np.allclose(np.array(q2), np.array(q0), atol=1e-9), (
        f"Leapfrog not reversible: max |q2-q0| = {np.abs(np.array(q2)-np.array(q0)).max():.2e}")
    assert np.allclose(np.array(-p2), np.array(p0), atol=1e-9), (
        "Momentum not reversed correctly.")


def _run_hmc(log_prob, d, n_walkers, n_steps, step_size, n_leapfrog,
             inv_mass=None, seed=0):
    """Run vmapped static HMC for n_steps, return samples (n_steps, n_walkers, d)."""
    grad_fn = jax.grad(log_prob)
    if inv_mass is None:
        inv_mass = jnp.ones(d, dtype=jnp.float64)

    key = jax.random.PRNGKey(seed)
    q = jax.random.normal(key, (n_walkers, d), dtype=jnp.float64)
    lp = jax.vmap(log_prob)(q)
    g = jax.vmap(grad_fn)(q)
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_walkers)

    step = jax.jit(jax.vmap(
        lambda q, lp, g, k: hmc.hmc_step(
            q, lp, g, k, step_size, inv_mass, log_prob, grad_fn, n_leapfrog)))

    samples = np.empty((n_steps, n_walkers, d))
    accepts = np.empty((n_steps, n_walkers))
    for t in range(n_steps):
        q, lp, g, keys, acc, ap = step(q, lp, g, keys)
        samples[t] = np.array(q)
        accepts[t] = np.array(acc)
    return samples, accepts


def test_hmc_unit_gaussian_moments():
    """Static HMC on N(0, I_10) recovers mean 0 and unit variance."""
    d = 10
    def log_prob(x):
        return -0.5 * jnp.sum(x * x)

    samples, accepts = _run_hmc(log_prob, d, n_walkers=32, n_steps=1500,
                                step_size=0.5, n_leapfrog=8, seed=0)
    prod = samples[500:].reshape(-1, d)   # discard 500 burn-in
    acc_rate = accepts[500:].mean()

    mean = prod.mean(axis=0)
    var = prod.var(axis=0)

    assert acc_rate > 0.6, f"HMC accept rate {acc_rate:.2f} too low (expect ~0.8)"
    assert np.abs(mean).max() < 0.1, f"mean off: max|mean|={np.abs(mean).max():.3f}"
    assert np.abs(var - 1.0).max() < 0.15, f"var off: {var.round(3)}"


def test_hmc_correlated_gaussian_moments():
    """Static HMC recovers a correlated 2D covariance (rho=0.8)."""
    rho = 0.8
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
    prec = jnp.asarray(np.linalg.inv(cov))

    def log_prob(x):
        return -0.5 * x @ prec @ x

    samples, accepts = _run_hmc(log_prob, 2, n_walkers=32, n_steps=2000,
                                step_size=0.3, n_leapfrog=12, seed=1)
    prod = samples[500:].reshape(-1, 2)
    emp_cov = np.cov(prod, rowvar=False)

    fro = np.linalg.norm(emp_cov - cov) / np.linalg.norm(cov)
    assert fro < 0.1, (
        f"Correlated-Gaussian cov error {fro*100:.1f}% too high.\n"
        f"emp_cov=\n{emp_cov.round(3)}\ntrue=\n{cov}")
    rho_hat = emp_cov[0, 1] / np.sqrt(emp_cov[0, 0] * emp_cov[1, 1])
    assert abs(rho_hat - rho) < 0.05, f"rho recovered {rho_hat:.3f} vs {rho}"


if __name__ == "__main__":
    test_leapfrog_energy_conservation()
    print("PASS: leapfrog energy conservation")
    test_leapfrog_reversibility()
    print("PASS: leapfrog reversibility")
    test_hmc_unit_gaussian_moments()
    print("PASS: HMC unit Gaussian moments")
    test_hmc_correlated_gaussian_moments()
    print("PASS: HMC correlated Gaussian moments")
    print("\nPhase 1 (leapfrog + static HMC) all green.")
