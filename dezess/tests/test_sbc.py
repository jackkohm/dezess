"""Layer 5: Simulation-Based Calibration (SBC).

Uses Bayesian linear regression (known posterior) as the generative
model. For each replication:
  1. Draw true parameters from the prior
  2. Generate synthetic data
  3. Run posterior inference with dezess
  4. Compute rank of true parameter in posterior samples

If the sampler is correct, ranks should be uniformly distributed.

Reference: Talts et al. (2018), "Validating Bayesian Inference Algorithms
with Simulation-Based Calibration", arXiv:1804.06788.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from dezess.sampler import run_demcz_slice

jax.config.update("jax_enable_x64", True)


# -- Bayesian linear regression with known posterior --
# y = X @ beta + eps, eps ~ N(0, sigma^2 I)
# Prior: beta ~ N(0, tau^2 I)
# Posterior: beta | y ~ N(mu_post, Sigma_post)
#   Sigma_post = (X'X / sigma^2 + I / tau^2)^{-1}
#   mu_post = Sigma_post @ X' y / sigma^2

N_OBS = 30       # observations
N_BETA = 5       # parameters
SIGMA = 1.0      # observation noise
TAU = 3.0        # prior std


def _generate_data(key):
    """Draw true beta from prior, generate y."""
    k1, k2, k3 = jax.random.split(key, 3)
    X = jax.random.normal(k1, (N_OBS, N_BETA), dtype=jnp.float64)
    beta_true = jax.random.normal(k2, (N_BETA,), dtype=jnp.float64) * TAU
    eps = jax.random.normal(k3, (N_OBS,), dtype=jnp.float64) * SIGMA
    y = X @ beta_true + eps
    return X, y, beta_true


def _make_log_prob(X, y):
    """Build log_prob for Bayesian linear regression."""
    XtX = X.T @ X
    Xty = X.T @ y
    prec_prior = jnp.eye(N_BETA) / TAU**2

    def log_prob(beta):
        resid = y - X @ beta
        ll = -0.5 * jnp.sum(resid**2) / SIGMA**2
        lp = -0.5 * beta @ prec_prior @ beta
        return ll + lp

    return log_prob


def _exact_posterior(X, y):
    """Compute exact posterior mean and covariance."""
    prec_post = X.T @ X / SIGMA**2 + jnp.eye(N_BETA) / TAU**2
    Sigma_post = jnp.linalg.inv(prec_post)
    mu_post = Sigma_post @ (X.T @ y / SIGMA**2)
    return mu_post, Sigma_post


def _one_sbc_replication(key):
    """Single SBC replication: generate data, run inference, compute ranks."""
    k_data, k_init, k_sampler = jax.random.split(key, 3)

    X, y, beta_true = _generate_data(k_data)
    log_prob = _make_log_prob(X, y)

    # Initialize near prior (not at truth — that would bias the test)
    init = jax.random.normal(k_init, (32, N_BETA), dtype=jnp.float64) * TAU

    result = run_demcz_slice(
        log_prob, init,
        n_steps=2000, n_warmup=500,
        key=k_sampler, mu=1.0, tune=True,
        verbose=False,
    )

    samples = np.asarray(result["samples"]).reshape(-1, N_BETA)

    # Rank of true parameter: number of posterior samples below truth
    beta_true_np = np.asarray(beta_true)
    ranks = np.array([
        (samples[:, d] < beta_true_np[d]).sum()
        for d in range(N_BETA)
    ])

    return ranks, len(samples)


def test_sbc(n_rep=80):
    """SBC: rank uniformity across many replications."""
    all_ranks = []
    n_samples = None

    for rep in range(n_rep):
        key = jax.random.PRNGKey(rep * 17 + 5)
        ranks, ns = _one_sbc_replication(key)
        all_ranks.append(ranks)
        if n_samples is None:
            n_samples = ns

    all_ranks = np.array(all_ranks)  # (n_rep, N_BETA)

    print(f"\n  SBC: {n_rep} replications, {n_samples} posterior samples each")
    print(f"  Parameters: {N_BETA}")

    # Per-parameter uniformity test
    p_values = []
    for d in range(N_BETA):
        # Normalize ranks to [0, 1]
        normalized = all_ranks[:, d] / n_samples
        stat, p = stats.kstest(normalized, 'uniform')
        p_values.append(p)
        print(f"    beta_{d}: rank_mean={normalized.mean():.3f} "
              f"rank_std={normalized.std():.3f} KS_p={p:.4f}")

    p_values = np.array(p_values)
    bonf_threshold = 0.01 / N_BETA
    n_reject = (p_values < bonf_threshold).sum()

    print(f"\n  Bonferroni threshold: {bonf_threshold:.4f}")
    print(f"  Rejections: {n_reject}/{N_BETA}")

    assert n_reject == 0, (
        f"SBC rejected {n_reject}/{N_BETA} parameters "
        f"(min_p={p_values.min():.6f}, threshold={bonf_threshold:.6f})"
    )


def test_sbc_posterior_accuracy():
    """Verify dezess posterior matches the known exact posterior."""
    key = jax.random.PRNGKey(42)
    X, y, beta_true = _generate_data(key)
    log_prob = _make_log_prob(X, y)
    mu_post, Sigma_post = _exact_posterior(X, y)

    init = jax.random.normal(jax.random.PRNGKey(1), (64, N_BETA), dtype=jnp.float64) * TAU

    result = run_demcz_slice(
        log_prob, init,
        n_steps=5000, n_warmup=1000,
        key=jax.random.PRNGKey(2), mu=1.0, tune=True,
        verbose=False,
    )

    samples = np.asarray(result["samples"]).reshape(-1, N_BETA)
    sample_mean = samples.mean(0)
    sample_cov = np.cov(samples.T)

    mu_post_np = np.asarray(mu_post)
    Sigma_post_np = np.asarray(Sigma_post)

    mean_err = np.max(np.abs(sample_mean - mu_post_np))
    # Compare diagonal of covariance
    std_err = np.max(np.abs(np.sqrt(np.diag(sample_cov)) - np.sqrt(np.diag(Sigma_post_np))))

    print(f"\n  Posterior accuracy (exact vs sampled):")
    print(f"    max |mean err|: {mean_err:.4f}")
    print(f"    max |std err|:  {std_err:.4f}")
    for d in range(N_BETA):
        print(f"    beta_{d}: exact={mu_post_np[d]:.3f} "
              f"sampled={sample_mean[d]:.3f} "
              f"exact_std={np.sqrt(Sigma_post_np[d,d]):.3f} "
              f"sampled_std={np.sqrt(sample_cov[d,d]):.3f}")

    assert mean_err < 0.15, f"Posterior mean error {mean_err:.4f} > 0.15"
    assert std_err < 0.1, f"Posterior std error {std_err:.4f} > 0.1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
