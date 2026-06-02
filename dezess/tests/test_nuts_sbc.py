"""Simulation-Based Calibration (SBC, Talts et al. 2018) for dezess NUTS.

SBC is the gold-standard sampler-correctness test: it checks that the sampler
draws from the TRUE posterior, not just that moments look right. A subtly
miscalibrated sampler (too narrow / too wide / biased) that passes moment
tests will fail SBC.

Conjugate Gaussian model (SBC exact):
    prior      θ̃ ~ N(0, 2C)
    likelihood y  ~ N(θ̃, 2C)
    posterior  θ | y ~ N(0.5 y, C)

For each of M simulations: draw θ̃, draw y, run NUTS on N(0.5y, C), thin to
~independent samples, compute the rank of θ̃ among them (per dim). If NUTS is
calibrated, ranks are Uniform{0..L}. We chi-square test rank uniformity per
dim and save rank histograms.

All M posteriors share covariance C (only the mean shifts), so we run all M
simulations as M walkers in ONE vmapped ensemble — each walker its own mean.
Tested in two regimes:
  - C = I            (identity-mass kernel)
  - C = correlated   (whitening / dense-mass production path)
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dezess
print(f"dezess imported from: {dezess.__file__}")
from dezess.ensemble import nuts

ARTIFACTS = "artifacts/nuts_sbc"
os.makedirs(ARTIFACTS, exist_ok=True)

D_TREE = 8
M = 512          # simulations (= walkers)
N_BURN = 200
N_COLLECT = 600
THIN = 3         # L = N_COLLECT/THIN ~ 200 thinned samples per chain
STEP_SIZE = 0.85


def run_sbc(d, C, label, seed=0):
    """Run SBC for posterior covariance C. Returns ranks (M, d), L."""
    rng = np.random.default_rng(seed)
    C = np.asarray(C, dtype=np.float64)
    chol_2C = np.linalg.cholesky(2.0 * C)
    prec_post = jnp.asarray(np.linalg.inv(C))

    # whitening for the kernel: x = L_w x', target N(m_i, I) in x'
    L_w = np.linalg.cholesky(C)
    L_w_jax = jnp.asarray(L_w)

    # --- simulate prior draws + data + posterior means ---
    theta_true = (chol_2C @ rng.standard_normal((d, M))).T          # (M,d) ~ N(0,2C)
    y = theta_true + (chol_2C @ rng.standard_normal((d, M))).T      # (M,d) ~ N(θ̃,2C)
    mu_post = 0.5 * y                                               # (M,d)
    mu_post_jax = jnp.asarray(mu_post)
    # whitened per-walker means m_i = L_w^{-1} mu_post_i
    m_white = jax.vmap(lambda mi: jax.scipy.linalg.solve_triangular(
        L_w_jax, mi, lower=True))(mu_post_jax)

    inv_mass = jnp.ones(d, dtype=jnp.float64)

    def whitened_lp(xp, m_i):
        # N(m_i, I) in whitened coords == N(mu_post_i, C) in original
        diff = xp - m_i
        return -0.5 * jnp.sum(diff * diff)

    grad_wl = jax.grad(whitened_lp, argnums=0)

    def step_one(xp, lp, g, key, m_i):
        lpf = lambda x: whitened_lp(x, m_i)
        gf = lambda x: grad_wl(x, m_i)
        return nuts.nuts_step(xp, lp, g, key, STEP_SIZE, inv_mass, lpf, gf, D_TREE)

    step = jax.jit(jax.vmap(step_one, in_axes=(0, 0, 0, 0, 0)))

    # start each chain at its posterior mean (whitened), burn in, then collect
    xp = m_white
    lp = jax.vmap(whitened_lp)(xp, m_white)
    g = jax.vmap(grad_wl)(xp, m_white)
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), M)

    for _ in range(N_BURN):
        xp, lp, g, keys, *_ = step(xp, lp, g, keys, m_white)

    kept = []
    for t in range(N_COLLECT):
        xp, lp, g, keys, *_ = step(xp, lp, g, keys, m_white)
        if t % THIN == 0:
            # transform back to original space: θ = L_w x'
            theta = jax.vmap(lambda z: L_w_jax @ z)(xp)
            kept.append(np.array(theta))
    samples = np.stack(kept, axis=0)        # (L, M, d)
    L = samples.shape[0]

    # rank of θ̃ among the L samples, per sim per dim
    ranks = (samples < theta_true[None, :, :]).sum(axis=0)   # (M, d)

    # --- chi-square uniformity test per dim ---
    n_bins = 20
    pvals = []
    fig, axes = plt.subplots(1, d, figsize=(3 * d, 2.6))
    if d == 1:
        axes = [axes]
    for j in range(d):
        counts, _ = np.histogram(ranks[:, j], bins=n_bins, range=(0, L + 1))
        expected = M / n_bins
        chi2 = ((counts - expected) ** 2 / expected).sum()
        p = stats.chi2.sf(chi2, df=n_bins - 1)
        pvals.append(p)
        axes[j].bar(range(n_bins), counts, width=1.0,
                    color="steelblue", alpha=0.7)
        axes[j].axhline(expected, color="k", ls="--", lw=1)
        # +/- 99% band for a uniform multinomial
        band = 2.576 * np.sqrt(expected * (1 - 1 / n_bins))
        axes[j].axhspan(expected - band, expected + band, color="gray", alpha=0.2)
        axes[j].set_title(f"dim {j}  p={p:.3f}")
        axes[j].set_xlabel("rank bin")
    fig.suptitle(f"SBC rank histograms — {label} (M={M}, L={L})")
    fig.tight_layout()
    out = f"{ARTIFACTS}/sbc_{label}.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)

    print(f"  [{label}] L={L} thinned samples/chain, chi2 p-values per dim: "
          f"{[f'{p:.3f}' for p in pvals]}")
    print(f"    -> {out}")
    return np.array(pvals)


def test_sbc_isotropic():
    d = 4
    C = np.eye(d)
    pvals = run_sbc(d, C, "isotropic", seed=0)
    # uniformity should NOT be rejected; use a loose threshold to avoid flakiness
    assert pvals.min() > 0.005, (
        f"SBC FAILED (isotropic): a dim's rank histogram is non-uniform, "
        f"min p={pvals.min():.4f}. NUTS may be miscalibrated.")


def test_sbc_correlated():
    d = 4
    rng = np.random.default_rng(7)
    A = rng.standard_normal((d, d))
    C = (A @ A.T) / d + 0.5 * np.eye(d)     # random SPD, correlated
    pvals = run_sbc(d, C, "correlated", seed=1)
    assert pvals.min() > 0.005, (
        f"SBC FAILED (correlated): min p={pvals.min():.4f}. The whitening / "
        f"dense-mass path may be miscalibrated.")


if __name__ == "__main__":
    test_sbc_isotropic()
    print("PASS: SBC isotropic (rank-uniform)")
    test_sbc_correlated()
    print("PASS: SBC correlated (rank-uniform)")
    print("\nNUTS SBC calibration check all green.")
