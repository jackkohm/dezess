"""NUTS warmup adaptation for dezess — dual-averaging step size + mass matrix.

Mass adaptation is implemented by WHITENING reparameterization, which reuses
the validated identity/diagonal NUTS kernel without modification:

  identity-mass NUTS on  lp'(x') = lp(L x')   ==   mass-matrix NUTS with M = Σ⁻¹
  where Σ = L Lᵀ.

So we estimate the posterior covariance Σ during warmup, set L = chol(Σ),
run plain identity-mass NUTS in the whitened frame x', and map samples back
with x = L x'. mass_type selects how Σ̂ is built:
  'identity' -> L = I            (no preconditioning)
  'diag'     -> L = diag(√var)   (per-dim scale)
  'dense'    -> L = chol(cov)    (full covariance)

Step size is adapted by Nesterov dual averaging (NUTS paper) targeting a
chosen mean Metropolis acceptance. Mass is estimated from windows of warmup
samples POOLED across walkers (the ensemble's one structural benefit here),
with Stan-style shrinkage toward a small diagonal.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dezess.ensemble import nuts

Array = jnp.ndarray


class _DualAvg:
    """Nesterov dual averaging for the leapfrog step size (NUTS paper)."""

    def __init__(self, eps0, target=0.8, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = np.log(10.0 * eps0)
        self.log_eps = np.log(eps0)
        self.log_eps_bar = 0.0
        self.H_bar = 0.0
        self.t = 0
        self.target = target
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa

    def update(self, accept_mean):
        self.t += 1
        eta = 1.0 / (self.t + self.t0)
        self.H_bar = (1 - eta) * self.H_bar + eta * (self.target - accept_mean)
        self.log_eps = self.mu - (np.sqrt(self.t) / self.gamma) * self.H_bar
        w = self.t ** (-self.kappa)
        self.log_eps_bar = w * self.log_eps + (1 - w) * self.log_eps_bar
        return float(np.exp(self.log_eps))

    def finalize(self):
        return float(np.exp(self.log_eps_bar))


def _estimate_L(samples_2d, mass_type, d):
    """Build the whitening matrix L (Σ = L Lᵀ) from pooled warmup samples.

    samples_2d: (n, d). Shrinkage toward a small diagonal (Stan-style).
    """
    n = samples_2d.shape[0]
    shrink = n / (n + 5.0)
    if mass_type == "identity":
        return np.eye(d)
    if mass_type == "diag":
        var = samples_2d.var(axis=0)
        var = shrink * var + (1 - shrink) * 1e-3
        var = np.maximum(var, 1e-8)
        return np.diag(np.sqrt(var))
    if mass_type == "dense":
        cov = np.cov(samples_2d, rowvar=False)
        cov = shrink * cov + (1 - shrink) * 1e-3 * np.eye(d)
        # symmetrize + jitter for a safe Cholesky
        cov = 0.5 * (cov + cov.T) + 1e-9 * np.eye(d)
        return np.linalg.cholesky(cov)
    raise ValueError(f"unknown mass_type {mass_type!r}")


def run_nuts(log_prob, init, n_warmup, n_prod, mass_type="diag",
             max_tree_depth=10, target_accept=0.8, step_size0=0.5, seed=0,
             verbose=False):
    """Ensemble NUTS with warmup adaptation. Each walker is an independent chain.

    Returns dict with samples (n_prod, n_walkers, d) in the ORIGINAL space,
    log_prob, mean tree depth, divergence rate, final step_size, and L.
    """
    init = jnp.asarray(init, dtype=jnp.float64)
    n_walkers, d = init.shape
    D = int(max_tree_depth)

    # --- whitened-space NUTS factory (re-created when L changes) ---
    def make_stepper(L):
        L_jax = jnp.asarray(L, dtype=jnp.float64)

        def lp_white(xp):
            return log_prob(L_jax @ xp)

        grad_white = jax.grad(lp_white)
        inv_mass = jnp.ones(d, dtype=jnp.float64)   # identity mass in whitened frame

        step = jax.jit(jax.vmap(
            lambda xp, lp, g, k, ss: nuts.nuts_step(
                xp, lp, g, k, ss, inv_mass, lp_white, grad_white, D),
            in_axes=(0, 0, 0, 0, None)))
        return step, lp_white, grad_white, L_jax

    def to_white(x, L_jax):
        # x' = L^{-1} x  (solve, per walker)
        return jax.vmap(lambda xi: jax.scipy.linalg.solve_triangular(
            L_jax, xi, lower=True))(x)

    def to_orig(xp, L_jax):
        return jax.vmap(lambda xpi: L_jax @ xpi)(xp)

    # --- warmup with windowed mass + dual-averaging ---
    L = np.eye(d)
    step, lp_white, grad_white, L_jax = make_stepper(L)
    xp = to_white(init, L_jax)
    lp = jax.vmap(lp_white)(xp)
    g = jax.vmap(grad_white)(xp)
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_walkers)

    da = _DualAvg(step_size0, target=target_accept)
    step_size = step_size0

    # schedule: init buffer (step-size only), growing mass windows, term buffer
    init_buf = max(int(0.15 * n_warmup), 25)
    term_buf = max(int(0.10 * n_warmup), 25)
    mass_steps = n_warmup - init_buf - term_buf
    # growing windows: 3 windows doubling in size
    windows = []
    if mass_steps > 0:
        w = max(mass_steps // 7, 10)
        used = 0
        while used < mass_steps:
            wlen = min(w, mass_steps - used)
            windows.append(wlen)
            used += wlen
            w *= 2

    def warmup_phase(n_steps, collect):
        nonlocal xp, lp, g, keys, step_size
        buf = []
        for _ in range(n_steps):
            xp, lp, g, keys, acc, depth, div = step(xp, lp, g, keys, step_size)
            step_size = da.update(float(jnp.mean(acc)))
            if collect:
                buf.append(np.array(to_orig(xp, L_jax)))
        return buf

    # init buffer
    warmup_phase(init_buf, collect=False)
    # mass windows
    for wi, wlen in enumerate(windows):
        buf = warmup_phase(wlen, collect=True)
        samples_win = np.concatenate(buf, axis=0).reshape(-1, d)  # pooled across walkers+steps
        L = _estimate_L(samples_win, mass_type, d)
        # rebuild stepper in the new whitened frame, re-express current walkers
        x_cur = np.array(to_orig(xp, L_jax))
        step, lp_white, grad_white, L_jax = make_stepper(L)
        xp = to_white(jnp.asarray(x_cur), L_jax)
        lp = jax.vmap(lp_white)(xp)
        g = jax.vmap(grad_white)(xp)
        # reset dual averaging around the new geometry
        da = _DualAvg(step_size, target=target_accept)
        if verbose:
            print(f"  [nuts warmup] window {wi+1}/{len(windows)} "
                  f"({wlen} steps) -> step_size={step_size:.4f}", flush=True)
    # term buffer (freeze mass, finalize step size)
    warmup_phase(term_buf, collect=False)
    step_size = da.finalize()
    if verbose:
        print(f"  [nuts warmup] done. final step_size={step_size:.4f}, "
              f"mass_type={mass_type}", flush=True)

    # --- production (frozen L, frozen step size) ---
    samples = np.empty((n_prod, n_walkers, d))
    lps = np.empty((n_prod, n_walkers))
    depths = np.empty((n_prod, n_walkers))
    divs = np.empty((n_prod, n_walkers))
    for t in range(n_prod):
        xp, lp, g, keys, acc, depth, div = step(xp, lp, g, keys, step_size)
        x_orig = to_orig(xp, L_jax)
        samples[t] = np.array(x_orig)
        lps[t] = np.array(lp)
        depths[t] = np.array(depth)
        divs[t] = np.array(div)

    return {
        "samples": samples,
        "log_prob": lps,
        "mean_depth": float(depths.mean()),
        "div_rate": float(divs.mean()),
        "step_size": float(step_size),
        "L": np.array(L),
        "mass_type": mass_type,
    }
