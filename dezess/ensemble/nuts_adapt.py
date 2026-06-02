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

    samples_2d: (n, d). Shrinkage toward unit-variance (Stan-style). Robust
    to few/near-singular samples: dense falls back to diagonal when there
    aren't enough samples for a reliable covariance, and the Cholesky retries
    with growing jitter before falling back to the diagonal.
    """
    n = samples_2d.shape[0]
    shrink = n / (n + 5.0)
    if mass_type == "identity":
        return np.eye(d)

    var = samples_2d.var(axis=0)
    var = shrink * var + (1 - shrink) * 1.0          # shrink toward unit variance
    var = np.maximum(var, 1e-8)
    diag_L = np.diag(np.sqrt(var))

    if mass_type == "diag":
        return diag_L
    if mass_type == "dense":
        # Need clearly more samples than dimensions for a usable dense cov.
        if n < 2 * d:
            return diag_L
        cov = np.cov(samples_2d, rowvar=False)
        cov = shrink * cov + (1 - shrink) * np.eye(d)   # shrink toward I
        cov = 0.5 * (cov + cov.T)
        mean_diag = max(np.mean(np.diag(cov)), 1e-8)
        for jit in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
            try:
                return np.linalg.cholesky(cov + jit * mean_diag * np.eye(d))
            except np.linalg.LinAlgError:
                continue
        return diag_L     # last resort: diagonal whitening
    raise ValueError(f"unknown mass_type {mass_type!r}")


def run_nuts(log_prob, init, n_warmup, n_prod, mass_type="diag",
             max_tree_depth=10, target_accept=0.8, step_size0=0.5, seed=0,
             verbose=False, L_init=None, step_size_init=None, skip_warmup=False,
             stream_path=None, config_name="nuts", save_every=50):
    """Ensemble NUTS with warmup adaptation. Each walker is an independent chain.

    The leapfrog/NUTS step is JIT-compiled ONCE with the whitening matrix L
    threaded as a dynamic argument, so changing L across mass-adaptation
    windows (or on resume) does NOT trigger recompilation.

    Resume: pass `skip_warmup=True` with `L_init` and `step_size_init` to skip
    adaptation and run production directly with a frozen geometry.

    Streaming: pass `stream_path` to write production chunks + NUTS state to
    disk (live-readable via dezess.read_streaming, resumable via
    dezess.resume_streaming).

    Returns dict: samples (n_prod, n_walkers, d) in ORIGINAL space, log_prob,
    mean_depth (doublings), div_rate, step_size, L, mass_type.
    """
    init = jnp.asarray(init, dtype=jnp.float64)
    n_walkers, d = init.shape
    D = int(max_tree_depth)
    grad_orig = jax.grad(log_prob)
    inv_mass = jnp.ones(d, dtype=jnp.float64)   # identity mass in whitened frame

    # --- ONE jit, L threaded as a dynamic arg (no re-jit when L changes) ---
    def _whitened_step(xp, lp, g, k, ss, L):
        def lp_white(z):
            return log_prob(L @ z)
        def grad_white(z):
            return L.T @ grad_orig(L @ z)
        return nuts.nuts_step(xp, lp, g, k, ss, inv_mass, lp_white, grad_white, D)

    step = jax.jit(jax.vmap(_whitened_step, in_axes=(0, 0, 0, 0, None, None)))
    _lp_white = jax.jit(jax.vmap(lambda xp, L: log_prob(L @ xp), in_axes=(0, None)))
    _grad_white = jax.jit(jax.vmap(lambda xp, L: L.T @ grad_orig(L @ xp), in_axes=(0, None)))
    to_white = jax.jit(jax.vmap(
        lambda xi, L: jax.scipy.linalg.solve_triangular(L, xi, lower=True),
        in_axes=(0, None)))
    to_orig = jax.jit(jax.vmap(lambda xpi, L: L @ xpi, in_axes=(0, None)))

    keys = jax.random.split(jax.random.PRNGKey(seed + 1), n_walkers)

    if skip_warmup:
        # Resume / frozen-geometry mode.
        L = np.eye(d) if L_init is None else np.asarray(L_init, dtype=np.float64)
        step_size = float(step_size_init if step_size_init is not None else step_size0)
        L_jax = jnp.asarray(L, dtype=jnp.float64)
        xp = to_white(init, L_jax)
        if verbose:
            print(f"  [nuts] resume/frozen: step_size={step_size:.4f}, "
                  f"mass dims={L.shape}", flush=True)
    else:
        # --- warmup: windowed mass + dual-averaging ---
        L = np.eye(d)
        L_jax = jnp.asarray(L, dtype=jnp.float64)
        xp = to_white(init, L_jax)
        da = _DualAvg(step_size0, target=target_accept)
        step_size = step_size0

        init_buf = max(int(0.15 * n_warmup), 25)
        term_buf = max(int(0.10 * n_warmup), 25)
        mass_steps = n_warmup - init_buf - term_buf
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
            nonlocal xp, keys, step_size
            buf = []
            for _ in range(n_steps):
                xp, lp_, g_, keys, acc, depth, div, nleaf = step(
                    xp, _lp_white(xp, L_jax), _grad_white(xp, L_jax), keys,
                    step_size, L_jax)
                step_size = da.update(float(jnp.mean(acc)))
                if collect:
                    buf.append(np.array(to_orig(xp, L_jax)))
            return buf

        warmup_phase(init_buf, collect=False)
        for wi, wlen in enumerate(windows):
            buf = warmup_phase(wlen, collect=True)
            samples_win = np.concatenate(buf, axis=0).reshape(-1, d)
            x_cur = np.array(to_orig(xp, L_jax))
            L = _estimate_L(samples_win, mass_type, d)
            L_jax = jnp.asarray(L, dtype=jnp.float64)   # value change only -> no re-jit
            xp = to_white(jnp.asarray(x_cur), L_jax)
            da = _DualAvg(step_size, target=target_accept)
            if verbose:
                print(f"  [nuts warmup] window {wi+1}/{len(windows)} "
                      f"({wlen} steps) -> step_size={step_size:.4f}", flush=True)
        warmup_phase(term_buf, collect=False)
        step_size = da.finalize()
        if verbose:
            print(f"  [nuts warmup] done. final step_size={step_size:.4f}, "
                  f"mass_type={mass_type}", flush=True)

    # --- production (frozen L, frozen step size) ---
    streamer = None
    if stream_path is not None:
        from dezess.streaming import Streamer
        streamer = Streamer(stream_path, n_walkers, d, n_prod, config_name,
                            z_capacity=0)
        streamer.open_chunk()

    samples = np.empty((n_prod, n_walkers, d))
    lps = np.empty((n_prod, n_walkers))
    depths = np.empty((n_prod, n_walkers))
    divs = np.empty((n_prod, n_walkers))
    total_grad_evals = 0   # total leapfrog steps = gradient evaluations
    batch_s, batch_lp = [], []
    try:
        for t in range(n_prod):
            xp, lp_, g_, keys, acc, depth, div, nleaf = step(
                xp, _lp_white(xp, L_jax), _grad_white(xp, L_jax), keys,
                step_size, L_jax)
            x_orig = np.array(to_orig(xp, L_jax))
            lp_np = np.array(lp_)
            samples[t] = x_orig
            lps[t] = lp_np
            depths[t] = np.array(depth)
            divs[t] = np.array(div)
            total_grad_evals += int(np.sum(np.array(nleaf)))
            if streamer is not None:
                batch_s.append(x_orig)
                batch_lp.append(lp_np)
                if len(batch_s) >= save_every or t == n_prod - 1:
                    streamer.append_batch(np.stack(batch_s), np.stack(batch_lp))
                    streamer.save_nuts_state(
                        last_positions=x_orig, last_lps=lp_np,
                        step_size=step_size, L=np.asarray(L),
                        max_tree_depth=D, n_steps_done_in_chunk=t + 1)
                    batch_s, batch_lp = [], []
    finally:
        if streamer is not None:
            streamer.close()

    return {
        "samples": samples,
        "log_prob": lps,
        "mean_depth": float(depths.mean()),
        "div_rate": float(divs.mean()),
        "step_size": float(step_size),
        "L": np.array(L),
        "mass_type": mass_type,
        "grad_evals": int(total_grad_evals),   # total leapfrog steps in production
    }
