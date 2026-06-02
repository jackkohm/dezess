"""Hamiltonian Monte Carlo + NUTS ensemble for dezess.

Each walker is an INDEPENDENT HMC/NUTS chain (vmap over walkers). Unlike
DE-MCz / block-Gibbs, walkers do not share a Z-matrix — the ensemble exists
only to provide multiple chains for diagnostics (R-hat, ESS) and pooled
warmup adaptation. Gradient-based: requires a differentiable log_prob.

Built from scratch (no BlackJAX runtime dependency).

Conventions / physics
---------------------
Hamiltonian  H(q, p) = U(q) + K(p),  with potential U = -log_prob(q) and
kinetic K(p) = 0.5 pᵀ M⁻¹ p. Hamilton's equations give
  dq/dt = M⁻¹ p
  dp/dt = -∂U/∂q = +∂log_prob/∂q = grad_logprob(q)
so the leapfrog momentum half-kick uses +grad_logprob.

Mass matrix: this module's kernels take the *diagonal* inverse mass
`inv_mass_diag` (= 1/m per dim). Dense mass support is added in a later
phase. Momentum is drawn p ~ N(0, M), i.e. p = sqrt(m) * z = z / sqrt(inv_mass).

Phase 1: leapfrog integrator + static HMC kernel (diagonal mass).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


def leapfrog(q, p, grad_q, step_size, inv_mass_diag, grad_fn, n_steps):
    """`n_steps` leapfrog steps with diagonal mass.

    Parameters
    ----------
    q : (d,) position
    p : (d,) momentum
    grad_q : (d,) grad_logprob at q (passed in to avoid recomputing)
    step_size : scalar leapfrog step size ε
    inv_mass_diag : (d,) diagonal of M⁻¹
    grad_fn : q -> grad_logprob(q)
    n_steps : int, number of leapfrog steps L

    Returns (q_new, p_new, grad_new).
    """
    def one_step(carry, _):
        q, p, g = carry
        p = p + 0.5 * step_size * g                  # half momentum kick
        q = q + step_size * (inv_mass_diag * p)      # full position drift
        g = grad_fn(q)
        p = p + 0.5 * step_size * g                  # half momentum kick
        return (q, p, g), None

    (q, p, g), _ = lax.scan(one_step, (q, p, grad_q), None, length=n_steps)
    return q, p, g


def kinetic_energy(p, inv_mass_diag):
    """K(p) = 0.5 pᵀ M⁻¹ p for diagonal M."""
    return 0.5 * jnp.sum(inv_mass_diag * p * p)


def sample_momentum(key, inv_mass_diag):
    """Draw p ~ N(0, M) for diagonal M (M = 1/inv_mass_diag)."""
    z = jax.random.normal(key, inv_mass_diag.shape, dtype=jnp.float64)
    # p = sqrt(m) z = z / sqrt(inv_mass)
    return z / jnp.sqrt(inv_mass_diag)


def hmc_step(q, lp, grad_q, key, step_size, inv_mass_diag,
             log_prob_fn, grad_fn, n_leapfrog):
    """One static-HMC transition for a single walker.

    Returns (q_out, lp_out, grad_out, key, accepted, accept_prob).
    `accepted` is a bool; `accept_prob` = min(1, exp(H0 - H1)) (for adaptation).
    """
    key, k_mom, k_acc = jax.random.split(key, 3)
    p0 = sample_momentum(k_mom, inv_mass_diag)
    H0 = -lp + kinetic_energy(p0, inv_mass_diag)

    q_new, p_new, grad_new = leapfrog(
        q, p0, grad_q, step_size, inv_mass_diag, grad_fn, n_leapfrog)
    lp_new = log_prob_fn(q_new)
    H1 = -lp_new + kinetic_energy(p_new, inv_mass_diag)

    log_accept = H0 - H1
    # Guard non-finite (e.g. divergence → inf energy) → reject.
    log_accept = jnp.where(jnp.isfinite(log_accept), log_accept, -jnp.inf)
    accept_prob = jnp.clip(jnp.exp(jnp.minimum(log_accept, 0.0)), 0.0, 1.0)

    u = jax.random.uniform(k_acc, dtype=jnp.float64)
    accept = jnp.log(u + 1e-30) < log_accept

    q_out = jnp.where(accept, q_new, q)
    lp_out = jnp.where(accept, lp_new, lp)
    grad_out = jnp.where(accept, grad_new, grad_q)
    return q_out, lp_out, grad_out, key, accept, accept_prob
