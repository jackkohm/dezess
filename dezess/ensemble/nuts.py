"""No-U-Turn Sampler (NUTS) for dezess — from scratch, no BlackJAX runtime dep.

Multinomial NUTS (Betancourt 2017) with the generalized momentum-sum U-turn
criterion (Stan / NumPyro style), reimplemented as a single early-terminating
`lax.while_loop` over leaves so the trajectory length is genuinely adaptive.

Algorithm structure
-------------------
The trajectory is grown by repeated doubling. Doubling j contributes 2^j new
leaves in a randomly chosen direction. We flatten all doublings into ONE loop
over leaf index i = 0 .. (2^D - 1), recovering the doubling index j and the
within-subtree leaf number m = 1..2^j from i. Per leaf we:

  1. leapfrog ONE step in the current direction from the active edge
  2. progressive (biased) multinomial sampling of the proposal
  3. checkpoint WRITE/CHECK for the within-subtree U-turn test:
       - m odd  -> WRITE this leaf's momentum at levels k where (m-1) mod 2^k == 0
       - m even -> CHECK U-turn for levels k where (m-1) mod 2^k == 2^k - 1
     (only O(D) checkpoints are stored — exactly the pending left-endpoints)
  4. at the last leaf of a doubling: combine the subtree into the main tree
     (biased acceptance), update the global momentum sum, and run the
     cross-tree U-turn test between the two outermost edges.

Stopping: U-turn (sub or cross), divergence (|ΔH| > max_delta or non-finite),
or reaching max_tree_depth. Returns the multinomial-selected sample (NUTS has
no separate accept/reject — the proposal IS the next state).

U-turn criterion (generalized, diagonal mass): a (sub)tree is turning if
  (r_sum · M⁻¹ r_left) < 0   OR   (r_sum · M⁻¹ r_right) < 0
where r_left/r_right are the end momenta and r_sum the sum of momenta over it.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

from dezess.ensemble.hmc import kinetic_energy, sample_momentum

Array = jnp.ndarray


def _leapfrog1(q, p, g, eps_signed, inv_mass_diag, grad_fn):
    """Single leapfrog step with signed step size (direction folded into sign)."""
    p = p + 0.5 * eps_signed * g
    q = q + eps_signed * (inv_mass_diag * p)
    g = grad_fn(q)
    p = p + 0.5 * eps_signed * g
    return q, p, g


def _is_turning_single(r_left, r_right, r_sum, inv_mass_diag):
    """Generalized U-turn for one (sub)tree (diagonal mass)."""
    msum = inv_mass_diag * r_sum
    return (jnp.sum(msum * r_left) < 0.0) | (jnp.sum(msum * r_right) < 0.0)


class _State(NamedTuple):
    i: Array                # leaf counter (int)
    theta_minus: Array; r_minus: Array; grad_minus: Array
    theta_plus: Array; r_plus: Array; grad_plus: Array
    theta_prop: Array; lp_prop: Array
    log_w_total: Array      # logsumexp of leaf weights over whole tree
    r_sum_total: Array      # sum of momenta over whole trajectory (incl initial)
    ckpt_r: Array           # (D, d) pending left-endpoint momenta
    ckpt_Sbefore: Array     # (D, d) subtree momentum-sum before each left endpoint
    v: Array                # current doubling direction (+1/-1)
    S_sub: Array            # (d,) momentum sum within current subtree
    log_w_sub: Array        # logsumexp of leaf weights in current subtree
    theta_prop_sub: Array; lp_prop_sub: Array
    sub_turning: Array      # bool
    stop: Array             # bool
    diverged: Array         # bool — any leaf diverged
    sum_accept: Array       # running sum of min(1, exp(ΔH)) for step-size adapt
    n_leaf: Array           # number of leaves taken
    depth: Array            # max doubling depth reached
    key: Array


def nuts_step(q, lp, grad_q, key, step_size, inv_mass_diag,
              log_prob_fn, grad_fn, max_tree_depth=10, max_delta=1000.0):
    """One multinomial-NUTS transition for a single walker.

    Returns (q_out, lp_out, grad_out, key, accept_stat, depth, diverged).
    accept_stat = mean leaf Metropolis prob (for dual-averaging step-size adapt).
    """
    d = q.shape[0]
    D = int(max_tree_depth)
    max_leaves = (1 << D) - 1
    levels = jnp.arange(1, D + 1)               # 1..D
    mask2k = (jnp.left_shift(1, levels) - 1)    # 2^k - 1, shape (D,)

    key, k_mom = jax.random.split(key)
    p0 = sample_momentum(k_mom, inv_mass_diag)
    H0 = -lp + kinetic_energy(p0, inv_mass_diag)

    init = _State(
        i=jnp.asarray(0, jnp.int64),
        theta_minus=q, r_minus=p0, grad_minus=grad_q,
        theta_plus=q, r_plus=p0, grad_plus=grad_q,
        theta_prop=q, lp_prop=lp,
        log_w_total=jnp.float64(0.0),           # weight of initial state = exp(0)
        r_sum_total=p0,
        ckpt_r=jnp.zeros((D, d), dtype=jnp.float64),
        ckpt_Sbefore=jnp.zeros((D, d), dtype=jnp.float64),
        v=jnp.float64(1.0),
        S_sub=jnp.zeros(d, dtype=jnp.float64),
        log_w_sub=jnp.float64(-jnp.inf),
        theta_prop_sub=q, lp_prop_sub=lp,
        sub_turning=jnp.bool_(False),
        stop=jnp.bool_(False),
        diverged=jnp.bool_(False),
        sum_accept=jnp.float64(0.0),
        n_leaf=jnp.asarray(0, jnp.int64),
        depth=jnp.asarray(0, jnp.int64),
        key=key,
    )

    def cond(s):
        return (s.i < max_leaves) & (~s.stop)

    def body(s):
        i = s.i
        ip1 = i + 1
        # doubling index j = #{k in 1..D : i+1 >= 2^k}; within-subtree m = 1..2^j
        j = jnp.sum((ip1 >= jnp.left_shift(1, levels)).astype(jnp.int64))
        pow2j = jnp.left_shift(jnp.asarray(1, jnp.int64), j)
        m = i - (pow2j - 1) + 1
        is_new = (m == 1)
        is_last = (m == pow2j)

        key = s.key
        key, k_dir = jax.random.split(key)
        v_new = jnp.where(jax.random.uniform(k_dir) < 0.5, -1.0, 1.0)
        v = jnp.where(is_new, v_new, s.v)

        # reset subtree-local state at the start of a doubling
        S_sub = jnp.where(is_new, jnp.zeros(d), s.S_sub)
        log_w_sub = jnp.where(is_new, -jnp.inf, s.log_w_sub)
        sub_turning = jnp.where(is_new, jnp.bool_(False), s.sub_turning)
        ckpt_r = jnp.where(is_new, jnp.zeros((D, d)), s.ckpt_r)
        ckpt_Sbefore = jnp.where(is_new, jnp.zeros((D, d)), s.ckpt_Sbefore)

        # active edge = the one in direction v
        extend_plus = v > 0
        q_edge = jnp.where(extend_plus, s.theta_plus, s.theta_minus)
        r_edge = jnp.where(extend_plus, s.r_plus, s.r_minus)
        g_edge = jnp.where(extend_plus, s.grad_plus, s.grad_minus)

        # one leapfrog step
        q_new, r_new, g_new = _leapfrog1(
            q_edge, r_edge, g_edge, v * step_size, inv_mass_diag, grad_fn)
        lp_new = log_prob_fn(q_new)
        H_new = -lp_new + kinetic_energy(r_new, inv_mass_diag)
        dH = H_new - H0
        lw = jnp.where(jnp.isfinite(H_new), -dH, -jnp.inf)   # leaf log-weight = H0 - H_new
        diverging = (~jnp.isfinite(H_new)) | (dH > max_delta)
        leaf_accept = jnp.clip(jnp.exp(jnp.minimum(-dH, 0.0)), 0.0, 1.0)
        leaf_accept = jnp.where(jnp.isfinite(leaf_accept), leaf_accept, 0.0)

        # update extended edge
        theta_plus = jnp.where(extend_plus, q_new, s.theta_plus)
        r_plus = jnp.where(extend_plus, r_new, s.r_plus)
        grad_plus = jnp.where(extend_plus, g_new, s.grad_plus)
        theta_minus = jnp.where(extend_plus, s.theta_minus, q_new)
        r_minus = jnp.where(extend_plus, s.r_minus, r_new)
        grad_minus = jnp.where(extend_plus, s.grad_minus, g_new)

        # progressive (biased) multinomial sampling within subtree
        S_before = S_sub
        S_sub2 = S_sub + r_new
        first_leaf = is_new
        log_w_sub_new = jnp.where(first_leaf, lw, jnp.logaddexp(log_w_sub, lw))
        key, k_as = jax.random.split(key)
        acc_sub = jnp.where(
            first_leaf, jnp.bool_(True),
            jnp.log(jax.random.uniform(k_as) + 1e-30) < (lw - log_w_sub_new))
        theta_prop_sub = jnp.where(acc_sub, q_new, s.theta_prop_sub)
        lp_prop_sub = jnp.where(acc_sub, lp_new, s.lp_prop_sub)

        # ---- checkpoint WRITE (m odd) / CHECK (m even) ----
        x = m - 1
        m_odd = (m % 2) == 1
        write_k = m_odd & ((jnp.bitwise_and(x, mask2k)) == 0)            # (D,)
        check_k = (~m_odd) & ((jnp.bitwise_and(x, mask2k)) == mask2k)    # (D,)

        # WRITE: store this leaf momentum + subtree-sum-before at qualifying levels
        ckpt_r = jnp.where(write_k[:, None], r_new[None, :], ckpt_r)
        ckpt_Sbefore = jnp.where(write_k[:, None], S_before[None, :], ckpt_Sbefore)

        # CHECK: U-turn between stored left endpoint and current leaf
        rsum_sub = S_sub2[None, :] - ckpt_Sbefore             # (D, d)
        msum = inv_mass_diag[None, :] * rsum_sub               # M^-1 r_sum
        dot_left = jnp.sum(msum * ckpt_r, axis=1)              # (D,)
        dot_right = jnp.sum(msum * r_new[None, :], axis=1)     # (D,)
        turn_k = (dot_left < 0.0) | (dot_right < 0.0)
        any_turn = jnp.sum((check_k & turn_k).astype(jnp.int32)) > 0
        sub_turning2 = sub_turning | any_turn

        # ---- combine subtree into main tree at the last leaf ----
        valid_sub = (~sub_turning2) & (~diverging)
        do_combine = is_last & valid_sub
        log_w_total_new = jnp.where(
            do_combine, jnp.logaddexp(s.log_w_total, log_w_sub_new), s.log_w_total)
        key, k_am = jax.random.split(key)
        # biased acceptance: prob = min(1, w_sub / w_main_before)
        acc_main = (jnp.log(jax.random.uniform(k_am) + 1e-30)
                    < jnp.minimum(0.0, log_w_sub_new - s.log_w_total)) & do_combine
        theta_prop = jnp.where(acc_main, theta_prop_sub, s.theta_prop)
        lp_prop = jnp.where(acc_main, lp_prop_sub, s.lp_prop)
        r_sum_total = jnp.where(do_combine, s.r_sum_total + S_sub2, s.r_sum_total)

        # cross-tree U-turn between the two outermost edges
        cross_turn = _is_turning_single(r_minus, r_plus, r_sum_total, inv_mass_diag)
        stop2 = s.stop | diverging | (is_last & ((~valid_sub) | cross_turn))

        depth = jnp.maximum(s.depth, j)

        return _State(
            i=i + 1,
            theta_minus=theta_minus, r_minus=r_minus, grad_minus=grad_minus,
            theta_plus=theta_plus, r_plus=r_plus, grad_plus=grad_plus,
            theta_prop=theta_prop, lp_prop=lp_prop,
            log_w_total=log_w_total_new,
            r_sum_total=r_sum_total,
            ckpt_r=ckpt_r, ckpt_Sbefore=ckpt_Sbefore,
            v=v, S_sub=S_sub2, log_w_sub=log_w_sub_new,
            theta_prop_sub=theta_prop_sub, lp_prop_sub=lp_prop_sub,
            sub_turning=sub_turning2,
            stop=stop2,
            diverged=s.diverged | diverging,
            sum_accept=s.sum_accept + leaf_accept,
            n_leaf=s.n_leaf + 1,
            depth=depth,
            key=key,
        )

    final = lax.while_loop(cond, body, init)

    grad_out = grad_fn(final.theta_prop)
    accept_stat = final.sum_accept / jnp.maximum(final.n_leaf, 1).astype(jnp.float64)
    # Report number of doublings (= Stan/BlackJAX num_trajectory_expansions),
    # which is the 0-indexed max doubling index + 1 (>=1 doubling always runs).
    n_expansions = final.depth + 1
    return (final.theta_prop, final.lp_prop, grad_out, final.key,
            accept_stat, n_expansions, final.diverged, final.n_leaf)
