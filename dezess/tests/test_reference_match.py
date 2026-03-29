"""Layer 2: Match between GPU kernel and uncapped reference.

Strategy: compare the brackets (L, R) from expansion, then compare
outputs only when brackets match. When the reference expanded more
than N_EXPAND, the brackets differ by design -- this is expected.

Test categories:
1. Same brackets → same x_new (exact float64 match) → PASS
2. Different brackets → both found valid points → EXPECTED (not a bug)
3. GPU found=False → GPU stayed put → PASS (cap-hit in shrinking)
4. Same brackets, different x_new → FAILURE (bug)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dezess.sampler import (
    N_EXPAND,
    N_SHRINK,
    _safe_log_prob,
    _slice_sample_fixed,
    _update_one_walker,
)
from dezess.reference_sampler import reference_slice_sample
from dezess.targets import correlated_gaussian, isotropic_gaussian, student_t

jax.config.update("jax_enable_x64", True)


def _make_frozen_z(target, key, n_z=2000, overdispersed=False):
    samples = target.sample(key, n_z)
    if overdispersed:
        samples = samples * 3.0
    return samples


@pytest.fixture(params=["isotropic_10", "correlated_10", "student_t_10"])
def target(request):
    factories = {
        "isotropic_10": lambda: isotropic_gaussian(10),
        "correlated_10": lambda: correlated_gaussian(10),
        "student_t_10": lambda: student_t(10),
    }
    return factories[request.param]()


@pytest.fixture(params=[False, True], ids=["calibrated_z", "overdispersed_z"])
def overdispersed(request):
    return request.param


def test_slice_sample_bracket_match(target, overdispersed):
    """When brackets (L,R) match, x_new must match exactly."""
    n_test = 300
    z = _make_frozen_z(target, jax.random.PRNGKey(0), overdispersed=overdispersed)

    key = jax.random.PRNGKey(42)
    k_starts, k_dirs, k_tests = jax.random.split(key, 3)
    x_all = target.sample(k_starts, n_test)
    mu = jnp.float64(1.0)

    # Random directions from Z-matrix pairs
    def _pick_dir(ki):
        ki, k1, k2 = jax.random.split(ki, 3)
        i1 = jax.random.randint(k1, (), 0, z.shape[0])
        i2 = jax.random.randint(k2, (), 0, z.shape[0])
        i2 = jnp.where(i1 == i2, (i2 + 1) % z.shape[0], i2)
        diff = z[i1] - z[i2]
        return diff / jnp.maximum(jnp.sqrt(jnp.sum(diff ** 2)), 1e-30)

    d_all = jax.vmap(_pick_dir)(jax.random.split(k_dirs, n_test))
    lp_all = jax.vmap(target.log_prob)(x_all)
    keys = jax.random.split(k_tests, n_test)

    # GPU kernel (returns L, R)
    def _gpu(x, d, lp, k):
        return _slice_sample_fixed(target.log_prob, x, d, lp, mu, k)

    x_gpu, lp_gpu, _, found_gpu, L_gpu, R_gpu = jax.jit(jax.vmap(_gpu))(
        x_all, d_all, lp_all, keys
    )

    # Reference kernel (returns L, R)
    def _ref(x, d, lp, k):
        return reference_slice_sample(target.log_prob, x, d, lp, mu, k)

    x_ref, lp_ref, _, found_ref, n_exp, n_shr, L_ref, R_ref = jax.jit(jax.vmap(_ref))(
        x_all, d_all, lp_all, keys
    )

    # Compare
    L_g, R_g = np.asarray(L_gpu), np.asarray(R_gpu)
    L_r, R_r = np.asarray(L_ref), np.asarray(R_ref)
    x_g = np.asarray(x_gpu)
    x_r = np.asarray(x_ref)
    x_0 = np.asarray(x_all)

    same_bracket = (L_g == L_r) & (R_g == R_r)
    same_x = np.all(x_g == x_r, axis=1)

    n_same_bracket = same_bracket.sum()
    n_diff_bracket = (~same_bracket).sum()
    n_bracket_match_x_match = (same_bracket & same_x).sum()
    n_bracket_match_x_diff = (same_bracket & ~same_x).sum()

    # Stay-put check: when GPU says not found, it must stay put
    fg = np.asarray(found_gpu)
    stayed = np.all(x_g == x_0, axis=1)
    cap_hits = ~fg
    bad_cap = cap_hits & ~stayed  # found=False but didn't stay put

    print(f"\n  {target.name} (overdispersed={overdispersed}):")
    print(f"    same bracket:     {n_same_bracket}/{n_test}")
    print(f"    diff bracket:     {n_diff_bracket}/{n_test} (expected when ref expands more)")
    print(f"    bracket match → x match: {n_bracket_match_x_match}/{n_same_bracket}")
    print(f"    bracket match → x diff:  {n_bracket_match_x_diff}/{n_same_bracket} (BUG if >0)")
    print(f"    cap hits (found=False):   {cap_hits.sum()}")
    print(f"    bad cap (didn't stay put): {bad_cap.sum()}")
    print(f"    ref expand: median={np.median(np.asarray(n_exp)):.0f} max={np.asarray(n_exp).max()}")

    # CRITICAL: when brackets match, outputs MUST match
    assert n_bracket_match_x_diff == 0, (
        f"{n_bracket_match_x_diff} cases with same bracket but different x_new"
    )
    # Stay-put invariant
    assert bad_cap.sum() == 0, "found=False but x_new != x_old"


def test_stay_put_on_cap_hit():
    """found=False always produces x_new == x_old."""
    target = isotropic_gaussian(5)
    mu = jnp.float64(0.001)
    n_test = 500

    key = jax.random.PRNGKey(77)
    k1, k2, k3 = jax.random.split(key, 3)
    x_all = target.sample(k1, n_test)
    lp_all = jax.vmap(target.log_prob)(x_all)
    d_all = jax.random.normal(k2, (n_test, 5), dtype=jnp.float64)
    d_all = d_all / jnp.sqrt(jnp.sum(d_all ** 2, axis=1, keepdims=True))
    keys = jax.random.split(k3, n_test)

    def _one(x, d, lp, k):
        return _slice_sample_fixed(target.log_prob, x, d, lp, mu, k)

    x_new, _, _, found, _, _ = jax.jit(jax.vmap(_one))(x_all, d_all, lp_all, keys)

    cap_hits = ~np.asarray(found)
    if cap_hits.sum() > 0:
        stayed = np.all(np.asarray(x_new)[cap_hits] == np.asarray(x_all)[cap_hits], axis=1)
        print(f"\n  Stay-put: {cap_hits.sum()} cap-hits, {stayed.sum()} stayed put")
        assert stayed.all(), f"Only {stayed.sum()}/{cap_hits.sum()} stayed put"
    else:
        print("\n  Stay-put: no cap-hits (slice always found within budget)")


def test_sampler_end_to_end():
    """Full sampler produces correct mean and std on a known target."""
    from dezess.sampler import run_demcz_slice

    target = correlated_gaussian(10)
    key = jax.random.PRNGKey(0)
    init = target.sample(key, 64)

    result = run_demcz_slice(
        target.log_prob, init,
        n_steps=3000, n_warmup=500,
        key=jax.random.PRNGKey(1), mu=1.0,
        verbose=False,
    )

    samples = np.asarray(result["samples"]).reshape(-1, 10)
    mean_err = np.max(np.abs(samples.mean(0) - np.asarray(target.mean)))
    std_err = np.max(np.abs(samples.std(0) - np.sqrt(np.diag(np.asarray(target.cov)))))

    print(f"\n  End-to-end: mean_err={mean_err:.4f} std_err={std_err:.4f}")
    # Relaxed thresholds: smoke test, not convergence test (see test_convergence.py)
    assert mean_err < 0.5, f"Mean error {mean_err:.4f} > 0.5"
    assert std_err < 1.5, f"Std error {std_err:.4f} > 1.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
