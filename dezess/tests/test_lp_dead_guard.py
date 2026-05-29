"""Regression tests for the LP_DEAD guard in block-MH paths.

Reproduces the snooker-pump runaway from the production traces: with snooker
and a target that floors to -1e8 outside a feasible ball, tightly-clustered
walkers should NOT escape to large norms. Without the guard, the (bsize-1)*log(r)
Jacobian pumps walkers outward on the flat floor.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig


NDIM = 30
N_WALKERS = 64
N_WARMUP = 200
N_PROD = 400
FLOOR = -1e8
FEASIBLE_R = 3.0


@jax.jit
def floored_log_prob(x):
    """Smooth Gaussian inside r=3, hard -1e8 floor outside."""
    r2 = jnp.sum(x * x)
    return jnp.where(r2 < FEASIBLE_R ** 2, -0.5 * r2, jnp.float64(FLOOR))


def _make_cfg(lp_dead=None):
    ens = {
        "block_sizes": [NDIM],          # single 30-dim block, like the real bug
        "use_mh": True,
        "delayed_rejection": True,
        "complementary_prob": 0.5,
        "snooker_prob": 0.10,           # snooker active — the runaway driver
        "gamma_jump_prob": 0.05,
    }
    if lp_dead is not None:
        ens["lp_dead"] = lp_dead
    return VariantConfig(
        name=f"bgmh_floortest_lpdead{lp_dead}",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="block_gibbs",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
        ensemble_kwargs=ens,
    )


def _max_norm(samples_3d):
    return float(np.linalg.norm(np.array(samples_3d), axis=-1).max())


def _frac_on_floor(log_probs_2d):
    return float((np.array(log_probs_2d) < -1e6).mean())


def test_default_guard_keeps_walkers_bounded():
    """With the default lp_dead = -1e6, walkers must stay near the feasible ball.

    Init at a tight cluster well inside r=3. If the guard works, walkers
    diffuse inside the ball and never accumulate beyond r ≈ 4 (a generous
    margin for the rejected-but-still-counted brief excursions).
    """
    rng = np.random.default_rng(0)
    init = 0.3 * rng.standard_normal((N_WALKERS, NDIM))
    init_jax = jnp.asarray(init, dtype=jnp.float64)

    result = run_variant(
        floored_log_prob, init_jax,
        n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
        config=_make_cfg(),   # default lp_dead = -1e6
        key=jax.random.PRNGKey(0), verbose=False,
    )
    samples = result["samples"]
    log_probs = result["log_probs"]

    max_norm = _max_norm(samples)
    frac_floor = _frac_on_floor(log_probs)

    # Reasonable bound: walkers may briefly drift to ~r=4 from the boundary,
    # but should NEVER reach z-norm ≈ 10+ (the runaway signature was 5000+).
    assert max_norm < 6.0, (
        f"Walkers escaped to max norm {max_norm:.1f} despite the guard "
        f"(expected < 6 inside the r=3 feasible ball with margin).")
    assert frac_floor < 0.05, (
        f"{frac_floor:.1%} of production samples are on the floor — guard "
        f"should keep walkers off the dead zone (expected < 5%).")


def test_without_guard_runaway_reproduces():
    """Without the guard (lp_dead = -inf), the snooker pump should let
    walkers escape onto the flat floor. This is the *bug we are fixing*.

    Setting lp_dead = -1e10 disables the guard (floor is at -1e8 > -1e10,
    so prop_ok = True everywhere). Without it, walkers should accumulate
    samples on the floor.
    """
    rng = np.random.default_rng(0)
    init = 0.3 * rng.standard_normal((N_WALKERS, NDIM))
    init_jax = jnp.asarray(init, dtype=jnp.float64)

    result = run_variant(
        floored_log_prob, init_jax,
        n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
        config=_make_cfg(lp_dead=-1e10),
        key=jax.random.PRNGKey(0), verbose=False,
    )
    samples = result["samples"]
    log_probs = result["log_probs"]

    max_norm = _max_norm(samples)
    frac_floor = _frac_on_floor(log_probs)

    # If the runaway reproduces, we should see either many floor samples OR
    # large norms. If neither happens (the bug doesn't trigger on this seed),
    # the test is still informative — but we expect SOME divergence.
    runaway = (max_norm > 6.0) or (frac_floor > 0.05)
    assert runaway, (
        "Expected the snooker pump to push walkers into the dead zone with "
        f"the guard disabled, but max_norm={max_norm:.1f} and floor_frac="
        f"{frac_floor:.1%}. Bug may not reproduce on this seed/config; "
        "consider strengthening init clustering or snooker_prob.")


def test_byte_identical_when_target_never_hits_floor():
    """On a target where the floor is never reached, guard vs no-guard
    must produce identical samples (byte-identical reproducibility)."""
    @jax.jit
    def smooth_log_prob(x):
        return -0.5 * jnp.sum(x * x)

    rng = np.random.default_rng(42)
    init = rng.standard_normal((N_WALKERS, NDIM))   # init at sigma=1, matched
    init_jax = jnp.asarray(init, dtype=jnp.float64)

    result_default = run_variant(
        smooth_log_prob, init_jax,
        n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
        config=_make_cfg(),
        key=jax.random.PRNGKey(42), verbose=False,
    )
    result_relaxed = run_variant(
        smooth_log_prob, init_jax,
        n_steps=N_WARMUP + N_PROD, n_warmup=N_WARMUP,
        config=_make_cfg(lp_dead=-1e10),
        key=jax.random.PRNGKey(42), verbose=False,
    )
    s_default = np.array(result_default["samples"])
    s_relaxed = np.array(result_relaxed["samples"])

    # All lp values are near -d/2 = -15, well above both lp_dead values,
    # so the guard never fires → samples must be byte-identical.
    assert np.array_equal(s_default, s_relaxed), (
        "Guard should not change samples when target never reaches lp_dead. "
        f"Max abs diff: {np.abs(s_default - s_relaxed).max():.2e}")


if __name__ == "__main__":
    test_default_guard_keeps_walkers_bounded()
    print("PASS: default guard keeps walkers bounded")
    test_without_guard_runaway_reproduces()
    print("PASS: without guard, runaway reproduces")
    test_byte_identical_when_target_never_hits_floor()
    print("PASS: byte-identical on smooth target")
