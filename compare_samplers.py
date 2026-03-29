#!/usr/bin/env python
"""Head-to-head comparison: dezess vs emcee vs zeus vs blackjax NUTS vs numpyro NUTS.

All on the same analytical targets with the same metrics (ESS/sec, ESS/eval).
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import gc
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.stdout.reconfigure(line_buffering=True)

print(f"JAX devices: {jax.devices()}", flush=True)

# ── Metrics ──────────────────────────────────────────────────────────────

def ess_fft(chain):
    n = len(chain)
    if n < 10:
        return 0.0
    x = chain - chain.mean()
    var = np.var(x)
    if var < 1e-30:
        return 0.0
    fft_x = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real / (n * var)
    tau = 1.0
    for lag in range(1, n):
        tau += 2 * acf[lag]
        if lag >= 5 * tau:
            break
    return n / max(tau, 1.0)


def compute_ess(samples_3d):
    """(n_steps, n_walkers, n_dim) -> ESS per dim."""
    mean_chain = samples_3d.mean(axis=1)
    return np.array([ess_fft(mean_chain[:, d]) for d in range(mean_chain.shape[1])])


# ── Targets ──────────────────────────────────────────────────────────────

def make_targets():
    """Return dict of targets, each with log_prob, ndim, name, numpy_log_prob."""
    targets = {}

    # Correlated Gaussian (condition ~50)
    ndim = 10
    rng = np.random.default_rng(42)
    A = rng.standard_normal((ndim, ndim))
    cov = A @ A.T / ndim + np.eye(ndim)
    evals_raw, evecs = np.linalg.eigh(cov)
    evals_new = np.linspace(1.0, 50.0, ndim)
    cov = evecs @ np.diag(evals_new) @ evecs.T
    cov = (cov + cov.T) / 2
    mu = rng.standard_normal(ndim)
    prec = np.linalg.inv(cov)
    L = np.linalg.cholesky(cov)

    mu_j = jnp.array(mu, dtype=jnp.float64)
    prec_j = jnp.array(prec, dtype=jnp.float64)

    def jax_lp_corr(x):
        d = x - mu_j
        return -0.5 * d @ prec_j @ d

    def np_lp_corr(x):
        d = x - mu
        return -0.5 * d @ prec @ d

    targets["correlated_10"] = {
        "jax_lp": jax_lp_corr,
        "np_lp": np_lp_corr,
        "ndim": ndim,
        "mu": mu,
        "cov": cov,
        "L": L,
    }

    # Ill-conditioned Gaussian (condition 1000)
    ndim = 10
    rng2 = np.random.default_rng(123)
    A2 = rng2.standard_normal((ndim, ndim))
    Q, _ = np.linalg.qr(A2)
    evals2 = np.geomspace(1.0, 1000.0, ndim)
    cov2 = Q @ np.diag(evals2) @ Q.T
    cov2 = (cov2 + cov2.T) / 2
    prec2 = np.linalg.inv(cov2)
    L2 = np.linalg.cholesky(cov2)
    mu2 = np.zeros(ndim)

    mu2_j = jnp.array(mu2, dtype=jnp.float64)
    prec2_j = jnp.array(prec2, dtype=jnp.float64)

    def jax_lp_ill(x):
        d = x - mu2_j
        return -0.5 * d @ prec2_j @ d

    def np_lp_ill(x):
        d = x - mu2
        return -0.5 * d @ prec2 @ d

    targets["ill_conditioned_10"] = {
        "jax_lp": jax_lp_ill,
        "np_lp": np_lp_ill,
        "ndim": ndim,
        "mu": mu2,
        "cov": cov2,
        "L": L2,
    }

    # Rosenbrock (banana)
    ndim = 4

    def jax_lp_rosen(x):
        return -jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

    def np_lp_rosen(x):
        return -np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

    targets["rosenbrock_4"] = {
        "jax_lp": jax_lp_rosen,
        "np_lp": np_lp_rosen,
        "ndim": ndim,
        "mu": np.ones(ndim),
        "cov": None,
        "L": None,
    }

    return targets


# ── Samplers ─────────────────────────────────────────────────────────────

def run_dezess(target, n_walkers=64, n_steps=5000, n_warmup=1000):
    """Run dezess (snooker_stochastic default)."""
    from dezess import run_variant

    ndim = target["ndim"]
    key = jax.random.PRNGKey(42)
    if target["L"] is not None:
        init = jax.random.normal(key, (n_walkers, ndim)) @ jnp.array(target["L"]).T + jnp.array(target["mu"])
    else:
        init = jax.random.normal(key, (n_walkers, ndim)) * 0.5 + 1.0

    t0 = time.time()
    result = run_variant(
        target["jax_lp"], init,
        n_steps=n_steps, n_warmup=n_warmup,
        key=jax.random.PRNGKey(1), verbose=False,
    )
    wall = time.time() - t0

    samples = np.asarray(result["samples"])
    ess = compute_ess(samples)
    n_prod = n_steps - n_warmup

    return {
        "ess_min": ess.min(),
        "ess_per_sec": ess.min() / wall,
        "wall_time": wall,
        "n_prod": n_prod,
        "n_walkers": n_walkers,
    }


def run_emcee(target, n_walkers=64, n_steps=5000, n_warmup=1000):
    """Run emcee (affine-invariant ensemble sampler)."""
    import emcee

    ndim = target["ndim"]
    rng = np.random.default_rng(42)
    if target["L"] is not None:
        init = rng.standard_normal((n_walkers, ndim)) @ target["L"].T + target["mu"]
    else:
        init = rng.standard_normal((n_walkers, ndim)) * 0.5 + 1.0

    sampler = emcee.EnsembleSampler(n_walkers, ndim, target["np_lp"])

    t0 = time.time()
    sampler.run_mcmc(init, n_steps, progress=False)
    wall = time.time() - t0

    samples = sampler.get_chain(discard=n_warmup)  # (n_prod, n_walkers, ndim)
    ess = compute_ess(samples)

    return {
        "ess_min": ess.min(),
        "ess_per_sec": ess.min() / wall,
        "wall_time": wall,
        "n_prod": n_steps - n_warmup,
        "n_walkers": n_walkers,
    }


def run_zeus(target, n_walkers=64, n_steps=5000, n_warmup=1000):
    """Run zeus (ensemble slice sampler)."""
    import zeus

    ndim = target["ndim"]
    rng = np.random.default_rng(42)
    if target["L"] is not None:
        init = rng.standard_normal((n_walkers, ndim)) @ target["L"].T + target["mu"]
    else:
        init = rng.standard_normal((n_walkers, ndim)) * 0.5 + 1.0

    sampler = zeus.EnsembleSampler(n_walkers, ndim, target["np_lp"])

    t0 = time.time()
    sampler.run_mcmc(init, n_steps)
    wall = time.time() - t0

    samples = sampler.get_chain(discard=n_warmup)  # (n_prod, n_walkers, ndim)
    ess = compute_ess(samples)

    return {
        "ess_min": ess.min(),
        "ess_per_sec": ess.min() / wall,
        "wall_time": wall,
        "n_prod": n_steps - n_warmup,
        "n_walkers": n_walkers,
    }


def run_blackjax_nuts(target, n_chains=4, n_steps=5000, n_warmup=1000):
    """Run blackjax NUTS (gradient-based, GPU-accelerated)."""
    import blackjax

    ndim = target["ndim"]
    key = jax.random.PRNGKey(42)

    if target["L"] is not None:
        init_pos = jax.random.normal(key, (n_chains, ndim)) @ jnp.array(target["L"]).T + jnp.array(target["mu"])
    else:
        init_pos = jax.random.normal(key, (n_chains, ndim)) * 0.5 + 1.0

    log_prob = target["jax_lp"]

    # Warmup to find step size and mass matrix
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob, num_integration_steps=50)

    @jax.jit
    def one_chain(init, key):
        key_warmup, key_sample = jax.random.split(key)
        (state, params), _ = warmup.run(key_warmup, init, n_warmup)
        kernel = blackjax.nuts(log_prob, **params).step

        def step_fn(carry, k):
            state, info = kernel(k, carry)
            return state, state.position

        keys = jax.random.split(key_sample, n_steps - n_warmup)
        _, samples = jax.lax.scan(step_fn, state, keys)
        return samples

    t0 = time.time()
    keys = jax.random.split(jax.random.PRNGKey(1), n_chains)
    all_samples = jax.vmap(one_chain)(init_pos, keys)  # (n_chains, n_prod, ndim)
    all_samples.block_until_ready()
    wall = time.time() - t0

    # Reshape to (n_prod, n_chains, ndim) for ESS computation
    samples = np.asarray(jnp.transpose(all_samples, (1, 0, 2)))
    ess = compute_ess(samples)

    return {
        "ess_min": ess.min(),
        "ess_per_sec": ess.min() / wall,
        "wall_time": wall,
        "n_prod": n_steps - n_warmup,
        "n_walkers": n_chains,
    }


def run_numpyro_nuts(target, n_chains=4, n_steps=5000, n_warmup=1000):
    """Run numpyro NUTS."""
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(1)

    ndim = target["ndim"]
    log_prob = target["jax_lp"]

    def model():
        x = numpyro.sample("x", dist.Normal(jnp.zeros(ndim), jnp.ones(ndim) * 10.0))
        numpyro.factor("lp", log_prob(x))

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_steps - n_warmup,
                num_chains=n_chains, progress_bar=False)

    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(42))
    wall = time.time() - t0

    samples_raw = mcmc.get_samples()["x"]  # (n_prod * n_chains, ndim) or (n_prod, ndim)
    n_prod = n_steps - n_warmup
    if samples_raw.shape[0] == n_prod * n_chains:
        samples = np.asarray(samples_raw).reshape(n_chains, n_prod, ndim).transpose(1, 0, 2)
    else:
        samples = np.asarray(samples_raw).reshape(n_prod, 1, ndim)

    ess = compute_ess(samples)

    return {
        "ess_min": ess.min(),
        "ess_per_sec": ess.min() / wall,
        "wall_time": wall,
        "n_prod": n_prod,
        "n_walkers": n_chains,
    }


# ── Main ─────────────────────────────────────────────────────────────────

SAMPLERS = {
    "dezess": run_dezess,
    "emcee": run_emcee,
    "zeus": run_zeus,
    "blackjax_nuts": run_blackjax_nuts,
    "numpyro_nuts": run_numpyro_nuts,
}

def main():
    targets = make_targets()
    n_steps = 5000
    n_warmup = 1000

    results = []
    total = len(SAMPLERS) * len(targets)
    idx = 0

    for tname, target in targets.items():
        for sname, run_fn in SAMPLERS.items():
            idx += 1
            print(f"[{idx}/{total}] {sname:20s} x {tname:25s} ... ", end="", flush=True)
            try:
                t0 = time.time()
                r = run_fn(target, n_steps=n_steps, n_warmup=n_warmup)
                r["sampler"] = sname
                r["target"] = tname
                results.append(r)
                print(f"ESS/s={r['ess_per_sec']:8.1f}  ESS_min={r['ess_min']:8.1f}  "
                      f"wall={r['wall_time']:6.1f}s", flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                results.append({"sampler": sname, "target": tname, "error": str(e)})

            gc.collect()

    # Print comparison table
    print("\n" + "=" * 90)
    print(f"  {'SAMPLER COMPARISON':^86s}")
    print("=" * 90)

    for tname in targets:
        print(f"\n  TARGET: {tname}")
        print(f"  {'Sampler':<20s} {'ESS_min':>10s} {'ESS/sec':>10s} {'Wall(s)':>10s} {'Walkers':>10s} {'Grad-free':>10s}")
        print("  " + "-" * 72)

        target_results = sorted(
            [r for r in results if r.get("target") == tname and "error" not in r],
            key=lambda r: r.get("ess_per_sec", 0),
            reverse=True,
        )

        for r in target_results:
            grad_free = "yes" if r["sampler"] in ("dezess", "emcee", "zeus") else "no"
            marker = " <-- best" if r == target_results[0] else ""
            print(f"  {r['sampler']:<20s} {r['ess_min']:>10.1f} {r['ess_per_sec']:>10.1f} "
                  f"{r['wall_time']:>10.1f} {r['n_walkers']:>10d} {grad_free:>10s}{marker}")

        # Errors
        for r in results:
            if r.get("target") == tname and "error" in r:
                print(f"  {r['sampler']:<20s} {'ERROR':>10s}: {r['error'][:50]}")

    # Summary
    print("\n" + "=" * 90)
    print("  WINNERS")
    print("=" * 90)
    for tname in targets:
        valid = [r for r in results if r.get("target") == tname and "error" not in r]
        if valid:
            best = max(valid, key=lambda r: r.get("ess_per_sec", 0))
            best_gf = max([r for r in valid if r["sampler"] in ("dezess", "emcee", "zeus")],
                          key=lambda r: r.get("ess_per_sec", 0), default=None)
            print(f"  {tname:<25s} Overall: {best['sampler']:<15s} ({best['ess_per_sec']:.1f} ESS/s)")
            if best_gf and best_gf != best:
                print(f"  {'':25s} Grad-free: {best_gf['sampler']:<15s} ({best_gf['ess_per_sec']:.1f} ESS/s)")


if __name__ == "__main__":
    main()
