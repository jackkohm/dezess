"""Learned directions via normalizing flow.

Trains a simple RealNVP flow on warmup samples, then uses the flow's
learned transformation to generate informed directions. Inspired by
Cabezas et al. (2024) "Markovian Flow Matching", but adapted for
slice sampling rather than full Metropolis proposals.

Two direction strategies are available:
  1. Transport direction: d = flow(x) - x (normalized). This points
     from x toward where the flow maps it in reference space,
     providing a locally-informed direction.
  2. JVP probes: random JVP (Jacobian-vector product) through the flow,
     which is O(d) per direction instead of O(d^2) for full Jacobian.

The flow is trained during warmup using standard maximum likelihood.
Direction generation during production is cheap: one forward pass (or JVP)
per walker, fully vectorizable with vmap.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray


# --- Simple RealNVP building blocks ---

def _make_affine_coupling_params(key: Array, n_dim: int, hidden: int = 64) -> dict:
    """Initialize parameters for one affine coupling layer."""
    half = n_dim // 2
    input_dim = half
    output_dim = n_dim - half

    k1, k2, k3, k4 = jax.random.split(key, 4)
    scale = 0.01
    return {
        "w1": jax.random.normal(k1, (input_dim, hidden), dtype=jnp.float64) * scale,
        "b1": jnp.zeros(hidden, dtype=jnp.float64),
        "w_s": jax.random.normal(k2, (hidden, output_dim), dtype=jnp.float64) * scale,
        "b_s": jnp.zeros(output_dim, dtype=jnp.float64),
        "w_t": jax.random.normal(k3, (hidden, output_dim), dtype=jnp.float64) * scale,
        "b_t": jnp.zeros(output_dim, dtype=jnp.float64),
        "half": half,
    }


def _coupling_forward(params: dict, x: Array) -> Array:
    """Forward pass of one affine coupling layer."""
    half = params["half"]
    x1, x2 = x[:half], x[half:]
    h = jnp.tanh(x1 @ params["w1"] + params["b1"])
    log_s = jnp.clip(h @ params["w_s"] + params["b_s"], -5.0, 5.0)
    t = h @ params["w_t"] + params["b_t"]
    y2 = x2 * jnp.exp(log_s) + t
    return jnp.concatenate([x1, y2])


def init_flow(key: Array, n_dim: int, n_layers: int = 3, hidden: int = 64) -> list:
    """Initialize a RealNVP flow with alternating coupling layers."""
    params_list = []
    for i in range(n_layers):
        key, k = jax.random.split(key)
        p = _make_affine_coupling_params(k, n_dim, hidden)
        p["flip"] = bool(i % 2)
        params_list.append(p)
    return params_list


def flow_forward(params_list: list, x: Array) -> Array:
    """Forward pass through the full flow.

    The Python for-loop is unrolled at JIT trace time since params_list
    length is fixed. The if-checks on p["flip"] are resolved at trace
    time (static booleans from dict).
    """
    for p in params_list:
        if p["flip"]:
            x = jnp.flip(x)
        x = _coupling_forward(p, x)
        if p["flip"]:
            x = jnp.flip(x)
    return x


def train_flow(
    key: Array,
    samples: Array,
    n_dim: int,
    n_epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_layers: int = 3,
    hidden: int = 64,
) -> list:
    """Train a RealNVP flow on samples using maximum likelihood.

    Training is JIT-compiled per gradient step for GPU efficiency.
    """
    key, k_init = jax.random.split(key)
    params_list = init_flow(k_init, n_dim, n_layers, hidden)
    n_samples = samples.shape[0]

    mean = jnp.mean(samples, axis=0)
    std = jnp.std(samples, axis=0)
    std = jnp.maximum(std, 1e-6)
    samples_std = (samples - mean) / std

    static_configs = [{"flip": p["flip"], "half": p["half"]} for p in params_list]
    diff_params = [{k: v for k, v in p.items() if k not in ("flip", "half")}
                   for p in params_list]

    def neg_log_likelihood(diff_params, batch):
        def per_sample(x):
            full_params = [{**dp, **sc} for dp, sc in zip(diff_params, static_configs)]
            z = flow_forward(full_params, x)
            log_pz = -0.5 * jnp.sum(z ** 2) - 0.5 * n_dim * jnp.log(2 * jnp.pi)
            log_det = 0.0
            y = x
            for dp, sc in zip(diff_params, static_configs):
                p = {**dp, **sc}
                if sc["flip"]:
                    y = jnp.flip(y)
                y1 = y[:sc["half"]]
                h = jnp.tanh(y1 @ p["w1"] + p["b1"])
                log_s = jnp.clip(h @ p["w_s"] + p["b_s"], -5.0, 5.0)
                log_det = log_det + jnp.sum(log_s)
                y = _coupling_forward(p, y)
                if sc["flip"]:
                    y = jnp.flip(y)
            return -(log_pz + log_det)
        return jnp.mean(jax.vmap(per_sample)(batch))

    @jax.jit
    def train_step(diff_params, batch):
        grads = jax.grad(neg_log_likelihood)(diff_params, batch)
        return jax.tree.map(lambda p, g: p - lr * g, diff_params, grads)

    for epoch in range(n_epochs):
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, n_samples)
        for start in range(0, n_samples - batch_size + 1, batch_size):
            batch = samples_std[perm[start:start + batch_size]]
            diff_params = train_step(diff_params, batch)

    params_list = [{**dp, **sc} for dp, sc in zip(diff_params, static_configs)]
    params_list = [dict(p, _mean=mean, _std=std) for p in params_list]
    return params_list


def precompute_flow_directions(
    flow_params: list,
    z_matrix: Array,
    z_count: Array,
    n_directions: int = 500,
    key: Optional[Array] = None,
) -> Array:
    """Precompute direction bank using transport directions and JVP probes.

    Two types of directions:
    1. Transport: d = flow(x_std) - x_std (where the flow maps each point)
    2. JVP probe: J_flow(x_std) @ random_vec (O(d) per direction, not O(d^2))

    Returns (n_directions, n_dim) array of unit direction vectors.
    """
    n_dim = z_matrix.shape[1]
    mean = flow_params[0].get("_mean", jnp.zeros(n_dim))
    std = flow_params[0].get("_std", jnp.ones(n_dim))

    if key is None:
        key = jax.random.PRNGKey(0)

    n_points = min(n_directions, int(z_count))
    key, k_idx = jax.random.split(key)
    point_indices = jax.random.choice(k_idx, int(z_count), (n_points,), replace=False)
    points = z_matrix[point_indices]
    points_std = (points - mean) / std

    # Strategy 1: Transport directions (half the bank)
    # d = flow(x) - x: where the flow wants to send this point
    @jax.jit
    def transport_direction(x_std):
        fx = flow_forward(flow_params, x_std)
        d = fx - x_std
        d_norm = jnp.sqrt(jnp.sum(d ** 2))
        return d / jnp.maximum(d_norm, 1e-30)

    n_transport = n_points // 2
    transport_dirs = jax.vmap(transport_direction)(points_std[:n_transport])

    # Strategy 2: JVP probes (other half)
    # d = J_flow(x) @ v for random v — O(d) via jax.jvp, not O(d^2)
    @jax.jit
    def jvp_direction(x_std, v):
        _, jvp_val = jax.jvp(
            lambda x: flow_forward(flow_params, x),
            (x_std,), (v,)
        )
        d_norm = jnp.sqrt(jnp.sum(jvp_val ** 2))
        return jvp_val / jnp.maximum(d_norm, 1e-30)

    n_jvp = n_points - n_transport
    key, k_vecs = jax.random.split(key)
    random_vecs = jax.random.normal(k_vecs, (n_jvp, n_dim), dtype=jnp.float64)
    random_vecs = random_vecs / jnp.sqrt(jnp.sum(random_vecs ** 2, axis=1, keepdims=True))
    jvp_dirs = jax.vmap(jvp_direction)(points_std[n_transport:n_transport + n_jvp], random_vecs)

    # Combine
    directions = jnp.concatenate([transport_dirs, jvp_dirs], axis=0)

    # Pad or tile to n_directions
    if directions.shape[0] < n_directions:
        reps = (n_directions // directions.shape[0]) + 1
        directions = jnp.tile(directions, (reps, 1))[:n_directions]

    return directions


def sample_direction(
    x_i: Array,
    z_matrix: Array,
    z_count: Array,
    z_log_probs: Array,
    key: Array,
    aux: Array,
    flow_directions: Optional[Array] = None,
    flow_mix: float = 0.5,
    **kwargs,
) -> tuple[Array, Array, Array]:
    """Sample direction from precomputed flow direction bank.

    With probability flow_mix, pick a random direction from the bank
    (with random sign for symmetry). Otherwise, use standard DE-MCz.

    This is O(1) per walker per step — just an array lookup + sign flip.
    No Jacobians, no forward passes, no ODE solves in the hot loop.
    """
    n_dim = x_i.shape[0]
    key, k_choice, k_dir_idx, k_sign, k_idx1, k_idx2 = jax.random.split(key, 6)

    # DE-MCz fallback
    idx1 = jax.random.randint(k_idx1, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jax.random.randint(k_idx2, (), 0, z_matrix.shape[0]) % z_count
    idx2 = jnp.where(idx1 == idx2, (idx2 + 1) % z_count, idx2)
    diff = z_matrix[idx1] - z_matrix[idx2]
    norm = jnp.sqrt(jnp.sum(diff ** 2))
    d_demcz = diff / jnp.maximum(norm, 1e-30)

    # Flow direction: O(1) lookup from precomputed bank
    n_bank = flow_directions.shape[0]
    dir_idx = jax.random.randint(k_dir_idx, (), 0, n_bank)
    d_flow = flow_directions[dir_idx]

    # Random sign for symmetry
    sign = 2.0 * jax.random.bernoulli(k_sign).astype(jnp.float64) - 1.0
    d_flow = d_flow * sign

    # Mix
    use_flow = jax.random.bernoulli(k_choice, flow_mix)
    d = jnp.where(use_flow, d_flow, d_demcz)

    # Store DE-MCz pair distance for scale_aware width compatibility
    aux = aux._replace(direction_scale=norm)

    return d, key, aux
