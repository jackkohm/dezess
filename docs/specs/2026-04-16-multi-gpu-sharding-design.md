# Multi-GPU Sharding Design

## Goal

Add multi-GPU support to dezess via `jax.sharding`. Enable running larger ensembles (e.g., 256 walkers across 4 GPUs) with replicated Z-matrix, periodic sample gathering, and zero-overhead fallback to single-GPU when `n_gpus=1`.

## Constraints

- Modern JAX sharding (`jax.sharding.Mesh`, `NamedSharding`, `PartitionSpec`), NOT legacy `pmap`
- Backwards compatible: `n_gpus=1` (default) runs identically to current dezess
- All 72 existing tests must continue to pass
- Z-matrix replicated across GPUs (full diversity benefit, manageable communication cost)
- Configurable walker count: `n_walkers_per_gpu` and `n_gpus` separate parameters

## API

Two new parameters on `run_variant`:

```python
result = run_variant(
    log_prob_fn,
    init_positions,            # shape (n_gpus * n_walkers_per_gpu, ndim)
    n_steps=24000,
    config=config,
    n_warmup=4000,
    n_gpus=4,                  # NEW: default 1
    n_walkers_per_gpu=64,      # NEW: default 64 when n_gpus > 1
    checkpoint_path="run.npz",
    verbose=True,
)
```

Total walkers = `n_gpus * n_walkers_per_gpu`. The user must provide `init_positions` with the matching first-axis size.

When `n_gpus == 1`: existing single-GPU code path runs unchanged. The new params are inert.

When `n_gpus > 1`: shardings are set up, the step functions get `in_shardings`/`out_shardings` annotations, and the sample storage uses K=100 batched gathering.

## Mesh and shardings

```python
mesh = Mesh(jax.devices()[:n_gpus], ('walkers',))
walker_sharding = NamedSharding(mesh, P('walkers'))      # walker axis sharded
replicated = NamedSharding(mesh, P())                     # full replication
```

Shardings applied to:
- **Walker-sharded** (`P('walkers')`): `positions`, `log_probs`, walker auxiliary state, per-walker keys, samples-in-flight
- **Replicated** (`P()`): `z_padded`, `z_count`, `z_log_probs`, `mu`, `mu_blocks`, `block_L_padded`, GMM parameters

The first axis of every walker-sharded array is the walker axis (size `n_gpus * n_walkers_per_gpu`).

## Z-matrix synchronization

The Z-matrix is replicated on every GPU. The `circular_zmatrix.append` function operates on replicated arrays. To append walker positions to the Z-matrix, we use `with_sharding_constraint` to all-gather:

```python
# After each warmup step:
positions_full = jax.lax.with_sharding_constraint(positions, replicated)
log_probs_full = jax.lax.with_sharding_constraint(log_probs, replicated)
z_padded, z_count, z_log_probs = circular_zmatrix.append(
    z_padded, z_count, z_log_probs,
    positions_full, log_probs_full, z_max_size)
```

The `with_sharding_constraint` triggers a single NCCL all-gather collective per step. For 256 walkers × 21 dims × 8 bytes = ~43 KB per gather — trivially fast.

**During production:** Z-matrix is frozen. Each GPU's walkers read from the replicated local copy. Zero cross-GPU traffic in the inner loop.

## Sample storage with K=100 batching

Production samples are accumulated on each GPU's HBM for K=100 steps, then gathered to host:

```python
SAMPLE_BATCH_K = 100

@jax.jit  # with sharding declared via in/out_shardings
def _scan_steps_sharded(carry, _):
    new_carry, sample = step_fn(carry)
    return new_carry, sample

# Run K steps on-device, samples shape (K, total_walkers, ndim) sharded
carry, batch_samples = jax.lax.scan(
    _scan_steps_sharded, carry, None, length=SAMPLE_BATCH_K)

# Gather + transfer to host (one collective + one D2H per K steps)
batch_host = np.asarray(jax.device_get(batch_samples))
all_samples[step_idx:step_idx + SAMPLE_BATCH_K] = batch_host
```

**Memory cost per GPU per chunk:** `K × walkers_per_gpu × ndim × 8 bytes` = 100 × 64 × 21 × 8 = 1 MB. Negligible.

**Communication per K steps:** one all-gather of (100, 256, 21) ≈ 4.3 MB → host transfer.

**Streaming diagnostics + checkpointing:** also fire every K=100 steps, using the gathered host arrays. Matches dezess's existing diagnostic cadence.

## Initial position handling

If `init_positions.shape[0] != n_gpus * n_walkers_per_gpu`, raise `ValueError` with a clear message. We do NOT auto-tile or auto-truncate — the user must provide the correct shape.

The init array is sharded onto the mesh via `jax.device_put(init_positions, walker_sharding)`. Each GPU receives its slice of the walkers.

## What stays unchanged

- `log_prob_fn`: called per-walker, no sharding awareness needed
- All direction strategies (`de_mcz`, `global_move`, etc.): per-walker logic, JAX vmap handles sharding transparently
- All width strategies (`scale_aware`, `scalar`, etc.): per-walker scalar computation
- All slice strategies (`fixed`, `adaptive`, `nurs`, `multi_try`): per-walker control flow
- All ensemble strategies including `block_gibbs` MH+DR: per-walker logic
- Z-matrix circular append logic: same code, just operates on gathered positions
- Checkpoint format: gathers samples from all GPUs before saving (host-side numpy arrays)
- Resume: loads to host, then redistributes to mesh via `device_put`

## What changes in `loop.py`

1. **Sharding setup** at the top of `run_variant`:
   - If `n_gpus > 1`, build mesh + shardings via new helper `_setup_sharding(n_gpus, n_walkers)`
   - Validate `init_positions.shape[0]` matches expected total

2. **Step function JIT decorations** become conditional on `n_gpus`:
   - Single-GPU: `@jax.jit` (current behavior)
   - Multi-GPU: `@jax.jit(in_shardings=..., out_shardings=...)`

3. **Production loop** uses K=100 batched scan with sharded samples (already partially exists via `SCAN_BATCH=50`, generalized for sharding)

4. **Z-matrix append** during warmup uses `with_sharding_constraint` to all-gather positions before append

5. **Initial log-prob computation** vmaps over the sharded positions (JAX handles distribution)

6. **`_call_step` and similar wrappers** updated to handle the sharded carry tuple

## What changes in other files

- `dezess/core/sharding.py` (new): Mesh setup, sharding spec utilities, validation
- `dezess/checkpoint.py`: gather samples from all devices before saving (`jax.device_get`)
- `dezess/api.py`: pass-through for `n_gpus`/`n_walkers_per_gpu` params

## Testing

- `dezess/tests/test_sharding.py` (new):
  - **Skipped** if `len(jax.devices()) < 2`
  - Test 1: `n_gpus=1` produces identical results to current single-GPU code (same seed, same target)
  - Test 2: `n_gpus=2` runs to completion on a 10D Gaussian, recovers correct variance
  - Test 3: Sharded Z-matrix grows correctly during warmup (size matches expected)
  - Test 4: Block-Gibbs MH+DR works with sharding on a 21D target

For CI without multi-GPU access, all sharding tests are auto-skipped. They run only on the H200 nodes (or whenever `jax.devices()` returns >= 2 devices).

## Files Changed

- Create: `dezess/core/sharding.py` — Mesh + sharding helpers
- Create: `dezess/tests/test_sharding.py` — multi-GPU tests
- Modify: `dezess/core/loop.py` — sharding setup + sharded JIT + sharded scan
- Modify: `dezess/checkpoint.py` — host-side gather before save/load
- Modify: `dezess/api.py` — expose `n_gpus`, `n_walkers_per_gpu` params

## Out of scope (deliberately deferred)

- Sharding the log_prob across GPUs (option 3 from brainstorming) — not needed for current targets
- Sharded Z-matrix (option C from Q1) — adds cross-GPU read complexity for marginal memory savings
- Per-step sample gather — K=100 batching is strictly better
- Block-Gibbs with conditional independence parallelism + sharding — already conditional, just needs to inherit shardings
- Multi-host (multiple machines) — single-host multi-GPU only for now
