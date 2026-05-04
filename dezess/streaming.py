"""Streaming write/read/resume helpers for dezess.

Writes samples + sampler state to disk during a `run_variant` call so
runs can be inspected live (mmap reads) and resumed after kill.

Disk layout::

    <stream_path>/
        manifest.json
        state/                 (overwritten each batch, atomic via tmp+rename)
            z_matrix.npy
            z_log_probs.npy
            z_count.npy
            mu.npy
            walker_aux_prev_dir.npy
            walker_aux_bw.npy
            walker_aux_da.npy
            walker_aux_ds.npy
            last_positions.npy
            last_lps.npy
        chunk_NNN/             (one per `run_variant` call)
            samples.npy        (mmap, shape (n_production, n_walkers, n_dim))
            log_probs.npy      (mmap, shape (n_production, n_walkers))
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


PathLike = Union[str, Path]


class Streamer:
    """Stream-writer for one `run_variant` call.

    Each `Streamer` writes ONE chunk. Multiple chunks accumulate in the
    same `stream_path` directory across multiple calls (e.g. resume after
    kill). The chunk index is determined from the existing manifest.
    """

    def __init__(
        self,
        stream_path: PathLike,
        n_walkers: int,
        n_dim: int,
        n_production: int,
        config_name: str,
        z_capacity: int,
    ):
        self.stream_path = Path(stream_path)
        self.n_walkers = int(n_walkers)
        self.n_dim = int(n_dim)
        self.n_production = int(n_production)
        self.config_name = str(config_name)
        self.z_capacity = int(z_capacity)

        self.stream_path.mkdir(parents=True, exist_ok=True)
        (self.stream_path / "state").mkdir(parents=True, exist_ok=True)

        self._chunk_idx: Optional[int] = None
        self._chunk_dir: Optional[Path] = None
        self._samples_mm: Optional[np.memmap] = None
        self._lps_mm: Optional[np.memmap] = None
        self._cursor = 0       # next row to write within this chunk

    @property
    def chunk_idx(self) -> Optional[int]:
        """1-indexed chunk number assigned by open_chunk(); None before open_chunk()."""
        return self._chunk_idx

    def open_chunk(self) -> None:
        """Allocate a new chunk_NNN/ with mmaps sized to `n_production`."""
        manifest = _read_manifest(self.stream_path)
        next_idx = len(manifest.get("chunks", [])) + 1
        chunk_name = f"chunk_{next_idx:03d}"
        self._chunk_idx = next_idx
        self._chunk_dir = self.stream_path / chunk_name
        self._chunk_dir.mkdir(parents=True, exist_ok=True)

        self._samples_mm = np.lib.format.open_memmap(
            self._chunk_dir / "samples.npy",
            mode="w+", dtype=np.float64,
            shape=(self.n_production, self.n_walkers, self.n_dim),
        )
        self._lps_mm = np.lib.format.open_memmap(
            self._chunk_dir / "log_probs.npy",
            mode="w+", dtype=np.float64,
            shape=(self.n_production, self.n_walkers),
        )
        self._cursor = 0

        # Append to manifest immediately so a kill mid-run still reflects
        # the chunk that was started.
        manifest.setdefault("chunks", []).append(chunk_name)
        manifest["n_walkers"] = self.n_walkers
        manifest["n_dim"] = self.n_dim
        manifest["config_name"] = self.config_name
        _write_json_atomic(self.stream_path / "manifest.json", manifest)

    def append_batch(self, samples_b: np.ndarray, log_probs_b: np.ndarray) -> None:
        """Write a batch of (k, n_walkers, n_dim) samples + (k, n_walkers) log_probs."""
        if self._samples_mm is None or self._lps_mm is None:
            raise RuntimeError("open_chunk() must be called before append_batch()")
        k = samples_b.shape[0]
        if self._cursor + k > self.n_production:
            raise ValueError(
                f"chunk overflow: cursor={self._cursor} + batch={k} > "
                f"capacity={self.n_production}"
            )
        self._samples_mm[self._cursor:self._cursor + k] = samples_b
        self._lps_mm[self._cursor:self._cursor + k] = log_probs_b
        self._samples_mm.flush()
        self._lps_mm.flush()
        self._cursor += k

    def save_state(
        self,
        *,
        z_matrix: np.ndarray,
        z_log_probs: np.ndarray,
        z_count: int,
        mu: float,
        walker_aux: Dict[str, np.ndarray],
        last_positions: np.ndarray,
        last_lps: np.ndarray,
        n_steps_done_in_chunk: int,
    ) -> None:
        """Atomically overwrite the state/ directory with current sampler state."""
        state = self.stream_path / "state"
        _save_npy_atomic(state / "z_matrix.npy", z_matrix)
        _save_npy_atomic(state / "z_log_probs.npy", z_log_probs)
        _save_npy_atomic(state / "z_count.npy", np.int64(z_count))
        _save_npy_atomic(state / "mu.npy", np.float64(mu))
        _save_npy_atomic(state / "walker_aux_prev_dir.npy", walker_aux["prev_direction"])
        _save_npy_atomic(state / "walker_aux_bw.npy", walker_aux["bracket_widths"])
        _save_npy_atomic(state / "walker_aux_da.npy", walker_aux["direction_anchor"])
        _save_npy_atomic(state / "walker_aux_ds.npy", walker_aux["direction_scale"])
        _save_npy_atomic(state / "last_positions.npy", last_positions)
        _save_npy_atomic(state / "last_lps.npy", last_lps)

        # Cumulative across all chunks
        manifest = _read_manifest(self.stream_path)
        per_chunk_done = manifest.get("steps_done_per_chunk", {})
        per_chunk_done[f"chunk_{self._chunk_idx:03d}"] = int(n_steps_done_in_chunk)
        manifest["steps_done_per_chunk"] = per_chunk_done
        _write_json_atomic(self.stream_path / "manifest.json", manifest)

    def close(self) -> None:
        """Final flush; mmaps will be closed when garbage-collected."""
        if self._samples_mm is not None:
            self._samples_mm.flush()
        if self._lps_mm is not None:
            self._lps_mm.flush()
        self._samples_mm = None
        self._lps_mm = None


# ────────────────────────── module-private helpers ──────────────────────────


def _read_manifest(stream_path: Path) -> dict:
    p = stream_path / "manifest.json"
    if not p.exists():
        return {"chunks": []}
    return json.loads(p.read_text())


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Use a file handle so np.save does not auto-append ".npy" to the tmp name.
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def read_streaming(stream_path: PathLike) -> dict:
    """Read all completed chunks at `stream_path` and return a result dict.

    Truncates each chunk to its recorded `steps_done_per_chunk` so partial
    final chunks (kill mid-run) are not padded with the zero-init tail.

    Returns a dict with keys: samples, log_probs, n_steps_total,
    n_walkers, n_dim, config_name.
    """
    stream_path = Path(stream_path)
    manifest = _read_manifest(stream_path)
    chunks = manifest.get("chunks", [])
    if not chunks:
        raise FileNotFoundError(f"no chunks in manifest at {stream_path}")
    per_chunk_done = manifest.get("steps_done_per_chunk", {})

    samples_list: List[np.ndarray] = []
    lps_list: List[np.ndarray] = []
    for chunk_name in chunks:
        chunk_dir = stream_path / chunk_name
        # Steps done — fall back to full chunk length if not recorded
        # (e.g. older chunk written without save_state at end)
        s_full = np.load(chunk_dir / "samples.npy", mmap_mode="r")
        lp_full = np.load(chunk_dir / "log_probs.npy", mmap_mode="r")
        n_done = per_chunk_done.get(chunk_name, s_full.shape[0])
        n_done = min(int(n_done), s_full.shape[0])
        samples_list.append(np.asarray(s_full[:n_done]))
        lps_list.append(np.asarray(lp_full[:n_done]))

    samples = np.concatenate(samples_list, axis=0)
    log_probs = np.concatenate(lps_list, axis=0)
    return {
        "samples": samples,
        "log_probs": log_probs,
        "n_steps_total": samples.shape[0],
        "n_walkers": int(manifest["n_walkers"]),
        "n_dim": int(manifest["n_dim"]),
        "config_name": str(manifest.get("config_name", "")),
    }


def resume_streaming(
    stream_path: PathLike,
    log_prob_fn,
    n_more_steps: int,
    key=None,
    verbose: bool = True,
    target_ess: Optional[float] = None,
    config=None,
) -> dict:
    """Resume a streamed run for `n_more_steps` more production steps.

    Reads sampler state from `<stream_path>/state/`, calls `run_variant`
    with `n_warmup=0`, supplied `mu` / `z_initial` / `init_positions` /
    `init_log_probs`, and `stream_path=stream_path` so the new samples
    write to a new chunk. The Markov chain continues; auto-tuning is off.

    Parameters
    ----------
    stream_path : str or Path
        The directory written by a previous streaming run.
    log_prob_fn : callable
        Same log-prob function used in the original run.
    n_more_steps : int
        Number of additional production steps.
    key : jax.random.PRNGKey, optional
        PRNG key for the resumed run. Defaults to PRNGKey(0).
    verbose : bool
    target_ess : float, optional
        Early stop threshold passed to run_variant.
    config : VariantConfig, optional
        Override the variant. If None, the variant is looked up in
        dezess.benchmark.registry.VARIANTS using manifest["config_name"].

    Returns
    -------
    dict — the resumed `run_variant` result.
    """
    import jax
    import jax.numpy as jnp
    from dezess.core.loop import run_variant

    stream_path = Path(stream_path)
    state = stream_path / "state"
    if not (state / "last_positions.npy").exists():
        raise FileNotFoundError(f"no state found at {state}")

    manifest = _read_manifest(stream_path)
    if config is None:
        from dezess.benchmark.registry import VARIANTS
        config_name = str(manifest.get("config_name", ""))
        if config_name not in VARIANTS:
            raise ValueError(
                f"resume_streaming: config '{config_name}' not in registry. "
                "Pass `config=...` to override."
            )
        config = VARIANTS[config_name]

    init = np.load(state / "last_positions.npy")
    init_lps = np.load(state / "last_lps.npy")
    z_matrix = np.load(state / "z_matrix.npy")
    mu_val = float(np.load(state / "mu.npy"))

    if verbose:
        per_chunk = manifest.get("steps_done_per_chunk", {})
        n_done_total = sum(per_chunk.values())
        print(f"dezess.resume_streaming: continuing from {n_done_total} steps "
              f"(latest chunk: {manifest['chunks'][-1] if manifest.get('chunks') else 'n/a'}, "
              f"mu={mu_val:.4f})", flush=True)

    if key is None:
        key = jax.random.PRNGKey(0)

    return run_variant(
        log_prob_fn,
        jnp.asarray(init, dtype=jnp.float64),
        n_steps=int(n_more_steps),
        n_warmup=0,
        config=config,
        key=key,
        mu=mu_val,
        tune=False,
        z_initial=jnp.asarray(z_matrix, dtype=jnp.float64),
        init_log_probs=jnp.asarray(init_lps, dtype=jnp.float64),
        target_ess=target_ess,
        verbose=verbose,
        stream_path=str(stream_path),
    )
