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
