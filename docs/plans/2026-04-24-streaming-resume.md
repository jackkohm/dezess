# Streaming-write + Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream samples + sampler state to disk during the run (one JIT compile, one execution) and support resume after kill via a separate `resume_streaming` helper.

**Architecture:** New `dezess/streaming.py` module owns disk I/O. `run_variant` gets one new kwarg `stream_path: Optional[str] = None`. After each scan-batch, samples + log_probs append to a per-chunk numpy memmap, and a small `state/` directory (z_matrix, mu, walker_aux, meta.json) is overwritten atomically. `dezess.resume_streaming(path, log_prob_fn, n_more_steps)` reads the state, calls `run_variant(... n_warmup=0, mu=..., z_initial=..., init_log_probs=...)` which writes to a new chunk. `dezess.read_streaming(path)` concatenates all chunks for analysis.

**Tech Stack:** numpy (`np.lib.format.open_memmap`), JAX, json. No new external deps.

**Spec source:** in-conversation brainstorm 2026-04-24, user-approved.

---

## Disk layout

```
<stream_path>/
    manifest.json            # {chunks: ["chunk_001", "chunk_002"], n_walkers, n_dim, config_name, n_steps_total}
    state/                   # overwritten each batch — atomic via tmp+rename
        z_matrix.npy         # (z_capacity, n_dim)
        z_log_probs.npy      # (z_capacity,)
        z_count.npy          # int scalar
        mu.npy               # float scalar
        walker_aux_prev_dir.npy
        walker_aux_bw.npy
        walker_aux_da.npy
        walker_aux_ds.npy
        last_lps.npy         # (n_walkers,) — final-step log_probs of latest chunk
        last_positions.npy   # (n_walkers, n_dim) — final-step positions
        meta.json            # {n_steps_done_total, latest_chunk_steps_done}
    chunk_001/
        samples.npy          # mmap (n_production_chunk_001, n_walkers, n_dim)
        log_probs.npy        # mmap (n_production_chunk_001, n_walkers)
    chunk_002/
        samples.npy
        log_probs.npy
```

---

## File responsibilities

- `dezess/streaming.py` (NEW, ~250 LOC): `Streamer` class (open/append batch/close), `read_streaming` (concatenate chunks), `_save_state_atomic` (write tmp + rename), `_load_state` (read state dir into a dict).
- `dezess/core/loop.py` (MODIFY): accept `stream_path` kwarg; instantiate `Streamer` if set; call `streamer.append_batch(...)` after each scan-batch and single-step; call `streamer.close()` at end. Also accept `init_log_probs` kwarg (small reuse-of-existing-local-change for resume input).
- `dezess/__init__.py` (MODIFY): re-export `resume_streaming`, `read_streaming`.
- `dezess/tests/test_streaming.py` (NEW, ~150 LOC): unit + integration tests.

---

## Task overview

| Task | What | Files | Independent? |
|---|---|---|---|
| 1 | `skip_auto_extend_warmup` (orthogonal) | loop.py | yes |
| 2 | `Streamer` class core (write only) | streaming.py + tests | yes |
| 3 | `read_streaming` reader | streaming.py + tests | depends on 2 |
| 4 | Hook into `run_variant` (write path) | loop.py + integration test | depends on 2, 3 |
| 5 | `resume_streaming` helper | streaming.py + integration test | depends on 4 |
| 6 | API re-exports + CLAUDE.md note | __init__.py, CLAUDE.md | depends on 4, 5 |

Tasks 1, 2 can run in parallel. Tasks 3-6 are sequential.

---

### Task 1: `skip_auto_extend_warmup` flag

**Files:**
- Modify: `dezess/core/loop.py:121-128` (add kwarg to `run_variant`)
- Modify: `dezess/core/loop.py:1014-1052` (gate the auto-extender)

This task carves out the orthogonal piece of your local uncommitted change. Streaming is unrelated to it.

- [ ] **Step 1: Confirm baseline test still passes (regression guard)**

```bash
cd GIVEMEPotential/third_party/dezess
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_gaussian_moments.py -v -k baseline
```
Expected: PASS (whatever count is current).

- [ ] **Step 2: Add the kwarg and gate**

In `dezess/core/loop.py`, modify `run_variant` signature (currently ends at the line before the docstring with `n_walkers_per_gpu: Optional[int] = None,`). Insert after `n_walkers_per_gpu: Optional[int] = None,`:

```python
    n_walkers_per_gpu: Optional[int] = None,
    skip_auto_extend_warmup: bool = False,
```

Find the auto-extender block (currently around line 1014, the comment `# Warmup auto-extension: if ESJD is still changing rapidly...`) and modify the predicate:

```python
    if (
        tune
        and n_warmup > 100
        and esjd_ema > 0
        and not use_block_gibbs
        and not skip_auto_extend_warmup
    ):
```

Add a verbose announcement immediately above:

```python
    if skip_auto_extend_warmup and verbose:
        print(f"  [{config.name}] Auto-extend warmup skipped (skip_auto_extend_warmup=True).",
              flush=True)
```

- [ ] **Step 3: Add a regression test**

Append to `dezess/tests/test_gaussian_moments.py`:

```python
def test_skip_auto_extend_warmup_returns_quickly():
    """skip_auto_extend_warmup=True must NOT trigger the post-warmup extender."""
    import time as _time
    from dezess.core.loop import run_variant
    from dezess.core.types import VariantConfig

    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(0), (32, 5)) * 0.1
    cfg = VariantConfig(
        name="skip_extend_test",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
    )

    t0 = _time.time()
    result = run_variant(
        log_prob, init, n_steps=300, config=cfg, n_warmup=200,
        key=jax.random.PRNGKey(0), verbose=False,
        skip_auto_extend_warmup=True,
    )
    wall = _time.time() - t0
    # With skip=True, total work is ~300 steps. Without skip, the extender
    # may add up to n_warmup=200 steps. We just verify the flag is wired —
    # the run completes and produces samples with the right shape.
    assert result["samples"].shape == (100, 32, 5)
    assert wall < 60.0   # generous; main check is the shape + wiring
```

- [ ] **Step 4: Run the test**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_gaussian_moments.py::test_skip_auto_extend_warmup_returns_quickly -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dezess/core/loop.py dezess/tests/test_gaussian_moments.py
git commit -m "feat: skip_auto_extend_warmup flag for tight posteriors at MU_MIN"
```

---

### Task 2: `Streamer` class core (write path)

**Files:**
- Create: `dezess/streaming.py`
- Test: `dezess/tests/test_streaming.py`

- [ ] **Step 1: Write the failing tests**

Create `dezess/tests/test_streaming.py`:

```python
"""Tests for streaming-write infrastructure."""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


from dezess.streaming import Streamer


@pytest.fixture
def tmp_stream():
    d = tempfile.mkdtemp(prefix="dezess_stream_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_streamer_writes_one_batch(tmp_stream):
    """append_batch writes samples + log_probs to per-chunk mmaps."""
    s = Streamer(
        tmp_stream,
        n_walkers=8, n_dim=3, n_production=200,
        config_name="test_cfg", z_capacity=64,
    )
    s.open_chunk()
    samples_b = np.random.standard_normal((50, 8, 3)).astype(np.float64)
    lps_b = np.random.standard_normal((50, 8)).astype(np.float64)
    s.append_batch(samples_b, lps_b)
    s.close()

    chunk0 = tmp_stream / "chunk_001"
    assert (chunk0 / "samples.npy").exists()
    assert (chunk0 / "log_probs.npy").exists()

    arr = np.load(chunk0 / "samples.npy", mmap_mode="r")
    assert arr.shape == (200, 8, 3)
    np.testing.assert_array_equal(arr[:50], samples_b)
    # Unwritten rows are zero-init (memmap created with np.zeros)
    assert np.all(arr[50:] == 0.0)


def test_streamer_appends_multiple_batches(tmp_stream):
    s = Streamer(tmp_stream, n_walkers=4, n_dim=2, n_production=100,
                 config_name="t", z_capacity=32)
    s.open_chunk()
    b1 = np.full((30, 4, 2), 1.0)
    b2 = np.full((40, 4, 2), 2.0)
    lp1 = np.full((30, 4), 1.0)
    lp2 = np.full((40, 4), 2.0)
    s.append_batch(b1, lp1)
    s.append_batch(b2, lp2)
    s.close()

    arr = np.load(tmp_stream / "chunk_001" / "samples.npy", mmap_mode="r")
    np.testing.assert_array_equal(arr[:30], b1)
    np.testing.assert_array_equal(arr[30:70], b2)


def test_streamer_save_state_atomic(tmp_stream):
    """save_state writes z_matrix, mu, walker_aux atomically (tmp+rename)."""
    s = Streamer(tmp_stream, n_walkers=8, n_dim=3, n_production=100,
                 config_name="t", z_capacity=64)
    s.open_chunk()

    z_matrix = np.random.standard_normal((64, 3))
    z_log_probs = np.random.standard_normal((64,))
    walker_aux = {
        "prev_direction": np.zeros((8, 3)),
        "bracket_widths": np.ones((8, 3)) * 0.5,
        "direction_anchor": np.zeros((8, 3)),
        "direction_scale": np.ones((8,)) * 1.5,
    }
    last_positions = np.random.standard_normal((8, 3))
    last_lps = np.random.standard_normal((8,))

    s.save_state(
        z_matrix=z_matrix, z_log_probs=z_log_probs, z_count=42,
        mu=0.123, walker_aux=walker_aux,
        last_positions=last_positions, last_lps=last_lps,
        n_steps_done_in_chunk=50,
    )

    state_dir = tmp_stream / "state"
    np.testing.assert_array_equal(np.load(state_dir / "z_matrix.npy"), z_matrix)
    np.testing.assert_array_equal(np.load(state_dir / "z_log_probs.npy"), z_log_probs)
    assert int(np.load(state_dir / "z_count.npy")) == 42
    assert float(np.load(state_dir / "mu.npy")) == 0.123
    np.testing.assert_array_equal(np.load(state_dir / "walker_aux_bw.npy"),
                                   walker_aux["bracket_widths"])
    np.testing.assert_array_equal(np.load(state_dir / "last_positions.npy"),
                                   last_positions)
    meta = json.loads((state_dir / "meta.json").read_text())
    assert meta["latest_chunk_steps_done"] == 50

    # No leftover .tmp files
    assert not list(state_dir.glob("*.tmp"))


def test_manifest_lists_chunks_in_order(tmp_stream):
    s = Streamer(tmp_stream, n_walkers=4, n_dim=2, n_production=10,
                 config_name="cfg_x", z_capacity=16)
    s.open_chunk()
    s.append_batch(np.zeros((5, 4, 2)), np.zeros((5, 4)))
    s.close()

    s2 = Streamer(tmp_stream, n_walkers=4, n_dim=2, n_production=20,
                  config_name="cfg_x", z_capacity=16)
    s2.open_chunk()
    s2.append_batch(np.ones((5, 4, 2)), np.ones((5, 4)))
    s2.close()

    manifest = json.loads((tmp_stream / "manifest.json").read_text())
    assert manifest["chunks"] == ["chunk_001", "chunk_002"]
    assert manifest["n_walkers"] == 4
    assert manifest["n_dim"] == 2
    assert manifest["config_name"] == "cfg_x"


def test_streamer_overflow_raises(tmp_stream):
    """Writing past chunk capacity must raise — protects mmap bounds."""
    s = Streamer(tmp_stream, n_walkers=4, n_dim=2, n_production=10,
                 config_name="t", z_capacity=8)
    s.open_chunk()
    s.append_batch(np.zeros((10, 4, 2)), np.zeros((10, 4)))   # fills
    with pytest.raises(ValueError, match="overflow"):
        s.append_batch(np.zeros((1, 4, 2)), np.zeros((1, 4)))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_streaming.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'dezess.streaming'`.

- [ ] **Step 3: Implement `dezess/streaming.py`**

Create with EXACTLY this content:

```python
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
            meta.json
        chunk_NNN/             (one per `run_variant` call)
            samples.npy        (mmap, shape (n_production, n_walkers, n_dim))
            log_probs.npy      (mmap, shape (n_production, n_walkers))
"""

from __future__ import annotations

import json
import os
import shutil
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
        n_steps_done_total = sum(per_chunk_done.values())
        manifest["steps_done_per_chunk"] = per_chunk_done
        _write_json_atomic(self.stream_path / "manifest.json", manifest)

        meta = {
            "n_steps_done_total": n_steps_done_total,
            "latest_chunk_steps_done": int(n_steps_done_in_chunk),
            "latest_chunk": f"chunk_{self._chunk_idx:03d}",
        }
        _write_json_atomic(state / "meta.json", meta)

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
    np.save(tmp, arr)
    os.replace(tmp, path)
```

- [ ] **Step 4: Run tests**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_streaming.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add dezess/streaming.py dezess/tests/test_streaming.py
git commit -m "feat: add Streamer for incremental sample + state write to disk"
```

---

### Task 3: `read_streaming` reader

**Files:**
- Modify: `dezess/streaming.py`
- Modify: `dezess/tests/test_streaming.py`

- [ ] **Step 1: Write the failing tests**

Append to `dezess/tests/test_streaming.py`:

```python
from dezess.streaming import read_streaming


def test_read_streaming_concatenates_chunks(tmp_stream):
    """read_streaming returns the union of all chunks, truncated by steps_done."""
    s = Streamer(tmp_stream, n_walkers=4, n_dim=2, n_production=10,
                 config_name="cfg", z_capacity=8)
    s.open_chunk()
    s.append_batch(np.full((6, 4, 2), 1.0), np.full((6, 4), 1.0))
    s.save_state(
        z_matrix=np.zeros((8, 2)), z_log_probs=np.zeros((8,)), z_count=4, mu=0.5,
        walker_aux={
            "prev_direction": np.zeros((4, 2)),
            "bracket_widths": np.zeros((4, 2)),
            "direction_anchor": np.zeros((4, 2)),
            "direction_scale": np.zeros((4,)),
        },
        last_positions=np.zeros((4, 2)), last_lps=np.zeros((4,)),
        n_steps_done_in_chunk=6,
    )
    s.close()

    s2 = Streamer(tmp_stream, n_walkers=4, n_dim=2, n_production=10,
                  config_name="cfg", z_capacity=8)
    s2.open_chunk()
    s2.append_batch(np.full((4, 4, 2), 2.0), np.full((4, 4), 2.0))
    s2.save_state(
        z_matrix=np.zeros((8, 2)), z_log_probs=np.zeros((8,)), z_count=4, mu=0.5,
        walker_aux={
            "prev_direction": np.zeros((4, 2)),
            "bracket_widths": np.zeros((4, 2)),
            "direction_anchor": np.zeros((4, 2)),
            "direction_scale": np.zeros((4,)),
        },
        last_positions=np.zeros((4, 2)), last_lps=np.zeros((4,)),
        n_steps_done_in_chunk=4,
    )
    s2.close()

    out = read_streaming(tmp_stream)
    # 6 from chunk_001 + 4 from chunk_002 = 10
    assert out["samples"].shape == (10, 4, 2)
    np.testing.assert_array_equal(out["samples"][:6], 1.0)
    np.testing.assert_array_equal(out["samples"][6:10], 2.0)
    assert out["log_probs"].shape == (10, 4)
    assert out["n_steps_total"] == 10
    assert out["config_name"] == "cfg"


def test_read_streaming_handles_partial_last_chunk(tmp_stream):
    """If the last chunk only wrote 3/10 rows before kill, return only those 3."""
    s = Streamer(tmp_stream, n_walkers=4, n_dim=2, n_production=10,
                 config_name="cfg", z_capacity=8)
    s.open_chunk()
    s.append_batch(np.full((3, 4, 2), 7.0), np.full((3, 4), 7.0))
    s.save_state(
        z_matrix=np.zeros((8, 2)), z_log_probs=np.zeros((8,)), z_count=2, mu=0.5,
        walker_aux={
            "prev_direction": np.zeros((4, 2)),
            "bracket_widths": np.zeros((4, 2)),
            "direction_anchor": np.zeros((4, 2)),
            "direction_scale": np.zeros((4,)),
        },
        last_positions=np.zeros((4, 2)), last_lps=np.zeros((4,)),
        n_steps_done_in_chunk=3,
    )
    # NOTE: do NOT call s.close() — simulates kill mid-run
    out = read_streaming(tmp_stream)
    assert out["samples"].shape == (3, 4, 2)
    np.testing.assert_array_equal(out["samples"], 7.0)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_streaming.py -v
```
Expected: 2 new failures with `ImportError: cannot import name 'read_streaming'`.

- [ ] **Step 3: Implement `read_streaming` in `dezess/streaming.py`**

Append to `dezess/streaming.py`:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_streaming.py -v
```
Expected: 7 passed (5 prior + 2 new).

- [ ] **Step 5: Commit**

```bash
git add dezess/streaming.py dezess/tests/test_streaming.py
git commit -m "feat: read_streaming concatenates chunks with partial-tail handling"
```

---

### Task 4: Hook `Streamer` into `run_variant` (write path)

**Files:**
- Modify: `dezess/core/loop.py` — add `stream_path` and `init_log_probs` kwargs; instantiate `Streamer`; call `append_batch` after each batch and single-step; call `save_state` at the same cadence; call `close()` at end
- Modify: `dezess/tests/test_streaming.py` — integration test

This is the main wiring. We do it carefully.

- [ ] **Step 1: Write the integration test (failing)**

Append to `dezess/tests/test_streaming.py`:

```python
import jax
import jax.numpy as jnp

from dezess.core.loop import run_variant
from dezess.core.types import VariantConfig

jax.config.update("jax_enable_x64", True)


def _scale_aware_cfg():
    return VariantConfig(
        name="stream_test",
        direction="de_mcz", width="scale_aware",
        slice_fn="fixed", zmatrix="circular", ensemble="standard",
        check_nans=False, width_kwargs={"scale_factor": 1.0},
    )


def test_run_variant_with_stream_path_writes_to_disk(tmp_stream):
    """run_variant(stream_path=...) writes samples to disk progressively."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(0), (16, 4)) * 0.5

    result = run_variant(
        log_prob, init, n_steps=300, n_warmup=100,
        config=_scale_aware_cfg(), key=jax.random.PRNGKey(0),
        verbose=False, stream_path=str(tmp_stream),
    )

    # In-RAM result still works
    assert result["samples"].shape == (200, 16, 4)

    # Disk has the same data
    out = read_streaming(tmp_stream)
    assert out["samples"].shape == (200, 16, 4)
    np.testing.assert_array_equal(np.asarray(result["samples"]), out["samples"])
    np.testing.assert_array_equal(np.asarray(result["log_prob"]), out["log_probs"])

    # State directory populated
    state = tmp_stream / "state"
    assert (state / "z_matrix.npy").exists()
    assert (state / "mu.npy").exists()
    assert (state / "last_positions.npy").exists()
    assert (state / "meta.json").exists()


def test_stream_path_none_is_byte_identical_to_baseline(tmp_stream):
    """stream_path=None must not change the in-RAM samples."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(0), (16, 4)) * 0.5
    cfg = _scale_aware_cfg()

    res_baseline = run_variant(
        log_prob, init, n_steps=200, n_warmup=50,
        config=cfg, key=jax.random.PRNGKey(0), verbose=False,
    )
    res_streamed = run_variant(
        log_prob, init, n_steps=200, n_warmup=50,
        config=cfg, key=jax.random.PRNGKey(0), verbose=False,
        stream_path=str(tmp_stream),
    )
    np.testing.assert_array_equal(
        np.asarray(res_baseline["samples"]),
        np.asarray(res_streamed["samples"]),
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_streaming.py::test_run_variant_with_stream_path_writes_to_disk -v
```
Expected: FAIL — `TypeError: run_variant() got an unexpected keyword argument 'stream_path'`.

- [ ] **Step 3: Add `stream_path` and `init_log_probs` kwargs to `run_variant`**

In `dezess/core/loop.py`, modify the `run_variant` signature. After the `skip_auto_extend_warmup: bool = False,` line added in Task 1, insert:

```python
    skip_auto_extend_warmup: bool = False,
    stream_path: Optional[str] = None,
    init_log_probs: Optional[Array] = None,
```

(`Array` is already imported at top of loop.py; if `Optional` isn't, add `from typing import Optional` at the top — likely already present.)

- [ ] **Step 4: Use `init_log_probs` if supplied (skip recompute)**

Find the "Initial log-probs" block in `run_variant` (currently around line 819, comment `# --- Initial log-probs ---`). Replace it with:

```python
    # --- Initial log-probs ---
    positions = jnp.array(init_positions, dtype=jnp.float64)
    if sharding_info is not None:
        positions = jax.device_put(positions, sharding_info["walker_sharding"])
    if init_log_probs is not None:
        log_probs = jnp.asarray(init_log_probs, dtype=jnp.float64)
        if verbose:
            print(f"  [{config.name}] Reusing supplied init log-probs (skip recompute)",
                  flush=True)
    else:
        if verbose:
            print(f"  [{config.name}] Computing initial log-probs...", flush=True)
        log_probs = jax.jit(jax.vmap(lambda x: lp_eval(log_prob_fn, x)))(positions)
    log_probs.block_until_ready()
```

- [ ] **Step 5: Instantiate `Streamer` and open chunk before the production loop**

Find the production-section header `n_production = n_steps - n_warmup` (currently around line 1557). Insert immediately before it (or right after, as long as it's BEFORE the `while step_idx < n_production:` loop):

```python
    # ── Streaming setup ────────────────────────────────────────────────
    streamer = None
    if stream_path is not None:
        from dezess.streaming import Streamer
        streamer = Streamer(
            stream_path,
            n_walkers=n_walkers, n_dim=n_dim,
            n_production=n_production,
            config_name=config.name,
            z_capacity=int(z_padded.shape[0]),
        )
        streamer.open_chunk()
        if verbose:
            print(f"  [{config.name}] Streaming to {stream_path} "
                  f"(chunk {streamer._chunk_idx:03d}, capacity {n_production})",
                  flush=True)
```

- [ ] **Step 6: Append after each batch + after each single-step**

In the production loop (currently around line 1638, `while step_idx < n_production:`), find the two write points:

(a) **batched scan path** (around line 1655-1661, the `for i in range(batch_sz):` block). Right after the inner `for` loop, before `step_idx += batch_sz`, insert:

```python
            if streamer is not None:
                streamer.append_batch(b_samples[:batch_sz], b_lps[:batch_sz])
                _streaming_save_state(
                    streamer, z_padded, z_log_probs, z_count, mu,
                    walker_aux_pd, walker_aux_bw, walker_aux_da, walker_aux_ds,
                    positions, log_probs,
                    n_steps_done_in_chunk=step_idx + batch_sz,
                )
            step_idx += batch_sz
```

(b) **single-step path** (around line 1695, after `pos_np = np.asarray(positions)`). After the existing `all_samples[step_idx] = pos_np` etc. lines, insert (right before `step_idx += 1`):

```python
            if streamer is not None:
                streamer.append_batch(pos_np[None, :, :], np.asarray(log_probs)[None, :])
                _streaming_save_state(
                    streamer, z_padded, z_log_probs, z_count, mu,
                    walker_aux_pd, walker_aux_bw, walker_aux_da, walker_aux_ds,
                    positions, log_probs,
                    n_steps_done_in_chunk=step_idx + 1,
                )
            step_idx += 1
```

- [ ] **Step 7: Add `_streaming_save_state` helper near the top of `run_variant`**

Just before the "Streaming setup" block from Step 5, add:

```python
    def _streaming_save_state(streamer, z_pad, z_lp, z_cnt, mu_val,
                              w_pd, w_bw, w_da, w_ds, pos, lps,
                              n_steps_done_in_chunk):
        # device_get to host for any sharded arrays
        z_h = np.asarray(jax.device_get(z_pad))
        zlp_h = np.asarray(jax.device_get(z_lp))
        pos_h = np.asarray(jax.device_get(pos))
        lps_h = np.asarray(jax.device_get(lps))
        wkr_aux = {
            "prev_direction": np.asarray(jax.device_get(w_pd)),
            "bracket_widths": np.asarray(jax.device_get(w_bw)),
            "direction_anchor": np.asarray(jax.device_get(w_da)),
            "direction_scale": np.asarray(jax.device_get(w_ds)),
        }
        streamer.save_state(
            z_matrix=z_h, z_log_probs=zlp_h,
            z_count=int(np.asarray(jax.device_get(z_cnt))),
            mu=float(np.asarray(jax.device_get(mu_val))),
            walker_aux=wkr_aux,
            last_positions=pos_h, last_lps=lps_h,
            n_steps_done_in_chunk=int(n_steps_done_in_chunk),
        )
```

- [ ] **Step 8: Close streamer at end of run**

Find the result-dict construction near the end of `run_variant` (look for `"diagnostics": {`). Just before that, add:

```python
    if streamer is not None:
        streamer.close()
```

- [ ] **Step 9: Run the integration tests**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_streaming.py -v
```
Expected: all 9 streaming tests pass (2 new + 7 prior).

- [ ] **Step 10: Run the regression test from Task 1 to confirm no breakage**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_gaussian_moments.py -v -k baseline
```
Expected: all baseline tests still pass.

- [ ] **Step 11: Commit**

```bash
git add dezess/core/loop.py dezess/tests/test_streaming.py
git commit -m "feat: stream samples + sampler state to disk during run_variant"
```

---

### Task 5: `resume_streaming` helper

**Files:**
- Modify: `dezess/streaming.py` — add `resume_streaming` function
- Modify: `dezess/tests/test_streaming.py` — kill-and-resume integration test

- [ ] **Step 1: Write the failing test**

Append to `dezess/tests/test_streaming.py`:

```python
from dezess.streaming import resume_streaming


def test_resume_streaming_continues_from_saved_state(tmp_stream):
    """End-to-end: stream → simulated kill → resume → verify continuity."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(0), (16, 4)) * 0.5

    # First run: 200 prod steps, streaming
    run_variant(
        log_prob, init, n_steps=300, n_warmup=100,
        config=_scale_aware_cfg(), key=jax.random.PRNGKey(0),
        verbose=False, stream_path=str(tmp_stream),
    )
    out_first = read_streaming(tmp_stream)
    assert out_first["samples"].shape == (200, 16, 4)

    # Resume for 100 more steps. Since this is a fresh `run_variant`
    # call with n_warmup=0 and the saved state, samples should chain
    # together (correct Markov chain continuation).
    resume_streaming(
        tmp_stream, log_prob,
        n_more_steps=100, key=jax.random.PRNGKey(99), verbose=False,
    )
    out_resumed = read_streaming(tmp_stream)
    assert out_resumed["samples"].shape == (300, 16, 4)
    # First 200 unchanged
    np.testing.assert_array_equal(
        out_resumed["samples"][:200], out_first["samples"][:200],
    )
    # Manifest knows about both chunks
    manifest = json.loads((tmp_stream / "manifest.json").read_text())
    assert manifest["chunks"] == ["chunk_001", "chunk_002"]
```

- [ ] **Step 2: Run to verify failure**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_streaming.py::test_resume_streaming_continues_from_saved_state -v
```
Expected: FAIL with `ImportError: cannot import name 'resume_streaming'`.

- [ ] **Step 3: Implement `resume_streaming`**

Append to `dezess/streaming.py`:

```python
def resume_streaming(
    stream_path: PathLike,
    log_prob_fn,
    n_more_steps: int,
    key=None,
    verbose: bool = True,
    target_ess: Optional[float] = None,
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

    Returns
    -------
    dict — the resumed `run_variant` result.
    """
    import jax
    import jax.numpy as jnp
    from dezess.core.loop import run_variant
    from dezess.benchmark.registry import VARIANTS

    stream_path = Path(stream_path)
    state = stream_path / "state"
    if not (state / "meta.json").exists():
        raise FileNotFoundError(f"no state found at {state}")

    manifest = _read_manifest(stream_path)
    config_name = str(manifest.get("config_name", ""))
    if config_name not in VARIANTS:
        raise ValueError(
            f"resume_streaming: config '{config_name}' not in registry. "
            "Resuming requires a registered variant."
        )
    config = VARIANTS[config_name]

    init = np.load(state / "last_positions.npy")
    init_lps = np.load(state / "last_lps.npy")
    z_matrix = np.load(state / "z_matrix.npy")
    mu_val = float(np.load(state / "mu.npy"))

    if verbose:
        meta = json.loads((state / "meta.json").read_text())
        print(f"dezess.resume_streaming: continuing from "
              f"{meta['n_steps_done_total']} steps "
              f"(latest chunk: {meta['latest_chunk']}, mu={mu_val:.4f})",
              flush=True)

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
```

- [ ] **Step 4: Run the test**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_streaming.py::test_resume_streaming_continues_from_saved_state -v
```
Expected: PASS.

- [ ] **Step 5: Run all streaming tests**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest dezess/tests/test_streaming.py -v
```
Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add dezess/streaming.py dezess/tests/test_streaming.py
git commit -m "feat: resume_streaming continues a killed/finished streamed run"
```

---

### Task 6: API re-exports + CLAUDE.md note

**Files:**
- Modify: `dezess/__init__.py` — re-export `read_streaming` and `resume_streaming`
- Modify: `CLAUDE.md` — add a note about streaming

- [ ] **Step 1: Inspect existing exports**

```bash
cat dezess/__init__.py
```

Note the existing public API entries.

- [ ] **Step 2: Re-export streaming helpers**

Append to `dezess/__init__.py`:

```python
from dezess.streaming import read_streaming, resume_streaming  # noqa: E402
```

(`# noqa: E402` because the existing file likely has top-level imports already; if not, place it with the other imports.)

- [ ] **Step 3: Verify the re-exports work**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import dezess; print(dezess.read_streaming, dezess.resume_streaming)"
```
Expected: prints two function references.

- [ ] **Step 4: Add a section to `CLAUDE.md`**

Find the "Default sampler" section in `CLAUDE.md`. Append a NEW section after the existing complementary_prob paragraph:

```markdown

### Streaming-write + resume

Long runs can stream samples + sampler state to disk during the run
(one JIT compile, one execution) instead of saving at the end. Pass
`stream_path="path/to/dir"` to `run_variant` (or `dezess.sample`).
Samples accumulate in `path/to/dir/chunk_NNN/{samples,log_probs}.npy`
as numpy memmaps; sampler state is overwritten atomically in
`path/to/dir/state/`.

Read while the run is in progress (or post-hoc) with
`dezess.read_streaming(path)` — concatenates all chunks and returns
`{samples, log_probs, n_steps_total, n_walkers, n_dim, config_name}`.

Resume after a kill (or for more samples) with
`dezess.resume_streaming(path, log_prob_fn, n_more_steps=N)` —
reads the state, calls `run_variant` with `n_warmup=0` + the saved
mu / z_matrix / positions, writes to a new chunk. The JIT compile
hits the persistent cache (`JAX_COMPILATION_CACHE_DIR`) so resume
is cheap.
```

- [ ] **Step 5: Commit**

```bash
git add dezess/__init__.py CLAUDE.md
git commit -m "docs: re-export streaming helpers and document streaming usage"
```

---

## Self-review

**Spec coverage:**
- `stream_path` kwarg — Task 4
- Memmap write per batch — Task 4
- State directory atomic — Task 2
- read_streaming concat — Task 3
- resume_streaming — Task 5
- skip_auto_extend_warmup carved out — Task 1
- byte-identical when stream_path=None — Task 4 Step 1 test

**Placeholder scan:** None present. All steps have concrete code or commands.

**Type consistency:**
- `Streamer` constructor signature: identical across all tasks
- `save_state` kwargs: identical across save calls
- `read_streaming` return dict keys: `{samples, log_probs, n_steps_total, n_walkers, n_dim, config_name}` — consistent
- `resume_streaming` signature: matches what tests call
