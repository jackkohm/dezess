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
    manifest = json.loads((tmp_stream / "manifest.json").read_text())
    assert manifest["steps_done_per_chunk"]["chunk_001"] == 50

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
