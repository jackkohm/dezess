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
    """If 5 rows are written but only 3 marked done before kill, return only 3."""
    s = Streamer(tmp_stream, n_walkers=4, n_dim=2, n_production=10,
                 config_name="cfg", z_capacity=8)
    s.open_chunk()
    # Write 5 rows: first 3 are 7.0, next 2 are SENTINEL 99.0 (must not appear in output)
    samples = np.concatenate([
        np.full((3, 4, 2), 7.0),
        np.full((2, 4, 2), 99.0),
    ], axis=0)
    lps = np.concatenate([
        np.full((3, 4), 7.0),
        np.full((2, 4), 99.0),
    ], axis=0)
    s.append_batch(samples, lps)
    # Mark only first 3 as done — simulates kill 2 rows after the last save_state
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
    # Sentinel must NOT have leaked into the result
    assert not np.any(out["samples"] == 99.0), \
        "truncation failed — rows beyond n_steps_done_in_chunk were returned"


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


from dezess.streaming import resume_streaming


def test_resume_streaming_continues_from_saved_state(tmp_stream):
    """End-to-end: stream → simulated kill → resume → verify continuity."""
    log_prob = jax.jit(lambda x: -0.5 * jnp.sum(x ** 2))
    init = jax.random.normal(jax.random.PRNGKey(0), (16, 4)) * 0.5
    cfg = _scale_aware_cfg()

    # First run
    run_variant(
        log_prob, init, n_steps=300, n_warmup=100,
        config=cfg, key=jax.random.PRNGKey(0),
        verbose=False, stream_path=str(tmp_stream),
    )
    out_first = read_streaming(tmp_stream)
    assert out_first["samples"].shape == (200, 16, 4)

    # Resume for 100 more steps. Pass `config` explicitly because the
    # test variant isn't in the registry.
    resume_streaming(
        tmp_stream, log_prob,
        n_more_steps=100, key=jax.random.PRNGKey(99), verbose=False,
        config=cfg,
    )
    out_resumed = read_streaming(tmp_stream)
    assert out_resumed["samples"].shape == (300, 16, 4)
    np.testing.assert_array_equal(
        out_resumed["samples"][:200], out_first["samples"][:200],
    )
    manifest = json.loads((tmp_stream / "manifest.json").read_text())
    assert manifest["chunks"] == ["chunk_001", "chunk_002"]
