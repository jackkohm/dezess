"""Baseline fixed-iteration slice execution (N_EXPAND=3, N_SHRINK=12)."""

from __future__ import annotations

from dezess.core.slice_sample import slice_sample_fixed

# Re-export the core function as the strategy interface
execute = slice_sample_fixed
