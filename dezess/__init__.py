"""dezess: DE-MCz Slice Sampler — fully GPU-parallel ensemble slice sampling."""

from dezess.sampler import run_demcz_slice
from dezess.core.loop import run_variant, DEFAULT_CONFIG
from dezess.core.types import VariantConfig
from dezess.api import sample, SampleResult
from dezess.utils import flatten_samples, thin_samples, summary_stats, print_summary

__all__ = [
    "sample",
    "SampleResult",
    "flatten_samples",
    "thin_samples",
    "summary_stats",
    "print_summary",
    "run_demcz_slice",
    "run_variant",
    "VariantConfig",
    "DEFAULT_CONFIG",
]
