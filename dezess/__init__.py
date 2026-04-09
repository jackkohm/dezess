"""dezess: DE-MCz Slice Sampler — fully GPU-parallel ensemble slice sampling."""

from dezess.sampler import run_demcz_slice
from dezess.core.loop import run_variant, DEFAULT_CONFIG
from dezess.core.types import VariantConfig
from dezess.api import sample, SampleResult

__all__ = [
    "sample",
    "SampleResult",
    "run_demcz_slice",
    "run_variant",
    "VariantConfig",
    "DEFAULT_CONFIG",
]
