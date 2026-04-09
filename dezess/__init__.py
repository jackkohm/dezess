"""dezess: DE-MCz Slice Sampler — fully GPU-parallel ensemble slice sampling."""

__version__ = "0.2.0"

from dezess.sampler import run_demcz_slice
from dezess.core.loop import run_variant, DEFAULT_CONFIG
from dezess.core.types import VariantConfig
from dezess.api import sample, SampleResult, init_walkers, diagnose, run_chains
from dezess.utils import (
    LogProbCounter,
    flatten_samples, thin_samples, summary_stats, print_summary,
    autocorrelation, integrated_autocorr_time,
)
from dezess.checkpoint import save_checkpoint, load_checkpoint, resume
from dezess.tuning import recommend_walkers, estimate_n_steps
from dezess.arviz_compat import to_inference_data

__all__ = [
    "sample",
    "SampleResult",
    "init_walkers",
    "diagnose",
    "run_chains",
    "flatten_samples",
    "thin_samples",
    "summary_stats",
    "print_summary",
    "LogProbCounter",
    "autocorrelation",
    "integrated_autocorr_time",
    "save_checkpoint",
    "load_checkpoint",
    "resume",
    "to_inference_data",
    "recommend_walkers",
    "estimate_n_steps",
    "run_demcz_slice",
    "run_variant",
    "VariantConfig",
    "DEFAULT_CONFIG",
]
