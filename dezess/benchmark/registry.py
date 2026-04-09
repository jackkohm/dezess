"""Registry of named variant configurations for benchmarking.

Each variant is a VariantConfig that specifies which strategies to compose.
The registry also defines standard target sets for benchmarking.
"""

from __future__ import annotations

from dezess.core.types import VariantConfig

# --- Variant Configurations ---

VARIANTS = {
    # Baseline: original DE-MCz slice sampler
    "baseline": VariantConfig(
        name="baseline",
        direction="de_mcz",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
    ),

    # Width variants
    "stochastic_width": VariantConfig(
        name="stochastic_width",
        direction="de_mcz",
        width="stochastic",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"sigma": 0.5},
    ),
    "stochastic_width_wide": VariantConfig(
        name="stochastic_width_wide",
        direction="de_mcz",
        width="stochastic",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"sigma": 1.0},
    ),
    "per_direction_width": VariantConfig(
        name="per_direction_width",
        direction="de_mcz",
        width="per_direction",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"ema_alpha": 0.1},
    ),

    # Direction variants
    "snooker": VariantConfig(
        name="snooker",
        direction="snooker",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
    ),
    "weighted_pair": VariantConfig(
        name="weighted_pair",
        direction="weighted_pair",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"temperature": 1.0},
    ),
    "weighted_pair_cold": VariantConfig(
        name="weighted_pair_cold",
        direction="weighted_pair",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"temperature": 0.5},
    ),
    "momentum_03": VariantConfig(
        name="momentum_03",
        direction="momentum",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"alpha": 0.3},
    ),
    "momentum_05": VariantConfig(
        name="momentum_05",
        direction="momentum",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"alpha": 0.5},
    ),
    "pca_directions": VariantConfig(
        name="pca_directions",
        direction="pca",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"pca_mix": 0.5},
    ),
    "riemannian": VariantConfig(
        name="riemannian",
        direction="riemannian",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"k_neighbors": 30, "riemannian_mix": 0.5},
    ),
    "flow_directions": VariantConfig(
        name="flow_directions",
        direction="flow",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"flow_mix": 0.5, "flow_epochs": 200, "flow_layers": 3},
    ),

    # Slice variants
    "adaptive_budget": VariantConfig(
        name="adaptive_budget",
        direction="de_mcz",
        width="scalar",
        slice_fn="adaptive_budget",
        zmatrix="circular",
        ensemble="standard",
    ),
    "delayed_rejection": VariantConfig(
        name="delayed_rejection",
        direction="de_mcz",
        width="scalar",
        slice_fn="delayed_rejection",
        zmatrix="circular",
        ensemble="standard",
        slice_kwargs={"retry_mu_factor": 0.3},
    ),

    # Ensemble variants
    "parallel_tempering": VariantConfig(
        name="parallel_tempering",
        direction="de_mcz",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="parallel_tempering",
        ensemble_kwargs={"n_temps": 4, "t_max": 10.0},
    ),

    # Combined variants (best ideas together)
    "snooker_stochastic": VariantConfig(
        name="snooker_stochastic",
        direction="snooker",
        width="stochastic",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"sigma": 0.5},
    ),
    "momentum_delayed": VariantConfig(
        name="momentum_delayed",
        direction="momentum",
        width="scalar",
        slice_fn="delayed_rejection",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"alpha": 0.3},
        slice_kwargs={"retry_mu_factor": 0.3},
    ),
    "pca_per_direction": VariantConfig(
        name="pca_per_direction",
        direction="pca",
        width="per_direction",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"pca_mix": 0.5},
        width_kwargs={"ema_alpha": 0.1},
    ),
    "full_kitchen_sink": VariantConfig(
        name="full_kitchen_sink",
        direction="snooker",
        width="stochastic",
        slice_fn="delayed_rejection",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"sigma": 0.5},
        slice_kwargs={"retry_mu_factor": 0.3},
    ),

    # --- New experimental variants ---
    "scale_aware": VariantConfig(
        name="scale_aware",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
    ),
    "zeus_gamma": VariantConfig(
        name="zeus_gamma",
        direction="de_mcz",
        width="zeus_gamma",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"sigma": 1.0},
    ),
    "whitened": VariantConfig(
        name="whitened",
        direction="whitened",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
    ),
    "whitened_stochastic": VariantConfig(
        name="whitened_stochastic",
        direction="whitened",
        width="stochastic",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"sigma": 0.5},
    ),
    "esjd_tuned": VariantConfig(
        name="esjd_tuned",
        direction="de_mcz",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        tune_method="esjd",
    ),
    "dual_avg": VariantConfig(
        name="dual_avg",
        direction="de_mcz",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        tune_method="dual_avg",
    ),
    "dual_avg_scale": VariantConfig(
        name="dual_avg_scale",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        tune_method="dual_avg",
        width_kwargs={"scale_factor": 1.0},
    ),
    "multi_direction_2": VariantConfig(
        name="multi_direction_2",
        direction="de_mcz",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        n_slices_per_step=2,
    ),
    "local_pair_scale": VariantConfig(
        name="local_pair_scale",
        direction="local_pair",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        # local_mix capped at 0.3: higher values cause variance contraction
        # (same state-dependent bias as snooker/gradient)
        direction_kwargs={"n_candidates": 10, "local_mix": 0.3},
        width_kwargs={"scale_factor": 1.0},
    ),

    "momentum_scale": VariantConfig(
        name="momentum_scale",
        direction="momentum",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"alpha": 0.3},
        width_kwargs={"scale_factor": 1.0},
    ),
    "pca_scale": VariantConfig(
        name="pca_scale",
        direction="pca",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        direction_kwargs={"pca_mix": 0.5},
        width_kwargs={"scale_factor": 1.0},
    ),

    "scale_aware_lean": VariantConfig(
        name="scale_aware_lean",
        direction="de_mcz",
        width="scale_aware",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        width_kwargs={"scale_factor": 1.0},
        slice_kwargs={"n_expand": 1, "n_shrink": 3},
    ),

    # NOTE: coordinate directions tested but underperform DE-MCz on
    # correlated targets because axis-aligned slices don't follow the
    # posterior geometry. Available as direction="coordinate" but not
    # recommended as default.

    # NOTE: gradient directions are experimental and NOT recommended.
    # The gradient d = normalize(grad(log_prob)(x)) is state-dependent and
    # radially biased, causing variance contraction (same issue as uncorrected
    # snooker). See dezess/directions/gradient.py for details.

    "multi_direction_3": VariantConfig(
        name="multi_direction_3",
        direction="de_mcz",
        width="scalar",
        slice_fn="fixed",
        zmatrix="circular",
        ensemble="standard",
        n_slices_per_step=3,
    ),
}


# --- Target sets for benchmarking ---

TARGET_SETS = {
    "quick": ["isotropic_10", "correlated_10"],
    "standard": ["isotropic_10", "correlated_10", "student_t_10", "mixture_5"],
    "full": [
        "isotropic_10", "correlated_10", "student_t_10", "mixture_5",
        "funnel_10", "rosenbrock_4", "ill_conditioned_10", "ring_2",
    ],
    "hard": ["funnel_10", "rosenbrock_4", "ill_conditioned_10", "mixture_5"],
    "high_dim": ["isotropic_30", "isotropic_60", "correlated_30", "correlated_60",
                  "ill_conditioned_30", "ill_conditioned_60"],
}


# --- Variant sets ---

VARIANT_SETS = {
    "width_variants": ["baseline", "stochastic_width", "stochastic_width_wide", "per_direction_width"],
    "direction_variants": ["baseline", "snooker", "weighted_pair", "momentum_03", "momentum_05", "pca_directions"],
    "slice_variants": ["baseline", "adaptive_budget", "delayed_rejection"],
    "ensemble_variants": ["baseline", "parallel_tempering"],
    "combined_variants": ["baseline", "snooker_stochastic", "momentum_delayed", "pca_per_direction", "full_kitchen_sink"],
    "all": list(VARIANTS.keys()),
}
