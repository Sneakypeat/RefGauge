from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd

SIM_DIR = Path(__file__).parents[1] / "validation" / "simulations"
sys.path.insert(0, str(SIM_DIR))
SPEC = importlib.util.spec_from_file_location(
    "aggregate_transport_stress_results",
    SIM_DIR / "aggregate_transport_stress_results.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(method: str, selection: str, cell_index: int) -> dict[str, object]:
    prohibited = selection == "target_informed_prohibited"
    return {
        "cell_index": cell_index,
        "method": method,
        "selection": selection,
        "p": 100,
        "target_bias": 0.0,
        "target_scale": 1.0,
        "truth": "null",
        "block_rho": 0.0,
        "k": 6,
        "missingness": "mcar_0",
        "latent_coverage": 0.95,
        "observed_coverage": 0.96,
        "latent_mean_width": 1.0,
        "observed_mean_width": 1.2,
        "pi_failure_rate": 0.0,
        "shared_point_method": "dl_selection_diagnostic",
        "features_selection_eligible_mean": 100.0,
        "features_selected_mean": 20.0,
        "features_selected_target_evaluable_mean": 20.0,
        "shared_transport_pearson_mean": 0.8 if prohibited else 0.2,
        "shared_transport_spearman_mean": 0.7 if prohibited else 0.1,
        "shared_transport_bias_mean": 0.0,
        "shared_transport_rmse_mean": 0.2,
        "shared_sign_error_rate_mean": 0.1,
        "shared_eiv_intercept_mean": 0.0,
        "shared_eiv_slope_mean": 1.0,
        "pi_features_attempted": 12,
        "pi_features_successful": 12,
        "pi_features_failed": 0,
        "features_training_k_failed_mean": 0.0,
        "features_target_missing_mean": 0.0,
    }


def test_prohibited_selection_pairs_across_distinct_cell_indices() -> None:
    frame = pd.DataFrame(
        [
            _row(method, selection, cell_index)
            for selection, cell_index in (
                ("training_only", 0),
                ("target_informed_prohibited", 1),
            )
            for method in ("nnf_cd", "reml_mkh_approx")
        ]
    )

    _, _, controls, _ = MODULE._summaries(frame)

    assert len(controls) == 1
    assert controls.loc[0, "cell_index_training_only"] == 0
    assert controls.loc[0, "cell_index_target_informed_prohibited"] == 1
    assert math.isclose(
        controls.loc[0, "delta_shared_transport_pearson_mean_prohibited_minus_training"],
        0.6,
    )


def test_training_only_alias_grid_allows_empty_control_table() -> None:
    frame = pd.DataFrame(
        [_row(method, "training_only", 0) for method in ("nnf_cd", "reml_mkh_approx")]
    )

    _, _, controls, _ = MODULE._summaries(frame, require_control_pairs=False)

    assert controls.empty
    assert "delta_shared_transport_pearson_mean_prohibited_minus_training" in controls
