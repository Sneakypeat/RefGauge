import warnings

import numpy as np
import pandas as pd
import pytest

import refgauge.calibration as calibration_module
from refgauge.calibration import (
    _errors_in_variables_calibration,
    calibrate_loso,
    calibration_summary,
)
from refgauge.prediction import NNFComputationError
from refgauge.simulation import simulate_dataset


def test_loso_excludes_target_and_returns_each_feature_study():
    effects, _ = simulate_dataset(n_features=12, n_studies=5, seed=11)
    result = calibrate_loso(effects, interval_method="reml_mkh_approx")
    assert len(result) == 60
    assert set(result["n_training_studies"]) == {4}
    assert set(result["interval_distribution"]) == {"reml_mkh_approx_t(df=2)"}
    assert result[["feature", "target_study", "stratum"]].duplicated().sum() == 0
    assert np.isfinite(result["standardized_discrepancy"]).all()
    assert np.isfinite(result[["cochran_q", "cochran_q_pvalue"]]).all().all()
    assert (result["cochran_q_df"] == result["n_training_studies"] - 1).all()


def test_prediction_interval_contains_pooled_like_direct_effects():
    effects, _ = simulate_dataset(n_features=100, n_studies=7, seed=21)
    result = calibrate_loso(effects, interval_method="reml_mkh_approx")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = calibration_summary(result)
    overall = summary.iloc[0]
    assert overall["scope"] == "overall_macro_study"
    assert 0 <= overall["coverage_success_conditional"] <= 1
    assert (
        overall["coverage_failure_as_miss_lower"]
        <= overall["coverage_success_conditional"]
        <= overall["coverage_failure_as_hit_upper"]
    )
    assert overall["pearson"] > 0.5
    assert overall["eiv_calibration_slope"] > 0
    assert overall["naive_wls_calibration_slope_sensitivity"] > 0


@pytest.mark.parametrize(
    ("interval_method", "point_method"),
    [
        ("fixed_normal", "fixed_inverse_variance"),
        ("dl_normal", "dl_inverse_variance"),
        ("reml_normal", "reml_inverse_variance"),
        ("reml_mkh_approx", "reml_mkh"),
        ("unweighted_t", "unweighted_mean"),
    ],
)
def test_all_approximate_comparators_return_finite_intervals(interval_method, point_method):
    effects, _ = simulate_dataset(n_features=5, n_studies=6, seed=22)
    result = calibrate_loso(effects, interval_method=interval_method)
    assert len(result) == 30
    assert set(result["point_method"]) == {point_method}
    assert result["interval_success"].all()
    assert np.isfinite(result[["pi_true_low", "pi_true_high", "pi_low", "pi_high"]]).all().all()
    assert (result["pi_observed_width"] >= result["pi_true_width"]).all()
    assert set(result["observed_interval_conditioning"]) == {
        "retrospective_known_held_out_target_se"
    }


def test_validation_rejects_only_two_studies():
    effects, _ = simulate_dataset(n_features=5, n_studies=3, seed=1)
    two = effects.loc[effects["study"].isin(["study_1", "study_2"])]
    try:
        calibrate_loso(two)
    except ValueError as exc:
        assert "three studies" in str(exc)
    else:
        raise AssertionError("two-study input did not fail")


def test_three_studies_do_not_receive_a_two_study_normal_fallback():
    effects, _ = simulate_dataset(n_features=4, n_studies=3, seed=2)
    try:
        calibrate_loso(effects, interval_method="reml_mkh_approx")
    except ValueError as exc:
        assert "enough training studies" in str(exc)
    else:
        raise AssertionError("two-study training folds should not receive an inferential interval")


def test_nnf_returns_separate_latent_and_observed_intervals():
    effects, _ = simulate_dataset(n_features=2, n_studies=6, seed=9)
    result = calibrate_loso(effects, interval_method="nnf_cd", nnf_draws=5_000, seed=12)
    assert len(result) == 12
    assert set(result["interval_method"]) == {"nnf_cd"}
    assert (result["pi_observed_width"] >= result["pi_true_width"]).all()
    assert np.isfinite(result["standardized_discrepancy"]).all()
    assert set(result["standardized_discrepancy_method"]) == {"predictive_pit_normal_score"}
    assert result["predictive_pit"].between(0, 1, inclusive="neither").all()
    assert np.isfinite(result["interval_relative_discrepancy"]).all()


def test_nnf_failure_can_be_recorded_without_fallback(monkeypatch):
    effects, _ = simulate_dataset(n_features=1, n_studies=6, seed=19)

    def fail(*args, **kwargs):
        raise NNFComputationError("deliberate test failure")

    monkeypatch.setattr(calibration_module, "nnf_prediction_interval", fail)
    result = calibrate_loso(
        effects,
        interval_method="nnf_cd",
        nnf_failure_policy="record",
    )
    assert len(result) == 6
    assert not result["interval_success"].any()
    assert result["covered"].isna().all()
    assert result["interval_failure_reason"].str.contains("deliberate test failure").all()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = calibration_summary(result)
    assert summary.iloc[0]["interval_failure_rate"] == pytest.approx(1.0)
    assert summary.iloc[0]["coverage_failure_as_miss_lower"] == pytest.approx(0.0)
    assert summary.iloc[0]["coverage_failure_as_hit_upper"] == pytest.approx(1.0)


def test_macro_summary_weights_target_studies_equally():
    effects, _ = simulate_dataset(n_features=3, n_studies=5, seed=24)
    result = calibrate_loso(effects, interval_method="reml_mkh_approx")
    large = result.loc[result["target_study"].eq("study_1")].copy()
    large = large.loc[large.index.repeat(100)].reset_index(drop=True)
    large["covered"] = True
    small = result.loc[result["target_study"].eq("study_2")].copy()
    small["covered"] = False
    summary = calibration_summary(pd.concat([large, small], ignore_index=True))

    macro = summary.loc[summary["scope"].eq("overall_macro_study")].iloc[0]
    micro = summary.loc[summary["scope"].eq("overall_micro_gene_fold")].iloc[0]
    assert macro["coverage_success_conditional"] == pytest.approx(0.5)
    assert micro["coverage_success_conditional"] > 0.98
    assert macro["aggregation_unit"] == "equal_target_study"
    assert macro["n_studies"] == 2


def test_macro_summary_does_not_weight_studies_by_number_of_strata():
    effects, _ = simulate_dataset(n_features=3, n_studies=5, seed=25)
    result = calibrate_loso(effects, interval_method="reml_mkh_approx")
    study_1 = result.loc[result["target_study"].eq("study_1")].copy()
    study_1["covered"] = True
    extra = study_1.copy()
    extra["stratum"] = "extra_stratum"
    study_2 = result.loc[result["target_study"].eq("study_2")].copy()
    study_2["covered"] = False

    combined = pd.concat([study_1, extra, study_2], ignore_index=True)
    summary = calibration_summary(combined)
    macro = summary.loc[summary["scope"].eq("overall_macro_study")].iloc[0]
    assert macro["coverage_success_conditional"] == pytest.approx(0.5)
    assert macro["n_studies"] == 2
    assert (
        macro["n_study_strata"] == combined[["target_study", "stratum"]].drop_duplicates().shape[0]
    )


def test_york_calibration_is_finite_for_heteroscedastic_linear_data():
    x = np.linspace(-2.0, 2.0, 101)
    y = 0.3 + 1.4 * x
    x_variance = np.linspace(0.01, 0.08, len(x))
    y_variance = np.linspace(0.03, 0.12, len(x))[::-1]
    intercept, slope = _errors_in_variables_calibration(x, y, x_variance, y_variance)
    assert intercept == pytest.approx(0.3, abs=1e-10)
    assert slope == pytest.approx(1.4, abs=1e-10)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": np.nan}, "alpha"),
        ({"min_training_studies": 3.5}, "integer"),
        ({"seed": True}, "seed"),
        ({"nnf_draws": 99}, "nnf_draws"),
    ],
)
def test_calibrate_rejects_invalid_global_configuration(kwargs, message):
    effects, _ = simulate_dataset(n_features=2, n_studies=6, seed=33)
    with pytest.raises(ValueError, match=message):
        calibrate_loso(effects, **kwargs)


def test_calibration_summary_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one row"):
        calibration_summary(pd.DataFrame())


@pytest.mark.parametrize("strong_threshold", [-0.1, np.inf, np.nan])
def test_calibration_summary_rejects_invalid_strong_threshold(strong_threshold):
    effects, _ = simulate_dataset(n_features=2, n_studies=4, seed=34)
    predictions = calibrate_loso(effects, interval_method="reml_mkh_approx")
    with pytest.raises(ValueError, match="strong_threshold"):
        calibration_summary(predictions, strong_threshold=strong_threshold)
