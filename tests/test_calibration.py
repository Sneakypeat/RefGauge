import numpy as np

from contrastcal.calibration import calibrate_loso, calibration_summary
from contrastcal.simulation import simulate_dataset


def test_loso_excludes_target_and_returns_each_feature_study():
    effects, _ = simulate_dataset(n_features=12, n_studies=5, seed=11)
    result = calibrate_loso(effects)
    assert len(result) == 60
    assert set(result["n_training_studies"]) == {4}
    assert set(result["interval_distribution"]) == {"t(df=2)"}
    assert result[["feature", "target_study", "stratum"]].duplicated().sum() == 0
    assert np.isfinite(result["standardized_discrepancy"]).all()


def test_prediction_interval_contains_pooled_like_direct_effects():
    effects, _ = simulate_dataset(n_features=100, n_studies=7, seed=21)
    result = calibrate_loso(effects)
    summary = calibration_summary(result)
    overall = summary.iloc[0]
    assert overall["scope"] == "overall"
    assert 0 <= overall["coverage"] <= 1
    assert overall["pearson"] > 0.5
    assert overall["calibration_slope"] > 0


def test_validation_rejects_only_two_studies():
    effects, _ = simulate_dataset(n_features=5, n_studies=3, seed=1)
    two = effects.loc[effects["study"].isin(["study_1", "study_2"])]
    try:
        calibrate_loso(two)
    except ValueError as exc:
        assert "three studies" in str(exc)
    else:
        raise AssertionError("two-study input did not fail")


def test_three_studies_records_two_training_study_fallback():
    effects, _ = simulate_dataset(n_features=4, n_studies=3, seed=2)
    result = calibrate_loso(effects)
    assert set(result["interval_distribution"]) == {"normal_fallback_k2"}
