from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from refgauge.prediction import NNFComputationError, nnf_prediction_interval

BASE_Y = np.array([0.18, 0.32, 0.27, 0.41, 0.23])
BASE_SE = np.array([0.10, 0.12, 0.09, 0.14, 0.11])


@pytest.mark.parametrize(
    ("estimate", "se", "message"),
    [
        ([0.1, 0.2], [0.1, 0.1], "at least three"),
        ([0.1, np.nan, 0.3], [0.1, 0.1, 0.1], "finite"),
        ([0.1, 0.2, 0.3], [0.1, 0.0, 0.1], "greater than zero"),
        ([0.1, 0.2, 0.3], [0.1, 0.1], "equal length"),
    ],
)
def test_rejects_invalid_effect_inputs(estimate, se, message):
    with pytest.raises(ValueError, match=message):
        nnf_prediction_interval(estimate, se, draws=500)


@pytest.mark.parametrize("target_se", [-0.1, np.inf, np.nan])
def test_rejects_invalid_target_standard_error(target_se):
    with pytest.raises(ValueError, match="target_se"):
        nnf_prediction_interval(BASE_Y, BASE_SE, target_se=target_se, draws=500)


@pytest.mark.parametrize("seed", [True, np.bool_(False), -1, 1.5])
def test_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        nnf_prediction_interval(BASE_Y, BASE_SE, draws=500, seed=seed)


def test_is_reproducible_for_fixed_seed():
    first = nnf_prediction_interval(BASE_Y, BASE_SE, observed_value=0.3, draws=5_000, seed=73)
    second = nnf_prediction_interval(BASE_Y, BASE_SE, observed_value=0.3, draws=5_000, seed=73)
    assert first == second
    assert first.draws == 5_000
    assert first.seed == 73
    assert first.status.startswith("ok")


def test_observed_target_interval_widens_with_target_error():
    result = nnf_prediction_interval(
        BASE_Y,
        BASE_SE,
        target_se=0.35,
        observed_value=0.3,
        draws=10_000,
        seed=91,
    )
    latent_width = result.latent.high - result.latent.low
    observed_width = result.observed.high - result.observed.low
    assert observed_width > latent_width
    assert result.target_se == 0.35
    assert 0 < result.observed_pit < 1
    assert np.isfinite(result.observed_pit_z)


def test_null_case_has_large_zero_boundary_mass_and_finite_interval():
    result = nnf_prediction_interval(
        np.array([0.20, 0.21, 0.19, 0.20, 0.205]),
        np.full(5, 0.20),
        draws=5_000,
        seed=17,
    )
    assert result.zero_boundary_mass > 0.95
    assert np.isfinite([result.latent.low, result.latent.high]).all()
    assert result.latent.low < 0.20 < result.latent.high


def test_identical_effects_export_low_dispersion_width_warning():
    result = nnf_prediction_interval(
        np.full(5, 0.2),
        np.full(5, 0.2),
        draws=5_000,
        seed=18,
    )
    assert result.q_value == pytest.approx(0.0)
    assert result.low_dispersion_width_warning
    assert result.latent_width_to_fixed_mean_ci_width < 0.5


def test_heterogeneous_case_has_positive_tau_distribution():
    result = nnf_prediction_interval(
        np.array([-1.1, -0.2, 0.4, 1.0, 1.7, 2.4]),
        np.full(6, 0.15),
        draws=5_000,
        seed=31,
    )
    assert result.zero_boundary_mass < 0.05
    assert result.tau2_median > 0
    assert result.tau2_upper_95 > result.tau2_median
    assert result.diagnostics.inversion_max_probability_error <= 5e-4


def test_reports_cdf_nonconvergence_without_fallback():
    with pytest.raises(NNFComputationError, match="did not converge"):
        nnf_prediction_interval(
            np.array([-1.0, 0.0, 0.5, 1.0]),
            np.array([0.001, 0.1, 0.2, 1.0]),
            draws=5_000,
            max_cdf_terms=100,
        )


def test_approximately_matches_fixed_pimeta_boot_reference():
    # pimeta 1.1.3: pima(BASE_Y, BASE_SE, method="boot", B=250000, seed=2468).
    # The Python and R generators differ, so this checks Monte Carlo agreement,
    # not bitwise identity. Expected values are frozen after local R evaluation.
    result = nnf_prediction_interval(BASE_Y, BASE_SE, draws=25_000, seed=2468)
    expected = np.array([0.1061, 0.4388])
    actual = np.array([result.latent.low, result.latent.high])
    assert actual == pytest.approx(expected, abs=0.015)


def test_matches_frozen_pimeta_oracle_across_edge_cases():
    oracle_path = Path(__file__).parents[1] / "validation" / "oracles" / "pimeta_nnf_reference.tsv"
    oracle = pd.read_csv(oracle_path, sep="\t")
    for row in oracle.itertuples(index=False):
        result = nnf_prediction_interval(
            np.fromstring(row.y, sep=";"),
            np.fromstring(row.se, sep=";"),
            draws=250_000,
            seed=int(row.seed),
        )
        actual = np.array([result.latent.low, result.latent.high])
        expected = np.array([row.pi_true_low, row.pi_true_high])
        assert actual == pytest.approx(expected, abs=0.012)


def test_low_draw_count_emits_strong_warning():
    with pytest.warns(RuntimeWarning, match="below 5,000"):
        nnf_prediction_interval(BASE_Y, BASE_SE, draws=500, seed=1)
