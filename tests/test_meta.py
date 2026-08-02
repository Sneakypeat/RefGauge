import numpy as np
import pytest

from refgauge.meta import dersimonian_laird_tau2, random_effects, reml_tau2


def test_fixed_like_pool_is_weighted_mean():
    y = np.array([0.2, 0.3, 0.4])
    se = np.array([0.1, 0.1, 0.1])
    fit = random_effects(y, se, method="reml")
    assert fit.k == 3
    assert abs(fit.estimate - 0.3) < 1e-6
    assert fit.se > 0
    assert fit.ci_low < fit.estimate < fit.ci_high


def test_heterogeneous_effects_have_positive_tau2():
    y = np.array([-1.0, 0.0, 1.0, 2.0])
    se = np.repeat(0.1, 4)
    fit = random_effects(y, se)
    assert fit.tau2 > 0.5
    assert fit.i2 > 0.9
    assert dersimonian_laird_tau2(y, se) > 0


def test_reml_snaps_a_boundary_solution_to_exact_zero():
    y = np.array([0.49, 0.50, 0.51, 0.50])
    se = np.repeat(0.2, len(y))
    assert reml_tau2(y, se) == 0.0


def test_single_study_is_explicit():
    fit = random_effects(np.array([0.4]), np.array([0.2]))
    assert fit.k == 1
    assert fit.method == "single-study"
    assert fit.estimate == 0.4


def test_invalid_method_fails():
    try:
        random_effects(np.array([0.1, 0.2]), np.array([0.1, 0.1]), method="bad")
    except ValueError as exc:
        assert "method" in str(exc)
    else:
        raise AssertionError("invalid method did not fail")


def test_invalid_method_fails_for_single_study_too():
    with pytest.raises(ValueError, match="method"):
        random_effects(np.array([0.2]), np.array([0.1]), method="bad")


@pytest.mark.parametrize("alpha", [0.0, 1.0, np.nan, np.inf])
def test_random_effects_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        random_effects(np.array([0.2, 0.3]), np.array([0.1, 0.1]), alpha=alpha)
