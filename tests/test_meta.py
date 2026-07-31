import numpy as np

from contrastcal.meta import dersimonian_laird_tau2, random_effects


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
