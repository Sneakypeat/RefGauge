"""Nagashima-Noma-Furukawa confidence-distribution prediction intervals.

This module implements the equations described by Nagashima, Noma, and
Furukawa independently.  It has no runtime dependency on ``pimeta`` or other
meta-analysis packages.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import interpolate, linalg, special, stats


class NNFComputationError(RuntimeError):
    """Raised when the confidence distribution cannot be computed reliably."""


@dataclass(frozen=True)
class PredictionInterval:
    """A two-sided prediction interval."""

    low: float
    high: float
    level: float


@dataclass(frozen=True)
class NNFDiagnostics:
    """Numerical diagnostics for the weighted-chi-square inversion."""

    cdf_method: str
    cdf_evaluations: int
    cdf_max_terms: int
    cdf_max_truncation_error: float
    inversion_grid_size: int
    inversion_max_probability_error: float
    bracket_upper: float


@dataclass(frozen=True)
class NNFPredictionResult:
    """NNF prediction intervals for latent and observed target effects."""

    latent: PredictionInterval
    observed: PredictionInterval
    target_se: float
    observed_pit: float
    observed_pit_z: float
    zero_boundary_mass: float
    tau2_median: float
    tau2_upper_95: float
    draws: int
    seed: int
    k: int
    status: str
    q_value: float
    latent_width_to_fixed_mean_ci_width: float
    low_dispersion_width_warning: bool
    diagnostics: NNFDiagnostics


@dataclass(frozen=True)
class _CDFResult:
    value: float
    terms: int
    truncation_error: float


def _weighted_chi_square_cdf(
    value: float,
    weights: np.ndarray,
    *,
    tolerance: float,
    max_terms: int,
) -> _CDFResult:
    """CDF of ``sum(weights * chi2_1)`` using an exact gamma-mixture series.

    A weighted chi-square with positive weights is a sum of independent gamma
    variables.  The Moschopoulos expansion expresses that sum as a positive
    mixture of gamma CDFs.  The unallocated mixture mass is an explicit upper
    bound on the CDF truncation error.  Failure to meet ``tolerance`` raises;
    there is no approximation fallback.
    """
    lam = np.asarray(weights, dtype=float)
    if value < 0 or not np.isfinite(value):
        raise ValueError("weighted chi-square value must be finite and non-negative")
    if lam.ndim != 1 or len(lam) == 0 or not np.isfinite(lam).all() or (lam <= 0).any():
        raise ValueError("weighted chi-square weights must be finite and positive")

    if np.allclose(lam, lam[0], rtol=1e-12, atol=0.0):
        return _CDFResult(float(stats.chi2.cdf(value / lam[0], df=len(lam))), 0, 0.0)

    alpha = np.full(len(lam), 0.5)
    scales = 2.0 * lam
    base_scale = float(np.min(scales))
    ratios = 1.0 - base_scale / scales
    log_constant = float(np.sum(alpha * np.log(base_scale / scales)))
    if log_constant < np.log(np.finfo(float).tiny):
        raise NNFComputationError("gamma-mixture normalising constant underflowed")
    constant = float(np.exp(log_constant))
    shape = float(np.sum(alpha))

    delta = np.zeros(max_terms + 1, dtype=float)
    gamma = np.zeros(max_terms + 1, dtype=float)
    delta[0] = 1.0
    powers = np.ones_like(ratios)
    mixture_mass = constant
    cdf = constant * float(special.gammainc(shape, value / base_scale))

    for term in range(1, max_terms + 1):
        powers *= ratios
        gamma[term] = float(np.sum(alpha * powers))
        delta[term] = float(np.dot(gamma[1 : term + 1], delta[term - 1 :: -1]) / term)
        component_mass = constant * delta[term]
        if not np.isfinite(component_mass) or component_mass < 0:
            raise NNFComputationError("gamma-mixture recurrence became non-finite")
        mixture_mass += component_mass
        cdf += component_mass * float(special.gammainc(shape + term, value / base_scale))
        remainder = max(0.0, 1.0 - mixture_mass)
        if remainder <= tolerance:
            return _CDFResult(float(np.clip(cdf, 0.0, 1.0)), term, float(remainder))

    raise NNFComputationError(
        f"weighted chi-square CDF did not converge within {max_terms} terms "
        f"(remaining mixture mass {max(0.0, 1.0 - mixture_mass):.3g})"
    )


def _validate_inputs(
    estimate: np.ndarray,
    se: np.ndarray,
    target_se: float,
    alpha: float,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(estimate, dtype=float)
    s = np.asarray(se, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or y.shape != s.shape:
        raise ValueError("estimate and se must be one-dimensional arrays of equal length")
    if len(y) < 3:
        raise ValueError("NNF prediction intervals require at least three studies")
    if not np.isfinite(y).all() or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("estimate and se must be finite; se must be greater than zero")
    if not np.isfinite(target_se) or target_se < 0:
        raise ValueError("target_se must be finite and non-negative")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    if not isinstance(draws, (int, np.integer)) or draws < 100:
        raise ValueError("draws must be an integer of at least 100")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return y, s


def nnf_prediction_interval(
    estimate: np.ndarray,
    se: np.ndarray,
    *,
    target_se: float = 0.0,
    observed_value: float | None = None,
    alpha: float = 0.05,
    draws: int = 25_000,
    seed: int = 0,
    cdf_tolerance: float = 1e-9,
    inversion_tolerance: float = 5e-4,
    initial_grid_size: int = 129,
    max_grid_size: int = 1_025,
    max_cdf_terms: int = 20_000,
) -> NNFPredictionResult:
    """Compute NNF confidence-distribution prediction intervals.

    ``latent`` predicts the true effect in a new study. ``observed`` predicts a
    future estimate having known standard error ``target_se``.  Monte Carlo
    output is deterministic for a fixed ``seed``.
    """
    y, s = _validate_inputs(estimate, se, target_se, alpha, draws, seed)
    if observed_value is not None and not np.isfinite(observed_value):
        raise ValueError("observed_value must be finite when supplied")
    if draws < 5_000:
        warnings.warn(
            "NNF draws below 5,000 are suitable only for tests or exploratory runs",
            RuntimeWarning,
            stacklevel=2,
        )
    if not 0 < cdf_tolerance < 0.01:
        raise ValueError("cdf_tolerance must be between zero and 0.01")
    if not 0 < inversion_tolerance < 0.01:
        raise ValueError("inversion_tolerance must be between zero and 0.01")
    if initial_grid_size < 17 or max_grid_size < initial_grid_size:
        raise ValueError("grid sizes must satisfy 17 <= initial_grid_size <= max_grid_size")
    if max_cdf_terms < 100:
        raise ValueError("max_cdf_terms must be at least 100")

    k = len(y)
    variance = s**2
    fixed_weights = 1.0 / variance
    matrix_a = np.diag(fixed_weights) - np.outer(fixed_weights, fixed_weights) / np.sum(
        fixed_weights
    )
    q_value = float(y @ matrix_a @ y)
    boundary_mass = float(stats.chi2.sf(q_value, df=k - 1))
    boundary_mass = float(np.clip(boundary_mass, 0.0, 1.0))

    cdf_evaluations = 0
    cdf_max_terms_used = 0
    cdf_max_error = 0.0

    def heterogeneity_cdf(tau2: float) -> float:
        nonlocal cdf_evaluations, cdf_max_terms_used, cdf_max_error
        if tau2 == 0:
            return boundary_mass
        root_variance = np.sqrt(variance + tau2)
        quadratic = matrix_a * np.outer(root_variance, root_variance)
        eigenvalues = linalg.eigvalsh(quadratic, check_finite=False)
        threshold = max(float(np.max(eigenvalues)) * 1e-11, np.finfo(float).eps)
        positive = eigenvalues[eigenvalues > threshold]
        if len(positive) != k - 1:
            raise NNFComputationError(
                f"expected {k - 1} positive quadratic-form eigenvalues, found {len(positive)}"
            )
        result = _weighted_chi_square_cdf(
            q_value,
            positive,
            tolerance=cdf_tolerance,
            max_terms=max_cdf_terms,
        )
        cdf_evaluations += 1
        cdf_max_terms_used = max(cdf_max_terms_used, result.terms)
        cdf_max_error = max(cdf_max_error, result.truncation_error)
        return float(1.0 - result.value)

    rng = np.random.default_rng(seed)
    uniform = rng.random(draws)
    tau_draws = np.zeros(draws, dtype=float)
    positive_mask = uniform > boundary_mass
    bracket_upper = 0.0
    inversion_error = 0.0
    grid_used = 1

    if positive_mask.any():
        maximum_probability = float(np.max(uniform[positive_mask]))
        scale = max(float(np.var(y, ddof=1)), float(np.max(variance)), 1e-12)
        bracket_upper = scale
        upper_probability = heterogeneity_cdf(bracket_upper)
        for _ in range(60):
            if upper_probability >= maximum_probability:
                break
            bracket_upper *= 2.0
            upper_probability = heterogeneity_cdf(bracket_upper)
        else:
            raise NNFComputationError("failed to bracket the tau-squared confidence distribution")

        grid_size = initial_grid_size
        inverse = None
        while grid_size <= max_grid_size:
            positive_grid = np.geomspace(bracket_upper * 1e-12, bracket_upper, grid_size - 1)
            tau_grid = np.concatenate(([0.0], positive_grid))
            probability_grid = np.array([heterogeneity_cdf(t) for t in tau_grid])
            decreases = np.diff(probability_grid)
            if np.min(decreases) < -10.0 * cdf_tolerance:
                raise NNFComputationError(
                    "tau-squared confidence distribution was materially non-monotone"
                )
            probability_grid = np.maximum.accumulate(probability_grid)
            keep = np.concatenate(([True], np.diff(probability_grid) > cdf_tolerance / 10.0))
            probability_unique = probability_grid[keep]
            tau_unique = tau_grid[keep]
            if len(probability_unique) < 2 or probability_unique[-1] < maximum_probability:
                raise NNFComputationError("insufficient support for tau-squared inversion")
            inverse = interpolate.PchipInterpolator(
                probability_unique,
                tau_unique,
                extrapolate=False,
            )

            check_low = max(boundary_mass + 1e-8, float(probability_unique[1]))
            check_probabilities = np.linspace(check_low, maximum_probability, 33)
            check_tau = np.asarray(inverse(check_probabilities), dtype=float)
            reconstructed = np.array([heterogeneity_cdf(t) for t in check_tau])
            inversion_error = float(np.max(np.abs(reconstructed - check_probabilities)))
            grid_used = grid_size
            if inversion_error <= inversion_tolerance:
                break
            grid_size = 2 * grid_size - 1
        else:
            raise NNFComputationError(
                "tau-squared inversion did not meet its probability-error tolerance "
                f"({inversion_error:.3g} > {inversion_tolerance:.3g})"
            )

        if inverse is None:
            raise NNFComputationError("tau-squared inverse was not constructed")
        tau_draws[positive_mask] = np.asarray(inverse(uniform[positive_mask]), dtype=float)
        if not np.isfinite(tau_draws).all() or (tau_draws < 0).any():
            raise NNFComputationError("tau-squared inversion produced invalid draws")

    total_variance = variance[None, :] + tau_draws[:, None]
    random_weights = 1.0 / total_variance
    pooled = np.sum(random_weights * y, axis=1) / np.sum(random_weights, axis=1)
    residual_variance = np.sum(random_weights * (y - pooled[:, None]) ** 2, axis=1)
    residual_variance /= (k - 1) * np.sum(random_weights, axis=1)
    residual_variance = np.maximum(residual_variance, 0.0)

    heterogeneity_noise = rng.standard_normal(draws)
    mean_noise = rng.standard_t(df=k - 1, size=draws)
    target_noise = rng.standard_normal(draws)
    latent_draws = (
        pooled + heterogeneity_noise * np.sqrt(tau_draws) - mean_noise * np.sqrt(residual_variance)
    )
    observed_draws = latent_draws + target_noise * target_se
    probabilities = [alpha / 2.0, 1.0 - alpha / 2.0]
    latent_bounds = np.quantile(latent_draws, probabilities)
    observed_bounds = np.quantile(observed_draws, probabilities)
    level = 1.0 - alpha
    if observed_value is None:
        observed_pit = float("nan")
        observed_pit_z = float("nan")
    else:
        observed_rank = int(np.count_nonzero(observed_draws <= observed_value))
        observed_pit = float((observed_rank + 0.5) / (draws + 1.0))
        observed_pit_z = float(stats.norm.ppf(observed_pit))

    fixed_mean_se = float(np.sqrt(1.0 / np.sum(fixed_weights)))
    fixed_mean_ci_width = float(2.0 * stats.t.ppf(1.0 - alpha / 2.0, df=k - 1) * fixed_mean_se)
    latent_width = float(latent_bounds[1] - latent_bounds[0])
    width_ratio = latent_width / fixed_mean_ci_width
    low_dispersion_warning = bool(width_ratio < 0.5)

    diagnostics = NNFDiagnostics(
        cdf_method="positive_gamma_mixture",
        cdf_evaluations=cdf_evaluations,
        cdf_max_terms=cdf_max_terms_used,
        cdf_max_truncation_error=float(cdf_max_error),
        inversion_grid_size=grid_used,
        inversion_max_probability_error=inversion_error,
        bracket_upper=bracket_upper,
    )
    return NNFPredictionResult(
        latent=PredictionInterval(float(latent_bounds[0]), float(latent_bounds[1]), level),
        observed=PredictionInterval(float(observed_bounds[0]), float(observed_bounds[1]), level),
        target_se=float(target_se),
        observed_pit=observed_pit,
        observed_pit_z=observed_pit_z,
        zero_boundary_mass=boundary_mass,
        tau2_median=float(np.median(tau_draws)),
        tau2_upper_95=float(np.quantile(tau_draws, 0.95)),
        draws=int(draws),
        seed=int(seed),
        k=k,
        status="ok_all_tau2_zero" if not positive_mask.any() else "ok",
        q_value=q_value,
        latent_width_to_fixed_mean_ci_width=float(width_ratio),
        low_dispersion_width_warning=low_dispersion_warning,
        diagnostics=diagnostics,
    )
