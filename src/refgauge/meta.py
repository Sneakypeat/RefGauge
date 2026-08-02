"""Small-study random-effects estimators used by RefGauge."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class MetaAnalysisResult:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    tau2: float
    q: float
    i2: float
    k: int
    method: str

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _clean(estimate: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(estimate, dtype=float)
    s = np.asarray(se, dtype=float)
    keep = np.isfinite(y) & np.isfinite(s) & (s > 0)
    return y[keep], s[keep]


def dersimonian_laird_tau2(estimate: np.ndarray, se: np.ndarray) -> float:
    """DerSimonian-Laird moment estimate of between-study variance."""
    y, s = _clean(estimate, se)
    if len(y) < 2:
        return 0.0
    w = 1.0 / s**2
    mean = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - mean) ** 2)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    return float(max(0.0, (q - (len(y) - 1)) / c)) if c > 0 else 0.0


def reml_tau2(estimate: np.ndarray, se: np.ndarray) -> float:
    """Estimate tau-squared by restricted maximum likelihood."""
    y, s = _clean(estimate, se)
    if len(y) < 2:
        return 0.0
    variance = s**2

    def objective(tau2: float) -> float:
        total = variance + tau2
        weights = 1.0 / total
        mean = np.sum(weights * y) / np.sum(weights)
        return float(
            0.5
            * (np.sum(np.log(total)) + np.log(np.sum(weights)) + np.sum(weights * (y - mean) ** 2))
        )

    scale = max(float(np.var(y, ddof=1)), float(np.max(variance)), 1e-8)
    fit = optimize.minimize_scalar(objective, bounds=(0.0, 100.0 * scale), method="bounded")
    if not fit.success or not np.isfinite(fit.x):
        return dersimonian_laird_tau2(y, s)
    tau2 = float(max(0.0, fit.x))
    boundary_tolerance = max(1e-12, scale * 1e-8)
    return 0.0 if objective(0.0) <= objective(tau2) + boundary_tolerance else tau2


def random_effects(
    estimate: np.ndarray,
    se: np.ndarray,
    *,
    method: str = "reml",
    alpha: float = 0.05,
) -> MetaAnalysisResult:
    """Pool estimates with a random-effects model.

    REML is the default. Confidence intervals use a t critical value when at least
    three studies are present, avoiding overconfident normal intervals in small
    collections. A one-study group is returned unchanged and should not be treated
    as a cross-study estimate.
    """
    if method not in {"reml", "dl"}:
        raise ValueError("method must be 'reml' or 'dl'")
    if not isinstance(alpha, (int, float, np.integer, np.floating)) or not (
        np.isfinite(alpha) and 0 < alpha < 1
    ):
        raise ValueError("alpha must be finite and between zero and one")
    y, s = _clean(estimate, se)
    k = len(y)
    if k == 0:
        return MetaAnalysisResult(*(float("nan"),) * 7, k=0, method=method)
    if k == 1:
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        return MetaAnalysisResult(
            float(y[0]),
            float(s[0]),
            float(y[0] - z * s[0]),
            float(y[0] + z * s[0]),
            0.0,
            0.0,
            0.0,
            1,
            "single-study",
        )
    tau2 = reml_tau2(y, s) if method == "reml" else dersimonian_laird_tau2(y, s)
    weights = 1.0 / (s**2 + tau2)
    pooled = float(np.sum(weights * y) / np.sum(weights))
    conventional_se = float(np.sqrt(1.0 / np.sum(weights)))
    kh_scale = float(np.sum(weights * (y - pooled) ** 2) / (k - 1))
    kh_se = float(np.sqrt(max(0.0, kh_scale) / np.sum(weights)))
    # Modified Knapp-Hartung avoids intervals becoming narrower than the
    # conventional random-effects interval when residual dispersion is small.
    pooled_se = max(conventional_se, kh_se)
    fixed_weights = 1.0 / s**2
    fixed = np.sum(fixed_weights * y) / np.sum(fixed_weights)
    q = float(np.sum(fixed_weights * (y - fixed) ** 2))
    i2 = float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=k - 1))
    return MetaAnalysisResult(
        pooled,
        pooled_se,
        pooled - critical * pooled_se,
        pooled + critical * pooled_se,
        tau2,
        q,
        i2,
        k,
        method,
    )
