"""Leave-one-study-out transport calibration."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .meta import random_effects
from .validation import validate_effects


def calibrate_loso(
    effects: pd.DataFrame,
    *,
    method: str = "reml",
    alpha: float = 0.05,
    min_training_studies: int = 2,
) -> pd.DataFrame:
    """Predict each direct study effect using all other studies.

    The prediction interval includes uncertainty in the pooled estimate,
    between-study heterogeneity, and sampling uncertainty in the held-out direct
    estimate. This makes interval coverage a test of transport, not merely whether
    the pooled mean is estimated precisely.
    """
    data = validate_effects(effects)
    rows: list[dict[str, float | int | str | bool]] = []
    for (feature, stratum), group in data.groupby(["feature", "stratum"], sort=False):
        for target in group.itertuples(index=False):
            training = group.loc[group["study"].ne(target.study)]
            fit = random_effects(
                training["estimate"].to_numpy(), training["se"].to_numpy(), method=method
            )
            if fit.k < min_training_studies:
                continue
            prediction_se = float(np.sqrt(fit.se**2 + fit.tau2 + target.se**2))
            if fit.k >= 3:
                critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=fit.k - 2))
                interval_distribution = f"t(df={fit.k - 2})"
            else:
                critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
                interval_distribution = "normal_fallback_k2"
            pi_low = fit.estimate - critical * prediction_se
            pi_high = fit.estimate + critical * prediction_se
            rows.append(
                {
                    "feature": feature,
                    "stratum": stratum,
                    "target_study": target.study,
                    "direct_estimate": float(target.estimate),
                    "direct_se": float(target.se),
                    "atlas_estimate": fit.estimate,
                    "atlas_se": fit.se,
                    "tau2": fit.tau2,
                    "n_training_studies": fit.k,
                    "interval_distribution": interval_distribution,
                    "prediction_se": prediction_se,
                    "pi_low": pi_low,
                    "pi_high": pi_high,
                    "covered": bool(pi_low <= target.estimate <= pi_high),
                    "discrepancy": float(target.estimate - fit.estimate),
                    "standardized_discrepancy": float(
                        (target.estimate - fit.estimate) / prediction_se
                    ),
                    "sign_agreement": bool(np.sign(target.estimate) == np.sign(fit.estimate)),
                }
            )
    if not rows:
        raise ValueError("No feature had enough training studies after target-study exclusion")
    return pd.DataFrame(rows)


def _safe_correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(x.corr(y, method=method))


def _weighted_calibration(
    x: np.ndarray, y: np.ndarray, variance: np.ndarray
) -> tuple[float, float]:
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(variance) & (variance > 0)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0:
        return float("nan"), float("nan")
    design = np.column_stack([np.ones(keep.sum()), x[keep]])
    weights = 1.0 / variance[keep]
    xtw = design.T * weights
    beta = np.linalg.pinv(xtw @ design) @ (xtw @ y[keep])
    return float(beta[0]), float(beta[1])


def calibration_summary(
    predictions: pd.DataFrame,
    *,
    strong_threshold: float = 0.5,
) -> pd.DataFrame:
    """Summarize calibration overall and by target-study/stratum."""
    required = {
        "direct_estimate",
        "direct_se",
        "atlas_estimate",
        "atlas_se",
        "target_study",
        "stratum",
        "covered",
        "sign_agreement",
        "standardized_discrepancy",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    def one(group: pd.DataFrame, scope: str) -> dict[str, float | int | str]:
        direct = group["direct_estimate"]
        atlas = group["atlas_estimate"]
        strong = group.loc[group["atlas_estimate"].abs().ge(strong_threshold)]
        intercept, slope = _weighted_calibration(
            atlas.to_numpy(),
            direct.to_numpy(),
            (group["direct_se"] ** 2 + group["atlas_se"] ** 2).to_numpy(),
        )
        return {
            "scope": scope,
            "n": len(group),
            "pearson": _safe_correlation(atlas, direct, "pearson"),
            "spearman": _safe_correlation(atlas, direct, "spearman"),
            "sign_agreement": float(group["sign_agreement"].mean()),
            "strong_sign_agreement": float(strong["sign_agreement"].mean())
            if len(strong)
            else np.nan,
            "n_strong": len(strong),
            "coverage": float(group["covered"].mean()),
            "mean_bias": float(group["discrepancy"].mean()),
            "mae": float(group["discrepancy"].abs().mean()),
            "rmse": float(np.sqrt(np.mean(group["discrepancy"] ** 2))),
            "median_abs_z": float(group["standardized_discrepancy"].abs().median()),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
        }

    rows = [one(predictions, "overall")]
    for (study, stratum), group in predictions.groupby(["target_study", "stratum"], sort=True):
        rows.append(one(group, f"study={study};stratum={stratum}"))
    return pd.DataFrame(rows)
