"""Leave-one-study-out transport calibration."""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .meta import dersimonian_laird_tau2, random_effects, reml_tau2
from .prediction import NNFComputationError, nnf_prediction_interval
from .validation import validate_effects

INTERVAL_METHODS = {
    "nnf_cd",
    "fixed_normal",
    "dl_normal",
    "reml_normal",
    "reml_mkh_approx",
    "unweighted_t",
}
NNF_FAILURE_POLICIES = {"raise", "record"}


def _fold_seed(seed: int, feature: str, stratum: str, target_study: str) -> int:
    payload = f"{seed}\0{feature}\0{stratum}\0{target_study}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _cochran_q(estimate: np.ndarray, se: np.ndarray) -> tuple[float, int, float]:
    """Return fixed-effect Cochran Q, degrees of freedom, and upper-tail p value."""
    weights = 1.0 / se**2
    fixed_mean = float(np.sum(weights * estimate) / np.sum(weights))
    q_value = float(np.sum(weights * (estimate - fixed_mean) ** 2))
    q_df = len(estimate) - 1
    return q_value, q_df, float(stats.chi2.sf(q_value, df=q_df))


def calibrate_loso(
    effects: pd.DataFrame,
    *,
    method: str = "reml",
    interval_method: str = "nnf_cd",
    alpha: float = 0.05,
    min_training_studies: int | None = None,
    nnf_draws: int = 25_000,
    seed: int = 20260801,
    nnf_failure_policy: str = "raise",
) -> pd.DataFrame:
    """Predict each direct study effect using all other studies.

    The primary NNF confidence-distribution interval propagates uncertainty in
    between-study heterogeneity. Separate intervals are returned for the latent
    effect in a new study and for an observed target estimate with its supplied
    sampling error. Only the observed-target interval defines LOSO coverage.
    """
    if interval_method not in INTERVAL_METHODS:
        raise ValueError(f"interval_method must be one of {sorted(INTERVAL_METHODS)}")
    if method not in {"reml", "dl"}:
        raise ValueError("method must be 'reml' or 'dl'")
    if not isinstance(alpha, (int, float, np.integer, np.floating)) or not (
        np.isfinite(alpha) and 0 < alpha < 1
    ):
        raise ValueError("alpha must be finite and between zero and one")
    if min_training_studies is None:
        min_training_studies = 5 if interval_method == "nnf_cd" else 3
    if isinstance(min_training_studies, (bool, np.bool_)) or not isinstance(
        min_training_studies, (int, np.integer)
    ):
        raise ValueError("min_training_studies must be an integer")
    if min_training_studies < 3:
        raise ValueError("min_training_studies must be at least 3")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if interval_method == "nnf_cd" and (
        isinstance(nnf_draws, (bool, np.bool_))
        or not isinstance(nnf_draws, (int, np.integer))
        or nnf_draws < 100
    ):
        raise ValueError("nnf_draws must be an integer of at least 100")
    if nnf_failure_policy not in NNF_FAILURE_POLICIES:
        raise ValueError(f"nnf_failure_policy must be one of {sorted(NNF_FAILURE_POLICIES)}")

    data = validate_effects(effects)
    rows: list[dict[str, float | int | str | bool]] = []
    for (feature, stratum), group in data.groupby(["feature", "stratum"], sort=False):
        for target in group.itertuples(index=False):
            training = group.loc[group["study"].ne(target.study)]
            training_estimate = training["estimate"].to_numpy()
            training_se = training["se"].to_numpy()
            k = len(training_estimate)
            if k < min_training_studies:
                continue
            cochran_q, cochran_q_df, cochran_q_pvalue = _cochran_q(training_estimate, training_se)

            if interval_method == "nnf_cd":
                fit = random_effects(training_estimate, training_se, method=method)
                atlas_estimate, atlas_se, tau2 = fit.estimate, fit.se, fit.tau2
                point_method = f"{method}_mkh_summary"
                try:
                    nnf = nnf_prediction_interval(
                        training_estimate,
                        training_se,
                        target_se=float(target.se),
                        observed_value=float(target.estimate),
                        alpha=alpha,
                        draws=nnf_draws,
                        seed=_fold_seed(seed, str(feature), str(stratum), str(target.study)),
                    )
                    pi_true_low, pi_true_high = nnf.latent.low, nnf.latent.high
                    pi_low, pi_high = nnf.observed.low, nnf.observed.high
                    interval_distribution = "nnf_confidence_distribution"
                    interval_status = nnf.status
                    interval_failure_reason = ""
                    interval_success = True
                    zero_boundary_mass = nnf.zero_boundary_mass
                    tau2_cd_median = nnf.tau2_median
                    tau2_cd_upper_95 = nnf.tau2_upper_95
                    nnf_probability_error = nnf.diagnostics.inversion_max_probability_error
                    nnf_cdf_error = nnf.diagnostics.cdf_max_truncation_error
                    predictive_pit = nnf.observed_pit
                    standardized_discrepancy = nnf.observed_pit_z
                    standardized_discrepancy_method = "predictive_pit_normal_score"
                    low_dispersion_width_ratio = nnf.latent_width_to_fixed_mean_ci_width
                    low_dispersion_width_warning = nnf.low_dispersion_width_warning
                except (NNFComputationError, ValueError, FloatingPointError) as error:
                    if nnf_failure_policy == "raise":
                        raise
                    pi_true_low = pi_true_high = pi_low = pi_high = float("nan")
                    interval_distribution = "nnf_confidence_distribution"
                    interval_status = "numerical_failure_no_fallback"
                    interval_failure_reason = f"{type(error).__name__}: {error}"
                    interval_success = False
                    zero_boundary_mass = tau2_cd_median = tau2_cd_upper_95 = float("nan")
                    nnf_probability_error = nnf_cdf_error = float("nan")
                    predictive_pit = float("nan")
                    standardized_discrepancy = float("nan")
                    standardized_discrepancy_method = "predictive_pit_normal_score"
                    low_dispersion_width_ratio = float("nan")
                    low_dispersion_width_warning = float("nan")
                prediction_se = float("nan")
            elif interval_method == "reml_mkh_approx":
                fit = random_effects(training_estimate, training_se, method="reml")
                atlas_estimate, atlas_se, tau2 = fit.estimate, fit.se, fit.tau2
                point_method = "reml_mkh"
                latent_se = float(np.sqrt(fit.se**2 + fit.tau2))
                prediction_se = float(np.sqrt(latent_se**2 + target.se**2))
                critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=fit.k - 2))
                pi_true_low = fit.estimate - critical * latent_se
                pi_true_high = fit.estimate + critical * latent_se
                pi_low = fit.estimate - critical * prediction_se
                pi_high = fit.estimate + critical * prediction_se
                interval_distribution = f"reml_mkh_approx_t(df={fit.k - 2})"
                standardized_discrepancy = float((target.estimate - fit.estimate) / prediction_se)
                standardized_discrepancy_method = "normal_predictive_residual"
                interval_status = "approximate_plugin_tau2"
                interval_failure_reason = ""
                interval_success = True
                zero_boundary_mass = float("nan")
                tau2_cd_median = float("nan")
                tau2_cd_upper_95 = float("nan")
                nnf_probability_error = float("nan")
                nnf_cdf_error = float("nan")
                predictive_pit = float(stats.norm.cdf(standardized_discrepancy))
                low_dispersion_width_ratio = float("nan")
                low_dispersion_width_warning = float("nan")
            else:
                if interval_method == "unweighted_t":
                    atlas_estimate = float(np.mean(training_estimate))
                    observed_variance = float(np.var(training_estimate, ddof=1))
                    average_sampling_variance = float(np.mean(training_se**2))
                    tau2 = max(0.0, observed_variance - average_sampling_variance)
                    atlas_se = float(np.sqrt((tau2 + average_sampling_variance) / k))
                    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=k - 1))
                    interval_distribution = f"unweighted_moment_t(df={k - 1})"
                    point_method = "unweighted_mean"
                else:
                    if interval_method == "fixed_normal":
                        tau2 = 0.0
                        point_method = "fixed_inverse_variance"
                    elif interval_method == "dl_normal":
                        tau2 = dersimonian_laird_tau2(training_estimate, training_se)
                        point_method = "dl_inverse_variance"
                    else:
                        tau2 = reml_tau2(training_estimate, training_se)
                        point_method = "reml_inverse_variance"
                    weights = 1.0 / (training_se**2 + tau2)
                    atlas_estimate = float(np.sum(weights * training_estimate) / np.sum(weights))
                    atlas_se = float(np.sqrt(1.0 / np.sum(weights)))
                    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
                    interval_distribution = f"{interval_method}_z"

                latent_se = float(np.sqrt(atlas_se**2 + tau2))
                prediction_se = float(np.sqrt(latent_se**2 + target.se**2))
                pi_true_low = atlas_estimate - critical * latent_se
                pi_true_high = atlas_estimate + critical * latent_se
                pi_low = atlas_estimate - critical * prediction_se
                pi_high = atlas_estimate + critical * prediction_se
                standardized_discrepancy = float((target.estimate - atlas_estimate) / prediction_se)
                standardized_discrepancy_method = "normal_predictive_residual"
                interval_status = "approximate_plugin_tau2"
                interval_failure_reason = ""
                interval_success = True
                zero_boundary_mass = float("nan")
                tau2_cd_median = float("nan")
                tau2_cd_upper_95 = float("nan")
                nnf_probability_error = float("nan")
                nnf_cdf_error = float("nan")
                predictive_pit = float(stats.norm.cdf(standardized_discrepancy))
                low_dispersion_width_ratio = float("nan")
                low_dispersion_width_warning = float("nan")

            pi_midpoint = (pi_low + pi_high) / 2.0
            pi_half_width = (pi_high - pi_low) / 2.0
            rows.append(
                {
                    "feature": feature,
                    "stratum": stratum,
                    "target_study": target.study,
                    "direct_estimate": float(target.estimate),
                    "direct_se": float(target.se),
                    "atlas_estimate": atlas_estimate,
                    "atlas_se": atlas_se,
                    "tau2": tau2,
                    "cochran_q": cochran_q,
                    "cochran_q_df": cochran_q_df,
                    "cochran_q_pvalue": cochran_q_pvalue,
                    "n_training_studies": k,
                    "point_method": point_method,
                    "interval_method": interval_method,
                    "interval_distribution": interval_distribution,
                    "interval_status": interval_status,
                    "interval_success": interval_success,
                    "interval_failure_reason": interval_failure_reason,
                    "prediction_se": prediction_se,
                    "pi_true_low": pi_true_low,
                    "pi_true_high": pi_true_high,
                    "pi_low": pi_low,
                    "pi_high": pi_high,
                    "pi_true_width": pi_true_high - pi_true_low,
                    "pi_observed_width": pi_high - pi_low,
                    "covered": bool(pi_low <= target.estimate <= pi_high)
                    if interval_success
                    else float("nan"),
                    "discrepancy": float(target.estimate - atlas_estimate),
                    "standardized_discrepancy": standardized_discrepancy,
                    "standardized_discrepancy_method": standardized_discrepancy_method,
                    "predictive_pit": predictive_pit,
                    "interval_relative_discrepancy": float(
                        (target.estimate - pi_midpoint) / pi_half_width
                    )
                    if interval_success and pi_half_width > 0
                    else float("nan"),
                    "sign_agreement": bool(np.sign(target.estimate) == np.sign(atlas_estimate)),
                    "tau2_zero_boundary_mass": zero_boundary_mass,
                    "tau2_cd_median": tau2_cd_median,
                    "tau2_cd_upper_95": tau2_cd_upper_95,
                    "nnf_inversion_probability_error": nnf_probability_error,
                    "nnf_cdf_truncation_error": nnf_cdf_error,
                    "nnf_latent_width_to_fixed_mean_ci_width": low_dispersion_width_ratio,
                    "nnf_low_dispersion_width_warning": low_dispersion_width_warning,
                    "observed_interval_conditioning": "retrospective_known_held_out_target_se",
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


def _safe_median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else float("nan")


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


def _errors_in_variables_calibration(
    x: np.ndarray,
    y: np.ndarray,
    x_variance: np.ndarray,
    y_variance: np.ndarray,
) -> tuple[float, float]:
    """York heteroscedastic errors-in-variables fit with uncorrelated errors."""
    keep = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(x_variance)
        & np.isfinite(y_variance)
        & (x_variance > 0)
        & (y_variance > 0)
    )
    x, y = x[keep], y[keep]
    x_variance, y_variance = x_variance[keep], y_variance[keep]
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan")

    weight_x = 1.0 / x_variance
    weight_y = 1.0 / y_variance
    slope = float(np.polyfit(x, y, 1)[0])
    converged = False
    for _ in range(1_000):
        weights = weight_x * weight_y / (weight_x + slope**2 * weight_y)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            return float("nan"), float("nan")
        x_mean = float(np.sum(weights * x) / weight_sum)
        y_mean = float(np.sum(weights * y) / weight_sum)
        x_centered = x - x_mean
        y_centered = y - y_mean
        beta = weights * (x_centered / weight_y + slope * y_centered / weight_x)
        denominator = float(np.sum(weights * beta * x_centered))
        if not np.isfinite(denominator) or abs(denominator) <= np.finfo(float).tiny:
            return float("nan"), float("nan")
        updated = float(np.sum(weights * beta * y_centered) / denominator)
        if not np.isfinite(updated):
            return float("nan"), float("nan")
        if abs(updated - slope) <= 1e-12 * max(1.0, abs(slope)):
            slope = updated
            converged = True
            break
        slope = updated
    if not converged:
        return float("nan"), float("nan")

    weights = weight_x * weight_y / (weight_x + slope**2 * weight_y)
    x_mean = float(np.sum(weights * x) / np.sum(weights))
    y_mean = float(np.sum(weights * y) / np.sum(weights))
    return y_mean - slope * x_mean, slope


def calibration_summary(
    predictions: pd.DataFrame,
    *,
    strong_threshold: float = 0.5,
) -> pd.DataFrame:
    """Summarize calibration overall and by target-study/stratum."""
    if predictions.empty:
        raise ValueError("Prediction table must contain at least one row")
    if not isinstance(strong_threshold, (int, float, np.integer, np.floating)) or not (
        np.isfinite(strong_threshold) and strong_threshold >= 0
    ):
        raise ValueError("strong_threshold must be finite and non-negative")
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
        "standardized_discrepancy_method",
        "interval_relative_discrepancy",
        "pi_true_width",
        "pi_observed_width",
        "interval_success",
        "interval_method",
        "point_method",
        "tau2",
        "cochran_q",
        "cochran_q_df",
        "cochran_q_pvalue",
        "nnf_low_dispersion_width_warning",
        "observed_interval_conditioning",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    def one(group: pd.DataFrame, scope: str, aggregation_unit: str) -> dict[str, float | int | str]:
        direct = group["direct_estimate"]
        atlas = group["atlas_estimate"]
        strong = group.loc[group["atlas_estimate"].abs().ge(strong_threshold)]
        successful = group.loc[group["interval_success"]]
        nnf_width_warnings = group["nnf_low_dispersion_width_warning"].dropna()
        n_success = len(successful)
        n_failed = len(group) - n_success
        n_covered = int(successful["covered"].fillna(False).astype(bool).sum())
        naive_intercept, naive_slope = _weighted_calibration(
            atlas.to_numpy(),
            direct.to_numpy(),
            (group["direct_se"] ** 2 + group["atlas_se"] ** 2).to_numpy(),
        )
        eiv_intercept, eiv_slope = _errors_in_variables_calibration(
            atlas.to_numpy(),
            direct.to_numpy(),
            group["atlas_se"].pow(2).to_numpy(),
            (group["direct_se"].pow(2) + group["tau2"]).to_numpy(),
        )
        return {
            "scope": scope,
            "aggregation_unit": aggregation_unit,
            "interval_method": ",".join(sorted(group["interval_method"].unique())),
            "point_method": ",".join(sorted(group["point_method"].unique())),
            "standardized_discrepancy_method": ",".join(
                sorted(group["standardized_discrepancy_method"].unique())
            ),
            "observed_interval_conditioning": ",".join(
                sorted(group["observed_interval_conditioning"].unique())
            ),
            "n_studies": int(group["target_study"].nunique()),
            "n_study_strata": int(group[["target_study", "stratum"]].drop_duplicates().shape[0]),
            "n": len(group),
            "pearson": _safe_correlation(atlas, direct, "pearson"),
            "spearman": _safe_correlation(atlas, direct, "spearman"),
            "sign_agreement": float(group["sign_agreement"].mean()),
            "strong_sign_agreement": float(strong["sign_agreement"].mean())
            if len(strong)
            else np.nan,
            "n_strong": len(strong),
            "coverage_success_conditional": float(successful["covered"].astype(float).mean())
            if n_success
            else np.nan,
            "coverage_failure_as_miss_lower": float(n_covered / len(group)),
            "coverage_failure_as_hit_upper": float((n_covered + n_failed) / len(group)),
            "coverage_status": "conditional_on_numerical_success_with_failure_bounds",
            "n_interval_success": n_success,
            "n_interval_failed": n_failed,
            "interval_failure_rate": float(n_failed / len(group)),
            "median_pi_true_width": _safe_median(group["pi_true_width"]),
            "median_pi_observed_width": _safe_median(group["pi_observed_width"]),
            "mean_bias": float(group["discrepancy"].mean()),
            "mae": float(group["discrepancy"].abs().mean()),
            "rmse": float(np.sqrt(np.mean(group["discrepancy"] ** 2))),
            "median_abs_z": _safe_median(group["standardized_discrepancy"].abs()),
            "median_abs_interval_relative_discrepancy": _safe_median(
                group["interval_relative_discrepancy"].abs()
            ),
            "median_cochran_q": float(group["cochran_q"].median()),
            "median_cochran_q_pvalue": float(group["cochran_q_pvalue"].median()),
            "cochran_q_p_lt_0_05_rate": float(group["cochran_q_pvalue"].lt(0.05).mean()),
            "nnf_low_dispersion_width_warning_rate": (
                float(nnf_width_warnings.astype(float).mean())
                if len(nnf_width_warnings)
                else np.nan
            ),
            "eiv_calibration_intercept": eiv_intercept,
            "eiv_calibration_slope": eiv_slope,
            "eiv_calibration_status": (
                "descriptive_y_variance_includes_target_tau2_genes_not_independent"
            ),
            "naive_wls_calibration_intercept_sensitivity": naive_intercept,
            "naive_wls_calibration_slope_sensitivity": naive_slope,
        }

    study_stratum_rows = []
    for (study, stratum), group in predictions.groupby(["target_study", "stratum"], sort=True):
        study_stratum_rows.append(
            one(
                group,
                f"study={study};stratum={stratum}",
                "within_target_study_stratum_gene",
            )
        )

    study_rows = []
    for study, group in predictions.groupby("target_study", sort=True):
        study_rows.append(
            one(
                group,
                f"study={study}",
                "within_target_study_gene_stratum",
            )
        )
    study_frame = pd.DataFrame(study_rows)
    study_stratum_frame = pd.DataFrame(study_stratum_rows)
    micro = one(predictions, "overall_micro_gene_fold", "gene_fold_weighted_descriptive")

    macro_metrics = [
        "pearson",
        "spearman",
        "sign_agreement",
        "strong_sign_agreement",
        "coverage_success_conditional",
        "coverage_failure_as_miss_lower",
        "coverage_failure_as_hit_upper",
        "interval_failure_rate",
        "median_pi_true_width",
        "median_pi_observed_width",
        "mean_bias",
        "mae",
        "rmse",
        "median_abs_z",
        "median_abs_interval_relative_discrepancy",
        "median_cochran_q",
        "median_cochran_q_pvalue",
        "cochran_q_p_lt_0_05_rate",
        "nnf_low_dispersion_width_warning_rate",
        "eiv_calibration_intercept",
        "eiv_calibration_slope",
        "naive_wls_calibration_intercept_sensitivity",
        "naive_wls_calibration_slope_sensitivity",
    ]
    macro: dict[str, float | int | str] = {
        "scope": "overall_macro_study",
        "aggregation_unit": "equal_target_study",
        "interval_method": micro["interval_method"],
        "point_method": micro["point_method"],
        "standardized_discrepancy_method": micro["standardized_discrepancy_method"],
        "observed_interval_conditioning": micro["observed_interval_conditioning"],
        "n_studies": len(study_frame),
        "n_study_strata": int(study_stratum_frame.shape[0]),
        "n": int(study_frame["n"].sum()),
        "n_strong": int(study_frame["n_strong"].sum()),
        "coverage_status": "equal_study_mean_with_success_conditional_and_failure_bounds",
        "n_interval_success": int(study_frame["n_interval_success"].sum()),
        "n_interval_failed": int(study_frame["n_interval_failed"].sum()),
        "eiv_calibration_status": micro["eiv_calibration_status"],
    }
    for metric in macro_metrics:
        macro[metric] = float(study_frame[metric].mean())

    return pd.concat(
        [pd.DataFrame([macro, micro]), study_frame, study_stratum_frame],
        ignore_index=True,
        sort=False,
    )
