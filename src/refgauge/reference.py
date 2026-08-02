"""Reference-anchor sensitivity analysis."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .validation import validate_references

MIN_SHARED_FEATURES_FOR_RANK = 10

FEATURE_SENSITIVITY_COLUMNS = [
    "target",
    "feature",
    "n_references",
    "estimate_min",
    "estimate_max",
    "estimate_range",
    "estimate_sd",
    "mean_abs_estimate",
    "sign_stability",
    "raw_sign_flip_noise_unadjusted",
    "n_pairwise_comparisons",
    "inference_status",
]

RANK_STABILITY_COLUMNS = [
    "target",
    "reference_1",
    "reference_2",
    "n_shared_features",
    "min_shared_features_required",
    "rank_metrics_reported",
    "spearman_rank",
    "top10_jaccard",
]


def _compare_reference_pairs(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in x.groupby("target", sort=True):
        references = sorted(group["reference"].unique())
        for left, right in itertools.combinations(references, 2):
            left_data = group.loc[
                group["reference"].eq(left), ["feature", "estimate", "se"]
            ].rename(columns={"estimate": "estimate_1", "se": "se_1"})
            right_data = group.loc[
                group["reference"].eq(right), ["feature", "estimate", "se"]
            ].rename(columns={"estimate": "estimate_2", "se": "se_2"})
            pair = left_data.merge(right_data, on="feature", how="inner")
            for record in pair.itertuples(index=False):
                difference = float(record.estimate_1 - record.estimate_2)
                rows.append(
                    {
                        "target": target,
                        "feature": record.feature,
                        "reference_1": left,
                        "reference_2": right,
                        "estimate_1": float(record.estimate_1),
                        "estimate_2": float(record.estimate_2),
                        "se_1": float(record.se_1),
                        "se_2": float(record.se_2),
                        "estimate_difference": difference,
                        "raw_sign_change_noise_unadjusted": bool(
                            record.estimate_1 * record.estimate_2 < 0
                        ),
                        "inference_status": (
                            "descriptive_only_shared_target_covariance_not_supplied"
                        ),
                    }
                )

    columns = [
        "target",
        "feature",
        "reference_1",
        "reference_2",
        "estimate_1",
        "estimate_2",
        "se_1",
        "se_2",
        "estimate_difference",
        "raw_sign_change_noise_unadjusted",
        "inference_status",
    ]
    comparisons = pd.DataFrame(rows)
    return comparisons.loc[:, columns] if not comparisons.empty else pd.DataFrame(columns=columns)


def compare_reference_pairs(data: pd.DataFrame) -> pd.DataFrame:
    """Describe pairwise anchor changes without invalid shared-target inference."""
    return _compare_reference_pairs(validate_references(data))


def audit_references(
    data: pd.DataFrame,
    *,
    min_shared_features: int = MIN_SHARED_FEATURES_FOR_RANK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize anchor sensitivity without assuming shared-target covariance."""
    if isinstance(min_shared_features, (bool, np.bool_)) or not isinstance(
        min_shared_features, (int, np.integer)
    ):
        raise ValueError("min_shared_features must be an integer")
    if min_shared_features < 2:
        raise ValueError("min_shared_features must be at least 2")
    x = validate_references(data)
    comparisons = _compare_reference_pairs(x)
    comparison_counts = (
        comparisons.groupby(["target", "feature"], sort=False).size().to_dict()
        if not comparisons.empty
        else {}
    )
    feature_rows = []
    for (target, feature), group in x.groupby(["target", "feature"], sort=False):
        if len(group) < 2:
            continue
        values = group["estimate"].to_numpy(float)
        signs = np.sign(values)
        nonzero = signs[signs != 0]
        raw_range = float(np.ptp(values))
        raw_sign_flip = bool(len(np.unique(nonzero)) > 1)
        feature_rows.append(
            {
                "target": target,
                "feature": feature,
                "n_references": len(group),
                "estimate_min": float(np.min(values)),
                "estimate_max": float(np.max(values)),
                "estimate_range": raw_range,
                "estimate_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "mean_abs_estimate": float(np.mean(np.abs(values))),
                "sign_stability": float(max(np.mean(nonzero > 0), np.mean(nonzero < 0)))
                if len(nonzero)
                else 1.0,
                "raw_sign_flip_noise_unadjusted": raw_sign_flip,
                "n_pairwise_comparisons": int(comparison_counts.get((target, feature), 0)),
                "inference_status": "descriptive_only_shared_target_covariance_not_supplied",
            }
        )
    feature_sensitivity = pd.DataFrame(feature_rows, columns=FEATURE_SENSITIVITY_COLUMNS)
    feature_sensitivity.attrs["raw_metrics_are_noise_unadjusted"] = True

    rank_rows = []
    for target, group in x.groupby("target", sort=True):
        wide = group.pivot(index="feature", columns="reference", values="estimate")
        for left, right in itertools.combinations(wide.columns, 2):
            pair = wide[[left, right]].dropna()
            eligible = len(pair) >= min_shared_features
            if eligible:
                rho = pair[left].corr(pair[right], method="spearman")
                top_n = max(1, int(np.ceil(0.1 * len(pair))))
                top_left = set(pair[left].abs().nlargest(top_n).index)
                top_right = set(pair[right].abs().nlargest(top_n).index)
                union = top_left | top_right
                top10_jaccard = len(top_left & top_right) / len(union) if union else np.nan
            else:
                rho = np.nan
                top10_jaccard = np.nan
            rank_rows.append(
                {
                    "target": target,
                    "reference_1": left,
                    "reference_2": right,
                    "n_shared_features": len(pair),
                    "min_shared_features_required": min_shared_features,
                    "rank_metrics_reported": eligible,
                    "spearman_rank": float(rho),
                    "top10_jaccard": top10_jaccard,
                }
            )
    return feature_sensitivity, pd.DataFrame(rank_rows, columns=RANK_STABILITY_COLUMNS)
