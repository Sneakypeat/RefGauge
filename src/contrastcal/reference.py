"""Reference-anchor sensitivity analysis."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .validation import validate_references


def audit_references(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify magnitude/sign sensitivity and reference-dependent rank stability."""
    x = validate_references(data)
    feature_rows = []
    for (target, feature), group in x.groupby(["target", "feature"], sort=False):
        values = group["estimate"].to_numpy(float)
        signs = np.sign(values)
        nonzero = signs[signs != 0]
        feature_rows.append(
            {
                "target": target,
                "feature": feature,
                "n_references": len(group),
                "estimate_min": float(np.min(values)),
                "estimate_max": float(np.max(values)),
                "estimate_range": float(np.ptp(values)),
                "estimate_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "mean_abs_estimate": float(np.mean(np.abs(values))),
                "sign_stability": float(max(np.mean(nonzero > 0), np.mean(nonzero < 0)))
                if len(nonzero)
                else 1.0,
                "sign_flip": bool(len(np.unique(nonzero)) > 1),
            }
        )
    feature_sensitivity = pd.DataFrame(feature_rows)

    rank_rows = []
    for target, group in x.groupby("target", sort=True):
        wide = group.pivot(index="feature", columns="reference", values="estimate")
        for left, right in itertools.combinations(wide.columns, 2):
            pair = wide[[left, right]].dropna()
            rho = pair[left].corr(pair[right], method="spearman") if len(pair) >= 3 else np.nan
            top_n = max(1, int(np.ceil(0.1 * len(pair))))
            top_left = set(pair[left].abs().nlargest(top_n).index)
            top_right = set(pair[right].abs().nlargest(top_n).index)
            union = top_left | top_right
            rank_rows.append(
                {
                    "target": target,
                    "reference_1": left,
                    "reference_2": right,
                    "n_shared_features": len(pair),
                    "spearman_rank": float(rho),
                    "top10_jaccard": len(top_left & top_right) / len(union) if union else np.nan,
                }
            )
    return feature_sensitivity, pd.DataFrame(rank_rows)
