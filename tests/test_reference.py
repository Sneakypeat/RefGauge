import numpy as np
import pandas as pd

from refgauge.reference import (
    MIN_SHARED_FEATURES_FOR_RANK,
    audit_references,
    compare_reference_pairs,
)
from refgauge.simulation import simulate_dataset


def test_reference_audit_detects_sensitivity():
    _, references = simulate_dataset(n_features=50, n_studies=4, seed=31)
    sensitivity, ranks = audit_references(references)
    assert len(sensitivity) == 100
    assert len(ranks) == 6
    assert sensitivity["estimate_range"].median() > 0
    assert ranks["spearman_rank"].between(-1, 1).all()
    assert ranks["top10_jaccard"].between(0, 1).all()


def test_reference_sign_flip_is_reported():
    _, references = simulate_dataset(n_features=20, n_studies=4, seed=41)
    sensitivity, _ = audit_references(references)
    assert sensitivity["raw_sign_flip_noise_unadjusted"].dtype == bool
    assert sensitivity["raw_sign_flip_noise_unadjusted"].any()
    assert sensitivity["inference_status"].str.startswith("descriptive_only").all()


def test_one_reference_feature_is_not_reported_as_stable():
    references = pd.DataFrame(
        {
            "feature": ["eligible", "eligible", "one_only"],
            "target": ["target"] * 3,
            "reference": ["ref_a", "ref_b", "ref_a"],
            "estimate": [0.2, 0.3, 1.0],
            "se": [0.1, 0.1, 0.1],
        }
    )
    sensitivity, _ = audit_references(references)
    assert sensitivity["feature"].tolist() == ["eligible"]
    assert sensitivity["n_references"].eq(2).all()


def test_no_shared_features_returns_stable_empty_schema():
    references = pd.DataFrame(
        {
            "feature": ["only_a", "only_b"],
            "target": ["target", "target"],
            "reference": ["ref_a", "ref_b"],
            "estimate": [0.2, 0.3],
            "se": [0.1, 0.1],
        }
    )
    sensitivity, ranks = audit_references(references)
    assert sensitivity.empty
    assert "estimate_range" in sensitivity
    assert len(ranks) == 1
    assert not ranks.loc[0, "rank_metrics_reported"]


def test_low_overlap_rank_metrics_are_withheld():
    n_shared = MIN_SHARED_FEATURES_FOR_RANK - 1
    references = pd.DataFrame(
        {
            "feature": [f"gene_{i}" for i in range(n_shared)] * 2,
            "target": ["target"] * (2 * n_shared),
            "reference": ["ref_a"] * n_shared + ["ref_b"] * n_shared,
            "estimate": list(range(n_shared)) + list(reversed(range(n_shared))),
            "se": [0.1] * (2 * n_shared),
        }
    )
    _, ranks = audit_references(references)
    row = ranks.iloc[0]
    assert row["n_shared_features"] == n_shared
    assert not row["rank_metrics_reported"]
    assert np.isnan(row["spearman_rank"])
    assert np.isnan(row["top10_jaccard"])


def test_pairwise_comparison_is_descriptive_without_shared_target_covariance():
    references = pd.DataFrame(
        {
            "feature": ["precise", "precise", "noisy", "noisy"],
            "target": ["target"] * 4,
            "reference": ["ref_a", "ref_b"] * 2,
            "estimate": [1.0, -1.0, 0.1, -0.1],
            "se": [0.1, 0.1, 10.0, 10.0],
        }
    )
    comparisons = compare_reference_pairs(references).set_index("feature")

    assert np.isclose(comparisons.loc["precise", "estimate_difference"], 2.0)
    assert "difference_se" not in comparisons
    assert "p_value" not in comparisons
    assert "q_value" not in comparisons
    assert comparisons.loc["noisy", "raw_sign_change_noise_unadjusted"]
    assert comparisons["inference_status"].str.startswith("descriptive_only").all()

    sensitivity, _ = audit_references(references)
    summary = sensitivity.set_index("feature")
    assert summary.loc["precise", "raw_sign_flip_noise_unadjusted"]
    assert summary.loc["noisy", "raw_sign_flip_noise_unadjusted"]


def test_pairwise_count_is_precomputed_per_target_feature():
    references = pd.DataFrame(
        [
            {
                "feature": feature,
                "target": target,
                "reference": reference,
                "estimate": estimate,
                "se": 0.1,
            }
            for target in ("target_a", "target_b")
            for feature in ("gene_a", "gene_b")
            for reference, estimate in (("ref_a", 0.1), ("ref_b", 0.2), ("ref_c", 0.3))
        ]
    )
    sensitivity, _ = audit_references(references)
    assert len(sensitivity) == 4
    assert sensitivity["n_pairwise_comparisons"].eq(3).all()
