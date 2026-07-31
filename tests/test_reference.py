from contrastcal.reference import audit_references
from contrastcal.simulation import simulate_dataset


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
    assert sensitivity["sign_flip"].dtype == bool
    assert sensitivity["sign_flip"].any()
