import numpy as np
import pandas as pd
import pytest

from refgauge.validation import validate_effects, validate_references


def effects_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["gene_a"] * 3,
            "study": ["study_1", "study_2", "study_3"],
            "estimate": [0.1, 0.2, 0.3],
            "se": [0.1, 0.1, 0.1],
        }
    )


def references_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["gene_a", "gene_a"],
            "target": ["target_1", "target_1"],
            "reference": ["reference_1", "reference_2"],
            "estimate": [0.1, 0.2],
            "se": [0.1, 0.1],
        }
    )


@pytest.mark.parametrize("column", ["estimate", "se"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_effects_reject_nonfinite_values(column, value):
    data = effects_table()
    data.loc[0, column] = value
    with pytest.raises(ValueError, match="must be finite"):
        validate_effects(data)


@pytest.mark.parametrize("column", ["feature", "study"])
@pytest.mark.parametrize("value", [None, "   "])
def test_effects_reject_missing_or_blank_identifiers(column, value):
    data = effects_table()
    data.loc[0, column] = value
    with pytest.raises(ValueError, match="missing or blank"):
        validate_effects(data)


@pytest.mark.parametrize("column", ["estimate", "se"])
def test_references_reject_nonfinite_values(column):
    data = references_table()
    data.loc[0, column] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        validate_references(data)


@pytest.mark.parametrize("column", ["feature", "target", "reference"])
@pytest.mark.parametrize("value", [pd.NA, "\t"])
def test_references_reject_missing_or_blank_identifiers(column, value):
    data = references_table()
    data.loc[0, column] = value
    with pytest.raises(ValueError, match="missing or blank"):
        validate_references(data)


def test_references_require_two_references_for_every_target():
    data = pd.concat(
        [
            references_table(),
            pd.DataFrame(
                {
                    "feature": ["gene_b"],
                    "target": ["target_2"],
                    "reference": ["reference_1"],
                    "estimate": [0.3],
                    "se": [0.1],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="target_2"):
        validate_references(data)
