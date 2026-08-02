"""Input contracts and explicit failure messages."""

from __future__ import annotations

import numpy as np
import pandas as pd

EFFECT_COLUMNS = {"feature", "study", "estimate", "se"}
REFERENCE_COLUMNS = {"feature", "target", "reference", "estimate", "se"}


def _normalise_identifiers(data: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Strip identifiers in place and reject missing or blank values."""
    for column in columns:
        values = data[column].astype("string").str.strip()
        invalid = values.isna() | values.eq("")
        if invalid.any():
            raise ValueError(f"Identifier column {column!r} contains missing or blank values")
        data[column] = values.astype(str)


def _validate_numeric_columns(data: pd.DataFrame) -> None:
    values = data[["estimate", "se"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (data["se"] <= 0).any():
        raise ValueError("estimate and se must be finite; se must be greater than zero")


def validate_effects(data: pd.DataFrame) -> pd.DataFrame:
    missing = EFFECT_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Effects table is missing columns: {sorted(missing)}")
    out = data.copy()
    if "stratum" not in out:
        out["stratum"] = "all"
    _normalise_identifiers(out, ("feature", "study", "stratum"))
    out["estimate"] = pd.to_numeric(out["estimate"], errors="coerce")
    out["se"] = pd.to_numeric(out["se"], errors="coerce")
    if out[["feature", "study", "stratum"]].duplicated().any():
        raise ValueError("Each feature-study-stratum combination must occur once")
    _validate_numeric_columns(out)
    if out["study"].nunique() < 3:
        raise ValueError("At least three studies are required for leave-one-study-out calibration")
    return out


def validate_references(data: pd.DataFrame) -> pd.DataFrame:
    missing = REFERENCE_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Reference table is missing columns: {sorted(missing)}")
    out = data.copy()
    _normalise_identifiers(out, ("feature", "target", "reference"))
    out["estimate"] = pd.to_numeric(out["estimate"], errors="coerce")
    out["se"] = pd.to_numeric(out["se"], errors="coerce")
    if out[["feature", "target", "reference"]].duplicated().any():
        raise ValueError("Each feature-target-reference combination must occur once")
    _validate_numeric_columns(out)
    reference_counts = out.groupby("target", sort=False)["reference"].nunique()
    invalid_targets = reference_counts.index[reference_counts.lt(2)].tolist()
    if invalid_targets:
        raise ValueError(
            f"Each target requires at least two references; invalid targets: {invalid_targets}"
        )
    return out
