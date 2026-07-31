"""Input contracts and explicit failure messages."""

from __future__ import annotations

import pandas as pd

EFFECT_COLUMNS = {"feature", "study", "estimate", "se"}
REFERENCE_COLUMNS = {"feature", "target", "reference", "estimate", "se"}


def validate_effects(data: pd.DataFrame) -> pd.DataFrame:
    missing = EFFECT_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Effects table is missing columns: {sorted(missing)}")
    out = data.copy()
    out["feature"] = out["feature"].astype(str)
    out["study"] = out["study"].astype(str)
    out["estimate"] = pd.to_numeric(out["estimate"], errors="coerce")
    out["se"] = pd.to_numeric(out["se"], errors="coerce")
    if "stratum" not in out:
        out["stratum"] = "all"
    out["stratum"] = out["stratum"].fillna("all").astype(str)
    if out[["feature", "study", "stratum"]].duplicated().any():
        raise ValueError("Each feature-study-stratum combination must occur once")
    if out[["estimate", "se"]].isna().any().any() or (out["se"] <= 0).any():
        raise ValueError("estimate and se must be finite; se must be greater than zero")
    if out["study"].nunique() < 3:
        raise ValueError("At least three studies are required for leave-one-study-out calibration")
    return out


def validate_references(data: pd.DataFrame) -> pd.DataFrame:
    missing = REFERENCE_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Reference table is missing columns: {sorted(missing)}")
    out = data.copy()
    for col in ("feature", "target", "reference"):
        out[col] = out[col].astype(str)
    out["estimate"] = pd.to_numeric(out["estimate"], errors="coerce")
    out["se"] = pd.to_numeric(out["se"], errors="coerce")
    if out[["feature", "target", "reference"]].duplicated().any():
        raise ValueError("Each feature-target-reference combination must occur once")
    if out[["estimate", "se"]].isna().any().any() or (out["se"] <= 0).any():
        raise ValueError("estimate and se must be finite; se must be greater than zero")
    if out["reference"].nunique() < 2:
        raise ValueError("At least two references are required for a reference audit")
    return out
