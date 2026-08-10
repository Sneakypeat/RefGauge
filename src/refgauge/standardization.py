"""Finite-reference standardization with explicit abstention gates.

This module is intentionally separate from the descriptive reference audit.  It
only emits a standardised finite-family discrepancy when direct crossed data,
source-balanced primary baselines, joint bootstrap uncertainty, and a frozen
reference-stability tolerance all support that interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EDGE_COLUMNS = (
    "domain",
    "feature",
    "edge_type",
    "target",
    "target_unit",
    "target_source",
    "reference",
    "reference_source",
    "estimate",
    "admissible",
    "leakage",
)
EDGE_KEY_COLUMNS = EDGE_COLUMNS[:8]

DESIGN_COLUMNS = (
    "domain",
    "feature",
    "edge_type",
    "target",
    "target_unit",
    "target_source",
    "reference_source",
)

SCORE_COLUMNS = (
    "domain",
    "feature",
    "target",
    "delta",
    "ci_low",
    "ci_high",
    "one_sided_low",
    "one_sided_high",
    "anchor_dependence",
    "anchor_dependence_upper",
    "max_source_deletion_shift",
    "max_source_deletion_shift_upper",
    "n_primary_sources",
    "n_target_sources",
    "n_complete_bootstrap_draws",
    "complete_bootstrap_fraction",
    "delta_ref",
    "delta_bio",
    "status",
    "biological_decision",
    "inference_scope",
    "descriptive_only",
    "failed_gates",
)

ANCHOR_COLUMNS = (
    "domain",
    "feature",
    "target",
    "reference",
    "reference_source",
    "anchor_excess",
    "delta",
    "absolute_deviation_from_delta",
    "status",
    "descriptive_only",
)

BASELINE_COLUMNS = (
    "domain",
    "feature",
    "target",
    "reference",
    "reference_source",
    "baseline",
    "n_contributing_primary_sources",
    "status",
    "descriptive_only",
)

GATE_COLUMNS = (
    "domain",
    "feature",
    "target",
    "gate_id",
    "passed",
    "observed",
    "required",
    "failure_status",
    "detail",
    "headline_status",
)

DELETION_COLUMNS = (
    "domain",
    "feature",
    "target",
    "deleted_primary_source",
    "delta_full",
    "delta_deleted",
    "absolute_shift",
    "absolute_shift_upper",
    "point_within_delta_ref",
    "within_delta_ref",
    "full_biological_decision",
    "deleted_biological_decision",
    "decision_stable",
    "status",
    "descriptive_only",
)


@dataclass(frozen=True)
class ReferenceStandardizationResult:
    """Tables returned by :func:`standardize_reference_family`."""

    scores: pd.DataFrame
    anchor_excess: pd.DataFrame
    reference_baselines: pd.DataFrame
    gates: pd.DataFrame
    source_deletions: pd.DataFrame


@dataclass(frozen=True)
class _PointMetrics:
    delta: float
    anchor_dependence: float
    anchors: pd.DataFrame
    baselines: pd.DataFrame


@dataclass(frozen=True)
class _BootstrapMetrics:
    delta: pd.Series
    anchor_dependence: pd.Series


def _empty_result() -> ReferenceStandardizationResult:
    return ReferenceStandardizationResult(
        pd.DataFrame(columns=SCORE_COLUMNS),
        pd.DataFrame(columns=ANCHOR_COLUMNS),
        pd.DataFrame(columns=BASELINE_COLUMNS),
        pd.DataFrame(columns=GATE_COLUMNS),
        pd.DataFrame(columns=DELETION_COLUMNS),
    )


def _validate_positive_integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _validate_probability(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be numeric")
    out = float(value)
    if not np.isfinite(out) or not 0 < out < 0.5:
        raise ValueError(f"{name} must be finite and between zero and 0.5")
    return out


def _validate_tolerance(value: Any, name: str, *, allow_none: bool) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        suffix = " or None" if allow_none else ""
        raise ValueError(f"{name} must be a positive finite number{suffix}")
    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return out


def _normalise_identifiers(data: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = data[column].astype("string").str.strip()
        invalid = values.isna() | values.eq("")
        if invalid.any():
            raise ValueError(f"Identifier column {column!r} contains missing or blank values")
        data[column] = values.astype(str)


def _validate_boolean(data: pd.DataFrame, column: str) -> None:
    valid = data[column].map(lambda value: isinstance(value, (bool, np.bool_)))
    if not valid.all():
        raise ValueError(f"Column {column!r} must contain booleans")
    data[column] = data[column].astype(bool)


def _validate_edges(
    data: pd.DataFrame,
    *,
    table_name: str,
    require_unique_edges: bool = True,
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{table_name} must be a pandas DataFrame")
    if data.empty:
        raise ValueError(f"{table_name} must contain at least one row")
    missing = set(EDGE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"{table_name} is missing columns: {sorted(missing)}")

    out = data.copy()
    _normalise_identifiers(out, EDGE_KEY_COLUMNS)
    if not out["edge_type"].isin({"OP", "PP"}).all():
        invalid = sorted(out.loc[~out["edge_type"].isin({"OP", "PP"}), "edge_type"].unique())
        raise ValueError(f"edge_type must be 'OP' or 'PP'; found {invalid}")
    out["estimate"] = pd.to_numeric(out["estimate"], errors="coerce")
    if not np.isfinite(out["estimate"].to_numpy(float)).all():
        raise ValueError("estimate must contain only finite numeric values")
    _validate_boolean(out, "admissible")
    _validate_boolean(out, "leakage")
    if require_unique_edges and out[list(EDGE_KEY_COLUMNS)].duplicated().any():
        raise ValueError(f"{table_name} contains duplicate edge keys")
    if set(out["edge_type"]) != {"OP", "PP"}:
        raise ValueError(f"{table_name} must contain both OP and PP edges")
    return out


def _validate_design(data: pd.DataFrame | None) -> pd.DataFrame | None:
    if data is None:
        return None
    if not isinstance(data, pd.DataFrame):
        raise TypeError("design must be a pandas DataFrame or None")
    if data.empty:
        raise ValueError("design must contain at least one expected cell")
    missing = set(DESIGN_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"design is missing columns: {sorted(missing)}")

    out = data.loc[:, list(DESIGN_COLUMNS)].copy()
    _normalise_identifiers(out, DESIGN_COLUMNS)
    if not out["edge_type"].isin({"OP", "PP"}).all():
        invalid = sorted(out.loc[~out["edge_type"].isin({"OP", "PP"}), "edge_type"].unique())
        raise ValueError(f"design edge_type must be 'OP' or 'PP'; found {invalid}")
    if out[list(DESIGN_COLUMNS)].duplicated().any():
        raise ValueError("design contains duplicate expected-cell identifiers")

    edge_types = out.groupby(["domain", "feature"], sort=False)["edge_type"].agg(set)
    if not edge_types.map(lambda values: values == {"OP", "PP"}).all():
        raise ValueError("every design domain-feature group must contain both OP and PP cells")

    same_source_pp = out["edge_type"].eq("PP") & out["target_source"].eq(
        out["reference_source"]
    )
    if same_source_pp.any():
        raise ValueError("design must not contain same-source PP cells")

    unit_targets = out.groupby(
        ["domain", "feature", "edge_type", "target_source", "target_unit"],
        sort=False,
    )["target"].nunique()
    if (unit_targets > 1).any():
        raise ValueError("each design source-level biological unit must have one target identifier")
    return out


def _assert_observed_cells_declared(edges: pd.DataFrame, design: pd.DataFrame) -> None:
    same_source_pp = edges["edge_type"].eq("PP") & edges["target_source"].eq(
        edges["reference_source"]
    )
    admitted = edges.loc[
        edges["admissible"] & ~same_source_pp, list(DESIGN_COLUMNS)
    ].drop_duplicates()
    if admitted.empty:
        return
    checked = admitted.merge(
        design.assign(_declared=True),
        on=list(DESIGN_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    undeclared = checked["_declared"].isna()
    if undeclared.any():
        examples = checked.loc[undeclared, list(DESIGN_COLUMNS)].head(3).to_dict("records")
        raise ValueError(
            f"{int(undeclared.sum())} admitted observed source-level cells are not present "
            f"in design; examples: {examples}"
        )


def _diagnostic_design(edges: pd.DataFrame) -> pd.DataFrame:
    return (
        edges.loc[_eligible(edges), list(DESIGN_COLUMNS)]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _validate_bootstrap(data: pd.DataFrame | None, edges: pd.DataFrame) -> pd.DataFrame | None:
    if data is None:
        return None
    if not isinstance(data, pd.DataFrame):
        raise TypeError("bootstrap must be a pandas DataFrame or None")
    if data.empty:
        return None
    if "draw" not in data:
        raise ValueError("bootstrap is missing column: 'draw'")

    out = _validate_edges(data, table_name="bootstrap", require_unique_edges=False)
    _normalise_identifiers(out, ("draw",))
    draw_keys = ["draw", *EDGE_KEY_COLUMNS]
    if out[draw_keys].duplicated().any():
        raise ValueError("bootstrap contains duplicate draw-edge keys")

    point_attributes = edges[[*EDGE_KEY_COLUMNS, "admissible", "leakage"]]
    checked = out.merge(
        point_attributes,
        on=list(EDGE_KEY_COLUMNS),
        how="left",
        suffixes=("", "_point"),
        indicator=True,
        validate="many_to_one",
    )
    if checked["_merge"].ne("both").any():
        raise ValueError("bootstrap contains an edge key not present in edges")
    mismatch = checked["admissible"].ne(checked["admissible_point"]) | checked["leakage"].ne(
        checked["leakage_point"]
    )
    if mismatch.any():
        raise ValueError("bootstrap admissible/leakage flags must match edges")
    return out


def _eligible(data: pd.DataFrame) -> pd.Series:
    same_source_pp = data["edge_type"].eq("PP") & data["target_source"].eq(
        data["reference_source"]
    )
    return data["admissible"] & ~data["leakage"] & ~same_source_pp


def _point_metrics(data: pd.DataFrame) -> _PointMetrics:
    x = data.loc[_eligible(data)].copy()
    pp = x.loc[x["edge_type"].eq("PP")]
    op = x.loc[x["edge_type"].eq("OP")]

    if pp.empty:
        baselines = pd.DataFrame(
            columns=["reference", "reference_source", "baseline", "n_contributing_primary_sources"]
        )
    else:
        pp_unit = (
            pp.groupby(
                ["reference", "reference_source", "target_source", "target_unit"],
                sort=True,
                as_index=False,
            )["estimate"]
            .mean()
        )
        pp_source = (
            pp_unit.groupby(
                ["reference", "reference_source", "target_source"],
                sort=True,
                as_index=False,
            )["estimate"]
            .mean()
        )
        baselines = (
            pp_source.groupby(["reference", "reference_source"], sort=True)
            .agg(
                baseline=("estimate", "mean"),
                n_contributing_primary_sources=("target_source", "nunique"),
            )
            .reset_index()
        )

    joined = op.merge(
        baselines[["reference", "reference_source", "baseline"]],
        on=["reference", "reference_source"],
        how="inner",
        validate="many_to_one",
    )
    if joined.empty:
        anchors = pd.DataFrame(columns=["reference", "reference_source", "anchor_excess"])
        return _PointMetrics(np.nan, np.nan, anchors, baselines)

    joined["excess"] = joined["estimate"] - joined["baseline"]
    by_unit = (
        joined.groupby(
            ["target_source", "target_unit", "reference", "reference_source"],
            sort=True,
            as_index=False,
        )["excess"]
        .mean()
    )
    by_target_source = (
        by_unit.groupby(
            ["target_source", "reference", "reference_source"],
            sort=True,
            as_index=False,
        )["excess"].mean()
    )
    anchors = (
        by_target_source.groupby(
            ["reference", "reference_source"], sort=True, as_index=False
        )["excess"]
        .mean()
        .rename(columns={"excess": "anchor_excess"})
    )
    source_means = anchors.groupby("reference_source", sort=True)["anchor_excess"].mean()
    delta = float(source_means.mean())
    dependence = float((anchors["anchor_excess"] - delta).abs().max())
    return _PointMetrics(delta, dependence, anchors, baselines)


def _bootstrap_metrics(data: pd.DataFrame) -> _BootstrapMetrics:
    pp = data.loc[data["edge_type"].eq("PP")]
    op = data.loc[data["edge_type"].eq("OP")]
    if pp.empty or op.empty:
        return _BootstrapMetrics(pd.Series(dtype=float), pd.Series(dtype=float))

    pp_unit = (
        pp.groupby(
            ["draw", "reference", "reference_source", "target_source", "target_unit"],
            sort=True,
            as_index=False,
        )["estimate"]
        .mean()
    )
    pp_source = (
        pp_unit.groupby(
            ["draw", "reference", "reference_source", "target_source"],
            sort=True,
            as_index=False,
        )["estimate"]
        .mean()
    )
    baselines = (
        pp_source.groupby(["draw", "reference", "reference_source"], sort=True, as_index=False)[
            "estimate"
        ]
        .mean()
        .rename(columns={"estimate": "baseline"})
    )
    joined = op.merge(
        baselines,
        on=["draw", "reference", "reference_source"],
        how="inner",
        validate="many_to_one",
    )
    joined["excess"] = joined["estimate"] - joined["baseline"]
    by_unit = (
        joined.groupby(
            [
                "draw",
                "target_source",
                "target_unit",
                "reference",
                "reference_source",
            ],
            sort=True,
            as_index=False,
        )["excess"]
        .mean()
    )
    by_target_source = (
        by_unit.groupby(
            ["draw", "target_source", "reference", "reference_source"],
            sort=True,
            as_index=False,
        )["excess"]
        .mean()
    )
    anchors = (
        by_target_source.groupby(
            ["draw", "reference", "reference_source"], sort=True, as_index=False
        )["excess"]
        .mean()
        .rename(columns={"excess": "anchor_excess"})
    )
    source_means = (
        anchors.groupby(["draw", "reference_source"], sort=True, as_index=False)[
            "anchor_excess"
        ].mean()
    )
    delta = source_means.groupby("draw", sort=True)["anchor_excess"].mean().rename("delta")
    centered = anchors.join(delta, on="draw")
    dependence = (
        (centered["anchor_excess"] - centered["delta"])
        .abs()
        .groupby(centered["draw"], sort=True)
        .max()
        .rename("anchor_dependence")
    )
    return _BootstrapMetrics(delta, dependence)


def _decision(draws: pd.Series, delta_bio: float | None, alpha: float) -> tuple[str, float, float]:
    if draws.empty:
        return "NOT_EVALUABLE", np.nan, np.nan
    low = float(draws.quantile(alpha))
    high = float(draws.quantile(1.0 - alpha))
    if delta_bio is None:
        return "NOT_REQUESTED", low, high
    if high <= delta_bio:
        return "WITHIN_ENVELOPE", low, high
    if low > delta_bio:
        return "DIVERGENT", low, high
    return "UNRESOLVED", low, high


def _gate(
    gate_id: str,
    passed: bool,
    observed: Any,
    required: str,
    failure_status: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
        "failure_status": failure_status,
        "detail": detail,
    }


def _headline_status(gates: list[dict[str, Any]], n_target_sources: int) -> str:
    passed = {row["gate_id"]: row["passed"] for row in gates}
    precedence = (
        ("G1_DECLARED_DESIGN", "ABSTAIN_UNDECLARED_REFERENCE_DESIGN"),
        ("G2_LEAKAGE_ADMISSIBILITY", "ABSTAIN_REFERENCE_LEAKAGE"),
        ("G1_PRIMARY_SOURCE_COUNT", "ABSTAIN_INCOMPLETE_REFERENCE_DESIGN"),
        ("G1_DIRECT_OP_CROSSING", "ABSTAIN_INCOMPLETE_REFERENCE_DESIGN"),
        ("G1_DIRECT_PP_CROSSING", "ABSTAIN_INCOMPLETE_REFERENCE_DESIGN"),
        ("G2_BIOLOGICAL_UNITS", "ABSTAIN_INSUFFICIENT_INFORMATION"),
        ("G3_JOINT_BOOTSTRAP", "ABSTAIN_DEPENDENCE_UNRECOVERABLE"),
        ("G3_COVARIANCE_PSD", "ABSTAIN_NUMERICAL_FAILURE"),
        ("G3_CONTRAST_FULL_RANK", "ABSTAIN_NUMERICAL_FAILURE"),
        ("G4_FROZEN_DELTA_REF", "ABSTAIN_UNFROZEN_TOLERANCE"),
        ("G4_REFERENCE_STABILITY", "ABSTAIN_REFERENCE_NONINVARIANT"),
    )
    for gate_id, status in precedence:
        if not passed.get(gate_id, False):
            return status
    return "STANDARDISED" if n_target_sources >= 3 else "PREPARATION_SPECIFIC"


def _complete_bootstrap(
    bootstrap: pd.DataFrame | None,
    relevant: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int, float]:
    if bootstrap is None:
        return pd.DataFrame(), 0, 0, 0.0

    expected = relevant.loc[_eligible(relevant), list(EDGE_KEY_COLUMNS)].drop_duplicates()
    expected_count = len(expected)
    if expected_count == 0:
        return pd.DataFrame(), int(bootstrap["draw"].nunique()), 0, 0.0

    marker = expected.assign(_expected=True)
    candidate = bootstrap.merge(
        marker,
        on=list(EDGE_KEY_COLUMNS),
        how="inner",
        validate="many_to_one",
    )
    total_draws = int(bootstrap["draw"].nunique())
    counts = candidate.groupby("draw", sort=True).size()
    complete_ids = counts.index[counts.eq(expected_count)]
    complete = candidate.loc[candidate["draw"].isin(complete_ids)].drop(columns="_expected")
    n_complete = len(complete_ids)
    fraction = n_complete / total_draws if total_draws else 0.0
    return complete, total_draws, n_complete, float(fraction)


def _crossing_diagnostics(
    relevant: pd.DataFrame,
    expected_design: pd.DataFrame,
    primary_sources: list[str],
) -> tuple[bool, bool, int, int]:
    eligible = relevant.loc[_eligible(relevant)]
    declared_op = expected_design.loc[expected_design["edge_type"].eq("OP")]
    declared_pp = expected_design.loc[expected_design["edge_type"].eq("PP")]

    expected_op = {
        (target, source, unit, reference_source)
        for target, source, unit in declared_op[
            ["target", "target_source", "target_unit"]
        ].drop_duplicates().itertuples(index=False, name=None)
        for reference_source in primary_sources
    }
    observed_op = set(
        eligible.loc[eligible["edge_type"].eq("OP")][
            ["target", "target_source", "target_unit", "reference_source"]
        ].itertuples(index=False, name=None)
    )

    primary_units = declared_pp[
        ["target", "target_source", "target_unit"]
    ].drop_duplicates()
    expected_pp = {
        (target, source, unit, reference_source)
        for target, source, unit in primary_units.itertuples(index=False, name=None)
        for reference_source in primary_sources
        if source != reference_source
    }
    observed_pp = set(
        eligible.loc[eligible["edge_type"].eq("PP")][
            ["target", "target_source", "target_unit", "reference_source"]
        ].itertuples(index=False, name=None)
    )
    return (
        expected_op.issubset(observed_op),
        expected_pp.issubset(observed_pp),
        len(expected_op - observed_op),
        len(expected_pp - observed_pp),
    )


def standardize_reference_family(
    edges: pd.DataFrame,
    *,
    design: pd.DataFrame | None = None,
    bootstrap: pd.DataFrame | None = None,
    delta_ref: float | None = None,
    delta_bio: float | None = None,
    min_primary_sources: int = 6,
    min_target_units: int = 3,
    min_bootstrap_draws: int = 5000,
    alpha: float = 0.05,
) -> ReferenceStandardizationResult:
    """Standardize discrepancies over a declared finite primary-reference family.

    ``design`` declares every expected source-level OP and PP cell independently of
    observed results. It is required for inference; when omitted, RefGauge retains
    point diagnostics but returns a declared-design abstention. ``bootstrap`` must
    contain jointly refitted edge estimates and the same edge
    columns as ``edges``, plus ``draw``.  The function cannot verify how those draws
    were generated; supplying them asserts that biological units and all learned
    metric components were jointly resampled as required by the frozen contract.
    Point diagnostics are retained after any gate failure but are then labelled
    ``descriptive_only``.
    """

    min_primary_sources = _validate_positive_integer(
        min_primary_sources, "min_primary_sources", 6
    )
    min_target_units = _validate_positive_integer(min_target_units, "min_target_units", 3)
    min_bootstrap_draws = _validate_positive_integer(
        min_bootstrap_draws, "min_bootstrap_draws", 5000
    )
    alpha = _validate_probability(alpha, "alpha")
    delta_ref = _validate_tolerance(delta_ref, "delta_ref", allow_none=True)
    delta_bio = _validate_tolerance(delta_bio, "delta_bio", allow_none=True)

    point = _validate_edges(edges, table_name="edges")
    declared_design = _validate_design(design)
    if declared_design is not None:
        _assert_observed_cells_declared(point, declared_design)
    boot = _validate_bootstrap(bootstrap, point)
    if point.empty:
        return _empty_result()

    score_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []

    contract_design = (
        declared_design if declared_design is not None else _diagnostic_design(point)
    )
    group_source = declared_design if declared_design is not None else point
    op_groups = group_source.loc[
        group_source["edge_type"].eq("OP"), ["domain", "feature", "target"]
    ]
    ordered_groups = op_groups.drop_duplicates().sort_values(list(op_groups))
    for domain, feature, target in ordered_groups.itertuples(index=False, name=None):
        mask_df = point["domain"].eq(domain) & point["feature"].eq(feature)
        relevant = point.loc[
            mask_df
            & (point["edge_type"].eq("PP") | point["target"].eq(target))
        ].copy()
        expected_relevant = contract_design.loc[
            contract_design["domain"].eq(domain)
            & contract_design["feature"].eq(feature)
            & (
                contract_design["edge_type"].eq("PP")
                | contract_design["target"].eq(target)
            )
        ].copy()
        expected_pp = expected_relevant.loc[expected_relevant["edge_type"].eq("PP")]
        expected_op = expected_relevant.loc[expected_relevant["edge_type"].eq("OP")]
        primary_sources = sorted(
            set(expected_pp["target_source"]) | set(expected_relevant["reference_source"])
        )
        target_sources = sorted(expected_op["target_source"].unique())
        metrics = _point_metrics(relevant)

        op_crossed, pp_crossed, missing_op, missing_pp = _crossing_diagnostics(
            relevant, expected_relevant, primary_sources
        )
        target_unit_counts = expected_op.groupby("target_source")["target_unit"].nunique()
        primary_unit_counts = expected_pp.groupby("target_source")["target_unit"].nunique()
        units_pass = (
            len(target_unit_counts) == len(target_sources)
            and len(primary_unit_counts) == len(primary_sources)
            and bool((target_unit_counts >= min_target_units).all())
            and bool((primary_unit_counts >= min_target_units).all())
        )
        leaked = relevant["admissible"] & relevant["leakage"]

        if boot is None:
            boot_relevant = None
        else:
            boot_relevant = boot.loc[
                boot["domain"].eq(domain)
                & boot["feature"].eq(feature)
                & (boot["edge_type"].eq("PP") | boot["target"].eq(target))
            ]
        complete, total_draws, n_complete, complete_fraction = _complete_bootstrap(
            boot_relevant, relevant
        )
        joint_bootstrap_pass = (
            n_complete >= min_bootstrap_draws and complete_fraction >= 0.95
        )
        boot_metrics = (
            _bootstrap_metrics(complete)
            if joint_bootstrap_pass
            else _BootstrapMetrics(pd.Series(dtype=float), pd.Series(dtype=float))
        )

        if boot_metrics.delta.empty:
            variance = np.nan
            lambda_min = np.nan
            lambda_max = np.nan
            psd_pass = False
            rank_pass = False
        else:
            variance = float(boot_metrics.delta.var(ddof=1))
            lambda_min = variance
            lambda_max = variance
            scale = max(abs(lambda_max), np.finfo(float).tiny)
            psd_pass = bool(np.isfinite(variance) and lambda_min >= -1e-8 * scale)
            rank_pass = bool(psd_pass and variance > np.finfo(float).eps * scale)

        if boot_metrics.delta.empty:
            ci_low = ci_high = one_low = one_high = np.nan
            anchor_upper = np.nan
            biological_decision = "NOT_EVALUABLE"
        else:
            ci_low = float(boot_metrics.delta.quantile(alpha / 2.0))
            ci_high = float(boot_metrics.delta.quantile(1.0 - alpha / 2.0))
            biological_decision, one_low, one_high = _decision(
                boot_metrics.delta, delta_bio, alpha
            )
            anchor_upper = float(boot_metrics.anchor_dependence.quantile(1.0 - alpha))

        source_deletion_data: list[dict[str, Any]] = []
        deletion_movement_draws: dict[str, pd.Series] = {}
        for deleted_source in primary_sources:
            keep_point = ~(
                relevant["reference_source"].eq(deleted_source)
                | (
                    relevant["edge_type"].eq("PP")
                    & relevant["target_source"].eq(deleted_source)
                )
            )
            deleted_metrics = _point_metrics(relevant.loc[keep_point])
            shift = (
                abs(deleted_metrics.delta - metrics.delta)
                if np.isfinite(deleted_metrics.delta) and np.isfinite(metrics.delta)
                else np.nan
            )

            if boot_metrics.delta.empty:
                deleted_decision = "NOT_EVALUABLE"
                shift_upper = np.nan
            else:
                deleted_boot = complete.loc[
                    ~(
                        complete["reference_source"].eq(deleted_source)
                        | (
                            complete["edge_type"].eq("PP")
                            & complete["target_source"].eq(deleted_source)
                        )
                    )
                ]
                deleted_boot_metrics = _bootstrap_metrics(deleted_boot)
                deleted_decision, _, _ = _decision(
                    deleted_boot_metrics.delta, delta_bio, alpha
                )
                aligned = pd.concat(
                    [
                        boot_metrics.delta.rename("full"),
                        deleted_boot_metrics.delta.rename("deleted"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                movement_draws = (aligned["deleted"] - aligned["full"]).abs()
                deletion_movement_draws[deleted_source] = movement_draws
                shift_upper = (
                    float(movement_draws.quantile(1.0 - alpha))
                    if not movement_draws.empty
                    else np.nan
                )
            decision_stable = (
                True
                if delta_bio is None
                else biological_decision == deleted_decision
                and biological_decision != "NOT_EVALUABLE"
            )
            source_deletion_data.append(
                {
                    "deleted_primary_source": deleted_source,
                    "delta_deleted": deleted_metrics.delta,
                    "absolute_shift": shift,
                    "absolute_shift_upper": shift_upper,
                    "point_within_delta_ref": bool(
                        delta_ref is not None and shift <= delta_ref
                    )
                    if np.isfinite(shift)
                    else False,
                    "within_delta_ref": bool(
                        delta_ref is not None and shift_upper <= delta_ref
                    )
                    if np.isfinite(shift_upper)
                    else False,
                    "deleted_biological_decision": deleted_decision,
                    "decision_stable": bool(decision_stable),
                }
            )

        shifts = np.array(
            [row["absolute_shift"] for row in source_deletion_data], dtype=float
        )
        max_shift = float(np.max(shifts)) if len(shifts) and np.isfinite(shifts).all() else np.nan
        movement_matrix = (
            pd.concat(deletion_movement_draws, axis=1).dropna()
            if deletion_movement_draws
            else pd.DataFrame()
        )
        max_shift_upper = (
            float(movement_matrix.max(axis=1).quantile(1.0 - alpha))
            if len(deletion_movement_draws) == len(primary_sources)
            and not movement_matrix.empty
            else np.nan
        )
        stability_pass = bool(
            delta_ref is not None
            and np.isfinite(anchor_upper)
            and anchor_upper <= delta_ref
            and np.isfinite(max_shift_upper)
            and max_shift_upper <= delta_ref
            and all(row["decision_stable"] for row in source_deletion_data)
        )

        gates = [
            _gate(
                "G0_DOMAIN",
                True,
                domain,
                "nonblank frozen domain identifier",
                "ABSTAIN_OUT_OF_DOMAIN",
                "Domain fields passed the strict input contract.",
            ),
            _gate(
                "G1_DECLARED_DESIGN",
                declared_design is not None,
                "supplied" if declared_design is not None else "absent",
                "independent expected-cell design table supplied",
                "ABSTAIN_UNDECLARED_REFERENCE_DESIGN",
                "Observed edges cannot define their own expected crossing universe.",
            ),
            _gate(
                "G1_PRIMARY_SOURCE_COUNT",
                len(primary_sources) >= min_primary_sources,
                len(primary_sources),
                f">={min_primary_sources} independent primary sources",
                "ABSTAIN_INCOMPLETE_REFERENCE_DESIGN",
                "Primary sources are counted once regardless of units or anchors.",
            ),
            _gate(
                "G1_DIRECT_OP_CROSSING",
                op_crossed,
                missing_op,
                "0 missing target-unit by reference-source cells",
                "ABSTAIN_INCOMPLETE_REFERENCE_DESIGN",
                "The expected OP Cartesian crossing is fixed by design; only admissible, "
                "nonleaked, directly observed edges count.",
            ),
            _gate(
                "G1_DIRECT_PP_CROSSING",
                pp_crossed,
                missing_pp,
                "0 missing cross-source primary-unit by reference-source cells",
                "ABSTAIN_INCOMPLETE_REFERENCE_DESIGN",
                "The expected PP Cartesian crossing is fixed by design. Same-source PP "
                "cells and graph-path imputations are not admitted.",
            ),
            _gate(
                "G2_BIOLOGICAL_UNITS",
                units_pass,
                (
                    f"target_min={target_unit_counts.min() if len(target_unit_counts) else 0};"
                    f"primary_min={primary_unit_counts.min() if len(primary_unit_counts) else 0}"
                ),
                f">={min_target_units} target and primary units per source",
                "ABSTAIN_INSUFFICIENT_INFORMATION",
                "Unit counts are evaluated at the independent source level.",
            ),
            _gate(
                "G2_LEAKAGE_ADMISSIBILITY",
                not leaked.any(),
                int(leaked.sum()),
                "0 admitted edges flagged leakage",
                "ABSTAIN_REFERENCE_LEAKAGE",
                "Same-source PP rows are structurally excluded and are not leakage.",
            ),
            _gate(
                "G3_JOINT_BOOTSTRAP",
                joint_bootstrap_pass,
                f"{n_complete}/{total_draws} ({complete_fraction:.3f})",
                f">={min_bootstrap_draws} complete draws and >=0.95 complete",
                "ABSTAIN_DEPENDENCE_UNRECOVERABLE",
                "Completeness requires every admitted edge in the reported contrast.",
            ),
            _gate(
                "G3_COVARIANCE_PSD",
                psd_pass,
                f"lambda_min={lambda_min};lambda_max={lambda_max}",
                "lambda_min >= -1e-8 * lambda_max",
                "ABSTAIN_NUMERICAL_FAILURE",
                "PSD is checked for the reported scalar contrast covariance.",
            ),
            _gate(
                "G3_CONTRAST_FULL_RANK",
                rank_pass,
                f"variance={variance}",
                "positive rank for the reported scalar contrast",
                "ABSTAIN_NUMERICAL_FAILURE",
                "A constant or nonfinite bootstrap contrast is not inferential.",
            ),
            _gate(
                "G4_FROZEN_DELTA_REF",
                delta_ref is not None,
                delta_ref,
                "positive prespecified delta_ref",
                "ABSTAIN_UNFROZEN_TOLERANCE",
                "The software does not select a tolerance from observed outcomes.",
            ),
            _gate(
                "G4_REFERENCE_STABILITY",
                stability_pass,
                f"A_upper={anchor_upper};max_deletion_shift_upper={max_shift_upper}",
                "A and every deletion upper bound <= delta_ref; decisions unchanged",
                "ABSTAIN_REFERENCE_NONINVARIANT",
                "A is already simultaneous over admitted reference anchors.",
            ),
            _gate(
                "G5_PROTOCOL_SCOPE",
                len(target_sources) >= 3,
                len(target_sources),
                ">=3 independent target sources for protocol scope",
                "PREPARATION_SPECIFIC",
                "This scope gate does not invalidate preparation-specific standardisation.",
            ),
        ]
        structural_status = _headline_status(gates, len(target_sources))
        status = (
            "UNRESOLVED"
            if structural_status in {"STANDARDISED", "PREPARATION_SPECIFIC"}
            and biological_decision == "UNRESOLVED"
            else structural_status
        )
        inferential_statuses = {"STANDARDISED", "PREPARATION_SPECIFIC", "UNRESOLVED"}
        descriptive_only = status not in inferential_statuses
        failed_gates = ";".join(row["gate_id"] for row in gates if not row["passed"])
        if structural_status in {"STANDARDISED", "PREPARATION_SPECIFIC"}:
            inference_scope = "protocol" if len(target_sources) >= 3 else "preparation"
        else:
            inference_scope = "descriptive_only"

        score_rows.append(
            {
                "domain": domain,
                "feature": feature,
                "target": target,
                "delta": metrics.delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "one_sided_low": one_low,
                "one_sided_high": one_high,
                "anchor_dependence": metrics.anchor_dependence,
                "anchor_dependence_upper": anchor_upper,
                "max_source_deletion_shift": max_shift,
                "max_source_deletion_shift_upper": max_shift_upper,
                "n_primary_sources": len(primary_sources),
                "n_target_sources": len(target_sources),
                "n_complete_bootstrap_draws": n_complete,
                "complete_bootstrap_fraction": complete_fraction,
                "delta_ref": delta_ref,
                "delta_bio": delta_bio,
                "status": status,
                "biological_decision": biological_decision,
                "inference_scope": inference_scope,
                "descriptive_only": descriptive_only,
                "failed_gates": failed_gates,
            }
        )
        for row in gates:
            gate_rows.append(
                {
                    "domain": domain,
                    "feature": feature,
                    "target": target,
                    **row,
                    "headline_status": status,
                }
            )
        for row in metrics.anchors.itertuples(index=False):
            anchor_rows.append(
                {
                    "domain": domain,
                    "feature": feature,
                    "target": target,
                    "reference": row.reference,
                    "reference_source": row.reference_source,
                    "anchor_excess": row.anchor_excess,
                    "delta": metrics.delta,
                    "absolute_deviation_from_delta": abs(row.anchor_excess - metrics.delta),
                    "status": status,
                    "descriptive_only": descriptive_only,
                }
            )
        for row in metrics.baselines.itertuples(index=False):
            baseline_rows.append(
                {
                    "domain": domain,
                    "feature": feature,
                    "target": target,
                    "reference": row.reference,
                    "reference_source": row.reference_source,
                    "baseline": row.baseline,
                    "n_contributing_primary_sources": row.n_contributing_primary_sources,
                    "status": status,
                    "descriptive_only": descriptive_only,
                }
            )
        for row in source_deletion_data:
            deletion_rows.append(
                {
                    "domain": domain,
                    "feature": feature,
                    "target": target,
                    "deleted_primary_source": row["deleted_primary_source"],
                    "delta_full": metrics.delta,
                    "delta_deleted": row["delta_deleted"],
                    "absolute_shift": row["absolute_shift"],
                    "absolute_shift_upper": row["absolute_shift_upper"],
                    "point_within_delta_ref": row["point_within_delta_ref"],
                    "within_delta_ref": row["within_delta_ref"],
                    "full_biological_decision": biological_decision,
                    "deleted_biological_decision": row["deleted_biological_decision"],
                    "decision_stable": row["decision_stable"],
                    "status": status,
                    "descriptive_only": descriptive_only,
                }
            )

    return ReferenceStandardizationResult(
        pd.DataFrame(score_rows, columns=SCORE_COLUMNS),
        pd.DataFrame(anchor_rows, columns=ANCHOR_COLUMNS),
        pd.DataFrame(baseline_rows, columns=BASELINE_COLUMNS),
        pd.DataFrame(gate_rows, columns=GATE_COLUMNS),
        pd.DataFrame(deletion_rows, columns=DELETION_COLUMNS),
    )
