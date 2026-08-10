import numpy as np
import pandas as pd
import pytest

from refgauge.standardization import (
    DESIGN_COLUMNS,
    ReferenceStandardizationResult,
    standardize_reference_family,
)


def _synthetic_edges(
    *,
    n_primary_sources=6,
    n_target_sources=3,
    units=3,
    unstable=False,
):
    primary_sources = [f"primary_{index}" for index in range(n_primary_sources)]
    rows = []
    for q_index, target_source in enumerate(primary_sources):
        for unit in range(units):
            for r_index, reference_source in enumerate(primary_sources):
                if target_source == reference_source:
                    continue
                rows.append(
                    {
                        "domain": "astrocyte_rna",
                        "feature": "discrepancy",
                        "edge_type": "PP",
                        "target": f"{target_source}_primary",
                        "target_unit": f"{target_source}_donor_{unit}",
                        "target_source": target_source,
                        "reference": f"{reference_source}_anchor",
                        "reference_source": reference_source,
                        "estimate": 0.10 + 0.0005 * q_index + 0.0001 * unit + 0.001 * r_index,
                        "admissible": True,
                        "leakage": False,
                    }
                )
    for source_index in range(n_target_sources):
        target_source = f"lab_{source_index}"
        for unit in range(units):
            for r_index, reference_source in enumerate(primary_sources):
                anchor_shift = 0.08 if unstable and r_index == 0 else 0.0
                rows.append(
                    {
                        "domain": "astrocyte_rna",
                        "feature": "discrepancy",
                        "edge_type": "OP",
                        "target": "protocol_a",
                        "target_unit": f"{target_source}_organoid_{unit}",
                        "target_source": target_source,
                        "reference": f"{reference_source}_anchor",
                        "reference_source": reference_source,
                        "estimate": (
                            0.12
                            + 0.0004 * source_index
                            + 0.0001 * unit
                            + 0.001 * r_index
                            + anchor_shift
                        ),
                        "admissible": True,
                        "leakage": False,
                    }
                )
    return pd.DataFrame(rows)


def _bootstrap(edges, *, draws=5000, seed=18):
    rng = np.random.default_rng(seed)
    repeated = edges.loc[edges.index.repeat(draws)].reset_index(drop=True)
    repeated["draw"] = np.tile(np.arange(draws), len(edges)).astype(str)
    repeated["estimate"] = repeated["estimate"] + rng.normal(0, 0.0004, len(repeated))
    return repeated


def _design(edges):
    same_source_pp = edges["edge_type"].eq("PP") & edges["target_source"].eq(
        edges["reference_source"]
    )
    return (
        edges.loc[~same_source_pp, list(DESIGN_COLUMNS)]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _run(edges, bootstrap, **kwargs):
    return standardize_reference_family(
        edges,
        design=kwargs.pop("design", _design(edges)),
        bootstrap=bootstrap,
        delta_ref=kwargs.pop("delta_ref", 0.02),
        min_bootstrap_draws=kwargs.pop("min_bootstrap_draws", 5000),
        **kwargs,
    )


def test_passing_synthetic_case_emits_source_balanced_standardized_score():
    edges = _synthetic_edges()
    result = _run(edges, _bootstrap(edges), delta_bio=0.05)

    assert isinstance(result, ReferenceStandardizationResult)
    score = result.scores.iloc[0]
    assert score["status"] == "STANDARDISED"
    assert score["biological_decision"] == "WITHIN_ENVELOPE"
    assert score["inference_scope"] == "protocol"
    assert not score["descriptive_only"]
    assert score["delta"] == pytest.approx(0.01915, abs=1e-12)
    assert score["anchor_dependence_upper"] <= 0.02
    assert result.gates["passed"].all()
    assert len(result.reference_baselines) == 6
    assert len(result.anchor_excess) == 6
    assert len(result.source_deletions) == 6
    assert score["max_source_deletion_shift_upper"] >= result.source_deletions[
        "absolute_shift_upper"
    ].max()


def test_missing_design_abstains_but_preserves_point_diagnostics():
    edges = _synthetic_edges()
    result = standardize_reference_family(
        edges,
        bootstrap=_bootstrap(edges),
        delta_ref=0.02,
    )
    score = result.scores.iloc[0]

    assert score["status"] == "ABSTAIN_UNDECLARED_REFERENCE_DESIGN"
    assert score["descriptive_only"]
    assert score["delta"] == pytest.approx(0.01915, abs=1e-12)
    gate = result.gates.set_index("gate_id").loc["G1_DECLARED_DESIGN"]
    assert not gate["passed"]
    assert gate["observed"] == "absent"


def test_declared_design_exposes_entirely_missing_primary_source_cells():
    complete = _synthetic_edges()
    design = _design(complete)
    missing_source = "primary_5"
    keep = ~(
        complete["reference_source"].eq(missing_source)
        | (
            complete["edge_type"].eq("PP")
            & complete["target_source"].eq(missing_source)
        )
    )
    observed = complete.loc[keep].reset_index(drop=True)
    result = _run(observed, _bootstrap(observed), design=design)
    gates = result.gates.set_index("gate_id")

    assert result.scores.iloc[0]["n_primary_sources"] == 6
    assert gates.loc["G1_PRIMARY_SOURCE_COUNT", "passed"]
    assert not gates.loc["G1_DIRECT_OP_CROSSING", "passed"]
    assert not gates.loc["G1_DIRECT_PP_CROSSING", "passed"]
    assert int(gates.loc["G1_DIRECT_OP_CROSSING", "observed"]) > 0
    assert int(gates.loc["G1_DIRECT_PP_CROSSING", "observed"]) > 0


def test_declared_design_counts_expected_units_even_when_one_is_wholly_missing():
    complete = _synthetic_edges()
    design = _design(complete)
    missing_unit = "lab_0_organoid_2"
    observed = complete.loc[~complete["target_unit"].eq(missing_unit)].reset_index(drop=True)
    result = _run(observed, _bootstrap(observed), design=design)
    gates = result.gates.set_index("gate_id")

    assert "target_min=3" in gates.loc["G2_BIOLOGICAL_UNITS", "observed"]
    assert gates.loc["G2_BIOLOGICAL_UNITS", "passed"]
    assert not gates.loc["G1_DIRECT_OP_CROSSING", "passed"]
    assert int(gates.loc["G1_DIRECT_OP_CROSSING", "observed"]) == 6


def test_four_primary_sources_abstains_but_keeps_diagnostics():
    edges = _synthetic_edges(n_primary_sources=4)
    result = _run(edges, _bootstrap(edges))
    score = result.scores.iloc[0]

    assert score["status"] == "ABSTAIN_INCOMPLETE_REFERENCE_DESIGN"
    assert score["descriptive_only"]
    assert np.isfinite(score["delta"])
    failed = result.gates.loc[~result.gates["passed"], "gate_id"].tolist()
    assert "G1_PRIMARY_SOURCE_COUNT" in failed


def test_missing_bootstrap_abstains_for_unrecoverable_dependence():
    edges = _synthetic_edges()
    result = standardize_reference_family(edges, design=_design(edges), delta_ref=0.02)

    assert result.scores.iloc[0]["status"] == "ABSTAIN_DEPENDENCE_UNRECOVERABLE"
    gate = result.gates.set_index("gate_id").loc["G3_JOINT_BOOTSTRAP"]
    assert not gate["passed"]


def test_admissible_leakage_has_headline_precedence_and_is_retained():
    edges = _synthetic_edges()
    leaked_index = edges.index[edges["edge_type"].eq("OP")][0]
    edges.loc[leaked_index, "leakage"] = True
    bootstrap = _bootstrap(edges)
    result = _run(edges, bootstrap)

    assert result.scores.iloc[0]["status"] == "ABSTAIN_REFERENCE_LEAKAGE"
    leakage_gate = result.gates.set_index("gate_id").loc["G2_LEAKAGE_ADMISSIBILITY"]
    assert not leakage_gate["passed"]
    assert "G1_DIRECT_OP_CROSSING" in result.scores.iloc[0]["failed_gates"]


def test_incomplete_direct_crossing_abstains():
    edges = _synthetic_edges()
    design = _design(edges)
    missing_index = edges.index[edges["edge_type"].eq("OP")][0]
    edges = edges.drop(index=missing_index).reset_index(drop=True)
    result = _run(edges, _bootstrap(edges), design=design)

    assert result.scores.iloc[0]["status"] == "ABSTAIN_INCOMPLETE_REFERENCE_DESIGN"
    gate = result.gates.set_index("gate_id").loc["G1_DIRECT_OP_CROSSING"]
    assert not gate["passed"]
    assert int(gate["observed"]) == 1


def test_missing_predeclared_tolerance_abstains():
    edges = _synthetic_edges()
    result = standardize_reference_family(
        edges,
        design=_design(edges),
        bootstrap=_bootstrap(edges),
        delta_ref=None,
        min_bootstrap_draws=5000,
    )

    assert result.scores.iloc[0]["status"] == "ABSTAIN_UNFROZEN_TOLERANCE"
    assert not result.gates.set_index("gate_id").loc["G4_FROZEN_DELTA_REF", "passed"]


def test_reference_noninvariance_abstains_on_simultaneous_bound():
    edges = _synthetic_edges(unstable=True)
    result = _run(edges, _bootstrap(edges), delta_ref=0.02)
    score = result.scores.iloc[0]

    assert score["status"] == "ABSTAIN_REFERENCE_NONINVARIANT"
    assert score["anchor_dependence_upper"] > 0.02
    assert not result.gates.set_index("gate_id").loc["G4_REFERENCE_STABILITY", "passed"]


def test_deletion_upper_bound_catches_bootstrap_only_instability():
    edges = _synthetic_edges()
    bootstrap = _bootstrap(edges)
    perturbed_draw = bootstrap["draw"].astype(int).ge(4500)
    source_as_target = (
        bootstrap["edge_type"].eq("PP")
        & bootstrap["target_source"].eq("primary_0")
        & bootstrap["reference_source"].ne("primary_0")
    )
    source_as_anchor = (
        bootstrap["edge_type"].eq("PP")
        & bootstrap["reference_source"].eq("primary_0")
        & bootstrap["target_source"].ne("primary_0")
    )
    bootstrap.loc[perturbed_draw & source_as_target, "estimate"] += 0.20
    bootstrap.loc[perturbed_draw & source_as_anchor, "estimate"] += 0.04

    result = _run(edges, bootstrap)
    score = result.scores.iloc[0]
    deletion = result.source_deletions.set_index("deleted_primary_source").loc["primary_0"]

    assert score["max_source_deletion_shift"] < 0.02
    assert score["anchor_dependence_upper"] < 0.02
    assert deletion["absolute_shift"] < 0.02
    assert deletion["absolute_shift_upper"] > 0.02
    assert deletion["point_within_delta_ref"]
    assert not deletion["within_delta_ref"]
    assert score["max_source_deletion_shift_upper"] > 0.02
    assert score["status"] == "ABSTAIN_REFERENCE_NONINVARIANT"
    gate = result.gates.set_index("gate_id").loc["G4_REFERENCE_STABILITY"]
    assert "max_deletion_shift_upper" in gate["observed"]


def test_anchor_diagnostic_does_not_hide_within_source_anchor_variation():
    edges = _synthetic_edges()
    second_anchor = edges.copy()
    second_anchor["reference"] = second_anchor["reference"] + "_replicate"
    shifted = second_anchor["edge_type"].eq("OP") & second_anchor["reference_source"].eq(
        "primary_0"
    )
    second_anchor.loc[shifted, "estimate"] += 0.08
    combined = pd.concat([edges, second_anchor], ignore_index=True)

    result = standardize_reference_family(
        combined, design=_design(combined), delta_ref=0.02
    )

    assert len(result.anchor_excess) == 12
    assert result.scores.iloc[0]["anchor_dependence"] > 0.07


def test_unresolved_biological_margin_is_inferential_headline_status():
    edges = _synthetic_edges()
    bootstrap = _bootstrap(edges)
    rng = np.random.default_rng(902)
    common_target_shift = {
        str(draw): shift for draw, shift in enumerate(rng.normal(0, 0.003, 5000))
    }
    op = bootstrap["edge_type"].eq("OP")
    bootstrap.loc[op, "estimate"] += bootstrap.loc[op, "draw"].map(common_target_shift)
    result = _run(edges, bootstrap, delta_bio=0.01915)
    score = result.scores.iloc[0]

    assert score["status"] == "UNRESOLVED"
    assert score["biological_decision"] == "UNRESOLVED"
    assert score["inference_scope"] == "protocol"
    assert not score["descriptive_only"]
    assert all(
        result.source_deletions["deleted_biological_decision"].eq("UNRESOLVED")
    )


def test_same_source_pp_is_excluded_without_being_called_leakage():
    edges = _synthetic_edges()
    same_source = []
    for source_index in range(6):
        source = f"primary_{source_index}"
        for unit in range(3):
            same_source.append(
                {
                    "domain": "astrocyte_rna",
                    "feature": "discrepancy",
                    "edge_type": "PP",
                    "target": f"{source}_primary",
                    "target_unit": f"{source}_donor_{unit}",
                    "target_source": source,
                    "reference": f"{source}_anchor",
                    "reference_source": source,
                    "estimate": 999.0,
                    "admissible": True,
                    "leakage": False,
                }
            )
    combined = pd.concat([edges, pd.DataFrame(same_source)], ignore_index=True)
    result = _run(combined, _bootstrap(combined))

    assert result.scores.iloc[0]["status"] == "STANDARDISED"
    leakage_gate = result.gates.set_index("gate_id").loc["G2_LEAKAGE_ADMISSIBILITY"]
    assert leakage_gate["passed"]
    assert result.scores.iloc[0]["delta"] == pytest.approx(0.01915, abs=1e-12)


def test_declared_target_source_remains_in_universe_when_observation_is_inadmissible():
    edges = _synthetic_edges()
    design = _design(edges)
    excluded_source = edges["edge_type"].eq("OP") & edges["target_source"].eq("lab_2")
    edges.loc[excluded_source, "admissible"] = False

    result = standardize_reference_family(edges, design=design, delta_ref=0.02)

    assert result.scores.iloc[0]["n_target_sources"] == 3
    assert result.gates.set_index("gate_id").loc["G5_PROTOCOL_SCOPE", "observed"] == 3
    assert not result.gates.set_index("gate_id").loc["G1_DIRECT_OP_CROSSING", "passed"]


def test_one_lab_can_only_emit_preparation_specific_scope():
    edges = _synthetic_edges(n_target_sources=1)
    result = _run(edges, _bootstrap(edges))
    score = result.scores.iloc[0]

    assert score["status"] == "PREPARATION_SPECIFIC"
    assert score["inference_scope"] == "preparation"
    assert not score["descriptive_only"]
    assert not result.gates.set_index("gate_id").loc["G5_PROTOCOL_SCOPE", "passed"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data.drop(columns="domain"), "missing columns"),
        (lambda data: pd.concat([data, data.iloc[[0]]], ignore_index=True), "duplicate edge"),
        (lambda data: data.assign(edge_type="XX"), "edge_type"),
        (lambda data: data.loc[data["edge_type"].eq("OP")], "both OP and PP"),
        (lambda data: data.assign(estimate=np.inf), "finite"),
        (lambda data: data.assign(admissible="yes"), "booleans"),
    ],
)
def test_invalid_point_inputs_fail_loudly(mutator, message):
    with pytest.raises(ValueError, match=message):
        standardize_reference_family(mutator(_synthetic_edges()), delta_ref=0.02)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data.drop(columns="domain"), "missing columns"),
        (
            lambda data: pd.concat([data, data.iloc[[0]]], ignore_index=True),
            "duplicate expected-cell",
        ),
        (lambda data: data.assign(edge_type="XX"), "edge_type"),
        (
            lambda data: pd.concat(
                [
                    data,
                    pd.DataFrame(
                        [
                            {
                                **data.iloc[0].to_dict(),
                                "target": "different_target",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            ),
            "one target identifier",
        ),
    ],
)
def test_invalid_design_inputs_fail_loudly(mutator, message):
    edges = _synthetic_edges()
    with pytest.raises(ValueError, match=message):
        standardize_reference_family(edges, design=mutator(_design(edges)), delta_ref=0.02)


def test_design_rejects_same_source_pp_cells():
    edges = _synthetic_edges()
    design = _design(edges)
    same_source = design.loc[design["edge_type"].eq("PP")].iloc[[0]].copy()
    same_source["reference_source"] = same_source["target_source"]
    design = pd.concat([design, same_source], ignore_index=True)

    with pytest.raises(ValueError, match="same-source PP"):
        standardize_reference_family(edges, design=design, delta_ref=0.02)


@pytest.mark.parametrize("design", [pd.DataFrame(), "not-a-data-frame"])
def test_design_requires_a_nonempty_dataframe(design):
    edges = _synthetic_edges()
    expected_error = ValueError if isinstance(design, pd.DataFrame) else TypeError
    with pytest.raises(expected_error, match="design"):
        standardize_reference_family(edges, design=design, delta_ref=0.02)


def test_admitted_observed_cell_must_be_declared():
    edges = _synthetic_edges()
    design = _design(edges)
    omitted = design.index[design["edge_type"].eq("OP")][0]
    design = design.drop(index=omitted).reset_index(drop=True)

    with pytest.raises(ValueError, match="not present in design"):
        standardize_reference_family(edges, design=design, delta_ref=0.02)


def test_bootstrap_rejects_unknown_edge_keys():
    edges = _synthetic_edges()
    bootstrap = _bootstrap(edges, draws=5000)
    bootstrap.loc[0, "reference"] = "unknown_anchor"
    with pytest.raises(ValueError, match="not present"):
        standardize_reference_family(
            edges,
            bootstrap=bootstrap,
            delta_ref=0.02,
            min_bootstrap_draws=5000,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0.5}, "alpha"),
        ({"delta_ref": 0}, "delta_ref"),
        ({"delta_bio": -0.1}, "delta_bio"),
        ({"min_primary_sources": True}, "integer"),
        ({"min_primary_sources": 5}, "at least"),
        ({"min_target_units": 2}, "at least"),
        ({"min_bootstrap_draws": 4999}, "at least"),
    ],
)
def test_invalid_configuration_fails_loudly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        standardize_reference_family(_synthetic_edges(), **kwargs)
