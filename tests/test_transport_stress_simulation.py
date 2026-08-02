import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "validation/simulations/transport_stress_simulation.py"
NOMINAL_MANIFEST = SCRIPT.with_name("nominal_selection_manifest.json")
SPEC = importlib.util.spec_from_file_location("transport_stress_simulation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
transport = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = transport
SPEC.loader.exec_module(transport)


def test_frozen_fractional_design_identity_and_balance():
    manifest, cells = transport.load_design()
    assert len(cells) == 378
    assert cells[0].cell_index == 0
    assert cells[-1].cell_index == 377
    assert transport.design_sha256(cells) == manifest["design_sha256"]
    assert manifest["full_replicates_by_p"] == {"100": 1000, "1000": 1000, "10000": 250}
    assert "full only" in manifest["full_profile_note"]

    for factor in transport.CORE_FACTORS:
        counts = {}
        for cell in cells:
            value = getattr(cell, factor)
            counts[value] = counts.get(value, 0) + 1
        assert set(counts.values()) == {126}

    for left_index, left in enumerate(transport.CORE_FACTORS):
        for right in transport.CORE_FACTORS[left_index + 1 :]:
            pair_counts = {}
            for cell in cells:
                pair = (getattr(cell, left), getattr(cell, right))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            assert len(pair_counts) == 9
            assert set(pair_counts.values()) == {42}


def test_missingness_and_selection_are_fully_crossed():
    manifest, cells = transport.load_design()
    combinations = {(cell.missingness, cell.selection) for cell in cells}
    assert len(combinations) == 14
    assert {cell.missingness for cell in cells} == {
        item["label"] for item in manifest["missingness_levels"]
    }
    assert {cell.selection for cell in cells} == set(manifest["selection_levels"])


def test_nominal_selection_alias_resolution_design_is_complete():
    manifest, cells = transport.load_design(manifest_path=NOMINAL_MANIFEST)
    assert len(cells) == 81
    assert transport.design_sha256(cells) == manifest["design_sha256"]
    assert {cell.k for cell in cells} == {6, 15, 30}
    assert {cell.target_bias for cell in cells} == {0.0}
    assert {cell.target_scale for cell in cells} == {1.0}
    assert {cell.selection for cell in cells} == {"training_only"}
    assert {(cell.k, cell.truth, cell.block_rho, cell.missingness) for cell in cells} == {
        (k, truth, rho, missingness)
        for k in (6, 15, 30)
        for truth in ("null", "sparse_10pct", "dense_70pct")
        for rho in (0.0, 0.3, 0.7)
        for missingness in ("mcar_0", "mcar_50", "effect_50")
    }


def test_cell_is_deterministic_and_has_separate_targets():
    manifest, cells = transport.load_design()
    cell = next(cell for cell in cells if cell.p == 100 and cell.k == 6)
    first = transport.run_cell(cell, replicates=1, nnf_draws=100, pi_features=2, manifest=manifest)
    second = transport.run_cell(cell, replicates=1, nnf_draws=100, pi_features=2, manifest=manifest)
    assert first == second
    assert {row["method"] for row in first} == set(transport.METHODS)
    for row in first:
        if row["pi_features_successful"]:
            assert row["observed_mean_width"] > row["latent_mean_width"]
        assert row["inferential_unit"] == "study"
        assert row["feature_level_p_value"].startswith("not_computed")


def test_prohibited_target_selection_is_identified_as_negative_control():
    manifest, cells = transport.load_design()
    leakage = next(cell for cell in cells if cell.selection == "target_informed_prohibited")
    primary = next(
        cell
        for cell in cells
        if cell.cell_index == leakage.cell_index - 1 and cell.selection == "training_only"
    )
    assert transport._control_type(leakage).startswith("prohibited")
    assert transport._control_type(primary) != transport._control_type(leakage)

    rng = np.random.default_rng(42)
    atlas = rng.standard_normal(1000)
    target = rng.standard_normal(1000)
    variance = np.full(1000, 0.1)
    eligible = np.ones(1000, dtype=bool)
    training_selected = transport._select_features(
        primary, atlas, variance, target, variance, eligible, 0.2
    )
    leaked_selected = transport._select_features(
        leakage, atlas, variance, target, variance, eligible, 0.2
    )
    training_r = np.corrcoef(atlas[training_selected], target[training_selected])[0, 1]
    leaked_r = np.corrcoef(atlas[leaked_selected], target[leaked_selected])[0, 1]
    assert leaked_r > training_r + 0.5


def test_training_selection_precedes_target_availability_filter():
    manifest, cells = transport.load_design()
    training = next(
        cell
        for cell in cells
        if cell.p == 100
        and cell.k == 6
        and cell.missingness == "effect_50"
        and cell.selection == "training_only"
    )
    prohibited = next(
        cell
        for cell in cells
        if cell.p == 100
        and cell.k == 6
        and cell.missingness == "effect_50"
        and cell.selection == "target_informed_prohibited"
    )
    training_rows = transport.run_cell(
        training, replicates=1, nnf_draws=100, pi_features=1, manifest=manifest
    )
    prohibited_rows = transport.run_cell(
        prohibited, replicates=1, nnf_draws=100, pi_features=1, manifest=manifest
    )

    training_row = training_rows[0]
    assert (
        training_row["features_selected_mean"]
        > training_row["features_selected_target_evaluable_mean"]
    )
    assert (
        prohibited_rows[0]["features_selected_mean"]
        == prohibited_rows[0]["features_selected_target_evaluable_mean"]
    )
    for rows in (training_rows, prohibited_rows):
        assert rows[0]["shared_point_method"] == "dl_selection_diagnostic"
        assert rows[0]["method_point_metrics_status"].startswith("shared_not")
        shared = [
            tuple(
                row[key]
                for key in (
                    "shared_transport_pearson_mean",
                    "shared_transport_spearman_mean",
                    "shared_transport_bias_mean",
                    "shared_transport_rmse_mean",
                )
            )
            for row in rows
        ]
        assert shared[0] == shared[1]


def test_nnf_failure_is_recorded_without_reml_substitution(monkeypatch):
    manifest, cells = transport.load_design()
    cell = next(cell for cell in cells if cell.p == 100 and cell.k == 6)

    def fail_nnf(*args, **kwargs):
        raise transport.NNFComputationError("deliberate test failure")

    monkeypatch.setattr(transport, "nnf_prediction_interval", fail_nnf)
    rows = transport.run_cell(cell, replicates=1, nnf_draws=100, pi_features=2, manifest=manifest)
    by_method = {row["method"]: row for row in rows}
    assert by_method["nnf_cd"]["pi_features_successful"] == 0
    assert by_method["nnf_cd"]["pi_features_failed"] > 0
    assert "deliberate test failure" in by_method["nnf_cd"]["pi_failure_types"]
    assert by_method["reml_mkh_approx"]["pi_features_successful"] > 0


def test_output_json_is_strict_and_schema_is_stable(tmp_path: Path):
    manifest, cells = transport.load_design()
    cell = next(cell for cell in cells if cell.p == 100 and cell.k == 6)
    rows = transport.run_cell(cell, replicates=1, nnf_draws=100, pi_features=1, manifest=manifest)
    tsv, structured = transport.write_outputs(
        rows,
        output_dir=tmp_path,
        cell=cell,
        metadata={"cell_index": cell.cell_index},
    )
    assert tsv.read_text().splitlines()[0].split("\t") == list(transport.OUTPUT_COLUMNS)
    payload = json.loads(structured.read_text())
    assert payload["metadata"]["cell_index"] == cell.cell_index
    assert len(payload["results"]) == 2
    assert all(set(row) == set(transport.OUTPUT_COLUMNS) for row in payload["results"])
    assert not any(
        isinstance(value, float) and not np.isfinite(value)
        for row in payload["results"]
        for value in row.values()
    )
