import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "validation/simulations/coverage_simulation.py"
SPEC = importlib.util.spec_from_file_location("coverage_simulation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
coverage_simulation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = coverage_simulation
SPEC.loader.exec_module(coverage_simulation)

METHODS = coverage_simulation.METHODS
OUTPUT_COLUMNS = coverage_simulation.OUTPUT_COLUMNS
_load_json = coverage_simulation._load_json
load_full_grid = coverage_simulation.load_full_grid
run_cell = coverage_simulation.run_cell
run_cells = coverage_simulation.run_cells
write_outputs = coverage_simulation.write_outputs


@pytest.fixture(scope="module")
def manifest_and_cell():
    manifest, cells = load_full_grid()
    return manifest, cells[377]


def test_frozen_grid_identity_and_schema():
    manifest, cells = load_full_grid()
    assert len(cells) == 750
    assert cells[0].cell_index == 0
    assert cells[-1].cell_index == 749
    assert tuple(manifest["axis_order"]) == (
        "k",
        "tau_se_ratio",
        "se_pattern",
        "random_effect",
        "reported_se_multiplier",
    )


def test_deterministic_for_fixed_cell_and_seed(manifest_and_cell):
    manifest, cell = manifest_and_cell
    first = run_cell(cell, replicates=2, nnf_draws=100, jobs=1, manifest=manifest)
    second = run_cell(cell, replicates=2, nnf_draws=100, jobs=1, manifest=manifest)
    assert first == second


def test_latent_and_observed_targets_are_distinct(manifest_and_cell):
    manifest, cell = manifest_and_cell
    rows = run_cell(cell, replicates=3, nnf_draws=200, jobs=1, manifest=manifest)
    by_method = {row["method"]: row for row in rows}
    assert set(by_method) == set(METHODS)
    for row in rows:
        assert row["observed_mean_width"] > row["latent_mean_width"]
        assert row["observed_center_rmse"] != row["latent_center_rmse"]
        assert 0 <= row["latent_sign_error_rate"] <= 1
        assert 0 <= row["observed_sign_error_rate"] <= 1


def test_single_cell_chunk_matches_multi_cell_selection(manifest_and_cell):
    manifest, cell = manifest_and_cell
    _, cells = load_full_grid()
    neighbor = cells[cell.cell_index - 1]
    alone = run_cell(cell, replicates=2, nnf_draws=100, jobs=1, manifest=manifest)
    combined = run_cells([neighbor, cell], replicates=2, nnf_draws=100, jobs=1, manifest=manifest)
    selected = [row for row in combined if row["cell_index"] == cell.cell_index]
    assert selected == alone


def test_tsv_and_json_have_stable_schema(tmp_path: Path, manifest_and_cell):
    manifest, cell = manifest_and_cell
    rows = run_cell(cell, replicates=1, nnf_draws=100, jobs=1, manifest=manifest)
    tsv_path, json_path = write_outputs(
        rows,
        output_dir=tmp_path,
        stem="chunk",
        metadata={"cell_indices": [cell.cell_index]},
    )
    assert tsv_path.read_text().splitlines()[0].split("\t") == list(OUTPUT_COLUMNS)
    payload = json.loads(json_path.read_text())
    assert payload["metadata"]["cell_indices"] == [cell.cell_index]
    assert [row["method"] for row in payload["results"]] == list(METHODS)
    assert all(set(row) == set(OUTPUT_COLUMNS) for row in payload["results"])


def test_smoke_profile_points_into_frozen_grid():
    manifest, cells = load_full_grid()
    smoke = _load_json(Path(__file__).parents[1] / "validation/simulations/smoke_profile.json")
    assert manifest["schema_version"] == smoke["schema_version"]
    assert len(set(smoke["full_grid_cell_indices"])) == len(smoke["full_grid_cell_indices"])
    assert all(0 <= index < len(cells) for index in smoke["full_grid_cell_indices"])
