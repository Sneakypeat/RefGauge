#!/usr/bin/env python3
"""Strictly validate and deterministically aggregate 378 transport-stress cells."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "transport_stress_manifest.json")
    parser.add_argument("--runner", type=Path, default=HERE / "run_transport_stress_grid.slurm")
    parser.add_argument(
        "--simulation-source",
        type=Path,
        default=HERE / "transport_stress_simulation.py",
    )
    parser.add_argument("--output-prefix", default="transport_stress")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_simulation_api(path: Path) -> Any:
    """Load the exact simulator supplied for aggregation and provenance checks."""

    resolved = path.resolve()
    module_name = f"_refgauge_transport_stress_{_sha256(resolved)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load simulation source: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    required = {"METHODS", "OUTPUT_COLUMNS", "design_sha256", "load_design"}
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise AttributeError(f"Simulation source lacks required API: {missing}")
    return module


def _load_design_from_source(
    simulation: Any,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[Any]]:
    """Support both current and frozen simulator ``load_design`` signatures."""

    parameters = inspect.signature(simulation.load_design).parameters
    if "manifest_path" in parameters:
        return simulation.load_design(manifest_path=manifest_path)

    source_manifest = getattr(simulation, "MANIFEST_PATH", None)
    if source_manifest is None or Path(source_manifest).resolve() != manifest_path.resolve():
        raise ValueError(
            "Frozen simulator cannot load an alternate manifest; its MANIFEST_PATH "
            f"is {source_manifest!r}, requested {manifest_path}"
        )
    return simulation.load_design()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_provenance(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(frame.columns) != ["key", "value"] or frame["key"].duplicated().any():
        raise ValueError(f"Malformed provenance sidecar: {path}")
    return dict(zip(frame["key"], frame["value"], strict=True))


def _same_value(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return bool(np.isclose(float(observed), expected, rtol=0.0, atol=1e-12))
    return str(observed) == str(expected)


def _validate_frame(
    frame: pd.DataFrame,
    cell: Any,
    path: Path,
    expected_replicates: int,
    *,
    methods: tuple[str, ...],
    output_columns: tuple[str, ...],
) -> None:
    if list(frame.columns) != list(output_columns):
        raise ValueError(f"Output schema mismatch in {path}")
    if len(frame) != len(methods) or set(frame["method"]) != set(methods):
        raise ValueError(f"Method set is incomplete in {path}")
    for key, expected in asdict(cell).items():
        if not all(_same_value(value, expected) for value in frame[key]):
            raise ValueError(f"Cell field {key!r} does not match the frozen design in {path}")
    if not frame["replicates_requested"].eq(expected_replicates).all():
        raise ValueError(f"Non-full replicate count in {path}")
    if not frame["replicates_successful"].eq(expected_replicates).all():
        raise ValueError(f"Incomplete simulation replicates in {path}")
    attempted = frame["pi_features_attempted"]
    successful = frame["pi_features_successful"]
    failed = frame["pi_features_failed"]
    if not (attempted.eq(successful + failed) & attempted.ge(0) & failed.ge(0)).all():
        raise ValueError(f"PI attempt/failure accounting is inconsistent in {path}")
    prohibited = frame["selection"].eq("target_informed_prohibited")
    expected_label = "prohibited_target_informed_negative_control"
    if prohibited.any() and not frame.loc[prohibited, "control_type"].eq(expected_label).all():
        raise ValueError(f"Target-informed selection is not labelled prohibited in {path}")
    if not frame["inferential_unit"].eq("study").all():
        raise ValueError(f"Inferential unit is not study in {path}")
    expected_p = "not_computed_features_are_not_replicates"
    if not frame["feature_level_p_value"].eq(expected_p).all():
        raise ValueError(f"A feature-as-replicate p-value appeared in {path}")
    expected_point = "shared_not_an_interval_method_comparison"
    if not frame["method_point_metrics_status"].eq(expected_point).all():
        raise ValueError(f"Shared point diagnostics are mislabelled in {path}")


def _validate_metadata(
    metadata: dict[str, Any],
    provenance: dict[str, str],
    *,
    cell: Any,
    expected_replicates: int,
    manifest: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[str, str, str, str]:
    expected = {
        "schema_version": manifest["schema_version"],
        "cell_index": cell.cell_index,
        "design_sha256": manifest["design_sha256"],
        "replicates": expected_replicates,
        "nnf_draws": manifest["default_nnf_draws"],
        "pi_features": manifest["default_pi_features"],
        "failure_policy": "record_failure_no_fallback",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"JSON metadata {key!r} failed for cell {cell.cell_index}")
    selection_warning = metadata.get("selection_warning", "")
    required_selection_labels = ["training_only", "target availability"]
    if "target_informed_prohibited" in manifest["selection_levels"]:
        required_selection_labels.append("target_informed_prohibited")
    if not all(label in selection_warning for label in required_selection_labels):
        raise ValueError(f"Prohibited-selection warning missing for cell {cell.cell_index}")
    expected_point = "shared DL selection diagnostics are not attributed to interval methods"
    if metadata.get("point_metric_status") != expected_point:
        raise ValueError(f"Point-metric status missing for cell {cell.cell_index}")

    provenance_expected = {
        "schema_version": str(manifest["schema_version"]),
        "status": "complete",
        "cell_index": str(cell.cell_index),
        "p": str(cell.p),
        "replicates": str(expected_replicates),
        "nnf_draws": str(manifest["default_nnf_draws"]),
        "pi_features": str(manifest["default_pi_features"]),
        "design_sha256": manifest["design_sha256"],
        "failure_policy": "record_failure_no_fallback",
        "inferential_unit": "study",
        "point_metric_status": (
            "shared DL selection diagnostics are not attributed to interval methods"
        ),
        "simulation_code_sha256": _sha256(paths["simulation"]),
        "nnf_code_sha256": _sha256(paths["nnf"]),
        "manifest_sha256": _sha256(paths["manifest"]),
        "runner_code_sha256": _sha256(paths["runner"]),
    }
    for key, value in provenance_expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Provenance {key!r} failed for cell {cell.cell_index}")
    if metadata["simulation_code_sha256"] != provenance["simulation_code_sha256"]:
        raise ValueError(f"JSON/sidecar code identity differs for cell {cell.cell_index}")
    return (
        provenance["simulation_code_sha256"],
        provenance["nnf_code_sha256"],
        provenance["manifest_sha256"],
        provenance["runner_code_sha256"],
    )


def _summaries(
    results: pd.DataFrame,
    *,
    require_control_pairs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interval_summary = (
        results.groupby(["method", "selection", "p", "k", "truth", "missingness"], sort=True)
        .agg(
            n_cells=("cell_index", "size"),
            latent_coverage=("latent_coverage", "mean"),
            observed_coverage=("observed_coverage", "mean"),
            latent_width=("latent_mean_width", "mean"),
            observed_width=("observed_mean_width", "mean"),
            max_pi_failure_rate=("pi_failure_rate", "max"),
        )
        .reset_index()
    )

    cell_keys = [
        "cell_index",
        "p",
        "target_bias",
        "target_scale",
        "truth",
        "block_rho",
        "k",
        "missingness",
        "selection",
    ]
    shared_columns = [
        "shared_point_method",
        "features_selection_eligible_mean",
        "features_selected_mean",
        "features_selected_target_evaluable_mean",
        "shared_transport_pearson_mean",
        "shared_transport_spearman_mean",
        "shared_transport_bias_mean",
        "shared_transport_rmse_mean",
        "shared_sign_error_rate_mean",
        "shared_eiv_intercept_mean",
        "shared_eiv_slope_mean",
    ]
    shared = results[cell_keys + shared_columns].drop_duplicates()
    if len(shared) != results["cell_index"].nunique():
        raise ValueError("Shared point diagnostics differ between interval-method rows")
    point_summary = (
        shared.groupby(["selection", "p", "k", "truth", "missingness"], sort=True)
        .agg(
            n_cells=("cell_index", "size"),
            mean_selection_eligible=("features_selection_eligible_mean", "mean"),
            mean_selected=("features_selected_mean", "mean"),
            mean_selected_target_evaluable=(
                "features_selected_target_evaluable_mean",
                "mean",
            ),
            mean_pearson=("shared_transport_pearson_mean", "mean"),
            mean_spearman=("shared_transport_spearman_mean", "mean"),
            mean_bias=("shared_transport_bias_mean", "mean"),
            mean_rmse=("shared_transport_rmse_mean", "mean"),
            mean_sign_error=("shared_sign_error_rate_mean", "mean"),
            mean_eiv_intercept=("shared_eiv_intercept_mean", "mean"),
            mean_eiv_slope=("shared_eiv_slope_mean", "mean"),
        )
        .reset_index()
    )
    point_summary["point_metric_status"] = (
        "shared_dl_selection_diagnostic_not_an_interval_method_comparison"
    )

    # Selection is encoded as a separate design cell, so paired negative-control
    # rows necessarily have different cell_index values. Pair on the biological
    # simulation coordinates and retain both source cell IDs for provenance.
    control_keys = [key for key in cell_keys if key not in {"cell_index", "selection"}]
    comparison_metrics = [
        "shared_transport_pearson_mean",
        "shared_transport_spearman_mean",
        "shared_transport_bias_mean",
        "shared_transport_rmse_mean",
        "shared_sign_error_rate_mean",
    ]
    training = shared.loc[
        shared["selection"].eq("training_only"),
        ["cell_index", *control_keys, *comparison_metrics],
    ].rename(columns={"cell_index": "cell_index_training_only"})
    prohibited = shared.loc[
        shared["selection"].eq("target_informed_prohibited"),
        ["cell_index", *control_keys, *comparison_metrics],
    ].rename(columns={"cell_index": "cell_index_target_informed_prohibited"})
    if prohibited.empty and not require_control_pairs:
        controls = pd.DataFrame(
            columns=[
                "cell_index_training_only",
                "cell_index_target_informed_prohibited",
                *control_keys,
                *[f"{metric}_training_only" for metric in comparison_metrics],
                *[f"{metric}_target_informed_prohibited" for metric in comparison_metrics],
                *[f"delta_{metric}_prohibited_minus_training" for metric in comparison_metrics],
                "interpretation",
            ]
        )
    else:
        controls = training.merge(
            prohibited,
            on=control_keys,
            suffixes=("_training_only", "_target_informed_prohibited"),
            validate="one_to_one",
        )
        if len(controls) != len(training) or len(controls) != len(prohibited):
            raise ValueError(
                "Training-only and prohibited-selection cells do not form complete pairs"
            )
        for metric in comparison_metrics:
            controls[f"delta_{metric}_prohibited_minus_training"] = (
                controls[f"{metric}_target_informed_prohibited"]
                - controls[f"{metric}_training_only"]
            )
        controls["interpretation"] = "negative_control_not_an_admissible_analysis"

    failures = (
        results.groupby(["method", "p", "k", "missingness", "selection"], sort=True)
        .agg(
            n_cells=("cell_index", "size"),
            attempted=("pi_features_attempted", "sum"),
            successful=("pi_features_successful", "sum"),
            failed=("pi_features_failed", "sum"),
            max_failure_rate=("pi_failure_rate", "max"),
            mean_training_k_failures=("features_training_k_failed_mean", "mean"),
            mean_target_missing=("features_target_missing_mean", "mean"),
        )
        .reset_index()
    )
    return interval_summary, point_summary, controls, failures


def main() -> int:
    args = _parse_args()
    simulation = _load_simulation_api(args.simulation_source)
    manifest, cells = _load_design_from_source(simulation, args.manifest)
    methods = tuple(simulation.METHODS)
    output_columns = tuple(simulation.OUTPUT_COLUMNS)
    paths = {
        "simulation": args.simulation_source,
        "manifest": args.manifest,
        "runner": args.runner,
        "nnf": HERE.parents[1] / "src/refgauge/prediction.py",
    }
    frames: list[pd.DataFrame] = []
    checksums: list[dict[str, str]] = []
    code_identities: set[tuple[str, str, str, str]] = set()
    schedule = manifest["full_replicates_by_p"]

    for cell in cells:
        stem = f"transport_stress_cell_{cell.cell_index:04d}"
        files = {
            "tsv": args.input_dir / f"{stem}.tsv",
            "json": args.input_dir / f"{stem}.json",
            "provenance": args.input_dir / f"{stem}.provenance.tsv",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing frozen cell {cell.cell_index}: {missing}")
        expected_replicates = int(schedule[str(cell.p)])
        frame = pd.read_csv(files["tsv"], sep="\t", keep_default_na=False)
        _validate_frame(
            frame,
            cell,
            files["tsv"],
            expected_replicates,
            methods=methods,
            output_columns=output_columns,
        )
        metadata = json.loads(files["json"].read_text(encoding="utf-8"))["metadata"]
        provenance = _read_provenance(files["provenance"])
        code_identities.add(
            _validate_metadata(
                metadata,
                provenance,
                cell=cell,
                expected_replicates=expected_replicates,
                manifest=manifest,
                paths=paths,
            )
        )
        frames.append(frame)
        checksums.extend({"file": path.name, "sha256": _sha256(path)} for path in files.values())

    if len(code_identities) != 1:
        raise ValueError("Cells were produced by more than one code identity")
    results = pd.concat(frames, ignore_index=True).sort_values(
        ["cell_index", "method"], kind="stable"
    )
    if len(results) != len(cells) * len(methods):
        raise ValueError("Aggregated row count does not match the frozen design")
    require_controls = "target_informed_prohibited" in manifest["selection_levels"]
    interval_summary, point_summary, controls, failures = _summaries(
        results,
        require_control_pairs=require_controls,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "all": args.output_dir / f"{args.output_prefix}_all_cells.tsv",
        "interval": args.output_dir / f"{args.output_prefix}_interval_summary.tsv",
        "point": args.output_dir / f"{args.output_prefix}_shared_point_summary.tsv",
        "controls": args.output_dir / f"{args.output_prefix}_prohibited_selection_control.tsv",
        "failures": args.output_dir / f"{args.output_prefix}_failure_summary.tsv",
        "checksums": args.output_dir / f"{args.output_prefix}_input_checksums.tsv",
    }
    tables = {
        "all": results,
        "interval": interval_summary,
        "point": point_summary,
        "controls": controls,
        "failures": failures,
        "checksums": pd.DataFrame(checksums).sort_values("file", kind="stable"),
    }
    for key, path in outputs.items():
        _atomic_write(path, tables[key].to_csv(sep="\t", index=False, na_rep="NA"))

    identity = next(iter(code_identities))
    run_manifest = {
        "schema_version": manifest["schema_version"],
        "design_sha256": simulation.design_sha256(cells),
        "n_cells": len(cells),
        "methods": list(methods),
        "full_replicates_by_p": schedule,
        "nnf_draws": manifest["default_nnf_draws"],
        "pi_features": manifest["default_pi_features"],
        "target_informed_selection": (
            "prohibited_negative_control_only" if require_controls else "not_included"
        ),
        "training_only_selection": "independent_of_target_availability_and_outcomes",
        "point_metrics": "shared_dl_diagnostics_not_attributed_to_interval_methods",
        "inferential_unit": "study",
        "n_result_rows": len(results),
        "code_identities": {
            "simulation_code_sha256": identity[0],
            "nnf_code_sha256": identity[1],
            "manifest_sha256": identity[2],
            "runner_code_sha256": identity[3],
            "aggregation_code_sha256": _sha256(Path(__file__).resolve()),
        },
        "outputs": {path.name: _sha256(path) for path in outputs.values()},
    }
    manifest_path = args.output_dir / f"{args.output_prefix}_aggregation_manifest.json"
    _atomic_write(manifest_path, json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
