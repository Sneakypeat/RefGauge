#!/usr/bin/env python3
"""Validate and aggregate all frozen coverage-simulation array cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from coverage_simulation import METHODS, grid_sha256, load_full_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    manifest, cells = load_full_grid()
    expected_indices = {cell.cell_index for cell in cells}
    frames = []
    input_checksums = []
    code_identities = set()
    for index in sorted(expected_indices):
        stem = f"coverage_cell_{index:04d}"
        tsv = args.input_dir / f"{stem}.tsv"
        metadata_path = args.input_dir / f"{stem}.json"
        if not tsv.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing TSV or JSON for frozen cell {index}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["metadata"]
        if metadata["cell_indices"] != [index]:
            raise ValueError(f"Cell-index mismatch in {metadata_path}")
        if metadata["grid_sha256"] != manifest["grid_sha256"]:
            raise ValueError(f"Grid-hash mismatch in {metadata_path}")
        code_identities.add((metadata["simulation_code_sha256"], metadata["nnf_code_sha256"]))
        frame = pd.read_csv(tsv, sep="\t")
        if set(frame["method"]) != set(METHODS) or len(frame) != len(METHODS):
            raise ValueError(f"Method set is incomplete in {tsv}")
        if set(frame["cell_index"]) != {index}:
            raise ValueError(f"TSV cell-index mismatch in {tsv}")
        frames.append(frame)
        input_checksums.extend(
            [
                {"file": tsv.name, "sha256": file_sha256(tsv)},
                {"file": metadata_path.name, "sha256": file_sha256(metadata_path)},
            ]
        )

    results = pd.concat(frames, ignore_index=True).sort_values(
        ["cell_index", "method"], kind="stable"
    )
    if len(results) != len(cells) * len(METHODS):
        raise ValueError("Aggregated result count does not match the frozen design")
    if len(code_identities) != 1:
        raise ValueError("Simulation cells were produced by more than one code identity")
    results["latent_coverage_error"] = results["latent_coverage"] - 0.95
    results["observed_coverage_error"] = results["observed_coverage"] - 0.95
    results["latent_undercoverage_gt_5pp"] = results["latent_coverage"] < 0.90
    results["observed_undercoverage_gt_5pp"] = results["observed_coverage"] < 0.90

    nominal = results.loc[
        results["reported_se_multiplier"].eq(1.0) & results["random_effect"].eq("normal")
    ]
    gate = (
        nominal.groupby("method", sort=False)
        .agg(
            n_cells=("cell_index", "size"),
            mean_latent_coverage=("latent_coverage", "mean"),
            worst_latent_coverage=("latent_coverage", "min"),
            mean_observed_coverage=("observed_coverage", "mean"),
            worst_observed_coverage=("observed_coverage", "min"),
            median_latent_width=("latent_mean_width", "median"),
            median_observed_width=("observed_mean_width", "median"),
            mean_latent_interval_score=("latent_mean_interval_score", "mean"),
            mean_observed_interval_score=("observed_mean_interval_score", "mean"),
            mean_latent_sign_error=("latent_sign_error_rate", "mean"),
            mean_observed_sign_error=("observed_sign_error_rate", "mean"),
            max_failure_rate=("failure_rate", "max"),
            n_latent_undercoverage_gt_5pp=("latent_undercoverage_gt_5pp", "sum"),
            n_observed_undercoverage_gt_5pp=("observed_undercoverage_gt_5pp", "sum"),
        )
        .reset_index()
    )

    stress = (
        results.groupby(["method", "random_effect", "reported_se_multiplier"], sort=False)
        .agg(
            n_cells=("cell_index", "size"),
            mean_latent_coverage=("latent_coverage", "mean"),
            mean_observed_coverage=("observed_coverage", "mean"),
            median_observed_width=("observed_mean_width", "median"),
            mean_observed_interval_score=("observed_mean_interval_score", "mean"),
            mean_latent_sign_error=("latent_sign_error_rate", "mean"),
            mean_observed_sign_error=("observed_sign_error_rate", "mean"),
            max_failure_rate=("failure_rate", "max"),
        )
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "coverage_all_cells.tsv.gz"
    gate_path = args.output_dir / "coverage_nominal_gate.tsv"
    stress_path = args.output_dir / "coverage_stress_summary.tsv"
    checksums_path = args.output_dir / "input_checksums.tsv.gz"
    results.to_csv(all_path, sep="\t", index=False, compression="gzip")
    gate.to_csv(gate_path, sep="\t", index=False)
    stress.to_csv(stress_path, sep="\t", index=False)
    pd.DataFrame(input_checksums).to_csv(checksums_path, sep="\t", index=False, compression="gzip")
    run_manifest = {
        "grid_sha256": grid_sha256(cells),
        "n_cells": len(cells),
        "methods": list(METHODS),
        "n_result_rows": len(results),
        "simulation_code_sha256": next(iter(code_identities))[0],
        "nnf_code_sha256": next(iter(code_identities))[1],
        "outputs": {
            path.name: file_sha256(path)
            for path in (all_path, gate_path, stress_path, checksums_path)
        },
    }
    (args.output_dir / "aggregation_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, indent=2))
    print(gate.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
