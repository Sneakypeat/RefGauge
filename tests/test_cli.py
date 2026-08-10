import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from refgauge.cli import main


def test_demo_writes_complete_reports(tmp_path: Path):
    assert main(["demo", "--output", str(tmp_path), "--seed", "51"]) == 0
    expected = [
        tmp_path / "inputs" / "synthetic_effects.tsv",
        tmp_path / "calibration" / "loso_predictions.tsv",
        tmp_path / "calibration" / "calibration_summary.json",
        tmp_path / "calibration" / "report.html",
        tmp_path / "reference_audit" / "reference_feature_sensitivity.tsv",
        tmp_path / "reference_audit" / "reference_pairwise_descriptive.tsv",
        tmp_path / "reference_audit" / "reference_summary.tsv",
        tmp_path / "reference_audit" / "report.html",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)


def test_reference_audit_accepts_compressed_tsv(tmp_path: Path):
    contrasts = pd.DataFrame(
        [
            {
                "feature": f"gene_{feature}",
                "target": "organoid",
                "reference": reference,
                "estimate": feature / 10 + offset,
                "se": 0.2,
            }
            for feature in range(12)
            for reference, offset in (("fetal", 0.0), ("adult", 0.5))
        ]
    )
    input_path = tmp_path / "references.tsv.gz"
    output_path = tmp_path / "reference_audit"
    contrasts.to_csv(input_path, sep="\t", index=False, compression="gzip")

    assert (
        main(["reference-audit", "--contrasts", str(input_path), "--output", str(output_path)]) == 0
    )
    assert (output_path / "reference_summary.tsv").is_file()


def test_cli_import_does_not_import_matplotlib():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import refgauge.cli; assert 'matplotlib' not in sys.modules",
        ],
        check=False,
    )
    assert result.returncode == 0


def test_package_module_exposes_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "refgauge", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "refgauge 0.3.0"
    assert result.stderr == ""


def _underidentified_reference_edges() -> pd.DataFrame:
    primary_sources = [f"primary_{index}" for index in range(4)]
    rows = []
    for target_source in primary_sources:
        for reference_source in primary_sources:
            if target_source == reference_source:
                continue
            rows.append(
                {
                    "domain": "marker_transfer",
                    "feature": "neural_progenitor",
                    "edge_type": "PP",
                    "target": target_source,
                    "target_unit": f"{target_source}_batch",
                    "target_source": target_source,
                    "reference": f"{reference_source}_anchor",
                    "reference_source": reference_source,
                    "estimate": 0.1,
                    "admissible": True,
                    "leakage": False,
                }
            )
    for reference_source in primary_sources:
        rows.append(
            {
                "domain": "marker_transfer",
                "feature": "neural_progenitor",
                "edge_type": "OP",
                "target": "organoid_protocol",
                "target_unit": "cell_line_1",
                "target_source": "organoid_study",
                "reference": f"{reference_source}_anchor",
                "reference_source": reference_source,
                "estimate": 0.2,
                "admissible": True,
                "leakage": False,
            }
        )
    return pd.DataFrame(rows)


def _reference_design(edges: pd.DataFrame) -> pd.DataFrame:
    return edges[
        [
            "domain",
            "feature",
            "edge_type",
            "target",
            "target_unit",
            "target_source",
            "reference_source",
        ]
    ].drop_duplicates()


def test_reference_standardize_writes_abstention_and_diagnostics(tmp_path: Path):
    edges_path = tmp_path / "edges.tsv"
    design_path = tmp_path / "design.tsv"
    output_path = tmp_path / "standardization"
    edges = _underidentified_reference_edges()
    edges.to_csv(edges_path, sep="\t", index=False)
    _reference_design(edges).to_csv(design_path, sep="\t", index=False)

    assert (
        main(
            [
                "reference-standardize",
                "--edges",
                str(edges_path),
                "--design",
                str(design_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    scores = pd.read_csv(output_path / "reference_standardized_scores.tsv", sep="\t")
    assert scores.loc[0, "status"] == "ABSTAIN_INCOMPLETE_REFERENCE_DESIGN"
    assert bool(scores.loc[0, "descriptive_only"])
    expected = {
        "anchor_excess.tsv",
        "reference_baselines.tsv",
        "reference_gates.tsv",
        "source_deletions.tsv",
        "reference_standardization_summary.json",
        "reference_standardization_diagnostic.png",
        "report.html",
    }
    assert expected.issubset({path.name for path in output_path.iterdir()})


def test_reference_standardize_without_design_writes_named_abstention(tmp_path: Path):
    edges_path = tmp_path / "edges.tsv"
    output_path = tmp_path / "standardization"
    _underidentified_reference_edges().to_csv(edges_path, sep="\t", index=False)

    assert (
        main(
            [
                "reference-standardize",
                "--edges",
                str(edges_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    scores = pd.read_csv(output_path / "reference_standardized_scores.tsv", sep="\t")
    assert scores.loc[0, "status"] == "ABSTAIN_UNDECLARED_REFERENCE_DESIGN"
    summary = json.loads(
        (output_path / "reference_standardization_summary.json").read_text(encoding="utf-8")
    )
    assert summary["contract"]["design_supplied"] is False


def test_reference_standardize_cli_cannot_weaken_frozen_source_minimum(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "reference-standardize",
                "--edges",
                str(tmp_path / "missing.tsv"),
                "--output",
                str(tmp_path / "output"),
                "--min-primary-sources",
                "5",
            ]
        )
