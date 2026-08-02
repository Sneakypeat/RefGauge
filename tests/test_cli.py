import subprocess
import sys
from pathlib import Path

import pandas as pd

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
    assert result.stdout.strip() == "refgauge 0.2.0"
    assert result.stderr == ""
