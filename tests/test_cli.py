from pathlib import Path

from contrastcal.cli import main


def test_demo_writes_complete_reports(tmp_path: Path):
    assert main(["demo", "--output", str(tmp_path), "--seed", "51"]) == 0
    expected = [
        tmp_path / "inputs" / "synthetic_effects.tsv",
        tmp_path / "calibration" / "loso_predictions.tsv",
        tmp_path / "calibration" / "calibration_summary.json",
        tmp_path / "calibration" / "report.html",
        tmp_path / "reference_audit" / "reference_feature_sensitivity.tsv",
        tmp_path / "reference_audit" / "report.html",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)
