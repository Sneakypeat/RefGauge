"""Command line interface for ContrastCal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .calibration import calibrate_loso, calibration_summary
from .reference import audit_references
from .report import calibration_figure, reference_figure, write_html_report
from .simulation import write_demo
from .validation import validate_effects


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")


def _json_ready(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict(orient="records")


def run_calibrate(args: argparse.Namespace) -> None:
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    effects = validate_effects(_read(args.effects))
    predictions = calibrate_loso(effects, method=args.method, alpha=args.alpha)
    summary = calibration_summary(predictions, strong_threshold=args.strong_threshold)
    predictions.to_csv(outdir / "loso_predictions.tsv", sep="\t", index=False)
    summary.to_csv(outdir / "calibration_summary.tsv", sep="\t", index=False)
    (outdir / "calibration_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    calibration_figure(predictions, outdir / "calibration_diagnostic.png")
    write_html_report(
        "ContrastCal leave-one-study-out calibration",
        summary,
        "calibration_diagnostic.png",
        outdir / "report.html",
    )


def run_reference(args: argparse.Namespace) -> None:
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    sensitivity, ranks = audit_references(_read(args.contrasts))
    sensitivity.to_csv(outdir / "reference_feature_sensitivity.tsv", sep="\t", index=False)
    ranks.to_csv(outdir / "reference_rank_stability.tsv", sep="\t", index=False)
    summary = pd.DataFrame(
        [
            {
                "n_features": len(sensitivity),
                "median_estimate_range": sensitivity["estimate_range"].median(),
                "fraction_sign_flips": sensitivity["sign_flip"].mean(),
                "median_rank_spearman": ranks["spearman_rank"].median(),
                "median_top10_jaccard": ranks["top10_jaccard"].median(),
            }
        ]
    )
    summary.to_csv(outdir / "reference_summary.tsv", sep="\t", index=False)
    reference_figure(sensitivity, ranks, outdir / "reference_diagnostic.png")
    write_html_report(
        "ContrastCal reference-anchor audit",
        summary,
        "reference_diagnostic.png",
        outdir / "report.html",
    )


def run_demo(args: argparse.Namespace) -> None:
    inputs = args.output / "inputs"
    effect_path, reference_path = write_demo(inputs, seed=args.seed)
    run_calibrate(
        argparse.Namespace(
            effects=effect_path,
            output=args.output / "calibration",
            method="reml",
            alpha=0.05,
            strong_threshold=0.5,
        )
    )
    run_reference(
        argparse.Namespace(contrasts=reference_path, output=args.output / "reference_audit")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contrastcal", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="Run leave-one-study-out transport calibration")
    cal.add_argument("--effects", type=Path, required=True)
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--method", choices=["reml", "dl"], default="reml")
    cal.add_argument("--alpha", type=float, default=0.05)
    cal.add_argument("--strong-threshold", type=float, default=0.5)
    cal.set_defaults(func=run_calibrate)

    ref = sub.add_parser("reference-audit", help="Quantify sensitivity to reference choice")
    ref.add_argument("--contrasts", type=Path, required=True)
    ref.add_argument("--output", type=Path, required=True)
    ref.set_defaults(func=run_reference)

    demo = sub.add_parser("demo", help="Generate synthetic data and a complete example report")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--seed", type=int, default=20260801)
    demo.set_defaults(func=run_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
