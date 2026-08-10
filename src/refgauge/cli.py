"""Command line interface for RefGauge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .calibration import calibrate_loso, calibration_summary
from .reference import audit_references, compare_reference_pairs
from .simulation import write_demo
from .validation import validate_effects


def _read(path: Path) -> pd.DataFrame:
    suffixes = {suffix.lower() for suffix in path.suffixes}
    separator = "\t" if suffixes.intersection({".tsv", ".txt"}) else ","
    return pd.read_csv(path, sep=separator)


def _json_ready(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return clean.to_dict(orient="records")


def _at_least(minimum: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return parsed

    return parse


def _confidence_alpha(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 0.05:
        raise argparse.ArgumentTypeError("must be greater than 0 and no larger than 0.05")
    return parsed


def run_calibrate(args: argparse.Namespace) -> None:
    from .report import calibration_figure, write_html_report

    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    effects = validate_effects(_read(args.effects))
    predictions = calibrate_loso(
        effects,
        method=args.method,
        interval_method=args.interval_method,
        alpha=args.alpha,
        min_training_studies=args.min_training_studies,
        nnf_draws=args.nnf_draws,
        seed=args.seed,
        nnf_failure_policy=args.nnf_failure_policy,
    )
    summary = calibration_summary(predictions, strong_threshold=args.strong_threshold)
    predictions.to_csv(outdir / "loso_predictions.tsv", sep="\t", index=False)
    summary.to_csv(outdir / "calibration_summary.tsv", sep="\t", index=False)
    (outdir / "calibration_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    calibration_figure(predictions, outdir / "calibration_diagnostic.png")
    write_html_report(
        f"RefGauge leave-one-study-out calibration ({args.interval_method})",
        summary,
        "calibration_diagnostic.png",
        outdir / "report.html",
    )


def run_reference(args: argparse.Namespace) -> None:
    from .report import reference_figure, write_html_report

    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    contrasts = _read(args.contrasts)
    sensitivity, ranks = audit_references(contrasts)
    comparisons = compare_reference_pairs(contrasts)
    sensitivity.to_csv(outdir / "reference_feature_sensitivity.tsv", sep="\t", index=False)
    ranks.to_csv(outdir / "reference_rank_stability.tsv", sep="\t", index=False)
    comparisons.to_csv(outdir / "reference_pairwise_descriptive.tsv", sep="\t", index=False)
    reported_ranks = ranks.loc[ranks["rank_metrics_reported"]]
    summary = pd.DataFrame(
        [
            {
                "n_features": len(sensitivity),
                "median_estimate_range_noise_unadjusted": sensitivity["estimate_range"].median(),
                "fraction_raw_sign_flips_noise_unadjusted": sensitivity[
                    "raw_sign_flip_noise_unadjusted"
                ].mean(),
                "n_pairwise_descriptive_comparisons": len(comparisons),
                "pairwise_inference": "not_performed_shared_target_covariance_not_supplied",
                "n_rank_pairs_reported": len(reported_ranks),
                "n_rank_pairs_withheld_low_overlap": int((~ranks["rank_metrics_reported"]).sum()),
                "median_rank_spearman_reported_pairs": reported_ranks["spearman_rank"].median(),
                "median_top10_jaccard_reported_pairs": reported_ranks["top10_jaccard"].median(),
            }
        ]
    )
    summary.to_csv(outdir / "reference_summary.tsv", sep="\t", index=False)
    reference_figure(sensitivity, ranks, outdir / "reference_diagnostic.png")
    write_html_report(
        "RefGauge reference-anchor audit",
        summary,
        "reference_diagnostic.png",
        outdir / "report.html",
    )


def run_reference_standardize(args: argparse.Namespace) -> None:
    from .report import reference_standardization_figure, write_html_report
    from .standardization import standardize_reference_family

    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    design = _read(args.design) if args.design is not None else None
    bootstrap = _read(args.bootstrap) if args.bootstrap is not None else None
    result = standardize_reference_family(
        _read(args.edges),
        design=design,
        bootstrap=bootstrap,
        delta_ref=args.delta_ref,
        delta_bio=args.delta_bio,
        min_primary_sources=args.min_primary_sources,
        min_target_units=args.min_target_units,
        min_bootstrap_draws=args.min_bootstrap_draws,
        alpha=args.alpha,
    )

    outputs = {
        "reference_standardized_scores.tsv": result.scores,
        "anchor_excess.tsv": result.anchor_excess,
        "reference_baselines.tsv": result.reference_baselines,
        "reference_gates.tsv": result.gates,
        "source_deletions.tsv": result.source_deletions,
    }
    for filename, frame in outputs.items():
        frame.to_csv(outdir / filename, sep="\t", index=False)

    status_counts = (
        result.scores["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="n")
        if "status" in result.scores
        else pd.DataFrame([{"status": "UNKNOWN", "n": len(result.scores)}])
    )
    gate_column = "gate_id" if "gate_id" in result.gates else "gate"
    gate_counts = (
        result.gates.groupby([gate_column, "passed"], dropna=False)
        .size()
        .reset_index(name="n")
        if {gate_column, "passed"}.issubset(result.gates.columns)
        else pd.DataFrame()
    )
    summary = status_counts.copy()
    summary["standardised_scalar_emitted"] = ~summary["status"].astype(str).str.startswith(
        "ABSTAIN_"
    )
    payload = {
        "status_counts": _json_ready(status_counts),
        "gate_counts": _json_ready(gate_counts),
        "contract": {
            "design_supplied": args.design is not None,
            "delta_ref": args.delta_ref,
            "delta_bio": args.delta_bio,
            "alpha": args.alpha,
            "min_primary_sources": args.min_primary_sources,
            "min_target_units": args.min_target_units,
            "min_bootstrap_draws": args.min_bootstrap_draws,
        },
    }
    (outdir / "reference_standardization_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    reference_standardization_figure(
        result.anchor_excess, result.gates, outdir / "reference_standardization_diagnostic.png"
    )
    write_html_report(
        "RefGauge reference-family standardisation",
        summary,
        "reference_standardization_diagnostic.png",
        outdir / "report.html",
        note=(
            "A score is emitted only when the declared finite reference family passes "
            "the expected-cell declaration, direct-crossing, "
            "independence, joint-uncertainty, and reference-stability gates. Failed gates preserve "
            "anchor-specific diagnostics and produce a machine-readable abstention."
        ),
    )


def run_demo(args: argparse.Namespace) -> None:
    inputs = args.output / "inputs"
    effect_path, reference_path = write_demo(inputs, seed=args.seed)
    run_calibrate(
        argparse.Namespace(
            effects=effect_path,
            output=args.output / "calibration",
            method="reml",
            interval_method="reml_mkh_approx",
            alpha=0.05,
            min_training_studies=3,
            nnf_draws=5_000,
            seed=args.seed,
            nnf_failure_policy="raise",
            strong_threshold=0.5,
        )
    )
    run_reference(
        argparse.Namespace(contrasts=reference_path, output=args.output / "reference_audit")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="refgauge", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="Run leave-one-study-out transport calibration")
    cal.add_argument("--effects", type=Path, required=True)
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--method", choices=["reml", "dl"], default="reml")
    cal.add_argument(
        "--interval-method",
        choices=[
            "nnf_cd",
            "fixed_normal",
            "dl_normal",
            "reml_normal",
            "reml_mkh_approx",
            "unweighted_t",
        ],
        default="nnf_cd",
    )
    cal.add_argument("--alpha", type=float, default=0.05)
    cal.add_argument(
        "--min-training-studies",
        type=int,
        default=None,
        help=(
            "Defaults to 5 for NNF and 3 for comparators; NNF below 5 is for method validation only"
        ),
    )
    cal.add_argument("--nnf-draws", type=int, default=25_000)
    cal.add_argument(
        "--nnf-failure-policy",
        choices=["record", "raise"],
        default="record",
        help="Record numerical failures without fallback, or abort immediately",
    )
    cal.add_argument("--seed", type=int, default=20260801)
    cal.add_argument("--strong-threshold", type=float, default=0.5)
    cal.set_defaults(func=run_calibrate)

    ref = sub.add_parser("reference-audit", help="Quantify sensitivity to reference choice")
    ref.add_argument("--contrasts", type=Path, required=True)
    ref.add_argument("--output", type=Path, required=True)
    ref.set_defaults(func=run_reference)

    standardize = sub.add_parser(
        "reference-standardize",
        help="Standardise over a finite primary-reference family or abstain",
    )
    standardize.add_argument("--edges", type=Path, required=True)
    standardize.add_argument(
        "--design",
        type=Path,
        default=None,
        help=(
            "Expected source-level OP/PP cells; omission preserves point diagnostics "
            "but forces a declared-design abstention"
        ),
    )
    standardize.add_argument(
        "--bootstrap",
        type=Path,
        default=None,
        help="Joint source-level bootstrap edge table; omission forces an uncertainty abstention",
    )
    standardize.add_argument("--output", type=Path, required=True)
    standardize.add_argument(
        "--delta-ref",
        type=float,
        default=None,
        help="Prespecified maximum tolerable residual reference dependence",
    )
    standardize.add_argument(
        "--delta-bio",
        type=float,
        default=None,
        help="Optional prespecified biological excess-discrepancy margin",
    )
    standardize.add_argument("--alpha", type=_confidence_alpha, default=0.05)
    standardize.add_argument("--min-primary-sources", type=_at_least(6), default=6)
    standardize.add_argument("--min-target-units", type=_at_least(3), default=3)
    standardize.add_argument("--min-bootstrap-draws", type=_at_least(5_000), default=5_000)
    standardize.set_defaults(func=run_reference_standardize)

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
