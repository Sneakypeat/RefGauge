#!/usr/bin/env python3
"""Chunkable coverage simulations for RefGauge prediction intervals.

Each full-grid cell is independently reproducible.  A Slurm array can map its
task ID directly to ``--cell-index`` and safely write one TSV/JSON pair per
task.  NNF errors are recorded as failures and never trigger a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from refgauge.prediction import NNFComputationError, nnf_prediction_interval

HERE = Path(__file__).resolve().parent
FULL_MANIFEST = HERE / "full_grid_manifest.json"
SMOKE_MANIFEST = HERE / "smoke_profile.json"

METHODS = (
    "nnf_cd",
    "fixed_normal",
    "dl_normal",
    "reml_normal",
    "reml_mkh_approx",
    "unweighted_t",
)
OUTPUT_COLUMNS = (
    "cell_index",
    "method",
    "k",
    "tau_se_ratio",
    "true_tau2",
    "se_pattern",
    "random_effect",
    "reported_se_multiplier",
    "base_se",
    "alpha",
    "replicates_requested",
    "replicates_successful",
    "replicates_failed",
    "failure_rate",
    "failure_types",
    "latent_coverage",
    "latent_coverage_mcse",
    "observed_coverage",
    "observed_coverage_mcse",
    "latent_mean_width",
    "observed_mean_width",
    "latent_mean_interval_score",
    "observed_mean_interval_score",
    "latent_center_bias",
    "latent_center_rmse",
    "observed_center_bias",
    "observed_center_rmse",
    "latent_sign_error_rate",
    "observed_sign_error_rate",
    "tau2_bias",
    "tau2_rmse",
    "mean_zero_boundary_mass",
    "tau2_boundary_rate",
    "nnf_draws",
    "master_seed",
)


@dataclass(frozen=True)
class CellSpec:
    """One immutable cell in the frozen simulation grid."""

    cell_index: int
    k: int
    tau_se_ratio: float
    se_pattern: str
    random_effect: str
    reported_se_multiplier: float


@dataclass(frozen=True)
class MethodResult:
    """Per-replicate interval and estimation results for one method."""

    latent_low: float
    latent_high: float
    observed_low: float
    observed_high: float
    center: float
    tau2_estimate: float
    zero_boundary_mass: float
    tau2_at_boundary: bool


@dataclass(frozen=True)
class ReplicateResult:
    """Simulation truth and method results for one deterministic replicate."""

    latent_target: float
    observed_target: float
    methods: dict[str, MethodResult]
    failures: dict[str, str]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_grid_payload(cells: list[CellSpec]) -> bytes:
    records = [asdict(cell) for cell in cells]
    return json.dumps(records, sort_keys=True, separators=(",", ":")).encode()


def grid_sha256(cells: list[CellSpec]) -> str:
    """Return the stable SHA-256 identity of an expanded grid."""

    return hashlib.sha256(_canonical_grid_payload(cells)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_full_grid(*, verify_hash: bool = True) -> tuple[dict[str, Any], list[CellSpec]]:
    """Load and expand the frozen Cartesian grid in its declared axis order."""

    manifest = _load_json(FULL_MANIFEST)
    axis_order = manifest["axis_order"]
    axes = manifest["axes"]
    combinations = itertools.product(*(axes[name] for name in axis_order))
    cells = [
        CellSpec(cell_index=index, **dict(zip(axis_order, values, strict=True)))
        for index, values in enumerate(combinations)
    ]
    if len(cells) != manifest["expected_cells"]:
        raise RuntimeError(
            f"expanded grid has {len(cells)} cells, expected {manifest['expected_cells']}"
        )
    actual_hash = grid_sha256(cells)
    expected_hash = manifest["grid_sha256"]
    if verify_hash and actual_hash != expected_hash:
        raise RuntimeError(
            "full-grid hash mismatch; the frozen manifest changed "
            f"({actual_hash} != {expected_hash})"
        )
    return manifest, cells


def _seed(master_seed: int, cell_index: int, replicate_index: int, stream: int) -> int:
    sequence = np.random.SeedSequence([master_seed, cell_index, replicate_index, stream])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _standardized_random_effect(rng: np.random.Generator, size: int, kind: str) -> np.ndarray:
    if kind == "normal":
        return rng.standard_normal(size)
    if kind == "t3":
        return rng.standard_t(df=3, size=size) / math.sqrt(3.0)
    if kind == "skew":
        return rng.exponential(scale=1.0, size=size) - 1.0
    raise ValueError(f"unknown random-effect distribution: {kind}")


def _true_standard_errors(k: int, base_se: float, pattern: str) -> np.ndarray:
    if pattern == "balanced":
        return np.full(k, base_se, dtype=float)
    if pattern == "unbalanced_4fold":
        # The geometric mean remains base_se while max(SE) / min(SE) is exactly four.
        return base_se * np.geomspace(0.5, 2.0, k)
    raise ValueError(f"unknown SE pattern: {pattern}")


def _reml_tau2(estimate: np.ndarray, reported_se: np.ndarray) -> float:
    """Strict independent REML implementation for the approximate comparator."""

    variance = reported_se**2

    def objective(tau2: float) -> float:
        total = variance + tau2
        weights = 1.0 / total
        mean = float(np.sum(weights * estimate) / np.sum(weights))
        return float(
            0.5
            * (
                np.sum(np.log(total))
                + np.log(np.sum(weights))
                + np.sum(weights * (estimate - mean) ** 2)
            )
        )

    scale = max(float(np.var(estimate, ddof=1)), float(np.max(variance)), 1e-12)
    upper = 100.0 * scale
    fitted = optimize.minimize_scalar(
        objective,
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": max(1e-14, scale * 1e-10)},
    )
    if not fitted.success or not np.isfinite(fitted.fun) or not np.isfinite(fitted.x):
        raise RuntimeError(f"REML optimization failed: {fitted.message}")

    tau2 = float(max(0.0, fitted.x))
    boundary_tolerance = max(1e-12, scale * 1e-8)
    if objective(0.0) <= objective(tau2) + boundary_tolerance:
        return 0.0
    return tau2


def _dl_tau2(estimate: np.ndarray, reported_se: np.ndarray) -> float:
    weights = 1.0 / reported_se**2
    pooled = float(np.sum(weights * estimate) / np.sum(weights))
    q_value = float(np.sum(weights * (estimate - pooled) ** 2))
    denominator = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    return max(0.0, (q_value - (len(estimate) - 1)) / denominator) if denominator > 0 else 0.0


def normal_plugin_interval(
    estimate: np.ndarray,
    reported_se: np.ndarray,
    *,
    target_se: float,
    alpha: float,
    tau2_method: str,
) -> MethodResult:
    """Conventional normal prediction interval with a plug-in tau-squared."""

    if tau2_method == "fixed":
        tau2 = 0.0
    elif tau2_method == "dl":
        tau2 = _dl_tau2(estimate, reported_se)
    elif tau2_method == "reml":
        tau2 = _reml_tau2(estimate, reported_se)
    else:
        raise ValueError(f"unknown tau-squared method: {tau2_method}")

    weights = 1.0 / (reported_se**2 + tau2)
    pooled = float(np.sum(weights * estimate) / np.sum(weights))
    mean_variance = float(1.0 / np.sum(weights))
    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    latent_half_width = critical * math.sqrt(tau2 + mean_variance)
    observed_half_width = critical * math.sqrt(tau2 + mean_variance + target_se**2)
    return MethodResult(
        latent_low=pooled - latent_half_width,
        latent_high=pooled + latent_half_width,
        observed_low=pooled - observed_half_width,
        observed_high=pooled + observed_half_width,
        center=pooled,
        tau2_estimate=tau2,
        zero_boundary_mass=float("nan"),
        tau2_at_boundary=tau2 == 0.0,
    )


def unweighted_t_interval(
    estimate: np.ndarray,
    reported_se: np.ndarray,
    *,
    target_se: float,
    alpha: float,
) -> MethodResult:
    """Unweighted mean predictor with a moment heterogeneity estimate."""

    k = len(estimate)
    pooled = float(np.mean(estimate))
    observed_variance = float(np.var(estimate, ddof=1))
    average_sampling_variance = float(np.mean(reported_se**2))
    tau2 = max(0.0, observed_variance - average_sampling_variance)
    mean_variance = (tau2 + average_sampling_variance) / k
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=k - 1))
    latent_half_width = critical * math.sqrt(tau2 + mean_variance)
    observed_half_width = critical * math.sqrt(tau2 + mean_variance + target_se**2)
    return MethodResult(
        latent_low=pooled - latent_half_width,
        latent_high=pooled + latent_half_width,
        observed_low=pooled - observed_half_width,
        observed_high=pooled + observed_half_width,
        center=pooled,
        tau2_estimate=tau2,
        zero_boundary_mass=float("nan"),
        tau2_at_boundary=tau2 == 0.0,
    )


def reml_mkh_approx_interval(
    estimate: np.ndarray,
    reported_se: np.ndarray,
    *,
    target_se: float,
    alpha: float,
) -> MethodResult:
    """REML/mKH Riley-style approximate latent and observed prediction intervals."""

    k = len(estimate)
    if k < 3:
        raise ValueError("REML/mKH prediction intervals require at least three studies")
    tau2 = _reml_tau2(estimate, reported_se)
    weights = 1.0 / (reported_se**2 + tau2)
    pooled = float(np.sum(weights * estimate) / np.sum(weights))
    residual_scale = float(np.sum(weights * (estimate - pooled) ** 2) / (k - 1))
    mean_variance = float(max(1.0, residual_scale) / np.sum(weights))
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=k - 2))
    latent_half_width = critical * math.sqrt(tau2 + mean_variance)
    observed_half_width = critical * math.sqrt(tau2 + mean_variance + target_se**2)
    return MethodResult(
        latent_low=pooled - latent_half_width,
        latent_high=pooled + latent_half_width,
        observed_low=pooled - observed_half_width,
        observed_high=pooled + observed_half_width,
        center=pooled,
        tau2_estimate=tau2,
        zero_boundary_mass=float("nan"),
        tau2_at_boundary=tau2 == 0.0,
    )


def _simulate_replicate(
    arguments: tuple[CellSpec, int, int, int, float, float, int],
) -> ReplicateResult:
    cell, replicate_index, master_seed, nnf_draws, true_mean, base_se, alpha_scaled = arguments
    alpha = alpha_scaled / 1_000_000.0
    rng = np.random.default_rng(_seed(master_seed, cell.cell_index, replicate_index, 0))

    true_se = _true_standard_errors(cell.k, base_se, cell.se_pattern)
    true_tau = cell.tau_se_ratio * base_se
    latent_studies = true_mean + true_tau * _standardized_random_effect(
        rng, cell.k, cell.random_effect
    )
    estimate = latent_studies + rng.normal(0.0, true_se)
    reported_se = true_se * cell.reported_se_multiplier

    latent_target = float(
        true_mean + true_tau * _standardized_random_effect(rng, 1, cell.random_effect)[0]
    )
    true_target_se = base_se
    observed_target = float(latent_target + rng.normal(0.0, true_target_se))
    reported_target_se = true_target_se * cell.reported_se_multiplier

    methods: dict[str, MethodResult] = {}
    failures: dict[str, str] = {}

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="NNF draws below 5,000",
                category=RuntimeWarning,
            )
            nnf = nnf_prediction_interval(
                estimate,
                reported_se,
                target_se=reported_target_se,
                alpha=alpha,
                draws=nnf_draws,
                seed=_seed(master_seed, cell.cell_index, replicate_index, 1),
            )
        tau2_point = _reml_tau2(estimate, reported_se)
        point_weights = 1.0 / (reported_se**2 + tau2_point)
        point_estimate = float(np.sum(point_weights * estimate) / np.sum(point_weights))
        methods["nnf_cd"] = MethodResult(
            latent_low=nnf.latent.low,
            latent_high=nnf.latent.high,
            observed_low=nnf.observed.low,
            observed_high=nnf.observed.high,
            center=point_estimate,
            tau2_estimate=nnf.tau2_median,
            zero_boundary_mass=nnf.zero_boundary_mass,
            tau2_at_boundary=nnf.tau2_median == 0.0,
        )
    except (NNFComputationError, ValueError, FloatingPointError) as error:
        failures["nnf_cd"] = f"{type(error).__name__}: {error}"

    try:
        methods["reml_mkh_approx"] = reml_mkh_approx_interval(
            estimate,
            reported_se,
            target_se=reported_target_se,
            alpha=alpha,
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        failures["reml_mkh_approx"] = f"{type(error).__name__}: {error}"

    for method, tau2_method in (
        ("fixed_normal", "fixed"),
        ("dl_normal", "dl"),
        ("reml_normal", "reml"),
    ):
        try:
            methods[method] = normal_plugin_interval(
                estimate,
                reported_se,
                target_se=reported_target_se,
                alpha=alpha,
                tau2_method=tau2_method,
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            failures[method] = f"{type(error).__name__}: {error}"

    try:
        methods["unweighted_t"] = unweighted_t_interval(
            estimate,
            reported_se,
            target_se=reported_target_se,
            alpha=alpha,
        )
    except (RuntimeError, ValueError, FloatingPointError) as error:
        failures["unweighted_t"] = f"{type(error).__name__}: {error}"

    return ReplicateResult(
        latent_target=latent_target,
        observed_target=observed_target,
        methods=methods,
        failures=failures,
    )


def _interval_score(low: float, high: float, target: float, alpha: float) -> float:
    score = high - low
    if target < low:
        score += 2.0 * (low - target) / alpha
    elif target > high:
        score += 2.0 * (target - high) / alpha
    return float(score)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _rmse(values: list[float]) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def _coverage_mcse(coverage: float | None, n: int) -> float | None:
    return float(math.sqrt(coverage * (1.0 - coverage) / n)) if n and coverage is not None else None


def _aggregate_method(
    cell: CellSpec,
    method: str,
    replicates: list[ReplicateResult],
    *,
    manifest: dict[str, Any],
    nnf_draws: int,
) -> dict[str, Any]:
    successes: list[tuple[ReplicateResult, MethodResult]] = []
    failure_counter: Counter[str] = Counter()
    for replicate in replicates:
        if method in replicate.methods:
            successes.append((replicate, replicate.methods[method]))
        else:
            message = replicate.failures.get(method, "unclassified failure")
            failure_counter[message] += 1

    latent_covered: list[float] = []
    observed_covered: list[float] = []
    latent_width: list[float] = []
    observed_width: list[float] = []
    latent_score: list[float] = []
    observed_score: list[float] = []
    latent_error: list[float] = []
    observed_error: list[float] = []
    latent_sign_error: list[float] = []
    observed_sign_error: list[float] = []
    tau_error: list[float] = []
    zero_mass: list[float] = []
    at_boundary: list[float] = []
    alpha = float(manifest["alpha"])
    true_tau2 = (cell.tau_se_ratio * float(manifest["base_se"])) ** 2

    for replicate, result in successes:
        latent_covered.append(
            float(result.latent_low <= replicate.latent_target <= result.latent_high)
        )
        observed_covered.append(
            float(result.observed_low <= replicate.observed_target <= result.observed_high)
        )
        latent_width.append(result.latent_high - result.latent_low)
        observed_width.append(result.observed_high - result.observed_low)
        latent_score.append(
            _interval_score(result.latent_low, result.latent_high, replicate.latent_target, alpha)
        )
        observed_score.append(
            _interval_score(
                result.observed_low, result.observed_high, replicate.observed_target, alpha
            )
        )
        latent_error.append(result.center - replicate.latent_target)
        observed_error.append(result.center - replicate.observed_target)
        latent_sign_error.append(float(np.sign(result.center) != np.sign(replicate.latent_target)))
        observed_sign_error.append(
            float(np.sign(result.center) != np.sign(replicate.observed_target))
        )
        tau_error.append(result.tau2_estimate - true_tau2)
        if np.isfinite(result.zero_boundary_mass):
            zero_mass.append(result.zero_boundary_mass)
        at_boundary.append(float(result.tau2_at_boundary))

    n_success = len(successes)
    latent_coverage = _mean(latent_covered)
    observed_coverage = _mean(observed_covered)
    row = {
        "cell_index": cell.cell_index,
        "method": method,
        "k": cell.k,
        "tau_se_ratio": cell.tau_se_ratio,
        "true_tau2": true_tau2,
        "se_pattern": cell.se_pattern,
        "random_effect": cell.random_effect,
        "reported_se_multiplier": cell.reported_se_multiplier,
        "base_se": manifest["base_se"],
        "alpha": alpha,
        "replicates_requested": len(replicates),
        "replicates_successful": n_success,
        "replicates_failed": len(replicates) - n_success,
        "failure_rate": 1.0 - n_success / len(replicates),
        "failure_types": json.dumps(failure_counter, sort_keys=True),
        "latent_coverage": latent_coverage,
        "latent_coverage_mcse": _coverage_mcse(latent_coverage, n_success),
        "observed_coverage": observed_coverage,
        "observed_coverage_mcse": _coverage_mcse(observed_coverage, n_success),
        "latent_mean_width": _mean(latent_width),
        "observed_mean_width": _mean(observed_width),
        "latent_mean_interval_score": _mean(latent_score),
        "observed_mean_interval_score": _mean(observed_score),
        "latent_center_bias": _mean(latent_error),
        "latent_center_rmse": _rmse(latent_error),
        "observed_center_bias": _mean(observed_error),
        "observed_center_rmse": _rmse(observed_error),
        "latent_sign_error_rate": _mean(latent_sign_error),
        "observed_sign_error_rate": _mean(observed_sign_error),
        "tau2_bias": _mean(tau_error),
        "tau2_rmse": _rmse(tau_error),
        "mean_zero_boundary_mass": _mean(zero_mass),
        "tau2_boundary_rate": _mean(at_boundary),
        "nnf_draws": nnf_draws if method == "nnf_cd" else 0,
        "master_seed": manifest["master_seed"],
    }
    return {column: row[column] for column in OUTPUT_COLUMNS}


def run_cell(
    cell: CellSpec,
    *,
    replicates: int,
    nnf_draws: int,
    jobs: int,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run one full-grid cell and return one aggregate row per method."""

    if replicates < 1:
        raise ValueError("replicates must be at least one")
    if nnf_draws < 100:
        raise ValueError("nnf_draws must be at least 100")
    if jobs < 1:
        raise ValueError("jobs must be at least one")

    alpha_scaled = int(round(float(manifest["alpha"]) * 1_000_000))
    arguments = [
        (
            cell,
            replicate_index,
            int(manifest["master_seed"]),
            nnf_draws,
            float(manifest["true_mean"]),
            float(manifest["base_se"]),
            alpha_scaled,
        )
        for replicate_index in range(replicates)
    ]
    if jobs == 1:
        simulated = [_simulate_replicate(item) for item in arguments]
    else:
        try:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                simulated = list(executor.map(_simulate_replicate, arguments, chunksize=1))
        except (OSError, PermissionError) as error:
            warnings.warn(
                f"process workers unavailable ({error}); using {jobs} thread workers",
                RuntimeWarning,
                stacklevel=2,
            )
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                simulated = list(executor.map(_simulate_replicate, arguments))

    return [
        _aggregate_method(cell, method, simulated, manifest=manifest, nnf_draws=nnf_draws)
        for method in METHODS
    ]


def run_cells(
    cells: list[CellSpec],
    *,
    replicates: int,
    nnf_draws: int,
    jobs: int,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run cells in stable index order."""

    rows: list[dict[str, Any]] = []
    for cell in sorted(cells, key=lambda item: item.cell_index):
        rows.extend(
            run_cell(
                cell,
                replicates=replicates,
                nnf_draws=nnf_draws,
                jobs=jobs,
                manifest=manifest,
            )
        )
    return rows


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_outputs(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    stem: str,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically write tabular and structured versions of one chunk."""

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    tsv_path = output_dir / f"{stem}.tsv"
    json_path = output_dir / f"{stem}.json"
    _atomic_write_text(tsv_path, frame.to_csv(sep="\t", index=False, na_rep="NA"))
    json_rows = [
        {
            key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in row.items()
        }
        for row in rows
    ]
    payload = {"metadata": metadata, "results": json_rows}
    _atomic_write_text(json_path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return tsv_path, json_path


def _selected_cells(profile: str, cell_index: int | None) -> tuple[dict[str, Any], list[CellSpec]]:
    manifest, full_grid = load_full_grid()
    if cell_index is not None:
        if not 0 <= cell_index < len(full_grid):
            raise ValueError(f"cell_index must be between 0 and {len(full_grid) - 1}")
        return manifest, [full_grid[cell_index]]
    if profile == "full":
        return manifest, full_grid
    smoke = _load_json(SMOKE_MANIFEST)
    return manifest, [full_grid[index] for index in smoke["full_grid_cell_indices"]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--cell-index",
        type=int,
        help="zero-based full-grid cell; ideal for SLURM_ARRAY_TASK_ID",
    )
    parser.add_argument("--replicates", type=int, help="override profile replicate count")
    parser.add_argument("--nnf-draws", type=int, help="override profile NNF Monte Carlo draws")
    parser.add_argument("--jobs", type=int, default=1, help="parallel replicate workers")
    parser.add_argument("--output-dir", type=Path, default=Path("coverage_results"))
    parser.add_argument(
        "--print-grid-hash",
        action="store_true",
        help="print the expanded grid hash without running simulations",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest, full_grid = load_full_grid(verify_hash=False)
    if args.print_grid_hash:
        print(grid_sha256(full_grid))
        return 0

    manifest, cells = _selected_cells(args.profile, args.cell_index)
    profile_manifest = _load_json(SMOKE_MANIFEST) if args.profile == "smoke" else manifest
    replicates = args.replicates or int(profile_manifest["default_replicates"])
    nnf_draws = args.nnf_draws or int(profile_manifest["default_nnf_draws"])
    rows = run_cells(
        cells,
        replicates=replicates,
        nnf_draws=nnf_draws,
        jobs=args.jobs,
        manifest=manifest,
    )

    stem = (
        f"coverage_cell_{args.cell_index:04d}"
        if args.cell_index is not None
        else f"coverage_{args.profile}"
    )
    metadata = {
        "schema_version": 1,
        "profile": args.profile,
        "cell_indices": [cell.cell_index for cell in cells],
        "grid_sha256": grid_sha256(full_grid),
        "replicates": replicates,
        "nnf_draws": nnf_draws,
        "jobs": args.jobs,
        "nnf_failure_policy": "record_failure_no_fallback",
        "latent_target": "true effect in an independent new study",
        "observed_target": "new-study estimate with true sampling SE; method receives reported SE",
        "simulation_code_sha256": _file_sha256(Path(__file__).resolve()),
        "nnf_code_sha256": _file_sha256(
            Path(__file__).resolve().parents[2] / "src/refgauge/prediction.py"
        ),
    }
    tsv_path, json_path = write_outputs(
        rows,
        output_dir=args.output_dir,
        stem=stem,
        metadata=metadata,
    )
    print(f"Wrote {tsv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
