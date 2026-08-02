#!/usr/bin/env python3
"""Chunkable multifeature transport-stress simulations for RefGauge.

The frozen design is a balanced fractional experiment rather than the
10,206-cell full Cartesian product. Each cell simulates multiple correlated
features, independent training studies, and a new target study. Features are
never treated as independent replicates for a paper-level significance test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from refgauge.prediction import NNFComputationError, nnf_prediction_interval

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "transport_stress_manifest.json"
METHODS = ("nnf_cd", "reml_mkh_approx")
CORE_FACTORS = ("p", "target_bias", "target_scale", "truth", "block_rho", "k")


@dataclass(frozen=True)
class CellSpec:
    """One immutable fractional-design cell."""

    cell_index: int
    p: int
    target_bias: float
    target_scale: float
    truth: str
    block_rho: float
    k: int
    missingness: str
    missingness_mechanism: str
    missingness_rate: float
    selection: str


@dataclass(frozen=True)
class IntervalResult:
    """Latent- and observed-target intervals for one feature and method."""

    latent_low: float
    latent_high: float
    observed_low: float
    observed_high: float


@dataclass
class ReplicateSummary:
    """Transport metrics plus feature-level interval outcomes."""

    metrics: dict[str, float | int]
    intervals: dict[str, list[dict[str, float]]]
    failures: dict[str, Counter[str]]


OUTPUT_COLUMNS = (
    "cell_index",
    "method",
    "p",
    "target_bias",
    "target_scale",
    "truth",
    "block_rho",
    "k",
    "missingness",
    "missingness_mechanism",
    "missingness_rate",
    "selection",
    "control_type",
    "replicates_requested",
    "replicates_successful",
    "features_total_mean",
    "features_selection_eligible_mean",
    "features_selected_mean",
    "features_selected_target_evaluable_mean",
    "features_transport_evaluable_mean",
    "features_target_missing_mean",
    "features_training_k_failed_mean",
    "shared_point_method",
    "method_point_metrics_status",
    "shared_transport_pearson_mean",
    "shared_transport_spearman_mean",
    "shared_transport_bias_mean",
    "shared_transport_rmse_mean",
    "shared_sign_error_rate_mean",
    "shared_eiv_intercept_mean",
    "shared_eiv_slope_mean",
    "pi_features_attempted",
    "pi_features_successful",
    "pi_features_failed",
    "pi_failure_rate",
    "pi_failure_types",
    "latent_coverage",
    "observed_coverage",
    "latent_mean_width",
    "observed_mean_width",
    "nnf_draws",
    "pi_evaluation_rule",
    "inferential_unit",
    "feature_level_p_value",
    "master_seed",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_payload(cells: list[CellSpec]) -> bytes:
    return json.dumps(
        [asdict(cell) for cell in cells], sort_keys=True, separators=(",", ":")
    ).encode()


def design_sha256(cells: list[CellSpec]) -> str:
    """Return the stable SHA-256 identity of the expanded fractional design."""

    return hashlib.sha256(_canonical_payload(cells)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand_design(manifest: dict[str, Any]) -> list[CellSpec]:
    levels = manifest["core_levels"]
    missingness = manifest["missingness_levels"]
    selection = manifest["selection_levels"]
    cells: list[CellSpec] = []
    index = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                core = {
                    "p": levels["p"][a],
                    "target_bias": levels["target_bias"][b],
                    "target_scale": levels["target_scale"][c],
                    "truth": levels["truth"][(a + b) % 3],
                    "block_rho": levels["block_rho"][(a + c) % 3],
                    "k": levels["k"][(b + c) % 3],
                }
                for miss in missingness:
                    for select in selection:
                        cells.append(
                            CellSpec(
                                cell_index=index,
                                **core,
                                missingness=miss["label"],
                                missingness_mechanism=miss["mechanism"],
                                missingness_rate=miss["rate"],
                                selection=select,
                            )
                        )
                        index += 1
    return cells


def load_design(*, verify_hash: bool = True) -> tuple[dict[str, Any], list[CellSpec]]:
    """Load, expand, and optionally verify the frozen fractional design."""

    manifest = _load_json(MANIFEST_PATH)
    cells = _expand_design(manifest)
    if len(cells) != manifest["expected_cells"]:
        raise RuntimeError(
            f"expanded design has {len(cells)} cells, expected {manifest['expected_cells']}"
        )
    actual = design_sha256(cells)
    if verify_hash and actual != manifest["design_sha256"]:
        raise RuntimeError(
            f"transport-stress design hash mismatch ({actual} != {manifest['design_sha256']})"
        )
    return manifest, cells


def _seed(master: int, cell: int, replicate: int, stream: int) -> int:
    sequence = np.random.SeedSequence([master, cell, replicate, stream])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _stable_rank(feature_indices: np.ndarray, *, salt: str) -> np.ndarray:
    keys = [
        hashlib.blake2b(f"{salt}\0{int(index)}".encode(), digest_size=8).digest()
        for index in feature_indices
    ]
    return np.argsort(np.asarray(keys, dtype="S8"), kind="stable")


def _block_normal(
    rng: np.random.Generator,
    n_rows: int,
    p: int,
    rho: float,
    block_size: int,
) -> np.ndarray:
    """Generate unit-variance Gaussian rows with exchangeable feature blocks."""

    if not 0 <= rho < 1:
        raise ValueError("block_rho must satisfy 0 <= rho < 1")
    n_blocks = math.ceil(p / block_size)
    common = rng.standard_normal((n_rows, n_blocks))
    independent = rng.standard_normal((n_rows, p))
    block_index = np.arange(p) // block_size
    return math.sqrt(rho) * common[:, block_index] + math.sqrt(1.0 - rho) * independent


def _truth_vector(rng: np.random.Generator, p: int, truth: str) -> np.ndarray:
    fractions = {"null": 0.0, "sparse_10pct": 0.1, "dense_70pct": 0.7}
    if truth not in fractions:
        raise ValueError(f"unknown truth regime: {truth}")
    beta = np.zeros(p, dtype=float)
    n_active = int(round(p * fractions[truth]))
    if n_active:
        active = rng.choice(p, size=n_active, replace=False)
        magnitude = np.exp(rng.normal(math.log(0.55), 0.25, size=n_active))
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_active)
        beta[active] = signs * magnitude
    return beta


def _missing_probabilities(
    mechanism: str,
    rate: float,
    abundance: np.ndarray,
    latent: np.ndarray,
) -> np.ndarray:
    """Return bounded probabilities with approximately the requested mean rate."""

    if rate == 0:
        return np.zeros_like(latent)
    if mechanism == "mcar":
        return np.full_like(latent, rate)
    if mechanism == "abundance_dependent":
        risk = np.broadcast_to(0.1 + 1.8 * (1.0 - abundance), latent.shape).copy()
    elif mechanism == "effect_dependent":
        absolute = np.abs(latent)
        ranks = np.argsort(np.argsort(absolute, axis=1), axis=1) / max(1, absolute.shape[1] - 1)
        risk = 0.1 + 1.8 * (1.0 - ranks)
    else:
        raise ValueError(f"unknown missingness mechanism: {mechanism}")
    probability = risk * (rate / float(np.mean(risk)))
    return np.clip(probability, 0.0, 0.95)


def _vectorized_random_effects(
    estimates: np.ndarray,
    ses: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DL random-effects summaries used for multifeature transport metrics."""

    safe_se = np.where(observed, ses, np.nan)
    safe_y = np.where(observed, estimates, np.nan)
    k_feature = np.sum(observed, axis=0)
    w0 = np.where(observed, 1.0 / safe_se**2, 0.0)
    sum_w0 = np.sum(w0, axis=0)
    fixed = np.divide(
        np.nansum(w0 * safe_y, axis=0),
        sum_w0,
        out=np.full(estimates.shape[1], np.nan),
        where=sum_w0 > 0,
    )
    q_value = np.nansum(w0 * (safe_y - fixed) ** 2, axis=0)
    c_value = sum_w0 - np.divide(
        np.sum(w0**2, axis=0),
        sum_w0,
        out=np.zeros_like(sum_w0),
        where=sum_w0 > 0,
    )
    tau2 = np.divide(
        q_value - np.maximum(k_feature - 1, 0),
        c_value,
        out=np.zeros_like(q_value),
        where=c_value > 0,
    )
    tau2 = np.maximum(tau2, 0.0)
    w = np.where(observed, 1.0 / (safe_se**2 + tau2), 0.0)
    sum_w = np.sum(w, axis=0)
    pooled = np.divide(
        np.nansum(w * safe_y, axis=0),
        sum_w,
        out=np.full(estimates.shape[1], np.nan),
        where=sum_w > 0,
    )
    pooled_variance = np.divide(
        1.0,
        sum_w,
        out=np.full(estimates.shape[1], np.nan),
        where=sum_w > 0,
    )
    return pooled, pooled_variance, k_feature


def _strict_reml_tau2(estimate: np.ndarray, se: np.ndarray) -> float:
    variance = se**2

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
    fit = optimize.minimize_scalar(
        objective,
        bounds=(0.0, 100.0 * scale),
        method="bounded",
        options={"xatol": max(1e-14, scale * 1e-10)},
    )
    if not fit.success or not np.isfinite(fit.fun) or not np.isfinite(fit.x):
        raise RuntimeError(f"REML optimization failed: {fit.message}")
    tau2 = float(max(0.0, fit.x))
    tolerance = max(1e-12, scale * 1e-8)
    return 0.0 if objective(0.0) <= objective(tau2) + tolerance else tau2


def _reml_mkh_interval(
    estimate: np.ndarray,
    se: np.ndarray,
    *,
    target_se: float,
    alpha: float,
) -> IntervalResult:
    k = len(estimate)
    if k < 3:
        raise ValueError("REML/mKH requires at least three studies")
    tau2 = _strict_reml_tau2(estimate, se)
    weights = 1.0 / (se**2 + tau2)
    pooled = float(np.sum(weights * estimate) / np.sum(weights))
    residual_scale = float(np.sum(weights * (estimate - pooled) ** 2) / (k - 1))
    mean_variance = float(max(1.0, residual_scale) / np.sum(weights))
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=k - 2))
    latent_half = critical * math.sqrt(tau2 + mean_variance)
    observed_half = critical * math.sqrt(tau2 + mean_variance + target_se**2)
    return IntervalResult(
        pooled - latent_half,
        pooled + latent_half,
        pooled - observed_half,
        pooled + observed_half,
    )


def _eiv_calibration(
    x: np.ndarray,
    y: np.ndarray,
    x_variance: np.ndarray,
    y_variance: np.ndarray,
) -> tuple[float, float]:
    keep = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(x_variance)
        & np.isfinite(y_variance)
        & (x_variance > 0)
        & (y_variance > 0)
    )
    x, y = x[keep], y[keep]
    x_variance, y_variance = x_variance[keep], y_variance[keep]
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    initial = np.polyfit(x, y, 1)

    def objective(parameters: np.ndarray) -> float:
        intercept, slope = parameters
        variance = y_variance + slope**2 * x_variance
        return float(np.sum((y - intercept - slope * x) ** 2 / variance))

    fit = optimize.minimize(
        objective,
        x0=np.array([initial[1], initial[0]]),
        method="BFGS",
    )
    if not fit.success or not np.isfinite(fit.x).all():
        return float("nan"), float("nan")
    return float(fit.x[0]), float(fit.x[1])


def _safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return float("nan")
    if method == "pearson":
        return float(stats.pearsonr(x[keep], y[keep]).statistic)
    return float(stats.spearmanr(x[keep], y[keep]).statistic)


def _select_features(
    cell: CellSpec,
    atlas: np.ndarray,
    atlas_variance: np.ndarray,
    target: np.ndarray,
    target_variance: np.ndarray,
    eligible: np.ndarray,
    fraction: float,
) -> np.ndarray:
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        return candidates
    n_select = max(3, int(math.ceil(len(candidates) * fraction)))
    if cell.selection == "training_only":
        score = np.abs(atlas[candidates]) / np.sqrt(atlas_variance[candidates])
        order = np.lexsort((candidates, -score))
    elif cell.selection == "target_informed_prohibited":
        discrepancy = np.abs(target[candidates] - atlas[candidates])
        discrepancy /= np.sqrt(atlas_variance[candidates] + target_variance[candidates])
        order = np.lexsort((candidates, discrepancy))
    else:
        raise ValueError(f"unknown selection mode: {cell.selection}")
    return candidates[order[: min(n_select, len(candidates))]]


def _pi_subset(selected: np.ndarray, *, n: int, cell: int, replicate: int) -> np.ndarray:
    if not len(selected):
        return selected
    order = _stable_rank(selected, salt=f"pi-evaluation:{cell}:{replicate}")
    return selected[order[: min(n, len(selected))]]


def _simulate_replicate(
    cell: CellSpec,
    *,
    replicate_index: int,
    manifest: dict[str, Any],
    nnf_draws: int,
    pi_features: int,
) -> ReplicateSummary:
    master = int(manifest["master_seed"])
    rng = np.random.default_rng(_seed(master, cell.cell_index, replicate_index, 0))
    p, k = cell.p, cell.k
    block_size = int(manifest["block_size"])
    beta = _truth_vector(rng, p, cell.truth)
    abundance = stats.rankdata(rng.lognormal(0.0, 1.0, p), method="average") / (p + 1.0)
    tau = 0.10 + 0.15 * np.minimum(np.abs(beta), 1.0)

    heterogeneity = _block_normal(rng, k + 1, p, cell.block_rho, block_size)
    sampling = _block_normal(rng, k + 1, p, cell.block_rho * 0.5, block_size)
    study_scale = rng.lognormal(0.0, 0.18, size=k + 1)
    base_se = 0.12 + 0.30 * (1.0 - abundance)
    true_se = study_scale[:, None] * base_se[None, :]

    training_latent = beta[None, :] + tau[None, :] * heterogeneity[:k]
    training_estimate = training_latent + true_se[:k] * sampling[:k]
    target_latent = cell.target_bias + cell.target_scale * beta + tau * heterogeneity[k]
    target_estimate = target_latent + true_se[k] * sampling[k]

    latent_all = np.vstack([training_latent, target_latent])
    missing_probability = _missing_probabilities(
        cell.missingness_mechanism,
        cell.missingness_rate,
        abundance,
        latent_all,
    )
    observed = rng.random((k + 1, p)) >= missing_probability
    training_observed, target_observed = observed[:k], observed[k]
    atlas, atlas_variance, k_feature = _vectorized_random_effects(
        training_estimate, true_se[:k], training_observed
    )

    min_k = int(manifest["minimum_training_studies"])
    training_valid = k_feature >= min_k
    selection_eligible = training_valid & np.isfinite(atlas)
    transport_valid = training_valid & target_observed & np.isfinite(atlas)
    selection_input = selection_eligible.copy()
    if cell.selection == "target_informed_prohibited":
        selection_input &= target_observed
    selected = _select_features(
        cell,
        atlas,
        atlas_variance,
        target_estimate,
        true_se[k] ** 2,
        selection_input,
        float(manifest["selection_fraction"]),
    )
    selected_evaluable = selected[target_observed[selected]]
    x, y = atlas[selected_evaluable], target_estimate[selected_evaluable]
    eiv_intercept, eiv_slope = _eiv_calibration(
        x,
        y,
        atlas_variance[selected_evaluable],
        true_se[k, selected_evaluable] ** 2 + tau[selected_evaluable] ** 2,
    )
    discrepancies = y - x
    sign_error = np.sign(x) != np.sign(y)
    metrics: dict[str, float | int] = {
        "features_total": p,
        "features_selection_eligible": int(np.sum(selection_eligible)),
        "features_selected": len(selected),
        "features_selected_target_evaluable": len(selected_evaluable),
        "features_transport_evaluable": int(np.sum(transport_valid)),
        "features_target_missing": int(np.sum(~target_observed)),
        "features_training_k_failed": int(np.sum(~training_valid)),
        "shared_transport_pearson": _safe_correlation(x, y, "pearson"),
        "shared_transport_spearman": _safe_correlation(x, y, "spearman"),
        "shared_transport_bias": float(np.mean(discrepancies))
        if len(selected_evaluable)
        else float("nan"),
        "shared_transport_rmse": float(np.sqrt(np.mean(discrepancies**2)))
        if len(selected_evaluable)
        else float("nan"),
        "shared_sign_error_rate": float(np.mean(sign_error))
        if len(selected_evaluable)
        else float("nan"),
        "shared_eiv_intercept": eiv_intercept,
        "shared_eiv_slope": eiv_slope,
    }

    interval_rows: dict[str, list[dict[str, float]]] = {method: [] for method in METHODS}
    failures: dict[str, Counter[str]] = {method: Counter() for method in METHODS}
    evaluation = _pi_subset(
        selected_evaluable,
        n=pi_features,
        cell=cell.cell_index,
        replicate=replicate_index,
    )
    for feature in evaluation:
        keep = training_observed[:, feature]
        estimates = training_estimate[keep, feature]
        ses = true_se[:k, feature][keep]
        target_se = float(true_se[k, feature])
        for method in METHODS:
            try:
                if method == "nnf_cd":
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="NNF draws below 5,000",
                            category=RuntimeWarning,
                        )
                        result = nnf_prediction_interval(
                            estimates,
                            ses,
                            target_se=target_se,
                            alpha=float(manifest["alpha"]),
                            draws=nnf_draws,
                            seed=_seed(
                                master,
                                cell.cell_index,
                                replicate_index,
                                10_000 + int(feature),
                            ),
                        )
                    interval = IntervalResult(
                        result.latent.low,
                        result.latent.high,
                        result.observed.low,
                        result.observed.high,
                    )
                else:
                    interval = _reml_mkh_interval(
                        estimates,
                        ses,
                        target_se=target_se,
                        alpha=float(manifest["alpha"]),
                    )
                interval_rows[method].append(
                    {
                        "latent_covered": float(
                            interval.latent_low <= target_latent[feature] <= interval.latent_high
                        ),
                        "observed_covered": float(
                            interval.observed_low
                            <= target_estimate[feature]
                            <= interval.observed_high
                        ),
                        "latent_width": interval.latent_high - interval.latent_low,
                        "observed_width": interval.observed_high - interval.observed_low,
                    }
                )
            except (NNFComputationError, RuntimeError, ValueError, FloatingPointError) as error:
                failures[method][f"{type(error).__name__}: {error}"] += 1
    for method in METHODS:
        failures[method]["__attempted__"] += len(evaluation)
    return ReplicateSummary(metrics, interval_rows, failures)


def _nanmean(values: list[float | int]) -> float | None:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else None


def _control_type(cell: CellSpec) -> str:
    if cell.selection == "target_informed_prohibited":
        return "prohibited_target_informed_negative_control"
    if (
        cell.truth == "null"
        and cell.target_bias == 0
        and cell.target_scale == 1
        and cell.missingness == "mcar_0"
    ):
        return "positive_null_control"
    return "primary_training_only"


def _aggregate(
    cell: CellSpec,
    replicates: list[ReplicateSummary],
    *,
    manifest: dict[str, Any],
    nnf_draws: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_keys = tuple(replicates[0].metrics) if replicates else ()
    metric_means = {
        key: _nanmean([replicate.metrics[key] for replicate in replicates]) for key in metric_keys
    }
    for method in METHODS:
        intervals = [row for replicate in replicates for row in replicate.intervals[method]]
        failures: Counter[str] = Counter()
        attempted = 0
        for replicate in replicates:
            counter = replicate.failures[method].copy()
            attempted += counter.pop("__attempted__", 0)
            failures.update(counter)
        successful = len(intervals)
        row = {
            **asdict(cell),
            "method": method,
            "control_type": _control_type(cell),
            "replicates_requested": len(replicates),
            "replicates_successful": len(replicates),
            **{f"{key}_mean": value for key, value in metric_means.items()},
            "shared_point_method": "dl_selection_diagnostic",
            "method_point_metrics_status": "shared_not_an_interval_method_comparison",
            "pi_features_attempted": attempted,
            "pi_features_successful": successful,
            "pi_features_failed": attempted - successful,
            "pi_failure_rate": (attempted - successful) / attempted if attempted else None,
            "pi_failure_types": json.dumps(failures, sort_keys=True),
            "latent_coverage": _nanmean([item["latent_covered"] for item in intervals]),
            "observed_coverage": _nanmean([item["observed_covered"] for item in intervals]),
            "latent_mean_width": _nanmean([item["latent_width"] for item in intervals]),
            "observed_mean_width": _nanmean([item["observed_width"] for item in intervals]),
            "nnf_draws": nnf_draws if method == "nnf_cd" else 0,
            "pi_evaluation_rule": (
                "stable_hash_subset_after_training_only_selection_then_target_availability"
            ),
            "inferential_unit": "study",
            "feature_level_p_value": "not_computed_features_are_not_replicates",
            "master_seed": manifest["master_seed"],
        }
        rows.append({column: row[column] for column in OUTPUT_COLUMNS})
    return rows


def run_cell(
    cell: CellSpec,
    *,
    replicates: int,
    nnf_draws: int,
    pi_features: int,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run one design cell deterministically and aggregate by interval method."""

    if replicates < 1:
        raise ValueError("replicates must be at least one")
    if nnf_draws < 100:
        raise ValueError("nnf_draws must be at least 100")
    if pi_features < 1:
        raise ValueError("pi_features must be at least one")
    simulated = [
        _simulate_replicate(
            cell,
            replicate_index=index,
            manifest=manifest,
            nnf_draws=nnf_draws,
            pi_features=pi_features,
        )
        for index in range(replicates)
    ]
    return _aggregate(cell, simulated, manifest=manifest, nnf_draws=nnf_draws)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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


def write_outputs(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    cell: CellSpec,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically write one Slurm-safe TSV/JSON result pair."""

    stem = f"transport_stress_cell_{cell.cell_index:04d}"
    tsv_path, json_path = output_dir / f"{stem}.tsv", output_dir / f"{stem}.json"
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    _atomic_write(tsv_path, frame.to_csv(sep="\t", index=False, na_rep="NA"))
    payload = {
        "metadata": metadata,
        "results": [{key: _json_safe(value) for key, value in row.items()} for row in rows],
    }
    _atomic_write(json_path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return tsv_path, json_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-index", type=int, default=0)
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--nnf-draws", type=int)
    parser.add_argument("--pi-features", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("transport_stress_results"))
    parser.add_argument("--print-design-hash", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest, cells = load_design(verify_hash=not args.print_design_hash)
    if args.print_design_hash:
        print(design_sha256(cells))
        return 0
    if not 0 <= args.cell_index < len(cells):
        raise ValueError(f"cell-index must be between 0 and {len(cells) - 1}")
    cell = cells[args.cell_index]
    replicates = args.replicates or int(manifest["full_replicates_by_p"][str(cell.p)])
    nnf_draws = args.nnf_draws or int(manifest["default_nnf_draws"])
    pi_features = args.pi_features or int(manifest["default_pi_features"])
    rows = run_cell(
        cell,
        replicates=replicates,
        nnf_draws=nnf_draws,
        pi_features=pi_features,
        manifest=manifest,
    )
    metadata = {
        "schema_version": manifest["schema_version"],
        "cell_index": cell.cell_index,
        "design_sha256": design_sha256(cells),
        "replicates": replicates,
        "nnf_draws": nnf_draws,
        "pi_features": pi_features,
        "failure_policy": "record_failure_no_fallback",
        "selection_warning": (
            "training_only selection excludes target availability and outcomes; "
            "target_informed_prohibited is a deliberate leakage negative control"
        ),
        "point_metric_status": (
            "shared DL selection diagnostics are not attributed to interval methods"
        ),
        "paper_inference": "No feature-as-replicate p-value is computed; study is the unit",
        "simulation_code_sha256": _file_sha256(Path(__file__).resolve()),
    }
    tsv, structured = write_outputs(rows, output_dir=args.output_dir, cell=cell, metadata=metadata)
    print(f"Wrote {tsv}")
    print(f"Wrote {structured}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
