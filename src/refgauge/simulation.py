"""Deterministic synthetic data for tutorials and smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def simulate_dataset(
    *,
    n_features: int = 400,
    n_studies: int = 8,
    seed: int = 20260801,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    features = np.array([f"feature_{i:04d}" for i in range(1, n_features + 1)])
    truth = rng.normal(0, 0.7, n_features)
    effects = []
    for study_index in range(n_studies):
        study_bias = rng.normal(0, 0.16)
        study_scale = rng.normal(1.0, 0.12)
        se = rng.uniform(0.08, 0.24, n_features)
        observed = study_bias + study_scale * truth + rng.normal(0, np.sqrt(se**2 + 0.18**2))
        for feature, estimate, stderr in zip(features, observed, se, strict=True):
            effects.append(
                {
                    "feature": feature,
                    "study": f"study_{study_index + 1}",
                    "stratum": "neuronal" if int(feature[-4:]) % 2 else "glial",
                    "estimate": estimate,
                    "se": stderr,
                }
            )

    references = []
    anchors = {"matched": (0.0, 1.0), "younger": (0.35, 0.82), "mixed": (-0.20, 1.25)}
    targets = {"early_organoid": truth, "late_organoid": 0.65 * truth + 0.1}
    for target, target_effect in targets.items():
        for reference, (offset, scale) in anchors.items():
            se = rng.uniform(0.07, 0.18, n_features)
            estimate = offset + scale * target_effect + rng.normal(0, se)
            for feature, value, stderr in zip(features, estimate, se, strict=True):
                references.append(
                    {
                        "feature": feature,
                        "target": target,
                        "reference": reference,
                        "estimate": value,
                        "se": stderr,
                    }
                )
    return pd.DataFrame(effects), pd.DataFrame(references)


def write_demo(outdir: Path, seed: int = 20260801) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    effects, references = simulate_dataset(seed=seed)
    effect_path = outdir / "synthetic_effects.tsv"
    reference_path = outdir / "synthetic_references.tsv"
    effects.to_csv(effect_path, sep="\t", index=False)
    references.to_csv(reference_path, sep="\t", index=False)
    return effect_path, reference_path
