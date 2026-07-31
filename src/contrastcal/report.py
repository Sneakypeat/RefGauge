"""Compact, dependency-light analysis reports."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calibration_figure(predictions: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    ax.scatter(predictions["atlas_estimate"], predictions["direct_estimate"], s=7, alpha=0.25)
    limits = np.nanpercentile(
        np.r_[predictions["atlas_estimate"], predictions["direct_estimate"]], [1, 99]
    )
    ax.plot(limits, limits, color="#D1495B", linewidth=1.4, label="identity")
    ax.set(xlabel="LOSO cross-study estimate", ylabel="Held-out direct estimate", title="Transport")
    ax.legend(frameon=False)

    ax = axes[1]
    z = predictions["standardized_discrepancy"].dropna()
    ax.hist(z, bins=35, color="#276FBF", alpha=0.8)
    ax.axvline(-1.96, color="#D1495B", linestyle="--", linewidth=1)
    ax.axvline(1.96, color="#D1495B", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Standardized direct-minus-predicted discrepancy",
        ylabel="Count",
        title="Calibration",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def reference_figure(
    feature_sensitivity: pd.DataFrame, rank_stability: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].hist(feature_sensitivity["estimate_range"], bins=35, color="#2A9D8F", alpha=0.85)
    axes[0].set(
        xlabel="Estimate range across references", ylabel="Features", title="Magnitude sensitivity"
    )
    values = rank_stability["spearman_rank"].dropna()
    axes[1].hist(values, bins=np.linspace(-1, 1, 31), color="#E9C46A", alpha=0.9)
    axes[1].set(
        xlabel="Pairwise Spearman rank correlation",
        ylabel="Reference pairs",
        title="Ranking stability",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_html_report(title: str, summary: pd.DataFrame, figure_name: str, path: Path) -> None:
    table = summary.to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)
    style = """
body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto;
       padding: 0 1rem; color: #202124; }
h1 { font-size: 1.7rem; }
img { max-width: 100%; height: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: .45rem .55rem; border-bottom: 1px solid #ddd; text-align: right; }
th:first-child, td:first-child { text-align: left; }
.note { background: #f5f7f8; border-left: 4px solid #276FBF; padding: .8rem 1rem; }
"""
    note = (
        "Cross-study estimates are predictions. Coverage and discrepancy must be interpreted "
        "with the declared target-study exclusion and reference definition."
    )
    body = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width">'
        f"<title>{html.escape(title)}</title><style>{style}</style></head><body>"
        f'<h1>{html.escape(title)}</h1><p class="note">{note}</p>'
        f'<img src="{html.escape(figure_name)}" alt="Analysis diagnostic">'
        f"<h2>Summary</h2>{table}</body></html>"
    )
    path.write_text(body, encoding="utf-8")


def write_json(data: dict | list, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
