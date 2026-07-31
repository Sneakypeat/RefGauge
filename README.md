# ContrastCal

**Calibrate cross-study biological contrasts against direct within-study evidence.**

[![Tests](https://github.com/Sneakypeat/ContrastCal/actions/workflows/tests.yml/badge.svg)](https://github.com/Sneakypeat/ContrastCal/actions/workflows/tests.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

Cross-study comparisons are often interpreted as properties of a model or condition,
even though their magnitude can depend on the study, reference population, age,
assay, and preprocessing pipeline. ContrastCal turns that concern into two explicit,
auditable analyses:

1. **Transport calibration:** can an estimate pooled from other studies predict a
   direct effect in a held-out study?
2. **Reference audit:** how much do effect magnitude, sign, and feature ranking change
   when the reference anchor changes?

The package was motivated by human cortical-organoid fidelity studies, but its input
contract is generic: one estimate and standard error per feature, study, and optional
biological stratum.

## What it reports

- target-study-excluded REML random-effects predictions with modified
  Knapp-Hartung uncertainty;
- between-study heterogeneity (`tau2`, Cochran's Q, I-squared);
- prediction intervals that include held-out sampling error;
- interval coverage, bias, MAE, RMSE, and standardized discrepancy;
- Pearson and Spearman transport correlations;
- sign agreement and strong-effect sign agreement;
- weighted calibration intercept and slope;
- reference-dependent magnitude ranges, sign flips, rank correlations, and top-set overlap;
- machine-readable TSV/JSON outputs, diagnostic PNGs, and a self-contained HTML report.

## Installation

From a tagged GitHub release:

```bash
pip install "contrastcal @ git+https://github.com/Sneakypeat/ContrastCal.git@v0.1.0"
```

For development:

```bash
git clone https://github.com/Sneakypeat/ContrastCal.git
cd ContrastCal
python -m pip install -e ".[dev]"
pytest
```

## Five-minute demo

```bash
contrastcal demo --output demo_results
open demo_results/calibration/report.html  # macOS
```

The demo creates deterministic synthetic inputs, runs both analyses, and writes a
complete report tree.

## Your data

Transport-calibration input (`effects.tsv`):

| column | meaning |
|---|---|
| `feature` | gene, peak, pathway, motif, or other shared feature |
| `study` | independent study identifier |
| `stratum` | optional matched lineage/age/assay stratum; defaults to `all` |
| `estimate` | signed effect on a common scale |
| `se` | standard error of that effect |

```bash
contrastcal calibrate \
  --effects effects.tsv \
  --output calibration_results
```

Reference-audit input (`references.tsv`):

| column | meaning |
|---|---|
| `feature` | shared feature identifier |
| `target` | fixed target condition or model |
| `reference` | reference population used for the contrast |
| `estimate` | target-minus-reference effect |
| `se` | standard error |

```bash
contrastcal reference-audit \
  --contrasts references.tsv \
  --output reference_results
```

See [Input contracts](docs/input_contracts.md) and
[Statistical methods](docs/statistical_methods.md) before analysing real data.

## Interpretation boundary

ContrastCal does not produce a universal fidelity score, repair confounded study
designs, or make heterogeneous cell identities comparable. A high cross-study
correlation can coexist with poor interval coverage; a low correlation can reflect
restricted effect ranges. The primary inferential objects are held-out prediction,
coverage, standardized discrepancy, and declared reference sensitivity.

## Cortical-organoid benchmark

The motivating benchmark and its current evidential status are summarized in
[`benchmarks/cortical_organoid/README.md`](benchmarks/cortical_organoid/README.md).
Raw public datasets are not redistributed. The benchmark currently demonstrates
feasibility and reference-anchor sensitivity; it is not yet sufficient for a broad
biological claim.

## Citation

Until a methods article is published, cite the software using [`CITATION.cff`](CITATION.cff).
