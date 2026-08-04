# RefGauge

**Audit reference dependence and held-out study transport in cross-study omics.**

[![Tests](https://github.com/Sneakypeat/RefGauge/actions/workflows/tests.yml/badge.svg)](https://github.com/Sneakypeat/RefGauge/actions/workflows/tests.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21794555.svg)](https://doi.org/10.5281/zenodo.21794555)

Cross-study comparisons are often interpreted as properties of a model or condition,
even though their magnitude can depend on the study, reference population, age,
assay, and preprocessing pipeline. RefGauge turns that concern into two explicit,
auditable analyses:

1. **Transport audit:** can an estimate pooled from other studies predict a
   direct effect in a held-out study?
2. **Reference audit:** how much do effect magnitude, sign, and feature ranking change
   when the reference anchor changes?

The package was motivated by human cortical-organoid fidelity studies, but its input
contract is generic: one estimate and standard error per feature, study, and optional
biological stratum.

## What it reports

- target-study-excluded random-effects predictions;
- independently implemented Nagashima-Noma-Furukawa confidence-distribution
  prediction intervals that propagate heterogeneity uncertainty;
- separate intervals for a latent new-study effect and an observed target estimate;
- five named approximate comparators: fixed-effect normal, DerSimonian-Laird
  normal, REML normal, REML with modified Knapp-Hartung/Riley, and an
  unweighted moment/t interval;
- between-study heterogeneity (`tau2`, Cochran's Q, I-squared);
- prediction intervals that include held-out sampling error;
- equal-study macro summaries plus explicitly labelled gene-fold micro summaries;
- success-conditional interval coverage with failure-as-miss and failure-as-hit bounds;
- bias, MAE, RMSE, and a predictive-PIT normal-score discrepancy for NNF;
- Pearson and Spearman transport correlations;
- sign agreement and strong-effect sign agreement;
- descriptive errors-in-variables calibration intercept and slope, including target
  heterogeneity, with naive WLS labelled as a sensitivity;
- reference-dependent raw ranges, point-sign changes, rank correlations, and top-set
  overlap; no pairwise p-values are emitted without shared-target covariance;
- machine-readable TSV/JSON outputs, diagnostic PNGs, and a self-contained HTML report.

## Installation

From a tagged GitHub release:

```bash
pip install "refgauge @ git+https://github.com/Sneakypeat/RefGauge.git@v0.2.0"
```

For development:

```bash
git clone https://github.com/Sneakypeat/RefGauge.git
cd RefGauge
python -m pip install -e ".[dev]"
pytest
```

The command is also available as `python -m refgauge`. To exercise the bundled
input templates after installation:

```bash
refgauge calibrate \
  --effects examples/effects_template.tsv \
  --output example_calibration
refgauge reference-audit \
  --contrasts examples/references_template.tsv \
  --output example_reference_audit
```

## Five-minute demo

```bash
refgauge demo --output demo_results
open demo_results/calibration/report.html  # macOS
```

The demo creates deterministic synthetic inputs, runs both analyses, and writes a
complete report tree.

## Your data

Transport-audit input (`effects.tsv`):

| column | meaning |
|---|---|
| `feature` | gene, peak, pathway, motif, or other shared feature |
| `study` | independent study identifier |
| `stratum` | optional matched lineage/age/assay stratum; defaults to `all` |
| `estimate` | signed effect on a common scale |
| `se` | standard error of that effect |

```bash
refgauge calibrate \
  --effects effects.tsv \
  --output calibration_results \
  --interval-method nnf_cd \
  --nnf-draws 25000 \
  --nnf-failure-policy record
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
refgauge reference-audit \
  --contrasts references.tsv \
  --output reference_results
```

See [Input contracts](docs/input_contracts.md) and
[Statistical methods](docs/statistical_methods.md) before analysing real data.

## Interpretation boundary

RefGauge does not produce a universal fidelity score, repair confounded study
designs, or make heterogeneous cell identities comparable. A high cross-study
correlation can coexist with poor interval coverage; a low correlation can reflect
restricted effect ranges. The primary objects are retrospective compatibility of an
excluded target with an exchangeable-study predictive distribution, equal-study
coverage summaries, interval width, predictive-PIT discrepancy, and declared
reference sensitivity. Observed-target coverage is conditional on the held-out
target's supplied SE; it is not a target-free prospective forecast. The latent-effect
interval is reported separately and is not scored against a noisy direct estimate.
NNF also exports a low-dispersion width warning when its unmodified Hartung component
becomes much narrower than a conventional fixed-effect mean interval.
Numerical NNF failures are retained as failed folds and are never replaced with an
approximate interval; use `--nnf-failure-policy raise` for fail-fast execution.
Select one comparator with `--interval-method`; methods are never substituted within
a failed fold. Run methods separately so every output table has one unambiguous
interval definition.

## Cortical-organoid benchmark

The motivating benchmark and its current evidential status are summarized in
[`benchmarks/cortical_organoid/README.md`](benchmarks/cortical_organoid/README.md).
Raw public datasets are not redistributed. Four direct studies support bounded
descriptive point transport, while calibration remains withheld because each fold
has only three training studies. The benchmark is not a field-wide biological claim.

## Citation

Until a methods article is published, cite the software using [`CITATION.cff`](CITATION.cff).
