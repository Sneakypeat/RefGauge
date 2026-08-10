# Changelog

## 0.3.0 - 2026-08-10

- Added a finite-reference-family excess-discrepancy estimand calibrated against
  direct primary-primary disagreement on the same metric.
- Added hierarchical equal-source weighting, anchor-specific residuals, and
  leave-one-primary-source diagnostics.
- Added joint source-bootstrap confidence limits that retain shared target and
  reference dependence.
- Added strict direct-design, independence, covariance, reference-stability, and
  protocol-transport gates.
- Added an explicit expected source-level OP/PP design contract. Observed edges no
  longer define their own crossing universe; omission produces
  `ABSTAIN_UNDECLARED_REFERENCE_DESIGN` while retaining point diagnostics.
- Added machine-readable abstention states and retained all descriptive diagnostics
  when a standardised scalar is not identifiable.
- Added the `reference-standardize` CLI command, tabular/JSON outputs, and an HTML
  identification report.
- Preserved the v0.2 descriptive `reference-audit` and held-out transport behavior
  without changing their estimands.

## 0.2.0 - 2026-08-03

- Renamed the project, Python package, CLI and public repository from ContrastCal to
  RefGauge; no compatibility alias is installed.

- Added an independently implemented NNF confidence-distribution prediction interval
  with separate latent and observed-target outputs and frozen `pimeta` oracle cases.
- Added explicit genome-wide NNF failure recording with no method fallback; fail-fast
  behavior remains available through the Python API and CLI.
- Added fixed-effect normal, DerSimonian-Laird normal, REML normal, REML/mKH, and
  unweighted moment/t comparators using the same real-data and simulation labels.
- Removed the two-study normal prediction fallback.
- Added errors-in-variables calibration and labelled naive WLS as a sensitivity.
- Added reference-pair descriptive and overlap gates.
- Replaced gene-fold-weighted headline summaries with equal-study macro summaries;
  retained the former as an explicitly descriptive micro view.
- Made coverage success-conditional by name and added failure-as-miss and
  failure-as-hit bounds.
- Added NNF predictive-PIT residuals and a low-dispersion width warning.
- Labelled observed-target intervals as retrospective and conditional on the held-out
  target SE.
- Removed invalid reference-pair p-values and FDR sign-change labels when shared-target
  covariance is unavailable.
- Included fitted target heterogeneity in the descriptive York calibration slope.
- Added bounded dependency ranges, single-source package versioning, wheel/sdist
  validation, minimum-dependency CI, and a clean-install artifact smoke test.
- Made the bundled Slurm validation launchers portable through repository-root and
  Python-interpreter environment overrides.
- Added the completed four-study cortical descriptive benchmark and corrected the
  interpretation of Trevino thresholded discovery-rate sensitivities.

## 0.1.0 - 2026-08-01

- Initial research release.
- Added REML and DerSimonian-Laird random-effects pooling.
- Added leave-one-study-out transport calibration.
- Added reference-anchor sensitivity and ranking audits.
- Added CLI, synthetic demo, HTML reports, tests, and GitHub Actions CI.
