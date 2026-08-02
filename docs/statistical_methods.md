# Statistical methods

## Cross-study pooling

For feature *g* and held-out target study *s*, RefGauge pools effects from all
other studies under

`y_gj ~ Normal(mu_g, se_gj^2 + tau_g^2)`.

REML is the default point estimator and DerSimonian-Laird is a sensitivity. These
point estimates do not by themselves provide a calibrated new-study interval.

## Primary NNF prediction interval

The primary interval is the Nagashima-Noma-Furukawa confidence-distribution (NNF-CD)
procedure, implemented independently from the published equations. It samples the
confidence distribution of `tau^2`, then propagates both heterogeneity and Hartung
mean uncertainty. The implementation has no runtime dependency on `pimeta`; frozen
oracle cases generated with `pimeta` are retained only for validation.

Two intervals are returned:

- `pi_true_low/high`: the latent biological effect in a new study;
- `pi_low/high`: a future observed estimate after adding the held-out target's
  supplied sampling variance.

Only the second interval defines leave-one-study-out coverage. A noisy direct
estimate must not be used to claim coverage of a latent-effect interval.
Because the observed interval uses the held-out study's supplied SE, its coverage is
retrospective and conditional on that known target precision. Prospective use requires
precision fixed from design information or modelled separately before outcomes.

NNF uses 25,000 draws by default. Runs below 5,000 draws issue a strong warning.
Numerical inversion and weighted-chi-square truncation diagnostics are exported.
The command-line workflow records a numerical failure for the affected fold and
continues, while the Python API can instead raise immediately. Neither policy
triggers a hidden fallback.

NNF standardized discrepancy is the normal score of the held-out estimate's empirical
predictive CDF position. This probability-integral-transform score respects the NNF
mixture rather than inventing a scalar plug-in SE. The implementation also reports the
ratio of NNF latent width to a conventional fixed-effect mean CI and flags ratios below
0.5. This warning exposes the published NNF procedure's false-precision boundary when
Q is almost zero; it does not silently alter the interval.

## Approximate comparators

`reml_mkh_approx` uses a modified Knapp-Hartung mean variance, REML heterogeneity,
and a Riley-style `t_(k-2)` prediction critical value. Latent and observed-target
intervals are again reported separately. This plug-in interval is fast but does not
fully propagate uncertainty in `tau^2`; it cannot replace NNF in a calibration claim.
Two-training-study normal fallbacks are prohibited.

Four additional comparators expose which modelling choice drives a result:

- `fixed_normal`: inverse-variance fixed-effect centre and normal interval;
- `dl_normal`: DerSimonian-Laird heterogeneity with a normal plug-in interval;
- `reml_normal`: REML heterogeneity with a normal plug-in interval;
- `unweighted_t`: unweighted centre, moment heterogeneity, and `t_(k-1)` interval.

All five comparators use the held-out study's supplied sampling variance in the
observed-target interval. They are deliberately named in every output row and no
method serves as a fallback for another.

## Study-count policy

- fewer than 5 training studies: no primary NNF inferential prediction interval;
- 5-14: exploratory, with no nominal-coverage claim;
- 15-29: usable with a small-study warning;
- 30 or more: most defensible, especially under low heterogeneity.

This policy applies per feature after missingness and target-study exclusion.
Approximate comparators may be computed from three or four training studies only as
diagnostics; that does not relax the primary study-count gate.

## Calibration endpoints

RefGauge reports Pearson and Spearman agreement, sign agreement, latent and
observed interval widths, bias, MAE, RMSE, and held-out predictive position. NNF uses
the predictive-PIT normal score; approximate intervals use a normal predictive
residual.

Study-stratum rows are reported explicitly. Metrics are then calculated within each
target study, pooling that study's eligible gene-stratum rows. The headline
`overall_macro_study` row gives each target study equal weight, irrespective of its
number of genes or strata. The separate `overall_micro_gene_fold` row is gene-fold
weighted and descriptive only. Coverage is reported among numerically successful
folds together with two unconditional bounds:
all failed folds counted as misses and all failed folds counted as hits. A method
cannot pass by selectively failing difficult folds.

Calibration intercept and slope use the iterative York heteroscedastic
errors-in-variables equations with predictor uncertainty and target variance equal to
the supplied target sampling variance plus fitted heterogeneity. The older weighted
least-squares slope is kept as a labelled sensitivity. Both are descriptive: no
feature-level p-value or confidence interval is emitted because genes and LOSO folds
are dependent.

## Reference-anchor audit

For every fixed target-feature pair, the audit reports raw range, standard deviation,
and point-sign changes across references. These diagnostics are noise-unadjusted.
Contrasts sharing a target are correlated, so contrast-level SEs alone do not identify
the variance of their difference. RefGauge therefore emits no pairwise p-values,
FDR values, or inferential sign-change labels without covariance. Rank correlation and
top-set overlap are withheld when fewer than 10 features are shared.

## Current limits

- NNF can be computationally expensive for very large feature-by-study matrices.
- Standard errors are assumed valid and effects must be on a common scale.
- Shared-reference covariance is not identifiable from contrast-level standard
  errors alone.
- Exchangeable-study prediction does not repair systematic target bias, scale shifts,
  reference mismatch, or effect modification.
- Feature dependence, missing-not-at-random effects, and target-informed selection
  require study-specific sensitivity analyses.
- Controlled reference-RNA benchmarks test technical transport, not transport across
  biological populations.
