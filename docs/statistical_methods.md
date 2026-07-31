# Statistical methods

## Cross-study pooling

For feature *g* and held-out target study *s*, ContrastCal pools effects from all
other studies using a random-effects model:

`y_gj ~ Normal(mu_g, se_gj^2 + tau_g^2)`.

The default between-study variance is restricted maximum likelihood (REML).
DerSimonian-Laird is available as a sensitivity. Uncertainty uses the modified
Knapp-Hartung standard error (never narrower than the conventional random-effects
standard error) and a Student-t critical value with `k - 1` degrees of freedom.

## Held-out prediction interval

The held-out prediction standard error is:

`sqrt(SE(mu_g)^2 + tau_g^2 + SE(y_gs)^2)`.

The final term matters: calibration asks whether the observed direct estimate is
compatible with the cross-study prediction, so uncertainty in the direct estimate
must be included. Prediction intervals use a Student-t critical value with `k - 2`
degrees of freedom for three or more training studies. With only two training
studies, where this interval is undefined, the output explicitly records a normal
fallback. Coverage is the fraction of direct effects inside these intervals.

## Calibration endpoints

ContrastCal reports transport correlations, sign agreement, prediction-interval
coverage, direct-minus-predicted bias, MAE, RMSE, and the standardized discrepancy.
A weighted regression of direct effects on cross-study predictions gives a
calibration intercept and slope. No one metric is sufficient:

- correlation ignores systematic bias and scale;
- sign agreement is unstable near zero;
- coverage depends on the quality of supplied standard errors and heterogeneity;
- pooled feature-level rows may be biologically dependent.

The package therefore reports all metrics and leaves multiplicity and hierarchical
feature dependence to the study-specific analysis plan.

## Reference-anchor audit

For every fixed target-feature pair, the audit computes the range and standard
deviation of estimates across references and whether signs change. For each pair of
references it calculates Spearman rank correlation and Jaccard overlap among the top
10% of absolute effects. These are sensitivity diagnostics, not tests that identify
the biologically correct reference.

## Current limits

- REML is fitted feature by feature and is unstable with very few studies.
- Two-training-study prediction intervals require a recorded normal fallback and
  should be treated as exploratory.
- Feature dependence is not modelled.
- Standard errors are assumed valid and on a common scale.
- Missing-not-at-random features and selection on significance can bias calibration.
