# Input contracts

## Effects table

Each row is one independently estimated, signed effect for one feature in one study.
The effect scale and direction must be identical across studies. `study` must identify
the unit that can be excluded without leakage; it is not a donor, cell, or technical
replicate. `stratum` should encode a biologically matched comparison such as lineage,
age window, and assay.

Required columns are `feature`, `study`, `estimate`, and `se`. `stratum` is optional.
The tuple `(feature, study, stratum)` must be unique. Standard errors must be finite
and strictly positive. The parser accepts at least three studies, but NNF inference
is withheld unless every held-out target has at least five training studies. Folds
with 5-14 training studies are exploratory; a strong calibration claim requires at
least 15. Lowering `min_training_studies` below five is available for method
validation and simulation only; it does not make those NNF folds inferential.

Cell-level analyses should first aggregate to independent biological preparations or
fit a model that respects donor, line, organoid, and study structure. RefGauge
does not correct pseudoreplication in supplied summary statistics.

## Reference table

Each row is a target-minus-reference effect. Keep the target data, feature universe,
effect definition, and preprocessing fixed while changing `reference`. Required
columns are `feature`, `target`, `reference`, `estimate`, and `se`.

Contrasts sharing the same target are correlated. Contrast-level standard errors do
not identify that covariance, so the default audit is descriptive and emits no
pairwise p-values or FDR sign-change labels. Raw ranges and point-sign changes are
always labelled noise-unadjusted. Covariance-aware inference must be performed from
component effects or a separately justified covariance model outside the default
contrast-only workflow.

Every target must have at least two distinct references. Targets with only one
anchor are rejected rather than silently omitted from the audit.

If changing a reference also changes feature eligibility, report the intersection
analysis and the reference-specific universe separately. Do not silently impute
missing effects as zero.

## Study exclusion

Leave-one-study-out calibration is only clean when the target study is absent from:

- the training effects;
- any reference atlas used to construct those effects;
- label-transfer, feature-selection, and hyperparameter-training steps;
- learned age or cell-identity references.

RefGauge can exclude rows by study identifier, but cannot discover upstream
leakage. Record upstream provenance in the manuscript and released data dictionary.
