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

## Reference-family standardisation tables

The `reference-standardize` command is a separate inferential path. It does not turn
the descriptive reference table above into a corrected score. It requires direct
target-primary (`OP`) and primary-primary (`PP`) discrepancies constructed with one
frozen metric and feature definition.

Inference first requires an expected-cell design table written independently of the
observed results. It has exactly these identifiers:

| column | meaning |
|---|---|
| `domain` | frozen biological and analytical domain identifier |
| `feature` | endpoint or programme standardised independently |
| `edge_type` | exactly `OP` or `PP` |
| `target` | target preparation for OP; primary comparison target for PP |
| `target_unit` | expected independent donor, line, organoid, or preparation unit |
| `target_source` | expected independent laboratory or source-study identifier |
| `reference_source` | expected independent source study supplying the anchor |

The seven-column tuple must be unique, every domain-feature group must contain OP and
PP cells, and same-source PP cells are forbidden. The table has no `estimate` and no
anchor identifier because it declares source-level cells, not observed values. The
primary-source, target-source, and biological-unit universes are taken from this
table. RefGauge expands those declared units over the declared primary sources to
test the complete direct OP and cross-source PP Cartesian crossings. Consequently, a
cell that is expected but wholly absent from the point table remains a counted
missing cell. The software cannot discover a source or unit omitted from the design;
the design itself is therefore part of the scientific declaration.

The point-edge table requires:

| column | meaning |
|---|---|
| `domain` | frozen biological and analytical domain identifier |
| `feature` | endpoint or programme standardised independently |
| `edge_type` | exactly `OP` or `PP` |
| `target` | target preparation for OP; primary comparison target for PP |
| `target_unit` | independent donor, line, organoid, or preparation unit |
| `target_source` | independent laboratory or source-study identifier |
| `reference` | directly evaluated primary anchor |
| `reference_source` | independent source study supplying that anchor |
| `estimate` | finite discrepancy, with one declared orientation for every edge |
| `admissible` | whether the edge belongs to the frozen reference family |
| `leakage` | whether target information entered reference or metric fitting |

The eight-column tuple `(domain, feature, edge_type, target, target_unit,
target_source, reference, reference_source)` must be unique. Every admitted observed
source-level cell must occur in the design. Inadmissible rows may remain as an audit
trail but cannot satisfy a declared cell. Every expected target must be crossed
directly with every declared reference source, and every expected cross-source PP
comparison must be observed. Graph-path imputation does not satisfy this contract.
At least six independent primary sources and three biological units per source are
required by the v0.3 inferential contract.

Supplying `--design` is optional only to support descriptive recovery of legacy
edge tables. Without it, RefGauge still reports point baselines and anchor excesses
but fails `G1_DECLARED_DESIGN` and returns
`ABSTAIN_UNDECLARED_REFERENCE_DESIGN`.

The optional joint-bootstrap table has the same columns plus `draw`. Each draw must
resample biological units at source level and refit marker selection, latent mapping,
feature filtering, and every other learned metric component. The same source draw
must be reused across all edges containing that source. Marginal edge standard errors
or independently bootstrapped edges do not recover the required covariance. Version
0.3 requires at least 5,000 draws and at least 95% complete draws.

`delta_ref` is the largest scientifically tolerable residual change caused only by
the admitted reference. It must be chosen before target outcomes are inspected. If it
is omitted, RefGauge preserves point diagnostics and returns
`ABSTAIN_UNFROZEN_TOLERANCE`. An optional positive `delta_bio` may classify the
excess discrepancy as within-envelope, divergent, or unresolved; it is never learned
from the observed target scores.
