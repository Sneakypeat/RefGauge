# Cortical-organoid motivating benchmark

This directory records compact, non-identifying benchmark summaries. Raw public
expression and accessibility data are not redistributed.

## Reference-anchor result

A deposited Trevino comparison was reproduced, then re-estimated while holding the
organoid data and analysis fixed and changing the primary anchor. In the in-house
thresholded test (`q < 0.01`), the day-200 glial differential-feature rate ranged
from 0.93% to 26.37%. The resulting 28.3-fold value is a ratio of thresholded
discovery rates, not an effect-magnitude ratio. The day-200 neuronal rate ranged
from 9.92% with a whole-fetal-tissue anchor to 29.01% with a deliberately
identity-mismatched fetal-astrocyte anchor; the upper value is a stress test and not
a plausible matched-reference estimate. Primary-only comparisons reached 20.48%,
showing that reference composition contributes materially to the contrast.

## Transport feasibility result

A biological-preparation-level Camp organoid-primary neuronal contrast was compared
with HNOCA protocol effects across six pre-specified cell-selection sensitivities.
Pearson correlation ranged from 0.150 to 0.212, strong-effect sign agreement from
55.8% to 58.0%, and nominal 95% interval coverage from 62.4% to 69.6%.

This is superseded feasibility evidence, not final validation: Camp contributes to
HNOCA's published primary reference, ages are not matched, and sample counts are
small.

## Four-study descriptive transport result

Direct organoid-minus-primary effects from Camp, Cheroni, Herring and Kang shared
12,538 genes. Mean pairwise Pearson agreement was 0.388 and mean pairwise sign
agreement was 0.633. Exploratory REML/mKH leave-one-study-out point predictions gave
Pearson 0.504, Spearman 0.448, sign agreement 0.667 and strong-effect sign agreement
0.767. These point metrics were stable to the declared Cheroni mapping-QC, Camp
preparation-definition and Kang standard-error sensitivities.

Calibration is withheld. Four total studies leave only `k = 3` training studies in
each held-out fold, below RefGauge's five-training-study minimum for primary NNF
inference. The approximate intervals had median latent and observed widths of 14.2
and 18.2 log2 units; their apparent 0.9998 coverage is therefore uninformative. This
application supports bounded descriptive transport only, not calibrated prediction
or a field-wide organoid-fidelity estimate.

See `benchmark_status.tsv` for the machine-readable gate.
