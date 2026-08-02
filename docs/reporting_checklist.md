# Reporting checklist

- Define target and reference populations separately.
- State effect direction and scale.
- Identify the independent biological and study units.
- Report donor, line, organoid, and cell counts.
- State how pseudobulks or mixed models were constructed.
- Declare the shared feature universe and missing-feature rule.
- Report developmental age, lineage, assay, and preparation matching.
- Confirm complete target-study exclusion from every training stage.
- Report latent and observed-target intervals separately.
- Report NNF and every named comparator separately; never silently substitute a
  comparator for a failed NNF fold.
- Report training-study count, zero-boundary mass, and numerical failures.
- Lead with the equal-study macro summary; label the gene-fold micro row descriptive.
- Report success-conditional coverage, both failure bounds, interval width, numerical
  failure patterns, discrepancy, and sign agreement together.
- State that observed-target coverage is conditional on the held-out target SE.
- Report the NNF low-dispersion width warning and a conventional interval sensitivity.
- Show raw reference sensitivity; do not infer pairwise anchor changes unless
  shared-target covariance or component effects are supplied.
- Avoid calling a cross-study contrast an intrinsic property of the target model.
