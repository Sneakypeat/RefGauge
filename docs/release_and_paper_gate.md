# Release and manuscript gate

## Software release gate

- Clean install from wheel in a new Python environment.
- Python 3.10-3.13 CI matrix.
- Unit tests for pooling, target exclusion, input failure, metrics, reference audit,
  and complete CLI output.
- Lint-clean source and at least 90% statement coverage.
- Deterministic demo with machine-readable and visual outputs.
- Versioned input/output contracts, license, citation, changelog, and checksums.
- No raw human data in the repository or release artifacts.

## Methods-paper gate

Software quality is necessary but not sufficient for a high-impact methods paper.
The empirical paper proceeds only when all of the following hold:

1. a controlled cross-laboratory benchmark and an independent naturalistic benchmark
   with at least 16 total studies are available, leaving 15 training studies per fold;
2. atlas predictions exclude the target from feature selection, reference learning,
   cell annotation, age estimation, and effect pooling;
3. results survive age, cell identity, assay, pseudobulk, effect-scale, and reference
   sensitivities;
4. simulation benchmarks cover realistic study counts, heterogeneity, missingness,
   selection, and standard-error misspecification;
5. RefGauge changes at least one substantive inference relative to conventional
   pooled analysis without overstating which reference is biologically correct;
6. the cortical-organoid application is kept separate from the general calibration
   claim and does not manufacture study replication from cells, regions, or batches.

Without items 1-5, the software remains a research release and the cortical analysis
remains bounded descriptive evidence rather than calibrated validation. A Nature
Methods submission additionally requires NNF coverage/sharpness simulations, an
established-method comparison, project-level technical transport, and a credible
biological use case.
