# Benchmark provenance

## Trevino reference audit

- Study: Trevino et al., *Science* (2020), DOI `10.1126/science.aay1645`
- Public accession: `GSE132403`
- Source universe: 244,044 deposited merged ATAC peaks
- Compact output: `trevino_reference_audit.tsv`
- SHA-256: `4546d27f609e0779e3ccffd0e746a0282add8c1884261ff3051618ece7222f0e`
- Deterministic analysis seed: `20260731`

The table contains deposited-result reproduction, in-house calibration, single-factor
reference/age/identity designs, primary-only contrasts, and sensitivity summaries.
Absolute percentages from the in-house moderated test are not substitutes for the
authors' DESeq2 percentages; only internally matched design comparisons are used.

## Camp-HNOCA feasibility calibration

- Camp study: `GSE75140`
- HNOCA collection: CELLxGENE collection `de379e5f-52d0-498c-9801-0f850823c847`
- HNOCA supplementary source: Table S4
- Compact output: `camp_hnoca_sensitivity.tsv`
- SHA-256: `ca5e68eeeab722cb5eb21958b8ab343ae11ee1ca5d1c5a6211f781bdf7ab4f65`

The Camp deposited matrix was verified empirically as `log2(FPKM + 1)`. The primary
specification back-transforms, averages FPKM within biological preparation, and then
logs the preparation mean. Mean deposited log expression is retained as a scale
sensitivity. This benchmark is marked feasibility-only because Camp contributes to
HNOCA's published primary reference and developmental ages are not matched.

No participant-level expression matrix, genotype, or controlled-access metadata is
distributed in this repository.

