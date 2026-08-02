#!/usr/bin/env Rscript

# Optional GPL validation oracle. RefGauge does not import or translate pimeta;
# this script records published-implementation bounds for independent parity tests.

suppressPackageStartupMessages(library(pimeta))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: generate_pimeta_nnf_oracle.R <output.tsv>")
}

cases <- list(
  balanced_low_heterogeneity = list(
    y = c(0.10, 0.20, 0.30, 0.40, 0.50),
    se = rep(0.20, 5)
  ),
  balanced_high_heterogeneity = list(
    y = c(-1.10, -0.35, 0.20, 0.90, 1.55, 2.10, 2.75),
    se = rep(0.18, 7)
  ),
  unbalanced_precision = list(
    y = c(-0.40, 0.05, 0.22, 0.47, 0.80, 1.10),
    se = c(0.05, 0.08, 0.15, 0.30, 0.45, 0.60)
  ),
  ten_studies_boundary = list(
    y = c(0.06, 0.13, 0.20, 0.18, 0.24, 0.31, 0.27, 0.22, 0.16, 0.25),
    se = c(0.12, 0.10, 0.09, 0.14, 0.11, 0.13, 0.08, 0.15, 0.10, 0.12)
  )
)

B <- 250000L
seed <- 20260801L
rows <- lapply(seq_along(cases), function(i) {
  case <- cases[[i]]
  fit <- pima(
    case$y,
    case$se,
    method = "boot",
    B = B,
    seed = seed + i - 1L
  )
  data.frame(
    case = names(cases)[[i]],
    k = length(case$y),
    y = paste(format(case$y, trim = TRUE, scientific = FALSE), collapse = ";"),
    se = paste(format(case$se, trim = TRUE, scientific = FALSE), collapse = ";"),
    B = B,
    seed = seed + i - 1L,
    estimate = fit$muhat,
    tau2_dl = fit$tau2h,
    pi_true_low = fit$lpi,
    pi_true_high = fit$upi,
    ci_mean_low = fit$lci,
    ci_mean_high = fit$uci,
    tau2_zero_mass = mean(fit$rnd == 0),
    stringsAsFactors = FALSE
  )
})

out <- do.call(rbind, rows)
dir.create(dirname(args[[1]]), recursive = TRUE, showWarnings = FALSE)
write.table(out, args[[1]], sep = "\t", row.names = FALSE, quote = FALSE)
