# RefGauge coverage simulations

This directory contains the frozen, chunkable validation harness for NNF-CD and
five independently implemented comparators: fixed-effect normal, DL-normal,
REML-normal, REML/mKH approximate, and an unweighted t predictor.

## Frozen design

`full_grid_manifest.json` defines 750 cells spanning:

- `k`: 3, 4, 6, 10, 20 training studies;
- heterogeneity `tau / base SE`: 0, 0.5, 1.25, 2.5, 5;
- balanced precision or a four-fold range of true study SEs;
- normal, variance-standardized t(3), or centered exponential random effects;
- reported/true SE multipliers: 0.5, 0.75, 1, 1.25, 2.

The true target SE is the base SE. The reported-SE multiplier is applied to
both training and target SEs, allowing under- and over-reporting to affect the
method while data continue to arise from the true sampling variance.

Each replicate simulates a fresh latent target effect and a noisy observed
target estimate. Latent and observed coverage are therefore distinct. NNF-CD
failures are counted and retained; the REML/mKH result is never substituted.

## Local smoke run

From the repository root after `pip install -e .`:

```bash
python validation/simulations/coverage_simulation.py \
  --profile smoke \
  --jobs 2 \
  --output-dir validation/simulations/results/smoke
```

The smoke profile uses only four replicates and 250 NNF draws. It verifies the
pipeline but is not suitable for inference.

## HPC array

The full grid maps directly to array indices `0-749`. Each task writes a unique
pair of files atomically, so completed chunks can be resumed without collisions.
The bundled `.slurm` launchers should be submitted from the repository root. Set
`REFGAUGE_ROOT` when submitting elsewhere and `REFGAUGE_PYTHON` when
`python3` is not the intended interpreter. Their partition and resource directives
are examples and should be adapted to the local scheduler; log files are written to
the Slurm submission directory.

```bash
#SBATCH --array=0-749%40

python validation/simulations/coverage_simulation.py \
  --profile full \
  --cell-index "${SLURM_ARRAY_TASK_ID}" \
  --replicates 10000 \
  --nnf-draws 25000 \
  --jobs "${SLURM_CPUS_PER_TASK:-1}" \
  --output-dir validation/simulations/results/full
```

Outputs are `coverage_cell_NNNN.tsv` and `coverage_cell_NNNN.json`. The JSON
retains run metadata and the frozen grid hash. Deterministic simulation and NNF
seeds are derived from `(master seed, cell index, replicate index, stream)`, so
results do not depend on task order or worker count.

Process workers are used by default. If the operating system denies process-pool
semaphores, the harness emits a runtime warning and reruns the cell with thread
workers. This execution fallback does not alter seeds or substitute one
statistical method for another.

## Interpretation

Coverage, width, interval score, target-center bias/RMSE, tau-squared bias/RMSE,
failure rate, and heterogeneity-boundary behavior are reported separately by
method and cell. The REML/mKH comparator uses a `t(k-2)` predictive critical
value and modified Knapp-Hartung mean variance. The unweighted comparator uses
a moment estimate of heterogeneity and a `t(k-1)` critical value. No comparator
is substituted for a failed NNF calculation.

The full profile is computationally substantial. Benchmark a representative
cell on the intended cluster before launching all 750 cells.
