#!/bin/bash
#SBATCH --job-name=ccal_pi_sim
#SBATCH --partition=mediumq
#SBATCH --array=0-749%40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=refgauge_pi_%A_%a.out
#SBATCH --error=refgauge_pi_%A_%a.err

set -Eeuo pipefail

ROOT="${REFGAUGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${REFGAUGE_PYTHON:-python3}"
OUTDIR="${ROOT}/validation/simulations/results/full"
[[ -f "${ROOT}/pyproject.toml" ]] || {
    echo "ERROR: set REFGAUGE_ROOT to the RefGauge repository root" >&2
    exit 1
}
mkdir -p "${OUTDIR}"

printf -v CELL "%04d" "${SLURM_ARRAY_TASK_ID}"
if [[ -s "${OUTDIR}/coverage_cell_${CELL}.tsv" && \
      -s "${OUTDIR}/coverage_cell_${CELL}.json" ]]; then
    echo "Cell ${CELL} already complete"
    exit 0
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date -Is) | host=$(hostname) | cell=${CELL}"
sha256sum src/refgauge/prediction.py validation/simulations/coverage_simulation.py
"${PYTHON}" validation/simulations/coverage_simulation.py \
    --profile full \
    --cell-index "${SLURM_ARRAY_TASK_ID}" \
    --replicates 10000 \
    --nnf-draws 25000 \
    --jobs "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${OUTDIR}"
echo "Done:  $(date -Is) | cell=${CELL}"
