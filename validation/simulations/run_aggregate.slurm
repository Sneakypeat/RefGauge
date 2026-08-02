#!/bin/bash
#SBATCH --job-name=ccal_pi_agg
#SBATCH --partition=mediumq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=refgauge_pi_aggregate_%j.out
#SBATCH --error=refgauge_pi_aggregate_%j.err

set -Eeuo pipefail

ROOT="${REFGAUGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${REFGAUGE_PYTHON:-python3}"
INPUT="${ROOT}/validation/simulations/results/full"
OUTPUT="${ROOT}/validation/simulations/results/aggregate"
[[ -f "${ROOT}/pyproject.toml" ]] || {
    echo "ERROR: set REFGAUGE_ROOT to the RefGauge repository root" >&2
    exit 1
}
mkdir -p "${OUTPUT}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date -Is) | host=$(hostname)"
"${PYTHON}" validation/simulations/aggregate_coverage_results.py \
    --input-dir "${INPUT}" \
    --output-dir "${OUTPUT}"
echo "Done:  $(date -Is)"
