#!/bin/bash
#SBATCH --job-name=ccal_trstress_agg
#SBATCH --partition=mediumq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=refgauge_trstress_aggregate_%j.out
#SBATCH --error=refgauge_trstress_aggregate_%j.err

set -Eeuo pipefail

ROOT="${REFGAUGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${REFGAUGE_PYTHON:-python3}"
SIMDIR="${ROOT}/validation/simulations"
FROZEN_SIM="${SIMDIR}/transport_stress_simulation_frozen_20260801.py"
INPUT="${SIMDIR}/results/transport_stress/raw"
OUTPUT="${SIMDIR}/results/transport_stress/aggregate"
[[ -f "${ROOT}/pyproject.toml" ]] || {
    echo "ERROR: set REFGAUGE_ROOT to the RefGauge repository root" >&2
    exit 1
}
mkdir -p "${OUTPUT}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${SIMDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Start: $(date -Is) | host=$(hostname) | dependency=${SLURM_JOB_DEPENDENCY:-not_exported}"
sha256sum \
    "${SIMDIR}/aggregate_transport_stress_results.py" \
    "${FROZEN_SIM}" \
    "${SIMDIR}/transport_stress_manifest.json" \
    "${ROOT}/src/refgauge/prediction.py" \
    "${SIMDIR}/run_transport_stress_grid.slurm"
"${PYTHON}" "${SIMDIR}/aggregate_transport_stress_results.py" \
    --input-dir "${INPUT}" \
    --output-dir "${OUTPUT}" \
    --simulation-source "${FROZEN_SIM}"
sha256sum "${OUTPUT}"/*
echo "Done: $(date -Is)"
