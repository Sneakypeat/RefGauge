#!/bin/bash
#SBATCH --job-name=ccal_nomsel_agg
#SBATCH --partition=longq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=refgauge_nomsel_agg_%j.out
#SBATCH --error=refgauge_nomsel_agg_%j.err

set -Eeuo pipefail

ROOT="${REFGAUGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${REFGAUGE_PYTHON:-python3}"
SIMDIR="${ROOT}/validation/simulations"
INPUT="${SIMDIR}/results/nominal_selection_alias/raw"
OUTPUT="${SIMDIR}/results/nominal_selection_alias/aggregate"
[[ -f "${ROOT}/pyproject.toml" ]] || {
    echo "ERROR: set REFGAUGE_ROOT to the RefGauge repository root" >&2
    exit 1
}
mkdir -p "${OUTPUT}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${SIMDIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" "${SIMDIR}/aggregate_transport_stress_results.py" \
    --input-dir "${INPUT}" \
    --output-dir "${OUTPUT}" \
    --manifest "${SIMDIR}/nominal_selection_manifest.json" \
    --runner "${SIMDIR}/run_nominal_selection_grid.slurm" \
    --output-prefix nominal_selection
