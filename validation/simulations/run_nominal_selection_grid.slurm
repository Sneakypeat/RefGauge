#!/bin/bash
#SBATCH --job-name=ccal_nomsel
#SBATCH --partition=longq
#SBATCH --array=0-80%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=refgauge_nomsel_%A_%a.out
#SBATCH --error=refgauge_nomsel_%A_%a.err

set -Eeuo pipefail

ROOT="${REFGAUGE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${REFGAUGE_PYTHON:-python3}"
SIMDIR="${ROOT}/validation/simulations"
MANIFEST="${SIMDIR}/nominal_selection_manifest.json"
OUTDIR="${SIMDIR}/results/nominal_selection_alias/raw"
[[ -f "${ROOT}/pyproject.toml" ]] || {
    echo "ERROR: set REFGAUGE_ROOT to the RefGauge repository root" >&2
    exit 1
}
mkdir -p "${OUTDIR}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${SIMDIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

read -r P REPLICATES DESIGN_SHA < <(
    "${PYTHON}" -c 'from pathlib import Path; from transport_stress_simulation import load_design; import sys; m,c=load_design(manifest_path=Path(sys.argv[2])); x=c[int(sys.argv[1])]; print(x.p, m["full_replicates_by_p"][str(x.p)], m["design_sha256"])' \
        "${SLURM_ARRAY_TASK_ID}" "${MANIFEST}"
)
NNF_DRAWS=25000
PI_FEATURES=12
printf -v CELL "%04d" "${SLURM_ARRAY_TASK_ID}"
STEM="transport_stress_cell_${CELL}"
PROVENANCE="${OUTDIR}/${STEM}.provenance.tsv"
PROVENANCE_TMP="${PROVENANCE}.tmp.${SLURM_JOB_ID}"
STARTED_AT="$(date -Is)"

SIM_SHA="$(sha256sum "${SIMDIR}/transport_stress_simulation.py" | awk '{print $1}')"
NNF_SHA="$(sha256sum "${ROOT}/src/refgauge/prediction.py" | awk '{print $1}')"
MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
RUNNER_SHA="$(sha256sum "${SIMDIR}/run_nominal_selection_grid.slurm" | awk '{print $1}')"

"${PYTHON}" "${SIMDIR}/transport_stress_simulation.py" \
    --manifest "${MANIFEST}" \
    --cell-index "${SLURM_ARRAY_TASK_ID}" \
    --replicates "${REPLICATES}" \
    --nnf-draws "${NNF_DRAWS}" \
    --pi-features "${PI_FEATURES}" \
    --output-dir "${OUTDIR}"

{
    printf 'key\tvalue\n'
    printf 'schema_version\t1\n'
    printf 'status\tcomplete\n'
    printf 'cell_index\t%s\n' "${SLURM_ARRAY_TASK_ID}"
    printf 'p\t%s\n' "${P}"
    printf 'replicates\t%s\n' "${REPLICATES}"
    printf 'nnf_draws\t%s\n' "${NNF_DRAWS}"
    printf 'pi_features\t%s\n' "${PI_FEATURES}"
    printf 'design_sha256\t%s\n' "${DESIGN_SHA}"
    printf 'simulation_code_sha256\t%s\n' "${SIM_SHA}"
    printf 'nnf_code_sha256\t%s\n' "${NNF_SHA}"
    printf 'manifest_sha256\t%s\n' "${MANIFEST_SHA}"
    printf 'runner_code_sha256\t%s\n' "${RUNNER_SHA}"
    printf 'selection_warning\ttraining_only excludes target availability and outcomes; no target-informed selection is included\n'
    printf 'point_metric_status\tshared DL selection diagnostics are not attributed to interval methods\n'
    printf 'failure_policy\trecord_failure_no_fallback\n'
    printf 'inferential_unit\tstudy\n'
    printf 'slurm_array_job_id\t%s\n' "${SLURM_ARRAY_JOB_ID}"
    printf 'slurm_array_task_id\t%s\n' "${SLURM_ARRAY_TASK_ID}"
    printf 'slurm_job_id\t%s\n' "${SLURM_JOB_ID}"
    printf 'host\t%s\n' "$(hostname)"
    printf 'started_at\t%s\n' "${STARTED_AT}"
    printf 'completed_at\t%s\n' "$(date -Is)"
} > "${PROVENANCE_TMP}"
mv "${PROVENANCE_TMP}" "${PROVENANCE}"

echo "Done: $(date -Is) | cell=${CELL}"
