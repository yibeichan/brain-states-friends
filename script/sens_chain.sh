#!/bin/bash
# sens_chain.sh - run the downstream chain for one (side root, subject, vt).
#
# Runs the production scripts unchanged, with SCRATCH_DIR pointed at a side
# root laid out by sens_side_root.py. Arguments mirror the stage wrappers'
# defaults (05e_a2 --n_permutations 2000, 05e_a3 primary run --n_perm 5000,
# everything else the script default).
#
# Environment:
#   ROOT     side-root parent (one subdirectory per configuration)
#   NAME     side-root name (config name, or gate_*)
#   SUB_ID   subject; or SUBS (space-separated) with SLURM_ARRAY_TASK_ID
#   VT       variance threshold string ("0.95", "0.90", "0.99")
#   CONFIG   sweep config name for stage 04s (unset: skip 04s)
#   STAGES   space-separated subset/order (default: all of DEFAULT_STAGES)
#   PUBLISHED_SCRATCH  published SCRATCH root (for 04s inputs and --protect;
#            default: SCRATCH_DIR from .env)
#   PYTHON   interpreter of the project environment (default: python on PATH)
# A stage writes status/<sub>/<stage>.done on success and is skipped when the
# marker exists, so a rerun resumes after the last finished stage.
set -euo pipefail

CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
PARC=atlas-4S156Parcels

if [ -n "${SUBS:-}" ]; then
    read -r -a _subs <<< "${SUBS}"
    SUB_ID="${_subs[${SLURM_ARRAY_TASK_ID:-0}]}"
fi
: "${ROOT:?}" "${NAME:?}" "${SUB_ID:?}" "${VT:?}"

set -a
# shellcheck disable=SC1091
source "${CODE_DIR}/.env"
set +a
PUBLISHED_SCRATCH="${PUBLISHED_SCRATCH:-${SCRATCH_DIR:?SCRATCH_DIR unset in .env}}"
export SCRATCH_DIR="${ROOT}/${NAME}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
_nt="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${_nt}" OPENBLAS_NUM_THREADS="${_nt}" MKL_NUM_THREADS="${_nt}"
export MPLBACKEND=Agg
[ -n "${ATLAS_DIR:-}" ] && [ -d "${ATLAS_DIR}" ] || { echo "ATLAS_DIR unset or missing" >&2; exit 2; }
[ -n "${DATA_DIR:-}" ] && [ -d "${DATA_DIR}" ] || { echo "DATA_DIR unset or missing" >&2; exit 2; }

DEFAULT_STAGES="04s 05a a1 a2 a3 a4 05f 06a 06b m10_03 m10_04 m10_05"
STAGES="${STAGES:-$DEFAULT_STAGES}"
STATUS="${SCRATCH_DIR}/status/${SUB_ID}"
mkdir -p "${STATUS}"

echo "code   $(git -C "${CODE_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "root   ${SCRATCH_DIR}  sub ${SUB_ID}  vt ${VT}  config ${CONFIG:-<none>}"
echo "stages ${STAGES}"

# refuse to start unless every regenerated path is outside the dataset
"${PY}" "${CODE_DIR}/script/sens_side_root.py" check --root "${ROOT}" --name "${NAME}" \
    --vt "${VT}" --subs "${SUB_ID}" --published_out "${PUBLISHED_SCRATCH}/output" \
    $([ -L "${SCRATCH_DIR}/output/m10_03_projected" ] && echo --link_m10_03)

S="${CODE_DIR}/script"
run_stage() {
    local st="$1"
    if [ -e "${STATUS}/${st}.done" ]; then echo "--- ${st}: done, skipping"; return 0; fi
    echo "=== ${st} start $(date -Is)"
    local t0=$SECONDS
    case "${st}" in
        04s)
            : "${CONFIG:?CONFIG required for 04s}"
            "${PY}" "${S}/04s_sweep_config_final.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" \
                --config "${CONFIG}" --source_scratch "${PUBLISHED_SCRATCH}" \
                --protect "${PUBLISHED_SCRATCH}/output" --force ;;
        05a) "${PY}" "${S}/05a_recurrence_analysis.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" --vt "${VT}" ;;
        a1)  "${PY}" "${S}/05e_temporal_trend_a1.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" --vt "${VT}" ;;
        a2)  "${PY}" "${S}/05e_temporal_trend_a2.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" \
                 --n_permutations 2000 --vt "${VT}" ;;
        a3)  "${PY}" "${S}/05e_temporal_trend_a3.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" \
                 --n_perm 5000 --vt "${VT}" ;;
        a4)  "${PY}" "${S}/05e_temporal_trend_a4.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" --vt "${VT}" ;;
        05f) "${PY}" "${S}/05f_state_fc.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" --vt "${VT}" ;;
        06a) "${PY}" "${S}/06a_state_temp_dynamics.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" --vt "${VT}" ;;
        06b) "${PY}" "${S}/06b_transition_structure.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" --vt "${VT}" ;;
        m10_03)
            if [ -L "${SCRATCH_DIR}/output/m10_03_projected" ]; then
                echo "m10_03_projected is linked to the published projection; nothing to run"
            else
                "${PY}" "${S}/m10_03_project_movie_pca.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" --vt "${VT}"
            fi ;;
        m10_04) "${PY}" "${S}/m10_04_score_and_decode.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" --vt "${VT}" ;;
        m10_05) "${PY}" "${S}/m10_05_cross_stimulus_validation.py" --sub_id "${SUB_ID}" --parcellation "${PARC}" --vt "${VT}" ;;
        06e_N2) "${PY}" "${S}/sens_06e_q_excess.py" --sub_id "${SUB_ID}" --vt "${VT}" --mode run --variant N2_E ;;
        06e_N3) "${PY}" "${S}/sens_06e_q_excess.py" --sub_id "${SUB_ID}" --vt "${VT}" --mode run --variant N3_E ;;
        06e_sum) "${PY}" "${S}/sens_06e_q_excess.py" --sub_id "${SUB_ID}" --vt "${VT}" --mode summarise ;;
        *) echo "unknown stage ${st}" >&2; return 2 ;;
    esac
    touch "${STATUS}/${st}.done"
    echo "=== ${st} done in $((SECONDS - t0)) s"
}

for st in ${STAGES}; do
    run_stage "${st}"
done
echo "all stages done $(date -Is)"
