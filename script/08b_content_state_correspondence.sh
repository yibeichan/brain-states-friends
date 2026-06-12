#!/bin/bash
#SBATCH --job-name=08b_content_state
#SBATCH --output=logs/08b_content_state_%A_%a.out
#SBATCH --error=logs/08b_content_state_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 08b - Content-State Correspondence
# =============================================================================
# Tests whether brain states carry information about narrative content.
# Seven analyses: per-state signatures (A1), decoding (A2), per-state multi-lag
# signatures (A3), TTAs (A4), consistency (A5), selectivity (A6), sensory-confound (A7).
# A1/A3 refactored 2026-04-23 to per-(state, feature) AUC framework -
# see the design notes.
#
# Prerequisites:
#   - 08a_content_features.py completed
#   - 04 decoded_states.pkl
#   - 05a recurrence_summary.json
#
# SLURM array job: one task per subject (array index 0-5 -> sub-01 to sub-06).
#
# Usage:
#   sbatch script/08b_content_state_correspondence.sh
#   sbatch --array=0 script/08b_content_state_correspondence.sh
#   sbatch --export=SUB_ID=sub-01 script/08b_content_state_correspondence.sh
# =============================================================================

set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate friends-states

# Subject ID: from export or array index
if [ -z "$SUB_ID" ]; then
    SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
    SUB_ID="${SUBJECTS[${SLURM_ARRAY_TASK_ID:-0}]}"
fi

PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
VT="${VT:-0.95}"
# ANALYSES: space-separated subset e.g. "A1 A2 A3" (default: all)
ANALYSES="${ANALYSES:-}"
# CONTROL_MODES: space-separated subset of {raw, partial, run_onset_anchored, mask33a}
# (default: unset = all four). Outputs that already exist are checkpoint-skipped
# unless FORCE=1. See the design notes.
CONTROL_MODES="${CONTROL_MODES:-}"
# FORCE: set to "1" to re-run even if checkpoint outputs exist
FORCE="${FORCE:-}"
# N_PERM_PER_STATE: permutations per (state, feature) for A1/A3 per-state signatures
# (2026-04-23 redesign). Default 500 inside the Python script; raise to 1000 for
# borderline q-values. See the design notes.
N_PERM_PER_STATE="${N_PERM_PER_STATE:-}"

echo "=============================================="
echo "08b - Content-State Correspondence"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "Joblib workers: ${N_JOBS}"
echo "VT: ${VT}"
echo "Analyses: ${ANALYSES:-all}"
echo "Control modes: ${CONTROL_MODES:-all}"
echo "Force: ${FORCE:-no}"
echo "=============================================="

# Subjob splitting examples (SLURM dependency chains):
#   # Light analyses (~1-2h each)
#   JOB_LIGHT=$(sbatch --parsable --export=ANALYSES="A1 A3 A4 A5 A7" script/08b_content_state_correspondence.sh)
#   # Heavy: decoding (~4-8h)
#   JOB_A2=$(sbatch --parsable --export=ANALYSES="A2" script/08b_content_state_correspondence.sh)
#   # Heavy: selectivity (~1-2h)
#   JOB_A6=$(sbatch --parsable --export=ANALYSES="A6" script/08b_content_state_correspondence.sh)

CMD="uv run --project ${PROJECT_DIR} --no-sync python ${PROJECT_DIR}/script/08b_content_state_correspondence.py \
    --sub_id ${SUB_ID} \
    --parcellation ${PARCELLATION} \
    --n_jobs ${N_JOBS} \
    --vt ${VT}"

if [ -n "$ANALYSES" ]; then
    CMD="${CMD} --analyses ${ANALYSES}"
fi

if [ -n "$CONTROL_MODES" ]; then
    CMD="${CMD} --control_modes ${CONTROL_MODES}"
fi

if [ -n "$N_PERM_PER_STATE" ]; then
    CMD="${CMD} --n_permutations_per_state ${N_PERM_PER_STATE}"
fi

if [ "$FORCE" = "1" ]; then
    CMD="${CMD} --force"
fi

eval $CMD

echo "=============================================="
echo "Content-state correspondence complete!"
echo "=============================================="
