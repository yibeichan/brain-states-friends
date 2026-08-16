#!/bin/bash
#SBATCH --job-name=rest_05_cross_valid
#SBATCH --partition=mit_normal
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# =============================================================================
# Resting-State Cross-Stimulus Validation - SLURM Submission Script
# =============================================================================
# Runs A1-A5 + C1 + B2 analyses comparing resting-state state usage against
# Friends recurrence scores, with Movie10 bootstrap baseline.
# All six subjects (rest includes sub-04).
#
# Prerequisites:
#   - rest_04_score_and_decode.py completed (decoded states, FO, LL)
#   - 05a_recurrence_analysis.py completed (Friends state categories)
#   - rest_03_project_rest_pca.py completed (PCA diagnostic)
#   - m10_04_score_and_decode.py completed (for B2 comparison)
#
# Usage:
#   sbatch script/rest_05_cross_stimulus_validation.sh
#   sbatch --array=0 script/rest_05_cross_stimulus_validation.sh
#   sbatch --export=PARCELLATION=atlas-4S456Parcels script/rest_05_cross_stimulus_validation.sh
# =============================================================================

# Configuration (all six subjects - rest includes sub-04, unlike HP/PP)
PARCELLATION=${PARCELLATION:-"atlas-4S156Parcels"}
VT=${VT:-"0.95"}
SUBJECTS=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")   # rest includes sub-04
SUBJECT_ID=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$SUBJECT_ID" ]; then
    echo "ERROR: Invalid array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=========================================="
echo "Resting-State Cross-Stimulus Validation"
echo "=========================================="
echo "Subject:      $SUBJECT_ID"
echo "Parcellation: $PARCELLATION"
echo "VT:           ${VT:-'(legacy)'}"
echo "=========================================="

VT_ARG=""
if [ -n "$VT" ]; then
    VT_ARG="--vt $VT"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/rest_05_cross_stimulus_validation.py" \
    --sub_id "$SUBJECT_ID" \
    --parcellation "$PARCELLATION" \
    $VT_ARG

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Cross-stimulus validation failed for $SUBJECT_ID (exit code: $EXIT_CODE)"
else
    echo "SUCCESS: Cross-stimulus validation complete for $SUBJECT_ID"
fi

exit $EXIT_CODE
