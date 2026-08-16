#!/bin/bash
#SBATCH --job-name=hp_05_cross_valid
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
source "${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh" 2>/dev/null || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/script/utils/_env.sh" || exit 1

# =============================================================================
# Harry Potter Cross-Stimulus Validation - SLURM Submission Script
# =============================================================================
# Runs A1–A5 + B1–B2 analyses comparing HP reading state usage against
# Friends recurrence scores, with Movie10 bootstrap baseline.
# 5 subjects (no sub-04 in Harry Potter).
#
# Prerequisites:
#   - hp_04_score_and_decode.py completed (decoded states, FO, LL)
#   - 05a_recurrence_analysis.py completed (Friends state categories)
#   - hp_03_project_hp_pca.py completed (PCA diagnostic)
#   - m10_04_score_and_decode.py completed (for B1/B2 comparison)
#
# Usage:
#   sbatch script/hp_05_cross_stimulus_validation.sh
#   sbatch --array=0 script/hp_05_cross_stimulus_validation.sh
#   sbatch --export=PARCELLATION=atlas-4S456Parcels script/hp_05_cross_stimulus_validation.sh
# =============================================================================


# Configuration (5 subjects - no sub-04 in Harry Potter)
PARCELLATION=${PARCELLATION:-"atlas-4S156Parcels"}
VT=${VT:-"0.95"}
SUBJECTS=("sub-01" "sub-02" "sub-03" "sub-05" "sub-06")
SUBJECT_ID=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$SUBJECT_ID" ]; then
    echo "ERROR: Invalid array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=========================================="
echo "Harry Potter Cross-Stimulus Validation"
echo "=========================================="
echo "Subject:      $SUBJECT_ID"
echo "Parcellation: $PARCELLATION"
echo "VT:           ${VT:-'(legacy)'}"
echo "=========================================="

VT_ARG=""
if [ -n "$VT" ]; then
    VT_ARG="--vt $VT"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/hp_05_cross_stimulus_validation.py" \
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
