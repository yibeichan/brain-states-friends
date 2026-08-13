#!/bin/bash
#SBATCH --job-name=rest_04_decode
#SBATCH --partition=mit_normal
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# =============================================================================
# Resting-State Score & Decode - SLURM Submission Script
# =============================================================================
# Scores and decodes hcptrt resting-state runs using the Friends-trained
# weak-limit HMM. All six subjects (rest includes sub-04).
#
# Prerequisites:
#   - rest_03_project_rest_pca.py completed (projected rest data)
#   - 04_combined_hdphmm.py (mode: select) completed (Friends model)
#
# Usage:
#   sbatch script/rest_04_score_and_decode.sh
#   sbatch --array=0 script/rest_04_score_and_decode.sh              # sub-01 only
#   sbatch --export=PARCELLATION=atlas-4S456Parcels script/rest_04_score_and_decode.sh
#   sbatch --export=VT=0.95 script/rest_04_score_and_decode.sh       # vt-specific model
# =============================================================================

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"

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
echo "Resting-State Score & Decode"
echo "=========================================="
echo "Subject:      $SUBJECT_ID"
echo "Parcellation: $PARCELLATION"
echo "VT:           ${VT:-'(legacy)'}"
echo "=========================================="

VT_ARG=""
if [ -n "$VT" ]; then
    VT_ARG="--vt $VT"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/rest_04_score_and_decode.py" \
    --sub_id "$SUBJECT_ID" \
    --parcellation "$PARCELLATION" \
    $VT_ARG

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Score & decode failed for $SUBJECT_ID (exit code: $EXIT_CODE)"
else
    echo "SUCCESS: Score & decode complete for $SUBJECT_ID"
fi

exit $EXIT_CODE
