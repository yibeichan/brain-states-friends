#!/bin/bash
#SBATCH --job-name=pp_03_pca_project
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# Petit Prince PCA Projection - SLURM Submission Script
# =============================================================================
# Projects Petit Prince parcel time series through Friends-trained PCA.
# One array task per subject (5 subjects, no sub-04).
#
# Prerequisites:
#   - pp_00_postproc.sh + pp_02_extract_parcel_ts.sh completed
#   - 04_combined_hdphmm.py (mode: select) completed for this subject
#
# Usage:
#   sbatch script/pp_03_project_pp_pca.sh
#   sbatch --array=0 script/pp_03_project_pp_pca.sh          # sub-01 only
#   sbatch --export=PARCELLATION=atlas-4S456Parcels script/pp_03_project_pp_pca.sh
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

# Configuration (5 subjects - no sub-04 in Petit Prince)
PARCELLATION=${PARCELLATION:-"atlas-4S156Parcels"}
VT="${VT:-"0.95"}"
SUBJECTS=("sub-01" "sub-02" "sub-03" "sub-05" "sub-06")
SUBJECT_ID=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$SUBJECT_ID" ]; then
    echo "ERROR: Invalid array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=========================================="
echo "Petit Prince PCA Projection"
echo "=========================================="
echo "Subject:      $SUBJECT_ID"
echo "Parcellation: $PARCELLATION"
echo "=========================================="

VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/pp_03_project_pp_pca.py" \
    --sub_id "$SUBJECT_ID" \
    --parcellation "$PARCELLATION" \
    ${VT_ARG}

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: PCA projection failed for $SUBJECT_ID (exit code: $EXIT_CODE)"
else
    echo "SUCCESS: PCA projection complete for $SUBJECT_ID"
fi

exit $EXIT_CODE
