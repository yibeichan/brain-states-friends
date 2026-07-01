#!/bin/bash
#SBATCH --job-name=03a_pca4combined_hmm
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# =============================================================================
# PCA Preparation for Combined HMM - SLURM Submission Script
# =============================================================================
#
# Fits PCA on primary training data and each of 6 LOSO fold training sets,
# then projects all splits. Run once per subject before combined HMM fitting.
#
# Memory: 16G covers ~110 MB training data (primary) + 6 LOSO folds with
#         cached training arrays (~80 MB each, freed between folds).
# Time:   ~10-15 min/subject (I/O-bound; 7 PCA fits total per subject).
#
# Sub-04 is included in the default array (0-5): the Python script internally
# restricts sub-04 to seasons 1-4 and creates 4 LOSO folds instead of 6.
#
# USAGE:
# ------
# 1. Default (all 6 subjects, 156-parcel atlas):
#    sbatch script/03a_pca4combined_hmm.sh
#
# 2. Single subject (e.g., for testing):
#    sbatch --array=0 script/03a_pca4combined_hmm.sh
#
# 3. Different parcellation:
#    sbatch --export=PARCELLATION=atlas-4S456Parcels script/03a_pca4combined_hmm.sh
#
# 4. Specific subjects only (e.g., sub-01 and sub-03):
#    sbatch --array=0,2 script/03a_pca4combined_hmm.sh
#
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

# =============================================================================
# Configuration (override with --export on sbatch command line)
# =============================================================================

PARCELLATION=${PARCELLATION:-"atlas-4S156Parcels"}
SPLIT_MODE=${SPLIT_MODE:-""}

# Map array task ID to subject (all 6 subjects; sub-04 handled internally)
SUBJECTS=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")
SUBJECT_ID=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$SUBJECT_ID" ]; then
    echo "ERROR: Invalid array task ID $SLURM_ARRAY_TASK_ID (max: $((${#SUBJECTS[@]}-1)))"
    exit 1
fi

# =============================================================================
# Validate inputs
# =============================================================================

# Get SCRATCH_DIR from .env if not already set
SCRATCH_DIR=${SCRATCH_DIR:-$(grep SCRATCH_DIR "${PROJECT_DIR}/.env" | cut -d'=' -f2 | tr -d '"' | tr -d "'")}

# Check episode ID file
EPISODE_FILE="${PROJECT_DIR}/${SUBJECT_ID}_episode_ids.txt"
if [ ! -f "$EPISODE_FILE" ]; then
    echo ""
    echo "=========================================="
    echo "SKIPPED: Episode file not found"
    echo "=========================================="
    echo "Subject: $SUBJECT_ID"
    echo "Expected: $EPISODE_FILE"
    echo "Exiting gracefully (not an error)."
    echo "=========================================="
    exit 0
fi

# Check parcel time series data directory
DATA_DIR="${SCRATCH_DIR}/output/02_parcel_ts_avg/${PARCELLATION}/${SUBJECT_ID}"
if [ ! -d "$DATA_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "SKIPPED: Parcel time series directory not found"
    echo "=========================================="
    echo "Subject: $SUBJECT_ID"
    echo "Expected: $DATA_DIR"
    echo "Run script 02 (02_extract_parcel_ts.sh) first."
    echo "Exiting gracefully (not an error)."
    echo "=========================================="
    exit 0
fi

# =============================================================================
# Print configuration
# =============================================================================

OUTPUT_DIR="${SCRATCH_DIR}/output/03a_pca4combined_hmm/${PARCELLATION}/${SUBJECT_ID}"

echo "=========================================="
echo "PCA Preparation for Combined HMM"
echo "=========================================="
echo "SLURM Job ID:      $SLURM_JOB_ID"
echo "Array Task ID:     $SLURM_ARRAY_TASK_ID"
echo "Subject:           $SUBJECT_ID"
echo "Parcellation:      $PARCELLATION"
echo "Project dir:       $PROJECT_DIR"
echo "Data dir:          $DATA_DIR"
echo "Output dir:        $OUTPUT_DIR"
echo "Episode file:      $EPISODE_FILE"
echo "=========================================="

# =============================================================================
# Run PCA preparation
# =============================================================================

echo ""
echo "Starting PCA preparation for combined HMM..."
echo ""

SPLIT_MODE_ARG=""
if [ -n "${SPLIT_MODE}" ]; then
    SPLIT_MODE_ARG="--split_mode ${SPLIT_MODE}"
    echo "Split mode:        $SPLIT_MODE"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/03a_pca4combined_hmm.py" \
    --sub_id "$SUBJECT_ID" \
    --parcellation "$PARCELLATION" \
    $SPLIT_MODE_ARG

EXIT_CODE=$?

# =============================================================================
# Report results
# =============================================================================

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: PCA preparation failed"
    echo "=========================================="
    echo "Subject:      $SUBJECT_ID"
    echo "Parcellation: $PARCELLATION"
    echo "Exit code:    $EXIT_CODE"
    echo "Check logs for details."
    echo "=========================================="
    exit $EXIT_CODE
else
    echo ""
    echo "=========================================="
    echo "SUCCESS: PCA preparation complete"
    echo "=========================================="
    echo "Subject:    $SUBJECT_ID"
    echo "Output dir: $OUTPUT_DIR"
    echo "=========================================="
fi

echo ""
echo "========== SLURM Job Resource Usage =========="
seff $SLURM_JOB_ID
echo "==============================================="
