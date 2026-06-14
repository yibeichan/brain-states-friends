#!/bin/bash
#SBATCH --job-name=03b_pca_loadings
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yibei@mit.edu

# =============================================================================
# PCA Loadings Analysis — SLURM Submission Script
# =============================================================================
#
# Generates diagnostic plots (A1-A5, A7) and CSVs from PCA models fitted in 03a.
# Includes motion artifact flagging and LOSO residual stability analysis.
#
# Memory: 8G covers loading up to 7 PCA models (primary + 6 LOSO) per subject.
# Time:   ~5 min/subject (plotting only, no heavy computation).
#
# USAGE:
# ------
# 1. Default (all 6 subjects, per-subject diagnostics A1-A5, A7):
#    sbatch script/03b_pca_loadings.sh
#
# 2. Single subject:
#    sbatch --array=0 script/03b_pca_loadings.sh
#
# 3. Cross-subject comparison (run once, not per-subject):
#    sbatch --array=0 --export=CROSS_SUBJECT=1 script/03b_pca_loadings.sh
#
# 4. Different parcellation:
#    sbatch --export=PARCELLATION=atlas-4S456Parcels script/03b_pca_loadings.sh
#
# =============================================================================

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

source ~/.bashrc
micromamba activate friends-states

# =============================================================================
# Configuration (override with --export on sbatch command line)
# =============================================================================

PARCELLATION=${PARCELLATION:-"atlas-4S156Parcels"}
CROSS_SUBJECT=${CROSS_SUBJECT:-"0"}

# Variance threshold for CSV outputs (pca_residual_variance.csv signal/residual
# split; A3/A5 plot k-cutoff). Both this wrapper and the Python script default
# to "0.95" (k~66-77 PCs), matching the production pipeline that the HMM
# downstream actually consumes. Pass a space-separated list (e.g. "0.95 0.90") to emit
# diagnostic plots at additional thresholds; the FIRST value is the one written
# to pca_residual_variance.csv.
VARIANCE_THRESHOLD=${VARIANCE_THRESHOLD:-"0.95"}

# Map array task ID to subject
SUBJECTS=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")
SUBJECT_ID=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$SUBJECT_ID" ]; then
    echo "ERROR: Invalid array task ID $SLURM_ARRAY_TASK_ID (max: $((${#SUBJECTS[@]}-1)))"
    exit 1
fi

# =============================================================================
# Validate inputs
# =============================================================================

SCRATCH_DIR=${SCRATCH_DIR:-$(grep SCRATCH_DIR "${PROJECT_DIR}/.env" | cut -d'=' -f2 | tr -d '"' | tr -d "'")}

PCA_DIR="${SCRATCH_DIR}/output/03a_pca4combined_hmm/${PARCELLATION}/${SUBJECT_ID}"
if [ ! -d "$PCA_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "SKIPPED: PCA output directory not found"
    echo "=========================================="
    echo "Subject: $SUBJECT_ID"
    echo "Expected: $PCA_DIR"
    echo "Run script 03a (03a_pca4combined_hmm.sh) first."
    echo "Exiting gracefully (not an error)."
    echo "=========================================="
    exit 0
fi

# =============================================================================
# Print configuration
# =============================================================================

OUTPUT_DIR="${SCRATCH_DIR}/output/03b_pca_loadings/${PARCELLATION}/${SUBJECT_ID}"

echo "=========================================="
echo "PCA Loadings Analysis"
echo "=========================================="
echo "SLURM Job ID:        $SLURM_JOB_ID"
echo "Array Task ID:       $SLURM_ARRAY_TASK_ID"
echo "Subject:             $SUBJECT_ID"
echo "Parcellation:        $PARCELLATION"
echo "Variance threshold:  $VARIANCE_THRESHOLD"
echo "Cross-subject:       $CROSS_SUBJECT"
echo "Project dir:         $PROJECT_DIR"
echo "PCA dir:             $PCA_DIR"
echo "Output dir:          $OUTPUT_DIR"
echo "=========================================="

# =============================================================================
# Run PCA loadings analysis
# =============================================================================

echo ""
echo "Starting PCA loadings analysis..."
echo ""

EXTRA_ARGS=""
if [ "$CROSS_SUBJECT" = "1" ]; then
    EXTRA_ARGS="--cross_subject"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/03b_pca_loadings.py" \
    --sub_id "$SUBJECT_ID" \
    --parcellation "$PARCELLATION" \
    --variance_threshold $VARIANCE_THRESHOLD \
    $EXTRA_ARGS

EXIT_CODE=$?

# =============================================================================
# Report results
# =============================================================================

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: PCA loadings analysis failed"
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
    echo "SUCCESS: PCA loadings analysis complete"
    echo "=========================================="
    echo "Subject:    $SUBJECT_ID"
    echo "Output dir: $OUTPUT_DIR"
    echo "=========================================="
fi

echo ""
echo "========== SLURM Job Resource Usage =========="
seff $SLURM_JOB_ID
echo "==============================================="
