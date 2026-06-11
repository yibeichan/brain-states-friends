#!/bin/bash
#SBATCH --job-name=05d_similarity
#SBATCH --output=logs/05d_similarity_%A_%a.out
#SBATCH --error=logs/05d_similarity_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5

# =============================================================================
# 05d - State Similarity Analysis
# =============================================================================
# Assesses whether discovered brain states are truly distinct or redundant.
#
# Computes:
#   - Activation similarity (Pearson correlation of state mean vectors)
#   - FC similarity (RV coefficient between state covariance matrices)
#   - Transition similarity (Pearson correlation of outgoing transition rows)
#   - Combined similarity (normalised average) + flagged high-similarity pairs
#   - Diagnosis of flagged pairs (split state vs genuinely distinct)
#
# Prerequisites:
#   - Step 04 completed in 'select' mode (state_means_parcel.npy,
#     state_covars_parcel.npy, best_model.pkl, decoded_states.pkl)
#   - Step 05a completed (recurrence_summary.json)
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/05d_state_similarity.sh
#   sbatch --export=SUB_ID=sub-01,PARCELLATION=atlas-4S456Parcels script/05d_state_similarity.sh
#   bash 05d_state_similarity.sh
# =============================================================================

set -e

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate friends-states

# Configuration (can be overridden via --export)
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

# Print header
echo "=============================================="
echo "05d - State Similarity Analysis"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "=============================================="

# Run state similarity analysis
VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

# Ensure JAX is available (needed for unpickling HMM model)
uv sync --project "${PROJECT_DIR}" --extra gpu --quiet

# Primary run: exclude sub-HRF states (default)
echo "--- Primary run (exclude sub-HRF) ---"
uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05d_state_similarity.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG}

# Sensitivity run: include all states
echo "--- Sensitivity run (all states) ---"
uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05d_state_similarity.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    --no-exclude_sub_hrf \
    ${VT_ARG}

echo "=============================================="
echo "State similarity analysis complete (both versions)!"
echo "=============================================="
