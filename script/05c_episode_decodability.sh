#!/bin/bash
#SBATCH --job-name=05c_decodability
#SBATCH --output=logs/05c_decodability_%A_%a.out
#SBATCH --error=logs/05c_decodability_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 05c - Episode Decodability Analysis
# =============================================================================
# Tests whether season identity can be decoded from brain-state fractional
# occupancy vectors using L2-regularized logistic regression (CLR-transformed
# FO features) with LOO-CV.
#
# Analyses:
#   - Season decoding accuracy + permutation p-value
#   - Nuisance control: session-order decoding
#   - Per-state Kruskal-Wallis test (which states carry season info?)
#
# Prerequisites:
#   - Step 05a completed (fractional_occupancy.pkl + recurrence_summary.json)
#
# Usage:
#   # All subjects (array job):
#   sbatch script/05c_episode_decodability.sh
#
#   # Single subject (override via --export):
#   sbatch --array=0 --export=SUB_ID=sub-01 script/05c_episode_decodability.sh
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

# Map array index to subject (ignored if SUB_ID is set explicitly)
SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
if [ -z "${SUB_ID}" ]; then
    SUB_ID="${SUBJECTS[${SLURM_ARRAY_TASK_ID:-0}]}"
fi

PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"

# Variance threshold (default: 0.95)
VT="${VT:-0.95}"
VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

# Print header
echo "=============================================="
echo "05c - Episode Decodability Analysis"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "VT: ${VT:-not set (legacy path)}"
echo "=============================================="

# Run decodability analysis
N_JOBS="${N_JOBS:-4}"

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05c_episode_decodability.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    --n_jobs "${N_JOBS}" \
    ${VT_ARG}

echo "=============================================="
echo "Episode decodability analysis complete!"
echo "=============================================="
