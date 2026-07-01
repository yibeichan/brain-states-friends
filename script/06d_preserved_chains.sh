#!/bin/bash
#SBATCH --job-name=06d_preserved_chains
#SBATCH --output=logs/06d_preserved_chains_%A_%a.out
#SBATCH --error=logs/06d_preserved_chains_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 06d - Preserved Transition Chains
# =============================================================================
# Identifies state transition chains (bigrams, trigrams) preserved across
# episodes beyond what a first-order Markov process predicts.
#
# Prerequisites:
#   - Step 04 completed in 'select' mode
#   - Step 05a completed
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/06d_preserved_chains.sh
#   sbatch --export=SUB_ID=sub-01,VT=0.99 script/06d_preserved_chains.sh
# =============================================================================

set -e

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"

# Configuration (can be overridden via --export)
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

# Print header
echo "=============================================="
echo "06d - Preserved Transition Chains"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "VT: ${VT}"
echo "=============================================="

# Run analysis
VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/06d_preserved_chains.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG}

echo "=============================================="
echo "Preserved chains analysis complete!"
echo "=============================================="
