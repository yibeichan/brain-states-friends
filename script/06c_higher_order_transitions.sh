#!/bin/bash
#SBATCH --job-name=06c_higher_order
#SBATCH --output=logs/06c_higher_order_%A_%a.out
#SBATCH --error=logs/06c_higher_order_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
source "${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh" 2>/dev/null || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/script/utils/_env.sh" || exit 1

# =============================================================================
# 06c - Model Adequacy Diagnostic for Transition Structure
# =============================================================================
# Tests how much sequential structure in the HMM's decoded state sequences
# exceeds what the model's 1st-order transition matrix predicts.
#
# Analyses:
#   1. Conditional entropy reduction (ΔH with bootstrap CI)
#   2. Context-dependence test per trigram (binomial, FDR-corrected)
#   3. BIC Markov order comparison (order 1 vs restricted order 2)
#   4. Hierarchical null for 4-grams (2nd-order Markov baseline)
#   5. Hub/role classification + state-type cross-reference
#
# Prerequisites:
#   - Step 04 completed in 'select' mode
#   - Step 05a completed
#   - Step 06b completed
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/06c_higher_order_transitions.sh
#   sbatch --export=SUB_ID=sub-01,PARCELLATION=atlas-4S456Parcels script/06c_higher_order_transitions.sh
#   bash 06c_higher_order_transitions.sh
# =============================================================================

set -e


# Configuration (can be overridden via --export)
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

# Print header
echo "=============================================="
echo "06c - Higher-Order Transition Structure"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "=============================================="

# Run analysis
VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/06c_higher_order_transitions.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG}

echo "=============================================="
echo "Higher-order transition analysis complete!"
echo "=============================================="
