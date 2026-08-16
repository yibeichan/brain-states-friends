#!/bin/bash
#SBATCH --job-name=06b_trans_struct
#SBATCH --output=logs/06b_trans_struct_%A_%a.out
#SBATCH --error=logs/06b_trans_struct_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# =============================================================================
# 06b - Transition Structure Analysis
# =============================================================================
# Analyzes the directed transition graph between brain states:
#   A1. Graph topology (community detection, centrality)
#   A2. Transition selectivity & asymmetry
#   A3. Transition ↔ state properties (assortativity, FC correlation, homophily)
#   A4. Mean first passage time & MDS landscape
#
# Prerequisites:
#   - Step 04 completed in 'select' mode
#   - Step 05a completed (recurrence_summary.json)
#   - Step 06a completed (transition_probabilities.npy)
#   - Step 05f completed (fc_similarity_corr_rv.npy; optional for A3)
#
# Usage:
#   sbatch --export=SUB_ID=sub-01 script/06b_transition_structure.sh
#   sbatch --export=SUB_ID=sub-01,PARCELLATION=atlas-4S456Parcels script/06b_transition_structure.sh
#   sbatch --array=0-5 script/06b_transition_structure.sh
# =============================================================================

set -e

# Configuration (can be overridden via --export)
subjects=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUB_ID:-${subjects[${SLURM_ARRAY_TASK_ID:-0}]}}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
VT="${VT:-0.95}"

# Print header
echo "=============================================="
echo "06b - Transition Structure Analysis"
echo "=============================================="
echo "Subject: ${SUB_ID}"
echo "Parcellation: ${PARCELLATION}"
echo "VT: ${VT}"
echo "=============================================="

# Run transition structure analysis
VT_ARG=""
if [ -n "${VT}" ]; then
    VT_ARG="--vt ${VT}"
fi

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/06b_transition_structure.py" \
    --sub_id "${SUB_ID}" \
    --parcellation "${PARCELLATION}" \
    ${VT_ARG}

echo "=============================================="
echo "Transition structure analysis complete!"
echo "=============================================="
