#!/bin/bash
#SBATCH --job-name=annotated_brain
#SBATCH --output=logs/annotated_brain_%A_%a.out
#SBATCH --error=logs/annotated_brain_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=mit_normal
#SBATCH --array=0-59

# =============================================================================
# Annotated brain plot — batch over all states (legacy pipeline)
# =============================================================================
# Generates annotated brain surface plots + parcel tables for every state.
# Uses SLURM array jobs (one task per state) for parallel execution.
#
# Data source: legacy/03d_combined_hdphmm (sub-01 only)
#
# Usage:
#   sbatch script/utils/annotated_brain_plot.sh
#
# To run a subset of states:
#   sbatch --array=0-9 script/utils/annotated_brain_plot.sh
# =============================================================================

set -e

# ---- Project directory ----
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

# ---- Load project paths from .env (BASE_DIR, SCRATCH_DIR, DATA_DIR) ----
set -a; [ -f "${PROJECT_DIR}/.env" ] && . "${PROJECT_DIR}/.env"; set +a

# ---- Environment ----
eval "$(micromamba shell hook --shell bash)"
micromamba activate friends-states
export PYVISTA_OFF_SCREEN=1

# ---- Configuration ----
SUB_ID="${SUB_ID:-sub-01}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"

: "${SCRATCH_DIR:?SCRATCH_DIR must be set (see .env)}"
OUTPUT_BASE="${SCRATCH_DIR}/output"
STATE_MEANS="${OUTPUT_BASE}/legacy/03d_combined_hdphmm/${PARCELLATION}/${SUB_ID}/final/state_means_parcel.npy"

if [ ! -f "${STATE_MEANS}" ]; then
    echo "ERROR: state_means_parcel.npy not found for ${SUB_ID}"
    echo "  Path: ${STATE_MEANS}"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_BASE}/05b_annotated_brain/${PARCELLATION}/${SUB_ID}"

# ---- State index from array task ID ----
STATE_IDX=${SLURM_ARRAY_TASK_ID}

# ---- Header ----
echo "=============================================="
echo "Annotated Brain Plot (legacy pipeline)"
echo "=============================================="
echo "Subject:       ${SUB_ID}"
echo "Parcellation:  ${PARCELLATION}"
echo "State index:   ${STATE_IDX}"
echo "State means:   ${STATE_MEANS}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "=============================================="

# ---- Run ----
uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/utils/annotated_brain_plot.py" \
    --state_means_path "${STATE_MEANS}" \
    --state_idx "${STATE_IDX}" \
    --parcellation "${PARCELLATION}" \
    --output_dir "${OUTPUT_DIR}"

echo "Done: state ${STATE_IDX}"
