#!/bin/bash
#SBATCH --job-name=pp_extract_parcel_ts
#SBATCH --partition=pi_satra
#SBATCH --output=logs/pp_02_extract_parcel_ts_%A_%a.out
#SBATCH --error=logs/pp_02_extract_parcel_ts_%A_%a.err
#SBATCH --array=0-17
# IMPORTANT: 18 PP episodes (9 FR + 9 EN) = 0-17. Adjust if episode list differs.
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# =============================================================================
# Petit Prince Parcel Time Series Extraction
# =============================================================================
# Reuses 02_extract_parcel_ts.py for Petit Prince episodes.
# Each array task processes one PP run for one subject.
#
# Prerequisites:
#   - pp_00_postproc.sh completed (cleaned PP CIFTIs in 00_postproc/{sub_id}/)
#   - Episode list file: {PROJECT_DIR}/petitprince_episode_ids.txt (one ID per line)
#
# Episode ID format (no "task-" prefix):
#   lppFR_run-1, ..., lppFR_run-9, lppEN_run-1, ..., lppEN_run-9
#
# Note: sub-06 has only 7 FR runs (no run-8, run-9). Those array tasks will
#       find no files and exit gracefully.
#
# Usage:
#   sbatch --export=SUBJECT_ID=sub-01 script/pp_02_extract_parcel_ts.sh
#   sbatch --export=SUBJECT_ID=sub-01,PARCELLATION=atlas-4S456Parcels script/pp_02_extract_parcel_ts.sh
#
# Documentation: the design notes
# =============================================================================

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

SCRIPT_DIR="${PROJECT_DIR}/script"
mkdir -p "${PROJECT_DIR}/logs"

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"

# --- Configuration (override via --export) ---
SUBJECT_ID="${SUBJECT_ID:-sub-01}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
EPISODE_FILE="${PROJECT_DIR}/petitprince_episode_ids.txt"

# Validate SLURM array context
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: This script must run as a SLURM array job via sbatch."
    exit 1
fi

# Validate episode file
if [ ! -f "$EPISODE_FILE" ]; then
    echo "Error: Episode file '$EPISODE_FILE' does not exist."
    echo "Create it with one PP episode ID per line (e.g., lppFR_run-1)."
    exit 1
fi

# Get episode ID for this array task
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
EPISODE_ID=$(sed -n "${LINE_NUMBER}p" "$EPISODE_FILE")

if [ -z "$EPISODE_ID" ]; then
    echo "No episode on line $LINE_NUMBER of '$EPISODE_FILE', exiting gracefully."
    exit 0
fi

echo "--- Petit Prince Parcel Extraction (Task: $SLURM_ARRAY_TASK_ID) ---"
echo "Subject: $SUBJECT_ID"
echo "Parcellation: $PARCELLATION"
echo "Episode: $EPISODE_ID"
echo "--------------------------------------------------------------"

uv run --project "${PROJECT_DIR}" --no-sync python "$SCRIPT_DIR/02_extract_parcel_ts.py" \
    --subject_id "$SUBJECT_ID" --parcellation "$PARCELLATION" --episode_id "$EPISODE_ID"

EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Successfully processed PP episode: '$EPISODE_ID'"
else
    echo "Error processing PP episode: '$EPISODE_ID' (exit status: $EXIT_STATUS)"
fi

exit $EXIT_STATUS
