#!/bin/bash
#SBATCH --job-name=hp_extract_parcel_ts
#SBATCH --partition=mit_normal
#SBATCH --output=logs/hp_02_extract_parcel_ts_%A_%a.out
#SBATCH --error=logs/hp_02_extract_parcel_ts_%A_%a.err
#SBATCH --array=0-6
# IMPORTANT: 7 HP runs = 0-6. Adjust if episode list differs.
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# =============================================================================
# Harry Potter Parcel Time Series Extraction
# =============================================================================
# Reuses 02_extract_parcel_ts.py for Harry Potter episodes.
# Each array task processes one HP run for one subject.
#
# Prerequisites:
#   - hp_00_postproc.sh completed (cleaned HP CIFTIs in 00_postproc/{sub_id}/)
#   - Episode list file: {PROJECT_DIR}/harrypotter_episode_ids.txt (one ID per line)
#
# Episode ID format (no "task-" prefix):
#   harrypotter_run-1, harrypotter_run-2, ..., harrypotter_run-7
#
# Usage:
#   sbatch --export=SUBJECT_ID=sub-01 script/hp_02_extract_parcel_ts.sh
#   sbatch --export=SUBJECT_ID=sub-01,PARCELLATION=atlas-4S456Parcels script/hp_02_extract_parcel_ts.sh
#
# Documentation: the design notes
# =============================================================================

# --- Configuration (override via --export) ---
SUBJECT_ID="${SUBJECT_ID:-sub-01}"
PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"
EPISODE_FILE="${PROJECT_DIR}/harrypotter_episode_ids.txt"

# Validate SLURM array context
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: This script must run as a SLURM array job via sbatch."
    exit 1
fi

# Validate episode file
if [ ! -f "$EPISODE_FILE" ]; then
    echo "Error: Episode file '$EPISODE_FILE' does not exist."
    echo "Create it with one HP episode ID per line (e.g., harrypotter_run-1)."
    exit 1
fi

# Get episode ID for this array task
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
EPISODE_ID=$(sed -n "${LINE_NUMBER}p" "$EPISODE_FILE")

if [ -z "$EPISODE_ID" ]; then
    echo "No episode on line $LINE_NUMBER of '$EPISODE_FILE', exiting gracefully."
    exit 0
fi

echo "--- Harry Potter Parcel Extraction (Task: $SLURM_ARRAY_TASK_ID) ---"
echo "Subject: $SUBJECT_ID"
echo "Parcellation: $PARCELLATION"
echo "Episode: $EPISODE_ID"
echo "--------------------------------------------------------------"

uv run --project "${PROJECT_DIR}" --no-sync python "$SCRIPT_DIR/02_extract_parcel_ts.py" \
    --subject_id "$SUBJECT_ID" --parcellation "$PARCELLATION" --episode_id "$EPISODE_ID"

EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Successfully processed HP episode: '$EPISODE_ID'"
else
    echo "Error processing HP episode: '$EPISODE_ID' (exit status: $EXIT_STATUS)"
fi

exit $EXIT_STATUS
