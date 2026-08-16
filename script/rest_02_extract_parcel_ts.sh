#!/bin/bash
#
# SLURM submission script for hcptrt resting-state parcel time series extraction
#
# Purpose: Extract parcel-averaged time series from cleaned restingstate CIFTIs.
#          Reuses 02_extract_parcel_ts.py; --episode_id restingstate matches and
#          loops over ALL rest runs for the subject (session token disambiguates).
#
# Prerequisites: rest_00_postproc.sh completed
#
# Usage:
#   sbatch script/rest_02_extract_parcel_ts.sh
#   sbatch --export=PARCELLATION=atlas-4S456Parcels script/rest_02_extract_parcel_ts.sh

#SBATCH --job-name=rest_parcel_ts
#SBATCH --partition=mit_normal
#SBATCH --output=logs/rest_02_extract_%A_%a.out
#SBATCH --error=logs/rest_02_extract_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-5

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
source "${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh" 2>/dev/null || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/script/utils/_env.sh" || exit 1


PARCELLATION="${PARCELLATION:-atlas-4S156Parcels}"

sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")
SUBJECT_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

uv run --project "${PROJECT_DIR}" --no-sync python "${SCRIPT_DIR}/02_extract_parcel_ts.py" \
    --subject_id "$SUBJECT_ID" --parcellation "$PARCELLATION" \
    --episode_id restingstate
