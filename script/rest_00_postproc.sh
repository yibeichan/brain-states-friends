#!/bin/bash
#
# SLURM submission script for hcptrt resting-state fMRIPrep post-processing
#
# Purpose: Apply minimal confound regression to hcptrt restingstate CIFTI outputs.
#          Reuses 00_postproc.py with --fmriprep_dir + --bids_task restingstate.
#
# Resting state: one ~15 min run (~600 TRs, TR 1.49 s) per hcptrt session;
#                5/5/5/5/4/6 runs for sub-01..06 (sub-04 present, unlike HP/PP)
#
# Output: {SCRATCH_DIR}/output/00_postproc/{sub_id}/*task-restingstate*_bold_cleaned.dtseries.nii
#         (coexists with Friends/Movie10/HP/PP files in same directory)
#
# Usage:
#   sbatch script/rest_00_postproc.sh

#SBATCH --job-name=rest_postproc
#SBATCH --partition=mit_normal
#SBATCH --output=logs/rest_00_postproc_%A_%a.out
#SBATCH --error=logs/rest_00_postproc_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# hcptrt fMRIPrep derivatives (read-only, datalad-retrieved)
HCPTRT_FMRIPREP="/orcd/data/satra/002/datasets/all_about_cneuromod/cneuromod.processed/fmriprep/hcptrt"

# All six subjects (rest has sub-04, unlike HP/PP)
sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")
TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

uv run --project "${PROJECT_DIR}" --no-sync python "${SCRIPT_DIR}/00_postproc.py" "${TASK_ID}" "hcptrt" \
    --fmriprep_dir "${HCPTRT_FMRIPREP}/${TASK_ID}" \
    --bids_task restingstate \
    --n_jobs "$SLURM_CPUS_PER_TASK"
