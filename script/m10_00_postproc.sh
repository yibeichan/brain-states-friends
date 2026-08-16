#!/bin/bash
#
# SLURM submission script for movie10 fMRIPrep post-processing
#
# Purpose: Apply minimal confound regression to movie10 fMRIPrep CIFTI outputs.
#          Reuses 00_postproc.py with task=movie10.
#
# Movie types: Bourne (10 runs), Wolf of Wall Street (17), Hidden Figures (24), Life (10)
# Total: 61 movie runs per subject
#
# Output: {SCRATCH_DIR}/output/00_postproc/{sub_id}/*task-{bourne,wolf,figures,life}*_cleaned.dtseries.nii
#         (coexists with Friends task-s* files in same directory)
#
# Usage:
#   sbatch script/m10_00_postproc.sh
#
# Documentation: the design notes

#SBATCH --job-name=m10_postproc
#SBATCH --partition=mit_normal
#SBATCH --output=logs/m10_00_postproc_%A_%a.out
#SBATCH --error=logs/m10_00_postproc_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# Subject array (6 subjects - identical to Friends)
sub_ids=("sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06")
TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "Movie10 Post-processing - Minimal Confound Strategy"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject: $TASK_ID"
echo "Task: movie10"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Run post-processing with task=movie10 (reuses 00_postproc.py unchanged)
uv run --project "${PROJECT_DIR}" --no-sync python "${SCRIPT_DIR}/00_postproc.py" "${TASK_ID}" "movie10" --n_jobs $SLURM_CPUS_PER_TASK

exit_code=$?

echo ""
echo "============================================================"
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "============================================================"

exit $exit_code
