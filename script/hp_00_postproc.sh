#!/bin/bash
#
# SLURM submission script for Harry Potter fMRIPrep post-processing
#
# Purpose: Apply minimal confound regression to Harry Potter fMRIPrep CIFTI outputs.
#          Reuses 00_postproc.py with task=harrypotter.
#
# Harry Potter: word-by-word reading (RSVP at 2Hz), 7 runs per subject
# Subjects: sub-01, sub-02, sub-03, sub-05, sub-06 (no sub-04)
#
# Output: {SCRATCH_DIR}/output/00_postproc/{sub_id}/*task-harrypotter*_cleaned.dtseries.nii
#         (coexists with Friends and Movie10 files in same directory)
#
# Usage:
#   sbatch script/hp_00_postproc.sh
#
# Documentation: the design notes

#SBATCH --job-name=hp_postproc
#SBATCH --partition=mit_normal
#SBATCH --output=logs/hp_00_postproc_%A_%a.out
#SBATCH --error=logs/hp_00_postproc_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Activate conda environment
source ~/.bashrc
micromamba activate friends-states

# Determine project directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

SCRIPT_DIR="${PROJECT_DIR}/script"
mkdir -p "${PROJECT_DIR}/logs"

# Subject array (5 subjects — no sub-04 in Harry Potter)
sub_ids=("sub-01" "sub-02" "sub-03" "sub-05" "sub-06")
TASK_ID=${sub_ids[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "Harry Potter Post-processing - Minimal Confound Strategy"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject: $TASK_ID"
echo "Task: harrypotter"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Run post-processing with task=harrypotter (reuses 00_postproc.py unchanged)
uv run --no-sync python "${SCRIPT_DIR}/00_postproc.py" "${TASK_ID}" "harrypotter" --n_jobs $SLURM_CPUS_PER_TASK

exit_code=$?

echo ""
echo "============================================================"
echo "Job completed with exit code: $exit_code"
echo "End time: $(date)"
echo "============================================================"

exit $exit_code
