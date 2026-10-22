#!/bin/bash
#SBATCH --job-name=06e_trans_modules
#SBATCH --output=logs/06e_trans_modules_%A_%a.out
#SBATCH --error=logs/06e_trans_modules_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --open-mode=append

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# =============================================================================
# 06e - Transition modules (R3)
# =============================================================================
# MODE=run (default): one (subject, variant) task per array index; null draws
#   are written in 100-draw chunks and finished chunks are skipped, so a
#   preempted task resumes where it stopped.
# MODE=aggregate: run once after every task has finished.
#
# Prerequisites: 04 (select + loso_fit), 05a, 05e_a4, 05f, 06b.
#
# Usage:
#   N=$(uv run --project . --no-sync python script/06e_transition_modules.py --mode tasks | wc -l)
#   jid=$(sbatch --parsable --array=0-$((N-1)) script/06e_transition_modules.sh)
#   sbatch --dependency=afterok:$jid --export=ALL,MODE=aggregate --time=04:00:00 \
#       --mem=12G --no-requeue --partition=pi_satra,mit_preemptable script/06e_transition_modules.sh
# =============================================================================

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MODE="${MODE:-run}"
echo "== $(date -Is) job ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:-none} restart ${SLURM_RESTART_COUNT:-0} mode ${MODE}"

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/06e_transition_modules.py" \
    --mode "${MODE}" --seed "${SEED:-0}" ${CAL_NULL:+--cal_null "${CAL_NULL}"}
