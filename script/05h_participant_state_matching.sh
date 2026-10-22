#!/bin/bash
#SBATCH --job-name=05h_state_matching
#SBATCH --output=logs/05h_state_matching_%j.out
#SBATCH --error=logs/05h_state_matching_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --open-mode=append

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# =============================================================================
# 05h - Matching states between participants
# =============================================================================
# All 15 participant pairs in one job (~4 min). Each pair's summary is written
# atomically and finished pairs are skipped (FORCE=1 recomputes them), so a
# preempted job resumes; the cross-pair summary is written at the end.
#
# Prerequisites: 04 (select + split_half fits), 05e_a4, 05g; ATLAS_DIR in .env.
# =============================================================================
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
echo "== $(date -Is) job ${SLURM_JOB_ID:-local} restart ${SLURM_RESTART_COUNT:-0}"

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05h_participant_state_matching.py" \
    --parcellation "${PARCELLATION:-atlas-4S156Parcels}" --vt "${VT:-0.95}" \
    --n_perm "${N_PERM:-1000}" --n_lock "${N_LOCK:-1000}" --seed "${SEED:-0}" ${FORCE:+--force}
