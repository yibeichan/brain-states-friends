#!/bin/bash
#SBATCH --job-name=05g_excluded_states
#SBATCH --output=logs/05g_excluded_states_%j.out
#SBATCH --error=logs/05g_excluded_states_%j.err
#SBATCH --time=01:00:00
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
# 05g - Excluded-state check
# =============================================================================
# Is each state excluded from content analyses (other than run-onset-anchored)
# an interruption of a neighbouring state, or a blend of the two states it sits
# between? Neighbours and blend partners are chosen among all active states. All six
# participants in one job (~2 min). Per-participant outputs are written
# atomically and the run is deterministic, so a preempted job reruns cleanly.
#
# Prerequisites: 03a (projected data), 04 (select), 05e_a4 (state_flags.csv).
# =============================================================================
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
echo "== $(date -Is) job ${SLURM_JOB_ID:-local} restart ${SLURM_RESTART_COUNT:-0}"

uv run --project "${PROJECT_DIR}" --no-sync python "${PROJECT_DIR}/script/05g_excluded_state_check.py" \
    --parcellation "${PARCELLATION:-atlas-4S156Parcels}" --vt "${VT:-0.95}" \
    --n_perm "${N_PERM:-5000}" --n_boot "${N_BOOT:-2000}" --seed "${SEED:-0}"
