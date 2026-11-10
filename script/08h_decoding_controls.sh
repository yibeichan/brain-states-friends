#!/bin/bash
# 08h_decoding_controls.sh - one 08h decoding-control cell per job.
#
# Usage (from the repo root):
#   sbatch --export=ALL,PART=friends,SUB=sub-01,MODEL=llama-3.2-3b,COND=loeo,OUT_ROOT=<side root> \
#       script/08h_decoding_controls.sh
#   sbatch --export=ALL,PART=transfer,SUB=sub-01,MODEL=llama-3.2-3b,STIM=movie10,OUT_ROOT=<side root> \
#       script/08h_decoding_controls.sh
#
# Per-layer partials make a Friends cell resumable, so a preempted job can be
# requeued and continues from the last finished layer (a partial is reused
# only when its fingerprint matches the run). Hence the preemptable partition
# and append mode, which keeps the log of every attempt.
#SBATCH --job-name=08h_ctrl
#SBATCH --partition=mit_preemptable,pi_satra
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --open-mode=append

_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }
set -euo pipefail

: "${PART:?PART=friends|transfer}" "${SUB:?}" "${MODEL:?}" "${OUT_ROOT:?}"
N_PERMS="${N_PERMS:-1000}"
N_JOBS="${N_JOBS:-2}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

ARGS=(--part "$PART" --sub_id "$SUB" --model "$MODEL" --out_root "$OUT_ROOT"
      --n_permutations "$N_PERMS" --n_jobs "$N_JOBS")
if [ "$PART" = "friends" ]; then
    ARGS+=(--condition "${COND:?COND required for PART=friends}")
    [ -n "${LAYERS:-}" ] && ARGS+=(--layers "$LAYERS")
else
    ARGS+=(--stimulus "${STIM:?STIM required for PART=transfer}")
fi

echo "== $(date -Is) job ${SLURM_JOB_ID:-none} restart ${SLURM_RESTART_COUNT:-0}"
cd "$PROJECT_DIR"
uv run --frozen --project "$PROJECT_DIR" python script/08h_decoding_controls.py "${ARGS[@]}"
