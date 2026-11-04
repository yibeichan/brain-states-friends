#!/bin/bash
#SBATCH --job-name=08e_transformer_cross_stim
#SBATCH --output=logs/08e_transformer_cross_stim_%A_%a.out
#SBATCH --error=logs/08e_transformer_cross_stim_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
_ENV="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)/script/utils/_env.sh"
[ -f "$_ENV" ] || _ENV="${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh"
source "$_ENV" || { echo "ERROR: cannot locate script/utils/_env.sh — submit from the repo root" >&2; exit 1; }

# Cap BLAS/OpenMP threads at the allocation; Python can otherwise see every
# core on the node.
n=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}
export OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n MKL_NUM_THREADS=$n

# =============================================================================
# 08e - Cross-stimulus aggregate transfer (D3a)
# =============================================================================
# Trains a per-layer Ridge classifier on Friends content_eligible ∩ test FO
# intersection, evaluates on the test stimulus, and builds a null by
# circular-shifting the Friends training labels within runs and refitting.
#
# Required exports:
#   STIMULUS    - one of movie10 / harrypotter / petitprince_fr / petitprince_en
#                 (default: movie10)
#   MODEL       - transformer model key (default: llama-3.2-3b)
#   VT          - 05e_a4 VT suffix       (default: 0.95)
#   N_PERMS     - permutation count      (default: 1000)
#   LAG         - brain-to-feature lag in TRs, or friends_best (default):
#                 the participant's best within-Friends lag for MODEL from
#                 08d D1. Non-zero lags write to <stimulus>_<model>_lag-best
#                 (or _lag-N); LAG=0 writes to the original folder.
#   FORCE       - set to 1 to recompute the cell even if its outputs exist
#                 (default: off). Implies FRESH=1: saved per-layer results are
#                 not reused, so leave FORCE unset when resubmitting a task
#                 that timed out, and do not use it to replace a stale
#                 best-lag output (it would restart every layer of every
#                 participant in the array on each resubmission).
#   FRESH       - set to 1 to ignore per-layer results saved by an earlier,
#                 interrupted run of the same cell (default: off). Without it,
#                 a resubmitted task continues from its last finished layer.
#   PER_SUBSET  - set to 1 to also emit per-subset breakdown JSON
#                 (currently honored only for STIMULUS=movie10, which splits
#                 by film: wolf / figures / bourne / life). Off by default.
#
# A completed best-lag cell whose lag or state eligibility disagrees with the
# current 08d D1 / 05e output stops with an error instead of being skipped;
# either side may be stale. If 08d D1 and 05e are current, move that
# participant's <stimulus>_<model>_lag-best folder aside and resubmit only
# that participant's array index (e.g. --array=5 for sub-06) without FORCE,
# with PER_SUBSET=1 if the folder held a D3a_per_subset file (movie10).
# 08e also stops, for any cell, when 08d D1 records a different eligibility
# than 05e gives now: bring 08d D1 up to date with 05e first.
#
# SLURM array index maps to subject (0 → sub-01, …, 5 → sub-06).
#
# HP and PP datasets do not include sub-04 - this is a CNeuroMod data-provenance
# limitation, NOT a quality-control exclusion. Array task 3 is auto-skipped (with
# exit 0) when STIMULUS is harrypotter / petitprince_fr / petitprince_en, so the
# uniform `--array=0-5` works for every stimulus without re-launching with a
# trimmed array.
#
# If 05e_a4 state_flags.csv is missing for a subject (not yet run, or a
# dropped annex object), a best-lag cell (LAG=friends_best, the default)
# stops and names the expected file: restore it (e.g. datalad get) or run
# `script/05e_temporal_trend_a4.sh` for that subject; do not rerun or move
# the 08e output. With an explicit LAG the script logs an ERROR, falls back
# to the 05a sub-HRF filter and writes `eligibility_source =
# sub_hrf_fallback` in the output JSON; content-encoding claims from such
# outputs are unreliable.
# =============================================================================

set -e

STIMULUS="${STIMULUS:-movie10}"
MODEL="${MODEL:-llama-3.2-3b}"
VT="${VT:-0.95}"
N_PERMS="${N_PERMS:-1000}"
LAG="${LAG:-friends_best}"
FORCE="${FORCE:-}"
PER_SUBSET="${PER_SUBSET:-}"
FRESH="${FRESH:-}"

SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
SUB_ID="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

# HP and PP datasets do not include sub-04 - this is a data-provenance
# limitation (CNeuroMod did not collect those stimuli for sub-04), not a
# quality-control exclusion. Skip array task 3 for those stimuli rather
# than failing in the python decode loader.
if [[ "${STIMULUS}" == "harrypotter" || "${STIMULUS}" == "petitprince_fr" || "${STIMULUS}" == "petitprince_en" ]] && [[ "${SUB_ID}" == "sub-04" ]]; then
    echo "Skipping ${SUB_ID} for ${STIMULUS} (subject not in dataset - CNeuroMod data-provenance limitation, not QC)"
    exit 0
fi

echo "=============================================="
echo "08e - Cross-stim aggregate transfer (D3a)"
echo "=============================================="
echo "Subject:        ${SUB_ID}"
echo "Test stimulus:  ${STIMULUS}"
echo "Model:          ${MODEL}"
echo "VT:             ${VT}"
echo "N perms:        ${N_PERMS}"
echo "Lag:            ${LAG}"
echo "Force:          ${FORCE:-no}"
echo "Per-subset:     ${PER_SUBSET:-no}"
echo "Fresh:          ${FRESH:-no}"
echo "=============================================="

CMD="uv run --project ${PROJECT_DIR} --no-sync python \
    ${PROJECT_DIR}/script/08e_transformer_cross_stim_aggregate.py \
    --sub_id ${SUB_ID} \
    --stimulus ${STIMULUS} \
    --model ${MODEL} \
    --vt ${VT} \
    --n_permutations ${N_PERMS} \
    --lag ${LAG}"

if [ "$FORCE" = "1" ]; then
    CMD="${CMD} --force --fresh"
fi
if [ "$PER_SUBSET" = "1" ]; then
    CMD="${CMD} --per_subset"
fi
if [ "$FRESH" = "1" ]; then
    CMD="${CMD} --fresh"
fi

eval $CMD

echo "=============================================="
echo "08e complete!"
echo "=============================================="
