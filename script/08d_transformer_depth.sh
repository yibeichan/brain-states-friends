#!/bin/bash
#SBATCH --job-name=08d_transformer_depth
#SBATCH --output=logs/08d_transformer_depth_%A_%a.out
#SBATCH --error=logs/08d_transformer_depth_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=ou_bcs_normal,pi_satra
#SBATCH --array=0-5

# =============================================================================
# 08d - Transformer Depth (within-stimulus)
# =============================================================================
# Runs D1, D1-net, D1 confound baseline, and D2 on a single subject / stimulus /
# model. Replaces the old monolithic 08d_transformer_state_correspondence.
#
# Required exports (with defaults):
#   MODEL       - transformer model key (default: llama-3.2-3b)
#   STIMULUS    - stimulus dataset       (default: friends)
#   ANALYSES    - subset of "D1 D1net D1confound D2" (default: all four)
#   VT          - VT suffix for 05e_a4   (default: 0.95)
#   N_PERMS     - permutation count      (default: 1000)
#   LAGS        - space-separated lag subset for per-lag D1 (default: unset)
#   PERLAGS     - set to "1" to auto-submit 9 per-lag jobs + merge + downstream
#
# SLURM array index maps to subject (0 → sub-01, …, 5 → sub-06).
#
# HP and PP datasets do not include sub-04 - this is a CNeuroMod data-provenance
# limitation, NOT a quality-control exclusion. Array task 3 is auto-skipped (with
# exit 0) when STIMULUS is harrypotter / petitprince_fr / petitprince_en, so the
# uniform `--array=0-5` works for every stimulus without re-launching with a
# trimmed array.
#
# If 05e_a4 state_flags.csv is missing for any subject, the script logs an
# ERROR (non-fatal - falls back to 05a sub-HRF filter) and writes
# `eligibility_source = sub_hrf_fallback` in the output JSON. Content-encoding
# claims downstream of that fallback are unreliable; rerun
# `script/05e_temporal_trend_a4.sh` for that subject before interpreting.
#
# Usage:
#   # Friends, llama, all analyses (default - monolithic, needs very long wall)
#   sbatch script/08d_transformer_depth.sh
#
#   # Per-lag parallel (recommended - submits 9 lag jobs + merge + downstream)
#   bash script/08d_transformer_depth.sh PERLAGS=1
#   bash script/08d_transformer_depth.sh PERLAGS=1 MODEL=w2v-bert-2.0
#   bash script/08d_transformer_depth.sh PERLAGS=1 MODEL=dinov2-large
#
#   # Friends, w2v-bert audio (monolithic)
#   sbatch --export=MODEL=w2v-bert-2.0 script/08d_transformer_depth.sh
#
#   # Movie10, llama, D1 + D2 only
#   sbatch --export=STIMULUS=movie10,MODEL=llama-3.2-3b,ANALYSES="D1 D2" \
#       script/08d_transformer_depth.sh
#
#   # Single lag (manual - for re-running a timed-out lag)
#   sbatch --export=ANALYSES="D1",LAGS="3" script/08d_transformer_depth.sh
# =============================================================================

set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
fi

mkdir -p "${PROJECT_DIR}/logs"

# ── Parse inline KEY=VALUE args (for non-SLURM invocation) ────────────
for arg in "$@"; do
    if [[ "$arg" == *"="* ]]; then
        export "$arg"
    fi
done

MODEL="${MODEL:-llama-3.2-3b}"
STIMULUS="${STIMULUS:-friends}"
ANALYSES="${ANALYSES:-D1 D1net D1confound D2}"
VT="${VT:-0.95}"
# LLaMA W-sweep paths - see 2026-05-01_08c_llama_local_window_design.md §11.6.
# Both empty → production paths (08c_transformer_features/, 08d_transformer_depth/).
# Set together (e.g. _sweep_w3) to read sweep features and write per-W outputs.
FEATURES_SUBDIR_SUFFIX="${FEATURES_SUBDIR_SUFFIX:-}"
OUTPUT_SUBDIR_SUFFIX="${OUTPUT_SUBDIR_SUFFIX:-}"
N_PERMS="${N_PERMS:-1000}"
FORCE="${FORCE:-}"
LAGS="${LAGS:-}"
PERLAGS="${PERLAGS:-}"
N_JOBS="${N_JOBS:-2}"

# ── Per-lag parallel submission mode ──────────────────────────────────
# Run from the shell (bash, not sbatch) to submit a fan of SLURM jobs:
#   bash script/08d_transformer_depth.sh PERLAGS=1 MODEL=llama-3.2-3b
# Recursion is prevented by passing PERLAGS="" to all sub-jobs.
if [ "$PERLAGS" = "1" ]; then
    echo "=============================================="
    echo "08d - Per-lag parallel submission"
    echo "Stimulus:  ${STIMULUS}"
    echo "Model:     ${MODEL}"
    echo "VT:        ${VT}"
    echo "N perms:   ${N_PERMS}"
    echo "=============================================="

    LAG_JOBIDS=""
    for LAG in 0 1 2 3 4 5 6 7 8; do
        JOB=$(sbatch --parsable \
            --job-name="08d_D1_lag${LAG}_${MODEL}" \
            --output="logs/08d_D1_lag${LAG}_${MODEL}_%A_%a.out" \
            --error="logs/08d_D1_lag${LAG}_${MODEL}_%A_%a.err" \
            --time=24:00:00 \
            --mem=48G \
            --cpus-per-task=4 \
            --partition=ou_bcs_normal,pi_satra,mit_preemptable \
            --array=0-5 \
            --export=ALL,ANALYSES="D1",LAGS="${LAG}",MODEL="${MODEL}",STIMULUS="${STIMULUS}",VT="${VT}",N_PERMS="${N_PERMS}",FORCE="${FORCE}",PERLAGS="" \
            "${PROJECT_DIR}/script/08d_transformer_depth.sh")

        if [ -z "$LAG_JOBIDS" ]; then
            LAG_JOBIDS="$JOB"
        else
            LAG_JOBIDS="${LAG_JOBIDS}:${JOB}"
        fi
        echo "  lag=${LAG} → job ${JOB}"
    done

    # Merge job (lightweight - no feature loading)
    MERGE_JOB=$(sbatch --parsable \
        --dependency=afterok:${LAG_JOBIDS} \
        --job-name="08d_D1merge_${MODEL}" \
        --output="logs/08d_D1merge_${MODEL}_%A_%a.out" \
        --error="logs/08d_D1merge_${MODEL}_%A_%a.err" \
        --time=00:30:00 \
        --mem=8G \
        --cpus-per-task=1 \
        --partition=ou_bcs_normal,pi_satra \
        --array=0-5 \
        --export=ALL,ANALYSES="D1merge",MODEL="${MODEL}",STIMULUS="${STIMULUS}",VT="${VT}",PERLAGS="" \
        "${PROJECT_DIR}/script/08d_transformer_depth.sh")
    echo "  D1merge → job ${MERGE_JOB} (depends on lag jobs)"

    # Downstream analyses (D1net + D1confound + D2)
    DOWN_JOB=$(sbatch --parsable \
        --dependency=afterok:${MERGE_JOB} \
        --job-name="08d_downstream_${MODEL}" \
        --output="logs/08d_downstream_${MODEL}_%A_%a.out" \
        --error="logs/08d_downstream_${MODEL}_%A_%a.err" \
        --time=24:00:00 \
        --mem=48G \
        --cpus-per-task=4 \
        --partition=ou_bcs_normal,pi_satra \
        --array=0-5 \
        --export=ALL,ANALYSES="D1net D1confound D2",MODEL="${MODEL}",STIMULUS="${STIMULUS}",VT="${VT}",N_PERMS="${N_PERMS}",PERLAGS="" \
        "${PROJECT_DIR}/script/08d_transformer_depth.sh")
    echo "  downstream → job ${DOWN_JOB} (depends on D1merge)"

    echo "=============================================="
    echo "Submitted: 9 lag jobs → merge → downstream"
    echo "=============================================="
    exit 0
fi

# ── Normal execution (inside SLURM or direct) ────────────────────────

# Ensure the user-local uv install is on PATH (SLURM jobs may not inherit it)
export PATH="$HOME/.local/bin:$PATH"

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
echo "08d - Transformer Depth"
echo "=============================================="
echo "Subject:   ${SUB_ID}"
echo "Stimulus:  ${STIMULUS}"
echo "Model:     ${MODEL}"
echo "Analyses:  ${ANALYSES}"
echo "VT:        ${VT}"
echo "N perms:   ${N_PERMS}"
echo "Force:     ${FORCE:-no}"
echo "Lags:      ${LAGS:-all}"
echo "N jobs:    ${N_JOBS}"
echo "=============================================="

# Prevent BLAS multi-threading from contending with joblib workers.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

CMD="uv run --project ${PROJECT_DIR} --no-sync python \
    ${PROJECT_DIR}/script/08d_transformer_depth.py \
    --sub_id ${SUB_ID} \
    --stimulus ${STIMULUS} \
    --model ${MODEL} \
    --vt ${VT} \
    --n_permutations ${N_PERMS} \
    --n_jobs ${N_JOBS} \
    --features_subdir_suffix \"${FEATURES_SUBDIR_SUFFIX}\" \
    --output_subdir_suffix \"${OUTPUT_SUBDIR_SUFFIX}\" \
    --analyses ${ANALYSES}"

if [ "$FORCE" = "1" ]; then
    CMD="${CMD} --force"
fi

if [ -n "$LAGS" ]; then
    CMD="${CMD} --lags ${LAGS}"
fi

eval $CMD

echo "=============================================="
echo "08d depth complete!"
echo "=============================================="
