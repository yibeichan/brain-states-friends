#!/bin/bash
#
# Submit the CNeuroMod hazard-10 end-to-end check: re-clean, parcellate, PCA and
# refit the combined HMM (full two-stage selection) for one or more subjects, once
# with the shared table's corrupt-mask CSF/WM regressors and once with its clean
# ones. Both variants use the same rebuild recipe, so their difference isolates the
# trilinear tissue-map defect.
#
# Each variant gets its own SCRATCH_DIR (RUN_ROOT/<variant>), so every stage reads
# and writes RUN_ROOT/<variant>/output/... and never touches the main outputs.
# Stage 2 and select use FIXED_VT=0.95, as the published fits did; Stage 1 still
# runs so its vt choice can be compared too.
#
# Usage (from the repo root, ideally a frozen worktree):
#   WMCSF_TABLE=<table v1 dir> RUN_ROOT=<new dir> bash script/hazard10_e2e_submit.sh sub-03 sub-04
#
# Partition: pi_satra, --no-requeue. Stage 00 overwrites its outputs and takes
# hours, and 02 skips existing files without atomic writes, so neither is safe to
# requeue mid-run.

set -euo pipefail

: "${WMCSF_TABLE:?set WMCSF_TABLE to the clean CSF/WM table directory}"
: "${RUN_ROOT:?set RUN_ROOT to a new directory for this check}"
[ "$#" -ge 1 ] || { echo "usage: $0 sub-XX [sub-YY ...]" >&2; exit 1; }
[ -f "${WMCSF_TABLE}/manifest.tsv" ] || { echo "ERROR: no manifest.tsv in ${WMCSF_TABLE}" >&2; exit 1; }
[ -f script/00_postproc.sh ] || { echo "ERROR: run from the repo root" >&2; exit 1; }

PROD_SCRATCH=$(grep '^SCRATCH_DIR=' .env | cut -d'=' -f2- | tr -d "\"'")
case "$(realpath -m "${RUN_ROOT}")/" in
    "$(realpath -m "${PROD_SCRATCH}")/output/"*)
        echo "ERROR: RUN_ROOT must not be inside the main output dataset" >&2; exit 1 ;;
esac

SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
PARCELLATION=atlas-4S156Parcels
FIXED_VT=0.95
N_STAGE2=18   # build_stage2_grid(0.95): 6 nc x 3 gamma x diag
COMMON=(--parsable --partition=pi_satra --no-requeue --mail-type=NONE)

for sub in "$@"; do
    idx=-1
    for i in "${!SUBJECTS[@]}"; do [ "${SUBJECTS[$i]}" = "$sub" ] && idx=$i; done
    [ "$idx" -ge 0 ] || { echo "ERROR: unknown subject $sub" >&2; exit 1; }

    for variant in corrupt clean; do
        root="${RUN_ROOT}/${variant}"
        if [ -e "${root}/output/04_combined_hdphmm/${PARCELLATION}/${sub}" ]; then
            echo "ERROR: ${root} already holds ${sub} fits; use a new RUN_ROOT" >&2; exit 1
        fi
        logs="${root}/logs"
        mkdir -p "${logs}"
        env_vars="ALL,SCRATCH_DIR=${root},SUBJECT_ID=${sub},PARCELLATION=${PARCELLATION}"
        tag="h10-${variant:0:2}-${sub#sub-}"

        j00=$(sbatch "${COMMON[@]}" --job-name="${tag}-00" --array="${idx}" \
            --cpus-per-task=8 --mem=48G --time=12:00:00 \
            --output="${logs}/00_%A_%a.out" --error="${logs}/00_%A_%a.err" \
            --export="${env_vars},WMCSF_TABLE=${WMCSF_TABLE},WMCSF_VARIANT=${variant}" \
            script/00_postproc.sh)

        j02=$(sbatch "${COMMON[@]}" --job-name="${tag}-02" --dependency="afterok:${j00}" \
            --cpus-per-task=2 --mem=16G --time=08:00:00 \
            --output="${logs}/02_%j.out" --error="${logs}/02_%j.err" \
            --export="${env_vars}" \
            --wrap="PATH=\$HOME/.local/bin:\$PATH uv run --project ${PWD} --no-sync python ${PWD}/script/02_extract_parcel_ts.py --subject_id ${sub} --parcellation ${PARCELLATION}")

        j03=$(sbatch "${COMMON[@]}" --job-name="${tag}-03a" --dependency="afterok:${j02}" \
            --array="${idx}" --cpus-per-task=1 --mem=16G --time=01:00:00 \
            --output="${logs}/03a_%A_%a.out" --error="${logs}/03a_%A_%a.err" \
            --export="${env_vars}" script/03a_pca4combined_hmm.sh)

        # Whole-array limits sized for the slowest configs (see the resource notes
        # in 04_combined_hdphmm.sh): stage 1 at vt=0.95/0.99 needs 36 h and stage 2
        # at nc>=80 needs 48 h. One task that times out cancels every afterok
        # dependant, so the faster configs share the longer limit.
        fit=(--dependency="afterok:${j03}" --cpus-per-task=8 --mem=32G)
        j41=$(sbatch "${COMMON[@]}" --job-name="${tag}-s1" "${fit[@]}" --array=0-4 \
            --time=36:00:00 \
            --output="${logs}/04s1_%A_%a.out" --error="${logs}/04s1_%A_%a.err" \
            --export="${env_vars},MODE=fit,STAGE=1" script/04_combined_hdphmm.sh)
        j42=$(sbatch "${COMMON[@]}" --job-name="${tag}-s2" "${fit[@]}" \
            --array=0-$((N_STAGE2 - 1)) --time=48:00:00 \
            --output="${logs}/04s2_%A_%a.out" --error="${logs}/04s2_%A_%a.err" \
            --export="${env_vars},MODE=fit,STAGE=2,FIXED_VT=${FIXED_VT}" script/04_combined_hdphmm.sh)

        jsel=$(sbatch "${COMMON[@]}" --job-name="${tag}-sel" \
            --dependency="afterok:${j41}:${j42}" --array=0 \
            --cpus-per-task=8 --mem=32G --time=10:00:00 \
            --output="${logs}/04sel_%A_%a.out" --error="${logs}/04sel_%A_%a.err" \
            --export="${env_vars},MODE=select,FIXED_VT=${FIXED_VT}" script/04_combined_hdphmm.sh)

        echo "${sub} ${variant}: 00=${j00} 02=${j02} 03a=${j03} s1=${j41} s2=${j42} select=${jsel} root=${root}"
    done
done
