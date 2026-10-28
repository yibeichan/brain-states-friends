#!/bin/bash
#
# Submit the CNeuroMod clean-vs-corrupt CSF/WM end-to-end check: re-clean,
# parcellate, PCA and refit the combined HMM (full two-stage selection) for one or
# more subjects, once with the shared CSF/WM table's corrupt-mask regressors and
# once with its clean ones, then compare the two fits (clean_wmcsf_compare.py).
# Both variants use the same rebuild recipe, so their difference isolates the
# trilinear tissue-map defect.
#
# Each variant gets its own SCRATCH_DIR (RUN_ROOT/<variant>), so every stage reads
# and writes RUN_ROOT/<variant>/output/... and never touches the main outputs.
# Stage 2 and the final refit use FIXED_VT=0.95, as the published fits did. Stage 1
# still runs so the compare can report its vt pick; stage 2 leaves out the config
# that stage 1 also fits (the same directory), so no two tasks write it at once.
#
# Chain per subject and variant: 00 -> 02 -> 03a -> 04 stage 1 + stage 2 ->
# select_seed (one task per final seed) -> select_finalize. Once both variants of a
# subject are final, its compare runs and writes RUN_ROOT/compare/<sub>.json.
#
# Usage (from the repo root, ideally a frozen worktree with its own .venv):
#   WMCSF_TABLE=<table dir> RUN_ROOT=<new absolute dir> \
#       bash script/clean_wmcsf_e2e_submit.sh sub-03 sub-04
#
# Every check runs before the first sbatch. RUN_ROOT must be an absolute path
# outside the home directory and outside the main output dataset, each subject is
# given once, and each subject and variant claims RUN_ROOT/<variant>/.claim_<sub>
# with mkdir, so a second submission for the same subject is refused. To resubmit
# after a failed chain, check with squeue that none of its jobs is queued or
# running, then remove the claim.
#
# Jobs inherit the calling shell's environment except the variables the stages read
# as options (STAGE_VARS), which come only from this script. UV_PROJECT_ENVIRONMENT
# is among them, so jobs use this worktree's .venv.
#
# Partition: pi_satra, --no-requeue. Stage 00 overwrites its outputs and takes
# hours, and 02 skips existing files without atomic writes, so neither is safe to
# requeue mid-run. Limits come from measured runtimes: stage 1 at vt=0.95/0.99 and
# stage 2 at nc>=80 ran past 14 h / 20 h, and one final seed at nc100 took 5-11 h
# (17 h on a slower node); one task that times out cancels every afterok dependant.

set -euo pipefail

: "${WMCSF_TABLE:?set WMCSF_TABLE to the clean CSF/WM table directory}"
: "${RUN_ROOT:?set RUN_ROOT to a new directory for this check}"
[ "$#" -ge 1 ] || { echo "usage: $0 sub-XX [sub-YY ...]" >&2; exit 1; }
[ -f "${WMCSF_TABLE}/manifest.tsv" ] || { echo "ERROR: no manifest.tsv in ${WMCSF_TABLE}" >&2; exit 1; }
[ -f script/00_postproc.sh ] || { echo "ERROR: run from the repo root" >&2; exit 1; }

SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06)
VARIANTS=(corrupt clean)
PARCELLATION=atlas-4S156Parcels
FIXED_VT=0.95
N_FINAL_SEEDS=10
COMMON=(--parsable --partition=pi_satra --no-requeue --mail-type=NONE)
UV_PY="PATH=\$HOME/.local/bin:\$PATH uv run --project ${PWD} --no-sync python"
# Options the stage wrappers and scripts read from the environment.
STAGE_VARS=(SCRATCH_DIR SUBJECT_ID PARCELLATION MODE STAGE FIXED_VT N_FIT_SEEDS
            N_FINAL_SEEDS N_JOBS FORCE_REFIT SPLIT_MODE WMCSF_TABLE WMCSF_VARIANT
            UV_PROJECT_ENVIRONMENT)
UNSET=()
for v in "${STAGE_VARS[@]}"; do UNSET+=(-u "$v"); done

die() { echo "ERROR: $*" >&2; exit 1; }

# submit VAR=value ... -- <sbatch args>: sbatch with only these stage options set.
submit() {
    local vars=()
    while [ "$1" != "--" ]; do vars+=("$1"); shift; done
    shift
    env "${UNSET[@]}" "${vars[@]}" sbatch "${COMMON[@]}" --export=ALL "$@"
}

# --- checks (nothing is submitted until all pass) ---------------------------------

[[ "${RUN_ROOT}" = /* ]] || die "RUN_ROOT must be an absolute path, got ${RUN_ROOT}"
run_root=$(realpath -m "${RUN_ROOT}")
# Both paths are spliced unquoted into --wrap commands.
for p in "${run_root}" "${PWD}"; do
    [[ "$p" =~ ^[A-Za-z0-9._/+-]+$ ]] || die "path has characters other than letters, digits and ._/+-: $p"
done
home=$(realpath -m "${HOME}")
case "${run_root}/" in
    "${home}/"*) die "RUN_ROOT is inside the home directory (${home}); a chain writes hundreds of GB" ;;
esac

prod_scratch=$(sed -n 's/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}SCRATCH_DIR=//p' .env 2>/dev/null \
               | head -n 1 | sed 's/[[:space:]]#.*$//' | tr -d "\"'" | sed 's/[[:space:]]*$//')
[ -n "${prod_scratch}" ] || die "no SCRATCH_DIR line in ./.env; cannot locate the main output dataset"
prod_outputs=("$(realpath -m "${prod_scratch}")/output")
[ -n "${SCRATCH_DIR:-}" ] && prod_outputs+=("$(realpath -m "${SCRATCH_DIR}")/output")

declare -A seen
for sub in "$@"; do
    case " ${SUBJECTS[*]} " in *" ${sub} "*) ;; *) die "unknown subject ${sub}" ;; esac
    [ -z "${seen[$sub]:-}" ] || die "${sub} is given twice"
    seen[$sub]=1
done

for variant in "${VARIANTS[@]}"; do
    root="${run_root}/${variant}"
    for out in "${prod_outputs[@]}"; do
        case "${root}/output/" in
            "${out}/"*) die "${root}/output is the main output dataset ${out}" ;;
        esac
        case "${root}/" in
            "${out}/"*) die "${root} is inside the main output dataset ${out}" ;;
        esac
    done
done

read -r N_STAGE1 N_STAGE2 STAGE2_SKIP < <(python3 - <<'EOF'
import sys
sys.path.insert(0, "script/config")
import combined_hmm_config as c
s1 = [c.config_name(x) for x in c.build_stage1_grid()]
s2 = [c.config_name(x) for x in c.build_stage2_grid(0.95)]
print(len(s1), len(s2), ",".join(str(i) for i, n in enumerate(s2) if n in s1) or "-")
EOF
)
[ -n "${N_STAGE2:-}" ] || die "could not read the stage-1/stage-2 grids from script/config"
STAGE2_TASKS=()
for ((i = 0; i < N_STAGE2; i++)); do
    [[ ",${STAGE2_SKIP}," == *",${i},"* ]] || STAGE2_TASKS+=("$i")
done
STAGE2_ARRAY=$(IFS=,; echo "${STAGE2_TASKS[*]}")

claimed=()
release() { for c in "${claimed[@]}"; do rmdir "$c"; done; }
for sub in "$@"; do
    for variant in "${VARIANTS[@]}"; do
        root="${run_root}/${variant}"
        mkdir -p "${root}/logs"
        if [ -e "${root}/output/04_combined_hdphmm/${PARCELLATION}/${sub}" ]; then
            release; die "${root} already holds ${sub} fits; use a new RUN_ROOT"
        fi
        if ! mkdir "${root}/.claim_${sub}" 2>/dev/null; then
            release
            die "${root}/.claim_${sub} exists: a chain for ${sub} was already submitted here"
        fi
        claimed+=("${root}/.claim_${sub}")
    done
done
mkdir -p "${run_root}/compare/logs"

# --- submission --------------------------------------------------------------------

echo "stage 1: ${N_STAGE1} configs; stage 2: tasks ${STAGE2_ARRAY} (skips ${STAGE2_SKIP}, fit by stage 1)"
for sub in "$@"; do
    idx=-1
    for i in "${!SUBJECTS[@]}"; do [ "${SUBJECTS[$i]}" = "$sub" ] && idx=$i; done
    fin=()
    for variant in "${VARIANTS[@]}"; do
        root="${run_root}/${variant}"
        logs="${root}/logs"
        base=(SCRATCH_DIR="${root}" SUBJECT_ID="${sub}" PARCELLATION="${PARCELLATION}")
        tag="wcsf-${variant:0:2}-${sub#sub-}"

        j00=$(submit "${base[@]}" WMCSF_TABLE="${WMCSF_TABLE}" WMCSF_VARIANT="${variant}" -- \
            --job-name="${tag}-00" --array="${idx}" \
            --cpus-per-task=8 --mem=48G --time=12:00:00 \
            --output="${logs}/00_%A_%a.out" --error="${logs}/00_%A_%a.err" \
            script/00_postproc.sh)

        j02=$(submit "${base[@]}" -- --job-name="${tag}-02" --dependency="afterok:${j00}" \
            --cpus-per-task=2 --mem=16G --time=08:00:00 \
            --output="${logs}/02_%j.out" --error="${logs}/02_%j.err" \
            --wrap="${UV_PY} ${PWD}/script/02_extract_parcel_ts.py --subject_id ${sub} --parcellation ${PARCELLATION}")

        j03=$(submit "${base[@]}" -- --job-name="${tag}-03a" --dependency="afterok:${j02}" \
            --array="${idx}" --cpus-per-task=1 --mem=16G --time=01:00:00 \
            --output="${logs}/03a_%A_%a.out" --error="${logs}/03a_%A_%a.err" \
            script/03a_pca4combined_hmm.sh)

        fit=(--dependency="afterok:${j03}" --cpus-per-task=8 --mem=32G)
        j41=$(submit "${base[@]}" MODE=fit STAGE=1 -- --job-name="${tag}-s1" "${fit[@]}" \
            --array=0-$((N_STAGE1 - 1)) --time=36:00:00 \
            --output="${logs}/04s1_%A_%a.out" --error="${logs}/04s1_%A_%a.err" \
            script/04_combined_hdphmm.sh)
        j42=$(submit "${base[@]}" MODE=fit STAGE=2 FIXED_VT="${FIXED_VT}" -- \
            --job-name="${tag}-s2" "${fit[@]}" --array="${STAGE2_ARRAY}" --time=48:00:00 \
            --output="${logs}/04s2_%A_%a.out" --error="${logs}/04s2_%A_%a.err" \
            script/04_combined_hdphmm.sh)

        jseed=$(submit "${base[@]}" MODE=select_seed FIXED_VT="${FIXED_VT}" -- \
            --job-name="${tag}-seed" --dependency="afterok:${j41}:${j42}" \
            --array=0-$((N_FINAL_SEEDS - 1)) --cpus-per-task=8 --mem=32G --time=48:00:00 \
            --output="${logs}/04seed_%A_%a.out" --error="${logs}/04seed_%A_%a.err" \
            script/04_combined_hdphmm.sh)
        jfin=$(submit "${base[@]}" MODE=select_finalize FIXED_VT="${FIXED_VT}" \
            N_FINAL_SEEDS="${N_FINAL_SEEDS}" -- \
            --job-name="${tag}-fin" --dependency="afterok:${jseed}" \
            --array=0 --cpus-per-task=8 --mem=32G --time=04:00:00 \
            --output="${logs}/04fin_%A_%a.out" --error="${logs}/04fin_%A_%a.err" \
            script/04_combined_hdphmm.sh)
        fin+=("${jfin}")

        echo "${sub} ${variant}: 00=${j00} 02=${j02} 03a=${j03} s1=${j41} s2=${j42}" \
             "seeds=${jseed} finalize=${jfin} root=${root}"
    done

    jcmp=$(submit SCRATCH_DIR="${run_root}" -- --job-name="wcsf-cmp-${sub#sub-}" \
        --dependency="afterok:${fin[0]}:${fin[1]}" \
        --cpus-per-task=1 --mem=8G --time=01:00:00 \
        --output="${run_root}/compare/logs/${sub}_%j.out" \
        --error="${run_root}/compare/logs/${sub}_%j.err" \
        --wrap="${UV_PY} ${PWD}/script/clean_wmcsf_compare.py --run_root ${run_root} --sub_id ${sub} --out ${run_root}/compare/${sub}.json")
    echo "${sub} compare: ${jcmp} -> ${run_root}/compare/${sub}.json"
done
