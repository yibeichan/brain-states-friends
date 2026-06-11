#!/bin/bash
# Launch fresh 08d D1 LLaMA jobs for ORPHANED (subject,lag) cells — incomplete cells
# that have NO currently-running job (chain_08d_successors.sh only continues running ones).
# Per-layer checkpointing means each task resumes from the saved partials.
#
# Cells are at n_jobs=8/cpus=8 (4× the prior 2-wide setting; layers single-threaded).
# Orphan sets re-derived 2026-05-29: killed all crawling n_jobs=2 jobs, relaunching
# every incomplete (28/28-layer) cell at n_jobs=8. Excludes sub-04 + sub-06 (fully done),
# sub-03 lag5 (done), and sub-03 lag7/lag8 (covered by surviving D1f jobs 14683752/53).
#
# Usage: bash script/utils/launch_08d_orphans.sh [--dry-run]

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

PARTITION="mit_preemptable"
EXCLUDE_NODES="node2803,node3805"
TIME="1-12:00:00"
# MEM 48G→96G→160G (2026-05-29). Real memory hog is per-worker: each of n_jobs
# workers runs batch_loro_ridge_classify with the 1000-perm batch, so peak
# scales with n_jobs (8 concurrent perm-batches spike to ~100G). The per-layer
# slice fix in 08d (passing only one layer per worker) cut shipping overhead
# but the perm-batch is dominant. Observed peak ~100G at n_jobs=8 → 160G gives
# ~60% margin. mit_preemptable has 2TB nodes, so 160G schedules fine.
MEM="160G"
CPUS=8
N_JOBS=8
N_PERMS=1000
VT="0.95"
MODEL="llama-3.2-3b"
SCRIPT="script/08d_transformer_depth.sh"

# array index → subject: 0=sub-01 1=sub-02 2=sub-03 3=sub-04 4=sub-05 5=sub-06
# Orphaned subject-array sets per lag:
# 2026-05-29 relaunch @160G: excludes sub-03/sub-05 lag0 (running survivors
# 14731343_2/_4), sub-03 lag5/lag7/lag8 (done), sub-04+sub-06 (fully done).
declare -A ORPHANS=(
  [0]="0,1"
  [1]="0,1,2,4"
  [2]="0,1,2,4"
  [3]="0,1,2,4"
  [4]="0,1,2,4"
  [5]="0,1,4"
  [6]="0,1,2,4"
  [7]="0,1,4"
  [8]="0,1,4"
)

count=0
for lag in 0 1 2 3 4 5 6 7 8; do
    arr="${ORPHANS[$lag]}"
    [[ -z "$arr" ]] && continue
    name="08d_D1f_lag${lag}_${MODEL}"
    cmd="sbatch --parsable \
        --job-name=${name} \
        --output=logs/${name}_%A_%a.out \
        --error=logs/${name}_%A_%a.err \
        --time=${TIME} \
        --mem=${MEM} \
        --cpus-per-task=${CPUS} \
        --partition=${PARTITION} \
        --exclude=${EXCLUDE_NODES} \
        --requeue \
        --array=${arr} \
        --export=ALL,ANALYSES=D1,LAGS=${lag},MODEL=${MODEL},STIMULUS=friends,VT=${VT},N_PERMS=${N_PERMS},N_JOBS=${N_JOBS},PERLAGS= \
        ${SCRIPT}"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY] lag=${lag} array=${arr}"
        echo "      $cmd"
    else
        jid=$(eval "$cmd")
        echo "  lag=${lag} array=${arr} → ${jid}"
    fi
    count=$((count + 1))
done

echo ""
echo "Launched ${count} per-lag orphan arrays (cpus=${CPUS}, n_jobs=${N_JOBS}, n_perms=${N_PERMS})"
