#!/bin/bash
# Run the assortativity nulls for all six subjects locally (each takes ~1-3 min).
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
for SUB_ID in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06; do
    uv run --project "${PROJECT_DIR}" python "${PROJECT_DIR}/script/sm_rel_r3_assortativity_null.py" \
        --sub_id "$SUB_ID" --n_perm "${N_PERM:-5000}" --n_bins "${N_BINS:-5}" --seed "${SEED:-0}"
done
