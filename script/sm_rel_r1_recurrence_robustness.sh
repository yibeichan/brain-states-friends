#!/bin/bash
# Recurrence threshold sensitivity + occupancy confound, all six subjects (seconds each).
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
for SUB_ID in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06; do
    uv run --project "${PROJECT_DIR}" python "${PROJECT_DIR}/script/sm_rel_r1_recurrence_robustness.py" --sub_id "$SUB_ID"
done
