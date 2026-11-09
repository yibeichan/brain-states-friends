#!/bin/bash
# R6 (FC-transition controls), R7 (LOSO module agreement), R8 (LOSO map baseline),
# all six subjects; each run takes seconds to a minute.
# GATE_RV_SUB06: the 05f RV file 06b read for sub-06, if 05f was rewritten after
# 06b ran (see docs/findings/sm_rel_r6_fc_transition_controls.md).
set -euo pipefail
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
OUT_ROOT="${OUT_ROOT:-}"
for SUB_ID in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06; do
    for S in sm_rel_r6_fc_transition_controls sm_rel_r7_loso_module_agreement sm_rel_r8_loso_map_null; do
        ARGS=(--sub_id "$SUB_ID")
        [ -n "$OUT_ROOT" ] && ARGS+=(--out_dir "$OUT_ROOT/$S/atlas-4S156Parcels/$SUB_ID/vt0.95")
        if [ "$S" = sm_rel_r6_fc_transition_controls ] && [ "$SUB_ID" = sub-06 ] && [ -n "${GATE_RV_SUB06:-}" ]; then
            ARGS+=(--gate_rv_path "$GATE_RV_SUB06")
        fi
        uv run --project "${PROJECT_DIR}" python "${PROJECT_DIR}/script/$S.py" "${ARGS[@]}"
    done
done
