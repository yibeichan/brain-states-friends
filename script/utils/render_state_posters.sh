#!/bin/bash
# Regenerate individual per-state brain-render PNGs for the project-page viewer.
#
# Self-contained on the PUBLIC repo. yabplot's atlas meshes are generated once
# (they are NOT shipped with the package); this script builds them into the
# repo's venv if absent, then renders each subject's top states under a headless
# X server. Also needs the per-subject HMM artifacts under $SCRATCH/output
# (shared scratch, auto-loaded from .env).
#
# Output: per-state cortical+subcortical PNGs in site-staging/static/states/.
# After running, update states.json to reference the produced PNGs, then copy
# both onto the gh-pages worktree (site Task 2).
#
# Env overrides: PARC, NSTATES, OUT, YABPLOT_TOOLS.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PARC="${PARC:-atlas-4S156Parcels}"
NSTATES="${NSTATES:-5}"
OUT="${OUT:-$CODE_DIR/site-staging/static/states}"
YABPLOT_TOOLS="${YABPLOT_TOOLS:-$HOME/yabplot/tools}"
mkdir -p "$OUT"

run() { uv run --project "$CODE_DIR" "$@"; }

# --- Ensure yabplot atlas meshes exist in the repo venv (built once, idempotent) ---
# yabplot reads meshes from <yabplot install>/data/atlases/{schaefer100_cortical,4s_subcortical}.
ATLAS_BASE="$(run python -c 'import os, yabplot; print(os.path.join(os.path.dirname(yabplot.__file__), "data", "atlases"))')"

if [ ! -d "$ATLAS_BASE/schaefer100_cortical" ] || [ -z "$(ls -A "$ATLAS_BASE/schaefer100_cortical" 2>/dev/null)" ]; then
  echo "Building cortical atlas meshes (schaefer100_cortical) ..."
  run python "$YABPLOT_TOOLS/build_schaefer100_cortical.py"
fi

if [ ! -d "$ATLAS_BASE/4s_subcortical" ] || ! ls "$ATLAS_BASE/4s_subcortical"/*.vtk >/dev/null 2>&1; then
  echo "Building subcortical atlas meshes (4s_subcortical) ..."
  run python "$YABPLOT_TOOLS/build_4s156_subcortical.py"
fi

# --- Headless rendering ---
# 05b self-starts an offscreen context (osmesa, else pv.start_xvfb()); xvfb-run
# provides a real DISPLAY as a safety net when osmesa is unavailable.
XVFB=""
if command -v xvfb-run >/dev/null 2>&1; then
  XVFB="xvfb-run -a"
fi

for SUB in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06; do
  echo "Rendering $SUB ..."
  $XVFB uv run --project "$CODE_DIR" python "$CODE_DIR/script/05b_visualize_recurring_states.py" \
    --sub_id "$SUB" --parcellation "$PARC" --n_states "$NSTATES" --poster_output_dir "$OUT"
done

echo "Renders written to $OUT"
echo "Next: hand-write/update states.json referencing the produced PNGs, then re-run"
echo "site Task 2 (copy states + manifest onto the gh-pages worktree)."
