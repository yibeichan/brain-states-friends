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
VT="${VT:-0.95}"   # variance-threshold subdir of 05a/04 outputs (final/vt{VT}); "" for legacy layout
OUT="${OUT:-$CODE_DIR/site-staging/static/states}"
# 05b also writes a combined montage + individual_states to --output_dir; we don't
# need those for the viewer, and its default scratch path may hold read-only
# (git-annex) files. Send that byproduct to a fresh, writable, gitignored work dir.
WORK="${WORK:-$CODE_DIR/site-staging/.render_work}"
YABPLOT_TOOLS="${YABPLOT_TOOLS:-$HOME/yabplot/tools}"
mkdir -p "$OUT"

run() { uv run --project "$CODE_DIR" "$@"; }

# --- Ensure yabplot atlas meshes exist in the repo venv (built once, idempotent) ---
# yabplot READS meshes from its INSTALLED data dir (dirname(yabplot.__file__)/data/atlases),
# but the build tools WRITE to the yabplot SOURCE tree ($YABPLOT_TOOLS/../yabplot/data/atlases).
# When yabplot is a non-editable install these differ, so we build (if absent) then copy
# the atlas dirs into the installed location the renderer actually reads.
ATLAS_BASE="$(run python -c 'import os, yabplot; print(os.path.join(os.path.dirname(yabplot.__file__), "data", "atlases"))')"
SRC_ATLAS="$(cd -- "$YABPLOT_TOOLS/.." && pwd)/yabplot/data/atlases"

ensure_atlas() {  # $1 = atlas subdir, $2 = build tool
  local sub="$1" tool="$2"
  if [ -d "$ATLAS_BASE/$sub" ] && [ -n "$(ls -A "$ATLAS_BASE/$sub" 2>/dev/null)" ]; then
    return 0  # already installed where the renderer looks
  fi
  if [ ! -d "$SRC_ATLAS/$sub" ] || [ -z "$(ls -A "$SRC_ATLAS/$sub" 2>/dev/null)" ]; then
    echo "Building $sub meshes ..."
    run python "$YABPLOT_TOOLS/$tool"
  fi
  echo "Installing $sub meshes into venv: $ATLAS_BASE/$sub"
  mkdir -p "$ATLAS_BASE"
  cp -r "$SRC_ATLAS/$sub" "$ATLAS_BASE/"
}

ensure_atlas schaefer100_cortical build_schaefer100_cortical.py
ensure_atlas 4s_subcortical build_4s156_subcortical.py

# --- Headless rendering ---
# 05b self-starts an offscreen context (osmesa, else pv.start_xvfb()); xvfb-run
# provides a real DISPLAY as a safety net when osmesa is unavailable.
XVFB=""
if command -v xvfb-run >/dev/null 2>&1; then
  XVFB="xvfb-run -a"
fi

# --vt selects the variance-threshold subdir (final/vt{VT}); omit it entirely for the legacy layout.
VT_ARG=()
[ -n "$VT" ] && VT_ARG=(--vt "$VT")

for SUB in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06; do
  echo "Rendering $SUB ..."
  # Poster components go to a per-subject subdir: state IDs are per-subject, so a
  # flat dir would collide/overwrite across subjects and lose subject identity.
  $XVFB uv run --project "$CODE_DIR" python "$CODE_DIR/script/05b_visualize_recurring_states.py" \
    --sub_id "$SUB" --parcellation "$PARC" --n_states "$NSTATES" "${VT_ARG[@]}" \
    --output_dir "$WORK/$SUB" --poster_output_dir "$OUT/$SUB"
done

echo "Poster components written to $OUT/<sub-XX>/ (montage byproduct in $WORK)"
echo "Next: hand-write/update states.json referencing the produced PNGs, then re-run"
echo "site Task 2 (copy states + manifest onto the gh-pages worktree)."
