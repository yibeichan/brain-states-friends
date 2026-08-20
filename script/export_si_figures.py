#!/usr/bin/env python3
"""export_si_figures.py - map generator outputs onto the SI figure filenames.

Several SI figures are emitted by scripts whose output filenames do not match
the ``S<nn>_*`` names used in ``docs/supplementary/``. Before this script that
mapping lived only in someone's shell history, so the SI could not be rebuilt
and nobody could tell whether a given PNG was current. This file makes the
mapping explicit, checkable, and re-runnable.

Three categories:

  DIRECT      the generator already writes straight into
              docs/supplementary/figures/ under the SI name. Nothing to do.
  EXPORT      the generator writes elsewhere; this script copies it into place.
  SUPPLEMENTS the figure is rendered on the orphan ``supplements`` branch and
              cannot be produced from ``main``. Reported, never copied.

Usage:
    # report status only (default; does not touch any file)
    uv run python script/export_si_figures.py

    # copy every available EXPORT source into docs/supplementary/figures/
    uv run python script/export_si_figures.py --copy

Notes:
  * Sources may be git-annex symlinks. Copies resolve the link and copy the
    annex object, so the destination is a real file rather than a dangling link.
  * A MISSING export source is not an error in itself: it means that generator
    has not been run since its outputs were last cleared. The reported
    generator command is what to run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
SI_DIR = REPO / "docs" / "supplementary" / "figures"

# (si_filename, category, generator, emitted_relpath_under_manuscript_figures)
SI_MAP: list[tuple[str, str, str, str | None]] = [
    ("S01_recurring_state_surface_maps.png", "DIRECT",
     "script/fig_S01_recurring_state_surface_maps.py", None),
    ("S02_pca_loadings_A.png", "DIRECT",
     "script/fig_S02_pca_loadings.py", None),
    ("S02_pca_loadings_B.png", "DIRECT",
     "script/fig_S02_pca_loadings.py", None),
    ("S02_pca_loadings_C.png", "DIRECT",
     "script/fig_S02_pca_loadings.py", None),

    ("S03_model_selection_A.png", "EXPORT",
     "script/fig_S03_model_selection.py", "figS03/figS03_A_pareto_ll_vs_states.png"),
    ("S03_model_selection_B.png", "EXPORT",
     "script/fig_S03_model_selection.py", "figS03/figS03_B_states_vs_capacity.png"),
    ("S03_model_selection_C.png", "EXPORT",
     "script/fig_S03_model_selection.py", "figS03/figS03_C_overfit_gap.png"),

    ("S04_reliability_A.png", "EXPORT",
     "script/fig_S04_reliability.py", "figS04/figS04_A_matched_pair_r.png"),
    ("S04_reliability_B.png", "EXPORT",
     "script/fig_S04_reliability.py", "figS04/figS04_B_structural_invariants.png"),
    ("S04_reliability_C.png", "EXPORT",
     "script/fig_S04_reliability.py", "figS04/figS04_C_fc_rv_raw.png"),
    ("S04_reliability_D.png", "EXPORT",
     "script/fig_S04_reliability.py", "figS04/figS04_D_fc_rv_delta.png"),

    ("S05_video_peak_depth.png", "DIRECT",
     "script/fig_S05_video_peak_depth.py", None),

    # Emitted by the Figure 5 script's companion supplementary renderer.
    ("S06_run_onset_negative_control.png", "EXPORT",
     "script/fig_F4_within_friends.py (render_supp_negcontrol)",
     "figS_R4b_negcontrol/figS_R4b_negcontrol_triple.png"),

    # 08e_plots.py shares the fig3/ output directory with
    # fig_F3_transition_structure.py. Only the *_depth.png files are S7.
    ("S07_decoding_depth_strips_A.png", "EXPORT",
     "script/08e_plots.py", "fig3/fig3_A_audio_depth.png"),
    ("S07_decoding_depth_strips_B.png", "EXPORT",
     "script/08e_plots.py", "fig3/fig3_B_text_depth.png"),
    ("S07_decoding_depth_strips_C.png", "EXPORT",
     "script/08e_plots.py", "fig3/fig3_C_video_depth.png"),

    ("S08_cross_stimulus_validity_A.png", "EXPORT",
     "script/fig_S08_cross_stimulus_validity.py", "figS08/figS08_A_pca_transfer.png"),
    ("S08_cross_stimulus_validity_B.png", "EXPORT",
     "script/fig_S08_cross_stimulus_validity.py", "figS08/figS08_B_fit_vs_transfer.png"),
    ("S08_cross_stimulus_validity_C.png", "EXPORT",
     "script/fig_S08_cross_stimulus_validity.py", "figS08/figS08_C_presence_donut.png"),

    ("S09_individual_differences.png", "EXPORT",
     "script/fig_S09_individual_differences.py", "figS09/figS09_radar_strip.png"),

    ("S10_ica_convergence_A.png", "SUPPLEMENTS",
     "script/fig_sm_alt_ica_matching.py (supplements branch)", None),
    ("S10_ica_convergence_B.png", "SUPPLEMENTS",
     "script/fig_sm_alt_ica_matching.py (supplements branch)", None),

    ("S11_network_participation.png", "EXPORT",
     "script/fig_S11_network_participation_categories.py",
     "supp_network_participation_categories/figS11_network_participation_categories.png"),

    ("S12_ica_oos_recurrence_m10_A_wta.png", "SUPPLEMENTS",
     "script/fig_sm_alt_ica_oos_recurrence.py (supplements branch)", None),
    ("S12_ica_oos_recurrence_m10_B_continuous.png", "SUPPLEMENTS",
     "script/fig_sm_alt_ica_oos_recurrence.py (supplements branch)", None),
    ("S12_ica_oos_recurrence_hp_A_wta.png", "SUPPLEMENTS",
     "script/fig_sm_alt_ica_oos_recurrence.py (supplements branch)", None),
    ("S12_ica_oos_recurrence_hp_B_continuous.png", "SUPPLEMENTS",
     "script/fig_sm_alt_ica_oos_recurrence.py (supplements branch)", None),
    ("S12_ica_oos_recurrence_pp_A_wta.png", "SUPPLEMENTS",
     "script/fig_sm_alt_ica_oos_recurrence.py (supplements branch)", None),
    ("S12_ica_oos_recurrence_pp_B_continuous.png", "SUPPLEMENTS",
     "script/fig_sm_alt_ica_oos_recurrence.py (supplements branch)", None),
]

# Legacy emitted paths, kept only so a stale tree can be diagnosed. These are
# the pre-2026-08-19 SI numbering. If an EXPORT source is missing but its
# legacy path exists, the generator predates the renumbering and must be re-run.
LEGACY_SOURCES: dict[str, str] = {
    "figS08/figS08_A_pca_transfer.png": "figS06/figS06_A_pca_transfer.png",
    "figS08/figS08_B_fit_vs_transfer.png": "figS06/figS06_B_fit_vs_transfer.png",
    "figS08/figS08_C_presence_donut.png": "figS06/figS06_C_presence_donut.png",
    "figS09/figS09_radar_strip.png": "figS7/figS7_radar_strip.png",
    "supp_network_participation_categories/figS11_network_participation_categories.png":
        "supp_network_participation_categories/figS9_network_participation_categories.png",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--copy", action="store_true",
                    help="copy available EXPORT sources into docs/supplementary/figures/")
    args = ap.parse_args()

    load_dotenv()
    scratch = os.environ.get("SCRATCH_DIR")
    if not scratch:
        print("SCRATCH_DIR is not set; cannot resolve generator outputs.", file=sys.stderr)
        return 2
    figroot = Path(scratch) / "output" / "manuscript_figures"

    copied = stale = missing = 0
    print(f"{'SI file':44}{'status':<12}source / note")
    print("-" * 110)

    for si_name, category, generator, rel in SI_MAP:
        dest = SI_DIR / si_name

        if category == "DIRECT":
            status = "ok" if dest.exists() else "ABSENT"
            print(f"{si_name:44}{status:<12}written directly by {generator}")
            if not dest.exists():
                missing += 1
            continue

        if category == "SUPPLEMENTS":
            status = "ok" if dest.exists() else "ABSENT"
            print(f"{si_name:44}{status:<12}{generator}")
            if not dest.exists():
                missing += 1
            continue

        src = figroot / rel
        if src.exists():
            if args.copy:
                real = src.resolve()  # git-annex symlinks -> real object
                shutil.copyfile(real, dest)
                print(f"{si_name:44}{'copied':<12}{rel}")
            else:
                print(f"{si_name:44}{'available':<12}{rel}")
            copied += 1
            continue

        legacy = LEGACY_SOURCES.get(rel)
        if legacy and (figroot / legacy).exists():
            print(f"{si_name:44}{'STALE':<12}"
                  f"expected {rel}; only legacy {legacy} exists -> re-run {generator}")
            stale += 1
        else:
            print(f"{si_name:44}{'MISSING':<12}expected {rel} -> run {generator}")
            missing += 1

    print("-" * 110)
    verb = "copied" if args.copy else "available"
    print(f"{copied} {verb}, {stale} stale (generator predates the 2026-08-19 "
          f"renumbering), {missing} missing")
    if stale and not args.copy:
        print("Stale entries keep their current committed PNG, which is still valid "
              "content; they simply cannot be regenerated until the generator is re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
