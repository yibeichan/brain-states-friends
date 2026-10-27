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
     "script/fig_S03_model_selection.py", "si_model_selection/figS03_A_pareto_ll_vs_states.png"),
    ("S03_model_selection_B.png", "EXPORT",
     "script/fig_S03_model_selection.py", "si_model_selection/figS03_B_states_vs_capacity.png"),
    ("S03_model_selection_C.png", "EXPORT",
     "script/fig_S03_model_selection.py", "si_model_selection/figS03_C_overfit_gap.png"),

    ("S04_reliability_A.png", "EXPORT",
     "script/fig_S04_reliability.py", "si_reliability/figS04_A_matched_pair_r.png"),
    ("S04_reliability_B.png", "EXPORT",
     "script/fig_S04_reliability.py", "si_reliability/figS04_B_structural_invariants.png"),
    ("S04_reliability_C.png", "EXPORT",
     "script/fig_S04_reliability.py", "si_reliability/figS04_C_fc_rv_raw.png"),
    ("S04_reliability_D.png", "EXPORT",
     "script/fig_S04_reliability.py", "si_reliability/figS04_D_fc_rv_delta.png"),

    ("S05_video_peak_depth.png", "DIRECT",
     "script/fig_S05_video_peak_depth.py", None),

    # Emitted by the Figure 5 script's companion supplementary renderer.
    ("S06_run_onset_negative_control.png", "EXPORT",
     "script/fig_F5_within_friends.py (render_supp_negcontrol)",
     "si_decoding_negative_control/figS_R4b_negcontrol_triple.png"),

    ("S07_decoding_depth_strips_A.png", "EXPORT",
     "script/08e_plots.py", "si_decoding_depth/si_decoding_depth_A_audio.png"),
    ("S07_decoding_depth_strips_B.png", "EXPORT",
     "script/08e_plots.py", "si_decoding_depth/si_decoding_depth_B_text.png"),
    ("S07_decoding_depth_strips_C.png", "EXPORT",
     "script/08e_plots.py", "si_decoding_depth/si_decoding_depth_C_video.png"),

    ("S08_cross_stimulus_validity_A.png", "EXPORT",
     "script/fig_S08_cross_stimulus_validity.py", "si_cross_stimulus_validity/figS08_A_pca_transfer.png"),
    ("S08_cross_stimulus_validity_B.png", "EXPORT",
     "script/fig_S08_cross_stimulus_validity.py", "si_cross_stimulus_validity/figS08_B_fit_vs_transfer.png"),
    ("S08_cross_stimulus_validity_C.png", "EXPORT",
     "script/fig_S08_cross_stimulus_validity.py", "si_cross_stimulus_validity/figS08_C_presence_donut.png"),

    ("S09_individual_differences.png", "EXPORT",
     "script/fig_S09_individual_differences.py", "si_individual_differences/figS09_radar_strip.png"),

    ("S10_ica_convergence_A.png", "SUPPLEMENTS",
     "script/fig_sm_alt_ica_matching.py (supplements branch)", None),
    ("S10_ica_convergence_B.png", "SUPPLEMENTS",
     "script/fig_sm_alt_ica_matching.py (supplements branch)", None),

    ("S11_network_participation.png", "EXPORT",
     "script/fig_S11_network_participation_categories.py",
     "si_network_participation_categories/figS11_network_participation_categories.png"),

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

    ("S13_phase_randomized_null.png", "SUPPLEMENTS",
     "script/fig_sm_rel_r5_phase_null.py (supplements branch)", None),

    ("S14_participant_umap_A_residual_matches.png", "EXPORT",
     "script/fig_F2E_participant_umap.py",
     "si_participant_umap/figS_participant_umap_A_residual_matches.png"),
    ("S14_participant_umap_B_active_categories.png", "EXPORT",
     "script/fig_F2E_participant_umap.py",
     "si_participant_umap/figS_participant_umap_B_active_categories.png"),
    ("S14_participant_umap_C_recurrence.png", "EXPORT",
     "script/fig_F2E_participant_umap.py",
     "si_participant_umap/figS_participant_umap_C_recurrence.png"),
    ("S14_participant_umap_D_robustness.png", "EXPORT",
     "script/fig_F2E_participant_umap.py",
     "si_participant_umap/figS_participant_umap_D_robustness.png"),
    ("S14_participant_umap_E_lagged_locking.png", "EXPORT",
     "script/fig_F2E_participant_umap.py",
     "si_participant_umap/figS_participant_umap_E_lagged_locking.png"),
    ("S14_participant_umap_legend_networks.png", "EXPORT",
     "script/fig_F2E_participant_umap.py",
     "fig2/fig2_E_legend_networks.png"),
    ("S14_participant_umap_legend_participants.png", "EXPORT",
     "script/fig_F2E_participant_umap.py",
     "fig2/fig2_E_legend_participants.png"),
    ("S14_participant_umap_legend_categories.png", "EXPORT",
     "script/fig_F2E_participant_umap.py",
     "si_participant_umap/figS_participant_umap_legend_categories.png"),
    ("S14_participant_umap_legend_recurrence.png", "EXPORT",
     "script/fig_F2E_participant_umap.py",
     "si_participant_umap/figS_participant_umap_legend_recurrence.png"),

    ("S15_transition_module_similarity_A_fc.png", "EXPORT",
     "script/fig_F3_transition_structure.py",
     "si_transition_modules/si_transition_modules_A_fc_similarity.png"),
    ("S15_transition_module_similarity_B_maps.png", "EXPORT",
     "script/fig_F3_transition_structure.py",
     "si_transition_modules/si_transition_modules_B_map_similarity.png"),
    ("S15_transition_module_similarity_legend_fc.png", "EXPORT",
     "script/fig_F3_transition_structure.py",
     "si_transition_modules/si_transition_modules_legend_fc.png"),
    ("S15_transition_module_similarity_legend_maps.png", "EXPORT",
     "script/fig_F3_transition_structure.py",
     "si_transition_modules/si_transition_modules_legend_map.png"),
    ("S15_transition_module_similarity_legend_networks.png", "EXPORT",
     "script/fig_F3_transition_structure.py",
     "fig3/fig3_A_legend_networks.png"),
]

# Legacy emitted paths, kept only so a stale tree can be diagnosed. These are
# the output paths used before the SI output folders were given topic names.
# If an EXPORT source is missing but its legacy path exists, the outputs have
# not been moved (or the generator re-run) since the rename.
LEGACY_SOURCES: dict[str, str] = {
    "si_model_selection/figS03_A_pareto_ll_vs_states.png": "figS03/figS03_A_pareto_ll_vs_states.png",
    "si_model_selection/figS03_B_states_vs_capacity.png": "figS03/figS03_B_states_vs_capacity.png",
    "si_model_selection/figS03_C_overfit_gap.png": "figS03/figS03_C_overfit_gap.png",
    "si_reliability/figS04_A_matched_pair_r.png": "figS04/figS04_A_matched_pair_r.png",
    "si_reliability/figS04_B_structural_invariants.png": "figS04/figS04_B_structural_invariants.png",
    "si_reliability/figS04_C_fc_rv_raw.png": "figS04/figS04_C_fc_rv_raw.png",
    "si_reliability/figS04_D_fc_rv_delta.png": "figS04/figS04_D_fc_rv_delta.png",
    "si_decoding_negative_control/figS_R4b_negcontrol_triple.png":
        "figS_R4b_negcontrol/figS_R4b_negcontrol_triple.png",
    "si_decoding_depth/si_decoding_depth_A_audio.png": "fig3/fig3_A_audio_depth.png",
    "si_decoding_depth/si_decoding_depth_B_text.png": "fig3/fig3_B_text_depth.png",
    "si_decoding_depth/si_decoding_depth_C_video.png": "fig3/fig3_C_video_depth.png",
    "si_cross_stimulus_validity/figS08_A_pca_transfer.png": "figS08/figS08_A_pca_transfer.png",
    "si_cross_stimulus_validity/figS08_B_fit_vs_transfer.png": "figS08/figS08_B_fit_vs_transfer.png",
    "si_cross_stimulus_validity/figS08_C_presence_donut.png": "figS08/figS08_C_presence_donut.png",
    "si_individual_differences/figS09_radar_strip.png": "figS09/figS09_radar_strip.png",
    "si_network_participation_categories/figS11_network_participation_categories.png":
        "supp_network_participation_categories/figS11_network_participation_categories.png",
    "si_participant_umap/figS_participant_umap_A_residual_matches.png":
        "fig2/supp_participant_umap/figS_participant_umap_A_residual_matches.png",
    "si_participant_umap/figS_participant_umap_B_active_categories.png":
        "fig2/supp_participant_umap/figS_participant_umap_B_active_categories.png",
    "si_participant_umap/figS_participant_umap_C_recurrence.png":
        "fig2/supp_participant_umap/figS_participant_umap_C_recurrence.png",
    "si_participant_umap/figS_participant_umap_D_robustness.png":
        "fig2/supp_participant_umap/figS_participant_umap_D_robustness.png",
    "si_participant_umap/figS_participant_umap_E_lagged_locking.png":
        "fig2/supp_participant_umap/figS_participant_umap_E_lagged_locking.png",
    "si_participant_umap/figS_participant_umap_legend_categories.png":
        "fig2/supp_participant_umap/figS_participant_umap_legend_categories.png",
    "si_participant_umap/figS_participant_umap_legend_recurrence.png":
        "fig2/supp_participant_umap/figS_participant_umap_legend_recurrence.png",
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
    print(f"{'SI file':52}{'status':<12}source / note")
    print("-" * 118)

    for si_name, category, generator, rel in SI_MAP:
        dest = SI_DIR / si_name

        if category == "DIRECT":
            status = "ok" if dest.exists() else "ABSENT"
            print(f"{si_name:52}{status:<12}written directly by {generator}")
            if not dest.exists():
                missing += 1
            continue

        if category == "SUPPLEMENTS":
            status = "ok" if dest.exists() else "ABSENT"
            print(f"{si_name:52}{status:<12}{generator}")
            if not dest.exists():
                missing += 1
            continue

        src = figroot / rel
        if src.exists():
            if args.copy:
                real = src.resolve()  # git-annex symlinks -> real object
                shutil.copyfile(real, dest)
                print(f"{si_name:52}{'copied':<12}{rel}")
            else:
                print(f"{si_name:52}{'available':<12}{rel}")
            copied += 1
            continue

        legacy = LEGACY_SOURCES.get(rel)
        if legacy and (figroot / legacy).exists():
            print(f"{si_name:52}{'STALE':<12}"
                  f"expected {rel}; only legacy {legacy} exists -> move it or re-run {generator}")
            stale += 1
        else:
            print(f"{si_name:52}{'MISSING':<12}expected {rel} -> run {generator}")
            missing += 1

    print("-" * 118)
    verb = "copied" if args.copy else "available"
    print(f"{copied} {verb}, {stale} stale (output still at its pre-rename path), "
          f"{missing} missing")
    if stale and not args.copy:
        print("Stale entries keep their current committed PNG, which is still valid "
              "content; they cannot be refreshed until the output is moved or the generator re-run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
