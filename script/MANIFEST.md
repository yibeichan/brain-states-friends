# Script Manifest

Authoritative classification of every analysis script by role and manuscript
tier. Filenames are **not** renamed by tier because each numeric prefix is also
the output-directory name on scratch/RIA and the DataLad `--stage` key (see
[AGENTS.md](../AGENTS.md)). This table is the source of truth for the
main/supplementary boundary.

**Tiers**

- **MAIN**: in the preprint (Version B scope: R1 repertoire + R2 taxonomy + R3
  transitions + R4b transformer depth (aggregate + cross-stim) + R5 cross-stim
  recurrence transfer).
- **SUPP**: supplementary / robustness / diagnostic; in the repo and in the
  Supplementary Material, not the main figures.
- **CROSS-STIM (MAIN)**: Movie10 / Harry Potter / Petit Prince decode + transfer;
  these are **main** (R5, Figure 4), not supplementary.

> **Figure-order note (2026-07 revision):** manuscript Figures 4 and 5 were
> swapped for narrative coherence. Script and output-directory IDs are unchanged
> (filenames encode the scratch/RIA dir + DataLad stage key, per the policy
> above): code `F4`/`fig4` (transformer depth, R4b) now maps to **manuscript
> Figure 5**, and code `F5`/`fig5` (cross-stimulus transfer, R5) now maps to
> **manuscript Figure 4**. The Manuscript column below shows current manuscript
> numbers; R-labels (R4b = decoding, R5 = cross-stimulus) are finding IDs and
> did not change.

---

## Main pipeline

| Script | Role | Tier | Manuscript |
|---|---|---|---|
| `00_postproc.py` | Confound regression + z-scoring of fMRIPrep outputs | MAIN | Methods §1 |
| `00_get_scan.py` | Preprocessing helper (scan enumeration) | MAIN (util) | n/a |
| `01_get_parcel_label.py` | Extract parcel labels from atlas | MAIN | Methods §2 |
| `02_extract_parcel_ts.py` | Extract parcel time series (avg + binary) | MAIN | Methods §2 |
| `03a_pca4combined_hmm.py` | PCA fit + train/valid/test split (feeds HMM) | MAIN | Methods §3 |
| `04_combined_hdphmm.py` | Combined weak-limit HMM per subject (fit/select/loso) | MAIN | Methods §3, R1 |
| `05a_recurrence_analysis.py` | Recurrence + season-specificity classification | MAIN | R1, Fig 1 |
| `05b_visualize_recurring_states.py` | Cortical + subcortical surface plots | MAIN | Fig 2C / Fig S1 |
| `05e_temporal_trend_a4.py` | State-flag synthesis (drift-anchored taxonomy / eligibility) | MAIN | Methods §4 (state classification) |
| `06a_state_temp_dynamics.py` | Dwell-time distributions, transition matrices | MAIN | R3, Fig 3 |
| `06b_transition_structure.py` | Graph topology, FC-Mantel, MFPT landscape | MAIN | R3, Fig 3 |
| `08c_transformer_features.py` | Layer-wise transformer features (GPU) | MAIN | R4b, Methods §5 |
| `08d_transformer_depth.py` | D1 representational depth per layer | MAIN | R4b, Fig 5 |
| `08e_transformer_cross_stim_aggregate.py` | Cross-stimulus aggregate depth profile | MAIN | R4b cross-stim, Fig 5 |

## Cross-stimulus validation (MAIN; R5, Figure 4)

| Script | Role | Tier | Manuscript |
|---|---|---|---|
| `m10_03_project_movie_pca.py` | Project Movie10 through Friends PCA | CROSS-STIM (MAIN) | R5, Fig 4 |
| `m10_04_score_and_decode.py` | Score/decode Movie10 with Friends HMM | CROSS-STIM (MAIN) | R5, Fig 4 |
| `m10_05_cross_stimulus_validation.py` | Movie10 cross-stimulus validation | CROSS-STIM (MAIN) | R5, Fig 4 |
| `hp_03_project_hp_pca.py` | Project Harry Potter through Friends PCA | CROSS-STIM (MAIN) | R5, Fig 4 |
| `hp_04_score_and_decode.py` | Score/decode Harry Potter | CROSS-STIM (MAIN) | R5, Fig 4 |
| `hp_05_cross_stimulus_validation.py` | Harry Potter cross-stimulus validation | CROSS-STIM (MAIN) | R5, Fig 4 |
| `pp_03_project_pp_pca.py` | Project Petit Prince (FR/EN) through Friends PCA | CROSS-STIM (MAIN) | R5, Fig 4 |
| `pp_04_score_and_decode.py` | Score/decode Petit Prince | CROSS-STIM (MAIN) | R5, Fig 4 |
| `pp_05_cross_stimulus_validation.py` | Petit Prince cross-stimulus validation | CROSS-STIM (MAIN) | R5, Fig 4 |
| `rest_03_project_rest_pca.py` | Project hcptrt rest through Friends PCA | CROSS-STIM (MAIN) | R5 ext |
| `rest_04_score_and_decode.py` | Score/decode hcptrt rest with Friends HMM | CROSS-STIM (MAIN) | R5 ext |
| `rest_05_cross_stimulus_validation.py` | Rest cross-stimulus validation (+C1 vigilance drift) | CROSS-STIM (MAIN) | R5 ext |

(Note: `pp_00_postproc.sh`, `pp_02_extract_parcel_ts.sh`, `m10_00`, `m10_02`,
`hp_00`, `hp_02`, `rest_00`, `rest_02` are the per-stimulus preprocessing wrappers for the above,
same MAIN tier.)

## Supplementary / validation / diagnostics

| Script | Role | Tier | Manuscript |
|---|---|---|---|
| `03b_pca_loadings.py` | PCA loadings diagnostics (A1–A7) | SUPP | Supp (PCA variance) |
| `04_patch_selection_metrics.py` | Backfill selection metrics into metadata | SUPP (util) | n/a |
| `04ra_loso_struct_comp.py` | LOSO structural comparison (reliability) | SUPP | Supp reliability |
| `04rb_split_half_reliability.py` | Split-half reliability | SUPP | Supp reliability |
| `04rc_reliability_fc.py` | Reliability of state FC | SUPP | Supp reliability |
| `04rv_reliability_vis.py` | Reliability visualization | SUPP | Supp reliability |
| `05a_sub_hrf_diagnostic.py` | Sub-HRF state diagnostic | SUPP | Supp |
| `05c_episode_decodability.py` | Season decodability from FO (exploratory) | SUPP | Supp |
| `05d_state_similarity.py` | Cross-state similarity | SUPP | Supp |
| `05e_temporal_trend_a1.py` | Cross-episode temporal trends | SUPP | Supp |
| `05e_temporal_trend_a2.py` | Within-run position anchoring | SUPP | Supp |
| `05e_temporal_trend_a3.py` | Within-session FO habituation (LME) | SUPP | Supp |
| `05f_state_fc.py` | Per-state functional connectivity | SUPP | Supp |
| `06c_higher_order_transitions.py` | Higher-order transition adequacy (entropy, BIC) | SUPP | Supp |
| `06d_preserved_chains.py` | Preserved transition chains | SUPP | Supp |

## Figure

Each manuscript figure is built by a dedicated `fig_*.py` that reads pipeline
outputs and renders the panels.

| Script | Figure | Manuscript |
|---|---|---|
| `fig_F1_recurrence_gradient.py` | Figure 1 | R1 |
| `fig_F2_recurrence_sources.py` | Figure 2 | R2 |
| `fig_F2_network_participation.py` | Figure 2 (Panel C batch renderer) | R2 |
| `fig_F3_transition_structure.py` | Figure 3 | R3 |
| `fig_F4_within_friends.py` | Figure 5 (lead) [code ID F4] | R4b |
| `fig_F4_per_film_video.py` | Figure 5 (Movie10 per-film panel) [code ID F4] | R4b |
| `fig_F5_cross_stimulus_transfer.py` | Figure 4 [code ID F5]; rest is supplementary-only (Fig S8) | R5 |
| `fig_S01_recurring_state_surface_maps.py` | Figure S1 | Supp |
| `fig_S02_pca_loadings.py` | Figure S2 | Supp |
| `fig_S03_model_selection.py` | Figure S3 (K_max selection sweep) | Supp |
| `fig_S04_reliability.py` | Figure S4 (LOSO + split-half reproducibility) | Supp |
| `fig_S05_video_peak_depth.py` | Figure S5 | Supp |
| `fig_S08_cross_stimulus_validity.py` | Figure S8 (all panels include Rest — the rest result's only figure home) | Supp |
| `fig_S09_individual_differences.py` | Figure S9 | Supp |
| `fig_S11_network_participation_categories.py` | Figure S11 | Supp |

SI figures with no dedicated `fig_S*` entry point (provenance verified
2026-08-19 by locating the emitting `savefig` call, not by name similarity):

| SI figure | Emitted by | Emitted filename |
|---|---|---|
| S6 | `fig_F4_within_friends.py` → `render_supp_negcontrol()` | `manuscript_figures/figS_R4b_negcontrol/figS_R4b_negcontrol_triple.{png,svg}` |
| S7 | `08e_plots.py` → `render_panel()`, one call per modality panel | `manuscript_figures/fig3/fig3_<panel>_depth.{pdf,png,svg}` |

**S10** and **S12** are rendered on the orphan `supplements` branch by
`fig_sm_alt_ica_matching.py` and `fig_sm_alt_ica_oos_recurrence.py`, so they
cannot be produced from `main`.

### Rebuilding the SI figure directory

`export_si_figures.py` owns the complete mapping from generator output to SI
filename, replacing what used to be an undocumented set of manual renames:

```bash
uv run python script/export_si_figures.py          # status only
uv run python script/export_si_figures.py --copy   # place the files
```

It classifies each SI figure as DIRECT (the generator already writes the SI
name: S1, S2, S5), EXPORT (copied by this script: S3, S4, S6, S7, S8, S9, S11),
or SUPPLEMENTS (S10, S12). It resolves git-annex symlinks before copying, so
destinations are real files, and it reports STALE when a generator still writes
a pre-2026-08-19 output name. Current status is **16 exported, 0 stale,
0 missing**.

The committed PNGs under `docs/supplementary/figures/` are byte-for-byte what
`--copy` emits, so re-running the exporter leaves the working tree clean. That
is the check to run if you suspect a committed SI figure has drifted from its
generator.

All three generators that were stale at renumbering time have been re-run and
verified: `fig_S09_individual_differences.py` and
`fig_S11_network_participation_categories.py` reproduce pixel-identically, so
their staleness was only in the output-path mapping. `fig_S08_cross_stimulus_
validity.py` had genuinely drifted — its committed panels predated the
resting-state addition (panel C had three rows, not four) — and the regenerated
version is what is committed now.

Two output-path gotchas the mapping encodes: `fig3/` is shared by
`fig_F3_transition_structure.py` and `08e_plots.py`, so only the `*_depth.png`
files in it belong to S7; and `fig_S11_*` writes into
`supp_network_participation_categories/`, not a `figS11/` directory.

Numbering note (2026-08-19): SI figures were renumbered to follow the Methods
reading order. `S3`/`S4` are new (model selection, reliability); former
`S3`–`S10` shifted to `S5`–`S12`.

Shared plotting helpers (imported by the figure scripts, not run directly):
`08d_plots.py`, `08e_plots.py`, `utils/network_participation.py`,
`utils/recurrence_plots.py`, `utils/temporal_plots.py`.

## Infrastructure (not analysis steps)

`__init__.py`, `utils/` (shared helpers: `stats.py`, `datalad_save.sh`,
plotting), `config/`, `tests/`, `dev/`, `__marimo__/`.
