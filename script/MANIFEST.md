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
  these are **main** (R5, Figure 5), not supplementary.

---

## Main pipeline

| Script | Role | Tier | Manuscript |
|---|---|---|---|
| `00_postproc.py` | Confound regression + z-scoring of fMRIPrep outputs | MAIN | Methods §1 |
| `00_get_scan.py` | Preprocessing helper (scan enumeration) | MAIN (util) | n/a |
| `01_get_parcel_label.py` | Extract parcel labels from atlas | MAIN | Methods §2 |
| `02_extract_parcel_ts.py` | Extract parcel time series (avg + binary) | MAIN | Methods §2 |
| `03a_pca4combined_hmm.py` | PCA fit + train/valid/test split (feeds HMM) | MAIN | Methods §3 |
| `04_combined_hdphmm.py` | Combined sHDP-HMM per subject (fit/select/loso) | MAIN | Methods §3, R1 |
| `05a_recurrence_analysis.py` | Recurrence + season-specificity classification | MAIN | R1, Fig 1 |
| `05b_visualize_recurring_states.py` | Cortical + subcortical surface plots | MAIN | Fig 2C / Fig S1 |
| `05e_temporal_trend_a4.py` | State-flag synthesis (drift-anchored taxonomy / eligibility) | MAIN | Methods §4 (state classification) |
| `06a_state_temp_dynamics.py` | Dwell-time distributions, transition matrices | MAIN | R3, Fig 3 |
| `06b_transition_structure.py` | Graph topology, FC-Mantel, MFPT landscape | MAIN | R3, Fig 3 |
| `08c_transformer_features.py` | Layer-wise transformer features (GPU) | MAIN | R4b, Methods §5 |
| `08d_transformer_depth.py` | D1 representational depth per layer | MAIN | R4b, Fig 4 |
| `08e_transformer_cross_stim_aggregate.py` | Cross-stimulus aggregate depth profile | MAIN | R4b cross-stim, Fig 4 |

## Cross-stimulus validation (MAIN; R5, Figure 5)

| Script | Role | Tier | Manuscript |
|---|---|---|---|
| `m10_03_project_movie_pca.py` | Project Movie10 through Friends PCA | CROSS-STIM (MAIN) | R5, Fig 5 |
| `m10_04_score_and_decode.py` | Score/decode Movie10 with Friends HMM | CROSS-STIM (MAIN) | R5, Fig 5 |
| `m10_05_cross_stimulus_validation.py` | Movie10 cross-stimulus validation | CROSS-STIM (MAIN) | R5, Fig 5 |
| `hp_03_project_hp_pca.py` | Project Harry Potter through Friends PCA | CROSS-STIM (MAIN) | R5, Fig 5 |
| `hp_04_score_and_decode.py` | Score/decode Harry Potter | CROSS-STIM (MAIN) | R5, Fig 5 |
| `hp_05_cross_stimulus_validation.py` | Harry Potter cross-stimulus validation | CROSS-STIM (MAIN) | R5, Fig 5 |
| `pp_03_project_pp_pca.py` | Project Petit Prince (FR/EN) through Friends PCA | CROSS-STIM (MAIN) | R5, Fig 5 |
| `pp_04_score_and_decode.py` | Score/decode Petit Prince | CROSS-STIM (MAIN) | R5, Fig 5 |
| `pp_05_cross_stimulus_validation.py` | Petit Prince cross-stimulus validation | CROSS-STIM (MAIN) | R5, Fig 5 |

(Note: `pp_00_postproc.sh`, `pp_02_extract_parcel_ts.sh`, `m10_00`, `m10_02`,
`hp_00`, `hp_02` are the per-stimulus preprocessing wrappers for the above,
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
| `fig_F4_within_friends.py` | Figure 4 (lead) | R4b |
| `fig_F4_per_film_video.py` | Figure 4 (Movie10 per-film panel) | R4b |
| `fig_F5_cross_stimulus_transfer.py` | Figure 5 | R5 |
| `fig_S01_recurring_state_surface_maps.py` | Figure S1 | Supp |
| `fig_S02_pca_loadings.py` | Figure S2 | Supp |
| `fig_S03_video_peak_depth.py` | Figure S3 | Supp |
| `fig_S06_cross_stimulus_validity.py` | Figure S6 | Supp |
| `fig_S07_individual_differences.py` | Figure S7 | Supp |
| `fig_S09_network_participation_categories.py` | Figure S9 | Supp |

Shared plotting helpers (imported by the figure scripts, not run directly):
`08d_plots.py`, `08e_plots.py`, `utils/network_participation.py`,
`utils/recurrence_plots.py`, `utils/temporal_plots.py`.

## Infrastructure (not analysis steps)

`__init__.py`, `utils/` (shared helpers: `stats.py`, `datalad_save.sh`,
plotting), `config/`, `tests/`, `dev/`, `__marimo__/`.
