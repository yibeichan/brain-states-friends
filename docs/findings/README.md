# Findings

Per-script findings documents: one Markdown file per analysis script, in a
pure-Results style (a Method-as-run header, tables of results, and output paths).
Every number is verified against the script's actual output files. Inference is
per-subject (n=6) unless a doc states otherwise; no group statistic is implied.

Tiers follow [`../../script/MANIFEST.md`](../../script/MANIFEST.md): **MAIN** =
in the preprint; **SUPP** = supplementary / robustness / diagnostic;
**CROSS-STIM** = Movie10 / Harry Potter / Petit Prince transfer (main, R5).

## Core pipeline

| Script | Tier | Findings | Manuscript |
|---|---|---|---|
| `00_postproc` | MAIN | [00_postproc.md](00_postproc.md) | Methods |
| `02_extract_parcel_ts` | MAIN | [02_extract_parcel_ts.md](02_extract_parcel_ts.md) | Methods |
| `03a_pca4combined_hmm` | MAIN | [03a_pca4combined_hmm.md](03a_pca4combined_hmm.md) | Methods |
| `03b_pca_loadings` | SUPP | [03b_pca_loadings.md](03b_pca_loadings.md) | Supp |
| `04_combined_hdphmm` | MAIN | [04_combined_hdphmm.md](04_combined_hdphmm.md) | Methods, R1 |
| `04ra_loso_struct_comp` | SUPP | [04ra_loso_struct_comp.md](04ra_loso_struct_comp.md) | Supp (reliability) |
| `04rb_split_half_reliability` | SUPP | [04rb_split_half_reliability.md](04rb_split_half_reliability.md) | Supp (reliability) |
| `04rc_reliability_fc` | SUPP | [04rc_reliability_fc.md](04rc_reliability_fc.md) | Supp (reliability) |
| `05a_recurrence_analysis` | MAIN | [05a_recurrence_analysis.md](05a_recurrence_analysis.md) | R1, Fig 1 |
| `05a_sub_hrf_diagnostic` | SUPP | [05a_sub_hrf_diagnostic.md](05a_sub_hrf_diagnostic.md) | Supp |
| `05b_recurring_states_visualization` | MAIN | [05b_recurring_states_visualization.md](05b_recurring_states_visualization.md) | Fig 2C / Fig S1 |
| `05c_episode_decodability` | SUPP | [05c_episode_decodability.md](05c_episode_decodability.md) | Supp |
| `05d_state_similarity` | SUPP | [05d_state_similarity.md](05d_state_similarity.md) | Supp |
| `05e_temporal_trend_a1` | SUPP | [05e_temporal_trend_a1.md](05e_temporal_trend_a1.md) | Supp |
| `05e_temporal_trend_a2` | SUPP | [05e_temporal_trend_a2.md](05e_temporal_trend_a2.md) | Supp |
| `05e_temporal_trend_a3` | SUPP | [05e_temporal_trend_a3.md](05e_temporal_trend_a3.md) | Supp |
| `05e_temporal_trend_a4` | MAIN | [05e_temporal_trend_a4.md](05e_temporal_trend_a4.md) | Methods (state classification) |
| `05f_state_fc` | SUPP | [05f_state_fc.md](05f_state_fc.md) | Supp |
| `06a_state_temp_dynamics` | MAIN | [06a_state_temp_dynamics.md](06a_state_temp_dynamics.md) | R3 |
| `06b_transition_structure` | MAIN | [06b_transition_structure.md](06b_transition_structure.md) | R3, Fig 3 |
| `06c_higher_order_transitions` | SUPP | [06c_higher_order_transitions.md](06c_higher_order_transitions.md) | Supp |
| `06d_preserved_chains` | SUPP | [06d_preserved_chains.md](06d_preserved_chains.md) | Supp |
| `08c_transformer_features` | MAIN | [08c_transformer_features.md](08c_transformer_features.md) | R4b, Methods |
| `08d_transformer_depth` | MAIN | [08d_transformer_depth.md](08d_transformer_depth.md) | R4b, Fig 4 |
| `08e_transformer_cross_stim_aggregate` | MAIN | [08e_transformer_cross_stim_aggregate.md](08e_transformer_cross_stim_aggregate.md) | R4b, Fig 4 |

## Cross-stimulus transfer (MAIN; R5, Figure 5)

| Script | Findings |
|---|---|
| `m10_03_project_movie_pca` | [m10_03_project_movie_pca.md](m10_03_project_movie_pca.md) |
| `m10_04_score_and_decode` | [m10_04_score_and_decode.md](m10_04_score_and_decode.md) |
| `m10_05_cross_stimulus_validation` | [m10_05_cross_stimulus_validation.md](m10_05_cross_stimulus_validation.md) |
| `hp_03_project_hp_pca` | [hp_03_project_hp_pca.md](hp_03_project_hp_pca.md) |
| `hp_04_score_and_decode` | [hp_04_score_and_decode.md](hp_04_score_and_decode.md) |
| `hp_05_cross_stimulus_validation` | [hp_05_cross_stimulus_validation.md](hp_05_cross_stimulus_validation.md) |
| `pp_03_project_pp_pca` | [pp_03_project_pp_pca.md](pp_03_project_pp_pca.md) |
| `pp_04_score_and_decode` | [pp_04_score_and_decode.md](pp_04_score_and_decode.md) |
| `pp_05_cross_stimulus_validation` | [pp_05_cross_stimulus_validation.md](pp_05_cross_stimulus_validation.md) |
| `rest_03_project_rest_pca` | [rest_03_project_rest_pca.md](rest_03_project_rest_pca.md) |
| `rest_04_score_and_decode` | [rest_04_score_and_decode.md](rest_04_score_and_decode.md) |
| `rest_05_cross_stimulus_validation` | [rest_05_cross_stimulus_validation.md](rest_05_cross_stimulus_validation.md) |
