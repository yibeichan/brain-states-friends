# Findings: 08d Transformer Depth (within-stimulus)

_Script: `script/08d_transformer_depth.py`. Tier: MAIN (R4b, Fig 4)._

_Per-layer LORO RidgeClassifier + per-state AUC decoder sweep across transformer depth for brain-state correspondence; per-subject (n=6), stimulus=friends, three models._

## Method (as run)

- **Stimulus / models**: Friends x DINOv2-large (24 layers), Wav2VecBert 2.0 (24 layers), LLaMA 3.2 3B (28 layers)
- **Parcellation**: atlas-4S156Parcels; **PCA**: per-layer PCA at vt=0.95 on training split; K ranges by model below
- **State eligibility**: content_eligible from 05e_a4 state_flags.csv (fallback: 05a sub-HRF); run_onset_anchored used as design-driven negative control for DINOv2/W2V (deprecated as of 2026-05-31; see gate files)
- **D1 main**: LORO RidgeClassifier on content_eligible TRs; lag grid = [0..8] TRs (~0-12 s); lag=0 excluded from peak search; global BH-FDR across full layers x lags grid; n_perm=1000
- **D1 confound baseline**: 6-regressor timing-only Ridge (run_onset indicator, linear/quadratic/cubic within-run drift, episode_idx_norm, season_idx_norm) on content_eligible TRs at best lag; n_perm=1000; run for all three models (LLaMA backfilled 2026-06-12)
- **D1-net**: DINOv2 only; per-network-polarity groups (NETWORK_ORDER x {pos, neg}); min 5 states and 200 TRs per group; best lag from D1 main; BH-FDR per group across layers
- **D2**: per-state x layer AUC (vectorised Ridge); best lag from D1 main; 10-fold StratifiedGroupKFold (runs-as-groups, season-stratified); n_perm=500; selectivity threshold primary = 0.05 (max_minus_median); sensitivity at 0.03 / 0.05 / 0.10 reported; schema v3; complete for all 18 cells

### PCA component counts (vt=0.95, sub-01 representative; others within 1-3 PCs)

| Model | n_layers | K min | K max | n_train_TRs (sub-01) |
|---|---|---|---|---|
| dinov2-large | 24 | 78 | 544 | 96266 |
| w2v-bert-2.0 | 24 | 196 | 393 | 96266 |
| llama-3.2-3b | 28 | 1381 | 1818 | 96266 |

Sub-04 has fewer training TRs (63041) due to fewer episodes; sub-02/03/05 have ~95811; sub-06 has 96266.

## Results

### D1 Main: peak layer and effect size per subject and model

Peak selected by maximum balanced accuracy across lag=1..8 x all layers. Relative depth = peak_layer / (n_layers - 1). All peaks: p_fdr = 0.001 (global BH-FDR).

| Subject | Model | Peak layer | Rel. depth | Best lag (TR) | Bal. acc | Chance | NES |
|---|---|---|---|---|---|---|---|
| sub-01 | dinov2-large | 22/24 | 0.96 | 3 | 0.100 | 0.032 | 0.070 |
| sub-02 | dinov2-large | 22/24 | 0.96 | 3 | 0.085 | 0.033 | 0.053 |
| sub-03 | dinov2-large | 22/24 | 0.96 | 3 | 0.122 | 0.038 | 0.087 |
| sub-04 | dinov2-large | 22/24 | 0.96 | 3 | 0.084 | 0.037 | 0.049 |
| sub-05 | dinov2-large | 22/24 | 0.96 | 3 | 0.090 | 0.035 | 0.057 |
| sub-06 | dinov2-large | 21/24 | 0.91 | 3 | 0.117 | 0.063 | 0.058 |
| sub-01 | w2v-bert-2.0 | 11/24 | 0.48 | 3 | 0.124 | 0.032 | 0.095 |
| sub-02 | w2v-bert-2.0 | 11/24 | 0.48 | 4 | 0.118 | 0.033 | 0.088 |
| sub-03 | w2v-bert-2.0 | 13/24 | 0.57 | 3 | 0.135 | 0.039 | 0.101 |
| sub-04 | w2v-bert-2.0 | 12/24 | 0.52 | 4 | 0.096 | 0.037 | 0.061 |
| sub-05 | w2v-bert-2.0 | 11/24 | 0.48 | 4 | 0.120 | 0.035 | 0.089 |
| sub-06 | w2v-bert-2.0 | 13/24 | 0.57 | 4 | 0.150 | 0.063 | 0.094 |
| sub-01 | llama-3.2-3b | 13/28 | 0.48 | 3 | 0.109 | 0.032 | 0.079 |
| sub-02 | llama-3.2-3b | 14/28 | 0.52 | 3 | 0.118 | 0.033 | 0.088 |
| sub-03 | llama-3.2-3b | 17/28 | 0.63 | 3 | 0.111 | 0.039 | 0.075 |
| sub-04 | llama-3.2-3b | 15/28 | 0.56 | 3 | 0.095 | 0.037 | 0.061 |
| sub-05 | llama-3.2-3b | 13/28 | 0.48 | 3 | 0.096 | 0.035 | 0.064 |
| sub-06 | llama-3.2-3b | 16/28 | 0.59 | 3 | 0.154 | 0.063 | 0.098 |

### D1 confound baseline: timing-floor NES and delta (all three models)

delta_conf = NES_main(peak cell) - NES_confound(same lag). Confound baseline: 6 timing regressors. LLaMA confound baseline backfilled 2026-06-12 (array 15915784); confound best_lag = 3 matched the D1-main peak lag for all six subjects.

| Subject | Model | lag* | L* | NES_main | NES_conf | delta_conf |
|---|---|---|---|---|---|---|
| sub-01 | dinov2-large | 3 | 22 | 0.070 | 0.011 | +0.059 |
| sub-02 | dinov2-large | 3 | 22 | 0.053 | 0.012 | +0.041 |
| sub-03 | dinov2-large | 3 | 22 | 0.087 | 0.015 | +0.073 |
| sub-04 | dinov2-large | 3 | 22 | 0.049 | 0.004 | +0.044 |
| sub-05 | dinov2-large | 3 | 22 | 0.057 | 0.011 | +0.046 |
| sub-06 | dinov2-large | 3 | 21 | 0.058 | 0.018 | +0.040 |
| sub-01 | w2v-bert-2.0 | 3 | 11 | 0.095 | 0.011 | +0.084 |
| sub-02 | w2v-bert-2.0 | 4 | 11 | 0.088 | 0.013 | +0.075 |
| sub-03 | w2v-bert-2.0 | 3 | 13 | 0.101 | 0.015 | +0.086 |
| sub-04 | w2v-bert-2.0 | 4 | 12 | 0.061 | 0.004 | +0.058 |
| sub-05 | w2v-bert-2.0 | 4 | 11 | 0.089 | 0.011 | +0.078 |
| sub-06 | w2v-bert-2.0 | 4 | 13 | 0.094 | 0.018 | +0.076 |
| sub-01 | llama-3.2-3b | 3 | 13 | 0.079 | 0.011 | +0.068 |
| sub-02 | llama-3.2-3b | 3 | 14 | 0.088 | 0.012 | +0.076 |
| sub-03 | llama-3.2-3b | 3 | 17 | 0.075 | 0.015 | +0.060 |
| sub-04 | llama-3.2-3b | 3 | 15 | 0.061 | 0.004 | +0.057 |
| sub-05 | llama-3.2-3b | 3 | 13 | 0.064 | 0.011 | +0.053 |
| sub-06 | llama-3.2-3b | 3 | 16 | 0.098 | 0.018 | +0.080 |

DINOv2: delta_conf range +0.040 to +0.073, median +0.045 (6/6 positive). W2V: delta_conf range +0.058 to +0.086, median +0.077 (6/6 positive). LLaMA: delta_conf range +0.053 to +0.080, median +0.064 (6/6 positive).

### D1 neg-control gate (DINOv2 and W2V; deprecated for LLaMA)

neg_control_passed is evaluated at the D1-main peak cell (main vs run_onset_anchored NES with 0.5-null-std margin). The run-onset neg control was deprecated for LLaMA (2026-05-31) because run-onset-anchored states are not content-free; gate files for LLaMA have status="deprecated".

| Subject | Model | main_peak_acc | neg_peak_acc | delta_peak | neg_control_passed |
|---|---|---|---|---|---|
| sub-01 | dinov2-large | 0.100 | 0.281 | -0.181 | false |
| sub-02 | dinov2-large | 0.085 | 0.300 | -0.215 | false |
| sub-03 | dinov2-large | 0.122 | 0.213 | -0.091 | false |
| sub-04 | dinov2-large | 0.084 | 0.805 | -0.721 | false |
| sub-05 | dinov2-large | 0.090 | 0.219 | -0.129 | false |
| sub-06 | dinov2-large | 0.117 | 0.236 | -0.119 | false |
| sub-01 | w2v-bert-2.0 | 0.124 | 0.370 | -0.246 | false |
| sub-02 | w2v-bert-2.0 | 0.118 | 0.378 | -0.260 | false |
| sub-03 | w2v-bert-2.0 | 0.135 | 0.313 | -0.178 | false |
| sub-04 | w2v-bert-2.0 | 0.096 | 0.901 | -0.805 | false |
| sub-05 | w2v-bert-2.0 | 0.120 | 0.313 | -0.193 | false |
| sub-06 | w2v-bert-2.0 | 0.150 | 0.277 | -0.127 | false |

Note: neg_control_passed=false reflects n_classes asymmetry (2-8 run-onset-anchored classes vs 16-31 content-eligible classes), not a reversal of the D1 existence claim; D1_confound_baseline is the apples-to-apples comparator.

### D1-net: network-stratified decoding (DINOv2 only, lag=3)

Only groups passing min-5-states and min-200-TRs gates are shown. sub-06: all groups skipped. D1-net is a vision-specific analysis (it tests whether the DINOv2 depth peak is localized to a network); it is not applicable to the audio (W2V) or text (LLaMA) models.

| Subject | Group | n_states | n_TRs | Peak layer/24 | Rel. depth | Max bal. acc | NES |
|---|---|---|---|---|---|---|---|
| sub-01 | Vis_neg | 8 | 27031 | 16 | 0.70 | 0.180 | 0.063 |
| sub-01 | DorsAttn_pos | 5 | 15909 | 22 | 0.96 | 0.350 | 0.187 |
| sub-02 | Vis_pos | 5 | 16800 | 17 | 0.74 | 0.351 | 0.189 |
| sub-02 | Vis_neg | 5 | 20012 | 17 | 0.74 | 0.252 | 0.065 |
| sub-02 | Default_pos | 6 | 22024 | 22 | 0.96 | 0.235 | 0.082 |
| sub-03 | Default_pos | 5 | 20109 | 21 | 0.91 | 0.301 | 0.126 |
| sub-04 | Vis_neg | 5 | 14616 | 22 | 0.96 | 0.261 | 0.076 |
| sub-04 | SomMot_pos | 6 | 11030 | 20 | 0.87 | 0.192 | 0.030 |
| sub-04 | DorsAttn_pos | 6 | 11304 | 22 | 0.96 | 0.335 | 0.201 |
| sub-05 | Vis_pos | 6 | 21028 | 23 | 1.00 | 0.343 | 0.212 |
| sub-05 | SomMot_pos | 5 | 19201 | 17 | 0.74 | 0.225 | 0.031 |
| sub-05 | Default_pos | 5 | 20895 | 19 | 0.83 | 0.269 | 0.086 |
| sub-06 | (all groups skipped - too few states per group) | | | | | | |

### D2: per-state layer selectivity (selective state counts)

Selectivity criterion: max_minus_median >= threshold across per-state AUC layer profile. Primary threshold = 0.05. Sensitivity at 0.03 and 0.10 also shown. n_states = number of content_eligible states with FO >= 1%.

| Subject | Model | n_states | n_sel @ 0.03 | n_sel @ 0.05 (primary) | n_sel @ 0.10 | Median peak layer (@ 0.05) |
|---|---|---|---|---|---|---|
| sub-01 | dinov2-large | 31 | 4 | 3 | 0 | 22.0 |
| sub-02 | dinov2-large | 30 | 4 | 3 | 1 | 23.0 |
| sub-03 | dinov2-large | 26 | 5 | 2 | 2 | 23.0 |
| sub-04 | dinov2-large | 27 | 8 | 2 | 0 | 23.0 |
| sub-05 | dinov2-large | 29 | 4 | 2 | 0 | 22.5 |
| sub-06 | dinov2-large | 16 | 4 | 2 | 0 | 22.5 |
| sub-01 | w2v-bert-2.0 | 31 | 4 | 0 | 0 | - |
| sub-02 | w2v-bert-2.0 | 30 | 5 | 0 | 0 | - |
| sub-03 | w2v-bert-2.0 | 26 | 1 | 0 | 0 | - |
| sub-04 | w2v-bert-2.0 | 27 | 6 | 0 | 0 | - |
| sub-05 | w2v-bert-2.0 | 29 | 2 | 0 | 0 | - |
| sub-06 | w2v-bert-2.0 | 16 | 1 | 0 | 0 | - |
| sub-01 | llama-3.2-3b | 31 | 0 | 0 | 0 | - |
| sub-02 | llama-3.2-3b | 30 | 0 | 0 | 0 | - |
| sub-03 | llama-3.2-3b | 26 | 0 | 0 | 0 | - |
| sub-04 | llama-3.2-3b | 27 | 0 | 0 | 0 | - |
| sub-05 | llama-3.2-3b | 29 | 0 | 0 | 0 | - |
| sub-06 | llama-3.2-3b | 16 | 2 | 0 | 0 | - |

### D2: per-state max AUC distribution across all eligible states

Max AUC = best AUC across layers for each state. 10-fold StratifiedGroupKFold, n_perm=500.

| Subject | Model | n_states | Median max AUC | Min max AUC | Max max AUC |
|---|---|---|---|---|---|
| sub-01 | dinov2-large | 31 | 0.610 | 0.516 | 0.893 |
| sub-02 | dinov2-large | 30 | 0.600 | 0.532 | 0.784 |
| sub-03 | dinov2-large | 26 | 0.601 | 0.530 | 0.906 |
| sub-04 | dinov2-large | 27 | 0.556 | 0.519 | 0.757 |
| sub-05 | dinov2-large | 29 | 0.586 | 0.520 | 0.881 |
| sub-06 | dinov2-large | 16 | 0.562 | 0.531 | 0.789 |
| sub-01 | w2v-bert-2.0 | 31 | 0.662 | 0.576 | 0.960 |
| sub-02 | w2v-bert-2.0 | 30 | 0.670 | 0.567 | 0.836 |
| sub-03 | w2v-bert-2.0 | 26 | 0.637 | 0.511 | 0.954 |
| sub-04 | w2v-bert-2.0 | 27 | 0.600 | 0.533 | 0.782 |
| sub-05 | w2v-bert-2.0 | 29 | 0.640 | 0.549 | 0.930 |
| sub-06 | w2v-bert-2.0 | 16 | 0.615 | 0.556 | 0.809 |
| sub-01 | llama-3.2-3b | 31 | 0.637 | 0.545 | 0.845 |
| sub-02 | llama-3.2-3b | 30 | 0.664 | 0.537 | 0.783 |
| sub-03 | llama-3.2-3b | 26 | 0.610 | 0.529 | 0.837 |
| sub-04 | llama-3.2-3b | 27 | 0.590 | 0.517 | 0.747 |
| sub-05 | llama-3.2-3b | 29 | 0.606 | 0.520 | 0.799 |
| sub-06 | llama-3.2-3b | 16 | 0.611 | 0.538 | 0.768 |

## Outputs

- output/08d_transformer_depth/atlas-4S156Parcels/sub-*/friends_dinov2-large/D1_depth_profile.json
- output/08d_transformer_depth/atlas-4S156Parcels/sub-*/friends_dinov2-large/D1_neg_control_run_onset_anchored.json
- output/08d_transformer_depth/atlas-4S156Parcels/sub-*/friends_dinov2-large/D1_neg_control_gate.json
- output/08d_transformer_depth/atlas-4S156Parcels/sub-*/friends_dinov2-large/D1_confound_baseline.json
- output/08d_transformer_depth/atlas-4S156Parcels/sub-*/friends_dinov2-large/D1_net.json
- output/08d_transformer_depth/atlas-4S156Parcels/sub-*/friends_dinov2-large/D2_state_layer_auc.json
- output/08d_transformer_depth/atlas-4S156Parcels/sub-*/friends_w2v-bert-2.0/ (same files; D1_net is DINOv2-only)
- output/08d_transformer_depth/atlas-4S156Parcels/sub-*/friends_llama-3.2-3b/ (D1_depth_profile.json + D2_state_layer_auc.json + D1_confound_baseline.json; no D1_net)
- output/08d_transformer_depth/atlas-4S156Parcels/sub-*/friends_*/pca_info.json
- output/08d_transformer_depth/_plots/ (cross-subject summary plots)
- Figure: script/fig_F4_within_friends.py
