# Findings: 05f Empirical Within-State Functional Connectivity

_Script: `script/05f_state_fc.py`. Tier: SUPP (Supp)._

_Computes state-conditioned parcel-space correlation matrices via Ledoit-Wolf shrinkage on Viterbi-assigned TRs, and derives delta FC, network-level aggregates, and RV-based FC similarity; per-subject n=6._

## Method (as run)

- Parcellation: atlas-4S156Parcels (156 parcels: 100 cortical Schaefer/Yeo-7 + 56 subcortical composite)
- Config: vt=0.95, nc=50, diagonal-covariance HMM
- Covariance estimator: Ledoit-Wolf shrinkage applied per state in parcel space
- Minimum TRs per state for inclusion: 30 (states below this threshold are excluded from FC estimation and RV computation)
- State assignments: Viterbi-decoded states from script 04 (mode: select); TRs pooled across all runs per state
- Network aggregation: 13 bins (Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default, BG, Midbrain-DA, Midbrain-Diencephalic, Thalamus, Hipp/Amyg, Cerebellum; parcel counts: 17, 14, 15, 12, 5, 13, 24, 16, 2, 8, 14, 6, 10)
- RV coefficient computed on active-state subset only; inactive states excluded to avoid identity-matrix inflation
- Delta FC defined as R_k - R_grand where R_grand is occupancy-weighted mean over active states
- No statistical testing; all delta FC values are descriptive
- Per-subject analysis; no group statistic

## Results

### Data coverage and FC differentiation

| Subject | Total TRs | Runs | Active states | Excluded states | Min TRs (active) | Max TRs (active) | Mean RV (full corr) | Min RV | Max RV | Mean delta/grand Frobenius ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 | 137,913 | 292 | 45 | 5 | 528 | 6,035 | 0.991 | 0.943 | 0.999 | 0.099 |
| sub-02 | 137,490 | 292 | 46 | 4 | 33 | 5,324 | 0.979 | 0.682 | 0.999 | 0.125 |
| sub-03 | 137,457 | 291 | 44 | 6 | 970 | 6,209 | 0.975 | 0.920 | 0.996 | 0.164 |
| sub-04 | 91,547 | 194 | 45 | 5 | 263 | 4,405 | 0.959 | 0.848 | 0.992 | 0.282 |
| sub-05 | 136,529 | 289 | 45 | 5 | 270 | 5,306 | 0.976 | 0.681 | 0.998 | 0.144 |
| sub-06 | 137,912 | 292 | 42 | 8 | 706 | 7,711 | 0.984 | 0.918 | 0.996 | 0.156 |

RV is computed on the full correlation matrices (156x156). Mean RV close to 1.0 reflects the dominant shared grand-mean FC backbone across all states. The delta/grand Frobenius ratio quantifies the relative magnitude of state-specific deviations; sub-04 has the highest (0.282) and sub-01 the lowest (0.099).

### Ledoit-Wolf shrinkage by subject

| Subject | Median alpha (active states) | Mean alpha | Min alpha | Max alpha (active states) |
|---|---|---|---|---|
| sub-01 | 0.0021 | 0.0026 | 0.0009 | 0.0116 |
| sub-02 | 0.0019 | 0.0170 | 0.0010 | 0.5846 |
| sub-03 | 0.0042 | 0.0048 | 0.0017 | 0.0123 |
| sub-04 | 0.0075 | 0.0087 | 0.0045 | 0.0418 |
| sub-05 | 0.0023 | 0.0036 | 0.0013 | 0.0355 |
| sub-06 | 0.0033 | 0.0038 | 0.0012 | 0.0124 |

Most active states have low shrinkage (alpha < 0.01), indicating reliable FC estimation. Sub-02 has a large mean alpha driven by state 47 (33 TRs, alpha=0.585), the only active state with fewer than 100 TRs where shrinkage strongly dominates.

### Strongest network-pair delta FC per subject

| Subject | Network pair | Delta R | State k |
|---|---|---|---|
| sub-01 | Limbic - Cerebellum | -0.242 | 17 |
| sub-02 | DorsAttn - Cerebellum | -0.429 | 47 |
| sub-03 | Vis - Cerebellum | +0.140 | 5 |
| sub-04 | DorsAttn - Default | +0.314 | 17 |
| sub-05 | SomMot - Cerebellum | -0.421 | 44 |
| sub-06 | Limbic - Cerebellum | -0.252 | 28 |

In 5 of 6 subjects the largest network-level delta FC involves the cerebellum. Sub-04 is the exception, where the strongest effect is cortical DorsAttn-Default coupling in state 17.

### Mean absolute cortical-subcortical delta FC by subcortical network

Rows are subjects; columns are subcortical networks. Values are mean |delta R| averaged across all active states and all 7 cortical networks.

| Subject | BG | Midbrain-DA | Midbrain-Diencephalic | Thalamus | Hipp/Amyg | Cerebellum |
|---|---|---|---|---|---|---|
| sub-01 | 0.012 | 0.014 | 0.014 | 0.024 | 0.017 | 0.025 |
| sub-02 | 0.026 | 0.032 | 0.026 | 0.044 | 0.035 | 0.042 |
| sub-03 | 0.013 | 0.014 | 0.010 | 0.023 | 0.018 | 0.024 |
| sub-04 | 0.029 | 0.029 | 0.024 | 0.054 | 0.040 | 0.051 |
| sub-05 | 0.024 | 0.029 | 0.023 | 0.044 | 0.033 | 0.042 |
| sub-06 | 0.022 | 0.026 | 0.020 | 0.038 | 0.024 | 0.036 |

Consistent across subjects: Thalamus shows the largest mean |delta R|, followed by Cerebellum and Hipp/Amyg. BG and Midbrain-Diencephalic show the smallest. Midbrain-DA has only 2 parcels (1 bilateral pair); its column is a single-structure estimate.

## Outputs

- output/05f_state_fc/atlas-4S156Parcels/{sub_id}/vt0.95/state_empirical_corr.npy: (K, 156, 156) per-state correlation matrices
- output/05f_state_fc/atlas-4S156Parcels/{sub_id}/vt0.95/state_delta_fc.npy: (K, 156, 156) delta correlation (R_k - R_grand)
- output/05f_state_fc/atlas-4S156Parcels/{sub_id}/vt0.95/grand_mean_corr.npy: (156, 156) occupancy-weighted grand mean correlation
- output/05f_state_fc/atlas-4S156Parcels/{sub_id}/vt0.95/fc_similarity_corr_rv.npy: (K, K) RV coefficient matrix (NaN for inactive states)
- output/05f_state_fc/atlas-4S156Parcels/{sub_id}/vt0.95/network_delta_fc.npy: (n_active, 13, 13) network-level delta FC
- output/05f_state_fc/atlas-4S156Parcels/{sub_id}/vt0.95/top_pairs_per_state.json: top 20 parcel pairs by |delta R| per active state
- output/05f_state_fc/atlas-4S156Parcels/{sub_id}/vt0.95/metadata.json: K, n_parcels, active/unreliable states, n_trs_per_state, shrinkage alphas, n_runs, total_trs
- output/05f_state_fc/atlas-4S156Parcels/{sub_id}/vt0.95/figures/: per-state network delta FC heatmaps + FC similarity heatmap sorted by 05e category
