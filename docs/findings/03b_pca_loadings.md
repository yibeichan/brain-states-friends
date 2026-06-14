# Findings: 03b PCA Loadings Diagnostics

_Script: `script/03b_pca_loadings.py`. Tier: SUPP (Supp)._

_Standalone diagnostic / QC for the PCA models fit in script 03a: loadings, residual variance, motion-artifact flags, and leave-one-season-out stability. It does not fit any model and is not part of the inferential pipeline; it characterizes the same per-subject PCA space the combined HMM (04) consumes. Per-subject, n=6, atlas-4S156Parcels._

## Method (as run)

- This is a diagnostic script: it re-opens the PCA model from 03a and reports QC. PCA is fit once in 03a; the variance threshold only selects how many already-computed components to retain.
- Parcellation: atlas-4S156Parcels (156 parcels: 100 cortical Schaefer/Yeo-17, 56 subcortical)
- Input: per-subject `pca_model.pkl` and `n_pcs_lookup.json` from script 03a
- Residual variance (A3) and LOSO stability (A5) computed at vt=0.95, the production HMM PCA truncation (script default aligned to 0.95)
- k at vt=0.95 varies by subject (66-77 PCs retained; see Results). The A7 panel sweeps additional thresholds (0.80-0.99) for reference.
- Motion artifact flags: SomMotA+SomMotB fraction threshold = 0.30 for PC1-3; all-subcortical fraction threshold = 0.25 for PC1-2. These are per-PC (PC1-3) loading energies and do not depend on the retained-PC count.
- LOSO stability flag threshold: coefficient of variation (CV) > 0.20 across season-leave-out folds
- sub-04 has 4 LOSO folds (seasons 1-4 only); all others have 6 folds (seasons 1-6)
- A6 cross-subject comparison was not run in production (no cross_subject output directory present)
- Per-subject results only; no group statistic

## Results

### PC count at vt=0.95 (production truncation, used for A3/A5)

| Subject | k (PCs retained at vt=0.95) |
|---|---|
| sub-01 | 75 |
| sub-02 | 72 |
| sub-03 | 72 |
| sub-04 | 77 |
| sub-05 | 66 |
| sub-06 | 74 |

### Motion artifact flags (A4 / flags JSON)

SomMot fraction = squared loading energy from SomMotA+SomMotB as fraction of total PC energy. Subcortical fraction = all 6 subcortical groups combined. Flag threshold: SomMot > 0.30 in PC1-3; Subcortical > 0.25 in PC1-2. These quantities are per-PC and identical regardless of variance threshold.

| Subject | SomMot PC1 | SomMot PC2 | SomMot PC3 | Subcort PC1 | Subcort PC2 | any_flag | flagged PC(s) |
|---|---|---|---|---|---|---|---|
| sub-01 | 0.128 | 0.070 | 0.192 | 0.024 | 0.007 | false | none |
| sub-02 | 0.113 | 0.040 | 0.031 | 0.026 | 0.004 | false | none |
| sub-03 | 0.091 | 0.049 | 0.005 | 0.020 | 0.006 | false | none |
| sub-04 | 0.115 | 0.088 | 0.032 | 0.021 | 0.002 | false | none |
| sub-05 | 0.115 | 0.049 | 0.198 | 0.031 | 0.005 | false | none |
| sub-06 | 0.116 | 0.085 | 0.301 | 0.029 | 0.009 | true | PC3 (SomMot) |

### Per-parcel residual variance at vt=0.95 (A3)

Residual fraction = variance not captured by the retained k PCs, expressed as a fraction of total parcel variance.

| Subject | Median residual fraction | Mean residual fraction | Min | Max |
|---|---|---|---|---|
| sub-01 | 0.042 | 0.152 | 0.000 | 0.903 |
| sub-02 | 0.043 | 0.144 | 0.000 | 0.884 |
| sub-03 | 0.047 | 0.167 | 0.000 | 0.930 |
| sub-04 | 0.052 | 0.185 | 0.000 | 0.943 |
| sub-05 | 0.047 | 0.141 | 0.000 | 0.904 |
| sub-06 | 0.047 | 0.152 | 0.000 | 0.885 |

Parcels with the highest residual fraction remain subcortical: Globus Pallidus (GPe/GPi), hypothalamus, and ventral thalamic parcels (e.g., sub-01 LH-GPe 0.90, RH-GPe 0.87, LH-HTH 0.84; sub-04 RH-GPe 0.94, LH-GPe 0.93).

### Per-network mean residual fraction at vt=0.95 (A3, cross-subject pattern)

Networks ordered by mean residual (highest at top). Values are mean residual fractions across subject-level parcel estimates.

| Network | sub-01 | sub-02 | sub-03 | sub-04 | sub-05 | sub-06 |
|---|---|---|---|---|---|---|
| Thalamus | 0.582 | 0.631 | 0.614 | 0.729 | 0.463 | 0.600 |
| Hipp/Amyg | 0.427 | 0.303 | 0.376 | 0.398 | 0.265 | 0.366 |
| Basal Ganglia | 0.252 | 0.208 | 0.368 | 0.387 | 0.340 | 0.306 |
| Midbrain-Diencephalic | 0.208 | 0.198 | 0.349 | 0.283 | 0.285 | 0.289 |
| LimbicB | 0.348 | 0.305 | 0.194 | 0.287 | 0.161 | 0.178 |
| LimbicA | 0.274 | 0.130 | 0.156 | 0.265 | 0.138 | 0.150 |
| Cerebellum | 0.169 | 0.157 | 0.158 | 0.246 | 0.155 | 0.105 |
| SalVentAttnA | 0.063 | 0.072 | 0.078 | 0.067 | 0.070 | 0.063 |
| DefaultC | 0.070 | 0.060 | 0.046 | 0.055 | 0.035 | 0.049 |
| SomMotA | 0.053 | 0.067 | 0.054 | 0.020 | 0.030 | 0.062 |
| VisCent | 0.019 | 0.020 | 0.016 | 0.018 | 0.021 | 0.023 |
| Midbrain-DA | 0.005 | 0.004 | 0.025 | 0.020 | 0.025 | 0.020 |

Subcortical networks (Thalamus, Hippocampus/Amygdala, Basal Ganglia, Midbrain-Diencephalic) retain the highest residual; unimodal cortical networks (visual, somatomotor) are nearly fully captured. Relative to the lower-PC 0.90 truncation, residuals drop across the board; the small Midbrain-DA group (SNc/PBP/VTA) is captured by the additional PCs (ranks ~44-75), moving from the highest residual at 0.90 to the lowest at 0.95.

### LOSO residual stability (A5)

CV = std/mean of per-parcel residual fraction across leave-one-season-out folds. High-CV flag threshold: CV > 0.20.

| Subject | LOSO folds | Median parcel CV | Max parcel CV | N parcels with CV > 0.20 | Flagged parcel(s) |
|---|---|---|---|---|---|
| sub-01 | 6 | 0.025 | 0.182 | 0 | none |
| sub-02 | 6 | 0.024 | 0.276 | 2 | RH-NAC, RH-SNr |
| sub-03 | 6 | 0.022 | 0.277 | 2 | LH-SNc_PBP_VTA, RH-SNc_PBP_VTA |
| sub-04 | 4 | 0.026 | 0.233 | 2 | LH-SNc_PBP_VTA, LH-RN |
| sub-05 | 6 | 0.025 | 0.291 | 3 | LH-DefaultC_PHC, LH-SNc_PBP_VTA, RH-GPi |
| sub-06 | 6 | 0.029 | 0.264 | 1 | LH-SNr |

High-CV parcels at vt=0.95 are predominantly midbrain/basal-ganglia (SNc/PBP/VTA, SNr, NAc, RN, GPi): the added PCs capture their variance, but which specific PCs do so varies across folds, so their residual fraction is low while its across-fold stability is lower.

## Outputs

- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/A1_pca_loadings_heatmap.png
- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/A2_pc{1-5}_top_parcels.png
- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/A3_residual_variance.png
- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/A4_network_variance_per_pc.png
- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/A5_loso_residual_stability.png
- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/A7_residual_vs_threshold.png
- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/pca_loadings_flags.json
- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/pca_loadings_top_parcels.csv
- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/pca_residual_variance.csv
- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/pca_network_variance.csv
- output/03b_pca_loadings/atlas-4S156Parcels/{sub_id}/pca_residual_variance_loso.csv
