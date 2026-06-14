# Findings: 03b PCA Loadings Diagnostics

_Script: `script/03b_pca_loadings.py`. Tier: SUPP (Supp)._

_Diagnostic figures and QC flags for PCA models from script 03a; per-subject, n=6, atlas-4S156Parcels._

## Method (as run)

- Parcellation: atlas-4S156Parcels (156 parcels: 100 cortical Schaefer/Yeo-17, 56 subcortical)
- Input: per-subject `pca_model.pkl` and `n_pcs_lookup.json` from script 03a
- Residual variance (A3) and LOSO stability (A5) computed at default variance threshold vt=0.90
- k at vt=0.90 varies by subject (37-47 PCs retained; see Results)
- Motion artifact flags: SomMotA+SomMotB fraction threshold = 0.30 for PC1-3; all-subcortical fraction threshold = 0.25 for PC1-2
- LOSO stability flag threshold: coefficient of variation (CV) > 0.20 across season-leave-out folds
- sub-04 has 4 LOSO folds (seasons 1-4 only); all others have 6 folds (seasons 1-6)
- A6 cross-subject comparison was not run in production (no cross_subject output directory present)
- Per-subject results only; no group statistic

## Results

### PC count at vt=0.90 (default for A3/A5)

| Subject | k (PCs retained at vt=0.90) |
|---|---|
| sub-01 | 43 |
| sub-02 | 39 |
| sub-03 | 42 |
| sub-04 | 47 |
| sub-05 | 37 |
| sub-06 | 43 |

### Motion artifact flags (A4 / flags JSON)

SomMot fraction = squared loading energy from SomMotA+SomMotB as fraction of total PC energy. Subcortical fraction = all 6 subcortical groups combined. Flag threshold: SomMot > 0.30 in PC1-3; Subcortical > 0.25 in PC1-2.

| Subject | SomMot PC1 | SomMot PC2 | SomMot PC3 | Subcort PC1 | Subcort PC2 | any_flag | flagged PC(s) |
|---|---|---|---|---|---|---|---|
| sub-01 | 0.128 | 0.070 | 0.192 | 0.024 | 0.007 | false | none |
| sub-02 | 0.113 | 0.040 | 0.031 | 0.026 | 0.004 | false | none |
| sub-03 | 0.091 | 0.049 | 0.005 | 0.020 | 0.006 | false | none |
| sub-04 | 0.115 | 0.088 | 0.032 | 0.021 | 0.002 | false | none |
| sub-05 | 0.115 | 0.049 | 0.198 | 0.031 | 0.005 | false | none |
| sub-06 | 0.116 | 0.085 | 0.301 | 0.029 | 0.009 | true | PC3 (SomMot) |

### Per-parcel residual variance at vt=0.90 (A3)

Residual fraction = variance not captured by the retained k PCs, expressed as a fraction of total parcel variance.

| Subject | Median residual fraction | Mean residual fraction | Min | Max |
|---|---|---|---|---|
| sub-01 | 0.103 | 0.244 | 0.000 | 0.967 |
| sub-02 | 0.114 | 0.241 | 0.000 | 0.942 |
| sub-03 | 0.113 | 0.261 | 0.000 | 0.970 |
| sub-04 | 0.110 | 0.269 | 0.000 | 0.971 |
| sub-05 | 0.103 | 0.224 | 0.000 | 0.944 |
| sub-06 | 0.109 | 0.237 | 0.000 | 0.932 |

Parcels with the highest residual fraction are consistently subcortical BG structures (e.g., GPi, GPe). Top three parcels by residual fraction (across subjects): LH-GPi, RH-GPi, LH-GPe or RH-GPe (Basal Ganglia network).

### Per-network mean residual fraction at vt=0.90 (A3, cross-subject pattern)

Networks ordered by typical rank (highest residual at top). Values are mean residual fractions averaged across subject-level parcel estimates.

| Network | sub-01 | sub-02 | sub-03 | sub-04 | sub-05 | sub-06 |
|---|---|---|---|---|---|---|
| Midbrain-DA | 0.561 | 0.867 | 0.931 | 0.925 | 0.848 | 0.837 |
| Thalamus | 0.635 | 0.691 | 0.719 | 0.768 | 0.529 | 0.646 |
| Basal Ganglia | 0.523 | 0.555 | 0.582 | 0.594 | 0.537 | 0.539 |
| Hipp/Amyg | 0.463 | 0.324 | 0.433 | 0.428 | 0.365 | 0.398 |
| Midbrain-Diencephalic | 0.445 | 0.428 | 0.422 | 0.443 | 0.385 | 0.432 |
| LimbicB | 0.435 | 0.325 | 0.298 | 0.378 | 0.229 | 0.236 |
| LimbicA | 0.347 | 0.183 | 0.213 | 0.351 | 0.223 | 0.223 |
| Cerebellum | 0.303 | 0.182 | 0.367 | 0.332 | 0.249 | 0.197 |
| DefaultC | 0.177 | 0.166 | 0.113 | 0.137 | 0.122 | 0.129 |
| SalVentAttnA | 0.132 | 0.147 | 0.146 | 0.146 | 0.145 | 0.138 |
| SomMotA | 0.121 | 0.136 | 0.137 | 0.065 | 0.094 | 0.147 |
| VisCent | 0.045 | 0.043 | 0.050 | 0.041 | 0.047 | 0.045 |

### LOSO residual stability (A5)

CV = std/mean of per-parcel residual fraction across leave-one-season-out folds. High-CV flag threshold: CV > 0.20.

| Subject | LOSO folds | Median parcel CV | Max parcel CV | N parcels with CV > 0.20 | Flagged parcel(s) |
|---|---|---|---|---|---|
| sub-01 | 6 | 0.018 | 0.278 | 1 | LH-EXA (Hipp/Amyg) |
| sub-02 | 6 | 0.017 | 0.156 | 0 | none |
| sub-03 | 6 | 0.017 | 0.132 | 0 | none |
| sub-04 | 4 | 0.019 | 0.181 | 0 | none |
| sub-05 | 6 | 0.022 | 0.163 | 0 | none |
| sub-06 | 6 | 0.021 | 0.160 | 0 | none |

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
