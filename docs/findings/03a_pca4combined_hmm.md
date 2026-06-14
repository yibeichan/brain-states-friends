# Findings: 03a PCA Preparation for Combined HDP-HMM

_Script: `script/03a_pca4combined_hmm.py`. Tier: MAIN (Methods)._

_Season-stratified PCA fit and train/valid/test split creation per subject (n=6); leakage-free projection for the combined cross-season HMM._

## Method (as run)

- Parcellation: atlas-4S156Parcels (156 parcels; background column dropped before PCA)
- Per-subject, no group statistic; sub-04 has seasons 1-4 only (seasons 5-6 excluded)
- PCA fitted on concatenated primary training TRs only (no leakage); valid/test runs projected through training mean and components
- Variance thresholds swept: 0.80, 0.85, 0.90, 0.95, 0.99; `n_pcs_max` = PCs needed for vt=0.99, stored per run; downstream scripts slice to the desired threshold
- Primary split: season-stratified 70/15/15 at episode level (multi-part episodes kept together); seed base 100
- LOSO splits: 6 folds (4 for sub-04); each fold holds out one full season as test; remaining 5 seasons split 80/20 train/valid
- Split-half splits: interleaved odd/even episodes per season; 80/20 train/valid within each half

## Results

### Primary split run counts and training set size

| Subject | Seasons | Total runs | Train runs | Valid runs | Test runs | Train TRs |
|---|---|---|---|---|---|---|
| sub-01 | 6 | 292 | 204 | 44 | 44 | 96,427 |
| sub-02 | 6 | 292 | 204 | 44 | 44 | 96,004 |
| sub-03 | 6 | 291 | 203 | 44 | 44 | 95,971 |
| sub-04 | 4 | 194 | 134 | 30 | 30 | 63,149 |
| sub-05 | 6 | 289 | 201 | 44 | 44 | 94,943 |
| sub-06 | 6 | 292 | 204 | 44 | 44 | 96,427 |

### PCA dimensionality by variance threshold (primary PCA)

Number of PCs required to reach each cumulative variance threshold.

| Subject | vt=0.80 | vt=0.85 | vt=0.90 | vt=0.95 | vt=0.99 | n_pcs_max |
|---|---|---|---|---|---|---|
| sub-01 | 19 | 28 | 43 | 75 | 124 | 124 |
| sub-02 | 16 | 25 | 39 | 72 | 123 | 123 |
| sub-03 | 20 | 28 | 42 | 72 | 122 | 122 |
| sub-04 | 24 | 33 | 47 | 77 | 124 | 124 |
| sub-05 | 16 | 24 | 37 | 67 | 121 | 121 |
| sub-06 | 20 | 28 | 43 | 74 | 123 | 123 |

### PCA effective dimensionality (participation ratio)

Participation ratio PR = (sum of eigenvalues)^2 / sum of squared eigenvalues; measures the effective number of dimensions carrying the signal.

| Subject | Participation ratio |
|---|---|
| sub-01 | 5.12 |
| sub-02 | 4.42 |
| sub-03 | 9.91 |
| sub-04 | 9.87 |
| sub-05 | 4.52 |
| sub-06 | 6.61 |

### LOSO fold run counts (primary season held out as test)

| Subject | LOSO folds | Train runs (range) | Valid runs (range) | Test runs (range) |
|---|---|---|---|---|
| sub-01 | 6 | 190-194 | 50-54 | 48-50 |
| sub-02 | 6 | 190-194 | 50-54 | 48-50 |
| sub-03 | 6 | 191-193 | 48-53 | 47-50 |
| sub-04 | 4 | 114-116 | 30-32 | 48-50 |
| sub-05 | 6 | 191-192 | 48-52 | 45-50 |
| sub-06 | 6 | 190-194 | 50-54 | 48-50 |

### Split-half run counts

| Subject | Half A train | Half A valid | Half A total | Half B train | Half B valid | Half B total |
|---|---|---|---|---|---|---|
| sub-01 | 128 | 26 | 154 | 114 | 24 | 138 |
| sub-02 | 128 | 26 | 154 | 114 | 24 | 138 |
| sub-03 | 128 | 26 | 154 | 113 | 24 | 137 |
| sub-04 | 82 | 18 | 100 | 78 | 16 | 94 |
| sub-05 | 124 | 26 | 150 | 115 | 24 | 139 |
| sub-06 | 128 | 26 | 154 | 114 | 24 | 138 |

## Outputs

- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/pca_model.pkl
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/n_pcs_lookup.json
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/pca_variance_summary.json
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/summary.json
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/splits/primary.json
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/splits/loso_season_{1-6}.json
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/splits/split_half_{A,B}.json
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/projected/train/{run_id}.npy (shape: n_TRs x n_pcs_max)
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/projected/valid/{run_id}.npy
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/projected/test/{run_id}.npy
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/loso/season_{1-6}/pca_model.pkl
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/loso/season_{1-6}/pca_variance_summary.json
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/loso/season_{1-6}/projected/test/{run_id}.npy
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/pca_scree_primary.png
- output/03a_pca4combined_hmm/atlas-4S156Parcels/sub-*/pca_cumvar_comparison.png
