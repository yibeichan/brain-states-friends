# Findings: pp_03 Project Petit Prince PCA

_Script: `script/pp_03_project_pp_pca.py`. Tier: CROSS-STIM (R5, Fig 5)._

_Projects Petit Prince (audiobook, French + English) parcel time series through the Friends-trained PCA; computes PCA transfer diagnostics per subject (n=5; sub-04 absent)._

## Method (as run)

- Parcellation: atlas-4S156Parcels (156 parcels: 100 cortical Schaefer + 56 subcortical)
- Variance threshold: vt=0.95 (67-75 PCs per subject)
- PCA model source: `04_combined_hdphmm/{parc}/{sub}/final/vt0.95/pca_model.pkl` (Friends training data)
- Input time series: `02_parcel_ts_avg/{parc}/{sub}/`, files matching `*task-lppFR*` or `*task-lppEN*`
- Subjects: sub-01, sub-02, sub-03, sub-05, sub-06 (no sub-04; Petit Prince data unavailable)
- sub-06 has 16 runs (7 FR + 9 EN); all other subjects have 18 runs (9 FR + 9 EN)
- R2 denominator: `SS_total = sum((X - pca.mean_)^2)` using Friends training mean
- Mean-corrected R2 decomposes transfer loss into mean shift vs covariance mismatch; after voxel-wise z-scoring in step 00, parcel mean shifts are machine-epsilon (~10^-11), so mean-corrected R2 equals standard R2 for all subjects
- Per-network R2 uses 13-group partition: Yeo-7 cortical networks + BG, Midbrain-DA, Midbrain-Diencephalic, Thalamus, Hipp/Amyg, Cerebellum
- Per-subject analysis; no group statistic

## Results

### PCA Transfer: Overall

| Subject | n_pcs | Friends R2 | PP R2 | PP R2 (mc) | Transfer gap | PP runs | Total TRs |
|---------|-------|-----------|-------|------------|-------------|---------|-----------|
| sub-01  | 75    | 0.9505    | 0.9491 | 0.9491    | 0.0014      | 18      | 7,798     |
| sub-02  | 72    | 0.9504    | 0.9472 | 0.9472    | 0.0032      | 18      | 7,798     |
| sub-03  | 72    | 0.9504    | 0.9549 | 0.9549    | -0.0045     | 18      | 7,798     |
| sub-05  | 67    | 0.9509    | 0.9516 | 0.9516    | -0.0007     | 18      | 7,798     |
| sub-06  | 74    | 0.9504    | 0.9496 | 0.9496    | 0.0008      | 16      | 6,941     |

Transfer gap = Friends R2 minus PP R2. Negative values indicate PP variance captured at a higher rate than Friends training variance. No subject has `flag_low_variance` (threshold 0.70) set.

### PCA Transfer: Per Language

| Subject | French R2 | French TRs | English R2 | English TRs |
|---------|-----------|------------|------------|-------------|
| sub-01  | 0.9488    | 3,990      | 0.9494     | 3,808       |
| sub-02  | 0.9397    | 3,990      | 0.9536     | 3,808       |
| sub-03  | 0.9583    | 3,990      | 0.9504     | 3,808       |
| sub-05  | 0.9498    | 3,990      | 0.9533     | 3,808       |
| sub-06  | 0.9438    | 3,133      | 0.9537     | 3,808       |

sub-06 French TRs reduced (7 runs instead of 9). No consistent French vs English advantage across subjects.

### Per-Network R2: Cortical Networks

| Network (n parcels) | sub-01 | sub-02 | sub-03 | sub-05 | sub-06 | Range |
|--------------------|--------|--------|--------|--------|--------|-------|
| Vis (17)           | 0.970  | 0.966  | 0.980  | 0.975  | 0.967  | 0.966-0.980 |
| SomMot (14)        | 0.970  | 0.956  | 0.974  | 0.971  | 0.968  | 0.956-0.974 |
| DorsAttn (15)      | 0.959  | 0.948  | 0.968  | 0.963  | 0.960  | 0.948-0.968 |
| SalVentAttn (12)   | 0.942  | 0.939  | 0.949  | 0.949  | 0.954  | 0.939-0.954 |
| Limbic (5)         | 0.664  | 0.814  | 0.841  | 0.854  | 0.856  | 0.664-0.856 |
| Cont (13)          | 0.967  | 0.971  | 0.974  | 0.971  | 0.968  | 0.967-0.974 |
| Default (24)       | 0.956  | 0.961  | 0.961  | 0.955  | 0.960  | 0.955-0.961 |

### Per-Network R2: Subcortical Networks

| Network (n parcels)      | sub-01 | sub-02 | sub-03 | sub-05 | sub-06 | Range |
|--------------------------|--------|--------|--------|--------|--------|-------|
| Midbrain-DA (2)          | 0.995  | 0.996  | 0.976  | 0.968  | 0.980  | 0.968-0.996 |
| Midbrain-Diencephalic (8)| 0.956  | 0.962  | 0.902  | 0.913  | 0.934  | 0.902-0.962 |
| BG (16)                  | 0.919  | 0.948  | 0.868  | 0.871  | 0.889  | 0.868-0.948 |
| Hipp/Amyg (6)            | 0.861  | 0.891  | 0.853  | 0.872  | 0.860  | 0.853-0.891 |
| Cerebellum (10)          | 0.801  | 0.766  | 0.877  | 0.695  | 0.907  | 0.695-0.907 |
| Thalamus (14)            | 0.440  | 0.414  | 0.493  | 0.522  | 0.496  | 0.414-0.522 |

Midbrain-DA n=2 parcels (SNc_PBP_VTA bilateral pair). Thalamus lowest across all subjects (R2 0.41-0.52).

### Parcel Mean Shift

| Subject | Mean abs shift | Max abs shift | Parcels > 1 SD | flag_mean_shift |
|---------|---------------|---------------|----------------|-----------------|
| sub-01  | 8.6e-12       | 1.1e-10       | 0              | False           |
| sub-02  | 2.9e-11       | 2.2e-10       | 0              | False           |
| sub-03  | 3.4e-11       | 8.7e-10       | 0              | False           |
| sub-05  | 2.3e-11       | 1.4e-10       | 0              | False           |
| sub-06  | 2.5e-11       | 1.7e-10       | 0              | False           |

Mean shift is machine-epsilon for all subjects; voxel-wise z-scoring in step 00 eliminates baseline mean differences across stimuli.

## Outputs

- `output/pp_03_projected/atlas-4S156Parcels/sub-*/vt0.95/{run_id}.npy`: PCA-projected PP run, shape (n_trs, n_pcs), one file per run
- `output/pp_03_projected/atlas-4S156Parcels/sub-*/vt0.95/pp_run_ids.json`: run IDs grouped by language type
- `output/pp_03_projected/atlas-4S156Parcels/sub-*/vt0.95/pca_transfer_diagnostic.json`: overall, per-type, and per-network R2 with transfer gap and mean shift flags
- `output/pp_03_projected/atlas-4S156Parcels/sub-*/vt0.95/pca_transfer_diagnostic.png`: diagnostic figure (brain surface + network bar chart)
