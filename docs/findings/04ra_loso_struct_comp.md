# Findings: 04ra LOSO Structural Comparison

_Script: `script/04ra_loso_struct_comp.py`. Tier: SUPP (Supp (reliability))._

_Compares structural invariants of leave-one-season-out HMM refits to the primary model, per subject (n=6; sub-04 has 4 folds)._

## Method (as run)

- Parcellation: atlas-4S156Parcels (156 parcels: Schaefer-100 cortical + 56 subcortical composite)
- Variance threshold: vt=0.95
- Active-state threshold: FO > 0.02 in at least one run (`ACTIVE_FO_THRESHOLD`, the same rule as 05a)
- K_used: posterior usage > 0.01 in the fitted model (fold models and seeds alike)
- Recurrence FO threshold: 0.02
- Hungarian matching threshold: r >= 0.3 (Pearson correlation in parcel space)
- Noise floor: 10 random seeds of the primary model
- Folds: 6 held-out seasons per subject (sub-04: 4, seasons 1-4 only)
- Per-subject analysis; no group statistic
- Recurrence treated as continuous score
- Pairwise KS tests between fold recurrence distributions corrected with BH-FDR

## Results

K_active counts decoded states with FO > 0.02 in at least one run of the fold (the rule used for recurrence and matching). K_used counts states with posterior usage > 0.01 in the fold model, the same quantity as the seed band; transition entropy and self-transition probability are computed over the K_used states.

### Per-fold K_active and K_used

| Subject | Folds | Per-fold K_active | Per-fold K_used | K_used mean | K_used std | K_used CV |
|---------|-------|-------------------|-----------------|------|-----|-----|
| sub-01 | 6 | 39, 41, 41, 42, 44, 44 | 37, 38, 39, 39, 40, 41 | 39.0 | 1.29 | 3.3% |
| sub-02 | 6 | 39, 41, 41, 42, 42, 43 | 37, 38, 39, 39, 39, 40 | 38.7 | 0.94 | 2.4% |
| sub-03 | 6 | 43, 44, 44, 45, 45, 46 | 38, 39, 40, 41, 42, 42 | 40.3 | 1.49 | 3.7% |
| sub-04 | 4 | 41, 42, 42, 42 | 36, 38, 39, 40 | 38.2 | 1.48 | 3.9% |
| sub-05 | 6 | 41, 42, 42, 43, 44, 44 | 38, 39, 41, 42, 42, 43 | 40.8 | 1.77 | 4.3% |
| sub-06 | 6 | 41, 41, 42, 42, 43, 43 | 38, 38, 39, 39, 39, 40 | 38.8 | 0.69 | 1.8% |

### Cross-fold scalar invariants

| Subject | Trans. entropy range | Self-trans. prob range | Dwell median (TR) | Recurrence mean range |
|---------|---------------------|----------------------|-------------------|-----------------------|
| sub-01 | [0.371, 0.380] | [0.720, 0.721] | 3-4 | [0.491, 0.542] |
| sub-02 | [0.382, 0.400] | [0.680, 0.697] | 3 | [0.483, 0.535] |
| sub-03 | [0.405, 0.412] | [0.658, 0.675] | 3 | [0.452, 0.503] |
| sub-04 | [0.408, 0.421] | [0.651, 0.663] | 3 | [0.473, 0.488] |
| sub-05 | [0.376, 0.389] | [0.683, 0.704] | 3 | [0.463, 0.522] |
| sub-06 | [0.416, 0.438] | [0.549, 0.589] | 2 | [0.465, 0.550] |

### Recurrence distribution KS tests across folds

Pairwise two-sample KS tests compare the sorted recurrence-score distributions between each pair of LOSO folds. BH-FDR applied within each subject.

| Subject | N pairs | Mean KS statistic | Median p (uncorrected) | N sig at 0.05 (uncorr.) | N sig at 0.05 (BH-FDR) |
|---------|---------|-------------------|------------------------|--------------------------|-------------------------|
| sub-01 | 15 | 0.149 | 0.586 | 0 | 0 |
| sub-02 | 15 | 0.166 | 0.526 | 0 | 0 |
| sub-03 | 15 | 0.159 | 0.545 | 0 | 0 |
| sub-04 | 6 | 0.159 | 0.692 | 0 | 0 |
| sub-05 | 15 | 0.188 | 0.388 | 1 | 0 |
| sub-06 | 15 | 0.187 | 0.383 | 0 | 0 |

### Hungarian matching: fold states to primary

Active states (FO > 0.02 in at least one run) from each fold matched to primary-model active states via Hungarian algorithm (cost = 1 - Pearson r in parcel space).

| Subject | Primary K_active | Mean fold K_active matched | Mean r (across folds) | Well-matched fraction (r >= 0.3) |
|---------|-----------------|---------------------------|----------------------|---------------------------------|
| sub-01 | 46 | 41.8 | 0.906 | 99.2% |
| sub-02 | 46 | 41.3 | 0.872 | 96.0% |
| sub-03 | 44 | 43.8 | 0.922 | 99.6% |
| sub-04 | 44 | 41.8 | 0.882 | 96.4% |
| sub-05 | 47 | 42.7 | 0.892 | 98.5% |
| sub-06 | 42 | 41.7 | 0.882 | 94.4% |

### Initialization sensitivity (seed noise floor)

Seed-to-seed variability of 10 primary-model seeds (same data, different EM starting points), compared to fold-to-fold variability. Both counts are K_used.

| Subject | Seeds loaded | Seed K_used mean | Seed K_used std | Fold K_used std | Fold/seed std ratio |
|---------|-------------|-------------------|-------------------|-------------------|---------------------|
| sub-01 | 10 | 32.6 | 4.72 | 1.29 | 0.27 |
| sub-02 | 10 | 34.5 | 4.92 | 0.94 | 0.19 |
| sub-03 | 10 | 35.8 | 3.60 | 1.49 | 0.41 |
| sub-04 | 10 | 35.1 | 5.39 | 1.48 | 0.27 |
| sub-05 | 10 | 34.5 | 4.70 | 1.77 | 0.38 |
| sub-06 | 10 | 32.1 | 4.68 | 0.69 | 0.15 |

Across seeds, transition-entropy std is 0.004-0.006 and self-transition-prob std is 0.007-0.022.

## Outputs

- output/04ra_loso_struct_comp/atlas-4S156Parcels/{sub_id}/fold_invariants.json
- output/04ra_loso_struct_comp/atlas-4S156Parcels/{sub_id}/cross_fold_consistency.json
- output/04ra_loso_struct_comp/atlas-4S156Parcels/{sub_id}/hungarian_matching.json
- output/04ra_loso_struct_comp/atlas-4S156Parcels/{sub_id}/noise_floor.json
