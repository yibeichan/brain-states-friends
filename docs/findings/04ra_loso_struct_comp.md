# Findings: 04ra LOSO Structural Comparison

_Script: `script/04ra_loso_struct_comp.py`. Tier: SUPP (Supp (reliability))._

_Compares structural invariants of leave-one-season-out HMM refits to the primary model, per subject (n=6; sub-04 has 4 folds)._

## Method (as run)

- Parcellation: atlas-4S156Parcels (156 parcels: Schaefer-100 cortical + 56 subcortical composite)
- Variance threshold: vt=0.95
- Active-state threshold: fractional occupancy >1% across all decoded TRs
- Recurrence FO threshold: 0.01
- Hungarian matching threshold: r >= 0.3 (Pearson correlation in parcel space)
- Noise floor: 10 random seeds of the primary model
- Folds: 6 held-out seasons per subject (sub-04: 4, seasons 1-4 only)
- Per-subject analysis; no group statistic
- Recurrence treated as continuous score
- Pairwise KS tests between fold recurrence distributions corrected with BH-FDR

## Results

### Per-fold K_active

| Subject | Folds | Per-fold K_active | Mean | Std | CV |
|---------|-------|-------------------|------|-----|-----|
| sub-01 | 6 | 37, 38, 39, 39, 39, 40 | 38.7 | 0.94 | 2.4% |
| sub-02 | 6 | 35, 36, 37, 39, 39, 40 | 37.7 | 1.80 | 4.8% |
| sub-03 | 6 | 37, 37, 38, 39, 41, 42 | 39.0 | 1.91 | 4.9% |
| sub-04 | 4 | 36, 36, 36, 37 | 36.2 | 0.43 | 1.2% |
| sub-05 | 6 | 38, 38, 38, 40, 41, 42 | 39.5 | 1.61 | 4.1% |
| sub-06 | 6 | 36, 37, 38, 39, 39, 41 | 38.3 | 1.60 | 4.2% |

### Cross-fold scalar invariants

| Subject | Trans. entropy range | Self-trans. prob range | Dwell median (TR) | Recurrence mean range |
|---------|---------------------|----------------------|-------------------|-----------------------|
| sub-01 | [0.370, 0.380] | [0.719, 0.727] | 3-4 | [0.768, 0.823] |
| sub-02 | [0.378, 0.392] | [0.686, 0.697] | 3 | [0.762, 0.798] |
| sub-03 | [0.401, 0.413] | [0.655, 0.675] | 3 | [0.753, 0.820] |
| sub-04 | [0.406, 0.421] | [0.649, 0.662] | 3 | [0.729, 0.774] |
| sub-05 | [0.374, 0.385] | [0.683, 0.704] | 3 | [0.746, 0.828] |
| sub-06 | [0.411, 0.442] | [0.535, 0.585] | 2 | [0.745, 0.831] |

### Recurrence distribution KS tests across folds

Pairwise two-sample KS tests compare the sorted recurrence-score distributions between each pair of LOSO folds. BH-FDR applied within each subject.

| Subject | N pairs | Mean KS statistic | Median p (uncorrected) | N sig at 0.05 (uncorr.) | N sig at 0.05 (BH-FDR) |
|---------|---------|-------------------|------------------------|--------------------------|-------------------------|
| sub-01 | 15 | 0.216 | 0.230 | 3 | 0 |
| sub-02 | 15 | 0.170 | 0.628 | 0 | 0 |
| sub-03 | 15 | 0.201 | 0.321 | 1 | 0 |
| sub-04 | 6 | 0.170 | 0.650 | 0 | 0 |
| sub-05 | 15 | 0.226 | 0.222 | 1 | 0 |
| sub-06 | 15 | 0.172 | 0.658 | 0 | 0 |

### Hungarian matching: fold states to primary

Active states from each fold matched to primary-model active states via Hungarian algorithm (cost = 1 - Pearson r in parcel space).

| Subject | Primary K_active | Mean fold K_active matched | Mean r (across folds) | Well-matched fraction (r >= 0.3) |
|---------|-----------------|---------------------------|----------------------|---------------------------------|
| sub-01 | 42 | 38.7 | 0.912 | 99.1% |
| sub-02 | 42 | 37.7 | 0.889 | 97.3% |
| sub-03 | 42 | 39.0 | 0.910 | 99.2% |
| sub-04 | 41 | 36.2 | 0.910 | 98.6% |
| sub-05 | 41 | 39.5 | 0.883 | 97.5% |
| sub-06 | 37 | 38.3 | 0.884 | 94.6% |

### Initialization sensitivity (seed noise floor)

Seed-to-seed variability of 10 primary-model seeds (same data, different EM starting points), compared to fold-to-fold variability.

| Subject | Seeds loaded | Seed K_active mean | Seed K_active std | Fold K_active std | Fold/seed std ratio |
|---------|-------------|-------------------|-------------------|-------------------|---------------------|
| sub-01 | 10 | 32.6 | 4.72 | 0.94 | 0.20 |
| sub-02 | 10 | 34.5 | 4.92 | 1.80 | 0.36 |
| sub-03 | 10 | 35.8 | 3.60 | 1.91 | 0.53 |
| sub-04 | 10 | 35.1 | 5.39 | 0.43 | 0.08 |
| sub-05 | 10 | 34.5 | 4.70 | 1.61 | 0.34 |
| sub-06 | 10 | 32.1 | 4.68 | 1.60 | 0.34 |

Seed transition-entropy std (0.004-0.006) and self-transition-prob std (0.008-0.022) bound the corresponding fold ranges for all subjects.

## Outputs

- output/04ra_loso_struct_comp/atlas-4S156Parcels/{sub_id}/fold_invariants.json
- output/04ra_loso_struct_comp/atlas-4S156Parcels/{sub_id}/cross_fold_consistency.json
- output/04ra_loso_struct_comp/atlas-4S156Parcels/{sub_id}/hungarian_matching.json
- output/04ra_loso_struct_comp/atlas-4S156Parcels/{sub_id}/noise_floor.json
