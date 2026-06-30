# Findings: 04rb Split-Half Reliability

_Script: `script/04rb_split_half_reliability.py`. Tier: SUPP (Supp (reliability))._

_Compares structural invariants and recurrence profiles between two independently-fit HMMs trained on interleaved odd/even episode halves; per-subject, n=6._

## Method (as run)

- **Parcellation:** atlas-4S156Parcels (156 parcels: Schaefer 100 cortical + 56 subcortical composite)
- **Halves:** Half A = odd episodes, Half B = even episodes; each half fits its own combined HMM independently (from `04_combined_hdphmm --mode split_half`)
- **Active-state criterion:** FO > 0.01 of total TRs in that half
- **Hungarian matching:** Parcel-space state-mean Pearson correlations; minimum threshold r > 0.3 for a pair to count as well-matched
- **Recurrence scores:** Continuous (fraction of episodes with FO > 0.01); computed per half independently
- **Primary reliability metric:** Spearman rho of recurrence scores across Hungarian-matched pairs (r > 0.3 threshold); Spearman-Brown correction is not applied (matched/thresholded pairs violate the parallel-form assumption)
- **Sorted-distribution comparison:** KS test on independently sorted recurrence vectors (descriptive only; state identity not aligned)
- **Network profile consistency:** Fraction of matched pairs sharing the same dominant-network label (single argmax of |parcel mean|; coarse descriptor, not a biological equivalence test)
- **n_decoded_runs per half:** A = 100-154 runs, B = 94-138 runs, varying by subject (sub-04 has fewer total episodes)
- Per-subject analysis; no group statistic

## Results

### Per-half structural invariants

| Subject | K_active A / B | Recurrence mean A / B | Transition entropy A / B | Self-trans prob A / B | Median dwell A / B (TRs) |
|---------|---------------|-----------------------|--------------------------|----------------------|--------------------------|
| sub-01 | 38 / 38 | 0.807 / 0.804 | 0.370 / 0.371 | 0.735 / 0.727 | 4.0 / 4.0 |
| sub-02 | 39 / 39 | 0.764 / 0.783 | 0.382 / 0.387 | 0.714 / 0.699 | 3.0 / 3.0 |
| sub-03 | 37 / 39 | 0.777 / 0.780 | 0.403 / 0.398 | 0.679 / 0.677 | 3.0 / 3.0 |
| sub-04 | 39 / 39 | 0.719 / 0.736 | 0.401 / 0.415 | 0.666 / 0.665 | 3.0 / 3.0 |
| sub-05 | 37 / 41 | 0.799 / 0.752 | 0.375 / 0.381 | 0.708 / 0.702 | 3.0 / 3.0 |
| sub-06 | 36 / 36 | 0.834 / 0.824 | 0.433 / 0.430 | 0.572 / 0.556 | 2.0 / 2.0 |

### Hungarian-matched spatial correlations

| Subject | Total pairs | Pairs above r>0.3 | Mean r (all pairs) | Mean r (above threshold) |
|---------|-------------|-------------------|--------------------|--------------------------|
| sub-01 | 38 | 37 | 0.823 | 0.839 |
| sub-02 | 39 | 38 | 0.811 | 0.828 |
| sub-03 | 37 | 37 | 0.867 | 0.867 |
| sub-04 | 39 | 37 | 0.815 | 0.846 |
| sub-05 | 37 | 37 | 0.850 | 0.850 |
| sub-06 | 36 | 35 | 0.882 | 0.906 |

### Matched recurrence correlation (primary reliability metric)

| Subject | Matched pairs (above threshold) | Spearman rho | p-value |
|---------|--------------------------------|--------------|---------|
| sub-01 | 37 | 0.653 | 1.2e-05 |
| sub-02 | 38 | 0.679 | 3.0e-06 |
| sub-03 | 37 | 0.759 | <1e-06 |
| sub-04 | 37 | 0.822 | <1e-06 |
| sub-05 | 37 | 0.598 | 9.2e-05 |
| sub-06 | 35 | 0.761 | <1e-06 |

### Network profile consistency (dominant-network label match)

| Subject | Matched pairs | Network-matching pairs | Match fraction |
|---------|---------------|------------------------|---------------|
| sub-01 | 37 | 24 | 64.9% |
| sub-02 | 38 | 25 | 65.8% |
| sub-03 | 37 | 27 | 73.0% |
| sub-04 | 37 | 31 | 83.8% |
| sub-05 | 37 | 26 | 70.3% |
| sub-06 | 35 | 25 | 71.4% |

### Occupancy-stratified spatial correlation (top-10 vs rest)

States ranked by mean recurrence across halves; top-10 are highest-recurrence states.

| Subject | Top-10 mean r | Top-10 median r | Rest mean r | Rest median r |
|---------|--------------|-----------------|-------------|---------------|
| sub-01 | 0.798 | 0.819 | 0.854 | 0.940 |
| sub-02 | 0.839 | 0.826 | 0.823 | 0.887 |
| sub-03 | 0.873 | 0.894 | 0.865 | 0.948 |
| sub-04 | 0.802 | 0.851 | 0.862 | 0.927 |
| sub-05 | 0.796 | 0.855 | 0.871 | 0.935 |
| sub-06 | 0.905 | 0.934 | 0.907 | 0.976 |

### Sorted recurrence distribution comparison (KS test, descriptive)

KS test on independently sorted recurrence vectors per half; descriptive only (no state alignment).

| Subject | KS statistic | KS p-value | n active A | n active B |
|---------|-------------|------------|-----------|-----------|
| sub-01 | 0.158 | 0.738 | 38 | 38 |
| sub-02 | 0.231 | 0.252 | 39 | 39 |
| sub-03 | 0.141 | 0.783 | 37 | 39 |
| sub-04 | 0.128 | 0.911 | 39 | 39 |
| sub-05 | 0.213 | 0.284 | 37 | 41 |
| sub-06 | 0.111 | 0.982 | 36 | 36 |

## Outputs

- output/04rb_split_half/atlas-4S156Parcels/{sub_id}/half_invariants.json - per-half scalar metrics and recurrence vectors
- output/04rb_split_half/atlas-4S156Parcels/{sub_id}/split_half_reliability.json - scalar comparison, matched recurrence rho, network consistency, KS test
- output/04rb_split_half/atlas-4S156Parcels/{sub_id}/hungarian_matching.json - matched pairs with correlations and occupancy stratification
