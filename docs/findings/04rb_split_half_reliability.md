# Findings: 04rb Split-Half Reliability

_Script: `script/04rb_split_half_reliability.py`. Tier: SUPP (Supp (reliability))._

_Compares structural invariants and recurrence profiles between two independently-fit HMMs trained on interleaved odd/even episode halves; per-subject, n=6._

## Method (as run)

- **Parcellation:** atlas-4S156Parcels (156 parcels: Schaefer 100 cortical + 56 subcortical composite)
- **Halves:** Half A = odd episodes, Half B = even episodes; each half fits its own combined HMM independently (from `04_combined_hdphmm --mode split_half`)
- **Active-state criterion:** FO > 0.02 in at least one run of that half (`ACTIVE_FO_THRESHOLD`, the same rule as 05a); Hungarian matching and recurrence use these states
- **Hungarian matching:** Parcel-space state-mean Pearson correlations; minimum threshold r > 0.3 for a pair to count as well-matched
- **Recurrence scores:** Continuous (fraction of runs with FO > 0.02); computed per half independently
- **K_used:** states with posterior usage > 0.01 in the half's fitted model; transition entropy and self-transition probability are computed over these
- **Primary reliability metric:** Spearman rho of recurrence scores across Hungarian-matched pairs (r > 0.3 threshold); Spearman-Brown correction is not applied (matched/thresholded pairs violate the parallel-form assumption)
- **Sorted-distribution comparison:** KS test on independently sorted recurrence vectors (descriptive only; state identity not aligned)
- **Network profile consistency:** Fraction of matched pairs sharing the same dominant-network label (single argmax of |parcel mean|; coarse descriptor, not a biological equivalence test)
- **n_decoded_runs per half:** A = 100-154 runs, B = 94-138 runs, varying by subject (sub-04 has fewer total episodes)
- Per-subject analysis; no group statistic

## Results

### Per-half structural invariants

K_active counts decoded states with FO > 0.02 in at least one run of that half; K_used counts states with posterior usage > 0.01 in the half's fitted model. Transition entropy and self-transition probability are computed over the K_used states.

| Subject | K_active A / B | K_used A / B | Recurrence mean A / B | Transition entropy A / B | Self-trans prob A / B | Median dwell A / B (TRs) |
|---------|---------------|-------------|-----------------------|--------------------------|----------------------|--------------------------|
| sub-01 | 43 / 44 | 38 / 38 | 0.508 / 0.489 | 0.370 / 0.371 | 0.735 / 0.727 | 4.0 / 4.0 |
| sub-02 | 45 / 44 | 39 / 39 | 0.458 / 0.483 | 0.382 / 0.387 | 0.714 / 0.699 | 3.0 / 3.0 |
| sub-03 | 46 / 44 | 37 / 38 | 0.455 / 0.477 | 0.403 / 0.397 | 0.679 / 0.677 | 3.0 / 3.0 |
| sub-04 | 44 / 42 | 39 / 39 | 0.458 / 0.490 | 0.401 / 0.415 | 0.666 / 0.665 | 3.0 / 3.0 |
| sub-05 | 44 / 45 | 37 / 41 | 0.472 / 0.461 | 0.375 / 0.381 | 0.708 / 0.702 | 3.0 / 3.0 |
| sub-06 | 43 / 44 | 36 / 36 | 0.502 / 0.490 | 0.433 / 0.430 | 0.572 / 0.556 | 2.0 / 2.0 |

### Hungarian-matched spatial correlations

| Subject | Total pairs | Pairs above r>0.3 | Mean r (all pairs) | Mean r (above threshold) |
|---------|-------------|-------------------|--------------------|--------------------------|
| sub-01 | 43 | 43 | 0.830 | 0.830 |
| sub-02 | 44 | 44 | 0.805 | 0.805 |
| sub-03 | 44 | 44 | 0.857 | 0.857 |
| sub-04 | 42 | 41 | 0.812 | 0.829 |
| sub-05 | 44 | 41 | 0.793 | 0.834 |
| sub-06 | 43 | 40 | 0.852 | 0.900 |

### Matched recurrence correlation (primary reliability metric)

| Subject | Matched pairs (above threshold) | Spearman rho | p-value |
|---------|--------------------------------|--------------|---------|
| sub-01 | 43 | 0.680 | 1.0e-06 |
| sub-02 | 44 | 0.599 | 1.7e-05 |
| sub-03 | 44 | 0.684 | <1e-06 |
| sub-04 | 41 | 0.597 | 3.8e-05 |
| sub-05 | 41 | 0.482 | 1.4e-03 |
| sub-06 | 40 | 0.839 | <1e-06 |

### Network profile consistency (dominant-network label match)

| Subject | Matched pairs | Network-matching pairs | Match fraction |
|---------|---------------|------------------------|---------------|
| sub-01 | 43 | 27 | 62.8% |
| sub-02 | 44 | 29 | 65.9% |
| sub-03 | 44 | 31 | 70.5% |
| sub-04 | 41 | 31 | 75.6% |
| sub-05 | 41 | 27 | 65.8% |
| sub-06 | 40 | 30 | 75.0% |

### Occupancy-stratified spatial correlation (top-10 vs rest)

States ranked by mean recurrence across halves; top-10 are highest-recurrence states.

| Subject | Top-10 mean r | Top-10 median r | Rest mean r | Rest median r |
|---------|--------------|-----------------|-------------|---------------|
| sub-01 | 0.795 | 0.835 | 0.840 | 0.941 |
| sub-02 | 0.827 | 0.826 | 0.799 | 0.866 |
| sub-03 | 0.839 | 0.859 | 0.862 | 0.921 |
| sub-04 | 0.814 | 0.851 | 0.833 | 0.916 |
| sub-05 | 0.841 | 0.901 | 0.832 | 0.934 |
| sub-06 | 0.944 | 0.943 | 0.885 | 0.972 |

### Sorted recurrence distribution comparison (KS test, descriptive)

KS test on independently sorted recurrence vectors per half; descriptive only (no state alignment).

| Subject | KS statistic | KS p-value | n active A | n active B |
|---------|-------------|------------|-----------|-----------|
| sub-01 | 0.113 | 0.891 | 43 | 44 |
| sub-02 | 0.191 | 0.334 | 45 | 44 |
| sub-03 | 0.193 | 0.314 | 46 | 44 |
| sub-04 | 0.153 | 0.625 | 44 | 42 |
| sub-05 | 0.150 | 0.615 | 44 | 45 |
| sub-06 | 0.104 | 0.941 | 43 | 44 |

## Outputs

- output/04rb_split_half/atlas-4S156Parcels/{sub_id}/half_invariants.json - per-half scalar metrics and recurrence vectors
- output/04rb_split_half/atlas-4S156Parcels/{sub_id}/split_half_reliability.json - scalar comparison, matched recurrence rho, network consistency, KS test
- output/04rb_split_half/atlas-4S156Parcels/{sub_id}/hungarian_matching.json - matched pairs with correlations and occupancy stratification
