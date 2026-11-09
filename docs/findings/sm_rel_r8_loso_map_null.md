# Findings: baselines for the LOSO matched-state map correlations (`sm_rel_r8_loso_map_null`)

## Question

`04ra` matches each leave-one-season-out fold's states to the primary model's states and reports the matched-pair Pearson r of parcel-space state means (pooled median 0.952–0.987 per participant). Each fold shares five of six seasons with the primary fit, and states share a common activation backbone. How high is r for pairs that are not matched?

## Method (as run)

- Gate: matched r recomputed from `state_means_parcel.npy` (primary and fold) with the `04rb` standardisation must equal the saved `matched_correlations` (`gate.matched_r_max_abs_delta`; 0 in all six).
- Derangement null: within each fold, the matched primary partners are permuted with no fixed points, and the pooled median r over all folds is recomputed; 10,000 draws (floor 0.0001).
- Best non-match: for each fold state, the highest r to any primary active state (recurrence > 0) other than its Hungarian partner.

## Results

| sub | pooled pairs | observed median r | null median | null 95th pct | null max | best non-match median | median gap | pairs matched ≤ best non-match |
|---|---|---|---|---|---|---|---|---|
| sub-01 | 251 | 0.966 | -0.024 | 0.023 | 0.089 | 0.687 | 0.260 | 36 |
| sub-02 | 248 | 0.952 | -0.031 | 0.016 | 0.085 | 0.700 | 0.202 | 48 |
| sub-03 | 263 | 0.987 | -0.031 | 0.006 | 0.072 | 0.703 | 0.260 | 32 |
| sub-04 | 167 | 0.963 | -0.033 | 0.013 | 0.084 | 0.622 | 0.320 | 25 |
| sub-05 | 256 | 0.970 | -0.034 | 0.005 | 0.052 | 0.712 | 0.248 | 45 |
| sub-06 | 250 | 0.979 | -0.031 | 0.002 | 0.049 | 0.718 | 0.240 | 42 |

Derangement p = 0.0001 (floor) in all six. "Median gap" is the median over pairs of matched r minus best non-match r.

## Interpretation (bounded)

A random pairing of matched states gives a median r near zero (null maximum ≤ 0.089), so the shared backbone does not by itself produce high map correlations: the median matched r of 0.95–0.99 reflects state-specific maps. The nearest competing primary state is much closer than the random pairing (median 0.62–0.72), so for a typical fold state the matched partner exceeds the runner-up by 0.20–0.32 in r.

In 25–48 pairs per participant (10–19% of pooled pairs), the Hungarian partner is not the fold state's highest-r primary state. The one-to-one assignment sends these fold states to a lower-r partner because their best primary state is taken by another fold state. They are the low-r tail of the matched distribution (their median r 0.48–0.66; 216 of 228 lie below the participant's 25th percentile of matched r), which is why the pooled mean (0.872–0.922) sits below the median.

## Caveats and audit notes

- Fold and primary fits share five of six seasons of data, so this checks state specificity, not independence. Split-half matching (`04rb`) is the comparison without shared data.
- Correlations are over 156 parcels of state means only; FC is compared separately (`fig_S04_reliability.py`, residual RV with its own derangement null).

## Connections

- `04ra_loso_struct_comp.py`, `04rb_split_half_reliability.py` (`hungarian_matching`), `fig_S04_reliability.py`.
