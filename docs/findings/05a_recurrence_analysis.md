# Findings: 05a Recurrence Analysis

_Script: `script/05a_recurrence_analysis.py`. Tier: MAIN (R1, Fig 1)._

_Scores each combined-HMM state's recurrence across episodes (a continuous gradient, not two discrete classes) via fractional occupancy, a recurrence score, a season-specificity index, and a season-label permutation test with FDR correction. Per-subject; n=6 (no group statistic)._

## Method (as run)
- Parcellation atlas-4S156Parcels; PCA variance threshold vt=0.95; production combined-HMM config.
- A state is "active" in a run when its fractional occupancy exceeds 0.02 (9.4 TRs, ~14.0-14.1 s at TR=1.49 s).
- Recurrence score = fraction of episodes in which a state is active; a continuous score, not a categorical grouping. Downstream eligibility is set by 05e state flags, not by this score.
- Season-specificity index = range of per-season recurrence (0 = invariant, 1 = specific).
- Permutation test: 5000 season-label shuffles per state; per-state p-value estimated as (count + 1) / (n_permutations + 1); Benjamini-Hochberg FDR correction across states.
- Inputs: 04 decoded_states.pkl, final_results.json. Per-subject only; no group/aggregate statistic.

## Results

### Repertoire and activity threshold

| Subject | States | Active states | Runs | FO threshold (TRs / s) |
|---------|--------|---------------|------|------------------------|
| sub-01  | 50     | 46            | 292  | 9.4 / 14.1             |
| sub-02  | 50     | 46            | 292  | 9.4 / 14.0             |
| sub-03  | 50     | 44            | 291  | 9.4 / 14.1             |
| sub-04  | 50     | 44            | 194  | 9.4 / 14.1             |
| sub-05  | 50     | 47            | 289  | 9.4 / 14.1             |
| sub-06  | 50     | 42            | 292  | 9.4 / 14.1             |

### Recurrence score distribution (active states)

| Subject | Mean  | Median | Min   | Max   |
|---------|-------|--------|-------|-------|
| sub-01  | 0.462 | 0.459  | 0.003 | 0.849 |
| sub-02  | 0.465 | 0.502  | 0.003 | 0.836 |
| sub-03  | 0.486 | 0.514  | 0.072 | 0.852 |
| sub-04  | 0.477 | 0.466  | 0.067 | 0.902 |
| sub-05  | 0.443 | 0.453  | 0.003 | 0.817 |
| sub-06  | 0.494 | 0.507  | 0.010 | 0.925 |

### Season-specificity index (active states)

| Subject | Mean  | Min   | Max   |
|---------|-------|-------|-------|
| sub-01  | 0.239 | 0.020 | 0.427 |
| sub-02  | 0.266 | 0.021 | 0.660 |
| sub-03  | 0.246 | 0.120 | 0.773 |
| sub-04  | 0.170 | 0.021 | 0.492 |
| sub-05  | 0.314 | 0.020 | 0.632 |
| sub-06  | 0.290 | 0.040 | 0.652 |

### Permutation test and FDR

| Subject | Permutations | Significant (uncorrected) | Significant (FDR) |
|---------|-------------|--------------------------|-------------------|
| sub-01  | 5000        | 17                       | 12                |
| sub-02  | 5000        | 20                       | 11                |
| sub-03  | 5000        | 19                       | 8                 |
| sub-04  | 5000        | 8                        | 3                 |
| sub-05  | 5000        | 34                       | 34                |
| sub-06  | 5000        | 23                       | 21                |

## Outputs
- output/05a_recurrence_analysis/atlas-4S156Parcels/sub-*/vt0.95/{recurrence_scores.npy, specificity_index.npy, fractional_occupancy.pkl, permutation_pvalues.json, recurrence_summary.json}
- Figure: fig_F1_recurrence_gradient.py (R1)
