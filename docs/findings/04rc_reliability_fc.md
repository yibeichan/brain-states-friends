# Findings: 04rc Reliability FC

_Script: `script/04rc_reliability_fc.py`. Tier: SUPP (Supp (reliability))._

_Computes state-conditioned Ledoit-Wolf empirical FC for every LOSO fold and split-half, saving (K x 156 x 156) correlation matrices alongside decoded-state outputs; per-subject, n=6._

## Method (as run)

- **Parcellation:** atlas-4S156Parcels (156 parcels: Schaefer 100 cortical + 56 subcortical composite)
- **Modes:** LOSO (6 folds per subject; sub-04 has 4 folds) and split-half (halves A and B per subject)
- **Inputs:** `decoded_states.pkl` and fold/half results JSON from script 04 loso\_fit / split\_half\_fit modes; raw parcel time series from script 02
- **Method:** Viterbi-decoded state labels pool all TRs assigned to each state across episodes; Ledoit-Wolf shrinkage covariance is computed per state; converted to Pearson correlation. States with fewer than 30 TRs are flagged as unreliable (not saved with FC).
- **K:** 50 (truncation capacity, as in primary model); n\_reliable\_states counts states that meet the 30-TR threshold.
- **Per-subject, no group statistic.** Outputs are saved in-place alongside existing LOSO/split-half model outputs under `output/04_combined_hdphmm/`.

## Results

### LOSO folds: reliable states and data volume

| Subject | Folds | n\_reliable (range) | n\_reliable (mean) | total\_TRs per fold (range) |
|---------|-------|--------------------|--------------------|----------------------------|
| sub-01 | 6 | 41-46 | 43.5 | 22345-23424 |
| sub-02 | 6 | 40-43 | 42.2 | 22345-23424 |
| sub-03 | 6 | 44-47 | 45.2 | 22345-23424 |
| sub-04 | 4 | 42-43 | 42.8 | 22345-23353 |
| sub-05 | 6 | 41-46 | 43.8 | 20961-23424 |
| sub-06 | 6 | 42-45 | 43.3 | 22345-23424 |

### LOSO folds: per-fold reliable states (n\_reliable by held-out season)

| Subject | S1 | S2 | S3 | S4 | S5 | S6 |
|---------|----|----|----|----|----|----|
| sub-01 | 46 | 42 | 41 | 45 | 44 | 43 |
| sub-02 | 43 | 43 | 43 | 42 | 42 | 40 |
| sub-03 | 45 | 44 | 45 | 46 | 44 | 47 |
| sub-04 | 43 | 42 | 43 | 43 | n/a | n/a |
| sub-05 | 43 | 41 | 44 | 44 | 46 | 45 |
| sub-06 | 42 | 44 | 44 | 42 | 43 | 45 |

### Split-half: reliable states and data volume

| Subject | Half | n\_runs | total\_TRs | n\_reliable | median TRs per state (all K=50) | median Ledoit-Wolf alpha (all K=50) |
|---------|------|---------|-----------|------------|---------------------|--------------------------|
| sub-01 | A | 154 | 72706 | 42 | 1623 | 0.0037 |
| sub-01 | B | 138 | 65207 | 42 | 1434 | 0.0040 |
| sub-02 | A | 154 | 72283 | 43 | 1544 | 0.0037 |
| sub-02 | B | 138 | 65207 | 40 | 1483 | 0.0042 |
| sub-03 | A | 154 | 72706 | 42 | 1427 | 0.0084 |
| sub-03 | B | 137 | 64751 | 44 | 1316 | 0.0088 |
| sub-04 | A | 100 | 47304 | 42 | 850 | 0.0132 |
| sub-04 | B | 94 | 44243 | 43 | 884 | 0.0156 |
| sub-05 | A | 150 | 70777 | 42 | 1655 | 0.0044 |
| sub-05 | B | 139 | 65752 | 44 | 1385 | 0.0054 |
| sub-06 | A | 154 | 72706 | 40 | 1624 | 0.0063 |
| sub-06 | B | 138 | 65206 | 42 | 1414 | 0.0073 |

### LOSO folds: shrinkage and TRs per state (season 1 fold, representative)

| Subject | median TRs per state (all K=50) | max TRs per state | median Ledoit-Wolf alpha (all K=50) |
|---------|---------------------|------------------|--------------------------|
| sub-01 | 518 | 1071 | 0.0109 |
| sub-02 | 454 | 1303 | 0.0113 |
| sub-03 | 469 | 1032 | 0.0218 |
| sub-04 | 396 | 1661 | 0.0260 |
| sub-05 | 486 | 1084 | 0.0163 |
| sub-06 | 450 | 1310 | 0.0222 |

### Matched-pair FC reproduction (RV)

For state pairs matched across fits, within-state FC is compared by the RV coefficient between the two Ledoit-Wolf correlation matrices (`04rv_reliability_vis.py`, `fc_rv_summary.json`; the same values as Fig 7b–7d). LOSO pairs are each fold's well-matched pairs against the primary model; split-half pairs are all Hungarian pairs between halves.

| Subject | median RV (LOSO) | LOSO pairs | median RV (split-half) | split-half pairs | mean-map r vs RV, Spearman rho (split-half) |
|---|---|---|---|---|---|
| sub-01 | 0.993 | 249 | 0.995 | 43 | 0.24 |
| sub-02 | 0.995 | 238 | 0.996 | 44 | 0.50 |
| sub-03 | 0.981 | 262 | 0.986 | 44 | 0.23 |
| sub-04 | 0.979 | 161 | 0.975 | 42 | 0.42 |
| sub-05 | 0.991 | 252 | 0.993 | 44 | 0.56 |
| sub-06 | 0.986 | 236 | 0.990 | 43 | 0.63 |

Matched-pair FC reproduces closely in every subject under both schemes (median RV 0.979–0.995 for LOSO and 0.975–0.996 for split-half). Agreement on the mean map does not guarantee agreement on FC for every pair: 26 of the 260 split-half pairs have RV below 0.9 (lowest 0.31), and 5 of them combine a mean-map correlation above 0.7 with an RV below 0.8. Mean-map similarity and RV are positively associated within subjects (Spearman rho 0.23–0.63). The lowest LOSO RV is 0.71 across 1398 pairs.

## Outputs

- output/04rv\_reliability\_vis/atlas-4S156Parcels/fc\_rv\_summary.json - per-pair and per-subject RV for matched states (LOSO and split-half), written by `04rv_reliability_vis.py` (`--fc_rv_only` writes only this file)
- output/04\_combined\_hdphmm/atlas-4S156Parcels/{sub\_id}/loso/season\_{N}/state\_empirical\_corr.npy - (50, 156, 156) Ledoit-Wolf Pearson correlation per state, per LOSO fold
- output/04\_combined\_hdphmm/atlas-4S156Parcels/{sub\_id}/loso/season\_{N}/fc\_metadata.json - K, n\_parcels, n\_runs, total\_trs, min\_trs, n\_reliable\_states, n\_trs\_per\_state, shrinkage\_alpha\_per\_state
- output/04\_combined\_hdphmm/atlas-4S156Parcels/{sub\_id}/split\_half/{A,B}/state\_empirical\_corr.npy - (50, 156, 156) as above, per split-half
- output/04\_combined\_hdphmm/atlas-4S156Parcels/{sub\_id}/split\_half/{A,B}/fc\_metadata.json - same fields as LOSO metadata
