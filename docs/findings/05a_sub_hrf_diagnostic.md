# Findings: 05a_sub_hrf_diagnostic Sub-HRF State Diagnostic

_Script: `script/05a_sub_hrf_diagnostic.py`. Tier: SUPP (Supp)._

_Multi-metric dwell-time comparison and transition structure description for sub-HRF states; per-subject, n=6, no group statistic._

## Method (as run)

- Parcellation: atlas-4S156Parcels; variance threshold: vt0.95
- Inputs: `state_recurrence_dwell_metrics.csv` from 05a recurrence analysis (per subject); empirical transition matrix derived from `decoded_states.pkl` (04 final; `best_model.pkl` requires unavailable `utils` module so the empirical fallback is used in practice)
- Sub-HRF threshold: HRF peak TR = round(5.0 / 1.49) = 3 TRs (~4.47 s)
- Three flagging criteria compared per decoded state (n_blocks > 0): mean_dwell_tr < 3 (mean criterion), median_dwell_tr < 3 (median criterion), 75th-percentile dwell_tr < 3 (p75 criterion)
- Transition structure (self-transition A_kk, bridge score, fraction-resolvable, bridge topology) reported for median-criterion sub-HRF states only; no classification thresholds applied
- Scope: per-subject diagnostics; no across-subject aggregation or group statistic

## Results

### Sub-HRF state counts by criterion

| Subject | Decoded active states | Sub-HRF (mean<3) | Sub-HRF (median<3) | Sub-HRF (p75<3) |
|---|---|---|---|---|
| sub-01 | 47 | 6 (13%) | 5 (11%) | 3 (6%) |
| sub-02 | 50 | 11 (22%) | 8 (16%) | 4 (8%) |
| sub-03 | 49 | 16 (33%) | 14 (29%) | 6 (12%) |
| sub-04 | 47 | 20 (43%) | 16 (34%) | 3 (6%) |
| sub-05 | 47 | 9 (19%) | 8 (17%) | 3 (6%) |
| sub-06 | 44 | 22 (50%) | 19 (43%) | 9 (20%) |

### Mean fraction of blocks that are sub-HRF (all decoded active states)

| Subject | Mean frac_blocks_sub_hrf | Mean dwell range (TR) | Median dwell range (TR) |
|---|---|---|---|
| sub-01 | 0.28 | 1.75 - 12.00 | 2.00 - 12.00 |
| sub-02 | 0.35 | 1.00 - 7.00 | 1.00 - 6.50 |
| sub-03 | 0.43 | 1.00 - 9.00 | 1.00 - 9.00 |
| sub-04 | 0.45 | 1.50 - 11.95 | 1.50 - 11.50 |
| sub-05 | 0.34 | 1.41 - 18.00 | 1.00 - 18.00 |
| sub-06 | 0.51 | 1.00 - 6.50 | 1.00 - 6.50 |

### Transition structure of median-criterion sub-HRF states

| Subject | Sub-HRF states (median) | Mean A_kk | Mean bridge score | Mean frac resolvable | Bridge topology (diff in/out) |
|---|---|---|---|---|---|
| sub-01 | 5 | 0.522 | 0.218 | 0.29 | 5/5 |
| sub-02 | 8 | 0.322 | 0.055 | 0.23 | 8/8 |
| sub-03 | 14 | 0.429 | 0.098 | 0.26 | 13/14 |
| sub-04 | 16 | 0.571 | 0.110 | 0.33 | 16/16 |
| sub-05 | 8 | 0.505 | 0.142 | 0.27 | 8/8 |
| sub-06 | 19 | 0.483 | 0.178 | 0.27 | 19/19 |

### Recurrence scores of median-criterion sub-HRF states

| Subject | n sub-HRF | Mean recurrence score | Min recurrence score | Max recurrence score |
|---|---|---|---|---|
| sub-01 | 5 | 0.198 | 0.031 | 0.394 |
| sub-02 | 8 | 0.199 | 0.000 | 0.661 |
| sub-03 | 14 | 0.331 | 0.000 | 0.828 |
| sub-04 | 16 | 0.399 | 0.000 | 0.820 |
| sub-05 | 8 | 0.204 | 0.003 | 0.561 |
| sub-06 | 19 | 0.527 | 0.000 | 0.925 |

## Outputs

- output/05a_sub_hrf_diagnostic/atlas-4S156Parcels/metrics_all_subjects.csv
- output/05a_sub_hrf_diagnostic/atlas-4S156Parcels/transitions_all_subjects.csv
