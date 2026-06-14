# Findings: 05e_a1 Cross-Episode Temporal Trends in Brain State Occupancy

_Script: `script/05e_temporal_trend_a1.py`. Tier: SUPP (Supp)._

_Tests whether fractional occupancy (FO) for each brain state trends systematically across episodes at three hierarchical timescales plus two diagnostics; per-subject, n=6._

## Method (as run)

- **Script:** `05e_temporal_trend_a1.py`
- **Parcellation:** atlas-4S156Parcels; **variance threshold:** vt=0.95
- **n subjects:** 6 (sub-01 through sub-06); sub-04 has 4 seasons, all others have 6
- **Inputs:** fractional_occupancy.pkl, per_season_mean_fo.json, recurrence_scores.npy, recurrence_summary.json (from 05a); decoded_states.pkl, best_model.pkl (from 04)
- **Episode unit:** broadcast episodes (standard: a+b runs -> one episode; 4-part episodes split into a+b and c+d pairs)
- **n_states per subject:** 50 (truncation capacity); eligible (non-sub-HRF) states flagged in output but all states included in analysis
- **Scale 1 - Cross-season:** Mann-Kendall tau of mean FO per season vs season number (n=4 or 6 data points; exploratory, very low power); BH-FDR per scale
- **Scale 2 - Within-season episode position:** Spearman rho of broadcast-episode FO vs episode position within each season, combined via permutation test on mean rho across seasons (5000 permutations); BH-FDR per scale
- **Scale 3 - Variance partitioning:** semi-partial R^2 decomposition with predictors: global position (real acquisition times in days when available, else ordinal index), season (categorical dummies), within-season position (normalized 0-1); per-predictor permutation test (5000 permutations); BH-FDR per predictor
- **Diagnostic 1 - Motion:** Spearman rho of median FD vs global run index; FD-controlled partial Spearman per state
- **Diagnostic 2 - State pairs:** emission-space anti-correlated pairs (Pearson r < -0.5 in back-projected parcel space); tests whether opposite FO trends accompany spatial opponency
- **Inference note:** Scale 2 and Scale 3 per-test p-values use Phipson and Smyth (count+1)/(n+1) permutation correction; Scale 1 uses asymptotic kendalltau p-values (scipy.stats.kendalltau method='auto'). Multiple comparisons across states handled by BH-FDR for all scales; analysis is per-subject with no group statistic

## Results

### Data Summary

| Subject | n_runs | n_broadcast_episodes | n_seasons | n_eligible_states | n_sub_HRF | lag-1 autocorr | effective n |
|---|---|---|---|---|---|---|---|
| sub-01 | 292 | 146 | 6 | 41 | 5 | 0.113 | 116 |
| sub-02 | 292 | 146 | 6 | 41 | 8 | 0.111 | 117 |
| sub-03 | 291 | 146 | 6 | 34 | 14 | 0.136 | 111 |
| sub-04 | 194 | 97 | 4 | 31 | 16 | 0.032 | 91 |
| sub-05 | 289 | 145 | 6 | 39 | 8 | 0.224 | 92 |
| sub-06 | 292 | 146 | 6 | 24 | 19 | 0.188 | 100 |

### Scale 1: Cross-Season Trend (Mann-Kendall tau)

| Subject | n_seasons | n_testable_states | n_sig_FDR_q005 |
|---|---|---|---|
| sub-01 | 6 | 47 | 0 |
| sub-02 | 6 | 50 | 0 |
| sub-03 | 6 | 49 | 0 |
| sub-04 | 4 | 47 | 0 |
| sub-05 | 6 | 47 | 0 |
| sub-06 | 6 | 44 | 0 |

No significant states at any FDR threshold. With 4-6 data points, power is negligible; Scale 1 is descriptive only.

### Scale 2: Within-Season Episode Position (mean Spearman rho, permutation + BH-FDR)

| Subject | n_testable_states | n_sig_FDR_q005 | Significant states (rho, q) |
|---|---|---|---|
| sub-01 | 47 | 2 | state 10 (rho=+0.276, q=0.019), state 38 (rho=-0.277, q=0.019) |
| sub-02 | 50 | 0 | - |
| sub-03 | 49 | 0 | - |
| sub-04 | 47 | 0 | - |
| sub-05 | 47 | 0 | - |
| sub-06 | 44 | 0 | - |

### Scale 3: Variance Partition (semi-partial R^2, permutation + BH-FDR)

| Subject | mean R^2_full | mean deltaR^2_global | mean deltaR^2_season | mean deltaR^2_within | mean shared_R^2 | n_sig_global | n_sig_season | n_sig_within |
|---|---|---|---|---|---|---|---|---|
| sub-01 | 0.112 | 0.011 | 0.070 | 0.006 | 0.034 | 1 | 8 | 0 |
| sub-02 | 0.111 | 0.013 | 0.064 | 0.010 | 0.037 | 3 | 7 | 0 |
| sub-03 | 0.122 | 0.016 | 0.070 | 0.010 | 0.038 | 3 | 10 | 0 |
| sub-04 | 0.083 | 0.018 | 0.048 | 0.014 | 0.021 | 1 | 1 | 1 |
| sub-05 | 0.173 | 0.016 | 0.100 | 0.010 | 0.054 | 1 | 27 | 0 |
| sub-06 | 0.152 | 0.005 | 0.114 | 0.008 | 0.029 | 0 | 20 | 0 |

Season identity is the dominant predictor (mean deltaR^2_season 0.048-0.114 across subjects). Global position uniquely explains 0.005-0.018. Within-season position uniquely explains 0.006-0.014. Approximately 83-92% of FO variance is unexplained by any temporal predictor.

### States with Significant Global Position Effect (Scale 3, BH-FDR q<0.05)

| Subject | States with sig global trend (q<0.05) |
|---|---|
| sub-01 | state 35 |
| sub-02 | state 42, state 47, state 49 |
| sub-03 | state 0, state 6, state 34 |
| sub-04 | state 36 |
| sub-05 | state 16 |
| sub-06 | none |

Total: 9 states across 5 subjects show significant unique global-position variance after BH-FDR correction.

### Diagnostic 1: Motion Confound Check

| Subject | FD trend rho | FD trend p | mean |rho_uncorr - rho_FDctrl| |
|---|---|---|---|
| sub-01 | +0.037 | 0.525 | 0.003 |
| sub-02 | +0.205 | 0.0004 | 0.013 |
| sub-03 | +0.164 | 0.005 | 0.012 |
| sub-04 | +0.015 | 0.840 | 0.003 |
| sub-05 | -0.353 | 6.6e-10 | 0.052 |
| sub-06 | -0.424 | 3.4e-14 | 0.059 |

Four of 6 subjects show a significant FD trend over the experiment (sub-02, sub-03 increasing; sub-05, sub-06 decreasing). The mean absolute shift between uncorrected and FD-controlled Spearman rho per state is small across all subjects (0.003-0.059).

### Diagnostic 2: Anti-Correlated State Pairs (emission Pearson r < -0.5)

| Subject | n_anti-correlated_pairs | n_opposite_FO_trend | opposite trend % | trend~emission Spearman r |
|---|---|---|---|---|
| sub-01 | 93 | 50 | 53.8 | 0.020 |
| sub-02 | 121 | 42 | 34.7 | -0.004 |
| sub-03 | 101 | 59 | 58.4 | 0.120 |
| sub-04 | 72 | 34 | 47.2 | -0.143 |
| sub-05 | 67 | 33 | 49.3 | -0.062 |
| sub-06 | 69 | 33 | 47.8 | 0.162 |

Opposite-trend rates range 35-58% across subjects (close to the 50% chance level). Spearman correlations between emission-space r and trend difference are near zero (-0.14 to +0.16). Spatial opponency does not reliably predict opposite temporal trends.

## Outputs

- output/05e_temporal_trend_a1/atlas-4S156Parcels/{sub_id}/vt0.95/temporal_trend_results.json
- output/05e_temporal_trend_a1/atlas-4S156Parcels/{sub_id}/vt0.95/temporal_trend_metrics.csv
- output/05e_temporal_trend_a1/atlas-4S156Parcels/{sub_id}/vt0.95/scale1_all_states_cross_season.png/.pdf
- output/05e_temporal_trend_a1/atlas-4S156Parcels/{sub_id}/vt0.95/scale2_within_season/state_NNN.png
- output/05e_temporal_trend_a1/atlas-4S156Parcels/{sub_id}/vt0.95/scale3_variance_partition.png/.pdf
- output/05e_temporal_trend_a1/atlas-4S156Parcels/{sub_id}/vt0.95/trend_vs_mean_fo.png/.pdf
- output/05e_temporal_trend_a1/atlas-4S156Parcels/{sub_id}/vt0.95/motion_confound_check.png/.pdf
- output/05e_temporal_trend_a1/atlas-4S156Parcels/{sub_id}/vt0.95/state_pair_trends.png/.pdf
