# Findings: 05e_a3 Within-Session FO Trends (LME)

_Script: `script/05e_temporal_trend_a3.py`. Tier: SUPP (Supp)._

_Tests whether fractional occupancy for each brain state trends systematically across runs within scanning sessions, using a linear mixed-effects model with random intercepts per session; per-subject, n=6._

## Method (as run)

- **Script:** `05e_temporal_trend_a3.py`
- **Parcellation:** atlas-4S156Parcels; **variance threshold:** vt=0.95
- **Model per state:** `FO_k ~ run_index_centered + (1 | session)` (REML, lbfgs, maxiter=200); fixed slope beta1 = within-session FO trend rate; random intercepts only (no random slopes - sessions have 3-6 runs, insufficient to identify per-session slopes)
- **Inference:** permutation test, n_perm=5000, shuffling run indices within sessions with fixed variance components extracted from the LME fit; two-sided Phipson-Smyth p-value; BH-FDR correction across all states
- **Session assignment:** BIDS session_id from `00_get_scan` acquisition CSV; minimum session size = 3 runs
- **Inputs:** `05a_recurrence_analysis` fractional_occupancy.pkl + recurrence_scores.npy; `05e` eligible_states via state_flags.csv (sub-HRF states excluded from significance counts by default)
- **Scope:** per-subject; no group statistic; continuous scores throughout
- **Additional output:** session-detrended FO (`FO_detrended = FO - beta1 * run_idx_centered`, clipped to 0, saved as `fractional_occupancy_detrended.pkl`)

## Results

### Session Structure and LME Convergence

| Subject | n_states | n_sessions | n_usable (>=3 runs) | Session size (min-max, median) | n_converged | n_singular | n_failed |
|---|---|---|---|---|---|---|---|
| sub-01 | 50 | 63 | 57 | 1-10, 4 | 47 | 42 | 3 |
| sub-02 | 50 | 56 | 52 | 2-12, 6 | 50 | 41 | 0 |
| sub-03 | 50 | 78 | 67 | 1-6, 4 | 41 | 36 | 9 |
| sub-04 | 50 | 41 | 36 | 2-10, 4 | 47 | 40 | 3 |
| sub-05 | 50 | 44 | 40 | 2-12, 6 | 47 | 40 | 3 |
| sub-06 | 50 | 53 | 45 | 1-12, 5 | 44 | 37 | 6 |

The majority of converged fits are singular (sigma2_session = 0), meaning most states show no between-session baseline variance.

### Slope Distribution and Significant States

| Subject | Median beta1 | Min beta1 | Max beta1 | n_testable | n_sig (q<0.05) | % sig | neg (habituation) | pos (sensitization) |
|---|---|---|---|---|---|---|---|---|
| sub-01 | +0.000031 | -0.001907 | +0.001378 | 47 | 10 | 21% | 5 | 5 |
| sub-02 | -0.000127 | -0.002342 | +0.003704 | 50 | 15 | 30% | 7 | 8 |
| sub-03 | +0.000433 | -0.004720 | +0.003348 | 41 | 15 | 37% | 8 | 7 |
| sub-04 | -0.000866 | -0.004248 | +0.008209 | 47 | 33 | 70% | 21 | 12 |
| sub-05 | -0.000053 | -0.001270 | +0.001762 | 47 | 5 | 11% | 3 | 2 |
| sub-06 | -0.000228 | -0.002294 | +0.002530 | 44 | 28 | 64% | 14 | 14 |

Median slopes are near zero for all subjects. sub-04 and sub-06 show pervasive within-session trending (70% and 64% of testable states significant at q<0.05). sub-05 is the most stable (11% significant).

### ICC (Intraclass Correlation) Summary

ICC = sigma2_session / (sigma2_session + sigma2_resid). Median ICC is 0 across all subjects (dominant singular pattern). Summary of states with elevated ICC:

| Subject | Median ICC | Max ICC | State with max ICC | Recurrence score | States with ICC > 0.1 |
|---|---|---|---|---|---|
| sub-01 | 0.000 | 0.530 | state 35 | 0.096 | 3 |
| sub-02 | 0.000 | 0.533 | state 4 | 0.380 | 4 |
| sub-03 | 0.000 | 0.216 | state 30 | 0.852 | 5 |
| sub-04 | 0.000 | 0.323 | state 27 | 0.082 | 5 |
| sub-05 | 0.000 | 0.272 | state 44 | 0.003 | 2 |
| sub-06 | 0.000 | 0.579 | state 7 | 0.466 | 5 |

### Detrended FO Clipping Diagnostics

Session-detrended FO is clipped to 0 where detrending produces negative values. Clipping is minor for most subjects:

| Subject | n_clipped entries | % clipped | Max clip magnitude |
|---|---|---|---|
| sub-01 | 909 | 6.23% | 0.00667 |
| sub-02 | 1410 | 9.66% | 0.01297 |
| sub-03 | 466 | 3.20% | 0.01180 |
| sub-04 | 653 | 6.73% | 0.02052 |
| sub-05 | 899 | 6.22% | 0.00441 |
| sub-06 | 893 | 6.12% | 0.01063 |

## Outputs

- `output/05e_temporal_trend_a3/atlas-4S156Parcels/{sub_id}/vt0.95/habituation_results.json` - full LME results: slopes, SEs, ICCs, variance components, permutation p-values, BH-FDR q-values, session info, convergence stats, clip info
- `output/05e_temporal_trend_a3/atlas-4S156Parcels/{sub_id}/vt0.95/habituation_metrics.csv` - per-state table: state, recurrence_score, eligible, lme_slope, lme_se, lme_icc, sigma2_session, sigma2_resid, perm_p, q_fdr, converged, rho_global, p_global
- `output/05e_temporal_trend_a3/atlas-4S156Parcels/{sub_id}/vt0.95/fractional_occupancy_detrended.pkl` - session-detrended FO in same format as upstream fractional_occupancy.pkl
- `output/05e_temporal_trend_a3/atlas-4S156Parcels/{sub_id}/vt0.95/habituation_summary.png/.pdf` - 4-panel summary: (a) slope distribution, (b) ICC distribution, (c) volcano plot, (d) slope vs mean FO by network
- `output/05e_temporal_trend_a3/atlas-4S156Parcels/{sub_id}/vt0.95/habituation_per_state/state_NNN.png` - per-state FO vs run index with session boundaries
- `output/05e_temporal_trend_a3/atlas-4S156Parcels/{sub_id}/vt0.95/all_states/` - sensitivity run including sub-HRF states in significance counts (same structure)
