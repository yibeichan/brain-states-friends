# Findings: 05c Season Decodability from FO Profiles

_Script: `script/05c_episode_decodability.py`. Tier: SUPP (Supp)._

_Tests whether season identity can be decoded from run-level brain-state fractional occupancy vectors; per-subject, n=6._

## Method (as run)

- **Input:** Run-level fractional occupancy (FO) vectors from 05a (`fractional_occupancy.pkl`, `recurrence_summary.json`), vt=0.95 model (50 truncation states)
- **Feature matrix:** X[r, k] = FO(state k, run r); runs sorted chronologically; sub-04 has 194 runs (seasons 1-4), all others have 289-292 runs (seasons 1-6)
- **CLR transform:** Applied to raw FO before logistic regression (FO vectors sum to 1; CLR maps the simplex to Euclidean space; pseudocount = 1e-4)
- **Classifier:** L2-regularized multinomial logistic regression (C=0.1), leave-one-out cross-validation (LOO-CV)
- **Season decoding permutation test:** 1000 shuffles of season labels; p-value computed as (count + 1)/(n + 1) (Phipson-Smyth correction); this is a per-test p-value, not a multiple-comparison correction
- **Nuisance control:** Same LOO-CV classifier predicting ordinal run-order bins (same number of bins as seasons) to assess longitudinal drift confound
- **Per-state screening:** Kruskal-Wallis H test across season groups on raw FO; BH-FDR correction across states per subject; KW is descriptive (FO is compositional, so per-state tests are not independent - the CLR-based logistic regression is the primary inferential test)
- **Parcellation:** atlas-4S156Parcels (156 parcels: Schaefer 100 cortical + 56 subcortical composite); per-subject, no group statistic

## Results

### Season decoding accuracy

| Subject | n_runs | n_seasons | Chance | Accuracy | Null mean +/- SD | Perm p |
|---|---|---|---|---|---|---|
| sub-01 | 292 | 6 | 0.167 | 0.353 | 0.163 +/- 0.024 | 0.001 |
| sub-02 | 292 | 6 | 0.167 | 0.408 | 0.162 +/- 0.025 | 0.001 |
| sub-03 | 291 | 6 | 0.167 | 0.378 | 0.161 +/- 0.025 | 0.001 |
| sub-04 | 194 | 4 | 0.250 | 0.397 | 0.242 +/- 0.037 | 0.001 |
| sub-05 | 289 | 6 | 0.167 | 0.471 | 0.163 +/- 0.025 | 0.001 |
| sub-06 | 292 | 6 | 0.167 | 0.503 | 0.161 +/- 0.025 | 0.001 |

All six subjects decode season above chance (permutation p = 0.001 = minimum achievable at 1000 shuffles). Among the five 6-season subjects, accuracy ranges from 35.3% (sub-01) to 50.3% (sub-06), 2-3x above the 16.7% chance level.

### Nuisance control: session-order confound

| Subject | Season acc | Order acc | Order/season ratio |
|---|---|---|---|
| sub-01 | 0.353 | 0.356 | 1.010 |
| sub-02 | 0.408 | 0.414 | 1.017 |
| sub-03 | 0.378 | 0.364 | 0.964 |
| sub-04 | 0.397 | 0.366 | 0.922 |
| sub-05 | 0.471 | 0.453 | 0.963 |
| sub-06 | 0.503 | 0.479 | 0.952 |

Session-order accuracy matches or closely tracks season accuracy across all subjects (ratio range: 0.92-1.02). Sub-01 and sub-02 have ratios above 1.0 (order decodes better than season). The remaining four subjects (sub-03 through sub-06) have ratios ranging from 0.92 to 0.96. The classifier gains little or nothing from knowing the season label beyond knowing scan order.

### Per-state Kruskal-Wallis: FDR-significant states

| Subject | n_states | Active (H > 0) | FDR-sig (q < 0.05) | % of active | Top state (H) |
|---|---|---|---|---|---|
| sub-01 | 50 | 47 | 20 | 43% | State 10 (H = 62.7) |
| sub-02 | 50 | 50 | 21 | 42% | State 4 (H = 69.5) |
| sub-03 | 50 | 49 | 22 | 45% | State 44 (H = 84.1) |
| sub-04 | 50 | 47 | 8 | 17% | State 27 (H = 32.5) |
| sub-05 | 50 | 47 | 40 | 85% | State 43 (H = 59.1) |
| sub-06 | 50 | 44 | 32 | 73% | State 7 (H = 83.3) |

Sub-04 (4 seasons) shows far fewer FDR-significant states (8/47, 17%) than the 6-season subjects. Among 6-season subjects, sub-05 and sub-06 have the highest fractions (85% and 73%), consistent with their higher decoding accuracies.

### Top 5 KW states per subject

**sub-01**

| State | H | p_fdr |
|---|---|---|
| 10 | 62.7 | 1.67e-10 |
| 40 | 45.3 | 3.18e-07 |
| 38 | 37.2 | 9.11e-06 |
| 5 | 34.5 | 2.37e-05 |
| 45 | 31.0 | 8.54e-05 |

**sub-02**

| State | H | p_fdr |
|---|---|---|
| 4 | 69.5 | 6.47e-12 |
| 27 | 65.2 | 2.58e-11 |
| 45 | 62.1 | 7.39e-11 |
| 21 | 45.0 | 1.78e-07 |
| 20 | 39.5 | 1.86e-06 |

**sub-03**

| State | H | p_fdr |
|---|---|---|
| 44 | 84.1 | 5.93e-15 |
| 22 | 58.8 | 5.26e-10 |
| 30 | 46.1 | 1.45e-07 |
| 24 | 39.5 | 2.32e-06 |
| 31 | 31.1 | 8.57e-05 |

**sub-04**

| State | H | p_fdr |
|---|---|---|
| 27 | 32.5 | 2.01e-05 |
| 2 | 21.0 | 1.79e-03 |
| 8 | 21.0 | 1.79e-03 |
| 36 | 19.0 | 3.35e-03 |
| 3 | 16.4 | 9.41e-03 |

**sub-05**

| State | H | p_fdr |
|---|---|---|
| 43 | 59.1 | 6.42e-10 |
| 40 | 58.4 | 6.42e-10 |
| 44 | 49.2 | 2.90e-08 |
| 21 | 48.9 | 2.90e-08 |
| 6 | 47.4 | 4.73e-08 |

**sub-06**

| State | H | p_fdr |
|---|---|---|
| 7 | 83.3 | 8.52e-15 |
| 40 | 79.0 | 3.43e-14 |
| 9 | 63.2 | 4.41e-11 |
| 4 | 59.4 | 2.03e-10 |
| 1 | 56.0 | 8.07e-10 |

## Outputs

- output/05c_episode_decodability/atlas-4S156Parcels/{sub_id}/vt0.95/decodability_results.json
- output/05c_episode_decodability/atlas-4S156Parcels/{sub_id}/vt0.95/per_state_kruskal_wallis.json
- output/05c_episode_decodability/atlas-4S156Parcels/{sub_id}/vt0.95/confusion_matrix.png / .pdf
