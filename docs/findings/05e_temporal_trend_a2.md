# Findings: 05e_a2 Within-Run Position Anchoring

_Script: `script/05e_temporal_trend_a2.py`. Tier: SUPP (Supp)._

_Detects brain states that cluster at the start of scanner runs, using a/b suffix contrast to separate a-specific from shared run-onset effects; per-subject n=6._

## Method (as run)

- **Atlas:** atlas-4S156Parcels (156 parcels); **PCA variance threshold:** vt=0.95
- **Input:** `decoded_states.pkl` (from 04 select) + `recurrence_summary.json` (from 05a)
- **Eligibility filter:** sub-HRF states excluded via `eligible_states.json` (from 05e); their p-values are nulled before FDR correction so they do not inflate the denominator
- **Metric:** early fraction = fraction of a state's block onsets falling in the first 20% of the run (normalized position < 0.20)
- **Null:** block-level within-run shuffle (2000 permutations, seed 42 for "a", seed 43 for "b"); preserves block counts and dwell times, destroys temporal position; one-sided test for early anchoring; p-value = (count + 1) / (n_perm + 1) (Phipson & Smyth 2010)
- **Multiple comparisons:** BH-FDR applied separately per suffix family ("early_a" and "early_b"); threshold q < 0.10
- **Conjunction classification** (neutral, no causal assumption):
  - `a_start_specific`: q_early_a < 0.10, q_early_b >= 0.10
  - `ab_start_common`: both q_early_a < 0.10 and q_early_b < 0.10
  - `b_start_specific`: q_early_b < 0.10, q_early_a >= 0.10
- **Descriptive:** `theme_fraction_a` = fraction of a-run block onsets with start_tr < 33 (approx. 49 s at TR=1.49 s); approximate lower bound only, not used for inference
- **Transition confound check:** secondary anchoring (non-locked state receives p >= 0.15 from a locked state) and sub-HRF feeders (sub-HRF state transitions to an anchored state with p >= 0.15) assessed post-hoc from fitted model transmat_
- **Per-subject, no group statistic**

## Results

### Dataset and Eligibility Summary

| Subject | Runs | Total blocks | Eligible states | Excluded sub-HRF | Anchored states | % of eligible |
|---------|------|-------------|-----------------|-------------------|-----------------|---------------|
| sub-01 | 292 | 32,040 | 45 | 5 | 7 | 16% |
| sub-02 | 292 | 32,738 | 42 | 8 | 8 | 19% |
| sub-03 | 291 | 29,946 | 36 | 14 | 6 | 17% |
| sub-04 | 194 | 19,209 | 34 | 16 | 2 | 6% |
| sub-05 | 289 | 32,097 | 42 | 8 | 7 | 17% |
| sub-06 | 292 | 21,655 | 31 | 19 | 7 | 23% |

### Anchoring Type Counts

| Subject | a_start_specific | ab_start_common | b_start_specific | none |
|---------|-----------------|-----------------|-----------------|------|
| sub-01 | 1 | 5 | 1 | 38 |
| sub-02 | 4 | 4 | 0 | 34 |
| sub-03 | 4 | 2 | 0 | 30 |
| sub-04 | 1 | 1 | 0 | 32 |
| sub-05 | 1 | 4 | 2 | 35 |
| sub-06 | 4 | 3 | 0 | 24 |

### Per-State Early Fractions for Anchored States

#### sub-01

| State | Type | early_frac_a | early_frac_b | delta (a-b) | recurrence | theme_frac_a |
|-------|------|-------------|-------------|-------------|-----------|-------------|
| s5 | ab_start_common | 0.427 | 0.376 | 0.052 | 0.322 | 0.162 |
| s6 | ab_start_common | 0.422 | 0.443 | -0.020 | 0.428 | 0.346 |
| s11 | a_start_specific | 0.316 | 0.228 | 0.087 | 0.298 | 0.082 |
| s14 | ab_start_common | 0.510 | 0.382 | 0.128 | 0.634 | 0.294 |
| s33 | b_start_specific | 0.183 | 0.291 | -0.107 | 0.428 | 0.059 |
| s36 | ab_start_common | 0.612 | 0.448 | 0.165 | 0.271 | 0.409 |
| s41 | ab_start_common | 0.551 | 0.412 | 0.139 | 0.411 | 0.385 |

#### sub-02

| State | Type | early_frac_a | early_frac_b | delta (a-b) | recurrence | theme_frac_a |
|-------|------|-------------|-------------|-------------|-----------|-------------|
| s9 | a_start_specific | 0.275 | 0.218 | 0.057 | 0.445 | 0.057 |
| s20 | ab_start_common | 0.630 | 0.461 | 0.169 | 0.209 | 0.419 |
| s21 | ab_start_common | 0.429 | 0.353 | 0.075 | 0.308 | 0.260 |
| s27 | a_start_specific | 0.439 | 0.248 | 0.190 | 0.486 | 0.291 |
| s31 | ab_start_common | 0.285 | 0.272 | 0.013 | 0.377 | 0.182 |
| s35 | a_start_specific | 0.260 | 0.141 | 0.118 | 0.291 | 0.077 |
| s36 | a_start_specific | 0.278 | 0.197 | 0.081 | 0.408 | 0.074 |
| s43 | ab_start_common | 0.447 | 0.360 | 0.086 | 0.236 | 0.258 |

#### sub-03

| State | Type | early_frac_a | early_frac_b | delta (a-b) | recurrence | theme_frac_a |
|-------|------|-------------|-------------|-------------|-----------|-------------|
| s4 | a_start_specific | 0.365 | 0.229 | 0.136 | 0.430 | 0.094 |
| s10 | ab_start_common | 0.775 | 0.675 | 0.100 | 0.072 | 0.571 |
| s24 | a_start_specific | 0.365 | 0.224 | 0.141 | 0.320 | 0.165 |
| s28 | a_start_specific | 0.246 | 0.154 | 0.092 | 0.632 | 0.079 |
| s41 | ab_start_common | 0.499 | 0.370 | 0.129 | 0.636 | 0.266 |
| s43 | a_start_specific | 0.287 | 0.266 | 0.021 | 0.096 | 0.079 |

#### sub-04

| State | Type | early_frac_a | early_frac_b | delta (a-b) | recurrence | theme_frac_a |
|-------|------|-------------|-------------|-------------|-----------|-------------|
| s2 | a_start_specific | 0.297 | 0.253 | 0.045 | 0.526 | 0.120 |
| s26 | ab_start_common | 0.667 | 0.335 | 0.331 | 0.345 | 0.396 |

#### sub-05

| State | Type | early_frac_a | early_frac_b | delta (a-b) | recurrence | theme_frac_a |
|-------|------|-------------|-------------|-------------|-----------|-------------|
| s21 | ab_start_common | 0.329 | 0.273 | 0.055 | 0.419 | 0.164 |
| s24 | b_start_specific | 0.201 | 0.271 | -0.071 | 0.436 | 0.065 |
| s27 | ab_start_common | 0.485 | 0.359 | 0.126 | 0.453 | 0.262 |
| s31 | ab_start_common | 0.492 | 0.526 | -0.034 | 0.176 | 0.422 |
| s34 | a_start_specific | 0.309 | 0.211 | 0.098 | 0.298 | 0.100 |
| s43 | ab_start_common | 0.558 | 0.430 | 0.128 | 0.225 | 0.339 |
| s46 | b_start_specific | 0.207 | 0.253 | -0.046 | 0.567 | 0.078 |

#### sub-06

| State | Type | early_frac_a | early_frac_b | delta (a-b) | recurrence | theme_frac_a |
|-------|------|-------------|-------------|-------------|-----------|-------------|
| s13 | a_start_specific | 0.255 | 0.224 | 0.031 | 0.548 | 0.063 |
| s27 | ab_start_common | 0.289 | 0.256 | 0.033 | 0.363 | 0.125 |
| s29 | a_start_specific | 0.317 | 0.220 | 0.096 | 0.099 | 0.127 |
| s30 | a_start_specific | 0.354 | 0.241 | 0.112 | 0.216 | 0.041 |
| s31 | a_start_specific | 0.307 | 0.215 | 0.091 | 0.712 | 0.125 |
| s33 | ab_start_common | 0.355 | 0.349 | 0.006 | 0.305 | 0.254 |
| s34 | ab_start_common | 0.515 | 0.310 | 0.206 | 0.253 | 0.326 |

### Recurrence of Anchored vs Non-Anchored States

Mean recurrence score across eligible (non-sub-HRF) states, split by anchoring status.

| Subject | Anchored mean recurrence | Non-anchored mean recurrence |
|---------|------------------------|------------------------------|
| sub-01 | 0.399 | 0.459 |
| sub-02 | 0.345 | 0.500 |
| sub-03 | 0.364 | 0.485 |
| sub-04 | 0.436 | 0.429 |
| sub-05 | 0.368 | 0.474 |
| sub-06 | 0.357 | 0.342 |

### Transition Confound Check

Secondary anchoring threshold: p(transition) >= 0.15 from a position-locked source state to a non-locked target; sub-HRF feeder threshold: same probability criterion from an excluded sub-HRF state to a position-locked target.

| Subject | Secondary anchored states | Sub-HRF feeders into anchored states |
|---------|--------------------------|-------------------------------------|
| sub-01 | 0 | 0 |
| sub-02 | 1 (s24 <- s36, p=0.183) | 1 (s19->s20, p=0.917) |
| sub-03 | 0 | 1 (s31->s10, p=0.188) |
| sub-04 | 1 (s14 <- s2, p=0.244) | 0 |
| sub-05 | 0 | 0 |
| sub-06 | 0 | 0 |

## Outputs

- output/05e_temporal_trend_a2/atlas-4S156Parcels/{sub_id}/vt0.95/temporal_position_metrics.csv
- output/05e_temporal_trend_a2/atlas-4S156Parcels/{sub_id}/vt0.95/temporal_position_analysis.json
- output/05e_temporal_trend_a2/atlas-4S156Parcels/{sub_id}/vt0.95/ab_early_fraction_scatter.png/.pdf
- output/05e_temporal_trend_a2/atlas-4S156Parcels/{sub_id}/vt0.95/position_cdf_flagged_states.png/.pdf
- output/05e_temporal_trend_a2/atlas-4S156Parcels/{sub_id}/vt0.95/early_fraction_bar_chart.png/.pdf
- output/05e_temporal_trend_a2/atlas-4S156Parcels/{sub_id}/vt0.95/anchored_transition_chains.png/.pdf
