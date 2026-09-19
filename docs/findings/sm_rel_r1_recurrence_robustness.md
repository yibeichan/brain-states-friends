# Findings: recurrence threshold sensitivity and occupancy confound (`sm_rel_r1_recurrence_robustness`)

## Question

(1) How much does the recurrence ordering, the active repertoire, and the content-eligible set depend on the 0.02 FO activity threshold? (2) How closely does recurrence track the model's stationary occupancy π and mean fractional occupancy?

## Method (as run)

- Recurrence recomputed from `fractional_occupancy.pkl` at FO thresholds 0.01 / 0.02 / 0.03 / 0.05 (≈ 7 / 14 / 22 / 36 s of a 12-min run) with the main-pipeline definition (fraction of runs with FO strictly above threshold), gated to `recurrence_scores.npy` at 0.02 (`gate.recurrence_max_abs_delta`; Δ = 0 in all six).
- Eligibility churn counts states whose category among the flag-free set {eligible_for_content_analysis, rare, unused} changes; exclusion flags (sub-HRF, run-onset-anchored, drift-anchored) are threshold-independent and taken from `state_flags.csv`, whose categories the recomputation reproduces exactly at 0.02 (`gate.category_mismatches`; empty list in all six).
- π from `best_model.pkl` over active states (`utils.stationary`), gated to `stationary_distribution.npy` (`gate.stationary_max_abs_delta`; 0.0 in all six).
- Eligibility cut: recurrence > 0.10 (`eligibility_recurrence` in the JSON) among the flag-free states.

## Results

**Reference (0.02) range**, `thresholds["0.02"].range`:

| sub | n_active | min | max | p10 | p90 |
|---|---|---|---|---|---|
| sub-01 | 46 | 0.0034 | 0.8493 | 0.1610 | 0.7260 |
| sub-02 | 46 | 0.0034 | 0.8356 | 0.2226 | 0.7158 |
| sub-03 | 44 | 0.0722 | 0.8522 | 0.1564 | 0.8089 |
| sub-04 | 44 | 0.0670 | 0.9021 | 0.1443 | 0.8077 |
| sub-05 | 47 | 0.0035 | 0.8166 | 0.1142 | 0.6997 |
| sub-06 | 42 | 0.0103 | 0.9247 | 0.1113 | 0.8695 |

Across all six participants: min ranges 0.0034–0.0722 and max ranges 0.8166–0.9247, i.e. the overall span is ≈ 0.003–0.925 — consistent with the Results text's pooled 0.003–0.925. The p10/p90 band ranges 0.111–0.223 / 0.700–0.870 across participants, i.e. within the Results text's pooled middle-80% band of 0.11–0.87.

**Gate** (all six): `recurrence_max_abs_delta` = 0.0, `stationary_max_abs_delta` = 0.0, `category_mismatches` = [] (n_runs 194–292, n_states_total 50 in all six).

**Rank stability vs the 0.02 reference** (`thresholds[t].rank_stability_vs_reference.rho_reference_active`, Spearman ρ over the reference-active state set):

| sub | 0.01 | 0.02 | 0.03 | 0.05 |
|---|---|---|---|---|
| sub-01 | 0.9516 | 1.0000 | 0.9746 | 0.8464 |
| sub-02 | 0.9644 | 1.0000 | 0.9705 | 0.7986 |
| sub-03 | 0.9626 | 1.0000 | 0.9826 | 0.8089 |
| sub-04 | 0.9454 | 1.0000 | 0.9735 | 0.7753 |
| sub-05 | 0.9064 | 1.0000 | 0.9789 | 0.9138 |
| sub-06 | 0.9750 | 1.0000 | 0.9722 | 0.7666 |
| **min–max** | **0.9064–0.9750** | 1.0000 | 0.9705–0.9826 | 0.7666–0.9138 |

(0.02 is the reference threshold itself, so ρ = 1.0 and all churn counts are 0 by construction — omitted from the churn table below.)

**Active- and eligibility-category churn vs the 0.02 reference** (`thresholds[t].churn_vs_reference`; "active" = states with recurrence > 0 at that threshold; "eligible" = flag-free states with recurrence > 0.10):

| sub | 0.01 active/eligible Δ | 0.01 gained/lost ids | 0.03 active/eligible Δ | 0.03 gained/lost ids | 0.05 active/eligible Δ | 0.05 gained/lost ids |
|---|---|---|---|---|---|---|
| sub-01 | 1 / 1 | gained [10] | 1 / 1 | lost [24] | 4 / 20 | lost [1,2,4,8,9,16,19,20,21,22,24,26,29,30,32,34,38,40,43,46] |
| sub-02 | 1 / 1 | gained [16] | 2 / 0 | — | 4 / 19 | lost [1,2,3,5,8,10,12,13,23,24,26,28,29,32,33,34,37,41,44] |
| sub-03 | 1 / 0 | — | 1 / 4 | lost [20,27,36,38] | 3 / 18 | lost [1,3,9,11,12,13,17,18,20,23,26,27,29,33,36,38,39,40] |
| sub-04 | 1 / 1 | gained [20] | 0 / 1 | lost [38] | 4 / 17 | lost [1,5,7,10,11,16,18,21,23,29,30,31,32,37,38,42,44] |
| sub-05 | 0 / 0 | — | 1 / 1 | lost [1] | 4 / 14 | lost [0,1,7,13,15,17,20,22,28,30,36,39,40,42] |
| sub-06 | 1 / 1 | gained [37] | 1 / 2 | lost [26,41] | 1 / 11 | lost [0,4,5,14,17,22,24,25,26,32,41] |

`n_eligible_ref` (0.02) per participant: sub-01 31, sub-02 30, sub-03 26, sub-04 27, sub-05 29, sub-06 16.

**Occupancy confound** (`occupancy_confound`):

| sub | n_active | ρ(recurrence, π) | ρ(recurrence, mean FO) | ρ(π, mean FO) |
|---|---|---|---|---|
| sub-01 | 46 | 0.9670 | 0.9858 | 0.9763 |
| sub-02 | 46 | 0.9796 | 0.9805 | 0.9940 |
| sub-03 | 44 | 0.9739 | 0.9813 | 0.9937 |
| sub-04 | 44 | 0.9700 | 0.9727 | 0.9946 |
| sub-05 | 47 | 0.9819 | 0.9830 | 0.9966 |
| sub-06 | 42 | 0.9842 | 0.9859 | 0.9966 |

## Interpretation (bounded)

At 0.01 and 0.03, recurrence ranking is stable and category churn is small: Spearman ρ vs. the 0.02 reference is never below 0.9064 (sub-05, at 0.01) and reaches 1.0000 at 0.02 by construction; active-state churn is at most 2 states (sub-02, 0.03) and eligibility churn is at most 4 states (sub-03, 0.03) out of 16–31 reference-eligible states per participant. Both criteria clear the ρ ≥ 0.9 / single-digit-churn bar, so for the 0.01–0.03 range the ranking and the eligible set can be treated as effectively the same as at 0.02 — a one-sentence Methods note to that effect is warranted, bounded to that range and quoting the minimum observed ρ of 0.9064.

At 0.05 this breaks down: rank stability falls to 0.7666–0.9138 across participants (below the 0.9 bar in five of six), and eligibility churn becomes substantial — 11 to 20 states change category, with sub-01 losing 20 of its 31 reference-eligible states (65%) at 0.05, the largest churn of any participant. This is a limitation of the 0.10 eligibility cut specifically at the 0.05 threshold (36 s of a 12-min run is a much stricter activity bar than 14 s), not a general statement that the recurrence analysis is insensitive to the activity threshold — it clearly is not, once the threshold reaches 0.05.

The occupancy-confound correlations hold regardless of threshold sensitivity: recurrence (at the reference 0.02 threshold) correlates with the fitted model's stationary distribution π at ρ = 0.9670–0.9842 and with mean fractional occupancy at ρ = 0.9727–0.9859 across the six participants; π and mean FO correlate with each other at ρ = 0.9763–0.9966. These are properties of the 0.02-threshold recurrence definition and the fitted model, independent of the threshold-sweep result above.

## Caveats and audit notes

- Threshold variants change which states are "active"; the table above reports rank stability over the reference-active state set (`rho_reference_active`); the JSON also carries `rho_both_active` (Spearman restricted to states active under both thresholds), which differs from `rho_reference_active` only at 0.03 and 0.05 (e.g. sub-01 at 0.05: 0.8464 vs. 0.8073) since at 0.01/0.02 the active sets barely change.
- The eligibility churn is counterfactual: downstream analyses were not re-run at other thresholds; the churn counts describe what the eligible set *would* look like, not a re-analysis of content coding at those thresholds.
- The 0.05 churn lists (`gained_eligible` / `lost_eligible`) are state ids from each participant's own 06b/04 state numbering and are not comparable across participants (state 20 in sub-01 is not the same state as state 20 in sub-03).

## Connections

- `05a_recurrence_analysis.py`, `05e_temporal_trend_a4` (flags), `06b_transition_structure.py` (π). Manuscript: Methods §State classification; Results ¶2.
