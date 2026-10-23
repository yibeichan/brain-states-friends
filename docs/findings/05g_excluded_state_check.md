# Findings: are the excluded states interruptions or blends of their neighbours? (`05g_excluded_state_check`)

## Question

05e_a4 sets aside some active states from content analyses. Its low-confidence flag (sub-HRF: median dwell < 3 TRs) is a timing property and says nothing about a state's map. Is an excluded state an interruption of a neighbouring state (a copy split off it), a blend of the two states it sits between, or neither?

## Method (as run)

- **States.** Per participant, the active states (FO > 0.02 in at least one run) that are not content-eligible.
  - Run-onset-anchored states (05e_a4 raw `run_onset`, `a_anchored` or `b_anchored`) are set aside first. The raw booleans are used because the summary category ranks sub-HRF first.
  - States with fewer than 10 visits that are not at a run edge are "too few visits".
  - The rest are "tested": both rules below are applied to them.
- **Visits.** Decoded paths are collapsed into segments within each run. A visit's neighbours are the adjacent segments, and visits at a run's first or last segment are censored.
- **Maps.** Empirical maps: the mean of the 03a PC scores over a state's decoded TRs, back-projected to the 156 parcels (no PCA mean), row z-scored.
- **Neighbours.** The parent (the most frequent state seen both before and after a visit) and the two blend partners (the most frequent pair of different states before and after) are chosen among all states, not only content-eligible ones. `state_classes.csv` gives the 05e_a4 summary category of each (`parent_category`, `partner_a_category`, `partner_b_category`).
- **Interruption.** All three conditions are required:
  - excess returns, i.e. a share of visits with the same state before and after that exceeds its expectation under independence. The permutation null permutes the following state across visits within each run (5,000 draws), with BH within participant over the tested states with a defined p. Only informative runs enter the test: runs with at least 2 visits and at least 2 different following states, since no permutation changes the others. The test is undefined (p NaN, left out of BH) when fewer than 10 visits remain in informative runs or the null has no spread.
  - `r_parent` at least the 95th percentile of the closest-match correlations among content-eligible states (`ref_nn`).
  - `r_parent` at least the 5th percentile of a bootstrap in which non-overlapping parent windows, one per visit and of the same lengths, are correlated with the rest of the parent (`ref_boot`). The bootstrap is undefined when the state has at least as many TRs as its parent. Close to that limit the rest of the parent is small and the reference is low.
- **Blend.** At least half of visits pass through, and an unconstrained OLS fit on the two blend partners improves R² beyond the 95th percentile of the same rule applied to content-eligible states. The null has one value per content-eligible state, so blend is a descriptive cut.
- **Pass, fail, untested.** Each arm (interruption, blend) fails if any of its conditions can be evaluated and fails, is untested if none fails but at least one is undefined, and passes otherwise. A state is an interruption or a blend when that arm passes (both: interruption/blend), "neither" only when both arms are tested and fail, and "untested" otherwise. Both arm statuses are columns of `state_classes.csv` (`interruption_status`, `blend_status`).
- **HRF blend.** A blend whose map is as close to the average around direct switches between its partners as to its blend fit. Untested when there are fewer than 10 such switches.
- **Sensitivity.** `r_parent` divided by the median bootstrap correlation (noise-corrected), compared with `ref_nn`, over all tested states with a parent. Where the correction is undefined, the raw `r_parent` is used; it is a lower bound on the corrected value (`r_parent_sensitivity`, summary `n_noise_correction_undefined`).

## Results

| participant | active | content-eligible | examined | run-onset-anchored | too few visits | tested | excess-return test undefined | median return share | median expected | excess return q < 0.05 | median r_max | max r_max | ref_nn | r_max >= ref_nn | max r_parent_sensitivity | interruption | blend | neither | untested |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 | 46 | 31 | 15 | 7 | 1 | 7 | 0 | 0.03 | 0.03 | 0 | 0.54 | 0.81 | 0.771 | 1 | 0.48 | 0 | 0 | 7 | 0 |
| sub-02 | 46 | 30 | 16 | 8 | 3 | 5 | 0 | 0.03 | 0.03 | 0 | 0.60 | 0.68 | 0.854 | 0 | 0.68 | 0 | 1 | 4 | 0 |
| sub-03 | 44 | 26 | 18 | 6 | 0 | 12 | 0 | 0.01 | 0.01 | 0 | 0.62 | 0.78 | 0.805 | 0 | 0.73 | 0 | 1 | 11 | 0 |
| sub-04 | 44 | 27 | 17 | 2 | 0 | 15 | 1 | 0.01 | 0.02 | 0 | 0.55 | 0.68 | 0.727 | 0 | 0.69 | 0 | 1 | 14 | 0 |
| sub-05 | 47 | 29 | 18 | 7 | 2 | 9 | 0 | 0.03 | 0.04 | 0 | 0.50 | 0.72 | 0.815 | 0 | 0.25 | 0 | 1 | 8 | 0 |
| sub-06 | 42 | 16 | 26 | 7 | 0 | 19 | 0 | 0.01 | 0.02 | 0 | 0.49 | 0.80 | 0.808 | 0 | 0.81 | 0 | 4 | 15 | 0 |

Return share and expected are over the visits in informative runs. Arm status counts per participant (interruption and blend: pass / fail / untested over the tested states):

| participant | interruption pass | interruption fail | interruption untested | blend pass | blend fail | blend untested | bootstrap undefined | HRF blend | HRF blend untested |
|---|---|---|---|---|---|---|---|---|---|
| sub-01 | 0 | 7 | 0 | 0 | 7 | 0 | 2 | 0 | 0 |
| sub-02 | 0 | 5 | 0 | 1 | 4 | 0 | 1 | 0 | 0 |
| sub-03 | 0 | 12 | 0 | 1 | 11 | 0 | 5 | 0 | 0 |
| sub-04 | 0 | 15 | 0 | 1 | 14 | 0 | 4 | 0 | 0 |
| sub-05 | 0 | 9 | 0 | 1 | 8 | 0 | 1 | 0 | 0 |
| sub-06 | 0 | 19 | 0 | 4 | 15 | 0 | 7 | 0 | 1 |

- **Excess returns.** None of the 66 tested states with a defined excess-return test reaches q < 0.05; the test is undefined for 1 (sub-04).
- **Closest content-eligible state.** The median closest correlation per participant is 0.49-0.62, against `ref_nn` of 0.73-0.85. States above or within 0.01 of `ref_nn`:
  - sub-01 state 7 (low-confidence, 995 visits, neither): r_max 0.810 (closest: content-eligible state 19) against 0.771. Its parent is content-eligible state 2, but its noise-corrected `r_parent` is 0.33 and it shows no excess returns.
  - sub-01 state 31 and sub-05 state 45 (rare, too few visits, so untested): r_max 0.794 against 0.771 and 0.866 against 0.815.
  - sub-06 state 37 (rare, 333 visits, neither): r_max 0.800 against 0.808.
- **Noise-corrected parent correlation.** One tested state reaches `ref_nn`: sub-06 state 7 (low-confidence, a blend), 0.808 against 0.808, with parent state 9, which is itself low-confidence. sub-06 state 37 (rare) is 0.807 against 0.808, with content-eligible parent state 11. Neither shows excess returns, so neither is an interruption under the sensitivity rule either. For 20 tested states the correction is undefined and the raw `r_parent` is used.
- **Classes.** Of 67 tested states: interruption 0, blend 8 (HRF blend 0, untested 1), neither 59, untested 0. Every interruption arm was evaluated and failed; 20 had an undefined bootstrap reference but failed another condition.

## Interpretation (bounded)

None of the 67 tested states met the interruption criteria, and none showed excess returns. For these states, comparisons of state maps can include them without counting the same pattern twice. The statement does not extend to run-onset-anchored states or to the 6 states with too few visits, two of which (sub-01 state 31, sub-05 state 45) have maps close to a content-eligible state. Two tested states sit at the map threshold without excess returns. sub-06 state 37's map (noise-corrected) is as close to content-eligible state 11 as distinct content-eligible states are to each other, so an analysis that adds it to the content-eligible states could count that pattern twice. sub-06 state 7's map matches its parent, low-confidence state 9, which matters for analyses over all active states. The map criteria favour calling short states distinct when their maps are noisy. The excess-return test does not depend on the maps.

## Caveats

- The bootstrap reference is undefined for some tested states (summary JSON `n_ref_boot_untested`); their interruption arm is untested unless another condition fails.
- The parent and the blend partners can themselves be excluded states (see the `*_category` columns).
- The blend count is descriptive.

## Outputs and reproduce

`$SCRATCH_DIR/output/05g_excluded_state_check/atlas-4S156Parcels/{sub}/vt0.95/` holds `state_classes.csv` and `state_classes_summary.json`.

```bash
uv run --with pytest pytest script/tests/test_05g_excluded_state_check.py
sbatch script/05g_excluded_state_check.sh        # all six participants
```
