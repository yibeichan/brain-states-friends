# Findings: are the excluded states copies of content-eligible states? (`05g_excluded_state_check`)

## Question

05e_a4 sets aside some active states from content analyses. Its low-confidence flag (sub-HRF: median dwell < 3 TRs) is a timing property and says nothing about a state's map. Is an excluded state a brief interruption of one content-eligible state (a copy split off it), a blend of the two states it sits between, or neither?

## Method (as run)

- **States.** Per participant, the active states (FO > 0.02 in at least one run) that are not content-eligible.
  - Run-onset-anchored states (05e_a4 raw `run_onset`, `a_anchored` or `b_anchored`) are set aside first. The raw booleans are used because the summary category ranks sub-HRF first.
  - States with fewer than 10 visits that are not at a run edge are "too few visits".
- **Visits.** Decoded paths are collapsed into segments within each run. A visit's neighbours are the adjacent segments, and visits at a run's first or last segment are censored.
- **Maps.** Empirical maps: the mean of the 03a PC scores over a state's decoded TRs, back-projected to the 156 parcels (no PCA mean), row z-scored.
- **Interruption.** All three conditions are required:
  - excess returns, i.e. a share of visits with the same state before and after that exceeds its expectation under independence. The permutation null permutes the following state across visits within each run (5,000 draws), with BH within participant.
  - `r_parent` at least the 95th percentile of the closest-match correlations among content-eligible states (`ref_nn`).
  - `r_parent` at least the 5th percentile of a bootstrap in which parent windows shaped like the state's visits are correlated with the rest of the parent (`ref_boot`). The bootstrap is untested when the state's visits add up to half the parent's TRs or more.
- **Blend.** At least half of visits pass through, and an unconstrained OLS fit on the two most frequent pass-through partners improves R² beyond the 95th percentile of the same rule applied to content-eligible states. The null has one value per content-eligible state, so blend is a descriptive cut.
- **Sensitivity.** `r_parent` divided by the median bootstrap correlation (noise-corrected), compared with `ref_nn`.

## Results

| participant | active | content-eligible | examined | run-onset-anchored | too few visits | tested | median return share | median expected | excess return q < 0.05 | median r_max | max r_max | ref_nn | r_max >= ref_nn | max noise-corrected r_parent | interruption | blend | neither |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 | 46 | 31 | 15 | 7 | 1 | 7 | 0.024 | 0.025 | 0 | 0.54 | 0.81 | 0.771 | 1 | 0.48 | 0 | 0 | 7 |
| sub-02 | 46 | 30 | 16 | 8 | 3 | 5 | 0.027 | 0.031 | 0 | 0.60 | 0.68 | 0.854 | 0 | 0.68 | 0 | 1 | 4 |
| sub-03 | 44 | 26 | 18 | 6 | 0 | 12 | 0.008 | 0.013 | 0 | 0.62 | 0.78 | 0.805 | 0 | 0.12 | 0 | 1 | 11 |
| sub-04 | 44 | 27 | 17 | 2 | 0 | 15 | 0.009 | 0.021 | 0 | 0.55 | 0.68 | 0.727 | 0 | 0.52 | 0 | 1 | 14 |
| sub-05 | 47 | 29 | 18 | 7 | 2 | 9 | 0.032 | 0.037 | 0 | 0.50 | 0.72 | 0.815 | 0 | 0.23 | 0 | 1 | 8 |
| sub-06 | 42 | 16 | 26 | 7 | 0 | 19 | 0.011 | 0.016 | 0 | 0.49 | 0.80 | 0.808 | 0 | 0.60 | 0 | 4 | 15 |

- **No examined state shows excess returns** in any participant (q >= 0.05 for all).
- **Their maps are less similar to any content-eligible state than distinct content-eligible states are to each other.** Median closest correlation is 0.49-0.62, against `ref_nn` of 0.73-0.85. The exception is sub-01 state 7 (r_max 0.81, no excess returns).
- **The noise-corrected parent correlation stays below `ref_nn` in every participant.**
- **No state is an interruption.** 8 states in total are blends; none is labelled an HRF blend (one is untested for lack of direct switches).

## Interpretation (bounded)

No excluded state met the pre-stated interruption criteria, and none showed excess returns. Comparisons of state maps can therefore include them without counting the same pattern twice. The criteria favour calling short states distinct when their maps are noisy. The excess-return test does not depend on the maps, and it alone finds no interruptions.

## Caveats

- The bootstrap reference is untested for many examined states (summary JSON `n_ref_boot_untested`), so the excess-return test carries the result.
- The blend count is descriptive.

## Outputs and reproduce

`$SCRATCH_DIR/output/05g_excluded_state_check/atlas-4S156Parcels/{sub}/vt0.95/` holds `state_classes.csv` and `state_classes_summary.json`.

```bash
uv run python -m pytest script/tests/test_05g_excluded_state_check.py
sbatch script/05g_excluded_state_check.sh        # all six participants, ~3 min
```
