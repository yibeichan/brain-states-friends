# Findings: do two participants have matching states? (`05h_participant_state_matching`)

## Question

Each participant's HMM was fitted separately. Does one participant's repertoire of states have counterparts in another participant's, beyond what two repertoires with the same parcel-space covariance give by chance and beyond large-scale networks? And are matched states occupied at the same moments of the episodes both participants watched?

## Method (as run)

Per participant pair (15 pairs):
- **Maps and matching.** Parcel-space contrast maps (state mean without the PCA mean), row z-scored. Rectangular Hungarian matching on 1 - r; the pair statistic is the median matched r.
- **Primary null (state mixing).** One participant's centred maps (alternating participants) are mixed across states by a random orthogonal matrix that fixes the mean, 1,000 draws. It keeps the mean map and the parcel-space covariance of the repertoire, including principal axes shared with the other participant, and loses which state is which.
- **Calibration.** On simulated pairs without state correspondence (shared subspaces, a decaying spectrum, heavy-tailed states, and real maps against Gaussian surrogates with another participant's covariance), the pair test rejected at 0.035-0.055 (200 pairs each), and it detected planted counterparts in every simulated pair. A rotation of the maps within their span, the null used before, rejected 0.995-1.000 of the pairs whose participants share a dominant subspace. A max-statistic threshold for single matches on the state-mixing null was liberal for heavy-tailed states (0.235), so individual matches are descriptive.
- **Secondary null (PC blocks).** A rotation of the state means within blocks of 5 adjacent PCs, back-projected, then given the variant's scope and centring.
- **Network level.** Network-mean maps and within-network residual maps (each map minus its network means), each with its own state-mixing null.
- **Amplitude.** Uncentred cosine on raw maps.
- **Shared components.** The top 3 components of the other four participants' state maps are removed from both maps, with the null re-run (k = 1 and 5 descriptive).
- **Time-locking.** Pearson r of matched states' binary occupancy over shared runs, against within-run circular shifts (1,000 draws). Matched pairs are flagged with a family-wise threshold.
  - The locking of the map matches is compared with the locking of the pairs chosen by the null matchings.
  - It is recomputed with 10 TRs dropped at each run edge.
- **Timing properties.** Rank agreement of recurrence, FO and dwell against the null matchings.
- **Ceiling.** Split-half fits.
- **Corrections.** BH across pairs within each family: one per (null, variant) for the pair test, and one per (test, state set, scope) for the specificity and validation tests. Random streams are per pair and variant. Every pair file records its parameters, code commit and hashes of the code and of every input file.

## Results

**Eligible states, all parcels.** The p floor is 1/1001; q is BH across pairs.

| pair | states | median r | state-mixing null mean | q | PC-block null mean | PC-block q | network-mean r (null) | within-network r (null) | raw cosine (null) | top-3 removed r (null) | variance removed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 / sub-02 | 31 x 30 | 0.726 | 0.593 | 0.0012 | 0.522 | 0.0010 | 0.865 (0.806) | 0.658 (0.502) | 0.711 (0.616) | 0.557 (0.462) | 0.51 |
| sub-01 / sub-03 | 31 x 26 | 0.653 | 0.573 | 0.0012 | 0.529 | 0.0010 | 0.816 (0.785) | 0.643 (0.502) | 0.734 (0.599) | 0.537 (0.465) | 0.51 |
| sub-01 / sub-04 | 31 x 27 | 0.600 | 0.528 | 0.0021 | 0.499 | 0.0010 | 0.733 (0.745) | 0.547 (0.450) | 0.635 (0.556) | 0.562 (0.465) | 0.47 |
| sub-01 / sub-05 | 31 x 29 | 0.688 | 0.565 | 0.0012 | 0.505 | 0.0010 | 0.881 (0.776) | 0.631 (0.470) | 0.741 (0.592) | 0.585 (0.450) | 0.49 |
| sub-01 / sub-06 | 31 x 16 | 0.684 | 0.586 | 0.0012 | 0.543 | 0.0010 | 0.837 (0.788) | 0.657 (0.522) | 0.728 (0.617) | 0.549 (0.466) | 0.49 |
| sub-02 / sub-03 | 30 x 26 | 0.688 | 0.581 | 0.0012 | 0.528 | 0.0010 | 0.829 (0.783) | 0.636 (0.519) | 0.699 (0.608) | 0.537 (0.456) | 0.54 |
| sub-02 / sub-04 | 30 x 27 | 0.619 | 0.541 | 0.0050 | 0.500 | 0.0010 | 0.803 (0.745) | 0.527 (0.464) | 0.644 (0.573) | 0.541 (0.475) | 0.47 |
| sub-02 / sub-05 | 30 x 29 | 0.734 | 0.571 | 0.0012 | 0.502 | 0.0010 | 0.869 (0.774) | 0.654 (0.483) | 0.740 (0.606) | 0.613 (0.464) | 0.50 |
| sub-02 / sub-06 | 30 x 16 | 0.691 | 0.582 | 0.0012 | 0.529 | 0.0010 | 0.821 (0.790) | 0.623 (0.512) | 0.710 (0.630) | 0.533 (0.469) | 0.52 |
| sub-03 / sub-04 | 26 x 27 | 0.804 | 0.550 | 0.0012 | 0.486 | 0.0010 | 0.838 (0.757) | 0.736 (0.492) | 0.816 (0.582) | 0.671 (0.477) | 0.45 |
| sub-03 / sub-05 | 26 x 29 | 0.756 | 0.559 | 0.0012 | 0.511 | 0.0010 | 0.876 (0.766) | 0.663 (0.501) | 0.794 (0.589) | 0.558 (0.442) | 0.50 |
| sub-03 / sub-06 | 26 x 16 | 0.781 | 0.594 | 0.0012 | 0.526 | 0.0010 | 0.869 (0.780) | 0.683 (0.545) | 0.812 (0.630) | 0.684 (0.495) | 0.49 |
| sub-04 / sub-05 | 27 x 29 | 0.626 | 0.519 | 0.0012 | 0.485 | 0.0010 | 0.832 (0.718) | 0.609 (0.453) | 0.642 (0.546) | 0.556 (0.460) | 0.45 |
| sub-04 / sub-06 | 27 x 16 | 0.791 | 0.584 | 0.0012 | 0.499 | 0.0010 | 0.869 (0.784) | 0.691 (0.515) | 0.781 (0.622) | 0.661 (0.511) | 0.43 |
| sub-05 / sub-06 | 29 x 16 | 0.732 | 0.553 | 0.0012 | 0.505 | 0.0010 | 0.847 (0.776) | 0.608 (0.470) | 0.775 (0.612) | 0.580 (0.459) | 0.49 |

**Checks on occupancy, same matches:**

| pair | locking r | shift null | q | time-locked (family-wise) | pointwise | relabelled-matching locking | q | run edges dropped | recurrence rho (q) | FO rho (q) | dwell rho (q) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 / sub-02 | 0.0961 | 0.0013 | 0.0010 | 24 of 30 | 26 | 0.0013 | 0.0010 | 0.0962 | -0.02 (0.552) | -0.08 (0.673) | 0.34 (0.322) |
| sub-01 / sub-03 | 0.1054 | 0.0014 | 0.0010 | 18 of 26 | 23 | 0.0013 | 0.0010 | 0.1054 | 0.67 (0.015) | 0.67 (0.015) | 0.08 (0.883) |
| sub-01 / sub-04 | 0.0517 | 0.0006 | 0.0010 | 16 of 27 | 17 | 0.0012 | 0.0010 | 0.0515 | 0.40 (0.052) | 0.46 (0.018) | 0.15 (0.883) |
| sub-01 / sub-05 | 0.0835 | 0.0010 | 0.0010 | 22 of 29 | 25 | 0.0009 | 0.0010 | 0.0837 | 0.31 (0.073) | 0.24 (0.131) | 0.21 (0.653) |
| sub-01 / sub-06 | 0.0487 | 0.0005 | 0.0010 | 10 of 16 | 12 | 0.0020 | 0.0010 | 0.0484 | 0.26 (0.173) | 0.20 (0.243) | 0.15 (0.883) |
| sub-02 / sub-03 | 0.0727 | 0.0014 | 0.0010 | 18 of 26 | 20 | 0.0015 | 0.0010 | 0.0728 | 0.29 (0.081) | 0.32 (0.070) | -0.04 (0.883) |
| sub-02 / sub-04 | 0.0573 | 0.0004 | 0.0010 | 17 of 27 | 17 | 0.0012 | 0.0010 | 0.0573 | 0.35 (0.073) | 0.50 (0.018) | -0.01 (0.883) |
| sub-02 / sub-05 | 0.0697 | 0.0009 | 0.0010 | 20 of 29 | 25 | 0.0007 | 0.0010 | 0.0696 | 0.38 (0.052) | 0.31 (0.070) | 0.18 (0.653) |
| sub-02 / sub-06 | 0.0641 | 0.0010 | 0.0010 | 8 of 16 | 12 | 0.0026 | 0.0010 | 0.0639 | 0.59 (0.045) | 0.49 (0.070) | -0.29 (0.934) |
| sub-03 / sub-04 | 0.0800 | 0.0009 | 0.0010 | 16 of 26 | 19 | 0.0019 | 0.0010 | 0.0800 | 0.61 (0.015) | 0.56 (0.015) | 0.28 (0.594) |
| sub-03 / sub-05 | 0.0835 | 0.0006 | 0.0010 | 16 of 26 | 20 | 0.0010 | 0.0010 | 0.0835 | 0.38 (0.062) | 0.38 (0.056) | 0.35 (0.322) |
| sub-03 / sub-06 | 0.0840 | 0.0011 | 0.0010 | 11 of 16 | 13 | 0.0034 | 0.0010 | 0.0838 | 0.61 (0.030) | 0.69 (0.015) | -0.15 (0.934) |
| sub-04 / sub-05 | 0.0403 | 0.0004 | 0.0010 | 14 of 27 | 18 | 0.0005 | 0.0010 | 0.0405 | 0.34 (0.073) | 0.45 (0.037) | -0.01 (0.934) |
| sub-04 / sub-06 | 0.0894 | 0.0017 | 0.0010 | 9 of 16 | 11 | 0.0038 | 0.0010 | 0.0894 | 0.36 (0.100) | 0.31 (0.151) | -0.07 (0.934) |
| sub-05 / sub-06 | 0.0466 | 0.0005 | 0.0010 | 9 of 16 | 12 | 0.0018 | 0.0010 | 0.0463 | 0.40 (0.081) | 0.34 (0.135) | -0.32 (0.984) |

**All variants:**

| variant | median r (range over pairs) | state-mixing null mean | pairs with q < 0.05 | PC-block null mean | PC-block pairs with q < 0.05 |
|---|---|---|---|---|---|
| active, cortex, centred | 0.640-0.793 | 0.533-0.579 | 15 of 15 | 0.525-0.572 | 15 of 15 |
| active, cortex, raw | 0.630-0.799 | 0.533-0.582 | 15 of 15 | 0.527-0.574 | 15 of 15 |
| active, full, centred | 0.649-0.797 | 0.537-0.577 | 15 of 15 | 0.495-0.546 | 15 of 15 |
| active, full, raw | 0.685-0.804 | 0.537-0.576 | 15 of 15 | 0.498-0.546 | 15 of 15 |
| eligible, cortex, centred | 0.587-0.783 | 0.521-0.599 | 14 of 15 | 0.518-0.568 | 14 of 15 |
| eligible, cortex, raw | 0.576-0.808 | 0.512-0.603 | 14 of 15 | 0.510-0.569 | 15 of 15 |
| eligible, full, centred | 0.615-0.789 | 0.526-0.594 | 14 of 15 | 0.492-0.540 | 15 of 15 |
| eligible, full, raw | 0.600-0.804 | 0.519-0.594 | 15 of 15 | 0.485-0.543 | 15 of 15 |
| onset_control, full, raw | 0.376-0.713 | 0.431-0.588 | 0 of 10 | — | — |

**Ceiling:**

| ceiling set | within participant (half A vs half B) | across participants (half A vs half B) |
|---|---|---|
| all active states | 0.838-0.962 | 0.688-0.745 |
| half-fit states matched to the final content-eligible states | 0.853-0.966 | 0.610-0.823 |

- **Pair test.** On eligible states and all parcels, every pair matches beyond the state-mixing null (q 0.0012-0.0050) and the PC-block null. The state-mixing null mean is 0.52-0.59, against observed medians of 0.60-0.80: much of the matched r comes from covariance the repertoires share, and the excess is 0.07-0.25.
- **Other variants.** All 15 pairs pass for all active states in every scope. For eligible states, three variants each miss one pair: sub-01 / sub-04 on cortex (q 0.059), and sub-02 / sub-06 on centred maps (all parcels q 0.073, cortex q 0.098).
- **Network level and below:**
  - Network-mean maps match beyond their own null in 10 of 15 pairs. In sub-01 / sub-03, sub-01 / sub-04, sub-01 / sub-06, sub-02 / sub-03 and sub-02 / sub-06 they do not (q 0.059-0.678), so network-level agreement is largely what the repertoires' covariance gives.
  - Maps with their network means removed match beyond chance in every pair (q ≤ 0.009).
- **Amplitude:** keeping each map's amplitude (raw cosine) gives the same result in every pair (q ≤ 0.011).
- **Shared components:** removing the three largest components shared by the other four participants (43-54% of the pair's map variance) lowers the matching but leaves it beyond chance in every pair (q ≤ 0.016); on cortex alone, sub-02 / sub-06 falls short (q 0.097).
- **Run-onset-anchored states:** with 6-8 such states per participant (sub-04 has 2 and is left out), no pair passes after BH (raw p < 0.05 in 2 of 10). These sets are too small to serve as a positive control.
- **Time-locking:**
  - Matched states are weakly but reliably time-locked in every pair (mean r 0.040-0.105 against shift-null means of 0.0004-0.0017).
  - The pairs that the null matchings choose are not (relabelled locking 0.0005-0.0038).
  - Dropping run edges does not change this.
- **Timing properties:** FO agrees in 6 pairs and recurrence in 4 after BH; dwell agrees in none.
- **Ceiling:** matching between participants stays below matching within a participant. The second ceiling row selects both halves' states through the final fit, which used both halves, so its within-participant value is mildly circular; the first row has no selection step.

## Interpretation (bounded)

Per pair, the separately fitted repertoires match beyond what two repertoires with the same parcel-space covariance give by chance, and beyond network-level structure. Matched states are occupied at the same moments of the shared episodes. The test is on the repertoire, not on single states: which individual matches are real is not tested. These are 15 non-independent pairs of 6 people: no population claim, and no claim about covariance or connectivity, which the diagonal-covariance states do not model.

## Outputs and reproduce

`$SCRATCH_DIR/output/05h_participant_state_matching/atlas-4S156Parcels/` holds `pairs/{subA}_{subB}/summary.json`, `cross_subject_summary.json`, and `calibration/{state_mixing,configuration}.json`.

```bash
uv run --with pytest pytest script/tests/test_05h_participant_state_matching.py
sbatch script/05h_participant_state_matching.sh                      # needs 05g; ~12 min
MODE=calibrate CAL_NULL=state_mixing sbatch script/05h_participant_state_matching.sh
uv run --extra umap python script/fig_F2E_participant_umap.py        # Fig 2E + supplementary panels
```
