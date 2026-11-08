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
| sub-01 / sub-02 | 31 x 30 | 0.726 | 0.593 | 0.0012 | 0.522 | 0.0010 | 0.865 (0.806) | 0.658 (0.502) | 0.711 (0.616) | 0.544 (0.462) | 0.51 |
| sub-01 / sub-03 | 31 x 26 | 0.653 | 0.573 | 0.0012 | 0.529 | 0.0010 | 0.816 (0.785) | 0.643 (0.502) | 0.734 (0.599) | 0.535 (0.463) | 0.51 |
| sub-01 / sub-04 | 31 x 27 | 0.600 | 0.528 | 0.0023 | 0.499 | 0.0010 | 0.733 (0.745) | 0.547 (0.450) | 0.635 (0.556) | 0.566 (0.465) | 0.47 |
| sub-01 / sub-05 | 31 x 29 | 0.688 | 0.565 | 0.0012 | 0.505 | 0.0010 | 0.881 (0.776) | 0.631 (0.470) | 0.741 (0.592) | 0.581 (0.449) | 0.49 |
| sub-01 / sub-06 | 31 x 15 | 0.701 | 0.593 | 0.0012 | 0.545 | 0.0010 | 0.848 (0.795) | 0.654 (0.527) | 0.716 (0.626) | 0.556 (0.466) | 0.50 |
| sub-02 / sub-03 | 30 x 26 | 0.688 | 0.581 | 0.0012 | 0.528 | 0.0010 | 0.829 (0.783) | 0.636 (0.519) | 0.699 (0.608) | 0.522 (0.456) | 0.54 |
| sub-02 / sub-04 | 30 x 27 | 0.619 | 0.541 | 0.0050 | 0.500 | 0.0010 | 0.803 (0.745) | 0.527 (0.464) | 0.644 (0.573) | 0.549 (0.473) | 0.47 |
| sub-02 / sub-05 | 30 x 29 | 0.734 | 0.571 | 0.0012 | 0.502 | 0.0010 | 0.869 (0.774) | 0.654 (0.483) | 0.740 (0.606) | 0.605 (0.463) | 0.51 |
| sub-02 / sub-06 | 30 x 15 | 0.678 | 0.594 | 0.0032 | 0.534 | 0.0010 | 0.839 (0.800) | 0.623 (0.525) | 0.707 (0.641) | 0.540 (0.480) | 0.53 |
| sub-03 / sub-04 | 26 x 27 | 0.804 | 0.550 | 0.0012 | 0.486 | 0.0010 | 0.838 (0.757) | 0.736 (0.492) | 0.816 (0.582) | 0.662 (0.478) | 0.45 |
| sub-03 / sub-05 | 26 x 29 | 0.756 | 0.559 | 0.0012 | 0.511 | 0.0010 | 0.876 (0.766) | 0.663 (0.501) | 0.794 (0.589) | 0.594 (0.442) | 0.50 |
| sub-03 / sub-06 | 26 x 15 | 0.819 | 0.610 | 0.0012 | 0.532 | 0.0010 | 0.873 (0.783) | 0.731 (0.566) | 0.827 (0.645) | 0.703 (0.508) | 0.50 |
| sub-04 / sub-05 | 27 x 29 | 0.626 | 0.519 | 0.0012 | 0.485 | 0.0010 | 0.832 (0.718) | 0.609 (0.453) | 0.642 (0.546) | 0.561 (0.459) | 0.45 |
| sub-04 / sub-06 | 27 x 15 | 0.795 | 0.585 | 0.0012 | 0.506 | 0.0010 | 0.865 (0.791) | 0.738 (0.518) | 0.793 (0.628) | 0.681 (0.509) | 0.43 |
| sub-05 / sub-06 | 29 x 15 | 0.702 | 0.565 | 0.0012 | 0.511 | 0.0010 | 0.826 (0.783) | 0.612 (0.478) | 0.772 (0.626) | 0.575 (0.456) | 0.50 |

**Checks on occupancy, same matches:**

| pair | locking r | shift null | q | time-locked (family-wise) | pointwise | relabelled-matching locking | q | run edges dropped | recurrence rho (q) | FO rho (q) | dwell rho (q) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 / sub-02 | 0.0961 | 0.0013 | 0.0010 | 24 of 30 | 26 | 0.0013 | 0.0010 | 0.0962 | -0.02 (0.552) | -0.08 (0.673) | 0.34 (0.322) |
| sub-01 / sub-03 | 0.1054 | 0.0014 | 0.0010 | 18 of 26 | 23 | 0.0013 | 0.0010 | 0.1054 | 0.67 (0.010) | 0.67 (0.007) | 0.08 (0.835) |
| sub-01 / sub-04 | 0.0517 | 0.0006 | 0.0010 | 16 of 27 | 17 | 0.0012 | 0.0010 | 0.0515 | 0.40 (0.060) | 0.46 (0.018) | 0.15 (0.835) |
| sub-01 / sub-05 | 0.0835 | 0.0010 | 0.0010 | 22 of 29 | 25 | 0.0009 | 0.0010 | 0.0837 | 0.31 (0.067) | 0.24 (0.115) | 0.21 (0.544) |
| sub-01 / sub-06 | 0.0520 | 0.0007 | 0.0010 | 10 of 15 | 12 | 0.0027 | 0.0010 | 0.0516 | 0.10 (0.360) | 0.13 (0.320) | 0.13 (0.835) |
| sub-02 / sub-03 | 0.0727 | 0.0014 | 0.0010 | 18 of 26 | 20 | 0.0015 | 0.0010 | 0.0728 | 0.29 (0.081) | 0.32 (0.064) | -0.04 (0.835) |
| sub-02 / sub-04 | 0.0573 | 0.0004 | 0.0010 | 17 of 27 | 17 | 0.0012 | 0.0010 | 0.0573 | 0.35 (0.067) | 0.50 (0.018) | -0.01 (0.835) |
| sub-02 / sub-05 | 0.0697 | 0.0009 | 0.0010 | 20 of 29 | 25 | 0.0007 | 0.0010 | 0.0696 | 0.38 (0.060) | 0.31 (0.064) | 0.18 (0.544) |
| sub-02 / sub-06 | 0.0672 | 0.0010 | 0.0010 | 8 of 15 | 10 | 0.0027 | 0.0010 | 0.0669 | 0.54 (0.060) | 0.52 (0.054) | -0.29 (0.850) |
| sub-03 / sub-04 | 0.0800 | 0.0009 | 0.0010 | 16 of 26 | 19 | 0.0019 | 0.0010 | 0.0800 | 0.61 (0.007) | 0.56 (0.015) | 0.28 (0.544) |
| sub-03 / sub-05 | 0.0835 | 0.0006 | 0.0010 | 16 of 26 | 20 | 0.0010 | 0.0010 | 0.0835 | 0.38 (0.062) | 0.38 (0.054) | 0.35 (0.322) |
| sub-03 / sub-06 | 0.0902 | 0.0014 | 0.0010 | 11 of 15 | 13 | 0.0041 | 0.0010 | 0.0901 | 0.79 (0.007) | 0.81 (0.007) | 0.40 (0.544) |
| sub-04 / sub-05 | 0.0403 | 0.0004 | 0.0010 | 14 of 27 | 18 | 0.0005 | 0.0010 | 0.0405 | 0.34 (0.067) | 0.45 (0.037) | -0.01 (0.850) |
| sub-04 / sub-06 | 0.0953 | 0.0019 | 0.0010 | 9 of 15 | 10 | 0.0046 | 0.0010 | 0.0954 | 0.46 (0.067) | 0.44 (0.064) | -0.07 (0.850) |
| sub-05 / sub-06 | 0.0491 | 0.0007 | 0.0010 | 9 of 15 | 11 | 0.0022 | 0.0010 | 0.0489 | 0.33 (0.119) | 0.31 (0.115) | -0.31 (0.920) |

**All variants:**

| variant | median r (range over pairs) | state-mixing null mean | pairs with q < 0.05 | PC-block null mean | PC-block pairs with q < 0.05 |
|---|---|---|---|---|---|
| active, cortex, centred | 0.640-0.793 | 0.533-0.579 | 15 of 15 | 0.525-0.572 | 15 of 15 |
| active, cortex, raw | 0.630-0.799 | 0.533-0.582 | 15 of 15 | 0.527-0.574 | 15 of 15 |
| active, full, centred | 0.649-0.797 | 0.537-0.577 | 15 of 15 | 0.495-0.546 | 15 of 15 |
| active, full, raw | 0.685-0.804 | 0.537-0.576 | 15 of 15 | 0.498-0.546 | 15 of 15 |
| eligible, cortex, centred | 0.587-0.776 | 0.521-0.599 | 14 of 15 | 0.518-0.570 | 14 of 15 |
| eligible, cortex, raw | 0.576-0.810 | 0.512-0.620 | 14 of 15 | 0.510-0.572 | 15 of 15 |
| eligible, full, centred | 0.625-0.811 | 0.526-0.594 | 14 of 15 | 0.492-0.541 | 15 of 15 |
| eligible, full, raw | 0.600-0.819 | 0.519-0.610 | 15 of 15 | 0.485-0.545 | 15 of 15 |
| onset_control, full, raw | 0.376-0.713 | 0.431-0.588 | 0 of 10 | — | — |

**Ceiling:**

| ceiling set | within participant (half A vs half B) | across participants (half A vs half B) |
|---|---|---|
| all active states | 0.838-0.962 | 0.688-0.745 |
| half-fit states matched to the final content-eligible states | 0.853-0.963 | 0.610-0.836 |

- **Pair test.** On eligible states and all parcels, every pair matches beyond the state-mixing null (q 0.0012-0.0050) and the PC-block null. The state-mixing null mean is 0.52-0.61, against observed medians of 0.60-0.82: much of the matched r comes from covariance the repertoires share, and the excess is 0.07-0.25.
- **Other variants.** All 15 pairs pass for all active states in every scope. For eligible states, three variants each miss one pair: sub-01 / sub-04 on cortex (q 0.059), and sub-02 / sub-06 on centred maps (all parcels q 0.057, cortex q 0.151).
- **Network level and below:**
  - Network-mean maps match beyond their own null in 8 of 15 pairs. In sub-01 / sub-03, sub-01 / sub-04, sub-01 / sub-06, sub-02 / sub-03, sub-02 / sub-04, sub-02 / sub-06 and sub-05 / sub-06 they do not (q 0.055-0.678), so network-level agreement is largely what the repertoires' covariance gives.
  - Maps with their network means removed match beyond chance in every pair (q ≤ 0.009).
- **Amplitude:** keeping each map's amplitude (raw cosine) gives the same result in every pair (q ≤ 0.017).
- **Shared components:** removing the three largest components shared by the other four participants (43-54% of the pair's map variance) lowers the matching but leaves it beyond chance in every pair (q ≤ 0.014); on cortex alone, sub-02 / sub-06 falls short (q 0.076).
- **Run-onset-anchored states:** with 6-8 such states per participant (sub-04 has 2 and is left out), no pair passes after BH (raw p < 0.05 in 2 of 10). These sets are too small to serve as a positive control.
- **Time-locking:**
  - Matched states are weakly but reliably time-locked in every pair (mean r 0.040-0.105 against shift-null means of 0.0004-0.0019).
  - The pairs that the null matchings choose are not (relabelled locking 0.0005-0.0046).
  - Dropping run edges does not change this.
- **Timing properties:** FO agrees in 6 pairs and recurrence in 3 after BH; dwell agrees in none.
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
