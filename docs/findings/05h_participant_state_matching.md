# Findings: do two participants have matching states? (`05h_participant_state_matching`)

## Question

Each participant's HMM was fitted separately. Do one participant's states have counterparts in another participant's repertoire, beyond what two repertoires with the same internal geometry give by chance and beyond large-scale networks? And are matched states occupied at the same moments of the episodes both participants watched?

## Method (as run)

Per participant pair (15 pairs):
- **Maps and matching.** Parcel-space contrast maps (state mean without the PCA mean), row z-scored. Rectangular Hungarian matching on 1 - r; the statistic is the median matched r.
- **Primary null.** A Haar rotation of one participant's centred maps within their own span (alternating participants), 1,000 draws. It keeps each repertoire's internal similarities and subspace.
- **Secondary null.** A rotation within blocks of 5 adjacent PCs.
- **Network level.** Network-mean maps and within-network residual maps (each map minus its network means), each with its own rotation null.
- **Amplitude.** Uncentred cosine on raw maps.
- **Shared components.** The top 3 components of the other four participants' state maps are removed, with the null re-run (k = 1 and 5 descriptive).
- **Per-match threshold.** A match passes when its r exceeds the 95th percentile of the largest matched r per null draw.
- **Time-locking.** Pearson r of matched states' binary occupancy over shared runs, against within-run circular shifts (1,000 draws). Pairs are flagged with a family-wise threshold.
  - The locking of the map matches is compared with the locking of the pairs chosen by the rotation-null matchings.
  - It is recomputed with 10 TRs dropped at each run edge.
- **Timing properties.** Rank agreement of recurrence, FO and dwell against the rotation-null matchings.
- **Ceiling.** Split-half fits.
- **Corrections.** BH across the 15 pairs for every p-family. Random streams are per pair and variant, and every pair file records its parameters and code commit.

## Results

**Eligible states, all parcels.** The p floor is 1/1001; q is BH across pairs.

| pair | states | median r | null mean | q | PC-block q | per-match passes | network-mean r (null) | within-network r (null) | raw cosine (null) | top-3 removed r (null) | variance removed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 / sub-02 | 31 x 30 | 0.726 | 0.280 | 0.0010 | 0.0010 | 22 of 30 | 0.865 (0.429) | 0.658 (0.283) | 0.711 (0.287) | 0.557 (0.288) | 0.51 |
| sub-01 / sub-03 | 31 x 26 | 0.653 | 0.305 | 0.0010 | 0.0010 | 16 of 26 | 0.816 (0.450) | 0.643 (0.302) | 0.734 (0.309) | 0.537 (0.305) | 0.51 |
| sub-01 / sub-04 | 31 x 27 | 0.600 | 0.303 | 0.0010 | 0.0010 | 12 of 27 | 0.733 (0.458) | 0.547 (0.304) | 0.635 (0.310) | 0.562 (0.303) | 0.47 |
| sub-01 / sub-05 | 31 x 29 | 0.688 | 0.284 | 0.0010 | 0.0010 | 21 of 29 | 0.881 (0.430) | 0.631 (0.285) | 0.741 (0.293) | 0.585 (0.285) | 0.49 |
| sub-01 / sub-06 | 31 x 16 | 0.684 | 0.365 | 0.0010 | 0.0010 | 7 of 16 | 0.837 (0.486) | 0.657 (0.365) | 0.728 (0.375) | 0.549 (0.345) | 0.49 |
| sub-02 / sub-03 | 30 x 26 | 0.688 | 0.301 | 0.0010 | 0.0010 | 16 of 26 | 0.829 (0.446) | 0.636 (0.303) | 0.699 (0.309) | 0.537 (0.300) | 0.54 |
| sub-02 / sub-04 | 30 x 27 | 0.619 | 0.304 | 0.0010 | 0.0010 | 12 of 27 | 0.803 (0.453) | 0.527 (0.304) | 0.644 (0.311) | 0.541 (0.301) | 0.47 |
| sub-02 / sub-05 | 30 x 29 | 0.734 | 0.278 | 0.0010 | 0.0010 | 21 of 29 | 0.869 (0.421) | 0.654 (0.290) | 0.740 (0.280) | 0.613 (0.281) | 0.50 |
| sub-02 / sub-06 | 30 x 16 | 0.691 | 0.354 | 0.0010 | 0.0010 | 7 of 16 | 0.821 (0.479) | 0.623 (0.353) | 0.710 (0.361) | 0.533 (0.332) | 0.52 |
| sub-03 / sub-04 | 26 x 27 | 0.804 | 0.318 | 0.0010 | 0.0010 | 19 of 26 | 0.838 (0.460) | 0.736 (0.320) | 0.816 (0.327) | 0.671 (0.307) | 0.45 |
| sub-03 / sub-05 | 26 x 29 | 0.756 | 0.297 | 0.0010 | 0.0010 | 17 of 26 | 0.876 (0.430) | 0.663 (0.301) | 0.794 (0.298) | 0.558 (0.296) | 0.50 |
| sub-03 / sub-06 | 26 x 16 | 0.781 | 0.377 | 0.0010 | 0.0010 | 11 of 16 | 0.869 (0.481) | 0.683 (0.375) | 0.812 (0.382) | 0.684 (0.344) | 0.49 |
| sub-04 / sub-05 | 27 x 29 | 0.626 | 0.289 | 0.0010 | 0.0010 | 13 of 27 | 0.832 (0.428) | 0.609 (0.298) | 0.642 (0.294) | 0.556 (0.290) | 0.45 |
| sub-04 / sub-06 | 27 x 16 | 0.791 | 0.394 | 0.0010 | 0.0010 | 9 of 16 | 0.869 (0.514) | 0.691 (0.391) | 0.781 (0.397) | 0.661 (0.352) | 0.43 |
| sub-05 / sub-06 | 29 x 16 | 0.732 | 0.342 | 0.0010 | 0.0010 | 9 of 16 | 0.847 (0.463) | 0.608 (0.340) | 0.775 (0.358) | 0.580 (0.339) | 0.49 |

**Checks on occupancy, same matches:**

| pair | locking r | shift null | q | time-locked (family-wise) | pointwise | relabelled-matching locking | q | run edges dropped | recurrence rho (q) | FO rho (q) | dwell rho (q) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 / sub-02 | 0.0961 | 0.0013 | 0.0010 | 24 of 30 | 26 | 0.0010 | 0.0010 | 0.0962 | -0.02 (0.509) | -0.08 (0.639) | 0.34 (0.375) |
| sub-01 / sub-03 | 0.1054 | 0.0014 | 0.0010 | 18 of 26 | 23 | 0.0014 | 0.0010 | 0.1054 | 0.67 (0.007) | 0.67 (0.007) | 0.08 (0.845) |
| sub-01 / sub-04 | 0.0517 | 0.0006 | 0.0010 | 16 of 27 | 17 | 0.0013 | 0.0010 | 0.0515 | 0.40 (0.060) | 0.46 (0.047) | 0.15 (0.845) |
| sub-01 / sub-05 | 0.0835 | 0.0010 | 0.0010 | 22 of 29 | 25 | 0.0007 | 0.0010 | 0.0837 | 0.31 (0.078) | 0.24 (0.121) | 0.21 (0.637) |
| sub-01 / sub-06 | 0.0487 | 0.0005 | 0.0010 | 10 of 16 | 12 | 0.0021 | 0.0010 | 0.0484 | 0.26 (0.162) | 0.20 (0.242) | 0.15 (0.845) |
| sub-02 / sub-03 | 0.0727 | 0.0014 | 0.0010 | 18 of 26 | 20 | 0.0013 | 0.0010 | 0.0728 | 0.29 (0.104) | 0.32 (0.097) | -0.04 (0.845) |
| sub-02 / sub-04 | 0.0573 | 0.0004 | 0.0010 | 17 of 27 | 17 | 0.0013 | 0.0010 | 0.0573 | 0.35 (0.072) | 0.50 (0.030) | -0.01 (0.845) |
| sub-02 / sub-05 | 0.0697 | 0.0009 | 0.0010 | 20 of 29 | 25 | 0.0005 | 0.0010 | 0.0696 | 0.38 (0.060) | 0.31 (0.097) | 0.18 (0.671) |
| sub-02 / sub-06 | 0.0641 | 0.0010 | 0.0010 | 8 of 16 | 12 | 0.0023 | 0.0010 | 0.0639 | 0.59 (0.030) | 0.49 (0.049) | -0.29 (0.940) |
| sub-03 / sub-04 | 0.0800 | 0.0009 | 0.0010 | 16 of 26 | 19 | 0.0022 | 0.0010 | 0.0800 | 0.61 (0.007) | 0.56 (0.025) | 0.28 (0.500) |
| sub-03 / sub-05 | 0.0835 | 0.0006 | 0.0010 | 16 of 26 | 20 | 0.0009 | 0.0010 | 0.0835 | 0.38 (0.060) | 0.38 (0.069) | 0.35 (0.375) |
| sub-03 / sub-06 | 0.0840 | 0.0011 | 0.0010 | 11 of 16 | 13 | 0.0037 | 0.0010 | 0.0838 | 0.61 (0.030) | 0.69 (0.007) | -0.15 (0.940) |
| sub-04 / sub-05 | 0.0403 | 0.0004 | 0.0010 | 14 of 27 | 18 | 0.0006 | 0.0010 | 0.0405 | 0.34 (0.071) | 0.45 (0.030) | -0.01 (0.899) |
| sub-04 / sub-06 | 0.0894 | 0.0017 | 0.0010 | 9 of 16 | 11 | 0.0040 | 0.0010 | 0.0894 | 0.36 (0.104) | 0.31 (0.152) | -0.07 (0.899) |
| sub-05 / sub-06 | 0.0466 | 0.0005 | 0.0010 | 9 of 16 | 12 | 0.0017 | 0.0010 | 0.0463 | 0.40 (0.078) | 0.34 (0.121) | -0.32 (0.940) |

**All variants:**

| variant | median r (range over pairs) | configuration-null mean | pairs with q < 0.05 |
|---|---|---|---|
| active, cortex, centred | 0.640-0.793 | 0.251-0.265 | 15 of 15 |
| active, cortex, raw | 0.630-0.799 | 0.249-0.272 | 15 of 15 |
| active, full, centred | 0.649-0.797 | 0.243-0.262 | 15 of 15 |
| active, full, raw | 0.685-0.804 | 0.242-0.268 | 15 of 15 |
| eligible, cortex, centred | 0.587-0.783 | 0.282-0.369 | 15 of 15 |
| eligible, cortex, raw | 0.576-0.808 | 0.286-0.403 | 15 of 15 |
| eligible, full, centred | 0.615-0.789 | 0.277-0.360 | 15 of 15 |
| eligible, full, raw | 0.600-0.804 | 0.278-0.394 | 15 of 15 |
| onset_control, full, raw | 0.376-0.713 | 0.347-0.461 | 6 of 10 |

**Ceiling:**

| ceiling set | within participant (half A vs half B) | across participants (half A vs half B) |
|---|---|---|
| all active states | 0.838-0.962 | 0.688-0.745 |
| half-fit states matched to the final content-eligible states | 0.853-0.966 | 0.610-0.823 |

- **Every pair matches beyond the rotation and PC-block nulls**, on all parcels, on cortex alone, on centred maps, and for all active states.
- **Network level and below:**
  - Network-mean maps match strongly against their own null.
  - Maps with their network means removed still match beyond chance in every pair, so the matching is not only network-level.
- **Amplitude:** keeping each map's amplitude (raw cosine) gives the same result.
- **Shared components:** removing the three largest components shared by the other four participants (43-54% of the pair's map variance) lowers but does not remove the matching.
- **Time-locking:**
  - Matched states are weakly but reliably time-locked.
  - The pairs that rotation-null matchings choose are not (relabelled locking 0.000-0.004).
  - Dropping run edges does not change this.
- **Timing properties:** recurrence and FO agree in some pairs after BH; dwell agrees in none.
- **Ceiling:** matching between participants stays below matching within a participant.

## Interpretation (bounded)

Per pair, the separately fitted states have counterparts with similar spatial patterns of mean activity. The similarity holds beyond network-level structure, and the counterparts are occupied at the same moments of the shared episodes. These are 15 non-independent pairs of 6 people: no population claim, and no claim about covariance or connectivity, which the diagonal-covariance states do not model.

## Outputs and reproduce

`$SCRATCH_DIR/output/05h_participant_state_matching/atlas-4S156Parcels/` holds `pairs/{subA}_{subB}/summary.json` and `cross_subject_summary.json`.

```bash
uv run python -m pytest script/tests/test_05h_participant_state_matching.py
sbatch script/05h_participant_state_matching.sh          # needs 05g; ~8 min
uv run python script/fig_F2E_participant_umap.py         # Fig 2E + supplementary panels
```
