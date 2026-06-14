# Findings: 05d State Similarity Analysis

_Script: `script/05d_state_similarity.py`. Tier: SUPP (Supp)._

_Pairwise heuristic distinctness diagnostic for HMM brain states; per-subject, n=6._

## Method (as run)

- **Parcellation:** atlas-4S156Parcels (156 parcels: Schaefer 100 cortical + 56 subcortical)
- **Model:** nc=50, diagonal covariance, vt=0.95 per subject
- **Three similarity metrics computed pairwise across all K states:**
  1. Activation similarity: Pearson r of parcel-space state mean vectors
  2. Transition similarity: Pearson r of outgoing transition-probability rows
  3. Heuristic combined similarity: mean of [0,1]-normalized activation and transition Pearson r, i.e. (1+r)/2 averaged
- **Episode overlap metrics (supplementary):** binary Jaccard and FO-weighted overlap across episode sets, used only to diagnose flagged pairs
- **Flagging threshold:** combined similarity > 0.85
- **Primary analysis:** sub-HRF states excluded from flagged-pair detection (eligible_states.json from 05a); FC similarity not computed (diagonal covariance model; see 05f for empirical state-conditioned FC)
- **Sensitivity analysis:** all_states subdirectory re-runs with sub-HRF exclusion disabled
- **Scope:** per-subject diagnostic; no group statistic

## Results

### Combined similarity distribution (primary analysis, sub-HRF excluded)

| Subject | K states | Sub-HRF excluded | Pairwise comparisons (all K states) | Mean combined sim | Max combined sim | Flagged pairs (combined > 0.85) |
|---------|---------|-----------------|---------------------|------------------|-----------------|--------------------------------|
| sub-01 | 50 | 5 | 1225 | 0.4915 | 0.877 | 0 |
| sub-02 | 50 | 8 | 1225 | 0.4916 | 0.743 | 0 |
| sub-03 | 50 | 14 | 1225 | 0.4921 | 0.763 | 0 |
| sub-04 | 50 | 16 | 1225 | 0.4926 | 0.828 | 0 |
| sub-05 | 50 | 8 | 1225 | 0.4916 | 0.811 | 0 |
| sub-06 | 50 | 19 | 1225 | 0.4970 | 0.929 | 0 |

Note: for sub-01, 1 pair was skipped because both states are sub-HRF; for sub-06, 2 pairs were skipped because each involves at least one sub-HRF state.

### Activation and transition pairs above threshold (primary)

| Subject | Activation pairs (r > 0.85) | Transition pairs (r > 0.85) |
|---------|---------------------------|---------------------------|
| sub-01 | 3 | 0 |
| sub-02 | 2 | 1 |
| sub-03 | 3 | 0 |
| sub-04 | 6 | 0 |
| sub-05 | 4 | 0 |
| sub-06 | 21 | 1 |

### Flagged pairs - sensitivity analysis (all states, sub-HRF inclusion enabled)

| Subject | n flagged pairs | n possible split | n distinct |
|---------|----------------|-----------------|-----------|
| sub-01 | 1 | 1 | 0 |
| sub-02 | 0 | 0 | 0 |
| sub-03 | 0 | 0 | 0 |
| sub-04 | 0 | 0 | 0 |
| sub-05 | 0 | 0 | 0 |
| sub-06 | 2 | 2 | 0 |

### Flagged pair details (sensitivity analysis)

| Subject | State i | State j | Combined sim | Activation r | Transition r | Episode Jaccard | Diagnosis | Sub-HRF involved |
|---------|--------|--------|-------------|-------------|-------------|----------------|-----------|-----------------|
| sub-01 | 27 | 45 | 0.877 | 0.823 | 0.685 | 0.979 | possible_split | yes (both) |
| sub-06 | 0 | 21 | 0.865 | 0.617 | 0.842 | 0.951 | possible_split | yes (state 21) |
| sub-06 | 7 | 9 | 0.929 | 0.810 | 0.905 | 1.000 | possible_split | yes (state 7) |

All 3 flagged pairs (across 6 subjects) involve at least one sub-HRF state; none survive the primary (sub-HRF-excluded) analysis.

## Outputs

- output/05d_state_similarity/atlas-4S156Parcels/{sub_id}/vt0.95/similarity_summary.json
- output/05d_state_similarity/atlas-4S156Parcels/{sub_id}/vt0.95/flagged_pairs.json
- output/05d_state_similarity/atlas-4S156Parcels/{sub_id}/vt0.95/activation_similarity.npy
- output/05d_state_similarity/atlas-4S156Parcels/{sub_id}/vt0.95/transition_similarity.npy
- output/05d_state_similarity/atlas-4S156Parcels/{sub_id}/vt0.95/heuristic_combined_similarity.npy
- output/05d_state_similarity/atlas-4S156Parcels/{sub_id}/vt0.95/episode_overlap_jaccard.npy
- output/05d_state_similarity/atlas-4S156Parcels/{sub_id}/vt0.95/fo_weighted_overlap.npy
- output/05d_state_similarity/atlas-4S156Parcels/{sub_id}/vt0.95/combined_similarity_heatmap.{png,pdf}
- output/05d_state_similarity/atlas-4S156Parcels/{sub_id}/vt0.95/state_dendrogram.{png,pdf}
- output/05d_state_similarity/atlas-4S156Parcels/{sub_id}/vt0.95/all_states/ (sensitivity re-run, sub-HRF inclusion)
