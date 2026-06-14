# Findings: 05e_a4 State Flag Synthesis

_Script: `script/05e_temporal_trend_a4.py`. Tier: MAIN (Methods (state classification))._

_Synthesizes per-state temporal tags from a1, a2, and a3 into a unified eligibility classification; per-subject, n=6._

## Method (as run)

- Parcellation: atlas-4S156Parcels; VT=0.95; alpha=0.05 (FDR threshold for a1/a3 q-values); recurrence_floor=0.10
- n_states=50 per subject (truncation capacity)
- Inputs: 05a recurrence scores + eligible_states.json (sub-HRF exclusions), plus a1 temporal_trend_metrics.csv, a2 temporal_position_metrics.csv, a3 habituation_metrics.csv; all three upstream sources available for all 6 subjects
- Tags are non-exclusive boolean flags; summary categories are mutually exclusive (priority order: unused > low_confidence > run_onset_anchored > season_temporal > eligible_for_content_analysis > rare)
- Session-trend tags (session_trend_down, session_trend_up) are informational only and do not drive category membership; within-session drift is addressed via detrended FO from a3
- season_temporal requires co-occurrence of season_structured AND global_trend; season-only variation is not treated as temporal drift

## Results

### Tag counts per subject

| Subject | sub_hrf | unused | run_onset | a_anchored | b_anchored | session_trend_down | session_trend_up | season_structured | global_trend |
|---------|---------|--------|-----------|------------|------------|--------------------|------------------|-------------------|--------------|
| sub-01  | 5       | 4      | 5         | 1          | 1          | 5                  | 5                | 8                 | 1            |
| sub-02  | 8       | 4      | 4         | 4          | 0          | 7                  | 8                | 7                 | 3            |
| sub-03  | 14      | 6      | 2         | 4          | 0          | 8                  | 7                | 10                | 3            |
| sub-04  | 16      | 6      | 1         | 1          | 0          | 21                 | 12               | 1                 | 1            |
| sub-05  | 8       | 3      | 4         | 1          | 2          | 3                  | 2                | 27                | 1            |
| sub-06  | 19      | 8      | 3         | 4          | 0          | 14                 | 14               | 20                | 0            |

### Summary category counts per subject

| Subject | unused | low_confidence | run_onset_anchored | season_temporal | eligible_for_content_analysis | rare |
|---------|--------|----------------|--------------------|-----------------|-------------------------------|------|
| sub-01  | 4      | 5              | 7                  | 1               | 31                            | 2    |
| sub-02  | 4      | 5              | 8                  | 0               | 30                            | 3    |
| sub-03  | 6      | 10             | 6                  | 2               | 26                            | 0    |
| sub-04  | 6      | 13             | 2                  | 1               | 27                            | 1    |
| sub-05  | 3      | 8              | 7                  | 1               | 29                            | 2    |
| sub-06  | 8      | 18             | 7                  | 0               | 16                            | 1    |

### Eligible states and excluded fraction per subject

States in the eligible_for_content_analysis category are used by downstream script 08d. The remaining states are excluded or treated as informational controls.

| Subject | Eligible | % of 50 | Excluded (unused + low_conf + run_onset + season_temp + rare) |
|---------|----------|---------|---------------------------------------------------------------|
| sub-01  | 31       | 62%     | 19                                                            |
| sub-02  | 30       | 60%     | 20                                                            |
| sub-03  | 26       | 52%     | 24                                                            |
| sub-04  | 27       | 54%     | 23                                                            |
| sub-05  | 29       | 58%     | 21                                                            |
| sub-06  | 16       | 32%     | 34                                                            |

## Outputs

- output/05e_temporal_trend_a4/atlas-4S156Parcels/sub-*/vt0.95/state_flags.csv - per-state tags, summary category, dominant network, lme_slope, early_fraction_a/b, dr2_season
- output/05e_temporal_trend_a4/atlas-4S156Parcels/sub-*/vt0.95/state_flags_summary.json - tag counts, category counts, thresholds, source availability, detrended FO path
- output/05e_temporal_trend_a4/atlas-4S156Parcels/sub-*/vt0.95/state_flag_overview.png/pdf - binary heatmap of tags x states (sorted by recurrence)
