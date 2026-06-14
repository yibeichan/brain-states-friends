# Findings: m10_04 Score and Decode

_Script: `script/m10_04_score_and_decode.py`. Tier: CROSS-STIM (R5, Fig 5)._

_Applies the Friends-trained sHDP-HMM to Movie10 data (via Viterbi decode + LL scoring), per subject (n=6), vt=0.95, atlas-4S156Parcels._

## Method (as run)

- Parcellation: atlas-4S156Parcels; PCA variance threshold: vt=0.95
- Friends model loaded from `output/04_combined_hdphmm/atlas-4S156Parcels/{sub_id}/final/vt0.95/best_model.pkl`
- Movie10 PCA-projected data loaded from `output/m10_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/`
- 61 runs per subject; 4 genres: Bourne (action, 10 runs), Wolf of Wall Street (drama, 17 runs), Figures (biography/drama, 24 runs), Life (documentary, 10 runs)
- Scoring: `model.score(X) / n_trs` (LL per sample); Decoding: Viterbi MAP state sequence
- Fractional occupancy (FO) computed per run over the full model state space (n_states=50)
- LL gap = Friends test LL minus movie overall LL (positive = Friends better explained)
- Baseline = log(1/K_active): a heuristic uniform-assignment reference not on the same scale as Gaussian-emission LL; `movie_above_baseline` is uninformative
- Per-subject analysis only; no cross-subject aggregation

## Results

### Model configuration and run counts

| Subject | n_states | K_active | n_pcs | n_runs | total TRs |
|---------|----------|----------|-------|--------|-----------|
| sub-01  | 50       | 42       | 75    | 61     | 24,836    |
| sub-02  | 50       | 42       | 72    | 61     | 24,891    |
| sub-03  | 50       | 42       | 72    | 61     | 24,812    |
| sub-04  | 50       | 41       | 77    | 61     | 24,832    |
| sub-05  | 50       | 41       | 67    | 61     | 24,888    |
| sub-06  | 50       | 37       | 74    | 61     | 24,872    |

### Overall log-likelihood: Friends vs. Movie10

| Subject | Friends test LL | Movie overall LL | LL gap | Run SD |
|---------|----------------|-----------------|--------|--------|
| sub-01  | -3.83          | -6.56           | +2.73  | 3.50   |
| sub-02  | -0.42          | -4.70           | +4.28  | 2.29   |
| sub-03  | -13.05         | -15.44          | +2.38  | 1.89   |
| sub-04  | -11.05         | -10.89          | -0.17  | 2.05   |
| sub-05  | -9.70          | -11.12          | +1.43  | 2.39   |
| sub-06  | -9.39          | -11.16          | +1.77  | 1.71   |

LL = log-likelihood per sample (nats). LL gap positive = Friends better explained. Sub-04 is the only subject with a negative gap (movie LL marginally exceeds Friends test LL).

### Per-genre log-likelihood

| Subject | Bourne  | Wolf    | Figures | Life    |
|---------|---------|---------|---------|---------|
| sub-01  | -6.41   | -4.05   | -5.89   | -12.66  |
| sub-02  | -6.70   | -2.31   | -4.51   | -7.27   |
| sub-03  | -15.42  | -15.68  | -14.32  | -17.73  |
| sub-04  | -12.42  | -9.63   | -10.08  | -13.47  |
| sub-05  | -11.92  | -9.59   | -11.01  | -13.25  |
| sub-06  | -9.18   | -11.69  | -10.86  | -12.95  |

Wolf shows highest LL for 4/6 subjects (sub-01, sub-02, sub-04, sub-05); Life shows lowest for all 6. Sub-06 is the exception (Bourne best); sub-03 is also an exception (Figures best at −14.32, Wolf lowest among non-Life genres at −15.68).

### Per-run LL distribution (percentiles)

| Subject | p5     | p25    | p50    | p75    | p95    | Friends test LL |
|---------|--------|--------|--------|--------|--------|----------------|
| sub-01  | -14.73 | -7.17  | -5.81  | -4.59  | -2.89  | -3.83          |
| sub-02  | -8.24  | -6.39  | -4.98  | -2.78  | -1.45  | -0.42          |
| sub-03  | -19.48 | -16.54 | -15.39 | -14.31 | -12.69 | -13.05         |
| sub-04  | -14.02 | -12.28 | -10.82 | -9.45  | -7.87  | -11.05         |
| sub-05  | -14.28 | -11.78 | -10.99 | -9.64  | -8.30  | -9.70          |
| sub-06  | -13.52 | -12.52 | -11.14 | -9.96  | -8.31  | -9.39          |

For sub-04, the Friends test LL (-11.05) falls near the median of movie run LLs (-10.82).

### State coverage in movie runs

| Subject | K_active (Friends) | States used across all movies | States/run (mean) | States/run (range) |
|---------|--------------------|-------------------------------|-------------------|--------------------|
| sub-01  | 42                 | 46                            | 36.5              | 31-41              |
| sub-02  | 42                 | 44                            | 37.1              | 30-42              |
| sub-03  | 42                 | 44                            | 38.3              | 32-43              |
| sub-04  | 41                 | 46                            | 35.3              | 31-39              |
| sub-05  | 41                 | 46                            | 37.0              | 30-41              |
| sub-06  | 37                 | 43                            | 34.9              | 29-39              |

States used across all movies exceeds K_active (Friends) for all 6 subjects, reflecting Viterbi assignments to model slots that were inactive in Friends.

## Outputs

- `output/m10_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/decoded_states.pkl` - short-key run_id -> state sequence array
- `output/m10_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/fractional_occupancy.pkl` - short-key run_id -> FO array (n_states,)
- `output/m10_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/movie_ll_summary.json` - per-run LL, per-genre LL, overall LL, run counts, baseline
- `output/m10_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/run_id_map.json` - short/long run ID crosswalk
- `output/m10_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/decoded_states_legacy_keys.pkl` - long BIDS-key version (phase-1 compat)
- `output/m10_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/fractional_occupancy_legacy_keys.pkl` - long BIDS-key version (phase-1 compat)
- `output/m10_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/ll_diagnostic.png` - per-run LL dot chart + states-per-run bar chart
