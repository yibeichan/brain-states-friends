# Findings: hp_04 Score and Decode

_Script: `script/hp_04_score_and_decode.py`. Tier: CROSS-STIM (R5, Fig 5)._

_Applies the Friends-trained HMM to Harry Potter reading-task data (7 runs/subject, 5 subjects; no sub-04) to compute per-run log-likelihoods and Viterbi-decoded state sequences._

## Method (as run)

- Parcellation: atlas-4S156Parcels; vt=0.95; 5 subjects (sub-01, sub-02, sub-03, sub-05, sub-06; sub-04 has no HP data)
- Input: Friends-trained HMM from 04_combined_hdphmm (select mode); HP data projected through Friends PCA from hp_03
- 7 HP runs per subject (3,363 total TRs per subject; run lengths 373-566 TRs)
- Per-run: model.score() for LL/sample (total LL / n_TRs); model.decode() Viterbi for state sequence
- Overall HP LL: TR-weighted average across runs
- Baseline: log(1/K_active), heuristic only, not on the same scale as Gaussian-emission LL
- Outputs: decoded_states.pkl (short run-id keys, 08c-compatible), fractional_occupancy.pkl, hp_ll_summary.json

## Results

### Model Configuration and HP Data Volume

| Subject | K_total | K_active | n_PCs | HP runs | HP TRs |
|---------|---------|----------|-------|---------|--------|
| sub-01  | 50      | 42       | 75    | 7       | 3363   |
| sub-02  | 50      | 42       | 72    | 7       | 3363   |
| sub-03  | 50      | 42       | 72    | 7       | 3363   |
| sub-05  | 50      | 41       | 67    | 7       | 3363   |
| sub-06  | 50      | 37       | 74    | 7       | 3363   |

### Log-Likelihood: Friends Test vs HP Transfer

LL gap = Friends test LL minus HP LL. Positive = Friends better explained; negative = HP better explained.

| Subject | Friends test LL/sample | HP overall LL/sample | LL gap | Per-run SD | HP > baseline |
|---------|------------------------|----------------------|--------|------------|---------------|
| sub-01  | -3.83                  | -5.71                | +1.88  | 3.57       | No            |
| sub-02  | -0.42                  | +0.11                | -0.53  | 1.02       | Yes           |
| sub-03  | -13.05                 | -10.99               | -2.06  | 1.06       | No            |
| sub-05  | -9.70                  | -10.39               | +0.70  | 0.88       | No            |
| sub-06  | -9.39                  | -8.12                | -1.27  | 2.04       | No            |

3 of 5 subjects show negative LL gaps (HP LL higher than Friends test LL). Sub-02 is the only subject where HP LL is above the heuristic baseline.

### Per-Run Log-Likelihood (LL/sample)

| Subject | run-1  | run-2  | run-3  | run-4  | run-5  | run-6   | run-7  |
|---------|--------|--------|--------|--------|--------|---------|--------|
| sub-01  | -3.93  | -2.50  | -5.55  | -1.75  | -7.04  | -12.34  | -6.89  |
| sub-02  | +0.50  | +1.49  | -1.14  | +0.14  | -1.34  | +0.29   | +0.77  |
| sub-03  | -12.69 | -9.77  | -11.61 | -9.75  | -11.52 | -11.21  | -10.64 |
| sub-05  | -10.31 | -10.05 | -11.23 | -11.04 | -10.10 | -11.50  | -8.95  |
| sub-06  | -5.29  | -9.15  | -10.58 | -10.50 | -6.22  | -8.49   | -7.35  |

Run lengths: run-3 = 373 TRs (shortest); run-5 = 566 TRs (longest); others 453-536 TRs. Sub-01 run-6 is a clear outlier at -12.34 vs. mean -4.61 for other sub-01 runs.

### Active States in HP Sequences

K active across all HP runs and K active per run (mean and range) via Viterbi decoding.

| Subject | K_active (Friends model) | K used across all HP | Mean states/run | Range states/run |
|---------|--------------------------|----------------------|-----------------|------------------|
| sub-01  | 42                       | 44                   | 32.7            | 25-40            |
| sub-02  | 42                       | 41                   | 33.4            | 30-37            |
| sub-03  | 42                       | 42                   | 34.6            | 29-38            |
| sub-05  | 41                       | 45                   | 34.9            | 33-37            |
| sub-06  | 37                       | 40                   | 33.1            | 31-37            |

Sub-01, sub-05, and sub-06 activate more states across all HP runs than the Friends K_active count, meaning a small number of near-zero-probability states appear in Viterbi paths.

## Outputs

- output/hp_04_decoded/atlas-4S156Parcels/sub-*/vt0.95/decoded_states.pkl (short run-id keys, 08c-compatible)
- output/hp_04_decoded/atlas-4S156Parcels/sub-*/vt0.95/fractional_occupancy.pkl
- output/hp_04_decoded/atlas-4S156Parcels/sub-*/vt0.95/decoded_states_legacy_keys.pkl (BIDS long keys, for hp_05 compatibility)
- output/hp_04_decoded/atlas-4S156Parcels/sub-*/vt0.95/fractional_occupancy_legacy_keys.pkl
- output/hp_04_decoded/atlas-4S156Parcels/sub-*/vt0.95/run_id_map.json (short-to-long key mapping)
- output/hp_04_decoded/atlas-4S156Parcels/sub-*/vt0.95/hp_ll_summary.json (per-run LL, overall LL, baselines)
- output/hp_04_decoded/atlas-4S156Parcels/sub-*/vt0.95/ll_diagnostic.png (per-run LL dot chart + states/run bar chart)
