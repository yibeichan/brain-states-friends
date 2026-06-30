# Findings: pp_04 Score and Decode

_Script: `script/pp_04_score_and_decode.py`. Tier: CROSS-STIM (R5, Fig 5)._

_Applies the Friends-trained HMM to Petit Prince audiobook runs projected through the Friends PCA; computes per-run log-likelihood and Viterbi state sequences. Per-subject, n=5 (sub-04 absent from PP dataset)._

## Method (as run)

- Parcellation: atlas-4S156Parcels; vt=0.95
- Friends model loaded from `output/04_combined_hdphmm/atlas-4S156Parcels/{sub_id}/final/vt0.95/best_model.pkl`
- PP projected data loaded from `output/pp_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/`
- Each run scored via `model.score(X)` (total LL divided by n_TRs) and decoded via Viterbi (`model.decode(X)`)
- Overall PP LL is a TR-weighted average across all runs; per-type LL groups runs by language (lppFR, lppEN)
- Baseline LL = log(1/n_active_states); this is a heuristic lower bound, not on the same scale as Gaussian-emission LL
- sub-06 has 16 runs (7 FR, 9 EN); all others have 18 runs (9 FR, 9 EN)
- No group statistic; all results are per-subject

## Results

### Overall LL and Model Parameters

| Subject | n_states | n_active | n_pcs | Friends test LL | PP overall LL | LL gap | Per-run SD | PP > baseline |
|---------|----------|----------|-------|----------------|---------------|--------|-----------|---------------|
| sub-01  | 50       | 42       | 75    | -3.83          | -11.93        | +8.10  | 3.46      | No            |
| sub-02  | 50       | 42       | 72    | -0.42          | -0.88         | +0.46  | 1.33      | Yes           |
| sub-03  | 50       | 42       | 72    | -13.05         | -18.12        | +5.06  | 2.96      | No            |
| sub-05  | 50       | 41       | 67    | -9.70          | -18.47        | +8.77  | 4.93      | No            |
| sub-06  | 50       | 37       | 74    | -9.39          | -14.94        | +5.54  | 2.80      | No            |

LL gap = Friends test LL minus PP overall LL (positive = Friends better explained). LL values are nats per TR. Per-run SD is unweighted std across individual runs.

### Per-Language LL

| Subject | FR LL  | FR runs | FR TRs | EN LL  | EN runs | EN TRs |
|---------|--------|---------|--------|--------|---------|--------|
| sub-01  | -11.94 | 9       | 3990   | -11.93 | 9       | 3808   |
| sub-02  | -0.68  | 9       | 3990   | -1.09  | 9       | 3808   |
| sub-03  | -19.70 | 9       | 3990   | -16.46 | 9       | 3808   |
| sub-05  | -18.01 | 9       | 3990   | -18.95 | 9       | 3808   |
| sub-06  | -15.25 | 7       | 3133   | -14.68 | 9       | 3808   |

### Per-Run LL Distribution Summary

| Subject | n_runs | Total TRs | Min LL | Max LL | Mean LL | SD    |
|---------|--------|-----------|--------|--------|---------|-------|
| sub-01  | 18     | 7798      | -16.52 | -5.48  | -11.92  | 3.46  |
| sub-02  | 18     | 7798      | -4.14  | +1.66  | -0.85   | 1.33  |
| sub-03  | 18     | 7798      | -24.44 | -12.88 | -18.07  | 2.96  |
| sub-05  | 18     | 7798      | -25.01 | -10.81 | -18.41  | 4.93  |
| sub-06  | 16     | 6941      | -19.38 | -8.26  | -14.82  | 2.80  |

### Active States per PP Run (Viterbi-decoded)

| Subject | Friends K_active | PP min active | PP max active | PP mean active |
|---------|-----------------|--------------|--------------|---------------|
| sub-01  | 42              | 28           | 39           | 34.4          |
| sub-02  | 42              | 27           | 36           | 31.7          |
| sub-03  | 42              | 27           | 37           | 32.6          |
| sub-05  | 41              | 28           | 40           | 34.0          |
| sub-06  | 37              | 30           | 38           | 34.4          |

Active states per run = number of distinct state indices appearing in the Viterbi sequence for that run.

## Outputs

- output/pp_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/pp_ll_summary.json
- output/pp_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/decoded_states.pkl
- output/pp_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/fractional_occupancy.pkl
- output/pp_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/decoded_states_legacy_keys.pkl
- output/pp_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/fractional_occupancy_legacy_keys.pkl
- output/pp_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/run_id_map.json
- output/pp_04_decoded/atlas-4S156Parcels/{sub_id}/vt0.95/ll_diagnostic.png
