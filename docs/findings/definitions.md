# State definitions

Terms that select or count HMM states. Each has one meaning across the code, the findings docs and the supplementary material. The thresholds are named constants in `utils.recurrence_utils` (`ACTIVE_FO_THRESHOLD`, `USED_STATE_THRESHOLD`). The model fit takes its usage cut from config `min_state_usage`, and the figure scripts restate 0.02 as a literal; tests check that both agree with the constants. The held-out `_05` stages stop if their `--fo_threshold` differs from the one 05a recorded.

| Term | Definition | Code | Used by |
|---|---|---|---|
| Fractional occupancy (FO) | Share of a run's TRs that the Viterbi path assigns to a state | `utils.recurrence_utils.compute_fractional_occupancy` | Everything below |
| Active state | FO > 0.02 in at least one run (about 14 s of a 12-minute run). The set of active states is a participant's active repertoire; K_active is its size (42–47) | `ACTIVE_FO_THRESHOLD`, `active_state_indices()` | 05a and everything built on it, 04ra/04rb recurrence and Hungarian matching, held-out presence and coverage (`_05` stages, `fig_S08`, `fig_F4`) |
| Recurrence score | Fraction of runs in which a state is active | `compute_recurrence_scores()` | 05a, 04ra/04rb, `_05` stages |
| Present in a held-out stimulus | FO > 0.02 in at least one run of that stimulus, the same rule as active in Friends | `states_present_in_runs()` | `_05` A4 check, `fig_S08` Panel C |
| K_used | Number of states whose posterior usage at the last EM iteration exceeds 0.01, over the training and validation data (37–42). A property of the fitted model, not of the decoded sequence. Older output keys use "active" for both counts (see notes) | `USED_STATE_THRESHOLD`, config `min_state_usage`, `used_state_indices()`, `utils.hdphmm.infer_n_active_states` | Configuration choice (validation likelihood against K_used, Figure S3), log(1/K_used) baselines, seed and fold state counts with their transition entropy and self-transition probability (04ra/04rb, Figure S4B), ICA dimensionality |
| Content-eligible state | Active state that passes the 05e_a4 screen (recurrence ≥ 0.10, median dwell ≥ 3 TRs, not run-onset- or drift-anchored) | `05e_temporal_trend_a4.py`, `load_content_eligibility()` | Content decoding (08 stages), network-composition summaries, Table S3 |

Notes:

- Viterbi decoding assigns every TR to one of all 50 model states. Active and used are labels applied afterwards; decoding is not restricted to either set.
- Two counts are routinely compared and are not interchangeable: K_used (37–42) and K_active (42–47). K_active can exceed K_used because a state can pass 0.02 in a single run while its usage pooled over all data stays below 0.01.
- Older JSON keys use "active" for both counts. They hold K_used in the 04_combined fit outputs (`n_active_states`), `04_patch_selection_metrics` (`K_active`), the 04ra field `k_used_reported` (`k_active_reported` in outputs written before it was renamed) and the supplements ICA outputs (`K_active`). They hold the active-state count in 04ra `k_active`, 04rb `K_active` and 05a `n_active_states`. 04ra and 04rb also write `k_used` / `K_used` explicitly.
- Filters that are not state labels keep their own constants: the 30-TR minimum for within-state FC (`05f --min_trs`), the median-dwell rule for low-confidence states (`SUB_HRF_DWELL_THRESHOLD_TR`), and the pooled-FO ≥ 0.01 sample-size filter for cross-stimulus decoding (`utils.transformer_analysis.INTERSECTION_MIN_FO`).
