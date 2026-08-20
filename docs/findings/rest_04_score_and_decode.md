# Findings: rest_04 Score and Decode Resting State

_Script: `script/rest_04_score_and_decode.py`. Tier: CROSS-STIM (R5 extension, Fig S8)._

_Applies each participant's Friends-fit HMM to the projected resting-state components without re-fitting, and reports model fit and per-run fractional occupancy; per-subject, n=6._

## Method (as run)

- Parcellation: atlas-4S156Parcels; vt=0.95
- Inputs: projected resting components from `rest_03`; Friends-fit HMM from `04_combined_hdphmm` (`final/vt0.95`)
- The emission and transition parameters are reused unchanged. Nothing is re-fit or fine-tuned, so a resting state index refers to the same emission distribution as the corresponding Friends state index.
- Per-TR Viterbi decoding assigns every resting TR to one of the participant's active states. Viterbi forces an assignment, so occupancy is not evidence that a state is genuinely expressed; it is the input to the correspondence test in `rest_05`.
- Fractional occupancy is computed per run over the participant's active states, matching the within-Friends definition in `05a`.
- `baseline_ll_per_sample` = log(1/n_active_states), a uniform state-assignment reference. It is a heuristic anchor only and is not on the same scale as a Gaussian-emission HMM log-likelihood; treat "above baseline" as a coarse sanity flag, not a test.

## Results

### Log-likelihood and decoding summary

| Subject | n_active | rest LL/sample | LL/run SD | rest TRs | rest runs | above baseline |
|---------|----------|----------------|-----------|----------|-----------|----------------|
| sub-01 | 42 | −19.809 | 1.517 | 3000 | 5 | False |
| sub-02 | 42 | −0.916 | 1.073 | 3000 | 5 | True |
| sub-03 | 42 | −16.515 | 4.172 | 3000 | 5 | False |
| sub-04 | 41 | −12.309 | 0.963 | 3000 | 5 | False |
| sub-05 | 41 | −12.995 | 1.487 | 2400 | 4 | False |
| sub-06 | 37 | −11.377 | 1.436 | 3600 | 6 | False |

Active-state counts (42, 42, 42, 41, 41, 37) are inherited from the primary Friends model (`final_refit.n_active_states`) and are not re-derived from resting data.

Only sub-02 clears the uniform-assignment reference. Resting log-likelihood also varies far more across participants than the Friends test likelihood does, and sub-01 is an extreme case: its resting LL of −19.809 against a Friends test LL of −3.829 is the widest Friends-to-condition gap of any dataset in the project.

This is the point at which the resting-state comparison diverges from the narrative stimuli. The spatial subspace transferred cleanly (`rest_03`, transfer gaps −0.011 to +0.007), but the fitted temporal model does not describe resting dynamics with comparable quality. The dissociation between those two facts is the substantive result carried into `rest_05` and the manuscript.

## Caveats

- The uniform-assignment baseline is heuristic. A single participant clearing it is weak evidence and should not be read as five participants failing a test.
- Per-run LL standard deviations are computed over 4–6 runs, so they describe spread rather than supporting inference.
- Resting runs are collected across sessions and carry strong serial dependence (`rest_05` reports lag-1 FO autocorrelation of 0.57–0.88, with roughly 2 effective independent observations per participant). Run count is not sample size here.

## Outputs

- `output/rest_04_decoded/atlas-4S156Parcels/<sub>/vt0.95/rest_ll_summary.json` — the table above
- `output/rest_04_decoded/atlas-4S156Parcels/<sub>/vt0.95/decoded_states.pkl` — per-run Viterbi sequences
- `output/rest_04_decoded/atlas-4S156Parcels/<sub>/vt0.95/fractional_occupancy.pkl` — per-run, per-state FO
- `output/rest_04_decoded/atlas-4S156Parcels/<sub>/vt0.95/run_id_map.json`
- `output/rest_04_decoded/atlas-4S156Parcels/<sub>/vt0.95/ll_diagnostic.png`
