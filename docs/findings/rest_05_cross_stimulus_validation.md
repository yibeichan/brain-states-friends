# Findings: rest_05 Cross-Stimulus Validation (Resting State)

_Script: `script/rest_05_cross_stimulus_validation.py`. Tier: CROSS-STIM (R5 extension, Fig S8)._

_Tests whether the Friends recurrence ordering corresponds to state occupancy during task-free rest, where there is no external content stream; per-subject, n=6._

## Method (as run)

- Parcellation: atlas-4S156Parcels; vt=0.95; fo_threshold=0.01; exclude_sub_hrf=False
- Inputs: Friends recurrence scores from `05a`; resting FO and LL from `rest_04`; PCA diagnostic from `rest_03`
- n=6 subjects. Resting state is the only out-of-Friends condition covering all six participants.
- Resting runs: 5, 5, 5, 5, 4, 6 for sub-01 through sub-06
- Active states (Friends recurrence > 0): 46, 46, 44, 44, 47, 42
- Content-eligible states (`05e_a4` state_flags.csv): 31, 30, 26, 27, 29, 16
- A1: Spearman rho between Friends recurrence score and mean resting FO across all active states. p-values are approximate: FO is compositional (sums to 1 per run) and resting runs carry serial dependence, so p is an upper bound on confidence.
- A1 eligible: same analysis restricted to content-eligible states
- A3: LL per sample, Friends test vs resting overall, against the heuristic log(1/n_active) reference
- A4: recurrence vs resting coverage Spearman, where coverage is the fraction of resting runs with state FO > 0.01
- A5: PCA transfer R², Friends training vs rest, network-stratified
- B2: bootstrap reference, 1000 subsamples of Movie10 runs matched to the resting run count, positioning the resting rho against that distribution
- C1: within-run drift, first-third vs last-third FO L1 shift per run
- Serial dependence: lag-1 FO autocorrelation between consecutive resting runs
- **No surrogate null was run for rest.** All values here are descriptive. The phase-randomized null used for Movie10 (see `m10_05` and the R5 null) was not extended to resting state.

## Results

### A1: Recurrence-FO correlation (full repertoire)

| Subject | n_active | Spearman rho | p (approx) | positive |
|---------|----------|--------------|------------|----------|
| sub-01 | 46 | −0.289 | 0.051 | False |
| sub-02 | 46 | 0.324 | 0.028 | True |
| sub-03 | 44 | 0.138 | 0.371 | False |
| sub-04 | 44 | 0.099 | 0.524 | False |
| sub-05 | 47 | 0.274 | 0.062 | False |
| sub-06 | 42 | 0.445 | 0.003 | True |

Correspondence is weak and inconsistent in sign. Two of six participants reach uncorrected p < 0.05 (sub-02, sub-06); sub-01 is negative at −0.289. Mean rho across participants is 0.165.

Note for interpretation: 0.165 is **not** below the Petit Prince mean (0.101). Raw rho carries a stimulus-independent covariance and stationarity floor, established by the R5 phase-randomized null, so the ordering of rho among the weakly corresponding conditions is confounded with fit quality, language, and modality. Do not read rest as a specificity floor or as the low end of a modality gradient.

### A1 eligible: Recurrence-FO correlation (content-eligible states)

| Subject | n_eligible | Spearman rho | p (approx) |
|---------|-----------|--------------|------------|
| sub-01 | 31 | −0.167 | 0.370 |
| sub-02 | 30 | 0.287 | 0.124 |
| sub-03 | 26 | 0.129 | 0.530 |
| sub-04 | 27 | 0.023 | 0.908 |
| sub-05 | 29 | 0.109 | 0.573 |
| sub-06 | 16 | 0.592 | 0.016 |

Restricting to content-eligible states leaves only sub-06 significant.

### A3: Log-likelihood per sample

| Subject | Friends test | rest overall | baseline | gap (Friends − rest) | above baseline |
|---------|--------------|--------------|----------|----------------------|----------------|
| sub-01 | −3.829 | −19.809 | −3.738 | 15.980 | False |
| sub-02 | −0.421 | −0.916 | −3.738 | 0.495 | True |
| sub-03 | −13.054 | −16.515 | −3.738 | 3.461 | False |
| sub-04 | −11.052 | −12.309 | −3.714 | 1.257 | False |
| sub-05 | −9.699 | −12.995 | −3.714 | 3.296 | False |
| sub-06 | −9.393 | −11.377 | −3.611 | 1.985 | False |

Gaps span 0.495 to 15.980 per sample, the widest range of any condition in the project. sub-01 is the outlier on both fit (gap 15.980) and correspondence (rho −0.289).

### A4: Recurrence vs resting coverage

| Subject | Spearman rho | p | n |
|---------|--------------|-----|---|
| sub-01 | −0.275 | 0.064 | 46 |
| sub-02 | 0.281 | 0.058 | 46 |
| sub-03 | 0.168 | 0.276 | 44 |
| sub-04 | 0.140 | 0.364 | 44 |
| sub-05 | 0.285 | 0.052 | 47 |
| sub-06 | 0.586 | <0.001 | 42 |

Coverage tracks the A1 pattern, including sub-01's negative sign.

### A5: PCA transfer R²

| Subject | Friends R² | rest R² | transfer gap | n_pcs |
|---------|-----------|---------|--------------|-------|
| sub-01 | 0.9505 | 0.9512 | −0.0007 | 75 |
| sub-02 | 0.9504 | 0.9617 | −0.0113 | 72 |
| sub-03 | 0.9504 | 0.9475 | +0.0029 | 72 |
| sub-04 | 0.9500 | 0.9579 | −0.0078 | 77 |
| sub-05 | 0.9509 | 0.9453 | +0.0057 | 67 |
| sub-06 | 0.9504 | 0.9432 | +0.0072 | 74 |

The spatial subspace transfers essentially perfectly. This is what makes the result a dissociation rather than a simple failure: the Friends subspace covers resting data (A5), while the Friends temporal model does not describe resting dynamics (A3) and the recurrence ordering does not carry (A1).

### B2: Bootstrap reference against run-count-matched Movie10

| Subject | Movie10 bootstrap mean | 95% CI | rest rho | rest percentile |
|---------|-----------------------|--------|----------|-----------------|
| sub-01 | 0.484 | 0.107–0.710 | −0.289 | 0.0 |
| sub-02 | 0.273 | 0.032–0.533 | 0.324 | 64.9 |
| sub-03 | 0.421 | 0.146–0.666 | 0.138 | 1.9 |
| sub-04 | 0.658 | 0.398–0.852 | 0.099 | 0.0 |
| sub-05 | 0.543 | 0.302–0.713 | 0.274 | 1.8 |
| sub-06 | 0.698 | 0.396–0.845 | 0.445 | 3.8 |

Matching Movie10 to the resting run count rules out run count as the explanation: in five of six participants the resting rho sits below the 4th percentile of the run-count-matched Movie10 distribution. sub-02 is the exception at the 65th percentile, and sub-02 is also the only participant whose resting fit clears the uniform baseline.

### Serial dependence

| Subject | mean lag-1 FO autocorrelation | effective independent observations | runs |
|---------|-------------------------------|-----------------------------------|------|
| sub-01 | 0.876 | 2.0 | 5 |
| sub-02 | 0.687 | 2.0 | 5 |
| sub-03 | 0.569 | 2.0 | 5 |
| sub-04 | 0.661 | 2.0 | 5 |
| sub-05 | 0.619 | 2.0 | 4 |
| sub-06 | 0.674 | 2.0 | 6 |

Consecutive resting runs are strongly dependent, leaving roughly two effective independent observations per participant. The A1 and A4 p-values should be read with that in mind.

## Caveats

- **No surrogate null.** Every number here is descriptive. The A1 rho cannot be compared against a covariance-and-spectrum-preserving null the way the Movie10 result can.
- Viterbi forces every resting TR onto a state, so occupancy and coverage are biased upward and are existence measures rather than clean transfer tests.
- Approximate p-values throughout: FO compositionality plus serial dependence both act on the same statistic.
- Placement decision (2026-08-17): resting state is supplementary only. It appears in all three panels of Figure S8 and is deliberately absent from main Figure 4.

## Outputs

- `output/rest_05_cross_validation/atlas-4S156Parcels/<sub>/vt0.95/cross_stimulus_summary.json` — all tables above
- `A1_recurrence_fo_scatter.png`, `A2_per_type_scatter.png`, `A3_ll_comparison.png`, `A4_state_coverage_heatmap.png`, `A5_pca_diagnostic.png`, `B2_bootstrap_reference.png`, `serial_dependence.png` in the same directory
