# Findings: pp_05 Cross-Stimulus Validation (Petit Prince Audio)

_Script: `script/pp_05_cross_stimulus_validation.py`. Tier: CROSS-STIM (R5, Fig 5)._

_Tests whether Friends-recurring brain states generalize to unimodal audio narration (Petit Prince, lppFR + lppEN); per-subject, n=5 (sub-04 absent)._

## Method (as run)

- Parcellation: atlas-4S156Parcels; vt=0.95; fo_threshold=0.01; exclude_sub_hrf=False
- Inputs: Friends recurrence scores from 05a; PP fractional occupancy and LL from pp_04; PCA diagnostic from pp_03
- n=5 subjects (sub-01, sub-02, sub-03, sub-05, sub-06); sub-04 has no PP output
- PP runs: 18 per subject except sub-06 (16 total; 7 lppFR + 9 lppEN)
- All other subjects: 9 lppFR + 9 lppEN
- Active states (Friends recurrence > 0): 46, 46, 44, 47, 42 for sub-01 through sub-06
- Content-eligible states (05e_a4 state_flags.csv): 31, 30, 26, 29, 16
- A1: Spearman rho between Friends recurrence score and mean PP FO across all active states; p-values are approximate (FO compositionality + serial dependence inflate rho upward - treat as upper bound)
- A2: Per-language breakdown (lppFR, lppEN), FDR-corrected across 2 tests per subject
- A1/A2 eligible: same analyses restricted to content-eligible states (05e_a4)
- A3: LL per sample - Friends test, PP overall, heuristic baseline log(1/n_active)
- A4: Recurrence vs PP coverage Spearman (coverage = fraction of PP runs where state FO > 0.01)
- A5: PCA transfer R2 - Friends training vs PP audio (network-stratified)
- B1: Movie10 A1-equivalent on same subject (full 61 runs)
- B2: Bootstrap reference - 1000 subsamples of n_pp_runs Movie10 runs; PP rho vs distribution
- Serial dependence: lag-1 FO autocorrelation between consecutive PP chapters per language
- Language comparison: cosine similarity and Pearson r between lppFR and lppEN mean FO vectors

## Results

### A1: Recurrence-FO Correlation (full repertoire)

| Subject | n_active | Spearman rho | p (approx) | positive_correlation |
|---------|----------|-------------|------------|----------------------|
| sub-01 | 46 | 0.056 | 0.710 | False |
| sub-02 | 46 | 0.332 | 0.024 | True |
| sub-03 | 44 | 0.133 | 0.389 | False |
| sub-05 | 47 | -0.000 | 0.998 | False |
| sub-06 | 42 | -0.018 | 0.908 | False |

### A1 Eligible: Recurrence-FO Correlation (content-eligible states, 05e_a4)

| Subject | n_eligible_active | Spearman rho | p (approx) |
|---------|-------------------|-------------|------------|
| sub-01 | 31 | 0.053 | 0.778 |
| sub-02 | 30 | 0.287 | 0.124 |
| sub-03 | 26 | 0.178 | 0.384 |
| sub-05 | 29 | -0.273 | 0.153 |
| sub-06 | 16 | 0.126 | 0.641 |

### A2: Per-Language Recurrence-FO Correlation (full repertoire, FDR across 2 tests)

| Subject | lppFR rho | lppFR p | lppFR p_fdr | lppEN rho | lppEN p | lppEN p_fdr | lppFR n_runs | lppEN n_runs |
|---------|-----------|---------|-------------|-----------|---------|-------------|--------------|--------------|
| sub-01 | 0.066 | 0.661 | 0.661 | 0.079 | 0.604 | 0.661 | 9 | 9 |
| sub-02 | 0.419 | 0.004 | 0.007 | 0.243 | 0.104 | 0.104 | 9 | 9 |
| sub-03 | 0.134 | 0.387 | 0.387 | 0.179 | 0.246 | 0.387 | 9 | 9 |
| sub-05 | 0.018 | 0.907 | 0.955 | -0.008 | 0.955 | 0.955 | 9 | 9 |
| sub-06 | 0.059 | 0.712 | 0.858 | -0.028 | 0.858 | 0.858 | 7 | 9 |

### A3: Log-Likelihood per Sample

| Subject | Friends test LL | PP overall LL | Baseline LL | PP > baseline |
|---------|----------------|--------------|-------------|---------------|
| sub-01 | -3.829 | -11.934 | -3.738 | False |
| sub-02 | -0.421 | -0.880 | -3.738 | True |
| sub-03 | -13.054 | -18.119 | -3.738 | False |
| sub-05 | -9.699 | -18.471 | -3.714 | False |
| sub-06 | -9.393 | -14.936 | -3.611 | False |

### A3: Per-Language LL per Sample

| Subject | lppFR LL | lppEN LL |
|---------|----------|----------|
| sub-01 | -11.939 | -11.928 |
| sub-02 | -0.679 | -1.091 |
| sub-03 | -19.703 | -16.458 |
| sub-05 | -18.014 | -18.949 |
| sub-06 | -15.249 | -14.678 |

### A4: Recurrence vs PP Coverage Spearman

| Subject | rho | p | n_active_states |
|---------|-----|---|-----------------|
| sub-01 | 0.097 | 0.522 | 46 |
| sub-02 | 0.378 | 0.010 | 46 |
| sub-03 | 0.167 | 0.277 | 44 |
| sub-05 | -0.084 | 0.573 | 47 |
| sub-06 | 0.165 | 0.297 | 42 |

### A5: PCA Transfer R2 (Overall)

| Subject | n_PCs | Friends R2 | PP R2 | Transfer gap | flag_low_variance |
|---------|-------|-----------|-------|-------------|-------------------|
| sub-01 | 75 | 0.951 | 0.949 | 0.001 | False |
| sub-02 | 72 | 0.950 | 0.947 | 0.003 | False |
| sub-03 | 72 | 0.950 | 0.955 | -0.005 | False |
| sub-05 | 67 | 0.951 | 0.952 | -0.001 | False |
| sub-06 | 74 | 0.950 | 0.950 | 0.001 | False |

### A5: PCA Transfer R2 by Network (PP audio, sub-01 shown; network order matches NETWORK_ORDER)

Networks with consistently lower PP R2 (all subjects, values approx): Thalamus (0.41-0.52), Hipp/Amyg (0.28-0.55), BG (0.71-0.86); Limbic also reduced (0.66-0.86). Neocortical networks (Vis, SomMot, DorsAttn, Cont, Default) all above 0.94.

| Network | sub-01 R2 | sub-02 R2 | sub-03 R2 | sub-05 R2 | sub-06 R2 |
|---------|-----------|-----------|-----------|-----------|-----------|
| Vis | 0.970 | 0.966 | 0.980 | 0.975 | 0.967 |
| SomMot | 0.970 | 0.956 | 0.974 | 0.971 | 0.968 |
| DorsAttn | 0.959 | 0.948 | 0.968 | 0.963 | 0.960 |
| SalVentAttn | 0.942 | 0.939 | 0.949 | 0.949 | 0.954 |
| Limbic | 0.664 | 0.814 | 0.841 | 0.855 | 0.856 |
| Cont | 0.967 | 0.971 | 0.974 | 0.971 | 0.968 |
| Default | 0.956 | 0.961 | 0.961 | 0.955 | 0.960 |
| BG | 0.781 | 0.862 | 0.708 | 0.707 | 0.764 |
| Brainstem | 0.980 | 0.982 | 0.946 | 0.949 | 0.962 |
| Thalamus | 0.440 | 0.414 | 0.493 | 0.522 | 0.496 |
| Hipp/Amyg | 0.283 | 0.554 | 0.463 | 0.480 | 0.434 |
| Cerebellum | 0.801 | 0.766 | 0.877 | 0.695 | 0.907 |

### B1: Movie10 Baseline (full 61 runs, same subject)

| Subject | n_movie_runs | Movie10 Spearman rho | Movie10 p |
|---------|-------------|---------------------|-----------|
| sub-01 | 61 | 0.549 | 7.9e-05 |
| sub-02 | 61 | 0.264 | 0.077 |
| sub-03 | 61 | 0.518 | 3.2e-04 |
| sub-05 | 61 | 0.610 | 5.4e-06 |
| sub-06 | 61 | 0.773 | 2.0e-09 |

### B2: Bootstrap Reference (Movie10 subsampled to n_pp_runs, 1000 draws)

| Subject | n_subsample | M10 bootstrap rho mean | M10 95% CI | PP rho | PP rho percentile |
|---------|-------------|----------------------|------------|--------|------------------|
| sub-01 | 18 | 0.535 | [0.364, 0.679] | 0.056 | 0.0 |
| sub-02 | 18 | 0.276 | [0.119, 0.448] | 0.332 | 74.6 |
| sub-03 | 18 | 0.491 | [0.323, 0.638] | 0.133 | 0.0 |
| sub-05 | 18 | 0.583 | [0.450, 0.683] | -0.000 | 0.0 |
| sub-06 | 16 | 0.746 | [0.625, 0.827] | -0.018 | 0.0 |

### Serial Dependence (lag-1 FO autocorrelation between consecutive PP chapters)

| Subject | lppFR mean lag-1 | lppFR n_eff | lppEN mean lag-1 | lppEN n_eff |
|---------|-----------------|-------------|-----------------|-------------|
| sub-01 | 0.488 | 3.1 | 0.649 | 2.0 |
| sub-02 | 0.666 | 2.0 | 0.739 | 2.0 |
| sub-03 | 0.805 | 2.0 | 0.688 | 2.0 |
| sub-05 | 0.514 | 2.9 | 0.636 | 2.0 |
| sub-06 | 0.650 | 2.0 | 0.672 | 2.0 |

### Language Comparison (French vs English mean FO profiles)

| Subject | Cosine similarity | Pearson r | Pearson p |
|---------|-----------------|-----------|-----------|
| sub-01 | 0.949 | 0.911 | 4.6e-20 |
| sub-02 | 0.873 | 0.757 | 1.9e-10 |
| sub-03 | 0.921 | 0.896 | 1.6e-18 |
| sub-05 | 0.959 | 0.927 | 3.9e-22 |
| sub-06 | 0.898 | 0.789 | 1.0e-11 |

## Outputs

- output/pp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/cross_stimulus_summary.json
- output/pp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A1_recurrence_fo_scatter.png
- output/pp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A2_per_type_scatter.png
- output/pp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A3_ll_comparison.png
- output/pp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A4_state_coverage_heatmap.png
- output/pp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A5_pca_diagnostic.png
- output/pp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/B2_bootstrap_reference.png
- output/pp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/language_comparison.png
- output/pp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/serial_dependence.png
