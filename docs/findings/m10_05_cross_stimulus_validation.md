# Findings: m10_05 Cross-Stimulus Validation

_Script: `script/m10_05_cross_stimulus_validation.py`. Tier: CROSS-STIM (R5, Fig 5)._

_Cross-stimulus test of whether Friends-derived brain state recurrence scores predict Movie10 fractional occupancy; per-subject, n=6._

## Method (as run)

- Atlas: atlas-4S156Parcels; vt=0.95; 61 Movie10 runs per subject (Bourne x10, Wolf of Wall Street x17, Hidden Figures x24, Life x10)
- Input: Friends recurrence scores from 05a; movie fractional occupancy and log-likelihood from m10_04; PCA transfer diagnostic from m10_03
- Active states defined as recurrence > 0 (all-repertoire A1/A2/A4); content-eligible subset restricted to states flagged in 05e_a4 state_flags.csv (eligibility_source = 05e_a4) intersected with active states
- FO threshold for state-active-in-run: 0.01 (A4 coverage counts)
- Sub-HRF exclusion: off (default); FO-based validation does not require per-block BOLD evidence
- No group statistics; all results are per-subject

## Results

### A1: Recurrence-FO Correlation (all active states)

Spearman rank correlation between Friends recurrence score and mean movie FO across all 61 runs, restricted to active states (recurrence > 0).

| Subject | n_active | Spearman rho | p-value |
|---------|----------|--------------|---------|
| sub-01 | 46 | 0.549 | 7.9e-05 |
| sub-02 | 46 | 0.264 | 0.077 |
| sub-03 | 44 | 0.518 | 3.2e-04 |
| sub-04 | 44 | 0.753 | 3.6e-09 |
| sub-05 | 47 | 0.610 | 5.4e-06 |
| sub-06 | 42 | 0.773 | 2.0e-09 |

### A1 (content-eligible subset)

Same correlation restricted to the 05e_a4 content-eligible subset intersected with active states.

| Subject | n_eligible_active | Spearman rho | p-value |
|---------|-------------------|--------------|---------|
| sub-01 | 31 | 0.301 | 0.100 |
| sub-02 | 30 | -0.082 | 0.665 |
| sub-03 | 26 | 0.457 | 0.019 |
| sub-04 | 27 | 0.657 | 2.0e-04 |
| sub-05 | 29 | 0.502 | 0.006 |
| sub-06 | 16 | 0.818 | 1.1e-04 |

### A2: Per-Movie-Type Correlation (all active states, FDR-corrected)

Spearman correlation between Friends recurrence and per-genre mean movie FO; FDR correction (Benjamini-Hochberg) across the four genre types within each subject.

| Subject | Bourne rho (q) | Wolf rho (q) | Figures rho (q) | Life rho (q) |
|---------|----------------|--------------|-----------------|--------------|
| sub-01 | 0.450 (0.002) | 0.779 (7.4e-10) | 0.584 (4.1e-05) | -0.093 (0.538) |
| sub-02 | 0.048 (0.795) | 0.609 (2.8e-05) | 0.293 (0.097) | 0.039 (0.795) |
| sub-03 | 0.419 (0.009) | 0.395 (0.011) | 0.654 (6.0e-06) | 0.197 (0.200) |
| sub-04 | 0.528 (3.0e-04) | 0.797 (1.9e-10) | 0.825 (2.3e-11) | 0.244 (0.110) |
| sub-05 | 0.502 (4.3e-04) | 0.703 (1.5e-07) | 0.651 (1.5e-06) | 0.271 (0.065) |
| sub-06 | 0.425 (0.007) | 0.866 (5.6e-13) | 0.790 (1.0e-09) | 0.300 (0.054) |

q = FDR-corrected p-value (Benjamini-Hochberg within subject across 4 genres).

### A3: Log-Likelihood Comparison

Friends-trained HMM log-likelihood per sample on Friends test data vs. Movie10 vs. uniform-state baseline. Baseline = log(1/n_active_states); heuristic reference only, not on the same scale as Gaussian-emission HMM log-likelihood.

| Subject | Friends test LL | Movie overall LL | LL gap (Friends - Movie) | Movie > baseline |
|---------|----------------|------------------|--------------------------|-----------------|
| sub-01 | -3.829 | -6.560 | +2.732 | False |
| sub-02 | -0.421 | -4.697 | +4.276 | False |
| sub-03 | -13.054 | -15.437 | +2.382 | False |
| sub-04 | -11.052 | -10.886 | -0.166 | False |
| sub-05 | -9.699 | -11.124 | +1.425 | False |
| sub-06 | -9.393 | -11.163 | +1.770 | False |

Per-genre movie LL (log-likelihood per sample):

| Subject | Bourne | Wolf | Figures | Life |
|---------|--------|------|---------|------|
| sub-01 | -6.412 | -4.055 | -5.886 | -12.655 |
| sub-02 | -6.700 | -2.312 | -4.511 | -7.268 |
| sub-03 | -15.417 | -15.680 | -14.322 | -17.728 |
| sub-04 | -12.421 | -9.628 | -10.079 | -13.465 |
| sub-05 | -11.923 | -9.592 | -11.012 | -13.253 |
| sub-06 | -9.176 | -11.686 | -10.859 | -12.948 |

### A4: State Coverage Across Movie Runs

Fraction of the 61 movie runs each active state exceeds FO threshold 0.01; Spearman correlation between Friends recurrence score and movie coverage fraction across active states.

| Subject | n_active states | n active in >50% runs | Coverage-recurrence rho | p-value |
|---------|----------------|----------------------|------------------------|---------|
| sub-01 | 46 | 33 | 0.573 | 3.1e-05 |
| sub-02 | 46 | 35 | 0.366 | 0.012 |
| sub-03 | 44 | 35 | 0.667 | 7.8e-07 |
| sub-04 | 44 | 29 | 0.850 | 3.1e-13 |
| sub-05 | 47 | 32 | 0.607 | 6.1e-06 |
| sub-06 | 42 | 31 | 0.792 | 4.3e-10 |

No Friends-inactive states (recurrence = 0) had mean movie FO above threshold for any subject.

### A5: PCA Transfer Diagnostic

Variance explained (R2) by Friends-trained PCs applied to Friends test data vs. Movie10 data. Flag raised if movie R2 < 0.70.

| Subject | Friends R2 | Movie10 R2 | Transfer gap | Low-variance flag |
|---------|-----------|-----------|-------------|-------------------|
| sub-01 | 0.951 | 0.936 | 0.014 | False |
| sub-02 | 0.950 | 0.935 | 0.016 | False |
| sub-03 | 0.950 | 0.947 | 0.004 | False |
| sub-04 | 0.950 | 0.941 | 0.009 | False |
| sub-05 | 0.951 | 0.934 | 0.017 | False |
| sub-06 | 0.950 | 0.940 | 0.011 | False |

Transfer gap is the raw friends_r2 - movie_r2 field (rounded to 3 decimals); it can differ from the rounded R2 columns by up to 0.002 due to rounding.

## Outputs

- output/m10_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/cross_stimulus_summary.json
- output/m10_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A1_recurrence_fo_scatter.png
- output/m10_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A2_per_type_scatter.png
- output/m10_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A3_ll_comparison.png
- output/m10_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A4_state_coverage_heatmap.png
- output/m10_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A5_pca_diagnostic.png
