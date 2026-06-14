# Findings: hp_05 Cross-Stimulus Validation (Harry Potter Reading)

_Script: `script/hp_05_cross_stimulus_validation.py`. Tier: CROSS-STIM (R5, Fig 5)._

_Tests whether Friends-recurring brain states generalize to unimodal word-by-word reading (Harry Potter RSVP); per-subject, n=5 (sub-04 absent from HP data)._

## Method (as run)

- Parcellation: atlas-4S156Parcels; vt=0.95; FO threshold for "active" state = 0.01
- Inputs: Friends recurrence scores (05a), HP decoded states and FO (hp_04), PCA transfer diagnostic (hp_03), Movie10 FO (m10_04, for B1/B2)
- 7 HP runs per subject (consecutive book chapters); n_active_states: sub-01=46, sub-02=46, sub-03=44, sub-05=47, sub-06=42
- Eligibility filtering: content-eligible states from 05e_a4 state_flags.csv; n_content_eligible: sub-01=31, sub-02=30, sub-03=26, sub-05=29, sub-06=16
- A1/A2: Spearman rho between Friends recurrence score and mean HP FO over active states (full repertoire and eligible subset); p-values approximate due to FO compositionality
- B1: Movie10 A1-equivalent rho recomputed per subject for direct comparison
- B2: 1000 bootstrap samples of 7 Movie10 runs each; HP rho percentile in that distribution
- Serial dependence: lag-1 FO autocorrelation across consecutive HP chapters; n_effective always 2.0 (all subjects)

## Results

### A1: Recurrence-FO Correlation (full repertoire)

Spearman rho between Friends recurrence score and mean HP FO, restricted to active states.

| Subject | n_active | rho   | p      | Significant |
|---------|----------|-------|--------|-------------|
| sub-01  | 46       | 0.232 | 0.121  | No          |
| sub-02  | 46       | 0.444 | 0.002  | Yes         |
| sub-03  | 44       | 0.367 | 0.014  | Yes         |
| sub-05  | 47       | 0.258 | 0.080  | No          |
| sub-06  | 42       | 0.294 | 0.059  | No          |

2/5 subjects significant (sub-02, sub-03). All correlations positive.

### A1 Eligible: Recurrence-FO Correlation (content-eligible subset, 05e_a4)

| Subject | n_eligible_active | rho    | p      | Significant |
|---------|------------------|--------|--------|-------------|
| sub-01  | 31               | 0.223  | 0.227  | No          |
| sub-02  | 30               | 0.343  | 0.063  | No          |
| sub-03  | 26               | 0.404  | 0.041  | Yes         |
| sub-05  | 29               | -0.028 | 0.887  | No          |
| sub-06  | 16               | 0.231  | 0.389  | No          |

1/5 subjects significant in eligible-subset analysis (sub-03).

### A3: Log-Likelihood Comparison

Friends test LL vs HP LL per sample (both relative to uniform baseline).

| Subject | Friends test LL | HP LL   | Baseline LL | LL gap (Friends minus HP) | HP above baseline |
|---------|----------------|---------|-------------|--------------------------|-------------------|
| sub-01  | -3.829         | -5.709  | -3.738      | +1.880                   | No                |
| sub-02  | -0.421         | +0.109  | -3.738      | -0.530                   | Yes               |
| sub-03  | -13.054        | -10.994 | -3.738      | -2.061                   | No                |
| sub-05  | -9.699         | -10.394 | -3.713      | +0.696                   | No                |
| sub-06  | -9.393         | -8.122  | -3.611      | -1.271                   | No                |

Negative LL gap means HP is better explained than Friends test data. 3/5 subjects show negative LL gap (sub-02, sub-03, sub-06).

### A4: State Coverage (recurrence vs HP coverage Spearman)

Coverage = fraction of 7 HP runs in which each state has FO > 0.01. No Friends-inactive states were activated in HP for any subject.

| Subject | n_active_states | Recurrence-coverage rho | p      | Significant |
|---------|----------------|------------------------|--------|-------------|
| sub-01  | 46             | 0.296                  | 0.046  | Yes         |
| sub-02  | 46             | 0.414                  | 0.004  | Yes         |
| sub-03  | 44             | 0.310                  | 0.041  | Yes         |
| sub-05  | 47             | 0.179                  | 0.229  | No          |
| sub-06  | 42             | 0.445                  | 0.003  | Yes         |

4/5 subjects significant (all except sub-05).

### A5: PCA Transfer Diagnostic (overall R2)

Variance explained by Friends PCA (n_pcs components) applied to HP data.

| Subject | n_pcs | Friends R2 | HP R2  | Transfer gap | Flag low variance |
|---------|-------|-----------|--------|--------------|-------------------|
| sub-01  | 75    | 0.951     | 0.944  | 0.006        | No                |
| sub-02  | 72    | 0.950     | 0.941  | 0.009        | No                |
| sub-03  | 72    | 0.950     | 0.941  | 0.010        | No                |
| sub-05  | 67    | 0.951     | 0.948  | 0.003        | No                |
| sub-06  | 74    | 0.950     | 0.945  | 0.006        | No                |

All subjects: HP R2 > 0.94, transfer gap < 0.01. No low-variance flags.

### A5: PCA Transfer - Per-Network HP R2

Network-stratified HP R2 (per PCs from Friends training). Values shown are HP R2; sub-01 values representative; others similar.

| Network      | sub-01 | sub-02 | sub-03 | sub-05 | sub-06 |
|--------------|--------|--------|--------|--------|--------|
| Vis          | 0.956  | 0.957  | 0.962  | 0.966  | 0.963  |
| SomMot       | 0.966  | 0.952  | 0.961  | 0.964  | 0.965  |
| DorsAttn     | 0.952  | 0.943  | 0.954  | 0.958  | 0.957  |
| SalVentAttn  | 0.938  | 0.934  | 0.936  | 0.951  | 0.950  |
| Limbic       | 0.661  | 0.797  | 0.801  | 0.861  | 0.820  |
| Cont         | 0.969  | 0.968  | 0.971  | 0.973  | 0.968  |
| Default      | 0.960  | 0.956  | 0.957  | 0.958  | 0.957  |
| BG           | 0.786  | 0.864  | 0.692  | 0.731  | 0.720  |
| Brainstem    | 0.984  | 0.982  | 0.948  | 0.956  | 0.963  |
| Thalamus     | 0.424  | 0.342  | 0.424  | 0.549  | 0.434  |
| Hipp/Amyg    | 0.289  | 0.496  | 0.410  | 0.602  | 0.342  |
| Cerebellum   | 0.818  | 0.784  | 0.832  | 0.829  | 0.893  |

Cortical networks (Vis, SomMot, DorsAttn, SalVentAttn, Cont, Default): all >0.93. Subcortical networks (Thalamus, Hipp/Amyg): 0.29-0.60 across subjects.

### B1: Movie10 Baseline Rho (same subjects, full 61-run M10)

| Subject | M10 rho (full) | M10 p       |
|---------|----------------|-------------|
| sub-01  | 0.549          | 7.9e-05     |
| sub-02  | 0.264          | 0.077       |
| sub-03  | 0.518          | 3.2e-04     |
| sub-05  | 0.610          | 5.4e-06     |
| sub-06  | 0.773          | 2.0e-09     |

HP rho is lower than M10 rho for 4/5 subjects; sub-02 is the exception (HP rho 0.444 > M10 rho 0.264).

### B2: Bootstrap Matched-Sample Reference (n=7 M10 runs, 1000 samples)

| Subject | HP rho | M10 bootstrap mean | M10 95% CI           | HP percentile |
|---------|--------|--------------------|----------------------|---------------|
| sub-01  | 0.232  | 0.508              | [0.208, 0.710]       | 3.5%          |
| sub-02  | 0.444  | 0.273              | [0.038, 0.505]       | 91.4%         |
| sub-03  | 0.367  | 0.437              | [0.198, 0.657]       | 28.7%         |
| sub-05  | 0.258  | 0.559              | [0.362, 0.707]       | 0.0%          |
| sub-06  | 0.294  | 0.709              | [0.445, 0.844]       | 0.0%          |

sub-02 HP rho exceeds 91% of matched M10 samples. sub-05 and sub-06 HP rho falls below all 1000 M10 bootstrap samples (0th percentile).

### Serial Dependence (HP chapters)

HP runs are consecutive book chapters; FO vectors are serially correlated.

| Subject | n_runs | Mean lag-1 autocorr | n_effective |
|---------|--------|---------------------|-------------|
| sub-01  | 7      | 0.585               | 2.0         |
| sub-02  | 7      | 0.642               | 2.0         |
| sub-03  | 7      | 0.811               | 2.0         |
| sub-05  | 7      | 0.789               | 2.0         |
| sub-06  | 7      | 0.584               | 2.0         |

All subjects: n_effective = 2.0 (Bartlett approximation). A1 p-values should be interpreted with this constraint in mind.

## Outputs

- output/hp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/cross_stimulus_summary.json
- output/hp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A1_recurrence_fo_scatter.png
- output/hp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A2_per_type_scatter.png
- output/hp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A3_ll_comparison.png
- output/hp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A4_state_coverage_heatmap.png
- output/hp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/A5_pca_diagnostic.png
- output/hp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/B2_bootstrap_reference.png
- output/hp_05_cross_validation/atlas-4S156Parcels/sub-*/vt0.95/serial_dependence.png
