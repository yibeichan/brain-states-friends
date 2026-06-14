# Findings: 06a State Temporal Dynamics

_Script: `script/06a_state_temp_dynamics.py`. Tier: MAIN (R3)._

_Characterizes dwell time, transition structure, and recurrence assortativity of decoded brain states; per-subject, n=6._

## Method (as run)

- **Parcellation:** atlas-4S156Parcels (156 parcels: Schaefer 100 cortical + 56 subcortical composite)
- **Config:** Pareto-selected nc50_g1, diagonal covariance, vt=0.95 (67-77 PCs per subject)
- **Inputs:** `04_combined_hdphmm` decoded_states.pkl + best_model.pkl; `05a_recurrence_analysis` recurrence_summary.json
- **State eligibility:** 05e_a4 state_flags.csv used for category annotations in scatter plots; all active states (recurrence_score > 0) included in numeric summaries
- **Self-transition probability** taken from the fitted HMM transmat_ (model parameter); transition entropy computed from empirical TR-to-TR transition counts
- **Assortativity** edge threshold P > 0.005; permutation test n_perm=5000, per-test permutation p-value p = (count+1)/(n_perm+1) (one assortativity test per subject; no multiple-comparison correction); bootstrap CI n_bootstrap=1000 episode-level resampling
- **Analyses run per-subject and as cross-subject summary** (mode=cross_subject_summary generates a 2x3 multi-panel figure)

## Results

### Active states and recurrence distribution

| Subject | Runs | Active states | Recurrence min | Recurrence max | Mean recurrence |
|---------|------|---------------|---------------|---------------|-----------------|
| sub-01 | 292 | 46 | 0.003 | 0.849 | 0.462 |
| sub-02 | 292 | 46 | 0.003 | 0.836 | 0.465 |
| sub-03 | 291 | 44 | 0.072 | 0.852 | 0.486 |
| sub-04 | 194 | 44 | 0.067 | 0.902 | 0.477 |
| sub-05 | 289 | 47 | 0.003 | 0.817 | 0.443 |
| sub-06 | 292 | 42 | 0.010 | 0.925 | 0.494 |

### Dwell time statistics (all active states)

| Subject | Blocks | Mean (s) | Median (s) | Std (s) |
|---------|--------|----------|-----------|---------|
| sub-01 | 35,829 | 5.74 | 4.47 | 3.37 |
| sub-02 | 36,812 | 5.57 | 4.47 | 3.41 |
| sub-03 | 42,509 | 4.82 | 4.47 | 2.99 |
| sub-04 | 30,565 | 4.46 | 4.47 | 2.67 |
| sub-05 | 38,237 | 5.32 | 4.47 | 3.19 |
| sub-06 | 52,735 | 3.90 | 2.98 | 2.58 |

Median dwell = 4.47 s (3 TRs) for sub-01 through sub-05; sub-06 is 2.98 s (2 TRs). Mean exceeds median by 1-2 s for all subjects, consistent with a right-skewed dwell distribution.

### Scatter correlations: recurrence vs temporal dynamics

Spearman rho computed from per-subject state_summary_table.csv (all active states).

| Subject | Rec vs Dwell rho | p | Rec vs a_kk rho | p | Rec vs Entropy rho | p |
|---------|-----------------|---|----------------|---|-------------------|---|
| sub-01 | 0.251 | 9.19e-02 | -0.164 | 2.77e-01 | 0.086 | 5.68e-01 |
| sub-02 | 0.055 | 7.14e-01 | -0.195 | 1.93e-01 | 0.270 | 6.97e-02 |
| sub-03 | 0.052 | 7.36e-01 | -0.245 | 1.09e-01 | -0.022 | 8.86e-01 |
| sub-04 | -0.139 | 3.67e-01 | **-0.524** | **2.60e-04** | **0.392** | **8.57e-03** |
| sub-05 | **0.332** | **2.28e-02** | 0.144 | 3.34e-01 | 0.009 | 9.52e-01 |
| sub-06 | -0.260 | 9.60e-02 | **-0.351** | **2.26e-02** | 0.293 | 5.96e-02 |

Bold = p < 0.05. Recurrence vs dwell: significant only for sub-05. Recurrence vs a_kk: negative in 5/6 subjects; significant for sub-04 and sub-06. Recurrence vs entropy: mostly null; sub-04 significant positive.

### Recurrence assortativity

Weighted directed assortativity of the empirical transition graph; edges with P > 0.005 retained.

| Subject | r | 95% CI | perm p | n_edges |
|---------|---|--------|--------|---------|
| sub-01 | 0.215 | [0.177, 0.238] | 2.0e-04 | 671 |
| sub-02 | 0.143 | [0.122, 0.172] | 2.0e-04 | 705 |
| sub-03 | 0.111 | [0.092, 0.149] | 1.0e-03 | 709 |
| sub-04 | 0.232 | [0.200, 0.263] | 2.0e-04 | 660 |
| sub-05 | 0.297 | [0.261, 0.315] | 2.0e-04 | 724 |
| sub-06 | 0.122 | [0.093, 0.147] | 4.0e-04 | 626 |

All six subjects show significant positive recurrence assortativity (r = 0.111 to 0.297). High-recurrence states preferentially transition to other high-recurrence states.

### Cross-subject recurrence vs median dwell (summary mode)

From cross_subject_summary_stats.json (computed per subject on state_summary_table.csv).

| Subject | rho | p | n_states |
|---------|-----|---|----------|
| sub-01 | 0.251 | 0.092 | 46 |
| sub-02 | 0.055 | 0.714 | 46 |
| sub-03 | 0.052 | 0.735 | 44 |
| sub-04 | -0.139 | 0.367 | 44 |
| sub-05 | **0.332** | **0.023** | 47 |
| sub-06 | -0.260 | 0.096 | 42 |

Bold = p < 0.05. Only sub-05 significant; recurrence and median dwell are largely independent dimensions.

## Outputs

- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/state_blocks.csv`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/state_summary_table.csv`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/dwell_time_statistics.json`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/recurrence_assortativity.json`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/transition_counts.npy`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/transition_probabilities.npy`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/transition_probabilities.png/.pdf`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/state_sequence_barcodes.png/.pdf`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/recurrence_vs_temporal.png/.pdf`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/dwell_time_distribution.png/.pdf`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/sub-*/vt0.95/network_dwell_comparison.png/.pdf`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/cross_subject_summary/vt0.95/cross_subject_summary_stats.json`
- `output/06a_state_temp_dynamics/atlas-4S156Parcels/cross_subject_summary/vt0.95/cross_subject_recurrence_vs_dwell.png/.pdf`
