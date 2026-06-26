# Findings: ICA out-of-stimulus recurrence supplement (`sm_alt_ica_oos_recurrence`)

Findings for the ICA out-of-stimulus recurrence supplement. This is the ICA analogue of main-analysis Results R5. Code: `script/sm_alt_ica_oos_recurrence.py` (+`.sh`); companion figure `script/fig_sm_alt_ica_oos_recurrence.py` (OOS panels).

## Question

Do the consensus ICA maps fit to Friends also emerge as recurring states when the subject watches independent Movie10 films (Bourne, Wolf, Figures of Speech, Life)? Under the main HMM analysis (R5), states identified from Friends re-appear across held-out Movie10 runs. Here we ask the same question using the ICA decomposition: the ICA projection is frozen from Friends and applied — without any refit — to Movie10 time courses; recurrence is judged by the same fractional-occupancy criterion used for HMM states. This is strictly a per-subject analysis across all 6 subjects. There is no group-level or cohort statistic — all values reported are per-subject descriptive quantities.

## Method (as run)

- **Inputs, frozen:** the same vt=0.95 PCA subspace used in the main analysis (saved from `script/03a_pca.py`), per-subject K_active consensus ICA maps from `sm_alt_ica_states`, and per-subject PC scores for Movie10 runs (from `script/m10_03_apply_pca.py`).
- **Projection (no refit):** given the consensus ICA component matrix `C` (K_active × n_pcs) and the PCA component matrix `M` (n_pcs × n_parcels), the ICA projection into parcel space is reconstructed as `proj = C @ M @ pinv(MᵀM)`. This projects each Movie10 TR into the ICA component space using only the frozen Friends-derived components; no parameters are estimated from Movie10 data.
- **Winner-take-all (WTA) labels:** each Movie10 run's PC scores are z-scored across time; the signed argmax across ICA components at each TR gives the WTA label. Recurrence is scored per component: a component is "recurrent" in a given Movie10 run if its fractional occupancy (FO) exceeds 0.02. The primary metric is the Spearman correlation (ρ) between per-component Friends recurrence (fraction of Friends runs with FO > 0.02) and per-component Movie10 WTA fractional occupancy (mean across the 61 Movie10 runs).
- **Continuous robustness arm:** as a secondary check, per-run z-scored ICA activations are L1-normalized to give magnitude shares (continuous occupancy) and the same Spearman ρ is computed against Friends recurrence. This arm is less sensitive to the arbitrary WTA threshold.
- **Model naming:** the upstream HMM whose PCA subspace and K_active value are inherited here is a finite (weak-limit, K_max=50) Gaussian-emission HMM with sticky self-transition and hierarchical-Dirichlet transition-concentration priors — not the nonparametric Bayesian model those priors are borrowed from.
- **Non-circularity:** the x-axis (Friends recurrence) derives entirely from Friends data; the y-axis (Movie10 occupancy) derives entirely from Movie10 data. The ICA components themselves were fit on Friends. No Movie10 information enters the ICA fitting or the projection.

## Results

All statistics are per-subject descriptive values (Spearman ρ with two-sided p, n = K_active components). There is no across-subject test, pooled ρ, or cohort-level significance. The analysis covers Movie10 only (four audiovisual films: Bourne, Wolf, Figures of Speech, Life); Phase 2 stimuli (HP, PP) are out of scope and no modality-ordering claim is made.

### Overall (across all Movie10 runs)

| sub | K_active | WTA ρ | WTA p | continuous ρ | continuous p |
|-----|----------|--------|-------|--------------|--------------|
| 01 | 42 | 0.952 | 3.3e-22 | 0.641 | 4.7e-06 |
| 02 | 42 | 0.916 | 1.9e-17 | 0.541 | 2.2e-04 |
| 03 | 42 | 0.942 | 1.3e-20 | 0.829 | 1.2e-11 |
| 04 | 41 | 0.909 | 1.9e-16 | 0.689 | 6.5e-07 |
| 05 | 41 | 0.860 | 5.6e-13 | 0.431 | 4.9e-03 |
| 06 | 37 | 0.923 | 4.6e-16 | 0.516 | 1.1e-03 |

WTA ρ ranges from 0.860 (sub-05) to 0.952 (sub-01). Continuous ρ ranges from 0.431 (sub-05) to 0.829 (sub-03). WTA ρ is consistently higher than continuous ρ within each subject; both arms are positive in all 6 subjects.

### Per-film breakdown

#### WTA ρ per film

| sub | K_active | bourne | wolf | figures | life |
|-----|----------|--------|------|---------|------|
| 01 | 42 | 0.814 | 0.937 | 0.924 | 0.761 |
| 02 | 42 | 0.814 | 0.899 | 0.920 | 0.741 |
| 03 | 42 | 0.908 | 0.941 | 0.887 | 0.836 |
| 04 | 41 | 0.783 | 0.840 | 0.883 | 0.838 |
| 05 | 41 | 0.728 | 0.831 | 0.851 | 0.730 |
| 06 | 37 | 0.926 | 0.912 | 0.833 | 0.788 |

#### Continuous ρ per film

| sub | K_active | bourne | wolf | figures | life |
|-----|----------|--------|------|---------|------|
| 01 | 42 | 0.596 | 0.590 | 0.613 | 0.312 |
| 02 | 42 | 0.507 | 0.523 | 0.496 | 0.430 |
| 03 | 42 | 0.784 | 0.831 | 0.788 | 0.815 |
| 04 | 41 | 0.479 | 0.639 | 0.710 | 0.429 |
| 05 | 41 | 0.221 | 0.469 | 0.411 | 0.360 |
| 06 | 37 | 0.603 | 0.525 | 0.416 | 0.410 |

Per-film WTA ρ values are all positive and range from 0.728 to 0.941. Continuous ρ values are more variable across films and subjects (0.221 to 0.831); the weakest continuous ρ values are in sub-05. No film-level ordering or modality claim is made: these are descriptive per-subject values from a single stimulus type (Movie10 audiovisual films); comparison to other stimulus modalities (HP, PP) requires Phase 2 data, which is out of scope.

### Notes on inter-subject variation

Sub-03 shows the strongest continuous ρ (overall 0.829), consistent across all four films. Sub-05 shows the weakest WTA ρ (0.860) and weakest continuous ρ (0.431). Subject-level variation is the dominant axis; within-subject variation across the four films is secondary. The subject-level pattern here mirrors what was observed in `sm_alt_ica_states`: sub-03 and sub-01 show the strongest correspondence across analyses, while sub-05 is more variable.

## Conclusion

For all 6 subjects, ICA components fit on Friends show positive and nominally significant Spearman correlation between Friends recurrence and Movie10 WTA fractional occupancy (ρ = 0.860 to 0.952). The continuous robustness arm likewise gives positive ρ for all subjects (0.431 – 0.829), confirming the result does not depend on the FO > 0.02 threshold. However, because per-run z-scoring equalizes each component's variance, the continuous arm can only exploit distributional shape — not amplitude spread — to discriminate components, making it a weaker, shape-driven signal; the WTA arm retains the full discriminative spread and is the primary measure. These are per-subject descriptive findings from Movie10 (Phase 1 only); no group statistic is computed and no comparison to other stimulus types or modalities is made. The analysis is non-circular: ICA components and the x-axis (Friends recurrence) are derived entirely from Friends data; the y-axis (Movie10 occupancy) is derived entirely from held-out Movie10 data.

## Outputs

`output/sm_ica_oos_recurrence/atlas-4S156Parcels/{sub-*/oos_recurrence_summary.json}` (in SCRATCH, not tracked in this repository).

Fields: `sub_id`, `parcellation`, `vt`, `stimulus`, `K_active`, `n_components`, `fo_threshold`, `n_movie_runs`, `friends_recurrence` (per-component), `movie_occupancy_wta` (per-component), `movie_occupancy_continuous` (per-component), `overall` (WTA + continuous ρ/p/n), `per_film` (bourne/wolf/figures/life, WTA + continuous ρ/p/n).

## Companion figure

`script/fig_sm_alt_ica_oos_recurrence.py` (OOS panels). Panel: scatter of Friends recurrence (x) vs Movie10 WTA fractional occupancy (y) per component, one panel per subject (6 panels), with ρ annotated. A secondary panel shows the continuous-arm scatter. These panels parallel the HMM R5 figure, enabling direct visual comparison.
