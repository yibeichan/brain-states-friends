# Findings: ICA out-of-stimulus recurrence supplement (`sm_alt_ica_oos_recurrence`)

Findings for the ICA out-of-stimulus recurrence supplement. This is the ICA analogue of main-analysis Results R5. Code: `script/sm_alt_ica_oos_recurrence.py` (+`.sh`); companion figure `script/fig_sm_alt_ica_oos_recurrence.py` (OOS panels).

## Question

Do the consensus ICA maps fit to Friends also emerge as recurring states when the subject is scanned on independent, held-out stimuli? Under the main HMM analysis (R5), states identified from Friends re-appear across held-out runs. Here we ask the same question using the ICA decomposition: the ICA projection is frozen from Friends and applied — without any refit — to the out-of-stimulus (OOS) time courses; recurrence is judged by the same fractional-occupancy criterion used for HMM states. The analysis now spans **three OOS datasets**: Movie10 (four audiovisual films: Bourne, Wolf, Figures of Speech, Life), Harry Potter (`harrypotter`, audio narrative), and Le Petit Prince (`petitprince`, audio narrative in French + English). This is strictly a per-subject analysis. There is no group-level or cohort statistic, and **no cross-stimulus or modality comparison** — all values reported are per-subject descriptive quantities.

## Method (as run)

- **Inputs, frozen:** the same vt=0.95 PCA subspace used in the main analysis (saved from `script/03a_pca.py`), per-subject K_active consensus ICA maps from `sm_alt_ica_states`, and per-subject PC scores for the OOS runs (Movie10 from `m10_03`, Harry Potter from `hp_03`, Petit Prince from `pp_03`). The Friends side (recurrence x-axis) is identical across all three stimuli — only the OOS occupancy (y-axis) changes.
- **Stimulus registry:** the three datasets are resolved through a `STIMULI` registry (`{proj_dir, run_ids_file}`); the per-group ("per-film") breakdown comes straight from each dataset's `run_ids.json` keys (Movie10 → 4 films; Harry Potter → single `harrypotter` group; Petit Prince → `lppFR` + `lppEN`).
- **Projection (no refit):** given the consensus ICA component matrix `C` (K_active × n_pcs) and the PCA component matrix `M` (n_pcs × n_parcels), the ICA projection into parcel space is reconstructed as `proj = C @ M @ pinv(MᵀM)`. This projects each OOS TR into the ICA component space using only the frozen Friends-derived components; no parameters are estimated from OOS data.
- **Winner-take-all (WTA) labels:** each OOS run's PC scores are z-scored across time; the signed argmax across ICA components at each TR gives the WTA label. Recurrence is scored per component: a component is "recurrent" in a given run if its fractional occupancy (FO) exceeds 0.02. The primary metric is the Spearman correlation (ρ) between per-component Friends recurrence (fraction of Friends runs with FO > 0.02) and per-component OOS WTA fractional occupancy (mean across the dataset's runs).
- **Continuous arm (secondary):** per-run z-scored ICA activations are L1-normalized to give magnitude shares (continuous occupancy) and the same Spearman ρ is computed against Friends recurrence. Because per-run z-scoring equalizes each component's variance, this arm can only exploit distributional shape, not amplitude spread.
- **Phase-randomized null:** the raw ρ above is dominated by structure shared between any two scans of the same brain in the same PC subspace (power spectrum + covariance), independent of stimulus content. To isolate the genuinely stimulus-specific component we compute a multivariate phase-randomized (Prichard–Theiler) null: each OOS run's PC scores are FFT phase-scrambled with a shared random phase across PCs (preserving each PC's power spectrum *and* the cross-PC covariance, destroying only stimulus-specific phase/higher-order structure), re-projected through the frozen ICA, and the overall Spearman recomputed. **1000 draws** per subject give a null mean, a z = (real − null mean)/null sd, a one-sided empirical p = (1 + #{null ≥ real}) / (1 + n_draws), and a residual = real − null mean. The residual is the part of the correlation attributable to the stimulus's stimulus-specific structure. (Phase 1 used 100 draws; the n=100 movie10 values are preserved below as a frozen block. The empirical p floor is 1/1001 ≈ 0.001 at 1000 draws.)
- **Run-count asymmetry:** Movie10 has ~61 runs/subject, Harry Potter 7, Petit Prince 18 (16 for sub-06). The phase-randomized null per draw has only as many independent segments as there are runs, so the null mean/sd (hence z and the residual) are estimated more noisily for Harry Potter (7 runs) than for Movie10. For Harry Potter the Gaussian-style z is read as descriptive only; the empirical p and the null distribution are the appropriate summaries.
- **Recurrence ≈ marginal occupancy (caveat, applies to all three stimuli):** the Friends recurrence score is nearly identical to each component's marginal Friends WTA-occupancy (Spearman ρ = 0.985–0.994 across subjects). This is an x-axis-only property (derived entirely from Friends), so it carries **identically** to all three OOS datasets. "Recurrence" here is essentially a long-run occupancy measure, and the cross-stimulus result is best read as "intrinsically high-occupancy components are preferentially occupied out-of-stimulus," not as transfer of a distinct recurrence property.
- **Model naming:** the upstream HMM whose PCA subspace and K_active value are inherited here is a finite (weak-limit, K_max=50) Gaussian-emission HMM with sticky self-transition and hierarchical-Dirichlet transition-concentration priors — not the nonparametric Bayesian model those priors are borrowed from.
- **Independent data sources:** the x-axis (Friends recurrence) derives entirely from Friends data; the y-axis (OOS occupancy) derives entirely from the held-out stimulus. The ICA components were fit on Friends. No OOS information enters the ICA fitting or the projection.

## Subject coverage

| stimulus | runs/subject | subjects | notes |
|---|---|---|---|
| movie10 | 61 | 01–06 (n=6) | full cohort |
| harrypotter | 7 | 01, 02, 03, 05, 06 (n=5) | sub-04 has no HP scans |
| petitprince | 18 | 01, 02, 03, 05, 06 (n=5) | sub-04 has no PP scans; **sub-06 = 16 runs** (lppFR 7 of 9; lppEN 9) |

## Results

All statistics are per-subject descriptive values (Spearman ρ with two-sided p, n = K_active components). There is no across-subject test, pooled ρ, or cohort-level significance, and **no cross-stimulus / modality comparison is made** — the run-count differences (61 / 7 / 18) and subject-set differences (n=6 vs n=5) make any apparent Movie10-vs-HP/PP difference uninterpretable even descriptively. Movie10 is audiovisual; Harry Potter and Petit Prince are auditory; we draw no audiovisual-vs-auditory contrast.

### Movie10 (audiovisual)

#### Phase 1, as published (n_null = 100, frozen for provenance)

Overall WTA arm, real ρ vs the 100-draw phase-randomized null (these are the originally published Phase-1 values; the live numbers below are the 1000-draw re-run):

| sub | WTA real ρ | null mean | z | residual |
|-----|-----------|-----------|------|----------|
| 01 | 0.952 | 0.828 | +5.30 | +0.124 |
| 02 | 0.916 | 0.867 | +1.97 | +0.049 |
| 03 | 0.942 | 0.865 | +5.37 | +0.077 |
| 04 | 0.909 | 0.864 | +2.39 | +0.046 |
| 05 | 0.860 | 0.778 | +3.20 | +0.082 |
| 06 | 0.923 | 0.857 | +3.19 | +0.066 |

#### Overall (n_null = 1000, live)

| sub | K_active | WTA ρ | WTA p | continuous ρ | continuous p |
|-----|----------|--------|-------|--------------|--------------|
| 01 | 42 | 0.952 | 3.3e-22 | 0.641 | 4.7e-06 |
| 02 | 42 | 0.916 | 1.9e-17 | 0.541 | 2.2e-04 |
| 03 | 42 | 0.942 | 1.3e-20 | 0.829 | 1.2e-11 |
| 04 | 41 | 0.909 | 1.9e-16 | 0.689 | 6.5e-07 |
| 05 | 41 | 0.860 | 5.6e-13 | 0.431 | 4.9e-03 |
| 06 | 37 | 0.923 | 4.6e-16 | 0.516 | 1.1e-03 |

Null control (phase-randomized, 1000 draws), overall WTA arm:

| sub | WTA real ρ | null mean | z | p (null) | residual | recurrence≈marginal ρ |
|-----|-----------|-----------|------|----------|----------|-----------------------|
| 01 | 0.952 | 0.827 | +5.57 | 0.0010 | +0.125 | 0.990 |
| 02 | 0.916 | 0.861 | +2.08 | 0.0130 | +0.055 | 0.994 |
| 03 | 0.942 | 0.863 | +4.92 | 0.0010 | +0.079 | 0.985 |
| 04 | 0.909 | 0.866 | +2.19 | 0.0080 | +0.044 | 0.988 |
| 05 | 0.860 | 0.782 | +2.49 | 0.0030 | +0.078 | 0.991 |
| 06 | 0.923 | 0.858 | +3.28 | 0.0010 | +0.064 | 0.986 |

The 1000-draw re-run reproduces the real ρ exactly (it is RNG-independent) and gives essentially the same residuals as the published 100-draw values (±0.006); the null mean/sd are re-estimated over all draws, so the z values shift slightly (e.g. sub-05 +3.20 → +2.49) and the empirical p now resolves below the old 1/101 floor. All 6 subjects remain significant (p ≤ 0.013; the weakest is sub-02 at 0.013). The result is unchanged: a small but significant stimulus-specific component (residual +0.044 to +0.125 above a null mean of ≈0.78–0.87), with recurrence ≈ marginal Friends occupancy (ρ ≈ 0.99). High-occupancy ICA components are preferentially occupied on Movie10, modestly above shared structure; the raw recurrence ρ≈0.9 does not transfer.

### Harry Potter (auditory; 7 runs/subject, n=5)

Overall WTA arm, real ρ vs the 1000-draw null:

| sub | K_active | WTA real ρ | null mean | z | p (null) | residual | recurrence≈marginal ρ |
|-----|----------|-----------|-----------|------|----------|----------|-----------------------|
| 01 | 42 | 0.876 | 0.743 | +3.33 | 0.0020 | +0.133 | 0.990 |
| 02 | 42 | 0.632 | 0.606 | +0.39 | 0.3506 | +0.026 | 0.994 |
| 03 | 42 | 0.796 | 0.746 | +0.98 | 0.1588 | +0.050 | 0.985 |
| 05 | 41 | 0.786 | 0.612 | +2.55 | 0.0020 | +0.175 | 0.991 |
| 06 | 37 | 0.806 | 0.790 | +0.42 | 0.3656 | +0.016 | 0.986 |

The residual is **positive for all 5 subjects but exceeds the null significantly in only 2** (sub-01, sub-05; p ≈ 0.002); sub-02, sub-03, sub-06 do not (p = 0.16–0.37). With only 7 runs the null is wide (sd 0.04–0.07, vs ≈0.02 for Movie10), so the z is read descriptively and the empirical p is the appropriate summary. On Harry Potter the stimulus-specific component is weak and subject-variable. The single `harrypotter` group means the per-group ρ equals the overall ρ (not an independent estimate).

### Petit Prince (auditory; 18 runs/subject, n=5; FR + EN pooled)

`petitprince` is treated as one stimulus with `lppFR` and `lppEN` as per-group breakdowns. **Assumption:** each subject heard both languages of the same narrative, so pooling assumes language-invariance of recurrence (and the within-run phase-randomized null does not absorb the FR/EN block structure). The per-group ρ below are reported to check FR/EN concordance.

Overall WTA arm, real ρ vs the 1000-draw null:

| sub | K_active | WTA real ρ | null mean | z | p (null) | residual | recurrence≈marginal ρ |
|-----|----------|-----------|-----------|------|----------|----------|-----------------------|
| 01 | 42 | 0.881 | 0.821 | +2.19 | 0.0100 | +0.060 | 0.990 |
| 02 | 42 | 0.808 | 0.694 | +2.31 | 0.0120 | +0.115 | 0.994 |
| 03 | 42 | 0.830 | 0.772 | +1.62 | 0.0519 | +0.058 | 0.985 |
| 05 | 41 | 0.821 | 0.662 | +3.45 | 0.0010 | +0.159 | 0.991 |
| 06 | 37 | 0.864 | 0.817 | +1.68 | 0.0390 | +0.046 | 0.986 |

The residual is **positive for all 5 subjects**; it exceeds the null at p < 0.05 for 3 (sub-01, sub-02, sub-05), is marginal for sub-06 (p = 0.039) and borderline for sub-03 (p = 0.052). Petit Prince is intermediate between Movie10 (all significant) and Harry Potter (2/5) — consistent with its larger run count tightening the null relative to HP. **sub-06 footnote:** sub-06's Petit Prince is 16 runs (lppFR 7 of 9 present; lppEN 9), so its pooled ρ rests on an FR/EN-imbalanced set.

#### Petit Prince per-group WTA ρ (lppFR vs lppEN — concordance check)

| sub | lppFR (n) | lppEN (n) |
|-----|-----------|-----------|
| 01 | 0.836 (9) | 0.853 (9) |
| 02 | 0.836 (9) | 0.668 (9) |
| 03 | 0.832 (9) | 0.757 (9) |
| 05 | 0.794 (9) | 0.738 (9) |
| 06 | 0.746 (7) | 0.882 (9) |

FR and EN per-group ρ are broadly concordant (both in the 0.67–0.88 range, no systematic FR>EN or EN>FR direction across subjects), so the pooled petitprince ρ is not dominated by one language. These per-group values are raw, not null-corrected.

### Continuous arm does not survive the null (all three stimuli)

Against the phase-randomized null the continuous-occupancy residual is mostly **negative or near-zero**, so the continuous arm carries no reliable stimulus-specific signal beyond shared structure. It is reported for completeness and does **not** corroborate the WTA result.

| stimulus | continuous residual by subject (sign pattern) |
|---|---|
| movie10 | 01 +0.091; 02 −0.200; 03 −0.042; 04 −0.034; 05 −0.168; 06 −0.009 (negative in 5/6) |
| harrypotter | 01 +0.066; 02 −0.119; 03 −0.191; 05 +0.103; 06 +0.003 (small, mixed sign) |
| petitprince | 01 −0.080; 02 −0.103; 03 −0.080; 05 +0.087; 06 +0.001 (negative in 3/5, near-zero otherwise) |

### Movie10 per-film breakdown (raw, not null-corrected)

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

These per-film values are **raw, not null-corrected** (the phase-randomized null is computed only for the pooled overall arm), so the "mostly shared structure" caveat applies and they should be read as descriptive only. No film-level ordering is made.

## Conclusion

The ICA decomposition reproduces the main-analysis R5 pattern across three held-out stimuli, but the null-controlled result is narrower than the raw ρ suggests and varies by stimulus. On Movie10 the WTA recurrence→occupancy correlation exceeds the phase-randomized null in all 6 subjects (1000-draw p ≤ 0.013), though the residual is small (+0.044 to +0.125 above a null mean ≈0.78–0.87). On Petit Prince it is significant in 3 of 5 subjects (marginal/borderline in the other two), and on Harry Potter in 2 of 5, where the null is run-count-limited (7 runs). Across all three, the Friends recurrence score is ≈ each component's marginal Friends occupancy (ρ ≈ 0.99): high-occupancy ICA components are preferentially occupied out-of-stimulus, modestly above shared structure, and the raw recurrence ρ≈0.9 does not transfer. The continuous arm does not exceed the null for any stimulus. These are per-subject descriptive findings (n=6 Movie10, n=5 HP/PP); no group statistic and no cross-stimulus or modality comparison is computed, since the differing run counts and subject sets preclude it. The x-axis (Friends recurrence) derives entirely from Friends data and the y-axis (out-of-stimulus occupancy) entirely from the held-out stimulus; the two share only the frozen projection. The same phase-randomized null leaves a comparably small, significant residual in the main HMM analysis (R5).

## Outputs

`output/sm_ica_oos_recurrence/atlas-4S156Parcels/{sub}/{stimulus}/oos_recurrence_summary.json` (in SCRATCH, not tracked in this repository), one per (subject, stimulus). The published Phase-1 movie10 100-draw summaries are archived at `…/{sub}/_phase1_n100/oos_recurrence_summary.json`.

Fields: `sub_id`, `parcellation`, `vt`, `stimulus`, `n_null`, `K_active`, `n_components`, `fo_threshold`, `n_movie_runs` (historical name = total OOS runs for the stimulus), `friends_recurrence` (per-component), `movie_occupancy_wta` (per-component OOS WTA occupancy; historical key name), `movie_occupancy_continuous` (per-component), `recurrence_vs_friends_marginal_wta_rho` (the ≈0.99 caveat), `overall` (WTA + continuous ρ/p/n, each with a `null` block `{mean, sd, z, p, n_draws, residual}` from the phase-randomized null), `per_film` (per-group, each with `n_runs` + WTA + continuous ρ/p/n — raw, not null-corrected). Note: the `movie_*` key names are historical/generic and apply to all three stimuli.

## Companion figure

`script/fig_sm_alt_ica_oos_recurrence.py` (OOS panels), rendered per stimulus as `fig_sm_ica_oos_recurrence_{stimulus}_{A_wta,B_continuous}.{png,svg}`. Panel A: scatter of Friends recurrence (x) vs OOS WTA fractional occupancy (y) per component, one cell per subject (sub-04 cell blank for HP/PP), annotated with ρ, the null z (vs null mean), and the run count. Panel B: the continuous-arm scatter, annotated with ρ, the null residual (Δ), and the run count. The run-count annotation makes the per-stimulus precision asymmetry visible; the stimuli are not compared. These panels parallel the HMM R5 figure.
