# Findings: ICA out-of-stimulus recurrence supplement (`sm_alt_ica_oos_recurrence`)

Findings for the ICA out-of-stimulus recurrence supplement. This is the ICA analogue of main-analysis Results R5. Code: `script/sm_alt_ica_oos_recurrence.py` (+`.sh`); companion figure `script/fig_sm_alt_ica_oos_recurrence.py` (OOS panels).

## Question

Do the consensus ICA maps fit to Friends also emerge as recurring states when the subject is scanned on independent, held-out stimuli? Under the main HMM analysis (R5), states identified from Friends re-appear across held-out runs. Here we ask the same question using the ICA decomposition: the ICA projection is frozen from Friends and applied — without any refit — to the out-of-stimulus (OOS) time courses; recurrence is judged by the same fractional-occupancy criterion used for HMM states. The analysis now spans **three OOS datasets**: Movie10 (four audiovisual films: Bourne, Wolf, Figures of Speech, Life), Harry Potter (`harrypotter`, audio narrative), and Le Petit Prince (`petitprince`, audio narrative in French + English). This is strictly a per-subject analysis. There is no group-level or cohort statistic, and **no cross-stimulus or modality comparison** — all values reported are per-subject descriptive quantities.

## Method (as run)

- **Inputs, frozen:** the same vt=0.95 PCA subspace used in the main analysis (saved from `script/03a_pca.py`), per-subject K_used consensus ICA maps from `sm_alt_ica_states`, fit in the top-K PC subspace (K_used is the fitted model's occupied-state count (posterior usage above 0.01 at the final EM iteration; 37–42 per subject), stored as `K_active` in the output JSONs. It is distinct from the active-state count (fractional occupancy above 0.02 in at least one run; 42–47 per subject).), and per-subject PC scores for the OOS runs (Movie10 from `m10_03`, Harry Potter from `hp_03`, Petit Prince from `pp_03`). The Friends side (recurrence x-axis) is identical across all three stimuli — only the OOS occupancy (y-axis) changes.
- **Stimulus registry:** the three datasets are resolved through a `STIMULI` registry (`{proj_dir, run_ids_file}`); the per-group ("per-film") breakdown comes straight from each dataset's `run_ids.json` keys (Movie10 → 4 films; Harry Potter → single `harrypotter` group; Petit Prince → `lppFR` + `lppEN`).
- **Projection (no refit):** given the consensus ICA component matrix `C` (K_used × n_pcs) and the PCA component matrix `M` (n_pcs × n_parcels), the ICA projection into parcel space is reconstructed as `proj = C @ M @ pinv(MᵀM)`. This projects each OOS TR into the ICA component space using only the frozen Friends-derived components; no parameters are estimated from OOS data.
- **Winner-take-all (WTA) labels:** each OOS run's PC scores are z-scored across time; the signed argmax across ICA components at each TR gives the WTA label. Recurrence is scored per component: a component is "recurrent" in a given run if its fractional occupancy (FO) exceeds 0.02. The primary metric is the Spearman correlation (ρ) between per-component Friends recurrence (fraction of Friends runs with FO > 0.02) and per-component OOS WTA fractional occupancy (mean across the dataset's runs).
- **Continuous arm (secondary):** per-run z-scored ICA activations are L1-normalized to give magnitude shares (continuous occupancy) and the same Spearman ρ is computed against Friends recurrence. Because per-run z-scoring equalizes each component's variance, this arm can only exploit distributional shape, not amplitude spread.
- **Phase-randomized null:** the raw ρ above is dominated by structure shared between any two scans of the same brain in the same PC subspace (power spectrum + covariance), independent of stimulus content. To isolate the genuinely stimulus-specific component we compute a multivariate phase-randomized (Prichard–Theiler) null: each OOS run's PC scores are FFT phase-scrambled with a shared random phase across PCs (preserving each PC's power spectrum *and* the cross-PC covariance, destroying only stimulus-specific phase/higher-order structure), re-projected through the frozen ICA, and the overall Spearman recomputed. **1000 draws** per subject give a null mean, a z = (real − null mean)/null sd, a one-sided empirical p = (1 + #{null ≥ real}) / (1 + n_draws), and a residual = real − null mean. The residual is the part of the correlation attributable to the stimulus's stimulus-specific structure. The empirical p floor is 1/1001 ≈ 0.001 at 1000 draws.
- **Run-count asymmetry:** Movie10 has ~61 runs/subject, Harry Potter 7, Petit Prince 18 (16 for sub-06). The phase-randomized null per draw has only as many independent segments as there are runs, so the null mean/sd (hence z and the residual) are estimated more noisily for Harry Potter (7 runs) than for Movie10. For Harry Potter the Gaussian-style z is read as descriptive only; the empirical p and the null distribution are the appropriate summaries.
- **Recurrence ≈ marginal occupancy (caveat, applies to all three stimuli):** the Friends recurrence score is nearly identical to each component's marginal Friends WTA-occupancy (Spearman ρ = 0.984–0.994 across subjects). This is an x-axis-only property (derived entirely from Friends), so it carries **identically** to all three OOS datasets. "Recurrence" here is essentially a long-run occupancy measure, and the cross-stimulus result is best read as "intrinsically high-occupancy components are preferentially occupied out-of-stimulus," not as transfer of a distinct recurrence property.
- **Model naming:** the upstream HMM whose PCA subspace and K_used value are inherited here is a finite (weak-limit, K_max=50) Gaussian-emission HMM with sticky self-transition and hierarchical-Dirichlet transition-concentration priors — not the nonparametric Bayesian model those priors are borrowed from.
- **Independent data sources:** the x-axis (Friends recurrence) derives entirely from Friends data; the y-axis (OOS occupancy) derives entirely from the held-out stimulus. The ICA components were fit on Friends. No OOS information enters the ICA fitting or the projection.

## Subject coverage

| stimulus | runs/subject | subjects | notes |
|---|---|---|---|
| movie10 | 61 | 01–06 (n=6) | full cohort |
| harrypotter | 7 | 01, 02, 03, 05, 06 (n=5) | sub-04 has no HP scans |
| petitprince | 18 | 01, 02, 03, 05, 06 (n=5) | sub-04 has no PP scans; **sub-06 = 16 runs** (lppFR 7 of 9; lppEN 9) |

## Results

All statistics are per-subject descriptive values (Spearman ρ with two-sided p, n = K_used components). There is no across-subject test, pooled ρ, or cohort-level significance, and **no cross-stimulus / modality comparison is made** — the run-count differences (61 / 7 / 18) and subject-set differences (n=6 vs n=5) make any apparent Movie10-vs-HP/PP difference uninterpretable even descriptively. Movie10 is audiovisual; Harry Potter and Petit Prince are auditory; we draw no audiovisual-vs-auditory contrast.

### Movie10 (audiovisual)

#### Overall

| sub | K_used | WTA ρ | WTA p | continuous ρ | continuous p |
|-----|----------|--------|-------|--------------|--------------|
| 01 | 42 | 0.869 | 8.4e-14 | 0.628 | 8.6e-06 |
| 02 | 42 | 0.923 | 3.7e-18 | 0.459 | 2.2e-03 |
| 03 | 42 | 0.932 | 3.1e-19 | 0.524 | 3.7e-04 |
| 04 | 41 | 0.881 | 3.1e-14 | 0.622 | 1.4e-05 |
| 05 | 41 | 0.910 | 1.7e-16 | 0.540 | 2.7e-04 |
| 06 | 37 | 0.901 | 3.0e-14 | 0.515 | 1.1e-03 |

Null control (phase-randomized, 1000 draws), overall WTA arm:

| sub | WTA real ρ | null mean | z | p (null) | residual | recurrence≈marginal ρ |
|-----|-----------|-----------|------|----------|----------|-----------------------|
| 01 | 0.869 | 0.699 | +5.17 | 0.0010 | +0.170 | 0.989 |
| 02 | 0.923 | 0.619 | +8.15 | 0.0010 | +0.304 | 0.991 |
| 03 | 0.932 | 0.718 | +6.87 | 0.0010 | +0.214 | 0.994 |
| 04 | 0.881 | 0.684 | +7.57 | 0.0010 | +0.197 | 0.988 |
| 05 | 0.910 | 0.681 | +6.27 | 0.0010 | +0.228 | 0.984 |
| 06 | 0.901 | 0.844 | +2.23 | 0.0080 | +0.057 | 0.985 |

All 6 subjects exceed the null (p ≤ 0.008; the weakest is sub-06). The stimulus-specific residual is +0.057 to +0.304 above a null mean of 0.62–0.84, and recurrence ≈ marginal Friends occupancy (ρ = 0.984–0.994). High-occupancy ICA components are preferentially occupied on Movie10 beyond the structure any scan of the same brain shares; the raw recurrence ρ ≈ 0.9 is still mostly shared structure.

### Harry Potter (auditory; 7 runs/subject, n=5)

Overall WTA arm, real ρ vs the 1000-draw null:

| sub | K_used | WTA real ρ | null mean | z | p (null) | residual | recurrence≈marginal ρ |
|-----|----------|-----------|-----------|------|----------|----------|-----------------------|
| 01 | 42 | 0.641 | 0.526 | +1.52 | 0.0519 | +0.115 | 0.989 |
| 02 | 42 | 0.792 | 0.568 | +3.19 | 0.0010 | +0.223 | 0.991 |
| 03 | 42 | 0.680 | 0.603 | +1.11 | 0.1309 | +0.077 | 0.994 |
| 05 | 41 | 0.835 | 0.626 | +3.40 | 0.0010 | +0.210 | 0.984 |
| 06 | 37 | 0.722 | 0.622 | +1.39 | 0.0689 | +0.100 | 0.985 |

The residual is **positive for all 5 subjects but exceeds the null in only 2** (sub-02, sub-05; p = 0.001); sub-01, sub-03 and sub-06 do not (p = 0.052–0.131). With only 7 runs the null is wide (sd 0.062–0.076, against 0.026–0.037 for Movie10), so the z is read descriptively and the empirical p is the appropriate summary. On Harry Potter the stimulus-specific component is weak and subject-variable. The single `harrypotter` group means the per-group ρ equals the overall ρ (not an independent estimate).

### Petit Prince (auditory; 18 runs/subject, n=5; FR + EN pooled)

`petitprince` is treated as one stimulus with `lppFR` and `lppEN` as per-group breakdowns. **Assumption:** each subject heard both languages of the same narrative, so pooling assumes language-invariance of recurrence (and the within-run phase-randomized null does not absorb the FR/EN block structure). The per-group ρ below are reported to check FR/EN concordance.

Overall WTA arm, real ρ vs the 1000-draw null:

| sub | K_used | WTA real ρ | null mean | z | p (null) | residual | recurrence≈marginal ρ |
|-----|----------|-----------|-----------|------|----------|----------|-----------------------|
| 01 | 42 | 0.789 | 0.627 | +3.31 | 0.0010 | +0.162 | 0.989 |
| 02 | 42 | 0.851 | 0.562 | +5.81 | 0.0010 | +0.289 | 0.991 |
| 03 | 42 | 0.705 | 0.529 | +3.19 | 0.0010 | +0.176 | 0.994 |
| 05 | 41 | 0.840 | 0.668 | +3.29 | 0.0010 | +0.172 | 0.984 |
| 06 | 37 | 0.832 | 0.695 | +2.29 | 0.0030 | +0.136 | 0.985 |

The residual is **positive and exceeds the null for all 5 subjects** (p ≤ 0.003; residual +0.136 to +0.289). **sub-06 footnote:** sub-06's Petit Prince is 16 runs (lppFR 7 of 9 present; lppEN 9), so its pooled ρ rests on an FR/EN-imbalanced set.

#### Petit Prince per-group WTA ρ (lppFR vs lppEN — concordance check)

| sub | lppFR (n) | lppEN (n) |
|-----|-----------|-----------|
| 01 | 0.745 (9) | 0.778 (9) |
| 02 | 0.790 (9) | 0.839 (9) |
| 03 | 0.665 (9) | 0.759 (9) |
| 05 | 0.782 (9) | 0.821 (9) |
| 06 | 0.768 (7) | 0.803 (9) |

FR and EN per-group ρ fall in a similar range (0.67–0.84), but EN is higher than FR in all five subjects, by 0.03–0.09, so the language-invariance assumption behind pooling holds only approximately. These per-group values are raw, not null-corrected, and each rests on 7–9 runs, so the direction is not interpreted further.

### Continuous arm: no reliable signal beyond the null (all three stimuli)

Against the phase-randomized null the continuous-occupancy residual is mostly **negative or near-zero**, so the continuous arm carries no reliable stimulus-specific signal beyond shared structure. Two subject-stimulus pairs exceed the null (Harry Potter sub-01, p = 0.026; Petit Prince sub-05, p = 0.003) with no consistent pattern across subjects or stimuli. The arm is reported for completeness and does **not** corroborate the WTA result.

| stimulus | continuous residual by subject (sign pattern) |
|---|---|
| movie10 | 01 +0.039; 02 −0.288; 03 −0.180; 04 −0.062; 05 +0.028; 06 −0.222 (negative in 4/6) |
| harrypotter | 01 +0.124; 02 −0.103; 03 −0.059; 05 −0.010; 06 −0.228 (negative in 4/5) |
| petitprince | 01 +0.014; 02 −0.144; 03 +0.023; 05 +0.204; 06 −0.193 (negative in 2/5) |

### Movie10 per-film breakdown (raw, not null-corrected)

#### WTA ρ per film

| sub | K_used | bourne | wolf | figures | life |
|-----|----------|--------|------|---------|------|
| 01 | 42 | 0.826 | 0.816 | 0.855 | 0.856 |
| 02 | 42 | 0.746 | 0.877 | 0.912 | 0.716 |
| 03 | 42 | 0.794 | 0.891 | 0.889 | 0.771 |
| 04 | 41 | 0.703 | 0.841 | 0.871 | 0.857 |
| 05 | 41 | 0.813 | 0.904 | 0.909 | 0.780 |
| 06 | 37 | 0.773 | 0.844 | 0.891 | 0.864 |

#### Continuous ρ per film

| sub | K_used | bourne | wolf | figures | life |
|-----|----------|--------|------|---------|------|
| 01 | 42 | 0.503 | 0.646 | 0.571 | 0.679 |
| 02 | 42 | 0.415 | 0.463 | 0.435 | 0.381 |
| 03 | 42 | 0.418 | 0.450 | 0.537 | 0.544 |
| 04 | 41 | 0.418 | 0.630 | 0.574 | 0.567 |
| 05 | 41 | 0.523 | 0.537 | 0.476 | 0.595 |
| 06 | 37 | 0.443 | 0.492 | 0.462 | 0.517 |

These per-film values are **raw, not null-corrected** (the phase-randomized null is computed only for the pooled overall arm), so the "mostly shared structure" caveat applies and they should be read as descriptive only. No film-level ordering is made.

## Conclusion

The ICA decomposition reproduces the main-analysis R5 pattern across three held-out stimuli, and the null-controlled result is narrower than the raw ρ suggests. The WTA recurrence→occupancy correlation exceeds the phase-randomized null in all 6 subjects on Movie10 (p ≤ 0.008; residual +0.057 to +0.304), in all 5 on Petit Prince (p ≤ 0.003), and in 2 of 5 on Harry Potter, where the null is run-count-limited (7 runs). Across all three, the Friends recurrence score is ≈ each component's marginal Friends occupancy (ρ ≈ 0.99): high-occupancy ICA components are preferentially occupied out-of-stimulus beyond shared structure, and the raw recurrence ρ ≈ 0.9 does not transfer as such. The continuous arm does not reliably exceed the null. These are per-subject descriptive findings (n=6 Movie10, n=5 HP/PP); no group statistic and no cross-stimulus or modality comparison is computed, since the differing run counts and subject sets preclude it. The x-axis (Friends recurrence) derives entirely from Friends data and the y-axis (out-of-stimulus occupancy) entirely from the held-out stimulus; the two share only the frozen projection. The same phase-randomized null leaves a small, significant residual in the main HMM analysis (R5).

An earlier version of this analysis used ICA maps from the superseded truncated whitening (see `sm_alt_ica_states.md`). With those maps the Movie10 residual was smaller (+0.044 to +0.125) and fewer Petit Prince subjects exceeded the null (3 of 5); the results above replace them.

## Outputs

`output/sm_ica_oos_recurrence_topk/atlas-4S156Parcels/{sub}/{stimulus}/oos_recurrence_summary.json` (in SCRATCH, not tracked in this repository; directory name `ICA_OOS_DIR` in `script/utils/ica_states.py`), one per (subject, stimulus). The earlier `output/sm_ica_oos_recurrence/` outputs come from the superseded ICA maps.

Fields: `sub_id`, `parcellation`, `vt`, `stimulus`, `n_null`, `K_active` (= K_used), `n_components`, `fo_threshold`, `n_movie_runs` (historical name = total OOS runs for the stimulus), `friends_recurrence` (per-component), `movie_occupancy_wta` (per-component OOS WTA occupancy; historical key name), `movie_occupancy_continuous` (per-component), `recurrence_vs_friends_marginal_wta_rho` (the ≈0.99 caveat), `overall` (WTA + continuous ρ/p/n, each with a `null` block `{mean, sd, z, p, n_draws, residual}` from the phase-randomized null), `per_film` (per-group, each with `n_runs` + WTA + continuous ρ/p/n — raw, not null-corrected). Note: the `movie_*` key names are historical/generic and apply to all three stimuli.

## Reproduce

```bash
# needs the sm_alt_ica_states outputs and the m10_03 / hp_03 / pp_03 projections
uv run python -m pytest script/tests/test_alt_ica_oos_recurrence.py
sbatch script/sm_alt_ica_oos_recurrence.sh                                 # all three stimuli, 1000 null draws
uv run python script/fig_sm_alt_ica_oos_recurrence.py                      # companion figure
```

## Companion figure

`script/fig_sm_alt_ica_oos_recurrence.py` (OOS panels), rendered per stimulus as `fig_sm_ica_oos_recurrence_{stimulus}_{A_wta,B_continuous}.{png,svg}`. Panel A: scatter of Friends recurrence (x) vs OOS WTA fractional occupancy (y) per component, one cell per subject (sub-04 cell blank for HP/PP), annotated with ρ, the null z (vs null mean), and the run count. Panel B: the continuous-arm scatter, annotated with ρ, the null residual (Δ), and the run count. The run-count annotation makes the per-stimulus precision asymmetry visible; the stimuli are not compared. These panels parallel the HMM R5 figure.
