# Findings: R5 phase-randomized null (`sm_rel_r5_phase_null`)

Canonical null for main-analysis Results R5. Code: `script/sm_rel_r5_phase_null.py` (+`.sh` runner); tests `script/tests/test_rel_r5_phase_null.py`.

## Question

R5 reports, per subject, the Spearman correlation between each state's Friends recurrence score and its mean fractional occupancy when the frozen Friends PCA+HMM is decoded on Movie10. Both sides of that correlation inherit low-level temporal structure from the same brain in the same PC subspace, so a raw rho overstates how much of the correspondence is specific to the stimulus. How much of the observed correlation survives a null that keeps that shared structure intact?

## Method (as run)

- **Statistic.** Per subject, Spearman rho between the Friends recurrence score and the per-state mean fractional occupancy over Movie10 runs, computed over active states (recurrence > 0). Identical to the statistic reported in R5.
- **Faithfulness gate.** The script rebuilds a standalone diagonal-Gaussian Viterbi from the saved model parameters (`startprob_`, `transmat_`, `means_`, `covars_`) and requires it to reproduce the published R5 rho before any null is drawn. Tolerance 1e-12; observed |delta| = 0 for all six subjects. The gate also aborts if the observed rho is non-finite, if the loaded Movie10 run count or active-state count does not match the published `cross_stimulus_summary.json`, or if the saved model's covariance type is not diagonal or its dimensionality does not match the subject's `n_pcs` lookup. A null computed on a different statistic than the manuscript reports would be worthless, so this is a hard abort rather than a warning.
- **Surrogate.** Multivariate (shared-phase) Prichard-Theiler FFT surrogate of each Movie10 run's PC-score matrix. One random phase vector is drawn per frequency and applied identically across all principal components; DC is held at zero phase and, for even run lengths, so is Nyquist. Only the Movie10 side is resampled; the Friends recurrence scores are fixed.
- **What the null preserves:** each component's power spectrum (hence its autocorrelation), each component's variance, the cross-component covariance, the run count, and the run lengths. Because the phase rotation is common to every component, the full cross-spectrum survives, so the covariance matrix is preserved exactly. Per-component independent phase randomization would destroy it; `test_rel_r5_phase_null.py` asserts both halves of that contrast.
- **What the null destroys:** stimulus-locked phase alignment and all higher-order temporal structure.
- **Draws and seeds.** 10,000 draws per subject; the CLI's `--n_null` default is 10,000, matching this published run, and `--out_dir` overrides the output location for exploratory runs without touching the published tree. Draw s uses `numpy.random.default_rng(s)`, so the sequence is deterministic, a longer run is a strict superset of a shorter one, and any single draw can be regenerated alone. The earlier 100-draw exploratory run is therefore an exact prefix of this one.
- **Inference.** One-sided empirical p = (1 + #{null >= real}) / (1 + n_finite), computed by `utils.stats.permutation_pvalue`, which excludes any non-finite surrogate draws and adjusts the denominator accordingly. Floor is 1/10001 = 0.0001 when all 10,000 draws are finite, which holds for all six subjects here. z = (real - null mean) / null sd is reported as a descriptive standardized distance, not as a second test; `utils.stats.null_summary` reports z as JSON null when the null sd is zero, a case that does not occur in this run. Per-subject only; no statistic is pooled across subjects.
- **Reporting the margin.** The quantity `delta_rho = rho_observed - E[rho_null]` measures how far the observed correspondence sits above the null expectation. It is **not** a decomposition: a Spearman correlation does not split additively into a shared-structure part and a stimulus-specific part, so delta_rho must not be described as "the stimulus-specific component" or as a residual.
- **HP/PP arrays.** Submitted 2026-09-19. Harry Potter: SLURM array job `23087550` (`r5null_hp`), tasks 0-5, all COMPLETED (`sacct -j 23087550,23087551 -X -n -o JobID,JobName,State,Elapsed`); elapsed 00:06:14-00:07:16 per subject, sub-04 00:00:06 via the skip path (no HP scans). Petit Prince: SLURM array job `23087551` (`r5null_pp`), tasks 0-5, all COMPLETED; elapsed 00:14:28-00:17:41 per subject, sub-04 00:00:06 via the skip path (no PP scans). All 12 tasks COMPLETED; no errors in the array logs.

## Results (10,000 draws, gate |delta| = 0 for all subjects)

| sub | observed rho | null mean | null sd | null 2.5% | null 97.5% | delta rho | z | p | null as % of observed |
|---|---|---|---|---|---|---|---|---|---|
| sub-01 | 0.5486 | 0.4502 | 0.0161 | 0.4169 | 0.4805 | +0.0984 | +6.11 | 0.0001 | 82% |
| sub-02 | 0.2637 | 0.1364 | 0.0159 | 0.1044 | 0.1665 | +0.1273 | +8.00 | 0.0001 | 52% |
| sub-03 | 0.5179 | 0.2942 | 0.0248 | 0.2443 | 0.3415 | +0.2237 | +9.02 | 0.0001 | 57% |
| sub-04 | 0.7533 | 0.6841 | 0.0116 | 0.6609 | 0.7061 | +0.0692 | +5.97 | 0.0001 | 91% |
| sub-05 | 0.6096 | 0.3974 | 0.0136 | 0.3708 | 0.4240 | +0.2122 | +15.62 | 0.0001 | 65% |
| sub-06 | 0.7731 | 0.6596 | 0.0156 | 0.6288 | 0.6894 | +0.1135 | +7.28 | 0.0001 | 85% |

Ranges across subjects: observed rho 0.264 to 0.773; null mean 0.136 to 0.684; delta rho 0.069 to 0.224; z 5.97 to 15.62; p at the 0.0001 floor in all six.

## Figure

`script/fig_sm_rel_r5_phase_null.py` (marimo; headless via `uv run python script/fig_sm_rel_r5_phase_null.py`) renders `fig_sm_r5_phase_null_A_null_vs_observed.{png,svg}` to `$SCRATCH_DIR/output/manuscript_figures/fig_sm_r5_phase_null/`: 2x3 small multiples (sub-01 to sub-06, row-major), each cell the 10,000-draw null histogram with the observed rho as a vertical accent line, annotated with delta rho and z. Shared x-range across cells keeps the observed-null distance comparable across subjects. The loader cell asserts the draws reproduce the summary JSON's null moments and that the faithfulness gate passed, so the figure cannot silently render stale draws.

**Two readings, both true.** The observed correlation exceeded the null in every subject, at the empirical floor, so the correspondence is not an artifact of Movie10's spectrum and covariance. At the same time the null mean accounts for 52% to 91% of the observed rho in each subject, so most of the raw correlation's magnitude is compatible with structure that has nothing to do with stimulus content. The effect is real and modest; reporting the raw rho alone overstates it.

## Relation to the earlier 100-draw run

The exploratory 100-draw run (memo `2026-06-29_R5_covariance_null_finding.md` in the backup repo) reported z = +6.45 to +17.85 and delta rho 0.069 to 0.220. Because seeds are the draw index, its 100 draws are the first 100 of this run. The observed rhos and delta rhos agree; the z values differ because the null sd is estimated far more stably at 10,000 draws, and the p floor moves from 0.01 to 0.0001. **The 10,000-draw table above supersedes the memo for every reported statistic.**

## Results — Harry Potter (reading, 7 runs, n = 5)

sub-04 has no Harry Potter scans and is skipped. 10,000 draws, gate |delta| = 0 for all five subjects.

| sub | observed rho | null mean | null sd | null 2.5% | null 97.5% | delta rho | z | p | null as % of observed |
|---|---|---|---|---|---|---|---|---|---|
| sub-01 | 0.2318 | 0.2531 | 0.0331 | 0.1899 | 0.3185 | -0.0212 | -0.64 | 0.7338 | 109% |
| sub-02 | 0.4439 | 0.4364 | 0.0257 | 0.3855 | 0.4862 | +0.0075 | +0.29 | 0.3877 | 98% |
| sub-03 | 0.3672 | 0.4471 | 0.0248 | 0.3966 | 0.4948 | -0.0799 | -3.22 | 0.9988 | 122% |
| sub-05 | 0.2581 | 0.1700 | 0.0318 | 0.1091 | 0.2334 | +0.0881 | +2.77 | 0.0037 | 66% |
| sub-06 | 0.2940 | 0.3571 | 0.0262 | 0.3047 | 0.4072 | -0.0631 | -2.41 | 0.9897 | 121% |

1 of 5 participants exceeds the null at p < 0.05 (sub-05, p = 0.0037). The other four are not distinguishable from their null (p = 0.39-1.00), and two of them (sub-03, sub-06) sit below their null mean (negative delta rho). Null sd ranges 0.025 to 0.033 across subjects — wider than the Movie10 range (0.012-0.025) — consistent with the run-count-asymmetry caveat below. The null mean accounts for 66% to 122% of the observed rho. Per the caveat above, the empirical p is the inferential statistic here; z is reported for consistency with the Movie10 table but is descriptive only.

## Results — Le Petit Prince (listening, FR + EN pooled, 16–18 runs, n = 5)

sub-04 has no Petit Prince scans and is skipped; sub-06 has 16 runs (lppFR 7 of 9 present, lppEN 9), the other four subjects have 18. 10,000 draws, gate |delta| = 0 for all five subjects.

| sub | observed rho | null mean | null sd | null 2.5% | null 97.5% | delta rho | z | p | null as % of observed |
|---|---|---|---|---|---|---|---|---|---|
| sub-01 | 0.0564 | -0.0042 | 0.0271 | -0.0582 | 0.0468 | +0.0606 | +2.24 | 0.0109 | -7% |
| sub-02 | 0.3323 | 0.3275 | 0.0167 | 0.2949 | 0.3613 | +0.0048 | +0.29 | 0.3865 | 99% |
| sub-03 | 0.1331 | 0.1257 | 0.0252 | 0.0766 | 0.1749 | +0.0074 | +0.29 | 0.3848 | 94% |
| sub-05 | -0.0003 | -0.1867 | 0.0205 | -0.2266 | -0.1467 | +0.1863 | +9.08 | 0.0001 | n/a\* |
| sub-06 | -0.0185 | -0.0211 | 0.0234 | -0.0679 | 0.0246 | +0.0026 | +0.11 | 0.4603 | 114% |

\*sub-05's observed rho (-0.0003) is ≈0, so the null-mean-as-percentage-of-observed ratio (53,801%) is not informative; delta rho (+0.1863) and the null 95% interval (-0.2266, -0.1467), which sits entirely below zero while the observed value sits at zero, are the interpretable summaries for that subject.

2 of 5 participants exceed the null at p < 0.05 (sub-01, p = 0.0109; sub-05, p = 0.0001). The other three (sub-02, sub-03, sub-06) are not distinguishable from their null (p = 0.38-0.46). Null sd ranges 0.017 to 0.027 across subjects. Excluding sub-05's degenerate ratio, the null mean's share of the observed rho ranges from -7% (sub-01, whose null mean is slightly negative) to 114% (sub-06). **Serial-dependence caveat (carried from `pp_05`).** Consecutive Petit Prince chapters show substantial lag-1 FO autocorrelation (mean 0.49-0.81 across subjects and languages; `pp_05_cross_stimulus_validation.md`), and this null draws an independent phase rotation per run, so within-run phase randomization does not absorb that across-run carryover. As in `pp_05`, this is an upper-bound caveat: the Petit Prince p-values here should be read as approximate, and the true null variance may be understated.

## Caveats and audit notes

- **The surrogate is not a model of what the brain would do without stimulus structure.** It is a null for one specific alternative: that the correspondence follows from second-order structure (spectrum plus covariance) shared between any two scans of the same subject in the same PC subspace. Other alternatives, including head motion and arousal, are addressed by different analyses.
- **Recurrence is close to stationary occupancy.** The recurrence score correlates with the fitted model's stationary distribution at rho = 0.97 to 0.98 across subjects. That relation is a property of the Friends side alone and is untouched by this null, so it constrains the interpretation of a surviving effect rather than its significance.
- **z is descriptive.** With 10,000 draws the empirical p is floored for every subject, so z is the only quantity that still ranks subjects. It assumes an approximately Gaussian null and should not be read as a separate test.
- **Run-count asymmetry.** Harry Potter (7 runs) and Le Petit Prince (18; 16 for sub-06) yield wider nulls than Movie10 (61): the surrogate statistic averages FO over the same few runs the observed statistic does, so the null is valid but the test is lower-powered. For these stimuli the empirical p is the inferential statistic and z is descriptive; the modality comparison across stimuli remains descriptive (no test of Δρ differences).
- **The ICA analogue lives elsewhere.** `sm_alt_ica_oos_recurrence.py` computes the same style of null for the ICA supplement, calling the same `utils.ica_oos_recurrence.phase_randomize` helper, so both nulls preserve identically what they claim to preserve.

## Connections

- Main analysis: Results R5, Methods "Cross-stimulus correspondence".
- `docs/findings/sm_alt_ica_oos_recurrence.md` — ICA analogue of the same test. That arm already covers all three OOS stimuli (Movie10, Harry Potter, Petit Prince; Figure S12); with this doc's HP and PP sections, the main-text HMM arm now matches that coverage rather than being Movie10-only.
