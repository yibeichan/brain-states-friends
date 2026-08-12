# Findings: R5 phase-randomized null (`sm_rel_r5_phase_null`)

Canonical null for main-analysis Results R5. Code: `script/sm_rel_r5_phase_null.py` (+`.sh` runner); tests `script/tests/test_rel_r5_phase_null.py`.

## Question

R5 reports, per subject, the Spearman correlation between each state's Friends recurrence score and its mean fractional occupancy when the frozen Friends PCA+HMM is decoded on Movie10. Both sides of that correlation inherit low-level temporal structure from the same brain in the same PC subspace, so a raw rho overstates how much of the correspondence is specific to the stimulus. How much of the observed correlation survives a null that keeps that shared structure intact?

## Method (as run)

- **Statistic.** Per subject, Spearman rho between the Friends recurrence score and the per-state mean fractional occupancy over Movie10 runs, computed over active states (recurrence > 0). Identical to the statistic reported in R5.
- **Faithfulness gate.** The script rebuilds a standalone diagonal-Gaussian Viterbi from the saved model parameters (`startprob_`, `transmat_`, `means_`, `covars_`) and requires it to reproduce the published R5 rho before any null is drawn. Tolerance 1e-12; observed |delta| = 0 for all six subjects. A null computed on a different statistic than the manuscript reports would be worthless, so this is a hard abort rather than a warning.
- **Surrogate.** Multivariate (shared-phase) Prichard-Theiler FFT surrogate of each Movie10 run's PC-score matrix. One random phase vector is drawn per frequency and applied identically across all principal components; DC is held at zero phase and, for even run lengths, so is Nyquist. Only the Movie10 side is resampled; the Friends recurrence scores are fixed.
- **What the null preserves:** each component's power spectrum (hence its autocorrelation), each component's variance, the cross-component covariance, the run count, and the run lengths. Because the phase rotation is common to every component, the full cross-spectrum survives, so the covariance matrix is preserved exactly. Per-component independent phase randomization would destroy it; `test_rel_r5_phase_null.py` asserts both halves of that contrast.
- **What the null destroys:** stimulus-locked phase alignment and all higher-order temporal structure.
- **Draws and seeds.** 10,000 draws per subject. Draw s uses `numpy.random.default_rng(s)`, so the sequence is deterministic, a longer run is a strict superset of a shorter one, and any single draw can be regenerated alone. The earlier 100-draw exploratory run is therefore an exact prefix of this one.
- **Inference.** One-sided empirical p = (1 + #{null >= real}) / (1 + n_draws), floor 1/10001 = 0.0001 at 10,000 draws. z = (real - null mean) / null sd is reported as a descriptive standardized distance, not as a second test. Per-subject only; no statistic is pooled across subjects.
- **Reporting the margin.** The quantity `delta_rho = rho_observed - E[rho_null]` measures how far the observed correspondence sits above the null expectation. It is **not** a decomposition: a Spearman correlation does not split additively into a shared-structure part and a stimulus-specific part, so delta_rho must not be described as "the stimulus-specific component" or as a residual.

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

## Caveats and audit notes

- **The surrogate is not a model of what the brain would do without stimulus structure.** It is a null for one specific alternative: that the correspondence follows from second-order structure (spectrum plus covariance) shared between any two scans of the same subject in the same PC subspace. Other alternatives, including head motion and arousal, are addressed by different analyses.
- **Recurrence is close to stationary occupancy.** The recurrence score correlates with the fitted model's stationary distribution at rho = 0.97 to 0.98 across subjects. That relation is a property of the Friends side alone and is untouched by this null, so it constrains the interpretation of a surviving effect rather than its significance.
- **z is descriptive.** With 10,000 draws the empirical p is floored for every subject, so z is the only quantity that still ranks subjects. It assumes an approximately Gaussian null and should not be read as a separate test.
- **Movie10 only.** Harry Potter and Le Petit Prince have far fewer runs per subject (7 and 18 versus 61), so a per-run surrogate null would be estimated from too few independent segments; this analysis does not extend to them.
- **The ICA analogue lives elsewhere.** `sm_alt_ica_oos_recurrence.py` computes the same style of null for the ICA supplement, calling the same `utils.ica_oos_recurrence.phase_randomize` helper, so both nulls preserve identically what they claim to preserve.

## Connections

- Main analysis: Results R5, Methods "Cross-stimulus correspondence".
- `docs/findings/sm_alt_ica_oos_recurrence.md` — ICA analogue of the same test.
