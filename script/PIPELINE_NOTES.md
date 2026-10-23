# Pipeline design notes

Design decisions behind the main pipeline stages: preprocessing, parcel time
series extraction, state discovery and recurrence scoring. For what each script
does and its manuscript tier, see [MANIFEST.md](MANIFEST.md); for state
definitions and the constants that implement them, see
[docs/findings/definitions.md](../docs/findings/definitions.md).

## Preprocessing Strategy

The preprocessing pipeline (`00_postproc.py`) follows a **minimal confound regression approach** optimized for naturalistic viewing:

**Key Preprocessing Decisions:**
1. **Minimal Confounds:** 6 motion parameters + 2 WM/CSF means + high-pass filter (~13-18 regressors total)
2. **Conservative Denoising:** WM/CSF mean signals instead of CompCor to preserve stimulus-evoked responses
3. **Voxel-wise Z-scoring:** Applied BEFORE parcellation to ensure equal contribution of all voxels within parcels
4. **No Scrubbing:** Maintains narrative continuity (dataset has low motion)
5. **No Global Signal Regression:** Preserves global patterns in naturalistic responses

**Rationale:**
- NeuroMod dataset documentation recommends "minimal strategy" for low-motion data
- Naturalistic viewing literature favors conservative preprocessing (Nastase et al., 2020; Chen et al., 2017)
- Brain state discovery requires preserving genuine spatiotemporal patterns
- Voxel-wise normalization before parcellation prevents baseline intensity artifacts

## Parcel Time Series Extraction Strategy

The parcel time series extraction pipeline (`02_extract_parcel_ts.py`) extracts both **unthresholded average** and **binary activation** representations for brain state analysis:

**Key Design Decisions:**
1. **Two Output Types:** Average activation (continuous) + Binary activation (thresholded)
2. **Binary Activation:** Direct thresholding on the temporal z-scores from script 00 (no double z-scoring), so it represents parcels with significant temporal activation above baseline
3. **Episode-Level Parallelization:** Each SLURM task processes one episode for 96× speedup
4. **Robust Failure Detection:** Proper exit codes enable easy reprocessing of failed episodes
5. **Path Robustness:** Absolute path handling prevents directory-dependent failures

**Parallel Processing:**
- Single episode mode: `--subject_id sub-01 --parcellation atlas-4S156Parcels --episode_id task-s01e01a`
- Batch mode: `--subject_id sub-01 --parcellation atlas-4S156Parcels`
- SLURM array: 96 episodes processed in parallel (~15 min vs ~24 hours sequential)

**Reprocessing Failed Episodes:**
1. Identify failures: `grep "ERROR" logs/*.err`
2. Create episode list file with failed episode IDs
3. Update shell script variables and array size
4. Resubmit: `sbatch 02_extract_parcel_ts.sh`

## HDP-HMM State Discovery Strategy

The brain state discovery pipeline uses a per-subject **Gaussian hidden Markov model** with **sticky and hierarchical-Dirichlet transition priors** (borrowed from the sticky HDP-HMM, sHDP-HMM, but applied under a fixed-capacity weak-limit truncation rather than full nonparametric inference) to identify latent brain states from fMRI time series. The primary analysis fits **one combined model per subject across all episodes** (`04_combined_hdphmm.py`), taking PCA-projected data from `03a_pca4combined_hmm.py` as input.

**Current main-analysis config (all 6 subjects, vt=0.95): `vt0.95_covdiag_nc50_g1`**

| Parameter | Value | Role |
|---|---|---|
| Parcellation | atlas-4S156Parcels | 156-parcel ROI-level (Schaefer 100 cortical + 56 subcortical composite) |
| PCA space | 67–77 PCs (vt=0.95) | HMM fits in PC space, NOT in parcel space |
| nc | 50 | Truncation capacity. Two different counts circulate, and they are not interchangeable: final-refit **K_used** (posterior usage > 0.01 at the last EM iteration) is 37–42 per subject (mean 40.8, 74–84% utilization), while the **active repertoire** used by 05a and the cross-stimulus scripts (recurrence > 0, i.e. FO > 0.02 in at least one episode) is larger at 42–47 (mean 44.8, 269 states total). LOSO folds give K_used 36–43 and split halves 36–41. |
| γ (gamma) | 1 | Low HDP concentration; prior prefers fewer states, so arriving at K_used ≈ 41 is strong evidence those states are real. Compare like with like: the prior-predictive simulation counts states above a 1% usage threshold, so the matching data-side count is K_used, not the larger active repertoire |
| κ (kappa) | 10 | Sticky bias; encourages multi-TR state persistence (~hemodynamic timescale) |
| α (alpha) | 1 | Row-level transition concentration |
| ρ (rho) | 1 | Sticky bias scaling |
| covariance | **diag** (in PC space) | Each state = per-PC mean + diagonal variance. Back-projected to parcel space, state covariance is non-diagonal (PCA mixing), so states still carry within-state coupling structure when visualized in brain-region space. |

**Model selection uses Pareto analysis (LL vs K_used), NOT BIC or any LL-based metric:** all LL-based scores (raw LL, LL/dim, BIC, 6-metric battery) systematically favor higher complexity. The Pareto middle cluster across subjects is K≈35–45; nc=50 γ=1 is the simplest model in that cluster.

**Validation:**
- LOSO refits (6 folds per subject, each with its own PCA to avoid leakage) test whether discovered states generalize across narrative contexts.
- Split-half (interleaved episode halves) reliability: 04rb Spearman ρ = 0.48–0.84 (recurrence at FO > 0.02 per run).

## Brain State Recurrence Analysis

Script `05a_recurrence_analysis.py` scores each brain state's recurrence across episodes, placing states along a **continuous recurrence gradient** rather than two discrete classes. This addresses the central research question: **Do brain states reflect recurring structure across episodes, or are they shaped by episode-specific content?**

**Prerequisites:**
- `04_combined_hdphmm.py` (mode: select) completed; produces `decoded_states.pkl` and `final_results.json`

**Core Analysis Pipeline:**
1. **Fractional Occupancy (FO):** Per-state, per-episode occupancy fraction
2. **Recurrence Score:** Fraction of episodes where each state is active (FO > threshold)
3. **Season Specificity Index:** Range of per-season recurrence (0 = invariant, 1 = specific)
4. **Permutation Test:** Shuffles season labels to test whether specificity exceeds chance
5. **FDR Correction:** Benjamini-Hochberg correction for multiple comparisons across states

**State scoring (continuous, not categorical):** Recurrence is treated as a continuous score, NOT binned into categorical groupings (Recurring / Specific / Partial / Inactive). Downstream scripts (05d similarity, 06a dynamics) all consume the continuous scores directly. Eligibility filtering (e.g., sub-HRF states) is handled via the `state_flags.csv` mechanism produced by 05e, not by the recurrence score.

**Key Outputs (saved to `{SCRATCH_DIR}/output/05a_recurrence_analysis/{parcellation}/{sub_id}/`):**
- `recurrence_scores.npy`: Per-state recurrence scores
- `specificity_index.npy`: Per-state season-specificity indices
- `fractional_occupancy.pkl`: Per-run, per-state FO
- `permutation_pvalues.json`: Uncorrected and FDR-corrected p-values
- `recurrence_summary.json`: Full results including per-state recurrence scores and specificity indices

**Usage:**
```bash
# Standard recurrence analysis (after 04 select)
python script/05a_recurrence_analysis.py \
    --sub_id sub-01 --parcellation atlas-4S156Parcels

# With threshold sweep for robustness check
python script/05a_recurrence_analysis.py \
    --sub_id sub-01 --parcellation atlas-4S156Parcels --threshold_sweep
```
