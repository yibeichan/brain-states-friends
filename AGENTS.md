# AGENTS.md

This file provides guidance to AI coding agents (e.g. Claude Code) when working with code in this repository.

## MANDATORY: Three Parallel Review Agents

For every planning session and code review, you MUST spawn **three parallel review agents** using the Task tool (subagent_type=Explore or Plan as appropriate). Launch all three in a single message. Do not skip any agent, even for seemingly simple tasks.

| Agent role | Focus |
|---|---|
| **Stats agent** | Verify every mathematical and statistical formula, derivation, and assumption step by step. Flag incorrect formulas, inappropriate model choices, or invalid statistical claims. |
| **Coding agent** | Verify all code implementations for correctness, efficiency, and optimization. Check for bugs, off-by-one errors, vectorization opportunities, memory usage, and cluster-appropriate practices (SLURM, scratch dir usage). |
| **Neuroscience agent** | Verify all decisions against cognitive neuroscience and neuroimaging best practices. Flag biologically implausible assumptions, inappropriate parcellation choices, invalid fMRI analysis steps, or claims inconsistent with the literature. |

**Veto rule:** Any concern raised by a review agent must be explicitly resolved before the plan is finalized or code is written. Document the resolution in the plan file or code comment.

## Project Overview

This project analyzes brain states during longitudinal TV watching (specifically the TV show "Friends") using fMRI data. It quantifies context-invariant and context-variant brain states across multiple subjects and episodes using Sticky Hierarchical Dirichlet Process Hidden Markov Models (HDP-HMM).

## Analysis Scope

- Treat analyses, interpretations, threshold choices, and follow-up recommendations as **subject-specific** unless the user explicitly asks otherwise.
- Do **not** propose or assume across-subject aggregation, group statistics, or multi-subject validation as the default next step.
- When discussing outputs from scripts such as `04`, `05a`, `05d` (state similarity), or `06`, frame conclusions at the single-subject level.

### Introduction

What does the brain do when we watch TV shows? To track who is doing what, where, how, and why, the brain transitions through dynamic states, spatiotemporal patterns of activity and coupling that shape perception, inference, and memory. But do these states reflect shared, recurring structure across episodes, or are they shaped by episode-specific content? Using fMRI data from six seasons of Friends, we fit a single combined sticky hierarchical Dirichlet process hidden Markov model (sHDP-HMM) per subject across all episodes (with PCA dimensionality reduction to manage data volume) and classified the resulting states as context-invariant or episode-specific using fractional occupancy and permutation tests.

### Methods

We analyzed fMRI data from six participants who each viewed all six seasons of the television show Friends (292 runs total, ~11 minutes each, TR = 1.49 s). Brain activity was parcellated using the Schaefer 100 cortical atlas (Yeo 7-network) and a subcortical composite (CIT168 + ThalamusHCP + SubcorticalHCP + Cerebellum; 56 parcels) for the primary 156-parcel analysis. PCA was fitted on training data only (70/15/15 season-stratified split) and used to project each episode's time series before model fitting (script 03a). PCA loadings diagnostics were generated (script 03b). We then fit a single combined sHDP-HMM per subject across all episodes to discover latent brain states without pre-specifying their number (script 04). States were classified as context-invariant or episode-specific by computing fractional occupancy per episode, a recurrence score (fraction of episodes where each state is active), a season specificity index, and a permutation test with FDR correction (script 05a). Dwell time distributions and temporal dynamics were examined across states (script 06). Leave-one-season-out refits validated that discovered states generalize across narrative contexts.


## Environment Setup

The project uses a dual workflow:
- `environment.yml` for micromamba bootstrap (`python`, `pip`, `uv`)
- `pyproject.toml` for Python dependencies (managed by `uv`)

```bash
micromamba env create -f environment.yml
micromamba activate friends-states
uv sync
```

If using abbreviated command:
```bash
mmb activate friends-states
```

## Key Commands

### Running the Pipeline

The pipeline consists of sequential steps (00-06), each with corresponding Python scripts and SLURM submission scripts.

**Important**: Steps 03a onward require `SUB_ID` and `PARCELLATION` to be set (defaults: `sub-01`, `atlas-4S156Parcels`).

```bash
# Step 0: Post-process fMRIPrep data
sbatch script/00_postproc.sh

# Step 1: Extract parcel labels
sbatch script/01_get_parcel_label.sh

# Step 2: Extract parcel time series
sbatch script/02_extract_parcel_ts.sh

# Step 3a: PCA fitting + train/valid/test splits for combined HMM
sbatch --export=SUB_ID=sub-01 script/03a_pca4combined_hmm.sh

# Step 3b: PCA loadings analysis (A1-A7 diagnostics)
sbatch --export=SUB_ID=sub-01 script/03b_pca_loadings.sh

# Step 4: Fit combined cross-season sHDP-HMM (fit mode: grid search over configs)
sbatch --export=SUB_ID=sub-01 script/04_combined_hdphmm.sh

# Step 4: Select best config + decode all episodes (select mode)
sbatch --export=SUB_ID=sub-01,MODE=select script/04_combined_hdphmm.sh

# Step 5a: Classify states as context-invariant vs episode-specific
sbatch --export=SUB_ID=sub-01 script/05a_recurrence_analysis.sh

# Step 5b: Visualize top recurring states on brain surface
sbatch --export=SUB_ID=sub-01 script/05b_visualize_recurring_states.sh

# Step 6a: Analyze temporal dynamics (dwell time, transition matrices, assortativity)
sbatch --export=SUB_ID=sub-01 script/06a_state_temp_dynamics.sh

# Step 6a: Cross-subject summary (run after all 06a per-subject jobs)
sbatch script/06a_cross_subject_summary.sh

# Step 6b: Transition structure (graph topology, asymmetry, MFPT landscape)
sbatch --export=SUB_ID=sub-01 script/06b_transition_structure.sh

# Step 7a: Extract TR-aligned physio features (Friends)
sbatch --export=SUB_ID=sub-01 script/07a_physio_features.sh

# Step 7a: Extract TR-aligned physio features (Movie10)
sbatch --export=SUB_ID=sub-01 script/m10_07a_physio_features.sh

# Step 7b: Physio-state correspondence (Friends)
sbatch --export=SUB_ID=sub-01 script/07b_physio_state_correspondence.sh

# Step 7c: Cross-stimulus physio correspondence (Friends vs Movie10)
sbatch --export=SUB_ID=sub-01,VT=0.99 script/07c_cross_stimulus_physio.sh

# Petit Prince cross-stimulus pipeline (audio-only, 5 subjects, no sub-04)
# Step PP-00: Post-process Petit Prince fMRIPrep data
sbatch script/pp_00_postproc.sh

# Step PP-02: Extract parcel time series
sbatch --export=SUBJECT_ID=sub-01 script/pp_02_extract_parcel_ts.sh

# Step PP-03: Project PP data through Friends-trained PCA
sbatch script/pp_03_project_pp_pca.sh

# Step PP-04: Score and decode PP data with Friends HMM
sbatch script/pp_04_score_and_decode.sh

# Step PP-05: Cross-stimulus validation (A1-A5, B1-B2, language comparison)
sbatch script/pp_05_cross_stimulus_validation.sh

# Step 08c: Transformer feature extraction (GPU required, uv sync --extra torch)
sbatch script/08c_transformer_features.sh                                    # Friends, all 3 models
sbatch --export=STIMULUS=movie10 script/08c_transformer_features.sh          # Movie10
sbatch --export=STIMULUS=harrypotter --array=2 script/08c_transformer_features.sh  # HP, text only

# Step 08d: Transformer-state correspondence (CPU, per-subject)
sbatch script/08d_transformer_state_correspondence.sh                        # Friends, D1+D2
sbatch --export=STIMULUS=movie10,MODEL=llama-3.2-3b,ANALYSES="D1 D3a",TRANSFER_FROM=friends \
    script/08d_transformer_state_correspondence.sh                           # M10 transfer
```

### Saving Outputs with DataLad

Analysis outputs are versioned with DataLad and synced to persistent storage. After any pipeline stage completes, save and push:

```bash
# Save outputs for a specific subject and stage
bash script/utils/datalad_save.sh --stage 04 --subject sub-01

# Or chain as SLURM dependency
JOB=$(sbatch --parsable --export=SUBJECT_ID=sub-01,MODE=select script/04_combined_hdphmm.sh)
sbatch --dependency=afterok:$JOB --export=STAGE=04,SUBJECT_ID=sub-01 script/utils/datalad_save.sh

# Save all subjects for a stage
bash script/utils/datalad_save.sh --stage 05a --subject all
```

Valid stages: `00`, `02`, `03a`, `03b`, `04`, `05a`, `05b`, `05c`, `05d`, `05e`, `06`, `07a`, `07b`, `07c`, `m10_03`, `m10_04`, `m10_07a`, `diag`

**Note:** Stage `00` outputs have no parcellation level; the path is `00_postproc/{sub_id}/`. All other stages use `{stage_dir}/{parcellation}/{sub_id}/`.

**Recovery after scratch purge:**
```bash
cd "$SCRATCH_DIR"
datalad clone "ria+file:///path/to/ria-store/brain-states-friends-outputs" output
cd output && datalad get 04_combined_hdphmm/  # lazy: get only what you need
```

**Do NOT run `datalad save` concurrently**; serialize save jobs or use `--subject all`.

### Utility Commands

```bash
# Check for failed SLURM tasks
python bash_monitor/get_failed_task.py

# Find empty folders in results
bash bash_monitor/find_empty_folders.sh

# Find missing episodes in processed data
bash bash_monitor/find_missing_episodes.sh
```

## Architecture and Key Patterns

### Pipeline Architecture
- Sequential processing pipeline (00→02→03a→03b→04→05a→05b→06→07a→07b→07c) where each step depends on the previous
- SLURM array jobs for parallel processing across subjects
- Results stored hierarchically: `{SCRATCH_DIR}/results/{step_name}/{parcellation}/{subject}/`

### Environment Variables
The `.env` file defines critical paths:
- `BASE_DIR`: Main project directory
- `SCRATCH_DIR`: Where intermediate results are stored
- `DATA_DIR`: Location of raw fMRI data

### Key Processing Steps
1. **Post-processing** (00): Cleans fMRIPrep outputs, applies confound regression
2. **Parcellation** (01–02): Supports multiple schemes (atlas-4S156 primary, atlas-4S456/1056 for validation)
3. **PCA** (03a): Fits PCA on training data; projects episode time series for HMM input
4. **PCA Loadings** (03b): PCA loadings analysis (A1-A7 diagnostics)
5. **Combined HDP-HMM** (04): Fits a single sHDP-HMM per subject across all episodes (fit → select → optional loso_fit modes). Leave-one-season-out refits test whether discovered states generalize across narrative contexts.
6. **Recurrence Analysis** (05a): Classifies states as context-invariant or episode-specific via fractional occupancy, permutation tests, and FDR correction
7. **Visualization** (05b): Surface plots of top recurring states
8. **Temporal Dynamics** (06a): Dwell time distributions, transition matrices, recurrence assortativity, cross-subject summary. Key finding: recurrence and persistence are independent dimensions.
8b. **Transition Structure** (06b): Graph topology, transition asymmetry, MFPT landscape, FC-transition correlation. Replaces old chain analysis.
8c. **Higher-Order Transitions** (06c): Model adequacy diagnostic; conditional entropy reduction (~14%), BIC prefers order-1
9. **Physio Feature Extraction** (07a): TR-aligned physiological features (HR, HRV, breathing, RVT, EDA, SCR) from physprep data. Supports both Friends (`--stimulus friends`) and Movie10 (`--stimulus movie10`). Output is parcellation-independent.
10. **Physio-State Correspondence** (07b): Post-hoc test of physio-state associations (Friends). 5 analyses: state profiles, multi-lag, TTAs, cross-episode consistency, arousal-diversity.
11. **Cross-Stimulus Physio** (07c): Tests whether brain states maintain autonomic signatures across Friends and Movie10. 4 analyses: signature stability, genre profiles, arousal modulation, cross-stimulus TTAs.
12. **Content Features** (08a): TR-level content features from te-charnet narrative annotations (Friends only). 16 features: dialogue structure, scene features, character presence.
13. **Content-State Correspondence** (08b): Tests association between content features and brain states (Friends). 7 analyses. A1 (per-state content signatures) and A3 (per-state multi-lag) use a per-(state, feature) Mann-Whitney AUC framework with a within-run circular-shift permutation null and two-layer BH-FDR.
14. **Transformer Feature Extraction** (08c): Layer-wise features from frozen pretrained transformers (TRIBEv2-validated backbones). Models: Wav2VecBert 2.0 (audio, 24L), DINOv2-large (video, 24L), LLaMA 3.2 3B (text, 28L). Supports Friends, Movie10, HP, PP-FR, PP-EN. Output is stimulus-level (no sub_id). Install: `uv sync --extra torch`.
15. **Transformer-State Correspondence** (08d): Tests brain state correspondence with transformer layer features. Analyses: D1 (representational depth per layer), D2 (per-state layer selectivity), D3a (cross-stimulus transfer), D5 (annotation convergence with 08b).

### Parallel Processing
- Uses SLURM array jobs for subject-level parallelization
- Within-script parallelization using joblib for episode processing
- Typical array configuration: `--array=0-5` for 6 subjects

### Data Organization
- Follows BIDS-like structure for input data
- Output organized by: `{parcellation}/{subject}/` (steps 05-06)
- HMM models saved as pickle files with metadata
- Linkage matrices and dendrograms saved with parameter-specific names

## Research Questions

### Primary Research Question
Do brain states during naturalistic TV viewing reflect shared, recurring structure across episodes, or are they shaped by episode-specific content?

### Key Sub-Questions the Methods Can Address

#### 1. State Discovery & Characterization
- What are the fundamental brain states that spontaneously emerge during naturalistic viewing?
- How many distinct brain states exist, and what are their spatial patterns?
- **Method**: sHDP-HMM discovers states without pre-specifying numbers

#### 2. Context Invariance vs. Specificity
- Which brain states are context-invariant (consistent across episodes) vs. context-variant (episode-specific)?
- Do certain brain states represent general viewing processes vs. content-specific responses?
- **Method**: Per-episode fractional occupancy, a recurrence score, a season-specificity index, and a permutation test with FDR correction (script 05a)

#### 3. Hierarchical Organization
- How are brain states organized hierarchically - do similar spatial regions show both activation and deactivation patterns?
- Are there "opposing" states that represent different polarities of the same networks?
- **Method**: State similarity analysis (script 05d) comparing back-projected state-mean patterns

#### 4. Temporal Dynamics
- How do brain states transition during viewing? What are characteristic dwell times?
- Are some states more "sticky" or persistent than others?
- **Method**: State sequence analysis, transition matrices, dwell time quantification

#### 5. Spatial Scale Robustness
- Are discovered brain states consistent across different spatial resolutions?
- Do findings generalize from ROI-level (156 parcels) to finer resolutions (456-1056 parcels)?
- **Method**: Primary analysis at 156 parcels (optimal power), validation at 456, 1056 parcels

#### 6. Reliability & Stability
- How stable and reproducible are the identified brain states?
- **Method**: Leave-one-season-out refits (script 04ra) and split-half reliability (script 04rb)

#### 7. Individual vs. Shared Structure
- What aspects of brain dynamics are shared across subjects vs. individual-specific?
- Do all subjects show similar state repertoires and transitions?
- **Method**: Multi-subject analysis with fractional occupancy comparisons

#### 8. Longitudinal Consistency
- Do brain state patterns remain stable across viewing sessions and seasons?
- **Method**: Six seasons (292 runs) of longitudinal data analysis

### Methodological Strengths
- **Unsupervised, data-driven approach**: Discovers the brain's "natural" organizational structure without imposing a priori assumptions
- **Multi-scale, multi-temporal analysis**: Provides convergent evidence across spatial resolutions and temporal dynamics
- **Naturalistic paradigm**: Real-world-like viewing experience rather than artificial laboratory tasks

### Limitations
- State–content and state–representation correspondence (scripts 08a–08d) is correlational
- No direct behavioral/cognitive correlates measured
- Observational (not causal) findings
- Post-hoc analysis (not real-time state identification)

## Implementation Notes

### Script Organization
The analysis pipeline is organized around a combined cross-season HMM:

1. **Main Pipeline Scripts** (numbered 03a–08d):
   - `03a_pca4combined_hmm.py` / `.sh`: PCA fitting + train/valid/test splits for combined HMM
   - `03b_pca_loadings.py` / `.sh`: PCA loadings analysis (A1-A7 diagnostics)
   - `04_combined_hdphmm.py` / `.sh`: Fit combined sHDP-HMM (fit / select / loso_fit modes)
   - `05a_recurrence_analysis.py` / `.sh`: Classify states as recurring vs episode-specific
   - `05b_visualize_recurring_states.py` / `.sh`: Cortical + subcortical surface plots (yabplot)
   - `06a_state_temp_dynamics.py` / `.sh`: Dwell time distributions, transition matrices, recurrence assortativity. Supports `--mode cross_subject_summary` for multi-panel aggregate figure.
   - `06b_transition_structure.py` / `.sh`: Graph topology, transition asymmetry, FC-transition correlation, MFPT landscape.
   - `06c_higher_order_transitions.py` / `.sh`: Model adequacy diagnostic (conditional entropy, BIC comparison)
   - `07a_physio_features.py` / `.sh` / `m10_07a_physio_features.sh`: TR-aligned physio extraction (Friends + Movie10)
   - `07b_physio_state_correspondence.py` / `.sh`: Physio-state correspondence (Friends, 5 analyses)
   - `07c_cross_stimulus_physio.py` / `.sh`: Cross-stimulus physio correspondence (C1-C4)
   - `08a_content_features.py` / `.sh`: TR-level content features from te-charnet annotations (Friends)
   - `08b_content_state_correspondence.py` / `.sh`: Content-state correspondence (Friends, 7 analyses). A1/A3 use a per-state Mann-Whitney AUC framework; shared helpers `per_state_auc_mann_whitney`, `per_state_auc_grid`, `two_layer_bh_fdr` in `utils/stats.py`.
   - `08c_transformer_features.py` / `.sh`: Layer-wise transformer feature extraction (TRIBEv2-validated: Wav2VecBert, DINOv2-large, LLaMA 3.2 3B). GPU required, `uv sync --extra torch`. Supports Friends, M10, HP, PP.
   - `08d_transformer_state_correspondence.py` / `.sh`: Transformer-state correspondence (D1 depth, D2 selectivity, D3a transfer, D5 convergence)

2. **Cross-Stimulus Scripts** (`m10_`, `hp_`, `pp_` prefixes):
   - `m10_03/04/05`: Movie10 PCA projection, HMM decode, cross-stimulus validation
   - `hp_03/04/05`: Harry Potter PCA projection, HMM decode, cross-stimulus validation
   - `pp_00_postproc.sh`: Petit Prince post-processing (uses `--fmriprep_dir` for separate dataset root)
   - `pp_02_extract_parcel_ts.sh`: Petit Prince parcel time series extraction
   - `pp_03_project_pp_pca.py` / `.sh`: Project PP data through Friends PCA (2 types: lppFR, lppEN)
   - `pp_04_score_and_decode.py` / `.sh`: Score/decode PP with Friends HMM
   - `pp_05_cross_stimulus_validation.py` / `.sh`: Cross-stimulus validation (A1-A5, B1-B2, language comparison)

3. **Utility Scripts** (in `script/utils/`):
   - `physio_io.py`: Physprep I/O, BIDS entity parsing, TR alignment, feature extraction (Friends + Movie10)

### Preprocessing Strategy

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

### Parcel Time Series Extraction Strategy

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

### HDP-HMM State Discovery Strategy

The brain state discovery pipeline uses **Sticky Hierarchical Dirichlet Process Hidden Markov Models (sHDP-HMM)** to identify latent brain states from fMRI time series. The primary analysis fits **one combined model per subject across all episodes** (`04_combined_hdphmm.py`), taking PCA-projected data from `03a_pca4combined_hmm.py` as input.

**Current production config (all 6 subjects, vt=0.95): `vt0.95_covdiag_nc50_g1`**

| Parameter | Value | Role |
|---|---|---|
| Parcellation | atlas-4S156Parcels | 156-parcel ROI-level (Schaefer 100 cortical + 56 subcortical composite) |
| PCA space | 67–77 PCs (vt=0.95) | HMM fits in PC space, NOT in parcel space |
| nc | 50 | Truncation capacity; K_active 42–47 per subject (mean 44.8, ~90% utilization). LOSO refits and split-half subsets yield smaller K (≈35–42) by ~3–5 states. |
| γ (gamma) | 1 | Low HDP concentration; prior prefers fewer states, so arriving at K≈45 is strong evidence those states are real |
| κ (kappa) | 10 | Sticky bias; encourages multi-TR state persistence (~hemodynamic timescale) |
| α (alpha) | 1 | Row-level transition concentration |
| ρ (rho) | 1 | Sticky bias scaling |
| covariance | **diag** (in PC space) | Each state = per-PC mean + diagonal variance. Back-projected to parcel space, state covariance is non-diagonal (PCA mixing), so states still carry within-state coupling structure when visualized in brain-region space. |

**Model selection uses Pareto analysis (LL vs K_active), NOT BIC or any LL-based metric:** all LL-based scores (raw LL, LL/dim, BIC, 6-metric battery) systematically favor higher complexity. The Pareto middle cluster across subjects is K≈35–45; nc=50 γ=1 is the simplest model in that cluster.

**Validation:**
- LOSO refits (6 folds per subject, each with its own PCA to avoid leakage) test whether discovered states generalize across narrative contexts.
- Split-half (interleaved episode halves) reliability: 04rb Spearman ρ = 0.60–0.82.

### Brain State Recurrence Analysis

Script `05a_recurrence_analysis.py` classifies brain states as **context-invariant** or **episode-specific**, answering the central research question: **Do brain states represent recurring structure across episodes, or are they shaped by episode-specific content?**

**Prerequisites:**
- `04_combined_hdphmm.py` (mode: select) completed; produces `decoded_states.pkl` and `final_results.json`

**Core Analysis Pipeline:**
1. **Fractional Occupancy (FO):** Per-state, per-episode occupancy fraction
2. **Recurrence Score:** Fraction of episodes where each state is active (FO > threshold)
3. **Season Specificity Index:** Range of per-season recurrence (0 = invariant, 1 = specific)
4. **Permutation Test:** Shuffles season labels to test whether specificity exceeds chance
5. **FDR Correction:** Benjamini-Hochberg correction for multiple comparisons across states

**State scoring (continuous, not categorical):** Recurrence is treated as a continuous score, NOT binned into categorical groupings (Recurring / Specific / Partial / Inactive). Downstream scripts (05d similarity, 06a dynamics, 07b physio, 08b content) all consume the continuous scores directly. Eligibility filtering (e.g., sub-HRF states) is handled via the `state_flags.csv` mechanism produced by 05e, not by the recurrence score.

**Key Outputs (saved to `{SCRATCH_DIR}/output/05a_recurrence_analysis/{parcellation}/{sub_id}/`):**
- `recurrence_scores.npy`: Per-state recurrence scores
- `specificity_index.npy`: Per-state season-specificity indices
- `fractional_occupancy.pkl`: Per-run, per-state FO
- `permutation_pvalues.json`: Uncorrected and FDR-corrected p-values
- `recurrence_summary.json`: Full results including state classifications

**Usage:**
```bash
# Standard recurrence analysis (after 04 select)
python script/05a_recurrence_analysis.py \
    --sub_id sub-01 --parcellation atlas-4S156Parcels

# With threshold sweep for robustness check
python script/05a_recurrence_analysis.py \
    --sub_id sub-01 --parcellation atlas-4S156Parcels --threshold_sweep
```
