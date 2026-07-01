# brain-states-friends

Quantifying how brain states recur across narrative contexts during longitudinal naturalistic viewing under fMRI.

## Overview

This project maps the repertoire of brain states that recur during longitudinal naturalistic viewing, using fMRI recorded while six participants watched the television series Friends across many episodes. A Gaussian hidden Markov model, fit per subject in a PCA-reduced parcel space, segments each viewing run into latent states. Its transition priors (sticky self-transitions plus a hierarchical-Dirichlet concentration prior, borrowed from the sticky HDP-HMM but applied under a fixed-capacity weak-limit truncation rather than unbounded nonparametric inference) let the number of occupied states emerge from the data. For each state we compute a recurrence score (the fraction of episodes in which it is active) using fractional occupancy and a season-label permutation test; the resulting scores form a continuous recurrence gradient rather than a binary invariant-versus-specific split. We then characterize each state's canonical-network composition and transition dynamics. Finally, we test the depth at which the states are decodable from a transformer model of the stimulus (representational depth), and whether the Friends state repertoire recurs in three held-out narratives: Movie10, Harry Potter, and Petit Prince.

## Data

This repository contains only analysis code; it bundles no neuroimaging data or
stimuli. The pipeline starts from fMRIPrep-preprocessed fMRI plus several external,
separately licensed datasets. Obtain each from its source, then point the
corresponding `.env` variable (see [Configuration](#configuration)) at your local copy.

| Dataset | `.env` variable | Used by | Source / access |
|---|---|---|---|
| CNeuroMod fMRI (Friends, Movie10, Petit Prince) | `DATA_DIR` | `00`–`06` | [cneuromod.ca](https://www.cneuromod.ca/). Requires a signed data-access agreement; distributed via DataLad. The pipeline expects fMRIPrep derivatives (script `00` post-processes them). |
| Algonauts 2025 stimuli | `ALGONAUTS_DIR` | `08c` | [algonautsproject.com/2025](https://algonautsproject.com/2025/index.html) (`$ALGONAUTS_DIR/stimuli`) |
| 4S parcellation atlases (Schaefer-100 cortical + subcortical composite) | `ATLAS_DIR` | `01`, `02` | XCP-D atlas bundle, pinned to [`xcp_d` v0.7.1](https://github.com/PennLINC/xcp_d/archive/0.7.1.tar.gz) (PennLINC; the 4S family originates from AtlasPack). One subdirectory per atlas, e.g. `atlas-4S156Parcels/atlas-4S156Parcels_space-fsLR_den-91k_dseg.dlabel.nii` plus the matching `_dseg.tsv` labels. |

> **Note:** Access to the CNeuroMod dataset is governed by its own terms; this code
> does not redistribute it.

## Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  00_postproc.py         Preprocess fMRIPrep outputs                          │
│                         - Confound regression, z-scoring                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  01_get_parcel_label.py   Extract parcel labels from atlas                   │
│  02_extract_parcel_ts.py  Extract parcel time series                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  03a_pca4combined_hmm.py  PCA + train/valid/test splits for combined HMM     │
│  03b_pca_loadings.py      PCA loadings analysis                              │
│  04_combined_hdphmm.py    Fit combined HMM per subject across all episodes   │
│                           - fit mode: grid search over configs               │
│                           - select mode: pick best config, decode all runs   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  05a_recurrence_analysis.py        Score state recurrence across episodes    │
│                                    (continuous gradient; FO + permutation)   │
│  05b_visualize_recurring_states.py Cortical + subcortical surface plots      │
│                                    (yabplot, headless PyVista rendering)     │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  06a_state_temp_dynamics.py   Dwell-time distributions, transition matrices  │
│  06b_transition_structure.py  Graph topology, FC-Mantel, MFPT landscape      │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  08c_transformer_features.py  Layer-wise transformer features (GPU)          │
│  08d_transformer_depth.py     Representational depth per layer               │
│  08e_transformer_cross_stim_aggregate.py  Cross-stimulus depth profile       │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Cross-stimulus transfer  Movie10 / Harry Potter / Petit Prince              │
│  {m10,hp,pp}_03  Project stimulus through the Friends PCA basis              │
│  {m10,hp,pp}_04  Score + decode with the Friends HMM                         │
│  {m10,hp,pp}_05  Cross-stimulus recurrence-transfer validation               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Script organization

Scripts are tagged by manuscript tier in **[script/MANIFEST.md](script/MANIFEST.md)**
(the source of truth for the main/supplementary boundary). Filenames are not
prefixed by tier because each numeric prefix doubles as the output-directory
name and the DataLad `--stage` key.

Some numeric families straddle tiers (e.g. `05e_a4` is main while `05e_a1`–`a3`
are supplementary), so the table below lists discrete scripts rather than
ranges; see [MANIFEST.md](script/MANIFEST.md) for the per-script detail.

| Tier | Scripts | What it covers |
|---|---|---|
| **Main** | `00`, `01`, `02`, `03a`, `04`, `05a`, `05b`, `05e_a4`, `06a`, `06b`, `08c`, `08d`, `08e` | Repertoire, taxonomy, transitions, transformer depth |
| **Cross-stimulus (main)** | `m10_*`, `hp_*`, `pp_*` (03/04/05) | Recurrence transfer to Movie10 / Harry Potter / Petit Prince |
| **Supplementary** | `03b`, `04r*`, `05c`, `05d`, `05e_a1`–`a3`, `05f`, `06c`, `06d` | Reliability, robustness, diagnostics |

## Quick Start

```bash
# One-time setup (uv installs a matching Python automatically)
cd brain-states-friends
uv sync

# Configure paths (see "Configuration" below)
cp .env.example .env
$EDITOR .env

# Run full pipeline for one subject
sbatch script/00_postproc.sh
sbatch script/01_get_parcel_label.sh
sbatch script/02_extract_parcel_ts.sh
sbatch --export=SUB_ID=sub-01 script/03a_pca4combined_hmm.sh
sbatch --export=SUB_ID=sub-01 script/03b_pca_loadings.sh
sbatch --export=SUB_ID=sub-01 script/04_combined_hdphmm.sh          # fit mode
sbatch --export=SUB_ID=sub-01,MODE=select script/04_combined_hdphmm.sh  # select mode
sbatch --export=SUB_ID=sub-01 script/05a_recurrence_analysis.sh
sbatch --export=SUB_ID=sub-01 script/05b_visualize_recurring_states.sh
sbatch --export=SUB_ID=sub-01 script/06a_state_temp_dynamics.sh
```

### Running without SLURM

The `script/NN_*.sh` files are thin SLURM submission wrappers: they resolve a few
environment variables (`SUB_ID`, `PARCELLATION`, `MODE`, …) and then call the
matching `script/NN_*.py` via `uv run`. On a machine without a scheduler, run the
Python directly with the same arguments. For example:

```bash
uv run python script/04_combined_hdphmm.py --sub_id sub-01 \
    --parcellation atlas-4S156Parcels --mode fit
uv run python script/04_combined_hdphmm.py --sub_id sub-01 \
    --parcellation atlas-4S156Parcels --mode select
```

Pass `--help` to any `script/NN_*.py` for its full argument list. The `.sh`
wrappers are the canonical reference for which variables map to which flags.

## Reproducing the figures

Each manuscript figure is built by a dedicated `script/fig_*.py`. The scripts
read the analysis outputs produced by the pipeline stages listed below, so run
the relevant stages (for all six subjects, plus the cross-stimulus stages where
noted) before invoking a figure script.

| Figure | Script | Shows | Reads from |
|---|---|---|---|
| **1** | `fig_F1_recurrence_gradient.py` | Continuous recurrence gradient across states (R1) | `05a`, `06a` |
| **2** | `fig_F2_recurrence_sources.py` (all panels); `fig_F2_network_participation.py` (batch Panel C) | Recurrence-screening categories + network participation (R2) | `04`, `05a`, `05e` |
| **3** | `fig_F3_transition_structure.py` | Transition graph topology + FC–transition coupling (R3) | `05a`, `05f`, `06a`, `06b` |
| **4** | `fig_F4_within_friends.py` (lead) + `fig_F4_per_film_video.py` (Movie10 per-film panel) | Within-Friends representational depth + Movie10 per-film depth (R4b) | `08d`, `08e` |
| **5** | `fig_F5_cross_stimulus_transfer.py` | Cross-stimulus recurrence transfer to Movie10/HP/PP (R5) | `04` decode, `05a`, `m10_`/`hp_`/`pp_05` |
| **S1** | `fig_S01_recurring_state_surface_maps.py` | Cortical + subcortical surface maps of the top recurring states, per subject | `05a`, `05b` |
| **S2** | `fig_S02_pca_loadings.py` | PCA loadings diagnostics (one representative subject) | `03b` |
| **S3** | `fig_S03_video_peak_depth.py` | Network-stratified video (DINOv2) decoding depth | `08d` |
| **S6** | `fig_S06_cross_stimulus_validity.py` | Cross-stimulus validity & repertoire presence | `03` proj, `04` decode, `05a`, `m10_`/`hp_`/`pp_05` |
| **S7** | `fig_S07_individual_differences.py` | Per-subject individual differences | `05a`, `05e`, `06b`, `08d` (cross-subject) |
| **S9** | `fig_S09_network_participation_categories.py` | Canonical-network participation across all recurrence-screening categories (descriptive provenance) | `04`, `05e` |

Supplementary figures S4 and S5 have no `fig_*.py`: they are emitted directly by
the `08d`/`08e` analysis scripts (via `script/08d_plots.py` / `script/08e_plots.py`).
Figures S8 and S10 live on the orphan [`supplements`](https://github.com/yibeichan/brain-states-friends/tree/supplements)
branch (ICA alternative decomposition). See [docs/supplementary/](docs/supplementary/)
for the complete S1–S10 index with per-figure provenance and branch.

Shared plotting helpers live in `script/08d_plots.py`, `script/08e_plots.py`,
and `script/utils/{recurrence,temporal}_plots.py`. Network-participation metrics
(Figure 2C and Figure S9) share one implementation in
`script/utils/network_participation.py`. The figure scripts import these
helpers; do not run them directly.

## Documentation

- **[script/MANIFEST.md](script/MANIFEST.md)** - Per-script classification (main / supplementary / cross-stimulus)
- **[docs/supplementary/](docs/supplementary/)** - Supplementary material accompanying the manuscript
- **[docs/findings/](docs/findings/)** - Per-script findings: one Results-only doc per analysis script, with every number verified against the pipeline outputs
- **[AGENTS.md](AGENTS.md)** - Instructions for AI coding agents (e.g. Claude Code) working in this repo: conventions, methodology notes, and analysis-scope rules.
- **Alternative analysis (ICA)** - An independent-component-analysis decomposition of the same data, provided as a convergence check on the HMM state repertoire, is maintained as a self-contained supplement on the [`supplements`](https://github.com/yibeichan/brain-states-friends/tree/supplements) branch (alongside other supplementary analyses).

## Citation

If you use this code, please cite the accompanying paper:

> Chen Y, Ghavami M, St-Laurent M, Bellec L, Ghosh SS. Brain states recur across
> diverse narrative contexts during longitudinal viewing. bioRxiv 2026.05.31.729141.
> https://doi.org/10.64898/2026.05.31.729141

[`CITATION.cff`](CITATION.cff) carries machine-readable citation metadata; GitHub's
"Cite this repository" button reads it.

## Key Configuration

Production config: `vt0.95_covdiag_nc50_g1`.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Parcellation | atlas-4S156Parcels | 156 parcels (100 cortical + 56 subcortical composite) |
| PCA space | vt=0.95 (~67–77 PCs) | HMM fits in PC space, not parcel space |
| Truncation `nc` | 50 | Capacity; K_active ≈ 37–42 per subject |
| Stickiness `kappa` | 10 | Encourages multi-TR (hemodynamic-scale) persistence |
| HDP concentration `gamma` | 1 | Low; prior prefers fewer states |
| Covariance | diagonal (PC space) | Back-projected to parcel space it is non-diagonal (PCA mixing), so states still carry within-state coupling structure |

Model selection uses Pareto analysis of validation log-likelihood vs active-state count (not BIC).

## Environment

```bash
# Sync Python dependencies from pyproject.toml
# (uv provisions a matching Python interpreter automatically; no conda/micromamba needed)
cd ~/brain-states-friends
uv sync                     # core dependencies

# Optional extras (add only what you need):
uv sync --extra gpu         # JAX/dynamax for GPU model fitting (script 04, GPU nodes)
uv sync --extra torch       # transformer stack (script 08c)
uv sync --extra datalad     # DataLad for output archival (script/utils/datalad_save.sh)
```

DataLad additionally requires the **git-annex** binary at runtime. git-annex is a
Haskell program, not a PyPI package, so `uv` cannot install it. The `datalad`
extra bundles `datalad-installer`, which fetches a standalone git-annex into
user space (no root) — run it once after `uv sync --extra datalad`:

```bash
uv run --extra datalad datalad-installer \
  -E ~/.local/share/git-annex-env.sh \
  git-annex -m datalad/git-annex:release --install-dir ~/.local
source ~/.local/share/git-annex-env.sh   # or add its PATH line to ~/.bashrc
```

Run `uv sync` from the project root (the directory containing `pyproject.toml`).
Install [`uv`](https://docs.astral.sh/uv/) first if it is not already on your
`PATH` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

Surface visualizations (script `05b`) depend on [yabplot](https://github.com/yibeichan/yabplot),
which is declared as a git dependency in `pyproject.toml` and installed
automatically by `uv sync`:

```toml
[tool.uv.sources]
yabplot = { git = "https://github.com/yibeichan/yabplot.git" }
```

## Configuration

All machine-specific paths are read from a local `.env` file (gitignored).
Copy the template and fill in the values for your environment:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `BASE_DIR` | Project root (this repository). |
| `SCRATCH_DIR` | Scratch root for outputs. Scripts write under `$SCRATCH_DIR/output/`. |
| `DATA_DIR` | Root of the `all_about_cneuromod` dataset tree (Friends/Movie10/Petit Prince derive from it). |
| `ATLAS_DIR` | Directory of dseg atlases, one subdirectory per atlas. |
| `ALGONAUTS_DIR` | Root of the Algonauts 2025 dataset (script `08c` uses `$ALGONAUTS_DIR/stimuli`). |

Scripts fail fast with a clear message if a required variable is unset. See
`.env.example` for the full annotated list.

## Output Management

Analysis outputs are versioned with **DataLad** and synced to persistent storage via a RIA store.

```bash
# Save outputs after a pipeline run
bash script/utils/datalad_save.sh --stage 04 --subject sub-01

# Or chain as SLURM dependency
JOB=$(sbatch --parsable --export=SUB_ID=sub-01,MODE=select script/04_combined_hdphmm.sh)
sbatch --dependency=afterok:$JOB --export=STAGE=04,SUBJECT_ID=sub-01 script/utils/datalad_save.sh

# Recover after scratch purge
cd "$SCRATCH_DIR"
datalad clone "ria+file:///path/to/ria-store/brain-states-friends-outputs" output
cd output && datalad get 04_combined_hdphmm/  # lazy: get only what you need
```

RIA store and clone locations are site-specific; configure them in `output_dataset.yml`.
