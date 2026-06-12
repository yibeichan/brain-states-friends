# brain-states-friends: ICA alternative-analysis supplement

> **This is the `sm/ica-states` branch.** It presents a supplementary
> *alternative* state-discovery analysis. For the primary analysis (the sticky
> HDP-HMM brain states), the full pipeline, and the environment/data
> documentation, see the
> [`main` branch](https://github.com/yibeichan/brain-states-friends/tree/main).

## What this branch is

The main analysis discovers brain states with a sticky HDP-HMM. This branch
addresses a convergent-validity question: if we had used a different
unsupervised method instead of the HMM, would we recover comparable states? It
fits spatial **Independent Component Analysis (ICA)** on the same PCA subspace
and matches the resulting components against the HMM state maps. In other words,
this is the path we would have taken had we not gone the HMM route, kept here as
a check that the discovered states are not an artifact of one particular model.

The pipeline reuses the main upstream stages (preprocessing through HMM fitting)
and adds only the ICA scripts. Everything not needed for the ICA comparison
(transition structure, transformer depth, physiology, content features, and the
Movie10 / Harry Potter / Petit Prince cross-stimulus analyses) is omitted from
this branch; see `main` for those.

## What the supplement does

Per subject, on the vt=0.95 PC subspace:

1. **Spatial ICA** over a sweep of component counts K in {15, 25, 35} plus
   K_active (the subject's HMM active-state count).
2. **Match each HMM state to ICA components on three tiers:**
   - Tier 1 (spatial): correlation between the back-projected HMM state maps and
     the ICA component maps.
   - Tier 2 (temporal): correspondence between HMM state posteriors and ICA
     component time courses.
   - Tier 3 (label agreement): winner-take-all label concordance per TR.
3. Run the comparison for two HMM state sets: the **content-eligible** states
   (from the 05e_a4 `state_flags.csv` taxonomy) and **all active** states.

Significance uses a within-run circular-shift null with FDR correction across
eligible pairs.

**Outputs** (under `$SCRATCH_DIR/output/`):
- `sm_ica_states/{parcellation}/{sub}/ica_match_summary.json`: per-subject match
  results across the K sweep.
- `sm_ica_states/{parcellation}/category_correspondence_table.{csv,md}`:
  aggregated per-(subject, taxonomy category) correspondence.
- A two-panel convergence figure: (A) FDR-surviving fraction of eligible pairs
  across subject x ICA K, (B) per-state matched spatial |r| at K_active.

## Running the pipeline

One-time setup (identical to `main`; see the
[`main` README](https://github.com/yibeichan/brain-states-friends/tree/main) for
data-access detail):

```bash
micromamba env create -f environment.yml
micromamba activate friends-states
uv sync
cp .env.example .env   # set BASE_DIR, SCRATCH_DIR, DATA_DIR, ATLAS_DIR
```

**Data.** This supplement uses only the CNeuroMod *Friends* fMRI (`DATA_DIR`)
and the 4S parcellation atlas (`ATLAS_DIR`). No stimuli, narrative annotations,
or other datasets are required. Access to CNeuroMod is governed by its own data
agreement; this code does not redistribute it.

**Pipeline** (per subject; the array jobs cover all six subjects):

```bash
# Upstream: preprocessing -> parcellation -> PCA -> HMM
sbatch script/00_postproc.sh
sbatch script/01_get_parcel_label.sh
sbatch script/02_extract_parcel_ts.sh
sbatch --export=SUB_ID=sub-01 script/03a_pca4combined_hmm.sh
sbatch --export=SUB_ID=sub-01 script/04_combined_hdphmm.sh              # fit mode
sbatch --export=SUB_ID=sub-01,MODE=select script/04_combined_hdphmm.sh  # select + decode

# Recurrence + state taxonomy (provides the content-eligible set and category labels)
sbatch --export=SUB_ID=sub-01 script/05a_recurrence_analysis.sh
sbatch --export=SUB_ID=sub-01 script/05e_temporal_trend_a4.sh

# ICA supplement (array 0-5 covers all subjects)
sbatch script/sm_ica_states.sh

# After all six subjects finish: aggregate table + figure
uv run python script/sm_ica_category_table.py --parcellation atlas-4S156Parcels --vt 0.95
uv run marimo run script/fig_sm_ica_matching.py
```

`script/05a_sub_hrf_diagnostic.py` is an optional report on short-dwell
("sub-HRF") states; it is not on the critical path.

### Running without SLURM

The `script/*.sh` files are thin SLURM wrappers that resolve a few environment
variables (`SUB_ID`, `PARCELLATION`, `MODE`, `VT`) and call the matching
`script/*.py` via `uv run`. On a machine without a scheduler, run the Python
directly with the same arguments:

```bash
uv run python script/04_combined_hdphmm.py --sub_id sub-01 \
    --parcellation atlas-4S156Parcels --mode fit
uv run python script/sm_ica_states.py --sub_id sub-01 \
    --parcellation atlas-4S156Parcels --vt 0.95
```

Pass `--help` to any `script/*.py` for its full argument list.

## Relationship to `main`

This branch shares the upstream pipeline (`00`-`04`) and the recurrence /
eligibility stages (`05a`, `05e_a4`) with `main`, and adds only the ICA scripts
(`script/sm_ica_*`, `script/fig_sm_ica_matching.py`) plus the
`compute_state_posteriors` helper in `script/utils/hmm_io.py`. For the primary
HMM analysis, the model configuration, and the full environment/data
documentation, see the
[`main` branch](https://github.com/yibeichan/brain-states-friends/tree/main).

## Citation

If you use this code, please cite the accompanying paper:

> Chen Y, Ghavami M, St-Laurent M, Bellec L, Ghosh SS. Brain states recur across
> diverse narrative contexts during longitudinal viewing. bioRxiv
> 2026.05.31.729141. https://doi.org/10.64898/2026.05.31.729141

[`CITATION.cff`](CITATION.cff) carries machine-readable citation metadata;
GitHub's "Cite this repository" button reads it.

## Configuration

Machine-specific paths are read from a local `.env` file (gitignored). Copy the
template and fill in the values for your environment:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `BASE_DIR` | Project root (this repository). |
| `SCRATCH_DIR` | Scratch root for outputs. Scripts write under `$SCRATCH_DIR/output/`. |
| `DATA_DIR` | Root of the CNeuroMod dataset tree (Friends fMRI derivatives). |
| `ATLAS_DIR` | Directory of dseg atlases, one subdirectory per atlas. |

Scripts fail fast with a clear message if a required variable is unset.
