# brain-states-friends — supplements

Self-contained supplementary analyses for the manuscript *"Brain states recur
across diverse narrative contexts during longitudinal viewing"* (Chen et al.).

This is a permanent **orphan branch** with history disjoint from `main`. It holds
**only** supplementary analyses and the dependencies they bundle — not the core
pipeline. The main analysis lives on the
[`main`](https://github.com/yibeichan/brain-states-friends/tree/main) branch; this
branch is never merged into it.

## Why a separate branch

Each supplement here is a stand-alone check on the main results (an alternative
decomposition, a simulation, a robustness analysis). Keeping them off `main`
leaves the core pipeline uncluttered while still publishing the supporting code.
Any `utils/`, config, or helper a supplement imports is **bundled as a frozen
snapshot** on this branch (deliberately duplicated from `main`, not auto-synced),
so each supplement is reproducible from this branch's own tree.

## Layout

Flat, no subfolders; a category token in each filename marks the kind of
supplement (`alt` = alternative analysis, `sim` = simulation, `rel` =
reliability/robustness):

- analysis scripts — `script/sm_<cat>_<topic>.py` (+ optional `.sh` SLURM runner)
- figure scripts — `script/fig_sm_<cat>_<topic>.py`
- tests — `script/tests/test_<cat>_<topic>.py`
- findings docs — `docs/findings/sm_<cat>_<topic>.md`

## Contents

| Supplement | Scripts | Findings doc | Runs standalone? |
|---|---|---|---|
| **ICA convergent validity** (`alt`) — does the HMM state repertoire re-emerge under a decomposition with orthogonal assumptions (spatial ICA)? | `script/sm_alt_ica_states.py` (+`.sh`), `script/sm_alt_ica_category_table.py`, `script/fig_sm_alt_ica_matching.py` | [`docs/findings/sm_alt_ica_states.md`](docs/findings/sm_alt_ica_states.md) | No — reads main-pipeline outputs (stages `03a`/`04`/`05e`); the code is complete but the inputs are produced on `main`. |
| **Prior-predictive occupied-state count** (`sim`) — under the production `gamma=1` sticky-HDP prior, how many states does the model expect *a priori*? | `script/sm_sim_prior_state_count.py` | [`docs/findings/sm_sim_prior_state_count.md`](docs/findings/sm_sim_prior_state_count.md) | Yes — pure simulation, `numpy` only. |

The ICA bundle's frozen `utils/` snapshot (`ica_states`, `hmm_io`, `hdphmm`,
`stats`, `transformer_analysis`, `common`, `state_blocks`, `state_flags_io`,
`plot_style`) is duplicated from `main` so the supplement resolves its imports
from this branch alone.

## Running

The scripts share the `main` branch's environment (`friends-states`); see the
[`main` README](https://github.com/yibeichan/brain-states-friends/tree/main#environment)
for setup. Paths are read from a local `.env` (`SCRATCH_DIR`, `BASE_DIR`, …);
nothing is hardcoded. The `sim` supplement runs as-is; the `alt`/ICA supplement
expects the relevant `output/` stages from a `main` pipeline run.

```bash
# sim — standalone
python script/sm_sim_prior_state_count.py
python script/tests/test_sim_prior_state_count.py

# alt / ICA — needs main-pipeline outputs (stages 03a/04/05e)
python -m pytest script/tests/test_alt_ica_states.py   # unit tests (no data needed)
sbatch script/sm_alt_ica_states.sh                 # or: python script/sm_alt_ica_states.py --help
```

## Citation

See [`CITATION.cff`](https://github.com/yibeichan/brain-states-friends/blob/main/CITATION.cff)
on `main`:

> Chen Y, Ghavami M, St-Laurent M, Bellec L, Ghosh SS. Brain states recur across
> diverse narrative contexts during longitudinal viewing. bioRxiv 2026.05.31.729141.
> https://doi.org/10.64898/2026.05.31.729141
