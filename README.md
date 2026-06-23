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

### ICA convergent validity (`alt`)

Does the HMM state repertoire re-emerge under a decomposition with orthogonal
assumptions (spatial ICA)? Findings: [`docs/findings/sm_alt_ica_states.md`](docs/findings/sm_alt_ica_states.md).

- `script/sm_alt_ica_states.py` (+ `.sh` runner) — fit/match ICA vs HMM states
- `script/sm_alt_ica_category_table.py` — per-category breakdown
- `script/fig_sm_alt_ica_matching.py` — companion figure

**Runs standalone?** No — reads main-pipeline outputs (stages `03a`/`04`/`05e`);
the code is complete but the inputs are produced on `main`. The bundled `utils/`
snapshot (`ica_states`, `hmm_io`, `hdphmm`, `stats`, `transformer_analysis`,
`common`, `state_blocks`, `state_flags_io`, `plot_style`) is duplicated from
`main` so imports resolve from this branch alone.

### Prior-predictive occupied-state count (`sim`)

Under the production `gamma=1` sticky-HDP prior, how many states does the model
expect *a priori*? Findings: [`docs/findings/sm_sim_prior_state_count.md`](docs/findings/sm_sim_prior_state_count.md).

- `script/sm_sim_prior_state_count.py` — the simulation

**Runs standalone?** Yes — pure simulation, `numpy` only.

## Running

This branch is a self-contained `uv` project. Set up once with `uv sync`, then
run via `uv run`. Paths are read from a local `.env` (`SCRATCH_DIR`, `BASE_DIR`, …).

```bash
uv sync

# sim — standalone
uv run python script/sm_sim_prior_state_count.py
uv run pytest script/tests/test_sim_prior_state_count.py

# alt / ICA — needs main-pipeline outputs (stages 03a/04/05a/05e/06b)
uv run pytest script/tests/test_alt_ica_states.py script/tests/test_alt_ica_diagnostics.py
uv run python script/sm_alt_ica_diagnostics.py            # repertoire/convergence diagnostics
sbatch script/sm_alt_ica_states.sh                        # or: uv run python script/sm_alt_ica_states.py --help
```

## Citation

See [`CITATION.cff`](https://github.com/yibeichan/brain-states-friends/blob/main/CITATION.cff)
on `main`:

> Chen Y, Ghavami M, St-Laurent M, Bellec L, Ghosh SS. Brain states recur across
> diverse narrative contexts during longitudinal viewing. bioRxiv 2026.05.31.729141.
> https://doi.org/10.64898/2026.05.31.729141
