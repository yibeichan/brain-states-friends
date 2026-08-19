# Findings: rest_03 Project Resting State Through Friends PCA

_Script: `script/rest_03_project_rest_pca.py`. Tier: CROSS-STIM (R5 extension, Fig S8)._

_Projects resting-state parcel time-series through each participant's Friends-trained PCA basis, without re-fitting, and reports how much resting variance that basis recovers; per-subject, n=6._

## Method (as run)

- Parcellation: atlas-4S156Parcels; vt=0.95
- Inputs: resting-state parcel time-series from `rest_02_extract_parcel_ts`; Friends-trained PCA from `03a_pca4combined_hmm`
- The Friends PCA basis and the Friends training-set parcel mean are both reused unchanged. No component is re-estimated on resting data, so Friends state indices remain directly comparable to resting state indices.
- n=6 subjects. Resting state is the only out-of-Friends condition with no missing participant; Harry Potter and Petit Prince lack sub-04.
- Resting runs: 5, 5, 5, 5, 4, 6 for sub-01 through sub-06 (30 runs total), 600 TRs per run
- Reconstruction R² is computed over the retained components (`rest_r2_n_pcs`), centred by the Friends training-set parcel mean. A mean-corrected variant is also written; for this dataset the two are identical to four decimal places, so the parcel mean shift between Friends and rest does not affect the diagnostic.
- `transfer_gap` = Friends R² − rest R². Positive means the basis recovers less resting than Friends variance.

## Results

### PCA transfer diagnostic

| Subject | n_pcs | rest R² | Friends R² | transfer gap | rest TRs | rest runs |
|---------|-------|---------|------------|--------------|----------|-----------|
| sub-01 | 75 | 0.9512 | 0.9505 | −0.0007 | 3000 | 5 |
| sub-02 | 72 | 0.9617 | 0.9504 | −0.0113 | 3000 | 5 |
| sub-03 | 72 | 0.9475 | 0.9504 | +0.0029 | 3000 | 5 |
| sub-04 | 77 | 0.9579 | 0.9500 | −0.0078 | 3000 | 5 |
| sub-05 | 67 | 0.9453 | 0.9509 | +0.0057 | 2400 | 4 |
| sub-06 | 74 | 0.9432 | 0.9504 | +0.0072 | 3600 | 6 |

The Friends basis recovers resting variance about as well as it recovers Friends variance. Transfer gaps span −0.0113 to +0.0072, and in three of six participants the gap is negative, meaning the basis reconstructs rest slightly better than the training stimulus. No participant triggered `flag_low_variance`.

This is the load-bearing negative result for the resting-state comparison: whatever fails downstream (see `rest_04`, `rest_05`) cannot be attributed to the spatial subspace failing to cover resting data.

### R² by network (sub-01, lowest five)

| Network | R² | n_parcels |
|---------|-----|-----------|
| Thalamus | 0.437 | 14 |
| Limbic | 0.682 | 5 |
| Cerebellum | 0.734 | 10 |
| Hipp/Amyg | 0.826 | 6 |
| BG | 0.914 | 16 |

Cortical networks are recovered at 0.95–0.97. The subcortical shortfall matches the within-Friends pattern reported in `03b_pca_loadings` and the cross-stimulus pattern in `m10_03` / `hp_03` / `pp_03`, so it is a property of the parcellation and the variance threshold rather than something specific to rest. Resting-state conclusions therefore rest mainly on cortical alignment.

## Caveats

- `rest_run_ids.json` stores a mapping rather than a flat list, so a naive `len()` on it returns 0. Run counts above were taken from `n_rest_trs` divided by 600 and cross-checked against `rest_04`'s `rest_total_runs`.
- R² measures variance recovery only. It says nothing about whether the fitted temporal model describes resting dynamics; that is `rest_04`.

## Outputs

- `output/rest_03_projected/atlas-4S156Parcels/<sub>/vt0.95/pca_transfer_diagnostic.json` — the table above plus `r2_by_network` and `r2_by_type`
- `output/rest_03_projected/atlas-4S156Parcels/<sub>/vt0.95/pca_transfer_diagnostic.png`
- `output/rest_03_projected/atlas-4S156Parcels/<sub>/vt0.95/<run>.npy` — projected component time-series per run
- `output/rest_03_projected/atlas-4S156Parcels/<sub>/vt0.95/rest_run_ids.json`
