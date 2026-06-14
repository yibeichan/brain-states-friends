# Findings: m10_03 Project Movie PCA

_Script: `script/m10_03_project_movie_pca.py`. Tier: CROSS-STIM (R5, Fig 5)._

_Projects Movie10 parcel time series through the Friends-trained PCA model and computes a transfer diagnostic (R² per subject, per movie type, per network); per-subject, n=6._

## Method (as run)
- Parcellation: atlas-4S156Parcels (156 parcels; column 0 background dropped before projection)
- Variance threshold: vt=0.95; PCA loaded from `output/04_combined_hdphmm/atlas-4S156Parcels/{sub_id}/final/vt0.95/pca_model.pkl`
- n_pcs per subject taken from `final_results.json` `data_info.n_pcs`
- Input time series: `output/02_parcel_ts_avg/atlas-4S156Parcels/{sub_id}/*_parcel_avg.npy` (movie runs only; Friends episodes excluded)
- Movie runs: 61 per subject (Bourne x10, Wolf x17, Figures x24, Life x10)
- R² computed relative to the Friends training mean (SST = sum((X - pca.mean_)^2)); mean-corrected R² also computed to isolate covariance mismatch from mean-shift
- Network assignment: Yeo-7 cortical labels from atlas TSV; subcortical via `assign_network()` (13 groups: BG, Midbrain-DA, Midbrain-Diencephalic, Thalamus, Hipp/Amyg, Cerebellum + Yeo-7)
- flag_low_variance threshold: R² < 0.70; flag_mean_shift threshold: >10% of parcels with |shift| > 1 SD
- No group statistics; all results are per-subject

## Results

### Overall PCA Transfer

| Subject | n_pcs | Friends R2 | Movie R2 | Movie R2 (mc) | Transfer gap | Movie TRs |
|---------|-------|------------|----------|---------------|--------------|-----------|
| sub-01  | 75    | 0.9505     | 0.9361   | 0.9361        | 0.0144       | 24,836    |
| sub-02  | 72    | 0.9504     | 0.9347   | 0.9347        | 0.0157       | 24,891    |
| sub-03  | 72    | 0.9504     | 0.9467   | 0.9467        | 0.0037       | 24,812    |
| sub-04  | 77    | 0.9500     | 0.9410   | 0.9410        | 0.0091       | 24,832    |
| sub-05  | 67    | 0.9509     | 0.9337   | 0.9337        | 0.0172       | 24,888    |
| sub-06  | 74    | 0.9504     | 0.9396   | 0.9396        | 0.0108       | 24,872    |

flag_low_variance = false for all subjects. Mean-corrected R² equals standard R² for all subjects (parcel mean shift negligible; see below).

### Per-Movie-Type R2

| Subject | Bourne (n=10) | Wolf (n=17) | Figures (n=24) | Life (n=10) |
|---------|---------------|-------------|----------------|-------------|
| sub-01  | 0.9365        | 0.9315      | 0.9380         | 0.9382      |
| sub-02  | 0.9418        | 0.9242      | 0.9360         | 0.9387      |
| sub-03  | 0.9509        | 0.9456      | 0.9468         | 0.9440      |
| sub-04  | 0.9475        | 0.9313      | 0.9369         | 0.9533      |
| sub-05  | 0.9344        | 0.9301      | 0.9359         | 0.9334      |
| sub-06  | 0.9474        | 0.9390      | 0.9382         | 0.9364      |

Wolf of Wall Street has the lowest R² in 4 of 6 subjects (sub-01, sub-02, sub-04, sub-05); in sub-03 and sub-06 the minimum is Life of Pi. All values exceed 0.92.

### Per-Network R2 (Overall, across all movie TRs)

| Network | n_parcels | sub-01 | sub-02 | sub-03 | sub-04 | sub-05 | sub-06 |
|---------|-----------|--------|--------|--------|--------|--------|--------|
| Vis             | 17 | 0.963  | 0.958  | 0.972  | 0.968  | 0.962  | 0.966  |
| SomMot          | 14 | 0.952  | 0.938  | 0.962  | 0.970  | 0.948  | 0.953  |
| DorsAttn        | 15 | 0.951  | 0.945  | 0.964  | 0.965  | 0.951  | 0.956  |
| SalVentAttn     | 12 | 0.927  | 0.925  | 0.943  | 0.937  | 0.929  | 0.946  |
| Limbic          |  5 | 0.667  | 0.760  | 0.798  | 0.623  | 0.775  | 0.788  |
| Cont            | 13 | 0.963  | 0.962  | 0.971  | 0.966  | 0.962  | 0.964  |
| Default         | 24 | 0.953  | 0.947  | 0.962  | 0.950  | 0.946  | 0.955  |
| BG              | 16 | 0.905  | 0.941  | 0.863  | 0.860  | 0.864  | 0.876  |
| Midbrain-DA     |  2 | 0.995  | 0.996  | 0.974  | 0.981  | 0.971  | 0.979  |
| Midbrain-Diencephalic | 8 | 0.958 | 0.962 | 0.896 | 0.930 | 0.911 | 0.929 |
| Thalamus        | 14 | 0.368  | 0.314  | 0.405  | 0.230  | 0.436  | 0.344  |
| Hipp/Amyg       |  6 | 0.859  | 0.889  | 0.860  | 0.850  | 0.861  | 0.835  |
| Cerebellum      | 10 | 0.775  | 0.765  | 0.823  | 0.720  | 0.766  | 0.851  |

Cortical Yeo-7 networks (excluding Limbic) transfer well (R² > 0.92 in all subjects). Thalamus has the lowest R² (0.23-0.44). Limbic is the weakest cortical network (0.62-0.80). Midbrain-DA (n=2 parcels, SNc/PBP/VTA) is the highest-R² subcortical bin.

### Parcel Mean Shift

| Subject | Mean abs shift | Max abs shift | Parcels > 1 SD | flag_mean_shift |
|---------|---------------|---------------|----------------|-----------------|
| sub-01  | 8.6e-12       | 1.1e-10       | 0              | false           |
| sub-02  | 2.9e-11       | 2.2e-10       | 0              | false           |
| sub-03  | 3.4e-11       | 8.7e-10       | 0              | false           |
| sub-04  | 4.5e-11       | 3.8e-10       | 0              | false           |
| sub-05  | 2.3e-11       | 1.4e-10       | 0              | false           |
| sub-06  | 2.5e-11       | 1.7e-10       | 0              | false           |

Mean shifts are effectively zero (order 10^-11 to 10^-10) for all subjects; no parcels exceed the 1 SD threshold. This confirms that z-scoring in preprocessing normalizes parcel means across stimuli.

## Outputs
- `output/m10_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/{run_id}.npy` - PCA-projected movie run, shape (n_trs, n_pcs); 61 files per subject
- `output/m10_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/movie_run_ids.json` - run IDs grouped by movie type
- `output/m10_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/pca_transfer_diagnostic.json` - full diagnostic (overall + per-type + per-network R², transfer gap, mean-corrected R², parcel mean shift, flags)
- `output/m10_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/pca_transfer_diagnostic.png` - overall diagnostic figure
- `output/m10_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/pca_transfer_diagnostic_{bourne,wolf,figures,life}.png` - per-movie-type diagnostic figures
