# Findings: hp_03 Project Harry Potter PCA

_Script: `script/hp_03_project_hp_pca.py`. Tier: CROSS-STIM (R5, Fig 5)._

_Projects Harry Potter reading-task parcel time series through the Friends-trained PCA and computes a variance-explained transfer diagnostic; n=5 subjects (sub-04 has no HP data)._

## Method (as run)

- Parcellation: atlas-4S156Parcels (156 parcels; background column 0 dropped before projection)
- Variance threshold: vt=0.95
- PCA model source: `output/04_combined_hdphmm/atlas-4S156Parcels/{sub_id}/final/vt0.95/pca_model.pkl` (Friends training data)
- HP stimulus: unimodal word-by-word reading (RSVP at 2 Hz); 7 runs per subject (run-1 through run-7)
- Transfer R² computed relative to Friends training mean; mean-corrected variant also computed
- Per-network R² uses 13-group partition: Yeo-7 cortical + BG, Midbrain-DA (n=2 parcels), Midbrain-Diencephalic, Thalamus, Hipp/Amyg, Cerebellum
- Sub-04 excluded (no HP data available); all other subjects (sub-01, sub-02, sub-03, sub-05, sub-06) complete

## Results

### Overall PCA Transfer

| Subject | n_pcs | Friends R2 | HP R2 | HP R2 (mean-corrected) | Transfer gap | HP TRs |
|---------|-------|-----------|-------|----------------------|-------------|--------|
| sub-01  | 75    | 0.9505    | 0.9444 | 0.9444              | 0.0061      | 3,363  |
| sub-02  | 72    | 0.9504    | 0.9410 | 0.9410              | 0.0093      | 3,363  |
| sub-03  | 72    | 0.9504    | 0.9407 | 0.9407              | 0.0097      | 3,363  |
| sub-05  | 67    | 0.9509    | 0.9484 | 0.9484              | 0.0025      | 3,363  |
| sub-06  | 74    | 0.9504    | 0.9446 | 0.9446              | 0.0058      | 3,363  |

Mean-corrected R2 is identical to standard R2 for all subjects (parcel mean shift ~10^-11; preprocessing z-scoring eliminates baseline differences between stimuli). No `flag_low_variance` triggered (threshold: R2 < 0.70).

### Per-Network R2 (Cortical)

Parcel counts: Vis=17, SomMot=14, DorsAttn=15, SalVentAttn=12, Limbic=5, Cont=13, Default=24.

| Network | sub-01 | sub-02 | sub-03 | sub-05 | sub-06 | Range |
|---------|--------|--------|--------|--------|--------|-------|
| Vis | 0.956 | 0.957 | 0.962 | 0.966 | 0.963 | 0.956-0.966 |
| SomMot | 0.966 | 0.952 | 0.961 | 0.964 | 0.965 | 0.952-0.966 |
| DorsAttn | 0.952 | 0.943 | 0.954 | 0.958 | 0.957 | 0.943-0.958 |
| SalVentAttn | 0.938 | 0.934 | 0.936 | 0.951 | 0.950 | 0.934-0.951 |
| Limbic | 0.661 | 0.797 | 0.801 | 0.861 | 0.820 | 0.661-0.861 |
| Cont | 0.969 | 0.968 | 0.971 | 0.973 | 0.968 | 0.968-0.973 |
| Default | 0.960 | 0.956 | 0.957 | 0.958 | 0.957 | 0.956-0.960 |

### Per-Network R2 (Subcortical)

Parcel counts: BG=16, Midbrain-DA=2, Midbrain-Diencephalic=8, Thalamus=14, Hipp/Amyg=6, Cerebellum=10.

| Network | sub-01 | sub-02 | sub-03 | sub-05 | sub-06 | Range |
|---------|--------|--------|--------|--------|--------|-------|
| BG | 0.920 | 0.952 | 0.866 | 0.883 | 0.883 | 0.866-0.952 |
| Midbrain-DA* | 0.995 | 0.996 | 0.973 | 0.975 | 0.981 | 0.973-0.996 |
| Midbrain-Diencephalic | 0.963 | 0.959 | 0.906 | 0.923 | 0.936 | 0.906-0.963 |
| Thalamus | 0.424 | 0.342 | 0.424 | 0.549 | 0.434 | 0.342-0.549 |
| Hipp/Amyg | 0.871 | 0.879 | 0.866 | 0.889 | 0.857 | 0.857-0.889 |
| Cerebellum | 0.818 | 0.784 | 0.832 | 0.829 | 0.893 | 0.784-0.893 |

*Midbrain-DA n_parcels=2 (SNc_PBP_VTA bilateral pair); bin-level R2 is a single-structure estimate.

### Parcel Mean Shift

All subjects: mean absolute shift ~10^-11, max ~10^-10 to ~10^-9 (sub-03 max=8.7e-10), parcels exceeding 1 SD threshold = 0. `flag_mean_shift` = false for all subjects.

### HP Run Structure

| Stimulus type | Runs per subject | TRs per subject |
|---|---|---|
| harrypotter | 7 (run-1 through run-7) | 3,363 |

## Outputs

- `output/hp_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/{run_id}.npy` - PCA-projected HP run, shape (n_trs, n_pcs); 7 files per subject
- `output/hp_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/hp_run_ids.json` - Run IDs grouped by stimulus type
- `output/hp_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/pca_transfer_diagnostic.json` - Full diagnostic: overall R2, per-type R2, per-network R2, transfer gap, mean-corrected R2, parcel mean shift, flags
- `output/hp_03_projected/atlas-4S156Parcels/{sub_id}/vt0.95/pca_transfer_diagnostic.png` - Diagnostic figure (brain surface + per-network bar chart + R2 comparison)
