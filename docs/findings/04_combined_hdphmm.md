# Findings: 04 Combined HMM (sticky HDP-HMM priors)

_Script: `script/04_combined_hdphmm.py`. Tier: MAIN (Methods, R1)._

_Fits one combined Gaussian HMM per subject across all episodes (sticky and hierarchical-Dirichlet transition priors borrowed from the sticky HDP-HMM, under a fixed-capacity weak-limit truncation); selects best config via Pareto analysis; Viterbi-decodes all TRs; validates with leave-one-season-out refits. Per-subject, n=6._

## Method (as run)

- **Parcellation:** atlas-4S156Parcels (156 parcels: 100 cortical Schaefer + 56 subcortical)
- **Input:** PCA-projected episode time series from 03a (vt=0.95; n_pcs varies by subject, 67-77)
- **Selected config:** `vt0.95_covdiag_nc50_g1` (nc=50 capacity, gamma=1, kappa=10, alpha=1, rho=1, diagonal covariance in PC space, 5-seed grid search)
- **Config selection:** Pareto analysis of validation LL vs K_active (not BIC, not LL-based metrics); nc=50 gamma=1 sits in the middle Pareto cluster (K_active approx 35-45) with high capacity utilization and small overfit gap
- **Final refit:** 10 seeds on combined train+validation data; best seed selected by trainval_ll_per_sample; test set is 15% of episodes held out during all fitting
- **Decoding:** Viterbi (single most probable state sequence); all episodes decoded (train + valid + test)
- **sub-04 scope:** seasons 1-4 only (seasons 5-6 absent from data)
- **LOSO validation:** 6 folds per subject (4 for sub-04); each fold refits from scratch on 5 seasons, evaluates on held-out season; each fold has its own PCA fit to prevent data leakage

## Results

### Final model: active states and PCA dimensionality

| Subject | n_components | K_active | n_pcs (vt=0.95) | Capacity used |
|---|---|---|---|---|
| sub-01 | 50 | 42 | 75 | 84% |
| sub-02 | 50 | 42 | 72 | 84% |
| sub-03 | 50 | 42 | 72 | 84% |
| sub-04 | 50 | 41 | 77 | 82% |
| sub-05 | 50 | 41 | 67 | 82% |
| sub-06 | 50 | 37 | 74 | 74% |

Mean K_active: 40.8; range 37-42.

### Final model: generalization metrics (test set)

| Subject | trainval_ll/TR | test_ll/TR | baseline_ll/TR | overfit_gap | best_seed |
|---|---|---|---|---|---|
| sub-01 | -3.470 | -3.829 | -3.738 | 0.359 | 1 |
| sub-02 | -0.109 | -0.421 | -3.738 | 0.312 | 1 |
| sub-03 | -12.926 | -13.054 | -3.738 | 0.128 | 6 |
| sub-04 | -11.186 | -11.052 | -3.714 | -0.134 | 6 |
| sub-05 | -9.106 | -9.699 | -3.714 | 0.593 | 1 |
| sub-06 | -9.362 | -9.393 | -3.611 | 0.030 | 3 |

Baseline is log(1/K_active), the uniform-over-active-states null. Only sub-02 test_ll exceeds the active-state baseline (test=-0.421 vs baseline=-3.738); all other subjects fall below baseline.

### Per-season test log-likelihood

| Subject | s1 | s2 | s3 | s4 | s5 | s6 |
|---|---|---|---|---|---|---|
| sub-01 | -4.262 | -3.293 | -3.560 | -4.485 | -4.345 | -3.296 |
| sub-02 | -0.581 | -2.535 | 0.604 | -0.459 | -0.939 | 1.278 |
| sub-03 | -14.736 | -13.411 | -13.093 | -12.647 | -12.566 | -11.537 |
| sub-04 | -10.878 | -12.208 | -10.181 | -10.898 | n/a | n/a |
| sub-05 | -9.797 | -9.395 | -9.661 | -8.428 | -10.159 | -10.572 |
| sub-06 | -8.832 | -9.690 | -10.487 | -10.989 | -8.875 | -7.799 |

sub-05 shows the widest seasonal range (s4=-8.428 vs s6=-10.572); sub-02 s3 and s6 are positive (model exceeds baseline for those seasons).

### Grid search: chosen config summary (5-seed, vt=0.95_covdiag_nc50_g1)

| Subject | n_pcs | mean K_active (5 seeds) | mean valid_ll/TR (5 seeds) |
|---|---|---|---|
| sub-01 | 75 | 32.2 | -3.689 |
| sub-02 | 72 | 32.2 | -0.252 |
| sub-03 | 72 | 33.8 | -13.296 |
| sub-04 | 77 | 31.4 | -11.881 |
| sub-05 | 67 | 32.6 | -9.543 |
| sub-06 | 74 | 31.0 | -9.973 |

Final-refit K_active (42, 42, 42, 41, 41, 37) exceeds grid-search mean K across all subjects, consistent with 10-seed best selection finding a higher-utilization optimum.

### LOSO validation: K_active per fold

| Subject | s1 | s2 | s3 | s4 | s5 | s6 |
|---|---|---|---|---|---|---|
| sub-01 | 40 | 39 | 37 | 41 | 38 | 39 |
| sub-02 | 39 | 39 | 38 | 40 | 37 | 39 |
| sub-03 | 42 | 38 | 40 | 42 | 39 | 41 |
| sub-04 | 40 | 38 | 39 | 36 | n/a | n/a |
| sub-05 | 41 | 38 | 43 | 42 | 39 | 42 |
| sub-06 | 39 | 39 | 39 | 38 | 38 | 40 |

K_active is stable across folds: within-subject range 2-5 states (sub-06 range=2; sub-05 range=5); all folds remain in the 36-43 band.

### LOSO validation: test log-likelihood and overfit gap per fold

| Subject | Fold | test_ll/TR | overfit_gap | generalizes vs baseline |
|---|---|---|---|---|
| sub-01 | s1 | -3.340 | -0.215 | yes |
| sub-01 | s2 | -3.747 | 0.224 | no |
| sub-01 | s3 | -3.768 | 0.252 | no |
| sub-01 | s4 | -4.127 | 0.727 | no |
| sub-01 | s5 | -4.464 | 0.476 | no |
| sub-01 | s6 | -3.882 | 0.384 | no |
| sub-02 | s1 | -0.097 | -0.081 | yes |
| sub-02 | s2 | -1.554 | 1.707 | yes |
| sub-02 | s3 | -0.584 | -0.211 | yes |
| sub-02 | s4 | -0.893 | 0.195 | yes |
| sub-02 | s5 | -1.135 | 0.509 | yes |
| sub-02 | s6 | -0.380 | -0.422 | yes |
| sub-03 | s1 | -12.832 | 0.439 | no |
| sub-03 | s2 | -13.411 | 1.108 | no |
| sub-03 | s3 | -13.804 | 1.050 | no |
| sub-03 | s4 | -13.286 | 0.448 | no |
| sub-03 | s5 | -13.062 | 0.100 | no |
| sub-03 | s6 | -11.629 | -1.534 | no |
| sub-04 | s1 | -10.373 | -0.491 | no |
| sub-04 | s2 | -11.505 | 0.365 | no |
| sub-04 | s3 | -11.343 | 0.878 | no |
| sub-04 | s4 | -11.845 | 0.825 | no |
| sub-05 | s1 | -10.952 | 2.050 | no |
| sub-05 | s2 | -10.063 | 0.377 | no |
| sub-05 | s3 | -8.024 | -0.867 | no |
| sub-05 | s4 | -8.747 | -1.109 | no |
| sub-05 | s5 | -9.720 | 0.657 | no |
| sub-05 | s6 | -10.353 | 0.763 | no |
| sub-06 | s1 | -9.084 | -0.934 | no |
| sub-06 | s2 | -9.837 | 0.566 | no |
| sub-06 | s3 | -10.665 | 2.205 | no |
| sub-06 | s4 | -10.042 | 1.388 | no |
| sub-06 | s5 | -9.062 | -0.359 | no |
| sub-06 | s6 | -8.302 | -1.261 | no |

34 folds total (6 per subject for sub-01/02/03/05/06; 4 for sub-04). sub-02 is the only subject where all 6 folds exceed the active-state baseline. Negative overfit gaps indicate the held-out season was easier to model than training data.

### Split-half reliability: K_active and LL

| Subject | Half A K_active | Half B K_active | Half A trainval_ll | Half B trainval_ll | Half A valid_ll | Half B valid_ll |
|---|---|---|---|---|---|---|
| sub-01 | 38 | 38 | -3.446 | -3.510 | -3.579 | -3.604 |
| sub-02 | 39 | 39 | -0.718 | -0.079 | -1.036 | 0.061 |
| sub-03 | 37 | 38 | -12.930 | -12.785 | -12.788 | -12.817 |
| sub-04 | 39 | 39 | -11.306 | -10.825 | -12.030 | -9.746 |
| sub-05 | 37 | 41 | -9.751 | -9.028 | -9.660 | -9.609 |
| sub-06 | 36 | 36 | -8.728 | -9.471 | -8.691 | -9.556 |

Split-half K_active matches to within 0-4 states across halves (sub-05 the widest at 37 vs 41).

## Outputs

- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/final/vt0.95/final_results.json
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/final/vt0.95/decoded_states.pkl
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/final/vt0.95/state_means_parcel.npy
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/final/vt0.95/state_means_pca.npy
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/final/vt0.95/state_covars.npy
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/final/vt0.95/state_covars_parcel.npy
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/final/vt0.95/best_model.pkl
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/final/vt0.95/pca_model.pkl
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/final/vt0.95/seeds/seed_{N}.json
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/configs/vt0.95_covdiag_nc50_g1/config_summary.json
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/loso/season_{S}/loso_results.json
- output/04_combined_hdphmm/atlas-4S156Parcels/sub-*/split_half/{A,B}/split_half_results.json
