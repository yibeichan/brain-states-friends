# Findings: 08e Transformer Cross-Stim Aggregate (D3a)

_Script: `script/08e_transformer_cross_stim_aggregate.py`. Tier: MAIN (R4b, Fig 4)._

_Trains a per-layer RidgeClassifier on Friends TRs and evaluates it on a held-out test stimulus (Movie10, Harry Potter, Petit Prince FR/EN), using the shared Friends HMM state space. Per-subject n=6 for Movie10; n=5 for Harry Potter and Petit Prince (sub-04 absent)._

## Method (as run)
- Parcellation atlas-4S156Parcels; Friends PCA fit on training split only (vt=0.95), applied to both Friends and test-stimulus features.
- Test stimuli: movie10 (3 models: DINOv2-large video, LLaMA-3.2-3B text, Wav2Vec-BERT-2.0 audio); harrypotter (LLaMA-3.2-3B only; no audio stream); petitprince_fr and petitprince_en (LLaMA-3.2-3B and Wav2Vec-BERT-2.0).
- Intersection states: content-eligible states (eligibility_source = 05e_a4) with FO >= 1% in both Friends and the test stimulus for a given subject.
- Classifier: RidgeClassifier (alpha=1.0, balanced class weights) trained on all Friends intersection TRs, tested on test-stimulus intersection TRs.
- Lag: Friends and test-stimulus features are shifted by the same brain-to-feature lag. The default (`--lag friends_best`) is each participant's best within-Friends lag for the model: the (lag, layer) cell with the highest 08d D1 balanced accuracy, lag 0 excluded, chosen on Friends only. `--lag N` sets it explicitly; `--lag 0` is the synchronous decoder of the original runs.
- Null: 1000 circular-shift permutations of Friends training labels (within-run, preserving autocorrelation); Ridge refit per permutation; evaluated on real test-stimulus labels. BH-FDR correction across layers.
- Effect size: normalized = (balanced_accuracy - chance) / (1 - chance), where chance = 1 / n_intersection_states.
- sub-04 has no HP or Petit Prince decoded states; sub-04 results appear for Movie10 only.

## Results

All tables below are lag-0 values (`--lag 0`, the folders without a suffix). Figure 5D-G and Figure S7 now read the best-lag outputs (`_lag-best` folders), whose values these tables do not show.

### Data coverage and intersection state counts

| Subject | Friends TRs (train+test) | Movie10 test TRs | HP test TRs | PP-FR test TRs | PP-EN test TRs | Movie10 n_states | HP n_states | PP-FR n_states | PP-EN n_states |
|---------|--------------------------|------------------|-------------|----------------|----------------|------------------|-------------|----------------|----------------|
| sub-01  | 104,900                  | 16,651           | 2,115       | 2,286          | 2,018          | 29               | 18          | 22             | 22             |
| sub-02  | 106,798                  | 15,812           | 2,558       | 2,959          | 2,503          | 30               | 20          | 21             | 19             |
| sub-03  | 82,595                   | 13,918           | 2,197       | 2,350          | 2,178          | 25               | 18          | 17             | 17             |
| sub-04  | 57,653                   | 14,967           | n/a          | n/a             | n/a             | 26               | n/a          | n/a             | n/a             |
| sub-05  | 93,188                   | 16,389           | 2,143       | 1,850          | 1,651          | 26               | 21          | 23             | 22             |
| sub-06  | 49,216                   | 7,816            | 1,138       | 900            | 1,085          | 13               | 11          | 10             | 10             |

Note: Friends TRs shown for the movie10_llama run; Movie10 test TRs shown for DINOv2; HP/PP counts are model-independent (same decoded states).

### Peak-layer transfer accuracy: Movie10

Peak balanced accuracy and FDR-significant layer counts across layers (layer index 0-based).

| Subject | DINOv2-large peak bal_acc (layer) | n_FDR / 24 | LLaMA-3.2-3B peak bal_acc (layer) | n_FDR / 28 | Wav2Vec-BERT-2.0 peak bal_acc (layer) | n_FDR / 24 | Chance |
|---------|-----------------------------------|------------|-----------------------------------|------------|---------------------------------------|------------|--------|
| sub-01  | 0.0506 (23)                       | 24/24      | 0.0609 (10)                       | 28/28      | 0.0726 (12)                           | 24/24      | 0.0345 |
| sub-02  | 0.0500 (18)                       | 24/24      | 0.0679 (13)                       | 28/28      | 0.0694 (11)                           | 24/24      | 0.0333 |
| sub-03  | 0.0567 (23)                       | 24/24      | 0.0666 (6)                        | 28/28      | 0.0782 (11)                           | 24/24      | 0.0400 |
| sub-04  | 0.0511 (17)                       | 19/24      | 0.0654 (10)                       | 28/28      | 0.0650 (11)                           | 24/24      | 0.0385 |
| sub-05  | 0.0525 (18)                       | 24/24      | 0.0632 (13)                       | 28/28      | 0.0682 (11)                           | 24/24      | 0.0385 |
| sub-06  | 0.0957 (20)                       | 3/24       | 0.1180 (12)                       | 28/28      | 0.1164 (11)                           | 24/24      | 0.0769 |

### Peak-layer normalized effect size: Movie10

| Subject | DINOv2-large max eff | LLaMA-3.2-3B max eff | Wav2Vec-BERT-2.0 max eff |
|---------|----------------------|----------------------|--------------------------|
| sub-01  | 0.0166               | 0.0273               | 0.0395                   |
| sub-02  | 0.0173               | 0.0358               | 0.0373                   |
| sub-03  | 0.0174               | 0.0277               | 0.0398                   |
| sub-04  | 0.0132               | 0.0281               | 0.0276                   |
| sub-05  | 0.0146               | 0.0257               | 0.0309                   |
| sub-06  | 0.0204               | 0.0445               | 0.0427                   |

### Peak-layer transfer accuracy: Harry Potter and Petit Prince (LLaMA-3.2-3B)

| Subject | HP peak bal_acc (layer) | HP n_FDR / 28 | HP chance | PP-FR peak bal_acc (layer) | PP-FR n_FDR / 28 | PP-EN peak bal_acc (layer) | PP-EN n_FDR / 28 |
|---------|-------------------------|----------------|-----------|----------------------------|------------------|----------------------------|------------------|
| sub-01  | 0.0919 (10)             | 11/28          | 0.0556    | 0.0769 (11)                | 16/28            | 0.0743 (12)                | 6/28             |
| sub-02  | 0.0771 (9)              | 17/28          | 0.0500    | 0.0818 (11)                | 21/28            | 0.0844 (8)                 | 22/28            |
| sub-03  | 0.0857 (11)             | 16/28          | 0.0556    | 0.0839 (12)                | 8/28             | 0.0823 (3)                 | 8/28             |
| sub-04  | n/a                      | n/a             | n/a        | n/a                         | n/a               | n/a                         | n/a               |
| sub-05  | 0.0703 (6)              | 1/28           | 0.0476    | 0.0627 (13)                | 1/28             | 0.0581 (24)                | 0/28             |
| sub-06  | 0.1456 (6)              | 24/28          | 0.0909    | 0.1455 (25)                | 1/28             | 0.1437 (15)                | 0/28             |

### Wav2Vec-BERT-2.0 transfer: Petit Prince

| Subject | PP-FR peak bal_acc (layer) | PP-FR n_FDR / 24 | PP-FR chance | PP-EN peak bal_acc (layer) | PP-EN n_FDR / 24 |
|---------|----------------------------|------------------|--------------|----------------------------|------------------|
| sub-01  | 0.0576 (11)                | 0/24             | 0.0455       | 0.0632 (7)                 | 0/24             |
| sub-02  | 0.0595 (10)                | 0/24             | 0.0476       | 0.0652 (15)                | 0/24             |
| sub-03  | 0.0774 (11)                | 0/24             | 0.0588       | 0.0733 (19)                | 0/24             |
| sub-04  | n/a                         | n/a               | n/a           | n/a                         | n/a               |
| sub-05  | 0.0653 (14)                | 2/24             | 0.0435       | 0.0558 (23)                | 0/24             |
| sub-06  | 0.1354 (0)                 | 0/24             | 0.1000       | 0.1389 (11)                | 2/24             |

### Movie10 per-film breakdown: LLaMA-3.2-3B (peak balanced accuracy and FDR-significant layers)

Films: Wolf of Wall Street (wolf), Figures in a Landscape (figures), Bourne Supremacy (bourne), Life (life). FDR count out of 28 layers.

| Subject | wolf bal_acc / n_FDR | figures bal_acc / n_FDR | bourne bal_acc / n_FDR | life bal_acc / n_FDR |
|---------|----------------------|-------------------------|------------------------|----------------------|
| sub-01  | 0.0751 / 28          | 0.0650 / 28             | 0.0513 / 0             | 0.0572 / 0           |
| sub-02  | 0.0744 / 28          | 0.0704 / 28             | 0.0704 / 24            | 0.0521 / 0           |
| sub-03  | 0.0771 / 28          | 0.0742 / 28             | 0.0861 / 3             | 0.0661 / 0           |
| sub-04  | 0.0665 / 26          | 0.0692 / 27             | 0.0740 / 12            | 0.0543 / 0           |
| sub-05  | 0.0759 / 28          | 0.0708 / 28             | 0.0618 / 8             | 0.0639 / 4           |
| sub-06  | 0.1381 / 28          | 0.1267 / 22             | 0.1311 / 12            | 0.0959 / 0           |

## Outputs
Each cell writes to `output/08e_transformer_cross_stim_aggregate/atlas-4S156Parcels/sub-*/<folder>/`, where `<folder>` is `{stimulus}_{model}_lag-best` for `--lag friends_best` (the default), `{stimulus}_{model}` for `--lag 0`, and `{stimulus}_{model}_lag-N` for another explicit lag. Each folder holds:
- D3a_transfer_{stimulus}_{model}.json (records `lag`, `lag_source`, `eligibility_source`, `intersection_states`; best-lag files also record the 08d D1 path (relative to `SCRATCH_DIR` in files written by this version, absolute in older ones), best layer and accuracy; best-lag files written by this version also record the layer fingerprint, code and input identities (including the 05e `content_eligible` list) and which layers were reused from an interrupted run, which files written by earlier versions do not have)
- D3a_per_subset_{stimulus}_{model}.json (movie10 with `--per_subset`; records `lag`, `lag_source`, `eligibility_source`, `intersection_states`; files written by this version also record the 08d D1 fields and the same provenance fields as the transfer file)
- pca_info.json
- D3a_transfer_{stimulus}_{model}.png

While a cell runs, its folder also holds `partial_layers/` (one file per finished layer, removed when the cell completes) and, during each JSON write, a `*.tmp.<pid>` file that is renamed into place; a killed job can leave either behind. Neither is an output: the output dataset should ignore them (its `.gitignore` is kept with the dataset, not in this repository).

Figures: fig_F5_within_friends.py (Panel D grid and Panels E-G Movie10 per-film lines) and 08e_plots.py (Figure S7) read the `_lag-best` folders through `utils.transformer_analysis.load_d3a_lag_best`, which requires every expected participant's file and checks each file's lag against the participant's current 08d D1 best lag and its states against the participant's current 05e content eligibility. If a participant's 05e state flags are missing, the reader (and a best-lag 08e run) stops and names the expected file instead of falling back to the 05a sub-HRF filter. For files written by earlier versions, which do not record the eligible set, the check is that every intersection state is still content-eligible (and the eligibility source and state count agree); states that became eligible later cannot be detected in them.
