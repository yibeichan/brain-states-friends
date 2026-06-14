# Findings: 06d Preserved Transition Chains

_Script: `script/06d_preserved_chains.py`. Tier: SUPP (Supp)._

_For each subject (n=6), tests which state-change bigrams and trigrams recur across episodes beyond what a first-order Markov null predicts, and whether preserved chains are season-specific._

## Method (as run)

- Parcellation: atlas-4S156Parcels; variance threshold: vt=0.95 (67-77 PCs per subject)
- Input: decoded state sequences from 04 (select mode) and recurrence scores from 05a
- State sequences collapsed to state-change sequences (consecutive identical states merged; min_dwell_tr=2 TRs)
- N-gram orders tested: 2 (bigrams) and 3 (trigrams)
- Chain Preservation Score (CPS): fraction of episodes where a given n-gram appears at least once (binary per episode)
- Null model: 1000 Markov-1 surrogate sequences per episode, generated from the model's fitted state-change transition matrix (model.transmat_); surrogate lengths matched to observed
- Permutation p-value: one-tailed (Phipson-Smyth finite-sampling correction): p = (count + 1) / (n_sim + 1)
- Multiple comparisons: Benjamini-Hochberg FDR correction across all tested n-grams (threshold: q < 0.05)
- Pre-filter: n-grams present in fewer than 5 episodes excluded from testing
- Season-specificity test: for each preserved chain, Season Specificity Index (SSI = max - min per-season CPS) tested by permuting season labels (n=5000); FDR-corrected across preserved chains
- Per-subject analysis; no group statistic; sub-04 has 194 episodes (4 seasons) vs 289-292 for other subjects

## Results

### Preservation summary: bigrams (order 2)

| Subject | Episodes | N tested | N sig (FDR < 0.05) | Sig % | CPS min | CPS max | CPS median |
|---------|----------|----------|---------------------|-------|---------|---------|------------|
| sub-01  | 292      | 1350     | 81                  | 6.0%  | 0.0171  | 0.3288  | 0.034      |
| sub-02  | 292      | 1343     | 108                 | 8.0%  | 0.0171  | 0.1849  | 0.027      |
| sub-03  | 291      | 1381     | 227                 | 16.4% | 0.0172  | 0.3333  | 0.041      |
| sub-04  | 194      | 1120     | 157                 | 14.0% | 0.0258  | 0.6701  | 0.057      |
| sub-05  | 289      | 1319     | 95                  | 7.2%  | 0.0173  | 0.2180  | 0.035      |
| sub-06  | 292      | 1296     | 393                 | 30.3% | 0.0171  | 0.7363  | 0.051      |

### Preservation summary: trigrams (order 3)

| Subject | Episodes | N tested | N sig (FDR < 0.05) | Sig % | CPS min | CPS max | CPS median |
|---------|----------|----------|---------------------|-------|---------|---------|------------|
| sub-01  | 292      | 950      | 80                  | 8.4%  | 0.0171  | 0.1130  | 0.021      |
| sub-02  | 292      | 1003     | 45                  | 4.5%  | 0.0171  | 0.1644  | 0.024      |
| sub-03  | 291      | 1265     | 82                  | 6.5%  | 0.0172  | 0.0893  | 0.021      |
| sub-04  | 194      | 821      | 68                  | 8.3%  | 0.0258  | 0.1443  | 0.036      |
| sub-05  | 289      | 903      | 85                  | 9.4%  | 0.0173  | 0.1696  | 0.021      |
| sub-06  | 292      | 1446     | 392                 | 27.1% | 0.0171  | 0.5753  | 0.024      |

### Top preserved chain per subject (highest CPS among FDR-significant chains)

| Subject | Top bigram  | CPS   | Top trigram    | CPS   |
|---------|-------------|-------|----------------|-------|
| sub-01  | 44->27      | 0.329 | 44->27->24     | 0.113 |
| sub-02  | 25->15      | 0.185 | 22->45->7      | 0.164 |
| sub-03  | 36->39      | 0.333 | 5->36->39      | 0.089 |
| sub-04  | 38->5       | 0.670 | 40->2->26      | 0.144 |
| sub-05  | 0->7        | 0.218 | 25->0->7       | 0.170 |
| sub-06  | 4->40       | 0.736 | 1->4->40       | 0.575 |

State indices are subject-specific and cannot be compared across subjects.

### Season specificity of preserved chains

| Subject | Bigrams tested | Bigrams season-specific (FDR < 0.05) | SSI median (2-gram) | Trigrams tested | Trigrams season-specific (FDR < 0.05) | SSI median (3-gram) |
|---------|----------------|--------------------------------------|---------------------|-----------------|---------------------------------------|---------------------|
| sub-01  | 81             | 0                                    | 0.062               | 80              | 0                                     | 0.042               |
| sub-02  | 108            | 0                                    | 0.060               | 45              | 0                                     | 0.042               |
| sub-03  | 227            | 0                                    | 0.066               | 82              | 0                                     | 0.043               |
| sub-04  | 157            | 0                                    | 0.064               | 68              | 0                                     | 0.062               |
| sub-05  | 95             | 0                                    | 0.067               | 85              | 0                                     | 0.047               |
| sub-06  | 393            | 0                                    | 0.083               | 392             | 0                                     | 0.060               |

## Outputs

- output/06d_preserved_chains/atlas-4S156Parcels/{sub_id}/vt0.95/preservation_summary.json
- output/06d_preserved_chains/atlas-4S156Parcels/{sub_id}/vt0.95/preserved_chains_2grams.csv
- output/06d_preserved_chains/atlas-4S156Parcels/{sub_id}/vt0.95/preserved_chains_3grams.csv
- output/06d_preserved_chains/atlas-4S156Parcels/{sub_id}/vt0.95/season_specificity_2grams.csv
- output/06d_preserved_chains/atlas-4S156Parcels/{sub_id}/vt0.95/season_specificity_3grams.csv
- output/06d_preserved_chains/atlas-4S156Parcels/{sub_id}/vt0.95/fig_preservation_{2,3}grams.png/.pdf
- output/06d_preserved_chains/atlas-4S156Parcels/{sub_id}/vt0.95/fig_top_preserved_{2,3}grams.png/.pdf
- output/06d_preserved_chains/atlas-4S156Parcels/{sub_id}/vt0.95/fig_preservation_vs_recurrence_{2,3}grams.png/.pdf
