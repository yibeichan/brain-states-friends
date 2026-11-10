# Findings: held-out season transition modules vs the main partition (`sm_rel_r7_loso_module_agreement`)

## Question

`06e` LOSO2 re-partitions each held-out season (decoded by the model fitted without that season) and shows the partition is modular. Is it the same partition as the main one (M), or only some modular partition?

## Method (as run)

- Fold states are mapped to primary states through the `04ra` Hungarian matches, keeping well-matched pairs (r ≥ 0.3).
- Kept states: fold states that are nodes of the LOSO2 partition and whose matched primary state is a node of M; 38–44 per fold.
- ARI between the LOSO2 labels and M's labels on the kept states. p from 10,000 permutations of the fold labels (floor 0.0001).
- Reference: the same ARI for LOSO_, the partition of the fold model's own transition matrix, which is fitted on the five training seasons and uses no held-out data.
- sub-04 has four LOSO folds, the others six.

## Results

LOSO2 ARI / LOSO_ ARI (kept states), per fold:

| sub | main modules | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | LOSO2 median |
|---|---|---|---|---|---|---|---|---|
| sub-01 | 4 | 0.74 / 0.75 (43) | 0.62 / 0.56 (41) | 0.57 / 0.84 (39) | 0.52 / 1.00 (43) | 0.36 / 0.47 (42) | 0.71 / 0.79 (41) | 0.59 |
| sub-02 | 4 | 0.22 / 0.38 (41) | 0.52 / 0.86 (39) | 0.36 / 0.44 (40) | 0.28 / 0.59 (40) | 0.36 / 0.62 (40) | 0.34 / 0.80 (38) | 0.35 |
| sub-03 | 6 | 0.43 / 0.82 (44) | 0.45 / 0.50 (43) | 0.49 / 0.37 (44) | 0.23 / 0.59 (44) | 0.39 / 0.42 (43) | 0.42 / 0.42 (44) | 0.43 |
| sub-04 | 4 | 0.50 / 0.50 (40) | 0.69 / 0.60 (40) | 0.57 / 0.48 (41) | 0.46 / 0.49 (40) | — | — | 0.53 |
| sub-05 | 5 | 0.51 / 0.50 (42) | 0.41 / 0.44 (41) | 0.43 / 0.43 (42) | 0.28 / 0.36 (43) | 0.85 / 0.85 (43) | 0.33 / 0.54 (41) | 0.42 |
| sub-06 | 4 | 0.69 / 0.48 (39) | 0.68 / 0.74 (39) | 0.56 / 0.56 (40) | 0.55 / 0.84 (39) | 0.70 / 0.51 (41) | 0.30 / 0.63 (38) | 0.62 |

Permutation p: 33 of 34 folds at the 0.0001 floor; sub-02 fold 1 p = 0.0008.

## Interpretation (bounded)

Held-out seasons recover the main module structure well above chance in every fold (ARI 0.22–0.85, per-participant medians 0.35–0.62), but only partly: no fold reproduces M. The LOSO_ reference, from the fold model's five training seasons, gives ARI 0.36–1.00 and is higher than LOSO2 in 25 of 34 folds (equal in 2). Agreement is lower when the partition is estimated from one season's transitions than from the fold model's five training seasons.

## Caveats and audit notes

- The main partition M comes from the final model's transition matrix, which was fitted on all six seasons. Each held-out season is therefore one sixth of the data M was estimated from, so this measures how well one season's transitions recover M, not replication on independent data, and the shared data inflate the ARI.
- Louvain partitions depend on the seed. Seed variability in either partition is part of the measured disagreement.
- States not well matched, or absent from either partition, are left out (0–5 per fold), so ARI is over a slightly reduced set.
- The label-permutation null tests agreement against no relation. It is not a test against a specific alternative partition.

## Connections

- `06e_transition_modules.py` (LOSO and LOSO2 variants, `partitions.json`), `04ra_loso_struct_comp.py` (`hungarian_matching.json`). The bootstrap ARI in `06e` (run resamples of the full data) is the within-data counterpart.
