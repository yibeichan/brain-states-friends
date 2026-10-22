# Findings: 06e Transition Modules

_Script: `script/06e_transition_modules.py`. Tier: MAIN (R3, Fig 3)._

_Per subject (n=6), tests whether the state transition graph is modular beyond what state stickiness, state popularity and the run-level composition of states produce, and whether the modules carry information beyond the pairwise similarity of the states._

## Question

06b finds Louvain communities in each subject's transition graph but reports them
descriptively, with no modularity value and no null. Two questions follow:

1. Is the transition graph more modular than graphs that keep simpler structure
   (which states occur, for how long, how sticky and how popular they are)?
2. If so, do the modules add anything to what pairwise state similarity (spatial
   maps, FC, distributional distance, network label, co-occurrence) already
   predicts about transitions, to module-level dwell, or to stimulus locking
   across subjects?

## Method (as run)

- Parcellation: atlas-4S156Parcels; vt=0.95. Inputs: model and decoded runs from 04
  (select and LOSO folds), recurrence scores from 05a, state flags from 05e_a4, FC
  similarity from 05f, transition graph from 06b.
- Nodes: active states (recurrence > 0), as in 06b A1 (42-47 per subject).
- Graphs:
  - **M**: model `transmat_`, self-loops removed, edges with P >= 0.01 (rebuilt and
    checked against 06b's `transition_graph.graphml`). M's partition is the one
    characterised and shown in Fig 3A.
  - **E**: empirical transition probabilities from the decoded runs (counted within
    runs, normalised over all states like the model), self-loops removed, P >= 0.01.
    The decision rule tests E.
  - Threshold variants of M: M02 (P >= 0.02), M005 (P >= 0.005), Mrenorm
    (off-diagonal rows rescaled to sum to 1, then P >= 0.01), Mfull (all
    off-diagonal P).
- Statistic: directed weighted modularity Q (Leicht-Newman), best of 50 fixed Louvain
  seeds at resolution 1, applied the same way to every observed and null graph.
- Nulls:
  - **N2** (on E, 1000 draws): segment order shuffled within each run. Keeps which
    states occur in which run and every segment length; destroys only their order.
  - **N3** (on E, 500 draws): a non-modular Markov chain with each state's fitted
    stay probability and in-strength popularity, simulated with the observed run
    lengths and start states.
  - **N1** (on M and variants, 1000 draws; 200 for sensitivity graphs): degree- and
    reciprocity-preserving rewiring with strength-matched weight reassignment.
    **N1row** (on Mfull): off-diagonal entries shuffled within each row.
  - **PRED** (on M, 1000 draws): log P regressed on the pairwise predictors plus
    log in- and out-strength; null graphs are the fitted prediction plus permuted
    residuals. States without a defined FC similarity row are dropped.
- p-values: one-sided, (1 + #null >= observed) / (1 + draws). Effect = Q_obs minus
  the null mean.
- Stability: 200 run-level bootstraps of E; ARI of each bootstrap partition against
  E's partition. A consensus partition is also built from the bootstrap
  co-assignment matrix.
- LOSO2: for each leave-one-season-out fold, the held-out season decoded by the fold
  model, E built from it, and tested against N2 (1000 draws).
- **Decision rule.** A subject is modular when N2 p < .05, N3 p < .05, the N2 effect
  is >= 0.05, and the bootstrap ARI median is >= 0.5. N1 is not part of the rule:
  in calibration it rejected 38.5-42.5% of graphs with no modules (table below). A
  variant with permuted rather than strength-matched weights (N1perm) held its
  error rate but rejected planted-module graphs no more often than graphs without
  modules at these graph sizes, so it could not replace N1. N1 results are still
  computed and reported descriptively; the N1perm null was not run on the data.
- Three further criteria, computed for subjects that are modular and pass LOSO2:
  - (a) Similarity-predicted: Q of M beyond the PRED null, p < .05.
  - (b) Module dwell: mean run length within one M module (TRs) beyond the same
    segment shuffle as N2 (1000 draws), p < .05.
  - (c) Stimulus locking: for each subject pair, modules matched by centroid map
    (Hungarian on 1 - r); mean correlation of matched module occupancy over
    common runs, against circular shifts of one subject's runs (1000 draws).
    Passes when at least half of the subject's 5 pairs have p < .05.
- R3 mapping (fixed before the run): modules lead R3 when at least 5 subjects are
  modular, every modular subject passes LOSO2 in at least 4 folds, and at least 4
  of them meet one or more of (a)-(c). With at least 5 modular subjects but a
  failed LOSO2 or new-criterion condition, modules are display only; otherwise R3
  stays with the FC ordering from 06b.
- Calibration: graphs built from sub-01 (200 graphs per generator, 200 null draws,
  10 Louvain seeds each). N1 and N1perm were tested on their own null draws
  (exchangeable), on graphs re-estimated from a simulated non-modular chain, and
  on planted-module graphs matched to sub-01's size, density and weight spread
  (within/between edge-probability ratio 1.5 or 2). N2 was tested on simulated
  non-modular chains and on chains with planted modules (within-module transition
  mass multiplied by 1.5 or 2).
- Per-subject inference; no group statistic. sub-04 has 4 seasons (4 LOSO folds),
  the others 6.

## Results

### Calibration (rate of p < .05; 200 graphs per row)

| Null | Generator | Modules planted | Rate p < .05 | 95% CI | Mean effect |
|---|---|---|---|---|---|
| N1 | own null draws | no | 0.385 | [0.317, 0.456] | 0.017 |
| N1 | non-modular chain | no | 0.425 | [0.356, 0.497] | 0.013 |
| N1 | planted modules, ratio 1.5 | yes | 0.315 | [0.251, 0.384] | 0.014 |
| N1 | planted modules, ratio 2.0 | yes | 0.295 | [0.233, 0.363] | 0.014 |
| N1perm | own null draws | no | 0.050 | [0.024, 0.090] | -0.000 |
| N1perm | non-modular chain | no | 0.015 | [0.003, 0.043] | -0.017 |
| N1perm | planted modules, ratio 1.5 | yes | 0.050 | [0.024, 0.090] | -0.000 |
| N1perm | planted modules, ratio 2.0 | yes | 0.060 | [0.031, 0.102] | 0.002 |
| N2 | non-modular chain | no | 0.000 | [0.000, 0.018] | -0.012 |
| N2 | modular chain, ratio 1.5 | yes | 1.000 | [0.982, 1.000] | 0.126 |
| N2 | modular chain, ratio 2.0 | yes | 1.000 | [0.982, 1.000] | 0.310 |

N1 rejects far too often without modules, and no more often with them. N1perm keeps its
error rate but has no power on planted modules. N2 holds its error rate and detects
planted modules at both strengths. Without modules, the 95th percentile of its
effect was 0.002, well below the 0.05 effect floor.

### Decision rule (E graph)

| Subject | M nodes / edges | E nodes / edges | Q (E) | N2 effect, p | N3 effect, p | Bootstrap ARI median [IQR] | LOSO2 folds p < .05 | Modules in M | Modular |
|---|---|---|---|---|---|---|---|---|---|
| sub-01 | 46 / 335 | 46 / 296 | 0.440 | 0.242, 0.001 | 0.307, 0.002 | 0.66 [0.59, 0.83] | 6 / 6 | 4 | yes |
| sub-02 | 46 / 398 | 46 / 345 | 0.427 | 0.209, 0.001 | 0.260, 0.002 | 0.68 [0.59, 0.74] | 6 / 6 | 4 | yes |
| sub-03 | 44 / 378 | 44 / 341 | 0.355 | 0.267, 0.001 | 0.281, 0.002 | 0.82 [0.72, 0.92] | 6 / 6 | 6 | yes |
| sub-04 | 44 / 379 | 44 / 364 | 0.394 | 0.262, 0.001 | 0.275, 0.002 | 1.00 [0.91, 1.00] | 4 / 4 | 4 | yes |
| sub-05 | 47 / 381 | 47 / 343 | 0.468 | 0.294, 0.001 | 0.344, 0.002 | 0.61 [0.56, 0.73] | 6 / 6 | 5 | yes |
| sub-06 | 42 / 371 | 42 / 337 | 0.446 | 0.382, 0.001 | 0.413, 0.002 | 0.92 [0.89, 1.00] | 6 / 6 | 4 | yes |

All N2 and N3 p-values are at the floor for their draw counts (1/1001 and 1/501).
Every LOSO2 fold in every subject has p at the floor, with N2 effects of 0.167-0.384.
M module sizes: sub-01 19/11/9/7; sub-02 20/12/12/2; sub-03 12/10/7/6/5/4;
sub-04 20/15/5/4; sub-05 25/8/7/5/2; sub-06 18/10/8/6.

### Further criteria

| Subject | (a) PRED p | (b) module dwell / state dwell / null mean (TRs), p | (c) locking pairs p < .05 | Meets (a) / (b) / (c) |
|---|---|---|---|---|
| sub-01 | 0.252 | 8.12 / 3.85 / 5.73, 0.001 | 5 / 5 | no / yes / yes |
| sub-02 | 0.948 | 8.07 / 3.73 / 5.89, 0.001 | 5 / 5 | no / yes / yes |
| sub-03 | 0.968 | 5.45 / 3.23 / 4.10, 0.001 | 5 / 5 | no / yes / yes |
| sub-04 | 0.384 | 8.89 / 3.00 / 5.71, 0.001 | 5 / 5 | no / yes / yes |
| sub-05 | 0.293 | 8.28 / 3.57 / 5.64, 0.001 | 5 / 5 | no / yes / yes |
| sub-06 | 0.067 | 7.30 / 2.62 / 4.40, 0.001 | 5 / 5 | no / yes / yes |

Across the 15 subject pairs, mean matched-module occupancy correlation was
0.028-0.209 against circular-shift null means of -0.003 to 0.003; all 15 pairs have p
at the floor (1/1001).

### Descriptive (not in the rule)

N1 and threshold variants of M (effect = Q minus N1 or N1row null mean; every p is at
the 1/1001 floor, and N1 is miscalibrated, so these p-values are not evidence):

| Subject | M | M02 | M005 | Mrenorm | Mfull (N1row) |
|---|---|---|---|---|---|
| sub-01 | 0.422 (+0.087) | 0.540 (+0.062) | 0.328 (+0.080) | 0.253 (+0.064) | 0.266 (+0.061) |
| sub-02 | 0.406 (+0.101) | 0.551 (+0.072) | 0.321 (+0.085) | 0.246 (+0.067) | 0.270 (+0.065) |
| sub-03 | 0.346 (+0.094) | 0.453 (+0.100) | 0.276 (+0.073) | 0.248 (+0.066) | 0.233 (+0.054) |
| sub-04 | 0.388 (+0.121) | 0.483 (+0.125) | 0.324 (+0.103) | 0.302 (+0.093) | 0.282 (+0.083) |
| sub-05 | 0.453 (+0.116) | 0.586 (+0.105) | 0.369 (+0.106) | 0.290 (+0.078) | 0.312 (+0.085) |
| sub-06 | 0.433 (+0.095) | 0.529 (+0.094) | 0.372 (+0.086) | 0.311 (+0.073) | 0.327 (+0.047) |

Values are Q (effect). LOSO folds of M tested with N1 give p < .05 in every fold.

Partition agreement with M (ARI on shared nodes). Sensitivity graphs: Mreliable keeps
states with >= 30 TRs, >= 20 exits and no diagnostic flag; Eaba collapses A-B-A
flicker (middle segment <= 4 TRs); Emerge90 merges states whose maps correlate
above 0.9; Edrop10 drops the first 10 TRs of each run. Each was also tested with N1
(200 draws; every p at the 1/201 floor except sub-01 Mreliable, p = 0.020).

| Subject | E | Bootstrap consensus | Mreliable (nodes) | Eaba | Emerge90 | Edrop10 |
|---|---|---|---|---|---|---|
| sub-01 | 0.90 | 0.86 | 0.31 (33) | 0.86 | 0.90 | 0.94 |
| sub-02 | 0.74 | 0.93 | 0.48 (31) | 0.68 | 0.74 | 0.68 |
| sub-03 | 0.85 | 0.67 | 0.38 (28) | 0.82 | 0.85 | 0.85 |
| sub-04 | 1.00 | 1.00 | 0.72 (29) | 1.00 | 1.00 | 1.00 |
| sub-05 | 0.58 | 0.96 | 0.21 (30) | 1.00 | 0.58 | 1.00 |
| sub-06 | 0.92 | 0.92 | 0.10 (17) | 1.00 | 0.92 | 1.00 |

### Outcome

`r3_mapping.json` gives **modules lead R3**. All three conditions held: 6 of 6
subjects are modular, all 6 pass LOSO2 in every fold, and all 6 meet at least one
further criterion ((b) and (c); none meets (a)).

## Interpretation (bounded)

- In each of the six subjects, transitions between states are grouped into
  "transition modules": the empirical graph is more modular than graphs that keep
  each run's states and segment lengths (N2) or each state's stickiness and
  popularity (N3), the partition is stable under run bootstraps, and the result
  holds in every held-out season. This is per-subject inference; with six subjects
  there is no group claim.
- The test is on E, while Fig 3A shows M's partition. The two partitions agree
  closely in five subjects (ARI 0.74-1.00) and less in sub-05 (0.58).
- Criterion (a) failed in every subject: the modularity of M is no higher than in
  graphs predicted from pairwise state similarity (spatial map, FC, distributional
  distance, network label, recurrence difference, co-occurrence) plus strengths.
  The modules therefore do not add to what pairwise state similarity predicts about
  transitions; they are a compact description of that similarity structure, not a
  separate organising principle.
- Module dwell (b) is 1.7 to 3.0 times state dwell and above the segment-shuffle null.
- Locking (c) shows that matched modules are occupied at the same moments of the
  same episodes across subjects. It was not compared with locking at the level of
  single states, so it does not show that modules lock better than states do.
- Partitions on reliable states only (Mreliable) agree weakly with M (ARI
  0.10-0.72), so module membership of low-occupancy or flagged states should not be
  read state by state.

## Outputs and reproduce

- `output/06e_transition_modules/atlas-4S156Parcels/r3_mapping.json`: outcome,
  subject lists, frozen parameters, pairwise locking results
- `output/06e_transition_modules/atlas-4S156Parcels/{sub_id}/vt0.95/transition_modules_summary.json`:
  all tests, stability, LOSO (`loso` = LOSO2 under N2, `loso_N1`), further criteria,
  characterisation, module occupancy by season and run decile
- `output/06e_transition_modules/atlas-4S156Parcels/{sub_id}/vt0.95/partitions.json`
  (M partition, networks, consensus labels), `bootstrap_coassignment.npy`,
  `nulls/{variant}/chunk_###.npz` + `observed.json`
- `output/06e_transition_modules/atlas-4S156Parcels/calibration/calibration_summary.json`
  (N1), `calibration/N1perm/calibration_summary.json`,
  `calibration/N2/calibration_summary.json`

```bash
# Unit tests
uv run --with pytest pytest script/tests/test_transition_modules.py

# Null draws: one (subject, variant) task per array index (--mode tasks lists them),
# then aggregate. Rows 0-111 are the original variants and N1 folds, 112-193 the
# permuted-weight N1p rows (not used by the aggregate), 194-227 the LOSO2 folds.
jid=$(sbatch --parsable --array=0-111,194-227 script/06e_transition_modules.sh)
sbatch --dependency=afterok:$jid --export=ALL,MODE=aggregate --time=04:00:00 \
    --mem=12G --no-requeue script/06e_transition_modules.sh

# Calibration of one null (N1, N1perm or N2)
uv run --project . --no-sync python script/06e_transition_modules.py --mode calibrate_tasks --cal_null N2
sbatch --array=0-29 --export=ALL,MODE=calibrate,CAL_NULL=N2 script/06e_transition_modules.sh
uv run --project . --no-sync python script/06e_transition_modules.py --mode calibrate_summary --cal_null N2
```

The task table also lists `N1p_*` and `LOSOp_*` rows (the N1perm null on the data).
The aggregate step does not read them, and they were not run.
