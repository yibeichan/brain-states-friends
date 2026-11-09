# Findings: 06b Transition Structure

_Script: `script/06b_transition_structure.py`. Tier: MAIN (R3, Fig 3)._

_Directed graph topology, transition asymmetry, recurrence assortativity, network homophily, and MFPT landscape of brain states; per-subject, n=6._

## Method (as run)

- Script: `script/06b_transition_structure.py`
- Parcellation: atlas-4S156Parcels (156 parcels); vt=0.95 (67-77 PCs per subject)
- Config: Pareto-selected nc50_g1, diagonal covariance; 42-47 active states per subject
- A1/A4 use model `transmat_` (learned); A2/A3 use empirical transition matrix from 06a
- Edge thresholds: P > 0.01 for topology (A1); P > 0.005 for assortativity graph (A3)
- Community detection: Louvain with n=200 bootstrap resamples for consensus
- Permutation tests: n=5000; bootstrap CI: n=1000
- FC similarity input (A3 Mantel test, A4 MFPT-FC): `05f_state_fc/fc_similarity_corr_rv.npy`; NaN entries filtered before correlation
- Scope: per-subject, no group statistic; continuous recurrence scores from 05a

## Results

### A1: Graph Topology

| Subject | Active states | Nodes | Edges (P > 0.01) | Communities | Bidirectionality index |
|---------|--------------|-------|-------------------|-------------|------------------------|
| sub-01 | 46 | 46 | 335 | 4 | 0.838 |
| sub-02 | 46 | 46 | 398 | 4 | 0.854 |
| sub-03 | 44 | 44 | 378 | 5 | 0.812 |
| sub-04 | 44 | 44 | 379 | 4 | 0.762 |
| sub-05 | 47 | 47 | 381 | 5 | 0.844 |
| sub-06 | 42 | 42 | 371 | 4 | 0.797 |

### A2: Transition Selectivity

Concentration ratio = fraction of exit probability captured by the top-2 targets.

| Subject | Mean concentration ratio | Median top-1 prob | Median top-2 prob |
|---------|--------------------------|-------------------|-------------------|
| sub-01 | 0.326 | 0.032 | 0.022 |
| sub-02 | 0.314 | 0.036 | 0.026 |
| sub-03 | 0.336 | 0.049 | 0.031 |
| sub-04 | 0.385 | 0.072 | 0.036 |
| sub-05 | 0.330 | 0.030 | 0.021 |
| sub-06 | 0.424 | 0.077 | 0.035 |

### A3: Recurrence Assortativity

Weighted recurrence assortativity on empirical graph (P > 0.005); permutation test and bootstrap 95% CI.

| Subject | r | p | 95% CI | n edges |
|---------|------|--------|------------------|---------|
| sub-01 | 0.216 | 0.0002 | [0.177, 0.238] | 672 |
| sub-02 | 0.143 | 0.0002 | [0.122, 0.172] | 705 |
| sub-03 | 0.111 | 0.0010 | [0.092, 0.149] | 709 |
| sub-04 | 0.232 | 0.0002 | [0.200, 0.263] | 660 |
| sub-05 | 0.297 | 0.0002 | [0.261, 0.315] | 724 |
| sub-06 | 0.122 | 0.0004 | [0.093, 0.147] | 626 |

### A3: FC-Transition Correlation (Mantel Test)

Spearman correlation between symmetrized empirical transition probability and RV-coefficient FC similarity; permutation test.

| Subject | rho | p | n pairs |
|---------|------|--------|---------|
| sub-01 | 0.417 | 0.0002 | 990 |
| sub-02 | 0.483 | 0.0002 | 990 |
| sub-03 | 0.326 | 0.0002 | 946 |
| sub-04 | 0.551 | 0.0002 | 946 |
| sub-05 | 0.393 | 0.0002 | 990 |
| sub-06 | 0.434 | 0.0002 | 861 |

### A3: Network Homophily

Within- vs between-network mean empirical transition probability; permutation test on difference.

| Subject | Within-net mean | Between-net mean | Ratio | Difference | p |
|---------|-----------------|------------------|-------|------------|------|
| sub-01 | 0.01053 | 0.00678 | 1.55 | 0.00375 | 0.0084 |
| sub-02 | 0.01280 | 0.00725 | 1.77 | 0.00555 | 0.0008 |
| sub-03 | 0.01284 | 0.00812 | 1.58 | 0.00472 | 0.0002 |
| sub-04 | 0.01571 | 0.00891 | 1.76 | 0.00680 | 0.0002 |
| sub-05 | 0.00844 | 0.00861 | 0.98 | -0.00017 | 0.511 |
| sub-06 | 0.01975 | 0.00993 | 1.99 | 0.00982 | 0.0016 |

### A4: MFPT Landscape

SCC = largest strongly connected component of model transmat_. MFPT-FC: Spearman correlation between symmetrized MFPT distance and RV-coefficient FC dissimilarity (1 - RV); permutation test.

| Subject | SCC size | MFPT-FC rho | p |
|---------|----------|-------------|--------|
| sub-01 | 46 | 0.588 | 0.0002 |
| sub-02 | 46 | 0.582 | 0.0002 |
| sub-03 | 44 | 0.622 | 0.0002 |
| sub-04 | 44 | 0.680 | 0.0002 |
| sub-05 | 47 | 0.405 | 0.0012 |
| sub-06 | 42 | 0.449 | 0.0002 |

## Outputs

- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/transition_structure_summary.json`: per-subject summary of all A1-A4 metrics
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/graph_metrics.csv`: per-state in/out degree, strength, betweenness centrality
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/community_assignments.json`: Louvain community labels per state
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/selectivity_metrics.csv`: per-state top-1/2 targets and concentration ratio
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/asymmetry_matrix.npy`: pairwise asymmetry indices
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/recurrence_assortativity.json`: assortativity coefficient, p, bootstrap CI
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/fc_transition_correlation.json`: Mantel test rho and p
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/network_homophily.json`: within/between network transition probs and permutation p
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/mfpt_matrix.npy`: mean first passage time matrix (SCC states)
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/stationary_distribution.npy`: stationary distribution from model transmat_
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/mfpt_fc_correlation.json`: MFPT-FC dissimilarity Mantel test
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/transition_graph.graphml`: NetworkX graph for external visualization
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/fig_A1_transition_graph.png/.pdf`
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/fig_A1b_degree_centrality.png/.pdf`
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/fig_A2_asymmetry.png/.pdf`
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/fig_A3_assortativity_panel.png/.pdf`
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/fig_A4a_mfpt_matrix.png/.pdf`
- `output/06b_transition_structure/atlas-4S156Parcels/sub-*/vt0.95/fig_A4b_transition_landscape.png/.pdf`
