# Findings: strength-controlled nulls for recurrence assortativity (`sm_rel_r3_assortativity_null`)

## Question

Does the positive recurrence assortativity reported for the transition graph (Figure 3B; r = 0.111–0.297 across participants, the published point estimates) survive nulls that control for the fact that recurrence is nearly the model's stationary occupancy π and that node strength on the graph is itself a function of occupancy?

## Method (as run)

- **Graph.** Rebuilt exactly as `06b_transition_structure` did for the published test — directed edges i≠j with empirical transition probability P[i,j] ≥ 0.005 (from `06a_state_temp_dynamics/.../transition_probabilities.npy`) over the active states (recurrence > 0), node attribute = recurrence score; the saved `transition_graph.graphml` is the 0.01 model-matrix topology graph and is not used. Per-subject edge counts (`n_edges` in the summary JSON): sub-01 672, sub-02 705, sub-03 709, sub-04 660, sub-05 724, sub-06 626, over 42–47 nodes.
- **Coefficient.** Newman's numeric assortativity (`networkx.numeric_assortativity_coefficient`), gated to the published `recurrence_assortativity.json` point estimate (`gate.assortativity_abs_delta`, tolerance 1e-9; observed |Δ| = 0 in all six).
- **π** recomputed from `best_model.pkl` over the active states with `utils.stationary`, gated to `stationary_distribution.npy` (`gate.stationary_max_abs_delta`; max |Δ| = 0 in all six).
- **Null 1 (primary): strength-stratified.** Recurrence labels permuted within strength quintiles (strength = weighted in-degree + weighted out-degree on the rebuilt empirical graph; 5 bins, sizes 8–10 nodes per subject), 5,000 permutations, `numpy.random.default_rng(0)`.
- **Null 2: π-residualized.** Recurrence rank residualized on rank(π) (OLS with intercept); assortativity of the residual attribute against 5,000 unconditional label permutations of that residual.
- **Also:** assortativity of π itself (r_π, i.e. the same coefficient computed with π in place of recurrence as the node attribute), Spearman(recurrence, π), and Spearman(recurrence, strength) are reported per subject below, and the unconditional (non-stratified) label-permutation null is recomputed alongside the stratified one for comparison. One-sided (greater) empirical p with Phipson–Smyth correction (`utils.stats.permutation_pvalue`); floor 1/5001 ≈ 0.0002.

## Results

**Graph, gates, and observed statistics** (`observed` block of each JSON):

| sub | n_nodes | n_edges | r_recurrence (published) | r_π | r_recurrence \| π (residual) | ρ(recurrence, π) | ρ(recurrence, strength) |
|---|---|---|---|---|---|---|---|
| sub-01 | 46 | 672 | 0.2161 | 0.1756 | 0.0212 | 0.9670 | 0.3282 |
| sub-02 | 46 | 705 | 0.1430 | 0.1470 | 0.0490 | 0.9796 | 0.3682 |
| sub-03 | 44 | 709 | 0.1109 | 0.1051 | -0.0051 | 0.9739 | 0.6175 |
| sub-04 | 44 | 660 | 0.2316 | 0.1769 | 0.0932 | 0.9700 | 0.7188 |
| sub-05 | 47 | 724 | 0.2971 | 0.2673 | -0.0040 | 0.9819 | 0.1291 |
| sub-06 | 42 | 626 | 0.1220 | 0.1056 | -0.0056 | 0.9842 | 0.6170 |

Gate for every row above: `assortativity_abs_delta` = 0.0 and `stationary_max_abs_delta` = 0.0 (tolerance 1e-9), all six subjects.

**Null 1a — unconditional label permutation** (for comparison, `nulls.unconditional_label_permutation`):

| sub | observed | null mean | null sd | Δ | z | p (one-sided, greater) | p (two-sided) | published perm_p_value |
|---|---|---|---|---|---|---|---|---|
| sub-01 | 0.2161 | -0.0228 | 0.0300 | 0.2389 | 7.97 | 0.0002 | 0.0002 | 0.0002 |
| sub-02 | 0.1430 | -0.0240 | 0.0300 | 0.1670 | 5.58 | 0.0004 | 0.0004 | 0.0002 |
| sub-03 | 0.1109 | -0.0229 | 0.0292 | 0.1338 | 4.59 | 0.0002 | 0.0008 | 0.0010 |
| sub-04 | 0.2316 | -0.0239 | 0.0305 | 0.2555 | 8.38 | 0.0002 | 0.0002 | 0.0002 |
| sub-05 | 0.2971 | -0.0230 | 0.0303 | 0.3201 | 10.58 | 0.0002 | 0.0002 | 0.0002 |
| sub-06 | 0.1220 | -0.0237 | 0.0307 | 0.1456 | 4.74 | 0.0002 | 0.0004 | 0.0004 |

The published test (`06b_transition_structure`, `recurrence_assortativity.json` → `perm_p_value`) was two-sided, seed 42; this recompute uses a different seed, so the two-sided values agree with the published ones to the permutation resolution (~1/5000) rather than exactly.

**Null 1b (primary) — strength-stratified label permutation** (`nulls.strength_stratified_label_permutation`):

| sub | observed | null mean | null sd | Δ | z | p (one-sided) | n_bins | bin sizes |
|---|---|---|---|---|---|---|---|---|
| sub-01 | 0.2161 | -0.0219 | 0.0296 | 0.2379 | 8.04 | 0.0002 | 5 | [10, 9, 9, 9, 9] |
| sub-02 | 0.1430 | -0.0377 | 0.0288 | 0.1807 | 6.28 | 0.0002 | 5 | [10, 9, 9, 9, 9] |
| sub-03 | 0.1109 | -0.0164 | 0.0296 | 0.1273 | 4.30 | 0.0004 | 5 | [9, 9, 9, 9, 8] |
| sub-04 | 0.2316 | -0.0256 | 0.0303 | 0.2572 | 8.50 | 0.0002 | 5 | [9, 9, 9, 9, 8] |
| sub-05 | 0.2971 | -0.0257 | 0.0285 | 0.3228 | 11.34 | 0.0002 | 5 | [10, 10, 9, 9, 9] |
| sub-06 | 0.1220 | -0.0222 | 0.0283 | 0.1442 | 5.10 | 0.0002 | 5 | [9, 9, 8, 8, 8] |

**Null 2 — π-residualized attribute** (`nulls.residualized_attribute_label_permutation`; "observed" here is the residual assortativity r_recurrence\|π from the table above):

| sub | observed (resid.) | null mean | null sd | Δ | z | p (one-sided) | p (two-sided) |
|---|---|---|---|---|---|---|---|
| sub-01 | 0.0212 | -0.0226 | 0.0290 | 0.0438 | 1.51 | 0.0666 | 0.6033 |
| sub-02 | 0.0490 | -0.0247 | 0.0295 | 0.0737 | 2.50 | 0.0134 | 0.2112 |
| sub-03 | -0.0051 | -0.0234 | 0.0280 | 0.0183 | 0.65 | 0.2432 | 0.9066 |
| sub-04 | 0.0932 | -0.0240 | 0.0300 | 0.1172 | 3.91 | 0.0018 | 0.0056 |
| sub-05 | -0.0040 | -0.0236 | 0.0293 | 0.0196 | 0.67 | 0.2318 | 0.9312 |
| sub-06 | -0.0056 | -0.0233 | 0.0306 | 0.0177 | 0.58 | 0.2613 | 0.9030 |

5,000 permutations, `default_rng(0)`, in all subjects and all three nulls.

**Per-outcome summary.** Every participant's published assortativity clears both the unconditional null and the strength-stratified null at the empirical p floor (0.0002–0.0004): re-pairing recurrence labels within strength-matched bins does not remove the effect, so it is not merely an artifact of high-strength nodes carrying high recurrence. The stratified null's mean and sd are close to the unconditional null's in every subject (e.g. sub-01: -0.0219 vs -0.0228), so stratifying by strength barely moves the null here — strength alone is not doing much of the work either way. Against Null 2, only two participants have a one-sided p below 0.05 once π is regressed out of recurrence: sub-02 (r_resid = 0.0490, p = 0.0134) and sub-04 (r_resid = 0.0932, p = 0.0018). The other four (sub-01, sub-03, sub-05, sub-06) do not survive Null 2 (p = 0.067–0.261), and three of those four have a residual r near zero or slightly negative (between -0.0056 and -0.0040: sub-03 -0.0051, sub-05 -0.0040, sub-06 -0.0056). r_π sits close to r_recurrence in every subject (76–103% as large — sub-02's r_π of 0.1470 slightly exceeds its r_recurrence of 0.1430; e.g. sub-01: 0.1756/0.2161 = 81%; sub-05: 0.2673/0.2971 = 90%), consistent with Spearman(recurrence, π) = 0.967–0.984 across subjects.

## Interpretation (bounded)

The reported recurrence assortativity is largely carried by stationary occupancy π (Spearman(recurrence, π) ≈ 0.97–0.98 per subject, this JSON's `observed.rho_recurrence_pi`). Organization beyond node strength is detectable: the strength-stratified null is survived at the empirical floor by all six participants (sub-01 through sub-06), so the assortativity is not explained away by strength-matched relabeling. But once π is regressed out of recurrence, little assortativity remains in most participants — only sub-02 (p = 0.0134) and sub-04 (p = 0.0018) still show a residual effect distinguishable from the π-residualized null, and even there the residual coefficients are small (0.049 and 0.093) relative to the published r (0.143 and 0.232). For sub-01, sub-03, sub-05, and sub-06 the residual assortativity is not distinguishable from its null (p = 0.067–0.261). Each null controls for a different confound — Null 1 controls for a node's raw connectivity strength, Null 2 controls for the node's occupancy under the fitted model — and they give different answers per participant; neither reading ("organization beyond strength" vs "carried by occupancy") is uniformly correct across all six, so this is reported per participant rather than pooled.

## Caveats and audit notes

- The graph used here is the rebuilt empirical graph matching the published test, not the saved `transition_graph.graphml` (that file is a different, 0.01-threshold, model-matrix topology graph and does not reproduce `recurrence_assortativity.json` — see Method above); strength is measured on this rebuilt graph, not on the full unthresholded empirical transition matrix.
- Quintile binning with 42–47 nodes gives 8–10 nodes per bin; permutations within bins are far fewer than 5,000 distinct arrangements for small bins, so p is bounded by the floor (1/5001 ≈ 0.0002), not by bin combinatorics.
- Residualization is linear in ranks; a nonlinear dependence of recurrence on π would leave structure in the residual that this test would misattribute to "organization beyond occupancy."

## Connections

- Main analysis: `06b_transition_structure.py` (A3). Occupancy confound numbers: `sm_rel_r1_recurrence_robustness`. Manuscript: Results §Transitions, Methods §State transition structure, Figure 3B.
