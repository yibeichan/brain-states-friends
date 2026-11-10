# Findings: FC similarity vs transitions without the shared FC backbone (`sm_rel_r6_fc_transition_controls`)

## Question

`06b` reports that state pairs with more similar within-state FC transition more often (Mantel, Spearman ρ between the `05f` RV matrix and the symmetrised empirical transition matrix; ρ 0.33–0.55). Raw RV between two correlation matrices is 0.68–1.00 for every pair, because all states share the same grand-mean FC. (1) Does the association hold for the state-specific part of FC? (2) Does it hold after controlling for how many TRs each state has, which sets both FC estimation noise and transition counts, and for how similar the two states' activation maps are?

## Method (as run)

- Gate: the `06b` Mantel ρ is recomputed from the same inputs with a copy of `test_fc_transition_correlation`, and must equal `fc_transition_correlation.json` (`gate`).
- Pairs: upper triangle over active states with defined FC (states below 05f's 30-TR minimum are dropped); 861–990 pairs per participant, the same pairs as `06b`.
- Residual FC similarity: each state's Ledoit-Wolf correlation matrix minus the unweighted mean over those states, compared by unclipped matrix cosine (as the residual RV in `fig_S04_reliability.py`). A second version subtracts the 05f occupancy-weighted grand mean (`state_delta_fc.npy`).
- Covariates per pair: log(TR_i · TR_j), log(min(TR_i, TR_j)) (TR counts from 05f `metadata.json`), and Pearson r between the two states' parcel-space activation maps.
- Mantel: Spearman ρ, two-sided p from 5,000 state-label permutations (floor 0.0002).
- Partial Mantel: ranks of the transition and FC vectors are each regressed on the ranked covariates; ρ is the Pearson correlation of the residuals. The null permutes the state labels of the residualised FC matrix and regresses the covariates out of the permuted values again before correlating (double semi-partialling, Dekker et al. 2007), 5,000 permutations.

## Results

**Mantel, no covariates:**

| sub | n pairs | published raw ρ | raw ρ (current inputs) | residual ρ | residual, occupancy-weighted ρ |
|---|---|---|---|---|---|
| sub-01 | 990 | 0.417 | 0.417 (0.0002) | 0.297 (0.0002) | 0.196 (0.0002) |
| sub-02 | 990 | 0.483 | 0.483 (0.0002) | 0.304 (0.0002) | 0.268 (0.0002) |
| sub-03 | 946 | 0.326 | 0.326 (0.0002) | 0.369 (0.0002) | 0.331 (0.0002) |
| sub-04 | 946 | 0.551 | 0.551 (0.0002) | 0.520 (0.0002) | 0.499 (0.0002) |
| sub-05 | 990 | 0.393 | 0.393 (0.0002) | 0.390 (0.0002) | 0.325 (0.0002) |
| sub-06 | 861 | 0.437 | 0.437 (0.0002) | 0.338 (0.0002) | 0.283 (0.0002) |

**Partial Mantel, controlling for both TR terms and activation-map r** (ρ, p):

| sub | raw RV | residual | residual, occupancy-weighted |
|---|---|---|---|
| sub-01 | 0.156 (0.0002) | 0.160 (0.0002) | 0.135 (0.0002) |
| sub-02 | 0.245 (0.0002) | 0.229 (0.0002) | 0.236 (0.0002) |
| sub-03 | 0.136 (0.0022) | 0.255 (0.0002) | 0.265 (0.0002) |
| sub-04 | 0.403 (0.0002) | 0.415 (0.0002) | 0.438 (0.0002) |
| sub-05 | 0.188 (0.0004) | 0.271 (0.0002) | 0.264 (0.0002) |
| sub-06 | 0.337 (0.0002) | 0.286 (0.0002) | 0.267 (0.0002) |

**Transitions vs each covariate** (Spearman ρ) and the residual FC similarity range:

| sub | log TR product | log min TR | activation-map r | residual FC range |
|---|---|---|---|---|
| sub-01 | 0.34 | 0.39 | 0.39 | -0.82 to 0.87 |
| sub-02 | 0.39 | 0.42 | 0.39 | -0.93 to 0.94 |
| sub-03 | 0.23 | 0.27 | 0.33 | -0.76 to 0.87 |
| sub-04 | 0.35 | 0.41 | 0.25 | -0.92 to 0.94 |
| sub-05 | 0.33 | 0.38 | 0.31 | -0.94 to 0.93 |
| sub-06 | 0.27 | 0.28 | 0.33 | -0.90 to 0.87 |

## Interpretation (bounded)

Removing the shared FC backbone does not remove the association: the residual ρ is 0.30–0.52 (0.20–0.50 with the occupancy-weighted mean), p at the permutation floor in all six participants. Residual FC similarity spans about -0.9 to 0.9, against 0.68–1.00 for raw RV, so the test now has a usable range.

TR counts and activation-map similarity each correlate with transition probability (ρ 0.23–0.42) and account for a large share of the raw association. With all three controls, the residual FC ρ is 0.16–0.42 (p = 0.0002 in all six), 38–78% of each participant's published ρ (sub-01 0.38, sub-02 0.47, sub-06 0.65, sub-05 0.69, sub-04 0.75, sub-03 0.78). Pairs with more similar state-specific FC transition more often than their TR counts and activation-map similarity predict.

## Caveats and audit notes

- The null permutes state labels, as in `06b`. Both FC and transitions come from the same Viterbi assignment, so a shared dependence on segmentation is not removed by any of these controls.
- The covariates enter as ranks in a linear model. A nonlinear dependence on TR count would be only partly removed.
- The gate reproduces the published `06b` ρ from the same 05f files the analysis uses, in all six participants.
- p values of 0.0002 are the floor for 5,000 permutations.

## Connections

- `06b_transition_structure.py` (`test_fc_transition_correlation`), `05f_state_fc.py`, `fig_S04_reliability.py` (residual RV). `sm_rel_r3_assortativity_null` addresses the same occupancy concern for recurrence assortativity.
