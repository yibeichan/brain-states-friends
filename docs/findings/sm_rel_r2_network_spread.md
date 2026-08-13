# Findings: R2 network-spread null (`sm_rel_r2_network_spread`)

Null for main-analysis Results R2's multi-network claim. Code: `script/sm_rel_r2_network_spread.py` (+`.sh` runner); tests `script/tests/test_rel_r2_network_spread.py`.

## Question

R2 reports that content-eligible state maps mix several canonical networks (pooled medians across 159 states: top-1 network share 0.25, top-3 share 0.60, 4 networks >= 10%, normalized network entropy 0.81). Every state mean, however, is a vector in the participant's retained PCA subspace, and the PCA basis itself mixes networks. Could the multi-network spread be inherited from the representation, i.e. would an *arbitrary* direction in the same subspace look just as multi-network?

## Method (as run)

- **Statistic.** The same per-state network-participation metrics as R2 (mean |loading| per network, normalized to shares; top-1 share, top-3 share, count of networks >= 10%, entropy normalized by log of the 13-network total), summarized as the per-participant median over content-eligible states, plus the pooled 159-state median.
- **Faithfulness gates (hard aborts, before any null draw).** (1) Back-projection: `state_means_pca @ pca.components_[:n_pcs] + pca.mean_` must reproduce `state_means_parcel.npy` to 1e-8 (observed max error <= 3.3e-16 in all six subjects), so the null's subspace-to-parcel map is exactly the pipeline's. (2) Published medians: the pooled content-eligible metrics must reproduce the manuscript's R2 values after rounding (n = 159; 0.25 / 0.60 / 4 / 0.81) — they do.
- **Null draws.** Per subject, 10,000 random directions in the retained PC subspace, back-projected through the same PCA transform. Two variants: **variance-matched** (primary; component i scaled by sqrt(explained_variance_i), i.e. a random pattern with the training data's second-moment structure) and **isotropic** (robustness; all retained components weighted equally).
- **Inference.** Observed per-participant median vs a null distribution of medians (10,000 groups of K_eligible per-draw values resampled from the subject's null pool). Two-sided empirical p by doubled min-tail with the (count + 1) / (n + 1) correction; floor 0.0002 at 10,000 groups. z is descriptive. Deterministic seeds (documented in the module docstring).

## Results (10,000 draws, both gates passed)

Fitted states are **more network-concentrated** than arbitrary directions in the same subspace, in every subject and under both null variants. Observed entropy medians sit far **below** the null; observed top-1 shares sit far **above** it.

| sub | K_elig | obs entropy | vm null median [95% CI] | vm z | iso null median | iso z | obs top1 | vm null top1 (z) |
|---|---|---|---|---|---|---|---|---|
| sub-01 | 31 | 0.793 | 0.928 [0.918, 0.937] | -28.1 | 0.954 | -44.7 | 0.256 | 0.167 (+13.8) |
| sub-02 | 30 | 0.818 | 0.933 [0.925, 0.942] | -26.7 | 0.954 | -38.9 | 0.249 | 0.163 (+14.5) |
| sub-03 | 26 | 0.801 | 0.925 [0.914, 0.935] | -23.7 | 0.957 | -39.4 | 0.243 | 0.171 (+10.4) |
| sub-04 | 27 | 0.788 | 0.921 [0.911, 0.931] | -25.4 | 0.949 | -43.2 | 0.253 | 0.172 (+11.2) |
| sub-05 | 29 | 0.825 | 0.933 [0.923, 0.942] | -21.7 | 0.958 | -36.4 | 0.241 | 0.164 (+12.1) |
| sub-06 | 16 | 0.806 | 0.932 [0.919, 0.944] | -19.7 | 0.956 | -31.6 | 0.230 | 0.164 (+8.0) |

All p at the 0.0002 floor (two-sided, 10,000 groups), every subject, both variants, for both entropy and top-1 share. Pooled (159 states): observed entropy 0.807 vs variance-matched null 0.929 [0.924, 0.933], z = -57.5, p = 0.0002.

## Reading

The multi-network composition of content-eligible states is **not** inherited from the PCA basis. Arbitrary directions in the retained subspace are far *more* uniformly spread across networks (entropy ~0.93-0.96) than fitted states (~0.79-0.83). Fitted states occupy a middle ground: substantially more network-concentrated than generic subspace patterns, yet not confined to single networks (top-1 share only ~0.25). Both halves of R2's claim survive, and the null adds a positive statement: the model's states carry network structure beyond what the representation supplies.

## Caveats and audit notes

- **The null is about the representation, not the brain.** It tests whether the *basis* forces multi-network maps. It does not test whether network labels are the right frame (R2 already treats them as an annotation frame only).
- **Variance-matched vs isotropic differ in the expected direction.** Isotropic nulls are more uniform (higher entropy) because they weight low-variance, more localized components equally; variance-matched nulls concentrate on high-variance global components. Observed states sit below both.
- **Entropy floor.** The 13-network normalized entropy of a map can be low only if network means differ strongly; with only 156 parcels and networks of 4-30 parcels, sampling noise alone keeps entropy well above 0 for smooth maps. The comparison is therefore always against the null, never against the theoretical [0, 1] range.
- **Metric mirror, not import.** The metric functions mirror main-branch `utils/network_participation.py` rather than importing it (supplements is an orphan branch); the published-medians gate is what guarantees the mirror is faithful.

## Manuscript integration

- Results R2: one sentence citing the null (observed spread below random-direction spread, all subjects, p = 0.0002 floor).
- Methods "State classification" or "Network participation": two sentences describing the null construction.
- Supplementary: table + short section (mirrors the R5 phase-null supplement pattern).

## Connections

- Main analysis: Results R2, Figure 2C; `utils/network_participation.py` (main branch).
- `docs/findings/sm_rel_r5_phase_null.md` — same gate-then-null design pattern (faithfulness gate before surrogate inference).
