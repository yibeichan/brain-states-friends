# Findings: ICA convergent-validity supplement (`sm_alt_ica_states`)

Findings for the ICA-vs-HMM convergent-validity supplement. Code: `script/sm_alt_ica_states.py` (+`.sh`), `script/sm_alt_ica_category_table.py`, `script/sm_alt_ica_focal_components.py`; companion figure `script/fig_sm_alt_ica_matching.py`.

## Question

Do the recurring co-activation patterns the sticky-HDP-HMM discovers re-emerge under a decomposition with orthogonal assumptions (spatial independence, no temporal model)? ICA is run as a standalone parallel state-discovery method; convergence on the **spatial repertoire** would argue the HMM states are not artifacts of its Markov machinery. Inference is strictly per-subject (all 6); any 6-subject display is descriptive, no group statistic.

## Method (as run)

- **Inputs, frozen:** the same vt=0.95 PCA subspace and combined HMM the main analysis uses. HMM state-mean maps are `means_[:, :n_pcs] @ components` (state-contrast, no global mean).
- **ICA subspace:** at dimensionality K, ICA is fit on the loadings of the top K PCs (`components_[:K].T`, parcels as samples), whitened with the symmetric inverse square root of their covariance, then rotated by FastICA (`whiten=False`). This is the standard two-step spatial ICA: reduce to the leading principal components, then find the rotation that maximizes spatial independence.
  - Why not let FastICA reduce all n_pcs loadings to K itself: the loadings have orthonormal rows, so after centring all but one of their singular values equal 1. FastICA's PCA step would then keep an arbitrary basis of that tied eigenspace, which is neither the top-K PCs nor stable across BLAS kernels.
  - Why the top-K PCs: the HMM is fit on variance-weighted PC scores, and its state means concentrate in the leading PCs (for sub-01, 95% of state-mean energy lies in the top 15 PCs). The top-K subspace is where the two decompositions can be compared.
  - Why symmetric whitening: the inverse square root of a positive-definite matrix is unique, so the whitened data do not depend on which eigenbasis the solver returns.
- **ICASSO consensus:** FastICA logcosh, 25 restarts, consensus maps + I_q.
- **Dimensionality sweep:** K ∈ {15, 25, 35} (low-dim) + K_used + a K=41–47 sensitivity grid spanning the active-state counts (42–47); n_components ≤ n_pcs for all. K_used is the fitted model's occupied-state count (posterior usage above 0.01 at the final EM iteration; 37–42 per subject), stored as `K_active` in the output JSONs. It is distinct from the active-state count (fractional occupancy above 0.02 in at least one run; 42–47 per subject). The full sweep is reported as a circularity control; no single K is privileged.
- **HMM state sets:** content-eligible states, and "all" = the active repertoire (fractional occupancy above 0.02 in at least one run, recurrence > 0 in 05a), the same set used throughout the analysis.
- **Tier 1 (spatial):** Hungarian match on |r|; per-pair significance by per-rank comparison with a null in which the ICA maps are replaced by random rotations of the **same centred, whitened top-K basis that ICA rotates** (Hungarian re-run per draw, 1000 draws), BH-FDR within each state set. A whole-set test compares the observed median matched |r| with the medians of the same null draws (`median_excess`, `median_p`). Because every null draw spans the same subspace as the ICA maps, the test asks whether ICA's particular rotation recovers HMM maps beyond an arbitrary rotation of that subspace.
  - A secondary null draws random K-frames over the full n_pcs subspace, centred and whitened the same way (`spatial_p_full`, `spatial_q_full`). It is not a test of the ICA rotation alone: HMM maps sit in the leading PCs, so any map set confined to the top-K subspace has an advantage over frames spread across all PCs.
  - `hmm_topk_capture` records, for each matched HMM state, the share of its PC-space mean energy inside the top K PCs. It bounds how well any map in the ICA subspace can match that state.
- **Tier 2 (temporal):** Spearman(HMM posterior γ_k, sign-aligned ICA time course) per matched pair; within-run circular-shift null (1000); low-occupancy states excluded. p-values are **conditional on the Tier-1 spatial match**.
- **Per-category breakdown** (`sm_alt_ica_category_table.py`): the K_used "all"-set matched pairs labeled by HMM-state `summary_category` (05e state_flags).

## Results

### Tier 1: spatial convergence

FDR-surviving content-eligible matched pairs / total, against the same-subspace rotation null:

| sub | K_used | K15 | K25 | K35 | K41 | K42 | K43 | K44 | K45 | K46 | K47 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 42 | 8/15 | 14/25 | 18/31 | 20/31 | 19/31 | 18/31 | 17/31 | 22/31 | 23/31 | 21/31 |
| 02 | 42 | 9/15 | 14/25 | 18/30 | 23/30 | 20/30 | 24/30 | 24/30 | 21/30 | 24/30 | 21/30 |
| 03 | 42 | 2/15 | 13/25 | 23/26 | 21/26 | 23/26 | 21/26 | 23/26 | 23/26 | 23/26 | 21/26 |
| 04 | 41 | 8/15 | 12/25 | 15/27 | 20/27 | 19/27 | 21/27 | 19/27 | 19/27 | 22/27 | 17/27 |
| 05 | 41 | 10/15 | 16/25 | 15/29 | 23/29 | 21/29 | 23/29 | 23/29 | 24/29 | 24/29 | 25/29 |
| 06 | 37 | 2/15 | 7/16 | 14/16 | 14/16 | 15/16 | 14/16 | 14/16 | 14/16 | 14/16 | 14/16 |

Median matched |r| minus the mean of the null draws' medians (content-eligible pairs); * marks a whole-set test p < 0.05:

| sub | K_used | K15 | K25 | K35 | K41 | K42 | K43 | K44 | K45 | K46 | K47 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 42 | +0.07* | +0.06* | +0.06* | +0.06* | +0.05* | +0.05* | +0.02* | +0.04* | +0.08* | +0.06* |
| 02 | 42 | +0.08* | +0.06* | +0.06* | +0.08* | +0.08* | +0.06* | +0.07* | +0.07* | +0.07* | +0.08* |
| 03 | 42 | +0.03 | +0.07* | +0.04* | +0.06* | +0.04* | +0.05* | +0.05* | +0.04* | +0.05* | +0.05* |
| 04 | 41 | +0.05* | +0.02 | +0.07* | +0.07* | +0.05* | +0.07* | +0.07* | +0.09* | +0.07* | +0.06* |
| 05 | 41 | +0.11* | +0.05* | +0.02* | +0.04* | +0.03* | +0.04* | +0.04* | +0.05* | +0.04* | +0.03* |
| 06 | 37 | +0.04 | +0.02 | +0.05* | +0.06* | +0.07* | +0.07* | +0.08* | +0.06* | +0.07* | +0.08* |

- Most of the raw agreement comes from the subspace the two methods share. Random K-frames spread over all n_pcs PCs reach a mean matched |r| of 0.21–0.28; random rotations of the top-K basis reach 0.46–0.51 at K = 15 and 0.34–0.36 at K = 42. HMM state means lie almost entirely in the leading PCs (`hmm_topk_capture` median 0.93–0.96 at K = 15 and 0.99–1.00 at K = 42), so any map set confined to them matches the HMM well.
- ICA's rotation matches the HMM better than the same-subspace null in every subject and every K (median excess +0.02 to +0.11). The whole-set test is significant in 56 of 60 subject × K cells. The four exceptions are at low K: sub-03 and sub-06 at K = 15 (p = 0.15 and 0.08), and sub-04 and sub-06 at K = 25 (p = 0.13 and 0.14). From K = 35 upward every cell is significant.
- At K = 15, which is set independently of the HMM, the convergence is significant in four of six subjects.
- In the median excess the agreement does not grow as K approaches the HMM's state count. Raw |r| and the null both fall with K (median matched |r| 0.50–0.62 at K = 15 and 0.36–0.43 at K = 42), because random maps in a higher-dimensional space correlate less. The per-rank surviving fraction does rise with K in the two subjects with weak low-K convergence (sub-03 from 0.13 at K = 15 to 0.88 at K_used, sub-06 from 0.13 to 0.81).
- The surviving counts are per rank: the i-th strongest observed match is compared with the i-th strongest match of each null draw. Neighbouring ranks are correlated, so the counts are not the number of individually validated states; they should be read together with the excess and the per-rank pattern below.
- Individual maps are not recovered one to one. Per-subject median matched |r| is 0.35–0.62 across K, and single pairs range from near 0 to 0.93.
- The secondary full-subspace null passes 68–100% of pairs per subject and K. As described under Method, it does not isolate the ICA rotation.

### Focal ICA components and the per-rank pattern

At K_used, 8–11 matched pairs per subject fall below |r| = 0.15 (`sm_alt_ica_focal_components.py`, `focal_components.json`, over the full active set of 42–47 states):

- 7–11 of the 37–42 ICA components (17–26%) are focal: their best |r| to any HMM state is below 0.15, about 99% of their map energy lies in three parcels (median 0.55–0.64 for the other components), and their median kurtosis is 148–149 (21–29 for the others). They are the most stable components across ICASSO restarts (median I_q ≥ 0.998, against 0.94–0.99 for the others).
- No HMM state lacks a counterpart. Every state's best |r| to any ICA map is at least 0.30 (subject minima 0.30–0.37). The near-zero pairs arise because the one-to-one assignment gives the focal components to the states left over once the better matches are taken.
- Rank by rank, 24–30 of the observed matches exceed the 97.5th percentile of the rotation null at the same rank, and 9–15 (the focal pairs at the bottom ranks) fall below its 2.5th percentile.
- At K = 15 only 1–3 pairs per subject fall below 0.15.

ICA therefore spends part of its dimensions on focal, near-single-parcel sources that the HMM does not represent as states, and the remaining components align with HMM state maps clearly above chance. The median excess in the table above mixes the two groups and understates the agreement of the non-focal components.

### Per-category breakdown

Per-category breakdown (`category_correspondence_table.{csv,md}`, at each subject's K_used, over the active set): spatial mean r is 0.31–0.42 across subjects for content-eligible states and 0.24–0.42 for low-confidence states; categories with one or two states per subject are descriptive only. Run-onset states match somewhat better than content-eligible states in five of six subjects (mean r 0.40–0.50 vs 0.31–0.42), consistent with high-variance, run-locked patterns sitting in the leading PCs. Because the taxonomy is a *temporal* classification, whether a state's mean map is recoverable in the shared subspace is largely independent of its label.

**Caveat:** matches to low-confidence, unused, or rare states indicate recoverability of those states' mean maps within the shared subspace, not independent biological or content validation. The state quality flags stand. The convergent-validity signal rests on the content-eligible (and, more weakly, run-onset) repertoire.

### Tier 2: temporal correspondence

At K = 15, 35 and K_used, every tested content-eligible matched pair survives the within-run shift null in every subject except one pair (sub-05 at K = 35, 28/29). The shift null is weaker than the subspace-rotation null and these p-values are conditional on the Tier-1 spatial selection, so this is complementary, not independent, confirmation.

## Conclusion

ICA and the HMM share the dominant PC subspace, and most of their spatial agreement follows from that alone. Within the subspace, ICA's independence-maximizing axes align with HMM state-mean maps better than random rotations of the same subspace: the whole-set test is significant in 56 of 60 subject × dimensionality cells, in every subject from K = 35 upward, and at K = 15 (which does not depend on the HMM) in four of six subjects. In most subjects the agreement does not depend on matching the HMM's state count; in sub-03 and sub-06 it is weak at low K and strengthens with K. At the HMM's own dimensionality, about a quarter of the ICA components are focal, near-single-parcel sources with no HMM counterpart; the rest match HMM states clearly above chance, and every HMM state has a moderately similar ICA map, but the one-to-one correspondence is incomplete. A decomposition with no temporal model therefore converges on the HMM's spatial axes beyond chance, but does not reproduce its repertoire state by state.

An earlier version of this analysis let FastICA reduce all n_pcs loadings to K itself. Because the loadings' singular values are tied, that kept an arbitrary, BLAS-dependent K-dimensional subspace whose overlap with the top-K PCs was at chance level. Its reported pattern (null at coarse K, convergence only near the HMM's state count, none in sub-03) reflected how much of the leading PCs that arbitrary subspace happened to cover, and is superseded by the results above.

## Outputs

`output/sm_ica_states_topk/atlas-4S156Parcels/{sub-*/ica_match_summary.json, ica_maps_K*.npy, ica_timecourses_K*.npy, category_correspondence_table.{csv,md}, focal_components.json, diagnostics/}` (directory name: `ICA_STATES_DIR` in `script/utils/ica_states.py`). The earlier `output/sm_ica_states/` outputs come from the superseded truncated whitening and are not read by any script.

## Companion figure

`script/fig_sm_alt_ica_matching.py`. Panel A: K-sweep heatmap, 6 subjects × K; colour = `median_excess` (median matched |r| above the rotation null, content-eligible pairs), cell text = FDR-surviving/total pairs, dagger = whole-set test p ≥ 0.05. Panel B: per-state matched |r| at K_used, x = subject, colour = taxonomy category, solid tick = per-subject median, dashed tick = rotation-null mean.
