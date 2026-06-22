# Findings: ICA convergent-validity supplement (`sm_alt_ica_states`)

Findings for the ICA-vs-HMM convergent-validity supplement. Code: `script/sm_alt_ica_states.py` (+`.sh`), `script/sm_alt_ica_category_table.py`; companion figure `script/fig_sm_alt_ica_matching.py`.

## Question

Do the recurring co-activation patterns the sticky-HDP-HMM discovers re-emerge under a decomposition with orthogonal assumptions (spatial independence, no temporal model)? ICA is run as a standalone parallel state-discovery method; convergence on the **spatial repertoire** would argue the HMM states are not artifacts of its Markov machinery. Inference is strictly per-subject (all 6); any 6-subject display is descriptive, no group statistic.

## Method (as run)

- **Inputs, frozen:** the same vt=0.95 PCA subspace and combined HMM the main analysis uses. ICA is fit on the PCA loadings (`components_[:n_pcs].T`, parcels as samples) → spatially independent maps; HMM state-mean maps are `means_[:, :n_pcs] @ components` (state-contrast, no global mean). All three map sets (ICA, HMM, null) live in the **same n_pcs-dim PC subspace**.
- **ICASSO consensus:** FastICA logcosh, 25 restarts, consensus maps + I_q.
- **Dimensionality sweep:** K ∈ {15, 25, 35} (low-dim) + K_active + a K=41–47 sensitivity grid (matches each subject's K_active range 37–42; n_components ≤ n_pcs for all). The full sweep is reported as a circularity control; no single K is privileged.
- **Tier 1 (spatial):** Hungarian match on |r|; per-pair significance vs a **subspace-rotation null** (random orthonormal K-frames in the same whitened subspace, Hungarian re-run per draw, 1000 draws), BH-FDR within the subject's full active-set family. This null controls the shared-subspace confound directly and is deliberately conservative.
- **Tier 2 (temporal):** Spearman(HMM posterior γ_k, sign-aligned ICA time course) per matched pair; within-run circular-shift null (1000); low-occupancy states excluded. p-values are **conditional on the Tier-1 spatial match**.
- **Per-category breakdown** (`sm_alt_ica_category_table.py`): the K_active "all"-set matched pairs labeled by HMM-state `summary_category` (05e state_flags).

## Results

### Tier 1: spatial convergence

FDR-surviving content-eligible matched pairs / total, vs the subspace-rotation null:

| sub | K_act | K15 | K25 | K35 | K41 | K42 | K43 | K44 | K45 | K46 | K47 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 42 | 0 | 0 | 8/31 | 19 | 15 | 12 | 18 | 25 | 18 | 24 |
| 02 | 42 | 0 | 8/25 | 5/30 | 12 | 8 | 10 | 10 | 7 | 13 | 9 |
| 03 | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 04 | 41 | 0 | 0 | 7/27 | 2 | 3 | 4 | 2 | 10 | 2 | 2 |
| 05 | 41 | 0 | 0 | 18/29 | 18 | 28 | 29/29 | 29/29 | 25 | 28 | 24 |
| 06 | 37 | 0 | 1/16 | 11/16 | 7 | 7 | 9 | 14/16 | 5 | 10 | 13 |

- At the low-dim settings, ICA matches the HMM no better than a random rotation of the shared subspace: K=15 is null for all 6 subjects; K=25 is null for four subjects, with sub-02 at 8/25 and a single matched pair in sub-06.
- Surviving pairs appear at K=35 and increase toward K≈K_active (41–47): sub-01 and sub-05 have the most, sub-02 and sub-06 fewer, sub-04 few, and sub-03 none at any K. Because K≈K_active sets the ICA dimensionality equal to the HMM's, those columns are reported as a sensitivity analysis rather than the primary comparison.
- Mechanism: a random orthonormal rotation of the vt=0.95 PC subspace already matches the HMM state-mean maps about as well as ICA's independence-maximizing rotation; the agreement that exceeds the null is what counts. Verified not a deflation bug (z-scoring removes global mean; consensus maps used; all maps share the subspace; sub-05 K=35 matched r reproduced to 1e-16).

### Per-category breakdown

Per-category breakdown (`category_correspondence_table.{csv,md}`, at each subject's K_active): within each subject, spatial mean r and temporal correspondence are comparable across content-eligible, run-onset, season, and low-confidence states; the dominant axis is the **subject**, not the category. Only "Unused" (lowest-occupancy) states match poorly in some subjects (sub-01/02). Because the taxonomy is a *temporal* classification, this independence is expected: whether a state's mean map is recoverable in the shared subspace is largely orthogonal to its temporal label.

**Caveat:** matches to low-confidence, unused, or rare states indicate recoverability of those states' mean maps within the shared subspace, not independent biological or content validation. The state quality flags stand. The convergent-validity signal rests on the content-eligible (and, more weakly, run-onset) repertoire.

### Tier 2: temporal correspondence

Nearly every tested matched pair survives the within-run shift null across all categories and all subjects, including the spatial non-convergers (sub-03: content 20/21, run-onset 5/5). But the shift null is weaker than the subspace-rotation null and these p-values are conditional on the Tier-1 spatial selection, so this is complementary, not independent, confirmation.

## Conclusion

Spatial convergence between ICA and the HMM depends on granularity: it is null at coarse, HMM-independent K and appears only as the ICA dimensionality approaches the HMM's K_active. It varies by subject (most surviving pairs in sub-01 and sub-05, none in sub-03) and is not specific to content-eligible states (artifact-category matches reflect subspace recoverability, not validation). Temporal correspondence is broad but rests on a weaker null and is conditional on the spatial match. An independent decomposition does not cleanly rediscover the HMM repertoire.

## Outputs

`output/sm_ica_states/atlas-4S156Parcels/{sub-*/ica_match_summary.json, ica_maps_K*.npy, ica_timecourses_K*.npy, category_correspondence_table.{csv,md}}`.

## Companion figure

`script/fig_sm_alt_ica_matching.py`. Panel A: K-sweep convergence heatmap (FDR-surviving eligible fraction, 6 subjects x K). Panel B: per-state matched |r| at K_active, x = subject, colour = taxonomy category, per-subject median tick. Panel B is a per-state strip showing the full per-subject distribution and category mix without collapsing to means.
