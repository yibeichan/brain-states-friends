# Findings: 05b Recurring Brain States Visualization

_Script: `script/05b_visualize_recurring_states.py`. Tier: MAIN (Fig 2C / Fig S1)._

_Renders top recurring brain states (by recurrence score from 05a) as cortical and subcortical surface maps; per-subject (n=6), atlas-4S156Parcels, vt=0.95._

## Method (as run)

- **Input**: `state_means_parcel.npy` (back-projected state centroids, shape n_states x 156) from script 04 select; `recurrence_summary.json` and `fractional_occupancy.pkl` from script 05a.
- **Parcellation**: atlas-4S156Parcels (100 Schaefer cortical + 56 subcortical composite: CIT168, HCP thalamus, hippocampus/amygdala, cerebellum).
- **PCA variance threshold**: vt=0.95 (67-77 PCs per subject).
- **States visualized**: top 5 in multi-panel figure; top 10 in individual plots. Active states are those with recurrence score > 0 (all 42-47 active states qualify; top-N are selected by rank).
- **Color range**: symmetric, set to the 95th percentile of absolute z-scores pooled across the top-N states; applied identically to cortical and subcortical panels.
- **Rendering**: yabplot with headless PyVista/OSMesa backend. Cortical: Schaefer-100 on fsaverage5 surface. Subcortical: CIT168+HCP anatomically-defined 3D VTK meshes. Hippocampus and amygdala carry approximately 50-70% lower tSNR due to susceptibility artifacts.
- **No group statistic or cross-subject aggregation.** Each subject's states are visualized independently.
- **Eligibility filtering**: not applied in 05b; all active states (score > 0) are eligible. State flags from 05e are not consumed here.
- **Sub-04 note**: 194 runs (seasons 1-4 only; seasons 5-6 excluded for this subject).

## Results

### Active state counts and run totals per subject

| Subject | Active states (score > 0) | Total runs |
|---|---|---|
| sub-01 | 46 | 292 |
| sub-02 | 46 | 292 |
| sub-03 | 44 | 291 |
| sub-04 | 44 | 194 |
| sub-05 | 47 | 289 |
| sub-06 | 42 | 292 |

### Top-5 states visualized in multi-panel figure (by recurrence score)

| Subject | Rank 1 (state ID, score) | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
|---|---|---|---|---|---|
| sub-01 | state28, 0.849 | state23, 0.836 | state37, 0.798 | state3, 0.750 | state12, 0.726 |
| sub-02 | state25, 0.836 | state6, 0.812 | state30, 0.774 | state38, 0.771 | state18, 0.760 |
| sub-03 | state30, 0.852 | state19, 0.835 | state21, 0.828 | state6, 0.821 | state8, 0.811 |
| sub-04 | state43, 0.902 | state41, 0.845 | state22, 0.820 | state24, 0.820 | state4, 0.809 |
| sub-05 | state14, 0.817 | state6, 0.785 | state26, 0.727 | state32, 0.706 | state35, 0.706 |
| sub-06 | state8, 0.925 | state18, 0.914 | state6, 0.887 | state38, 0.887 | state23, 0.870 |

### Top-10 states in individual plots (run spread shown)

| Subject | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | Rank 6 | Rank 7 | Rank 8 | Rank 9 | Rank 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| sub-01 | state28 (248) | state23 (244) | state37 (233) | state3 (219) | state12 (212) | state15 (212) | state0 (202) | state4 (196) | state13 (187) | state32 (187) |
| sub-02 | state25 (244) | state6 (237) | state30 (226) | state38 (225) | state18 (222) | state42 (196) | state0 (195) | state17 (193) | state11 (192) | state24 (189) |
| sub-03 | state30 (248) | state19 (243) | state21 (241) | state6 (239) | state8 (236) | state2 (234) | state7 (226) | state0 (223) | state9 (199) | state15 (198) |
| sub-04 | state43 (175) | state41 (164) | state22 (159) | state24 (159) | state4 (157) | state9 (156) | state35 (154) | state34 (142) | state0 (138) | state31 (137) |
| sub-05 | state14 (236) | state6 (227) | state26 (210) | state32 (204) | state35 (204) | state5 (201) | state12 (196) | state19 (190) | state33 (187) | state2 (185) |
| sub-06 | state8 (270) | state18 (267) | state6 (259) | state38 (259) | state23 (254) | state42 (253) | state3 (248) | state9 (233) | state35 (223) | state0 (219) |

Run spread = number of runs in which the state was active (recurrence score x n_runs, rounded).

### Output file counts per subject

| Subject | Multi-panel PNG/PDF | Individual state PNGs | Total files |
|---|---|---|---|
| sub-01 | 2 | 10 | 12 |
| sub-02 | 2 | 10 | 12 |
| sub-03 | 2 | 10 | 12 |
| sub-04 | 2 | 10 | 12 |
| sub-05 | 2 | 10 | 12 |
| sub-06 | 2 | 10 | 12 |

## Outputs

- `output/05b_recurring_states_visualization/atlas-4S156Parcels/sub-*/vt0.95/{sub_id}_top5_recurring_states.png` - multi-panel figure (5 states x 4 columns: cortical, subcortical, metrics, top runs)
- `output/05b_recurring_states_visualization/atlas-4S156Parcels/sub-*/vt0.95/{sub_id}_top5_recurring_states.pdf` - PDF version of multi-panel figure
- `output/05b_recurring_states_visualization/atlas-4S156Parcels/sub-*/vt0.95/individual_states/recurring_state_{rank:02d}_state{id}.png` - individual cortical+subcortical plots for top 10 states per subject
