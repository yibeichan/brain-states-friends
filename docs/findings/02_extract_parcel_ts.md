# Findings: 02 Extract Parcel Time Series

_Script: `script/02_extract_parcel_ts.py`. Tier: MAIN (Methods)._

_Extracts parcel-level average BOLD activation from cleaned CIFTI files; one (TRs x 157) array per run, per subject (n=6)._

## Method (as run)

- Parcellation: atlas-4S156Parcels (Schaefer 100 cortical + 56 subcortical composite; 156 parcels, IDs 1-156; column 0 is background and is always zero)
- Input: grayordinate-wise z-scored cleaned CIFTI dtseries files from script 00 (`output/00_postproc/{sub_id}/`)
- Extraction: mean across all grayordinates within each parcel at each TR; output indexed by parcel ID (not sequential) so column 0 = background, columns 1-156 = valid parcels
- Episodes per subject: includes Friends (s01-s06), Movie10 (Bourne, Wolf, Life, Figures), Harry Potter, and Petit Prince (lppFR, lppEN); Friends episodes are the primary analysis input for downstream scripts 03a onward
- sub-04 has fewer runs than other subjects (259 vs ~377-379 for the others)
- Validation parcellations (4S256, 4S356, 4S456) are supported by the script but have not been extracted to disk

## Results

### Run Counts per Subject

| Subject | Total runs | Friends episodes | Other-stimulus runs |
|---|---|---|---|
| sub-01 | 378 | 292 | 86 |
| sub-02 | 379 | 293 | 86 |
| sub-03 | 377 | 291 | 86 |
| sub-04 | 259 | 198 | 61 |
| sub-05 | 375 | 289 | 86 |
| sub-06 | 377 | 293 | 84 |

### Output Array Dimensions per Subject (atlas-4S156Parcels)

Array shape per file: (n_TRs, 157) - column 0 is background (all zeros), columns 1-156 are valid parcels.

| Subject | n_runs | TR min | TR max | TR mean | Total TRs |
|---|---|---|---|---|---|
| sub-01 | 378 | 359 | 592 | 460.1 | 173,910 |
| sub-02 | 379 | 33 | 592 | 459.1 | 173,998 |
| sub-03 | 377 | 359 | 592 | 460.0 | 173,430 |
| sub-04 | 259 | 373 | 592 | 456.8 | 118,318 |
| sub-05 | 375 | 359 | 592 | 460.2 | 172,578 |
| sub-06 | 377 | 359 | 592 | 460.3 | 173,542 |

Note: sub-02 has a minimum of 33 TRs for at least one run, which likely reflects a truncated or partial episode.

## Outputs

- `output/02_parcel_ts_avg/atlas-4S156Parcels/{sub_id}/{sub_id}_{ses}_task-{episode_id}_space-fsLR_den-91k_parcel_avg.npy` - one file per run; shape (n_TRs, 157)
