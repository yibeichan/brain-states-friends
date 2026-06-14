# Findings: 00 Post-processing (Minimal Confound Regression)

_Script: `script/00_postproc.py`. Tier: MAIN (Methods)._

_Cleans fMRIPrep CIFTI outputs with minimal confound regression and voxel-wise z-scoring; run per-subject across all stimuli._

## Method (as run)

- **Input:** fMRIPrep CIFTI files (`*_space-fsLR_den-91k_bold.dtseries.nii`) and confound TSV files from the CNeuroMod dataset
- **Confound strategy:** 6 basic motion parameters (translations + rotations) + mean WM signal + mean CSF signal + high-pass filter (discrete cosine basis); approximately 13-18 regressors per run depending on run length
- **No global signal regression; no scrubbing; detrending applied**
- **Standardization:** voxel-wise z-scoring per run (`zscore_sample`); optional `--no_zscore` variant writes to `output/00_postproc_no_zscore/` (not materialized in current output)
- **Parallelization:** up to 8 concurrent workers per subject (memory cap for 91k-grayordinate CIFTI)
- **Subjects:** n=6 (sub-01 to sub-06); sub-04 has incomplete Friends coverage (seasons 1-6 available but fewer runs) and is missing Harry Potter and Petit Prince runs
- **Stimuli processed:** Friends (all 6 seasons), Movie10 (Bourne Identity, Figures, Life, Wolf of Wall Street), Harry Potter, Petit Prince (lppFR, lppEN); outputs are not per-parcellation (no parcellation level in path)

## Results

### Run counts per subject (Friends stimulus only)

| Subject | Friends runs | Total TRs | TR range | Median TRs/run |
|---|---|---|---|---|
| sub-01 | 292 | 137,913 | 428-592 | 470 |
| sub-02 | 293 | 137,946 | 33-592 | 470 |
| sub-03 | 291 | 137,457 | 428-592 | 470 |
| sub-04 | 198 | 93,486 | 437-592 | 469.5 |
| sub-05 | 289 | 136,529 | 428-592 | 470 |
| sub-06 | 293 | 138,366 | 428-592 | 470 |

Note: sub-02 has one short run (s05e09a, 33 TRs); sub-04 has substantially fewer Friends runs than other subjects.

### Total cleaned files per subject (all stimuli)

| Subject | Total cleaned files | Friends | Movie10 | Harry Potter | Petit Prince (lppFR+EN) |
|---|---|---|---|---|---|
| sub-01 | 378 | 292 | 61 | 7 | 18 |
| sub-02 | 379 | 293 | 61 | 7 | 18 |
| sub-03 | 377 | 291 | 61 | 7 | 18 |
| sub-04 | 259 | 198 | 61 | 0 | 0 |
| sub-05 | 375 | 289 | 61 | 7 | 18 |
| sub-06 | 377 | 293 | 61 | 7 | 16 |

### Output file properties

| Property | Value |
|---|---|
| CIFTI grayordinates | 91,282 |
| TR | 1.49 s (read from CIFTI header) |
| Typical run duration | ~470 TRs x 1.49 s = ~700 s (~11.7 min) |
| File size per run | ~413 MB |
| Standardization | voxel-wise z-score (mean=0, std=1 per grayordinate across time within run) |

## Outputs

- `output/00_postproc/{sub_id}/{sub_id}_{ses}_{task}_space-fsLR_den-91k_bold_cleaned.dtseries.nii` - one cleaned CIFTI per run per subject
