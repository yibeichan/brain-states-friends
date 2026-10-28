# Findings: 00 Post-processing (Minimal Confound Regression)

_Script: `script/00_postproc.py`. Tier: MAIN (Methods)._

_Cleans fMRIPrep CIFTI outputs with minimal confound regression and voxel-wise z-scoring; run per-subject across all stimuli._

## Method (as run)

- **Input:** fMRIPrep CIFTI files (`*_space-fsLR_den-91k_bold.dtseries.nii`) and confound TSV files from the CNeuroMod dataset
- **Confound strategy:** 6 basic motion parameters (translations + rotations) + mean WM signal + mean CSF signal + high-pass filter (discrete cosine basis); approximately 13-18 regressors per run depending on run length
- **No global signal regression; no scrubbing; detrending applied**
- **Standardization:** voxel-wise z-scoring per run (`zscore_sample`); optional `--no_zscore` variant writes to `output/00_postproc_no_zscore/` (not materialized in current output)
- **Optional CSF/WM override:** `--wmcsf_table` with `--wmcsf_variant clean|corrupt` (`WMCSF_TABLE` / `WMCSF_VARIANT` in `00_postproc.sh`) replaces the `csf` and `white_matter` columns with the clean-mask or corrupt-mask columns of the shared CNeuroMod CSF/WM table, demeaned like the other regressors (`utils/wmcsf_override.py`). Output names are unchanged, so the override is meant for a separate `SCRATCH_DIR`, as `clean_wmcsf_e2e_submit.sh` sets one per variant. Not used for the published outputs.
- **Exit status:** the script exits 1 if any run fails, after processing the others; this applies to every stimulus wrapper that calls it.
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

## Known limitation: CSF and WM regressors from corrupted tissue maps

The CNeuroMod anatomical derivatives (`anat.smriprep`) were built with fMRIPrep 20.2.1 `--anat-only`, whose sMRIPrep (< 0.7.2) resampled the FreeSurfer aseg with trilinear instead of nearest-neighbour interpolation (fixed upstream in sMRIPrep 0.7.2). The shipped GM/WM/CSF tissue maps therefore contain spurious intermediate labels, and the CSF class is inflated. Every CNeuroMod BOLD fMRIPrep run reused these maps (`--anat-derivatives`), so the `csf` and `white_matter` confound columns regressed here are mask means over affected masks. The BOLD data, motion parameters and cosine regressors are not affected. Because the masks are per participant, the same issue applies to every stimulus processed here; the Movie10, Harry Potter, Petit Prince and hcptrt derivatives all pin the same `anat.smriprep` tissue maps, and their fMRIPrep versions (20.2.1–20.2.6) share the same confound code.

**Measured effect.** One Friends run per participant (session 010) was cleaned exactly as above twice, once with the shipped `csf` / `white_matter` columns and once with columns recomputed from masks rebuilt from the clean FreeSurfer aseg using fMRIPrep 20.2.6's own mask code, and the cleaned data were averaged into the 4S156 parcels:

| sub | CSF mask voxels, shipped → clean | `csf` r | `white_matter` r | cortical parcel r, median / min | subcortical parcel r, median / min | top-15 PC subspace, min cosine |
|---|---|---|---|---|---|---|
| 01 | 500 → 274 | 0.954 | 0.962 | 0.990 / 0.977 | 0.995 / 0.962 | 0.991 |
| 02 | 742 → 494 | 0.987 | 0.997 | 0.997 / 0.992 | 0.998 / 0.982 | 0.983 |
| 03 | 507 → 52 | 0.714 | 0.988 | 0.991 / 0.968 | 0.987 / 0.958 | 0.969 |
| 04 | 581 → 42 | 0.813 | 0.997 | 0.995 / 0.946 | 0.994 / 0.936 | 0.972 |
| 05 | 521 → 115 | 0.851 | 0.996 | 0.987 / 0.968 | 0.991 / 0.965 | 0.992 |
| 06 | 473 → 397 | 0.929 | 0.983 | 0.996 / 0.991 | 0.998 / 0.979 | 0.995 |

The same check on one run per participant of each other stimulus gives the same pattern:

| stimulus | n | `csf` r | median parcel r, cortical / subcortical | lowest parcel r |
|---|---|---|---|---|
| Movie10 | 6 | 0.694–0.992 | 0.978–0.999 / 0.990–0.999 | 0.942 |
| Harry Potter | 5 | 0.790–0.992 | 0.978–0.996 / 0.987–0.998 | 0.954 |
| Petit Prince | 5 | 0.730–0.988 | 0.990–0.995 / 0.991–0.999 | 0.959 |
| Rest (hcptrt) | 6 | 0.652–0.979 | 0.977–0.999 / 0.989–0.999 | 0.937 |

Across stimuli, the `csf` regressor is substantially wrong in sub-03, sub-04 and sub-05, but because it is one of 13–18 regressors the cleaned parcel time series change little (Friends median r 0.987–0.998; the lowest parcels, mostly cerebellar, 0.936–0.992). All r values compare the shipped columns with the clean-mask rebuild, so they include the rebuild's own departure from fMRIPrep, which runs on native-space rather than T1w-space BOLD. A rebuild from the corrupted masks reproduces the shipped `csf` at r ≥ 0.96 in 24 of the 28 runs checked and at 0.90–0.95 in the other four (all rest), and the shipped `white_matter` at r 0.63 in rest sub-06 and 0.92 in Harry Potter sub-01. The low `white_matter` values in those two runs therefore reflect the rebuild, not the corrupted masks (corrupted vs clean rebuild: r 0.98 and 0.99). Comparing the corrupted-mask rebuild with the clean-mask rebuild isolates the tissue-map bug: `csf` r is 0.728–0.991 for Friends, 0.675–0.994 for Movie10, 0.718–0.994 for Harry Potter, 0.733–0.990 for Petit Prince and 0.640–0.992 for rest (lowest in sub-03, sub-04 and sub-05), while `white_matter` r is at least 0.966 in every run, so the bug acts almost entirely through the CSF regressor. Neither comparison bounds the other: the bug-only `csf` r is the lower of the two in 6 of the 28 runs. Measured as the bug alone, the cleaned parcel time series still change little: median parcel r is at least 0.966 in every run, and the lowest single parcel is 0.931–0.963 depending on the dataset (Friends 0.940, Movie10 0.931, Harry Potter 0.942, Petit Prince 0.963, rest 0.935). The PC subspace is computed here from a single run, a noisier proxy for the 03a PCA over all runs. The pipeline was not rerun; results are reported with this limitation.

## Outputs

- `output/00_postproc/{sub_id}/{sub_id}_{ses}_{task}_space-fsLR_den-91k_bold_cleaned.dtseries.nii` - one cleaned CIFTI per run per subject
