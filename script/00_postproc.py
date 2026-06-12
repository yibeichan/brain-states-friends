#!/usr/bin/env python
"""
Post-process fMRIPrep outputs with minimal confound regression strategy.

This script applies confound regression and standardization to fMRIPrep-processed
CIFTI files following a minimal preprocessing approach optimized for naturalistic
viewing paradigms.

Preprocessing Strategy (November 2025):
- Minimal confounds: motion (basic) + WM/CSF means + high-pass filter
- Voxel-wise z-scoring before parcellation (essential for unbiased parcel averaging)
- No global signal regression (preserves global patterns)
- No scrubbing (low-motion data, maintains narrative continuity)

For detailed rationale, see: the design notes

Author: Yibei Chen
Updated: November 2025
"""

import os
import glob
import nibabel as nib
from nilearn.interfaces.fmriprep import load_confounds
from nilearn import signal
from joblib import Parallel, delayed
from dotenv import load_dotenv
from argparse import ArgumentParser


def load_confounds_minimal(task_file):
    """
    Load minimal confound regressors using nilearn's standardized approach.

    Strategy: motion + high_pass + wm_csf (minimal denoising)
    - 6 motion parameters (basic: 3 translations + 3 rotations)
    - High-pass filter (discrete cosine basis)
    - 2 physiological regressors (mean WM + mean CSF signals)

    Total: ~13-18 regressors depending on run length

    Parameters
    ----------
    task_file : str
        Path to CIFTI functional file

    Returns
    -------
    confounds : pandas.DataFrame
        Confound regressors ready for signal.clean()

    Notes
    -----
    This follows the NeuroMod dataset recommendation for "minimal strategy"
    appropriate for low-motion naturalistic viewing data.
    """
    try:
        confounds, sample_mask = load_confounds(
            task_file,
            strategy=['motion', 'high_pass', 'wm_csf'],
            motion='basic',      # 6 motion params (not derivatives/quadratic)
            wm_csf='basic',      # Mean WM + CSF (not CompCor)
        )

        print(f"  Loaded {len(confounds.columns)} confound regressors: {confounds.columns.tolist()}")

        # Handle NaN values (can occur in first few timepoints)
        if confounds.isnull().any().any():
            n_nans = confounds.isnull().sum().sum()
            print(f"  Warning: {n_nans} NaN values in confounds, filling with 0")
            confounds = confounds.fillna(0)

        return confounds

    except Exception as e:
        print(f"  Error loading confounds with nilearn: {e}")
        print(f"  Attempting fallback to fMRIPrep confounds file...")

        # Fallback: load confounds file directly
        # C1 fix: match full BIDS prefix (sub/ses/task) to avoid loading wrong file
        basename = os.path.basename(task_file)
        bids_prefix = basename.split('_space-')[0]  # e.g. sub-01_ses-001_task-s01e02a
        confound_files = glob.glob(os.path.join(
            os.path.dirname(task_file),
            f"{bids_prefix}_desc-confounds*timeseries.tsv"))

        if not confound_files:
            raise FileNotFoundError(
                f"No confounds file found matching prefix: {bids_prefix}")

        import pandas as pd
        confounds_df = pd.read_csv(confound_files[0], sep='\t')

        # Minimal selection matching nilearn strategy
        motion_cols = [c for c in confounds_df.columns
                      if c in ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
        wm_csf_cols = [c for c in confounds_df.columns
                      if c in ['csf', 'white_matter']]
        cosine_cols = [c for c in confounds_df.columns if c.startswith('cosine')]

        selected_cols = motion_cols + wm_csf_cols + cosine_cols

        if len(selected_cols) == 0:
            raise ValueError("No matching confound columns found in confounds file")

        # C3 fix: demean confound columns (primary path via load_confounds demeans by default)
        confounds_out = confounds_df[selected_cols].copy()
        confounds_out = confounds_out - confounds_out.mean()
        confounds_out = confounds_out.fillna(0)

        print(f"  Fallback: loaded {len(selected_cols)} confound regressors (demeaned)")
        return confounds_out


def process_file(task_file, save_dir, standardize='zscore_sample'):
    """
    Process a single fMRI file with minimal confound regression.

    Processing steps:
    1. Load confounds (minimal strategy)
    2. Load functional data (CIFTI)
    3. Extract TR from CIFTI header
    4. Apply signal.clean():
       - Detrend (remove linear trends)
       - Regress confounds (motion + WM/CSF + high-pass)
       - Voxel-wise z-scoring (mean=0, std=1 per voxel across time)
    5. Save cleaned CIFTI

    Parameters
    ----------
    task_file : str
        Path to input CIFTI file
    save_dir : str
        Directory to save cleaned output
    standardize : str or False
        Standardization mode for signal.clean(). Default 'zscore_sample' applies
        voxel-wise z-scoring. Set to False for A3 sensitivity analysis
        (no per-episode z-scoring).

    Returns
    -------
    cleaned_file_path : str or None
        Path to saved cleaned file, or None if processing failed

    Notes
    -----
    Voxel-wise z-scoring is essential for unbiased parcel averaging in script 02.
    See the design notes for rationale.
    """
    cleaned_file_path = None

    try:
        print(f"\nProcessing: {os.path.basename(task_file)}")

        # Load confounds using minimal strategy
        confounds = load_confounds_minimal(task_file)

        # Load functional data
        task_img = nib.load(task_file)
        func_data = task_img.dataobj[:]

        # Extract TR from CIFTI header (more robust than hardcoding)
        try:
            tr = task_img.header.matrix.get_index_map(0).series_step
            print(f"  TR extracted from header: {tr:.3f} seconds")
        except Exception as e:
            print(f"  Warning: Could not extract TR from header ({e}), using default 1.49s")
            tr = 1.49

        # Apply confound regression and standardization
        std_label = standardize if standardize else 'none'
        print(f"  Cleaning data: detrend + confound regression + standardize={std_label}")
        cleaned_data = signal.clean(
            func_data,
            confounds=confounds,
            detrend=True,                    # Remove linear trends
            standardize=standardize,         # Voxel-wise z-scoring (or False for A3)
            t_r=tr
        )

        # C2 fix: free raw data early to reduce peak memory
        del func_data, task_img

        print(f"  Cleaned shape: {cleaned_data.shape}")

        # Save cleaned CIFTI - reload header from disk (task_img was deleted)
        header_img = nib.load(task_file)
        task_cln = nib.Cifti2Image(cleaned_data, header_img.header)
        cleaned_file_path = os.path.join(
            save_dir,
            os.path.basename(task_file).replace('.dtseries.nii', '_cleaned.dtseries.nii')
        )
        nib.save(task_cln, cleaned_file_path)
        print(f"  Saved: {cleaned_file_path}")

        # C2 fix: free cleaned data after save
        del cleaned_data, header_img, task_cln

    except Exception as e:
        print(f"  ERROR processing {task_file}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        return cleaned_file_path


def process_subject(subject_files, save_dir, n_jobs=-1, standardize='zscore_sample'):
    """
    Process all fMRI files for a subject in parallel.

    Parameters
    ----------
    subject_files : list of str
        Paths to CIFTI files to process
    save_dir : str
        Directory to save cleaned outputs
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all CPUs, capped at 8)
    standardize : str or False
        Passed to process_file(). Default 'zscore_sample'.

    Returns
    -------
    results : list
        List of output file paths (None for failed files)
    """
    if not subject_files:
        print("No subject files found.")
        return []

    # C2 fix: cap n_jobs to avoid OOM with 91k-grayordinate CIFTI (~1.2 GB/worker)
    if n_jobs == -1:
        n_jobs = min(8, os.cpu_count() or 8)

    print(f"\nProcessing {len(subject_files)} files with {n_jobs} parallel jobs...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(task_file, save_dir, standardize=standardize)
        for task_file in subject_files
    )

    # Summary
    successful = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)

    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful}/{len(subject_files)} files")
    if failed > 0:
        print(f"Failed: {failed} files")
    print(f"Output directory: {save_dir}")

    return results


if __name__ == "__main__":
    load_dotenv()

    parser = ArgumentParser(
        description="Post-process fMRIPrep CIFTI data with minimal confound regression",
        epilog="""
Examples:
  # Process all runs for subject
  python 00_postproc.py sub-01 friends

  # Process with fewer parallel jobs
  python 00_postproc.py sub-01 friends --n_jobs 4

Preprocessing Strategy:
  - Minimal confounds: 6 motion + 2 WM/CSF + high-pass (~13-18 regressors)
  - Voxel-wise z-scoring (essential for unbiased parcellation)
  - No global signal regression (preserves global patterns)
  - No scrubbing (maintains narrative continuity)

For detailed rationale: the design notes
        """
    )

    parser.add_argument("sub_id", type=str,
                       help="Subject ID (e.g., sub-01)")
    parser.add_argument("task", type=str,
                       help="Task name (e.g., friends)")
    parser.add_argument("--n_jobs", type=int, default=-1,
                       help="Number of parallel jobs (-1 = all CPUs, capped at 8)")
    parser.add_argument("--no_zscore", action="store_true",
                       help="Disable per-episode voxel-wise z-scoring (A3 sensitivity analysis)")
    parser.add_argument("--fmriprep_dir", type=str, default=None,
                       help="Override fMRIPrep directory (e.g., for petit-prince whose "
                            "fmriprep lives outside cneuromod.processed/)")

    args = parser.parse_args()

    # Load environment variables
    base_dir = os.getenv("BASE_DIR")
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.getenv("DATA_DIR")

    if not base_dir or not scratch_dir or not data_dir:
        print("ERROR: BASE_DIR, SCRATCH_DIR, or DATA_DIR environment variables not set")
        print("Please ensure .env file exists with these variables defined")
        exit(1)

    # A3 sensitivity: choose standardization mode and output directory
    if args.no_zscore:
        standardize = False
        output_subdir = "00_postproc_no_zscore"
    else:
        standardize = 'zscore_sample'
        output_subdir = "00_postproc"

    # Setup output directory (scratch output, tracked by DataLad)
    save_dir = os.path.join(scratch_dir, "output", output_subdir, args.sub_id)
    os.makedirs(save_dir, exist_ok=True)

    # Find subject files
    if args.fmriprep_dir:
        # Custom fmriprep directory (e.g., petit-prince.fmriprep/{sub_id})
        fmriprep_base = args.fmriprep_dir
    else:
        # Default: cneuromod.processed layout
        fmriprep_base = os.path.join(
            data_dir, "cneuromod.processed", "fmriprep",
            args.task, args.sub_id)

    # Session-based structure (friends, movie10, petit-prince)
    subject_files = glob.glob(os.path.join(
        fmriprep_base, "ses-*/func/*fsLR_den-91k*bold.dtseries.nii"))
    if not subject_files:
        # Fallback: session-less structure (e.g., harrypotter uses sub-XX/func/ directly)
        subject_files = glob.glob(os.path.join(
            fmriprep_base, "func/*fsLR_den-91k*bold.dtseries.nii"))

    if not subject_files:
        print(f"ERROR: No files found for subject {args.sub_id}, task {args.task}")
        print(f"Searched in: {fmriprep_base}/")
        exit(1)

    print(f"{'='*60}")
    print(f"fMRIPrep Post-processing - Minimal Confound Strategy")
    print(f"{'='*60}")
    print(f"Subject: {args.sub_id}")
    print(f"Task: {args.task}")
    print(f"Files found: {len(subject_files)}")
    print(f"Output directory: {save_dir}")
    print(f"Parallel jobs: {args.n_jobs if args.n_jobs > 0 else 'all CPUs'}")
    print(f"{'='*60}\n")

    # Process subject
    process_subject(subject_files, save_dir, n_jobs=args.n_jobs, standardize=standardize)
