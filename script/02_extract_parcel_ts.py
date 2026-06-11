#!/usr/bin/env python
"""
Extract Parcel Time Series - Average Activation

This script processes fMRI BOLD data to extract parcel-level average activation
for brain state analysis. It operates on cleaned CIFTI files to compute the mean
BOLD signal across all grayordinates within each parcel.

OUTPUT FILES:
=============
Average Activation (*_parcel_avg.npy)
--------------------------------------
- Shape: (n_timepoints, n_parcels)
- X[t, p] = average BOLD signal of all grayordinates in parcel p at time t
- Represents the mean activation level across all vertices within each parcel
- Useful for: connectivity analysis, brain state modeling, mean signal extraction

PARCELLATIONS:
==============
- atlas-4S156Parcels: 156 parcels combining cortical and subcortical regions

ALGORITHM:
==========
For each parcel at each timepoint:
1. Extract all grayordinate values within the parcel (already voxel-wise z-scored from script 00)
2. Compute mean activation across vertices → average activation output

USAGE:
======
# Single episode mode (typical SLURM array usage)
python 02_extract_parcel_ts.py --subject_id sub-01 --parcellation atlas-4S156Parcels --episode_id s01e01a

# All episodes for one subject and parcellation
python 02_extract_parcel_ts.py --subject_id sub-01 --parcellation atlas-4S156Parcels

# All episodes and all parcellations for one subject
python 02_extract_parcel_ts.py --subject_id sub-01

OUTPUT STRUCTURE:
=================
/scratch/output/
└── 02_parcel_ts_avg/
    └── atlas-4S156Parcels/
        └── sub-XX/
            └── *_parcel_avg.npy

Author: Yibei Chen (yc-nn branch)
"""

import os
import glob
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

# Fixed parcellations to use. Atlas files live under $ATLAS_DIR (see .env.example).
load_dotenv()
_ATLAS_DIR = os.getenv("ATLAS_DIR")
if _ATLAS_DIR is None:
    raise ValueError("ATLAS_DIR must be set in the .env file")
PARCELLATIONS = {
    name: os.path.join(_ATLAS_DIR, name, f"{name}_space-fsLR_den-91k_dseg.dlabel.nii")
    for name in (
        "atlas-4S156Parcels", "atlas-4S256Parcels",
        "atlas-4S356Parcels", "atlas-4S456Parcels",
    )
}

def load_parcellation(parc_path):
    """Load parcellation file and return the data array."""
    parc_img = nib.load(parc_path)
    return parc_img.get_fdata()

def extract_average_activation(bold_path, parcellation):
    """
    Extract average activation for each parcel from a BOLD run.
    
    Parameters
    ----------
    bold_path : str
        Path to the BOLD .dtseries.nii file
    parcellation : numpy.ndarray
        Parcellation data array
        
    Returns
    -------
    numpy.ndarray
        Array of shape (n_timepoints, n_parcels) - average activation per parcel
    """
    # Load BOLD data
    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata()
    
    print(f"BOLD shape: {bold_data.shape}")
    print(f"Parcellation shape: {parcellation.shape}")
    
    # Input validation
    if bold_data.ndim not in [2, 3]:
        raise ValueError(f"BOLD data should be 2D, or 3D, got shape {bold_data.shape}")
    
    # If bold_data is 3D, it has to be that it's (grayordinates, timepoints, 1) - reshape to 2D
    if bold_data.ndim == 3:
        orig_shape = bold_data.shape
        try:
            bold_data = bold_data.squeeze()
        except Exception as e:
            raise ValueError(f"Error squeezing BOLD data from shape {orig_shape}: {e}")
        print(f"Reshaped BOLD data from {orig_shape} to {bold_data.shape}")

    # Check orientation - ensure grayordinates × timepoints
    if bold_data.shape[0] < bold_data.shape[1]:
        print("Transposing to (grayordinates × timepoints) format")
        bold_data = bold_data.T
    
    print(f"Working with BOLD data of shape {bold_data.shape} (grayordinates × timepoints)")
    
    # Ensure parcellation is 1D
    if len(parcellation.shape) > 1:
        parcellation = np.squeeze(parcellation)
        if parcellation.ndim != 1:
            raise ValueError(f"Parcellation cannot be squeezed to 1D")
    
    # Handle NaN values
    if np.any(np.isnan(bold_data)):
        print("Warning: BOLD data contains NaN values, replacing with 0")
        bold_data = np.nan_to_num(bold_data, nan=0.0)
    
    # Dimension compatibility check
    if bold_data.shape[0] != len(parcellation):
        raise ValueError(
            f"Dimension mismatch: BOLD has {bold_data.shape[0]} grayordinates, "
            f"parcellation has {len(parcellation)} grayordinates"
        )
    
    # Get unique parcel IDs (excluding 0)
    parcel_ids = np.unique(parcellation)
    parcel_ids = parcel_ids[parcel_ids > 0]  # Exclude 0 (background)
    
    if len(parcel_ids) == 0:
        raise ValueError("No valid parcels found in parcellation")
    
    print(f"Found {len(parcel_ids)} unique parcels (IDs: {parcel_ids.min()} to {parcel_ids.max()})")
    
    # Initialize output array (TRs × max_parcel_id)
    # IMPORTANT: Index by parcel_id, not sequential index, to match label files
    n_timepoints = bold_data.shape[1]
    max_parcel_id = int(parcel_ids.max())
    
    # Create array sized by max parcel ID (not by number of parcels)
    average_activation = np.zeros((n_timepoints, max_parcel_id + 1))
    
    # Process each parcel
    for parcel_id in tqdm(parcel_ids, desc="Processing parcels"):
        # Convert to int for array indexing
        parcel_idx = int(parcel_id)
        parcel_mask = parcellation == parcel_id
        n_vertices = np.sum(parcel_mask)
        
        if n_vertices == 0:
            print(f"Warning: Parcel {parcel_id} is empty")
            continue
        
        # Extract data for this parcel (vertices × time)
        parcel_data = bold_data[parcel_mask, :]
        
        # Handle non-finite values
        if np.any(~np.isfinite(parcel_data)):
            parcel_data = np.nan_to_num(parcel_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Average activation - use parcel_idx as index to match label files
        average_activation[:, parcel_idx] = np.mean(parcel_data, axis=0)
    
    print(f"Average activation shape: {average_activation.shape}")
    
    return average_activation

def process_subject(subject_id, data_dir, scratch_dir, parcellation_filter=None, episode_id=None):
    """
    Process a single subject's data with specified parcellations and episodes.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., 'sub-01')
    data_dir : str
        Data directory path
    scratch_dir : str
        Scratch directory path
    parcellation_filter : str, optional
        If provided, only process this parcellation (e.g., 'atlas-4S156Parcels')
        If None, process all parcellations in PARCELLATIONS dict
    episode_id : str, optional
        If provided, only process this episode (e.g., 's01e01a')
        If None, process all episodes for the subject

    Returns
    -------
    bool
        True if processing succeeded, False if failed
    """

    # Filter parcellations if specified
    if parcellation_filter:
        if parcellation_filter not in PARCELLATIONS:
            print(f"ERROR: Parcellation '{parcellation_filter}' not found in PARCELLATIONS")
            print(f"Available parcellations: {list(PARCELLATIONS.keys())}")
            return False
        parcellations_to_process = {parcellation_filter: PARCELLATIONS[parcellation_filter]}
    else:
        parcellations_to_process = PARCELLATIONS

    # Track processing success
    files_processed = 0
    files_failed = 0

    # Process each parcellation
    for parc_name, parc_path in parcellations_to_process.items():
        print(f"\n{'='*60}")
        print(f"Processing {subject_id} with parcellation: {parc_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(parc_path):
            print(f"Warning: Parcellation file not found: {parc_path}")
            continue
        
        # Load parcellation
        try:
            parcellation = load_parcellation(parc_path)
        except Exception as e:
            print(f"Error loading parcellation {parc_path}: {e}")
            continue
        
        # Create output directory
        avg_output_dir = os.path.join(scratch_dir, "output", "02_parcel_ts_avg", parc_name, subject_id)
        os.makedirs(avg_output_dir, exist_ok=True)
        
        # Find BOLD files for this subject
        if episode_id:
            # Process only the specified episode
            # Use glob to match full BIDS naming (includes session and space entities)
            bold_pattern = os.path.join(scratch_dir, "output", "00_postproc",
                                       subject_id, f"{subject_id}*task-{episode_id}_*_bold_cleaned.dtseries.nii")
            bold_files = glob.glob(bold_pattern)
            if not bold_files:
                print(f"ERROR: No BOLD file found matching pattern: {bold_pattern}")
                print(f"Skipping episode {episode_id} for {subject_id} with {parc_name}")
                files_failed += 1
                continue
            print(f"Processing single episode: {episode_id}")
        else:
            # Process all episodes for this subject
            bold_files = glob.glob(
                os.path.join(scratch_dir, "output", "00_postproc",
                            subject_id, "*_bold_cleaned.dtseries.nii")
            )
            if not bold_files:
                print(f"WARNING: No BOLD files found for subject {subject_id}")
                continue
            print(f"Found {len(bold_files)} BOLD files to process")
        
        # Process each run
        for bold_file in tqdm(bold_files, desc=f"Processing runs"):
            # Extract run identifier from filename
            run_id = os.path.basename(bold_file).replace("_bold_cleaned.dtseries.nii", "")
            
            # Check if average file already exists
            avg_output_file = os.path.join(avg_output_dir, f"{run_id}_parcel_avg.npy")
            
            if os.path.exists(avg_output_file):
                print(f"Average file exists: {avg_output_file}, skipping...")
                files_processed += 1
                continue
            
            print(f"Processing BOLD data...")
            try:
                # Extract average activation
                avg_activation = extract_average_activation(bold_file, parcellation)
                
                # Save average activation
                np.save(avg_output_file, avg_activation)
                print(f"Saved average activation: {avg_output_file}")
                files_processed += 1
                
            except Exception as e:
                print(f"Error processing {bold_file}: {e}")
                files_failed += 1
                continue

    print(f"\nCompleted processing for subject {subject_id}")
    print(f"Files successfully processed: {files_processed}")
    print(f"Files failed: {files_failed}")

    # Return True only if at least one file was successfully processed
    if files_processed == 0:
        print("ERROR: No files were successfully processed")
        return False

    return True

def main():
    """Main function to process subjects with command-line arguments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract parcel time series - average activation only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single episode (typical SLURM array usage)
  python 02_extract_parcel_ts.py --subject_id sub-01 --parcellation atlas-4S156Parcels --episode_id s01e01a

  # Process all episodes for one subject and parcellation
  python 02_extract_parcel_ts.py --subject_id sub-01 --parcellation atlas-4S156Parcels

  # Process all episodes and all parcellations for one subject
  python 02_extract_parcel_ts.py --subject_id sub-01

  # Process all subjects (not recommended - use SLURM array instead)
  python 02_extract_parcel_ts.py
        """
    )
    # Command-line arguments
    parser.add_argument('--subject_id', '--sub-id', '--sub_id', type=str, default=None,
                       dest='subject_id',
                       help='Subject ID (e.g., sub-01). If not provided, processes all subjects.')
    parser.add_argument('--parcellation', type=str, default=None,
                       help='Parcellation name (e.g., atlas-4S156Parcels). If not provided, processes all parcellations.')
    parser.add_argument('--episode_id', type=str, default=None,
                       help='Episode ID without "task-" prefix (e.g., s01e01a). If not provided, processes all episodes.')
    args = parser.parse_args()

    # Validate arguments
    if args.episode_id and not args.subject_id:
        print("ERROR: --episode_id requires --subject_id to be specified")
        return 1
    if args.episode_id and not args.parcellation:
        print("ERROR: --episode_id requires --parcellation to be specified")
        print("       (Processing single episodes across multiple parcellations is not recommended)")
        return 1

    # Load environment variables
    load_dotenv()
    scratch_dir = os.getenv("SCRATCH_DIR")
    data_dir = os.getenv("DATA_DIR")

    if not scratch_dir or not data_dir:
        print("ERROR: SCRATCH_DIR and DATA_DIR must be set in .env file")
        return 1
    
    # Determine which subjects to process
    if args.subject_id:
        # Process single subject
        subjects = [args.subject_id]
        print(f"Processing single subject: {args.subject_id}")
    else:
        # Get all available subjects
        subjects_dir = os.path.join(scratch_dir, "output", "00_postproc")
        if not os.path.exists(subjects_dir):
            print(f"Error: Directory not found: {subjects_dir}")
            return
        
        subjects = [d for d in os.listdir(subjects_dir) 
                   if os.path.isdir(os.path.join(subjects_dir, d)) and d.startswith("sub-")]
        
        if not subjects:
            print("No subjects found")
            return
        
        print(f"Processing all {len(subjects)} subjects: {subjects}")
    
    print(f"\nConfiguration:")
    print(f"  - Subject(s): {subjects}")
    print(f"  - Parcellation: {args.parcellation if args.parcellation else 'all (' + ', '.join(PARCELLATIONS.keys()) + ')'}")
    print(f"  - Episode: {args.episode_id if args.episode_id else 'all'}")
    print(f"\nProcessing mode:")
    if args.episode_id:
        print(f"  - Single episode mode (SLURM array task)")
    else:
        print(f"  - Batch mode (all episodes)")
    print(f"\nNote: If average files exist, they will be skipped.")

    # Process subjects
    success_count = 0
    fail_count = 0
    for subject_id in subjects:
        success = process_subject(subject_id, data_dir, scratch_dir,
                                  parcellation_filter=args.parcellation,
                                  episode_id=args.episode_id)
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Subjects processed: {len(subjects)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    if args.episode_id:
        print(f"Episode: {args.episode_id}")
    if args.parcellation:
        print(f"Parcellation: {args.parcellation}")
    print(f"{'='*60}")

    # Return exit code based on success
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
